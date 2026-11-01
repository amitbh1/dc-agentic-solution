from __future__ import annotations

import json
import logging
import math

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from langgraph.types import Command

from ..agent import outcome_of
from ..bootstrap import Bundle
from ..core.errors import RateLimited
from ..memory import FeedbackRecord
from ..memory.plan_cache import PlanCache
from ..observability import audit, hash_text, request_id_var
from ..observability.metrics import FEEDBACK, RATE_LIMITED, render
from ..security import UserContext
from .dependencies import current_user, get_bundle
from .schemas import ApproveRequest, AskRequest, FeedbackRequest

log = logging.getLogger("dc_agent.api")
RECURSION_LIMIT = 25

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(b: Bundle = Depends(get_bundle)) -> dict:
    if not b.ready():
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> Response:
    """Expose only on an internal network or behind your metrics scraper's auth."""
    body, content_type = render()
    return Response(content=body, media_type=content_type)


def _thread(user: UserContext, session_id: str | None) -> str:
    return f"{user.user_id}:{session_id or 'default'}"


def _begin(b: Bundle, session_id: str | None, user: UserContext) -> str:
    """Single flight per session, then rate limit. Returns the checkpoint thread id; caller releases it."""
    thread_id = _thread(user, session_id)
    if not b.locks.try_acquire(thread_id):
        raise HTTPException(status_code=409, detail="another request in this session is still running")
    try:
        b.limiter.check(user.user_id, user.tenant)
    except RateLimited as exc:
        b.locks.release(thread_id)
        RATE_LIMITED.labels(exc.reason).inc()
        raise HTTPException(
            status_code=429, detail=exc.reason,
            headers={"Retry-After": str(max(1, math.ceil(exc.retry_after_s)))},
        )
    return thread_id


def _input(req: AskRequest, user: UserContext) -> dict:
    return {"question": req.question, "user": user.to_dict()}


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def _payload(state: dict, user: UserContext) -> dict:
    interrupts = state.get("__interrupt__") or ()
    if interrupts:
        info = interrupts[0].value
        return {
            "outcome": "approval_required",
            "answer": info.get("message"),
            "approval": {"estimated_bytes": info.get("estimated_bytes")},
            "columns": [], "rows": [], "warnings": [], "notes": [],
            "release": state.get("release_id"),
            "request_id": request_id_var.get(),
        }
    out = {
        "outcome": outcome_of(state),
        "answer": state.get("answer"),
        "columns": state.get("columns", []),
        "rows": state.get("rows", []),
        "warnings": state.get("warnings", []),
        "notes": state.get("notes", []),
        "release": state.get("release_id"),
        "request_id": request_id_var.get(),
    }
    if user.role == "admin":
        out["sql"] = state.get("sql")
    return out


def _invoke(b: Bundle, thread_id: str, payload, user: UserContext) -> dict:
    try:
        state = b.agent.invoke(payload, config=_config(thread_id))
    except Exception:
        log.exception("agent invocation failed")
        raise HTTPException(status_code=500, detail="internal error")
    finally:
        b.locks.release(thread_id)
    return _payload(state, user)


@router.post("/ask")
def ask(req: AskRequest, user: UserContext = Depends(current_user), b: Bundle = Depends(get_bundle)) -> dict:
    thread_id = _begin(b, req.session_id, user)
    return _invoke(b, thread_id, _input(req, user), user)


@router.post("/ask/approve")
def approve(req: ApproveRequest, user: UserContext = Depends(current_user), b: Bundle = Depends(get_bundle)) -> dict:
    """Resume a turn paused for approval (outcome approval_required) with the user's decision.
    Threads are keyed by user id, so only the user who asked can approve their own query."""
    thread_id = _begin(b, req.session_id, user)
    try:
        pending = b.agent.get_state(_config(thread_id)).interrupts
    except Exception:
        b.locks.release(thread_id)
        log.exception("could not read thread state")
        raise HTTPException(status_code=500, detail="internal error")
    if not pending:
        b.locks.release(thread_id)
        raise HTTPException(status_code=409, detail="nothing is waiting for approval in this session")
    return _invoke(b, thread_id, Command(resume={"approve": req.approve}), user)


@router.post("/feedback")
def feedback(req: FeedbackRequest, user: UserContext = Depends(current_user), b: Bundle = Depends(get_bundle)) -> dict:
    """Rate the latest answer in a session. Thumbs-up answers feed verified-query promotion."""
    thread_id = _thread(user, req.session_id)
    try:
        values = b.agent.get_state(_config(thread_id)).values or {}
    except Exception:
        log.exception("could not read thread state")
        raise HTTPException(status_code=500, detail="internal error")
    question = values.get("question")
    if not question or not values.get("answer"):
        raise HTTPException(status_code=409, detail="no answer to rate in this session")
    outcome = outcome_of(values)
    ok = outcome == "ok"
    rec = FeedbackRecord(
        user_id=user.user_id, role=user.role, tenant=user.tenant, thread_id=thread_id, rating=req.rating,
        q_hash=hash_text(PlanCache.key(0, "", question)[2]), outcome=outcome,
        request_id=request_id_var.get(), comment=req.comment,
        question=question if req.rating == "up" else None,
        plan=values.get("plan") if ok else None, sql=values.get("sql") if ok else None,
        dialect=values.get("dialect") if ok else None, release_id=values.get("release_id"),
    )
    try:
        b.feedback.add(rec)
    except Exception:
        log.exception("feedback write failed")
        raise HTTPException(status_code=503, detail="feedback is temporarily unavailable")
    FEEDBACK.labels(req.rating).inc()
    audit("feedback", user=user.user_id, rating=req.rating, outcome=outcome, q_hash=rec.q_hash,
          release=rec.release_id, has_comment=bool(req.comment))
    return {"status": "recorded", "request_id": request_id_var.get()}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/ask/stream")
def ask_stream(
    req: AskRequest, user: UserContext = Depends(current_user), b: Bundle = Depends(get_bundle)
) -> StreamingResponse:
    """Server-sent events: one `node` event per completed step, then a final `result` (which may be
    approval_required). Intermediate state (SQL, rows) is deliberately not streamed."""
    thread_id = _begin(b, req.session_id, user)
    rid = request_id_var.get()

    def events():
        # The body is iterated after the middleware returns, possibly in a different context per step,
        # so set the request id here rather than relying on the middleware's (already reset) value.
        request_id_var.set(rid)
        final: dict = {}
        try:
            for chunk in b.agent.stream(_input(req, user), config=_config(thread_id), stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        final["__interrupt__"] = update
                        continue
                    if isinstance(update, dict):
                        final.update(update)
                    yield _sse("node", {"node": node})
            request_id_var.set(rid)
            yield _sse("result", _payload(final, user))
        except Exception:
            request_id_var.set(rid)
            log.exception("agent stream failed")
            yield _sse("error", {"error": "internal error", "request_id": rid})
        finally:
            b.locks.release(thread_id)

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
