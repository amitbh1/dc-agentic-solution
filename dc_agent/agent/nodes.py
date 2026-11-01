"""Graph nodes. Each takes the state and returns a partial update; none of them raise for
expected failures. Unexpected exceptions are caught by graph.instrument."""
from __future__ import annotations

import logging
import math
import time

from langgraph.types import interrupt
from pydantic import ValidationError

from ..core.errors import AgentError, CircuitOpen, InputBlocked, Overloaded
from ..llm import PROMPT_VERSION, PlanRequest
from ..memory import PlanCache
from ..observability import audit, hash_text
from ..observability.metrics import ANSWERS, APPROVALS, BLOCKS, CACHE, COST_GATE, FAILURES, GROUNDING
from ..query import MergeTooLarge, QueryPlan, compile_plan, guard, merge_parts
from ..security import UserContext, check_input, redact_rows
from ..semantic import is_follow_up, select_examples
from ..semantic.text import phrase_windows
from .deps import Deps
from .messages import MSG, dependency_failure, fatal, human_bytes, message_for
from .render import render_rows
from .state import AgentState, fresh_turn, outcome_of

log = logging.getLogger("dc_agent.agent")


def _user(state: AgentState) -> UserContext:
    return UserContext.from_dict(state["user"])


def remaining_s(state: AgentState) -> float:
    return state.get("deadline", math.inf) - time.time()


class AgentNodes:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.s = deps.settings

    # --- helpers -------------------------------------------------------------------------

    @staticmethod
    def previous_plan(state: AgentState) -> dict | None:
        """The last answered plan, but only when this question refines it."""
        last = state.get("last_plan")
        return last if last and is_follow_up(state.get("question", "")) else None

    def _cache_key(self, state: AgentState, user: UserContext) -> tuple:
        return PlanCache.key(state.get("release_id") or 0, user.role, state["question"], self.previous_plan(state))

    def _embed_question(self, question: str, onto, user: UserContext):
        """Vectors for the question and its short windows (one batched call), or None when embeddings
        are off, unused by this release, or failing. The first vector is the whole question."""
        if self.deps.embedder is None or not onto.has_vectors:
            return None
        try:
            return self.deps.embedder.embed([question, *phrase_windows(question)], kind="query")
        except Exception as exc:
            audit("embed_failed", user=user.user_id, kind=type(exc).__name__)
            return None

    # --- nodes ---------------------------------------------------------------------------

    def guard_input(self, state: AgentState) -> dict:
        out = fresh_turn()
        out["deadline"] = time.time() + self.s.request_timeout_s
        user = _user(state)
        question = state.get("question", "")
        try:
            check_input(question)
        except InputBlocked as exc:
            escalated = self.deps.blocks.record(user.user_id)
            BLOCKS.labels("input").inc()
            audit("input_blocked", user=user.user_id, reason=str(exc), q_hash=hash_text(question), repeated=escalated)
            if escalated:
                log.warning("repeated blocked inputs from one user", extra={"fields": {"user": user.user_id}})
            out.update(error=str(exc), error_fatal=True, error_kind="blocked", user_message=MSG["blocked"])
        return out

    def ground(self, state: AgentState) -> dict:
        user = _user(state)
        onto = self.deps.provider.get()
        question = state["question"]
        qvecs = self._embed_question(question, onto, user)
        min_sim = getattr(self.deps.embedder, "min_similarity", 1.0)
        sl = onto.ground(question, user, qvecs, min_sim)
        how = "lexical"
        if sl is not None and qvecs and onto.ground(question, user) is None:
            how = "semantic"
        prev = self.previous_plan(state)
        if sl is None and prev:
            sl, how = onto.slice_for(prev.get("metrics", []), user), "follow_up"
        if sl is None:
            GROUNDING.labels("none").inc()
            hint = onto.catalogue_hint(user)
            return {"clarification": f"I couldn't match that to a known metric. I can report: {hint}."}
        GROUNDING.labels(how).inc()
        examples = select_examples(onto.examples, sl, question, self.s.few_shot_k, qvecs[0] if qvecs else None)
        return {"slice": sl, "examples": examples, "release_id": onto.release_id}

    def planner(self, state: AgentState) -> dict:
        user = _user(state)
        attempts = state.get("attempts", 0)
        err = state.get("error")
        if remaining_s(state) <= 0:
            audit("deadline_exceeded", user=user.user_id, node="planner", attempts=attempts)
            return {"plan": None, **fatal("timeout", "timeout")}

        key = self._cache_key(state, user)
        if attempts == 0 and not err:
            hit = self.deps.cache.get(key)
            CACHE.labels("hit" if hit else "miss").inc()
            if hit:
                audit("plan_cache_hit", user=user.user_id)
                return {"plan": hit, "attempts": 1}

        req = PlanRequest(
            question=state["question"], catalogue=state["slice"], examples=state.get("examples", []),
            error=err, failed_plan=state.get("plan") if err else None, previous_plan=self.previous_plan(state),
        )
        try:
            out = self.deps.llm.plan(req)
        except (CircuitOpen, Overloaded) as exc:
            audit("llm_unavailable", user=user.user_id, kind=type(exc).__name__)
            return {"plan": None, "attempts": attempts + 1, **dependency_failure(exc, "llm_unavailable")}
        except ValidationError as exc:
            # The model returned something that is not a PlannerOutput: worth one more try.
            audit("llm_malformed", user=user.user_id, attempt=attempts + 1, detail=str(exc)[:300])
            return {
                "plan": None, "attempts": attempts + 1, "error_fatal": False, "error_kind": "malformed_plan",
                "error": f"output did not match the QueryPlan schema: {exc.errors()[0]['msg']}",
                "user_message": MSG["plan"],
            }
        except Exception as exc:
            audit("llm_failed", user=user.user_id, error=repr(exc)[:300])
            return {"plan": None, "attempts": attempts + 1, **fatal("planner", "plan", "planner failed")}
        if out.plan is None:
            return {"clarification": out.clarification or MSG["plan"], "plan": None, "attempts": attempts + 1}
        return {"plan": out.plan.model_dump(), "error": None, "error_kind": None, "attempts": attempts + 1}

    def compile(self, state: AgentState) -> dict:
        user = _user(state)
        onto = self.deps.provider.get()
        try:
            qp = QueryPlan.model_validate(state["plan"])
            c = compile_plan(qp, onto, user, self.s.max_rows, dialects=self.deps.dialects,
                             max_partial_rows=self.s.max_partial_rows)
            cap = self.s.max_partial_rows + 1 if c.merge else self.s.max_rows
            parts = [
                {"engine": p.engine, "dialect": p.dialect, "sql": guard(p.sql, p.dialect, p.tables, cap),
                 "tables": sorted(p.tables), "columns": p.columns}
                for p in c.parts
            ]
        except ValidationError as exc:
            return {
                "error": f"invalid plan: {exc.errors()[0]['msg']}", "error_fatal": False,
                "error_kind": "ValidationError", "sql": None,
            }
        except AgentError as exc:
            audit("compile_failed", user=user.user_id, kind=type(exc).__name__, detail=str(exc))
            return {
                "error": f"{type(exc).__name__}: {exc}", "error_fatal": exc.fatal,
                "error_kind": type(exc).__name__, "user_message": message_for(exc), "sql": None,
            }
        sql = parts[0]["sql"] if len(parts) == 1 else "\n\n".join(f"-- engine: {p['engine']}\n{p['sql']}" for p in parts)
        return {
            "sql": sql, "dialect": parts[0]["dialect"], "tables": sorted(c.tables), "parts": parts,
            "merge": c.merge, "expected_columns": c.columns, "error": None, "error_kind": None,
        }

    def cost_check(self, state: AgentState) -> dict:
        """EXPLAIN-based cost gate: refuse above DC_MAX_SCAN_BYTES, ask the user above DC_APPROVAL_SCAN_BYTES.

        Asking pauses the graph (LangGraph interrupt); the API resumes it with the user's decision.
        If no estimate is available the query proceeds: timeouts and row caps still apply.
        """
        s = self.s
        if not (s.max_scan_bytes or s.approval_scan_bytes):
            return {}
        user = _user(state)
        total, known = 0, False
        for part in state.get("parts", []):
            ex = self.deps.executors.get(part["engine"])
            try:
                est = ex.estimate(part["sql"]) if ex is not None and hasattr(ex, "estimate") else None
            except Exception as exc:
                audit("estimate_failed", user=user.user_id, engine=part["engine"], error=repr(exc)[:200])
                est = None
            if est is not None and est.bytes is not None:
                total += est.bytes
                known = True
        if not known:
            COST_GATE.labels("unknown").inc()
            return {"estimate_bytes": None}
        if s.max_scan_bytes and total > s.max_scan_bytes:
            COST_GATE.labels("refused").inc()
            audit("cost_refused", user=user.user_id, bytes=total, limit=s.max_scan_bytes)
            return {"estimate_bytes": total, **fatal("too_expensive", "too_expensive")}
        if s.approval_scan_bytes and total > s.approval_scan_bytes:
            COST_GATE.labels("approval").inc()
            decision = interrupt({
                "kind": "approval", "estimated_bytes": total,
                "message": MSG["approval"].format(size=human_bytes(total)),
            })
            approved = decision is True or (isinstance(decision, dict) and decision.get("approve") is True)
            APPROVALS.labels("approved" if approved else "rejected").inc()
            audit("approval_decision", user=user.user_id, bytes=total, approved=approved)
            if not approved:
                return {"estimate_bytes": total, **fatal("cancelled", "cancelled")}
            # The user may have taken a while to decide: give execution a fresh budget.
            return {"estimate_bytes": total, "deadline": time.time() + s.request_timeout_s}
        COST_GATE.labels("allowed").inc()
        return {"estimate_bytes": total}

    def execute(self, state: AgentState) -> dict:
        user = _user(state)
        merge = state.get("merge")
        results = []
        for part in state.get("parts", []):
            left = remaining_s(state)
            if left <= 0:
                audit("deadline_exceeded", user=user.user_id, node="execute")
                return fatal("timeout", "timeout")
            ex = self.deps.executors.get(part["engine"])
            if ex is None:
                return {"error": "no executor", "error_kind": "execution", "user_message": MSG["exec"]}
            timeout_s = max(1, min(self.s.query_timeout_s, math.floor(left)))
            cap = self.s.max_partial_rows + 1 if merge else self.s.max_rows
            try:
                results.append(ex.run(part["sql"], cap, timeout_s))
            except (CircuitOpen, Overloaded) as exc:
                audit("execute_unavailable", user=user.user_id, kind=type(exc).__name__)
                return dependency_failure(exc, "unavailable")
            except TimeoutError:
                audit("execute_timeout", user=user.user_id, timeout_s=timeout_s, engine=part["engine"])
                return fatal("timeout", "timeout")
            except Exception as exc:
                audit("execute_failed", user=user.user_id, engine=part["engine"], error=repr(exc)[:300])
                return {"error": "execution failed", "error_kind": "execution", "user_message": MSG["exec"]}
        if merge:
            try:
                rows = merge_parts(merge, results[0][1], results[1][1])
            except MergeTooLarge as exc:
                audit("merge_too_large", user=user.user_id, detail=str(exc))
                return fatal("too_large", "too_large")
            cols = merge["columns"]
        else:
            cursor_cols, rows = results[0]
            expected = state.get("expected_columns") or []
            # Engines differ in identifier case (Snowflake upper-cases); report the compiler's names.
            cols = expected if len(expected) == len(cursor_cols) else cursor_cols
        if self.s.redact_pii:
            rows = redact_rows(rows, self.deps.redactor)
        return {"columns": cols, "rows": rows}

    def check_result(self, state: AgentState) -> dict:
        warnings = []
        rows, cols = state.get("rows", []), state.get("columns", [])
        if not rows:
            warnings.append("No rows matched the filters.")
        if len(rows) >= self.s.max_rows:
            warnings.append(f"Result truncated at {self.s.max_rows} rows.")
        for i, c in enumerate(cols):
            if rows and sum(1 for r in rows if r[i] is None) / len(rows) > 0.5:
                warnings.append(f"Column {c} is mostly empty.")
        if state.get("merge"):
            warnings.append("Combined from several data platforms.")
        return {"warnings": warnings}

    def enrich(self, state: AgentState) -> dict:
        """Optional role-scoped tools (for example runbook search). They see the question and the
        shape of the answer, never the rows."""
        user = _user(state)
        plan = state.get("plan") or {}
        context = {
            "question": state["question"], "metrics": plan.get("metrics", []),
            "dimensions": plan.get("dimensions", []), "columns": state.get("columns", []),
            "row_count": len(state.get("rows", [])),
        }
        return {"notes": self.deps.tools.enrich(user, context)}

    def respond(self, state: AgentState) -> dict:
        user = _user(state)
        outcome = outcome_of(state)
        ANSWERS.labels(outcome).inc()
        if outcome not in ("ok", "clarification"):
            FAILURES.labels(state.get("error_kind") or "unknown").inc()
        if outcome == "clarification":
            return {"answer": state["clarification"]}
        if outcome != "ok":
            return {"answer": state.get("user_message") or MSG["plan"]}
        self.deps.cache.set(self._cache_key(state, user), state["plan"])
        audit(
            "answered", user=user.user_id, role=user.role, tenant=user.tenant, release=state.get("release_id"),
            rows=len(state.get("rows", [])), sql=state["sql"], q_hash=hash_text(state["question"]),
            attempts=state.get("attempts", 0), engines=[p["engine"] for p in state.get("parts", [])],
            estimate_bytes=state.get("estimate_bytes"), prompt_version=PROMPT_VERSION, model=self.s.plan_model,
        )
        return {"answer": render_rows(state["columns"], state["rows"]), "last_plan": state["plan"]}
