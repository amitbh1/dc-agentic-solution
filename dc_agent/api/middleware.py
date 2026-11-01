from __future__ import annotations

import json
import re
import time

from fastapi import Request
from fastapi.responses import JSONResponse

from ..observability import audit, new_request_id, request_id_var
from ..observability.metrics import REQUESTS

_RID = re.compile(r"^[A-Za-z0-9_.-]{8,64}$")
MAX_BODY_BYTES = 16 * 1024


async def request_context(request: Request, call_next):
    """Request id propagation, fast body size check, request metric and one audit line per request."""
    rid = request.headers.get("x-request-id", "")
    rid = rid if _RID.match(rid) else new_request_id()
    token = request_id_var.set(rid)
    t0 = time.perf_counter()
    try:
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            resp = JSONResponse({"error": "payload too large", "request_id": rid}, status_code=413)
        else:
            resp = await call_next(request)
    finally:
        request_id_var.reset(token)
    route = getattr(request.scope.get("route"), "path", "unmatched")
    REQUESTS.labels(route, str(resp.status_code)).inc()
    resp.headers["X-Request-ID"] = rid
    audit("http", route=route, status=resp.status_code, ms=round((time.perf_counter() - t0) * 1000))
    return resp


class BodyLimit:
    """ASGI middleware enforcing the body cap on the bytes actually received, so chunked requests
    without a Content-Length header cannot bypass it. Bodies are small (16 KB), so the request body
    is read up front and replayed to the app; an oversized body gets 413 without reaching the app."""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        messages, seen = [], 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break  # disconnect: let the app see it
            seen += len(message.get("body", b""))
            if seen > self.max_bytes:
                body = json.dumps({"error": "payload too large"}).encode()
                await send({"type": "http.response.start", "status": 413,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
            if not message.get("more_body", False):
                break

        async def replay():
            return messages.pop(0) if messages else await receive()

        await self.app(scope, replay, send)
