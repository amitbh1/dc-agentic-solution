from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import uuid

request_id_var: contextvars.ContextVar = contextvars.ContextVar("request_id", default="-")
_log = logging.getLogger("dc_agent.audit")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        base.update(getattr(record, "fields", {}))
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def new_request_id() -> str:
    return uuid.uuid4().hex


def hash_text(text: str) -> str:
    """Stable short hash so logs can correlate questions without storing them."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def audit(event: str, **fields) -> None:
    """One JSON line per event. Never log question text or result rows."""
    _log.info(event, extra={"fields": fields})
