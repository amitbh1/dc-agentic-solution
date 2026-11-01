"""User-facing messages. Internal error detail never reaches the user; it goes to the audit log."""
from __future__ import annotations

from ..core.errors import AccessError, CrossEngineError, GuardError, InputBlocked, Overloaded

MSG = {
    "blocked": "I can't process that request.",
    "access": "Your role doesn't have access to some of the data needed for that question.",
    "engine": "That question spans several data platforms, which isn't supported yet.",
    "guard": "The generated query failed a safety check, so it was not run.",
    "plan": "I couldn't build a reliable query for that. Try naming the metric and the breakdown you want.",
    "exec": "The query couldn't be completed. Please try again or narrow the question.",
    "unavailable": "A required service is temporarily unavailable. Please retry shortly.",
    "busy": "The system is busy. Please retry shortly.",
    "timeout": "That took too long to answer. Please try a narrower question.",
    "internal": "Something went wrong on our side. Please try again.",
    "too_expensive": "That query would scan too much data. Please narrow the time range or add filters.",
    "too_large": "That question needs too much data from several platforms. Please narrow it.",
    "cancelled": "Okay, I didn't run that query.",
    "approval": "This query will scan about {size}. Do you want to run it?",
}


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def message_for(exc: Exception) -> str:
    if isinstance(exc, InputBlocked):
        return MSG["blocked"]
    if isinstance(exc, AccessError):
        return MSG["access"]
    if isinstance(exc, CrossEngineError):
        return MSG["engine"]
    if isinstance(exc, GuardError):
        return MSG["guard"]
    return MSG["plan"]


def dependency_failure(exc: Exception, kind: str) -> dict:
    return {
        "error": type(exc).__name__, "error_fatal": True, "error_kind": kind,
        "user_message": MSG["busy"] if isinstance(exc, Overloaded) else MSG["unavailable"],
    }


def fatal(kind: str, message_key: str, error: str | None = None) -> dict:
    return {"error": error or kind, "error_fatal": True, "error_kind": kind, "user_message": MSG[message_key]}
