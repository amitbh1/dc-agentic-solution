from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
from typing import Callable

from ..core.errors import AccessError
from ..observability import audit
from ..observability.metrics import TOOL_CALLS
from ..security.identity import UserContext

MAX_NOTE_CHARS = 500


@dataclass
class ToolSpec:
    """A role-scoped tool. Enrichment tools get a small, row-free context and return a short note.

    fn(question=..., metrics=[...], dimensions=[...], columns=[...], row_count=int) -> str | None
    """

    name: str
    fn: Callable
    roles: frozenset
    timeout_s: float = 5.0
    enrich: bool = True


class ToolRegistry:
    def __init__(self, max_workers: int = 4) -> None:
        self._tools: dict = {}
        self._pool = cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")

    def __bool__(self) -> bool:
        return bool(self._tools)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def for_role(self, role: str) -> list:
        return [s for s in self._tools.values() if role in s.roles]

    def call(self, name: str, user: UserContext, **kwargs):
        spec = self._tools.get(name)
        if spec is None or user.role not in spec.roles:
            raise AccessError(f"tool not available: {name}")
        audit("tool_call", tool=name, user=user.user_id, role=user.role)
        return spec.fn(**kwargs)

    def enrich(self, user: UserContext, context: dict) -> list:
        """Run every enrichment tool the role may use, in parallel and time-bounded. Never raises.

        Tool output is untrusted text: it is shown to the user as a note and never sent to the planner.
        """
        specs = [s for s in self.for_role(user.role) if s.enrich]
        futures = {s.name: (s, self._pool.submit(self.call, s.name, user, **context)) for s in specs}
        notes = []
        for name, (spec, fut) in futures.items():
            try:
                out = fut.result(timeout=spec.timeout_s)
                TOOL_CALLS.labels(name, "ok").inc()
                if out:
                    notes.append({"tool": name, "note": str(out)[:MAX_NOTE_CHARS]})
            except cf.TimeoutError:
                fut.cancel()
                TOOL_CALLS.labels(name, "timeout").inc()
                audit("tool_timeout", tool=name, user=user.user_id)
            except Exception as exc:
                TOOL_CALLS.labels(name, "error").inc()
                audit("tool_failed", tool=name, user=user.user_id, error=repr(exc)[:200])
        return notes

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
