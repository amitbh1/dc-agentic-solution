from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict


class PlanCache:
    """Caches validated plans (not results) keyed by ontology release, role, normalised question
    and, for follow-ups, the previous plan the question refines."""

    def __init__(self, ttl_s: int = 3600, max_items: int = 1000):
        self._ttl = ttl_s
        self._max = max_items
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(release_id: int, role: str, question: str, context: dict | None = None) -> tuple:
        ctx = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16] if context else ""
        return (release_id, role, " ".join(question.lower().split()), ctx)

    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            at, plan = item
            if time.monotonic() - at > self._ttl:
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return plan

    def set(self, key, plan: dict) -> None:
        with self._lock:
            self._d[key] = (time.monotonic(), plan)
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)
