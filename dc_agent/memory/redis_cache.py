from __future__ import annotations

import hashlib
import json
import logging

from ..observability.metrics import BACKEND_FALLBACK

log = logging.getLogger("dc_agent.memory")


class RedisPlanCache:
    """Plan cache shared by all replicas. Same get/set contract as PlanCache; never raises.

    Plans hold no claim values or rows, so sharing them across users of one role is safe.
    """

    def __init__(self, client, ttl_s: int = 3600, prefix: str = "dcagent:plan:"):
        self._r = client
        self._ttl = ttl_s
        self._prefix = prefix

    def _k(self, key: tuple) -> str:
        return self._prefix + hashlib.sha256(json.dumps(list(key)).encode()).hexdigest()

    def get(self, key):
        try:
            raw = self._r.get(self._k(key))
        except Exception:
            BACKEND_FALLBACK.labels("plan_cache").inc()
            log.warning("plan cache read failed", exc_info=True)
            return None
        return json.loads(raw) if raw else None

    def set(self, key, plan: dict) -> None:
        try:
            self._r.set(self._k(key), json.dumps(plan), ex=self._ttl)
        except Exception:
            BACKEND_FALLBACK.labels("plan_cache").inc()
            log.warning("plan cache write failed", exc_info=True)
