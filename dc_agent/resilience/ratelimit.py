from __future__ import annotations

import math
import threading
import time
from typing import Callable

from ..core.errors import RateLimited


def seconds_to_utc_midnight(wall: float) -> float:
    return 86400 - (wall % 86400)


class RateLimiter:
    """Per-user token bucket, a per-user daily query budget and an optional per-tenant daily budget.

    In-memory: correct for one process. With several replicas set DC_REDIS_URL to use
    RedisRateLimiter (same check() contract), or enforce limits at the API gateway.
    """

    def __init__(self, per_min: int = 30, burst: int = 10, daily_budget: int = 500, tenant_budget: int = 0,
                 now: Callable = time.monotonic, wall: Callable = time.time):
        self._rate = per_min / 60.0
        self._burst = float(burst)
        self._budget = daily_budget
        self._tenant_budget = tenant_budget
        self._now = now
        self._wall = wall
        self._buckets: dict = {}
        self._daily: dict = {}
        self._tenants: dict = {}
        self._lock = threading.Lock()

    def check(self, user_id: str, tenant: str | None = None) -> None:
        with self._lock:
            now = self._now()
            tokens, last = self._buckets.get(user_id, (self._burst, now))
            tokens = min(self._burst, tokens + (now - last) * self._rate)
            wall = self._wall()
            day = time.strftime("%Y-%m-%d", time.gmtime(wall))
            d_day, d_count = self._daily.get(user_id, (day, 0))
            if d_day != day:
                d_day, d_count = day, 0
            if self._budget and d_count >= self._budget:
                raise RateLimited(seconds_to_utc_midnight(wall), "daily budget exhausted")
            t_day, t_count = self._tenants.get(tenant, (day, 0)) if tenant else (day, 0)
            if t_day != day:
                t_day, t_count = day, 0
            if tenant and self._tenant_budget and t_count >= self._tenant_budget:
                raise RateLimited(seconds_to_utc_midnight(wall), "tenant budget exhausted")
            if tokens < 1.0:
                raise RateLimited(math.ceil((1.0 - tokens) / self._rate) if self._rate else 60, "rate limit")
            self._buckets[user_id] = (tokens - 1.0, now)
            self._daily[user_id] = (d_day, d_count + 1)
            if tenant:
                self._tenants[tenant] = (t_day, t_count + 1)
            if len(self._buckets) > 10_000:
                self._buckets = {u: v for u, v in self._buckets.items() if now - v[1] < 3600}
