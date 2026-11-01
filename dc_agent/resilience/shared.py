"""Redis-backed rate limiter and session lock, shared by all replicas.

Each has a fallback wrapper: if Redis is unreachable the process degrades to its in-memory
equivalent (limits become per replica) instead of failing every request. A circuit breaker stops
it from waiting on a dead Redis on every call.
"""
from __future__ import annotations

import logging
import math
import secrets
import threading
import time
from typing import Callable

from ..core.errors import RateLimited
from ..observability.metrics import BACKEND_FALLBACK
from .breaker import CircuitBreaker
from .ratelimit import seconds_to_utc_midnight

log = logging.getLogger("dc_agent.resilience")
PREFIX = "dcagent"

# Token bucket + daily counters, atomically. Returns {code, retry_after_s}:
# 0 ok, 1 daily budget, 2 rate limit, 3 tenant budget.
_RATE_LUA = """
local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local burst = tonumber(ARGV[3])
local budget = tonumber(ARGV[4])
local tenant_budget = tonumber(ARGV[5])
local to_midnight = tonumber(ARGV[6])
local b = redis.call('HMGET', KEYS[1], 't', 'ts')
local tokens = tonumber(b[1]) or burst
local ts = tonumber(b[2]) or now
tokens = math.min(burst, tokens + (now - ts) * rate)
local used = tonumber(redis.call('GET', KEYS[2]) or '0')
if budget > 0 and used >= budget then return {1, to_midnight} end
if KEYS[3] ~= '' and tenant_budget > 0 then
  local t_used = tonumber(redis.call('GET', KEYS[3]) or '0')
  if t_used >= tenant_budget then return {3, to_midnight} end
end
if tokens < 1 then
  if rate <= 0 then return {2, 60} end
  return {2, math.ceil((1 - tokens) / rate)}
end
redis.call('HSET', KEYS[1], 't', tokens - 1, 'ts', now)
redis.call('EXPIRE', KEYS[1], math.ceil(burst / math.max(rate, 0.0001)) + 60)
redis.call('INCR', KEYS[2])
redis.call('EXPIRE', KEYS[2], to_midnight + 3600)
if KEYS[3] ~= '' then
  redis.call('INCR', KEYS[3])
  redis.call('EXPIRE', KEYS[3], to_midnight + 3600)
end
return {0, 0}
"""
_REASONS = {1: "daily budget exhausted", 2: "rate limit", 3: "tenant budget exhausted"}

_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""


class RedisRateLimiter:
    def __init__(self, client, per_min: int = 30, burst: int = 10, daily_budget: int = 500,
                 tenant_budget: int = 0, wall: Callable = time.time):
        self._r = client
        self._rate = per_min / 60.0
        self._burst = burst
        self._budget = daily_budget
        self._tenant_budget = tenant_budget
        self._wall = wall
        self._script = client.register_script(_RATE_LUA)

    def check(self, user_id: str, tenant: str | None = None) -> None:
        wall = self._wall()
        day = time.strftime("%Y-%m-%d", time.gmtime(wall))
        keys = [f"{PREFIX}:rl:bucket:{user_id}", f"{PREFIX}:rl:day:{day}:{user_id}",
                f"{PREFIX}:rl:tenant:{day}:{tenant}" if tenant else ""]
        to_midnight = math.ceil(seconds_to_utc_midnight(wall))
        code, retry_after = self._script(
            keys=keys, args=[wall, self._rate, self._burst, self._budget, self._tenant_budget, to_midnight]
        )
        code = int(code)
        if code:
            raise RateLimited(float(retry_after), _REASONS[code])


class RedisKeyedLock:
    """One in-flight request per key across replicas. SET NX with expiry; release only our own token."""

    def __init__(self, client, ttl_s: float = 120.0):
        self._r = client
        self._ttl_ms = int(ttl_s * 1000)
        self._tokens: dict = {}
        self._mu = threading.Lock()
        self._release = client.register_script(_RELEASE_LUA)

    def try_acquire(self, key: str) -> bool:
        token = secrets.token_hex(8)
        if not self._r.set(f"{PREFIX}:lock:{key}", token, nx=True, px=self._ttl_ms):
            return False
        with self._mu:
            self._tokens[key] = token
        return True

    def release(self, key: str) -> None:
        with self._mu:
            token = self._tokens.pop(key, None)
        if token:
            self._release(keys=[f"{PREFIX}:lock:{key}"], args=[token])


class _Fallback:
    def __init__(self, name: str, primary, fallback, breaker: CircuitBreaker | None = None):
        self._name = name
        self._primary = primary
        self.fallback = fallback
        self._breaker = breaker or CircuitBreaker(f"redis-{name}", failure_threshold=3, reset_s=15)

    def _run(self, call_primary, call_fallback, passthrough=()):
        try:
            return self._breaker.call(call_primary, count_if=lambda exc: not isinstance(exc, passthrough))
        except passthrough:
            raise
        except Exception:
            BACKEND_FALLBACK.labels(self._name).inc()
            log.warning("shared backend unavailable; using local fallback", exc_info=True,
                        extra={"fields": {"backend": self._name}})
            return call_fallback()


class FallbackRateLimiter(_Fallback):
    def __init__(self, primary, fallback, breaker=None):
        super().__init__("ratelimit", primary, fallback, breaker)

    def check(self, user_id: str, tenant: str | None = None) -> None:
        self._run(lambda: self._primary.check(user_id, tenant), lambda: self.fallback.check(user_id, tenant),
                  passthrough=(RateLimited,))


class FallbackKeyedLock(_Fallback):
    """Remembers which backend granted each lock so release goes to the same place."""

    def __init__(self, primary, fallback, breaker=None):
        super().__init__("lock", primary, fallback, breaker)
        self._where: dict = {}
        self._mu = threading.Lock()

    def try_acquire(self, key: str) -> bool:
        used = {"backend": "primary"}

        def local():
            used["backend"] = "fallback"
            return self.fallback.try_acquire(key)

        ok = self._run(lambda: self._primary.try_acquire(key), local)
        if ok:
            with self._mu:
                self._where[key] = used["backend"]
        return ok

    def release(self, key: str) -> None:
        with self._mu:
            where = self._where.pop(key, "primary")
        if where == "fallback":
            self.fallback.release(key)
            return
        try:
            self._primary.release(key)
        except Exception:
            log.warning("lock release failed; it will expire", exc_info=True)
