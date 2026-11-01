import pytest

from dc_agent.core.errors import CircuitOpen, Overloaded, RateLimited
from dc_agent.resilience import RateLimiter
from dc_agent.resilience import Bulkhead, CircuitBreaker, KeyedLock, is_retryable_error, retry


class Boom(Exception):
    pass


def test_retry_succeeds_after_transient_failures():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("down")
        return "ok"

    assert retry(flaky, attempts=3, sleep=lambda s: None) == "ok"
    assert len(calls) == 3


def test_retry_does_not_retry_non_transient_or_timeouts():
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        retry(bad, attempts=3, sleep=lambda s: None)
    assert len(calls) == 1
    assert not is_retryable_error(TimeoutError())
    assert is_retryable_error(ConnectionError())


def test_breaker_opens_then_half_opens():
    t = [0.0]
    b = CircuitBreaker("t", failure_threshold=2, reset_s=10, now=lambda: t[0])

    def fail():
        raise ConnectionError("x")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            b.call(fail)
    with pytest.raises(CircuitOpen):
        b.call(lambda: "never")
    t[0] = 11.0
    assert b.call(lambda: "probe") == "probe"
    assert b.call(lambda: "closed") == "closed"


def test_breaker_ignores_non_transient_errors():
    b = CircuitBreaker("t", failure_threshold=1, reset_s=10)
    for _ in range(3):
        with pytest.raises(Boom):
            b.call(lambda: (_ for _ in ()).throw(Boom()))
    assert b.call(lambda: "still closed") == "still closed"


def test_bulkhead_rejects_when_full():
    bh = Bulkhead("t", limit=1, wait_s=0.01)
    with bh.slot():
        with pytest.raises(Overloaded):
            with bh.slot():
                pass
    with bh.slot():
        pass


def test_keyed_lock_blocks_then_expires():
    t = [0.0]
    lock = KeyedLock(ttl_s=5, now=lambda: t[0])
    assert lock.try_acquire("a")
    assert not lock.try_acquire("a")
    t[0] = 6.0
    assert lock.try_acquire("a")
    lock.release("a")
    assert lock.try_acquire("a")


def test_rate_limiter_burst_refill_and_daily_budget():
    t = [0.0]
    rl = RateLimiter(per_min=60, burst=2, daily_budget=3, now=lambda: t[0], wall=lambda: 1_000_000.0)
    rl.check("u")
    rl.check("u")
    with pytest.raises(RateLimited) as exc:
        rl.check("u")
    assert exc.value.reason == "rate limit"
    t[0] = 1.0
    rl.check("u")
    t[0] = 10.0
    with pytest.raises(RateLimited) as exc:
        rl.check("u")
    assert exc.value.reason == "daily budget exhausted"
    rl.check("someone-else")
