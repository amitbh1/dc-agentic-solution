"""Integration tests against a real Redis. Skipped unless DC_TEST_REDIS_URL is set:

    docker run -d --rm -p 56379:6379 redis:7-alpine
    DC_TEST_REDIS_URL=redis://localhost:56379/0 pytest tests/integration
"""
import os
import uuid

import pytest

URL = os.getenv("DC_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not URL, reason="DC_TEST_REDIS_URL not set")
redis = pytest.importorskip("redis")


@pytest.fixture()
def client():
    r = redis.Redis.from_url(URL)
    r.flushdb()
    yield r
    r.flushdb()
    r.close()


def test_rate_limiter_is_shared_between_instances(client):
    from dc_agent.core.errors import RateLimited
    from dc_agent.resilience import RedisRateLimiter

    replica_a = RedisRateLimiter(client, per_min=1, burst=2, daily_budget=100)
    replica_b = RedisRateLimiter(client, per_min=1, burst=2, daily_budget=100)
    replica_a.check("u")
    replica_b.check("u")
    with pytest.raises(RateLimited):
        replica_a.check("u")


def test_lock_is_shared_and_expires(client):
    import time

    from dc_agent.resilience import RedisKeyedLock

    a, b = RedisKeyedLock(client, ttl_s=0.3), RedisKeyedLock(client, ttl_s=0.3)
    key = uuid.uuid4().hex
    assert a.try_acquire(key) and not b.try_acquire(key)
    time.sleep(0.4)
    assert b.try_acquire(key)
    a.release(key)  # a's token expired; must not release b's lock
    assert not a.try_acquire(key)


def test_api_uses_redis_backends(client, monkeypatch):
    from fastapi.testclient import TestClient

    from dc_agent.api import create_app

    monkeypatch.setenv("DC_MODE", "demo")
    monkeypatch.setenv("DC_REDIS_URL", URL)
    monkeypatch.setenv("DC_RATE_BURST", "2")
    headers = {"Authorization": "Bearer dev-analyst"}
    q = {"question": "What is the average power by region?"}
    with TestClient(create_app()) as c1, TestClient(create_app()) as c2:  # two "replicas"
        assert c1.post("/ask", json=q, headers=headers).status_code == 200
        assert c2.post("/ask", json=q, headers=headers).status_code == 200
        assert c1.post("/ask", json=q, headers=headers).status_code == 429
    assert any(k.startswith(b"dcagent:plan:") for k in client.keys("dcagent:*"))
