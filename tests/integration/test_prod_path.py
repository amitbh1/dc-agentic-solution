"""The production wiring end to end: DC_MODE=prod with the ontology, checkpoints and feedback in
Postgres, rate limits and locks in Redis, and OIDC auth. Only Gemini (replaced by the rule-based
planner) and the warehouse (an in-memory SQLite registered as the 'databricks' engine) are stand-ins.

Needs DC_TEST_PG_DSN and DC_TEST_REDIS_URL (see the other integration tests).
"""
import os
import time
from pathlib import Path

import pytest

PG, REDIS = os.getenv("DC_TEST_PG_DSN"), os.getenv("DC_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not (PG and REDIS), reason="DC_TEST_PG_DSN and DC_TEST_REDIS_URL not set")
psycopg = pytest.importorskip("psycopg")
jwt = pytest.importorskip("jwt")
SCHEMA = (Path(__file__).parents[2] / "sql" / "ontology_schema.sql").read_text()


@pytest.fixture()
def prod_app(monkeypatch):
    import redis
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi.testclient import TestClient

    import dc_agent.llm as llm_mod
    import dc_agent.warehouse as wh
    from dc_agent.api import create_app
    from dc_agent.demo import build_sqlite
    from dc_agent.llm import RuleBasedLLM
    from dc_agent.security import auth
    from dc_agent.warehouse import SqliteExecutor

    with psycopg.connect(PG, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS ontology CASCADE; DROP SCHEMA IF EXISTS agent CASCADE")
        for t in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations"):
            conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        conn.execute(SCHEMA)
        conn.execute("""
            UPDATE ontology.concept SET physical_table = CASE name
                WHEN 'power_reading' THEN 'power_readings' WHEN 'rack' THEN 'racks' ELSE 'facilities' END;
            INSERT INTO ontology.role_grant (role, concept_id, max_sensitivity, row_scope_attribute_id, row_scope_claim)
            SELECT 'analyst', c.id, 'internal', a.id, 'regions'
              FROM ontology.concept c, ontology.attribute a JOIN ontology.concept ac ON ac.id = a.concept_id
             WHERE ac.name = 'facility' AND a.name = 'region';
        """)
    redis.Redis.from_url(REDIS).flushdb()

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class Jwks:
        def get_signing_key_from_jwt(self, _token):
            return type("K", (), {"key": key.public_key()})()

    class FakeGemini:
        def __init__(self, *_a, **_k):
            self._inner = RuleBasedLLM()

        def plan(self, req):
            return self._inner.plan(req)

    conn = build_sqlite()
    monkeypatch.setattr(auth, "_jwks_client", lambda url: Jwks())
    monkeypatch.setattr(llm_mod, "GeminiLLM", FakeGemini)
    monkeypatch.setattr(wh, "build_raw_executors", lambda: {"databricks": SqliteExecutor(conn, dialect="sqlite")})
    env = {
        "DC_MODE": "prod", "DC_PG_DSN": PG, "DC_REDIS_URL": REDIS, "GOOGLE_CLOUD_PROJECT": "test",
        "DC_AUTH_MODE": "oidc", "DC_OIDC_JWKS_URL": "https://idp/jwks", "DC_OIDC_AUDIENCE": "dc-agent",
        "DC_OIDC_ISSUER": "https://idp", "DC_TENANT_CLAIM": "tenant", "DC_EMBEDDINGS": "hashing",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    def token(sub="alice", roles=("analyst",), regions=("West",)):
        now = int(time.time())
        claims = {"sub": sub, "aud": "dc-agent", "iss": "https://idp", "iat": now, "exp": now + 300,
                  "roles": list(roles), "regions": list(regions), "tenant": "acme"}
        return {"Authorization": f"Bearer {jwt.encode(claims, key, algorithm='RS256')}"}

    with TestClient(create_app()) as client:
        yield client, token


def test_prod_stack_answers_scoped_questions_and_records_feedback(prod_app):
    client, token = prod_app
    assert client.get("/readyz").status_code == 200
    assert client.post("/ask", json={"question": "x"}, headers={"Authorization": "Bearer dev-admin"}).status_code == 401

    q = {"question": "What is the average power by region?", "session_id": "p1"}
    body = client.post("/ask", json=q, headers=token()).json()
    assert body["outcome"] == "ok" and {r[0] for r in body["rows"]} == {"West"} and body["release"] == 1
    assert "sql" not in body

    follow = client.post("/ask", json={"question": "now for tier III", "session_id": "p1"}, headers=token()).json()
    assert follow["outcome"] == "ok"  # follow-up context came back from the Postgres checkpointer

    assert client.post("/feedback", json={"session_id": "p1", "rating": "up"}, headers=token()).status_code == 200
    with psycopg.connect(PG) as conn:
        rows = conn.execute("SELECT user_id, tenant, rating, question FROM agent.feedback").fetchall()
    assert rows == [("alice", "acme", "up", "now for tier III")]

    denied = client.post("/ask", json={"question": "What is the average power by region?"},
                         headers=token(sub="bob", roles=("guest",))).json()
    assert denied["outcome"] == "clarification" and not denied["rows"]  # guest sees no catalogue
