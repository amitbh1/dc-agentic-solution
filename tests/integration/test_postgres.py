"""Integration tests against a real Postgres with pgvector. Skipped unless DC_TEST_PG_DSN is set:

    docker run -d --rm -e POSTGRES_PASSWORD=test -e POSTGRES_DB=dcagent -p 55432:5432 pgvector/pgvector:pg16
    DC_TEST_PG_DSN=postgresql://postgres:test@localhost:55432/dcagent pytest tests/integration

Each test starts from a freshly applied sql/ontology_schema.sql.
"""
import datetime as dt
import json
import os
from pathlib import Path

import pytest

DSN = os.getenv("DC_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="DC_TEST_PG_DSN not set")
psycopg = pytest.importorskip("psycopg")
SCHEMA = (Path(__file__).parents[2] / "sql" / "ontology_schema.sql").read_text()


@pytest.fixture()
def db():
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS ontology CASCADE; DROP SCHEMA IF EXISTS agent CASCADE")
        for t in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations"):
            conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        conn.execute(SCHEMA)
        conn.execute("""
            INSERT INTO ontology.metric (id, concept_id, attribute_id, agg, description, unit)
            SELECT 'max_power_kw', a.concept_id, a.id, 'max', 'Peak rack power draw', 'kW'
              FROM ontology.attribute a JOIN ontology.concept c ON c.id = a.concept_id
             WHERE c.name = 'power_reading' AND a.name = 'power_kw';
            INSERT INTO ontology.synonym (term, metric_id) VALUES ('peak power', 'max_power_kw');
            INSERT INTO ontology.role_grant (role, concept_id, max_sensitivity, row_scope_attribute_id, row_scope_claim)
            SELECT 'analyst', c.id, 'internal', a.id, 'regions'
              FROM ontology.concept c, ontology.attribute a JOIN ontology.concept ac ON ac.id = a.concept_id
             WHERE ac.name = 'facility' AND a.name = 'region';
            UPDATE ontology.relationship SET priority = 7
             WHERE left_concept_id = (SELECT id FROM ontology.concept WHERE name = 'rack');
        """)
        yield conn


def test_loader_reads_schema_and_seed(db):
    from dc_agent.semantic import load_from_postgres, validate_ontology

    onto = load_from_postgres(DSN)
    assert onto.release_id == 1
    assert set(onto.concepts) == {"power_reading", "rack", "facility"}
    assert set(onto.metrics) == {"avg_power_kw", "max_power_kw"}
    assert onto.grants[("analyst", "rack")].scope_ref == "facility.region"
    assert {e.priority for e in onto.edges} == {7, 100}
    assert {s.term for s in onto.synonyms} >= {"average power", "peak power", "geography"}
    assert not onto.has_vectors and onto.examples == []
    assert validate_ontology(onto) == []


def test_embed_job_and_verified_queries_feed_grounding(db):
    from dc_agent.jobs import embed
    from dc_agent.semantic import HashingEmbedder, load_from_postgres

    good = {"metrics": ["max_power_kw"], "dimensions": ["rack.row"]}
    stale = {"metrics": ["retired_metric"]}
    for q, p in (("peak power per row", good), ("old question", stale)):
        db.execute("INSERT INTO ontology.verified_query (question, plan, sql, dialect, release_id)"
                   " VALUES (%s, %s, 'SELECT 1', 'databricks', 1)", (q, json.dumps(p)))
    counts = embed.run(DSN, HashingEmbedder(768))
    assert counts == {"ontology.synonym": 4, "ontology.verified_query": 2}
    assert embed.run(DSN, HashingEmbedder(768)) == {"ontology.synonym": 0, "ontology.verified_query": 0}

    onto = load_from_postgres(DSN)
    assert onto.has_vectors
    assert [e.question for e in onto.examples] == ["peak power per row"]  # retired metric dropped
    assert len(onto.examples[0].vector) == 768


def test_drift_record_marks_concept_stale_and_loader_excludes_it(db):
    from dc_agent.jobs.drift import Finding, record
    from dc_agent.semantic import load_from_postgres

    finding = Finding("rack", "type_changed", "row", {"expected": "string", "actual": "bigint"})
    record(DSN, [finding])
    record(DSN, [finding])  # unresolved duplicate is not inserted twice
    assert db.execute("SELECT count(*) FROM ontology.drift_event").fetchone()[0] == 1
    assert db.execute("SELECT status FROM ontology.concept WHERE name = 'rack'").fetchone()[0] == "stale"
    onto = load_from_postgres(DSN)
    assert "rack" not in onto.concepts and not onto.edges


def test_feedback_store_and_promotion(db):
    from psycopg_pool import ConnectionPool

    from dc_agent.jobs import promote
    from dc_agent.memory import FeedbackRecord, PostgresFeedbackStore

    plan = {"metrics": ["avg_power_kw"], "dimensions": ["facility.region"]}
    with ConnectionPool(DSN, open=True) as pool:
        store = PostgresFeedbackStore(pool)
        for user in ("a", "b", "c"):
            store.add(FeedbackRecord(user, "analyst", f"{user}:s", "up", "h1", "ok", question="Average power by region",
                                     plan=plan, sql="SELECT 1", dialect="databricks", release_id=1))
        store.add(FeedbackRecord("a", "analyst", "a:s", "up", "h2", "ok", question="disputed", plan=plan,
                                 sql="SELECT 2", dialect="databricks", release_id=1))
        store.add(FeedbackRecord("b", "analyst", "b:s", "down", "h2", "ok", plan=plan, sql="SELECT 2",
                                 dialect="databricks", release_id=1, comment="wrong"))
    assert [c["question"] for c in promote.run(DSN, min_up=3, dry_run=True)] == ["Average power by region"]
    assert db.execute("SELECT count(*) FROM ontology.verified_query").fetchone()[0] == 0
    promote.run(DSN, min_up=3)
    row = db.execute("SELECT question, plan, verified_by FROM ontology.verified_query").fetchall()
    assert row == [("Average power by region", plan, "auto:feedback")]
    assert promote.run(DSN, min_up=3) == []  # already promoted
    assert promote.run(DSN, min_up=1, dry_run=True) == []  # 'disputed' has a down vote


def test_postgres_checkpointer_and_retention(db):
    from dc_agent.bootstrap import build_agent
    from dc_agent.core.config import Settings
    from dc_agent.jobs import retention
    from dc_agent.memory import make_postgres_checkpointer

    saver, pool = make_postgres_checkpointer(DSN)
    try:
        b = build_agent(Settings(mode="demo"), checkpointer=saver)
        user = b.dev_users["dev-admin"]
        cfg = {"configurable": {"thread_id": f"{user.user_id}:ret"}}
        b.agent.invoke({"question": "What is the average power by region?", "user": user.to_dict()}, config=cfg)
        follow = b.agent.invoke({"question": "now for tier III", "user": user.to_dict()}, config=cfg)
        assert {r[0] for r in follow["rows"]} == {"West", "South"}  # last_plan survived in Postgres
        db.execute("INSERT INTO agent.feedback (user_id, role, thread_id, rating, q_hash, outcome, created_at)"
                   " VALUES ('u', 'r', 't', 'down', 'h', 'ok', now() - interval '400 days')")

        assert retention.run(DSN, 30, 365, dry_run=True) == {"threads": 0, "feedback": 1, "dry_run": True}
        later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=31)
        assert retention.run(DSN, 30, 365, now=later)["threads"] == 1
        assert saver.get_tuple(cfg) is None
        assert db.execute("SELECT count(*) FROM agent.feedback").fetchone()[0] == 0
    finally:
        pool.close()
