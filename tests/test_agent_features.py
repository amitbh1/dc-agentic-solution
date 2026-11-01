import time
from dataclasses import replace

import pytest
from langgraph.types import Command

from dc_agent.agent import MSG, outcome_of
from dc_agent.bootstrap import build_agent
from dc_agent.core.config import Settings
from dc_agent.demo import build_ontology, build_sqlite
from dc_agent.llm import RuleBasedLLM
from dc_agent.semantic import Ontology
from dc_agent.tools import ToolRegistry, ToolSpec
from dc_agent.warehouse import SqliteExecutor


def bundle(**settings):
    return build_agent(Settings(mode="demo", **settings))


def ask(b, token, question, thread="t", payload=None):
    user = b.dev_users[token]
    return b.agent.invoke(
        payload if payload is not None else {"question": question, "user": user.to_dict()},
        config={"configurable": {"thread_id": f"{user.user_id}:{thread}"}, "recursion_limit": 25},
    )


def as_dict(state, key_col=0, val_col=1):
    return {r[key_col]: round(r[val_col], 6) for r in state["rows"]}


def test_mixed_grain_question_end_to_end():
    s = ask(bundle(), "dev-analyst", "What is the average power and rack count by region?")
    assert outcome_of(s) == "ok"
    assert s["columns"] == ["facility__region", "avg_power_kw", "rack_count"]
    assert {r[0]: r[2] for r in s["rows"]} == {"West": 4, "South": 4}


def test_trend_over_a_relative_window():
    s = ask(bundle(), "dev-admin", "Average power by hour for the last 6 hours")
    assert outcome_of(s) == "ok" and s["columns"] == ["power_reading__ts__hour", "avg_power_kw"]
    buckets = [r[0] for r in s["rows"]]
    assert 5 <= len(buckets) <= 6 and buckets == sorted(buckets)


def test_multiple_metrics_same_grain_keep_question_order():
    s = ask(bundle(), "dev-analyst", "Average and peak power by row")
    assert s["columns"] == ["rack__row", "avg_power_kw", "max_power_kw"]


def test_top_n():
    s = ask(bundle(), "dev-admin", "Peak power per facility, top 2")
    assert s["columns"] == ["facility__name", "max_power_kw"] and len(s["rows"]) == 2
    assert s["rows"][0][1] >= s["rows"][1][1]


def test_semantic_grounding_handles_misspellings_and_falls_back_to_lexical():
    s = ask(bundle(), "dev-analyst", "averge powr by region")
    assert outcome_of(s) == "ok" and s["plan"]["metrics"] == ["avg_power_kw"]
    lexical_only = ask(bundle(embeddings="off"), "dev-analyst", "averge powr by region")
    assert outcome_of(lexical_only) == "clarification"


def test_embedding_failure_degrades_to_lexical():
    b = bundle()

    class Broken:
        min_similarity = 0.5

        def embed(self, *_a, **_k):
            raise ConnectionError("embedding service down")

    b.deps.embedder = Broken()
    assert outcome_of(ask(b, "dev-analyst", "What is the average power by region?")) == "ok"


def test_verified_examples_reach_the_planner():
    b = bundle()
    seen = []

    class Spy(RuleBasedLLM):
        def plan(self, req):
            seen.append(req)
            return super().plan(req)

    b.deps.llm = Spy()
    ask(b, "dev-analyst", "peak power per row in tier III")
    assert seen and seen[0].examples[0]["question"] == "peak power per row in tier III"


def test_follow_up_keeps_time_window():
    b = bundle()
    first = ask(b, "dev-admin", "Average power by region for the last 6 hours", "fu")
    follow = ask(b, "dev-admin", "now for tier III", "fu")
    assert outcome_of(first) == outcome_of(follow) == "ok"
    assert follow["plan"]["time_range"]["last"] == 6
    assert set(as_dict(follow)) == {"West", "South"}


def test_cross_engine_question_end_to_end():
    conn = build_sqlite()
    onto = build_ontology()
    split = Ontology([replace(c, source="sqlite_b") if c.name == "facility" else c for c in onto.concepts.values()],
                     onto.edges, list(onto.metrics.values()), onto.synonyms, list(onto.grants.values()), 1, onto.examples)
    b = build_agent(Settings(mode="demo"), ontology=split,
                    executors={"sqlite": SqliteExecutor(conn), "sqlite_b": SqliteExecutor(conn, dialect="sqlite")})
    single = build_agent(Settings(mode="demo"), executors={"sqlite": SqliteExecutor(conn)})
    q = "What is the average power by region?"
    s, ref = ask(b, "dev-analyst", q), ask(single, "dev-analyst", q)
    assert outcome_of(s) == "ok" and len(s["parts"]) == 2
    assert as_dict(s) == as_dict(ref)
    assert "Combined from several data platforms." in s["warnings"]


# --- cost gate and human approval --------------------------------------------------------

def test_expensive_query_waits_for_approval_then_runs():
    b = bundle(approval_scan_bytes=1_000)
    paused = ask(b, "dev-admin", "What is the average power by region?", "ap")
    info = paused["__interrupt__"][0].value
    assert info["kind"] == "approval" and info["estimated_bytes"] > 1_000 and "Do you want to run it" in info["message"]
    assert not paused.get("rows")
    done = ask(b, "dev-admin", None, "ap", payload=Command(resume={"approve": True}))
    assert outcome_of(done) == "ok" and len(done["rows"]) == 3


def test_rejected_approval_does_not_run():
    b = bundle(approval_scan_bytes=1_000)
    ask(b, "dev-admin", "What is the average power by region?", "rj")
    done = ask(b, "dev-admin", None, "rj", payload=Command(resume={"approve": False}))
    assert outcome_of(done) == "cancelled" and done["answer"] == MSG["cancelled"] and not done["rows"]


def test_new_question_discards_pending_approval():
    b = bundle(approval_scan_bytes=1_000)
    ask(b, "dev-admin", "What is the average power by region?", "nq")
    s = ask(b, "dev-admin", "Show revenue by customer", "nq")
    assert outcome_of(s) == "clarification" and not s.get("__interrupt__")


def test_query_over_hard_cap_is_refused():
    s = ask(bundle(max_scan_bytes=1_000), "dev-admin", "What is the average power by region?")
    assert s["answer"] == MSG["too_expensive"] and outcome_of(s) == "error"


def test_cheap_query_runs_without_approval():
    s = ask(bundle(approval_scan_bytes=10**9), "dev-admin", "What is the average power by region?")
    assert outcome_of(s) == "ok" and s["estimate_bytes"] > 0


# --- enrichment tools --------------------------------------------------------------------

def test_enrich_adds_role_scoped_notes_without_rows():
    reg = ToolRegistry()
    calls = []

    def runbook(question, metrics, dimensions, columns, row_count):
        calls.append(row_count)
        return f"Runbook for {metrics[0]}"

    reg.register(ToolSpec("runbook", runbook, frozenset({"admin"})))
    reg.register(ToolSpec("slow", lambda **_: time.sleep(1), frozenset({"admin"}), timeout_s=0.05))
    b = build_agent(Settings(mode="demo"), tools=reg)
    admin = ask(b, "dev-admin", "What is the average power by region?")
    assert admin["notes"] == [{"tool": "runbook", "note": "Runbook for avg_power_kw"}] and calls == [3]
    analyst = ask(b, "dev-analyst", "What is the average power by region?")
    assert analyst["notes"] == []
    b.close()


@pytest.mark.parametrize("q", ["What is the average power by region?", "How many racks are in each region?"])
def test_results_identical_with_and_without_tools_graph(q):
    with_tools = build_agent(Settings(mode="demo"), tools=ToolRegistry())
    assert as_dict(ask(with_tools, "dev-admin", q)) == as_dict(ask(bundle(), "dev-admin", q))
