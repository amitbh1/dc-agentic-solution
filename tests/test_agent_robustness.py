import pytest
from langgraph.checkpoint.memory import MemorySaver

from dc_agent.agent import MSG, Deps, build_graph, outcome_of
from dc_agent.agent import nodes
from dc_agent.core.config import Settings
from dc_agent.demo import DEV_USERS, build_ontology, build_sqlite
from dc_agent.llm import PlanRequest, RuleBasedLLM, build_prompt
from dc_agent.memory import PlanCache
from dc_agent.query import PlannerOutput, QueryPlan
from dc_agent.semantic import OntologyProvider, is_follow_up
from dc_agent.warehouse import SqliteExecutor

ADMIN = DEV_USERS["dev-admin"]


def make_agent(llm=None, provider=None, **settings):
    onto = build_ontology()
    deps = Deps(
        provider=provider or OntologyProvider(lambda: onto, ttl_s=10**9),
        llm=llm or RuleBasedLLM(),
        executors={"sqlite": SqliteExecutor(build_sqlite())},
        cache=PlanCache(),
        settings=Settings(mode="demo", **settings),
    )
    return build_graph(deps, MemorySaver())


def ask(agent, question, thread="t1", user=ADMIN):
    return agent.invoke(
        {"question": question, "user": user.to_dict()},
        config={"configurable": {"thread_id": thread}, "recursion_limit": 25},
    )


def regions(state):
    i = state["columns"].index("facility__region")
    return {r[i] for r in state["rows"]}


def test_follow_up_refines_previous_plan():
    agent = make_agent()
    assert regions(ask(agent, "What is the average power by region?")) == {"West", "South", "North"}
    follow = ask(agent, "now for tier III")
    assert outcome_of(follow) == "ok"
    assert regions(follow) == {"West", "South"}


def test_self_contained_question_does_not_inherit_filters():
    agent = make_agent()
    ask(agent, "What is the average power by region for tier III?")
    assert regions(ask(agent, "What is the average power by region?")) == {"West", "South", "North"}


def test_follow_up_without_history_asks_for_clarification():
    s = ask(make_agent(), "now for tier III", thread="fresh")
    assert outcome_of(s) == "clarification"


def test_unexpected_node_exception_degrades_gracefully():
    def broken():
        raise RuntimeError("ontology store exploded")

    s = ask(make_agent(provider=OntologyProvider(broken)), "What is the average power by region?")
    assert s["answer"] == MSG["internal"]
    assert s["error_kind"] == "internal" and outcome_of(s) == "error"


class MalformedOnce:
    def __init__(self):
        self.calls = 0

    def plan(self, req):
        self.calls += 1
        if self.calls == 1:
            return PlannerOutput.model_validate_json("{not json")
        return RuleBasedLLM().plan(PlanRequest(req.question, req.catalogue))


def test_malformed_llm_output_is_repaired():
    llm = MalformedOnce()
    s = ask(make_agent(llm=llm), "What is the average power by region?")
    assert outcome_of(s) == "ok" and llm.calls == 2


def test_malformed_output_gives_up_after_max_repairs():
    class AlwaysMalformed:
        calls = 0

        def plan(self, req):
            self.calls += 1
            return PlannerOutput.model_validate_json("[]")

    llm = AlwaysMalformed()
    s = ask(make_agent(llm=llm, max_repairs=1), "What is the average power by region?")
    assert outcome_of(s) == "error" and llm.calls == 2
    assert s["answer"] == MSG["plan"]


def test_turn_deadline_stops_the_repair_loop(monkeypatch):
    clock = [1_000.0]
    monkeypatch.setattr(nodes.time, "time", lambda: clock[0])

    class SlowAndWrong:
        calls = 0

        def plan(self, req):
            self.calls += 1
            clock[0] += 120  # each call burns more than the whole budget
            return PlannerOutput(plan=QueryPlan(metrics=["no_such_metric"]))

    llm = SlowAndWrong()
    s = ask(make_agent(llm=llm, request_timeout_s=60), "What is the average power by region?")
    assert s["answer"] == MSG["timeout"] and llm.calls == 1


def test_sqlite_executor_enforces_timeout():
    ex = SqliteExecutor(build_sqlite())
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    with pytest.raises(TimeoutError):
        ex.run(endless, 10, 0)
    assert ex.run("SELECT count(*) FROM racks", 10, 5)[1] == [[12]]


def test_plan_cache_key_includes_follow_up_context():
    base = PlanCache.key(1, "admin", "now for tier III")
    assert PlanCache.key(1, "admin", "Now  for tier III") == base
    assert PlanCache.key(1, "admin", "now for tier III", {"metrics": ["a"]}) != base


def test_repair_prompt_shows_the_failed_plan():
    req = PlanRequest("q", {"metrics": []}, error="unknown metric: x", failed_plan={"metrics": ["x"]})
    prompt = build_prompt(req)
    assert '"metrics": ["x"]' in prompt and "unknown metric: x" in prompt
    assert "Previous plan in this conversation" not in prompt


@pytest.mark.parametrize("q,expected", [
    ("now for tier III", True), ("What about the West?", True), ("and by row", True),
    ("What is the average power by region?", False), ("Show tier III racks", False),
])
def test_is_follow_up(q, expected):
    assert is_follow_up(q) is expected
