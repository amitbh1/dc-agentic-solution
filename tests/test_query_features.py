import datetime as dt
from dataclasses import replace

import pytest

from dc_agent.core.errors import AccessError, CrossEngineError, GuardError, PlanError
from dc_agent.demo import DEV_USERS, build_ontology, build_sqlite
from dc_agent.query import MergeTooLarge, QueryPlan, compile_plan, guard, merge_parts
from dc_agent.query.plan import TimeRange
from dc_agent.query.timeutil import resolve_range
from dc_agent.security import UserContext
from dc_agent.semantic import Concept, Edge, Grant, Metric, Ontology
from dc_agent.semantic.models import Attribute

NOW = dt.datetime(2026, 11, 1, 12, tzinfo=dt.timezone.utc)
ONTO = build_ontology()
CONN = build_sqlite(now=NOW)
ADMIN, ANALYST = DEV_USERS["dev-admin"], DEV_USERS["dev-analyst"]


def run(plan, user=ADMIN, onto=ONTO):
    c = compile_plan(QueryPlan(**plan), onto, user, now=NOW)
    sql = guard(c.sql, c.dialect, c.tables)
    cur = CONN.execute(sql)
    return [d[0] for d in cur.description], cur.fetchall(), sql


def test_mixed_grain_metrics_are_each_computed_at_their_own_grain():
    cols, rows, sql = run({"metrics": ["avg_power_kw", "rack_count"], "dimensions": ["facility.region"]}, ANALYST)
    assert cols == ["facility__region", "avg_power_kw", "rack_count"]
    assert {r[0]: r[2] for r in rows} == {"West": 4, "South": 4}  # not 4 x 24 readings
    assert "WITH m0 AS" in sql and "FULL OUTER JOIN" in sql
    single = run({"metrics": ["avg_power_kw"], "dimensions": ["facility.region"]}, ANALYST)[1]
    assert {r[0]: round(r[1], 6) for r in rows} == {r[0]: round(r[1], 6) for r in single}


def test_mixed_grain_without_dimensions_cross_joins_single_rows():
    cols, rows, _ = run({"metrics": ["avg_power_kw", "rack_count"]})
    assert cols == ["avg_power_kw", "rack_count"] and len(rows) == 1 and rows[0][1] == 12


def test_relative_time_range_and_hourly_buckets():
    cols, rows, sql = run({
        "metrics": ["avg_power_kw"],
        "time_range": {"ref": "power_reading.ts", "last": 6, "unit": "hour"},
        "time_grain": {"ref": "power_reading.ts", "grain": "hour"},
    })
    assert cols == ["power_reading__ts__hour", "avg_power_kw"]
    assert [r[0] for r in rows] == [f"2026-11-01T{h:02d}:00:00" for h in range(6, 12)]  # ascending, last 6 hours
    assert "'2026-11-01T06:00:00'" in sql


def test_absolute_range_and_daily_bucket():
    _, rows, _ = run({
        "metrics": ["max_power_kw"],
        "time_range": {"ref": "power_reading.ts", "start": "2026-10-31", "end": "2026-11-01"},
        "time_grain": {"ref": "power_reading.ts", "grain": "day"},
    })
    assert [r[0] for r in rows] == ["2026-10-31"]


def test_month_window_uses_calendar_months():
    start, end = resolve_range(TimeRange(ref="x.ts", last=1, unit="month"),
                               dt.datetime(2026, 3, 31, tzinfo=dt.timezone.utc))
    assert start == dt.datetime(2026, 2, 28, tzinfo=dt.timezone.utc) and end.month == 3


def test_time_plan_validation():
    with pytest.raises(ValueError):
        TimeRange(ref="power_reading.ts")
    with pytest.raises(ValueError):
        TimeRange(ref="power_reading.ts", last=3, start="2026-01-01")
    with pytest.raises(ValueError):
        TimeRange(ref="power_reading.ts", start="2026-02-01", end="2026-01-01")
    with pytest.raises(PlanError):
        compile_plan(QueryPlan(metrics=["avg_power_kw"], time_grain={"ref": "facility.region", "grain": "day"}),
                     ONTO, ADMIN, now=NOW)


def test_databricks_and_snowflake_time_sql():
    onto = Ontology([replace(c, source="databricks") for c in ONTO.concepts.values()], ONTO.edges,
                    list(ONTO.metrics.values()), ONTO.synonyms, list(ONTO.grants.values()), 1)
    plan = QueryPlan(metrics=["avg_power_kw"], time_range={"ref": "power_reading.ts", "last": 7, "unit": "day"},
                     time_grain={"ref": "power_reading.ts", "grain": "day"})
    for dialect in ("databricks", "snowflake"):
        c = compile_plan(plan, onto, ADMIN, now=NOW, dialects={"databricks": dialect})
        sql = guard(c.sql, c.dialect, c.tables)
        assert "DATE_TRUNC('DAY'" in sql and "CAST('2026-10-25 12:00:00' AS TIMESTAMP)" in sql


def _scoped_on_intermediate():
    """'ops' may read facilities only through racks it is entitled to (scope on the intermediate concept)."""
    grants = [Grant("ops", "power_reading", "internal"), Grant("ops", "rack", "internal", "rack.row", "rows"),
              Grant("ops", "facility", "internal")]
    return Ontology(list(ONTO.concepts.values()), ONTO.edges, list(ONTO.metrics.values()), ONTO.synonyms, grants, 1)


def test_row_scope_on_intermediate_join_concept_is_applied():
    ops = UserContext("u-ops", "ops", {"rows": ["R1"]})
    c = compile_plan(QueryPlan(metrics=["avg_power_kw"], dimensions=["facility.region"]), _scoped_on_intermediate(), ops)
    assert "rack.row_label IN ('R1')" in c.sql


def test_missing_grant_on_intermediate_concept_is_denied():
    grants = [g for g in ONTO.grants.values() if not (g.role == "admin" and g.concept == "rack")]
    onto = Ontology(list(ONTO.concepts.values()), ONTO.edges, list(ONTO.metrics.values()), ONTO.synonyms, grants, 1)
    with pytest.raises(AccessError):
        compile_plan(QueryPlan(metrics=["avg_power_kw"], dimensions=["facility.region"]), onto, ADMIN)


def test_join_path_prefers_lower_priority_on_ties():
    a = Attribute("id", "id", "int", "key")
    concepts = [Concept(n, "sqlite", n, n, {"id": a}) for n in ("a", "b", "c", "d")]
    edges = [Edge("a", "b", "id", "id", "many_to_one", 50), Edge("b", "d", "id", "id", "many_to_one", 50),
             Edge("a", "c", "id", "id", "many_to_one", 10), Edge("c", "d", "id", "id", "many_to_one", 10)]
    onto = Ontology(concepts, edges, [Metric("n", "a", "count", "id")])
    assert [s.nxt for s in onto.path("a", "d")] == ["c", "d"]
    edges[2] = Edge("a", "c", "id", "id", "many_to_one", 500)
    assert [s.nxt for s in Ontology(concepts, edges, []).path("a", "d")] == ["b", "d"]


@pytest.mark.parametrize("sql", [
    "WITH m0 AS (DELETE FROM racks) SELECT * FROM m0",
    "WITH RECURSIVE c(x) AS (SELECT 1) SELECT * FROM c",
    "WITH m0 AS (SELECT * FROM secrets) SELECT * FROM m0",
    "WITH m0 AS (WITH x AS (SELECT 1) SELECT * FROM x) SELECT * FROM m0",
    "SELECT * FROM racks UNION SELECT * FROM racks",
])
def test_guard_rejects_unsafe_ctes(sql):
    with pytest.raises(GuardError):
        guard(sql, "sqlite", {"racks"})


def test_guard_accepts_compiler_ctes():
    guard("WITH m0 AS (SELECT rack_id FROM racks) SELECT m0.rack_id FROM m0", "sqlite", {"racks"})


# --- cross-engine ------------------------------------------------------------------------

def split_ontology():
    """Facilities live on a second engine ('sqlite_b')."""
    concepts = [replace(c, source="sqlite_b") if c.name == "facility" else c for c in ONTO.concepts.values()]
    return Ontology(concepts, ONTO.edges, list(ONTO.metrics.values()), ONTO.synonyms, list(ONTO.grants.values()), 1)


def run_split(plan, user):
    c = compile_plan(QueryPlan(**plan), split_ontology(), user, now=NOW, dialects={"sqlite_b": "sqlite"})
    assert [p.engine for p in c.parts] == ["sqlite", "sqlite_b"] and c.merge
    local, remote = (CONN.execute(guard(p.sql, p.dialect, p.tables, 100_001)).fetchall() for p in c.parts)
    return c, merge_parts(c.merge, [list(r) for r in local], [list(r) for r in remote])


@pytest.mark.parametrize("metric", ["avg_power_kw", "max_power_kw"])
def test_cross_engine_split_matches_single_engine(metric):
    plan = {"metrics": [metric], "dimensions": ["facility.region"]}
    _, merged = run_split(plan, ANALYST)
    single = run(plan, ANALYST)[1]
    assert {r[0]: round(r[1], 6) for r in merged} == {r[0]: round(r[1], 6) for r in single}


def test_cross_engine_scope_and_filter_on_remote_side():
    c, merged = run_split({"metrics": ["rack_count"], "dimensions": ["facility.region"],
                           "filters": [{"ref": "facility.tier", "op": "=", "values": ["III"]}]}, ANALYST)
    assert "facility.region IN ('West', 'South')" in c.parts[1].sql
    assert c.merge["join"] == "inner"
    assert sorted(merged) == [["South", 4], ["West", 4]]


def test_cross_engine_refuses_distinct_and_mixed_grain():
    onto = split_ontology()
    onto.metrics["distinct_racks"] = Metric("distinct_racks", "power_reading", "count", "ts", distinct=True)
    with pytest.raises(CrossEngineError):
        compile_plan(QueryPlan(metrics=["distinct_racks"], dimensions=["facility.region"]), onto, ADMIN)
    with pytest.raises(CrossEngineError):
        compile_plan(QueryPlan(metrics=["avg_power_kw", "rack_count"], dimensions=["facility.region"]), onto, ADMIN)


SPEC = {
    "outputs": [["r", 0]], "n_local": 0, "n_remote": 1,
    "metrics": [{"id": "avg", "agg": "avg"}, {"id": "n", "agg": "count"}, {"id": "hi", "agg": "max"}],
    "join": "left", "order_by": "n", "descending": True, "time_alias": None, "limit": 10,
    "max_partial_rows": 5, "columns": ["region", "avg", "n", "hi"],
}


def test_merge_combines_partials_correctly():
    local = [[1, 10.0, 2, 4, 7.0], [2, 6.0, 3, 1, 9.0], [3, 4.0, 1, 1, 4.0]]  # key, avg sum, avg count, n, hi
    remote = [["West", 1], ["West", 2]]
    rows = merge_parts(SPEC, local, remote)
    assert rows[0] == ["West", 16.0 / 5, 5, 9.0]
    assert rows[1] == [None, 4.0, 1, 4.0]  # left join keeps unmatched keys
    assert merge_parts({**SPEC, "join": "inner"}, local, remote) == rows[:1]


def test_merge_matches_keys_across_types_and_refuses_oversized_parts():
    assert merge_parts(SPEC, [[1.0, 2.0, 1, 1, 2.0]], [["West", "1"]])[0][0] == "West"
    with pytest.raises(MergeTooLarge):
        merge_parts(SPEC, [[i, 1.0, 1, 1, 1.0] for i in range(6)], [])
