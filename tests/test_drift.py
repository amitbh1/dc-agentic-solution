from dc_agent.demo import build_ontology, build_sqlite
from dc_agent.jobs.drift import check_ontology
from dc_agent.warehouse import SqliteExecutor
from dc_agent.semantic import Attribute, Concept, Ontology


def _executors():
    return {"sqlite": SqliteExecutor(build_sqlite())}


def test_no_drift_on_demo_data():
    assert check_ontology(build_ontology(), _executors()) == []


def test_missing_column_and_table_are_reported():
    base = build_ontology()
    broken_rack = Concept(
        "rack", "sqlite", "racks", "Physical rack",
        {"id": Attribute("id", "rack_id", "bigint", "key"), "row": Attribute("row", "row_label_v2", "string", "dimension")},
    )
    ghost = Concept("ghost", "sqlite", "no_such_table", "Missing", {})
    concepts = [c for c in base.concepts.values() if c.name != "rack"] + [broken_rack, ghost]
    edges = []
    onto = Ontology(concepts, edges, list(base.metrics.values()), base.synonyms, list(base.grants.values()), 1)
    kinds = {(f.concept, f.kind, f.attribute) for f in check_ontology(onto, _executors())}
    assert ("rack", "column_missing", "row") in kinds
    assert ("ghost", "table_missing", None) in kinds
