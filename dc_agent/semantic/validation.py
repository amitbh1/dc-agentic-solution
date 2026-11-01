"""Structural checks on an ontology release. Run on every load (problems are logged and counted)
and from `python -m dc_agent.jobs.validate` in CI before publishing a release."""
from __future__ import annotations

from .models import SENS
from .ontology import Ontology

AGGS = {"sum", "avg", "min", "max", "count"}
ROLES = {"key", "dimension", "measure", "time"}
CARDINALITIES = {"many_to_one", "one_to_many", "one_to_one"}


def validate_ontology(onto: Ontology) -> list:
    """Returns human-readable problems; an empty list means the release is consistent."""
    problems = []
    for c in onto.concepts.values():
        for a in c.attrs.values():
            if a.role not in ROLES:
                problems.append(f"{c.name}.{a.name}: unknown role {a.role!r}")
            if a.sensitivity not in SENS:
                problems.append(f"{c.name}.{a.name}: unknown sensitivity {a.sensitivity!r}")
    for m in onto.metrics.values():
        if m.agg not in AGGS:
            problems.append(f"metric {m.id}: unknown aggregate {m.agg!r}")
        if m.attr not in onto.concepts[m.concept].attrs:
            problems.append(f"metric {m.id}: attribute {m.concept}.{m.attr} does not exist")
    for e in onto.edges:
        if e.cardinality not in CARDINALITIES:
            problems.append(f"edge {e.left}->{e.right}: unknown cardinality {e.cardinality!r}")

    # Rule 4: if a role is row-scoped anywhere, every concept it can query must carry a scope.
    # An unscoped concept would expose rows the role is not entitled to.
    roles: dict = {}
    for (role, concept), g in onto.grants.items():
        roles.setdefault(role, []).append(g)
        if g.scope_ref:
            try:
                onto.attr(g.scope_ref)
            except Exception:
                problems.append(f"grant {role}/{concept}: scope attribute {g.scope_ref} does not exist")
            if not g.claim:
                problems.append(f"grant {role}/{concept}: scope without a claim")
        if g.max_sensitivity not in SENS:
            problems.append(f"grant {role}/{concept}: unknown max_sensitivity {g.max_sensitivity!r}")
    for role, grants in sorted(roles.items()):
        if any(g.scope_ref for g in grants):
            for g in grants:
                if not g.scope_ref:
                    problems.append(f"role {role}: concept {g.concept} has no row scope but other grants do")
    for s in onto.synonyms:
        if s.kind == "metric" and s.target not in onto.metrics:
            problems.append(f"synonym {s.term!r}: unknown metric {s.target}")
    return sorted(problems)
