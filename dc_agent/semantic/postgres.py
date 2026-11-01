from __future__ import annotations

import json
import logging

from .models import Attribute, Concept, Edge, Example, Grant, Metric, Synonym
from .ontology import Ontology

log = logging.getLogger("dc_agent.ontology")

MAX_EXAMPLES = 500


def _vector(text):
    """pgvector's text form '[0.1,0.2,...]' is valid JSON."""
    return tuple(json.loads(text)) if text else None


def load_from_postgres(dsn: str) -> Ontology:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM ontology.release WHERE is_active")
        row = cur.fetchone()
        release_id = row[0] if row else 0

        cur.execute(
            "SELECT c.name, a.name, a.physical_column, a.data_type, a.role, a.sensitivity,"
            " coalesce(a.description, ''), a.unit"
            " FROM ontology.attribute a JOIN ontology.concept c ON c.id = a.concept_id"
            " WHERE c.status = 'active'"
        )
        attrs: dict = {}
        for cname, name, col, dtype, role, sens, desc, unit in cur.fetchall():
            attrs.setdefault(cname, {})[name] = Attribute(name, col, dtype, role, sens, desc, unit)

        cur.execute(
            "SELECT name, source, physical_table, description FROM ontology.concept WHERE status = 'active'"
        )
        concepts = [Concept(n, s, t, d, attrs.get(n, {})) for n, s, t, d in cur.fetchall()]

        cur.execute(
            "SELECT l.name, r.name, e.left_key, e.right_key, e.cardinality, e.priority"
            " FROM ontology.relationship e"
            " JOIN ontology.concept l ON l.id = e.left_concept_id AND l.status = 'active'"
            " JOIN ontology.concept r ON r.id = e.right_concept_id AND r.status = 'active'"
        )
        edges = [Edge(*r) for r in cur.fetchall()]

        cur.execute(
            "SELECT m.id, c.name, a.name, m.agg, m.is_distinct, m.description, m.unit"
            " FROM ontology.metric m"
            " JOIN ontology.concept c ON c.id = m.concept_id AND c.status = 'active'"
            " JOIN ontology.attribute a ON a.id = m.attribute_id"
        )
        metrics = [Metric(i, c, agg, a, d, u, dist) for i, c, a, agg, dist, d, u in cur.fetchall()]

        cur.execute(
            "SELECT s.term,"
            " CASE WHEN s.metric_id IS NOT NULL THEN 'metric'"
            "      WHEN s.attribute_id IS NOT NULL THEN 'attribute' ELSE 'concept' END,"
            " COALESCE(s.metric_id, ac.name || '.' || a.name, c.name),"
            " s.embedding::text"
            " FROM ontology.synonym s"
            " LEFT JOIN ontology.attribute a ON a.id = s.attribute_id"
            " LEFT JOIN ontology.concept ac ON ac.id = a.concept_id"
            " LEFT JOIN ontology.concept c ON c.id = s.concept_id"
        )
        synonyms = [Synonym(t, k, tgt, _vector(v)) for t, k, tgt, v in cur.fetchall()]

        cur.execute(
            "SELECT g.role, c.name, g.max_sensitivity, sc.name || '.' || sa.name, g.row_scope_claim"
            " FROM ontology.role_grant g"
            " JOIN ontology.concept c ON c.id = g.concept_id"
            " LEFT JOIN ontology.attribute sa ON sa.id = g.row_scope_attribute_id"
            " LEFT JOIN ontology.concept sc ON sc.id = sa.concept_id"
        )
        grants = [Grant(*r) for r in cur.fetchall()]

        # Verified examples from any release; ones that no longer fit the ontology are dropped below.
        cur.execute(
            "SELECT question, plan, question_embedding::text FROM ontology.verified_query"
            " ORDER BY created_at DESC LIMIT %s",
            (MAX_EXAMPLES,),
        )
        raw_examples = [
            Example(q, p if isinstance(p, dict) else json.loads(p), _vector(v)) for q, p, v in cur.fetchall()
        ]

    onto = Ontology(concepts, edges, metrics, synonyms, grants, release_id)
    onto.examples = [ex for ex in raw_examples if _fits(ex.plan, onto)]
    if len(onto.examples) < len(raw_examples):
        log.info("dropped verified queries that no longer match the ontology",
                 extra={"fields": {"dropped": len(raw_examples) - len(onto.examples)}})
    onto.has_vectors = onto.has_vectors or any(e.vector for e in onto.examples)
    return onto


def _fits(plan: dict, onto: Ontology) -> bool:
    from .examples import plan_refs

    metrics, refs = plan_refs(plan)
    if not metrics or not all(m in onto.metrics for m in metrics):
        return False
    try:
        for ref in refs:
            onto.attr(ref)
    except Exception:
        return False
    return True
