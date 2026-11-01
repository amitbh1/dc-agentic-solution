"""Ontology drift check: compare the ontology with the live warehouse schemas.

    python -m dc_agent.jobs.drift --dry-run   # print findings, exit 1 if anything is broken
    python -m dc_agent.jobs.drift             # also record findings; the database trigger marks concepts stale

Run it on a schedule (for example every 15 minutes). After fixing a mapping, set the concept back
to active and set drift_event.resolved_at.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

_FAMILIES = [
    ("boolean", re.compile(r"^(bool|boolean)$")),
    ("number", re.compile(r"^(tinyint|smallint|int|integer|bigint|long|byte|short|float|double|real|decimal|"
                          r"numeric|number|int\d+|float\d+)$")),
    ("temporal", re.compile(r"^(date|datetime|time|timestamp|timestamp_ntz|timestamp_ltz|timestamp_tz)$")),
    ("string", re.compile(r"^(string|text|varchar|char|character|nvarchar|nchar|clob)$")),
]


def type_family(data_type: str, dialect: str = "") -> str | None:
    """Coarse type family, or None when unknown (unknown types are never reported as drift).
    SQLite stores timestamps as text, so there temporal and string are the same family."""
    base = re.split(r"[(<\s]", (data_type or "").strip().lower(), maxsplit=1)[0]
    for family, pattern in _FAMILIES:
        if pattern.match(base):
            return "string" if dialect == "sqlite" and family == "temporal" else family
    return None


@dataclass(frozen=True)
class Finding:
    concept: str
    kind: str  # table_missing | column_missing | type_changed | new_column
    attribute: str | None
    detail: dict

    @property
    def breaking(self) -> bool:
        return self.kind != "new_column"


def check_ontology(onto, executors: dict) -> list:
    findings = []
    for c in onto.concepts.values():
        ex = executors.get(c.source)
        if ex is None:
            continue
        try:
            cols = {k.lower(): v for k, v in ex.columns(c.table).items()}
        except Exception as exc:
            findings.append(Finding(c.name, "table_missing", None, {"error": repr(exc)[:200]}))
            continue
        if not cols:
            findings.append(Finding(c.name, "table_missing", None, {"error": "no columns returned"}))
            continue
        dialect = getattr(ex, "dialect", c.source)
        for a in c.attrs.values():
            live = cols.get(a.column.lower())
            if live is None:
                findings.append(Finding(c.name, "column_missing", a.name, {"column": a.column}))
                continue
            want, got = type_family(a.data_type, dialect), type_family(str(live), dialect)
            if want and got and want != got:
                findings.append(Finding(c.name, "type_changed", a.name,
                                        {"column": a.column, "expected": a.data_type, "actual": str(live)}))
        known = {a.column.lower() for a in c.attrs.values()}
        known |= {step[1].lower() for step in onto.adj.get(c.name, [])}
        for col in sorted(set(cols) - known):
            findings.append(Finding(c.name, "new_column", None, {"column": col}))
    return findings


_INSERT = """
INSERT INTO ontology.drift_event (concept_id, attribute_id, kind, detail)
SELECT c.id, a.id, %(kind)s, %(detail)s::jsonb
  FROM ontology.concept c
  LEFT JOIN ontology.attribute a ON a.concept_id = c.id AND a.name = %(attr)s
 WHERE c.name = %(concept)s
   AND NOT EXISTS (
         SELECT 1 FROM ontology.drift_event d
          WHERE d.concept_id = c.id AND d.kind = %(kind)s AND d.resolved_at IS NULL
            AND d.attribute_id IS NOT DISTINCT FROM a.id)
"""


def record(dsn: str, findings: list) -> None:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for f in findings:
            cur.execute(_INSERT, {"kind": f.kind, "detail": json.dumps(f.detail), "attr": f.attribute, "concept": f.concept})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from ..core.config import Settings
    from ..semantic.postgres import load_from_postgres
    from ..warehouse.factory import build_raw_executors

    settings = Settings.from_env()
    onto = load_from_postgres(settings.pg_dsn)
    findings = check_ontology(onto, build_raw_executors())
    for f in findings:
        print(f"{f.kind:15} {f.concept}.{f.attribute or '-'} {json.dumps(f.detail)}")
    if not args.dry_run and findings:
        record(settings.pg_dsn, findings)
    return 1 if any(f.breaking for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
