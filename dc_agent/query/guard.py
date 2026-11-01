from __future__ import annotations

import sqlglot
from sqlglot import exp

from ..core.errors import GuardError


def _arg(node, name: str):
    """sqlglot renamed some arg keys (with -> with_) across versions; accept both."""
    v = node.args.get(name)
    return v if v is not None else node.args.get(name + "_")


def guard(sql: str, dialect: str, allowed_tables: set, max_rows: int = 10_000) -> str:
    """Independent check on the final SQL: a single SELECT (optionally with SELECT-only CTEs),
    no subqueries, no SELECT INTO, allow-listed tables only, and a row cap."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as exc:
        raise GuardError(f"unparseable SQL: {exc}") from exc
    if not isinstance(tree, exp.Select):
        raise GuardError("only a single SELECT is allowed")
    if _arg(tree, "into") is not None:
        raise GuardError("SELECT INTO is not allowed")
    cte_names = set()
    with_ = _arg(tree, "with")
    if with_ is not None:
        if with_.args.get("recursive"):
            raise GuardError("recursive CTEs are not allowed")
        for cte in with_.expressions:
            body = cte.this
            if not isinstance(body, exp.Select) or _arg(body, "with") is not None or _arg(body, "into") is not None:
                raise GuardError("CTEs must be plain SELECTs")
            cte_names.add(cte.alias_or_name.lower())
    if tree.find(exp.Subquery) is not None:
        raise GuardError("subqueries are not allowed")
    allowed = {t.lower() for t in allowed_tables}
    for t in tree.find_all(exp.Table):
        name = ".".join(p for p in (t.catalog, t.db, t.name) if p).lower()
        if not (t.catalog or t.db) and name in cte_names:
            continue
        if name not in allowed:
            raise GuardError(f"table not allowed: {name}")
    if tree.args.get("limit") is None:
        tree = tree.limit(max_rows)
    return tree.sql(dialect=dialect, pretty=True)
