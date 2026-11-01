"""Deterministic plan-to-SQL compiler. Every identifier comes from the ontology; every value is a typed literal.

Shapes:
- one base concept, one engine: a single SELECT (the common case)
- several base concepts (mixed grain), one engine: one CTE per base, joined on the dimensions
- one base concept spanning two engines: a partial aggregate per engine, merged in Python (see merge.py)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlglot import exp

from ..core.errors import AccessError, CrossEngineError, FanOutError, PlanError
from ..security.identity import UserContext
from ..semantic.models import Attribute, Concept
from ..semantic.ontology import Ontology
from .plan import QueryPlan
from .timeutil import bucket, resolve_range, time_literal

FAN_OUT_SENSITIVE = {"sum", "count", "avg"}
CMP = {"=": exp.EQ, "!=": exp.NEQ, ">": exp.GT, ">=": exp.GTE, "<": exp.LT, "<=": exp.LTE}
AGG = {"sum": exp.Sum, "avg": exp.Avg, "min": exp.Min, "max": exp.Max, "count": exp.Count}
INT_TYPES = {"int", "integer", "bigint", "smallint", "tinyint", "long"}
FLOAT_TYPES = {"double", "float", "real", "decimal", "numeric", "number"}
TIME_TYPES = {"timestamp", "timestamp_ntz", "timestamp_ltz", "timestamp_tz", "datetime", "date"}
KEY = "__k"


@dataclass(frozen=True)
class Output:
    """A grouped output column."""

    alias: str
    concept: Concept
    attr: Attribute
    grain: str | None = None


@dataclass
class Part:
    """One SQL statement for one engine."""

    engine: str
    dialect: str
    sql: str
    tables: set
    columns: list


@dataclass
class Compiled:
    parts: list
    concepts: list
    columns: list
    merge: dict | None = None  # set when parts must be merged in Python

    @property
    def sql(self) -> str:
        if len(self.parts) == 1:
            return self.parts[0].sql
        return "\n\n".join(f"-- engine: {p.engine}\n{p.sql}" for p in self.parts)

    @property
    def dialect(self) -> str:
        return self.parts[0].dialect

    @property
    def tables(self) -> set:
        return set().union(*(p.tables for p in self.parts))


@dataclass
class _Group:
    base: str
    metrics: list
    steps: list = field(default_factory=list)
    joined: set = field(default_factory=set)
    scopes: list = field(default_factory=list)  # (concept, attribute, values)


def _base_type(attr) -> str:
    return attr.data_type.lower().split("(")[0].strip()


def _typed(attr, raw: str):
    base = _base_type(attr)
    try:
        if base in INT_TYPES:
            return int(raw)
        if base in FLOAT_TYPES:
            return float(raw)
    except ValueError:
        raise PlanError(f"value {raw!r} is not valid for {attr.name} ({attr.data_type})")
    if base in ("boolean", "bool"):
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        raise PlanError(f"value {raw!r} is not a boolean for {attr.name}")
    return raw


def _col(concept, attr) -> exp.Column:
    return exp.column(attr.column, table=concept.name)


def _table(onto: Ontology, concept_name: str) -> exp.Table:
    t = exp.to_table(onto.concepts[concept_name].table)
    t.set("alias", exp.TableAlias(this=exp.to_identifier(concept_name)))
    return t


def _require_time(ref: str, attr) -> None:
    if attr.role != "time" and _base_type(attr) not in TIME_TYPES:
        raise PlanError(f"{ref} is not a time attribute")


def _out_expr(o: Output, dialect: str) -> exp.Expression:
    col = _col(o.concept, o.attr)
    return bucket(col, o.grain, dialect) if o.grain else col


def _join_group(onto: Ontology, user: UserContext, base: str, needed: set, metrics: list) -> _Group:
    """Join tree from the base concept to every needed concept, with row scopes for every joined
    concept. Scopes can pull in further concepts, so iterate to a fixed point."""
    goals = set(needed)
    while True:
        steps, joined = [], {base}
        for goal in sorted(goals - {base}):
            for st in onto.path(base, goal):
                if st.nxt not in joined:
                    steps.append(st)
                    joined.add(st.nxt)
        scopes, seen, grew = [], set(), False
        for cname in sorted(joined):
            g = onto.grants.get((user.role, cname))
            if g is None:  # also covers concepts only passed through on the join path
                raise AccessError(f"{user.role} may not read {cname}")
            if not g.scope_ref or (g.scope_ref, g.claim) in seen:
                continue
            seen.add((g.scope_ref, g.claim))
            values = user.claims.get(g.claim, [])
            if not values:
                raise AccessError(f"no entitlement values for claim {g.claim}")
            sc, sa = onto.attr(g.scope_ref)
            scopes.append((sc, sa, list(values)))
            if sc.name not in joined:
                goals.add(sc.name)
                grew = True
        if not grew:
            break
    sensitive = any(m.agg in FAN_OUT_SENSITIVE and not m.distinct for m in metrics)
    for st in steps:
        if st.fan and sensitive:
            raise FanOutError(f"joining {st.cur} -> {st.nxt} multiplies rows; pre-aggregate first")
    return _Group(base, metrics, steps, joined, scopes)


def _where(q, conds, trange, scopes, dialect):
    for f, c, a in conds:
        col = _col(c, a)
        if f.op == "in":
            q = q.where(exp.In(this=col, expressions=[exp.convert(_typed(a, v)) for v in f.values]))
        else:
            if len(f.values) != 1:
                raise PlanError(f"operator {f.op} takes exactly one value")
            q = q.where(CMP[f.op](this=col, expression=exp.convert(_typed(a, f.values[0]))))
    if trange:
        c, a, start, end = trange
        col = _col(c, a)
        if start:
            q = q.where(exp.GTE(this=col, expression=time_literal(start, a.data_type, dialect)))
        if end:
            q = q.where(exp.LT(this=col, expression=time_literal(end, a.data_type, dialect)))
    for sc, sa, values in scopes:
        q = q.where(exp.In(this=_col(sc, sa), expressions=[exp.convert(_typed(sa, v)) for v in values]))
    return q


def _select(onto, root, steps, outputs, metric_cols, conds, trange, scopes, dialect, *,
            keys=(), group=True, distinct=False):
    sel = [exp.alias_(_out_expr(o, dialect), o.alias) for o in outputs]
    sel += [exp.alias_(e, alias) for e, alias in keys]
    q = exp.select(*sel, *metric_cols).from_(_table(onto, root))
    for st in steps:
        on = exp.EQ(this=exp.column(st.cur_key, table=st.cur), expression=exp.column(st.nxt_key, table=st.nxt))
        q = q.join(_table(onto, st.nxt), on=on, join_type="left")
    q = _where(q, conds, trange, scopes, dialect)
    group_exprs = [_out_expr(o, dialect) for o in outputs] + [e.copy() for e, _ in keys]
    if group and group_exprs:
        q = q.group_by(*group_exprs)
    if distinct:
        q = q.distinct()
    return q


def _metric_col(m, onto: Ontology) -> exp.Expression:
    c, a = onto.attr(f"{m.concept}.{m.attr}")
    col = _col(c, a)
    agg = exp.Count(this=exp.Distinct(expressions=[col])) if m.distinct else AGG[m.agg](this=col)
    return exp.alias_(agg, m.id)


def _finish(q, plan: QueryPlan, outputs: list, max_rows: int):
    if plan.order_by_metric:
        q = q.order_by(exp.Ordered(this=exp.column(plan.order_by_metric), desc=plan.descending))
    elif plan.time_grain:
        alias = next(o.alias for o in outputs if o.grain)
        q = q.order_by(exp.Ordered(this=exp.column(alias), desc=False))
    return q.limit(min(plan.limit, max_rows))


def compile_plan(plan: QueryPlan, onto: Ontology, user: UserContext, max_rows: int = 10_000, *,
                 now: dt.datetime | None = None, dialects: dict | None = None,
                 max_partial_rows: int = 100_000) -> Compiled:
    """dialects maps engine name (concept.source) to sqlglot dialect; by default they are the same."""
    now = now or dt.datetime.now(dt.timezone.utc)
    dialects = dialects or {}

    metrics, seen_ids = [], set()
    for mid in plan.metrics:
        m = onto.metrics.get(mid)
        if m is None:
            raise PlanError(f"unknown metric: {mid}")
        if mid not in seen_ids:
            seen_ids.add(mid)
            metrics.append(m)
    if plan.order_by_metric and plan.order_by_metric not in seen_ids:
        raise PlanError("order_by_metric must be one of the selected metrics")

    outputs, aliases = [], set()
    for ref in plan.dimensions:
        c, a = onto.attr(ref)
        alias = ref.replace(".", "__")
        if alias not in aliases:
            aliases.add(alias)
            outputs.append(Output(alias, c, a))
    if plan.time_grain:
        c, a = onto.attr(plan.time_grain.ref)
        _require_time(plan.time_grain.ref, a)
        outputs.append(Output(f"{plan.time_grain.ref.replace('.', '__')}__{plan.time_grain.grain}", c, a,
                              plan.time_grain.grain))
    conds = [(f, *onto.attr(f.ref)) for f in plan.filters]
    trange = None
    if plan.time_range:
        c, a = onto.attr(plan.time_range.ref)
        _require_time(plan.time_range.ref, a)
        trange = (c, a, *resolve_range(plan.time_range, now))

    checks = [(o.concept, o.attr) for o in outputs] + [(c, a) for _, c, a in conds]
    checks += [onto.attr(f"{m.concept}.{m.attr}") for m in metrics]
    if trange:
        checks.append(trange[:2])
    for c, a in checks:
        onto.check_access(user, c, a)

    needed = {o.concept.name for o in outputs} | {c.name for _, c, _ in conds}
    if trange:
        needed.add(trange[0].name)
    by_base: dict = {}
    for m in metrics:
        by_base.setdefault(m.concept, []).append(m)
    groups = [_join_group(onto, user, base, needed, ms) for base, ms in by_base.items()]

    columns = [o.alias for o in outputs] + [m.id for m in metrics]
    engines = {onto.concepts[n].source for g in groups for n in g.joined}
    if len(engines) == 1:
        engine = engines.pop()
        dialect = dialects.get(engine, engine)
        if len(groups) == 1:
            g = groups[0]
            q = _select(onto, g.base, g.steps, outputs, [_metric_col(m, onto) for m in g.metrics],
                        conds, trange, g.scopes, dialect)
        else:
            q = _mixed_grain(onto, groups, outputs, conds, trange, dialect, metrics)
        q = _finish(q, plan, outputs, max_rows)
        joined = set().union(*(g.joined for g in groups))
        part = Part(engine, dialect, q.sql(dialect=dialect, pretty=True),
                    {onto.concepts[n].table for n in joined}, columns)
        return Compiled([part], sorted(joined), columns)
    if len(groups) > 1:
        raise CrossEngineError("metrics with different grain across engines are not supported")
    if len(engines) > 2:
        raise CrossEngineError(f"plan spans more than two engines: {sorted(engines)}")
    return _cross_engine(onto, groups[0], outputs, conds, trange, plan, max_rows, max_partial_rows, dialects, columns)


def _mixed_grain(onto, groups, outputs, conds, trange, dialect, metrics):
    """One CTE per base concept, aggregated to the requested dimensions, then FULL OUTER JOINed on them.
    Each metric is computed at its own grain, so no aggregate is inflated by another's join."""
    names = [f"m{i}" for i in range(len(groups))]
    owner = {m.id: names[i] for i, g in enumerate(groups) for m in g.metrics}
    sel = []
    for o in outputs:
        cols = [exp.column(o.alias, table=n) for n in names]
        sel.append(exp.alias_(exp.Coalesce(this=cols[0], expressions=cols[1:]), o.alias))
    sel += [exp.alias_(exp.column(m.id, table=owner[m.id]), m.id) for m in metrics]
    q = exp.select(*sel).from_(exp.to_table(names[0]))
    for i in range(1, len(names)):
        if outputs:
            preds = []
            for o in outputs:
                prev = [exp.column(o.alias, table=n) for n in names[:i]]
                left = prev[0] if len(prev) == 1 else exp.Coalesce(this=prev[0], expressions=prev[1:])
                preds.append(exp.EQ(this=left, expression=exp.column(o.alias, table=names[i])))
            q = q.join(exp.to_table(names[i]), on=exp.and_(*preds), join_type="full outer")
        else:
            q = q.join(exp.to_table(names[i]), join_type="cross")
    for name, g in zip(names, groups):
        cte = _select(onto, g.base, g.steps, outputs, [_metric_col(m, onto) for m in g.metrics],
                      conds, trange, g.scopes, dialect)
        q = q.with_(name, as_=cte)
    return q


def _partials(m, onto) -> list:
    """Partial aggregates that can be combined after a merge."""
    if m.distinct:
        raise CrossEngineError("distinct counts cannot be merged across engines")
    c, a = onto.attr(f"{m.concept}.{m.attr}")
    col = _col(c, a)
    if m.agg == "avg":
        return [exp.alias_(exp.Sum(this=col), f"{m.id}__sum"), exp.alias_(exp.Count(this=col.copy()), f"{m.id}__count")]
    return [exp.alias_(AGG[m.agg](this=col), f"{m.id}__{m.agg}")]


def _cross_engine(onto, g, outputs, conds, trange, plan, max_rows, max_partial_rows, dialects, columns):
    """Split one plan at the single edge where it crosses engines.

    Local engine (the metric's): partial aggregates grouped by local dimensions and the boundary key.
    Remote engine: distinct (boundary key, remote dimensions) under remote filters and scopes.
    """
    src = {n: onto.concepts[n].source for n in g.joined}
    local_engine = src[g.base]
    remote = {n for n in g.joined if src[n] != local_engine}
    boundary = [st for st in g.steps if (st.cur in remote) != (st.nxt in remote)]
    if len(boundary) != 1 or boundary[0].cur in remote:
        raise CrossEngineError("plan crosses between engines more than once")
    b = boundary[0]
    remote_engine = src[b.nxt]
    ld, rd = dialects.get(local_engine, local_engine), dialects.get(remote_engine, remote_engine)

    def side(items, concept_of, want_remote):
        return [x for x in items if (concept_of(x) in remote) == want_remote]

    l_out, r_out = side(outputs, lambda o: o.concept.name, False), side(outputs, lambda o: o.concept.name, True)
    l_cond, r_cond = side(conds, lambda x: x[1].name, False), side(conds, lambda x: x[1].name, True)
    l_scope, r_scope = side(g.scopes, lambda x: x[0].name, False), side(g.scopes, lambda x: x[0].name, True)
    l_time = trange if trange and trange[0].name not in remote else None
    r_time = trange if trange and trange[0].name in remote else None

    partial_cols = [col for m in g.metrics for col in _partials(m, onto)]
    local = _select(onto, g.base, [st for st in g.steps if st.nxt not in remote], l_out, partial_cols,
                    l_cond, l_time, l_scope, ld, keys=[(exp.column(b.cur_key, table=b.cur), KEY)])
    local = local.limit(max_partial_rows + 1)
    remote_q = _select(onto, b.nxt, [st for st in g.steps if st.cur in remote and st.nxt in remote], r_out, [],
                       r_cond, r_time, r_scope, rd, keys=[(exp.column(b.nxt_key, table=b.nxt), KEY)],
                       group=False, distinct=True)
    remote_q = remote_q.limit(max_partial_rows + 1)

    local_names = [o.alias for o in l_out] + [KEY] + [c.alias for c in partial_cols]
    remote_names = [o.alias for o in r_out] + [KEY]
    parts = [
        Part(local_engine, ld, local.sql(dialect=ld, pretty=True),
             {onto.concepts[n].table for n in g.joined - remote}, local_names),
        Part(remote_engine, rd, remote_q.sql(dialect=rd, pretty=True),
             {onto.concepts[n].table for n in remote}, remote_names),
    ]
    merge = {
        "outputs": [["l", l_out.index(o)] if o in l_out else ["r", r_out.index(o)] for o in outputs],
        "n_local": len(l_out),
        "n_remote": len(r_out),
        "metrics": [{"id": m.id, "agg": m.agg} for m in g.metrics],
        "join": "inner" if (r_cond or r_time or r_scope) else "left",
        "order_by": plan.order_by_metric,
        "descending": plan.descending,
        "time_alias": next((o.alias for o in outputs if o.grain), None),
        "limit": min(plan.limit, max_rows),
        "max_partial_rows": max_partial_rows,
        "columns": columns,
    }
    return Compiled(parts, sorted(g.joined), columns, merge)
