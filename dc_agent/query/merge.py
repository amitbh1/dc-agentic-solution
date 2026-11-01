"""Merge partial results from two engines (see compiler._cross_engine). Pure function; positions, not
column names, are used because engines differ in identifier case."""
from __future__ import annotations

from ..core.errors import AgentError


class MergeTooLarge(AgentError):
    """A part returned more rows than can be merged safely."""

    fatal = True


def _key(v):
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return None if v is None else str(v)


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _pick(fn, a, b):
    if a is None:
        return b
    if b is None:
        return a
    return fn(a, b)


def merge_parts(spec: dict, local_rows: list, remote_rows: list) -> list:
    cap = spec["max_partial_rows"]
    if len(local_rows) > cap or len(remote_rows) > cap:
        raise MergeTooLarge(f"a part returned more than {cap} rows")
    nl, nr = spec["n_local"], spec["n_remote"]

    lookup: dict = {}
    for r in remote_rows:
        lookup.setdefault(_key(r[nr]), []).append(tuple(r[:nr]))
    miss = [tuple([None] * nr)] if spec["join"] == "left" else []

    groups: dict = {}
    for row in local_rows:
        lo, k, partials = row[:nl], row[nl], row[nl + 1:]
        for ro in lookup.get(_key(k), miss):
            dims = tuple(lo[i] if side == "l" else ro[i] for side, i in spec["outputs"])
            acc = groups.get(dims)
            if acc is None:
                groups[dims] = list(partials)
                continue
            i = 0
            for m in spec["metrics"]:
                agg = m["agg"]
                if agg in ("sum", "count"):
                    acc[i] = _add(acc[i], partials[i])
                    i += 1
                elif agg == "min":
                    acc[i] = _pick(min, acc[i], partials[i])
                    i += 1
                elif agg == "max":
                    acc[i] = _pick(max, acc[i], partials[i])
                    i += 1
                else:  # avg: sum and count
                    acc[i] = _add(acc[i], partials[i])
                    acc[i + 1] = _add(acc[i + 1], partials[i + 1])
                    i += 2

    rows = []
    for dims, acc in groups.items():
        values, i = [], 0
        for m in spec["metrics"]:
            if m["agg"] == "avg":
                s, c = acc[i], acc[i + 1]
                values.append(s / c if c else None)
                i += 2
            else:
                values.append(acc[i])
                i += 1
        rows.append(list(dims) + values)

    cols = spec["columns"]
    if spec["order_by"]:
        idx = cols.index(spec["order_by"])
        present = [r for r in rows if r[idx] is not None]
        present.sort(key=lambda r: r[idx], reverse=spec["descending"])
        rows = present + [r for r in rows if r[idx] is None]
    elif spec["time_alias"]:
        idx = cols.index(spec["time_alias"])
        rows.sort(key=lambda r: (r[idx] is None, str(r[idx])))
    return rows[: spec["limit"]]
