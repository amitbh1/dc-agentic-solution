from __future__ import annotations

import datetime as dt
import decimal
import math
import re
from dataclasses import dataclass
from typing import Protocol

_SAFE_TABLE = re.compile(r"^[A-Za-z0-9_.\-]+$")


@dataclass(frozen=True)
class Estimate:
    """Pre-execution cost estimate from the engine's planner. Either field may be unknown."""

    bytes: int | None = None
    rows: int | None = None


class Executor(Protocol):
    dialect: str

    def run(self, sql: str, max_rows: int, timeout_s: int) -> tuple: ...

    def columns(self, table: str) -> dict: ...

    def estimate(self, sql: str) -> Estimate | None: ...


def safe_value(v):
    """Convert driver types to JSON-friendly values. NaN and infinities become null: they are not valid JSON."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, decimal.Decimal):
        return float(v) if v.is_finite() else None
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    return v


def check_table(table: str) -> str:
    if not _SAFE_TABLE.match(table):
        raise ValueError(f"unsafe table name: {table!r}")
    return table


def fetch(cur, max_rows: int) -> tuple:
    cols = [d[0] for d in cur.description]
    rows = [[safe_value(v) for v in r] for r in cur.fetchmany(max_rows)]
    return cols, rows


_UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40, "PiB": 2**50, "EiB": 2**60}
_SIZE = re.compile(r"sizeInBytes=([\d.]+)\s*(EiB|PiB|TiB|GiB|MiB|KiB|B)")
_ROWS = re.compile(r"rowCount=([\d.E+]+)")


def parse_spark_cost(plan_text: str) -> Estimate | None:
    """Parse the top node of a Spark/Databricks `EXPLAIN COST` plan: 'Statistics(sizeInBytes=1.2 GiB, rowCount=...)'."""
    m = _SIZE.search(plan_text)
    if not m:
        return None
    r = _ROWS.search(plan_text[m.start():m.start() + 200])
    return Estimate(int(float(m.group(1)) * _UNITS[m.group(2)]), int(float(r.group(1))) if r else None)
