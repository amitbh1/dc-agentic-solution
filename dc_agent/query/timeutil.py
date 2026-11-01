"""Time ranges and buckets, rendered per dialect."""
from __future__ import annotations

import calendar
import datetime as dt

from sqlglot import exp

from .plan import TimeRange, parse_when

_DELTA = {"hour": dt.timedelta(hours=1), "day": dt.timedelta(days=1), "week": dt.timedelta(weeks=1)}
_SQLITE_BUCKET = {
    "hour": "%Y-%m-%dT%H:00:00",
    "day": "%Y-%m-%d",
    "month": "%Y-%m-01",
}


def _months_back(d: dt.datetime, n: int) -> dt.datetime:
    month = d.month - 1 - n
    year = d.year + month // 12
    month = month % 12 + 1
    return d.replace(year=year, month=month, day=min(d.day, calendar.monthrange(year, month)[1]))


def resolve_range(tr: TimeRange, now: dt.datetime) -> tuple:
    """(start, end) as aware UTC datetimes; either may be None. Rolling windows end at `now`."""
    if tr.last is not None:
        start = _months_back(now, tr.last) if tr.unit == "month" else now - tr.last * _DELTA[tr.unit]
        return start, now
    return (parse_when(tr.start) if tr.start else None), (parse_when(tr.end) if tr.end else None)


def is_date_type(data_type: str) -> bool:
    return data_type.lower().split("(")[0].strip() == "date"


def time_literal(value: dt.datetime, data_type: str, dialect: str) -> exp.Expression:
    """SQLite stores timestamps as ISO-8601 text ('YYYY-MM-DDTHH:MM:SS'), so compare as strings;
    other engines get a typed CAST."""
    value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    if is_date_type(data_type):
        text = value.strftime("%Y-%m-%d")
        return exp.Literal.string(text) if dialect == "sqlite" else exp.cast(exp.Literal.string(text), "DATE")
    if dialect == "sqlite":
        return exp.Literal.string(value.strftime("%Y-%m-%dT%H:%M:%S"))
    return exp.cast(exp.Literal.string(value.strftime("%Y-%m-%d %H:%M:%S")), "TIMESTAMP")


def bucket(col: exp.Expression, grain: str, dialect: str) -> exp.Expression:
    if dialect == "sqlite":
        if grain == "week":  # Monday on or before the timestamp
            return exp.Anonymous(this="date", expressions=[col, exp.Literal.string("-6 days"), exp.Literal.string("weekday 1")])
        return exp.Anonymous(this="strftime", expressions=[exp.Literal.string(_SQLITE_BUCKET[grain]), col])
    return exp.Anonymous(this="DATE_TRUNC", expressions=[exp.Literal.string(grain.upper()), col])
