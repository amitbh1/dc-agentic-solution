from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_ROWS_HARD = 10_000
Grain = Literal["hour", "day", "week", "month"]


def parse_when(value: str) -> dt.datetime:
    """ISO date or datetime; naive values are UTC."""
    try:
        d = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"not an ISO date or datetime: {value!r}") from exc
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


class Filter(BaseModel):
    ref: str = Field(description="Attribute reference, concept.attribute")
    op: Literal["=", "!=", ">", ">=", "<", "<=", "in"]
    values: list[str] = Field(min_length=1, description="Values as strings; one value unless op is in")


class TimeRange(BaseModel):
    """Either a rolling window ending now (`last` + `unit`) or absolute bounds (`start`, `end`)."""

    ref: str = Field(description="Time attribute reference, concept.attribute")
    last: int | None = Field(default=None, ge=1, le=10_000, description="Rolling window size, e.g. 7 for 'last 7 days'")
    unit: Grain = "day"
    start: str | None = Field(default=None, description="ISO date or datetime, inclusive")
    end: str | None = Field(default=None, description="ISO date or datetime, exclusive")

    @field_validator("start", "end")
    @classmethod
    def _iso(cls, v):
        if v is not None:
            parse_when(v)
        return v

    @model_validator(mode="after")
    def _shape(self):
        if self.last is None and self.start is None and self.end is None:
            raise ValueError("time_range needs last, start or end")
        if self.last is not None and (self.start or self.end):
            raise ValueError("use either last or start/end, not both")
        if self.start and self.end and parse_when(self.start) >= parse_when(self.end):
            raise ValueError("time_range start must be before end")
        return self


class TimeGrain(BaseModel):
    """Bucket a time attribute for trends, e.g. 'by day'."""

    ref: str = Field(description="Time attribute reference, concept.attribute")
    grain: Grain


class QueryPlan(BaseModel):
    metrics: list[str] = Field(min_length=1, description="Metric ids; may have different base concepts")
    dimensions: list[str] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    time_range: TimeRange | None = None
    time_grain: TimeGrain | None = None
    order_by_metric: str | None = None
    descending: bool = True
    limit: int = Field(default=100, ge=1, le=MAX_ROWS_HARD)


class PlannerOutput(BaseModel):
    plan: QueryPlan | None = None
    clarification: str | None = Field(
        default=None, description="One short question when the request is ambiguous or unanswerable"
    )
