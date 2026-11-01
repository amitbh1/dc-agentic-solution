from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..query.plan import PlannerOutput


@dataclass(frozen=True)
class PlanRequest:
    question: str
    catalogue: dict  # ontology slice the user may see
    examples: list = field(default_factory=list)
    error: str | None = None  # why the previous attempt in this turn failed
    failed_plan: dict | None = None  # the plan that failed, so the model can repair it
    previous_plan: dict | None = None  # last answered plan; set only for follow-up questions


class LLM(Protocol):
    def plan(self, req: PlanRequest) -> PlannerOutput: ...
