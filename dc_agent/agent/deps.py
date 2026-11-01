from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..core.config import Settings
from ..llm import LLM
from ..memory import PlanCache
from ..security import BlockTracker, make_redactor
from ..semantic import OntologyProvider


@dataclass
class Deps:
    """Everything the graph nodes need. Built once per process by bootstrap.build_agent."""

    provider: OntologyProvider
    llm: LLM
    executors: dict  # engine name (concept.source) -> executor
    cache: PlanCache
    settings: Settings
    blocks: BlockTracker = field(default_factory=BlockTracker)
    embedder: object | None = None  # semantic grounding; None means lexical only
    tools: object | None = None  # ToolRegistry; enables the enrich step when it has tools
    redactor: Callable | None = None

    def __post_init__(self):
        if self.redactor is None:
            self.redactor = make_redactor(self.settings.redact_patterns)

    @property
    def dialects(self) -> dict:
        return {name: getattr(ex, "dialect", name) for name, ex in self.executors.items()}
