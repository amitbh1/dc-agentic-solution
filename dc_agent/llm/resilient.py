from __future__ import annotations

import time

from ..observability.metrics import LLM_SECONDS
from ..query.plan import PlannerOutput
from ..resilience import Bulkhead, CircuitBreaker, is_retryable_error, is_transient_error, retry
from .base import PlanRequest


class ResilientLLM:
    """Concurrency cap, retries with backoff on transient errors, and a circuit breaker."""

    def __init__(self, inner, model: str, bulkhead: Bulkhead, breaker: CircuitBreaker, attempts: int = 3):
        self._inner = inner
        self._model = model
        self._bulkhead = bulkhead
        self._breaker = breaker
        self._attempts = attempts

    def plan(self, req: PlanRequest) -> PlannerOutput:
        t0 = time.perf_counter()
        try:
            with self._bulkhead.slot():
                return self._breaker.call(
                    lambda: retry(lambda: self._inner.plan(req), attempts=self._attempts, retry_on=is_retryable_error),
                    count_if=is_transient_error,
                )
        finally:
            LLM_SECONDS.labels(self._model).observe(time.perf_counter() - t0)
