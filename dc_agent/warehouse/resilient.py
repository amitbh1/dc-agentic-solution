from __future__ import annotations

import time

from ..observability.metrics import QUERY_SECONDS
from ..resilience import Bulkhead, CircuitBreaker, is_retryable_error, is_transient_error, retry


class ResilientExecutor:
    """Adds a concurrency cap, retries on transient connection errors and a circuit breaker.

    Only transient errors count towards the breaker, so a flood of bad queries cannot open it.
    """

    def __init__(self, inner, name: str, bulkhead: Bulkhead, breaker: CircuitBreaker, attempts: int = 2):
        self._inner = inner
        self._name = name
        self._bulkhead = bulkhead
        self._breaker = breaker
        self._attempts = attempts
        self.dialect = getattr(inner, "dialect", name)

    def _call(self, fn):
        with self._bulkhead.slot():
            return self._breaker.call(
                lambda: retry(fn, attempts=self._attempts, retry_on=is_retryable_error),
                count_if=is_transient_error,
            )

    def run(self, sql, max_rows, timeout_s):
        t0 = time.perf_counter()
        try:
            return self._call(lambda: self._inner.run(sql, max_rows, timeout_s))
        finally:
            QUERY_SECONDS.labels(self._name).observe(time.perf_counter() - t0)

    def estimate(self, sql):
        fn = getattr(self._inner, "estimate", None)
        return self._call(lambda: fn(sql)) if fn else None

    def columns(self, table):
        return self._inner.columns(table)
