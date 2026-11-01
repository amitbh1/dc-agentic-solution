from __future__ import annotations

import threading
import time
from typing import Callable

from ..core.errors import CircuitOpen
from ..observability.metrics import BREAKER_OPENED
from .retry import is_transient_error


class CircuitBreaker:
    """Opens after consecutive transient failures; lets one probe through after the reset window."""

    def __init__(self, name: str, failure_threshold: int = 5, reset_s: float = 30.0, now: Callable = time.monotonic):
        self.name = name
        self._threshold = failure_threshold
        self._reset_s = reset_s
        self._now = now
        self._failures = 0
        self._opened_at = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._now() - self._opened_at >= self._reset_s:
                self._opened_at = self._now()  # half-open: one probe per window
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                if self._opened_at is None:
                    BREAKER_OPENED.labels(self.name).inc()
                self._opened_at = self._now()

    def call(self, fn: Callable, count_if: Callable = is_transient_error):
        if not self.allow():
            raise CircuitOpen(self.name)
        try:
            result = fn()
        except Exception as exc:
            if count_if(exc):
                self.failure()
            raise
        self.success()
        return result
