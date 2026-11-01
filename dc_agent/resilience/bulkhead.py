from __future__ import annotations

import threading
from contextlib import contextmanager

from ..core.errors import Overloaded


class Bulkhead:
    """Caps concurrent calls to one dependency so it cannot starve the rest."""

    def __init__(self, name: str, limit: int, wait_s: float = 5.0):
        self.name = name
        self._sem = threading.BoundedSemaphore(limit)
        self._wait_s = wait_s

    @contextmanager
    def slot(self):
        if not self._sem.acquire(timeout=self._wait_s):
            raise Overloaded(self.name)
        try:
            yield
        finally:
            self._sem.release()
