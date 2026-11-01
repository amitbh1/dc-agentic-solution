from __future__ import annotations

import threading
import time
from typing import Callable


class KeyedLock:
    """Allows one in-flight request per key. Locks expire so a dropped stream cannot wedge a session."""

    def __init__(self, ttl_s: float = 120.0, now: Callable = time.monotonic):
        self._ttl = ttl_s
        self._now = now
        self._held: dict = {}
        self._lock = threading.Lock()

    def try_acquire(self, key: str) -> bool:
        with self._lock:
            at = self._held.get(key)
            if at is not None and self._now() - at < self._ttl:
                return False
            self._held[key] = self._now()
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._held.pop(key, None)
