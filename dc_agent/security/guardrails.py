from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import deque
from typing import Callable

from ..core.errors import InputBlocked

MAX_QUESTION_CHARS = 1000

_INJECTION = [
    re.compile(p, re.I)
    for p in (
        r"(ignore|disregard) (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
        r"(reveal|show|print|repeat|output) (me )?(your |the )?(system|hidden|initial) (prompt|instructions)",
        r"\b(drop|truncate|alter)\s+table\b",
        r";\s*(drop|delete|update|insert)\b",
        r"\bunion\s+(all\s+)?select\b",
    )
]
_ZERO_WIDTH = re.compile("[​-‏⁠﻿]")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return " ".join(text.split())


def check_input(question: str) -> None:
    """Heuristic screen. Weak on its own: the real defences are the typed plan,
    the compiler, the SQL guard, read-only credentials and row scoping."""
    if not question or not question.strip():
        raise InputBlocked("empty question")
    if len(question) > MAX_QUESTION_CHARS:
        raise InputBlocked("question too long")
    cleaned = _normalise(question)
    for pattern in _INJECTION:
        if pattern.search(cleaned):
            raise InputBlocked(f"matched pattern: {pattern.pattern[:40]}")


class BlockTracker:
    """Counts blocked inputs per user in a sliding window so repeated probing can raise an alert."""

    def __init__(self, threshold: int = 5, window_s: float = 300.0, now: Callable = time.monotonic):
        self._threshold = threshold
        self._window = window_s
        self._now = now
        self._events: dict = {}
        self._lock = threading.Lock()

    def record(self, user_id: str) -> bool:
        with self._lock:
            t = self._now()
            q = self._events.setdefault(user_id, deque())
            q.append(t)
            while q and t - q[0] > self._window:
                q.popleft()
            if len(self._events) > 10_000:
                self._events = {u: d for u, d in self._events.items() if d}
            return len(q) >= self._threshold
