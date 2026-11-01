"""User feedback on answers. Thumbs-up answers can be promoted to verified queries (jobs/promote.py).

Data minimisation: the question text is stored only with an "up" rating, because promotion needs
it; "down" ratings keep the question hash. Comments are free text from the user and are never logged.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import asdict, dataclass, field


@dataclass
class FeedbackRecord:
    user_id: str
    role: str
    thread_id: str
    rating: str  # up | down
    q_hash: str
    outcome: str
    tenant: str | None = None
    request_id: str | None = None
    comment: str | None = None
    question: str | None = None
    plan: dict | None = None
    sql: str | None = None
    dialect: str | None = None
    release_id: int | None = None
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class InMemoryFeedbackStore:
    def __init__(self, max_items: int = 10_000):
        self._items: list = []
        self._max = max_items
        self._lock = threading.Lock()

    def add(self, rec: FeedbackRecord) -> None:
        with self._lock:
            self._items.append(rec)
            del self._items[:-self._max]

    def all(self) -> list:
        with self._lock:
            return list(self._items)


_INSERT = """
INSERT INTO agent.feedback (user_id, role, tenant, thread_id, request_id, rating, comment, question,
                            q_hash, plan, sql, dialect, release_id, outcome, created_at)
VALUES (%(user_id)s, %(role)s, %(tenant)s, %(thread_id)s, %(request_id)s, %(rating)s, %(comment)s,
        %(question)s, %(q_hash)s, %(plan)s::jsonb, %(sql)s, %(dialect)s, %(release_id)s, %(outcome)s, %(created_at)s)
"""


class PostgresFeedbackStore:
    """Writes to agent.feedback (sql/ontology_schema.sql) through the shared connection pool."""

    def __init__(self, pool):
        self._pool = pool

    def add(self, rec: FeedbackRecord) -> None:
        row = asdict(rec)
        row["plan"] = json.dumps(rec.plan) if rec.plan is not None else None
        with self._pool.connection() as conn:
            conn.execute(_INSERT, row)
