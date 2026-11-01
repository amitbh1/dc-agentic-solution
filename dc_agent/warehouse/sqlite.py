from __future__ import annotations

import sqlite3
import threading
import time

import sqlglot
from sqlglot import exp

from .base import Estimate, check_table, fetch

DEMO_ROW_BYTES = 64  # SQLite has no byte estimate; demo heuristic for the cost gate


class SqliteExecutor:
    """Demo executor. The connection should be read-only (PRAGMA query_only = ON)."""

    def __init__(self, conn: sqlite3.Connection, dialect: str = "sqlite"):
        self._conn = conn
        self._lock = threading.Lock()
        self.dialect = dialect

    def run(self, sql, max_rows, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self._lock:
            # SQLite has no statement timeout; abort from the progress handler instead.
            self._conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
            try:
                return fetch(self._conn.execute(sql), max_rows)
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc):
                    raise TimeoutError(f"query exceeded {timeout_s}s") from exc
                raise
            finally:
                self._conn.set_progress_handler(None, 0)

    def columns(self, table):
        with self._lock:
            info = self._conn.execute(f'PRAGMA table_info("{check_table(table)}")').fetchall()
        return {r[1].lower(): r[2] for r in info}

    def estimate(self, sql):
        """Rows in every table the query reads, times a nominal row size. Rough, but monotonic in data size."""
        tables = {check_table(t.name) for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}
        rows = 0
        with self._lock:
            known = {r[0] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            for t in tables & known:
                rows += self._conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        return Estimate(rows * DEMO_ROW_BYTES, rows)
