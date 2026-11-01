from __future__ import annotations

import json

from .base import Estimate, check_table, fetch


class SnowflakeExecutor:
    """Use a read-only role and a dedicated warehouse; statement timeout is set per session."""

    dialect = "snowflake"

    def __init__(self, **connect_args):
        self._args = connect_args

    def _connect(self, timeout_s: int):
        import snowflake.connector

        params = {"STATEMENT_TIMEOUT_IN_SECONDS": int(timeout_s), "QUERY_TAG": "dc-agent"}
        return snowflake.connector.connect(session_parameters=params, **self._args)

    def run(self, sql, max_rows, timeout_s):
        conn = self._connect(timeout_s)
        try:
            cur = conn.cursor()
            cur.execute(sql, timeout=timeout_s)
            return fetch(cur, max_rows)
        finally:
            conn.close()

    def columns(self, table):
        conn = self._connect(60)
        try:
            cur = conn.cursor()
            cur.execute(f"DESCRIBE TABLE {check_table(table)}")
            return {r[0].lower(): r[1] for r in cur.fetchall()}
        finally:
            conn.close()

    def estimate(self, sql):
        """`EXPLAIN USING JSON`: GlobalStats.bytesAssigned is the bytes the plan will scan after pruning."""
        conn = self._connect(60)
        try:
            cur = conn.cursor()
            cur.execute(f"EXPLAIN USING JSON {sql}")
            row = cur.fetchone()
        finally:
            conn.close()
        stats = json.loads(row[0]).get("GlobalStats", {}) if row else {}
        if "bytesAssigned" not in stats:
            return None
        return Estimate(int(stats["bytesAssigned"]), None)
