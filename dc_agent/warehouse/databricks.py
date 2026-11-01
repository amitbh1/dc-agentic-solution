from __future__ import annotations

from .base import check_table, fetch, parse_spark_cost


class DatabricksExecutor:
    """Use a service principal with SELECT-only grants. Connects per query; add pooling for load.

    The STATEMENT_TIMEOUT setting is applied per session: verify it on your workspace.
    """

    dialect = "databricks"

    def __init__(self, host: str, http_path: str, token: str):
        self._args = dict(server_hostname=host, http_path=http_path, access_token=token)

    def run(self, sql, max_rows, timeout_s):
        from databricks import sql as dbsql

        with dbsql.connect(**self._args) as conn, conn.cursor() as cur:
            cur.execute(f"SET STATEMENT_TIMEOUT = {int(timeout_s)}")
            cur.execute(sql)
            return fetch(cur, max_rows)

    def columns(self, table):
        from databricks import sql as dbsql

        with dbsql.connect(**self._args) as conn, conn.cursor() as cur:
            cur.execute(f"DESCRIBE TABLE {check_table(table)}")
            out = {}
            for row in cur.fetchall():
                name = (row[0] or "").strip()
                if not name or name.startswith("#"):
                    break
                out[name.lower()] = row[1]
        return out

    def estimate(self, sql):
        """`EXPLAIN COST` statistics from the optimizer; quality depends on ANALYZE TABLE having run."""
        from databricks import sql as dbsql

        with dbsql.connect(**self._args) as conn, conn.cursor() as cur:
            cur.execute(f"EXPLAIN COST {sql}")
            text = "\n".join(str(r[0]) for r in cur.fetchall())
        return parse_spark_cost(text)
