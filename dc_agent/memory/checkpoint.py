from __future__ import annotations


def make_postgres_checkpointer(dsn: str):
    """Durable conversation state. Returns (saver, pool); close the pool on shutdown.

    Requires langgraph-checkpoint-postgres and psycopg[pool]. Set a retention job that deletes
    old checkpoints: conversation state contains questions and result rows.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(dsn, kwargs={"autocommit": True, "row_factory": dict_row}, open=True)
    saver = PostgresSaver(pool)
    saver.setup()
    return saver, pool
