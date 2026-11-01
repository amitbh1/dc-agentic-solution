"""Compute missing embeddings for synonyms and verified-query questions.

    python -m dc_agent.jobs.embed            # fill NULL embeddings
    python -m dc_agent.jobs.embed --all      # recompute everything (after changing the embedding model)

Run after ontology edits and after jobs.promote. Vectors must match vector(DC_EMBED_DIM) in the schema.
"""
from __future__ import annotations

import argparse
import sys

BATCH = 64
_TARGETS = [
    ("ontology.synonym", "term", "embedding"),
    ("ontology.verified_query", "question", "question_embedding"),
]


def _literal(vec) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def run(dsn: str, embedder, recompute: bool = False) -> dict:
    import psycopg

    counts = {}
    with psycopg.connect(dsn) as conn:
        for table, text_col, vec_col in _TARGETS:
            where = "" if recompute else f" WHERE {vec_col} IS NULL"
            rows = conn.execute(f"SELECT id, {text_col} FROM {table}{where} ORDER BY id").fetchall()
            for i in range(0, len(rows), BATCH):
                batch = rows[i:i + BATCH]
                vectors = embedder.embed([r[1] for r in batch], kind="document")
                with conn.cursor() as cur:
                    cur.executemany(
                        f"UPDATE {table} SET {vec_col} = %s::vector WHERE id = %s",
                        [(_literal(v), r[0]) for r, v in zip(batch, vectors)],
                    )
                conn.commit()
            counts[table] = len(rows)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="recompute every embedding")
    args = parser.parse_args(argv)

    from ..bootstrap import _embedder
    from ..core.config import Settings

    s = Settings.from_env()
    embedder = _embedder(s)
    if embedder is None or not s.pg_dsn:
        print("needs DC_PG_DSN and DC_EMBEDDINGS other than off", file=sys.stderr)
        return 2
    for table, n in run(s.pg_dsn, embedder, args.all).items():
        print(f"{table}: embedded {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
