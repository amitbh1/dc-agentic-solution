"""Promote well-rated answers to verified queries (few-shot examples and golden-set candidates).

    python -m dc_agent.jobs.promote --dry-run          # list candidates
    python -m dc_agent.jobs.promote --min-up 3         # insert them

A candidate is one (release, question, plan) with at least --min-up thumbs-up from distinct users,
no thumbs-down, and no existing verified query for the same question and plan. Rows are inserted with
verified_by = 'auto:feedback' so the ontology owner can review or delete them; then run jobs.embed.
"""
from __future__ import annotations

import argparse
import sys

_CANDIDATES = """
WITH votes AS (
    SELECT release_id, q_hash, plan,
           max(question) AS question, max(sql) AS sql, max(dialect) AS dialect,
           count(DISTINCT user_id) FILTER (WHERE rating = 'up') AS ups,
           count(*) FILTER (WHERE rating = 'down') AS downs
      FROM agent.feedback
     WHERE plan IS NOT NULL AND outcome = 'ok'
     GROUP BY release_id, q_hash, plan
)
SELECT v.release_id, v.question, v.plan, v.sql, v.dialect, v.ups
  FROM votes v
 WHERE v.ups >= %(min_up)s AND v.downs = 0 AND v.question IS NOT NULL
   AND EXISTS (SELECT 1 FROM ontology.release r WHERE r.id = v.release_id)
   AND NOT EXISTS (
        SELECT 1 FROM ontology.verified_query q
         WHERE q.plan = v.plan
           AND lower(regexp_replace(btrim(q.question), '\\s+', ' ', 'g'))
             = lower(regexp_replace(btrim(v.question), '\\s+', ' ', 'g')))
 ORDER BY v.ups DESC
"""

_INSERT = """
INSERT INTO ontology.verified_query (question, plan, sql, dialect, release_id, verified_by)
VALUES (%s, %s, %s, %s, %s, 'auto:feedback')
"""


def run(dsn: str, min_up: int = 3, dry_run: bool = False) -> list:
    import psycopg
    from psycopg.types.json import Jsonb

    with psycopg.connect(dsn) as conn:
        rows = conn.execute(_CANDIDATES, {"min_up": min_up}).fetchall()
        if not dry_run:
            with conn.cursor() as cur:
                cur.executemany(_INSERT, [(q, Jsonb(p), s, d, r) for r, q, p, s, d, _ in rows])
            conn.commit()
    return [{"release": r, "question": q, "ups": u} for r, q, _, _, _, u in rows]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-up", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from ..core.config import Settings

    s = Settings.from_env()
    if not s.pg_dsn:
        print("DC_PG_DSN is required", file=sys.stderr)
        return 2
    found = run(s.pg_dsn, args.min_up, args.dry_run)
    for c in found:
        print(f"{c['ups']:>3} up  release {c['release']}  {c['question']}")
    print(f"{'would promote' if args.dry_run else 'promoted'} {len(found)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
