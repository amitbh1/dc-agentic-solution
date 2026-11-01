"""Data retention: delete old conversation checkpoints and feedback.

    python -m dc_agent.jobs.retention --dry-run   # report what would be deleted
    python -m dc_agent.jobs.retention             # delete

Checkpoints hold questions and result rows. A thread is deleted when its newest checkpoint is older
than DC_CHECKPOINT_RETENTION_DAYS (default 30). Feedback older than DC_FEEDBACK_RETENTION_DAYS
(default 365) is deleted too. Run daily.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

_STALE_THREADS = """
SELECT thread_id
  FROM checkpoints
 GROUP BY thread_id
HAVING max((checkpoint->>'ts')::timestamptz) < %s
"""


def stale_threads(conn, cutoff: dt.datetime) -> list:
    return [r[0] for r in conn.execute(_STALE_THREADS, (cutoff,)).fetchall()]


def run(dsn: str, checkpoint_days: int, feedback_days: int, dry_run: bool = False, now=None) -> dict:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    now = now or dt.datetime.now(dt.timezone.utc)
    cp_cutoff = now - dt.timedelta(days=checkpoint_days)
    fb_cutoff = now - dt.timedelta(days=feedback_days)
    with psycopg.connect(dsn, autocommit=True) as conn:
        threads = stale_threads(conn, cp_cutoff)
        feedback = conn.execute("SELECT count(*) FROM agent.feedback WHERE created_at < %s", (fb_cutoff,)).fetchone()[0]
        if not dry_run:
            saver = PostgresSaver(conn)
            for thread_id in threads:
                saver.delete_thread(thread_id)
            conn.execute("DELETE FROM agent.feedback WHERE created_at < %s", (fb_cutoff,))
    return {"threads": len(threads), "feedback": feedback, "dry_run": dry_run}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from ..core.config import Settings

    s = Settings.from_env()
    if not s.pg_dsn:
        print("DC_PG_DSN is required", file=sys.stderr)
        return 2
    out = run(s.pg_dsn, s.checkpoint_retention_days, s.feedback_retention_days, args.dry_run)
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} {out['threads']} conversation threads and {out['feedback']} feedback rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
