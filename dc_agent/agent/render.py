"""Template-based answers: result rows never pass through an LLM."""
from __future__ import annotations


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def render_rows(columns: list, rows: list, limit: int = 10) -> str:
    if not rows:
        return "No rows matched."
    lines = [", ".join(f"{c}={_fmt(v)}" for c, v in zip(columns, r)) for r in rows[:limit]]
    more = f"\n... and {len(rows) - limit} more rows" if len(rows) > limit else ""
    return f"{len(rows)} row(s):\n" + "\n".join(lines) + more
