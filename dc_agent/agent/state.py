from __future__ import annotations

from typing import TypedDict

OUTCOMES = ("ok", "clarification", "blocked", "denied", "cancelled", "error")  # plus approval_required (API only)


class AgentState(TypedDict, total=False):
    question: str
    user: dict
    deadline: float  # wall-clock epoch seconds; the turn's end-to-end budget
    slice: dict | None
    examples: list  # verified few-shot examples chosen for this question
    plan: dict | None
    clarification: str | None
    sql: str | None  # every executed statement, for audit and the admin response
    dialect: str | None
    tables: list
    parts: list  # [{engine, dialect, sql, tables, columns}]; two parts for a cross-engine plan
    merge: dict | None  # cross-engine merge spec
    estimate_bytes: int | None
    expected_columns: list  # output names from the compiler
    error: str | None
    error_fatal: bool
    error_kind: str | None
    user_message: str | None
    attempts: int
    columns: list
    rows: list
    warnings: list
    notes: list  # enrichment tool notes
    answer: str | None
    release_id: int | None
    last_plan: dict | None  # carried across turns for follow-up questions


def fresh_turn() -> dict:
    """Reset per-turn fields: checkpointed state would otherwise leak between turns."""
    return {
        "slice": None, "examples": [], "plan": None, "clarification": None, "sql": None, "dialect": None,
        "tables": [], "parts": [], "merge": None, "estimate_bytes": None, "expected_columns": [], "error": None, "error_fatal": False,
        "error_kind": None, "user_message": None, "attempts": 0, "columns": [], "rows": [], "warnings": [],
        "notes": [], "answer": None, "release_id": None,
    }


def outcome_of(state: dict) -> str:
    """ok | clarification | blocked | denied | cancelled | error"""
    if state.get("clarification"):
        return "clarification"
    if state.get("error") or not state.get("sql"):
        return {"blocked": "blocked", "AccessError": "denied", "cancelled": "cancelled"}.get(
            state.get("error_kind"), "error")
    return "ok"
