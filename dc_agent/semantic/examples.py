"""Few-shot example selection from verified queries."""
from __future__ import annotations

from .embeddings import cosine, normalise
from .text import tokens


def plan_refs(plan: dict) -> tuple:
    """(metric ids, attribute refs) a plan uses."""
    refs = set(plan.get("dimensions", []))
    refs |= {f["ref"] for f in plan.get("filters", [])}
    for key in ("time_range", "time_grain"):
        if plan.get(key):
            refs.add(plan[key]["ref"])
    return set(plan.get("metrics", [])), refs


def select_examples(examples: list, catalogue: dict, question: str, k: int = 3, qvec=None) -> list:
    """Up to k verified examples relevant to the question.

    An example is offered only if every metric and attribute it uses is in the user's catalogue
    slice, so examples cannot reveal anything the role may not see.
    """
    if k <= 0 or not examples:
        return []
    metric_ids = {m["id"] for m in catalogue.get("metrics", [])}
    attr_refs = {a["ref"] for a in catalogue.get("attributes", [])}
    q = tokens(question)
    qn = normalise(qvec) if qvec else None
    scored = []
    for i, ex in enumerate(examples):
        metrics, refs = plan_refs(ex.plan)
        if not metrics or not metrics <= metric_ids or not refs <= attr_refs:
            continue
        score = len(q & tokens(ex.question)) + len(metrics & metric_ids) * 0.1
        if qn and ex.vector:
            score += 3 * cosine(qn, normalise(ex.vector))
        scored.append((-score, i, ex))
    scored.sort()
    return [{"question": ex.question, "plan": ex.plan} for _, _, ex in scored[:k]]
