from __future__ import annotations

import re

from ..query.plan import Filter, PlannerOutput, QueryPlan, TimeGrain, TimeRange
from .base import PlanRequest

_UNITS = {"hour": "hour", "hours": "hour", "day": "day", "days": "day", "week": "week", "weeks": "week",
          "month": "month", "months": "month"}
_GRAINS = {"hourly": "hour", "daily": "day", "weekly": "week", "monthly": "month"}


class RuleBasedLLM:
    """Offline stand-in for demos and tests. NOT an LLM: simple pattern matching over the catalogue."""

    _CUT = re.compile(r"\s+(?:for|in|where|with|top|limit|over|during)\s+|,")
    _SKIP_VALUES = {"and", "for", "in", "with", "by", "per", "is", "of", "the", "each"}

    @staticmethod
    def _words(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower())}

    def _metrics(self, question: str, catalogue: dict) -> list:
        """Every catalogue metric whose synonym phrase (or its first and last word) is in the question."""
        q, words = question.lower(), self._words(question)
        chosen = []
        for m in catalogue["metrics"]:
            for term in m.get("synonyms", []):
                tw = term.lower().split()
                if term.lower() in q or (len(tw) > 1 and tw[0].rstrip("s") in words and tw[-1].rstrip("s") in words):
                    first = re.search(rf"\b{re.escape(tw[0].rstrip('s'))}", q)
                    chosen.append((first.start() if first else len(q), m["id"]))
                    break
        return [mid for _, mid in sorted(chosen)]  # in the order the question names them

    def plan(self, req: PlanRequest) -> PlannerOutput:
        if req.error:
            return PlannerOutput(clarification="I couldn't build a valid query for that. Could you rephrase it?")
        question = req.question
        prev = req.previous_plan
        metrics = self._metrics(question, req.catalogue) or (
            list(prev["metrics"]) if prev else [req.catalogue["metrics"][0]["id"]]
        )
        attrs = req.catalogue["attributes"]
        time_attr = next((a["ref"] for a in attrs if a["role"] == "time"), None)

        time_grain = time_range = None
        last = re.search(r"\b(?:last|past)\s+(\d+)?\s*(hours?|days?|weeks?|months?)\b", question, re.I)
        if last and time_attr:
            time_range = TimeRange(ref=time_attr, last=int(last.group(1) or 1), unit=_UNITS[last.group(2).lower()])
        grain = re.search(r"\b(?:by|per|each)\s+(hour|day|week|month)\b|\b(hourly|daily|weekly|monthly)\b", question, re.I)
        if grain and time_attr:
            time_grain = TimeGrain(ref=time_attr, grain=(grain.group(1) or _GRAINS[grain.group(2).lower()]).lower())

        dims: list = []
        rest = question
        m = re.search(r"\b(by|per|each)\s+([a-z0-9_ ]+)", question.lower())
        if m:
            seg = self._CUT.split(m.group(2))[0].strip()
            for a in attrs:
                if a["role"] == "time":
                    continue
                terms = [a["name"]] + a["synonyms"]
                if any(re.search(rf"\b{re.escape(t.lower())}s?\b", seg) for t in terms):
                    dims.append(a["ref"])
            rest = re.sub(re.escape(f"{m.group(1)} {seg}"), " ", question, flags=re.I)

        filters = []
        for a in attrs:
            if a["ref"] in dims or a["role"] == "time":
                continue
            for t in [a["name"]] + a["synonyms"]:
                fm = re.search(rf"\b{re.escape(t)}\s+(?:is\s+|=\s*|of\s+)?([A-Za-z0-9][A-Za-z0-9\-]*)", rest, re.I)
                if fm and fm.group(1).lower() not in self._SKIP_VALUES:
                    filters.append(Filter(ref=a["ref"], op="=", values=[fm.group(1)]))
                    break

        limit, order = 100, None
        top = re.search(r"\btop\s+(\d+)", question, re.I)
        if top:
            limit, order = int(top.group(1)), metrics[0]

        if prev:
            # Follow-up: keep the previous breakdown, filters and period, override what the question names.
            dims = dims or list(prev.get("dimensions", []))
            named = {f.ref for f in filters}
            filters = [Filter(**f) for f in prev.get("filters", []) if f["ref"] not in named] + filters
            if time_range is None and prev.get("time_range"):
                time_range = TimeRange(**prev["time_range"])
            if time_grain is None and prev.get("time_grain"):
                time_grain = TimeGrain(**prev["time_grain"])
        return PlannerOutput(plan=QueryPlan(
            metrics=metrics, dimensions=dims, filters=filters, time_range=time_range, time_grain=time_grain,
            order_by_metric=order, limit=limit,
        ))
