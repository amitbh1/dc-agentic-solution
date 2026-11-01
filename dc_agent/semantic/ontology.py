from __future__ import annotations

import heapq
from collections import namedtuple

from ..core.errors import AccessError, PlanError
from ..security.identity import UserContext
from .embeddings import cosine, normalise
from .models import SENS, Attribute, Concept
from .text import tokens

MAX_SLICE_ATTRS = 80

Step = namedtuple("Step", "cur nxt cur_key nxt_key fan")


class Ontology:
    def __init__(self, concepts, edges, metrics, synonyms=(), grants=(), release_id: int = 0, examples=()):
        self.concepts = {c.name: c for c in concepts}
        self.metrics = {m.id: m for m in metrics if m.concept in self.concepts}
        self.synonyms = list(synonyms)
        self.grants = {(g.role, g.concept): g for g in grants if g.concept in self.concepts}
        self.release_id = release_id
        self.examples = list(examples)
        self.adj: dict = {}
        self.edges = []
        for e in edges:
            if e.left not in self.concepts or e.right not in self.concepts:
                continue
            self.edges.append(e)
            fwd_fan = e.cardinality == "one_to_many"
            rev_fan = e.cardinality == "many_to_one"
            self.adj.setdefault(e.left, []).append((e.right, e.left_key, e.right_key, fwd_fan, e.priority))
            self.adj.setdefault(e.right, []).append((e.left, e.right_key, e.left_key, rev_fan, e.priority))
        for steps in self.adj.values():
            steps.sort(key=lambda s: (s[4], s[0], s[1], s[2]))
        # lexical and semantic grounding indexes, built once per release
        self._metric_words = {m.id: tokens(m.id.replace("_", " ")) | tokens(m.description) for m in self.metrics.values()}
        self._metric_syn: dict = {m: set() for m in self.metrics}
        self._metric_vecs: dict = {m: [] for m in self.metrics}
        for s in self.synonyms:
            if s.kind == "metric" and s.target in self.metrics:
                self._metric_syn[s.target] |= tokens(s.term)
                if s.vector:
                    self._metric_vecs[s.target].append(normalise(s.vector))
        self.has_vectors = any(self._metric_vecs.values()) or any(e.vector for e in self.examples)

    def attr(self, ref: str):
        concept, _, name = ref.partition(".")
        c = self.concepts.get(concept)
        if c is None or name not in c.attrs:
            raise PlanError(f"unknown attribute reference: {ref}")
        return c, c.attrs[name]

    def check_access(self, user: UserContext, concept: Concept, attr: Attribute) -> None:
        g = self.grants.get((user.role, concept.name))
        if g is None or SENS[attr.sensitivity] > SENS[g.max_sensitivity]:
            raise AccessError(f"{user.role} may not read {concept.name}.{attr.name}")

    def path(self, start: str, goal: str) -> list:
        """Join path with the fewest hops; ties go to the lowest total edge priority, then by name."""
        best: dict = {start: (0, 0)}
        prev: dict = {start: None}
        heap = [(0, 0, start)]
        while heap:
            hops, prio, cur = heapq.heappop(heap)
            if (hops, prio) > best.get(cur, (hops, prio)):
                continue
            if cur == goal:
                break
            for nxt, cur_key, nxt_key, fan, p in self.adj.get(cur, []):
                cost = (hops + 1, prio + p)
                if nxt not in best or cost < best[nxt]:
                    best[nxt] = cost
                    prev[nxt] = Step(cur, nxt, cur_key, nxt_key, fan)
                    heapq.heappush(heap, (*cost, nxt))
        if goal not in prev:
            raise PlanError(f"no join path from {start} to {goal}")
        steps = []
        node = goal
        while prev[node] is not None:
            steps.append(prev[node])
            node = prev[node].cur
        return steps[::-1]

    def reachable(self, start: str, depth: int = 3) -> set:
        seen = {start}
        frontier = [start]
        for _ in range(depth):
            nxt_frontier = []
            for cur in frontier:
                for nxt, *_ in self.adj.get(cur, []):
                    if nxt not in seen:
                        seen.add(nxt)
                        nxt_frontier.append(nxt)
            frontier = nxt_frontier
        return seen

    def allowed_concepts(self, user: UserContext) -> set:
        return {c for (role, c) in self.grants if role == user.role}

    def score_metrics(self, question: str, user: UserContext, qvecs=None, min_sim: float = 1.0) -> list:
        """[(score, metric)] for metrics the role may access, best first.

        Lexical: |question ∩ id/description tokens| + 2·|question ∩ synonym tokens|.
        Semantic (when synonyms are embedded): + 4·s, where s is the best cosine between any question
        vector (the whole question and its short windows) and any synonym of the metric, if s >= min_sim.
        With min_sim >= 0.5 a semantic match alone qualifies (score >= 2), like a lexical synonym hit.
        """
        q = tokens(question)
        allowed = self.allowed_concepts(user)
        qns = [normalise(v) for v in qvecs] if qvecs else []
        scored = []
        for m in self.metrics.values():
            if m.concept not in allowed:
                continue
            score = float(len(q & self._metric_words[m.id]) + 2 * len(q & self._metric_syn[m.id]))
            if qns and self._metric_vecs[m.id]:
                sim = max(cosine(qn, v) for qn in qns for v in self._metric_vecs[m.id])
                if sim >= min_sim:
                    score += 4 * sim
            if score >= 2:
                scored.append((score, m))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return scored

    def ground(self, question: str, user: UserContext, qvecs=None, min_sim: float = 1.0) -> dict | None:
        """Pick up to 3 metrics within 60% of the best score and build the role-filtered catalogue slice."""
        scored = self.score_metrics(question, user, qvecs, min_sim)
        if not scored:
            return None
        best = scored[0][0]
        return self._slice([m for s, m in scored if s >= best * 0.6][:3], user)

    def slice_for(self, metric_ids: list, user: UserContext) -> dict | None:
        """Catalogue slice for known metric ids, used to ground follow-up questions on the previous plan."""
        allowed = self.allowed_concepts(user)
        chosen = [self.metrics[i] for i in metric_ids if i in self.metrics and self.metrics[i].concept in allowed]
        return self._slice(chosen, user) if chosen else None

    def _slice(self, chosen: list, user: UserContext) -> dict:
        allowed = self.allowed_concepts(user)
        concepts: set = set()
        for m in chosen:
            concepts |= self.reachable(m.concept)
        attrs = []
        for cname in sorted(concepts & allowed):
            g = self.grants[(user.role, cname)]
            for a in self.concepts[cname].attrs.values():
                if a.role not in ("dimension", "time") or SENS[a.sensitivity] > SENS[g.max_sensitivity]:
                    continue
                ref = f"{cname}.{a.name}"
                attrs.append(
                    {
                        "ref": ref,
                        "name": a.name,
                        "type": a.data_type,
                        "role": a.role,
                        "description": a.description,
                        "synonyms": [s.term for s in self.synonyms if s.kind == "attribute" and s.target == ref],
                    }
                )
        return {
            "release": self.release_id,
            "metrics": [
                {
                    "id": m.id, "description": m.description, "unit": m.unit, "concept": m.concept,
                    "synonyms": [s.term for s in self.synonyms if s.kind == "metric" and s.target == m.id],
                }
                for m in chosen
            ],
            "attributes": attrs[:MAX_SLICE_ATTRS],
        }

    def catalogue_hint(self, user: UserContext, limit: int = 8) -> str:
        allowed = self.allowed_concepts(user)
        names = [f"{m.id} ({m.description})" for m in self.metrics.values() if m.concept in allowed]
        return "; ".join(sorted(names)[:limit])
