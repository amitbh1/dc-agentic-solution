"""Text embeddings for semantic grounding and few-shot example selection.

Grounding never depends on embeddings being available: any failure falls back to lexical matching.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Protocol

from ..observability import audit
from ..observability.metrics import EMBED_SECONDS
from ..resilience import Bulkhead, CircuitBreaker, is_retryable_error, is_transient_error, retry


class Embedder(Protocol):
    dim: int
    min_similarity: float  # cosine at or above which a synonym counts as a semantic match

    def embed(self, texts: list, kind: str = "query") -> list: ...


def normalise(v) -> tuple:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def cosine(a, b) -> float:
    """Both vectors must already be normalised."""
    return sum(x * y for x, y in zip(a, b))


class HashingEmbedder:
    """Deterministic, offline embedder: hashed character trigrams of each word.

    Catches spelling variants and inflections ("averge powr"), not synonyms ("wattage"). Used in
    demo mode and tests; use VertexEmbedder for real semantic matching.
    """

    min_similarity = 0.5

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _one(self, text: str) -> tuple:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                h = int.from_bytes(hashlib.blake2b(padded[i:i + 3].encode(), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0 if (h >> 63) else -1.0
        return normalise(v)

    def embed(self, texts, kind="query"):
        return [self._one(t) for t in texts]


class VertexEmbedder:
    """Vertex AI text embeddings (default text-embedding-005, 768 dimensions). Untested against a live project."""

    min_similarity = 0.75

    def __init__(self, model: str, project: str, location: str, dim: int = 768, timeout_s: int = 10):
        from google import genai
        from google.genai import types

        self._types = types
        self._model = model
        self.dim = dim
        self._client = genai.Client(
            vertexai=True, project=project, location=location,
            http_options=types.HttpOptions(timeout=timeout_s * 1000),
        )

    def embed(self, texts, kind="query"):
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        resp = self._client.models.embed_content(
            model=self._model, contents=list(texts),
            config=self._types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
        )
        audit("embed_call", model=self._model, n=len(texts))
        return [normalise(e.values) for e in resp.embeddings]


class ResilientEmbedder:
    """Concurrency cap, retries and a circuit breaker around an embedder."""

    def __init__(self, inner, bulkhead: Bulkhead, breaker: CircuitBreaker, attempts: int = 2):
        self._inner = inner
        self._bulkhead = bulkhead
        self._breaker = breaker
        self._attempts = attempts
        self.dim = inner.dim
        self.min_similarity = inner.min_similarity

    def embed(self, texts, kind="query"):
        t0 = time.perf_counter()
        try:
            with self._bulkhead.slot():
                return self._breaker.call(
                    lambda: retry(lambda: self._inner.embed(texts, kind), attempts=self._attempts,
                                  retry_on=is_retryable_error),
                    count_if=is_transient_error,
                )
        finally:
            EMBED_SECONDS.observe(time.perf_counter() - t0)


def with_min_similarity(embedder, min_sim: float):
    """Override the embedder's match threshold (DC_SEMANTIC_MIN_SIM)."""
    if min_sim:
        embedder.min_similarity = min_sim
    return embedder
