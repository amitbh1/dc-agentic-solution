"""Composition root: builds the agent and everything around it from Settings."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .agent import Deps, build_graph
from .core.config import Settings
from .core.errors import ConfigError
from .memory import InMemoryFeedbackStore, PlanCache
from .resilience import Bulkhead, CircuitBreaker, KeyedLock, RateLimiter
from .security import BlockTracker

log = logging.getLogger("dc_agent.bootstrap")


@dataclass
class Bundle:
    agent: object
    deps: Deps
    settings: Settings
    dev_users: dict
    limiter: object
    locks: object
    feedback: object
    closers: list = field(default_factory=list)

    def ready(self) -> bool:
        try:
            self.deps.provider.get()
            return True
        except Exception:
            log.exception("readiness check failed")
            return False

    def close(self) -> None:
        for fn in self.closers:
            try:
                fn()
            except Exception:
                log.exception("shutdown hook failed")


def _shared_backends(settings: Settings, closers: list) -> tuple:
    """Rate limiter, session lock and plan cache: Redis-backed with local fallback when DC_REDIS_URL is set."""
    local_limiter = RateLimiter(settings.rate_per_min, settings.rate_burst, settings.daily_budget,
                                settings.tenant_daily_budget)
    local_locks = KeyedLock(ttl_s=settings.request_timeout_s + 30)
    local_cache = PlanCache(settings.cache_ttl_s)
    if not settings.redis_url:
        return local_limiter, local_locks, local_cache

    import redis

    from .memory import RedisPlanCache
    from .resilience import FallbackKeyedLock, FallbackRateLimiter, RedisKeyedLock, RedisRateLimiter

    client = redis.Redis.from_url(settings.redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
    closers.append(client.close)
    limiter = FallbackRateLimiter(
        RedisRateLimiter(client, settings.rate_per_min, settings.rate_burst, settings.daily_budget,
                         settings.tenant_daily_budget),
        local_limiter,
    )
    locks = FallbackKeyedLock(RedisKeyedLock(client, settings.request_timeout_s + 30), local_locks)
    return limiter, locks, RedisPlanCache(client, settings.cache_ttl_s)


def _embedder(settings: Settings):
    from .semantic import HashingEmbedder, ResilientEmbedder, VertexEmbedder
    from .semantic.embeddings import with_min_similarity

    mode = settings.embeddings_mode
    if mode == "off":
        return None
    if mode == "hashing":
        return with_min_similarity(HashingEmbedder(settings.embed_dim), settings.semantic_min_sim)
    inner = VertexEmbedder(settings.embed_model, settings.gcp_project, settings.gcp_location, settings.embed_dim)
    return with_min_similarity(
        ResilientEmbedder(inner, Bulkhead("embed", settings.max_concurrent_llm, settings.queue_wait_s),
                          CircuitBreaker("embed", settings.breaker_failures, settings.breaker_reset_s)),
        settings.semantic_min_sim,
    )


def build_agent(settings: Settings | None = None, checkpointer=None, tools=None, executors: dict | None = None,
                ontology=None) -> Bundle:
    """tools: a ToolRegistry (see tools/); executors and ontology override the demo defaults in tests."""
    settings = settings or Settings.from_env()
    problems = settings.validate()
    if problems:
        raise ConfigError("; ".join(problems))

    closers: list = []
    limiter, locks, cache = _shared_backends(settings, closers)
    blocks = BlockTracker(settings.block_alert_threshold)
    if tools is not None:
        closers.append(tools.close)

    if settings.mode == "demo":
        from langgraph.checkpoint.memory import MemorySaver

        from .demo import DEV_USERS, build_ontology, build_sqlite
        from .llm import RuleBasedLLM
        from .semantic import OntologyProvider
        from .semantic.embeddings import HashingEmbedder
        from .warehouse import SqliteExecutor

        embedder = _embedder(settings)
        onto = ontology or build_ontology()
        if isinstance(embedder, HashingEmbedder):
            onto = _embedded(onto, embedder)
        deps = Deps(
            provider=OntologyProvider(lambda: onto, ttl_s=10**9),
            llm=RuleBasedLLM(),
            executors=executors or {"sqlite": SqliteExecutor(build_sqlite())},
            cache=cache, settings=settings, blocks=blocks, embedder=embedder, tools=tools,
        )
        agent = build_graph(deps, checkpointer or MemorySaver())
        return Bundle(agent, deps, settings, DEV_USERS, limiter, locks, InMemoryFeedbackStore(), closers)

    from .llm import GeminiLLM, ResilientLLM
    from .memory import PostgresFeedbackStore, make_postgres_checkpointer
    from .semantic import OntologyProvider, load_from_postgres
    from .warehouse import ResilientExecutor, build_raw_executors

    executors = executors or {
        name: ResilientExecutor(
            raw, name,
            Bulkhead(f"exec-{name}", settings.max_concurrent_queries, settings.queue_wait_s),
            CircuitBreaker(f"exec-{name}", settings.breaker_failures, settings.breaker_reset_s),
        )
        for name, raw in build_raw_executors().items()
    }
    if not executors:
        raise ConfigError("no warehouse configured: set DATABRICKS_* and/or SNOWFLAKE_* variables")

    llm = ResilientLLM(
        GeminiLLM(settings.plan_model, settings.gcp_project, settings.gcp_location, settings.llm_timeout_s),
        settings.plan_model,
        Bulkhead("llm", settings.max_concurrent_llm, settings.queue_wait_s),
        CircuitBreaker("llm", settings.breaker_failures, settings.breaker_reset_s),
        settings.llm_attempts,
    )
    deps = Deps(
        provider=OntologyProvider(lambda: load_from_postgres(settings.pg_dsn), settings.ontology_ttl_s),
        llm=llm, executors=executors, cache=cache, settings=settings, blocks=blocks,
        embedder=_embedder(settings), tools=tools,
    )
    feedback = None
    if checkpointer is None:
        checkpointer, pool = make_postgres_checkpointer(settings.pg_dsn)
        closers.append(pool.close)
        feedback = PostgresFeedbackStore(pool)
    agent = build_graph(deps, checkpointer)
    return Bundle(agent, deps, settings, {}, limiter, locks, feedback or InMemoryFeedbackStore(), closers)


def _embedded(onto, embedder):
    """Demo stand-in for jobs/embed.py: a copy of the ontology with synonym and example vectors."""
    from dataclasses import replace

    from .semantic.ontology import Ontology

    syns = [replace(s, vector=embedder.embed([s.term], "document")[0]) for s in onto.synonyms]
    exs = [replace(e, vector=embedder.embed([e.question], "document")[0]) for e in onto.examples]
    return Ontology(list(onto.concepts.values()), onto.edges, list(onto.metrics.values()), syns,
                    list(onto.grants.values()), onto.release_id, exs)
