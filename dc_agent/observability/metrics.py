"""Prometheus metrics with a no-op fallback when prometheus_client is not installed."""
from __future__ import annotations

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    ENABLED = True
except ImportError:  # pragma: no cover
    ENABLED = False


class _Noop:
    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None

    def observe(self, *args, **kwargs):
        return None


def _counter(name: str, doc: str, labels: list):
    return Counter(name, doc, labels) if ENABLED else _Noop()


def _histogram(name: str, doc: str, labels: list = ()):
    return Histogram(name, doc, list(labels)) if ENABLED else _Noop()


def _gauge(name: str, doc: str):
    return Gauge(name, doc) if ENABLED else _Noop()


REQUESTS = _counter("dc_agent_requests_total", "HTTP requests", ["route", "status"])
ANSWERS = _counter("dc_agent_answers_total", "Agent outcomes", ["outcome"])
FAILURES = _counter("dc_agent_failures_total", "Agent failures by kind", ["kind"])
NODE_ERRORS = _counter("dc_agent_node_errors_total", "Unhandled exceptions caught at a graph node", ["node"])
NODE_SECONDS = _histogram("dc_agent_node_seconds", "Graph node latency", ["node"])
LLM_SECONDS = _histogram("dc_agent_llm_seconds", "LLM call latency", ["model"])
EMBED_SECONDS = _histogram("dc_agent_embed_seconds", "Embedding call latency")
QUERY_SECONDS = _histogram("dc_agent_query_seconds", "Warehouse query latency", ["dialect"])
BREAKER_OPENED = _counter("dc_agent_breaker_opened_total", "Circuit breaker openings", ["name"])
BLOCKS = _counter("dc_agent_input_blocks_total", "Blocked inputs", ["reason"])
CACHE = _counter("dc_agent_plan_cache_total", "Plan cache lookups", ["result"])
RATE_LIMITED = _counter("dc_agent_rate_limited_total", "Rate limited requests", ["reason"])
GROUNDING = _counter("dc_agent_grounding_total", "Grounding results", ["result"])  # lexical | semantic | none | follow_up
COST_GATE = _counter("dc_agent_cost_gate_total", "Cost gate decisions", ["decision"])
APPROVALS = _counter("dc_agent_approvals_total", "Human approval decisions", ["decision"])
TOOL_CALLS = _counter("dc_agent_tool_calls_total", "Enrichment tool calls", ["tool", "result"])
FEEDBACK = _counter("dc_agent_feedback_total", "User feedback", ["rating"])
BACKEND_FALLBACK = _counter("dc_agent_backend_fallback_total", "Shared backend unavailable, local fallback used", ["backend"])
ONTOLOGY_RELOADS = _counter("dc_agent_ontology_reloads_total", "Ontology reload attempts", ["result"])
ONTOLOGY_PROBLEMS = _gauge("dc_agent_ontology_problems", "Validation problems in the active ontology release")


def render() -> tuple:
    if not ENABLED:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST
