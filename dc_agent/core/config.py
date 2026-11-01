from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _json(name: str, default):
    raw = os.getenv(name, "").strip()
    return json.loads(raw) if raw else default


def _csv(name: str, default: str) -> tuple:
    return tuple(c.strip() for c in os.getenv(name, default).split(",") if c.strip())


REDACT_PATTERNS = ("email", "phone", "card", "ipv4", "ssn", "aadhaar", "pan")


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"  # demo | prod
    pg_dsn: str = ""
    redis_url: str = ""  # shared rate limits, session locks and plan cache across replicas
    # query limits
    max_rows: int = 1000
    max_repairs: int = 2
    query_timeout_s: int = 30
    request_timeout_s: int = 60  # end-to-end budget for one turn across all LLM calls and the query
    max_partial_rows: int = 100_000  # per part, for cross-engine split and merge
    # cost gate (0 disables): estimated bytes scanned, from the engine's EXPLAIN
    max_scan_bytes: int = 0  # refuse above this
    approval_scan_bytes: int = 0  # ask the user to confirm above this
    # caches
    cache_ttl_s: int = 3600
    ontology_ttl_s: int = 60
    # models
    plan_model: str = "gemini-2.5-flash"
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    llm_timeout_s: int = 30
    llm_attempts: int = 3
    # grounding
    embeddings: str = ""  # off | hashing | vertex; empty means hashing in demo, vertex in prod
    embed_model: str = "text-embedding-005"
    embed_dim: int = 768  # must match the vector(...) columns in sql/ontology_schema.sql
    semantic_min_sim: float = 0.0  # 0 means the embedder's own default
    few_shot_k: int = 3
    # auth
    auth_mode: str = "dev"  # dev | oidc
    oidc_jwks_url: str = ""
    oidc_audience: str = ""
    oidc_issuer: str = ""
    entitlement_claims: tuple = ("regions",)
    tenant_claim: str = ""  # OIDC claim naming the user's tenant, for per-tenant budgets
    # protection
    rate_per_min: int = 30
    rate_burst: int = 10
    daily_budget: int = 500
    tenant_daily_budget: int = 0  # 0 disables
    max_concurrent_queries: int = 8
    max_concurrent_llm: int = 16
    queue_wait_s: float = 5.0
    breaker_failures: int = 5
    breaker_reset_s: float = 30.0
    block_alert_threshold: int = 5
    redact_pii: bool = True
    redact_patterns: tuple = REDACT_PATTERNS
    # retention
    checkpoint_retention_days: int = 30
    feedback_retention_days: int = 365
    # tools: {"runbooks": {"url": "...", "transport": "streamable_http"}}
    mcp_servers: dict = field(default_factory=dict)
    # {"search_runbooks": {"roles": ["admin"], "arg": "query", "timeout_s": 5}}
    mcp_tools: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.getenv
        return cls(
            mode=g("DC_MODE", "demo"),
            pg_dsn=g("DC_PG_DSN", ""),
            redis_url=g("DC_REDIS_URL", ""),
            max_rows=int(g("DC_MAX_ROWS", "1000")),
            max_repairs=int(g("DC_MAX_REPAIRS", "2")),
            query_timeout_s=int(g("DC_QUERY_TIMEOUT_S", "30")),
            request_timeout_s=int(g("DC_REQUEST_TIMEOUT_S", "60")),
            max_partial_rows=int(g("DC_MAX_PARTIAL_ROWS", "100000")),
            max_scan_bytes=int(g("DC_MAX_SCAN_BYTES", "0")),
            approval_scan_bytes=int(g("DC_APPROVAL_SCAN_BYTES", "0")),
            cache_ttl_s=int(g("DC_CACHE_TTL_S", "3600")),
            ontology_ttl_s=int(g("DC_ONTOLOGY_TTL_S", "60")),
            plan_model=g("DC_PLAN_MODEL", "gemini-2.5-flash"),
            gcp_project=g("GOOGLE_CLOUD_PROJECT", ""),
            gcp_location=g("GOOGLE_CLOUD_LOCATION", "us-central1"),
            llm_timeout_s=int(g("DC_LLM_TIMEOUT_S", "30")),
            llm_attempts=int(g("DC_LLM_ATTEMPTS", "3")),
            embeddings=g("DC_EMBEDDINGS", ""),
            embed_model=g("DC_EMBED_MODEL", "text-embedding-005"),
            embed_dim=int(g("DC_EMBED_DIM", "768")),
            semantic_min_sim=float(g("DC_SEMANTIC_MIN_SIM", "0")),
            few_shot_k=int(g("DC_FEW_SHOT_K", "3")),
            auth_mode=g("DC_AUTH_MODE", "dev"),
            oidc_jwks_url=g("DC_OIDC_JWKS_URL", ""),
            oidc_audience=g("DC_OIDC_AUDIENCE", ""),
            oidc_issuer=g("DC_OIDC_ISSUER", ""),
            entitlement_claims=_csv("DC_ENTITLEMENT_CLAIMS", "regions"),
            tenant_claim=g("DC_TENANT_CLAIM", ""),
            rate_per_min=int(g("DC_RATE_PER_MIN", "30")),
            rate_burst=int(g("DC_RATE_BURST", "10")),
            daily_budget=int(g("DC_DAILY_BUDGET", "500")),
            tenant_daily_budget=int(g("DC_TENANT_DAILY_BUDGET", "0")),
            max_concurrent_queries=int(g("DC_MAX_CONCURRENT_QUERIES", "8")),
            max_concurrent_llm=int(g("DC_MAX_CONCURRENT_LLM", "16")),
            queue_wait_s=float(g("DC_QUEUE_WAIT_S", "5")),
            breaker_failures=int(g("DC_BREAKER_FAILURES", "5")),
            breaker_reset_s=float(g("DC_BREAKER_RESET_S", "30")),
            block_alert_threshold=int(g("DC_BLOCK_ALERT_THRESHOLD", "5")),
            redact_pii=_bool("DC_REDACT_PII", True),
            redact_patterns=_csv("DC_REDACT_PATTERNS", ",".join(REDACT_PATTERNS)),
            checkpoint_retention_days=int(g("DC_CHECKPOINT_RETENTION_DAYS", "30")),
            feedback_retention_days=int(g("DC_FEEDBACK_RETENTION_DAYS", "365")),
            mcp_servers=_json("DC_MCP_SERVERS", {}),
            mcp_tools=_json("DC_MCP_TOOLS", {}),
        )

    @property
    def embeddings_mode(self) -> str:
        return self.embeddings or ("hashing" if self.mode == "demo" else "vertex")

    def validate(self) -> list:
        problems = []
        if self.mode not in ("demo", "prod"):
            problems.append("DC_MODE must be demo or prod")
        positive = {
            "DC_MAX_ROWS": self.max_rows, "DC_QUERY_TIMEOUT_S": self.query_timeout_s,
            "DC_REQUEST_TIMEOUT_S": self.request_timeout_s,
            "DC_RATE_PER_MIN": self.rate_per_min, "DC_RATE_BURST": self.rate_burst,
            "DC_MAX_CONCURRENT_QUERIES": self.max_concurrent_queries,
            "DC_MAX_CONCURRENT_LLM": self.max_concurrent_llm,
            "DC_LLM_ATTEMPTS": self.llm_attempts, "DC_MAX_PARTIAL_ROWS": self.max_partial_rows,
            "DC_EMBED_DIM": self.embed_dim,
        }
        problems += [f"{k} must be >= 1" for k, v in positive.items() if v < 1]
        non_negative = {
            "DC_MAX_REPAIRS": self.max_repairs, "DC_MAX_SCAN_BYTES": self.max_scan_bytes,
            "DC_APPROVAL_SCAN_BYTES": self.approval_scan_bytes, "DC_FEW_SHOT_K": self.few_shot_k,
            "DC_TENANT_DAILY_BUDGET": self.tenant_daily_budget,
        }
        problems += [f"{k} must be >= 0" for k, v in non_negative.items() if v < 0]
        if self.max_rows > 10_000:
            problems.append("DC_MAX_ROWS must be <= 10000")
        if self.max_scan_bytes and self.approval_scan_bytes and self.approval_scan_bytes >= self.max_scan_bytes:
            problems.append("DC_APPROVAL_SCAN_BYTES must be below DC_MAX_SCAN_BYTES")
        if self.embeddings_mode not in ("off", "hashing", "vertex"):
            problems.append("DC_EMBEDDINGS must be off, hashing or vertex")
        if not 0 <= self.semantic_min_sim < 1:
            problems.append("DC_SEMANTIC_MIN_SIM must be in [0, 1)")
        unknown = set(self.redact_patterns) - set(REDACT_PATTERNS)
        if unknown:
            problems.append(f"DC_REDACT_PATTERNS has unknown names: {sorted(unknown)}")
        for name, spec in self.mcp_tools.items():
            if name and not (isinstance(spec, dict) and spec.get("roles")):
                problems.append(f"DC_MCP_TOOLS[{name}] needs a non-empty roles list")
        if self.mcp_tools and not self.mcp_servers:
            problems.append("DC_MCP_TOOLS is set but DC_MCP_SERVERS is empty")
        if self.mode == "prod":
            if not self.pg_dsn:
                problems.append("DC_PG_DSN is required")
            if not self.gcp_project:
                problems.append("GOOGLE_CLOUD_PROJECT is required")
            if self.auth_mode != "oidc":
                problems.append("DC_AUTH_MODE must be oidc in prod")
            elif not (self.oidc_jwks_url and self.oidc_audience and self.oidc_issuer):
                problems.append("DC_OIDC_JWKS_URL, DC_OIDC_AUDIENCE and DC_OIDC_ISSUER are required")
            if self.tenant_daily_budget and not self.tenant_claim:
                problems.append("DC_TENANT_DAILY_BUDGET needs DC_TENANT_CLAIM")
        return problems
