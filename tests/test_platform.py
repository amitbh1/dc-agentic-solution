import datetime as dt
import decimal
import time
from dataclasses import replace

import pytest

from dc_agent.core.config import Settings
from dc_agent.core.errors import AuthError, RateLimited
from dc_agent.demo import build_ontology, build_sqlite
from dc_agent.jobs.drift import check_ontology
from dc_agent.memory import PlanCache, RedisPlanCache
from dc_agent.resilience import (
    FallbackKeyedLock, FallbackRateLimiter, KeyedLock, RateLimiter, RedisKeyedLock, RedisRateLimiter,
)
from dc_agent.security import UserContext, make_redactor, redact_rows
from dc_agent.semantic import Example, Grant, HashingEmbedder, Ontology, select_examples, validate_ontology
from dc_agent.tools import ToolRegistry, ToolSpec
from dc_agent.warehouse import SqliteExecutor, parse_spark_cost
from dc_agent.warehouse.base import safe_value


# --- redaction ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("mail alice@example.com", "mail [email]"),
    ("call +91 98765 43210 now", "call [phone] now"),
    ("call (415) 555-0132", "call [phone]"),
    ("card 4111 1111 1111 1111", "card [card]"),
    ("ssn 123-45-6789", "ssn [ssn]"),
    ("aadhaar 2345 6789 0123", "aadhaar [aadhaar]"),
    ("pan ABCDE1234F", "pan [pan]"),
    ("host 10.20.30.40", "host [ip]"),
])
def test_redaction_masks_pii(text, expected):
    assert redact_rows([[text]]) == [[expected]]


@pytest.mark.parametrize("text", [
    "2026-11-01T05:00:00", "2026-11-01 05:00:00", "Mumbai-1", "R2", "III", "1234 5678 9012 3456",
    "version 1.2.3", "rack-0042",
])
def test_redaction_leaves_ordinary_values(text):
    assert redact_rows([[text]]) == [[text]]


def test_redaction_never_touches_numbers_and_respects_pattern_choice():
    assert redact_rows([[9876543210, 4111111111111111.0]]) == [[9876543210, 4111111111111111.0]]
    only_email = make_redactor(["email"])
    assert redact_rows([["a@b.io 10.0.0.1"]], only_email) == [["[email] 10.0.0.1"]]


def test_safe_value_makes_json_safe():
    assert safe_value(float("nan")) is None and safe_value(float("inf")) is None
    assert safe_value(decimal.Decimal("NaN")) is None and safe_value(decimal.Decimal("1.5")) == 1.5
    assert safe_value(dt.date(2026, 1, 2)) == "2026-01-02"


def test_spark_cost_parser():
    text = "== Optimized Logical Plan ==\nAggregate ..., Statistics(sizeInBytes=1.5 GiB, rowCount=2.0E+6)\n+- ..."
    est = parse_spark_cost(text)
    assert est.bytes == int(1.5 * 2**30) and est.rows == 2_000_000
    assert parse_spark_cost("no stats") is None


def test_sqlite_estimate_counts_rows_read():
    est = SqliteExecutor(build_sqlite()).estimate("SELECT * FROM racks JOIN facilities ON 1 = 1")
    assert est.rows == 15 and est.bytes == 15 * 64


# --- configuration -----------------------------------------------------------------------

def test_settings_validation_rules():
    assert Settings().validate() == []
    assert any("APPROVAL" in p for p in Settings(max_scan_bytes=10, approval_scan_bytes=20).validate())
    assert any("REDACT" in p for p in Settings(redact_patterns=("email", "dna")).validate())
    assert any("EMBEDDINGS" in p for p in Settings(embeddings="magic").validate())
    assert any("MCP" in p for p in Settings(mcp_tools={"x": {"roles": []}}).validate())
    prod = Settings(mode="prod", pg_dsn="x", gcp_project="p", auth_mode="oidc", oidc_jwks_url="u",
                    oidc_audience="a", oidc_issuer="i", tenant_daily_budget=5)
    assert any("TENANT_CLAIM" in p for p in prod.validate())
    assert Settings(mode="prod").embeddings_mode == "vertex" and Settings().embeddings_mode == "hashing"


def test_settings_from_env_parses_json_and_lists(monkeypatch):
    monkeypatch.setenv("DC_MCP_SERVERS", '{"rb": {"url": "http://x", "transport": "streamable_http"}}')
    monkeypatch.setenv("DC_REDACT_PATTERNS", "email, phone")
    s = Settings.from_env()
    assert s.mcp_servers["rb"]["url"] == "http://x" and s.redact_patterns == ("email", "phone")


# --- drift and ontology validation -------------------------------------------------------

def test_drift_reports_type_changes():
    base = build_ontology()
    rack = base.concepts["rack"]
    changed = replace(rack, attrs={**rack.attrs, "row": replace(rack.attrs["row"], data_type="bigint")})
    concepts = [changed if c.name == "rack" else c for c in base.concepts.values()]
    onto = Ontology(concepts, base.edges, list(base.metrics.values()), base.synonyms, list(base.grants.values()), 1)
    findings = check_ontology(onto, {"sqlite": SqliteExecutor(build_sqlite())})
    assert [(f.concept, f.kind, f.attribute) for f in findings] == [("rack", "type_changed", "row")]
    assert findings[0].breaking


def test_demo_ontology_is_valid_and_missing_scope_is_reported():
    onto = build_ontology()
    assert validate_ontology(onto) == []
    grants = [g for g in onto.grants.values() if (g.role, g.concept) != ("analyst", "rack")]
    grants.append(Grant("analyst", "rack", "internal"))  # scoped role, unscoped concept
    broken = Ontology(list(onto.concepts.values()), onto.edges, list(onto.metrics.values()), onto.synonyms, grants, 2)
    assert validate_ontology(broken) == ["role analyst: concept rack has no row scope but other grants do"]


# --- embeddings and examples -------------------------------------------------------------

def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(64)
    a, b = e.embed(["average power"]), e.embed(["average power"])
    assert a == b and abs(sum(x * x for x in a[0]) - 1) < 1e-9


def test_examples_are_filtered_to_what_the_user_can_see():
    examples = [
        Example("peak power per row", {"metrics": ["max_power_kw"], "dimensions": ["rack.row"]}),
        Example("secret metric", {"metrics": ["hidden"]}),
        Example("peak by restricted attr", {"metrics": ["max_power_kw"], "dimensions": ["facility.secret"]}),
    ]
    catalogue = {"metrics": [{"id": "max_power_kw"}], "attributes": [{"ref": "rack.row"}]}
    out = select_examples(examples, catalogue, "peak power by row", k=3)
    assert [e["question"] for e in out] == ["peak power per row"]
    assert select_examples(examples, catalogue, "x", k=0) == []


# --- rate limits, locks and caches -------------------------------------------------------

def test_local_tenant_budget():
    rl = RateLimiter(per_min=600, burst=100, daily_budget=100, tenant_budget=2)
    rl.check("a", "t1")
    rl.check("b", "t1")
    with pytest.raises(RateLimited) as exc:
        rl.check("c", "t1")
    assert exc.value.reason == "tenant budget exhausted"
    rl.check("c", "t2")
    rl.check("c")  # no tenant: tenant budget does not apply


fakeredis = pytest.importorskip("fakeredis")


def test_redis_rate_limiter_burst_daily_and_tenant():
    r = fakeredis.FakeRedis()
    wall = [1_000_000.0]
    rl = RedisRateLimiter(r, per_min=60, burst=2, daily_budget=3, tenant_budget=0, wall=lambda: wall[0])
    rl.check("u")
    rl.check("u")
    with pytest.raises(RateLimited) as exc:
        rl.check("u")
    assert exc.value.reason == "rate limit" and exc.value.retry_after_s >= 1
    wall[0] += 1.5
    rl.check("u")
    wall[0] += 10
    with pytest.raises(RateLimited) as exc:
        rl.check("u")
    assert exc.value.reason == "daily budget exhausted"
    tl = RedisRateLimiter(r, per_min=600, burst=50, daily_budget=0, tenant_budget=1, wall=lambda: wall[0])
    tl.check("x", "acme")
    with pytest.raises(RateLimited) as exc:
        tl.check("y", "acme")
    assert exc.value.reason == "tenant budget exhausted"


def test_redis_lock_is_exclusive_and_only_owner_releases():
    r = fakeredis.FakeRedis()
    a, b = RedisKeyedLock(r, ttl_s=60), RedisKeyedLock(r, ttl_s=60)
    assert a.try_acquire("s") and not b.try_acquire("s")
    b.release("s")  # not the owner: no effect
    assert not b.try_acquire("s")
    a.release("s")
    assert b.try_acquire("s")


class _Down:
    def register_script(self, _):
        def run(**_):
            raise ConnectionError("redis down")
        return run

    def set(self, *a, **k):
        raise ConnectionError("redis down")

    def get(self, *a, **k):
        raise ConnectionError("redis down")


def test_fallbacks_when_redis_is_down():
    rl = FallbackRateLimiter(RedisRateLimiter(_Down()), RateLimiter(per_min=60, burst=1))
    rl.check("u")
    with pytest.raises(RateLimited):  # local limiter now enforces
        rl.check("u")
    locks = FallbackKeyedLock(RedisKeyedLock(_Down()), KeyedLock())
    assert locks.try_acquire("k") and not locks.try_acquire("k")
    locks.release("k")
    assert locks.try_acquire("k")
    cache = RedisPlanCache(_Down())
    cache.set(("k",), {"p": 1})
    assert cache.get(("k",)) is None


def test_redis_plan_cache_round_trip():
    cache = RedisPlanCache(fakeredis.FakeRedis(), ttl_s=60)
    key = PlanCache.key(1, "analyst", "Average power by region")
    cache.set(key, {"metrics": ["avg_power_kw"]})
    assert cache.get(key) == {"metrics": ["avg_power_kw"]}


# --- tools -------------------------------------------------------------------------------

def test_tool_registry_is_role_scoped_and_time_bounded():
    reg = ToolRegistry()
    seen = {}

    def note(question, **ctx):
        seen.update(ctx)
        return "see runbook RB-7"

    reg.register(ToolSpec("runbook", note, frozenset({"admin"})))
    reg.register(ToolSpec("slow", lambda **_: time.sleep(1) or "late", frozenset({"admin"}), timeout_s=0.05))
    reg.register(ToolSpec("broken", lambda **_: 1 / 0, frozenset({"admin"})))
    ctx = {"question": "q", "metrics": ["m"], "dimensions": [], "columns": ["a"], "row_count": 3}
    admin, analyst = UserContext("u1", "admin"), UserContext("u2", "analyst")
    assert reg.enrich(admin, ctx) == [{"tool": "runbook", "note": "see runbook RB-7"}]
    assert "rows" not in seen and seen["row_count"] == 3
    assert reg.enrich(analyst, ctx) == []
    reg.close()


# --- OIDC --------------------------------------------------------------------------------

jwt = pytest.importorskip("jwt")


def test_oidc_maps_role_claims_and_tenant(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import rsa

    from dc_agent.security import auth

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class Jwks:
        def get_signing_key_from_jwt(self, _token):
            return type("K", (), {"key": key.public_key()})()

    monkeypatch.setattr(auth, "_jwks_client", lambda url: Jwks())
    s = Settings(auth_mode="oidc", oidc_jwks_url="https://idp/jwks", oidc_audience="dc-agent",
                 oidc_issuer="https://idp", tenant_claim="tenant")
    now = int(time.time())
    claims = {"sub": "alice", "aud": "dc-agent", "iss": "https://idp", "iat": now, "exp": now + 60,
              "roles": ["analyst"], "regions": ["West"], "tenant": "acme"}
    user = auth.authenticate(jwt.encode(claims, key, algorithm="RS256"), s)
    assert user == UserContext("alice", "analyst", {"regions": ["West"]}, "acme")
    for bad in ({**claims, "aud": "other"}, {**claims, "exp": now - 10}, {k: v for k, v in claims.items() if k != "sub"}):
        with pytest.raises(AuthError):
            auth.authenticate(jwt.encode(bad, key, algorithm="RS256"), s)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthError):
        auth.authenticate(jwt.encode(claims, other, algorithm="RS256"), s)
