from __future__ import annotations

from functools import lru_cache

from ..core.errors import AuthError
from .identity import UserContext


def authenticate(token: str, settings, dev_users: dict | None = None) -> UserContext:
    if not token:
        raise AuthError("missing token")
    if settings.auth_mode == "dev":
        user = (dev_users or {}).get(token)
        if user is None:
            raise AuthError("unknown token")
        return user
    return _from_oidc(token, settings)


@lru_cache(maxsize=4)
def _jwks_client(url: str):
    import jwt

    return jwt.PyJWKClient(url)


def _from_oidc(token: str, s) -> UserContext:
    import jwt

    try:
        key = _jwks_client(s.oidc_jwks_url).get_signing_key_from_jwt(token).key
        data = jwt.decode(
            token, key, algorithms=["RS256"], audience=s.oidc_audience, issuer=s.oidc_issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
    except Exception as exc:
        raise AuthError("invalid token") from exc
    roles = data.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    claims = {}
    for name in s.entitlement_claims:
        v = data.get(name, [])
        claims[name] = [str(v)] if isinstance(v, (str, int)) else [str(x) for x in v]
    tenant = data.get(s.tenant_claim) if s.tenant_claim else None
    return UserContext(str(data["sub"]), roles[0] if roles else "none", claims, str(tenant) if tenant else None)
