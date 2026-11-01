from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserContext:
    user_id: str
    role: str
    claims: dict = field(default_factory=dict)
    tenant: str | None = None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "claims": {k: list(v) for k, v in self.claims.items()},
            "tenant": self.tenant,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserContext":
        return cls(d["user_id"], d["role"], {k: list(v) for k, v in d.get("claims", {}).items()}, d.get("tenant"))
