class AgentError(Exception):
    fatal = False


class PlanError(AgentError):
    """Plan cannot be compiled. Repairable by asking the planner again."""


class FanOutError(PlanError):
    """A join would multiply rows and corrupt a sum, count or avg."""


class AccessError(PlanError):
    """The user's role may not read something the plan needs."""

    fatal = True


class CrossEngineError(PlanError):
    """Plan spans several engines; not supported in this version."""

    fatal = True


class GuardError(AgentError):
    """Generated SQL failed a safety check."""

    fatal = True


class InputBlocked(AgentError):
    """Question rejected by input guardrails."""

    fatal = True


class CircuitOpen(AgentError):
    """A dependency is failing; calls are short-circuited."""

    fatal = True


class Overloaded(AgentError):
    """Too many concurrent calls to a dependency."""

    fatal = True


class RateLimited(Exception):
    def __init__(self, retry_after_s: float, reason: str):
        super().__init__(reason)
        self.retry_after_s = retry_after_s
        self.reason = reason


class AuthError(Exception):
    """Missing or invalid credentials."""


class ConfigError(Exception):
    """Invalid configuration; refuse to start."""
