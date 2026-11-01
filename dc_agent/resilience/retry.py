from __future__ import annotations

import random
import time
from typing import Callable

_TRANSIENT_NAMES = {
    "OperationalError", "InterfaceError", "ServerError", "ServiceUnavailable",
    "ResourceExhausted", "DeadlineExceeded", "ConnectError", "ReadTimeout", "ConnectTimeout",
}


def is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    return getattr(exc, "code", None) in (429, 500, 502, 503, 504)


def is_retryable_error(exc: BaseException) -> bool:
    """Transient, but never retry a timeout: it would double the cost of a slow query."""
    return is_transient_error(exc) and not isinstance(exc, TimeoutError)


def retry(fn: Callable, *, attempts: int = 3, base_s: float = 0.5, max_s: float = 8.0,
          retry_on: Callable = is_retryable_error, sleep: Callable = time.sleep):
    """Exponential backoff with jitter. Only wrap idempotent calls."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1 or not retry_on(exc):
                raise
            sleep(min(max_s, base_s * 2 ** i) * (0.5 + random.random() / 2))
    raise RuntimeError("unreachable")
