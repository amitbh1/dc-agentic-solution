"""Protection for calls to external dependencies and for the API itself."""
from .breaker import CircuitBreaker
from .bulkhead import Bulkhead
from .locks import KeyedLock
from .ratelimit import RateLimiter
from .retry import is_retryable_error, is_transient_error, retry
from .shared import FallbackKeyedLock, FallbackRateLimiter, RedisKeyedLock, RedisRateLimiter

__all__ = [
    "Bulkhead", "CircuitBreaker", "FallbackKeyedLock", "FallbackRateLimiter", "KeyedLock", "RateLimiter",
    "RedisKeyedLock", "RedisRateLimiter", "is_retryable_error", "is_transient_error", "retry",
]
