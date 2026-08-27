"""Resilient LLM/embedding call wrapper — retry, backoff, circuit breaker.

Wraps any callable with:
- Exponential backoff on transient failures (timeout, 502, 503, 429)
- Circuit breaker after N consecutive failures (cooldown period)
- Jitter to prevent thundering herd
- Typed error classification (retryable vs permanent)
"""

from __future__ import annotations

import random
import threading
import time

import requests

from ..log import logger

try:
    import openai

    _OPENAI_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
        openai.APITimeoutError,
        openai.APIConnectionError,
    )
except ImportError:  # openai is an optional dependency
    _OPENAI_RETRYABLE_EXCEPTIONS = ()


# Errors that are worth retrying
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    requests.ConnectionError,
    requests.Timeout,
    ConnectionError,
    TimeoutError,
    OSError,
) + _OPENAI_RETRYABLE_EXCEPTIONS


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying."""
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    # Duck-typed SDK errors (openai/anthropic APIStatusError expose .status_code)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS_CODES
    return False


class CircuitBreaker:
    """Tracks consecutive failures and trips after threshold.

    Thread-safe: an internal lock guards failure-count mutation so concurrent
    callers (e.g. multiple request threads sharing a provider breaker) don't
    race on trip/reset state.
    """

    def __init__(self, threshold: int = 5, cooldown: float = 30.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._tripped_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        """True if circuit is tripped and still in cooldown."""
        with self._lock:
            if self._failures < self.threshold:
                return False
            elapsed = time.time() - self._tripped_at
            if elapsed >= self.cooldown:
                # Cooldown expired — reset
                self._failures = 0
                return False
            return True

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            tripped = self._failures >= self.threshold
            if tripped:
                self._tripped_at = time.time()
        if tripped:
            logger.warning(
                "Circuit breaker tripped after %d failures. Cooldown: %.0fs",
                self._failures,
                self.cooldown,
            )

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0


# Global circuit breakers per provider type
_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def _get_breaker(name: str) -> CircuitBreaker:
    with _breakers_lock:
        if name not in _breakers:
            from ..config import get_config

            cfg = get_config()
            _breakers[name] = CircuitBreaker(
                threshold=cfg.llm.circuit_breaker_threshold,
                cooldown=cfg.llm.circuit_breaker_cooldown,
            )
        return _breakers[name]


def reset_breaker(name: str) -> None:
    """Reset the circuit breaker for a provider (e.g. after config change)."""
    with _breakers_lock:
        _breakers.pop(name, None)


def resilient_call(fn, *args, provider_name: str = "default", **kwargs):
    """Call fn with retry + backoff + circuit breaker.

    Args:
        fn: The callable to execute (e.g., requests.post)
        *args: Positional args for fn
        provider_name: Name for circuit breaker grouping (e.g., "ollama", "openai")
        **kwargs: Keyword args for fn

    Returns:
        The return value of fn

    Raises:
        The last exception if all retries exhausted, or CircuitBreakerOpen
        if the breaker is tripped.
    """
    from ..config import get_config

    cfg = get_config()
    max_retries = cfg.llm.max_retries
    base_delay = cfg.llm.retry_base_delay
    max_delay = cfg.llm.retry_max_delay

    breaker = _get_breaker(provider_name)

    if breaker.is_open:
        raise ConnectionError(
            f"Circuit breaker open for {provider_name}. "
            f"Too many consecutive failures. Waiting for cooldown."
        )

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt >= max_retries:
                breaker.record_failure()
                raise

            breaker.record_failure()
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = delay * random.uniform(0.5, 1.0)
            logger.debug(
                "Retryable error (attempt %d/%d): %s. Retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                jitter,
            )
            time.sleep(jitter)

    raise last_exc  # type: ignore[misc]
