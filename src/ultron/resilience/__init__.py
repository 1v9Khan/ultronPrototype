"""Resilience primitives — circuit breakers, retry helpers, error logging.

Phase 4 of the Foundation phase. Used by every external-dependency
wrapper so the system degrades gracefully under partial failure.
"""

from ultron.resilience import fail_open_log
from ultron.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from ultron.resilience.error_log import ErrorLog, get_error_log, set_error_log
from ultron.resilience.phrases import phrase_for, reset_phrase_cache

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "ErrorLog",
    "fail_open_log",
    "get_error_log",
    "set_error_log",
    "phrase_for",
    "reset_phrase_cache",
]
