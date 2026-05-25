"""``safe_capture`` wrapper for fire-and-forget observability emits.

Adapted from cline's ``TelemetryService.safeCapture`` pattern (Apache
2.0; see ``THIRD_PARTY_NOTICES.md``). The wrapper ensures that an
emit-site failure (writer crash, schema mismatch, disk full) NEVER
propagates back into the call site that triggered the observation.

Ultron's observation framework already lives in
``src/ultron/observations/`` (writer, schema, integrations, etc.).
This module adds the canonical wrapper that every external caller
should route through. The contract is:

* The wrapped callable is invoked synchronously OR asynchronously,
  matching the caller's idiom.
* Any exception during the call is logged at WARN (via the standard
  ``ultron.observations.safe_capture`` logger) and swallowed.
* The wrapper returns the call's result on success; on failure it
  returns the optional ``fallback`` value (default ``None``).
* An optional per-call ``error_context`` string is included in the
  WARN log so the operator can identify which subsystem failed.

Two operator-facing surfaces sit on top of the wrapper:

* :func:`safe_capture` for synchronous callers (the bulk of voice/coding
  emit sites today).
* :func:`safe_capture_async` for async coroutines (used by the bus sink
  and any future MCP-side emit).

A bounded in-memory counter (:class:`SafeCaptureStats`) tracks how
often the wrapper has fired its fallback path so the operator can spot
chronic observation failure without trawling the log file.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar

LOGGER = logging.getLogger("ultron.observations.safe_capture")

T = TypeVar("T")


@dataclass
class SafeCaptureStats:
    """Mutable counters tracked by the wrapper.

    Attributes:
        total_calls: every wrapped invocation (success + failure).
        success_calls: invocations that returned normally.
        failure_calls: invocations whose callable raised.
        last_failure_message: short rendering of the most-recent error.
        last_failure_at: monotonic timestamp of the most-recent error.
        per_context_failures: failure count grouped by ``error_context``.
    """

    total_calls: int = 0
    success_calls: int = 0
    failure_calls: int = 0
    last_failure_message: Optional[str] = None
    last_failure_at: Optional[float] = None
    per_context_failures: dict[str, int] = field(default_factory=dict)


_STATS = SafeCaptureStats()
_STATS_LOCK = threading.RLock()


def safe_capture_stats() -> SafeCaptureStats:
    """Return a snapshot of the wrapper's accumulated counts.

    The returned object is a copy; mutating it does not affect the
    module-level singleton.
    """
    with _STATS_LOCK:
        return SafeCaptureStats(
            total_calls=_STATS.total_calls,
            success_calls=_STATS.success_calls,
            failure_calls=_STATS.failure_calls,
            last_failure_message=_STATS.last_failure_message,
            last_failure_at=_STATS.last_failure_at,
            per_context_failures=dict(_STATS.per_context_failures),
        )


def reset_safe_capture_stats() -> None:
    """Reset the wrapper's accumulated counts (test-only)."""
    with _STATS_LOCK:
        _STATS.total_calls = 0
        _STATS.success_calls = 0
        _STATS.failure_calls = 0
        _STATS.last_failure_message = None
        _STATS.last_failure_at = None
        _STATS.per_context_failures.clear()


def _record_success() -> None:
    with _STATS_LOCK:
        _STATS.total_calls += 1
        _STATS.success_calls += 1


def _record_failure(error: BaseException, error_context: str) -> None:
    with _STATS_LOCK:
        _STATS.total_calls += 1
        _STATS.failure_calls += 1
        _STATS.last_failure_message = f"{type(error).__name__}: {error}"
        _STATS.last_failure_at = time.monotonic()
        if error_context:
            _STATS.per_context_failures[error_context] = (
                _STATS.per_context_failures.get(error_context, 0) + 1
            )


def safe_capture(
    fn: Callable[..., T],
    *args: Any,
    error_context: str = "",
    fallback: Optional[T] = None,
    log_traceback: bool = False,
    **kwargs: Any,
) -> Optional[T]:
    """Invoke ``fn(*args, **kwargs)`` and swallow any exception.

    Args:
        fn: the (sync) callable to invoke.
        *args: positional arguments forwarded to ``fn``.
        error_context: optional short descriptor of the calling
            subsystem (``"bus.publish"``, ``"memory.add"``, etc.). The
            WARN log line includes it and the stats counter groups by it.
        fallback: value returned when ``fn`` raises.
        log_traceback: when True, the WARN log includes the full
            traceback. Default False (the type-and-message single-line
            form is usually enough for telemetry triage).
        **kwargs: keyword arguments forwarded to ``fn``.

    Returns:
        ``fn``'s return value on success, or ``fallback`` on failure.

    Notes:
        The wrapper deliberately does NOT propagate ``asyncio.CancelledError``
        because it is sync-only — async cancellation can never be triggered
        in this code path. Use :func:`safe_capture_async` for awaitables.
    """
    try:
        result = fn(*args, **kwargs)
        _record_success()
        return result
    except BaseException as exc:  # noqa: BLE001 - telemetry must not crash callers
        _record_failure(exc, error_context)
        if log_traceback:
            LOGGER.warning(
                "safe_capture(%s) raised %s",
                error_context or fn.__qualname__,
                type(exc).__name__,
                exc_info=True,
            )
        else:
            LOGGER.warning(
                "safe_capture(%s) raised %s: %s",
                error_context or fn.__qualname__,
                type(exc).__name__,
                exc,
            )
        return fallback


async def safe_capture_async(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    error_context: str = "",
    fallback: Optional[T] = None,
    log_traceback: bool = False,
    **kwargs: Any,
) -> Optional[T]:
    """Async twin of :func:`safe_capture`.

    Accepts a coroutine function ``fn`` and awaits the result. Any
    exception is swallowed exactly as in the sync variant.
    :class:`asyncio.CancelledError` propagates so the caller's own
    cancel chain is not eaten.
    """
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            value = await result
        else:
            value = result
        _record_success()
        return value
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # noqa: BLE001
        _record_failure(exc, error_context)
        if log_traceback:
            LOGGER.warning(
                "safe_capture_async(%s) raised %s",
                error_context or getattr(fn, "__qualname__", repr(fn)),
                type(exc).__name__,
                exc_info=True,
            )
        else:
            LOGGER.warning(
                "safe_capture_async(%s) raised %s: %s",
                error_context or getattr(fn, "__qualname__", repr(fn)),
                type(exc).__name__,
                exc,
            )
        return fallback


def safe_capture_decorator(
    *,
    error_context: str = "",
    fallback: Any = None,
    log_traceback: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., Optional[T]]]:
    """Decorator form of :func:`safe_capture` for emit-site convenience.

    Example:

    .. code-block:: python

        @safe_capture_decorator(error_context="memory.write")
        def emit_memory_write(turn_id: str) -> None:
            writer.write({"event": "memory_write", "turn_id": turn_id})
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., Optional[T]]:
        if inspect.iscoroutinefunction(fn):
            async def async_wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
                return await safe_capture_async(
                    fn,
                    *args,
                    error_context=error_context,
                    fallback=fallback,
                    log_traceback=log_traceback,
                    **kwargs,
                )
            async_wrapper.__name__ = fn.__name__
            async_wrapper.__doc__ = fn.__doc__
            return async_wrapper  # type: ignore[return-value]

        def sync_wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            return safe_capture(
                fn,
                *args,
                error_context=error_context,
                fallback=fallback,
                log_traceback=log_traceback,
                **kwargs,
            )
        sync_wrapper.__name__ = fn.__name__
        sync_wrapper.__doc__ = fn.__doc__
        return sync_wrapper

    return decorate


__all__ = [
    "SafeCaptureStats",
    "reset_safe_capture_stats",
    "safe_capture",
    "safe_capture_async",
    "safe_capture_decorator",
    "safe_capture_stats",
]
