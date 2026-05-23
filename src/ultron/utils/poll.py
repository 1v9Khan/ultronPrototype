"""Polling-with-bounded-retries as a graceful-degradation primitive.

Clean-room adaptation of the OpenHands ``_poll_for_title`` pattern from
``event_callback/set_title_callback_processor.py``. The original polls an
HTTP endpoint up to ``_NUM_POLL_ATTEMPTS=4`` times with ``_POLL_DELAY_S=3``
between attempts and returns ``None`` if the answer never arrives. This
module generalises the pattern: any callable that "might not be ready
yet" gets bounded retries with optional exponential backoff, and the
caller decides when a partial result is "done" via a custom predicate.

Sync (:func:`poll_until`) and async (:func:`apoll_until`) variants are
provided. Both return ``None`` (or the predicate's negative value) when
no attempt succeeded -- the caller is expected to handle that gracefully
(e.g. re-fire the polling from a future event callback). Pattern lineage
attributed in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_BACKOFF_FACTOR = 1.0  # 1.0 == constant delay; 2.0 == double each attempt
DEFAULT_MAX_DELAY_SECONDS = 60.0


@dataclass(frozen=True)
class PollResult(Generic[T]):
    """Outcome of a :func:`poll_until` / :func:`apoll_until` call.

    Attributes:
        value: The last value returned by ``fn``. ``None`` when every
            attempt raised. May be a falsy result when the predicate
            still rejected it.
        succeeded: ``True`` iff the predicate accepted at least one
            attempt's value.
        attempts: Number of times ``fn`` was actually invoked (>=1).
        elapsed_seconds: Wall-clock duration of the polling loop.
        last_error: Repr of the most recent exception from ``fn``,
            or ``None`` if every attempt completed without raising.
    """

    value: T | None
    succeeded: bool
    attempts: int
    elapsed_seconds: float
    last_error: str | None = None


def _is_present(value: Any) -> bool:
    """Default "done" predicate: anything not-``None`` is acceptable.

    Mirrors the OpenHands ``if title:`` short-circuit; matches the most
    common "the answer eventually shows up" use case.
    """

    return value is not None


def _next_delay(current: float, factor: float, ceiling: float) -> float:
    """Apply exponential backoff with a ceiling."""

    if factor <= 1.0:
        return current
    return min(current * factor, ceiling)


def _validate_inputs(
    *,
    max_attempts: int,
    delay_seconds: float,
    backoff_factor: float,
    max_delay_seconds: float,
) -> None:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be >= 0")
    if backoff_factor < 1.0:
        raise ValueError("backoff_factor must be >= 1.0")
    if max_delay_seconds < delay_seconds:
        raise ValueError("max_delay_seconds must be >= delay_seconds")


def poll_until(
    fn: Callable[[], T],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    is_done: Callable[[T], bool] = _is_present,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    swallow_exceptions: bool = True,
    label: str = "",
) -> PollResult[T]:
    """Synchronous bounded-retry poller.

    Args:
        fn: Zero-argument callable invoked once per attempt.
        max_attempts: Hard cap on attempts (>=1). Default 4.
        delay_seconds: Initial sleep between attempts. The FIRST attempt
            fires immediately; the delay applies BEFORE attempts 2..N.
            Mirrors the OpenHands ``_poll_for_title`` shape where the
            polling loop sleeps before each retry attempt.
        is_done: Predicate that accepts the returned value and returns
            ``True`` when polling should stop. Default accepts any
            non-``None`` value.
        backoff_factor: Multiplier applied to the delay between attempts.
            ``1.0`` (default) means constant delay; ``2.0`` doubles each
            attempt.
        max_delay_seconds: Upper bound on the delay even with backoff.
        swallow_exceptions: When ``True`` (default), exceptions from ``fn``
            are caught + logged at WARN; the loop continues. When ``False``,
            exceptions propagate.
        label: Optional short string included in WARN log lines.

    Returns:
        :class:`PollResult` describing the outcome.
    """

    _validate_inputs(
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
        backoff_factor=backoff_factor,
        max_delay_seconds=max_delay_seconds,
    )

    start = time.perf_counter()
    current_delay = delay_seconds
    last_value: T | None = None
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        if attempt > 1 and current_delay > 0:
            time.sleep(current_delay)
            current_delay = _next_delay(current_delay, backoff_factor, max_delay_seconds)

        try:
            last_value = fn()
        except Exception as exc:
            last_error = repr(exc)
            if label:
                logger.warning(
                    "poll_until[%s] attempt %d/%d raised: %s",
                    label,
                    attempt,
                    max_attempts,
                    last_error,
                )
            else:
                logger.warning(
                    "poll_until attempt %d/%d raised: %s",
                    attempt,
                    max_attempts,
                    last_error,
                )
            if not swallow_exceptions:
                raise
            continue

        if is_done(last_value):
            return PollResult(
                value=last_value,
                succeeded=True,
                attempts=attempt,
                elapsed_seconds=time.perf_counter() - start,
                last_error=last_error,
            )

    return PollResult(
        value=last_value,
        succeeded=False,
        attempts=max_attempts,
        elapsed_seconds=time.perf_counter() - start,
        last_error=last_error,
    )


async def apoll_until(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    is_done: Callable[[T], bool] = _is_present,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    swallow_exceptions: bool = True,
    label: str = "",
    cancel_check: Callable[[], bool] | None = None,
) -> PollResult[T]:
    """Async bounded-retry poller.

    Same contract as :func:`poll_until` but invokes a coroutine factory
    each attempt.

    Args:
        fn: Zero-argument callable returning a coroutine. Invoked once per
            attempt.
        cancel_check: Optional zero-argument callable returning ``True``
            when the loop should abandon polling (e.g. "the user resumed
            speaking"). Checked between attempts.
        Other args identical to :func:`poll_until`.

    Returns:
        :class:`PollResult` describing the outcome.
    """

    _validate_inputs(
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
        backoff_factor=backoff_factor,
        max_delay_seconds=max_delay_seconds,
    )

    start = time.perf_counter()
    current_delay = delay_seconds
    last_value: T | None = None
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        if attempt > 1 and current_delay > 0:
            await asyncio.sleep(current_delay)
            current_delay = _next_delay(current_delay, backoff_factor, max_delay_seconds)

        if cancel_check is not None:
            try:
                if cancel_check():
                    return PollResult(
                        value=last_value,
                        succeeded=False,
                        attempts=max(attempt - 1, 1),
                        elapsed_seconds=time.perf_counter() - start,
                        last_error="cancelled",
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("poll cancel_check raised: %r", exc)

        try:
            last_value = await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = repr(exc)
            if label:
                logger.warning(
                    "apoll_until[%s] attempt %d/%d raised: %s",
                    label,
                    attempt,
                    max_attempts,
                    last_error,
                )
            else:
                logger.warning(
                    "apoll_until attempt %d/%d raised: %s",
                    attempt,
                    max_attempts,
                    last_error,
                )
            if not swallow_exceptions:
                raise
            continue

        if is_done(last_value):
            return PollResult(
                value=last_value,
                succeeded=True,
                attempts=attempt,
                elapsed_seconds=time.perf_counter() - start,
                last_error=last_error,
            )

    return PollResult(
        value=last_value,
        succeeded=False,
        attempts=max_attempts,
        elapsed_seconds=time.perf_counter() - start,
        last_error=last_error,
    )
