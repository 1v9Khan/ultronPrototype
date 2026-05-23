"""Tests for the polling-with-bounded-retries helper (T14 from the OpenHands catalog)."""

from __future__ import annotations

import asyncio
import time

import pytest

from ultron.utils.poll import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    PollResult,
    apoll_until,
    poll_until,
)


# -- sync poll_until --


def test_first_attempt_succeeds_returns_value_no_sleep():
    calls = []

    def fn():
        calls.append(1)
        return "ready"

    start = time.perf_counter()
    result = poll_until(fn, max_attempts=4, delay_seconds=10.0)
    elapsed = time.perf_counter() - start

    assert result.succeeded is True
    assert result.value == "ready"
    assert result.attempts == 1
    assert result.last_error is None
    # No sleep between the (single) attempt and the conclusion.
    assert elapsed < 1.0
    assert len(calls) == 1


def test_eventual_success_after_n_retries():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1
        if counter["n"] < 3:
            return None
        return "found"

    result = poll_until(fn, max_attempts=5, delay_seconds=0.01)
    assert result.succeeded is True
    assert result.value == "found"
    assert result.attempts == 3


def test_all_attempts_fail_returns_succeeded_false():
    def fn():
        return None

    result = poll_until(fn, max_attempts=3, delay_seconds=0.01)
    assert result.succeeded is False
    assert result.value is None
    assert result.attempts == 3


def test_predicate_lets_caller_define_done():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1
        return counter["n"]  # always non-None

    result = poll_until(
        fn,
        max_attempts=10,
        delay_seconds=0,
        is_done=lambda v: v >= 4,
    )
    assert result.succeeded is True
    assert result.value == 4
    assert result.attempts == 4


def test_exception_caught_and_loop_continues_by_default():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1
        if counter["n"] == 1:
            raise RuntimeError("transient")
        return "ok"

    result = poll_until(fn, max_attempts=3, delay_seconds=0.01)
    assert result.succeeded is True
    assert result.value == "ok"
    assert result.attempts == 2
    assert result.last_error is not None
    assert "transient" in result.last_error


def test_exception_propagates_when_swallow_disabled():
    def fn():
        raise ValueError("hard fail")

    with pytest.raises(ValueError):
        poll_until(
            fn,
            max_attempts=3,
            delay_seconds=0.01,
            swallow_exceptions=False,
        )


def test_invalid_max_attempts_raises():
    with pytest.raises(ValueError):
        poll_until(lambda: None, max_attempts=0)


def test_invalid_delay_seconds_raises():
    with pytest.raises(ValueError):
        poll_until(lambda: None, delay_seconds=-1)


def test_invalid_backoff_factor_raises():
    with pytest.raises(ValueError):
        poll_until(lambda: None, backoff_factor=0.5)


def test_max_delay_less_than_initial_raises():
    with pytest.raises(ValueError):
        poll_until(
            lambda: None,
            delay_seconds=10.0,
            max_delay_seconds=5.0,
        )


def test_default_constants_match_openhands_shape():
    """The catalog says the OpenHands version uses 4 attempts; preserve the default."""
    assert DEFAULT_MAX_ATTEMPTS == 4
    assert DEFAULT_DELAY_SECONDS == pytest.approx(3.0)


def test_pollresult_is_frozen():
    result = PollResult(value=1, succeeded=True, attempts=1, elapsed_seconds=0.0)
    with pytest.raises(Exception):
        result.value = 99  # type: ignore[misc]


def test_label_shows_up_in_warning_logs(caplog):
    import logging

    caplog.set_level(logging.WARNING)

    def fn():
        raise RuntimeError("boom")

    poll_until(fn, max_attempts=2, delay_seconds=0.01, label="title-poll")
    assert any("title-poll" in record.message for record in caplog.records)


# -- async apoll_until --


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_async_first_attempt_succeeds():
    async def fn():
        return "hi"

    result = _run(apoll_until(fn, max_attempts=4, delay_seconds=10.0))
    assert result.succeeded is True
    assert result.value == "hi"
    assert result.attempts == 1


def test_async_eventual_success():
    counter = {"n": 0}

    async def fn():
        counter["n"] += 1
        if counter["n"] < 2:
            return None
        return "done"

    result = _run(apoll_until(fn, max_attempts=4, delay_seconds=0.01))
    assert result.succeeded is True
    assert result.value == "done"
    assert result.attempts == 2


def test_async_all_attempts_fail():
    async def fn():
        return None

    result = _run(apoll_until(fn, max_attempts=3, delay_seconds=0.01))
    assert result.succeeded is False
    assert result.attempts == 3


def test_async_exception_caught_continues():
    counter = {"n": 0}

    async def fn():
        counter["n"] += 1
        if counter["n"] == 1:
            raise RuntimeError("transient")
        return "after retry"

    result = _run(apoll_until(fn, max_attempts=3, delay_seconds=0.01))
    assert result.succeeded is True
    assert result.value == "after retry"


def test_async_exception_propagates_when_swallow_disabled():
    async def fn():
        raise ValueError("hard")

    with pytest.raises(ValueError):
        _run(
            apoll_until(
                fn,
                max_attempts=3,
                delay_seconds=0.01,
                swallow_exceptions=False,
            )
        )


def test_async_cancel_check_aborts_loop_early():
    cancel_after = {"n": 0}

    async def fn():
        cancel_after["n"] += 1
        return None

    def cancel():
        # Cancel BEFORE the second invocation.
        return cancel_after["n"] >= 1

    result = _run(
        apoll_until(
            fn,
            max_attempts=5,
            delay_seconds=0.01,
            cancel_check=cancel,
        )
    )
    assert result.succeeded is False
    assert result.last_error == "cancelled"
    # We made it through one attempt before cancel fired.
    assert cancel_after["n"] == 1


def test_async_cancellederror_propagates():
    async def fn():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _run(apoll_until(fn, max_attempts=2, delay_seconds=0.01))


def test_async_invalid_inputs_raise():
    async def fn():
        return None

    with pytest.raises(ValueError):
        _run(apoll_until(fn, max_attempts=0))
    with pytest.raises(ValueError):
        _run(apoll_until(fn, delay_seconds=-0.1))


def test_async_backoff_doubles_delay():
    """Exponential backoff capped at max_delay_seconds."""

    delays_observed: list[float] = []

    counter = {"n": 0}

    async def fn():
        counter["n"] += 1
        return None

    async def patched_sleep(delay):
        delays_observed.append(delay)
        # Skip the actual sleep so tests stay fast.
        return None

    import ultron.utils.poll as poll_mod

    original = asyncio.sleep
    poll_mod.asyncio.sleep = patched_sleep  # type: ignore[attr-defined]
    try:
        _run(
            apoll_until(
                fn,
                max_attempts=4,
                delay_seconds=0.1,
                backoff_factor=2.0,
                max_delay_seconds=10.0,
            )
        )
    finally:
        poll_mod.asyncio.sleep = original  # type: ignore[attr-defined]

    # Three retries -> three sleeps with backoff.
    assert delays_observed == pytest.approx([0.1, 0.2, 0.4])
    assert counter["n"] == 4


def test_backoff_respects_max_delay():
    delays_observed: list[float] = []

    counter = {"n": 0}

    async def fn():
        counter["n"] += 1
        return None

    async def patched_sleep(delay):
        delays_observed.append(delay)
        return None

    import ultron.utils.poll as poll_mod

    original = asyncio.sleep
    poll_mod.asyncio.sleep = patched_sleep  # type: ignore[attr-defined]
    try:
        _run(
            apoll_until(
                fn,
                max_attempts=4,
                delay_seconds=2.0,
                backoff_factor=10.0,
                max_delay_seconds=3.0,
            )
        )
    finally:
        poll_mod.asyncio.sleep = original  # type: ignore[attr-defined]

    # 2.0 -> capped at 3.0 -> 3.0 -> 3.0
    assert delays_observed == pytest.approx([2.0, 3.0, 3.0])


def test_sync_zero_delay_does_not_block():
    counter = {"n": 0}

    def fn():
        counter["n"] += 1
        return None

    start = time.perf_counter()
    poll_until(fn, max_attempts=5, delay_seconds=0)
    elapsed = time.perf_counter() - start
    assert counter["n"] == 5
    # 5 attempts with zero delay should complete in well under 100 ms.
    assert elapsed < 0.2
