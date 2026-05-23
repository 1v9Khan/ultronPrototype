"""Tests for :mod:`ultron.llm.cache_warmer`."""

from __future__ import annotations

import threading
import time

import pytest

from ultron.llm.cache_warmer import (
    CacheWarmer,
    DEFAULT_IDLE_GIVEUP_SECONDS,
    DEFAULT_INTERVAL_SECONDS,
    WarmerTelemetry,
)


def test_default_interval_is_295_seconds():
    """Catalog T9: 5*60 - 5 = 295 — under Anthropic's 5-min TTL."""
    assert DEFAULT_INTERVAL_SECONDS == 295


def test_idle_giveup_is_30_minutes():
    assert DEFAULT_IDLE_GIVEUP_SECONDS == 30 * 60


def test_constructor_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        CacheWarmer(lambda: True, interval_seconds=0)
    with pytest.raises(ValueError):
        CacheWarmer(lambda: True, interval_seconds=-1)


def test_initial_telemetry_is_zero():
    w = CacheWarmer(lambda: True)
    t = w.telemetry
    assert t.pings_sent == 0
    assert t.pings_succeeded == 0
    assert t.pings_failed == 0
    assert t.pings_skipped_idle == 0
    assert t.pings_skipped_empty == 0


def test_running_false_before_start():
    w = CacheWarmer(lambda: True)
    assert w.running is False


def test_running_true_while_active_then_false_after_stop():
    """Use a short interval so we observe the running state."""
    w = CacheWarmer(lambda: True, interval_seconds=10.0)
    w.start()
    assert w.running is True
    w.stop(timeout=2)
    assert w.running is False


def test_ping_fires_after_interval():
    """Use a TINY interval (0.05 s) so a few pings fire before we stop."""
    calls = {"count": 0}

    def send():
        calls["count"] += 1
        return True

    w = CacheWarmer(send, interval_seconds=0.05)
    w.start()
    time.sleep(0.2)
    w.stop(timeout=1)
    assert calls["count"] >= 1
    t = w.telemetry
    assert t.pings_sent >= 1
    assert t.pings_succeeded >= 1


def test_send_returning_false_counts_as_failed():
    w = CacheWarmer(lambda: False, interval_seconds=0.05)
    w.start()
    time.sleep(0.15)
    w.stop(timeout=1)
    t = w.telemetry
    assert t.pings_sent >= 1
    assert t.pings_failed >= 1
    assert t.pings_succeeded == 0


def test_send_raising_exception_counts_as_failed():
    def boom():
        raise RuntimeError("LLM down")

    w = CacheWarmer(boom, interval_seconds=0.05)
    w.start()
    time.sleep(0.15)
    w.stop(timeout=1)
    t = w.telemetry
    assert t.pings_sent >= 1
    assert t.pings_failed >= 1


def test_idle_guard_skips_ping():
    """Stale last_activity -> skip + tick skipped_idle counter."""
    calls = {"count": 0}

    def send():
        calls["count"] += 1
        return True

    # last_activity 100 s ago; giveup at 10 s; should skip every cycle.
    fixed_last = time.monotonic() - 100.0
    w = CacheWarmer(
        send,
        last_activity_provider=lambda: fixed_last,
        interval_seconds=0.05,
        idle_giveup_seconds=10.0,
    )
    w.start()
    time.sleep(0.15)
    w.stop(timeout=1)
    assert calls["count"] == 0
    t = w.telemetry
    assert t.pings_skipped_idle >= 1


def test_idle_guard_disabled_by_zero():
    """idle_giveup_seconds=0 disables the guard."""
    calls = {"count": 0}

    def send():
        calls["count"] += 1
        return True

    fixed_last = time.monotonic() - 9999.0  # ancient
    w = CacheWarmer(
        send,
        last_activity_provider=lambda: fixed_last,
        interval_seconds=0.05,
        idle_giveup_seconds=0.0,
    )
    w.start()
    time.sleep(0.15)
    w.stop(timeout=1)
    assert calls["count"] >= 1


def test_prefix_present_check_skips_when_empty():
    calls = {"count": 0}

    def send():
        calls["count"] += 1
        return True

    w = CacheWarmer(
        send,
        interval_seconds=0.05,
        prefix_present_check=lambda: False,
    )
    w.start()
    time.sleep(0.15)
    w.stop(timeout=1)
    assert calls["count"] == 0
    t = w.telemetry
    assert t.pings_skipped_empty >= 1


def test_prefix_present_check_allows_when_true():
    calls = {"count": 0}

    def send():
        calls["count"] += 1
        return True

    w = CacheWarmer(
        send,
        interval_seconds=0.05,
        prefix_present_check=lambda: True,
    )
    w.start()
    time.sleep(0.15)
    w.stop(timeout=1)
    assert calls["count"] >= 1


def test_start_is_idempotent():
    w = CacheWarmer(lambda: True, interval_seconds=10.0)
    w.start()
    first_thread = w._thread
    w.start()  # no-op
    assert w._thread is first_thread
    w.stop(timeout=1)


def test_stop_when_never_started_does_not_raise():
    w = CacheWarmer(lambda: True)
    w.stop()  # idempotent
    assert w.running is False


def test_warmer_telemetry_dataclass():
    t = WarmerTelemetry(pings_sent=5, pings_succeeded=3, pings_failed=2)
    assert t.pings_sent == 5
    assert t.pings_succeeded == 3
    assert t.pings_failed == 2
