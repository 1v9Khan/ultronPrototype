"""Tests for ultron.bus.service -- pub/sub semantics + race safety."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from ultron.bus.event import BusEvent
from ultron.bus.service import (
    DEFAULT_SLOW_SUBSCRIBER_WARN_MS,
    Bus,
    get_bus,
    publish,
    reset_bus_for_testing,
    set_slow_subscriber_recorder,
    subscribe,
    subscribe_all,
)


@pytest.fixture
def bus() -> Bus:
    """Fresh bus per test -- avoids singleton state bleed."""
    return reset_bus_for_testing()


# ---------------------------------------------------------------------------
# Subscribe + publish basics
# ---------------------------------------------------------------------------


def test_subscribe_fires_callback_on_publish(bus: Bus) -> None:
    event = BusEvent.define("test.basic", {"x": int})
    received = []
    bus.subscribe(event, lambda p: received.append(p))

    bus.publish(event, {"x": 42})

    assert len(received) == 1
    assert received[0].properties == {"x": 42}
    assert received[0].type == "test.basic"


def test_publish_to_no_subscribers_is_noop(bus: Bus) -> None:
    event = BusEvent.define("test.empty", {"x": int})
    # No subscribers -- must not raise.
    bus.publish(event, {"x": 1})
    assert bus.published_count() == 1


def test_multiple_subscribers_all_fire(bus: Bus) -> None:
    event = BusEvent.define("test.multi", {"x": int})
    received_a = []
    received_b = []
    received_c = []
    bus.subscribe(event, lambda p: received_a.append(p))
    bus.subscribe(event, lambda p: received_b.append(p))
    bus.subscribe(event, lambda p: received_c.append(p))

    bus.publish(event, {"x": 1})

    assert len(received_a) == 1
    assert len(received_b) == 1
    assert len(received_c) == 1


def test_subscriber_count_tracks_subscribers(bus: Bus) -> None:
    event = BusEvent.define("test.count", {"x": int})
    assert bus.subscriber_count(event) == 0
    bus.subscribe(event, lambda p: None)
    assert bus.subscriber_count(event) == 1
    bus.subscribe(event, lambda p: None)
    assert bus.subscriber_count(event) == 2


def test_subscriber_count_total(bus: Bus) -> None:
    e1 = BusEvent.define("test.a", {})
    e2 = BusEvent.define("test.b", {})
    bus.subscribe(e1, lambda p: None)
    bus.subscribe(e2, lambda p: None)
    bus.subscribe_all(lambda p: None)
    assert bus.subscriber_count() == 3


# ---------------------------------------------------------------------------
# Wildcard subscribers
# ---------------------------------------------------------------------------


def test_wildcard_receives_all_events(bus: Bus) -> None:
    e1 = BusEvent.define("test.alpha", {})
    e2 = BusEvent.define("test.beta", {})
    received = []
    bus.subscribe_all(lambda p: received.append(p.type))

    bus.publish(e1, {})
    bus.publish(e2, {})

    assert received == ["test.alpha", "test.beta"]


def test_typed_and_wildcard_both_fire(bus: Bus) -> None:
    event = BusEvent.define("test.both", {})
    typed = []
    wildcard = []
    bus.subscribe(event, lambda p: typed.append(p))
    bus.subscribe_all(lambda p: wildcard.append(p))

    bus.publish(event, {})

    assert len(typed) == 1
    assert len(wildcard) == 1


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------


def test_unsubscribe_stops_callback(bus: Bus) -> None:
    event = BusEvent.define("test.unsub", {})
    received = []
    unsub = bus.subscribe(event, lambda p: received.append(p))

    bus.publish(event, {})
    assert len(received) == 1

    unsub()
    bus.publish(event, {})
    assert len(received) == 1  # No new event after unsub.


def test_unsubscribe_idempotent(bus: Bus) -> None:
    event = BusEvent.define("test.unsub2", {})
    unsub = bus.subscribe(event, lambda p: None)
    unsub()
    unsub()  # Must not raise.


def test_unsubscribe_removes_one_subscriber_only(bus: Bus) -> None:
    event = BusEvent.define("test.unsub_one", {})
    received_a = []
    received_b = []
    unsub_a = bus.subscribe(event, lambda p: received_a.append(p))
    bus.subscribe(event, lambda p: received_b.append(p))

    unsub_a()
    bus.publish(event, {})

    assert len(received_a) == 0
    assert len(received_b) == 1


def test_unsubscribe_wildcard(bus: Bus) -> None:
    event = BusEvent.define("test.wildcard_unsub", {})
    received = []
    unsub = bus.subscribe_all(lambda p: received.append(p))

    bus.publish(event, {})
    assert len(received) == 1

    unsub()
    bus.publish(event, {})
    assert len(received) == 1


def test_subscriber_count_after_unsubscribe(bus: Bus) -> None:
    event = BusEvent.define("test.count_after_unsub", {})
    unsub = bus.subscribe(event, lambda p: None)
    assert bus.subscriber_count(event) == 1
    unsub()
    # Channel cleaned up to keep counts accurate.
    assert bus.subscriber_count(event) == 0


# ---------------------------------------------------------------------------
# Fail-open: subscriber exceptions don't break others
# ---------------------------------------------------------------------------


def test_callback_exception_does_not_break_other_subscribers(
    bus: Bus, caplog: pytest.LogCaptureFixture,
) -> None:
    event = BusEvent.define("test.fail_open", {})
    received_after_failure = []

    def bad_subscriber(_p):
        raise RuntimeError("intentional test failure")

    bus.subscribe(event, bad_subscriber)
    bus.subscribe(event, lambda p: received_after_failure.append(p))

    bus.publish(event, {})

    # Second subscriber still fired.
    assert len(received_after_failure) == 1


def test_callback_exception_swallowed_not_raised(bus: Bus) -> None:
    event = BusEvent.define("test.fail_swallow", {})
    bus.subscribe(event, lambda p: (_ for _ in ()).throw(ValueError("boom")))

    # Must not raise.
    bus.publish(event, {})


# ---------------------------------------------------------------------------
# Schema validation -- best-effort, delivered anyway
# ---------------------------------------------------------------------------


def test_schema_mismatch_still_delivers(bus: Bus) -> None:
    event = BusEvent.define("test.bad_schema", {"x": int})
    received = []
    bus.subscribe(event, lambda p: received.append(p))

    # Wrong type -- logged warning but still dispatched.
    bus.publish(event, {"x": "not_an_int"})

    assert len(received) == 1


# ---------------------------------------------------------------------------
# Eager subscribe -- no lost-events race (the bug opencode fixed)
# ---------------------------------------------------------------------------


def test_subscribe_then_publish_in_quick_succession(bus: Bus) -> None:
    """The opencode race: subscribe-then-publish must not lose events.

    The bug: if subscribe returns BEFORE the callback is in the dispatch
    table, the immediately-following publish misses it. Our
    implementation acquires the dispatch slot under the lock inside
    subscribe(), so this test passes by construction.
    """
    event = BusEvent.define("test.race", {})
    received = []

    # Tight loop: many subscribes then publishes per iteration.
    for _ in range(100):
        unsub = bus.subscribe(event, lambda p: received.append(p))
        bus.publish(event, {})
        unsub()

    assert len(received) == 100


def test_concurrent_subscribe_and_publish_no_crash(bus: Bus) -> None:
    """Two threads alternating subscribe + publish must not deadlock or crash."""
    event = BusEvent.define("test.concurrent", {})
    stop = threading.Event()
    received: list = []

    def publisher() -> None:
        for _ in range(200):
            bus.publish(event, {})
        stop.set()

    def subscriber_churn() -> None:
        unsubs = []
        while not stop.is_set():
            unsubs.append(bus.subscribe(event, lambda p: received.append(p)))
        for u in unsubs:
            u()

    t_pub = threading.Thread(target=publisher, name="bus-test-pub")
    t_sub = threading.Thread(target=subscriber_churn, name="bus-test-sub")
    t_pub.start()
    t_sub.start()
    t_pub.join(timeout=5.0)
    t_sub.join(timeout=5.0)
    assert not t_pub.is_alive() and not t_sub.is_alive()


# ---------------------------------------------------------------------------
# Subscriber list snapshot -- mutate during dispatch is safe
# ---------------------------------------------------------------------------


def test_subscribe_during_dispatch_does_not_double_fire(bus: Bus) -> None:
    event = BusEvent.define("test.snapshot", {})
    received_b = []

    def subscriber_a(_p):
        # Subscribe a SECOND callback while the first one is mid-dispatch.
        bus.subscribe(event, lambda p: received_b.append(p))

    bus.subscribe(event, subscriber_a)

    # First publish: only subscriber_a is registered, so received_b stays empty.
    bus.publish(event, {})
    assert len(received_b) == 0

    # Second publish: now both fire; subscriber_a registers a THIRD callback,
    # but the snapshot taken at start of publish has 2 entries.
    bus.publish(event, {})
    assert len(received_b) == 1


def test_unsubscribe_during_dispatch_safe(bus: Bus) -> None:
    event = BusEvent.define("test.unsub_during", {})
    received_b = []
    unsub_a = [None]

    def subscriber_a(_p):
        unsub_a[0]()  # Unsubscribe self mid-dispatch.

    unsub_a[0] = bus.subscribe(event, subscriber_a)
    bus.subscribe(event, lambda p: received_b.append(p))

    # subscriber_a unsubscribes itself; subscriber_b still fires (snapshot).
    bus.publish(event, {})
    assert len(received_b) == 1

    # Subscriber_a is gone for the next publish.
    bus.publish(event, {})
    assert len(received_b) == 2


# ---------------------------------------------------------------------------
# Module-level shortcuts
# ---------------------------------------------------------------------------


def test_module_publish_subscribe_use_singleton(bus: Bus) -> None:
    """The module-level publish/subscribe must hit the same instance."""
    event = BusEvent.define("test.module", {})
    received = []
    unsub = subscribe(event, lambda p: received.append(p))
    publish(event, {})
    assert len(received) == 1
    unsub()


def test_get_bus_returns_singleton(bus: Bus) -> None:
    a = get_bus()
    b = get_bus()
    assert a is b


def test_reset_bus_returns_fresh() -> None:
    a = get_bus()
    reset_bus_for_testing()
    b = get_bus()
    assert a is not b


def test_subscribe_all_module_level(bus: Bus) -> None:
    e1 = BusEvent.define("test.mod.a", {})
    e2 = BusEvent.define("test.mod.b", {})
    received = []
    unsub = subscribe_all(lambda p: received.append(p.type))
    publish(e1, {})
    publish(e2, {})
    assert received == ["test.mod.a", "test.mod.b"]
    unsub()


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def test_published_count_increments(bus: Bus) -> None:
    event = BusEvent.define("test.count_inc", {})
    assert bus.published_count() == 0
    bus.publish(event, {})
    bus.publish(event, {})
    bus.publish(event, {})
    assert bus.published_count() == 3


# ---------------------------------------------------------------------------
# Slow-subscriber watchdog (2026-05-22)
# ---------------------------------------------------------------------------


def test_default_warn_threshold_constant() -> None:
    """The constant is the runtime default; pin it to catch accidental edits."""
    assert DEFAULT_SLOW_SUBSCRIBER_WARN_MS == 15.0


def test_fast_subscriber_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A callback that returns immediately must not trip the watchdog."""
    bus = Bus()  # default 15 ms threshold
    event = BusEvent.define("test.fast", {})
    bus.subscribe(event, lambda p: None)

    with caplog.at_level(logging.WARNING, logger="ultron.bus"):
        bus.publish(event, {})

    assert bus.slow_subscriber_count() == 0
    assert not any("threshold" in r.message for r in caplog.records)


def test_slow_subscriber_warns_and_increments_counter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A callback that exceeds the threshold logs WARN + bumps the counter."""
    bus = Bus(slow_subscriber_warn_ms=5.0)
    event = BusEvent.define("test.slow", {})

    def slow_cb(_p):
        time.sleep(0.020)  # 20 ms > 5 ms threshold

    bus.subscribe(event, slow_cb)

    with caplog.at_level(logging.WARNING, logger="ultron.bus"):
        bus.publish(event, {})

    assert bus.slow_subscriber_count() == 1
    # One log line names the event type + a number followed by "ms".
    slow_logs = [r for r in caplog.records if "test.slow" in r.message and "threshold" in r.message]
    assert len(slow_logs) == 1


def test_slow_subscriber_counter_accumulates_across_publishes() -> None:
    """Each slow callback fire bumps the counter; counts per-occurrence not per-subscriber."""
    bus = Bus(slow_subscriber_warn_ms=5.0)
    event = BusEvent.define("test.slow_count", {})
    bus.subscribe(event, lambda p: time.sleep(0.020))

    bus.publish(event, {})
    bus.publish(event, {})
    bus.publish(event, {})

    assert bus.slow_subscriber_count() == 3


def test_subscriber_exception_does_not_bump_slow_counter() -> None:
    """The exception path short-circuits before the timing check.

    A callback that raises has done some unknown amount of work; we
    don't want to double-flag it (the exception warning already logged).
    """
    bus = Bus(slow_subscriber_warn_ms=5.0)
    event = BusEvent.define("test.slow_exception", {})

    def slow_then_raise(_p):
        time.sleep(0.020)
        raise RuntimeError("boom")

    bus.subscribe(event, slow_then_raise)

    bus.publish(event, {})

    # Exception path logged at WARN already; the elapsed-time path
    # was skipped so the slow-subscriber counter stays at 0.
    assert bus.slow_subscriber_count() == 0


def test_slow_subscriber_recorder_callback_fires() -> None:
    """When a recorder is installed, slow-subscriber events are forwarded."""
    bus = Bus(slow_subscriber_warn_ms=5.0)
    event = BusEvent.define("test.recorder", {})

    received: list[tuple[str, str]] = []

    def recorder(category: str, reason: str) -> None:
        received.append((category, reason))

    set_slow_subscriber_recorder(recorder)
    try:
        bus.subscribe(event, lambda p: time.sleep(0.020))
        bus.publish(event, {})

        assert received == [("bus_slow_subscriber", "test.recorder")]
    finally:
        set_slow_subscriber_recorder(None)


def test_recorder_exception_is_swallowed() -> None:
    """A buggy recorder must not break the bus's fail-open contract."""
    bus = Bus(slow_subscriber_warn_ms=5.0)
    event = BusEvent.define("test.recorder_buggy", {})

    def bad_recorder(_c: str, _r: str) -> None:
        raise RuntimeError("recorder is broken")

    set_slow_subscriber_recorder(bad_recorder)
    try:
        bus.subscribe(event, lambda p: time.sleep(0.020))
        # Must not raise.
        bus.publish(event, {})
        # Watchdog still recorded the slow subscriber locally.
        assert bus.slow_subscriber_count() == 1
    finally:
        set_slow_subscriber_recorder(None)


def test_slow_subscriber_warn_ms_accessor() -> None:
    """The configured threshold is readable for diagnostics."""
    bus = Bus(slow_subscriber_warn_ms=42.0)
    assert bus.slow_subscriber_warn_ms() == 42.0


def test_slow_subscriber_does_not_block_other_subscribers() -> None:
    """A slow subscriber bumps the counter but doesn't stop later ones firing.

    The watchdog is observational only -- the synchronous-on-publisher-thread
    contract is preserved.
    """
    bus = Bus(slow_subscriber_warn_ms=5.0)
    event = BusEvent.define("test.slow_then_fast", {})
    received: list = []
    bus.subscribe(event, lambda p: time.sleep(0.020))
    bus.subscribe(event, lambda p: received.append(p))

    bus.publish(event, {})

    assert len(received) == 1
    assert bus.slow_subscriber_count() == 1
