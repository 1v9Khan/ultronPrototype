"""Tests for ultron.subprocess.zombie_killer."""

from __future__ import annotations

import threading
import time
from typing import Optional

import pytest

from ultron.subprocess import zombie_killer as zk


class _FakeClock:
    """Manually-advanced monotonic clock for deterministic age tests."""

    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.RLock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


@pytest.fixture
def fake_clock() -> _FakeClock:
    return _FakeClock()


@pytest.fixture
def killed_pids() -> list[int]:
    """Mutable list capturing pid arguments passed to the terminator."""
    return []


@pytest.fixture
def killer_factory(fake_clock, killed_pids):
    """Factory returning a fresh ZombieKiller with test hooks."""

    def _make(
        *,
        hard_timeout_s: float = 100.0,
        warn_age_s: float = 50.0,
        warn_rss_mb: int = 100,
        rss_value: Optional[int] = 0,
        terminator_success: bool = True,
    ) -> zk.ZombieKiller:
        def terminator(pid: int) -> bool:
            killed_pids.append(pid)
            return terminator_success

        def rss_probe(_pid: int) -> Optional[int]:
            return rss_value

        return zk.ZombieKiller(
            hard_timeout_s=hard_timeout_s,
            poll_interval_s=1000.0,  # never auto-poll in tests
            warn_age_s=warn_age_s,
            warn_rss_mb=warn_rss_mb,
            clock=fake_clock,
            terminator=terminator,
            rss_probe=rss_probe,
        )

    return _make


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_returns_entry(self, killer_factory) -> None:
        killer = killer_factory()
        entry = killer.register(123, "test")
        assert entry.pid == 123
        assert entry.label == "test"
        assert entry.persistent is False

    def test_register_persistent(self, killer_factory) -> None:
        killer = killer_factory()
        entry = killer.register(456, "mcp", persistent=True)
        assert entry.persistent is True

    def test_re_register_updates_in_place(self, killer_factory) -> None:
        killer = killer_factory()
        first = killer.register(7, "a")
        second = killer.register(7, "b", persistent=True, owner="me")
        assert first is second
        assert second.label == "b"
        assert second.persistent is True
        assert second.owner == "me"

    def test_unregister(self, killer_factory) -> None:
        killer = killer_factory()
        killer.register(11, "x")
        assert killer.unregister(11) is True
        assert killer.unregister(11) is False
        assert killer.lookup(11) is None

    def test_mark_persistent(self, killer_factory) -> None:
        killer = killer_factory()
        killer.register(22, "x")
        assert killer.mark_persistent(22) is True
        entry = killer.lookup(22)
        assert entry is not None and entry.persistent is True
        assert killer.mark_persistent(999) is False

    def test_list_tracked_sorted(self, fake_clock, killer_factory) -> None:
        killer = killer_factory()
        killer.register(1, "a")
        fake_clock.advance(5.0)
        killer.register(2, "b")
        fake_clock.advance(5.0)
        killer.register(3, "c")
        order = [e.pid for e in killer.list_tracked()]
        assert order == [1, 2, 3]

    def test_find_by_label(self, killer_factory) -> None:
        killer = killer_factory()
        killer.register(1, "mcp")
        killer.register(2, "mcp")
        killer.register(3, "other")
        matches = killer.find_by_label("mcp")
        assert sorted(e.pid for e in matches) == [1, 2]


# ---------------------------------------------------------------------------
# Sweep semantics
# ---------------------------------------------------------------------------

class TestSweep:
    def test_no_kill_when_under_timeout(
        self, fake_clock, killer_factory, killed_pids,
    ) -> None:
        killer = killer_factory(hard_timeout_s=600.0)
        killer.register(1, "x")
        fake_clock.advance(60.0)
        reports = killer.sweep_once()
        assert reports == []
        assert killed_pids == []
        assert killer.lookup(1) is not None

    def test_kill_when_over_timeout(
        self, fake_clock, killer_factory, killed_pids,
    ) -> None:
        killer = killer_factory(hard_timeout_s=60.0)
        killer.register(1, "x")
        fake_clock.advance(120.0)
        reports = killer.sweep_once()
        assert len(reports) == 1
        assert reports[0].pid == 1
        assert reports[0].action == "terminated"
        assert killed_pids == [1]
        assert killer.lookup(1) is None

    def test_persistent_never_killed(
        self, fake_clock, killer_factory, killed_pids,
    ) -> None:
        killer = killer_factory(hard_timeout_s=10.0)
        killer.register(1, "mcp", persistent=True)
        fake_clock.advance(100.0)
        reports = killer.sweep_once()
        assert reports == []
        assert killed_pids == []
        assert killer.lookup(1) is not None

    def test_per_process_timeout_overrides_global(
        self, fake_clock, killer_factory, killed_pids,
    ) -> None:
        killer = killer_factory(hard_timeout_s=10.0)
        killer.register(1, "long", hard_timeout_s=1000.0)
        fake_clock.advance(500.0)
        reports = killer.sweep_once()
        assert reports == []

    def test_warning_tier_no_kill(
        self, fake_clock, killer_factory, killed_pids,
    ) -> None:
        killer = killer_factory(
            hard_timeout_s=1000.0,
            warn_age_s=50.0,
            warn_rss_mb=100,
            rss_value=500,
        )
        killer.register(1, "x")
        fake_clock.advance(100.0)
        reports = killer.sweep_once()
        assert len(reports) == 1
        assert reports[0].action == "warned"
        assert reports[0].rss_mb == 500
        assert killed_pids == []
        # The process is still tracked after a warning.
        assert killer.lookup(1) is not None

    def test_warning_skipped_when_rss_below_threshold(
        self, fake_clock, killer_factory,
    ) -> None:
        killer = killer_factory(
            hard_timeout_s=1000.0,
            warn_age_s=50.0,
            warn_rss_mb=1000,
            rss_value=50,
        )
        killer.register(1, "x")
        fake_clock.advance(100.0)
        reports = killer.sweep_once()
        assert reports == []

    def test_warning_skipped_when_age_below_threshold(
        self, fake_clock, killer_factory,
    ) -> None:
        killer = killer_factory(
            hard_timeout_s=1000.0,
            warn_age_s=200.0,
            warn_rss_mb=100,
            rss_value=500,
        )
        killer.register(1, "x")
        fake_clock.advance(100.0)
        reports = killer.sweep_once()
        assert reports == []

    def test_terminator_failure_keeps_process(
        self, fake_clock, killer_factory,
    ) -> None:
        killer = killer_factory(hard_timeout_s=10.0, terminator_success=False)
        killer.register(1, "x")
        fake_clock.advance(60.0)
        reports = killer.sweep_once()
        assert reports == []
        assert killer.lookup(1) is not None

    def test_on_terminate_callback_fires(
        self, fake_clock, killer_factory,
    ) -> None:
        killer = killer_factory(hard_timeout_s=10.0)
        captured: list[zk.ZombieReport] = []
        killer.register(1, "x", on_terminate=captured.append)
        fake_clock.advance(60.0)
        killer.sweep_once()
        assert len(captured) == 1
        assert captured[0].pid == 1

    def test_on_terminate_exception_does_not_break(
        self, fake_clock, killer_factory,
    ) -> None:
        def broken(_: zk.ZombieReport) -> None:
            raise RuntimeError("boom")
        killer = killer_factory(hard_timeout_s=10.0)
        killer.register(1, "x", on_terminate=broken)
        fake_clock.advance(60.0)
        reports = killer.sweep_once()
        # Sweep still succeeds despite the broken callback.
        assert len(reports) == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_idempotent(self, killer_factory) -> None:
        killer = killer_factory()
        killer.start()
        try:
            killer.start()  # second call is a no-op
            assert killer.is_running()
        finally:
            killer.shutdown()
        assert killer.is_running() is False

    def test_shutdown_idempotent(self, killer_factory) -> None:
        killer = killer_factory()
        killer.shutdown()
        killer.shutdown()  # safe to call without start

    def test_recent_reports_bounded(self, fake_clock, killer_factory) -> None:
        killer = killer_factory(hard_timeout_s=1.0)
        for pid in range(1, 5):
            killer.register(pid, f"p{pid}")
        fake_clock.advance(10.0)
        killer.sweep_once()
        reports = killer.recent_reports(limit=10)
        assert len(reports) == 4
        killer.clear_reports()
        assert killer.recent_reports() == []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_returns_same_instance(self) -> None:
        zk.reset_zombie_killer_for_testing()
        try:
            a = zk.get_zombie_killer()
            b = zk.get_zombie_killer()
            assert a is b
        finally:
            zk.reset_zombie_killer_for_testing()

    def test_reset_drops_singleton(self) -> None:
        a = zk.get_zombie_killer()
        zk.reset_zombie_killer_for_testing()
        b = zk.get_zombie_killer()
        assert a is not b
        zk.reset_zombie_killer_for_testing()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_hard_timeout_is_ten_minutes(self) -> None:
        assert zk.DEFAULT_HARD_TIMEOUT_S == 10 * 60.0

    def test_poll_interval_is_one_minute(self) -> None:
        assert zk.DEFAULT_POLL_INTERVAL_S == 60.0
