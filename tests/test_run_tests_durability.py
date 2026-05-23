"""Tests for the multi-layer safeguards in :mod:`scripts.run_tests`.

The runner itself is the most critical piece of test infrastructure
in the project — every other test relies on it being correct. So
the safeguard logic gets its own test coverage.

We can't directly import `scripts.run_tests` as a Python module by
its package path (``scripts/`` isn't a package), so the tests load
the file by absolute path via :mod:`importlib`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_tests_module() -> Any:
    """Load scripts/run_tests.py as a module for direct inspection."""
    here = Path(__file__).resolve().parent
    script = here.parent / "scripts" / "run_tests.py"
    spec = importlib.util.spec_from_file_location("scripts_run_tests", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pre-flight environment check
# ---------------------------------------------------------------------------


def test_preflight_environment_ok(run_tests_module):
    ok, reason = run_tests_module._preflight_environment()
    # On the dev machine + CI all three preconditions pass.
    assert ok, reason


# ---------------------------------------------------------------------------
# Sweep lock (mutex)
# ---------------------------------------------------------------------------


def test_acquire_release_lock_roundtrip(run_tests_module, monkeypatch, tmp_path: Path):
    fake_lock = tmp_path / "fake_lock"
    monkeypatch.setattr(run_tests_module, "SWEEP_LOCK_FILE", fake_lock)

    assert run_tests_module._acquire_sweep_lock() is True
    assert fake_lock.exists()
    assert fake_lock.read_text().strip() == str(os.getpid())

    run_tests_module._release_sweep_lock()
    assert not fake_lock.exists()


def test_release_lock_doesnt_touch_other_owners(run_tests_module, monkeypatch, tmp_path: Path):
    fake_lock = tmp_path / "fake_lock"
    monkeypatch.setattr(run_tests_module, "SWEEP_LOCK_FILE", fake_lock)

    # Pretend someone else holds it.
    fake_lock.write_text("99999999", encoding="utf-8")
    run_tests_module._release_sweep_lock()
    # The lock file should still exist — we don't own it.
    assert fake_lock.exists()


def test_acquire_lock_when_psutil_says_pid_dead_recovers(
    run_tests_module, monkeypatch, tmp_path: Path,
):
    """Stale lock (PID not alive) is recovered + overwritten."""
    fake_lock = tmp_path / "fake_lock"
    monkeypatch.setattr(run_tests_module, "SWEEP_LOCK_FILE", fake_lock)
    # Use a PID that's guaranteed to NOT be a python process —
    # PID 1 is init/system but psutil.pid_exists returns True for it
    # on most systems; use a large unlikely PID instead.
    fake_lock.write_text("999999999", encoding="utf-8")
    assert run_tests_module._acquire_sweep_lock() is True
    assert fake_lock.read_text().strip() == str(os.getpid())


def test_release_lock_is_idempotent(run_tests_module, monkeypatch, tmp_path: Path):
    fake_lock = tmp_path / "fake_lock"
    monkeypatch.setattr(run_tests_module, "SWEEP_LOCK_FILE", fake_lock)

    run_tests_module._acquire_sweep_lock()
    run_tests_module._release_sweep_lock()
    run_tests_module._release_sweep_lock()  # second call must not raise


# ---------------------------------------------------------------------------
# Competing-pytest discovery
# ---------------------------------------------------------------------------


def test_list_competing_excludes_self(run_tests_module):
    """The current python (running pytest right now) IS technically a
    competing pytest — but the function excludes the ancestor chain
    so we don't kill ourselves."""
    found = run_tests_module._list_competing_pytests()
    # All entries in `found` MUST have PIDs that are NOT in our ancestor chain.
    try:
        import psutil
        ancestors = {a.pid for a in psutil.Process(os.getpid()).parents()}
        ancestors.add(os.getpid())
    except ImportError:
        pytest.skip("psutil unavailable")
    for entry in found:
        assert entry["pid"] not in ancestors


def test_list_competing_include_age_attaches_age_field(run_tests_module):
    """include_age=True attaches an age_seconds field."""
    found = run_tests_module._list_competing_pytests(include_age=True)
    for entry in found:
        assert "age_seconds" in entry
        assert isinstance(entry["age_seconds"], float)


# ---------------------------------------------------------------------------
# Watchdog (heartbeat staleness + wall-clock deadline)
# ---------------------------------------------------------------------------


class _FakePopen:
    """Tiny fake of :class:`subprocess.Popen` for watchdog testing."""

    def __init__(self) -> None:
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls = 0
        self._poll_return: Any = None

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True

    def wait(self, timeout: float = None) -> int:  # noqa: ARG002
        self.wait_calls += 1
        return 0

    def poll(self) -> Any:
        return self._poll_return


def test_watchdog_fires_on_wall_clock(run_tests_module, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        run_tests_module,
        "HEARTBEAT_PATH",
        tmp_path / ".heartbeat",
    )
    monkeypatch.setattr(
        run_tests_module, "WATCHDOG_POLL_INTERVAL_SECONDS", 0.05,
    )
    fake_proc = _FakePopen()
    triggers: list[tuple[str, float]] = []
    wd = run_tests_module._Watchdog(
        fake_proc,
        max_runtime_seconds=0.1,
        stale_heartbeat_seconds=999.0,
        on_trigger=lambda reason, value: triggers.append((reason, value)),
    )
    wd.start()
    time.sleep(0.5)
    wd.stop()
    assert wd.triggered_reason == "wall_clock"
    assert fake_proc.terminate_called is True
    assert triggers and triggers[0][0] == "wall_clock"


def test_watchdog_fires_on_stale_heartbeat(
    run_tests_module, monkeypatch, tmp_path: Path,
):
    hb = tmp_path / ".heartbeat"
    hb.write_text(str(time.time()), encoding="utf-8")
    monkeypatch.setattr(run_tests_module, "HEARTBEAT_PATH", hb)
    monkeypatch.setattr(
        run_tests_module, "WATCHDOG_POLL_INTERVAL_SECONDS", 0.05,
    )

    # Set the heartbeat mtime to 100s ago so the watchdog sees it as stale.
    old_time = time.time() - 100
    os.utime(hb, (old_time, old_time))

    fake_proc = _FakePopen()
    triggers: list[tuple[str, float]] = []
    wd = run_tests_module._Watchdog(
        fake_proc,
        max_runtime_seconds=999.0,
        stale_heartbeat_seconds=10.0,
        on_trigger=lambda reason, value: triggers.append((reason, value)),
    )
    wd.start()
    time.sleep(0.3)
    wd.stop()
    assert wd.triggered_reason == "heartbeat"
    assert fake_proc.terminate_called is True


def test_watchdog_doesnt_fire_when_heartbeat_fresh(
    run_tests_module, monkeypatch, tmp_path: Path,
):
    hb = tmp_path / ".heartbeat"
    hb.write_text(str(time.time()), encoding="utf-8")
    monkeypatch.setattr(run_tests_module, "HEARTBEAT_PATH", hb)
    monkeypatch.setattr(
        run_tests_module, "WATCHDOG_POLL_INTERVAL_SECONDS", 0.05,
    )

    fake_proc = _FakePopen()

    def _refresh_heartbeat():
        for _ in range(20):
            try:
                hb.write_text(str(time.time()), encoding="utf-8")
            except OSError:
                pass
            time.sleep(0.05)

    refresher = threading.Thread(target=_refresh_heartbeat, daemon=True)
    refresher.start()

    wd = run_tests_module._Watchdog(
        fake_proc,
        max_runtime_seconds=999.0,
        stale_heartbeat_seconds=10.0,
        on_trigger=lambda *_args: None,
    )
    wd.start()
    time.sleep(0.4)
    wd.stop()
    refresher.join(timeout=2)
    assert wd.triggered_reason is None
    assert fake_proc.terminate_called is False


def test_watchdog_stops_when_subprocess_exits(
    run_tests_module, monkeypatch, tmp_path: Path,
):
    hb = tmp_path / ".heartbeat"
    hb.write_text(str(time.time()), encoding="utf-8")
    monkeypatch.setattr(run_tests_module, "HEARTBEAT_PATH", hb)
    monkeypatch.setattr(
        run_tests_module, "WATCHDOG_POLL_INTERVAL_SECONDS", 0.05,
    )

    fake_proc = _FakePopen()
    fake_proc._poll_return = 0  # simulate clean exit

    wd = run_tests_module._Watchdog(
        fake_proc,
        max_runtime_seconds=999.0,
        stale_heartbeat_seconds=999.0,
        on_trigger=lambda *_args: None,
    )
    wd.start()
    time.sleep(0.2)
    wd.stop()
    assert wd.triggered_reason is None
    assert fake_proc.terminate_called is False


# ---------------------------------------------------------------------------
# Session validation
# ---------------------------------------------------------------------------


def test_session_completed_cleanly_reads_session_end(
    run_tests_module, monkeypatch, tmp_path: Path,
):
    progress = tmp_path / ".progress.jsonl"
    progress.write_text(
        json.dumps({"event": "session_start", "ts": 1.0}) + "\n"
        + json.dumps({"event": "passed", "test": "t1", "ts": 2.0}) + "\n"
        + json.dumps({"event": "session_end", "exitstatus": 0, "ts": 3.0}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_tests_module, "PROGRESS_LOG_PATH", progress)
    end = run_tests_module._session_completed_cleanly()
    assert end is not None
    assert end["exitstatus"] == 0


def test_session_completed_cleanly_missing_file(run_tests_module, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        run_tests_module, "PROGRESS_LOG_PATH", tmp_path / "absent.jsonl",
    )
    assert run_tests_module._session_completed_cleanly() is None


def test_session_completed_cleanly_no_session_end(
    run_tests_module, monkeypatch, tmp_path: Path,
):
    progress = tmp_path / ".progress.jsonl"
    progress.write_text(
        json.dumps({"event": "session_start", "ts": 1.0}) + "\n"
        + json.dumps({"event": "passed", "test": "t1", "ts": 2.0}) + "\n",
        # NO session_end event — sweep was killed mid-stream
        encoding="utf-8",
    )
    monkeypatch.setattr(run_tests_module, "PROGRESS_LOG_PATH", progress)
    assert run_tests_module._session_completed_cleanly() is None


def test_session_completed_cleanly_handles_malformed_line(
    run_tests_module, monkeypatch, tmp_path: Path,
):
    progress = tmp_path / ".progress.jsonl"
    progress.write_text("not valid json\n", encoding="utf-8")
    monkeypatch.setattr(run_tests_module, "PROGRESS_LOG_PATH", progress)
    assert run_tests_module._session_completed_cleanly() is None


# ---------------------------------------------------------------------------
# Tunables sanity
# ---------------------------------------------------------------------------


def test_orphan_age_constant_sane(run_tests_module):
    assert run_tests_module.ORPHAN_AGE_SECONDS == 5 * 60


def test_default_max_runtime_constant_sane(run_tests_module):
    assert run_tests_module.DEFAULT_MAX_RUNTIME_SECONDS == 10 * 60


def test_default_stale_heartbeat_constant_sane(run_tests_module):
    # Should be 3x the per-test timeout (30s) at minimum.
    assert run_tests_module.DEFAULT_STALE_HEARTBEAT_SECONDS >= 60


def test_watchdog_poll_interval_constant_sane(run_tests_module):
    assert 0.5 <= run_tests_module.WATCHDOG_POLL_INTERVAL_SECONDS <= 10.0
