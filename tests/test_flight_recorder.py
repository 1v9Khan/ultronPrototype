"""Pins the crash flight recorder (2026-07-26).

Built to catch a silent native ``__fastfail`` (0xc0000409) that left no
traceback, no stderr text and no WER dump. Contract: default OFF, stdlib
only, atomic alternating A/B snapshots so a snapshot torn by the crash
still leaves the previous one readable.
"""

from __future__ import annotations

import time

import pytest

from kenning import flight_recorder as fr


def test_disabled_by_default(monkeypatch) -> None:
    # Opt-in: the live process must be unchanged unless the operator asks
    # for the recorder. It exists for the NEXT unexplained native abort.
    monkeypatch.delenv("KENNING_FLIGHT_RECORDER", raising=False)
    assert fr.enabled() is False
    assert fr.maybe_start_from_env() is False


@pytest.mark.parametrize("val,want", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("nope", False),
])
def test_env_gate(monkeypatch, val, want) -> None:
    monkeypatch.setenv("KENNING_FLIGHT_RECORDER", val)
    assert fr.enabled() is want


def test_writes_snapshots_with_thread_stacks(tmp_path, monkeypatch) -> None:
    # Reset the module-level idempotence latch so this test can start one.
    monkeypatch.setattr(fr, "_started", False, raising=False)
    assert fr.start(log_dir=tmp_path, interval_s=0.02) is True
    # Second call is a no-op, not a second thread.
    assert fr.start(log_dir=tmp_path, interval_s=0.02) is True

    deadline = time.monotonic() + 5.0
    a, b = tmp_path / "_flight_A.txt", tmp_path / "_flight_B.txt"
    while time.monotonic() < deadline and not (a.exists() and b.exists()):
        time.sleep(0.02)
    assert a.exists() and b.exists(), "both A/B snapshots should appear"

    body = a.read_text(encoding="utf-8")
    assert "snapshot #" in body and "pid=" in body
    # The recorder must capture OTHER threads' stacks, including its own loop.
    assert "--- thread " in body
    assert "flight_recorder.py" in body
    # No stray temp file left behind (writes are atomic via .tmp -> replace).
    assert not list(tmp_path.glob("*.tmp"))


def test_start_failure_is_fail_open(tmp_path, monkeypatch) -> None:
    # An unusable log dir must return False, never raise into the host app.
    monkeypatch.setattr(fr, "_started", False, raising=False)
    blocker = tmp_path / "notadir"
    blocker.write_text("x", encoding="utf-8")
    assert fr.start(log_dir=blocker / "sub") is False
