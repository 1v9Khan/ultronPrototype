"""Tests for cross-platform process-tree termination.

T8 (OpenClaw catalog port). Most coverage uses fake processes that
implement the psutil contract; one end-to-end test spawns a trivial
sleeping Python subprocess and verifies kill_pid_if_alive removes it.
All subprocesses are reaped via fixture teardown (R3 binding rule).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

import pytest

from ultron.subprocess import kill_tree


# ----------------------------------------------------------------------
# clamp_grace + KillTreeResult dataclass invariants


def test_clamp_grace_below_zero_returns_zero() -> None:
    assert kill_tree._clamp_grace(-5.0) == 0.0


def test_clamp_grace_above_max_clamps_to_max() -> None:
    assert kill_tree._clamp_grace(kill_tree.MAX_GRACE_SECONDS + 100) == kill_tree.MAX_GRACE_SECONDS


def test_clamp_grace_within_band_passes_through() -> None:
    assert kill_tree._clamp_grace(2.5) == 2.5


def test_kill_tree_result_total_killed_aggregates() -> None:
    result = kill_tree.KillTreeResult(
        root_pid=123,
        terminated=(1, 2),
        force_killed=(3,),
    )
    assert result.total_killed == 3


def test_kill_tree_result_total_killed_zero_when_only_unreachable() -> None:
    result = kill_tree.KillTreeResult(root_pid=123, unreachable=(99,))
    assert result.total_killed == 0


# ----------------------------------------------------------------------
# Argument validation


def test_kill_process_tree_negative_pid_returns_empty() -> None:
    result = kill_tree.kill_process_tree(-1)
    assert result.terminated == ()
    assert result.force_killed == ()


def test_kill_process_tree_zero_pid_returns_empty() -> None:
    result = kill_tree.kill_process_tree(0)
    assert result.terminated == ()


def test_kill_process_tree_none_pid_returns_empty() -> None:
    result = kill_tree.kill_process_tree(None)
    assert result.terminated == ()


def test_kill_pid_if_alive_negative_pid_returns_empty() -> None:
    result = kill_tree.kill_pid_if_alive(-1)
    assert result.terminated == ()


def test_kill_pid_if_alive_zero_pid_returns_empty() -> None:
    result = kill_tree.kill_pid_if_alive(0)
    assert result.terminated == ()


# ----------------------------------------------------------------------
# Fake psutil-style process for graceful + force flows


class _FakeProc:
    """Stand-in for ``psutil.Process`` used in unit tests."""

    def __init__(
        self,
        pid: int,
        *,
        graceful_succeeds: bool = True,
        force_succeeds: bool = True,
        children: Optional[list["_FakeProc"]] = None,
        running: bool = True,
    ) -> None:
        self.pid = pid
        self._graceful_called = False
        self._force_called = False
        self._graceful_succeeds = graceful_succeeds
        self._force_succeeds = force_succeeds
        self._children = children or []
        self._running = running

    def terminate(self) -> None:
        self._graceful_called = True
        if not self._graceful_succeeds:
            raise RuntimeError("terminate failed")

    def kill(self) -> None:
        self._force_called = True
        if not self._force_succeeds:
            raise RuntimeError("kill failed")
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def children(self, *, recursive: bool = False) -> list["_FakeProc"]:
        if recursive:
            out: list[_FakeProc] = []
            stack = list(self._children)
            while stack:
                child = stack.pop()
                out.append(child)
                stack.extend(child._children)
            return out
        return list(self._children)


def _install_fake_psutil(monkeypatch: pytest.MonkeyPatch, root_factory) -> None:
    """Wire up a fake psutil module that returns ``root_factory(pid)``."""
    class _FakePsutil:
        NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        AccessDenied = type("AccessDenied", (Exception,), {})
        STATUS_ZOMBIE = "zombie"

        @staticmethod
        def Process(pid: int):  # noqa: N802 — mimics psutil API
            proc = root_factory(pid)
            if proc is None:
                raise _FakePsutil.NoSuchProcess(pid)
            return proc

        @staticmethod
        def wait_procs(procs, timeout):
            gone = [p for p in procs if not p.is_running()]
            alive = [p for p in procs if p.is_running()]
            return gone, alive

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)


def test_kill_process_tree_root_only_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _FakeProc(123, graceful_succeeds=True)

    def factory(pid: int) -> Optional[_FakeProc]:
        if pid == 123:
            # After terminate, mark as not-running so wait_procs reports gone.
            root._running = False
            return root
        return None

    _install_fake_psutil(monkeypatch, lambda pid: root if pid == 123 else None)
    # Pre-mark as not-running so the wait_procs branch reports gone immediately.
    root._running = False
    result = kill_tree.kill_process_tree(123, grace_seconds=0.1)
    assert 123 in result.terminated
    assert result.force_killed == ()


def test_kill_process_tree_root_with_descendants(monkeypatch: pytest.MonkeyPatch) -> None:
    child_a = _FakeProc(101, running=False)
    child_b = _FakeProc(102, running=False)
    root = _FakeProc(100, running=False, children=[child_a, child_b])

    _install_fake_psutil(monkeypatch, lambda pid: root if pid == 100 else None)
    result = kill_tree.kill_process_tree(100, grace_seconds=0.1)
    assert set(result.terminated) >= {100, 101, 102}


def test_kill_process_tree_unreachable_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_psutil(monkeypatch, lambda pid: None)
    result = kill_tree.kill_process_tree(999)
    assert result.unreachable == (999,)
    assert result.terminated == ()


def test_kill_process_tree_force_kills_when_graceful_unsuccessful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Process stays running after terminate; kill_process_tree should
    # force-kill it in the second phase.
    proc = _FakeProc(200, graceful_succeeds=True, running=True)

    def factory(pid: int) -> Optional[_FakeProc]:
        return proc if pid == 200 else None

    _install_fake_psutil(monkeypatch, factory)
    result = kill_tree.kill_process_tree(200, grace_seconds=0.05)
    assert 200 in result.force_killed
    assert proc._force_called is True


def test_kill_process_tree_swallows_graceful_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(300, graceful_succeeds=False, running=False)
    _install_fake_psutil(monkeypatch, lambda pid: proc if pid == 300 else None)
    # No exception should escape even though terminate() raises.
    result = kill_tree.kill_process_tree(300, grace_seconds=0.05)
    assert 300 in result.unreachable


def test_kill_pid_if_alive_skips_dead_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    # If the pid isn't alive at call time, return immediately as unreachable.
    monkeypatch.setattr(kill_tree, "_is_alive", lambda pid: False)
    result = kill_tree.kill_pid_if_alive(404)
    assert result.unreachable == (404,)
    assert result.terminated == ()


def test_kill_pid_if_alive_runs_graceful_then_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(500, graceful_succeeds=True, running=True)
    monkeypatch.setattr(kill_tree, "_is_alive", lambda pid: True)
    _install_fake_psutil(monkeypatch, lambda pid: proc if pid == 500 else None)
    result = kill_tree.kill_pid_if_alive(500, grace_seconds=0.05)
    # Either terminated or force_killed; both are acceptable outcomes
    # (depends on which wait_procs phase the fake reports gone in).
    assert 500 in result.terminated or 500 in result.force_killed


# ----------------------------------------------------------------------
# Single end-to-end with a real subprocess (R3 reap discipline)


@pytest.fixture
def long_running_python_process():
    """Spawn a Python subprocess that sleeps; reap it on test exit."""
    flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        flags |= subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    yield proc
    # R3 binding rule: always reap, even if the test failed.
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
    except Exception:
        pass


def test_kill_pid_if_alive_real_subprocess(long_running_python_process) -> None:
    proc = long_running_python_process
    # Sanity: process is alive.
    assert proc.poll() is None
    result = kill_tree.kill_pid_if_alive(proc.pid, grace_seconds=1.0)
    assert result.total_killed >= 1
    # Give the OS table a tick to settle.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.05)
    assert proc.poll() is not None, "subprocess should be gone after kill"


def test_is_alive_returns_false_for_negative_pid() -> None:
    assert kill_tree._is_alive(-1) is False


def test_is_alive_returns_false_for_zero_pid() -> None:
    assert kill_tree._is_alive(0) is False


def test_is_alive_returns_false_for_none_pid() -> None:
    assert kill_tree._is_alive(None) is False


def test_collect_tree_returns_empty_on_missing_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_psutil(monkeypatch, lambda pid: None)
    procs, unreachable = kill_tree._collect_tree(99999999)
    assert procs == []
    assert unreachable == [99999999]


def test_is_posix_matches_os_name() -> None:
    assert kill_tree._is_posix() is (os.name == "posix")
