"""Guardrails against runaway orphan processes (2026-06-16).

Covers the three new layers that ensure NO child survives an Ultron exit:
  * kill_tree.kill_own_children -- the shutdown catch-all (reap every descendant).
  * sidecar_lock.reap_stray_embedders -- reap an embedder_server by cmdline even
    when it is NOT bound to the port (the gap that let a 20 GB orphan survive).
  * embedder_server._pid_alive -- the parent-death deadman's liveness check.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time

import psutil
import pytest

from kenning.subprocess.kill_tree import kill_own_children
from kenning.subprocess.sidecar_lock import reap_stray_embedders


def _spawn_sleeper(*extra_argv: str) -> subprocess.Popen:
    """A short-lived child python that just sleeps, with optional marker argv."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time,sys; time.sleep(40)", *extra_argv],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # let it register as a child
    for _ in range(50):
        if psutil.pid_exists(proc.pid):
            break
        time.sleep(0.02)
    return proc


def test_kill_own_children_reaps_descendant() -> None:
    proc = _spawn_sleeper()
    try:
        assert psutil.pid_exists(proc.pid)
        n = kill_own_children(grace_seconds=2.0)
        assert n >= 1
        # the child is gone; THIS process obviously survives
        for _ in range(50):
            if not psutil.pid_exists(proc.pid):
                break
            time.sleep(0.02)
        assert not psutil.pid_exists(proc.pid)
        assert psutil.pid_exists(os.getpid())
    finally:
        if proc.poll() is None:
            proc.kill()


def test_reap_stray_embedders_by_cmdline_unbound() -> None:
    # A UNIQUE marker so the test NEVER reaps the real sidecar (embedder_server).
    marker = "embedder_server_GUARDTEST_UNBOUND"
    proc = _spawn_sleeper(marker)
    try:
        n = reap_stray_embedders(script_hint=marker)
        assert n >= 1, "stray embedder-like process not reaped by cmdline"
        for _ in range(50):
            if not psutil.pid_exists(proc.pid):
                break
            time.sleep(0.02)
        assert not psutil.pid_exists(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_reap_stray_keeps_marked_pid() -> None:
    marker = "embedder_server_GUARDTEST_KEEP"
    proc = _spawn_sleeper(marker)
    try:
        n = reap_stray_embedders(keep_pid=proc.pid, script_hint=marker)
        assert n == 0
        assert psutil.pid_exists(proc.pid)  # keep_pid was spared
    finally:
        proc.kill()


def _load_embedder_module():
    """Import scripts/embedder_server.py with a safe argv (its module body reads
    sys.argv[1] as the port -> pytest's argv would crash it)."""
    saved = sys.argv
    sys.argv = ["embedder_server.py"]
    try:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "scripts", "embedder_server.py")
        spec = importlib.util.spec_from_file_location("_embsrv_guardtest", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # module body only; the model loads in main()
        return mod
    finally:
        sys.argv = saved


def test_embedder_pid_alive_deadman_check() -> None:
    mod = _load_embedder_module()
    assert mod._pid_alive(os.getpid()) is True       # this process is alive
    assert mod._pid_alive(2 ** 31 - 1) is False       # an impossible pid is dead
    assert mod._pid_alive(0) is True                  # unknown -> fail-SAFE (no self-kill)
    assert callable(mod._parent_watchdog)


def test_embedder_pid_alive_detects_dead_child() -> None:
    mod = _load_embedder_module()
    proc = _spawn_sleeper()
    assert mod._pid_alive(proc.pid) is True
    proc.kill()
    proc.wait(timeout=5)
    for _ in range(50):
        if not mod._pid_alive(proc.pid):
            break
        time.sleep(0.02)
    assert mod._pid_alive(proc.pid) is False
