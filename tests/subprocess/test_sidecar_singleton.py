"""Anti-stale-sidecar permanent guards (folded into the rigorous cleanup process).

Covers the fix for the 2026-06-21 incident (three guard sidecars co-bound to
:8774): the EXCLUSIVE single-instance bind (two processes can't co-bind a port),
the generalized cmdline reaper across all sidecar roles, the port reclaim, the
singleton guard, and the per-role pidfiles.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler

import psutil
import pytest

from kenning.subprocess import sidecar_lock
from kenning.subprocess.sidecar_server import SingletonThreadingHTTPServer


class _H(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):  # noqa: A002
        return


def _spawn_sleeper(*extra_argv: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(40)", *extra_argv],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        if psutil.pid_exists(proc.pid):
            break
        time.sleep(0.02)
    return proc


def test_exclusive_bind_refuses_a_second_instance_on_a_live_port() -> None:
    """The decisive fix: a second bind on a LIVE port must FAIL (not co-serve).
    The old ThreadingHTTPServer default (allow_reuse_address=True) silently allowed
    co-binding on Windows -> three stale sidecars on :8774."""
    s1 = SingletonThreadingHTTPServer(("127.0.0.1", 0), _H)
    try:
        port = s1.server_address[1]
        with pytest.raises(OSError):
            SingletonThreadingHTTPServer(("127.0.0.1", port), _H)
    finally:
        s1.server_close()


def test_reap_stray_sidecars_by_cmdline() -> None:
    marker = "twitch_guard_sidecar_GUARDTEST_STRAY"
    proc = _spawn_sleeper(marker)
    try:
        n = sidecar_lock.reap_stray_sidecars([marker])
        assert n >= 1, "stray sidecar not reaped by cmdline"
        for _ in range(50):
            if not psutil.pid_exists(proc.pid):
                break
            time.sleep(0.02)
        assert not psutil.pid_exists(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_reap_stray_sidecars_spares_keep_pid() -> None:
    marker = "twitch_read_sidecar_GUARDTEST_KEEP"
    proc = _spawn_sleeper(marker)
    try:
        n = sidecar_lock.reap_stray_sidecars([marker], keep_pid=proc.pid)
        assert n == 0
        assert psutil.pid_exists(proc.pid)
    finally:
        proc.kill()


def test_reap_stray_embedders_back_compat_delegates() -> None:
    marker = "embedder_server_GUARDTEST_DELEGATE"
    proc = _spawn_sleeper(marker)
    try:
        n = sidecar_lock.reap_stray_embedders(script_hint=marker)
        assert n >= 1
    finally:
        if proc.poll() is None:
            proc.kill()


def test_reclaim_port_never_kills_self_and_noops_when_clear() -> None:
    # nothing listening -> 0, fail-open
    assert sidecar_lock.reclaim_port("127.0.0.1", 59599) == 0
    # a listener owned by THIS process must NOT be reaped (would kill the test run)
    s = SingletonThreadingHTTPServer(("127.0.0.1", 0), _H)
    try:
        port = s.server_address[1]
        assert sidecar_lock.reclaim_port("127.0.0.1", port) == 0
        assert psutil.pid_exists(os.getpid())
    finally:
        s.server_close()


def test_guard_singleton_is_fail_open() -> None:
    # no strays, no listener -> returns 0, never raises
    assert sidecar_lock.guard_singleton("127.0.0.1", 59598, "twitch_guard") == 0


def test_role_pidfile_write_read_clear() -> None:
    role = "GUARDTEST_role"
    try:
        sidecar_lock.write_role(role, 12345, 8774)
        data = json.loads(sidecar_lock.role_pidfile(role).read_text(encoding="utf-8"))
        assert data["pid"] == 12345 and data["port"] == 8774 and data["role"] == role
    finally:
        sidecar_lock.clear_role(role)
        assert not sidecar_lock.role_pidfile(role).exists()


def test_all_known_sidecar_roles_registered() -> None:
    for role in ("embedder", "twitch_guard", "twitch_read", "twitch_overlay"):
        assert role in sidecar_lock.SIDECAR_HINTS
