"""Shared pytest fixtures and import shims.

Two safeguards live here:

1. **Pre-flight concurrent-run detection** (2026-05-21): if another
   pytest invocation against this codebase is already running, the
   incoming run fails fast with a clear message naming the existing
   PID. Without this, two concurrent ``pytest tests/`` calls (a real
   trap when bash background tasks aren't waited on) silently contend
   for fixture file locks, GPU memory, HF cache, and produce hung
   workers at ~0 % CPU. The contention manifests as "the sweep takes
   3-5x longer than usual AND some tests fail" -- exactly the failure
   mode that motivated the existing session-end cleanup hook below.

2. **Session-end subprocess cleanup**: terminates any leftover python
   subprocesses launched during the test run. Without this, a hung
   test or background-task pytest invocation that is never explicitly
   waited on can leave Python workers consuming hundreds of MB of RAM
   (and on the dev machine, hundreds of MB of VRAM too if the worker
   loaded torch/CUDA before failing). Best-effort and fail-open.

We never kill processes outside our own descendant tree, never kill
non-python processes, and never kill a process that has an open TCP
listener on the Ultron MCP port (19761) -- that's the live orchestrator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Suppress observation-writer IO during test runs.
#
# Otherwise every test that touches a wired call site (classify_routing,
# AddressingClassifier, ConversationMemory.retrieve, LLMEngine.generate*)
# would accumulate rows in data/observations.jsonl, polluting analytics
# runs and adding spurious IO to the test sweep.
#
# Tests that specifically want to observe an emit can override the
# singleton via :func:`ultron.observations.set_observation_writer`.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _disable_observation_io_for_tests():
    try:
        from ultron.observations import (
            ObservationWriter,
            set_observation_writer,
        )
    except Exception:
        # Module not importable for some reason -- don't block the run.
        yield
        return
    disabled = ObservationWriter(Path("data") / "observations.jsonl", enabled=False)
    set_observation_writer(disabled)
    try:
        yield
    finally:
        set_observation_writer(None)


# ---------------------------------------------------------------------------
# Session-end subprocess cleanup
# ---------------------------------------------------------------------------


_ULTRON_MCP_PORT = 19761


# ---------------------------------------------------------------------------
# Pre-flight: refuse to run if another pytest is in flight on this codebase.
# Prevents the "two concurrent sweeps both hang at 0 % CPU" failure mode
# that's a real trap when bash background tasks aren't waited on.
# ---------------------------------------------------------------------------


def pytest_configure(config):  # noqa: ARG001
    """Pytest hook: refuse to start if another pytest run on this
    codebase is already in flight."""
    try:
        import os
        import psutil  # type: ignore[import]
    except Exception:
        return  # psutil unavailable -> can't enforce, silent fail-open

    me_pid = os.getpid()
    try:
        me_proc = psutil.Process(me_pid)
        my_ancestors = {a.pid for a in me_proc.parents()}
    except Exception:
        my_ancestors = set()
    my_ancestors.add(me_pid)

    # Codebase signature: any process whose cmdline mentions pytest +
    # a 'tests' or 'tests/' arg is a candidate.
    candidates = []
    for p in psutil.process_iter(attrs=["pid", "name", "cmdline", "create_time"]):
        try:
            if not p.info["name"] or "python" not in p.info["name"].lower():
                continue
            cmdline = p.info["cmdline"] or []
            joined = " ".join(cmdline).lower()
            if "pytest" not in joined:
                continue
            if "tests" not in joined and "tests/" not in joined:
                continue
            if p.info["pid"] in my_ancestors:
                continue
            # Worker children of our own pytest run are fine.
            try:
                parents = {pa.pid for pa in p.parents()}
            except Exception:
                parents = set()
            if me_pid in parents:
                continue
            candidates.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not candidates:
        return

    lines = ["\n  Another pytest run on this codebase is already in flight:"]
    for c in candidates:
        lines.append(
            f"    PID {c['pid']}: {' '.join((c['cmdline'] or [])[:6])}"
        )
    lines.append(
        "  Kill it first (Stop-Process -Id <PID> -Force) and retry. "
        "Concurrent runs contend for fixture file locks, GPU memory, "
        "and the HF cache, which causes hangs at ~0 % CPU."
    )
    raise pytest.UsageError("\n".join(lines))


def _kill_test_descendants() -> None:
    """Terminate any python subprocesses descended from this pytest run.

    Skips the live Ultron orchestrator (the process listening on the
    MCP port) and its ancestors / descendants. Best-effort: anything
    we can't enumerate, terminate, or kill is silently ignored.
    """
    try:
        import psutil  # type: ignore[import]
    except Exception:
        return
    try:
        me = psutil.Process()
    except Exception:
        return

    # Build the protected set: any process tied to the running Ultron.
    preserved: set[int] = set()
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if (
                conn.laddr
                and conn.laddr.port == _ULTRON_MCP_PORT
                and conn.status == "LISTEN"
                and conn.pid
            ):
                preserved.add(conn.pid)
                try:
                    holder = psutil.Process(conn.pid)
                    for anc in holder.parents():
                        preserved.add(anc.pid)
                    for desc in holder.children(recursive=True):
                        preserved.add(desc.pid)
                except psutil.NoSuchProcess:
                    pass
    except (psutil.AccessDenied, PermissionError):
        # If we can't see TCP state, fail safe: leave everything alone
        # rather than risk killing the live orchestrator.
        return

    # Walk our descendants.
    try:
        descendants = me.children(recursive=True)
    except psutil.NoSuchProcess:
        return

    to_kill: list[psutil.Process] = []
    for child in descendants:
        if child.pid in preserved or child.pid == me.pid:
            continue
        try:
            name = (child.name() or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name not in {"python.exe", "python", "pythonw.exe", "pythonw"}:
            continue
        to_kill.append(child)

    if not to_kill:
        return

    for c in to_kill:
        try:
            c.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    try:
        _gone, alive = psutil.wait_procs(to_kill, timeout=3.0)
    except Exception:
        alive = []
    for c in alive:
        try:
            c.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def pytest_sessionfinish(session, exitstatus):  # noqa: D401, ARG001
    """Pytest hook: best-effort reap of python descendants at session end."""
    _kill_test_descendants()
