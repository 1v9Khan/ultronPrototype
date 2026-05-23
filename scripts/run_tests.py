"""Unified pytest runner with multi-layer durability safeguards.

THE single entry point for running the test sweep. Use this instead
of ``pytest tests/`` directly so every safeguard fires.

Why this exists
===============

During the 2026-05-22 catalog pass the sweep recurrently broke in
ways that wasted hours to debug. The wrapper now defends against
*every concrete failure mode* observed across that session:

  1. **Concurrent pytest runs.** Two ``pytest tests/`` invocations
     contend for fixture-file locks, the MCP port, the HF cache,
     and CUDA memory. Both hung at ~0 % CPU.
  2. **Harness-orphan pytests.** The Claude harness sometimes
     backgrounds a tool call and falsely reports it as "completed"
     while the pytest child is still alive. The orphan then blocks
     the next sweep's conftest concurrent-run guard.
  3. **Per-test hangs.** A test stuck in a C-extension call (the
     thread-method timeout can't interrupt) would freeze the whole
     sweep with no indication of which test was the culprit.
  4. **Background-mode opacity.** PowerShell + ``Select-Object
     -Last N`` buffers the entire stream before emitting. The
     operator can't tell whether the sweep is alive or dead.
  5. **Test pollution.** A test mutating module-level / class-
     level state without ``monkeypatch`` leaks into downstream
     tests. Failures vanish on isolated reruns; sweep stays red.

Defensive layers
================

The wrapper applies five **independent** safeguards. Each catches a
specific class of failure even when the others miss:

  1. **Pre-flight env check.** psutil is importable, the venv
     python is reachable, ``data/`` is writable. Hard failure
     refuses to spawn pytest at all.
  2. **Cross-instance process mutex.** ``data/.run_tests.lock``
     containing the active sweep's PID. Stale-lock recovery from
     crashed instances. Plus orphan kill: pytests older than
     ``ORPHAN_AGE_SECONDS`` (5 min) are killed unconditionally.
  3. **Heartbeat watchdog.** Separate thread polling
     ``data/.run_tests_heartbeat`` every 5 s. Stale > ``STALE_HEARTBEAT_SECONDS``
     (default 90 s) ⇒ the test is hung in a way pytest-timeout
     can't interrupt (C-extension etc.); kill the pytest subprocess
     + log which test was running per ``data/.run_tests_current``.
  4. **Wall-clock deadline.** ``--max-runtime=N`` (default 600 s).
     Hard upper bound regardless of any other timeout. Kills
     pytest + summarises progress.
  5. **Post-run validation + aggressive cleanup.** Verify
     ``data/.run_tests_progress.jsonl`` ends with a
     ``session_end`` event (truthful "did the sweep actually
     finish?" signal regardless of pytest's exit code). Walk for
     ALL pytest processes against this codebase + kill survivors.

Observability surface
=====================

The conftest hooks publish three external files for operators:

  * ``data/.run_tests_heartbeat`` — mtime + content timestamp
    updated before every test.
  * ``data/.run_tests_current`` — name of the currently-running
    test (or ``"(session_ended status=N)"`` on completion).
  * ``data/.run_tests_progress.jsonl`` — one JSON line per
    test start / outcome with duration + a session_start +
    session_end event.

Operators can ``cat data/.run_tests_current`` to see what the sweep
is on right now, ``tail data/.run_tests_progress.jsonl`` for the
event stream, and ``stat data/.run_tests_heartbeat`` to spot
staleness — all without consuming stdout (so background-mode pipes
don't hide anything).

Usage
=====

::

    python scripts/run_tests.py                  # full sweep
    python scripts/run_tests.py tests/memory/    # one dir
    python scripts/run_tests.py -k embedder      # matching
    python scripts/run_tests.py --fast           # skip @slow
    python scripts/run_tests.py --no-timeout     # disable per-test timeout
    python scripts/run_tests.py --kill-only      # cleanup, then exit
    python scripts/run_tests.py --dry-run        # env check, then exit
    python scripts/run_tests.py --wait           # wait for competing sweep
    python scripts/run_tests.py --max-runtime=300  # 5-minute hard cap
    python scripts/run_tests.py --stale-heartbeat=60  # tighter staleness

Exit codes
==========

  *  0  green sweep, all tests passed
  *  1  red sweep, pytest reported failures
  *  2  internal error (couldn't spawn pytest)
  *  3  killed by heartbeat watchdog (stale heartbeat)
  *  4  pre-flight failed (couldn't acquire mutex / clean orphans)
  *  5  killed by wall-clock deadline
  *  6  pre-flight environment check failed
  *  7  killed-mid-stream detected (no session_end in JSONL)
  *  130 user Ctrl-C
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# Durability tunables -------------------------------------------------------

# Any pytest process older than this is treated as an ORPHAN — killed
# unconditionally with no operator prompt. 5 min is more than enough
# for any legitimate sweep (typical: 75-80 s, slowest observed ~3 min
# with CUDA warmup).
ORPHAN_AGE_SECONDS = 5 * 60

# Wall-clock hard deadline (overridable via --max-runtime). 10 min is
# generous: a sweep that takes >10 min has something genuinely wrong
# and should be killed. Set lower in CI.
DEFAULT_MAX_RUNTIME_SECONDS = 10 * 60

# Heartbeat staleness threshold. Per-test timeout is 30 s by default,
# so a test that hasn't ticked the heartbeat in 90 s is hung in a way
# pytest-timeout (thread-method) couldn't kill. Override via
# ``--stale-heartbeat=N``.
DEFAULT_STALE_HEARTBEAT_SECONDS = 90.0

# Watchdog poll interval. Short enough that staleness detection
# matters; long enough that the watchdog itself doesn't churn.
WATCHDOG_POLL_INTERVAL_SECONDS = 2.0

# Paths to the observability surface (must match the conftest hooks).
DATA_DIR = ROOT / "data"
SWEEP_LOCK_FILE = DATA_DIR / ".run_tests.lock"
HEARTBEAT_PATH = DATA_DIR / ".run_tests_heartbeat"
CURRENT_TEST_PATH = DATA_DIR / ".run_tests_current"
PROGRESS_LOG_PATH = DATA_DIR / ".run_tests_progress.jsonl"


# ---------------------------------------------------------------------------
# Pre-flight environment check
# ---------------------------------------------------------------------------


def _preflight_environment() -> tuple[bool, str]:
    """Verify the runner's own dependencies before spawning anything.

    Returns ``(ok, reason)``. ``ok=False`` means refuse to run — the
    safeguards can't function under the missing dep. We don't try to
    install anything; the reason string tells the operator what to do.
    """
    try:
        import psutil  # type: ignore[import]  # noqa: F401
    except ImportError:
        return (
            False,
            "psutil is not installed in this Python. Without it the "
            "concurrent-run safeguard + watchdog can't function.\n"
            "    Fix: pip install psutil",
        )
    # Verify writable data/.
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        probe = DATA_DIR / ".run_tests_preflight_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return (
            False,
            f"data/ directory not writable ({exc}). The heartbeat + "
            f"progress files live there.",
        )
    # Verify venv python (used to spawn pytest below).
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.is_file() and not Path(sys.executable).is_file():
        return (
            False,
            f"Neither .venv python ({venv_python}) nor sys.executable "
            f"({sys.executable}) is reachable.",
        )
    return True, ""


# ---------------------------------------------------------------------------
# Cross-instance mutex
# ---------------------------------------------------------------------------


def _acquire_sweep_lock() -> bool:
    """Acquire the cross-instance sweep lock or return False.

    Lock semantics: a file at :data:`SWEEP_LOCK_FILE` containing the
    PID of the active sweep.

      * File doesn't exist → write our PID, return True.
      * File exists, PID is a live python process with cmdline
        containing ``run_tests.py`` or ``pytest`` → another sweep is
        alive, return False.
      * File exists, PID is dead (crashed previous instance) →
        overwrite + return True.

    Cleanup registered via ``atexit``. SIGTERM bypasses atexit on
    Windows, so stale-lock recovery on the next invocation is the
    real safety net.
    """
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return False  # pre-flight should have already rejected this

    SWEEP_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if SWEEP_LOCK_FILE.exists():
        try:
            existing_pid_text = SWEEP_LOCK_FILE.read_text(encoding="utf-8").strip()
            existing_pid = int(existing_pid_text)
        except (OSError, ValueError):
            existing_pid = -1
        if existing_pid > 0 and psutil.pid_exists(existing_pid):
            try:
                proc = psutil.Process(existing_pid)
                cmd_joined = " ".join(proc.cmdline()).lower()
                if "run_tests.py" in cmd_joined or "pytest" in cmd_joined:
                    print(
                        f"!!! Another scripts/run_tests.py instance is "
                        f"already active (PID {existing_pid}).\n"
                        f"    Use --wait to wait, or kill manually."
                    )
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        print(
            f"  Recovering stale sweep lock at {SWEEP_LOCK_FILE} "
            f"(was PID {existing_pid}, no longer alive)."
        )
    try:
        SWEEP_LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        print(f"!!! Could not write sweep lock file: {exc}")
        return False
    return True


def _release_sweep_lock() -> None:
    """Drop the sweep lock if we own it. Safe to call multiple times."""
    try:
        if not SWEEP_LOCK_FILE.exists():
            return
        try:
            held_by = SWEEP_LOCK_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if held_by != str(os.getpid()):
            return
        SWEEP_LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Competing-pytest discovery + kill
# ---------------------------------------------------------------------------


def _list_competing_pytests(include_age: bool = False) -> list[dict]:
    """Return process info for every pytest-on-this-codebase
    NOT in this script's own ancestor chain."""
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return []

    me_pid = os.getpid()
    try:
        ancestors = {a.pid for a in psutil.Process(me_pid).parents()}
    except Exception:
        ancestors = set()
    ancestors.add(me_pid)

    now = time.time()
    found = []
    for p in psutil.process_iter(attrs=["pid", "name", "cmdline", "create_time"]):
        try:
            if "python" not in (p.info.get("name") or "").lower():
                continue
            cmdline = p.info.get("cmdline") or []
            joined = " ".join(cmdline).lower()
            if "pytest" not in joined:
                continue
            if "tests" not in joined and "tests/" not in joined:
                continue
            if p.info["pid"] in ancestors:
                continue
            info = dict(p.info)
            if include_age:
                try:
                    info["age_seconds"] = now - float(p.info.get("create_time") or now)
                except (TypeError, ValueError):
                    info["age_seconds"] = 0.0
            found.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _kill_competing_pytests(yes: bool = False) -> bool:
    """Terminate every competing pytest process. Returns True on clean slate.

    Two passes:

      1. Orphans (>``ORPHAN_AGE_SECONDS`` old) killed unconditionally,
         no operator prompt — these are harness-leak artefacts.
      2. Recent competitors killed too, with a printed notice.

    Each pass: terminate first, wait up to 3 s for graceful exit, then
    SIGKILL survivors. A second ``wait_procs`` for SIGKILL to take
    effect on Windows. Final ``_list_competing_pytests()`` check
    verifies the slate is clear.
    """
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return True

    competing = _list_competing_pytests(include_age=True)
    if not competing:
        return True

    orphans = [c for c in competing if c.get("age_seconds", 0.0) >= ORPHAN_AGE_SECONDS]
    recent = [c for c in competing if c.get("age_seconds", 0.0) < ORPHAN_AGE_SECONDS]

    if orphans:
        print(
            f"\n!!! Found {len(orphans)} ORPHAN pytest process(es) "
            f"(>{ORPHAN_AGE_SECONDS:.0f}s old; killing unconditionally):"
        )
        for c in orphans:
            cmd_preview = " ".join((c["cmdline"] or [])[:5])
            print(f"      PID {c['pid']} ({c.get('age_seconds', 0):.0f}s): {cmd_preview}")
    if recent:
        print(
            f"\n!!! Found {len(recent)} other (recent) pytest process(es) "
            "running on this codebase:"
        )
        for c in recent:
            cmd_preview = " ".join((c["cmdline"] or [])[:5])
            print(f"      PID {c['pid']} ({c.get('age_seconds', 0):.0f}s): {cmd_preview}")
        if not yes:
            print("\n  Terminating before the new sweep starts.")
            print("  (--wait to wait for them to finish instead.)")

    killed_pids: list[int] = []
    for c in (orphans + recent):
        try:
            proc = psutil.Process(c["pid"])
            proc.terminate()
            killed_pids.append(c["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed_pids:
        try:
            _gone, alive = psutil.wait_procs(
                [psutil.Process(pid) for pid in killed_pids if psutil.pid_exists(pid)],
                timeout=3.0,
            )
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(
                [psutil.Process(pid) for pid in killed_pids if psutil.pid_exists(pid)],
                timeout=2.0,
            )
        except Exception:
            pass

    leftover = _list_competing_pytests()
    if leftover:
        print(
            f"\n!!! Could not terminate all competing pytest processes; "
            f"{len(leftover)} still running. Bailing out.\n"
            "    Manual fix: Get-Process -Name python | "
            "Where-Object {$_.CommandLine -match 'pytest'} | "
            "Stop-Process -Force"
        )
        return False
    return True


def _wait_for_competing_pytests(poll_seconds: float = 2.0) -> bool:
    """Block until no competing pytest is running, then return True."""
    try:
        import psutil  # type: ignore[import]  # noqa: F401
    except ImportError:
        return True
    waited = 0.0
    while True:
        competing = _list_competing_pytests()
        if not competing:
            if waited > 0:
                print(f"  Waited {waited:.0f}s for competing sweep(s) to finish.")
            return True
        if waited == 0:
            print(
                f"\n!!! --wait mode: {len(competing)} competing pytest "
                "process(es) detected. Waiting for them to finish..."
            )
            for c in competing:
                cmd_preview = " ".join((c["cmdline"] or [])[:5])
                print(f"      PID {c['pid']}: {cmd_preview}")
        time.sleep(poll_seconds)
        waited += poll_seconds


# ---------------------------------------------------------------------------
# Watchdog (heartbeat staleness + wall-clock deadline)
# ---------------------------------------------------------------------------


class _Watchdog:
    """Background thread monitoring the sweep for two failure modes.

    Args:
        proc: The pytest :class:`subprocess.Popen` to kill on trigger.
        max_runtime_seconds: Hard wall-clock deadline.
        stale_heartbeat_seconds: Per-test heartbeat staleness limit.
        on_trigger: Callable invoked with ``(reason: str)`` BEFORE
            the kill so the operator sees what fired. Receives one
            of ``"wall_clock"`` / ``"heartbeat"``.

    The thread polls ``HEARTBEAT_PATH``'s mtime every
    :data:`WATCHDOG_POLL_INTERVAL_SECONDS`. When the heartbeat is
    stale OR the wall-clock deadline is exceeded:

      1. ``on_trigger`` is called with the reason.
      2. The pytest subprocess is terminated, then killed if it
         doesn't exit within 3 s.
      3. The thread sets ``triggered_reason`` and exits.

    The thread NEVER kills anything else. The post-run cleanup phase
    handles descendant python processes separately.
    """

    def __init__(
        self,
        proc: subprocess.Popen,
        *,
        max_runtime_seconds: float,
        stale_heartbeat_seconds: float,
        on_trigger,
    ) -> None:
        self._proc = proc
        self._max_runtime_seconds = float(max_runtime_seconds)
        self._stale_heartbeat_seconds = float(stale_heartbeat_seconds)
        self._on_trigger = on_trigger
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = time.monotonic()
        self.triggered_reason: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="run_tests-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(WATCHDOG_POLL_INTERVAL_SECONDS):
                return
            # Subprocess exit check FIRST -- a fast sweep can finish
            # before the heartbeat file ever gets written. Checking
            # staleness against a stale-from-the-previous-run heartbeat
            # would then mis-fire the watchdog. Without this guard the
            # watchdog kills an already-exited pytest and the wrapper
            # reports exit 3 for a green sweep.
            if self._proc.poll() is not None:
                return
            # Wall-clock deadline
            elapsed = time.monotonic() - self._started_at
            if elapsed > self._max_runtime_seconds:
                self._fire("wall_clock", elapsed)
                return
            # Heartbeat staleness
            stale = self._heartbeat_staleness()
            if stale is not None and stale > self._stale_heartbeat_seconds:
                self._fire("heartbeat", stale)
                return

    def _heartbeat_staleness(self) -> Optional[float]:
        """Seconds since the heartbeat file was last written. None on missing."""
        try:
            st = HEARTBEAT_PATH.stat()
        except OSError:
            return None
        return time.time() - st.st_mtime

    def _fire(self, reason: str, value: float) -> None:
        self.triggered_reason = reason
        try:
            self._on_trigger(reason, value)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  (watchdog on_trigger raised: {exc})")
        # Terminate the pytest subprocess.
        try:
            self._proc.terminate()
        except Exception:                                         # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                self._proc.kill()
            except Exception:                                     # noqa: BLE001
                pass
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass


# ---------------------------------------------------------------------------
# Session validation (truthful "did the sweep finish?" signal)
# ---------------------------------------------------------------------------


def _session_completed_cleanly() -> Optional[dict]:
    """Read the LAST event from the progress JSONL and return it if
    it's a ``session_end`` event, else None.

    A missing or absent session_end means the sweep was killed
    mid-stream — the heartbeat watchdog, the wall-clock deadline,
    or an external kill (harness, Ctrl-C, OOM) interrupted it
    before pytest's session-finish hook could fire.
    """
    if not PROGRESS_LOG_PATH.exists():
        return None
    try:
        # Read backwards to find the last non-empty line. The JSONL is
        # always small enough that reading the whole file is fine.
        text = PROGRESS_LOG_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    last_line = text.splitlines()[-1]
    try:
        evt = json.loads(last_line)
    except json.JSONDecodeError:
        return None
    if not isinstance(evt, dict):
        return None
    if evt.get("event") != "session_end":
        return None
    return evt


# ---------------------------------------------------------------------------
# Post-run cleanup
# ---------------------------------------------------------------------------


def _post_run_cleanup() -> None:
    """Aggressive sweep of any pytest leftovers + descendant pythons.

    Catches two things the per-instance lifecycle misses:

      * Python descendants of this runner that didn't exit (test
        subprocesses, lingering watchers).
      * Other pytest-on-this-codebase processes that started up
        DURING our sweep (e.g. an operator launching a second
        sweep believing the first was dead).

    Logs each kill at INFO. Never raises.
    """
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return
    me_pid = os.getpid()
    killed_any = False
    try:
        for child in psutil.Process(me_pid).children(recursive=True):
            try:
                if "python" not in (child.name() or "").lower():
                    continue
                print(f"  cleanup: terminating descendant python PID {child.pid}")
                child.terminate()
                killed_any = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    # Walk for any LATE-arriving pytest processes against this codebase.
    leftovers = _list_competing_pytests()
    for c in leftovers:
        try:
            print(f"  cleanup: terminating leftover pytest PID {c['pid']}")
            psutil.Process(c["pid"]).terminate()
            killed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed_any:
        time.sleep(1.0)
        for c in _list_competing_pytests():
            try:
                psutil.Process(c["pid"]).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Ultron test runner with multi-layer safeguards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else "",
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER,
        help="Args passed through to pytest (paths, -k, etc.)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip @pytest.mark.slow tests.",
    )
    parser.add_argument(
        "--no-timeout", action="store_true",
        help="Disable the per-test timeout (debug aid only).",
    )
    parser.add_argument(
        "--kill-only", action="store_true",
        help="Pre-flight cleanup then exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run pre-flight checks then exit without spawning pytest.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Don't prompt for the pre-flight kill confirmation.",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Wait for any competing pytest runs to finish instead of killing them.",
    )
    parser.add_argument(
        "--max-runtime", type=float, default=DEFAULT_MAX_RUNTIME_SECONDS,
        metavar="SECONDS",
        help=(
            f"Wall-clock hard deadline in seconds. Default "
            f"{DEFAULT_MAX_RUNTIME_SECONDS:.0f}s. Sweep is killed when "
            "exceeded; exit code 5."
        ),
    )
    parser.add_argument(
        "--stale-heartbeat", type=float,
        default=DEFAULT_STALE_HEARTBEAT_SECONDS,
        metavar="SECONDS",
        help=(
            f"Heartbeat staleness limit in seconds. Default "
            f"{DEFAULT_STALE_HEARTBEAT_SECONDS:.0f}s. Sweep is killed when "
            "the heartbeat file isn't updated for this long; exit code 3."
        ),
    )
    return parser


def main(argv: list[str]) -> int:
    args = _build_arg_parser().parse_args(argv)

    print("=" * 70)
    print(f"Ultron test runner  (PID {os.getpid()})")
    print("=" * 70)

    # Layer 1: pre-flight environment check
    ok, reason = _preflight_environment()
    if not ok:
        print(f"\n!!! Pre-flight environment check failed:\n    {reason}")
        return 6

    # Layer 2: cross-instance mutex + orphan kill
    if not _acquire_sweep_lock():
        if args.wait:
            print("  --wait mode: blocking until the other sweep finishes...")
            if not _wait_for_competing_pytests():
                return 4
            if not _acquire_sweep_lock():
                print("!!! Could not acquire sweep lock after wait.")
                return 4
        else:
            return 4
    atexit.register(_release_sweep_lock)
    atexit.register(_post_run_cleanup)

    if args.wait:
        if not _wait_for_competing_pytests():
            return 4
    else:
        if not _kill_competing_pytests(yes=args.yes):
            return 4

    if args.kill_only:
        print("\n  Pre-flight kill complete. Exiting.")
        return 0
    if args.dry_run:
        print("\n  Dry-run: pre-flight checks passed. Exiting without running pytest.")
        return 0

    # Assemble + spawn pytest
    pytest_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    if not pytest_exe.is_file():
        pytest_exe = Path(sys.executable)

    cmd = [str(pytest_exe), "-m", "pytest", "--no-header", "-q"]
    if args.no_timeout:
        cmd += ["-o", "addopts=-p no:hydra_pytest --durations=10"]
    if args.fast:
        cmd += ["-m", "not slow"]
    if args.pytest_args:
        forwarded = list(args.pytest_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        cmd += forwarded
    else:
        cmd += ["tests/", "--ignore=tests/coding/test_orchestration_real.py"]

    print(f"\n  Running: {' '.join(cmd)}")
    print(f"  Max runtime: {args.max_runtime:.0f}s; "
          f"stale-heartbeat threshold: {args.stale_heartbeat:.0f}s")
    print()
    print("-" * 70)

    # Clear stale observability files from prior runs so the watchdog
    # never sees a heartbeat mtime older than this sweep's pytest_sessionstart.
    # Without this, a previous run's heartbeat lingering on disk causes the
    # watchdog to mis-fire on fast sweeps that finish before the conftest's
    # pytest_sessionstart hook has run.
    for stale_path in (HEARTBEAT_PATH, CURRENT_TEST_PATH, PROGRESS_LOG_PATH):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"  (could not clear {stale_path.name}: {e}; continuing)")

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )
    except FileNotFoundError as e:
        print(f"!!! Could not spawn pytest: {e}")
        return 2

    # Layers 3 + 4: watchdog (heartbeat staleness + wall-clock deadline)
    watchdog_triggers: list[str] = []

    def _on_watchdog_trigger(reason: str, value: float) -> None:
        if reason == "wall_clock":
            print(
                f"\n!!! WATCHDOG: wall-clock deadline exceeded "
                f"({value:.0f}s > {args.max_runtime:.0f}s). Killing pytest.",
            )
        elif reason == "heartbeat":
            # Read current_test path to surface WHICH test was hung.
            current = "<unknown>"
            try:
                current = CURRENT_TEST_PATH.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            print(
                f"\n!!! WATCHDOG: heartbeat stale ({value:.0f}s > "
                f"{args.stale_heartbeat:.0f}s).\n"
                f"    Last-running test: {current}\n"
                f"    This is a hang the per-test thread-timeout couldn't "
                f"break (typically a C-extension call holding the GIL).\n"
                f"    Killing pytest.",
            )
        watchdog_triggers.append(reason)

    watchdog = _Watchdog(
        proc,
        max_runtime_seconds=args.max_runtime,
        stale_heartbeat_seconds=args.stale_heartbeat,
        on_trigger=_on_watchdog_trigger,
    )
    watchdog.start()

    try:
        # Live-stream stdout. The watchdog terminates proc out-of-band
        # if needed; the stdout loop exits naturally when the pipe closes.
        for line in proc.stdout:                                   # type: ignore[union-attr]
            print(line, end="")
        proc.wait()
        rc = proc.returncode or 0
    except KeyboardInterrupt:
        print("\n\n  Interrupted; terminating pytest...")
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        rc = 130
    finally:
        watchdog.stop()

    elapsed = time.monotonic() - t0
    print()
    print("-" * 70)
    print(f"  Run took {elapsed:.1f}s; pytest exit code {rc}")

    # Layer 5: post-run validation. Did the sweep actually finish?
    end_event = _session_completed_cleanly()
    if watchdog.triggered_reason == "wall_clock":
        print("=" * 70)
        return 5
    if watchdog.triggered_reason == "heartbeat":
        print("=" * 70)
        return 3
    if end_event is None and rc == 0:
        # Pytest reported success but no session_end event was written.
        # Most likely the harness or an external kill stopped pytest
        # AFTER it had cleaned up, OR before pytest_sessionfinish ran.
        # Honesty bit: we report this distinctly so the operator
        # doesn't trust a false "exit 0".
        print(
            "\n!!! WARNING: pytest exited 0 but no session_end event "
            "was written to data/.run_tests_progress.jsonl. The sweep "
            "may have been killed before finishing cleanly. Treat the "
            "exit code as suspect."
        )
        print("=" * 70)
        return 7
    print("=" * 70)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
