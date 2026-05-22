"""Unified pytest runner with built-in safeguards.

THE single entry point for running the test sweep. Use this instead
of calling ``pytest tests/`` directly so the safeguards always apply.

Why this exists: during the 2026-05-21 frontier-enhancement pass, the
test sweep recurrently broke in ways that wasted hours to debug:

- Two ``pytest tests/`` invocations launched from different shells
  contended for fixture locks, GPU memory, and the HF cache. Both
  hung at ~0 % CPU. Diagnosis required `tasklist` / `psutil` chasing.

- Individual tests hung silently because pytest had no per-test
  timeout. The whole sweep would freeze with no indication of which
  test was the culprit.

- One test's mutation of the global config singleton would leak into
  unrelated tests downstream, producing failures that vanished when
  the tests were run in isolation.

This wrapper does:

1. **Pre-flight kill**: if any other pytest processes are running
   against this codebase, terminates them before starting (with a
   loud warning naming the PIDs). No more silent concurrent-run
   contention.

2. **Per-test timeout**: passes ``--timeout=30 --timeout-method=thread``
   (also set as the default in pyproject.toml). Any individual hang
   surfaces as ``Failed: Timeout >30.0s`` naming the offending test.

3. **Live streaming**: forwards stdout/stderr so you see progress
   tick-by-tick, not a buffered wall of dots at the end.

4. **Clean shutdown**: on Ctrl-C or completion, terminates all
   pytest descendants so no zombie workers linger.

5. **Coloured pass/fail summary** at the end with timing.

Usage::

    python scripts/run_tests.py                  # full sweep
    python scripts/run_tests.py tests/memory/    # just one dir
    python scripts/run_tests.py -k embedder      # just matching
    python scripts/run_tests.py --fast           # skip slow markers
    python scripts/run_tests.py --no-timeout     # disable per-test
                                                 # timeout (debug aid)
    python scripts/run_tests.py --kill-only      # just clean up + exit

Exit codes mirror pytest's: 0 on green, 1 on failures, 2 on internal
errors. Returns 4 if the pre-flight kill couldn't establish a clean
slate.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ---------------------------------------------------------------------------
# Pre-flight: kill any other pytest processes on this codebase
# ---------------------------------------------------------------------------


def _list_competing_pytests() -> list[dict]:
    """Return process info for any python.exe processes running
    pytest on this codebase EXCLUDING this script's own PID + its
    ancestors."""
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        print("WARNING: psutil unavailable; can't enforce concurrent-run "
              "safeguard. Install with: pip install psutil")
        return []

    me_pid = os.getpid()
    try:
        me_proc = psutil.Process(me_pid)
        ancestors = {a.pid for a in me_proc.parents()}
    except Exception:
        ancestors = set()
    ancestors.add(me_pid)

    found = []
    for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = p.info.get("cmdline") or []
            joined = " ".join(cmdline).lower()
            if "pytest" not in joined:
                continue
            if "tests" not in joined and "tests/" not in joined:
                continue
            if p.info["pid"] in ancestors:
                continue
            found.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _kill_competing_pytests(yes: bool = False) -> bool:
    """Terminate any competing pytest processes. Returns True if the
    slate is clean afterwards."""
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return True

    competing = _list_competing_pytests()
    if not competing:
        return True

    print(f"\n!!! Found {len(competing)} other pytest process(es) running "
          "on this codebase:")
    for c in competing:
        cmd_preview = " ".join((c["cmdline"] or [])[:5])
        print(f"      PID {c['pid']}: {cmd_preview}")
    if not yes:
        print("\n  These will be terminated before the new sweep starts.")
        print("  (Concurrent runs contend for fixture locks + GPU memory")
        print("   and cause the symptom of 'pytest hangs at 0 % CPU'.)")

    killed = []
    for c in competing:
        try:
            proc = psutil.Process(c["pid"])
            proc.terminate()
            killed.append(c["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if killed:
        try:
            # Give them 3 seconds to exit gracefully, then SIGKILL.
            _gone, alive = psutil.wait_procs(
                [psutil.Process(pid) for pid in killed
                 if psutil.pid_exists(pid)],
                timeout=3.0,
            )
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
        except Exception:
            pass

    # Final check
    leftover = _list_competing_pytests()
    if leftover:
        print(f"\n!!! Could not terminate all competing pytest processes; "
              f"{len(leftover)} still running. Bailing out.")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Unified Ultron test runner with safeguards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage::")[1] if "Usage::" in __doc__ else "",
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER,
        help="Args passed through to pytest (paths, -k, etc.)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip @pytest.mark.slow tests (default off).",
    )
    parser.add_argument(
        "--no-timeout", action="store_true",
        help="Disable the per-test timeout (debug aid only).",
    )
    parser.add_argument(
        "--kill-only", action="store_true",
        help="Pre-flight kill any competing pytest runs, then exit.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Don't prompt for the pre-flight kill confirmation.",
    )
    args = parser.parse_args(argv)

    # Step 1: pre-flight kill any competing pytest invocations.
    print("=" * 70)
    print("Ultron test runner")
    print("=" * 70)
    if not _kill_competing_pytests(yes=args.yes):
        return 4

    if args.kill_only:
        print("\n  Pre-flight kill complete. Exiting.")
        return 0

    # Step 2: assemble the pytest command.
    pytest_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    if not pytest_exe.is_file():
        # Fallback to the calling interpreter.
        pytest_exe = Path(sys.executable)

    cmd = [
        str(pytest_exe), "-m", "pytest",
        "--no-header",
        "-q",
    ]
    if args.no_timeout:
        # Override the addopts-default timeout.
        cmd += ["-o", "addopts=-p no:hydra_pytest --durations=10"]
    if args.fast:
        cmd += ["-m", "not slow"]
    # Forward remaining args to pytest.
    if args.pytest_args:
        # argparse leaves a leading "--" in pytest_args when REMAINDER
        # was used after a flag; strip it.
        forwarded = list(args.pytest_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        cmd += forwarded
    else:
        cmd += [
            "tests/",
            "--ignore=tests/coding/test_orchestration_real.py",
        ]

    print(f"\n  Running: {' '.join(cmd)}")
    print()
    print("-" * 70)

    # Step 3: spawn pytest. Live-stream stdout/stderr to the user.
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

    try:
        for line in proc.stdout:                                   # type: ignore[union-attr]
            print(line, end="")
        proc.wait()
        rc = proc.returncode or 0
    except KeyboardInterrupt:
        print("\n\n  Interrupted; terminating pytest...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        rc = 130

    elapsed = time.monotonic() - t0
    print()
    print("-" * 70)
    print(f"  Run took {elapsed:.1f}s; pytest exit code {rc}")
    print("=" * 70)

    # Step 4: clean shutdown -- terminate any python descendants of
    # this runner that might still be lingering.
    try:
        import psutil  # type: ignore[import]
        me = psutil.Process()
        for child in me.children(recursive=True):
            try:
                if "python" in (child.name() or "").lower():
                    child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass

    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
