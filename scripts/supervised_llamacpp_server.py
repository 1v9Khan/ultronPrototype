"""Supervisor for llama-cpp-server.

Spawns ``scripts/start_llamacpp_server.py`` as a subprocess, watches
it, and restarts on death with exponential backoff up to a cap. Logs
stdout + stderr through the supervisor's own stream-tee so failures
have a permanent record. Catches Ctrl+C cleanly.

This is a lighter alternative to NSSM (which would convert the server
into a Windows Service). For day-to-day desk use, run this in a
PowerShell window and forget about it. For unattended deployment,
NSSM is still appropriate — see docs/openclaw_integration.md.

The supervisor itself is single-file, dependency-free Python (uses
only the stdlib + ``ultron`` for the DLL setup the launcher needs).

Run from main checkout (where models/ lives):

    cd C:\\STC\\ultronPrototype
    .venv\\Scripts\\python.exe scripts/supervised_llamacpp_server.py

Or from anywhere with explicit cwd via --cwd. Pass --max-restarts 0
to disable auto-restart (just runs once + reports exit). Pass
--child-arg "--n-ctx" --child-arg "32768" to forward CLI flags to
the underlying launcher.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


_LAUNCHER = Path(__file__).resolve().parent / "start_llamacpp_server.py"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    sys.stderr.write(f"[supervisor {_now_iso()}] {msg}\n")
    sys.stderr.flush()


class Supervisor:
    """One-instance supervisor with bounded restart backoff.

    Backoff schedule: ``initial`` seconds, doubling each consecutive
    failure, capped at ``max_backoff``. A successful run lasting at
    least ``healthy_after`` seconds resets the backoff.
    """

    def __init__(
        self,
        argv: List[str],
        cwd: Path,
        *,
        max_restarts: int = 1000,
        initial_backoff_s: float = 2.0,
        max_backoff_s: float = 60.0,
        healthy_after_s: float = 30.0,
    ) -> None:
        self.argv = argv
        self.cwd = cwd
        self.max_restarts = max_restarts
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self.healthy_after_s = healthy_after_s
        self._proc: Optional[subprocess.Popen] = None
        self._stop = False

    def stop(self) -> None:
        """Signal the supervisor to wind down on the next iteration.

        Also tries to terminate the active child gracefully (SIGTERM,
        then SIGKILL on Windows via taskkill /F /T).
        """
        self._stop = True
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        _log(f"stopping child pid={proc.pid}")
        try:
            if os.name == "nt":
                # On Windows, SIGTERM doesn't trickle through subprocess
                # trees reliably. taskkill /F /T kills the whole tree.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, check=False,
                )
            else:
                proc.send_signal(signal.SIGTERM)
        except Exception as e:
            _log(f"stop signal failed: {e}")

    def run(self) -> int:
        """Run forever (or until ``max_restarts`` is exceeded). Returns
        the last child exit code."""
        backoff = self.initial_backoff_s
        restart_count = 0
        last_exit: int = 0
        while not self._stop:
            _log(f"launching: cwd={self.cwd} argv={shlex.join(self.argv)}")
            started_at = time.monotonic()
            try:
                self._proc = subprocess.Popen(
                    self.argv,
                    cwd=str(self.cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=sys.stdout,  # tee through; user sees the launcher's logs
                    stderr=sys.stderr,
                )
            except FileNotFoundError as e:
                _log(f"launch failed: {e}")
                return 2

            try:
                last_exit = self._proc.wait()
            except KeyboardInterrupt:
                _log("KeyboardInterrupt — stopping")
                self.stop()
                last_exit = self._proc.wait() if self._proc else 130
                break

            ran_for = time.monotonic() - started_at
            _log(
                f"child exited code={last_exit} after {ran_for:.1f}s "
                f"(restart {restart_count + 1}/{self.max_restarts})"
            )

            if self._stop:
                break

            if restart_count >= self.max_restarts:
                _log("max_restarts reached; giving up")
                break

            if ran_for >= self.healthy_after_s:
                # Run was long enough to consider the model loaded
                # successfully — reset backoff so transient crashes
                # don't compound.
                backoff = self.initial_backoff_s

            _log(f"backing off {backoff:.1f}s before restart")
            try:
                time.sleep(backoff)
            except KeyboardInterrupt:
                _log("KeyboardInterrupt during backoff — stopping")
                break
            backoff = min(backoff * 2, self.max_backoff_s)
            restart_count += 1

        _log(f"supervisor exit code={last_exit}")
        return last_exit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cwd", type=str, default=r"C:\STC\ultronPrototype",
        help="Working directory for the launcher (default: main checkout).",
    )
    parser.add_argument(
        "--max-restarts", type=int, default=1000,
        help="Maximum total restarts before giving up. 0 disables restart.",
    )
    parser.add_argument(
        "--initial-backoff-s", type=float, default=2.0,
    )
    parser.add_argument(
        "--max-backoff-s", type=float, default=60.0,
    )
    parser.add_argument(
        "--healthy-after-s", type=float, default=30.0,
        help="A run lasting this long resets the backoff (defaults to 30 s).",
    )
    parser.add_argument(
        "--child-arg", action="append", default=[],
        help=(
            "Argument to forward to start_llamacpp_server.py. "
            "Repeat for multi-token args. Example: "
            "--child-arg --n-ctx --child-arg 16384"
        ),
    )
    args = parser.parse_args()

    cwd = Path(args.cwd)
    if not cwd.is_dir():
        sys.stderr.write(f"error: --cwd not a directory: {cwd}\n")
        return 2
    if not _LAUNCHER.is_file():
        sys.stderr.write(f"error: launcher missing: {_LAUNCHER}\n")
        return 2

    argv = [sys.executable, str(_LAUNCHER), *args.child_arg]
    sup = Supervisor(
        argv=argv,
        cwd=cwd,
        max_restarts=args.max_restarts,
        initial_backoff_s=args.initial_backoff_s,
        max_backoff_s=args.max_backoff_s,
        healthy_after_s=args.healthy_after_s,
    )

    def _on_sigterm(signum, frame):  # pragma: no cover - signal handler
        sup.stop()

    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _on_sigterm)
        except (OSError, ValueError):
            pass  # Some Windows console contexts disallow this.
    if hasattr(signal, "SIGINT"):
        try:
            signal.signal(signal.SIGINT, _on_sigterm)
        except (OSError, ValueError):
            pass

    return sup.run()


if __name__ == "__main__":
    raise SystemExit(main())
