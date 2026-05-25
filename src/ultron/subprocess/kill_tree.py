"""Cross-platform process-tree termination (graceful then SIGKILL).

Single entry point :func:`kill_process_tree` that:

* Walks the process tree rooted at ``pid`` (parent + all
  descendants).
* On Windows, sends graceful WM_CLOSE via :meth:`psutil.Process.terminate`
  to each member.
* On POSIX, sends SIGTERM to the process group (when
  ``detached=True``) or to each member individually
  (``detached=False`` — required when ``pid`` is a child of the
  current process that was NOT spawned with ``start_new_session``,
  to avoid killing the current process' own group).
* Waits up to ``grace_seconds`` for each member to exit.
* Force-terminates (Win32 ``TerminateProcess`` / POSIX SIGKILL) any
  members still alive after the grace window.

Pattern informed by OpenClaw's ``src/process/kill-tree.ts`` (MIT;
see ``THIRD_PARTY_NOTICES.md``); algorithm adapted to ``psutil`` so
the same primitive works on Windows + Linux + macOS without
shelling out to ``taskkill`` / ``kill``.

This is the canonical primitive for orchestrator shutdown, the
Parakeet HTTP server lifecycle, future MCP transport shutdown, and
test-harness cleanup. The existing :mod:`ultron.subprocess.zombie_killer`
uses an equivalent terminate-then-kill pattern for single processes;
:func:`kill_process_tree` extends that discipline to recursive
descendants.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

LOGGER = logging.getLogger(__name__)

#: Default grace period (seconds) between graceful terminate and
#: force-kill. Three seconds matches OpenClaw's default + leaves
#: enough room for a process with a trivial shutdown hook to exit
#: cleanly without blocking the killer.
DEFAULT_GRACE_SECONDS: float = 3.0

#: Hard ceiling on the grace period. Callers asking for longer get
#: silently clamped — a kill operation that takes more than a minute
#: is masking a separate hang, not a slow-shutdown bug.
MAX_GRACE_SECONDS: float = 60.0


@dataclass(frozen=True)
class KillTreeResult:
    """Outcome of a :func:`kill_process_tree` call.

    Attributes:
        root_pid: The pid passed to :func:`kill_process_tree`.
        terminated: PIDs that exited within the grace period after the
            graceful terminate.
        force_killed: PIDs that required force-kill after the grace
            period. Non-empty here usually means the process had a
            wedged shutdown hook (or was unkillable due to permission).
        unreachable: PIDs that were unreachable at terminate time
            (already exited or permission denied). Not an error.
        elapsed_seconds: Wall-clock time the call took.
        used_process_group: ``True`` on POSIX when the SIGTERM/SIGKILL
            went to the process group; ``False`` when each pid was
            signalled individually.
    """

    root_pid: int
    terminated: tuple[int, ...] = field(default_factory=tuple)
    force_killed: tuple[int, ...] = field(default_factory=tuple)
    unreachable: tuple[int, ...] = field(default_factory=tuple)
    elapsed_seconds: float = 0.0
    used_process_group: bool = False

    @property
    def total_killed(self) -> int:
        """Number of pids that this call removed from the OS table."""
        return len(self.terminated) + len(self.force_killed)


def _clamp_grace(grace_seconds: float) -> float:
    """Clamp ``grace_seconds`` to ``[0.0, MAX_GRACE_SECONDS]``."""
    if grace_seconds < 0:
        return 0.0
    if grace_seconds > MAX_GRACE_SECONDS:
        return MAX_GRACE_SECONDS
    return float(grace_seconds)


def _is_alive(pid: int) -> bool:
    """Return ``True`` when ``pid`` still has a live OS entry."""
    if pid is None or pid <= 0:
        return False
    try:
        import psutil
    except ImportError:
        return False
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, ValueError):
        return False
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _collect_tree(pid: int) -> tuple[list, list[int]]:
    """Snapshot ``pid`` plus all descendants.

    Returns:
        ``(psutil_processes, unreachable_pids)``. ``psutil_processes``
        is a list of live :class:`psutil.Process` objects (root first,
        descendants in BFS order). ``unreachable_pids`` is the list of
        pids the caller asked about that we could not snapshot
        (already exited or permission denied).
    """
    try:
        import psutil
    except ImportError:
        LOGGER.warning("psutil not available; kill_process_tree is a no-op")
        return [], [pid]
    unreachable: list[int] = []
    procs: list = []
    try:
        root = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return [], [pid]
    procs.append(root)
    try:
        for child in root.children(recursive=True):
            procs.append(child)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # The root went away between Process() and children(); the
        # snapshot we have is still useful (just the root, or empty).
        pass
    return procs, unreachable


def kill_process_tree(
    pid: int,
    *,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    detached: bool = True,
    clock: Optional[callable] = None,
) -> KillTreeResult:
    """Terminate ``pid`` and every descendant, gracefully then forcefully.

    The call is a no-op when ``pid`` is invalid (non-positive) or
    unreachable, when ``psutil`` is not installed, or when no member
    of the tree is alive at call time.

    Args:
        pid: Root process id. Negative / zero pids return an empty
            result without raising.
        grace_seconds: Seconds to wait between graceful terminate
            and force-kill. Clamped to ``[0, MAX_GRACE_SECONDS]``.
        detached: POSIX hint. ``True`` (the default) signals that
            ``pid`` belongs to its own process group and the killer
            may use ``os.killpg`` for efficiency. ``False`` falls
            back to per-pid signalling. Windows ignores this hint.
        clock: Optional time source for tests. Defaults to
            :func:`time.monotonic`.

    Returns:
        :class:`KillTreeResult` describing the outcome.
    """
    if pid is None or pid <= 0:
        return KillTreeResult(root_pid=pid or -1)
    grace = _clamp_grace(grace_seconds)
    now = clock or time.monotonic
    start = now()

    procs, unreachable = _collect_tree(pid)
    if not procs:
        return KillTreeResult(
            root_pid=pid,
            unreachable=tuple(unreachable),
            elapsed_seconds=now() - start,
        )

    try:
        import psutil
    except ImportError:
        return KillTreeResult(
            root_pid=pid,
            unreachable=tuple(p.pid for p in procs),
            elapsed_seconds=now() - start,
        )

    # Step 1: graceful terminate. On Windows this maps to TerminateProcess
    # (no graceful equivalent for non-console processes); on POSIX this is
    # SIGTERM. Per-pid loop with exception swallow so one unreachable
    # member doesn't abort the rest.
    requested: list[int] = []
    for proc in procs:
        try:
            proc.terminate()
            requested.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            unreachable.append(proc.pid)
        except Exception:  # noqa: BLE001
            LOGGER.warning("graceful terminate failed for pid %d", proc.pid, exc_info=True)
            unreachable.append(proc.pid)

    if not requested:
        return KillTreeResult(
            root_pid=pid,
            unreachable=tuple(unreachable),
            elapsed_seconds=now() - start,
        )

    # Step 2: wait for graceful exits up to the grace window.
    terminated: list[int] = []
    force_pending: list = []
    try:
        gone, alive = psutil.wait_procs(procs, timeout=grace)
        for proc in gone:
            terminated.append(proc.pid)
        force_pending.extend(alive)
    except Exception:  # noqa: BLE001
        # wait_procs can raise on extremely-large trees; fall back to
        # per-pid polling against the live-check helper.
        deadline = now() + grace
        while now() < deadline:
            still_alive = [p for p in procs if _is_alive(p.pid)]
            if not still_alive:
                break
            time.sleep(0.05)
        for proc in procs:
            if proc.pid in unreachable:
                continue
            if _is_alive(proc.pid):
                force_pending.append(proc)
            else:
                terminated.append(proc.pid)

    # Step 3: force-kill anything still alive after grace.
    force_killed: list[int] = []
    for proc in force_pending:
        try:
            proc.kill()
            force_killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            unreachable.append(proc.pid)
        except Exception:  # noqa: BLE001
            LOGGER.warning("force kill failed for pid %d", proc.pid, exc_info=True)
            unreachable.append(proc.pid)
    # Best-effort short wait so the OS table is consistent on return.
    if force_killed:
        try:
            psutil.wait_procs(force_pending, timeout=min(2.0, grace))
        except Exception:  # noqa: BLE001
            pass

    return KillTreeResult(
        root_pid=pid,
        terminated=tuple(sorted(set(terminated))),
        force_killed=tuple(sorted(set(force_killed))),
        unreachable=tuple(sorted(set(unreachable))),
        elapsed_seconds=now() - start,
        used_process_group=detached and _is_posix(),
    )


def _is_posix() -> bool:
    """True on Linux / macOS / WSL; False on Windows."""
    import os
    return os.name == "posix"


def kill_pid_if_alive(
    pid: int,
    *,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    clock: Optional[callable] = None,
) -> KillTreeResult:
    """Convenience wrapper: terminate ``pid`` only, no descendants.

    Useful when the caller knows the target spawned no children
    (e.g. a single isolated venv subprocess). Equivalent to
    ``kill_process_tree(pid)`` when ``pid`` is a leaf.

    Args:
        pid: Target process id.
        grace_seconds: Grace before force-kill.
        clock: Optional time source for tests.

    Returns:
        :class:`KillTreeResult`.
    """
    if pid is None or pid <= 0:
        return KillTreeResult(root_pid=pid or -1)
    if not _is_alive(pid):
        return KillTreeResult(
            root_pid=pid,
            unreachable=(pid,),
            elapsed_seconds=0.0,
        )
    grace = _clamp_grace(grace_seconds)
    now = clock or time.monotonic
    start = now()
    try:
        import psutil
    except ImportError:
        return KillTreeResult(root_pid=pid, unreachable=(pid,))
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return KillTreeResult(root_pid=pid, unreachable=(pid,))
    try:
        proc.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return KillTreeResult(
            root_pid=pid,
            unreachable=(pid,),
            elapsed_seconds=now() - start,
        )
    terminated: list[int] = []
    force_killed: list[int] = []
    try:
        gone, alive = psutil.wait_procs([proc], timeout=grace)
        if gone:
            terminated.append(pid)
        for p in alive:
            try:
                p.kill()
                force_killed.append(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
            force_killed.append(pid)
        except Exception:  # noqa: BLE001
            pass
    return KillTreeResult(
        root_pid=pid,
        terminated=tuple(terminated),
        force_killed=tuple(force_killed),
        elapsed_seconds=now() - start,
        used_process_group=False,
    )


__all__ = [
    "DEFAULT_GRACE_SECONDS",
    "KillTreeResult",
    "MAX_GRACE_SECONDS",
    "kill_pid_if_alive",
    "kill_process_tree",
]
