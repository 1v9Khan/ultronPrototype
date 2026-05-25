"""Periodic subprocess reaper with persistent-tag carve-out.

Adapted from cline's ``BACKGROUND_COMMAND_TIMEOUT_MS`` pattern
(Apache 2.0; see ``THIRD_PARTY_NOTICES.md``). Ultron's variant:

* Registers EVERY tracked subprocess in a single shared registry so a
  voice command can answer "what's running right now?".
* Tags persistent processes (MCP server, Parakeet HTTP daemon, Kokoro
  stream worker) so they are never killed by the timeout.
* Adds a resource-budget warning tier (RSS > N MB AND age > N s) that
  fires a notice without killing — operators decide whether to act.
* Single-threaded periodic check (no per-process timers); poll cadence
  defaults to 60 s.

The killer NEVER touches the live Ultron orchestrator process or its
ancestor chain (the safety contract from
``scripts/cleanup_stale_processes.py``). When the registry lacks a
process for a pid (a child spawned outside ultron's tracking), the
killer does NOT touch it — only registered children are managed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

LOGGER = logging.getLogger(__name__)

#: Default hard cap (seconds) after which a non-persistent subprocess
#: is forcibly terminated. Mirrors cline's 10-minute window.
DEFAULT_HARD_TIMEOUT_S: float = 10 * 60.0

#: Default poll cadence (seconds) for the killer thread.
DEFAULT_POLL_INTERVAL_S: float = 60.0

#: Default RSS-threshold (MB) at which the killer logs a warning for
#: a still-running process. Diagnostic-only; never auto-kills.
DEFAULT_WARN_RSS_MB: int = 2048

#: Default age-threshold (seconds) paired with the RSS warn; the
#: warning only fires once both apply (filters out fresh heavy starts).
DEFAULT_WARN_AGE_S: float = 5 * 60.0


def _now() -> float:
    """Monotonic wall-clock seconds for age math."""
    return time.monotonic()


def _safe_terminate(pid: int) -> bool:
    """Best-effort SIGTERM/TerminateProcess for ``pid``."""
    try:
        import psutil
    except ImportError:
        LOGGER.warning("psutil not available; cannot terminate pid %d", pid)
        return False
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except psutil.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True
    except Exception:  # noqa: BLE001
        LOGGER.warning("failed to terminate pid %d", pid, exc_info=True)
        return False


def _rss_mb(pid: int) -> Optional[int]:
    """Resident-set-size in MB for ``pid``, or None when unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(pid)
        return int(proc.memory_info().rss // (1024 * 1024))
    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):  # noqa: BLE001
        return None


@dataclass
class TrackedProcess:
    """Registry entry for one tracked subprocess.

    Attributes:
        pid: OS process id.
        label: human-readable description ("parakeet-server", "claude-cli",
            "kokoro-synth"). Used in audit logs + voice-side narration.
        persistent: when True, the process is NEVER auto-killed by the
            timeout regardless of age. MCP servers, Kokoro daemons, and
            the Parakeet HTTP server should use this.
        registered_at: monotonic timestamp the process was registered.
            Recorded via the owning killer's ``_clock`` so test hooks
            can inject deterministic time.
        hard_timeout_s: per-process override on the global cap. ``None``
            uses :data:`DEFAULT_HARD_TIMEOUT_S`. Persistent processes
            ignore this field entirely.
        owner: optional opaque caller token; useful for cancel-on-shutdown.
        on_terminate: optional callback invoked AFTER termination
            succeeds (receives the report).
    """

    pid: int
    label: str
    persistent: bool = False
    registered_at: float = field(default_factory=_now)
    hard_timeout_s: Optional[float] = None
    owner: Optional[str] = None
    on_terminate: Optional[Callable[["ZombieReport"], None]] = None

    def age_seconds(self, clock: Callable[[], float] = _now) -> float:
        """Wall-clock seconds since registration.

        Args:
            clock: callable returning monotonic seconds (test hook).
                Defaults to :func:`_now` (real monotonic time). The
                :class:`ZombieKiller` always passes its own ``_clock``
                so injected clocks propagate through the sweep.
        """
        return clock() - self.registered_at

    def effective_timeout(self, fallback: float) -> float:
        """Resolve the timeout to enforce for this process."""
        return self.hard_timeout_s if self.hard_timeout_s is not None else fallback


@dataclass(frozen=True)
class ZombieReport:
    """Diagnostic report for a single termination / warning event.

    Attributes:
        pid: terminated process id.
        label: the process label.
        age_seconds: age at the moment of the event.
        action: ``"terminated"`` for kills, ``"warned"`` for the
            resource-budget warning tier.
        rss_mb: resident-set-size at the event, if available.
    """

    pid: int
    label: str
    age_seconds: float
    action: str
    rss_mb: Optional[int] = None


class ZombieKiller:
    """Periodic reaper for tracked, non-persistent subprocesses.

    Args:
        hard_timeout_s: global age cap (seconds) after which a
            non-persistent subprocess is terminated.
        poll_interval_s: how often the killer's background thread
            wakes up to scan the registry.
        warn_rss_mb: RSS threshold (MB) at which a long-running process
            triggers a warning (no kill).
        warn_age_s: age threshold (seconds) that must also be met
            before the RSS warn fires.
        clock: optional callable returning monotonic seconds (test hook).
        terminator: optional callable accepting ``pid`` and returning
            True on success (test hook; defaults to psutil terminate/kill).
        rss_probe: optional callable returning the RSS in MB for a pid
            (test hook; defaults to psutil).

    Notes:
        The killer is NOT started automatically on construction. The
        caller must invoke :meth:`start` (typically the orchestrator
        on init). :meth:`shutdown` joins the worker thread and is
        safe to call multiple times.
    """

    def __init__(
        self,
        *,
        hard_timeout_s: float = DEFAULT_HARD_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        warn_rss_mb: int = DEFAULT_WARN_RSS_MB,
        warn_age_s: float = DEFAULT_WARN_AGE_S,
        clock: Optional[Callable[[], float]] = None,
        terminator: Optional[Callable[[int], bool]] = None,
        rss_probe: Optional[Callable[[int], Optional[int]]] = None,
    ) -> None:
        self._hard_timeout = hard_timeout_s
        self._poll_interval = poll_interval_s
        self._warn_rss_mb = warn_rss_mb
        self._warn_age_s = warn_age_s
        self._clock = clock or _now
        self._terminator = terminator or _safe_terminate
        self._rss_probe = rss_probe or _rss_mb
        self._registry: dict[int, TrackedProcess] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reports: list[ZombieReport] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background reaper thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="ultron-zombie-killer",
                daemon=True,
            )
            self._thread.start()

    def shutdown(self) -> None:
        """Signal stop and join the reaper thread (idempotent)."""
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop_event.set()
        if thread is not None:
            thread.join(timeout=5.0)

    def is_running(self) -> bool:
        """True when the background thread is alive."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register(
        self,
        pid: int,
        label: str,
        *,
        persistent: bool = False,
        hard_timeout_s: Optional[float] = None,
        owner: Optional[str] = None,
        on_terminate: Optional[Callable[[ZombieReport], None]] = None,
    ) -> TrackedProcess:
        """Track ``pid`` in the registry.

        Args:
            pid: process id to track.
            label: short descriptor (used in audit + voice rendering).
            persistent: when True, immune to the auto-kill timeout.
            hard_timeout_s: per-process timeout override.
            owner: opaque caller token for cancel-on-shutdown groupings.
            on_terminate: callback fired after termination succeeds.

        Returns:
            The :class:`TrackedProcess` entry that was registered.

        Notes:
            Re-registering an existing pid updates the entry in place
            (preserves ``registered_at``) so callers can flip the
            persistent flag without losing age tracking.
        """
        with self._lock:
            existing = self._registry.get(pid)
            if existing is not None:
                existing.label = label
                existing.persistent = persistent
                if hard_timeout_s is not None:
                    existing.hard_timeout_s = hard_timeout_s
                if owner is not None:
                    existing.owner = owner
                if on_terminate is not None:
                    existing.on_terminate = on_terminate
                return existing
            entry = TrackedProcess(
                pid=pid,
                label=label,
                persistent=persistent,
                hard_timeout_s=hard_timeout_s,
                owner=owner,
                on_terminate=on_terminate,
                registered_at=self._clock(),
            )
            self._registry[pid] = entry
            return entry

    def unregister(self, pid: int) -> bool:
        """Remove ``pid`` from the registry.

        Returns:
            True when an entry was removed, False otherwise.
        """
        with self._lock:
            return self._registry.pop(pid, None) is not None

    def mark_persistent(self, pid: int, persistent: bool = True) -> bool:
        """Flip the persistent flag on a tracked process.

        Returns:
            True when the pid was found, False otherwise.
        """
        with self._lock:
            entry = self._registry.get(pid)
            if entry is None:
                return False
            entry.persistent = persistent
            return True

    def list_tracked(self) -> list[TrackedProcess]:
        """Snapshot of every tracked process (sorted by registration time)."""
        with self._lock:
            return sorted(self._registry.values(), key=lambda e: e.registered_at)

    def lookup(self, pid: int) -> Optional[TrackedProcess]:
        """Return the registry entry for ``pid``, or ``None``."""
        with self._lock:
            return self._registry.get(pid)

    def find_by_label(self, label: str) -> list[TrackedProcess]:
        """Return every tracked entry whose label matches ``label``."""
        with self._lock:
            return [e for e in self._registry.values() if e.label == label]

    def recent_reports(self, limit: int = 50) -> list[ZombieReport]:
        """Return up to ``limit`` most-recent termination / warn reports."""
        with self._lock:
            return list(self._reports[-limit:])

    def clear_reports(self) -> None:
        """Reset the report buffer."""
        with self._lock:
            self._reports.clear()

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def sweep_once(self) -> list[ZombieReport]:
        """Run a single sweep and return the resulting reports.

        Suitable for tests that don't want to spawn the worker thread,
        and for the manual ``ultron stats`` CLI.
        """
        produced: list[ZombieReport] = []
        with self._lock:
            entries = list(self._registry.values())
        for entry in entries:
            if entry.persistent:
                continue
            age = entry.age_seconds(self._clock)
            timeout = entry.effective_timeout(self._hard_timeout)
            if age >= timeout:
                report = self._terminate_entry(entry, age)
                if report is not None:
                    produced.append(report)
                continue
            if age >= self._warn_age_s:
                rss = self._rss_probe(entry.pid)
                if rss is not None and rss >= self._warn_rss_mb:
                    warn = ZombieReport(
                        pid=entry.pid,
                        label=entry.label,
                        age_seconds=age,
                        action="warned",
                        rss_mb=rss,
                    )
                    self._record_report(warn)
                    produced.append(warn)
                    LOGGER.warning(
                        "long-running subprocess: pid=%d label=%s age=%ds rss=%dMB",
                        entry.pid,
                        entry.label,
                        int(age),
                        rss,
                    )
        return produced

    def _loop(self) -> None:
        """Background thread main loop."""
        while not self._stop_event.is_set():
            try:
                self.sweep_once()
            except Exception:  # noqa: BLE001
                LOGGER.warning("zombie-killer sweep raised", exc_info=True)
            # Wait with interruptibility.
            self._stop_event.wait(self._poll_interval)

    def _terminate_entry(
        self, entry: TrackedProcess, age: float,
    ) -> Optional[ZombieReport]:
        """Terminate ``entry`` and return the resulting report."""
        rss = self._rss_probe(entry.pid)
        success = self._terminator(entry.pid)
        if not success:
            return None
        with self._lock:
            self._registry.pop(entry.pid, None)
        report = ZombieReport(
            pid=entry.pid,
            label=entry.label,
            age_seconds=age,
            action="terminated",
            rss_mb=rss,
        )
        self._record_report(report)
        LOGGER.info(
            "terminated stale subprocess: pid=%d label=%s age=%ds rss=%s",
            entry.pid,
            entry.label,
            int(age),
            rss if rss is not None else "unknown",
        )
        if entry.on_terminate is not None:
            try:
                entry.on_terminate(report)
            except Exception:  # noqa: BLE001
                LOGGER.warning(
                    "on_terminate callback raised for pid %d", entry.pid,
                    exc_info=True,
                )
        return report

    def _record_report(self, report: ZombieReport) -> None:
        """Append a report to the bounded buffer."""
        with self._lock:
            self._reports.append(report)
            # Keep the buffer bounded; oldest-first eviction.
            if len(self._reports) > 256:
                del self._reports[:128]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_KILLER: Optional[ZombieKiller] = None
_DEFAULT_KILLER_LOCK = threading.RLock()


def get_zombie_killer(**kwargs: Any) -> ZombieKiller:
    """Return (and lazily construct) the module-level zombie killer.

    Args:
        **kwargs: forwarded to :class:`ZombieKiller` on first
            construction; ignored thereafter.
    """
    global _DEFAULT_KILLER
    with _DEFAULT_KILLER_LOCK:
        if _DEFAULT_KILLER is None:
            _DEFAULT_KILLER = ZombieKiller(**kwargs)
        return _DEFAULT_KILLER


def reset_zombie_killer_for_testing() -> None:
    """Drop the module-level singleton (test-only)."""
    global _DEFAULT_KILLER
    with _DEFAULT_KILLER_LOCK:
        if _DEFAULT_KILLER is not None:
            _DEFAULT_KILLER.shutdown()
        _DEFAULT_KILLER = None


__all__ = [
    "DEFAULT_HARD_TIMEOUT_S",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_WARN_AGE_S",
    "DEFAULT_WARN_RSS_MB",
    "TrackedProcess",
    "ZombieKiller",
    "ZombieReport",
    "get_zombie_killer",
    "reset_zombie_killer_for_testing",
]
