"""Per-session counter for fail-open events across subsystems.

Most of Ultron is fail-open by design: a reranker miss falls back to
the composite scorer, a provider chain falls through SearxNG -> Brave
-> DDG, a supervisor exception returns a NEW-action decision, a slow
bus subscriber logs WARN. The cumulative effect is graceful
degradation -- but it's easy to mask real bugs over time because no
single fail-open fire is loud enough to investigate.

This module tracks per-session counts and persists them so the next
startup can log a one-line summary ("previous session: N fail-opens
across categories X, Y, Z"). A spike in a particular category between
sessions surfaces a regression that would otherwise stay quiet.

API:
  * :func:`configure(log_path)` -- called once at orchestrator init.
    Resets the in-memory counts for the new session and remembers the
    log path for the flush.
  * :func:`record(category, reason="")` -- called from any subsystem's
    fail-open path. Increments the counter for ``category``. Reason
    is opt-in metadata (not currently aggregated; reserved for future
    per-reason summaries).
  * :func:`session_counts()` -- snapshot of current session's counters.
  * :func:`flush_to_disk()` -- append the session's totals as one
    JSON line to the log. Called at orchestrator shutdown OR
    periodically.
  * :func:`previous_session_counts()` -- read the last JSONL entry
    from the log file. Returns None when log is missing / empty /
    malformed.
  * :func:`render_summary(counts)` -- one-line human-readable string
    for log emission.

Fail-safe: every entry point swallows exceptions. The counter never
prevents the operation it's tracking.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("ultron.resilience.fail_open_log")

# Module-level state. Singleton by design -- the recorder is called
# from arbitrary subsystem paths so we don't pass an instance around.
_LOCK = threading.RLock()
_COUNTS: Dict[str, int] = {}
_LOG_PATH: Optional[Path] = None
_SESSION_START_TIME: float = 0.0


# Known categories. Adding a category here is optional but lets
# tests + the doc cross-reference the canonical set.
KNOWN_CATEGORIES: tuple[str, ...] = (
    "bus_slow_subscriber",
    "reranker_load_fail",
    "reranker_predict_fail",
    "provider_chain_fallthrough",
    "reader_chain_fallthrough",
    "supervisor_exception",
    "memory_retrieve_fail",
    "web_gate_fail",
    "intent_recognizer_fail",
    "kokoro_synth_fail",
    "openclaw_unreachable",
    "stt_swap_fail",
)


def configure(log_path: Optional[Path]) -> None:
    """Set the log path and start a fresh in-memory counter for this session.

    Args:
        log_path: append-only JSONL file. Pass None to disable
            disk persistence (counters still tracked in-process).
    """
    global _LOG_PATH, _SESSION_START_TIME
    with _LOCK:
        _LOG_PATH = log_path
        _SESSION_START_TIME = time.time()
        _COUNTS.clear()


def record(category: str, reason: str = "") -> None:
    """Increment the counter for ``category``. Fail-safe.

    Args:
        category: short string naming the fail-open path (see
            :data:`KNOWN_CATEGORIES`). Unknown categories work too --
            the counter dict is open-ended.
        reason: opt-in metadata describing the specific fall-through
            (e.g. event type for the bus path, exception class name
            for supervisor failures). Reserved for future per-reason
            aggregation; not currently summarized.
    """
    try:
        with _LOCK:
            _COUNTS[category] = _COUNTS.get(category, 0) + 1
    except Exception:                                              # noqa: BLE001
        # Counter must never break the wrapping subsystem.
        pass


def session_counts() -> Dict[str, int]:
    """Snapshot of the current session's per-category counters."""
    with _LOCK:
        return dict(_COUNTS)


def flush_to_disk() -> None:
    """Append the current session's counts to the JSONL log.

    No-op when :func:`configure` was called with ``None`` or when
    the session recorded zero events. Idempotent in the sense that
    calling it more than once appends each time -- callers control
    the cadence.
    """
    with _LOCK:
        if _LOG_PATH is None:
            return
        if not _COUNTS:
            return
        entry = {
            "session_start": _SESSION_START_TIME,
            "session_end": time.time(),
            "counts": dict(_COUNTS),
        }
        log_path = _LOG_PATH
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception as e:                                         # noqa: BLE001
        logger.debug("fail_open_log: flush_to_disk failed: %s", e)


def previous_session_counts(
    log_path: Optional[Path] = None,
) -> Optional[Dict[str, int]]:
    """Read the most recent JSONL entry and return its counts.

    Args:
        log_path: explicit path override. When None, uses the path
            registered via :func:`configure`.

    Returns:
        Dict of category -> count from the last logged session, or
        None when the file is missing / empty / malformed.
    """
    if log_path is None:
        with _LOCK:
            log_path = _LOG_PATH
    if log_path is None or not log_path.exists():
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            return None
        entry = json.loads(lines[-1])
        counts = entry.get("counts")
        if isinstance(counts, dict):
            # Drop any non-int values defensively.
            return {k: int(v) for k, v in counts.items() if isinstance(v, (int, float))}
        return None
    except Exception:                                              # noqa: BLE001
        return None


def render_summary(counts: Optional[Dict[str, int]]) -> str:
    """Render counts as a single-line summary suitable for log output.

    Returns ``"no fail-open events recorded"`` when counts is None
    or empty. Otherwise lists categories alphabetically with their
    counts, e.g. ``"bus_slow_subscriber=2, reranker_load_fail=1"``.
    """
    if not counts:
        return "no fail-open events recorded"
    return ", ".join(f"{cat}={cnt}" for cat, cnt in sorted(counts.items()))


def reset_for_testing() -> None:
    """Clear module state. Test-only escape hatch."""
    global _LOG_PATH, _SESSION_START_TIME
    with _LOCK:
        _LOG_PATH = None
        _SESSION_START_TIME = 0.0
        _COUNTS.clear()


__all__ = [
    "KNOWN_CATEGORIES",
    "configure",
    "flush_to_disk",
    "previous_session_counts",
    "record",
    "render_summary",
    "reset_for_testing",
    "session_counts",
]
