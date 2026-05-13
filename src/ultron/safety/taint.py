"""Cross-capability taint tracking.

The dangerous combos are usually two allowed capabilities chained:
Cap-1 captures the screen, then an outbound tool sends the bytes;
Cap-3 reads an authenticated browser page, then an email tool sends
the content. The individual capabilities are allowed (that's the
product); the chain is the exfil.

Solution: track recent capability OUTPUTS as input TAINTS. When a
tool call's argument bytes can be traced back (within a small time
window) to a tainted capability output, the OUT-gate fires.

Phase 5 wires this in. The module exposes:

* :class:`TaintTracker` -- per-orchestrator instance. Records the
  hash of each tainted capability output and the timestamp. Queries
  return True if a candidate argument matches a recent taint.
* :func:`get_taint_tracker()` -- module-level singleton.

Hash matching is deliberately tight: we hash the BYTES that a
capability output produced, then check whether new tool arguments
contain those exact bytes. This is a conservative match -- if the
model transforms the bytes (base64 encode, re-encode, paraphrase,
etc.), the taint is lost. That's acceptable for Phase 5 -- adversarial
laundering can be defended against in a later iteration.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ultron.safety.taint")

# How long a taint persists. After this window the bytes are
# considered no longer hot -- the model could have re-derived them
# from another source. 60 seconds is a sensible default; tunable
# via config in a later iteration.
DEFAULT_TAINT_TTL_SECONDS = 60.0


@dataclass(frozen=True)
class TaintEntry:
    """One recorded capability output.

    Attributes:
        digest: SHA-256 of the bytes the capability produced.
        capability: which capability produced the bytes
            (``screen_context``, ``browser_authenticated``, etc.).
        timestamp: monotonic time when the entry was recorded.
        size_bytes: total length of the bytes (for log context).
    """

    digest: str
    capability: str
    timestamp: float
    size_bytes: int


class TaintTracker:
    """Record + query capability-output taints.

    Thread-safe. Single in-memory rolling buffer; entries expire
    after ``ttl_seconds``. Tracker size is bounded -- once it
    exceeds ``max_entries``, the oldest entries are dropped.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TAINT_TTL_SECONDS,
        max_entries: int = 256,
    ) -> None:
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._entries: list[TaintEntry] = []
        self._lock = threading.Lock()

    def record(self, *, data: bytes, capability: str) -> str:
        """Record that ``capability`` produced ``data``. Returns digest.

        Skips empty payloads (no point taint-tracking empty bytes).
        """
        if not data:
            return ""
        digest = hashlib.sha256(data).hexdigest()
        entry = TaintEntry(
            digest=digest,
            capability=capability,
            timestamp=time.monotonic(),
            size_bytes=len(data),
        )
        with self._lock:
            self._entries.append(entry)
            self._prune_locked()
        return digest

    def _prune_locked(self) -> None:
        """Drop expired entries and trim to max size. Caller holds lock."""
        now = time.monotonic()
        cutoff = now - self._ttl
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        if len(self._entries) > self._max_entries:
            # Drop oldest.
            self._entries = self._entries[-self._max_entries:]

    def has_taint(self, *, data: bytes) -> Optional[TaintEntry]:
        """Return the taint entry that ``data`` was sourced from,
        or None if no recent match.

        Conservative byte-exact match: hash the candidate bytes and
        compare against the recorded digests. If the model transforms
        the bytes (encode, paraphrase), the taint is lost -- that's
        an accepted limitation for Phase 5.
        """
        if not data:
            return None
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            self._prune_locked()
            for e in self._entries:
                if e.digest == digest:
                    return e
        return None

    def has_taint_str(self, *, text: str) -> Optional[TaintEntry]:
        """Convenience wrapper for text payloads."""
        if not text:
            return None
        return self.has_taint(data=text.encode("utf-8", errors="ignore"))

    def clear(self) -> None:
        """Test hook -- drop all entries."""
        with self._lock:
            self._entries.clear()

    @property
    def size(self) -> int:
        """Current number of live entries (post-prune)."""
        with self._lock:
            self._prune_locked()
            return len(self._entries)


_tracker_singleton: Optional[TaintTracker] = None
_tracker_lock = threading.Lock()


def get_taint_tracker() -> TaintTracker:
    """Module-level singleton accessor."""
    global _tracker_singleton
    if _tracker_singleton is None:
        with _tracker_lock:
            if _tracker_singleton is None:
                _tracker_singleton = TaintTracker()
    return _tracker_singleton


def set_taint_tracker(tracker: Optional[TaintTracker]) -> None:
    """Test hook -- swap the singleton."""
    global _tracker_singleton
    with _tracker_lock:
        _tracker_singleton = tracker if tracker is not None else None
