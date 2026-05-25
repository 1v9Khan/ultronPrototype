"""Per-session file-read cache with mtime-validated short-circuit.

Adapted from cline's ``TaskState.fileReadCache`` pattern (Apache 2.0;
see ``THIRD_PARTY_NOTICES.md``). The cache lives at the session level
(one cache per coding task / voice session); on a repeated read whose
mtime is unchanged, the cached content is served and a short
``[Note] ... served from per-session cache (read N times; mtime
unchanged).`` notice is appended so the LLM stops loop-reading.

The cache is byte-content based — it does not depend on encoding or
file-system case-sensitivity. Path canonicalization is the caller's
responsibility (the safety validator already canonicalises paths
upstream).

This module is intentionally side-effect-free: it does not perform
the actual file read. Callers wrap their existing read primitive with
the :func:`maybe_serve_from_cache` -> read -> :func:`record_read`
pattern (see the bottom of this docstring for a recipe).

Recipe:

.. code-block:: python

    cache = FileReadCache(session_id="abc")

    def read_file(path: Path) -> tuple[str, str | None]:
        cached = cache.maybe_serve_from_cache(path)
        if cached is not None:
            return cached.content, cached.notice
        content = path.read_text(encoding="utf-8")
        cache.record_read(path, content)
        return content, None
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ultron.llm.response_format import cached_read_notice


@dataclass(frozen=True)
class CachedReadEntry:
    """Snapshot of a single cached file-read entry."""

    content: str
    mtime_ns: int
    read_count: int
    notice: str


@dataclass
class _Entry:
    """Mutable cache record (internal)."""

    content: str
    mtime_ns: int
    read_count: int = 0


class FileReadCache:
    """Thread-safe per-session file-read cache.

    Args:
        session_id: identifier for the owning session (informational;
            two caches with the same id are still independent objects).
        max_entries: optional cap on the number of files held; the cache
            evicts the lowest-read-count entry when the cap is exceeded.
            ``None`` (default) keeps every entry; for voice / coding
            sessions this typically tops out at low dozens.

    Notes:
        - mtime is read at ``record_read`` time and verified at
          ``maybe_serve_from_cache`` time; an entry whose underlying
          file has changed (mtime delta) is invalidated transparently.
        - ``read_count`` reflects how many times the file has been
          touched via this cache, including the first read (which is
          recorded by ``record_read``).
    """

    def __init__(
        self,
        session_id: str = "default",
        *,
        max_entries: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self._max_entries = max_entries
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_serve_from_cache(
        self, path: str | os.PathLike[str],
    ) -> Optional[CachedReadEntry]:
        """Return the cached content for ``path`` if mtime is unchanged.

        Args:
            path: file path to look up.

        Returns:
            :class:`CachedReadEntry` when a fresh cache hit is found,
            ``None`` when the file is not in the cache OR when its mtime
            has changed since the cached entry was recorded.

        Notes:
            On a cache hit, the entry's ``read_count`` is incremented
            and the returned notice reflects the new count.
        """
        resolved = self._key(path)
        with self._lock:
            entry = self._entries.get(resolved)
            if entry is None:
                return None
            current_mtime = self._current_mtime(resolved)
            if current_mtime is None or current_mtime != entry.mtime_ns:
                # Stale: drop the entry so the next read repopulates.
                self._entries.pop(resolved, None)
                return None
            entry.read_count += 1
            return CachedReadEntry(
                content=entry.content,
                mtime_ns=entry.mtime_ns,
                read_count=entry.read_count,
                notice=cached_read_notice(resolved, entry.read_count),
            )

    def record_read(
        self,
        path: str | os.PathLike[str],
        content: str,
    ) -> None:
        """Record a fresh read of ``path`` with its current content.

        Args:
            path: file path that was just read.
            content: contents that were read.
        """
        resolved = self._key(path)
        mtime = self._current_mtime(resolved)
        if mtime is None:
            # File disappeared in the gap between read and record; do
            # nothing (a future read will re-record it).
            return
        with self._lock:
            existing = self._entries.get(resolved)
            if existing is None:
                self._entries[resolved] = _Entry(
                    content=content, mtime_ns=mtime, read_count=1,
                )
            else:
                existing.content = content
                existing.mtime_ns = mtime
                existing.read_count = max(1, existing.read_count)
            self._evict_if_needed()

    def invalidate(self, path: str | os.PathLike[str]) -> bool:
        """Drop ``path`` from the cache if present.

        Args:
            path: file path to evict.

        Returns:
            True when an entry was removed, False otherwise.
        """
        resolved = self._key(path)
        with self._lock:
            return self._entries.pop(resolved, None) is not None

    def clear(self) -> None:
        """Drop every cached entry (e.g. on session end)."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, int]:
        """Return diagnostic counts for the cache."""
        with self._lock:
            total_reads = sum(e.read_count for e in self._entries.values())
            return {
                "entries": len(self._entries),
                "total_reads": total_reads,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(path: str | os.PathLike[str]) -> str:
        """Canonical cache key for ``path``.

        Uses absolute resolution to normalise the key; Windows
        case-insensitivity is left to the OS layer.
        """
        try:
            return str(Path(path).resolve(strict=False))
        except Exception:  # noqa: BLE001
            return str(path)

    @staticmethod
    def _current_mtime(path: str) -> Optional[int]:
        """Read the mtime_ns of ``path`` or return ``None`` on failure."""
        try:
            return os.stat(path).st_mtime_ns
        except OSError:
            return None

    def _evict_if_needed(self) -> None:
        """Evict the lowest-touched entry when the cache exceeds capacity."""
        if self._max_entries is None or len(self._entries) <= self._max_entries:
            return
        # Pick the entry with the smallest read_count (oldest-touched).
        victim = min(self._entries.items(), key=lambda kv: kv[1].read_count)
        self._entries.pop(victim[0], None)


# ---------------------------------------------------------------------------
# Module-level singletons (per-session registry)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, FileReadCache] = {}
_REGISTRY_LOCK = threading.RLock()


def get_file_read_cache(
    session_id: str,
    *,
    max_entries: Optional[int] = None,
) -> FileReadCache:
    """Return the file-read cache for ``session_id`` (creating if needed).

    Args:
        session_id: caller's session identifier.
        max_entries: optional cap on cache size (only applied on creation).

    Returns:
        Stable :class:`FileReadCache` instance for the session.
    """
    with _REGISTRY_LOCK:
        cache = _REGISTRY.get(session_id)
        if cache is None:
            cache = FileReadCache(session_id, max_entries=max_entries)
            _REGISTRY[session_id] = cache
        return cache


def reset_file_read_cache_registry() -> None:
    """Drop every per-session cache (test-only helper)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


__all__ = [
    "CachedReadEntry",
    "FileReadCache",
    "get_file_read_cache",
    "reset_file_read_cache_registry",
]
