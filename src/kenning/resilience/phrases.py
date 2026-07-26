"""Phase 4 — error-phrase selection helper.

Reads ``config.error_phrases.<failure_mode>`` and returns a shuffled,
non-repeating phrase. Mirrors the existing :class:`AcknowledgmentSource`
pattern from web_search/acknowledgments.py: shuffled cycles so the user
never hears the same phrase twice in a row, but each phrase shows up
once per cycle.

Per-failure-mode sources are cached at module level so multiple call
sites for the same failure share state (no two-in-a-row across them).

Usage::

    from kenning.resilience.phrases import phrase_for

    msg = phrase_for("brave_unavailable")
    if msg:
        speak(msg)  # voice-pipeline narration

Returns ``None`` if the configured pool is empty for that mode (the
operator silenced narration for that failure).
"""

from __future__ import annotations

import random
import threading
from typing import Dict, List, Optional

from kenning.config import get_config
from kenning.utils.logging import get_logger

logger = get_logger("resilience.phrases")


class _PhraseSource:
    """Round-robin shuffled selector. Cached per failure mode."""

    def __init__(self, phrases: List[str]) -> None:
        self._pool = list(phrases)
        self._lock = threading.Lock()
        self._cycle: List[str] = []

    def next(self) -> Optional[str]:
        if not self._pool:
            return None
        with self._lock:
            if not self._cycle:
                self._cycle = list(self._pool)
                random.shuffle(self._cycle)
            return self._cycle.pop()


_SOURCES: Dict[str, _PhraseSource] = {}
_SOURCES_LOCK = threading.Lock()


def phrase_for(failure_mode: str) -> Optional[str]:
    """Return one phrase for ``failure_mode``, or ``None`` if the pool is empty.

    ``failure_mode`` is the attribute name on
    :class:`ErrorPhrasesConfig` — e.g. ``"brave_unavailable"``,
    ``"qdrant_unavailable"``, ``"rvc_unavailable"``.
    """
    cfg = get_config().error_phrases
    pool = getattr(cfg, failure_mode, None)
    if pool is None:
        # Unknown failure mode -> log and bail. Don't crash the caller.
        logger.warning(
            "phrase_for: unknown failure_mode %r; no narration emitted",
            failure_mode,
        )
        return None
    if not pool:
        return None

    with _SOURCES_LOCK:
        src = _SOURCES.get(failure_mode)
        # If config was hot-reloaded with new phrases, the pool reference
        # diverges; rebuild the source.
        if src is None or src._pool != list(pool):
            src = _PhraseSource(list(pool))
            _SOURCES[failure_mode] = src
    return src.next()


def reset_phrase_cache() -> None:
    """Test-only: clear cached sources so a new test gets a fresh shuffle."""
    with _SOURCES_LOCK:
        _SOURCES.clear()


__all__ = ["phrase_for", "reset_phrase_cache"]
