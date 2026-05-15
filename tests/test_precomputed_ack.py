"""Tests for the pre-computed TTS clip cache.

Covers:
- Cache construction (de-dup, strip, sort, empty filter)
- Lookup semantics (exact match, strip, miss returns None)
- Prewarm population (success / synth-raises / empty-clip / partial)
- Thread safety (concurrent prewarm + get)
- Default phrase pool collector
- Background prewarm thread starts and joinable
"""

from __future__ import annotations

import threading
import time
from typing import List, Tuple

import numpy as np
import pytest

from ultron.tts.precomputed_ack import (
    PrecomputedAckClipCache,
    build_default_ack_clip_cache,
    collect_default_ack_phrases,
    prewarm_in_background,
)


# Helpers ------------------------------------------------------------


def _mk_clip(samples: int = 100, sr: int = 24000) -> Tuple[np.ndarray, int]:
    """Build a non-empty fake clip for tests."""
    return np.ones(samples, dtype=np.int16), sr


def _silent_clip(sr: int = 24000) -> Tuple[np.ndarray, int]:
    """Return an empty (size-0) clip — represents a failed synth."""
    return np.zeros(0, dtype=np.int16), sr


# Construction -------------------------------------------------------


class TestConstruction:
    def test_phrases_deduped(self) -> None:
        cache = PrecomputedAckClipCache(["Hello.", "Hello.", "World."])
        assert cache.phrases == ("Hello.", "World.")

    def test_phrases_stripped(self) -> None:
        cache = PrecomputedAckClipCache(["  Hello.  ", "World.\n"])
        assert cache.phrases == ("Hello.", "World.")

    def test_phrases_sorted(self) -> None:
        cache = PrecomputedAckClipCache(["Zoo.", "Alpha.", "Mango."])
        assert cache.phrases == ("Alpha.", "Mango.", "Zoo.")

    def test_empty_and_whitespace_dropped(self) -> None:
        cache = PrecomputedAckClipCache(["Hello.", "", "   ", "World."])
        assert cache.phrases == ("Hello.", "World.")

    def test_none_safe_in_iter(self) -> None:
        # The signature is Sequence[str] but the implementation defensively
        # filters falsy values, so a list containing a None survives.
        cache = PrecomputedAckClipCache(["Hello.", None, "World."])  # type: ignore[list-item]
        assert cache.phrases == ("Hello.", "World.")

    def test_starts_empty(self) -> None:
        cache = PrecomputedAckClipCache(["Hello."])
        assert cache.warmed_count == 0
        assert cache.is_warm() is False
        assert cache.get("Hello.") is None


# Lookup --------------------------------------------------------------


class TestLookup:
    def test_get_miss_returns_none(self) -> None:
        cache = PrecomputedAckClipCache(["Hello."])
        assert cache.get("Hello.") is None  # never warmed

    def test_get_after_manual_warm_hits(self) -> None:
        cache = PrecomputedAckClipCache(["Hello."])
        # Manual populate via prewarm
        cache.prewarm(lambda _t: _mk_clip(100, 24000))
        pcm, sr = cache.get("Hello.")
        assert pcm.size == 100
        assert sr == 24000

    def test_get_strips_input(self) -> None:
        cache = PrecomputedAckClipCache(["Hello."])
        cache.prewarm(lambda _t: _mk_clip(50, 24000))
        # Whitespace variants all hit the same entry.
        assert cache.get("  Hello.  ") is not None
        assert cache.get("\nHello.\t") is not None
        assert cache.get("Hello.") is not None

    def test_get_empty_returns_none(self) -> None:
        cache = PrecomputedAckClipCache(["Hello."])
        cache.prewarm(lambda _t: _mk_clip())
        assert cache.get("") is None
        assert cache.get("   ") is None
        assert cache.get(None) is None  # type: ignore[arg-type]

    def test_get_other_phrase_miss(self) -> None:
        cache = PrecomputedAckClipCache(["Hello."])
        cache.prewarm(lambda _t: _mk_clip())
        # Different phrase even with same prefix -- miss.
        assert cache.get("Hello, world.") is None
        assert cache.get("Hello") is None  # no period


# Prewarm semantics --------------------------------------------------


class TestPrewarm:
    def test_prewarm_populates_all(self) -> None:
        cache = PrecomputedAckClipCache(["A.", "B.", "C."])
        synth_calls: List[str] = []

        def synth(text: str):
            synth_calls.append(text)
            return _mk_clip(samples=len(text), sr=24000)

        n = cache.prewarm(synth)
        assert n == 3
        assert cache.is_warm() is True
        assert sorted(synth_calls) == ["A.", "B.", "C."]
        # Each cached entry distinct (by samples).
        a_pcm, _ = cache.get("A.")
        b_pcm, _ = cache.get("B.")
        c_pcm, _ = cache.get("C.")
        assert a_pcm.size == 2
        assert b_pcm.size == 2
        assert c_pcm.size == 2

    def test_prewarm_returns_count(self) -> None:
        cache = PrecomputedAckClipCache(["A.", "B.", "C."])
        n = cache.prewarm(lambda _t: _mk_clip())
        assert n == 3

    def test_prewarm_skips_empty_clip(self) -> None:
        cache = PrecomputedAckClipCache(["A.", "B."])

        def synth(text: str):
            return _silent_clip() if text == "A." else _mk_clip()

        n = cache.prewarm(synth)
        assert n == 1
        assert cache.is_warm() is False
        assert cache.get("A.") is None
        assert cache.get("B.") is not None

    def test_prewarm_swallows_synth_exception(self) -> None:
        cache = PrecomputedAckClipCache(["A.", "B."])

        def synth(text: str):
            if text == "A.":
                raise RuntimeError("server unreachable")
            return _mk_clip()

        n = cache.prewarm(synth)
        assert n == 1
        assert cache.get("A.") is None
        assert cache.get("B.") is not None

    def test_prewarm_partial_does_not_break_subsequent(self) -> None:
        cache = PrecomputedAckClipCache(["A.", "B.", "C."])

        def synth(text: str):
            if text == "B.":
                raise RuntimeError("transient")
            return _mk_clip()

        n = cache.prewarm(synth)
        assert n == 2
        # A and C cached; B is on the live path.
        assert cache.get("A.") is not None
        assert cache.get("B.") is None
        assert cache.get("C.") is not None

    def test_prewarm_idempotent(self) -> None:
        cache = PrecomputedAckClipCache(["A.", "B."])
        cache.prewarm(lambda _t: _mk_clip())
        # Second run re-synthesises every phrase. Warmed count saturates
        # at len(phrases) but the count goes up each time (the counter
        # is post-success, not unique-set).
        first = cache.warmed_count
        cache.prewarm(lambda _t: _mk_clip())
        assert cache.warmed_count == first + 2

    def test_prewarm_empty_pool_returns_zero(self) -> None:
        cache = PrecomputedAckClipCache([])
        n = cache.prewarm(lambda _t: _mk_clip())
        assert n == 0
        assert cache.is_warm() is True  # vacuously


# Thread safety ------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_get_during_prewarm(self) -> None:
        """Lookups during prewarm must never raise. A miss on a
        not-yet-populated phrase IS valid (caller falls back to live
        synth); the contract is just that ``get()`` is thread-safe."""
        cache = PrecomputedAckClipCache(["A.", "B.", "C.", "D.", "E."])

        def slow_synth(text: str):
            time.sleep(0.01)
            return _mk_clip()

        prewarm_thread = threading.Thread(
            target=cache.prewarm, args=(slow_synth,), daemon=True,
        )
        reader_errors: List[Exception] = []

        def reader():
            try:
                # Read every phrase 100x while prewarm runs. Mix of
                # eventually-cached and never-cached lookups.
                for _ in range(100):
                    for p in ["A.", "B.", "C.", "D.", "E.", "missing."]:
                        _ = cache.get(p)  # None or clip — both fine.
                    time.sleep(0.001)
            except Exception as e:  # pragma: no cover -- defensive
                reader_errors.append(e)

        readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
        prewarm_thread.start()
        for t in readers:
            t.start()
        prewarm_thread.join(timeout=5.0)
        for t in readers:
            t.join(timeout=5.0)

        # Thread safety: no exceptions during concurrent reads.
        assert reader_errors == []
        # Prewarm finished cleanly and all phrases ended up cached.
        assert cache.is_warm() is True
        for p in ("A.", "B.", "C.", "D.", "E."):
            assert cache.get(p) is not None, f"missing post-prewarm: {p!r}"
        assert cache.get("missing.") is None


# Default pool factory -----------------------------------------------


class TestDefaultPool:
    def test_collect_default_includes_conv_phrases(self) -> None:
        phrases = collect_default_ack_phrases()
        # The conversational pool has "Mm." etc.
        assert "Mm." in phrases
        assert "Right." in phrases

    def test_collect_default_includes_web_search_phrases(self) -> None:
        phrases = collect_default_ack_phrases()
        # The web-search pool has "Querying external sources." etc.
        assert "Querying external sources." in phrases
        assert "One moment." in phrases

    def test_build_default_cache_has_both_pools(self) -> None:
        cache = build_default_ack_clip_cache()
        # Spot-check both pools.
        assert "Mm." in cache.phrases
        assert "Querying external sources." in cache.phrases
        # And the cache is empty (not yet warmed).
        assert cache.warmed_count == 0
        assert cache.get("Mm.") is None


# Background prewarm -------------------------------------------------


class TestBackgroundPrewarm:
    def test_prewarm_in_background_returns_thread(self) -> None:
        cache = PrecomputedAckClipCache(["A.", "B."])
        t = prewarm_in_background(cache, lambda _t: _mk_clip())
        assert isinstance(t, threading.Thread)
        assert t.daemon is True
        t.join(timeout=3.0)
        assert not t.is_alive()

    def test_prewarm_in_background_populates(self) -> None:
        cache = PrecomputedAckClipCache(["A.", "B.", "C."])
        t = prewarm_in_background(cache, lambda _t: _mk_clip())
        t.join(timeout=3.0)
        assert cache.is_warm() is True
        assert cache.get("A.") is not None
        assert cache.get("B.") is not None
        assert cache.get("C.") is not None

    def test_prewarm_in_background_thread_name(self) -> None:
        cache = PrecomputedAckClipCache(["A."])
        t = prewarm_in_background(cache, lambda _t: _mk_clip(), name="custom-name")
        try:
            assert t.name == "custom-name"
        finally:
            t.join(timeout=3.0)
