"""Pre-computed TTS clip cache for common ack phrases.

The voice-path latency budget on the conversational branch is dominated
by three stages: end-of-turn detection (~500-1200 ms), Whisper STT
(~890 ms), and the LLM TTFT + TTS first-chunk path (~140 ms + ~350 ms).
Filler-ack ("Mm.", "Right.", etc.) and the web-search ack
("Querying external sources.", etc.) are the FIRST audio the user
hears on most turns — and they're synthesised through the same XTTS
HTTP + v3 pedalboard filter chain as everything else, paying the full
~350-400 ms first-chunk cost on a phrase that never changes.

This module flips that cost to startup. Phrases are synthesised once
via the live TTS engine path (so the cached clip is bit-identical to
what the live path would produce — voice character preserved), then
served from a dict on subsequent calls. Net latency saving on every
cache-hit turn: ~350-400 ms.

Architecture:

    Orchestrator.__init__
        └── tts engine ready + warmed
            └── build_default_ack_clip_cache()
                └── prewarm_in_background(engine, phrases) -> daemon thread
                    └── engine.set_ack_cache(cache)

    engine._synthesize(text)
        ├── cache.get(text.strip()) -> Optional[(pcm, sr)]
        │       └── hit: return immediately, no HTTP, no filter
        └── miss: existing live-synth path

Failure modes are fail-open: missing engine, server unreachable
during prewarm, partial population, all leave the cache in whatever
state it reached and the live path picks up the rest.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ultron.utils.logging import get_logger

logger = get_logger("tts.precomputed_ack")


# Type alias matching the ``Clip = Tuple[ndarray, int]`` contract both
# engines (xtts_v3 + legacy speech.py) use. Re-declared here so this
# module has no import dependency on either engine.
Clip = Tuple[np.ndarray, int]


class PrecomputedAckClipCache:
    """Cache of pre-synthesised audio clips keyed by exact stripped text.

    Thread-safe: ``prewarm`` and ``get`` can run concurrently. A lookup
    during prewarm returns ``None`` for phrases that haven't been
    synthesised yet, falling back to the live path. The cache itself
    never blocks the live path — the worst case on a lookup race is a
    cache miss and a live synth.
    """

    def __init__(self, phrases: Sequence[str]) -> None:
        """Initialise an empty cache.

        Args:
            phrases: Phrases to pre-render at prewarm time. Stripped of
                leading/trailing whitespace; empty / whitespace-only
                entries are dropped; duplicates are de-duplicated.
                Order is not preserved — internally the unique set is
                sorted for deterministic logs.
        """
        unique = sorted({p.strip() for p in phrases if p and p.strip()})
        self._phrases: Tuple[str, ...] = tuple(unique)
        self._clips: Dict[str, Clip] = {}
        self._lock = threading.Lock()
        self._warmed_count = 0

    @property
    def phrases(self) -> Tuple[str, ...]:
        """The unique stripped phrases this cache will populate.

        Order is sorted-ascending — deterministic for log audit.
        """
        return self._phrases

    @property
    def warmed_count(self) -> int:
        """Number of phrases successfully cached so far."""
        with self._lock:
            return self._warmed_count

    def is_warm(self) -> bool:
        """True iff every phrase has been synthesised at least once."""
        with self._lock:
            return self._warmed_count >= len(self._phrases)

    def get(self, text: str) -> Optional[Clip]:
        """Return cached audio for ``text``, or None on miss.

        Lookup is exact-match on the stripped text. Caller MUST
        strip with the same convention used at prewarm time
        (`.strip()`, no other normalisation). Mismatched whitespace
        or punctuation = miss = live synth — no harm beyond the
        missed optimisation.
        """
        key = (text or "").strip()
        if not key:
            return None
        with self._lock:
            return self._clips.get(key)

    def prewarm(self, synth_fn: Callable[[str], Clip]) -> int:
        """Synthesise every phrase via ``synth_fn`` and store the result.

        ``synth_fn`` must return ``(pcm, sample_rate)`` with the same
        processing the live path applies — for XTTS that's the HTTP
        synth + phantom-tail trim + v3 filter chain. The cached clip
        is what the live path would have produced bit-for-bit on
        first synth; using a different synth path would skew the
        cached audio away from the live path and break the voice
        character contract.

        Returns the count of phrases successfully cached. Synthesis
        failures (raise) are logged at WARN level; the cache continues
        with whatever phrases did succeed. Idempotent: re-running
        re-synthesises every phrase (useful after a TTS server
        restart).
        """
        logger.info(
            "PrecomputedAckClipCache: prewarming %d phrases",
            len(self._phrases),
        )
        succeeded = 0
        for phrase in self._phrases:
            try:
                pcm, sr = synth_fn(phrase)
            except Exception as e:
                logger.warning(
                    "PrecomputedAckClipCache: synth failed for %r (%s) -- "
                    "live path will handle it",
                    phrase, e,
                )
                continue
            if pcm is None or pcm.size == 0:
                logger.warning(
                    "PrecomputedAckClipCache: empty clip for %r -- skipping",
                    phrase,
                )
                continue
            with self._lock:
                self._clips[phrase] = (pcm, sr)
                self._warmed_count += 1
                succeeded += 1
            logger.debug(
                "PrecomputedAckClipCache: cached %r (%.2fs @ %d Hz)",
                phrase, pcm.shape[0] / max(sr, 1), sr,
            )
        logger.info(
            "PrecomputedAckClipCache: %d/%d phrases cached",
            self._warmed_count, len(self._phrases),
        )
        return succeeded


def collect_default_ack_phrases() -> List[str]:
    """Return the union of the conversational + web-search ack pools.

    Imports are inside the function so this module stays importable
    even when ``conversational_ack`` / ``web_search`` aren't loaded
    (e.g. in narrow unit-test environments).
    """
    phrases: List[str] = []
    try:
        from ultron.conversational_ack import _CONVERSATIONAL_PHRASES
        phrases.extend(_CONVERSATIONAL_PHRASES)
    except Exception as e:                                       # noqa: BLE001
        logger.warning(
            "Could not import conversational ack phrases (%s); "
            "cache will only cover web-search acks", e,
        )
    try:
        from ultron.web_search.acknowledgments import _PHRASES as _WEB_ACK_PHRASES
        phrases.extend(_WEB_ACK_PHRASES)
    except Exception as e:                                       # noqa: BLE001
        logger.warning(
            "Could not import web-search ack phrases (%s); "
            "cache will only cover conversational acks", e,
        )
    return phrases


def build_default_ack_clip_cache() -> PrecomputedAckClipCache:
    """Build an empty cache covering the standard ack phrase pools.

    The returned cache is NOT yet populated. The caller must run
    ``cache.prewarm(synth_fn)`` (typically on a daemon thread after
    the TTS engine is up) to actually synthesise the clips.
    """
    return PrecomputedAckClipCache(collect_default_ack_phrases())


def prewarm_in_background(
    cache: PrecomputedAckClipCache,
    synth_fn: Callable[[str], Clip],
    *,
    name: str = "ack-prewarm",
) -> threading.Thread:
    """Kick off ``cache.prewarm(synth_fn)`` on a daemon thread.

    The thread is returned so callers can ``join()`` if they want to
    wait — typically the orchestrator does NOT wait, letting the cache
    warm up while the first wake-word capture is in progress. The
    first turn may still hit the live synth path (cache not warm
    yet); the SECOND turn and beyond benefit.

    Returns the started thread.
    """
    thread = threading.Thread(
        target=cache.prewarm,
        args=(synth_fn,),
        daemon=True,
        name=name,
    )
    thread.start()
    return thread
