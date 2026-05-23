"""Architect-plan TTS narrator with sentence-boundary barge-in window.

Catalog T5 Phase 2 (2026-05-22 batch 14). After the architect
supervisor produces a plan (see ``architect_supervisor.py``), the
orchestrator may optionally narrate the plan aloud via the configured
TTS engine BEFORE dispatching the editor LLM. Between sentences the
narrator polls a caller-supplied ``should_stop`` callback so the user
can interrupt the narration with a wake-word or follow-up utterance.

Design constraints:

* **Sentence-boundary granularity.** TTS engines do their best work
  on sentence-sized clips (cadence, prosody, edge fades). Splitting
  on sentence boundaries gives the user a natural barge-in window
  every 1-3 seconds.
* **Fail-open.** Any narration failure (TTS error, splitter edge
  case, etc.) returns ``NarrationResult(completed=False)`` so the
  caller proceeds straight to dispatch instead of aborting.
* **No mutation of the plan.** The narrator NEVER trims the plan
  string before passing it to the editor LLM — the
  ``narrate_max_chars`` cap only bounds what's SPOKEN, not what's
  forwarded.

Public surface:

* :class:`NarrationResult` — frozen telemetry dataclass.
* :func:`split_into_sentences` — splitter used by the narrator.
* :class:`ArchitectNarrator` — instantiable class.
* :func:`narrate_plan` — convenience function for one-shot use.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol


logger = logging.getLogger("ultron.coding.architect_narrator")


# ---------------------------------------------------------------------------
# TTS protocol
# ---------------------------------------------------------------------------


class _SpeakableTTS(Protocol):
    """Minimal protocol every TTS engine in :mod:`ultron.tts` satisfies.

    The narrator uses synchronous ``speak`` so it doesn't have to
    juggle the producer-consumer queue; this is fine for the
    architect path which is off the voice hot path (it's a
    dispatch-time, not a real-time, surface).
    """

    def speak(self, text: str) -> None: ...

    def stop(self) -> None: ...


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NarrationResult:
    """Outcome of one narration pass.

    Attributes:
        completed: True when every selected sentence reached TTS without
            interruption or error.
        interrupted: True when ``should_stop()`` returned True between
            sentences. Mutually exclusive with ``completed`` in practice
            because an interruption short-circuits the loop.
        sentences_spoken: How many sentences actually reached the TTS
            engine. Includes the in-flight sentence at interruption
            time (since by the time we noticed, it was already spoken).
        chars_spoken: Total chars handed to the TTS engine. Useful for
            tuning the ``narrate_max_chars`` cap against real plans.
        elapsed_seconds: Wall-clock duration of the narration pass.
        error: Empty when the call ran cleanly; otherwise a short
            failure reason. Combined with ``completed=False`` this
            tells the caller to fall through to dispatch without
            narration.
    """

    completed: bool
    interrupted: bool = False
    sentences_spoken: int = 0
    chars_spoken: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------


# Conservative sentence splitter: end-of-line + standard punctuation
# stops. Tuned to match the splitter in :mod:`ultron.tts.kokoro_engine`'s
# flush logic so the narrator chunks the plan exactly the way the
# producer-consumer pipeline would chunk a streaming LLM response.
# Avoids splitting on decimals (3.14), ellipses (...), and abbreviated
# titles (e.g. "Mr.") via small lookahead heuristics.
_SENTENCE_END_RE = re.compile(
    r"""
    (?<![A-Z])              # not after a single capital letter (initial like "U.")
    [.!?]                   # sentence terminator
    (?:[\"\)\]]+)?          # optional closing quote/bracket
    (?=\s+(?:[A-Z0-9\(\"\[]|$)|\n|$)  # space-then-(capital|digit) or newline or EOS
    """,
    re.VERBOSE,
)


def split_into_sentences(text: str) -> List[str]:
    """Split ``text`` into sentence-sized chunks for TTS narration.

    The splitter is conservative -- it prefers leaving a too-long
    chunk over creating fragments that play awkwardly. Returns a list
    of stripped non-empty strings; an empty / whitespace-only input
    returns ``[]``.
    """
    if not text or not text.strip():
        return []
    # Find every sentence-terminator position and split there.
    chunks: List[str] = []
    last = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        chunk = text[last:end].strip()
        if chunk:
            chunks.append(chunk)
        last = end
    tail = text[last:].strip()
    if tail:
        chunks.append(tail)
    return chunks


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------


class ArchitectNarrator:
    """Speaks an architect plan with a per-sentence barge-in window.

    Args:
        tts: TTS engine instance from :mod:`ultron.tts`. Any object
            exposing ``speak(text)`` + ``stop()`` works.
        max_chars: Cap on characters spoken before the narrator gives
            up and returns ``completed=False, error="max_chars_exceeded"``.
            Pass 0 to disable the cap (the full plan will be spoken).
        inter_sentence_pause_seconds: Sleep this long between sentences
            so a wake-word interrupt has time to register before the
            next sentence starts playing.
    """

    def __init__(
        self,
        tts: _SpeakableTTS,
        *,
        max_chars: int = 400,
        inter_sentence_pause_seconds: float = 0.12,
    ) -> None:
        if max_chars < 0:
            raise ValueError("max_chars must be >= 0 (0 = no cap)")
        if inter_sentence_pause_seconds < 0:
            raise ValueError("inter_sentence_pause_seconds must be >= 0")
        self._tts = tts
        self._max_chars = max_chars
        self._pause = inter_sentence_pause_seconds

    def narrate(
        self,
        plan_text: str,
        *,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> NarrationResult:
        """Speak ``plan_text`` sentence-by-sentence.

        ``should_stop`` is polled BEFORE every sentence (including the
        first). When it returns True, the narrator calls
        ``self._tts.stop()`` to interrupt any in-flight playback and
        returns ``interrupted=True``.

        Args:
            plan_text: The architect's prose plan. ``None`` or empty
                returns ``completed=False, error="empty plan"``.
            should_stop: Caller-supplied predicate. Called once before
                every sentence. ``None`` disables barge-in (still safe
                to call -- the loop runs to completion).

        Returns:
            :class:`NarrationResult` with telemetry. Never raises;
            errors fall through to ``completed=False`` so the caller
            proceeds to dispatch without narration.
        """
        if not plan_text or not plan_text.strip():
            return NarrationResult(completed=False, error="empty plan")

        sentences = split_into_sentences(plan_text)
        if not sentences:
            return NarrationResult(completed=False, error="no sentences after split")

        t0 = time.monotonic()
        spoken = 0
        chars = 0
        for sentence in sentences:
            # Barge-in check BEFORE every sentence.
            if should_stop is not None:
                try:
                    if should_stop():
                        try:
                            self._tts.stop()
                        except Exception as exc:                  # noqa: BLE001
                            logger.debug("narrator: tts.stop() raised: %s", exc)
                        return NarrationResult(
                            completed=False,
                            interrupted=True,
                            sentences_spoken=spoken,
                            chars_spoken=chars,
                            elapsed_seconds=time.monotonic() - t0,
                        )
                except Exception as exc:                          # noqa: BLE001
                    logger.warning(
                        "narrator: should_stop callback raised (%s); "
                        "treating as no-interrupt.", exc,
                    )
            # Char cap check.
            if self._max_chars > 0 and chars + len(sentence) > self._max_chars:
                logger.debug(
                    "narrator: max_chars cap (%d) hit at sentence %d; "
                    "stopping narration.",
                    self._max_chars, spoken + 1,
                )
                return NarrationResult(
                    completed=False,
                    sentences_spoken=spoken,
                    chars_spoken=chars,
                    elapsed_seconds=time.monotonic() - t0,
                    error="max_chars_exceeded",
                )
            # Speak.
            try:
                self._tts.speak(sentence)
            except Exception as exc:                              # noqa: BLE001
                logger.warning(
                    "narrator: tts.speak raised (%s); aborting narration.",
                    exc,
                )
                return NarrationResult(
                    completed=False,
                    sentences_spoken=spoken,
                    chars_spoken=chars,
                    elapsed_seconds=time.monotonic() - t0,
                    error=f"tts.speak raised: {exc}",
                )
            spoken += 1
            chars += len(sentence)
            if self._pause > 0 and spoken < len(sentences):
                time.sleep(self._pause)

        return NarrationResult(
            completed=True,
            sentences_spoken=spoken,
            chars_spoken=chars,
            elapsed_seconds=time.monotonic() - t0,
        )


def narrate_plan(
    plan_text: str,
    tts: _SpeakableTTS,
    *,
    should_stop: Optional[Callable[[], bool]] = None,
    max_chars: int = 400,
    inter_sentence_pause_seconds: float = 0.12,
) -> NarrationResult:
    """One-shot convenience wrapper around :class:`ArchitectNarrator`."""
    return ArchitectNarrator(
        tts,
        max_chars=max_chars,
        inter_sentence_pause_seconds=inter_sentence_pause_seconds,
    ).narrate(plan_text, should_stop=should_stop)


__all__ = [
    "ArchitectNarrator",
    "NarrationResult",
    "narrate_plan",
    "split_into_sentences",
]
