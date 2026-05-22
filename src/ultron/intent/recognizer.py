"""Engine-agnostic intent recognizer.

Wraps ``moonshine_voice.IntentRecognizer`` so the orchestrator can
short-circuit common voice commands to local handlers without an LLM
roundtrip. Uses the bundled Gemma-300M embedding model (q4 = ~300 MB
on CPU) and cosine similarity over registered canonical phrases.

Engine-agnostic: the recognizer accepts arbitrary transcript text via
:meth:`process_utterance`. It runs identically whether the producing
STT was Moonshine, Parakeet, Whisper, or typed input.

Lifecycle:

- Constructed lazily by the orchestrator when ``intent.enabled=true``.
- The Gemma-300M model is downloaded by ``moonshine_voice`` on first
  use into the user's HF cache (~300 MB q4) and re-used across
  sessions.
- Module-level singleton accessible via :func:`get_intent_recognizer`
  / :func:`set_intent_recognizer` (mirrors the :mod:`ultron.desktop.vlm`
  pattern for cross-component access).

Failure modes (all log WARN, never raise to the voice loop):

- ``moonshine_voice`` not installed -> :attr:`is_available` returns
  False; calls are no-ops.
- Embedding model download fails -> same.
- Native lib call returns error -> :meth:`process_utterance` returns
  None and the orchestrator falls through to the LLM path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ultron.utils.logging import get_logger

logger = get_logger("intent.recognizer")


# Callable signature for intent handlers. Same shape as
# :class:`moonshine_voice.intent_recognizer.IntentHandler` so we can
# pass them through.
IntentHandler = Callable[[str, str, float], None]


@dataclass(frozen=True)
class IntentMatch:
    """A single recognized intent.

    Attributes:
        canonical_phrase: The registered trigger phrase that matched.
        utterance: The raw transcript text that was processed.
        similarity: Cosine similarity between the utterance embedding
            and the canonical-phrase embedding (range ~0.0-1.0).
    """

    canonical_phrase: str
    utterance: str
    similarity: float


@dataclass
class IntentRegistration:
    """One registered intent. Mutable so callers can edit handlers
    without re-registering."""

    canonical_phrase: str
    handler: Optional[IntentHandler] = None
    priority: int = 0


# ----------------------------------------------------------------------
# Recognizer
# ----------------------------------------------------------------------


class UltronIntentRecognizer:
    """Engine-agnostic intent matcher.

    Wraps ``moonshine_voice.IntentRecognizer`` with Ultron-friendly
    semantics:

    - Lazy load (no work at construction; first use triggers the
      embedding-model download + native handle creation).
    - Fail-open on missing library / failed load -- :attr:`is_available`
      reports False, all methods become no-ops.
    - Returns :class:`IntentMatch` explicitly from
      :meth:`process_utterance` so callers can dispatch their own
      handlers (in addition to the registered ones).
    - Thread-safe; the C library handle is guarded by a lock.

    Args:
        model_name: Embedding model name. Default ``"embeddinggemma-300m"``
            (the only one currently supported by ``moonshine_voice``).
        variant: Quantization variant. Default ``"q4"`` (~300 MB);
            other options ``"q8"``, ``"fp16"``, ``"fp32"``, ``"q4f16"``.
        threshold: Minimum cosine similarity for
            :meth:`process_utterance` to consider an utterance a match.
            Default 0.8 mirrors the moonshine_voice default.
    """

    def __init__(
        self,
        *,
        model_name: str = "embeddinggemma-300m",
        variant: str = "q4",
        threshold: float = 0.8,
    ) -> None:
        self.model_name = model_name
        self.variant = variant
        self._threshold = float(threshold)
        self._registrations: Dict[str, IntentRegistration] = {}
        self._handle = None  # underlying moonshine_voice.IntentRecognizer
        self._load_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._loaded = False
        self._load_failed = False
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Status / lifecycle
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if the recognizer can attempt loads.

        Returns False after a prior load failure (we cache the failure
        so we don't retry the slow native-lib import on every call).
        Returns True before the first load attempt (optimistic).
        """
        return not self._load_failed

    @property
    def loaded(self) -> bool:
        """True iff the native handle has been constructed successfully."""
        return self._loaded and self._handle is not None

    @property
    def threshold(self) -> float:
        """Minimum similarity for a match in :meth:`process_utterance`."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)
        if self.loaded:
            with self._call_lock:
                try:
                    self._handle.threshold = self._threshold
                except Exception as e:                            # noqa: BLE001
                    logger.warning("intent: threshold setter failed: %s", e)

    def ensure_loaded(self) -> bool:
        """Force the lazy load now. Returns True on success.

        Useful for warmup paths -- avoids the first-utterance latency
        spike that the lazy load would cause.
        """
        if self._loaded:
            return True
        if self._load_failed:
            return False
        return self._do_load()

    def _do_load(self) -> bool:
        with self._load_lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                from moonshine_voice.intent_recognizer import IntentRecognizer
                from moonshine_voice.download import get_embedding_model
            except ImportError as e:
                self._load_failed = True
                self._load_error = f"moonshine_voice not installed: {e}"
                logger.warning("intent: %s", self._load_error)
                return False
            try:
                model_path, model_arch = get_embedding_model(
                    self.model_name, self.variant,
                )
            except Exception as e:                                # noqa: BLE001
                self._load_failed = True
                self._load_error = (
                    f"embedding model download failed: {e}"
                )
                logger.warning("intent: %s", self._load_error)
                return False
            try:
                self._handle = IntentRecognizer(
                    model_path=model_path,
                    model_arch=model_arch,
                    model_variant=self.variant,
                    threshold=self._threshold,
                )
                # Re-register any phrases that were registered before
                # the load completed (the orchestrator wires intents
                # during init, the load runs on first utterance).
                for reg in self._registrations.values():
                    self._handle.register_intent(
                        reg.canonical_phrase,
                        handler=None,  # we dispatch handlers ourselves
                        priority=reg.priority,
                    )
                self._loaded = True
                logger.info(
                    "intent recognizer loaded (model=%s variant=%s, "
                    "phrases=%d, threshold=%.2f)",
                    self.model_name, self.variant,
                    len(self._registrations), self._threshold,
                )
                return True
            except Exception as e:                                # noqa: BLE001
                self._load_failed = True
                self._load_error = f"native lib init failed: {e}"
                logger.warning("intent: %s", self._load_error)
                return False

    def close(self) -> None:
        """Release the native handle. Idempotent."""
        with self._load_lock:
            handle = self._handle
            self._handle = None
            self._loaded = False
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register(
        self,
        canonical_phrase: str,
        *,
        handler: Optional[IntentHandler] = None,
        priority: int = 0,
    ) -> None:
        """Register a canonical phrase for matching.

        Args:
            canonical_phrase: The phrase users would naturally say.
                The recognizer computes its Gemma-300M embedding once
                at registration time and stores it for similarity
                lookups.
            handler: Optional callback fired when this phrase matches
                in :meth:`process_utterance`. Signature
                ``(canonical_phrase, utterance, similarity) -> None``.
                Pass ``None`` if the orchestrator dispatches based on
                the returned :class:`IntentMatch` instead.
            priority: Higher-priority intents win ties / are preferred
                in ranked output (see ``moonshine_voice`` docs).
        """
        if not canonical_phrase or not canonical_phrase.strip():
            raise ValueError("canonical_phrase must be non-empty")
        reg = IntentRegistration(
            canonical_phrase=canonical_phrase,
            handler=handler,
            priority=int(priority),
        )
        self._registrations[canonical_phrase] = reg
        if self.loaded:
            with self._call_lock:
                try:
                    self._handle.register_intent(
                        canonical_phrase,
                        handler=None,
                        priority=priority,
                    )
                except Exception as e:                            # noqa: BLE001
                    logger.warning(
                        "intent: register_intent(%r) failed: %s",
                        canonical_phrase, e,
                    )

    def unregister(self, canonical_phrase: str) -> bool:
        """Remove a previously-registered phrase.

        Returns True if the phrase was present, False otherwise.
        """
        was_present = canonical_phrase in self._registrations
        self._registrations.pop(canonical_phrase, None)
        if self.loaded:
            with self._call_lock:
                try:
                    self._handle.unregister_intent(canonical_phrase)
                except Exception as e:                            # noqa: BLE001
                    logger.warning(
                        "intent: unregister_intent(%r) failed: %s",
                        canonical_phrase, e,
                    )
        return was_present

    @property
    def registered_phrases(self) -> List[str]:
        """Snapshot of currently registered canonical phrases."""
        return list(self._registrations.keys())

    def clear(self) -> None:
        """Remove all registered phrases. Idempotent."""
        self._registrations.clear()
        if self.loaded:
            with self._call_lock:
                try:
                    self._handle.clear_intents()
                except Exception as e:                            # noqa: BLE001
                    logger.warning("intent: clear_intents failed: %s", e)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def process_utterance(self, utterance: str) -> Optional[IntentMatch]:
        """Match ``utterance`` against registered intents.

        Performs the lazy load on first call. Returns the best match
        above :attr:`threshold`, or ``None`` if no match qualifies (or
        if the recognizer is unavailable). Also fires the registered
        handler for the best match if one was provided.

        Args:
            utterance: Transcript text from any STT engine. Whitespace
                is preserved; case-insensitivity is the embedding
                model's responsibility.

        Returns:
            Best :class:`IntentMatch` above threshold, or ``None``.
        """
        if not utterance or not utterance.strip():
            return None
        if not self._loaded:
            if not self._do_load():
                return None
        if not self._registrations:
            return None
        with self._call_lock:
            try:
                matches = self._handle.get_closest_intents(
                    utterance, self._threshold,
                )
            except Exception as e:                                # noqa: BLE001
                logger.warning(
                    "intent: get_closest_intents failed (%s); "
                    "falling through", e,
                )
                return None
        if not matches:
            return None
        top = matches[0]
        canonical = top.canonical_phrase
        match = IntentMatch(
            canonical_phrase=canonical,
            utterance=utterance,
            similarity=float(top.similarity),
        )
        reg = self._registrations.get(canonical)
        if reg is not None and reg.handler is not None:
            try:
                reg.handler(canonical, utterance, match.similarity)
            except Exception as e:                                # noqa: BLE001
                logger.warning(
                    "intent: handler for %r raised: %s",
                    canonical, e,
                )
        return match

    def get_top_matches(
        self, utterance: str, *, n: int = 5,
        threshold: Optional[float] = None,
    ) -> List[IntentMatch]:
        """Return up to ``n`` ranked matches at or above ``threshold``.

        Useful for diagnostics and for callers that want to see the
        full ranking, not just the top hit.
        """
        if not utterance or not utterance.strip():
            return []
        if not self._loaded:
            if not self._do_load():
                return []
        if not self._registrations:
            return []
        thresh = threshold if threshold is not None else self._threshold
        with self._call_lock:
            try:
                matches = self._handle.get_closest_intents(utterance, thresh)
            except Exception as e:                                # noqa: BLE001
                logger.warning(
                    "intent: get_closest_intents failed (%s)", e,
                )
                return []
        out: List[IntentMatch] = []
        for m in matches[:n]:
            out.append(IntentMatch(
                canonical_phrase=m.canonical_phrase,
                utterance=utterance,
                similarity=float(m.similarity),
            ))
        return out


# ----------------------------------------------------------------------
# Module-level singleton (mirrors ultron.desktop.vlm pattern)
# ----------------------------------------------------------------------


_singleton: Optional[UltronIntentRecognizer] = None
_singleton_lock = threading.Lock()


def get_intent_recognizer() -> Optional[UltronIntentRecognizer]:
    """Return the process-wide recognizer or None if not yet set."""
    return _singleton


def set_intent_recognizer(r: Optional[UltronIntentRecognizer]) -> None:
    """Install (or detach) the process-wide recognizer."""
    global _singleton
    with _singleton_lock:
        _singleton = r
