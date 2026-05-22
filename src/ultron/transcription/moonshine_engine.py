"""Moonshine STT engine -- streaming + one-shot (2026-05-22 rewrite).

Drop-in replacement for :class:`WhisperEngine` -- same
``transcribe(audio: np.ndarray, language: Optional[str]) -> str``
interface -- PLUS an opt-in streaming protocol the orchestrator can
use to feed audio chunks live during capture and read partial
transcripts before speech-end fires.

Backed by ``moonshine-voice`` (the official Moonshine AI package) so
we get the v2 streaming model variants:

    ModelArch.TINY_STREAMING    --  26M params, lowest footprint
    ModelArch.BASE_STREAMING    -- larger, slightly better WER
    ModelArch.SMALL_STREAMING   -- mid-tier
    ModelArch.MEDIUM_STREAMING  -- 200M params, ~6.65% WER, the
                                   default for English -- beats
                                   Whisper Large V3 on the OpenASR
                                   leaderboard average. Default.

Non-streaming variants are also available (``ModelArch.TINY`` /
``ModelArch.BASE``) for the absolute-lowest-footprint case.

**Why Moonshine is fast on short clips:** Whisper pads everything to
30 s before encoding, so a 2 s clip costs the same as a 28 s clip.
Moonshine was trained on variable-length segments without padding --
runtime scales with actual audio length. Combined with the streaming
encoder, partial transcripts become available continuously as audio
arrives.

**Streaming protocol** (called from the orchestrator's capture loop):

    engine.start_stream()                       # on speech-start
    while capturing:
        engine.feed_audio(chunk_samples)        # 16 ms blocksize
        partial = engine.get_partial_text()     # peek any time
    final = engine.stop_stream()                # on speech-end

The one-shot ``transcribe(audio)`` method wraps the same machinery
for callers who don't want to think about streaming -- it submits
the whole buffer via ``transcribe_without_streaming`` for minimum
overhead.

**Easy reversibility** (the "variable switch"):
- ``stt.engine: whisper``  -> Whisper (faster-whisper / distil-small.en).
- ``stt.engine: moonshine`` -> this engine (streaming-native).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from config import settings
from ultron.errors import WhisperTranscriptionError
from ultron.resilience import get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("transcription.moonshine")


MOONSHINE_INSTALL_HINT = (
    "Moonshine requires the ``moonshine-voice`` package. Install with:\n"
    "    .venv/Scripts/pip install moonshine-voice\n"
    "(Bundled with platform-specific C++ binaries + ONNX runtime;\n"
    "no Keras / TF / PyTorch upgrade required.)\n"
    "\n"
    "Or revert to Whisper with stt.engine: whisper in config.yaml."
)


# Moonshine's audio-length sanity bounds. The v2 streaming variants
# don't have the 64 s ceiling that v1 used because the stream
# processes incrementally, but we still skip <100 ms clips upstream
# to avoid wasting the model's encoder context.
_MIN_AUDIO_SECONDS = 0.1


# Maps human-friendly model names (and the v1 ONNX names we used to
# expose via useful-moonshine-onnx) to v2 ModelArch enum values. The
# default for English is MEDIUM_STREAMING -- the headline 200M-param
# variant that beats Whisper Large V3 on the average WER benchmark.
_MODEL_ARCH_ALIASES = {
    # v2 streaming variants (recommended for live voice -- they are
    # the only ones that expose partial transcripts during capture):
    "medium-streaming-en": "MEDIUM_STREAMING",
    "small-streaming-en": "SMALL_STREAMING",
    "base-streaming-en": "BASE_STREAMING",
    "tiny-streaming-en": "TINY_STREAMING",
    # Non-streaming variants (lower footprint, single-shot only):
    "moonshine/base": "BASE",
    "moonshine/tiny": "TINY",
    "base": "BASE",
    "tiny": "TINY",
}


def is_moonshine_available() -> bool:
    """Return True iff the ``moonshine_voice`` package can be imported."""
    try:
        import importlib.util
        return importlib.util.find_spec("moonshine_voice") is not None
    except Exception:                                                  # noqa: BLE001
        return False


def _resolve_model(model_name: Optional[str], language: str = "en"):
    """Resolve a model name string to ``(model_path, ModelArch)``.

    Strategy:
    1. If ``model_name`` is in :data:`_MODEL_ARCH_ALIASES`, ask
       ``get_model_for_language`` to fetch that specific arch (handles
       caching + downloading streaming assets on first use).
    2. Otherwise, try parsing as a canonical arch name
       (``"MEDIUM_STREAMING"`` etc.).
    3. Final fallback: language default
       (``get_model_for_language(language)`` -- medium-streaming for
       English).
    """
    from moonshine_voice import (
        ModelArch, get_model_for_language, string_to_model_arch,
    )

    if model_name and model_name in _MODEL_ARCH_ALIASES:
        arch_name = _MODEL_ARCH_ALIASES[model_name]
        model_arch = getattr(ModelArch, arch_name)
        # get_model_for_language returns (path, arch). When the model
        # is already cached on disk this is fast.
        return get_model_for_language(language, model_arch)

    if model_name:
        try:
            model_arch = string_to_model_arch(model_name)
            return get_model_for_language(language, model_arch)
        except Exception:                                              # noqa: BLE001
            logger.warning(
                "Moonshine: unrecognised model_name %r; falling back to "
                "language default (%s).", model_name, language,
            )

    return get_model_for_language(language)


class _PartialTextCollector:
    """Listener that maintains a thread-safe dict of {line_id: text}.

    The orchestrator can read the current partial transcript at any
    time via :meth:`get_text`. Lines are kept in insertion order; new
    lines are appended on ``on_line_started`` and updated text replaces
    the cached entry on subsequent events.
    """

    def __init__(self):
        # Late import so a missing moonshine_voice doesn't blow up
        # the module-level definition for is_moonshine_available().
        from moonshine_voice.transcriber import TranscriptEventListener

        # ABC requires subclassing at class definition time; we attach
        # the methods imperatively to avoid a top-level import that
        # blocks the module from loading when moonshine_voice isn't
        # installed (the engine still exposes is_moonshine_available
        # which other code reads to decide whether to construct).
        self._listener_cls = TranscriptEventListener
        self._lock = threading.Lock()
        # Ordered dict keyed by line_id; values are the latest text.
        self._lines: Dict[int, str] = {}
        # Track completion so the final aggregate is stable.
        self._completed_ids: set[int] = set()
        # Latest in-flight latency report from the C++ core (for logs).
        self.last_latency_ms = 0
        # 2026-05-22 diag: tracks how many on_line_* callbacks fire per
        # streaming session, so we can detect "no events fired" -- the
        # failure mode where stop_stream returns empty even though
        # audio was fed.
        self.event_count = 0

    def _update_line(self, line) -> None:
        """Shared helper for every line-changed event."""
        with self._lock:
            text = line.text or ""
            self._lines[line.line_id] = text
            if getattr(line, "is_complete", False):
                self._completed_ids.add(line.line_id)
            latency = getattr(line, "last_transcription_latency_ms", 0)
            if latency:
                self.last_latency_ms = latency
            self.event_count += 1
            # Per-event DEBUG so we can verify listener wiring without
            # spamming production. Enable with logger level DEBUG to see.
            if logger.isEnabledFor(10):  # DEBUG
                logger.debug(
                    "Moonshine listener: line_id=%d is_complete=%s "
                    "text=%r",
                    line.line_id,
                    getattr(line, "is_complete", False),
                    text[:80],
                )

    def get_text(self, *, completed_only: bool = False) -> str:
        """Return the current accumulated transcript.

        Args:
            completed_only: When True, only lines marked ``is_complete``
                are joined -- safest for the final-result path. When
                False (default), all in-flight lines are joined --
                gives you the live partial.
        """
        with self._lock:
            if completed_only:
                items = [
                    (lid, self._lines[lid])
                    for lid in self._lines
                    if lid in self._completed_ids
                ]
            else:
                items = list(self._lines.items())
        items.sort(key=lambda kv: kv[0])
        return " ".join(t for _, t in items if t).strip()

    def reset(self) -> None:
        with self._lock:
            self._lines.clear()
            self._completed_ids.clear()
            self.last_latency_ms = 0
            self.event_count = 0


def _build_listener(collector: _PartialTextCollector):
    """Build a TranscriptEventListener bound to the collector.

    Returns an INSTANCE (not a class) so the C++ core can hold it.
    """
    from moonshine_voice.transcriber import TranscriptEventListener

    class _Listener(TranscriptEventListener):
        def on_line_started(self, event):                              # noqa: D401
            collector._update_line(event.line)

        def on_line_updated(self, event):                              # noqa: D401
            collector._update_line(event.line)

        def on_line_text_changed(self, event):                         # noqa: D401
            collector._update_line(event.line)

        def on_line_completed(self, event):                            # noqa: D401
            collector._update_line(event.line)

        def on_error(self, event):                                     # noqa: D401
            logger.warning(
                "Moonshine transcriber on_error: %s",
                getattr(event, "message", repr(event)),
            )

    return _Listener()


class MoonshineEngine:
    """Moonshine streaming + one-shot STT.

    Args:
        model_name: ``"medium-streaming-en"`` (default for English),
            ``"small-streaming-en"``, ``"base-streaming-en"``,
            ``"tiny-streaming-en"`` for streaming variants, or
            ``"moonshine/base"`` / ``"moonshine/tiny"`` for the
            non-streaming variants (slightly lower footprint, no live
            partials).
        device: kept for API parity; Moonshine runs on CPU via ONNX
            runtime (a future ``onnxruntime-gpu`` swap could honour
            cuda, not wired today).
        model_precision: kept for API parity with the prior
            ``useful-moonshine-onnx`` engine; ``moonshine-voice``
            handles precision via the model asset selection (the
            English default downloads as ``quantized``).
        update_interval_s: how often the C++ core emits a partial-
            transcript update during streaming. 0.2 s is the sweet
            spot for a voice agent -- frequent enough that the
            speculative LLM hand-off has fresh text, infrequent
            enough that the encoder isn't thrashing.

    Raises:
        ImportError: when ``moonshine_voice`` is not installed.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        model_precision: Optional[str] = None,
        update_interval_s: Optional[float] = None,
        language: str = "en",
    ) -> None:
        from ultron.config import get_config

        stt_cfg = get_config().stt

        self.model_name = model_name or getattr(
            stt_cfg, "moonshine_model", "medium-streaming-en",
        )
        self.device = device or getattr(stt_cfg, "moonshine_device", "cpu")
        if self.device.lower() != "cpu":
            logger.info(
                "Moonshine runs on CPU; ignoring device=%r and "
                "running on CPU.", self.device,
            )
        self.device = "cpu"
        self.model_precision = (
            model_precision
            or getattr(stt_cfg, "moonshine_precision", "float")
        )
        self.update_interval_s = float(
            update_interval_s
            if update_interval_s is not None
            else getattr(stt_cfg, "moonshine_update_interval_s", 0.2)
        )
        self.language = language

        if not is_moonshine_available():
            raise ImportError(MOONSHINE_INSTALL_HINT)

        logger.info(
            "Loading Moonshine '%s' (update_interval=%.2fs) on CPU...",
            self.model_name, self.update_interval_s,
        )
        t0 = time.monotonic()
        try:
            from moonshine_voice.transcriber import Transcriber

            model_path, model_arch = _resolve_model(
                self.model_name, language=self.language,
            )
            self._model_arch = model_arch
            self._transcriber = Transcriber(
                model_path=str(model_path),
                model_arch=model_arch,
                update_interval=self.update_interval_s,
            )
        except Exception as e:                                          # noqa: BLE001
            logger.error("Moonshine load failed: %s", e)
            raise

        # Wire a listener that maintains an in-memory partial transcript
        # the orchestrator can poll via :meth:`get_partial_text`.
        self._collector = _PartialTextCollector()
        self._transcriber.add_listener(_build_listener(self._collector))

        self._stream_active = False
        self._stream_lock = threading.Lock()
        # 2026-05-22 streaming integration: when the orchestrator's
        # capture loop drives the engine via start_stream / feed_audio
        # / stop_stream, ``stop_stream`` stores the final transcript
        # here so the orchestrator's subsequent ``transcribe(buffer)``
        # call returns the cached value instead of re-running the
        # model. Consumed (cleared) on first read.
        self._last_streaming_text: Optional[str] = None

        logger.info(
            "Moonshine ready in %.2fs (model_arch=%s, streaming_supported=%s)",
            time.monotonic() - t0,
            _arch_to_string(self._model_arch),
            self.supports_streaming(),
        )

    # ----------------------------------------------------------------
    # Streaming protocol (consumed by Orchestrator._capture_utterance).
    # ----------------------------------------------------------------

    def supports_streaming(self) -> bool:
        """Return True iff this model variant can emit partial
        transcripts during ``feed_audio`` calls.

        Only the ``*_STREAMING`` model arches expose partials. The
        non-streaming TINY / BASE variants must be used via the
        one-shot ``transcribe`` method instead.
        """
        try:
            return "streaming" in _arch_to_string(self._model_arch).lower()
        except Exception:                                              # noqa: BLE001
            return False

    def start_stream(self) -> None:
        """Begin streaming mode. Idempotent."""
        with self._stream_lock:
            if self._stream_active:
                return
            self._collector.reset()
            try:
                self._transcriber.start()
                self._stream_active = True
            except Exception as e:                                     # noqa: BLE001
                logger.warning("Moonshine start_stream failed: %s", e)

    def feed_audio(
        self, audio: np.ndarray, sample_rate: int = settings.SAMPLE_RATE,
    ) -> None:
        """Push an audio chunk to the streaming session.

        ``audio`` is a 1-D numpy array (mono float32). The C++ core
        accepts any iterable of floats; we pass the numpy array
        directly (no .tolist() needed).
        """
        if not self._stream_active:
            return
        if audio.size == 0:
            return
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        try:
            # The C++ core unpacks the iterable into a ctypes array,
            # so passing the numpy array works as long as it's
            # iterable + each element is float-castable.
            self._transcriber.add_audio(audio, sample_rate=sample_rate)
        except Exception as e:                                         # noqa: BLE001
            logger.warning("Moonshine feed_audio failed: %s", e)

    def get_partial_text(self, *, completed_only: bool = False) -> str:
        """Return the current accumulated transcript without stopping
        the stream. Safe to call any time after ``start_stream``."""
        return self._collector.get_text(completed_only=completed_only)

    def stop_stream(self) -> str:
        """Finalize streaming and return the full transcript. Idempotent.

        Stashes the result on ``self._last_streaming_text`` so a
        subsequent :meth:`transcribe` call (e.g. from the orchestrator's
        post-capture path) returns the cached value instead of running
        the model again.

        2026-05-22 stash semantics (the cache-miss path):
            - Streaming produced text -> stash it.
            - Streaming produced NOTHING (event_count == 0) -> stash
              ``None`` instead of "" so the post-capture
              ``transcribe(speech)`` falls through to a real one-shot
              transcribe on the audio buffer. Empty-from-listener and
              empty-from-no-events are different failure modes; the
              second means "we have audio, just no events yet" and the
              safe play is to re-run synchronously.
        """
        with self._stream_lock:
            if not self._stream_active:
                return self._last_streaming_text or ""
            try:
                # Force a final update before stopping so the partial
                # reflects any audio still in the encoder window.
                try:
                    self._transcriber.update_transcription()
                except Exception:                                       # noqa: BLE001
                    pass
                self._transcriber.stop()
            except Exception as e:                                     # noqa: BLE001
                logger.warning("Moonshine stop_stream failed: %s", e)
            finally:
                self._stream_active = False
        # Prefer completed lines for the final read; fall back to all
        # in-flight text if no lines completed (very short utterances).
        text = self._collector.get_text(completed_only=True)
        if not text:
            text = self._collector.get_text(completed_only=False)
        events = self._collector.event_count
        if events == 0 and not text:
            # Listener never fired during this session. Mark cache
            # miss so the downstream ``transcribe(buffer)`` call
            # re-runs synchronously on the full audio buffer instead
            # of returning a misleading empty string.
            logger.warning(
                "Moonshine stop_stream: 0 listener events fired during "
                "session. Cache miss -> post-capture transcribe will "
                "re-run synchronously on the buffer.",
            )
            self._last_streaming_text = None
            return ""
        self._last_streaming_text = text
        logger.info(
            "Moonshine stream finalized: %d events, %d chars",
            events, len(text),
        )
        return text

    # ----------------------------------------------------------------
    # One-shot transcribe (backward-compatible WhisperEngine API).
    # ----------------------------------------------------------------

    def transcribe(
        self, audio: np.ndarray, language: Optional[str] = "en",
    ) -> str:
        """Transcribe a complete audio buffer in one shot.

        Three cases (in priority order):
        1. The orchestrator's capture loop just called ``stop_stream``
           and stashed the final transcript -- return that cached text
           and clear the slot. This is the fast path on a normal turn.
        2. A streaming session is *currently active* (the speculative
           STT thread is calling us mid-capture). Return the current
           partial text via ``get_partial_text`` without stopping the
           stream; the main capture path still owns the lifecycle.
        3. No streaming activity. Run a synchronous one-shot
           transcribe: either ``transcribe_without_streaming`` on
           non-streaming arches, or a quick start/feed/stop cycle on
           streaming arches.
        """
        # Case 1: cached final-stream text waiting for pickup.
        cached = self._last_streaming_text
        if cached is not None:
            self._last_streaming_text = None
            return cached

        # Case 2: a streaming session is in flight -- the speculative
        # STT thread can read the live partial without disrupting the
        # main capture loop's session.
        if self._stream_active:
            try:
                self._transcriber.update_transcription()
            except Exception:                                          # noqa: BLE001
                pass
            return self.get_partial_text()

        # Case 3: fall through to a true one-shot transcribe.
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        num_seconds = audio.size / settings.SAMPLE_RATE
        if num_seconds < _MIN_AUDIO_SECONDS:
            logger.debug(
                "Moonshine: skipping %.3fs clip (below %.1fs minimum)",
                num_seconds, _MIN_AUDIO_SECONDS,
            )
            return ""

        t0 = time.monotonic()
        try:
            if self.supports_streaming():
                # Streaming models don't expose transcribe_without_streaming
                # directly -- run a synchronous start/feed/stop cycle.
                self.start_stream()
                self.feed_audio(audio)
                # Give the encoder one update tick to flush partials.
                try:
                    self._transcriber.update_transcription()
                except Exception:                                       # noqa: BLE001
                    pass
                text = self.stop_stream()
                # 2026-05-22 cache-leak fix: stop_stream stashes
                # ``_last_streaming_text`` for the orchestrator's
                # pre-ran-streaming + post-capture-transcribe pattern.
                # In case 3 (synchronous fresh transcribe) THIS call is
                # already consuming the result, so leaving a stash
                # would cause the NEXT transcribe call to return THIS
                # call's text as if it were its own. Clear the stash
                # explicitly so the next call's case 1 falls through.
                self._last_streaming_text = None
            else:
                # Non-streaming arch: use the single-shot API.
                transcript = self._transcriber.transcribe_without_streaming(
                    audio, sample_rate=settings.SAMPLE_RATE,
                )
                text = " ".join(
                    (line.text or "").strip()
                    for line in transcript.lines
                ).strip()
        except Exception as e:                                         # noqa: BLE001
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error(
                "Moonshine transcribe failed in %.0fms: %s "
                "(if this is recurring, swap to stt.engine: whisper)",
                elapsed_ms, e,
            )
            get_error_log().record(
                WhisperTranscriptionError(
                    f"Moonshine transcribe failed: {e}",
                    context={
                        "audio_seconds": num_seconds,
                        "model": self.model_name,
                        "model_arch": _arch_to_string(self._model_arch),
                        "engine": "moonshine",
                    },
                    recovery=(
                        "returned empty transcription; orchestrator "
                        "skips this turn. Operator: consider "
                        "``stt.engine: whisper`` to revert."
                    ),
                ),
                dependency="moonshine",
            )
            return ""

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Moonshine: %.2fs audio -> %d chars in %.0fms "
            "(RTF=%.3f, model=%s, last_partial_latency=%dms)",
            num_seconds, len(text), elapsed_ms,
            elapsed_ms / 1000 / max(num_seconds, 1e-6),
            _arch_to_string(self._model_arch),
            self._collector.last_latency_ms,
        )
        return text

    # ----------------------------------------------------------------
    # Lifecycle.
    # ----------------------------------------------------------------

    def warmup(self) -> None:
        """Trigger a tiny transcribe so the ONNX session and tokenizer
        finish JIT'ing before the first real turn. Saves ~100-300 ms
        on the cold path.
        """
        try:
            silence = np.zeros(
                int(0.5 * settings.SAMPLE_RATE), dtype=np.float32,
            )
            self.transcribe(silence)
            logger.info("Moonshine warmup complete")
        except Exception as e:                                         # noqa: BLE001
            logger.debug("Moonshine warmup skipped (%s)", e)

    def __enter__(self) -> "MoonshineEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._stream_active:
                self.stop_stream()
        except Exception:                                              # noqa: BLE001
            pass
        try:
            self._transcriber.close()
        except Exception:                                              # noqa: BLE001
            pass


def _arch_to_string(model_arch: Any) -> str:
    """Return a stable string label for a ``ModelArch`` enum value."""
    try:
        from moonshine_voice import model_arch_to_string
        return model_arch_to_string(model_arch)
    except Exception:                                                  # noqa: BLE001
        return repr(model_arch)


__all__ = [
    "MoonshineEngine",
    "is_moonshine_available",
    "MOONSHINE_INSTALL_HINT",
]
