"""faster-whisper wrapper.

Loads the model once at construction (not lazily on first transcribe), so the
hot path is just GPU inference. Audio is expected as mono float32 at 16 kHz —
the rest of the pipeline already standardizes on that, so no resampling here.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np

from config import settings
from kenning.errors import WhisperTranscriptionError
from kenning.resilience import get_error_log
from kenning.utils.logging import get_logger

logger = get_logger("transcription.whisper")

# The decoder-priming vocabulary + the non-speech blocklist live in
# ``_whisper_text`` (stdlib-only) so the OUT-OF-PROCESS sidecar
# (``scripts/stt_server.py``) applies exactly the same decode-quality rules --
# otherwise running STT on another GPU would silently cost agent-name accuracy
# and re-admit "Thank you."-type phantom callouts. Names re-exported under the
# historical private spellings so existing references keep working.
from kenning.transcription._whisper_text import (          # noqa: E402
    DOMAIN_PROMPT as _DOMAIN_PROMPT,
    WHISPER_HALLUCINATIONS as _WHISPER_HALLUCINATIONS,
    build_initial_prompt as _build_initial_prompt,
    is_whisper_hallucination as _is_whisper_hallucination,
)


class WhisperEngine:
    """Speech-to-text via faster-whisper on CUDA.

    Args:
        model_name: e.g. ``small.en``, ``base.en``, ``medium.en``.
        device: ``cuda`` or ``cpu``.
        compute_type: ``float16``, ``int8_float16``, ``int8``, ``float32``.
        beam_size: 1 for greedy decoding (fastest), >1 for beam search.
    """

    def __init__(
        self,
        model_name: str = settings.WHISPER_MODEL,
        device: str = settings.WHISPER_DEVICE,
        compute_type: str = settings.WHISPER_COMPUTE_TYPE,
        beam_size: int = settings.WHISPER_BEAM_SIZE,
        device_index: int = getattr(settings, "WHISPER_DEVICE_INDEX", 0),
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        # 2026-07-24 multi-GPU: pin Whisper to a SPECIFIC card so the primary
        # GPU holds nothing but Ultron's model. CTranslate2 ignores this on
        # CPU.
        self.device_index = int(device_index)

        logger.info(
            "Loading Whisper '%s' on %s%s (%s)…",
            model_name,
            device,
            f":{self.device_index}" if device != "cpu" else "",
            compute_type,
        )
        t0 = time.monotonic()
        try:
            self._model = WhisperModel(
                model_name, device=device, compute_type=compute_type,
                device_index=self.device_index,
            )
        except Exception as e:
            logger.error("Whisper load failed: %s", e)
            raise
        logger.info("Whisper ready in %.2fs", time.monotonic() - t0)

    def __enter__(self) -> "WhisperEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # faster-whisper has no explicit close; let GC reclaim CUDA memory.
        self._model = None

    def transcribe(self, audio: np.ndarray, language: Optional[str] = "en") -> str:
        """Transcribe an audio segment to text.

        Args:
            audio: mono float32 at 16 kHz.
            language: ISO code, or ``None`` to autodetect (slower).

        Returns:
            Stripped transcription text. May be empty for silence.
            On Whisper failure, returns ``""`` and logs to errors.jsonl;
            the orchestrator's repeated-failure counter takes over from
            there ("Speech recognition is having trouble." after 3+).
        """
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Silence gate: skip the GPU call on near-silent buffers. faster-whisper
        # hallucinates stock phrases ("Thank you.") on silence / faint room
        # tone; a real callout peaks far above this floor. Cheap insurance that
        # also saves an inference when upstream VAD lets a quiet buffer through.
        if float(np.max(np.abs(audio))) < 0.008:
            return ""

        t0 = time.monotonic()
        try:
            # Decode-time DOMAIN BIASING: prime the decoder with the Valorant
            # closed vocabulary so proper nouns (agent names) and callout terms
            # are recognised at the SOURCE -- fewer downstream corrections needed.
            # Additive + reversible: gated by WHISPER_DOMAIN_BIAS (default on);
            # initial_prompt is supported by every faster-whisper version. Reset
            # per turn (condition_on_previous_text stays off for command STT).
            _kw = dict(
                language=language,
                beam_size=self.beam_size,
                temperature=settings.WHISPER_TEMPERATURE,
                condition_on_previous_text=settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
                vad_filter=settings.WHISPER_VAD_FILTER,
            )
            if getattr(settings, "WHISPER_DOMAIN_BIAS", True):
                _kw["initial_prompt"] = _build_initial_prompt(
                    getattr(settings, "WHISPER_INITIAL_PROMPT", "") or "")
            segments, info = self._model.transcribe(audio, **_kw)
            kept = []
            for seg in segments:
                # Drop segments the model is highly confident are non-speech.
                if getattr(seg, "no_speech_prob", 0.0) > 0.85:
                    continue
                piece = (seg.text or "").strip()
                if piece:
                    kept.append(piece)
            text = " ".join(kept).strip()
            # Final guard: a whole-transcript stock phrase is a hallucination.
            if text and _is_whisper_hallucination(text):
                logger.debug("whisper: dropped non-speech hallucination %r", text)
                text = ""
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error(
                "Whisper transcribe failed in %.0fms: %s", elapsed_ms, e,
            )
            get_error_log().record(
                WhisperTranscriptionError(
                    f"transcribe failed: {e}",
                    context={
                        "audio_seconds": len(audio) / settings.SAMPLE_RATE,
                        "model": self.model_name,
                        "device": self.device,
                    },
                    recovery="returned empty transcription; orchestrator skips this turn",
                ),
                dependency="whisper",
            )
            return ""
        elapsed_ms = (time.monotonic() - t0) * 1000
        audio_seconds = len(audio) / settings.SAMPLE_RATE
        logger.info(
            "Whisper: %.2fs audio → %d chars in %.0fms (RTF=%.2f, lang=%s)",
            audio_seconds,
            len(text),
            elapsed_ms,
            elapsed_ms / 1000 / max(audio_seconds, 1e-6),
            info.language,
        )
        return text
