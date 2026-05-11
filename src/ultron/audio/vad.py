"""Silero VAD wrapper.

Silero v5 is a small ONNX/PyTorch model that classifies fixed-size windows
(512 samples at 16 kHz) as speech vs. non-speech. We accumulate audio across
arbitrary chunk boundaries, slice it into 512-sample windows, and run the
model. Hysteresis (min speech / min silence) prevents toggling on coughs and
brief pauses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from config import settings
from ultron.utils.logging import get_logger

logger = get_logger("audio.vad")


class SpeechEvent(Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VadResult:
    event: SpeechEvent
    is_speech: bool
    probability: float


class VoiceActivityDetector:
    """Streaming VAD that emits start/end events.

    Args:
        sample_rate: Must be 16000 (Silero requirement).
        threshold: Probability above which a window counts as speech.
        min_speech_ms: Sustained speech needed before emitting SPEECH_START.
        min_silence_ms: Sustained silence needed before emitting SPEECH_END.
        window_samples: Window size for the model. Silero v5 wants 512.
    """

    def __init__(
        self,
        sample_rate: int = settings.SAMPLE_RATE,
        threshold: float = settings.VAD_THRESHOLD,
        min_speech_ms: int = settings.MIN_SPEECH_DURATION_MS,
        min_silence_ms: int = settings.MIN_SILENCE_DURATION_MS,
        window_samples: int = settings.VAD_WINDOW_SAMPLES,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("Silero VAD requires 16 kHz audio")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window_samples = window_samples
        self.window_ms = (window_samples / sample_rate) * 1000

        # Convert hysteresis durations to consecutive-window counts.
        self._speech_windows_required = max(1, int(min_speech_ms / self.window_ms))
        # Default silence requirement -- captured separately from the
        # currently-active one so ``reset()`` can restore the baseline
        # after the orchestrator's adaptive end-of-turn bump.
        self._default_silence_windows_required = max(1, int(min_silence_ms / self.window_ms))
        self._silence_windows_required = self._default_silence_windows_required

        self._is_speech_active = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._tail = np.zeros(0, dtype=np.float32)

        self._model = self._load_model()

    # --- model loading -------------------------------------------------------

    @staticmethod
    def _load_model():
        """Load Silero VAD; tolerate either packaging style."""
        try:
            from silero_vad import load_silero_vad

            model = load_silero_vad(onnx=False)
            logger.info("Silero VAD loaded (PyTorch backend)")
            return model
        except Exception as e:
            logger.error("Failed to load Silero VAD: %s", e)
            raise

    # --- streaming API -------------------------------------------------------

    def reset(self) -> None:
        self._is_speech_active = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._tail = np.zeros(0, dtype=np.float32)
        # Restore baseline silence requirement; an adaptive bump from
        # the previous utterance shouldn't leak into the next one.
        self._silence_windows_required = self._default_silence_windows_required
        try:
            self._model.reset_states()
        except AttributeError:
            pass

    def set_min_silence_duration_ms(self, ms: int) -> None:
        """Adjust the trailing-silence requirement at runtime.

        Used by the orchestrator's adaptive end-of-turn policy: long
        utterances get a longer silence requirement so a thinking
        pause mid-description doesn't close the capture. ``reset()``
        restores the baseline configured at construction.
        """
        self._silence_windows_required = max(1, int(ms / self.window_ms))

    def process(self, audio: np.ndarray) -> Optional[VadResult]:
        """Feed a chunk of audio. Returns the most recent event for the chunk.

        If a chunk straddles multiple windows, the *last* event in the chunk
        wins — typical chunks are ~32 ms while the model window is also ~32 ms,
        so this is a non-issue in practice.
        """
        import torch  # local import to keep cold-start light

        if audio.ndim != 1:
            audio = audio.reshape(-1)
        buffer = np.concatenate([self._tail, audio.astype(np.float32, copy=False)])

        last_event = SpeechEvent.NONE
        last_prob = 0.0
        n = self.window_samples
        i = 0
        while i + n <= len(buffer):
            window = buffer[i : i + n]
            with torch.no_grad():
                prob = float(
                    self._model(torch.from_numpy(window), self.sample_rate).item()
                )
            event = self._update_state(prob)
            if event != SpeechEvent.NONE:
                last_event = event
            last_prob = prob
            i += n

        # Save trailing samples that didn't fill a window.
        self._tail = buffer[i:].copy()

        return VadResult(
            event=last_event,
            is_speech=self._is_speech_active,
            probability=last_prob,
        )

    # --- internal ------------------------------------------------------------

    def _update_state(self, prob: float) -> SpeechEvent:
        if prob >= self.threshold:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
            if (
                not self._is_speech_active
                and self._consecutive_speech >= self._speech_windows_required
            ):
                self._is_speech_active = True
                logger.debug("VAD: speech start (prob=%.2f)", prob)
                return SpeechEvent.SPEECH_START
        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0
            if (
                self._is_speech_active
                and self._consecutive_silence >= self._silence_windows_required
            ):
                self._is_speech_active = False
                logger.debug("VAD: speech end (prob=%.2f)", prob)
                return SpeechEvent.SPEECH_END
        return SpeechEvent.NONE
