"""Wake-word detection via openWakeWord.

The user-facing wake word is "Ultron", which is **not** a pretrained model in
openWakeWord. The detector tries to load a custom-trained ONNX at
``settings.WAKE_WORD_MODEL_PATH`` and, if missing, falls back to one of the
shipped pretrained words (default: ``hey_jarvis``) with a prominent warning.

Train your own model: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings
from ultron.utils.logging import get_logger

logger = get_logger("audio.wake_word")


class WakeWordDetector:
    """Streaming wake-word detector.

    Args:
        model_path: Path to a custom ONNX model (e.g. ``ultron.onnx``).
            If ``None`` or missing, falls back to ``fallback_name``.
        fallback_name: Pretrained word to use when no custom model is found.
            Must be one of openWakeWord's built-ins.
        threshold: Probability above which a frame counts as a detection.
        cooldown_seconds: Suppress repeat triggers within this window.
    """

    def __init__(
        self,
        model_path: Optional[Path] = settings.WAKE_WORD_MODEL_PATH,
        fallback_name: str = settings.WAKE_WORD_FALLBACK,
        threshold: float = settings.WAKE_WORD_THRESHOLD,
        cooldown_seconds: float = settings.WAKE_WORD_COOLDOWN_SECONDS,
    ) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger_ts = 0.0
        self._using_fallback = False
        self._active_word: str = ""

        self._model = self._load_model(model_path, fallback_name)

    # --- model loading -------------------------------------------------------

    def _load_model(self, model_path: Optional[Path], fallback_name: str):
        from openwakeword.model import Model

        if model_path is not None and Path(model_path).is_file():
            logger.info("Loading custom wake-word model: %s", model_path)
            self._active_word = settings.WAKE_WORD_NAME
            return Model(wakeword_models=[str(model_path)], inference_framework="onnx")

        # Fallback path
        self._using_fallback = True
        self._active_word = fallback_name
        msg = (
            f"Custom Ultron wake-word model not found at {model_path}. "
            f"Falling back to pretrained '{fallback_name}'. "
            f"To use 'Ultron' as the wake word, train a custom model and "
            f"place it at the path above."
        )
        logger.warning(msg)
        print(f"\n[!] {msg}\n")

        return Model(wakeword_models=[fallback_name], inference_framework="onnx")

    # --- properties ----------------------------------------------------------

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def active_word(self) -> str:
        return self._active_word

    # --- streaming API -------------------------------------------------------

    def process(self, audio: np.ndarray) -> bool:
        """Feed a chunk of audio. Returns True iff the wake word fired.

        openWakeWord expects int16 PCM at 16 kHz. We ingest float32 from
        sounddevice and convert here.
        """
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        # Convert float32 [-1, 1] → int16 PCM
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        scores = self._model.predict(pcm)
        score = max(scores.values()) if scores else 0.0

        if score < self.threshold:
            return False

        now = time.monotonic()
        if now - self._last_trigger_ts < self.cooldown_seconds:
            return False

        self._last_trigger_ts = now
        logger.info("Wake word '%s' detected (score=%.2f)", self._active_word, score)
        return True

    def reset(self) -> None:
        try:
            self._model.reset()
        except AttributeError:
            pass
        self._last_trigger_ts = 0.0

    def fired_recently(self, window_s: float = 0.5) -> bool:
        """Return True iff the wake word fired within the last ``window_s``
        seconds.

        A4 (pre-task confirmation): the orchestrator polls this during
        the confirmation TTS playback to detect a barge-in. The model
        already tracks ``_last_trigger_ts``; this accessor is a read-only
        view that doesn't reset internal state. Idempotent across calls.

        Returns False when the detector has never fired (initial
        ``_last_trigger_ts == 0``) so a stale or zeroed timestamp can't
        spoof a barge-in on the first task of a session.
        """
        if self._last_trigger_ts <= 0.0:
            return False
        return (time.monotonic() - self._last_trigger_ts) < float(window_s)
