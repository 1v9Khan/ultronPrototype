"""Wake-word detection via openWakeWord.

The user-facing wake word is "Kenning", a custom-trained ONNX at
``settings.WAKE_WORD_MODEL_PATH`` (``models/openwakeword/kenning.onnx``).
"kenning" and "ultron" are BOTH custom models living side by side in
``models/openwakeword/`` -- the active one is selected by
``settings.WAKE_WORD_NAME`` and can be hot-swapped at runtime via
:meth:`WakeWordDetector.reload_for_word` (the settings-panel "Wake word"
dropdown). When the selected model is missing the detector falls back to
the custom ``ultron.onnx`` (``settings.WAKE_WORD_FALLBACK``), NOT a
pretrained word. A pretrained openWakeWord word is used only as an
absolute last resort (neither the selected nor the fallback ONNX exists).

Train your own model: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings
from kenning.utils.logging import get_logger

logger = get_logger("audio.wake_word")


class WakeWordDetector:
    """Streaming wake-word detector.

    Args:
        model_path: Path to the selected custom ONNX (e.g. ``kenning.onnx``).
            If ``None`` or missing, falls back to the ``fallback_name``
            custom ONNX, then to a pretrained word as a last resort.
        fallback_name: Wake word used when the selected model is missing.
            Resolved FIRST as a custom ONNX (``{models_dir}/{name}.onnx``,
            e.g. ``ultron.onnx``); only if that is absent is it treated as
            a pretrained openWakeWord built-in.
        name: The selected wake word's display name (e.g. ``kenning``).
        threshold: Probability above which a frame counts as a detection.
        cooldown_seconds: Suppress repeat triggers within this window.
    """

    def __init__(
        self,
        model_path: Optional[Path] = settings.WAKE_WORD_MODEL_PATH,
        fallback_name: str = settings.WAKE_WORD_FALLBACK,
        threshold: float = settings.WAKE_WORD_THRESHOLD,
        cooldown_seconds: float = settings.WAKE_WORD_COOLDOWN_SECONDS,
        name: str = settings.WAKE_WORD_NAME,
    ) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger_ts = 0.0
        self._using_fallback = False
        self._active_word: str = ""
        self._name = (name or "kenning").strip().lower()
        self._fallback_name = (fallback_name or "ultron").strip().lower()
        # Directory that holds the side-by-side custom models
        # (kenning.onnx, ultron.onnx). Derived from the configured model
        # path so word->onnx resolution stays in one place.
        self._models_dir = (
            Path(model_path).parent if model_path is not None
            else Path("models/openwakeword")
        )

        self._model = self._load_model(model_path, self._fallback_name)

    # --- model loading -------------------------------------------------------

    def _model_path_for_word(self, word: str) -> Optional[Path]:
        """Resolve a wake word to its side-by-side custom ONNX, or None.

        ``kenning`` -> ``{models_dir}/kenning.onnx`` (if it exists).
        """
        word = (word or "").strip().lower()
        if not word:
            return None
        cand = self._models_dir / f"{word}.onnx"
        return cand if cand.is_file() else None

    def _load_model(self, model_path: Optional[Path], fallback_name: str):
        from openwakeword.model import Model

        # 1. The selected word's custom model.
        if model_path is not None and Path(model_path).is_file():
            logger.info("Loading custom wake-word model: %s", model_path)
            self._active_word = self._name
            self._using_fallback = False
            return Model(wakeword_models=[str(model_path)], inference_framework="onnx")

        # 2. Fallback to a CUSTOM ONNX (e.g. ultron.onnx) -- never a
        #    pretrained word while a real custom fallback exists.
        fb_path = self._model_path_for_word(fallback_name)
        if fb_path is not None:
            self._using_fallback = True
            self._active_word = fallback_name
            msg = (
                f"Wake-word model for '{self._name}' not found at "
                f"{model_path}. Falling back to custom '{fallback_name}' "
                f"({fb_path.name}). Train/deploy "
                f"{self._models_dir / (self._name + '.onnx')} to use "
                f"'{self._name}'."
            )
            logger.warning(msg)
            print(f"\n[!] {msg}\n")
            return Model(wakeword_models=[str(fb_path)], inference_framework="onnx")

        # 3. Absolute last resort: a pretrained openWakeWord built-in.
        self._using_fallback = True
        self._active_word = fallback_name
        msg = (
            f"Neither the selected '{self._name}' model nor a custom "
            f"fallback '{fallback_name}.onnx' was found under "
            f"{self._models_dir}. Using pretrained '{fallback_name}'."
        )
        logger.warning(msg)
        print(f"\n[!] {msg}\n")
        return Model(wakeword_models=[fallback_name], inference_framework="onnx")

    def reload_for_word(self, word: str) -> tuple[bool, str]:
        """Hot-swap the active wake word at runtime (settings-panel
        dropdown). Resolves ``word`` to its side-by-side custom ONNX and
        swaps the live model in place. When the requested model is
        missing, falls back to the custom ``self._fallback_name`` model
        (e.g. ultron). Resets the cooldown so the new word can fire
        immediately. Returns ``(changed_to_requested, message)``.
        """
        from openwakeword.model import Model

        word = (word or "").strip().lower()
        if not word:
            return False, "no wake word given"
        if word == self._active_word and not self._using_fallback:
            return True, word  # already active; no-op

        target = self._model_path_for_word(word)
        if target is not None:
            try:
                new_model = Model(
                    wakeword_models=[str(target)], inference_framework="onnx"
                )
            except Exception as e:                                   # noqa: BLE001
                return False, f"failed to load {word}: {e}"
            self._model = new_model
            self._name = word
            self._active_word = word
            self._using_fallback = False
            self._last_trigger_ts = 0.0
            logger.info("Wake word hot-swapped to '%s' (%s)", word, target.name)
            return True, word

        # Requested model missing -> custom fallback (e.g. ultron).
        fb_path = self._model_path_for_word(self._fallback_name)
        if fb_path is not None:
            try:
                new_model = Model(
                    wakeword_models=[str(fb_path)], inference_framework="onnx"
                )
            except Exception as e:                                   # noqa: BLE001
                return False, f"failed to load fallback {self._fallback_name}: {e}"
            self._model = new_model
            self._active_word = self._fallback_name
            self._using_fallback = True
            self._last_trigger_ts = 0.0
            logger.warning(
                "Wake word '%s' model missing; using fallback '%s'",
                word, self._fallback_name,
            )
            return False, f"{word} model not found; using {self._fallback_name}"
        return False, f"{word} model not found and no fallback available"

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
