"""Speech-to-text engines.

Two engines are wired:

- :class:`WhisperEngine` -- the long-standing default; faster-whisper
  on CUDA. Strong on accented / noisy audio; ~80 ms median on 5 s
  audio with ``base.en`` beam=1.
- :class:`ParakeetEngine` -- NVIDIA Parakeet TDT via NeMo.
  Frontier-enhancement Item 5 (2026-05-21). Streaming-native RNN-T;
  ~RTFx 2000+ on consumer GPUs. Requires
  ``pip install nemo_toolkit[asr]``.

Use :func:`make_stt_engine` to construct the configured engine
(respects ``stt.engine`` and falls back to Whisper when Parakeet's
dependencies aren't installed).
"""

from __future__ import annotations

from typing import Union, TYPE_CHECKING

from ultron.transcription.parakeet_engine import (
    PARAKEET_INSTALL_HINT,
    ParakeetEngine,
    is_nemo_available,
)
from ultron.transcription.whisper_engine import WhisperEngine
from ultron.utils.logging import get_logger

if TYPE_CHECKING:
    from ultron.config import STTConfig

logger = get_logger("transcription.factory")

# Type alias: any engine that quacks like the WhisperEngine
# transcribe interface. Both engines expose
# ``transcribe(audio: np.ndarray, language: Optional[str]) -> str``.
STTEngine = Union[WhisperEngine, ParakeetEngine]


def make_stt_engine(cfg: "STTConfig | None" = None) -> STTEngine:
    """Construct the STT engine selected by ``stt.engine``.

    Resolution:
    - ``auto``: Parakeet if NeMo is installed; else Whisper.
    - ``whisper``: always Whisper.
    - ``parakeet``: always Parakeet (raises if NeMo missing).

    The active choice is logged at INFO so it's visible at startup --
    important because the engine is the FIRST thing to suspect if
    voice transcription regresses after 2026-05-21.
    """
    if cfg is None:
        from ultron.config import get_config
        cfg = get_config().stt

    selector = getattr(cfg, "engine", "whisper")

    if selector == "parakeet":
        if not is_nemo_available():
            raise ImportError(PARAKEET_INSTALL_HINT)
        logger.info(
            "STT engine: parakeet (forced by config; frontier item 5)"
        )
        return ParakeetEngine(
            model_name=getattr(cfg, "parakeet_model", None),
            device=getattr(cfg, "parakeet_device", None),
        )

    if selector == "auto":
        if is_nemo_available():
            try:
                engine = ParakeetEngine(
                    model_name=getattr(cfg, "parakeet_model", None),
                    device=getattr(cfg, "parakeet_device", None),
                )
                logger.info(
                    "STT engine: parakeet (auto-detected NeMo; "
                    "frontier item 5. If voice quality regresses, "
                    "set ``stt.engine: whisper`` to swap back.)"
                )
                return engine
            except Exception as e:                                 # noqa: BLE001
                logger.warning(
                    "Parakeet auto-load failed (%s); falling back to "
                    "Whisper. Set ``stt.engine: parakeet`` explicitly "
                    "to surface this error.", e,
                )
        logger.info("STT engine: whisper (auto -- NeMo not available)")
        return WhisperEngine()

    # selector == "whisper" or anything unrecognised
    logger.info("STT engine: whisper")
    return WhisperEngine()


__all__ = [
    "make_stt_engine",
    "STTEngine",
    "WhisperEngine",
    "ParakeetEngine",
    "is_nemo_available",
    "PARAKEET_INSTALL_HINT",
]
