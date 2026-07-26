"""Text-to-speech engines.

Two engines are wired:

- :class:`KokoroSpeech` — StyleTTS2 + ISTFTNet (2026-05-20 swap).
  Current production default (``tts.engine: kokoro``). CUDA or CPU;
  the fine-tuned Kenning voice loads from ``models/kokoro/voices/kenning.pt``.
- :class:`XttsV3Speech` — XTTS v2 streaming + v3 filter (legacy
  high-quality option). Selected when ``tts.engine: xtts_v3``.

The legacy ``piper_rvc`` engine (Piper + RVC voice conversion) was retired
2026-07-23: production ran ``kokoro`` and the RVC path was dormant. The
trained RVC voicepack (``kenning_rvc_voice/``) is kept on disk under the
existing voice-baseline protections, but the engine, ``RvcConverter`` and the
Piper ``TextToSpeech`` wrapper are gone.

Use :func:`make_tts_engine` to construct the configured engine. The
orchestrator and measurement scripts both call into this factory so
they always exercise the same code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from kenning.tts.kokoro_engine import KokoroSpeech
from kenning.tts.xtts_v3 import XttsV3Speech
from kenning.utils.logging import get_logger

if TYPE_CHECKING:
    from kenning.config import TTSConfig

logger = get_logger("tts.factory")

# Type alias: any object that exposes the orchestrator-facing surface
# (``warmup``, ``speak``, ``speak_stream``, ``prepare_output_stream``,
# ``stop``). Both engines satisfy it.
TTSEngine = Union[KokoroSpeech, XttsV3Speech]


def make_tts_engine(
    cfg: "TTSConfig | None" = None,
) -> Tuple[Optional[object], TTSEngine]:
    """Construct the TTS engine selected by ``tts.engine``.

    Returns a ``(rvc_or_none, tts_engine)`` pair. The first element is ALWAYS
    ``None`` now that the legacy ``piper_rvc`` engine is retired — the tuple
    shape is preserved so the orchestrator (which keeps an ``rvc`` attribute,
    always ``None``, and None-guards its ``close()``) and the TTS injector need
    no change.

    Selectors:
    - ``kokoro``: :class:`KokoroSpeech` (production default).
    - ``xtts_v3``: :class:`XttsV3Speech`.

    Raises:
        RuntimeError: when ``tts.engine`` is set to an unknown value.
    """
    from kenning.config import get_config, resolve_path

    if cfg is None:
        cfg = get_config().tts

    engine_name = getattr(cfg, "engine", "kokoro")

    if engine_name == "xtts_v3":
        logger.info("TTS engine: xtts_v3 (XTTS v2 streaming + v3 filter)")
        return None, XttsV3Speech()

    if engine_name == "kokoro":
        kokoro_cfg = getattr(cfg, "kokoro", None)
        kwargs: dict = {}
        if kokoro_cfg is not None:
            kwargs = {
                "model_path": resolve_path(kokoro_cfg.model_path),
                "voice": kokoro_cfg.voice,
                "device": kokoro_cfg.device,
                "speed": kokoro_cfg.speed,
                "apply_runtime_filter": kokoro_cfg.apply_runtime_filter,
                "filter_preset": kokoro_cfg.filter_preset,
                "apply_spectral_smooth": kokoro_cfg.apply_spectral_smooth,
                "spectral_smooth_window": kokoro_cfg.spectral_smooth_window,
                "apply_trim_fade": kokoro_cfg.apply_trim_fade,
                "trim_fade_threshold_db": kokoro_cfg.trim_fade_threshold_db,
                "f0_contour_factor": kokoro_cfg.f0_contour_factor,
                "f0_shift_semitones": kokoro_cfg.f0_shift_semitones,
                "f0_max_excursion": kokoro_cfg.f0_max_excursion,
                "f0_energy_factor": kokoro_cfg.f0_energy_factor,
                "dur_final_factor": kokoro_cfg.dur_final_factor,
                "dur_internal_factor": kokoro_cfg.dur_internal_factor,
                "dur_stress_factor": kokoro_cfg.dur_stress_factor,
                "max_pause_cap_ms": kokoro_cfg.max_pause_cap_ms,
            }
        logger.info(
            "TTS engine: kokoro (StyleTTS2 + ISTFTNet, voice=%s, device=%s)",
            kwargs.get("voice", "af_alloy"),
            kwargs.get("device", "cpu"),
        )
        return None, KokoroSpeech(**kwargs)

    raise RuntimeError(
        f"Unknown tts.engine: {engine_name!r}. Valid: 'kokoro' | 'xtts_v3'."
    )


__all__ = [
    "KokoroSpeech",
    "XttsV3Speech",
    "TTSEngine",
    "make_tts_engine",
]
