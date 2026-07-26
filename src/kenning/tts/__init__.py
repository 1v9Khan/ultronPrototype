"""Text-to-speech engines.

Three engines are wired:

- :class:`KokoroSpeech` — StyleTTS2 + ISTFTNet (2026-05-20 swap).
  Current production default (``tts.engine: kokoro``). CUDA or CPU;
  the fine-tuned Kenning voice loads from ``models/kokoro/voices/kenning.pt``.
- :class:`XttsV3Speech` — XTTS v2 streaming + v3 filter (legacy
  high-quality option). Selected when ``tts.engine: xtts_v3``.
- :class:`TextToSpeech` — Piper + optional RVC (long-standing
  fallback). Selected when ``tts.engine: piper_rvc``.

Use :func:`make_tts_engine` to construct the configured engine. The
orchestrator and measurement scripts both call into this factory so
they always exercise the same code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from kenning.tts.rvc import RvcConverter
from kenning.tts.speech import TextToSpeech
from kenning.tts.kokoro_engine import KokoroSpeech
from kenning.tts.xtts_v3 import XttsV3Speech
from kenning.utils.logging import get_logger

if TYPE_CHECKING:
    from kenning.config import TTSConfig

logger = get_logger("tts.factory")

# Type alias: any object that exposes the orchestrator-facing surface
# (``warmup``, ``speak``, ``speak_stream``, ``prepare_output_stream``,
# ``stop``). All three engines satisfy it.
TTSEngine = Union[KokoroSpeech, XttsV3Speech, TextToSpeech]


def _load_rvc_if_enabled() -> Optional[RvcConverter]:
    """Construct an RVC converter iff config enables it AND the model is on
    disk. Returns ``None`` otherwise (the caller falls back to plain Piper).

    Replicates the legacy orchestrator helper so the factory is self-
    contained — pulled out of :mod:`kenning.pipeline.orchestrator` during the
    2026-05-22 measurement-script audit so ``scripts/measure_baseline.py``
    can build the same TTS engine without depending on the orchestrator.
    """
    from config import settings  # noqa: WPS433 — legacy shim, intentional

    if not settings.RVC_ENABLED:
        return None
    if not settings.RVC_MODEL_PATH.is_file():
        logger.warning(
            "RVC enabled but model missing at %s -- falling back to plain Piper",
            settings.RVC_MODEL_PATH,
        )
        return None
    try:
        return RvcConverter()
    except Exception as e:                                       # noqa: BLE001
        logger.warning("RVC load failed (%s) -- falling back to plain Piper", e)
        return None


def make_tts_engine(
    cfg: "TTSConfig | None" = None,
) -> Tuple[Optional[RvcConverter], TTSEngine]:
    """Construct the TTS engine selected by ``tts.engine``.

    Returns a ``(rvc_or_none, tts_engine)`` pair. ``rvc`` is non-None only
    for the legacy ``piper_rvc`` engine — kept in the return tuple so the
    orchestrator (which retains an ``rvc`` attribute for diagnostics) can
    drop in without changes.

    Selectors:
    - ``kokoro``: :class:`KokoroSpeech` (production default).
    - ``xtts_v3``: :class:`XttsV3Speech`.
    - ``piper_rvc``: :class:`TextToSpeech` plus optional RVC.

    Raises:
        RuntimeError: when ``tts.engine`` is set to an unknown value.
    """
    from kenning.config import get_config, resolve_path

    if cfg is None:
        cfg = get_config().tts

    engine_name = getattr(cfg, "engine", "piper_rvc")

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

    if engine_name == "piper_rvc":
        logger.info("TTS engine: piper_rvc (legacy Piper + RVC)")
        rvc = _load_rvc_if_enabled()
        return rvc, TextToSpeech(rvc=rvc)

    raise RuntimeError(
        f"Unknown tts.engine: {engine_name!r}. "
        f"Valid: 'kokoro' | 'xtts_v3' | 'piper_rvc'."
    )


__all__ = [
    "TextToSpeech",
    "RvcConverter",
    "KokoroSpeech",
    "XttsV3Speech",
    "TTSEngine",
    "make_tts_engine",
]
