"""Speech-to-text engines.

Three engines are wired:

- :class:`WhisperEngine` -- the long-standing default; faster-whisper
  on CUDA. Strong on accented / noisy audio; ~80 ms median on 5 s
  audio with ``base.en`` beam=1.
- :class:`ParakeetEngine` -- NVIDIA Parakeet TDT via NeMo.
  Frontier-enhancement Item 5 (2026-05-21). Streaming-native RNN-T;
  ~RTFx 2000+ on consumer GPUs. Requires
  ``pip install nemo_toolkit[asr]`` in an isolated venv.
- :class:`MoonshineEngine` -- Moonshine ONNX (2026-05-22). Lowest-
  footprint option (58 MB base model on CPU); streaming-native; ~5-
  15 ms on short voice clips. Requires
  ``pip install useful-moonshine-onnx`` -- pure ONNX runtime, no
  Keras / TF / PyTorch upgrade needed.

Use :func:`make_stt_engine` to construct the configured engine
(respects ``stt.engine`` and falls back gracefully when deps are
missing). Use :func:`make_dual_stt_engines` to build both a primary
and a gaming engine for runtime swap (see :class:`DualSTTRegistry`).
"""

from __future__ import annotations

from typing import Union, TYPE_CHECKING

from ultron.transcription.moonshine_engine import (
    MOONSHINE_INSTALL_HINT,
    MoonshineEngine,
    is_moonshine_available,
)
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
# transcribe interface. All engines expose
# ``transcribe(audio: np.ndarray, language: Optional[str]) -> str``.
STTEngine = Union[WhisperEngine, ParakeetEngine, MoonshineEngine]


def make_stt_engine(cfg: "STTConfig | None" = None) -> STTEngine:
    """Construct the STT engine selected by ``stt.engine``.

    Resolution:
    - ``auto``: Parakeet if NeMo is installed; else Whisper. (Moonshine
      is opt-in via the explicit selector because its WER trade-off vs
      Whisper depends on the user's audio characteristics and we don't
      want to silently switch.)
    - ``whisper``: always Whisper.
    - ``parakeet``: always Parakeet (raises if NeMo missing).
    - ``moonshine``: always Moonshine ONNX (raises if package missing).

    The active choice is logged at INFO so it's visible at startup --
    important because the engine is the FIRST thing to suspect if
    voice transcription regresses.
    """
    if cfg is None:
        from ultron.config import get_config
        cfg = get_config().stt

    selector = getattr(cfg, "engine", "whisper")

    if selector == "moonshine":
        if not is_moonshine_available():
            raise ImportError(MOONSHINE_INSTALL_HINT)
        logger.info(
            "STT engine: moonshine (forced by config; ONNX on CPU)"
        )
        return MoonshineEngine(
            model_name=getattr(cfg, "moonshine_model", None),
            device=getattr(cfg, "moonshine_device", None),
            model_precision=getattr(cfg, "moonshine_precision", None),
        )

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


# ---------------------------------------------------------------------------
# Dual-engine support (2026-05-22) -- gaming-mode STT swap
# ---------------------------------------------------------------------------


def _build_engine_by_name(name: str, cfg: "STTConfig"):
    """Construct a specific named engine. Used by the dual-engine path.

    Mirrors :func:`make_stt_engine` resolution but takes the engine
    name explicitly instead of reading ``cfg.engine``.
    """
    if name == "moonshine":
        if not is_moonshine_available():
            raise ImportError(MOONSHINE_INSTALL_HINT)
        return MoonshineEngine(
            model_name=getattr(cfg, "moonshine_model", None),
            device=getattr(cfg, "moonshine_device", None),
            model_precision=getattr(cfg, "moonshine_precision", None),
        )
    if name == "parakeet":
        if not is_nemo_available():
            raise ImportError(PARAKEET_INSTALL_HINT)
        return ParakeetEngine(
            model_name=getattr(cfg, "parakeet_model", None),
            device=getattr(cfg, "parakeet_device", None),
        )
    if name == "whisper":
        return WhisperEngine()
    raise ValueError(f"unknown STT engine name: {name!r}")


class DualSTTRegistry:
    """Holds a primary + optional gaming engine and supports runtime swap.

    The orchestrator constructs this at init time when
    ``stt.gaming_engine`` is set + different from ``stt.engine``. The
    ``active`` property exposes whichever engine the gaming-mode flip
    currently points at; ``swap_to(name)`` flips between them.

    Both engines stay loaded for the lifetime of the process so the
    swap is cheap (microseconds for the pointer flip). The gaming-mode
    callback also calls :meth:`ParakeetEngine.stop_server` separately
    to release VRAM held by the Parakeet HTTP server -- the engine
    handle here stays around and reconnects when stop_server's
    counterpart re-spawns the server on disengage.
    """

    def __init__(
        self,
        *,
        primary: "STTEngine",
        primary_name: str,
        gaming: "Optional[STTEngine]" = None,
        gaming_name: "Optional[str]" = None,
    ) -> None:
        self.primary = primary
        self.primary_name = primary_name
        self.gaming = gaming
        self.gaming_name = gaming_name
        self._active_name = primary_name
        self._active: "STTEngine" = primary

    @property
    def active(self) -> "STTEngine":
        return self._active

    @property
    def active_name(self) -> str:
        return self._active_name

    def has_gaming(self) -> bool:
        return self.gaming is not None and self.gaming_name is not None

    def swap_to(self, name: str) -> "STTEngine":
        """Flip the active pointer. Returns the now-active engine.

        Unknown / unconfigured names log a WARN and leave the active
        engine unchanged so the voice loop keeps working.
        """
        if name == self.primary_name:
            self._active = self.primary
            self._active_name = self.primary_name
            return self._active
        if self.has_gaming() and name == self.gaming_name:
            self._active = self.gaming  # type: ignore[assignment]
            self._active_name = name
            return self._active
        logger.warning(
            "DualSTTRegistry.swap_to(%r): unknown name; staying on %r",
            name, self._active_name,
        )
        return self._active


from typing import Optional


def make_dual_stt_engines(
    cfg: "STTConfig | None" = None,
) -> "DualSTTRegistry":
    """Construct primary + (optional) gaming engine registry.

    If ``stt.gaming_engine`` is empty or matches the resolved primary
    engine name, the registry has no gaming engine (gaming mode falls
    back to the engage / disengage hooks for Kokoro + VLM + LLM only).

    Returns a :class:`DualSTTRegistry`. Construction failures for the
    gaming engine are caught + logged WARN -- the primary engine still
    loads, just without the swap-on-engage capability.
    """
    if cfg is None:
        from ultron.config import get_config
        cfg = get_config().stt

    primary = make_stt_engine(cfg)
    primary_name = _resolved_engine_name(primary)
    gaming_selector = getattr(cfg, "gaming_engine", "") or ""
    if not gaming_selector or gaming_selector == primary_name:
        return DualSTTRegistry(
            primary=primary, primary_name=primary_name,
        )
    try:
        gaming = _build_engine_by_name(gaming_selector, cfg)
        logger.info(
            "Dual-STT: primary=%s, gaming=%s (swappable on gaming mode)",
            primary_name, gaming_selector,
        )
        return DualSTTRegistry(
            primary=primary, primary_name=primary_name,
            gaming=gaming, gaming_name=gaming_selector,
        )
    except Exception as e:                                    # noqa: BLE001
        logger.warning(
            "Dual-STT: gaming engine %r failed to load (%s); "
            "primary %s still active; gaming mode will skip STT swap",
            gaming_selector, e, primary_name,
        )
        return DualSTTRegistry(
            primary=primary, primary_name=primary_name,
        )


def _resolved_engine_name(engine) -> str:
    """Map a constructed engine instance back to its name."""
    if isinstance(engine, ParakeetEngine):
        return "parakeet"
    if isinstance(engine, MoonshineEngine):
        return "moonshine"
    if isinstance(engine, WhisperEngine):
        return "whisper"
    return type(engine).__name__.lower()


__all__ = [
    "make_stt_engine",
    "make_dual_stt_engines",
    "DualSTTRegistry",
    "STTEngine",
    "WhisperEngine",
    "ParakeetEngine",
    "MoonshineEngine",
    "is_nemo_available",
    "is_moonshine_available",
    "PARAKEET_INSTALL_HINT",
    "MOONSHINE_INSTALL_HINT",
]
