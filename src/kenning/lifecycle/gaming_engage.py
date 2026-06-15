"""Gaming-mode engage/disengage as a typed start-task state machine
(catalog 09 batch H wiring).

Replaces the prior synchronous ``_engage_extra`` / ``_disengage_extra``
callbacks in :mod:`kenning.pipeline.orchestrator` with an async-generator
that yields :class:`~kenning.lifecycle.start_task.StartTask` snapshots at
each substep. Driven by :func:`drive_start_task`, the orchestrator
receives a per-stage callback so it can speak a short voice ack between
substeps (e.g. "Stopping Parakeet... swapping language model...").

Sub-step coverage mirrors the legacy callbacks 1-to-1 so the actual
VRAM/RAM reclaim work is unchanged; the only difference is that the
state machine is observable -- each stage produces an audit row + a
voice ack opportunity. Every stage is fail-open: a failure logs WARN
and the state machine continues to the next stage (gaming mode is
purely additive -- a half-done engage still puts us in a better state
than the pre-engage configuration).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, AsyncIterator, Callable, Optional

from kenning.lifecycle.start_task import (
    StartTask,
    StartTaskStatus,
    create_start_task,
)

logger = logging.getLogger(__name__)


ENGAGE_TASK_NAME = "gaming_engage"
DISENGAGE_TASK_NAME = "gaming_disengage"


# ---------------------------------------------------------------------------
# Public dependency container -- lets callers (and tests) inject stubs
# ---------------------------------------------------------------------------


class GamingEngageDeps:
    """Bag of orchestrator-side hooks the state machine needs.

    Pulled out as a separate object so tests can construct a deps
    instance directly without standing up a real Orchestrator. Each
    field is optional -- the iterator skips substeps whose dependency
    is missing.
    """

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        tts: Optional[Any] = None,
        stt_registry: Optional[Any] = None,
        swap_stt_engine: Optional[Callable[[str], bool]] = None,
        get_vlm: Optional[Callable[[], Any]] = None,
        start_parakeet_server: Optional[Callable[..., Any]] = None,
        stop_parakeet_server: Optional[Callable[[], bool]] = None,
        gaming_llm_preset: str = "",
        gaming_llm_gpu_layers: Optional[int] = None,
        tts_kokoro_default_device: str = "cpu",
        tts_kokoro_engage_device: str = "cuda",
        llm_preset_holder: Optional[dict] = None,
        stt_name_holder: Optional[dict] = None,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.stt_registry = stt_registry
        self.swap_stt_engine = swap_stt_engine
        self.get_vlm = get_vlm
        self.start_parakeet_server = start_parakeet_server
        self.stop_parakeet_server = stop_parakeet_server
        self.gaming_llm_preset = gaming_llm_preset
        # Force the gaming LLM onto CPU (0) regardless of the env / config
        # gpu_layers override, so generation uses no GPU during a game. None
        # keeps config behaviour.
        self.gaming_llm_gpu_layers = gaming_llm_gpu_layers
        self.tts_kokoro_default_device = tts_kokoro_default_device
        # Device the Kokoro TTS runs on WHILE gaming. Default "cuda": keeping
        # the (tiny, ~330 MB) voice model on the GPU is what makes callouts
        # snappy AND frees the CPU so the audio-capture consumer never falls
        # behind (CPU saturation = dropped capture blocks = garbled STT). The
        # big VRAM saver is the 3B LLM on CPU; the voice model is not.
        self.tts_kokoro_engage_device = tts_kokoro_engage_device
        # Shared cells so engage can stash the pre-engage state and
        # disengage can read it. The orchestrator passes the same
        # dict references for both directions.
        self.llm_preset_holder = (
            llm_preset_holder if llm_preset_holder is not None else {"value": None}
        )
        self.stt_name_holder = (
            stt_name_holder if stt_name_holder is not None else {"value": None}
        )


# ---------------------------------------------------------------------------
# Engage state machine
# ---------------------------------------------------------------------------


async def gaming_engage_iterator(
    deps: GamingEngageDeps,
) -> AsyncIterator[StartTask]:
    """Yield :class:`StartTask` transitions for the gaming-engage flow.

    Stage order (mirrors the legacy synchronous callback so VRAM/RAM
    accounting is identical):

    1. ``SWAPPING_LLM`` -- hot-swap the LLM to the gaming preset
       (typically 3B abliterated).
    2. ``STOPPING_PARAKEET`` -- swap STT to gaming engine + kill the
       Parakeet HTTP server.
    3. ``MOVING_KOKORO`` -- flip Kokoro TTS engine to CPU.
    4. ``UNLOADING_VLM`` -- unload moondream2 if loaded.
    5. ``READY`` -- terminal state.

    Each substep is wrapped in try/except so a single failure doesn't
    short-circuit the rest of the engage cycle. The detail field is a
    short user-facing string the on_transition callback can speak.
    """

    task = create_start_task(
        ENGAGE_TASK_NAME,
        detail="engaging gaming mode",
    )
    yield task

    # ----- 1: LLM swap to gaming preset --------------------------------
    yield task.advance(
        StartTaskStatus.SWAPPING_LLM,
        detail="swapping language model",
        progress=0.1,
    )
    if deps.gaming_llm_preset and deps.llm is not None and hasattr(
        deps.llm, "reload_for_preset",
    ):
        try:
            from kenning.config import get_config

            current_preset = get_config().llm.preset
            if current_preset != deps.gaming_llm_preset:
                ok, msg = deps.llm.reload_for_preset(
                    deps.gaming_llm_preset,
                    gpu_layers=deps.gaming_llm_gpu_layers,  # 0 -> CPU for gaming
                )
                if ok:
                    deps.llm_preset_holder["value"] = current_preset
                    logger.info(
                        "gaming engage: LLM swapped %s -> %s",
                        current_preset, deps.gaming_llm_preset,
                    )
                else:
                    logger.warning(
                        "gaming engage: LLM swap to %s failed (%s); "
                        "keeping %s",
                        deps.gaming_llm_preset, msg, current_preset,
                    )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("gaming engage: LLM swap skipped (%s)", e)

    # ----- 2: STT swap + Parakeet shutdown -----------------------------
    yield task.advance(
        StartTaskStatus.STOPPING_PARAKEET,
        detail="stopping Parakeet",
        progress=0.4,
    )
    if (
        deps.stt_registry is not None
        and getattr(deps.stt_registry, "has_gaming", lambda: False)()
        and deps.swap_stt_engine is not None
    ):
        try:
            prior_stt = getattr(deps.stt_registry, "active_name", None)
            if deps.swap_stt_engine(deps.stt_registry.gaming_name):
                deps.stt_name_holder["value"] = prior_stt
                if deps.stop_parakeet_server is not None:
                    try:
                        if deps.stop_parakeet_server():
                            logger.info(
                                "gaming engage: Parakeet server stopped "
                                "(~700 MB VRAM freed)",
                            )
                    except Exception as e:                          # noqa: BLE001
                        logger.warning(
                            "gaming engage: Parakeet server stop "
                            "failed (%s); STT swap still in effect", e,
                        )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("gaming engage: STT swap skipped (%s)", e)

    # ----- 3: Kokoro TTS device move -----------------------------------
    yield task.advance(
        StartTaskStatus.MOVING_KOKORO,
        detail="moving voice engine",
        progress=0.7,
    )
    _engage_dev = getattr(deps, "tts_kokoro_engage_device", "cuda") or "cuda"
    if deps.tts is not None and hasattr(deps.tts, "move_to_device"):
        try:
            # Default "cuda": keep the voice model on the GPU so callouts stay
            # snappy and the CPU is free for audio capture + STT (CPU saturation
            # was dropping capture blocks -> garbled transcription + lag).
            deps.tts.move_to_device(_engage_dev)
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "gaming engage: Kokoro move-to-%s skipped (%s)", _engage_dev, e)

    # ----- 4: VLM unload -----------------------------------------------
    yield task.advance(
        StartTaskStatus.UNLOADING_VLM,
        detail="unloading vision model",
        progress=0.9,
    )
    if deps.get_vlm is not None:
        try:
            vlm = deps.get_vlm()
            if vlm is not None and getattr(vlm, "loaded", False):
                vlm.unload()
        except Exception as e:                                      # noqa: BLE001
            logger.warning("gaming engage: VLM unload skipped (%s)", e)

    # ----- 5: Terminal -------------------------------------------------
    yield task.advance(
        StartTaskStatus.READY,
        detail="gaming mode ready",
        progress=1.0,
    )


# ---------------------------------------------------------------------------
# Disengage state machine
# ---------------------------------------------------------------------------


async def gaming_disengage_iterator(
    deps: GamingEngageDeps,
) -> AsyncIterator[StartTask]:
    """Yield :class:`StartTask` transitions for the disengage flow.

    Stage order:

    1. ``MOVING_KOKORO`` -- restore Kokoro to its configured device.
    2. ``STOPPING_PARAKEET`` (reused enum, semantically "starting") --
       spawn the Parakeet HTTP server on a background thread and queue
       the STT swap-back for when /healthz reports ready.
    3. ``SWAPPING_LLM`` -- restore the LLM preset that was active
       before engage.
    4. ``READY``.
    """

    task = create_start_task(
        DISENGAGE_TASK_NAME,
        detail="disengaging gaming mode",
    )
    yield task

    # ----- 1: Kokoro TTS restore --------------------------------------
    yield task.advance(
        StartTaskStatus.MOVING_KOKORO,
        detail="restoring voice engine",
        progress=0.1,
    )
    if deps.tts is not None and hasattr(deps.tts, "move_to_device"):
        try:
            deps.tts.move_to_device(deps.tts_kokoro_default_device)
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "gaming disengage: Kokoro restore-to-%s skipped (%s)",
                deps.tts_kokoro_default_device, e,
            )

    # ----- 2: STT restore (spawn Parakeet on a daemon thread) ---------
    yield task.advance(
        StartTaskStatus.STOPPING_PARAKEET,
        detail="restarting Parakeet",
        progress=0.4,
    )
    prior_stt = deps.stt_name_holder["value"]
    if prior_stt is not None and deps.stt_registry is not None:
        if prior_stt == "parakeet" and deps.start_parakeet_server is not None:
            try:
                def _restore_when_ready() -> None:
                    try:
                        deps.start_parakeet_server(wait_for_ready=True)
                        if deps.swap_stt_engine is not None:
                            deps.swap_stt_engine(prior_stt)
                            logger.info(
                                "gaming disengage: Parakeet ready; "
                                "STT swapped back to %s", prior_stt,
                            )
                    except Exception as e:                          # noqa: BLE001
                        logger.warning(
                            "gaming disengage: Parakeet restore "
                            "failed (%s); staying on gaming engine", e,
                        )

                threading.Thread(
                    target=_restore_when_ready,
                    daemon=True,
                    name="parakeet-restore",
                ).start()
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "gaming disengage: failed to spawn restore "
                    "thread (%s)", e,
                )
        elif deps.swap_stt_engine is not None:
            try:
                deps.swap_stt_engine(prior_stt)
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "gaming disengage: STT swap-back skipped (%s)", e,
                )
        deps.stt_name_holder["value"] = None

    # ----- 3: LLM preset restore --------------------------------------
    yield task.advance(
        StartTaskStatus.SWAPPING_LLM,
        detail="restoring language model",
        progress=0.7,
    )
    prior_preset = deps.llm_preset_holder["value"]
    if prior_preset is not None and deps.llm is not None and hasattr(
        deps.llm, "reload_for_preset",
    ):
        try:
            ok, msg = deps.llm.reload_for_preset(prior_preset)
            if ok:
                logger.info(
                    "gaming disengage: LLM restored to %s",
                    prior_preset,
                )
            else:
                logger.warning(
                    "gaming disengage: LLM restore to %s failed (%s); "
                    "stuck on gaming preset",
                    prior_preset, msg,
                )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("gaming disengage: LLM restore skipped (%s)", e)
        deps.llm_preset_holder["value"] = None

    # ----- 4: Terminal -------------------------------------------------
    yield task.advance(
        StartTaskStatus.READY,
        detail="back to standby",
        progress=1.0,
    )
