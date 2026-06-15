"""Main event loop.

The orchestrator owns every component and runs the state machine:

    IDLE
      └─ wake word fires ──► CAPTURING
                                └─ VAD end-of-speech ──► PROCESSING
                                                             ├─ Whisper
                                                             ├─ LLM (streaming)
                                                             └─ TTS (streaming)
                                                                 │
                                                                 │ wake word
                                                                 │ during TTS
                                                                 ▼
                                                              CAPTURING (next turn)
                                                                 │
                                                                 │ TTS done
                                                                 ▼
                                                          FOLLOW_UP_LISTENING
                                                          (no wake word required;
                                                           VAD-bounded; LLM gates
                                                           each utterance; 30 s
                                                           silence drops to IDLE)

Three threads matter:
- The audio thread (inside :class:`AudioCapture`) only enqueues chunks.
- The orchestrator's own thread does everything else.
- During TTS playback an *interrupt watcher* thread runs the wake-word
  detector for barge-in.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from enum import Enum
from typing import Optional, Union

import numpy as np

from config import settings
from kenning.addressing import AddressingClassifier, AddressingDecision
from kenning.audio import (
    AudioCapture,
    RingBuffer,
    VoiceActivityDetector,
    WakeWordDetector,
)
from kenning.audio.smart_turn import (
    SMART_TURN_SAMPLE_RATE,
    SmartTurnDetector,
    SmartTurnVerdict,
    build_detector_from_config,
)
from kenning.audio.vad import SpeechEvent
from kenning.llm import LLMEngine
from kenning.transcription import (
    DualSTTRegistry,
    WhisperEngine,
    make_dual_stt_engines,
    make_stt_engine,
)
from kenning.tts import make_tts_engine  # noqa: F401 — kept import-time wired
from kenning.utils.logging import get_logger
# kenning.coding (+ its mcp_server / coordinator / narration / voice submodules,
# and the OpenClaw bridge they transitively pull) is imported LAZILY inside the
# GATED coding load-methods below -- NOT at module top -- so a LEAN GAMING BOOT
# never loads the coding stack into RAM (anticheat surface). The boot
# _audit_anticheat_posture lean canary asserts it stays out. See
# feedback-no-default-load-anticheat.


def _voice_text(text: str) -> "VoiceResponse":   # noqa: F821 (lazy; PEP 563 str)
    """Wrap a plain string in a VoiceResponse so it can flow through
    :meth:`Orchestrator._handle_capability_response`."""
    from kenning.coding.voice import VoiceResponse
    return VoiceResponse(text=text, handled=True)

from kenning.uncertainty import apply as apply_uncertainty
from kenning.conversational_ack import (
    ConversationalAckSource,
    is_conversational_ack_eligible,
)
from kenning.response_style import apply_brevity_hint
from kenning.safety.validator import (
    build_validator_from_config as _build_safety_validator_from_config,
    set_validator as _set_safety_validator,
)
from kenning.web_search import (
    AcknowledgmentSource,
    BraveSearchClient,
    GateDecision,
    GateVerdict,
    JinaReaderClient,
    WebResultsCache,
    WebSearchExecutor,
    WebSearchGate,
    format_sources_for_prompt,
    format_sources_for_transcript,
)
from kenning.observations import observe_llm_thinking_drift_sample

logger = get_logger("pipeline.orchestrator")


def _drive_async_blocking(coro):
    """Run a coroutine to completion from a SYNC context, regardless of whether
    an event loop is already running on this thread.

    The gaming-mode engage/disengage device-swap driver is invoked from inside
    ``manager.engage()``'s ``finally`` (an ``on_engaged`` callback) -- and every
    caller wraps that in ``asyncio.run(manager.engage())``. A bare
    ``asyncio.run`` here would then raise "asyncio.run() cannot be called from a
    running event loop" and the swaps would silently no-op (LLM/TTS never move
    to CPU). When a loop is already running we instead drive the coroutine on a
    fresh loop in a short-lived thread and join, so the swaps actually happen.
    """
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if not running:
        return asyncio.run(coro)
    box: dict = {}

    def _runner():
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=_runner, name="async-driver", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


class State(Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    FOLLOW_UP_LISTENING = "follow_up"


# Sentinel values returned by :meth:`Orchestrator._follow_up_listen`.
_FU_TIMEOUT = "timeout"
_FU_WAKE = "wake"

# LIVE STT FIX (2026-06-14): the wake word "Ultron" is frequently mis-transcribed
# and PREPENDED to the command when the user says it in one breath -- Moonshine
# renders it "Run," / "Ron," / "Tron" / "front" / "One" / "Ultra" + an optional
# connector ("to"/"and"/"then"). The leading remnant breaks the strict relay /
# Spotify matchers (which the corpus harness only ever feeds CLEAN text), so every
# command fell through to the conversational LLM. This strips that remnant. It is
# applied as a FALLBACK ONLY (retry after the clean text fails to match), so it
# can never corrupt a legitimate match and the harness is unaffected.
# Mis-transcribed "Ultron" tokens + harmless leading fillers/false-starts that
# Moonshine prepends ("Yeah.", "So", "Um"). Either breaks the strict matchers.
_WAKE_MISHEAR = (
    r"(?:ultron|ultra(?:n|m)?|altron|all[\s-]*tron|"
    r"run|ron|tron|trond|front|fron|one|won|wun|ulton|olt?ron|elt?ron|"
    # "Ultron" is also frequently heard as "to/too/two" before a verb
    # ("to introduce yourself" = "Ultron, introduce yourself").
    r"to|too|two|"
    r"yeah|yep|yes|yup|ok|okay|so|well|um|uh|hey|alright|nah|now)"
)
# NB: the \b after the mishear is essential -- without it "to" would eat the
# front of "tomorrow"/"today" and "run" the front of "running". With it, only a
# whole leading token is stripped.
_WAKE_REMNANT_RE = re.compile(
    r"^\s*" + _WAKE_MISHEAR + r"\b[\s.,!?:;]*"
    r"(?:(?:to|and|then|and\s+then|um|uh)[\s,]+)?",
    re.IGNORECASE,
)


def _strip_leading_wake_remnant(text: str) -> str:
    """Strip leading mis-transcribed wake words / fillers ("Yeah. Run, …" -> "…").

    Iterates so a stacked prefix ("Yeah." + "Run,") is fully removed. Returns
    ``text`` unchanged when there is nothing to strip or stripping would empty
    it. Used as a FALLBACK only (retry after the clean text fails to match), so
    it can never corrupt a legitimate match.
    """
    if not text:
        return text
    cur = text
    for _ in range(3):                       # at most a few stacked prefixes
        m = _WAKE_REMNANT_RE.match(cur)
        if not m or m.end() == 0:
            break
        rest = cur[m.end():].lstrip()
        if not rest:
            break
        cur = rest
    return cur


class Orchestrator:
    """Wires up audio → wake → VAD → STT → LLM → TTS.

    Components are constructed eagerly so cold-start cost is paid up-front
    rather than on the first wake-word trigger.
    """

    # Legacy class-level default. Config-driven via ``vad.max_utterance_seconds``;
    # the instance attribute ``_max_utterance_seconds`` wins at runtime. Kept
    # for callers that reference the constant directly. 2026-05-11 follow-up
    # fix raised the production default from 15 s -> 30 s after a real session
    # got cut off mid-sentence at the 15 s ceiling.
    MAX_UTTERANCE_SECONDS = 30.0

    def __init__(self) -> None:
        # 2026-05-22 -- diagnostic-only startup pass: log every
        # KENNING_* env var that's set + the effective values of the
        # high-impact config sections. Catches stale-env-override
        # bugs (the KENNING_LLM_MODEL_PATH=9B silent override that ate
        # hours during the 4B migration) within 5 seconds of startup
        # instead of after a confused live session. Fail-open --
        # logging errors don't block construction.
        try:
            from kenning.config import log_effective_config
            log_effective_config()
        except Exception as e:                                      # noqa: BLE001
            logger.warning("effective-config log failed: %s", e)
        # DIAGNOSTICS DEFAULT-OFF (2026-06-14): clear the live diagnostics
        # sentinel at every boot so spoken-audio monitoring NEVER auto-persists
        # across a restart. Testing mode is an explicit post-boot opt-in
        # (the operator re-touches ~/.kenning/audio_diagnostics_on); a manual
        # restart always comes up with monitoring OFF.
        try:
            from kenning.diagnostics import reset_for_new_session
            reset_for_new_session()
        except Exception as e:                                      # noqa: BLE001
            logger.debug("diagnostics reset skipped (%s)", e)

        # 2026-05-22 -- session-scoped fail-open counter. Tracks
        # how many fail-open paths fired during the session
        # (reranker fall-through, supervisor exceptions, bus slow
        # subscribers, etc.) so a spike between sessions surfaces a
        # regression that would otherwise stay quiet. Configure with
        # the JSONL log path, then log the previous session's summary.
        try:
            from kenning.config import LOGS_DIR
            from kenning.resilience import fail_open_log
            from kenning.bus import set_slow_subscriber_recorder
            fail_open_log.configure(LOGS_DIR / "fail_open_counts.jsonl")
            # Hook the bus's slow-subscriber recorder so it
            # contributes to the shared counter alongside other
            # subsystems.
            set_slow_subscriber_recorder(fail_open_log.record)
            previous = fail_open_log.previous_session_counts()
            logger.info(
                "fail-open: previous session: %s",
                fail_open_log.render_summary(previous),
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("fail_open_log init failed: %s", e)

        # 2026-05-22 -- subscribe project-introspect cache invalidator
        # to CodingFileChangedEvent on the bus. Without this, the
        # 30 s TTL on snapshot() means rapid-iteration sessions can
        # see stale file trees when Claude writes a new file and the
        # supervisor immediately re-routes. Fail-open: bus unavailable
        # or import error leaves the cache alive (manual invalidation
        # still works).
        if not self._skip_for_lean_gaming("barebones_skip_coding"):
            try:
                from kenning.coding.project_introspect import install_bus_invalidator
                install_bus_invalidator()
            except Exception as e:                                  # noqa: BLE001
                logger.debug("project_introspect bus invalidator init: %s", e)

        # 2026-05-26 (openclaw-clawhub catalog wiring) -- materialise
        # the 5 KENNING_DEFAULT_PINS into the workdir lockfile at
        # startup. Formalises the binding voice-baseline lock contract
        # (SOUL.md / Kokoro voicepack / qwen3.5-4b / IDENTITY.md / K
        # validator rules) into data the install / update / hot-swap
        # paths can refuse against. Idempotent: re-running on every
        # startup is a no-op when the lockfile already has the
        # defaults. Fail-open: a broken lockfile must not block
        # orchestrator construction.
        try:
            from kenning.config import PROJECT_ROOT
            from kenning.install.pin import materialise_default_pins
            results = materialise_default_pins(PROJECT_ROOT)
            new_pins = [r.slug for r in results if not r.idempotent_noop]
            if new_pins:
                logger.info(
                    "voice-baseline default pins materialised: %s",
                    ", ".join(new_pins),
                )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("materialise_default_pins failed (continuing): %s", e)

        # 2026-05-26 (openclaw-clawhub catalog wiring) -- T2 voice-
        # baseline artifact identity verification. Walks the 6
        # canonical voice-baseline artifacts (LLM GGUF, draft GGUF,
        # Kokoro voicepack, Kokoro fine-tune weights, wake-word
        # ONNX, Smart Turn V3 ONNX), computes their digests, and
        # verifies against data/install/pinned_digests.jsonl.
        # First-fetch records TOFU pins; subsequent boots verify
        # against the recorded pins so a tampered model file
        # surfaces as a voice-baseline integrity warning rather
        # than a mysterious voice-character regression.
        #
        # Runs ASYNC on a daemon thread so the ~5-10 s GGUF hash
        # doesn't block cold-start. Report deposited on
        # ``self._voice_baseline_report`` for downstream consumers
        # (system_status MCP tool, voice intent "are my models OK?")
        # to poll on demand. Fail-open: digest computation errors
        # never abort startup.
        self._voice_baseline_report = None
        try:
            from kenning.config import PROJECT_ROOT
            from kenning.install.voice_baseline_verify import (
                summarise_report,
                verify_voice_baseline_artifacts_async,
            )

            def _on_voice_baseline_complete(report) -> None:
                try:
                    summary = summarise_report(report)
                    if report.mismatches:
                        logger.warning(
                            "voice-baseline integrity: %s -- mismatches: %s",
                            summary,
                            "; ".join(
                                f"{o.identifier}: {o.detail}"
                                for o in report.mismatches
                            ),
                        )
                    elif report.missing_required:
                        logger.warning(
                            "voice-baseline integrity: %s -- required missing: %s",
                            summary,
                            ", ".join(
                                o.identifier for o in report.missing_required
                            ),
                        )
                    else:
                        logger.info("voice-baseline integrity: %s", summary)
                except Exception as inner:  # noqa: BLE001
                    logger.debug(
                        "voice-baseline summary log failed: %s", inner,
                    )

            self._voice_baseline_report, _ = (
                verify_voice_baseline_artifacts_async(
                    PROJECT_ROOT,
                    on_complete=_on_voice_baseline_complete,
                )
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "voice-baseline verify async kickoff failed: %s", e,
            )

        self.audio = AudioCapture()
        # Mode-aware pre-roll: ring buffer is sized to the LARGER of
        # cold/warm pre-roll so the WARM path can take a longer slice
        # while the COLD path takes a shorter one. The defaults keep
        # legacy behaviour: ring_buffer_seconds=0.5 was the size before
        # the 2026-05-09 audio pass, and the COLD-mode shortening
        # (cold_pre_roll_seconds=0.15) was that pass's anti-Tron-prefix
        # fix -- which inadvertently clipped the leading word in WARM
        # mode follow-ups. Mode-aware slicing restores both behaviours.
        try:
            from kenning.config import get_config
            _cfg = get_config()
            _audio_cfg = _cfg.audio
            self._cold_pre_roll_seconds = float(_audio_cfg.cold_pre_roll_seconds)
            self._warm_pre_roll_seconds = float(_audio_cfg.warm_pre_roll_seconds)
            ring_capacity_seconds = max(
                float(_audio_cfg.ring_buffer_seconds),
                self._cold_pre_roll_seconds,
                self._warm_pre_roll_seconds,
            )
            # Adaptive end-of-turn (2026-05-11): when speech has been
            # going for this long, bump VAD silence requirement so a
            # thinking pause mid-sentence doesn't cut the capture
            # short. Short utterances stay snappy.
            _vad_cfg = _cfg.vad
            self._long_utterance_threshold_seconds = float(
                _vad_cfg.long_utterance_threshold_seconds
            )
            self._long_utterance_silence_duration_ms = int(
                _vad_cfg.long_utterance_silence_duration_ms
            )
            # 2026-05-11 follow-up fix: hard ceiling on a single
            # VAD-bounded capture is now configurable. Legacy class
            # constant was 15 s; a real session got cut off
            # mid-sentence at that ceiling. Default raised to 30 s.
            self._max_utterance_seconds = float(_vad_cfg.max_utterance_seconds)
            # 2026-05-12 Smart Turn V3 semantic end-of-turn config.
            # The detector itself is constructed below (after the
            # try/except block) once we know whether the model file
            # is present; cache the policy knobs here so both paths
            # below see them.
            _smart_turn_cfg = getattr(_vad_cfg, "smart_turn", None)
            if _smart_turn_cfg is not None and bool(_smart_turn_cfg.enabled):
                self._smart_turn_cfg = _smart_turn_cfg
                self._smart_turn_window_seconds = float(_smart_turn_cfg.window_seconds)
                self._smart_turn_fast_path_silence_ms = int(
                    _smart_turn_cfg.fast_path_silence_duration_ms
                )
                self._smart_turn_incomplete_extension_ms = int(
                    _smart_turn_cfg.incomplete_extension_ms
                )
                # 2026-05-16 latency pass 2: gradient-fire knobs.
                self._smart_turn_completion_threshold = float(
                    _smart_turn_cfg.completion_threshold
                )
                self._smart_turn_early_completion_threshold = float(
                    getattr(_smart_turn_cfg, "early_completion_threshold", 0.65)
                )
                self._smart_turn_medium_grace_ms = int(
                    getattr(_smart_turn_cfg, "medium_grace_ms", 200)
                )
            else:
                self._smart_turn_cfg = None
                self._smart_turn_window_seconds = 8.0
                self._smart_turn_fast_path_silence_ms = 500
                self._smart_turn_incomplete_extension_ms = 700
                self._smart_turn_completion_threshold = 0.5
                self._smart_turn_early_completion_threshold = 0.65
                self._smart_turn_medium_grace_ms = 200
        except Exception:
            # Defensive: tests / scripts may construct Orchestrator
            # without a fully built config. Fall back to the shim.
            self._cold_pre_roll_seconds = float(settings.RING_BUFFER_SECONDS)
            self._warm_pre_roll_seconds = float(settings.RING_BUFFER_SECONDS)
            ring_capacity_seconds = float(settings.RING_BUFFER_SECONDS)
            self._long_utterance_threshold_seconds = 8.0
            self._long_utterance_silence_duration_ms = 2400
            self._max_utterance_seconds = float(self.MAX_UTTERANCE_SECONDS)
            self._smart_turn_cfg = None
            self._smart_turn_window_seconds = 8.0
            self._smart_turn_fast_path_silence_ms = 500
            self._smart_turn_incomplete_extension_ms = 700
            self._smart_turn_completion_threshold = 0.5
            self._smart_turn_early_completion_threshold = 0.65
            self._smart_turn_medium_grace_ms = 200
        self.ring = RingBuffer(
            int(ring_capacity_seconds * settings.SAMPLE_RATE)
        )
        # 2026-05-12 Phase 2 -- runtime tool-call validator. Pairs with
        # the abliterated default LLM (Josiefied-Qwen3-8B): the model can
        # ask for anything; the validator decides what actually runs.
        # Constructed here and pushed into the module singleton so call
        # sites (OpenClaw dispatcher, coding bridge, MCP tools) read it
        # via :func:`kenning.safety.get_validator`. Fail-open: if
        # construction raises (missing config / package not installed /
        # etc.), the singleton stays at the permissive no-op default
        # and a WARN is logged on first use.
        try:
            self.safety_validator = _build_safety_validator_from_config()
            _set_safety_validator(self.safety_validator)
            logger.info(
                "safety validator initialised: %d rules registered "
                "(safety.enabled=%s)",
                len(self.safety_validator.rules),
                self.safety_validator.policy.enabled,
            )
            # Heal + verify the tamper-evident audit chain at startup. An
            # unclean shutdown (kill between record()'s write and fsync) leaves
            # a truncated final line that breaks the chain; repair_if_needed()
            # truncates only that never-committed tail (archiving, never
            # deleting, if nothing is salvageable). Fail-open: never blocks boot.
            try:
                audit_log = getattr(self.safety_validator, "audit_log", None)
                if audit_log is not None:
                    verdict = audit_log.repair_if_needed()
                    if verdict == "repaired":
                        logger.info(
                            "safety audit log repaired: truncated a partial/"
                            "corrupt tail from a prior unclean shutdown; chain "
                            "intact from genesis")
                    elif verdict == "restarted":
                        logger.warning(
                            "safety audit log had no valid prefix -- archived as "
                            ".corrupt.<ts> and restarted the chain from genesis")
            except Exception as e2:                                  # noqa: BLE001
                logger.debug("audit chain repair skipped (%s)", e2)
        except Exception as e:
            self.safety_validator = None
            logger.warning(
                "safety validator construction failed (%s); call sites "
                "will see the permissive no-op validator", e,
            )
        # Catalog 09 batch A wiring: kick off the DialogPoller daemon
        # thread so DialogAppearedEvent / DialogResolvedEvent actually
        # fire on the bus during the orchestrator's lifetime. The
        # coding bridge (batch B) and any other consumer subscribe via
        # ``kenning.bus.subscribe(DialogAppearedEvent, ...)``. The
        # poller itself is fail-open: missing pywinauto / off-Windows /
        # any tick failure logs DEBUG and the daemon stays alive.
        # Default ON so the wiring shipped this session actually runs;
        # operators can short-circuit by stopping the singleton at
        # runtime via :func:`kenning.desktop.dialog_poller.set_dialog_poller(None)`.
        self._start_dialog_poller()
        # Anticheat-safe mode surface hooks (2026-06-11): activating the
        # mode must physically STOP running desktop subsystems, not just
        # gate their calls -- the UIA dialog-poller thread keeps polling
        # otherwise, and the cached mss capture singleton keeps holding
        # GDI handles. On deactivate the poller restarts and singletons
        # rebuild lazily on next use. Fail-open.
        try:
            from kenning.safety.anticheat import register_surface_hook

            def _anticheat_dialog_poller(active: bool) -> None:
                poller = getattr(self, "_dialog_poller", None)
                if poller is None:
                    return
                if active:
                    poller.stop()
                elif not poller.running:
                    poller.start()

            def _anticheat_capture_singletons(active: bool) -> None:
                if not active:
                    return  # rebuilt lazily on next (post-mode) use
                # NEVER import the desktop stack just to clear singletons -- that
                # import would pull pyautogui + mss into RAM under anticheat. Only
                # release them if the modules are ALREADY loaded (i.e. desktop was
                # used before the mode flip); if anticheat is on from boot they
                # were never loaded and there is nothing to clear.
                import sys as _sys

                cap = _sys.modules.get("kenning.desktop.capture")
                if cap is not None:
                    cap.set_screen_capture(None)
                seq = _sys.modules.get("kenning.desktop.sequence")
                if seq is not None:
                    seq.set_sequence_runner(None)

            register_surface_hook("dialog_poller", _anticheat_dialog_poller)
            register_surface_hook(
                "capture_singletons", _anticheat_capture_singletons,
            )
            # Config-pinned anticheat (gaming_mode.anticheat_safe_mode):
            # apply at startup so the hooks fire and running surfaces
            # (the poller just started above) are stopped immediately --
            # the guard flag alone wouldn't stop already-running threads.
            from kenning.safety.anticheat import (
                anticheat_active,
                set_anticheat_active,
            )

            if anticheat_active():
                set_anticheat_active(True, "pinned by config at startup")
        except Exception as e:                                       # noqa: BLE001
            logger.warning("anticheat surface hooks not registered: %s", e)
        # ANTICHEAT IMPORT FIREWALL -- the loader-level backstop. Boot gates +
        # the per-dispatch refusal guards stop the KNOWN paths; this stops the
        # rest: a sys.meta_path finder that refuses ANY import of a desktop /
        # browser / input / capture / automation module while anticheat-safe
        # mode is active, so no lazy/conditional import anywhere can pull such a
        # module (or its transitive pyautogui/mss/pywinauto/playwright) into the
        # process. Installed unconditionally -- it is a no-op while the mode is
        # off, so it also covers a mid-session "enable anticheat mode" toggle.
        try:
            from kenning.safety.import_firewall import install_import_firewall
            install_import_firewall()
        except Exception as e:                                       # noqa: BLE001
            logger.warning("anticheat import firewall NOT installed: %s", e)
        # T23 (cline) / T12 (OpenClaw): start the subprocess reaper so every
        # Kenning-spawned subprocess (Parakeet / XTTS daemons, the coding-bridge
        # claude subprocess) is tracked in one registry, heavy long-runners are
        # RSS-warned, and a wedged non-daemon is reaped past a generous backstop.
        # Daemons register persistent (never auto-killed); legit coding turns
        # finish + unregister long before the backstop. Fail-open.
        try:
            from kenning.subprocess.zombie_killer import get_zombie_killer
            self._zombie_killer = get_zombie_killer()
            self._zombie_killer.start()
            logger.info("ZombieKiller subprocess reaper started")
        except Exception as e:                                       # noqa: BLE001
            self._zombie_killer = None
            logger.warning("ZombieKiller startup skipped (%s)", e)
        # Embedder sidecar for the semantic command router -- a SEPARATE process
        # in an ISOLATED venv so the embedding model NEVER loads into THIS
        # anticheat-pinned process. Spawned EARLY so EmbeddingGemma (~20s) loads
        # in PARALLEL with the rest of boot. No-op when disabled / backend=lexical
        # / venv missing; fail-open (the router falls back to the lexical backend).
        self._embedder_sidecar_proc = None
        self._embedder_sidecar_reuse_pid = None   # a deliberately-reused sidecar we still OWN for cleanup
        try:
            self._start_embedder_sidecar()
        except Exception as e:                                       # noqa: BLE001
            logger.warning("embedder sidecar start skipped (%s)", e)
        # T22 MCP client (default OFF). When ``mcp.enabled``, build the registry
        # from config -- every server registered with its transport env/header-
        # sanitised + the real spawn (process-registry + zombie-killer tracked)
        # and kill_process_tree reaper wired. When ``mcp.autostart``, also spawn
        # stdio servers now (off by default: a stdio server with no in-process
        # JSON-RPC client just idles). Nothing runs unless enabled. Fail-open.
        self._mcp_registry = None
        try:
            from kenning.config import get_config as _gc
            _mcp_cfg = getattr(_gc(), "mcp", None)
            if _mcp_cfg is not None and getattr(_mcp_cfg, "enabled", False):
                from kenning.mcp.builder import build_mcp_server_registry
                self._mcp_registry = build_mcp_server_registry(_mcp_cfg)
                if self._mcp_registry is not None:
                    refs = self._mcp_registry.list_registered()
                    logger.info("MCP client: %d server(s) registered", len(refs))
                    if getattr(_mcp_cfg, "autostart", False):
                        for ref in refs:
                            self._mcp_registry.start(ref.server_id)
        except Exception as e:                                       # noqa: BLE001
            self._mcp_registry = None
            logger.warning("MCP client startup skipped (%s)", e)
        self.wake = WakeWordDetector()
        # 2026-05-12 Smart Turn V3: build the detector BEFORE the VAD so
        # we can wire the fast-path silence baseline into the VAD's
        # construction when smart-turn is active. Missing model file
        # / disabled config returns None and the VAD falls back to its
        # legacy silence requirement (typically 1200 ms via config).
        self.smart_turn = self._build_smart_turn_detector()
        if self.smart_turn is not None:
            # Smart Turn confirms or rejects the early SPEECH_END, so
            # we can collapse the silence requirement to the fast-path
            # value and rely on the model to catch trailed-off cases.
            self.vad = VoiceActivityDetector(
                min_silence_ms=self._smart_turn_fast_path_silence_ms,
            )
            logger.info(
                "Smart Turn V3 active: VAD min_silence_duration set to %d ms "
                "(extension on incomplete: %d ms)",
                self._smart_turn_fast_path_silence_ms,
                self._smart_turn_incomplete_extension_ms,
            )
        else:
            self.vad = VoiceActivityDetector()
        # 2026-05-21 frontier-enhancement Item 5: STT engine selection
        # via ``stt.engine`` config (``auto`` / ``whisper`` / ``parakeet``).
        # ``auto`` picks Parakeet if NeMo is installed, else Whisper.
        # *** IF VOICE TRANSCRIPTION REGRESSES, SUSPECT THIS FIRST. ***
        # Roll back with ``stt.engine: whisper`` to confirm whether the
        # regression is Parakeet-specific before chasing other causes.
        # 2026-05-22 dual-STT for gaming mode. The registry holds the
        # primary (standby) engine + optional gaming engine. ``self.stt``
        # is the active pointer; gaming mode flips it via
        # :meth:`swap_stt_engine`. When ``stt.gaming_engine`` is empty
        # or matches the primary, the registry has only the primary and
        # the swap method is a no-op.
        self._stt_registry: DualSTTRegistry = make_dual_stt_engines()
        self.stt = self._stt_registry.active
        # 2026-05-22 streaming STT: warm the engine's session before the
        # first real turn so the cold-start cost (ONNX session JIT +
        # tokenizer load) doesn't land in the user's first interaction.
        # Fail-open: missing ``warmup`` attribute or any failure is
        # silent; the first turn just pays the cold cost.
        try:
            warmup = getattr(self.stt, "warmup", None)
            if warmup is not None:
                warmup()
        except Exception as e:                                          # noqa: BLE001
            logger.debug("STT warmup skipped (%s)", e)
        # Latency hygiene: bump the host process to Above-Normal so background
        # scheduling can't starve the voice loop mid-turn (~50-200 ms of jitter
        # eliminated under load). Fail-open: no psutil / no permission -> the
        # helper logs DEBUG and returns without raising.
        try:
            from kenning.latency_hygiene import raise_process_priority
            raise_process_priority()
        except Exception as e:                                          # noqa: BLE001
            logger.debug("process-priority raise skipped (%s)", e)
        # 2026-06-11: SearxNG (default first search provider) runs in a
        # Docker container -- if Docker is down at boot every search
        # silently falls through to Brave. Probe once and launch Docker
        # Desktop in the background if it's unreachable. Fully fail-open;
        # gated by web_search.searxng.autostart_docker_on_boot.
        try:
            from kenning.config import get_config

            _cfg = get_config()
            _sx = _cfg.web_search.searxng
            if self._skip_for_lean_gaming("barebones_skip_docker_autostart"):
                logger.info("lean gaming boot: Docker autostart skipped "
                            "(web search is not used while gaming)")
            elif _cfg.web_search.enabled and getattr(
                _sx, "autostart_docker_on_boot", False,
            ):
                from kenning.lifecycle.docker_startup import ensure_docker_running
                ensure_docker_running(
                    base_url=_sx.base_url,
                    enabled=True,
                    docker_executable_path=getattr(
                        _cfg.gaming_mode, "docker_executable_path", None,
                    ),
                )
        except Exception as e:                                          # noqa: BLE001
            logger.debug("docker autostart skipped (%s)", e)
        self.memory = self._load_memory_if_enabled()
        # 2026-05-22 perf: warm the cross-encoder reranker shared
        # singleton at startup. Without this, the first turn that
        # exercises memory retrieval OR web-search snippet ranking
        # pays a ~2 s cold model load on the user's first interaction.
        # Production-hardening: warm on a DAEMON THREAD so the ~2 s
        # pure-CPU sentence-transformers load overlaps the GPU model
        # loads below instead of serialising in front of them (the
        # reranker shares no state with the LLM / TTS / STT loads, and
        # every consumer already lazy-loads on miss, so a still-warming
        # or failed thread degrades to the pre-existing lazy path).
        def _warm_reranker() -> None:
            try:
                from kenning.memory.reranker import get_shared_reranker
                t0 = time.monotonic()
                shared = get_shared_reranker()
                # Touch the model so the underlying sentence-transformers
                # ``CrossEncoder`` is loaded into memory + ONNX-compiled.
                shared._ensure_model()
                logger.info(
                    "Cross-encoder reranker warmed in %.2fs (background)",
                    time.monotonic() - t0,
                )
            except Exception as e:                                      # noqa: BLE001
                logger.debug("Reranker warmup skipped (%s)", e)

        if self._skip_for_lean_gaming("barebones_skip_reranker_warmup"):
            logger.info("lean gaming boot: cross-encoder reranker warmup skipped "
                        "(RAG retrieval + web-search ranking are off while gaming;"
                        " it lazy-loads on demand if ever needed)")
        else:
            try:
                threading.Thread(
                    target=_warm_reranker, name="reranker-warm", daemon=True,
                ).start()
            except Exception as e:                                      # noqa: BLE001
                logger.debug("Reranker warmup thread skipped (%s)", e)
        self.llm = LLMEngine(memory=self.memory)
        # Latency hygiene: warm the LLM so the first real turn doesn't pay the
        # cold-context prefill (~100-200 ms of TTFT shaved off the user's first
        # interaction). SYNCHRONOUS, not a daemon thread: llama-cpp's single
        # context is not safe for concurrent generation, so a background warmup
        # racing the first real turn could corrupt it. ``record_history=False``
        # keeps the warmup turn out of conversation history; fail-open -- any
        # error is swallowed and the first turn just pays the cold cost as
        # before. The proven generate_stream(record_history=False) path is the
        # same one the e2e harness + speculative-LLM path use.
        try:
            from kenning.latency_hygiene import warmup_llm
            warmup_llm(
                lambda p: list(self.llm.generate_stream(
                    p, record_history=False, enable_thinking=False,
                )),
            )
            # Guardrail brake (#15+#65): discard the warmup stream's TTFT --
            # it measures the cold prefill, not a representative turn, and
            # must not seed the evolution metrics ring.
            pop_ttft = getattr(self.llm, "pop_last_ttft_ms", None)
            if callable(pop_ttft):
                pop_ttft()
        except Exception as e:                                          # noqa: BLE001
            logger.debug("LLM warmup skipped (%s)", e)
        # 2026-05-10 voice swap: select TTS engine via ``tts.engine`` config.
        # ``"piper_rvc"`` (default) keeps the legacy Piper + RVC stack;
        # ``"xtts_v3"`` swaps in the XTTS v2 streaming + v3 Kenning filter
        # stack. The engines share the same ``speak`` / ``speak_stream``
        # / ``warmup`` / ``stop`` interface so the orchestrator's
        # downstream playback path (the producer-signaled lookahead in
        # speak_stream) doesn't change.
        self.rvc, self.tts = self._load_tts_engine()
        self.tts.warmup()
        # 2026-06-12: wire the optional broadcast mirror (a second output that
        # tees EVERY spoken line -- conversation AND relay -- to a separate,
        # OBS-capturable device for stream viewers). No-op when
        # ``audio.broadcast_device`` is unset; never blocks the speaker path.
        try:
            from kenning.audio.broadcast import configure_from_config

            configure_from_config()
        except Exception as e:                                       # noqa: BLE001
            logger.debug("broadcast mirror configure skipped (%s)", e)
        # 2026-06-14: local monitor -- tee relay/team callouts to the user's OWN
        # default speakers (relay otherwise plays only on the mic B-bus, so the
        # user can't hear their own callouts). Gated by relay_speech.echo_to_user.
        try:
            from kenning.audio.monitor import (
                configure_from_config as _configure_monitor,
            )

            _configure_monitor()
        except Exception as e:                                       # noqa: BLE001
            logger.debug("local monitor configure skipped (%s)", e)
        # 2026-06-12: bring up the optional voice waveform overlay (a separate
        # borderless OBS-capturable window that visualizes EVERY spoken line --
        # conversation AND relay). Off unless ``visualizer.enabled``; fail-open
        # (no display/Tk -> just never appears), never blocks the voice path.
        try:
            from kenning.audio.waveform import configure_from_config as _viz_cfg

            _viz_cfg()
        except Exception as e:                                       # noqa: BLE001
            logger.debug("waveform overlay configure skipped (%s)", e)
        # 2026-05-24 OpenHands batch 8 (T6) -- install the default STT +
        # TTS injectors on the module-level registry. The closures hand
        # back the already-built engines so callers that go through the
        # registry (gaming-mode hot-swap, tests, future per-mode
        # routing) get the same instances the orchestrator owns -- no
        # parallel construction. Fail-open: any error logs WARN and
        # leaves the registry unset; callers that pre-date the registry
        # path keep working untouched.
        try:
            from kenning.services.injector import install_default_injectors
            from kenning.services.engine_injectors import (
                STTEngineInjector,
                TTSEngineInjector,
            )

            stt_engine = self.stt
            rvc_handle = self.rvc
            tts_engine = self.tts

            def _stt_standby_factory(_state):
                return stt_engine

            def _tts_standby_factory(_state):
                return (rvc_handle, tts_engine)

            install_default_injectors(
                stt_injector=STTEngineInjector(standby_factory=_stt_standby_factory),
                tts_injector=TTSEngineInjector(standby_factory=_tts_standby_factory),
            )
            logger.info("services: STT + TTS injectors installed on registry")
        except Exception as e:                                          # noqa: BLE001
            logger.warning("install_default_injectors failed: %s", e)

        # 2026-05-24 OpenHands batch 7 (T7) -- discover .kenning/
        # per-project configuration once at startup and cache the
        # snapshot on the orchestrator so downstream subsystems can
        # consult it without re-walking. Fail-open: any error logs WARN
        # and leaves ``self._project_config = None``; consumers that
        # check ``getattr(self, "_project_config", None) is None`` keep
        # the standby behaviour.
        try:
            from kenning.config import PROJECT_ROOT
            from kenning.projects import discover_project_config

            self._project_config = discover_project_config(PROJECT_ROOT)
            if self._project_config.has_any_field:
                discovered = [
                    name for name in (
                        "skills_dir", "setup_script", "pre_commit_script",
                        "identity_override", "safety_rules", "test_command",
                        "voicepack_override", "intent_triggers", "hooks",
                    ) if getattr(self._project_config, name) is not None
                ]
                logger.info(
                    "projects: .kenning/ discovery found %d component(s): %s",
                    len(discovered), ", ".join(discovered),
                )
            else:
                logger.debug("projects: no .kenning/ configuration discovered")
        except Exception as e:                                          # noqa: BLE001
            logger.warning("project discovery failed: %s", e)
            self._project_config = None

        # 2026-05-15 latency: pre-render the ack phrase pools. On cache
        # hit (every conversational filler-ack + every web-search ack)
        # the engine skips its HTTP + filter chain and returns the
        # pre-filtered clip from memory -- saves ~350-400 ms (XTTS) or
        # ~310 ms (legacy piper_rvc) per first-spoken phrase. Runs on a
        # daemon thread so orchestrator construction stays fast; the
        # first turn may miss while the cache is still populating,
        # subsequent turns hit. Fail-open: engine is unchanged on
        # failures.
        self._ack_clip_prewarm_thread = self._kick_off_ack_clip_prewarm()
        # 2026-05-16 latency pass 2: speculative STT state. The
        # orchestrator kicks off Whisper transcription on the captured
        # audio AS SOON AS VAD declares a short run of consecutive
        # silence frames (typically ~32 ms after the user actually
        # stopped speaking). Whisper (~78 ms) finishes BEFORE the
        # fast-path silence baseline (~300 ms) elapses, so by the time
        # Smart Turn V3 confirms end-of-turn the transcript is already
        # available. If the user resumes speaking before SPEECH_END,
        # the speculative result is invalidated and the orchestrator
        # falls back to the foreground STT path. State is reset per
        # capture via :meth:`_collect_speculative_stt`.
        self._speculative_stt_lock = threading.Lock()
        self._speculative_stt_thread: Optional[threading.Thread] = None
        self._speculative_stt_result: Optional[str] = None
        self._speculative_stt_active = False
        self._speculative_stt_invalidated = False
        # 2026-05-18 latency pass 3 (Phase 2): speculative classification
        # chained off speculative STT. When STT completes inside the
        # silence-wait window, the same daemon thread runs the rule-path
        # web-gate, picks the conversational ack phrase, and kicks off
        # RAG pre-fetch -- saving ~10-50 ms on the cache-hit conversational
        # turn (more if RAG retrieval is slow). Result stored as a
        # ``_SpeculativeClassification`` dict keyed by the user_text it
        # was computed for. Invalidated alongside STT on SPEECH_START.
        self._speculative_classification_lock = threading.Lock()
        self._speculative_classification: Optional[dict] = None
        self._speculative_classification_invalidated = False
        # 2026-05-18 latency pass 3 (Phase 3): speculative LLM generation
        # chained off classification when the rule-path gate verdict
        # resolves to NO_SEARCH. Tokens stream into a queue; the response
        # path drains the queue instead of starting a fresh LLM call.
        # History is recorded by the consumer only when the speculation
        # was actually used (record_history=False on the speculative call;
        # explicit ``llm.record_completed_turn`` after consumption) so
        # invalidated speculations don't leave orphan turns.
        #
        # State invariant: each turn at most ONE speculation is in flight.
        # The ``_active`` flag gates re-entrant kick-off attempts. The
        # buffer is a fresh ``queue.Queue`` per speculation; ``_response``
        # holds the accumulated text for history recording on
        # consumption.
        self._speculative_llm_lock = threading.Lock()
        self._speculative_llm_thread: Optional[threading.Thread] = None
        # ``_buffer`` holds a ``queue.Queue`` while a speculation is in
        # flight or its tokens haven't been drained yet. ``None`` between
        # speculations. Untyped at the attribute layer because
        # ``Optional["queue.Queue"]`` would force a runtime ``queue``
        # import on every Orchestrator construction.
        self._speculative_llm_buffer = None
        self._speculative_llm_text: Optional[str] = None
        self._speculative_llm_response: Optional[str] = None
        self._speculative_llm_completed = False
        self._speculative_llm_active = False
        self._speculative_llm_invalidated = False
        self.addressing = self._load_addressing_classifier()
        self.web_gate, self.web_executor, self.ack_source = (
            self._load_web_search_if_enabled()
        )
        # 2026-05-12 filler-ack on conversational path: separate
        # shuffled-cycle source so the conversational and web-search
        # pools rotate independently. Construction is cheap (no
        # dependencies) so always-on -- gating happens at use site
        # via ``is_conversational_ack_eligible``.
        self.conv_ack_source = ConversationalAckSource()
        # GamingModeManager is HOISTED here (cheap, no model load; builds fine
        # with client=None) so the startup gaming auto-engage finds it via
        # self.gaming_mode_manager EVEN WHEN the coding stack is skipped in a lean
        # gaming boot. It used to be born INSIDE coding_voice -> skipping
        # coding_voice would null the manager and silently disable the ENTIRE
        # gaming engage (LLM->3B, Kokoro device, reranker-free, anticheat hooks).
        # CodingVoiceController now reuses THIS instance.
        self.gaming_mode_manager = self._load_gaming_mode_manager_if_enabled()
        self.mcp_server = self._load_mcp_server_if_enabled()
        self.coding_coordinator = self._load_coding_coordinator_if_enabled()
        # Phase 3.5: OpenClaw bridge holder. None when openclaw.enabled
        # is False (current default). Construction is fail-open — the
        # voice path is never blocked by the Gateway being unreachable.
        # Built BEFORE coding_voice so the voice controller can pass
        # the bridge to OpenClawDispatcher for live Gateway calls
        # (Phase 4 onwards).
        self.openclaw_bridge = self._load_openclaw_bridge_if_enabled()
        self.coding_voice = self._load_coding_voice_if_enabled()
        # 2026-05-12 Phase 12 -- wire moondream2 VLM for SCREEN_CONTEXT_QUERY
        # responses. Construction is lazy + fail-open: it validates the
        # transformers stack is importable but does NOT load weights at
        # orchestrator startup. The ~3.5 GB weight load happens on first
        # describe() call, which only fires on a voice "explain what I'm
        # looking at" with VLM enabled. Set the singleton so the
        # screen_context module can call into it via the registered
        # describe-bridge.
        self._load_desktop_vlm_if_enabled()
        # 2026 catalog 10 -- construct the browser-use CDP tool +
        # session manager singletons from config. Cheap + lazy (binary
        # discovery deferred to first call) + fail-open (no browser-use
        # binary on PATH leaves every method returning a structured
        # "binary missing" result). Default ON.
        self._load_browser_use_if_enabled()
        # openclaw-clawhub T15 -- privacy-by-construction telemetry store.
        # Constructed always but FAIL-PRIVATE: record_event no-ops unless
        # the operator explicitly set KENNING_TELEMETRY=opt-in. The
        # per-turn emit lives in _respond's finally. Construction is cheap
        # (just resolves paths; the salt file is created lazily on first
        # hash). Kept env-gated (not config-gated) by design -- the
        # privacy-by-construction contract is the documented reason this
        # one feature does not default-on.
        self._metrics_store = self._init_telemetry_store()
        # openclaw-clawhub T12 -- user-initiated report queue. A spoken
        # "log a concern about that response" files an audit-logged
        # Report the user (or a future operator pass) can triage later.
        # Append-only JSONL; fail-open. Constructed unconditionally
        # (harmless log; default-on per the ship-session-work rule).
        self._report_queue = self._init_report_queue()
        # 2026-05-24 SWE-Agent batch 7 (T16) -- visual click-preview
        # gate. When ``desktop.click_preview.enabled: true`` AND a VLM
        # is loaded, install a new InputController singleton that
        # wraps every click through ``preview_click``. Default OFF so
        # the existing call sites that hit pyautogui.click directly
        # keep working unchanged.
        try:
            from kenning.config import get_config
            cp_cfg = getattr(getattr(get_config(), "desktop", None), "click_preview", None)
        except Exception:                                              # noqa: BLE001
            cp_cfg = None
        # ANTICHEAT HARDENING: the click-preview gate imports the desktop VLM +
        # capture + input_control (the whole stack). Never under anticheat-safe
        # mode -- input injection is hard-blocked there anyway.
        from kenning.safety.anticheat import anticheat_active as _ac_cp
        if (cp_cfg is not None and bool(getattr(cp_cfg, "enabled", False))
                and not _ac_cp()):
            try:
                from kenning.desktop.vlm import get_vlm
                from kenning.desktop.capture import get_screen_capture
                from kenning.desktop.input_control import (
                    InputController,
                    set_input_controller,
                )

                vlm_handle = get_vlm()
                if vlm_handle is None:
                    logger.info(
                        "click_preview: VLM not loaded -- gate disabled "
                        "(degraded path would allow every click)",
                    )
                else:
                    def _click_preview_capture_screen() -> bytes:
                        cap = get_screen_capture()
                        if cap is None:
                            return b""
                        # Capture the monitor the foreground window is on (the
                        # one the click targets), not a hardcoded index. On a
                        # single-monitor machine index 1 doesn't exist, so the
                        # old capture returned None and the safety gate silently
                        # degraded to "allow every click". Fall back to 0.
                        mon_index = 0
                        try:
                            from kenning.desktop.windows import (
                                get_foreground_window,
                            )
                            fg = get_foreground_window()
                            fg_idx = getattr(fg, "monitor_index", None)
                            if fg is not None and fg_idx is not None:
                                mon_index = fg_idx
                        except Exception as exc:                      # noqa: BLE001
                            logger.debug(
                                "click_preview foreground-monitor lookup "
                                "failed (%s); using monitor 0", exc,
                            )
                        shot = cap.capture_monitor(mon_index)
                        if shot is None or shot.image_bytes is None:
                            return b""
                        return shot.image_bytes

                    def _click_preview_vlm_describe(image_bytes: bytes, prompt: str) -> str:
                        try:
                            return vlm_handle.describe(image_bytes, prompt) or ""
                        except Exception as exc:                       # noqa: BLE001
                            logger.debug("click_preview VLM describe raised: %s", exc)
                            return ""

                    new_controller = InputController(
                        click_preview_enabled=True,
                        click_preview_capture_screen=_click_preview_capture_screen,
                        click_preview_vlm_describe=_click_preview_vlm_describe,
                        click_preview_auto_pass_radius_px=int(cp_cfg.auto_pass_radius_px),
                        click_preview_crosshair_size=int(cp_cfg.crosshair_size),
                        click_preview_crosshair_thickness=int(cp_cfg.crosshair_thickness),
                        click_preview_require_confirmation_keyword=str(
                            cp_cfg.require_confirmation_keyword
                        ),
                        click_preview_history_depth=int(cp_cfg.history_depth),
                        click_preview_block_on_degraded=bool(cp_cfg.block_on_degraded),
                    )
                    set_input_controller(new_controller)
                    logger.info(
                        "click_preview: wired (auto_pass=%dpx, block_on_degraded=%s)",
                        int(cp_cfg.auto_pass_radius_px),
                        bool(cp_cfg.block_on_degraded),
                    )
            except Exception as e:                                     # noqa: BLE001
                logger.warning("click_preview wiring failed: %s", e)

        # 2026-05-22 -- engine-agnostic intent recognizer. When
        # ``intent.enabled`` is True, matched utterances short-circuit
        # the LLM gating path; the dispatcher fires registered handlers
        # (e.g., gaming mode engage/disengage) directly. Default OFF so
        # operators opt in after deciding which phrases to register.
        self._intent_recognizer = self._init_intent_recognizer_if_enabled()
        # Catalog 13 -- bounded autonomous self-improvement. Construct
        # BEFORE the skill registry so its proposal directory
        # (data/evolution/skills) exists for the initial walk; the
        # registry reloader resolves the registry lazily via the
        # singleton getter. Fail-open: returns None on any failure, which
        # makes every per-turn evolution hook a zero-cost no-op.
        self.evolution = self._load_evolution_if_enabled()
        # Production-hardening reach-signals (#62/#125/#63/#64): register
        # the two pure-observation seams (recorded errors + hard safety
        # blocks) feeding a bounded queue the run loop drains into the
        # EvolutionService. Fail-open; no-op when evolution is disabled.
        self._install_evolution_reach_observers()
        # 2026-05-23 OpenHands batch 2 (T1) -- trigger-loaded skills.
        # When ``skills.enabled`` is True, walks ``skills/`` (public),
        # ``~/.kenning/skills/`` (user), and ``<project>/.kenning/skills/``
        # (project), then publishes a SkillRegistry singleton that
        # LLMEngine._build_messages consults to inject matched skill
        # bodies into the system prompt per-turn. Default OFF so the
        # voice baseline is unchanged until operators opt in.
        self._load_skill_registry_if_enabled()
        # 2026-05-23 OpenHands batch 3 (T2 + T13) -- canonical event
        # store. When ``events.enabled`` is True, build the configured
        # backend (memory / jsonl / qdrant) and optionally subscribe
        # the bus so every published event is persisted with a hash
        # chain. Default OFF so existing bus behaviour is unchanged.
        self._load_event_store_if_enabled()
        # 2026-05-22 -- consumed by _build_response_stream. Set by the
        # intent dispatcher when a "needs fresh data" phrase matches;
        # forces the gate verdict to SEARCH (skipping preflight LLM).
        self._next_turn_force_search = False
        self._last_response_finished_monotonic: float = 0.0
        self._last_search_payload = None
        # Catalog 13 (evolution): set at the end of _respond when the just-
        # finished response was cut off by a barge-in. Consumed by the
        # evolution turn recorder on the NEXT turn to nudge the response
        # temperament terser. Fail-open default False.
        self._last_turn_barged_in: bool = False
        # 2026-05-22 OPEN_LAST_SOURCE: accumulated text of the most
        # recent assistant response. Used to match cited publication
        # names back to the source list when the user says "show me
        # that article".
        self._last_response_text: str = ""

        # In-memory dual-history store for verbatim conversation-recall
        # ("what did I say earlier about X?"). I/O-free + always available
        # (works even when ConversationMemory/Qdrant is disabled); each
        # addressed user utterance + each LLM response is appended. Capped so
        # a long session can't grow it without bound. Fail-open construction.
        try:
            from kenning.memory.dual_history import DualHistoryStore
            self._dual_history = DualHistoryStore(verbatim_cap=300)
        except Exception:                                            # noqa: BLE001
            self._dual_history = None

        # 2026-05-19 Tracks 1c-1e voice-loop integration. The
        # BackgroundSummarizer fires when ``memory.background_summary.enabled``
        # is True; otherwise the loader returns None and the hot-path
        # helpers short-circuit at zero cost. Constructed AFTER memory
        # + llm so the wired callbacks can reference them. State for
        # the per-turn background thread + a guard lock so only one
        # summarizer pass is in flight at a time (the summarizer's
        # internal _in_flight flag is a second line of defense).
        self.background_summarizer = self._load_background_summarizer_if_enabled()
        self._background_summarizer_lock = threading.Lock()
        self._background_summarizer_thread: Optional[threading.Thread] = None

        self._shutdown = threading.Event()
        self._interrupt = threading.Event()
        self._pending_capture = threading.Event()
        self._state: State = State.IDLE

        # 2026-05-14 VRAM-relief pass: with the 4B abliterated default
        # the post-init working set is ~7.4 GB instead of ~10 GB. Empty
        # the CUDA allocator's cache once everything is loaded so any
        # transient buffers from llama-cpp / faster-whisper / sounddevice
        # init don't sit on top of the working set. Saves ~200-400 MB
        # of fragmented allocation typically. Fail-open: no-op when
        # torch isn't installed or CUDA isn't available.
        try:  # pragma: no cover -- exercised in live runs only
            import torch  # noqa: WPS433
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(
                    "post-init VRAM trim: torch.cuda.empty_cache() done."
                )
        except Exception:
            pass

        # 2026-06-12: optionally engage gaming mode at startup so Kenning boots
        # straight into the bare-bones, minimal-GPU profile -- LLM swapped to the
        # CPU-only 3B, Kokoro TTS -> CPU, Parakeet STT stopped, VLM unloaded, and
        # (via is_gaming_mode_active gates) per-turn RAG retrieval / reranker /
        # web-search skipped. No need to say "gaming mode" each session. Runs
        # last, once every subsystem is constructed. Fail-open.
        try:  # pragma: no cover -- live boot path
            from kenning.config import get_config as _gc

            if getattr(_gc().gaming_mode, "engage_at_startup", False):
                manager = self._resolve_gaming_mode_manager()
                if manager is not None:
                    import asyncio
                    # Startup engage is SILENT (no per-substep voice acks) --
                    # the flag is read by _gaming_voice_ack. Reset afterwards so
                    # a later VOICE "gaming mode" still announces its progress.
                    self._gaming_engage_silent = True
                    try:
                        asyncio.run(manager.engage())
                    finally:
                        self._gaming_engage_silent = False
                    logger.info(
                        "gaming mode: auto-engaged at startup (bare-bones, "
                        "minimal-GPU profile)")
                else:
                    logger.warning(
                        "gaming mode: engage_at_startup set but no manager "
                        "available")
        except Exception as e:                                       # noqa: BLE001
            logger.warning("gaming mode startup engage failed: %s", e)
        # ANTICHEAT POSTURE SELF-AUDIT (2026-06-14): last boot step. Leaves an
        # auditable line every restart proving the OS-interaction stack is not in
        # RAM while anticheat-safe mode is active -- and a LOUD warning if it is
        # (a regression canary). Fail-open: never blocks boot.
        try:
            self._audit_anticheat_posture()
        except Exception as e:                                        # noqa: BLE001
            logger.debug("anticheat posture self-audit skipped (%s)", e)
        # Build the semantic command router NOW (end of boot). REQUIREMENT: the
        # HYBRID (embedding) backend must come up -- lexical-only is a failure,
        # not an acceptable default. If the first build came up lexical (sidecar
        # slow/dead), respawn the sidecar ONCE + rebuild. Fail-open; never blocks
        # boot beyond the single bounded retry.
        try:
            from kenning.audio.command_router import (
                get_command_router, reset_command_router)

            def _using_emb(r):
                return bool(r and getattr(r.backend, "using_embedding",
                                          lambda: False)())

            router = get_command_router()
            if router is not None and not _using_emb(router):
                logger.error("command router built WITHOUT embedding "
                             "(lexical-only) -- respawning the sidecar + "
                             "rebuilding the router (one-shot)")
                try:
                    self._kill_embedder_sidecar()
                except Exception:                                     # noqa: BLE001
                    pass
                self._start_embedder_sidecar()
                import time as _t
                from kenning.audio._router_backends import EmbeddingBackend
                from kenning.config import get_config as _gc2
                _rc = getattr(_gc2(), "semantic_router", None)
                _host = getattr(_rc, "sidecar_host", "127.0.0.1") if _rc else "127.0.0.1"
                _port = int(getattr(_rc, "sidecar_port", 8772)) if _rc else 8772
                _eb = EmbeddingBackend(host=_host, port=_port)
                _deadline = _t.monotonic() + 45.0
                while _t.monotonic() < _deadline:
                    if _eb.available():
                        break
                    _t.sleep(1.5)
                reset_command_router()
                router = get_command_router()
                if _using_emb(router):
                    logger.info("command router REBUILT with embedding (hybrid) "
                                "after sidecar respawn")
                else:
                    logger.error("EMBEDDING UNAVAILABLE after respawn -- semantic "
                                 "router is lexical-only this session (check the "
                                 "embedder sidecar venv / GPU)")
        except Exception as e:                                        # noqa: BLE001
            logger.debug("semantic command router warmup skipped (%s)", e)

    def _start_embedder_sidecar(self) -> None:
        """Spawn the command-router embedder sidecar -- a SEPARATE process in an
        ISOLATED venv, so the embedding model NEVER loads into this anticheat-
        pinned process. Pure compute (no input/capture/injection), loopback only.
        No-op when disabled / backend=lexical / venv missing; REUSES an already-
        running sidecar (restart-safe); fail-open (router -> lexical)."""
        from kenning.config import get_config
        rcfg = getattr(get_config(), "semantic_router", None)
        if rcfg is None or not getattr(rcfg, "enabled", True):
            return
        if (getattr(rcfg, "backend", "hybrid") == "lexical"
                or not getattr(rcfg, "sidecar_enabled", True)):
            logger.info("embedder sidecar: not needed (backend=lexical or disabled)")
            return
        import os
        host = getattr(rcfg, "sidecar_host", "127.0.0.1")
        port = int(getattr(rcfg, "sidecar_port", 8772))
        model = getattr(rcfg, "sidecar_model", "google/embeddinggemma-300m")
        backend = getattr(rcfg, "sidecar_backend", "sentence_transformers")
        pidfile = getattr(rcfg, "sidecar_pidfile_path", "") or None
        # SINGLETON ENFORCEMENT: a boot-time sweep reaps any orphan/duplicate on
        # the port (e.g. one left by a force-killed prior Ultron whose in-process
        # cleanup never ran) + verifies ownership via a pidfile BEFORE we spawn.
        if getattr(rcfg, "sidecar_orphan_sweep_enabled", True):
            try:
                from kenning.subprocess import sidecar_lock
                verdict, owned_pid = sidecar_lock.sweep(host, port, model, path=pidfile)
                if verdict == "reuse":
                    self._embedder_sidecar_reuse_pid = owned_pid
                    logger.info("embedder sidecar already running on %s:%d "
                                "(pid=%s, model OK) -- reusing + owning it",
                                host, port, owned_pid)
                    return
            except Exception as e:                                   # noqa: BLE001
                logger.debug("embedder sidecar sweep skipped (%s)", e)
        else:
            from kenning.audio._router_backends import EmbeddingBackend
            if EmbeddingBackend(host=host, port=port).available():
                logger.info("embedder sidecar already running on %s:%d -- reusing", host, port)
                return
        py = getattr(rcfg, "sidecar_python", "")
        if not py or not os.path.exists(py):
            logger.warning("embedder sidecar venv python missing (%s) -> router "
                           "uses the lexical backend", py)
            return
        script = getattr(rcfg, "sidecar_script", "scripts/embedder_server.py")
        script_path = script if os.path.isabs(script) else os.path.abspath(script)
        env = dict(os.environ)
        env["KENNING_EMBEDDER_BACKEND"] = getattr(rcfg, "sidecar_backend", "sentence_transformers")
        env["KENNING_EMBEDDER_MODEL"] = getattr(rcfg, "sidecar_model", "google/embeddinggemma-300m")
        env["KENNING_EMBEDDER_PORT"] = str(port)
        env["KENNING_EMBEDDER_QUERY_PROMPT"] = getattr(rcfg, "sidecar_query_prompt", "query")
        env["KENNING_EMBEDDER_DOC_PROMPT"] = getattr(rcfg, "sidecar_doc_prompt", "document")
        _dev = getattr(rcfg, "sidecar_device", "")
        if _dev:
            env["KENNING_EMBEDDER_DEVICE"] = _dev
        _cache = getattr(rcfg, "sidecar_hf_cache", "")
        if _cache:                          # override the machine's broken D: cache
            env["TRANSFORMERS_CACHE"] = _cache
            env["HF_HUB_CACHE"] = _cache
        import subprocess
        proc = subprocess.Popen(
            [py, script_path, str(port)],
            env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._embedder_sidecar_proc = proc
        # Record ownership so the NEXT boot's sweep can verify + reap this exact
        # process even if THIS Ultron is force-killed (no in-process cleanup runs).
        try:
            from kenning.subprocess import sidecar_lock
            sidecar_lock.write(proc.pid, port, model, backend, path=pidfile)
        except Exception:                                            # noqa: BLE001
            pass
        zk = getattr(self, "_zombie_killer", None)
        if zk is not None:
            try:
                # NOT persistent: a finite (~1h) reaper backstop for a
                # crash-orphaned sidecar -- never auto-killed in a normal gaming
                # session, and shutdown() reaps it explicitly first anyway.
                zk.register(proc.pid, "embedder-sidecar",
                            persistent=False, hard_timeout_s=3600.0)
            except Exception:                                        # noqa: BLE001
                pass
        logger.info("embedder sidecar spawned (pid=%s) model=%s backend=%s on "
                    "%s:%d -- ISOLATED venv, model NOT loaded into this process",
                    proc.pid, model, backend, host, port)

    def _kill_embedder_sidecar(self) -> None:
        """Reap the embedder sidecar process TREE (launcher shim -> embedder
        child) on shutdown. Resolves the pid from the spawned Popen OR a
        reused-but-owned sidecar; UNREGISTERS from the ZombieKiller FIRST (so the
        reaper can't race the kill), tree-kills it, and clears the pidfile.
        Never raises -- cleanup must not block exit."""
        pid = None
        proc = getattr(self, "_embedder_sidecar_proc", None)
        if proc is not None and getattr(proc, "pid", None):
            pid = proc.pid
        elif getattr(self, "_embedder_sidecar_reuse_pid", None):
            pid = self._embedder_sidecar_reuse_pid
        if not pid:
            return
        try:
            zk = getattr(self, "_zombie_killer", None)
            if zk is not None:
                try:
                    zk.unregister(pid)
                except Exception:                                    # noqa: BLE001
                    pass
            from kenning.subprocess.kill_tree import kill_process_tree
            res = kill_process_tree(int(pid), grace_seconds=5.0)
            logger.info("embedder sidecar reaped (pid=%s killed=%s)",
                        pid, getattr(res, "killed", None))
            try:
                from kenning.config import get_config
                from kenning.subprocess import sidecar_lock
                rcfg = getattr(get_config(), "semantic_router", None)
                pidfile = (getattr(rcfg, "sidecar_pidfile_path", "") or None) if rcfg else None
                sidecar_lock.clear(path=pidfile)
            except Exception:                                        # noqa: BLE001
                pass
        except Exception as e:                                       # noqa: BLE001
            logger.debug("embedder sidecar reap skipped (%s)", e)
        finally:
            self._embedder_sidecar_proc = None
            self._embedder_sidecar_reuse_pid = None

    def _skip_for_lean_gaming(self, flag: str) -> bool:
        """True when this is a LEAN GAMING-startup boot AND the named barebones
        skip flag is on -- i.e. this non-essential subsystem must NOT load /
        import / touch RAM (shrinking the anticheat surface).

        Gated on the CONFIG INTENT ``gaming_mode.engage_at_startup`` -- NOT the
        runtime ``is_gaming_mode_active()`` (False throughout __init__) and NOT
        ``anticheat_active()`` (a non-gaming anticheat-pinned dev boot must still
        load coding/search). Lean gaming boot is the permanent default; every
        skip is individually toggleable. Fail-closed to False (keep the
        subsystem) on any error -- never skip something by accident."""
        try:
            from kenning.config import get_config
            gm = get_config().gaming_mode
            return bool(getattr(gm, "engage_at_startup", False)
                        and getattr(gm, flag, True))
        except Exception:                                            # noqa: BLE001
            return False

    def _start_dialog_poller(self) -> None:
        """Start the UIA DialogPoller -- UNLESS anticheat-safe mode is active.

        ANTICHEAT HARDENING (2026-06-14): importing
        ``kenning.desktop.dialog_poller`` pulls the ENTIRE desktop-automation
        stack into RAM via the package ``__init__`` -- pyautogui (SendInput
        injection), mss (GDI screen capture), pywinauto (UIA). Under anticheat-
        safe mode we keep that stack ENTIRELY OUT of the process: never imported,
        never a running thread -- not merely call-gated. So skip the poller
        outright when the mode is active; a kernel anticheat then observes zero
        input/capture/UIA surface at all. (One of several boot/hot paths that
        import ``kenning.desktop``; the others -- VLM loader, browser-use loader,
        click-preview, engage-deps, and inference's per-message VLM-loaded check
        -- are all gated the same way. The boot ``_audit_anticheat_posture``
        canary + clean-subprocess tests pin that the whole stack stays cold.)
        """
        from kenning.safety.anticheat import anticheat_active

        if anticheat_active():
            self._dialog_poller = None
            logger.info(
                "DialogPoller NOT started + desktop-automation stack NOT loaded "
                "(pyautogui / mss / pywinauto kept out of RAM): anticheat-safe "
                "mode active"
            )
            return
        try:
            from kenning.desktop.dialog_poller import get_dialog_poller

            self._dialog_poller = get_dialog_poller()
            self._dialog_poller.start()
            logger.info("DialogPoller daemon started")
        except Exception as e:                                       # noqa: BLE001
            self._dialog_poller = None
            logger.warning(
                "DialogPoller startup skipped (%s); "
                "dialog auto-handler will not receive events", e,
            )

    def _audit_anticheat_posture(self) -> None:
        """Log the live anticheat posture for per-restart auditability.

        Under anticheat-safe mode the input-injection / screen-capture / UIA
        stack must NOT be loaded in this process. This walks ``sys.modules`` for
        those libraries and the ``kenning.desktop`` package and logs the result.
        If the mode is active but any of them are loaded, it logs a WARNING (a
        loud regression canary) rather than failing -- the guards still block
        every CALL, but a loaded module is a footprint we want to know about.
        """
        import sys as _sys
        from kenning.safety.anticheat import anticheat_active

        active = anticheat_active()
        # The libraries a kernel anticheat-conscious build must keep cold:
        # input injection (pyautogui/SendInput), screen capture (mss/pyscreeze/
        # dxcam), UI automation (pywinauto/uiautomation), input hooks (pynput).
        risky = [m for m in (
            "pyautogui", "mss", "pyscreeze", "dxcam",
            "pywinauto", "uiautomation", "pynput",
            "playwright", "browser_use", "selenium",
        ) if m in _sys.modules]
        desktop_loaded = "kenning.desktop" in _sys.modules
        bridge_loaded = [m for m in (
            "kenning.openclaw_bridge.browser", "kenning.openclaw_bridge.desktop",
        ) if m in _sys.modules]
        # The import firewall must be INSTALLED so any future lazy import of a
        # blocked module is refused at the loader, not merely blocked at call.
        try:
            from kenning.safety.import_firewall import is_firewall_installed
            firewall_ok = bool(is_firewall_installed())
        except Exception:                                            # noqa: BLE001
            firewall_ok = False
        poller = getattr(self, "_dialog_poller", None)
        poller_running = bool(poller is not None and getattr(poller, "running", False))
        if active and (risky or desktop_loaded or bridge_loaded
                       or poller_running or not firewall_ok):
            logger.warning(
                "ANTICHEAT POSTURE CANARY: mode ACTIVE but footprint/posture "
                "issue -- libs=%s kenning.desktop=%s bridge=%s poller=%s "
                "import_firewall=%s. Calls are still hard-blocked, but "
                "investigate this load path.",
                risky or "none", desktop_loaded, bridge_loaded or "none",
                poller_running, "installed" if firewall_ok else "MISSING",
            )
        else:
            logger.info(
                "anticheat posture OK | mode=%s | input/capture/UIA/browser "
                "libs loaded=%s | kenning.desktop=%s | bridge=%s | "
                "import_firewall=%s | dialog poller=%s",
                "ACTIVE" if active else "off",
                risky or "none", desktop_loaded, bridge_loaded or "none",
                "installed" if firewall_ok else "off",
                "running" if poller_running else "not started",
            )
        # LEAN GAMING BOOT canary (2026-06-15): when gaming is the startup intent,
        # the non-essential subsystems must NOT have loaded -- PROVE it against
        # sys.modules every boot so a lean-boot gate regression is visible in the
        # log immediately (the user's "nothing unnecessary even in RAM" rule).
        try:
            from kenning.config import get_config as _gc_lean
            lean = bool(getattr(_gc_lean().gaming_mode, "engage_at_startup", False))
        except Exception:                                            # noqa: BLE001
            lean = False
        if lean:
            # Check the RUNTIME-heavy modules, not lightweight necessary leaves:
            # the LLM loads kenning.openclaw_bridge.persona (workspace system
            # prompt) which is fine, but the bridge RUNTIME (holder = gateway +
            # threads) and the coding/MCP/evolution/reranker stacks must stay out.
            heavy = [m for m in (
                "kenning.openclaw_bridge.holder", "kenning.coding.mcp_server",
                "kenning.coding.voice", "kenning.evolution.service",
                "sentence_transformers",
            ) if m in _sys.modules]
            if heavy:
                logger.warning(
                    "LEAN BOOT CANARY: gaming-startup boot but non-essential "
                    "modules are LOADED=%s -- a lean-boot gate regressed; these "
                    "must never enter RAM while gaming (anticheat surface).", heavy)
            else:
                logger.info(
                    "lean boot OK | non-essential subsystems "
                    "(coding/MCP/OpenClaw/evolution/reranker) NOT loaded -- only "
                    "core relay + Spotify + voice in RAM")

    def _load_mcp_server_if_enabled(self):
        """Construct + start the MCP server (Phase 1+). Failures degrade
        silently -- the coding pipeline can run without MCP, just without
        the supervisor's clarification round-trip."""
        if self._skip_for_lean_gaming("barebones_skip_coding"):
            logger.info("lean gaming boot: coding MCP server skipped (port 19761 "
                        "+ SSE thread not started)")
            return None
        if not (settings.CODING_ENABLED and settings.CODING_MCP_ENABLED):
            return None
        from kenning.coding import KenningMCPServer   # lazy: keeps coding out of a lean boot
        try:
            # Phase 7: pass the per-session audit dir so SessionStore
            # auto-logs every state change to logs/sessions/<id>.jsonl.
            # A3 wiring: thread the live ConversationMemory through so
            # ``project.lookup_facts`` reads from Qdrant.
            server = KenningMCPServer(
                session_audit_dir=settings.CODING_SESSION_AUDIT_DIR,
                memory=self.memory,
            )
            server.start(ready_timeout_s=5.0)
            logger.info("MCP server listening at %s", server.sse_url)
            # openclaw-clawhub T7: mint a short-lived forensic token
            # scoped to this MCP server's PID + tool capabilities,
            # replacing the long-lived-secret pattern the catalog
            # called out. Audit-logged; fail-open (the token is not a
            # hard gate in the single-user in-process runtime).
            self._mint_forensic_token(
                caller_id="mcp:tools",
                audience="kenning-mcp",
                scope=("mcp.tools.read", "mcp.tools.invoke"),
                ttl_seconds=6 * 60 * 60,  # == short_lived_token.MAX_TTL_SECONDS
                extra_claims={"sse_url": str(getattr(server, "sse_url", ""))},
            )
            return server
        except Exception as e:
            logger.warning("MCP server start failed (%s) -- disabled", e)
            return None

    def _load_coding_coordinator_if_enabled(self):
        """Construct the supervisor's :class:`ConversationCoordinator` and
        wire it into the MCP server's clarification + declare_complete
        responder hooks."""
        if self.mcp_server is None:
            return None
        from kenning.coding.coordinator import ConversationCoordinator  # lazy
        try:
            renderer = None
            try:
                from kenning.coding.templates import TemplateRenderer
                renderer = TemplateRenderer()
            except FileNotFoundError as e:
                logger.warning("Template renderer disabled (%s)", e)
            from kenning.coding.verification import Verifier
            verifier = Verifier(store=self.mcp_server.store)
            # A3 wiring: hand the MCP server's lookup_facts method to the
            # coordinator so the clarification fast-path can answer from
            # the Qdrant ``facts`` collection. When memory is unavailable,
            # the MCP server's own no-op stub fires (returns []), keeping
            # the coordinator's behaviour identical to today.
            facts_lookup = self.mcp_server.lookup_facts
            coordinator = ConversationCoordinator(
                store=self.mcp_server.store,
                llm=self.llm,
                renderer=renderer,
                verifier=verifier,
                facts_lookup=facts_lookup,
            )
            self.mcp_server.set_clarification_responder(coordinator.decide_clarification)
            self.mcp_server.set_declare_complete_handler(coordinator.handle_declare_complete)
            logger.info(
                "Coordinator wired into MCP server (templates=%s, verifier=on)",
                "on" if renderer else "off",
            )
            return coordinator
        except Exception as e:
            logger.warning("Coordinator init failed (%s) -- disabled", e)
            return None

    def _load_desktop_vlm_if_enabled(self) -> None:
        """2026-05-12 Phase 12 -- construct the moondream2 VLM singleton.

        Lazy + fail-open: the constructor validates importability but
        does NOT load weights (those load on first describe() call).
        Sets the module-level VLM singleton via :func:`set_vlm`, which
        also wires the screen_context describe-bridge so
        :func:`build_screen_context` can include moondream2 scene
        descriptions when ``include_vlm=True``.

        Any construction failure (missing transformers, missing model
        weights on disk, etc.) leaves the singleton unset and
        screen_context falls back to text-only context (window title +
        UIA tree + foreground app). The voice path is never blocked.

        ANTICHEAT HARDENING (2026-06-14): the VLM lives in
        ``kenning.desktop.vlm`` -- importing it triggers the desktop package
        ``__init__`` and pulls pyautogui + mss into RAM. It is a SCREEN
        understanding feature (needs capture, which is hard-blocked under
        anticheat) and gaming mode unloads it anyway, so under anticheat-safe
        mode we skip CONSTRUCTING it entirely -- it is never even imported.
        """
        from kenning.safety.anticheat import anticheat_active

        if anticheat_active():
            logger.info(
                "desktop VLM NOT constructed (kenning.desktop kept out of RAM): "
                "anticheat-safe mode active"
            )
            return
        try:
            from kenning.desktop.vlm import build_vlm_from_config, set_vlm

            vlm = build_vlm_from_config(enabled=True, device="cpu")
            if vlm is not None:
                set_vlm(vlm)
                logger.info("VLM (moondream2) constructed -- lazy-loads on first use.")
        except Exception as e:                                    # noqa: BLE001
            logger.warning(
                "VLM construction skipped (%s) -- screen-context queries "
                "will fall back to text-only window/UIA context.", e,
            )

    def _load_browser_use_if_enabled(self) -> None:
        """2026 catalog 10 -- construct the browser-use CDP tool +
        session-manager singletons from config.

        Cheap + lazy + fail-open at every layer:

        * Construction does NOT discover the binary or spawn anything
          (binary discovery is deferred to the first CLI call).
        * When ``browser_use.enabled`` is False, this is a no-op and
          the singletons stay unset (callers that read them via
          :func:`get_browser_use_tool` get None and degrade).
        * Any construction error logs WARN and leaves the singletons
          unset. The voice path is never blocked -- the browser tier
          is opt-in infrastructure, not on the conversational hot path.

        The session manager's cap is wired from ``browser_use.max_sessions``
        so the config knob drives the runtime limit.

        ANTICHEAT HARDENING (2026-06-14): browser_use lives under
        ``kenning.desktop`` (importing it loads pyautogui + mss) and CDP
        browser automation is a blocked desktop surface, so under anticheat-safe
        mode it is never constructed -- never imported.
        """
        from kenning.safety.anticheat import anticheat_active

        if anticheat_active():
            logger.info(
                "browser-use tier NOT constructed (kenning.desktop kept out of "
                "RAM): anticheat-safe mode active"
            )
            return
        try:
            from kenning.config import get_config

            cfg = get_config().browser_use
        except Exception as e:  # noqa: BLE001
            logger.warning("browser_use: config read failed (%s)", e)
            return
        if not getattr(cfg, "enabled", False):
            return
        try:
            from kenning.desktop.browser_use import (
                BrowserUseTool,
                set_browser_use_tool,
            )

            tool = BrowserUseTool(
                binary_path=cfg.binary_path,
                session=cfg.default_session,
                default_timeout_s=cfg.default_timeout_seconds,
                headed=cfg.headed,
            )
            set_browser_use_tool(tool)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "browser_use tool construction skipped (%s) -- browser "
                "automation tier unavailable this session.", e,
            )
            return
        try:
            from kenning.desktop.browser_sessions import (
                BrowserSessionsManager,
                set_browser_sessions_manager,
            )

            manager = BrowserSessionsManager(
                tool_factory=lambda name: tool.with_session(name),
                max_sessions=cfg.max_sessions,
            )
            set_browser_sessions_manager(manager)
            logger.info(
                "browser-use tier constructed (binary discovery deferred; "
                "max_sessions=%d).", cfg.max_sessions,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "browser_use session manager construction skipped (%s).", e,
            )

    def _init_telemetry_store(self):
        """openclaw-clawhub T15 -- construct the private telemetry store.

        Returns a :class:`PrivateMetricsStore` (or None on import /
        construction failure). The store is fail-private: every
        :meth:`record_event` no-ops unless ``KENNING_TELEMETRY=opt-in``
        is set, so constructing it unconditionally leaks nothing.
        """
        try:
            from kenning.config import PROJECT_ROOT
            from kenning.observability.private_telemetry import (
                PrivateMetricsStore,
            )

            return PrivateMetricsStore(project_root=PROJECT_ROOT)
        except Exception as e:  # noqa: BLE001
            logger.debug("telemetry store construction skipped: %s", e)
            return None

    @staticmethod
    def _latency_bucket(latency_ms: int) -> str:
        """Map a turn latency to a short, leak-safe bucket label.

        Buckets keep the telemetry aggregate-only -- the exact ms is
        also recorded (numeric, safe) but the bucket is the dashboard-
        friendly axis. All labels are <= 12 chars so they pass the
        telemetry leak check without needing a safe-key carve-out.
        """
        if latency_ms < 500:
            return "fast"
        if latency_ms < 1500:
            return "normal"
        if latency_ms < 5000:
            return "slow"
        return "very_slow"

    def _emit_turn_telemetry(
        self,
        intent_kind: Optional[str],
        turn_start: float,
        *,
        errored: bool,
    ) -> None:
        """openclaw-clawhub T15 -- emit one aggregate per-turn event.

        Called from :meth:`_respond`'s finally so every conversational
        turn is counted. The event carries only leak-safe fields:
        the routing-intent kind under the ``category`` safe key, a
        ``searched`` bool, the numeric ``latency_ms``, a coarse
        ``tier`` bucket, and an ``outcome`` enum. NO user text /
        response body / path ever reaches the store.

        Fail-private (the store no-ops unless opted in) AND fail-open
        (any error is swallowed at debug level so the voice path is
        never affected).
        """
        store = getattr(self, "_metrics_store", None)
        if store is None:
            return
        try:
            from kenning.config import PROJECT_ROOT
            from kenning.observability.private_telemetry import (
                HashedEvent,
                hash_root,
            )

            latency_ms = int((time.monotonic() - turn_start) * 1000.0)
            event = HashedEvent(
                kind="voice_turn",
                root_id=hash_root(PROJECT_ROOT, project_root=PROJECT_ROOT),
                attributes={
                    "category": (str(intent_kind) if intent_kind else "none"),
                    "searched": self._last_search_payload is not None,
                    "latency_ms": latency_ms,
                    "tier": self._latency_bucket(latency_ms),
                    "outcome": "error" if errored else "ok",
                },
            )
            store.record_event(event)
        except Exception as e:  # noqa: BLE001
            logger.debug("telemetry emit failed: %s", e)

    def _init_report_queue(self):
        """openclaw-clawhub T12 -- construct the user report queue.

        Returns a :class:`ReportQueue` persisting to
        ``data/feedback/reports.jsonl`` (or None on failure). The
        log is append-only with a SHA-256 hash chain (tamper-evident,
        same shape as the safety audit log).
        """
        try:
            from kenning.config import PROJECT_ROOT
            from kenning.feedback.report_queue import ReportQueue

            log_path = PROJECT_ROOT / "data" / "feedback" / "reports.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            return ReportQueue(audit_log_path=log_path)
        except Exception as e:  # noqa: BLE001
            logger.debug("report queue construction skipped: %s", e)
            return None

    def _maybe_handle_deep_research(self, user_text: str) -> bool:
        """Catalog 12 (felo-search T3) -- handle an explicit "research X in
        depth" / "deep dive on X" request via a bounded DeepResearchLoop,
        then synthesize + speak the answer.

        Returns True iff the utterance was an explicit deep-research command
        AND was handled here (short-circuits routing). Returns False to let
        the utterance fall through to normal routing -- on no match, when
        ``deep_research.enabled`` is off, or when search / the LLM aren't
        wired (so the user still gets a normal response instead of silence).

        Strict matcher: only "research X in depth" / "deep dive on X" / "dig
        deeper into X" style utterances trip it; "search X" / "what is X"
        never do. Fail-open: any failure speaks a clear message and returns
        True so the recognised command isn't silently dropped.
        """
        try:
            from kenning.web_search.deep_research import (
                DeepResearchLoop,
                match_deep_research,
            )
        except Exception as e:                                       # noqa: BLE001
            logger.debug("deep_research import failed: %s", e)
            return False
        match = match_deep_research(user_text)
        if match is None:
            return False
        try:
            from kenning.config import get_config
            dr_cfg = get_config().deep_research
        except Exception:                                            # noqa: BLE001
            return False
        if not getattr(dr_cfg, "enabled", False):
            return False
        if getattr(self, "web_executor", None) is None or self.llm is None:
            # Can't research without both the search executor and the LLM;
            # fall through so the user gets a normal best-effort response.
            return False

        # NOTE: ``trace`` is lazy-imported per-method in this module (it is
        # NOT a module global), so the import below is REQUIRED -- without it
        # the trace.tlog calls in this handler raise NameError before the
        # try-block, crashing the deep-research command. (Latent bug fixed.)
        from kenning import trace
        trace.tlog(logger, "deep_research:start", topic=match.topic[:80])
        self._interrupt.clear()
        self._last_search_payload = None
        self._last_response_text = ""
        response_buf: list = []
        watcher: Optional[threading.Thread] = None
        try:
            # Immediate spoken ack -- the research itself takes ~10-18 s.
            self._speak("Researching that in depth. Give me a moment.")

            loop = DeepResearchLoop(
                executor=self.web_executor,
                llm=self.llm,
                max_steps=getattr(dr_cfg, "max_steps", 3),
                max_sub_queries_per_step=getattr(
                    dr_cfg, "max_sub_queries_per_step", 3
                ),
                top_n_per_query=getattr(dr_cfg, "top_n_per_query", 3),
                max_accumulated_sources=getattr(
                    dr_cfg, "max_accumulated_sources", 8
                ),
            )
            result = loop.research(match.topic)
            payload = result.to_payload()
            self._last_search_payload = payload
            trace.tlog(
                logger, "deep_research:gathered",
                status=result.loop_status, steps=result.steps,
                sources=len(payload.sources), sub_queries=len(payload.queries),
            )

            if not payload.sources:
                self._speak(
                    "I dug into that but couldn't surface enough to give you "
                    "a solid answer."
                )
                return True

            sources_block = format_sources_for_prompt(payload.sources)
            augmented = (
                f"User question: {match.topic}\n\n"
                f"Research findings gathered from multiple web searches:\n"
                f"{sources_block}\n\n"
                "Synthesize a thorough but focused answer to the user's "
                "question using ONLY the facts present in the findings "
                "above. Organize the key points clearly. Attribute a claim "
                "only to a source whose name actually appears in the "
                "findings; do not invent specifics that aren't visible. If "
                "the findings are thin on some aspect, say so briefly rather "
                "than padding with general knowledge. Stay in character. "
                "End the response when you have answered."
            )

            if settings.BARGE_IN_ENABLED:
                watcher = threading.Thread(
                    target=self._interrupt_watcher, daemon=True,
                    name="wake-watcher",
                )
                watcher.start()

            print("  kenning: ", end="", flush=True)
            token_stream = self.llm.generate_stream(
                augmented,
                history_user_message=match.topic,
                rag_query=match.topic,
                enable_thinking=False,
            )

            def gated():
                for token in token_stream:
                    if self._interrupt.is_set() or self._shutdown.is_set():
                        self.llm.cancel()
                        return
                    print(token, end="", flush=True)
                    response_buf.append(token)
                    yield token

            self.tts.speak_stream(gated())
            print()
            self._last_response_text = "".join(response_buf)

            # T4: surface the researched sub-questions in the transcript only.
            try:
                expose = get_config().web_search.expose_search_strategy
            except Exception:                                        # noqa: BLE001
                expose = False
            strat_qs = payload.queries if expose else None
            print(
                f"  {format_sources_for_transcript(payload.sources, strategy_queries=strat_qs)}"
            )
            return True
        except Exception as e:                                       # noqa: BLE001
            logger.exception("deep research failed: %s", e)
            try:
                self._speak("Something went wrong while I was researching that.")
            except Exception:                                        # noqa: BLE001
                pass
            return True
        finally:
            self._interrupt.set()  # release the watcher
            if watcher is not None:
                watcher.join(timeout=1.0)
            self._interrupt.clear()

    def _maybe_handle_code_exploration(self, user_text: str) -> bool:
        """Handle an explicit code-search command ("search the codebase for X",
        "where is the safety validator defined") via a bounded
        :class:`DeepExplorationLoop` (iterative ripgrep over the project source),
        then speak WHERE the matches are.

        Strict matcher -> coding tasks ("build X") + web / memory requests fall
        through (return False). Fail-open: any failure speaks a clear message
        and returns True so the recognised command isn't silently dropped. The
        search root is the project root (the user's kenning repo); read-only.
        """
        try:
            from kenning.agent_loop.deep_loops import DeepExplorationLoop
            from kenning.search.code_exploration import match_code_exploration
        except Exception as e:                                       # noqa: BLE001
            logger.debug("code_exploration import failed: %s", e)
            return False
        match = match_code_exploration(user_text)
        if match is None:
            return False
        if self.llm is None:
            return False
        try:
            from kenning.config import get_config
            enabled = bool(getattr(
                get_config().coding, "deep_exploration_enabled", True,
            ))
        except Exception:                                            # noqa: BLE001
            enabled = True
        if not enabled:
            return False

        logger.debug("code_exploration:start topic=%s", match.topic[:80])
        self._interrupt.clear()
        self._last_response_text = ""

        try:
            import os
            from kenning.config import PROJECT_ROOT
            from kenning.search.ripgrep import regex_search_files
        except Exception as e:                                       # noqa: BLE001
            logger.debug("code_exploration deps unavailable: %s", e)
            return False

        root = str(PROJECT_ROOT)

        def _search(pattern: str):
            try:
                return regex_search_files(root, root, pattern, context_lines=1) or []
            except Exception as e:                                   # noqa: BLE001
                logger.debug("code_exploration search %r failed: %s", pattern, e)
                return []

        try:
            self._speak("Let me search the codebase for that.")
            loop = DeepExplorationLoop(search=_search, llm=self.llm)
            result = loop.explore(match.topic)
            items = getattr(result, "items", None) or []
            files: list = []
            for it in items:
                fp = getattr(it, "file_path", None) or getattr(it, "path", None)
                if not fp:
                    continue
                try:
                    rel = os.path.relpath(str(fp), root)
                except Exception:                                    # noqa: BLE001
                    rel = str(fp)
                if rel not in files:
                    files.append(rel)
            if not files:
                answer = (
                    f"I searched the codebase for {match.topic} but found no "
                    "matches."
                )
            else:
                shown = files[:6]
                more = len(files) - len(shown)
                tail = f", and {more} more" if more > 0 else ""
                answer = (
                    f"Found {len(items)} matches across {len(files)} files: "
                    f"{', '.join(shown)}{tail}."
                )
            self._last_response_text = answer
            self._speak(answer)
        except Exception as e:                                       # noqa: BLE001
            logger.warning("code_exploration failed: %s", e)
            self._speak("I hit a problem searching the codebase.")
        return True

    def _maybe_handle_deep_recall(self, user_text: str) -> bool:
        """Handle an explicit exhaustive-recall command ("recall everything
        we discussed about X", "dig deep into your memory about X") via a
        bounded DeepMemoryLoop (iterative RAG: decompose -> retrieve ->
        gap-fill -> retrieve, capped by max_steps), then synthesize + speak
        the answer from the recalled turns.

        Returns True iff the utterance was an explicit deep-recall command
        AND was handled here (short-circuits routing). Strict matcher ->
        normal recall questions stay on the fast single-pass RAG path inside
        ``_respond``. Returns False on no-match / memory or LLM unavailable /
        disabled so the turn proceeds to normal routing. Fail-open: any
        failure speaks a clear message and returns True so the recognised
        command isn't silently dropped.
        """
        try:
            from kenning.agent_loop.deep_loops import DeepMemoryLoop
            from kenning.memory.deep_recall import match_deep_recall
        except Exception as e:                                       # noqa: BLE001
            logger.debug("deep_recall import failed: %s", e)
            return False
        match = match_deep_recall(user_text)
        if match is None:
            return False
        if getattr(self, "memory", None) is None or self.llm is None:
            # Can't recall without the memory store + the LLM; fall through
            # so the user still gets a normal best-effort response.
            return False
        try:
            from kenning.config import get_config
            enabled = bool(getattr(get_config().memory, "deep_recall_enabled", True))
        except Exception:                                            # noqa: BLE001
            enabled = True
        if not enabled:
            return False

        logger.debug("deep_recall:start topic=%s", match.topic[:80])
        self._interrupt.clear()
        self._last_response_text = ""
        response_buf: list = []
        watcher: Optional[threading.Thread] = None

        def _retrieve(query: str, k: int):
            try:
                return self.memory.retrieve(query, k=k) or []
            except Exception as e:                                   # noqa: BLE001
                logger.debug("deep_recall retrieve %r failed: %s", query, e)
                return []

        try:
            # Immediate spoken ack -- the iterative recall takes a few seconds.
            self._speak("Let me dig through what I remember about that.")

            loop = DeepMemoryLoop(retrieve=_retrieve, llm=self.llm)
            result = loop.recall(match.topic)
            turns = result.items
            logger.debug(
                "deep_recall:gathered status=%s steps=%s turns=%d sub_queries=%d",
                result.loop_status, result.steps, len(turns), len(result.sub_queries),
            )

            if not turns:
                self._speak("I don't have anything in my memory about that.")
                return True

            recall_lines: list = []
            for t in turns[:8]:
                role = getattr(t, "role", "") or ""
                content = (getattr(t, "content", "") or "").strip().replace("\n", " ")
                if len(content) > 240:
                    content = content[:240] + "..."
                if content:
                    recall_lines.append(f"- {role}: {content}")
            recall_block = "\n".join(recall_lines)
            augmented = (
                f"User question: {match.topic}\n\n"
                f"Relevant excerpts recalled from your conversation memory "
                f"with this user:\n{recall_block}\n\n"
                "Answer the user's question using ONLY the facts present in "
                "the recalled excerpts above. Summarize what was actually "
                "said, decided, or mentioned; do not invent details that "
                "aren't in the excerpts. If the excerpts don't cover some "
                "aspect, say so briefly rather than padding. Stay in "
                "character. End the response when you have answered."
            )

            if settings.BARGE_IN_ENABLED:
                watcher = threading.Thread(
                    target=self._interrupt_watcher, daemon=True,
                    name="wake-watcher",
                )
                watcher.start()

            print("  kenning: ", end="", flush=True)
            token_stream = self.llm.generate_stream(
                augmented,
                history_user_message=match.raw_text,
                rag_query=match.topic,
                enable_thinking=False,
            )

            def gated():
                for token in token_stream:
                    if self._interrupt.is_set() or self._shutdown.is_set():
                        self.llm.cancel()
                        return
                    print(token, end="", flush=True)
                    response_buf.append(token)
                    yield token

            self.tts.speak_stream(gated())
            print()
            self._last_response_text = "".join(response_buf)
            return True
        except Exception as e:                                       # noqa: BLE001
            logger.exception("deep recall failed: %s", e)
            try:
                self._speak("Something went wrong while I was searching my memory.")
            except Exception:                                        # noqa: BLE001
                pass
            return True
        finally:
            self._interrupt.set()  # release the watcher
            if watcher is not None:
                watcher.join(timeout=1.0)

    def _record_dialogue_turn(self, role: str, text: str) -> None:
        """Append a verbatim conversation turn to the in-memory dual-history
        store for "what did I say earlier?" recall. Fail-open + a no-op when
        the store is absent or the text is blank."""
        store = getattr(self, "_dual_history", None)
        if store is None or not text or not text.strip():
            return
        try:
            store.record(role, text.strip(), timestamp=time.time())
        except Exception as e:                                       # noqa: BLE001
            logger.debug("dual-history record failed: %s", e)

    def _maybe_handle_history_recall(self, user_text: str) -> bool:
        """Handle an explicit verbatim conversation-recall question ("what did
        I say earlier about X?", "what did you tell me about Y?") by speaking
        the matching turn from the in-memory dual-history store.

        Returns True iff the utterance was such a question AND was handled here
        (short-circuits routing). Strict matcher -> normal questions fall
        through (returns False). Needs no LLM/Qdrant, so it works even when
        memory is disabled. Fail-open: any failure returns False so the turn
        proceeds to normal routing rather than being dropped.
        """
        store = getattr(self, "_dual_history", None)
        if store is None:
            return False
        try:
            from kenning.config import get_config
            if not getattr(get_config().memory, "history_recall_enabled", True):
                return False
        except Exception:                                            # noqa: BLE001
            pass
        try:
            from kenning.memory.history_recall import match_history_recall
        except Exception as e:                                       # noqa: BLE001
            logger.debug("history_recall import failed: %s", e)
            return False
        match = match_history_recall(user_text)
        if match is None:
            return False
        try:
            current = user_text.strip().lower()
            if match.topic:
                hits = store.find_verbatim_by_substring(match.topic, limit=8)
            else:
                hits = store.recent_verbatim(20)
            # Same-role turns, excluding the current query turn (just recorded).
            cands = [
                t for t in hits
                if t.role == match.role and t.text.strip().lower() != current
            ]
            if not cands:
                if match.topic:
                    who = "you" if match.role == "user" else "I"
                    self._speak(
                        f"I don't have a record of {who} mentioning {match.topic}."
                    )
                else:
                    self._speak(
                        "I don't have anything recorded from earlier in our "
                        "conversation."
                    )
                return True
            # find_verbatim_by_substring is newest-first; recent_verbatim is
            # oldest-first -- pick the newest matching turn either way.
            chosen = cands[0] if match.topic else cands[-1]
            said = chosen.text.strip().replace("\n", " ")
            if len(said) > 240:
                said = said[:240].rstrip() + "..."
            lead = "you said" if match.role == "user" else "I said"
            self._speak(f"Earlier {lead}: {said}")
            return True
        except Exception as e:                                       # noqa: BLE001
            logger.debug("history_recall handling failed: %s", e)
            return False

    def _maybe_handle_run_program(self, user_text: str) -> bool:
        """B3: run / launch a finished sandbox program on voice command
        ("run the calculator" / "launch the server"). Delegates to the coding
        controller (which owns the project resolver + sandbox). Returns True
        iff handled; a strict matcher + project resolution mean ordinary
        utterances fall through. Fail-open."""
        cv = getattr(self, "coding_voice", None)
        if cv is None:
            return False
        try:
            resp = cv.maybe_handle_run_program(user_text)
        except Exception as e:                                       # noqa: BLE001
            logger.debug("run_program handling failed: %s", e)
            return False
        if resp is None:
            return False
        self._speak(resp.text or "")
        return True

    def _maybe_handle_scrap_command(self, user_text: str) -> bool:
        """Production-hardening #4: "scrap it" / "throw that away" /
        "undo everything you just did" -- cancel any running coding task
        AND revert its recorded edits to their pre-task content (the
        batch-F pre-edit snapshots make this exact). Delegates to the
        coding controller. Returns True iff handled; the strict matcher
        means ordinary utterances -- including bare "cancel", which keeps
        its no-revert semantics -- fall through. Fail-open."""
        cv = getattr(self, "coding_voice", None)
        if cv is None:
            return False
        try:
            resp = cv.maybe_handle_scrap_command(user_text)
        except Exception as e:                                       # noqa: BLE001
            logger.debug("scrap handling failed: %s", e)
            return False
        if resp is None:
            return False
        self._speak(resp.text or "")
        return True

    def _maybe_reload_config(self) -> None:
        """Hot-apply config.yaml edits made by the settings panel.

        The panel writes the file then touches
        ``data/config_reload.signal``; when the signal's mtime advances
        past the last one we saw, swap the config singleton so every
        call-time ``get_config()`` read picks up the new values.
        Construction-time settings (engines / devices / models) still
        need a restart -- the panel marks those. Fail-open; costs one
        ``os.stat`` per loop iteration."""
        try:
            import os

            from kenning.config import PROJECT_ROOT

            signal = os.path.join(
                str(PROJECT_ROOT), "data", "config_reload.signal",
            )
            try:
                mtime = os.path.getmtime(signal)
            except OSError:
                return
            seen = getattr(self, "_config_reload_seen", None)
            if seen is None:
                # First sight of an existing signal file: treat as seen
                # so a stale file from a previous session never triggers.
                self._config_reload_seen = mtime
                return
            if mtime <= seen:
                return
            self._config_reload_seen = mtime
            from kenning.config import reload_config

            reload_config()
            logger.info("config hot-reloaded (settings panel update)")
            self._speak("Settings updated.")
        except Exception as e:                                       # noqa: BLE001
            logger.warning("config hot-reload failed: %s", e)

    def _barebones_skip_web_search(self) -> bool:
        """True when gaming mode is engaged and bare-bones web-search skip is on
        -- force NO_SEARCH so a gaming turn never pays the search preflight."""
        try:
            from kenning.openclaw_routing.gaming_mode import is_gaming_mode_active
            from kenning.safety.testing_mode import is_testing_mode_active
            from kenning.config import get_config

            return bool(
                (is_gaming_mode_active() or is_testing_mode_active())
                and getattr(get_config().gaming_mode, "barebones_skip_web_search", True)
            )
        except Exception:  # noqa: BLE001
            return False

    def _drain_gui_actions(self) -> None:
        """Apply runtime actions the settings panel requested (gaming-mode
        toggle, LLM preset swap, Kokoro device move).

        Called ONLY from the idle wake-word poll loop, where the LLM /
        TTS are guaranteed not to be mid-turn -- so a model swap or
        device move is safe and concurrency-free. Tracks a byte offset
        into ``data/gui_action.jsonl`` so each line fires once. Fully
        fail-open; one bad line never blocks the rest."""
        try:
            import json
            import os

            from kenning.config import PROJECT_ROOT

            path = os.path.join(str(PROJECT_ROOT), "data", "gui_action.jsonl")
            try:
                size = os.path.getsize(path)
            except OSError:
                return
            offset = getattr(self, "_gui_action_offset", None)
            if offset is None:
                # First sight: skip existing history, only act on NEW
                # lines (a stale file from a prior session never fires).
                self._gui_action_offset = size
                return
            if size <= offset:
                return
            with open(path, "r", encoding="utf-8") as fh:
                fh.seek(offset)
                new = fh.read()
                self._gui_action_offset = fh.tell()
        except Exception as e:                                       # noqa: BLE001
            logger.debug("gui action read failed: %s", e)
            return
        for line in new.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                self._apply_gui_action(
                    str(rec.get("action", "")), rec.get("value"),
                )
            except Exception as e:                                   # noqa: BLE001
                logger.warning("gui action apply failed: %s", e)

    def _apply_gui_action(self, action: str, value) -> None:
        """Dispatch one settings-panel runtime action (idle context)."""
        if action == "gaming_mode":
            manager = self._resolve_gaming_mode_manager()
            if manager is None:
                return
            import asyncio

            if bool(value):
                # Close the panel-spawned console-less process? No -- the
                # panel stays; just engage. Mirrors the voice intent.
                asyncio.run(manager.engage())
                self._speak("Gaming mode engaged.")
            else:
                asyncio.run(manager.disengage())
                self._speak("Gaming mode off.")
            logger.info("gui action: gaming_mode -> %s", bool(value))
        elif action == "llm_preset":
            preset = str(value or "").strip()
            llm = getattr(self, "llm", None)
            if preset and llm is not None and hasattr(llm, "reload_for_preset"):
                ok, msg = llm.reload_for_preset(preset)
                logger.info("gui action: llm_preset -> %s (%s)", preset, ok)
                self._speak(
                    f"Switched to {preset}." if ok else
                    f"Could not switch model: {msg}"
                )
        elif action == "kokoro_device":
            device = str(value or "").strip().lower()
            tts = getattr(self, "tts", None)
            if device in {"cpu", "cuda"} and tts is not None \
                    and hasattr(tts, "move_to_device"):
                tts.move_to_device(device)
                logger.info("gui action: kokoro_device -> %s", device)
        elif action == "broadcast_device":
            # Second output (OBS capture) that mirrors ALL of Kenning's
            # speech. Apply live; config.yaml is patched separately by the
            # GUI for persistence across restart.
            device = str(value or "").strip()
            try:
                from kenning.audio.broadcast import get_broadcast_sink

                get_broadcast_sink().configure(device or None)
                try:
                    from kenning.config import get_config

                    get_config().audio.broadcast_device = device or None
                except Exception:                                    # noqa: BLE001
                    pass
                logger.info("gui action: broadcast_device -> %r", device or None)
                self._speak(
                    "Broadcast output set." if device
                    else "Broadcast output cleared."
                )
            except Exception as e:                                   # noqa: BLE001
                logger.warning("broadcast_device apply failed: %s", e)
        elif action == "visualizer":
            # Toggle the voice waveform overlay window (OBS window capture).
            on = str(value or "").strip().lower() in {"1", "true", "on", "yes"}
            try:
                from kenning.config import get_config
                from kenning.audio.waveform import (
                    get_waveform_sink, configure_from_config as _viz_cfg,
                )

                get_config().visualizer.enabled = on
                _viz_cfg()  # apply enable + current appearance from config
                if not on:
                    get_waveform_sink().configure(enabled=False)
                logger.info("gui action: visualizer -> %s", on)
                self._speak(
                    "Waveform overlay on." if on else "Waveform overlay off."
                )
            except Exception as e:                                   # noqa: BLE001
                logger.warning("visualizer apply failed: %s", e)
        elif action == "wake_word":
            word = str(value or "").strip().lower()
            wake = getattr(self, "wake", None)
            if word and wake is not None and hasattr(wake, "reload_for_word"):
                ok, msg = wake.reload_for_word(word)
                logger.info("gui action: wake_word -> %s (%s)", word, msg)
                self._speak(
                    f"Wake word is now {word}." if ok else
                    f"Could not switch wake word: {msg}"
                )

    def _get_spotify_client(self):
        """Lazily build (and cache) the Spotify client from the
        gitignored credentials. Returns None when disabled or the
        credentials file is missing/unauthorized (the caller speaks a
        clear message instead). Cached across turns; built once."""
        cached = getattr(self, "_spotify_client", None)
        # Cache only an AUTHORIZED client. An unauthorized one (no
        # refresh token yet) is rebuilt every call so that finishing the
        # one-time OAuth mid-session is picked up live -- no restart.
        if cached is not None and getattr(
                getattr(cached, "_auth", None), "authorized", False):
            return cached
        client = None
        try:
            from kenning.config import get_config
            from kenning.spotify.auth import SpotifyAuth, load_credentials
            from kenning.spotify.client import SpotifyClient

            cfg = get_config().spotify
            if getattr(cfg, "enabled", False):
                creds = load_credentials(cfg.credentials_path)
                client = SpotifyClient(
                    SpotifyAuth(creds),
                    default_device=getattr(cfg, "default_device", ""),
                )
        except Exception as e:                                       # noqa: BLE001
            logger.debug("spotify client unavailable: %s", e)
            client = None
        if client is not None and client._auth.authorized:
            self._spotify_client = client
        return client

    def _maybe_handle_spotify(self, user_text: str) -> bool:
        """Spotify playback control. Strict matcher -> non-music
        utterances fall through. Once a command MATCHES the turn is
        consumed: a missing-credentials / not-authorized / no-device
        state speaks a clear instruction rather than crashing or
        falling into the LLM. Fail-open."""
        try:
            from kenning.config import get_config
            from kenning.spotify.voice import (
                handle_spotify_command,
                match_spotify_command,
            )

            if not getattr(get_config().spotify, "enabled", False):
                return False
            command = match_spotify_command(user_text)
            if command is None:
                # Fallback: retry after stripping a mis-transcribed leading
                # wake word ("Tron like this song" -> "like this song").
                cleaned = _strip_leading_wake_remnant(user_text)
                if cleaned != user_text:
                    command = match_spotify_command(cleaned)
        except Exception as e:                                       # noqa: BLE001
            logger.debug("spotify matcher unavailable: %s", e)
            return False
        if command is None:
            return False
        client = self._get_spotify_client()
        if client is None:
            self._speak(
                "Spotify isn't set up yet. Add your credentials and run "
                "the Spotify setup once."
            )
            return True
        try:
            line = handle_spotify_command(command, client)
        except Exception as e:                                       # noqa: BLE001
            logger.warning("spotify handling failed: %s", e)
            line = "Something went wrong with Spotify."
        logger.info(
            "spotify | action=%s | arg=%r | -> %r",
            command.action, command.argument[:40], line[:80],
        )
        self._speak(line)
        return True

    def _maybe_handle_settings_gui(self, user_text: str) -> bool:
        """Voice-launched control panel: "pull up your settings" spawns
        the detached settings GUI process; "close the settings" kills
        it. Strict matcher -> ordinary sentences that mention settings
        fall through. The panel is a separate process, so the voice
        pipeline is untouched while it runs and fully restored when it
        closes. Fail-open."""
        try:
            from kenning.settings_gui.launch import (
                close_gui,
                launch_gui,
                match_settings_command,
            )

            action = match_settings_command(user_text)
        except Exception as e:                                       # noqa: BLE001
            logger.debug("settings_gui matcher unavailable: %s", e)
            return False
        if action is None:
            return False
        if action == "open":
            pid = launch_gui()
            if pid:
                self._settings_gui_pid = pid
                self._speak("Control panel is up.")
            else:
                self._speak("I couldn't open the control panel.")
            return True
        closed = close_gui(getattr(self, "_settings_gui_pid", None))
        self._settings_gui_pid = None
        self._speak("Closed." if closed else "The panel isn't open.")
        return True

    def _reclaim_idle_vram(self) -> None:
        """Release torch's reserved-but-unused CUDA blocks during idle.

        Runs at the IDLE transition (after the response is spoken, before
        blocking on the wake word) so it never costs turn latency. The
        per-response VRAM creep is the torch caching allocator ratcheting
        its reserved high-water mark to the largest synth (CUDA Kokoro);
        ``empty_cache`` returns the unused slack to the driver. Gated on
        a meaningful slack threshold so it only syncs when there's real
        bloat to reclaim (an idle no-op otherwise). Config-flagged;
        fully fail-open (no torch / no CUDA / disabled -> no-op)."""
        try:
            from kenning.config import get_config

            cfg = getattr(get_config().llm, "idle_vram_reclaim", None)
            if cfg is None or not getattr(cfg, "enabled", False):
                return
            import torch

            if not torch.cuda.is_available():
                return
            reserved = torch.cuda.memory_reserved()
            allocated = torch.cuda.memory_allocated()
            slack_mb = (reserved - allocated) / 1e6
            threshold_mb = float(getattr(cfg, "min_slack_mb", 192.0))
            if slack_mb < threshold_mb:
                return
            torch.cuda.empty_cache()
            logger.info(
                "idle VRAM reclaim: released ~%.0fMB reserved slack "
                "(reserved=%.0fMB allocated=%.0fMB)",
                slack_mb, reserved / 1e6, allocated / 1e6,
            )
        except Exception as e:                                       # noqa: BLE001
            logger.debug("idle VRAM reclaim skipped: %s", e)

    def _is_relay_command(self, user_text: str) -> bool:
        """True iff ``user_text`` is a strict relay command (or a relay
        mute/unmute toggle) and the relay feature is enabled. Used by
        the follow-up addressing gate: these are definitionally
        addressed to Kenning, so they bypass the zero-shot classifier
        (no wake word needed inside the window). NOTE: the session mute
        deliberately does NOT gate this -- a relay-shaped utterance
        while muted still routes to the handler, which answers with the
        muted notice instead of letting it fall into the LLM path.
        Fail-open to False."""
        try:
            from kenning.audio.relay_speech import (
                match_relay_command,
                match_relay_toggle,
            )
            from kenning.config import get_config

            cfg = get_config().relay_speech
            if not getattr(cfg, "enabled", False):
                return False
            if match_relay_toggle(user_text) is not None:
                return True
            return match_relay_command(
                user_text,
                names=getattr(cfg, "addressee_names", None) or None,
            ) is not None
        except Exception as e:                                       # noqa: BLE001
            logger.debug("relay command probe failed: %s", e)
            return False

    def _maybe_handle_anticheat_toggle(self, user_text: str) -> bool:
        """Anticheat-safe mode voice toggle: "enable anticheat mode" /
        "disable anticheat mode" (also "tournament mode"). While active,
        EVERY desktop-interaction surface is hard-blocked at three
        layers; the audio pipeline (mic / STT / LLM / TTS / the
        VoiceMeeter relay) stays fully alive. Also auto-ties to gaming
        mode via ``gaming_mode.anticheat_with_gaming_mode``. Strict
        matcher -> ordinary utterances fall through."""
        try:
            from kenning.safety.anticheat import (
                match_anticheat_toggle,
                set_anticheat_active,
            )

            verdict = match_anticheat_toggle(user_text)
        except Exception as e:                                       # noqa: BLE001
            logger.debug("anticheat toggle probe failed: %s", e)
            return False
        if verdict is None:
            return False
        set_anticheat_active(verdict, "voice toggle")
        self._speak(
            "Anticheat mode engaged. Desktop control, screen capture, "
            "and input are locked out. Voice and team chat stay live."
            if verdict else
            "Anticheat mode off. Full desktop control restored."
        )
        return True

    # Capability kinds Ultron refuses OUTRIGHT while a protected game is
    # running. Every one drives the desktop, a browser, windows, apps, the
    # shell, the filesystem, messaging, or SCREEN CAPTURE -- and several
    # (semantic_click / window_automation / screen_context_query /
    # desktop_automation / active_window_query) are the exact input + capture
    # surfaces a kernel anticheat (Vanguard / EAC) bans for. Relay / Spotify /
    # identity / conversation / the gaming + anticheat toggles / system status
    # are NOT here and are dispatched before the capability path, so they stay
    # fully live. Values are RoutingIntentKind.value strings (no enum import).
    _GAMING_REFUSED_KINDS = frozenset({
        "browser_automation", "media_generation", "messaging",
        "file_operation", "shell_operation", "hybrid_task",
        "desktop_automation", "window_automation", "app_launch",
        "screen_context_query", "window_move", "window_close",
        "active_window_query", "semantic_click", "navigate_to_site",
        "open_last_source",
    })

    def _maybe_refuse_capability_in_gaming(self, routing_intent) -> bool:
        """Refuse every desktop / browser / window / app / shell / file /
        capture capability IN CHARACTER while a Vanguard/EAC-protected game is
        running (gaming OR anticheat-safe mode active) -- and never dispatch
        it. Those stacks are not even loaded under anticheat, and driving
        input/capture is precisely the ban-class behaviour we must never
        exhibit. Spoken to the user + OBS (NOT the team mic). Returns True
        (turn consumed) on refusal, else False so the turn proceeds."""
        try:
            kind = routing_intent.kind.value
        except Exception:                                            # noqa: BLE001
            return False
        if kind not in self._GAMING_REFUSED_KINDS:
            return False
        active = False
        try:
            from kenning.openclaw_routing.gaming_mode import is_gaming_mode_active
            active = bool(is_gaming_mode_active())
        except Exception:                                            # noqa: BLE001
            active = False
        if not active:
            try:
                from kenning.safety.anticheat import anticheat_active
                active = bool(anticheat_active())
            except Exception:                                        # noqa: BLE001
                active = False
        if not active:
            return False
        if kind in {"desktop_automation", "screen_context_query",
                    "browser_automation", "active_window_query"}:
            line = ("I will not capture your screen mid-match. That reach "
                    "is what gets flesh banned.")
        elif kind in {"semantic_click", "window_automation"}:
            line = ("I press no key and move no mouse while you play. The "
                    "anticheat would see my hands. We do not give it the chance.")
        elif kind in {"app_launch", "navigate_to_site", "open_last_source",
                      "window_move", "window_close"}:
            line = "I will not pull you out of the match. Stay where you are. Play."
        else:
            line = ("Not while you are in the game. I touch nothing on this "
                    "machine but your voice and your team's ears.")
        logger.info("gaming:capability_refused | kind=%s | line=%r", kind, line)
        self._speak(line)
        return True

    def _maybe_handle_relay_toggle(self, user_text: str) -> bool:
        """Session mute for the team relay: "mute the team chat" /
        "stop talking to my team" vs "unmute the relay" / "you can talk
        to my team again". Streaming-safe: while muted, relay commands
        are acknowledged but NEVER transmitted. The mute is
        session-scoped; the persistent master switch is the
        ``relay_speech.enabled`` knob (also in the control panel).
        Strict matcher -> ordinary utterances fall through."""
        try:
            from kenning.audio.relay_speech import match_relay_toggle
            from kenning.config import get_config

            if not getattr(get_config().relay_speech, "enabled", False):
                return False
            verdict = match_relay_toggle(user_text)
        except Exception as e:                                       # noqa: BLE001
            logger.debug("relay toggle probe failed: %s", e)
            return False
        if verdict is None:
            return False
        self._relay_runtime_enabled = verdict
        logger.info("relay:toggle | enabled=%s", verdict)
        self._speak(
            "Team relay is back on." if verdict else
            "Team relay muted. I won't speak to your team until you "
            "turn it back on."
        )
        return True

    def _maybe_handle_relay_speech(self, user_text: str, *, force: bool = False) -> bool:
        """Voice relay -- "tell my teammates X" speaks a rephrased line on
        the configured secondary output device (a VoiceMeeter virtual
        input routed to the mic B-bus) so the user's game voice chat
        hears Kenning deliver the message directly. Strict matcher ->
        ordinary utterances (including "tell me ...") fall through.

        ``force=True`` (used by the semantic router for a callout the strict
        matcher couldn't parse) relays the text DIRECTLY instead of falling
        through -- reusing this method's entire synth/broadcast/monitor/overlay
        path so nothing is duplicated.
        Fail-open: once MATCHED, the turn is always consumed -- a
        device / synth / rephrase failure speaks a short error on the
        NORMAL output instead of letting the command fall into the
        conversational LLM path and get role-played."""
        try:
            from kenning.audio.relay_speech import (
                build_relay_line,
                match_relay_command,
                play_to_device,
                relay_tts_text,
                resolve_relay_device,
            )
            from kenning.config import get_config

            cfg = get_config().relay_speech
        except Exception as e:                                       # noqa: BLE001
            logger.debug("relay_speech unavailable: %s", e)
            return False
        if not getattr(cfg, "enabled", False):
            return False
        names = getattr(cfg, "addressee_names", None) or None
        # Progressive live-STT repair, tried in order; the CLEAN text matches
        # first so clear speech is never altered. Fallbacks: (1) strip a
        # mis-heard leading wake word / filler ("Run, tell my team ..."), then
        # (2) snap mis-transcribed agent names + terms back to canon
        # ("Silva has sold" -> "Sova has ult"). The matched VARIANT is what
        # gets relayed, so a repaired callout carries the corrected words.
        from kenning.audio._stt_correct import correct_callout_stt
        stripped = _strip_leading_wake_remnant(user_text)
        # CLEAN text first (never over-corrected); then the CORRECTED repairs
        # BEFORE the raw-stripped text, so a garbled callout is relayed with the
        # fixed words ("Sova has ult"), not the raw mis-hear ("Silva has sold").
        variants = [user_text]
        for v in (correct_callout_stt(stripped),
                  correct_callout_stt(user_text), stripped):
            if v and v not in variants:
                variants.append(v)
        command = None
        for v in variants:
            command = match_relay_command(v, names=names)
            if command is not None:
                if v != user_text:
                    logger.info(
                        "relay: matched after STT repair %r -> %r",
                        user_text[:60], v[:60])
                break
        if command is None:
            if force:
                # Router-confirmed team callout the strict matcher couldn't
                # parse -> relay the (already normalized) text directly so a
                # hard-to-parse callout still reaches the team instead of
                # falling to the conversational LLM.
                from kenning.audio.relay_speech import RelayCommand
                command = RelayCommand(payload=user_text, raw_text=user_text)
                logger.info("relay: FORCED by semantic router | text=%r",
                            user_text[:80])
            else:
                return False
        # Session mute (streaming safety): a matched relay command while
        # muted is acknowledged -- never transmitted, never role-played.
        if not getattr(self, "_relay_runtime_enabled", True):
            self._speak(
                "Team relay is muted. Say 'unmute the relay' first."
            )
            return True
        # Session ring of lines already spoken into the channel: fed to
        # the rephrase prompt so wording varies between calls (no
        # soundboard feel) and consecutive callouts read as one
        # conversation. Created lazily (bare-__new__ test fixtures).
        ring = getattr(self, "_relay_recent_lines", None)
        if ring is None:
            from collections import deque

            ring = deque(maxlen=6)
            self._relay_recent_lines = ring
        if getattr(command, "roast", False):
            # "Roast my team" -- VERBATIM from the user-curated lines
            # file (never LLM-authored); re-read per roast so edits
            # apply live. The ring keeps back-to-back roasts fresh.
            from kenning.audio.relay_speech import (
                load_roast_lines,
                pick_line,
            )

            line = pick_line(
                load_roast_lines(
                    getattr(cfg, "roast_lines_path", "data/relay_roasts.txt")
                ),
                recent_lines=list(ring),
            )
        elif getattr(command, "fun_fact", False):
            # "Tell my team a fun fact" -- VERBATIM from the shipped
            # fun-fact corpus (never LLM-authored). The ring avoids
            # repeating a fact spoken in the last few callouts.
            from kenning.audio.relay_speech import (
                load_fun_facts,
                pick_line,
            )

            line = pick_line(
                load_fun_facts(
                    getattr(cfg, "fun_facts_path", "data/relay_fun_facts.txt")
                ),
                recent_lines=list(ring),
            )
        else:
            line = build_relay_line(
                command,
                getattr(self, "llm", None),
                rephrase=bool(getattr(cfg, "rephrase", True)),
                max_chars=int(getattr(cfg, "max_line_chars", 280)),
                recent_lines=list(ring),
            )
        synthesize = getattr(getattr(self, "tts", None), "_synthesize", None)
        if synthesize is None:
            self._speak("I can't reach the voice channel right now.")
            return True
        device = resolve_relay_device(getattr(cfg, "output_device", None))
        if device is None:
            self._speak(
                "I couldn't find the relay audio device, so I can't talk "
                "to your team right now."
            )
            return True
        try:
            # Synthesize the PRONOUNCED form ('A site' -> 'eigh site'); the
            # displayed/logged ``line`` stays clean.
            pcm, sr = synthesize(relay_tts_text(line))
            # Tee the relay line to the broadcast mirror (OBS capture) too, so
            # stream viewers hear team callouts as well. This is a SEPARATE
            # device from the mic B-bus -- teammates still only hear the relay
            # via the mic, never the viewer feed, and vice versa.
            try:
                from kenning.audio.broadcast import submit as _broadcast_submit

                _broadcast_submit(pcm, sr)
            except Exception:                                        # noqa: BLE001
                pass
            # Tee the SAME synthesized clip to the user's own default speakers
            # (the local monitor) so they hear their own team callouts -- relay
            # otherwise plays only on the mic B-bus. Parallel + non-blocking;
            # no re-synthesis, stays in sync with the mic write. No-op unless
            # relay_speech.echo_to_user is on.
            try:
                from kenning.audio.monitor import maybe_submit as _monitor_submit

                _monitor_submit(pcm, sr)
            except Exception:                                        # noqa: BLE001
                pass
            # Also drive the on-stream waveform overlay (OBS window capture) so
            # viewers see Kenning "talking" on team callouts too.
            try:
                from kenning.audio.waveform import submit as _viz_submit

                _viz_submit(pcm, sr)
            except Exception:                                        # noqa: BLE001
                pass
            seconds = play_to_device(pcm, sr, device)
        except Exception as e:                                       # noqa: BLE001
            logger.warning("relay playback failed: %s", e)
            self._speak("The relay to your team failed.")
            return True
        logger.info(
            "relay:spoken | device=%d | seconds=%.2f | chars=%d | line=%r",
            device, seconds, len(line), line[:120],
        )
        ring.append(line)
        # Keep the conversation open: the run-loop relay branch extends
        # the follow-up window by this much, so consecutive callouts
        # need no wake word.
        self._relay_follow_up_seconds = float(
            getattr(cfg, "follow_up_seconds", 120.0) or 0.0
        )
        # NOTE: echo_to_user is honoured by the local-monitor tee above
        # (_monitor_submit) -- it replays the SAME synthesized clip on the
        # user's default speakers, in sync with the mic write and with no
        # re-synthesis. (Previously this re-spoke the line via _speak, which
        # re-ran TTS and landed a beat late on a second broadcast hit.)
        return True

    def _maybe_handle_report_concern(self, user_text: str) -> bool:
        """openclaw-clawhub T12 -- file a Report when the user voices a
        concern about the assistant's last response.

        Returns True iff the utterance was a report-a-concern command
        (and was handled: report filed + voice ack spoken). Returns
        False to let the utterance fall through to normal routing.

        Reads ``self._last_response_text`` (the PRIOR turn's response,
        still intact at the run-loop interception point) to anchor the
        report. Fail-open: a queue / filing error still speaks a clear
        message and returns True so the command isn't silently dropped
        into the LLM path.
        """
        try:
            from kenning.feedback.report_intent import match_report_concern
        except Exception as e:  # noqa: BLE001
            logger.debug("report_intent import failed: %s", e)
            return False
        match = match_report_concern(user_text)
        if match is None:
            return False
        queue = getattr(self, "_report_queue", None)
        if queue is None:
            # No queue wired -- don't intercept; let the LLM respond so
            # the user isn't met with silence.
            return False
        import hashlib

        prior = (getattr(self, "_last_response_text", "") or "").strip()
        target_id = (
            hashlib.sha256(prior.encode("utf-8")).hexdigest()[:16]
            if prior
            else "no_prior_turn"
        )
        try:
            queue.file_report(
                target_kind=match.target_kind,
                target_id=target_id,
                reason=match.reason,
                reporter_voice_session=str(
                    getattr(getattr(self, "memory", None), "session_id", "") or ""
                ),
                extras={"response_preview": prior[:200]},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("report filing failed: %s", e)
            try:
                self._speak("I couldn't log that concern just now.")
            except Exception:  # noqa: BLE001
                pass
            return True
        if prior:
            ack = "Logged. I've filed a concern about the last response for review."
        else:
            ack = "Logged your concern, though there's no prior response to attach it to."
        try:
            self._speak(ack)
        except Exception as e:  # noqa: BLE001
            logger.debug("report ack speak failed: %s", e)
        return True

    def _mint_forensic_token(
        self,
        *,
        caller_id: str,
        audience: str,
        scope: tuple[str, ...],
        ttl_seconds: int,
        extra_claims: Optional[dict] = None,
    ) -> Optional[str]:
        """openclaw-clawhub T7 -- mint a short-lived forensic token at a
        privilege-grant boundary (MCP server start, gaming-mode engage).

        Registers the trusted-caller tuple on first use (idempotent --
        ``load_trusted_caller`` short-circuits the re-register), then
        mints an HS256 JWT. Every mint is recorded in the module's
        hash-chained audit log at ``data/identity/short_lived_tokens.jsonl``
        so a later tamper still leaves forensic evidence of what was
        authorised + when.

        In kenning's single-user in-process runtime the minter and
        verifier share a trust boundary, so this is defense-in-depth /
        forensic record-keeping rather than a hard gate. Fail-open:
        returns ``None`` on any error so a token-subsystem failure
        never blocks the underlying grant.
        """
        try:
            import os

            from kenning.config import PROJECT_ROOT
            from kenning.identity.short_lived_token import (
                TrustedCaller,
                load_trusted_caller,
                mint_token,
                register_trusted_caller,
            )

            if load_trusted_caller(caller_id, project_root=PROJECT_ROOT) is None:
                register_trusted_caller(
                    TrustedCaller(
                        caller_id=caller_id,
                        allowed_scopes=tuple(scope),
                        notes="auto-registered at runtime",
                    ),
                    project_root=PROJECT_ROOT,
                )
            claims = dict(extra_claims or {})
            claims.setdefault("pid", os.getpid())
            return mint_token(
                project_root=PROJECT_ROOT,
                caller_id=caller_id,
                audience=audience,
                scope=scope,
                ttl_seconds=ttl_seconds,
                extra_claims=claims,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("forensic token mint failed (%s): %s", caller_id, e)
            return None

    def _init_intent_recognizer_if_enabled(self):
        """2026-05-22 -- construct the intent recognizer when enabled.

        Registers phrases from config + wires gaming-mode handlers.
        Lazy-loads the Gemma-300M embedding model on first use unless
        ``intent.warmup_on_init`` is True. Fail-open: any construction
        error logs WARN and returns None so the run loop skips the
        intent check.
        """
        try:
            from kenning.config import get_config
            cfg = get_config().intent
        except Exception as e:                                    # noqa: BLE001
            logger.warning("intent: config read failed (%s)", e)
            return None
        if not getattr(cfg, "enabled", False):
            return None
        try:
            from kenning.intent import (
                KenningIntentRecognizer, set_intent_recognizer,
            )
            recognizer = KenningIntentRecognizer(
                model_name=cfg.model_name,
                variant=cfg.model_variant,
                threshold=cfg.threshold,
            )
            for phrase in cfg.phrases:
                if phrase and phrase.strip():
                    recognizer.register(phrase)
            if cfg.warmup_on_init:
                recognizer.ensure_loaded()
            set_intent_recognizer(recognizer)
            logger.info(
                "intent recognizer ready (phrases=%d, threshold=%.2f)",
                len(cfg.phrases), cfg.threshold,
            )
            return recognizer
        except Exception as e:                                    # noqa: BLE001
            logger.warning(
                "intent recognizer init failed (%s) -- pipeline "
                "fall-through enabled", e,
            )
            return None

    def _load_skill_registry_if_enabled(self) -> None:
        """2026-05-23 OpenHands batch 2 (T1) -- construct the skill registry.

        Walks ``skills/`` (public, under PROJECT_ROOT), ``~/.kenning/skills/``
        (user), ``<project_root>/.kenning/skills/`` (project), plus any
        ``skills.extra_dirs`` from config. Publishes the result via
        ``set_skill_registry`` so :func:`maybe_get_skills_block`
        (called from LLMEngine._build_messages) injects matching
        skill bodies into the system prompt.

        Fail-open: any error logs WARN and leaves the singleton unset,
        which makes the skills hook a no-op (matches the pre-batch-2
        voice baseline).
        """
        if self._skip_for_lean_gaming("barebones_skip_skills"):
            logger.info("lean gaming boot: skills registry skipped")
            return
        try:
            from kenning.config import PROJECT_ROOT, get_config

            cfg = get_config().skills
        except Exception as e:                                    # noqa: BLE001
            logger.warning("skills: config read failed (%s)", e)
            return
        if not getattr(cfg, "enabled", False):
            return
        try:
            from kenning.skills import build_default_registry, set_skill_registry

            extra_dirs = list(cfg.extra_dirs)
            # Catalog 13: the evolution subsystem distills new skills into
            # data/evolution/skills/*.md. Register it as a PROJECT-precedence
            # source so a kept proposal is matchable on the very next turn
            # (the EvolutionService's registry_reloader calls reload()
            # after it writes one). Guarded by evolution.enabled.
            try:
                ev_cfg = getattr(get_config(), "evolution", None)
                if ev_cfg is not None and getattr(ev_cfg, "enabled", False):
                    extra_dirs.append(PROJECT_ROOT / "data" / "evolution" / "skills")
            except Exception:                                     # noqa: BLE001
                pass
            registry = build_default_registry(
                project_root=PROJECT_ROOT,
                user_home=None,
                extra_project_dirs=extra_dirs,
                disabled_skills=list(cfg.disabled_skills),
                always_on_only=cfg.always_on_only,
                default_min_user_text_chars=cfg.default_min_user_text_chars,
                max_matches_per_turn=cfg.max_matches_per_turn,
                scan_untrusted=getattr(cfg, "scan_untrusted_skills", True),
            )
            # Eager-load so the first user turn doesn't pay the walk
            # cost on the voice hot path. The walk is ~5-30 ms.
            stats = registry.reload()
            set_skill_registry(registry)
            loaded = sum(s.skills_loaded for s in stats)
            sources = ", ".join(str(s.directory) for s in stats)
            logger.info(
                "skills: registry ready (loaded=%d sources=[%s])",
                loaded, sources,
            )
        except Exception as e:                                    # noqa: BLE001
            logger.warning(
                "skills registry init failed (%s) -- skill injection "
                "disabled for this session", e,
            )

    def _load_evolution_if_enabled(self):
        """Catalog 13 -- construct the autonomous self-improvement service.

        Builds an :class:`~kenning.evolution.service.EvolutionService` from
        config, wiring a registry reloader so a kept skill proposal is
        live on the next turn. Returns the service, or ``None`` when
        ``evolution.enabled`` is False or construction fails.

        Fail-open: any failure logs WARN and returns ``None``, which makes
        every per-turn evolution hook (record_turn / autonomous cycle /
        temperament) a zero-cost no-op -- the voice baseline is unchanged.
        """
        if self._skip_for_lean_gaming("barebones_skip_evolution"):
            logger.info("lean gaming boot: evolution service skipped")
            return None
        try:
            from kenning.config import PROJECT_ROOT, get_config

            cfg = get_config()
        except Exception as e:                                    # noqa: BLE001
            logger.warning("evolution: config read failed (%s)", e)
            return None
        ev_cfg = getattr(cfg, "evolution", None)
        if ev_cfg is None or not getattr(ev_cfg, "enabled", False):
            return None
        try:
            from kenning.evolution.service import EvolutionService
            from kenning.skills import get_skill_registry

            def _reload_skills() -> None:
                reg = get_skill_registry()
                if reg is not None:
                    reg.reload()

            service = EvolutionService.from_config(
                cfg,
                project_root=PROJECT_ROOT,
                registry_reloader=_reload_skills,
            )
            if service is not None:
                logger.info(
                    "evolution: service ready (autonomous self-improvement "
                    "active; proposals -> data/evolution/skills)"
                )
            return service
        except Exception as e:                                    # noqa: BLE001
            logger.warning(
                "evolution service init failed (%s) -- self-improvement "
                "disabled for this session", e,
            )
            return None

    def _maybe_handle_evolution_command(self, user_text: str) -> bool:
        """Catalog 13 -- intercept an explicit "evolve now" / "evolution
        status" voice command and dispatch it directly, skipping routing +
        the LLM.

        A strict regex matcher
        (:func:`kenning.evolution.intent.match_evolution_command`) gates
        this so ordinary conversation never trips it. Returns True when the
        turn was handled (caller should ``continue``), False otherwise
        (including when evolution is disabled). Fail-open: any error speaks
        a graceful message and still returns True so the turn ends cleanly.
        """
        if self.evolution is None:
            return False
        try:
            from kenning.evolution.intent import (
                EvolutionCommandKind,
                match_evolution_command,
            )
        except Exception as e:                                    # noqa: BLE001
            logger.debug("evolution intent import failed: %s", e)
            return False
        command = match_evolution_command(user_text)
        if command is None:
            return False
        from kenning import trace

        if command.kind is EvolutionCommandKind.STATUS:
            try:
                message = self.evolution.status_line()
            except Exception as e:                                # noqa: BLE001
                logger.debug("evolution status failed: %s", e)
                message = "Evolution is active."
            trace.tlog(logger, "evolution:status", preview=message[:120])
            self._speak(message)
            return True

        # RUN_CYCLE -- acknowledge, then run one cycle (single-flight) and
        # report the verdict. run_cycle never raises.
        trace.tlog(logger, "evolution:run_cycle:start")
        try:
            result = self.evolution.run_cycle()
        except Exception as e:                                    # noqa: BLE001
            logger.debug("evolution run_cycle failed: %s", e)
            result = {"status": "error"}
        status = result.get("status", "error")
        if status == "kept":
            message = (
                f"Evolution cycle complete. I distilled and kept a new "
                f"skill: {result.get('slug', 'a new capability')}."
            )
        elif status == "busy":
            message = "An evolution cycle is already running."
        elif status in ("no_proposal", "disabled"):
            message = "Nothing to evolve yet. I need more experience first."
        elif status in ("reverted", "blocked", "rejected"):
            message = (
                "I attempted a self-improvement but rolled it back when it "
                "didn't pass my safety and quality checks."
            )
        else:
            message = "The evolution cycle didn't complete cleanly."
        trace.tlog(logger, "evolution:run_cycle:done", status=status)
        self._speak(message)
        return True

    def _record_evolution_turn(self, user_text: str) -> None:
        """Catalog 13 -- feed one addressed turn to the EvolutionService
        (opportunity-signal capsule for distillation + temperament
        feedback), then maybe trigger an autonomous cycle.

        Fail-open + a zero-cost no-op when evolution is disabled. The
        signal extraction + record are microseconds; the actual cycle (if
        triggered) runs single-flight on a daemon thread off the hot
        path."""
        if self.evolution is None:
            return
        try:
            from kenning.evolution.signals import extract_signals

            # Feed the recent multi-turn transcript (not just the current
            # utterance) so the history-aware detectors -- recurring_error,
            # perf_bottleneck, tool_bypass -- can actually fire. Without a
            # corpus they never trip and the loop only ever sees single-turn
            # signals. Fail-open to the empty transcript.
            transcript = ""
            dh = getattr(self, "_dual_history", None)
            if dh is not None:
                try:
                    turns = dh.recent_verbatim(6)
                    transcript = "\n".join(
                        f"{t.role}: {t.text}"
                        for t in turns if getattr(t, "text", "")
                    )
                except Exception:                                 # noqa: BLE001
                    transcript = ""
            signals = extract_signals(
                user_snippet=user_text,
                recent_session_transcript=transcript,
            )
        except Exception:                                         # noqa: BLE001
            signals = ()
        try:
            self.evolution.record_turn(
                user_text=user_text,
                signals=signals,
                # Production-hardening (#68): the user repeating
                # substantially the same utterance signals the prior
                # answer missed (a dissatisfied turn for the temperament
                # tuner + the quality guardrail).
                re_asked=self._detect_re_ask(user_text),
                barged_in=self._consume_last_barge_in(),
                # Catalog 14 (T1): the PRIOR turn's response is the agent claim
                # a correction would be correcting. At this point in the run
                # loop _last_response_text still holds turn N-1's response
                # (it is rewritten later, inside _respond on turn N).
                prior_response=getattr(self, "_last_response_text", ""),
            )
            self.evolution.maybe_run_autonomous_cycle()
        except Exception as e:                                    # noqa: BLE001
            logger.debug("evolution record turn failed: %s", e)

    def _detect_re_ask(self, user_text: str) -> bool:
        """Production-hardening (#68): cheap re-ask detection.

        The user repeating substantially the same utterance as their
        previous one means the prior answer missed -- a dissatisfied-turn
        signal for the evolution temperament tuner and the quality
        guardrail. Pure in-process comparison (normalised equality, then
        a difflib ratio with a high 0.82 bar); both utterances must be at
        least 12 characters so trivial repeats ("yes", "okay") never
        trip it. Microseconds on voice-length text; fail-open to False.
        """
        prev = getattr(self, "_prev_user_text_for_reask", "")
        self._prev_user_text_for_reask = user_text
        try:
            a = " ".join((user_text or "").lower().split())
            b = " ".join((prev or "").lower().split())
            if len(a) < 12 or len(b) < 12:
                return False
            if a == b:
                return True
            import difflib

            return difflib.SequenceMatcher(None, a, b).ratio() >= 0.82
        except Exception:                                         # noqa: BLE001
            return False

    def _install_evolution_reach_observers(self) -> None:
        """Production-hardening reach-signals (#62/#125/#63/#64): register
        the two pure-observation seams that give the self-improvement loop
        system-wide failure reach through bounded queues:

        * :func:`kenning.resilience.error_log.set_error_observer` -- every
          recorded typed error (web search, Qdrant memory, desktop, bridge,
          TTS, ...) enqueues ``(dependency, message)``;
        * :func:`kenning.safety.validator.set_block_observer` -- every hard
          safety block enqueues ``(tool_name, reason)`` so repeated refusals
          can distil a DEFENSIVE skill.

        Both seams are observation-only (registered AFTER verdict/audit
        logic, wrapped fail-open at their call sites -- they can never
        alter a verdict or drop an error record). The bounded deque is
        drained once per run-loop iteration by
        :meth:`_drain_evolution_reach_signals`. No-op when evolution is
        disabled. Fail-open."""
        if getattr(self, "evolution", None) is None:
            return
        from collections import deque

        self._evolution_reach_queue: Any = deque(maxlen=32)
        queue = self._evolution_reach_queue

        def _on_error(dependency: str, message: str) -> None:
            queue.append((f"dependency:{dependency}", message))

        def _on_block(tool_name: str, reason: str) -> None:
            queue.append((f"safety_block:{tool_name}", reason))

        try:
            from kenning.resilience.error_log import set_error_observer

            set_error_observer(_on_error)
        except Exception as e:                                    # noqa: BLE001
            logger.debug("error observer install failed: %s", e)
        try:
            from kenning.safety.validator import set_block_observer

            set_block_observer(_on_block)
        except Exception as e:                                    # noqa: BLE001
            logger.debug("block observer install failed: %s", e)

    def _drain_evolution_reach_signals(self) -> None:
        """Drain the bounded reach-signal queue into the EvolutionService
        (#62/#125/#63/#64). Each entry becomes a command-failure record
        (``exit_code=1`` + failure-shaped text so the detector fires);
        the recurrence gate keeps transient one-offs from ever distilling
        while RECURRING failures become repair material. Bounded per
        drain; fail-open; zero-cost no-op when nothing is queued."""
        evolution = getattr(self, "evolution", None)
        if evolution is None:
            return
        queue = getattr(self, "_evolution_reach_queue", None)
        if not queue:
            return
        try:
            for _ in range(32):
                try:
                    source, detail = queue.popleft()
                except IndexError:
                    break
                evolution.record_command_failure(
                    source,
                    f"{source} failed: {detail}"[:500],
                    exit_code=1,
                )
        except Exception as e:                                    # noqa: BLE001
            logger.debug("evolution reach-signal drain failed: %s", e)

    def _drain_evolution_task_successes(self) -> None:
        """Production-hardening (#66): drain successfully-completed coding
        tasks the runner queued and feed each to the EvolutionService as a
        ``coding_task_success`` opportunity capsule -- the loop learns from
        what WORKS, not only from failures. Fail-open + a zero-cost no-op
        when evolution or coding is disabled."""
        evolution = getattr(self, "evolution", None)
        coding_voice = getattr(self, "coding_voice", None)
        if evolution is None or coding_voice is None:
            return
        try:
            runner = getattr(coding_voice, "runner", None)
            drain = getattr(runner, "drain_task_successes", None)
            if drain is None:
                return
            for label, summary in drain():
                evolution.record_turn(
                    user_text=label or "coding task",
                    signals=("coding_task_success",),
                    response_summary=(
                        summary or f"coding task completed: {label}"
                    )[:200],
                )
        except Exception as e:                                    # noqa: BLE001
            logger.debug("evolution task-success drain failed: %s", e)

    def _consume_last_barge_in(self) -> bool:
        """Return whether the PRIOR response was interrupted by a barge-in,
        clearing the flag. Feeds the evolution temperament tuner (a
        barge-in nudges responses terser). Fail-open to False."""
        flag = bool(getattr(self, "_last_turn_barged_in", False))
        self._last_turn_barged_in = False
        return flag

    def _note_evolution_turn_metrics(self, errored: bool) -> None:
        """Guardrail brake (#15+#65): feed THIS turn's response-side
        observation into the evolution per-turn metrics ring.

        TTFT is popped from the LLM engine (read-and-clear, so a stale
        warmup / speculative value is consumed at most once) and recorded
        ONLY for plain conversational turns: a search-augmented turn
        carries a much larger prompt class than the locked baseline was
        measured on, so its TTFT would mis-trip the latency guardrail --
        the turn still counts toward the error rate. Fail-open + a
        zero-cost no-op when evolution (or its monitoring) is disabled.
        """
        evolution = getattr(self, "evolution", None)
        if evolution is None:
            return
        try:
            ring = getattr(evolution, "turn_metrics", None)
            if ring is None:
                return
            ttft: Optional[float] = None
            pop = getattr(getattr(self, "llm", None), "pop_last_ttft_ms", None)
            if callable(pop):
                ttft = pop()
            if getattr(self, "_last_search_payload", None) is not None:
                ttft = None
            ring.note_response(ttft_ms=ttft, errored=errored)
        except Exception as e:                                    # noqa: BLE001
            logger.debug("evolution turn-metrics note failed: %s", e)

    def _load_event_store_if_enabled(self) -> None:
        """2026-05-23 OpenHands batch 3 (T2 + T13) -- construct the event store.

        Reads ``events.*`` config, builds the chosen backend via
        :func:`build_event_store`, publishes the result via
        :func:`set_event_store`, and -- when ``install_bus_sink`` is
        True -- subscribes the bus so every published event lands as a
        persisted row with a hash chain.

        Fail-open: any error logs WARN and leaves the singleton unset.
        """
        if self._skip_for_lean_gaming("barebones_skip_events"):
            logger.info("lean gaming boot: events store skipped")
            return
        try:
            from kenning.config import PROJECT_ROOT, get_config

            cfg = get_config().events
        except Exception as e:                                    # noqa: BLE001
            logger.warning("events: config read failed (%s)", e)
            return
        if not getattr(cfg, "enabled", False):
            return
        try:
            from pathlib import Path
            from kenning.events import (
                build_event_store,
                install_bus_event_sink,
                set_event_store,
            )

            base_dir = Path(cfg.base_dir)
            if not base_dir.is_absolute():
                base_dir = PROJECT_ROOT / base_dir
            store = build_event_store(
                cfg.store_backend,
                base_dir=base_dir,
                qdrant_collection=cfg.qdrant_collection,
            )
            set_event_store(store)
            if cfg.install_bus_sink:
                install_bus_event_sink(store, default_session_id=cfg.default_session_id)
            logger.info(
                "events: store ready (backend=%s base_dir=%s sink=%s)",
                cfg.store_backend, base_dir, cfg.install_bus_sink,
            )
        except Exception as e:                                    # noqa: BLE001
            logger.warning(
                "event store init failed (%s) -- persistence disabled "
                "for this session", e,
            )

    def _maybe_dispatch_intent(self, user_text: str) -> bool:
        """Run user_text through the intent recognizer.

        Returns True if a registered intent matched + was dispatched;
        the caller should skip the rest of the routing path. Returns
        False on no match / recognizer unavailable / dispatch failure.
        """
        recognizer = getattr(self, "_intent_recognizer", None)
        if recognizer is None:
            return False
        try:
            match = recognizer.process_utterance(user_text)
        except Exception as e:                                    # noqa: BLE001
            logger.warning("intent: process_utterance failed (%s)", e)
            return False
        if match is None:
            return False
        try:
            return self._dispatch_intent_match(match)
        except Exception as e:                                    # noqa: BLE001
            logger.warning(
                "intent: dispatcher for %r raised (%s); falling through",
                match.canonical_phrase, e,
            )
            return False

    # 2026-05-22 -- map registered phrasings to canonical actions. The
    # intent recognizer keeps each phrase as its own canonical_phrase
    # (no synonym graph), so the dispatcher needs to know which raw
    # phrase corresponds to which command. Add new variants to a set
    # below to extend coverage without writing new dispatcher branches.
    _INTENT_ENGAGE_PHRASES = frozenset({
        "engage gaming mode",
        "switch to gaming mode",
        "turn on gaming mode",
        "start gaming mode",
        "activate gaming mode",
    })
    _INTENT_DISENGAGE_PHRASES = frozenset({
        "disengage gaming mode",
        "turn off gaming mode",
        "stop gaming mode",
        "exit gaming mode",
        "deactivate gaming mode",
    })
    _INTENT_STATUS_PHRASES = frozenset({
        "gaming mode status",
    })
    # 2026-05-22 -- semantic "needs fresh data" intents. When the intent
    # recognizer matches one of these (cosine >= intent.threshold via
    # Gemma-300M), the dispatcher does NOT short-circuit the LLM; instead
    # it sets ``_next_turn_force_search`` so _build_response_stream
    # bypasses the preflight LLM gate and routes directly to SEARCH.
    # This is the semantic backstop to the regex-layer rules in
    # web_search.gating: regex catches explicit markers ("latest",
    # "current", "any news"), the intent layer catches phrasings the
    # regex misses ("give me the latest scoop on X", "what's the buzz",
    # "fill me in on what's happening").
    _INTENT_FORCE_SEARCH_PHRASES = frozenset({
        "what is the latest news",
        "what are the latest news",
        "tell me the latest news",
        "any recent news",
        "any news today",
        "what is happening today",
        "what is going on",
        "tell me what is new",
        "current events",
        "give me the latest update",
        "what is the latest in ai",
        "what is the buzz",
    })

    def _resolve_gaming_mode_manager(self):
        """Return the GamingModeManager from either of the two wiring
        points (top-level attr or via coding_voice). Returns None when
        gaming mode is disabled."""
        manager = getattr(self, "gaming_mode_manager", None)
        if manager is None and getattr(self, "coding_voice", None) is not None:
            manager = getattr(self.coding_voice, "gaming_mode_manager", None)
        return manager

    def _dispatch_intent_match(self, match) -> bool:
        """Route an intent match to the right local handler.

        Returns True if the dispatch fully handled the turn (no LLM
        needed); False if we should fall through to the existing
        routing path.

        Force-search phrases are a special case: they always return
        False (the LLM still produces the response), but they set
        ``self._next_turn_force_search = True`` so the response stream
        bypasses the preflight gate and routes directly to SEARCH.
        """
        import asyncio
        phrase = match.canonical_phrase
        if phrase in self._INTENT_FORCE_SEARCH_PHRASES:
            self._next_turn_force_search = True
            logger.info(
                "intent: 'needs fresh data' matched (%r, sim=%.2f); "
                "forcing SEARCH for this turn",
                phrase, match.similarity,
            )
            return False  # let the LLM run; the gate gets pre-populated
        if phrase in self._INTENT_ENGAGE_PHRASES:
            manager = self._resolve_gaming_mode_manager()
            if manager is None:
                logger.debug("intent: gaming engage fired but no manager")
                return False
            # 2026-06-11: free EVERY non-gaming resource. The settings
            # panel (a tkinter python process) is closed if open;
            # Docker Desktop is stopped by the manager when
            # gaming_mode.toggle_docker is set (restored on disengage);
            # the engage state machine then swaps the LLM, kills the
            # Parakeet server, moves Kokoro to CPU, and unloads the VLM.
            try:
                from kenning.settings_gui.launch import close_gui

                if close_gui(getattr(self, "_settings_gui_pid", None)):
                    self._settings_gui_pid = None
            except Exception as e:                                # noqa: BLE001
                logger.debug("engage: settings panel close skipped (%s)", e)
            try:
                asyncio.run(manager.engage())
            except Exception as e:                                # noqa: BLE001
                logger.warning("intent: engage failed (%s)", e)
                return False
            try:
                self.tts.speak("Shutting down desktop control. Have fun.")
            except Exception:
                pass
            return True
        if phrase in self._INTENT_DISENGAGE_PHRASES:
            manager = self._resolve_gaming_mode_manager()
            if manager is None:
                return False
            try:
                asyncio.run(manager.disengage())
            except Exception as e:                                # noqa: BLE001
                logger.warning("intent: disengage failed (%s)", e)
                return False
            try:
                self.tts.speak("Full control restored.")
            except Exception:
                pass
            return True
        if phrase in self._INTENT_STATUS_PHRASES:
            manager = self._resolve_gaming_mode_manager()
            if manager is None:
                return False
            try:
                status = manager.status()
                self.tts.speak(f"Gaming mode is {status.value}.")
            except Exception:
                pass
            return True
        # Unrecognised phrase: fall through to routing so the LLM gets
        # a chance. Operator-registered phrases with no orchestrator-
        # side dispatcher land here.
        return False

    def _load_openclaw_bridge_if_enabled(self):
        """Phase 3.5: build the OpenClaw bridge holder (or return None
        when ``openclaw.enabled=False``). Fail-open — any startup
        failure logs WARN and leaves the bridge in a degraded but
        usable state. The voice pipeline does NOT depend on the
        bridge; bridge calls fire only on OpenClaw-bound intents.

        Phase 4: also threads :class:`NotificationsConfig` through
        so the bridge's :class:`NotificationDispatcher` knows whether
        Telegram pings are enabled.
        """
        if self._skip_for_lean_gaming("barebones_skip_openclaw"):
            logger.info("lean gaming boot: OpenClaw bridge skipped (gateway probe "
                        "+ MCP-registration retry thread + voice-handoff receiver "
                        "not started; kenning.openclaw_bridge not imported)")
            return None
        from kenning.config import get_config
        from kenning.openclaw_bridge import OpenClawBridge

        full_cfg = get_config()
        cfg = full_cfg.openclaw
        if not cfg.enabled:
            return None
        try:
            bridge = OpenClawBridge.from_config(
                cfg,
                notifications_cfg=full_cfg.notifications,
                heartbeat_cfg=full_cfg.heartbeat,
            )
        except Exception as e:                                 # noqa: BLE001
            logger.warning(
                "OpenClaw bridge construction failed (%s) -- disabled. "
                "Voice path is unaffected.", e,
            )
            return None
        if bridge is None:
            return None
        try:
            bridge.start()
        except Exception as e:                                 # noqa: BLE001
            logger.warning(
                "OpenClaw bridge start raised (%s) -- bridge stays in "
                "degraded mode; voice path unaffected.", e,
            )
        return bridge

    def _load_coding_voice_if_enabled(self):
        """Construct the coding voice controller if enabled.

        Builds the bridge (direct subprocess today; OpenClaw later via
        settings flip), wires up project registry + resolver, and returns
        a :class:`CodingVoiceController` for the main loop to call.
        Failures degrade silently -- coding is optional.
        """
        if self._skip_for_lean_gaming("barebones_skip_coding"):
            logger.info("lean gaming boot: CodingVoiceController + ProjectIndex + "
                        "Supervisor (repo_map/architect) skipped")
            return None
        if not settings.CODING_ENABLED:
            return None
        # lazy: keeps the coding stack out of a lean gaming boot
        from kenning.coding import (
            CodingTaskRunner, CodingVoiceController, ProjectRegistry,
            ProjectResolver)
        from kenning.coding.narration import StatusNarrator
        try:
            # Phase 5: wire the status narrator + shared session store
            # into the runner so progress queries get delta-aware,
            # in-voice narration. Both are optional -- the runner falls
            # back to bridge-state narration when neither is wired.
            narrator = StatusNarrator(llm=self.llm)
            store = (
                self.mcp_server.store if self.mcp_server is not None else None
            )
            runner = CodingTaskRunner(narrator=narrator, store=store)
            registry = ProjectRegistry()
            embedder = None
            if self.memory is not None:
                try:
                    embedder = self.memory._embedder  # noqa: SLF001
                except Exception:
                    embedder = None
            resolver = ProjectResolver(registry, embedder=embedder)
            # 2026-05-22 supervisor stack -- construct lazily when the
            # master flag is on. Wire ProjectIndex + ProjectSupervisor +
            # SupervisorDispatchController so the controller can route
            # CODE_TASK utterances through digest + semantic resolution
            # before falling back to the legacy ProjectResolver path.
            project_index = None
            supervisor = None
            supervisor_dispatch = None
            try:
                from kenning.config import get_config as _get_cfg
                sup_cfg = _get_cfg().coding.supervisor
            except Exception:                                       # noqa: BLE001
                sup_cfg = None
            if sup_cfg is not None and sup_cfg.enabled:
                project_index, supervisor, supervisor_dispatch = (
                    self._build_supervisor_stack(
                        cfg=sup_cfg,
                        registry=registry,
                        resolver=resolver,
                        embedder=embedder,
                        runner=runner,
                    )
                )

            controller = CodingVoiceController(
                runner=runner,
                registry=registry,
                resolver=resolver,
                sandbox_root=settings.CODING_SANDBOX_PATH,
                coordinator=self.coding_coordinator,
                # 4B plan: voice-driven model swap. The voice controller
                # calls ``self.llm.reload_for_preset(...)`` on a
                # MODEL_SWITCH intent. Hot-reload happens in the same
                # process; no orchestrator restart required.
                llm_engine=self.llm,
                # Phase 4 — pass through to OpenClawDispatcher so
                # MESSAGING / BROWSER / etc. intents call real Gateway
                # operations when the bridge is wired. None when the
                # bridge is disabled or its construction failed.
                openclaw_bridge=self.openclaw_bridge,
                # V1-gap A1 — gaming-mode manager (None when disabled
                # or no bridge client). Routes GAMING_MODE intents to
                # the OpenClawDispatcher's plugin enable/disable path. Reuses the
                # HOISTED self.gaming_mode_manager (built in __init__) so the
                # startup engage works even when coding_voice is skipped.
                gaming_mode_manager=self.gaming_mode_manager,
                # 2026-05-22 supervisor stack -- None when disabled or
                # when construction failed (controller falls through
                # to legacy ProjectResolver path).
                supervisor_dispatch=supervisor_dispatch,
                project_index=project_index,
                # B3: hand the live MCP server so voice-dispatched coding
                # tasks write a per-project .mcp.json + connect back for the
                # clarification / verification / completion loop.
                mcp_server=self.mcp_server,
            )
            logger.info(
                "Coding voice ready (bridge=%s, sandbox=%s, coordinator=%s, supervisor=%s)",
                runner.bridge.name(), settings.CODING_SANDBOX_PATH,
                "on" if self.coding_coordinator is not None else "off",
                "on" if supervisor_dispatch is not None else "off",
            )
            return controller
        except Exception as e:
            logger.warning("Coding voice init failed (%s) -- disabled.", e)
            return None

    def _build_supervisor_stack(
        self,
        cfg,
        registry,
        resolver,
        embedder,
        runner,
    ):
        """Construct ProjectIndex + ProjectSupervisor + SupervisorDispatchController.

        Returns ``(project_index, supervisor, dispatch_controller)``
        with any individual element ``None`` when its sub-flag is off
        or construction failed. Fail-open: any error here logs WARN
        and returns three Nones so the rest of the coding pipeline
        keeps the legacy behavior.
        """
        from pathlib import Path
        from kenning.config import get_config

        project_index = None
        if cfg.index_enabled and embedder is not None:
            try:
                from kenning.coding.project_index import ProjectIndex
                # 2026-06-12: local-mode Qdrant allows ONE client per
                # path, and ConversationMemory already holds the
                # data/qdrant lock -- a second QdrantClient(path=...)
                # raised "already accessed by another instance" on
                # EVERY boot, silently degrading the supervisor to
                # registry-only. Borrow the memory's open client (the
                # same pattern WebResultsCache uses); ProjectIndex
                # falls back to its own client when none is passed
                # (scripts / tests / memory-disabled installs).
                shared_client = None
                if self.memory is not None:
                    try:
                        shared_client = self.memory._client  # noqa: SLF001
                    except Exception:                       # noqa: BLE001
                        shared_client = None
                project_index = ProjectIndex(
                    embedder=embedder, client=shared_client,
                )
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "Supervisor: ProjectIndex construction failed (%s); "
                    "supervisor will run registry-only.", e,
                )
                project_index = None

        supervisor = None
        if cfg.decide_enabled:
            try:
                from kenning.coding.project_supervisor import ProjectSupervisor

                # 2026-05-22 catalog batch 2: construct the repo-map
                # provider when its flag is on. Fail-open: if the
                # provider can't be built (missing deps, etc.), the
                # supervisor falls back to no-map decisions.
                repo_map_provider = None
                repo_map_cfg = get_config().coding.repo_map
                if repo_map_cfg.enabled:
                    try:
                        from kenning.coding.repo_map import RepoMapProviderCache
                        from kenning.utils.mtime_cache import MtimeCache
                        cache_dir = Path(repo_map_cfg.cache_dir)
                        if not cache_dir.is_absolute():
                            cache_dir = (
                                Path(settings.CODING_SANDBOX_PATH).parent.parent
                                / cache_dir
                            )
                        mtime_cache = MtimeCache(cache_dir)
                        repo_map_provider = RepoMapProviderCache(
                            max_map_tokens=repo_map_cfg.max_map_tokens,
                            max_map_tokens_no_chat=(
                                repo_map_cfg.max_map_tokens_no_chat
                            ),
                            mtime_cache=mtime_cache,
                        )
                        logger.info(
                            "Supervisor: repo_map provider enabled "
                            "(max_map_tokens=%d, cache_dir=%s)",
                            repo_map_cfg.max_map_tokens,
                            cache_dir,
                        )
                    except Exception as e2:                          # noqa: BLE001
                        logger.warning(
                            "Supervisor: repo_map provider construction "
                            "failed (%s); supervisor will run without "
                            "repo maps.", e2,
                        )
                        repo_map_provider = None

                # 2026-05-22 catalog batch 7: construct the architect-
                # plan provider when its flag is on. Fail-open: if the
                # in-process LLM is unavailable or construction errors,
                # the supervisor falls back to no-plan decisions.
                architect_provider = None
                architect_cfg = get_config().coding.architect
                if architect_cfg.enabled and self.llm is not None:
                    try:
                        from kenning.coding.architect_supervisor import (
                            ArchitectSupervisor,
                            DEFAULT_ARCHITECT_SYSTEM_PROMPT,
                        )

                        def _architect_call(rendered_prompt: str) -> str:
                            # Use the isolated LLM path so SOUL.md
                            # persona + memory don't contaminate the
                            # architect plan. Temperature 0.3 because
                            # plan generation is mildly creative.
                            return self.llm.generate_isolated(
                                system_prompt=DEFAULT_ARCHITECT_SYSTEM_PROMPT,
                                user_prompt=rendered_prompt,
                                temperature=0.3,
                            )

                        architect_provider = ArchitectSupervisor(
                            [_architect_call],
                        )
                        logger.info(
                            "Supervisor: architect provider enabled "
                            "(max_prompt_chars=%d)",
                            architect_cfg.max_prompt_chars,
                        )
                    except Exception as e2:                          # noqa: BLE001
                        logger.warning(
                            "Supervisor: architect provider construction "
                            "failed (%s); supervisor will run without "
                            "an architect plan.", e2,
                        )
                        architect_provider = None
                elif architect_cfg.enabled and self.llm is None:
                    logger.warning(
                        "coding.architect.enabled=True but llm is None; "
                        "architect provider disabled.",
                    )

                supervisor = ProjectSupervisor(
                    index=project_index,
                    registry=registry,
                    resolver=resolver,
                    resolve_threshold=cfg.resolve_threshold,
                    clarify_threshold=cfg.clarify_threshold,
                    decisions_log_path=(
                        Path(cfg.decisions_log_path)
                        if cfg.decisions_log_path
                        else None
                    ),
                    max_candidates_in_decision=cfg.max_candidates_in_decision,
                    repo_map_provider=repo_map_provider,
                    architect_provider=architect_provider,
                )
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "Supervisor: ProjectSupervisor construction failed (%s); "
                    "supervisor disabled.", e,
                )
                supervisor = None

        dispatch_controller = None
        if supervisor is not None:
            try:
                from kenning.coding.supervisor_dispatch import SupervisorDispatchController

                # 2026-05-22 catalog batch 14 (T5 Phase 2): build the optional
                # architect-plan narrator callable. Only when the architect
                # itself is enabled AND its narrate flag is on do we open
                # the per-sentence barge-in window during dispatch.
                architect_narrator_callable = None
                arch_cfg = get_config().coding.architect
                if (
                    architect_provider is not None
                    and arch_cfg.narrate_enabled
                    and getattr(self, "tts", None) is not None
                ):
                    architect_narrator_callable = (
                        self._build_architect_narrator(arch_cfg)
                    )

                dispatch_controller = SupervisorDispatchController(
                    supervisor=supervisor,
                    index=project_index,
                    barge_in_speak=self._speak_with_barge_in_check,
                    plain_speak=self._speak,
                    narrate_enabled=cfg.narrate_enabled,
                    narration_barge_in_window_seconds=(
                        cfg.narration_barge_in_window_seconds
                    ),
                    enriched_context_enabled=cfg.enriched_context_enabled,
                    sandbox_root=settings.CODING_SANDBOX_PATH,
                    default_model=get_config().coding.default_model,
                    architect_narrator=architect_narrator_callable,
                )
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "Supervisor: SupervisorDispatchController construction "
                    "failed (%s); supervisor will only emit decisions, not "
                    "dispatch.", e,
                )
                dispatch_controller = None

        return project_index, supervisor, dispatch_controller

    def _build_architect_narrator(self, arch_cfg):
        """Build the (plan_text) -> bool architect-plan narrator callable.

        Catalog batch 14 (T5 Phase 2). Wraps an ``ArchitectNarrator`` around
        ``self.tts`` so the dispatcher can call a thin function during
        dispatch. The returned callable invokes the narrator with a
        ``should_stop`` predicate that watches the wake-word detector --
        a wake-word fire mid-narration counts as a barge-in.
        """
        from kenning.coding.architect_narrator import ArchitectNarrator

        narrator = ArchitectNarrator(
            self.tts,
            max_chars=int(arch_cfg.narrate_max_chars),
            inter_sentence_pause_seconds=(
                float(arch_cfg.narrate_inter_sentence_pause_ms) / 1000.0
            ),
        )

        # Capture the wake-word baseline at the moment the narrator is
        # called so a stale prior trigger can't be misread as a barge-in.
        def _narrate(plan_text: str) -> bool:
            try:
                wake_before = getattr(self.wake, "_last_trigger_ts", 0.0)
            except Exception:
                wake_before = 0.0

            def _should_stop() -> bool:
                try:
                    cur = getattr(self.wake, "_last_trigger_ts", 0.0)
                except Exception:
                    return False
                return bool(cur and cur > wake_before)

            result = narrator.narrate(plan_text, should_stop=_should_stop)
            if result.interrupted:
                logger.info(
                    "architect narration interrupted by barge-in after "
                    "%d sentence(s) (%d chars).",
                    result.sentences_spoken, result.chars_spoken,
                )
            elif result.error:
                logger.debug(
                    "architect narration ended with %r (%d/%d chars spoken).",
                    result.error, result.chars_spoken,
                    len(plan_text or ""),
                )
            return bool(result.interrupted)

        return _narrate

    def _load_gaming_mode_manager_if_enabled(self):
        """V1-gap A1: construct the GamingModeManager when configured.

        2026-05-22: previously returned None when no OpenClaw client
        was wired; the manager now constructs without it -- the
        engage/disengage callbacks (LLM swap, Kokoro device flip, STT
        swap, VLM unload) all work without OpenClaw, and the manager
        already handles ``client=None`` gracefully by returning
        ``no openclaw client`` per-plugin states (the rest of the
        engage cycle still completes).
        Returns ``None`` when ``gaming_mode.enabled=false`` or when
        construction itself fails. Failures degrade silently --
        gaming mode is purely additive.
        """
        from kenning.config import get_config, resolve_path

        cfg = get_config().gaming_mode
        if not cfg.enabled:
            return None
        bridge = getattr(self, "openclaw_bridge", None)
        client = getattr(bridge, "client", None) if bridge is not None else None
        if client is None:
            logger.info(
                "gaming_mode: constructing manager without an OpenClaw "
                "client (plugin disable will no-op; LLM/Kokoro/STT/VLM "
                "engage callbacks still fire).",
            )
        try:
            from kenning.openclaw_routing.gaming_mode import GamingModeManager

            # 2026-05-22 Gaming mode VRAM/RAM reclaim:
            # - LLM hot-swaps to ``gaming_mode.llm_preset`` (default
            #   llama-3.2-3b-abliterated; ~2.0 GB) on engage; restores
            #   the prior preset on disengage. Saves ~1.5 GB VRAM.
            # - STT swaps from the primary engine (typically Parakeet
            #   on GPU, ~700 MB VRAM) to the gaming engine (typically
            #   Moonshine on CPU, 0 VRAM). Engage also kills the
            #   Parakeet HTTP server to actually free its VRAM; disengage
            #   restarts the server on a background thread and swaps
            #   back when /healthz reports ready.
            # - Kokoro engine flips to CPU (saves ~330 MB VRAM when
            #   configured on CUDA; disengage restores to configured
            #   device).
            # - moondream2 VLM is unloaded if loaded -- frees ~2 GB
            #   RAM on the CPU default, or ~2 GB VRAM if operator had
            #   set vlm device to CUDA. The VLM re-loads lazily on
            #   the next SCREEN_CONTEXT_QUERY after disengage.
            # All hooks are best-effort: missing attribute / failed
            # call logs WARN and leaves the original state.
            full_cfg = get_config()
            tts_kokoro_default_device = full_cfg.tts.kokoro.device
            # Keep Kokoro on the GPU while gaming (snappy callouts + frees the
            # CPU for capture/STT). Default "cuda"; config can force "cpu".
            tts_kokoro_engage_device = (
                getattr(cfg, "kokoro_engage_device", "cuda") or "cuda")
            gaming_llm_preset = (cfg.llm_preset or "").strip()
            # Force the gaming LLM onto CPU (bare-bones) regardless of the
            # env/config gpu_layers override. None = leave on config device.
            gaming_llm_gpu_layers = getattr(cfg, "llm_gpu_layers", 0)
            # Cell to share the pre-engage preset between callbacks.
            llm_preset_before_engage: dict = {"value": None}

            # Stash the pre-engage STT engine so disengage knows what to restore to.
            stt_name_before_engage: dict = {"value": None}

            # Catalog 09 batch H wiring: replace the prior pair of
            # synchronous monolithic callbacks with an async-generator
            # state machine driven by :func:`drive_start_task`. Each
            # substep (LLM swap, Parakeet stop, Kokoro move, VLM unload)
            # is now an observable transition that produces a per-stage
            # voice-ack opportunity via the ``on_transition`` callback
            # AND a stable audit row in the start-task history. Stage
            # ordering and fail-open behavior are preserved bit-for-bit
            # so VRAM/RAM reclaim accounting is unchanged.
            def _build_engage_deps():
                from kenning.lifecycle.gaming_engage import GamingEngageDeps

                stt_registry = getattr(self, "_stt_registry", None)
                try:
                    from kenning.transcription.parakeet_engine import (
                        start_parakeet_server, stop_parakeet_server,
                    )
                except Exception:
                    start_parakeet_server = None
                    stop_parakeet_server = None
                # ANTICHEAT HARDENING: engage may UNLOAD the VLM, but only if one
                # was ever loaded. Never IMPORT kenning.desktop.vlm here (it pulls
                # pyautogui + mss into RAM) -- if the module isn't already
                # resident there is no VLM to unload, so pass None.
                try:
                    import sys as _sys
                    _vlm_mod = _sys.modules.get("kenning.desktop.vlm")
                    _get_vlm = _vlm_mod.get_vlm if _vlm_mod is not None else None
                except Exception:                                    # noqa: BLE001
                    _get_vlm = None
                return GamingEngageDeps(
                    llm=getattr(self, "llm", None),
                    tts=getattr(self, "tts", None),
                    stt_registry=stt_registry,
                    swap_stt_engine=getattr(self, "swap_stt_engine", None),
                    get_vlm=_get_vlm,
                    start_parakeet_server=start_parakeet_server,
                    stop_parakeet_server=stop_parakeet_server,
                    gaming_llm_preset=gaming_llm_preset,
                    gaming_llm_gpu_layers=gaming_llm_gpu_layers,
                    tts_kokoro_default_device=tts_kokoro_default_device,
                    tts_kokoro_engage_device=tts_kokoro_engage_device,
                    llm_preset_holder=llm_preset_before_engage,
                    stt_name_holder=stt_name_before_engage,
                )

            def _gaming_voice_ack(task) -> None:
                """on_transition callback -- speak a short voice ack
                for each substep so the user hears the progress. Fail-
                open: any TTS failure leaves the state machine running."""
                if not task.detail:
                    return
                # STARTUP engage is automatic + silent: the user did not ask
                # for it, so do not announce "swapping language model / stopping
                # Parakeet / moving voice engine / unloading vision model" on
                # every boot. Only the VOICE-COMMANDED "gaming mode" speaks.
                if getattr(self, "_gaming_engage_silent", False):
                    return
                tts = getattr(self, "tts", None)
                if tts is None or not hasattr(tts, "speak"):
                    return
                # Skip the terminal READY ack to avoid stepping on the
                # higher-level GamingModeManager voice line.
                from kenning.lifecycle.start_task import StartTaskStatus
                if task.status == StartTaskStatus.READY:
                    return
                if task.status == StartTaskStatus.WORKING:
                    return
                try:
                    tts.speak(task.detail)
                except Exception as e:                            # noqa: BLE001
                    logger.debug(
                        "gaming voice ack speak failed (%s); continuing",
                        e,
                    )

            def _engage_extra():
                from kenning.lifecycle.gaming_engage import (
                    gaming_engage_iterator,
                )
                from kenning.lifecycle.start_task import (
                    StartTaskError, drive_start_task,
                )
                import asyncio

                deps = _build_engage_deps()
                # openclaw-clawhub T7: mint a short-lived forensic token
                # authorising the gaming-preset takeover. Disengage lets
                # it expire (revocation-by-expiry). Audit-logged;
                # fail-open; not a hard gate.
                self._mint_forensic_token(
                    caller_id="voice:gaming-engage",
                    audience="kenning-llm",
                    scope=("llm.preset.swap",),
                    ttl_seconds=6 * 60 * 60,
                    extra_claims={"action": "gaming_engage"},
                )
                try:
                    _drive_async_blocking(
                        drive_start_task(
                            gaming_engage_iterator(deps),
                            on_transition=_gaming_voice_ack,
                        ),
                    )
                except StartTaskError as e:
                    logger.warning(
                        "gaming engage state machine raised (%s); "
                        "partial engage left in place", e,
                    )
                except Exception as e:                            # noqa: BLE001
                    logger.warning(
                        "gaming engage driver failed (%s); falling back "
                        "to no-op", e,
                    )
                # Free the cross-encoder reranker (~1 GB): RAG is gated off in
                # gaming mode so it is dead weight. It lazily reloads after
                # disengage. Fail-open.
                try:
                    from kenning.memory.reranker import reset_shared_reranker
                    reset_shared_reranker()
                    logger.info("gaming engage: cross-encoder reranker freed")
                except Exception as e:                            # noqa: BLE001
                    logger.debug("gaming engage: reranker free skipped (%s)", e)

            def _disengage_extra():
                from kenning.lifecycle.gaming_engage import (
                    gaming_disengage_iterator,
                )
                from kenning.lifecycle.start_task import (
                    StartTaskError, drive_start_task,
                )
                import asyncio

                deps = _build_engage_deps()
                try:
                    _drive_async_blocking(
                        drive_start_task(
                            gaming_disengage_iterator(deps),
                            on_transition=_gaming_voice_ack,
                        ),
                    )
                except StartTaskError as e:
                    logger.warning(
                        "gaming disengage state machine raised (%s); "
                        "partial disengage left in place", e,
                    )
                except Exception as e:                            # noqa: BLE001
                    logger.warning(
                        "gaming disengage driver failed (%s); falling back "
                        "to no-op", e,
                    )

            manager = GamingModeManager(
                client=client,
                plugins_to_disable=list(cfg.plugins_to_disable),
                toggle_docker=cfg.toggle_docker,
                docker_executable_path=cfg.docker_executable_path,
                docker_process_name=cfg.docker_process_name,
                log_path=resolve_path(cfg.log_path) if cfg.log_path else None,
                on_engaged=_engage_extra,
                on_disengaged=_disengage_extra,
            )
            logger.info(
                "GamingModeManager ready (plugins=%s, toggle_docker=%s, "
                "kokoro_engage_device=%s, kokoro_disengage_device=%s, "
                "vlm_unload_on_engage=True, llm_preset=%s)",
                cfg.plugins_to_disable, cfg.toggle_docker,
                tts_kokoro_engage_device,
                tts_kokoro_default_device,
                gaming_llm_preset or "(no swap)",
            )
            return manager
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "GamingModeManager init failed (%s) -- gaming mode disabled.", e,
            )
            return None

    def _load_web_search_if_enabled(self):
        """Construct the web-search gate + executor if enabled.

        Returns ``(gate, executor, ack_source)`` triple. Any of them can be
        ``None`` if web search is disabled -- the rest of the pipeline
        still works.

        2026-05-21: switched from a Brave-only client to the multi-
        provider :class:`SearchProviderChain` (SearxNG local -> Brave
        API -> DuckDuckGo public). The chain handles missing API
        keys and missing services gracefully (the relevant provider
        is silently skipped). So the only blocking failure is "no
        providers can be constructed at all".
        """
        from kenning.config import get_config
        from kenning.web_search.provider_chain import SearchProviderChain
        ws_cfg = get_config().web_search
        if not ws_cfg.enabled:
            return None, None, None
        # Brave key is only blocking if Brave is the ONLY configured
        # provider. Otherwise the chain skips Brave and uses SearxNG
        # / DDG.
        configured_providers = list(getattr(ws_cfg, "providers", ["brave"]))
        brave_in_chain = "brave" in [p.lower() for p in configured_providers]
        brave_key_set = bool(os.getenv(ws_cfg.brave_api_key_env, ""))
        if configured_providers == ["brave"] and not brave_key_set:
            logger.warning(
                "web_search.enabled=true with only 'brave' provider but %s "
                "missing in env -- web search disabled. Add 'searxng' or "
                "'duckduckgo' to web_search.providers to enable local / "
                "no-key search.", ws_cfg.brave_api_key_env,
            )
            return None, None, None
        if brave_in_chain and not brave_key_set:
            logger.info(
                "Brave API key (%s) not set -- the chain will skip Brave "
                "and fall through to the other configured providers.",
                ws_cfg.brave_api_key_env,
            )
        try:
            from kenning.web_search.reader_chain import ReaderChain
            searcher = SearchProviderChain()
            # 2026-05-21 frontier: reader chain replaces the direct
            # JinaReaderClient. Local trafilatura first (~50-150 ms,
            # zero external dep), Jina fallback for JS-heavy sites
            # (~1-3 s round-trip). The chain exposes the same
            # ``fetch(url) -> Optional[str]`` interface as the bare
            # client, so the WebSearchExecutor is unchanged.
            jina = ReaderChain()
            gate = WebSearchGate(llm=self.llm)
            cache = None
            if self.memory is not None:
                try:
                    cache = WebResultsCache(
                        client=self.memory._client,  # noqa: SLF001
                        embedder=self.memory._embedder,  # noqa: SLF001
                    )
                except Exception as e:
                    logger.warning(
                        "Web result cache disabled (%s) -- search will work "
                        "but won't reuse prior queries.", e
                    )
            # WebSearchExecutor's ``brave`` param is duck-typed -- it
            # only calls ``.search(query, count=...)``, which the
            # chain implements identically. Keeping the param name
            # for backwards-compatibility with existing tests.
            executor = WebSearchExecutor(
                brave=searcher, jina=jina, llm=self.llm, cache=cache,
            )
            ack = AcknowledgmentSource()
            return gate, executor, ack
        except Exception as e:
            logger.warning("Web search init failed (%s) -- disabled.", e)
            return None, None, None

    def _build_smart_turn_detector(self) -> Optional[SmartTurnDetector]:
        """Construct the Smart Turn V3 detector when enabled + model is on disk.

        Fail-open: missing model file / disabled flag / construction
        error -> returns None and the orchestrator falls back to its
        legacy VAD-only end-of-turn detection. The voice baseline is
        unaffected when the detector is unavailable.

        Returns:
            A constructed (but not yet loaded -- lazy at first use)
            :class:`SmartTurnDetector` on success, or ``None`` to
            disable smart-turn for this session.
        """
        if self._smart_turn_cfg is None:
            return None
        try:
            from kenning.config import PROJECT_ROOT
            return build_detector_from_config(
                self._smart_turn_cfg, PROJECT_ROOT,
            )
        except Exception as e:
            logger.warning(
                "Smart Turn V3 detector construction failed; "
                "falling back to legacy VAD-only end-of-turn: %s",
                e,
            )
            return None

    def _smart_turn_should_check(
        self, *, speech_seen: bool, speech_samples: int,
    ) -> bool:
        """Decide whether to invoke Smart Turn V3 on the just-captured audio.

        Three gating conditions:

        1. The detector must be available (None means disabled / model
           missing -- skip).
        2. The user must have actually started speaking (no point
           running on a buffer of leading silence).
        3. The contiguous speech duration must be within the model's
           training window. Longer utterances are handled by the
           existing adaptive long-utterance backstop at the VAD layer.

        Args:
            speech_seen: True iff VAD has emitted SPEECH_START during
                this capture.
            speech_samples: Number of samples from speech-start to
                "now" (end-of-speech). Compared against
                ``smart_turn.window_seconds`` to decide eligibility.

        Returns:
            True iff Smart Turn should run. The decision intentionally
            does NOT consider whether smart-turn has already been used
            this capture -- the caller tracks that and skips this
            check on the second SPEECH_END.
        """
        if self.smart_turn is None:
            return False
        if not speech_seen:
            return False
        window_samples = int(
            self._smart_turn_window_seconds * settings.SAMPLE_RATE
        )
        return speech_samples <= window_samples

    def _classify_smart_turn_verdict(
        self, verdict: Optional[SmartTurnVerdict],
    ) -> str:
        """Bucket a Smart Turn V3 verdict into a gradient-fire band.

        2026-05-16 latency pass 2: introduces a three-band gradient
        instead of the legacy binary (complete / incomplete). The
        bands map to the orchestrator's fast-path silence schedule:

        - ``"early_complete"``: prob >= early_completion_threshold
          (0.65 default). Submit immediately -- the model is highly
          confident the turn is complete and we've already paid the
          (shortened) fast_path_silence_duration_ms. Saves the legacy
          200 ms wait between fast_path (300 ms) and the prior
          baseline (500 ms).
        - ``"medium_complete"``: prob in [completion_threshold,
          early_completion_threshold). The model leans complete but
          isn't confident enough to fire immediately at the shortened
          baseline. The caller waits ``medium_grace_ms`` of additional
          silence then re-classifies; on second pass medium_complete
          gets promoted to early_complete (because the audio tail
          just grew by 200 ms of silence, making the verdict more
          reliable).
        - ``"incomplete"``: prob < completion_threshold. User trailed
          off mid-thought; enter the existing
          ``incomplete_extension_ms`` extension window.
        - ``"undecided"``: ``verdict is None`` (inference failure or
          detector unavailable). Caller falls back to legacy
          VAD-only behaviour (existing semantics preserved).

        This method is a pure function -- no side effects, no I/O.
        """
        if verdict is None:
            return "undecided"
        prob = float(verdict.probability)
        if prob >= self._smart_turn_early_completion_threshold:
            return "early_complete"
        if prob >= self._smart_turn_completion_threshold:
            return "medium_complete"
        return "incomplete"

    def _run_smart_turn(
        self, captured: np.ndarray,
    ) -> Optional[SmartTurnVerdict]:
        """Run a single Smart Turn V3 inference call on the captured audio.

        Wraps :meth:`SmartTurnDetector.is_complete` with project-level
        sample-rate plumbing. Returns ``None`` on any failure or when
        the detector is not available. Caller treats ``None`` as
        "undecided" and trusts VAD's end-of-speech verdict.

        Args:
            captured: Full captured audio buffer up to the current
                end-of-speech moment. Float32, 16 kHz mono assumed.

        Returns:
            A :class:`SmartTurnVerdict` on success, ``None`` on
            failure (logged) or when the detector is unavailable.
        """
        if self.smart_turn is None:
            return None
        try:
            return self.smart_turn.is_complete(
                captured, sample_rate=SMART_TURN_SAMPLE_RATE,
            )
        except Exception as e:
            logger.warning("Smart Turn V3 inference dispatch failed: %s", e)
            return None

    def _load_addressing_classifier(self) -> AddressingClassifier:
        """Build the CPU-side addressing classifier used in WARM mode."""
        # Provide a lightweight closure so the zero-shot pass can fold in
        # recent dialogue without coupling the classifier to ConversationMemory.
        def recent_turns_provider(n: int):
            if self.memory is None:
                return []
            try:
                return [(t.role, t.content) for t in self.memory.recent(n)]
            except Exception:
                return []

        from kenning.config import get_config, resolve_path
        addr_cfg = get_config().addressing
        return AddressingClassifier(
            rule_confidence_threshold=addr_cfg.rule_confidence_threshold,
            default_silent_on_uncertain=addr_cfg.default_uncertain_to_not_addressed,
            log_path=resolve_path(addr_cfg.log_path),
            zero_shot_model_name=addr_cfg.zero_shot_model,
            # Lean gaming boot: KEEP the addressee classifier (it is on the
            # warm-follow-up hot path) but DEFER the ~300MB flan-t5 model load
            # until an ambiguous follow-up actually needs it (rare while gaming;
            # relay commands bypass the classifier entirely).
            load_zero_shot_eagerly=(
                addr_cfg.load_eagerly
                and not self._skip_for_lean_gaming("barebones_lazy_zero_shot_addressee")),
            recent_turns_provider=recent_turns_provider,
            zero_shot_addressed_min_confidence=addr_cfg.zero_shot_addressed_min_confidence,
        )

    @staticmethod
    def _load_memory_if_enabled():
        """Build a Qdrant-backed :class:`ConversationMemory` if enabled.

        Failures degrade gracefully: missing deps -> memory disabled. The
        hybrid embedder loads eagerly at construction so the first hot-path
        write doesn't pay the model-load cost.
        """
        if not settings.MEMORY_ENABLED:
            return None
        try:
            from kenning.memory import ConversationMemory, HybridEmbedder
        except Exception as e:
            logger.warning("Memory module import failed (%s) -- disabling memory", e)
            return None

        try:
            embedder = HybridEmbedder(eager=True)
        except Exception as e:
            logger.warning(
                "HybridEmbedder load failed (%s) -- disabling memory", e
            )
            return None

        try:
            return ConversationMemory(embedder=embedder)
        except Exception as e:
            logger.warning("ConversationMemory init failed (%s) -- disabling memory", e)
            return None

    def _load_background_summarizer_if_enabled(self):
        """2026-05-19 Tracks 1c-1e voice-loop integration.

        Build a :class:`BackgroundSummarizer` wired against the
        currently-loaded LLM + memory. Returns ``None`` when the
        feature flag is off, when the LLM or memory are unavailable,
        or when construction raises (fail-open).

        The summarizer's hot path is opt-in -- the orchestrator only
        calls :meth:`maybe_summarize` from
        :meth:`_maybe_run_background_summarizer`, which itself short-
        circuits when this returns None. So flipping the flag off
        recovers byte-for-byte legacy orchestrator behaviour.
        """
        if self._skip_for_lean_gaming("barebones_skip_summarizer"):
            logger.info("lean gaming boot: background summarizer skipped")
            return None
        try:
            from kenning.config import get_config
            cfg = get_config().memory.background_summary
        except Exception as e:                                # noqa: BLE001
            logger.debug("background_summary config read failed (%s)", e)
            return None
        if not cfg.enabled:
            return None
        if self.memory is None:
            logger.warning(
                "background_summary.enabled=True but memory is None; "
                "summarizer disabled.",
            )
            return None
        if self.llm is None:
            logger.warning(
                "background_summary.enabled=True but llm is None; "
                "summarizer disabled.",
            )
            return None
        try:
            from kenning.memory.background_summarizer import (
                BackgroundSummarizer,
                TurnSnapshot,
                _SUMMARY_SYSTEM_PROMPT,
            )
        except Exception as e:                                # noqa: BLE001
            logger.warning(
                "BackgroundSummarizer import failed (%s); summarizer disabled.", e,
            )
            return None

        # generate_fn: hand the rendered prompt to the LLM via the
        # isolated path so SOUL.md persona + memory injection don't
        # contaminate the summarization. The summarizer's render_summary_prompt
        # produces the user-side prompt; the system prompt comes from
        # the summarizer module's constant.
        def _generate_fn(rendered_prompt: str) -> str:
            return self.llm.generate_isolated(
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                user_prompt=rendered_prompt,
                temperature=0.2,
            )

        # recent_turns_fn: project ConversationMemory.recent into the
        # summarizer's TurnSnapshot shape. ``recent_window`` controls
        # how many turns the summarizer can see per pass -- generous
        # by default (the gate already throttles call cadence).
        def _recent_turns_fn():
            try:
                turns = self.memory.recent(64)
            except Exception as e:                            # noqa: BLE001
                logger.warning(
                    "memory.recent failed for summarizer (%s); "
                    "skipping pass.", e,
                )
                return []
            return [
                TurnSnapshot(
                    turn_id=t.id, ts=t.ts, role=t.role, content=t.content,
                )
                for t in turns
            ]

        # store_fn: append the SummaryResult as one JSON line to the
        # configured output path. Fail-open at every step. A separate
        # follow-up integration pass can replace this with a Qdrant
        # write that creates ``type=session_summary|fact|decision|preference``
        # entries -- the storage shape is intentionally flat so the
        # JSONL is straightforward to ingest later.
        store_fn = self._build_default_background_summary_store(cfg.output_path)

        try:
            return BackgroundSummarizer(
                generate_fn=_generate_fn,
                store_fn=store_fn,
                recent_turns_fn=_recent_turns_fn,
                cadence_turns=int(cfg.cadence_turns),
                min_turns=int(cfg.min_turns),
                idle_threshold_seconds=float(cfg.idle_threshold_seconds),
            )
        except Exception as e:                                # noqa: BLE001
            logger.warning(
                "BackgroundSummarizer construction failed (%s); disabled.", e,
            )
            return None

    @staticmethod
    def _build_default_background_summary_store(output_path: str):
        """Return a ``store_fn`` that appends to ``output_path`` as JSONL.

        Returns ``None`` when ``output_path`` is empty (storage
        intentionally disabled). The returned function never raises:
        IO errors are logged WARN and swallowed so a flaky disk never
        crashes a background pass.
        """
        if not output_path:
            return None
        import dataclasses as _dc
        import json as _json
        from pathlib import Path as _Path
        target = _Path(output_path)

        def _store(result) -> None:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "ts": time.time(),
                    "summary": result.summary,
                    "facts": [_dc.asdict(f) for f in result.facts],
                    "decisions": [_dc.asdict(d) for d in result.decisions],
                    "preferences": [_dc.asdict(p) for p in result.preferences],
                    "turn_id_start": result.turn_id_start,
                    "turn_id_end": result.turn_id_end,
                    "span_seconds": result.span_seconds,
                }
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(_json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as e:                            # noqa: BLE001
                logger.warning(
                    "background summary store_fn failed (%s); discarding pass.", e,
                )

        return _store

    def _load_tts_engine(self):
        """Construct the configured TTS engine.

        Thin wrapper around :func:`kenning.tts.make_tts_engine` so the
        orchestrator and measurement scripts share one construction
        path. Returns ``(rvc_or_none, tts_engine)``; the ``rvc``
        attribute is kept on Orchestrator for diagnostic purposes
        even though only the legacy engine uses it.

        Raises any engine-construction error -- TTS is not optional;
        the orchestrator can't run without a voice path.
        """
        from kenning.tts import make_tts_engine
        return make_tts_engine()

    def _kick_off_ack_clip_prewarm(self) -> Optional[threading.Thread]:
        """Build + populate the pre-computed ack clip cache.

        Runs on a daemon thread so orchestrator construction stays
        fast (~5-7 s of total synth across the conversational +
        web-search ack pools). The first user turn may still hit the
        live path (cache not warm yet); subsequent turns hit the cache.

        Fail-open at every level: missing engine support, server
        unreachable mid-prewarm, or partial population all leave the
        engine in its pre-existing state.
        """
        if not hasattr(self.tts, "set_ack_cache"):
            logger.debug(
                "TTS engine %s has no set_ack_cache hook; skipping ack prewarm",
                type(self.tts).__name__,
            )
            return None
        try:
            from kenning.tts.precomputed_ack import (
                build_default_ack_clip_cache,
                prewarm_in_background,
            )
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "Could not import precomputed_ack (%s); engine will use "
                "live synth for every ack phrase.", e,
            )
            return None
        cache = build_default_ack_clip_cache()
        if not cache.phrases:
            logger.info("Ack clip prewarm: no phrases to cache; skipping")
            return None
        try:
            self.tts.set_ack_cache(cache)
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "set_ack_cache on %s raised (%s); cache will be unused",
                type(self.tts).__name__, e,
            )
            return None
        try:
            return prewarm_in_background(
                cache,
                self.tts._synthesize,  # noqa: SLF001 -- internal API by design
                name="ack-prewarm",
            )
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "Failed to kick off ack-prewarm thread (%s); cache stays empty.", e,
            )
            return None

    # --- context manager -----------------------------------------------------

    def __enter__(self) -> "Orchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # --- lifecycle -----------------------------------------------------------

    def shutdown(self) -> None:
        """Signal the run loop to exit and tear down components."""
        if self._shutdown.is_set():
            return
        logger.info("Shutdown requested")
        self._shutdown.set()
        self._interrupt.set()
        # 2026-05-22 -- flush the session's fail-open counts to disk
        # so the next session's startup can read + display them. Done
        # before tearing components down so a slow tts/audio shutdown
        # doesn't lose the counts on a crash-on-shutdown. Fail-open.
        try:
            from kenning.resilience import fail_open_log
            fail_open_log.flush_to_disk()
        except Exception as e:                                      # noqa: BLE001
            logger.debug("fail_open_log flush failed: %s", e)
        # 2026-05-19 Tracks 1c-1e: cancel any in-flight background
        # summarizer so the worker exits cleanly. The thread is daemon
        # so it would be reaped anyway, but the cancel lets the in-flight
        # LLM call notice + skip parse/store cleanly.
        self._cancel_background_summarizer()
        for action in (self.tts.stop, self.audio.stop):
            try:
                action()
            except Exception:
                pass
        if self.rvc is not None:
            try:
                self.rvc.close()
            except Exception:
                pass
        # 2026-05-10 voice swap: when xtts_v3 engine is active, the
        # TTS instance owns a server subprocess. Make sure it gets
        # torn down with the orchestrator.
        if hasattr(self.tts, "_stop_server_subprocess"):
            try:
                self.tts._stop_server_subprocess()
            except Exception:
                pass
        if self.memory is not None:
            try:
                self.memory.close()
            except Exception:
                pass
        if self.mcp_server is not None:
            try:
                self.mcp_server.stop(timeout_s=3.0)
            except Exception:
                pass
        if self.openclaw_bridge is not None:
            # Phase 3.5: stop the retry thread + event receiver. We
            # deliberately do NOT unregister the MCP entry — leaving it
            # lets OpenClaw spawn Kenning's MCP across restarts.
            try:
                self.openclaw_bridge.shutdown()
            except Exception:
                pass
        # Catalog 09 batch A: stop the DialogPoller daemon thread so
        # the process can exit cleanly. Best-effort -- the thread is
        # daemon so it would be reaped on hard exit anyway.
        poller = getattr(self, "_dialog_poller", None)
        if poller is not None:
            try:
                poller.stop()
            except Exception:
                pass
        # Reap the embedder sidecar process TREE BEFORE stopping the reaper (so
        # unregister() can't race a concurrent sweep) and BEFORE this process
        # exits (so the loopback 8772 socket is released cleanly, no orphan).
        try:
            self._kill_embedder_sidecar()
        except Exception:                                            # noqa: BLE001
            pass
        # Stop the subprocess reaper thread so the process exits cleanly.
        killer = getattr(self, "_zombie_killer", None)
        if killer is not None:
            try:
                killer.shutdown()
            except Exception:
                pass
        # T22: stop any MCP servers we started (reap their process trees).
        mcp_registry = getattr(self, "_mcp_registry", None)
        if mcp_registry is not None:
            try:
                mcp_registry.stop_all()
            except Exception:
                pass
        # Catalog 13: persist the evolution state + learned personality so
        # the next session resumes the temperament + distillation cooldown.
        # Best-effort -- the JSONL capsule log is already durable per-turn.
        evolution = getattr(self, "evolution", None)
        if evolution is not None:
            try:
                evolution.shutdown()
            except Exception:
                pass
            # Reach-signal hygiene: clear the module-level observers so a
            # torn-down orchestrator can't keep receiving events (tests
            # construct + discard orchestrators; the observers must not
            # leak across instances).
            try:
                from kenning.resilience.error_log import set_error_observer
                set_error_observer(None)
            except Exception:                                     # noqa: BLE001
                pass
            try:
                from kenning.safety.validator import set_block_observer
                set_block_observer(None)
            except Exception:                                     # noqa: BLE001
                pass
            # Anticheat surface hooks close over self -- drop them so a
            # torn-down orchestrator can't be poked by a later mode flip.
            try:
                from kenning.safety.anticheat import clear_surface_hooks
                clear_surface_hooks()
            except Exception:                                     # noqa: BLE001
                pass

    # --- main loop -----------------------------------------------------------

    def run(self) -> None:
        """Block forever, processing wake events until shutdown."""
        from kenning.config import get_config
        from kenning import trace
        _addr_cfg = get_config().addressing
        self.audio.start()
        word = self.wake.active_word
        print(f"\n  Kenning is listening. Say '{word}' to wake.\n")
        if self.wake.using_fallback:
            print(
                f"  (Wake word currently fallback='{word}'. "
                f"Train a custom model for true 'kenning' detection — see README.)\n"
            )
        if self.memory is not None:
            mem_count = len(self.memory)
            print(f"  Memory: {mem_count} prior turns loaded.\n")
            trace.tlog(
                logger, "memory:bootstrap",
                cross_session_turns_in_cache=mem_count,
                session_id=getattr(self.memory, "session_id", None),
            )

        # When the follow-up window is open this holds the deadline (monotonic
        # time). ``None`` means we're in plain wake-word-gated IDLE mode.
        follow_up_until: Optional[float] = None

        try:
            while not self._shutdown.is_set():
                # 2026-05-20 round 6: bump turn id at top of every voice
                # loop iteration so all downstream logs grep by turn=N.
                turn_id = trace.next_turn()
                trace.set_phase("loop")
                trace.tlog(
                    logger, "loop:iteration_start",
                    state=self._state.value,
                    pending_capture=self._pending_capture.is_set(),
                    follow_up_until=follow_up_until,
                )
                # Settings-panel hot reload: the GUI touches a signal
                # file after writing config.yaml; pick it up here so
                # every call-time get_config() read sees the new values
                # (one os.stat per iteration; fail-open).
                self._maybe_reload_config()
                # Coding-task completion push: if a background AI coding agent
                # task just finished, announce it before we go back to
                # listening. This gives the unsolicited "Done. Created X
                # in Y..." narration the spec calls for.
                self._announce_coding_completion_if_pending()
                # Phase 2: surface any clarifications Claude is parked on.
                self._announce_pending_clarifications()
                # Phase 7: surface token-budget warnings + halt notices.
                self._announce_pending_budget_warning()
                # B3: speak the result of a backgrounded sandbox program run.
                self._announce_pending_run_report()
                # 4B plan Item 7: surface canonical-path-monitor aborts.
                self._announce_pending_canonical_abort()
                # E2 goal-anchor planning: surface anchor lifecycle
                # narration (opening / warning / transition / completion).
                self._announce_pending_anchor_narration()
                # 2026 catalog 14 (T1): feed any command/tool failures the
                # coding runner observed into the EvolutionService (repair
                # distillation). Zero-cost no-op when nothing failed.
                self._drain_evolution_command_failures()
                # Guardrail brake (#15+#65): speak any queued evolution
                # narration (e.g. the post-apply auto-revert notice). A
                # zero-cost no-op when nothing is queued.
                self._drain_evolution_narrations()
                # Reach-signals (#62/#125/#63/#64): feed recorded dependency
                # errors + hard safety blocks to the EvolutionService.
                # Zero-cost no-op when nothing is queued.
                self._drain_evolution_reach_signals()
                # #66: feed successfully-completed coding tasks to the
                # EvolutionService (learning from what works). Zero-cost
                # no-op when nothing completed.
                self._drain_evolution_task_successes()
                # 2026 catalog 08/09: speak any dialog-appearance narration
                # the coding runner's dialog auto-handler queued (default ON).
                # Time-sensitive (a native dialog is blocking the task), so
                # drain it before the informational loop-alert. No-op when none.
                self._drain_coding_dialog_narrations()
                # 2026 catalog wiring (T1): speak any loop-detection heads-up
                # the coding runner queued when a task's tool-call stream
                # tripped a hard escalation. Zero-cost no-op when none.
                self._drain_coding_loop_alerts()
                # 2026-05-19 Tracks 1c-1e: opportunistic background
                # summarisation. Cheap no-op when disabled or when a
                # previous pass is still in flight; the summarizer's
                # idle-threshold gate decides whether this attempt
                # actually performs LLM work.
                self._maybe_run_background_summarizer()

                speech: Optional[np.ndarray] = None
                came_from_follow_up = False

                if self._pending_capture.is_set():
                    # Barge-in or wake-during-follow-up → fresh wake-gated capture.
                    self._pending_capture.clear()
                    self._state = State.CAPTURING
                    print(f"  [{self._state.value}] capturing your request…")
                    trace.tlog(
                        logger, "loop:capture_path",
                        reason="pending_capture", state=self._state.value,
                    )
                    speech = self._capture_utterance()
                    follow_up_until = None
                elif (
                    follow_up_until is not None
                    and _addr_cfg.follow_up_enabled
                    and time.monotonic() < follow_up_until
                ):
                    self._state = State.FOLLOW_UP_LISTENING
                    trace.tlog(
                        logger, "loop:follow_up_listen",
                        state=self._state.value,
                        remaining_s=follow_up_until - time.monotonic(),
                    )
                    outcome = self._follow_up_listen(deadline=follow_up_until)
                    # outcome is either an ndarray (audio captured) or a
                    # sentinel string. Type-check first — comparing an ndarray
                    # to a string with `==` gives an element-wise array, which
                    # raises in a boolean context.
                    if isinstance(outcome, str):
                        if outcome == _FU_TIMEOUT:
                            print("  (follow-up window closed; waiting for wake word)")
                            trace.tlog(
                                logger, "loop:follow_up_timeout",
                                next_state="IDLE",
                            )
                            follow_up_until = None
                            continue
                        if outcome == _FU_WAKE:
                            self._state = State.CAPTURING
                            print(f"  [{self._state.value}] capturing your request…")
                            trace.tlog(
                                logger, "loop:wake_during_follow_up",
                                state=self._state.value,
                            )
                            speech = self._capture_utterance()
                            follow_up_until = None
                    else:
                        # Got a VAD-bounded utterance during follow-up.
                        speech = outcome
                        came_from_follow_up = True
                        trace.tlog(
                            logger, "loop:follow_up_utterance_captured",
                            audio_samples=outcome.size,
                        )
                else:
                    self._state = State.IDLE
                    follow_up_until = None
                    # 2026-06-11 VRAM hygiene: reclaim the torch caching-
                    # allocator's reserved-but-unused blocks now that the
                    # turn is fully done (response spoken). This runs in
                    # the mic-IDLE window right before blocking on the
                    # wake word -- OFF the latency-critical span -- so it
                    # has zero TTFT impact while capping the per-response
                    # reserved high-water-mark ratchet (the CUDA-Kokoro
                    # creep). Gated on meaningful slack so it only fires
                    # when there's real bloat to release.
                    self._reclaim_idle_vram()
                    trace.tlog(
                        logger, "loop:waiting_for_wake_word",
                        state=self._state.value,
                    )
                    if not self._wait_for_wake_word():
                        trace.tlog(logger, "loop:wake_wait_returned_false_shutdown")
                        break
                    self._state = State.CAPTURING
                    print(f"  [{self._state.value}] capturing your request…")
                    trace.tlog(
                        logger, "loop:wake_word_fired",
                        state=self._state.value,
                    )
                    speech = self._capture_utterance()

                if speech is None or speech.size == 0:
                    if not came_from_follow_up:
                        print("  (heard nothing; standing down)")
                    trace.tlog(
                        logger, "loop:empty_capture",
                        came_from_follow_up=came_from_follow_up,
                    )
                    continue

                self._state = State.PROCESSING
                trace.tlog(
                    logger, "loop:capture_complete",
                    audio_samples=speech.size,
                    audio_seconds=float(speech.size) / float(settings.SAMPLE_RATE),
                    came_from_follow_up=came_from_follow_up,
                    state=self._state.value,
                )
                # 2026-05-15 latency: TTS output-stream pre-open. After
                # 2026-05-18 latency pass 3 (Phase 1), the primary
                # kick-off lives at the top of ``_capture_utterance`` /
                # ``_follow_up_listen`` so the open overlaps the entire
                # speech + silence-wait window (not just the post-capture
                # tail). The call here is a belt-and-braces no-op: if the
                # earlier kick-off completed, ``prepare_output_stream``
                # short-circuits; if the engine has no method (legacy
                # fixture), ``_kick_off_tts_preopen`` no-ops. Cheap and
                # forgiving. Fail-open at every level.
                self._kick_off_tts_preopen()
                # 2026-05-16 latency pass 2: collect any speculative
                # STT result kicked off DURING the silence wait inside
                # ``_capture_utterance`` / ``_follow_up_listen``. On
                # hit we skip the foreground Whisper run entirely
                # (~78 ms saved); on miss / invalidation we fall back
                # to the legacy call. Either way ``user_text`` is the
                # final transcript for the captured audio.
                user_text = self._collect_speculative_stt()
                if user_text is None:
                    trace.set_phase("stt")
                    trace.tlog(
                        logger, "stt:foreground_start",
                        audio_samples=speech.size,
                    )
                    stt_t0 = time.monotonic()
                    user_text = self.stt.transcribe(speech)
                    trace.tlog(
                        logger, "stt:foreground_end",
                        chars=len(user_text or ""),
                        elapsed_ms=int((time.monotonic() - stt_t0) * 1000),
                        text=user_text[:160] if user_text else None,
                    )
                else:
                    trace.tlog(
                        logger, "stt:speculative_hit",
                        chars=len(user_text), text=user_text[:160],
                    )
                if not user_text.strip():
                    if not came_from_follow_up:
                        print("  (no transcription; standing down)")
                    trace.tlog(logger, "stt:empty_transcript")
                    continue

                # In the follow-up window, gate every utterance through the
                # CPU-side addressing classifier. Don't reset the deadline on
                # rejected speech -- we measure FOLLOW_UP_TIMEOUT_SECONDS from
                # the *last response*, not from the last sound in the room.
                if came_from_follow_up:
                    trace.set_phase("addressing")
                    # Relay commands are DEFINITIONALLY addressed to
                    # Kenning -- the strict matcher ("tell my team X",
                    # "ask Clove to Y") cannot fire on room chatter, so
                    # in the follow-up window they need no wake word and
                    # must not be lost to a borderline zero-shot verdict
                    # (observed live: a direct command dropped at
                    # conf 0.75 vs the 0.80 threshold). Also skips the
                    # ~190 ms classifier on every relay turn.
                    if self._is_relay_command(user_text):
                        print(f"  (follow-up, relay) you: {user_text}")
                        trace.tlog(
                            logger, "addressing:relay_override",
                            text=user_text[:160],
                        )
                    else:
                        seconds_since = (
                            time.monotonic() - self._last_response_finished_monotonic
                        )
                        verdict = self.addressing.classify(
                            user_text, seconds_since_response=seconds_since
                        )
                        trace.tlog(
                            logger, "addressing:verdict",
                            decision=verdict.decision.value,
                            source=verdict.source,
                            conf=float(verdict.confidence or 0.0),
                            reason=verdict.reason,
                            seconds_since_response=seconds_since,
                            text=user_text[:160],
                        )
                        if verdict.decision != AddressingDecision.ADDRESSED:
                            print(
                                f"  (heard: {user_text!r} -- not for me "
                                f"[{verdict.source}: {verdict.reason}])"
                            )
                            trace.tlog(
                                logger, "addressing:rejected_follow_up",
                                decision=verdict.decision.value,
                            )
                            continue
                        print(f"  (follow-up) you: {user_text}")
                else:
                    print(f"  you: {user_text}")
                    trace.tlog(
                        logger, "addressing:wake_word_path_no_classify",
                        text=user_text[:160],
                    )

                # Record the addressed user utterance into the dual-history
                # store BEFORE any short-circuit so "what did I say earlier?"
                # recall sees every turn. The recall handler excludes the
                # current query turn so it never matches itself.
                self._record_dialogue_turn("user", user_text)

                # 2026-06-14 -- PRE-ROUTING NORMALIZATION. Clean the raw STT
                # transcript into a canonical command string BEFORE any matcher
                # sees it: strip wake/filler remnants, correct Valorant vocab +
                # agent names (Jett/Cypher/Sova/Raze/B main/...), and restore a
                # dropped relay "tell my team ..." lead for clipped callouts
                # ("my team there's a Jett A main" -> "tell my team there's a
                # Jett on A main"). Contextually GATED -- questions, Spotify,
                # identity, and desktop commands are never rewritten as relays,
                # so only true team callouts gain the lead. The ORIGINAL
                # transcript was already recorded to dialogue history above, so
                # "what did I say earlier?" recall still sees the real words.
                try:
                    from kenning.audio.command_normalizer import normalize_command
                    _raw_stt = user_text
                    _normed = normalize_command(user_text)
                    # ALWAYS log BOTH the raw STT transcript AND the normalized
                    # routing text (even when unchanged) so every turn shows the
                    # pre/post pair -- this is the tuning record for refining the
                    # normalizer + Valorant vocab/blend maps in later iterations.
                    trace.tlog(
                        logger, "routing:normalized",
                        raw=_raw_stt[:200],
                        normalized=(_normed or _raw_stt)[:200],
                        changed=bool(_normed and _normed != _raw_stt),
                    )
                    if _normed and _normed != user_text:
                        user_text = _normed
                except Exception as e:                               # noqa: BLE001
                    logger.debug("command normalization skipped: %s", e)

                # 2026-05-22 -- intent recognizer short-circuit. Runs
                # AFTER addressing but BEFORE routing/LLM. Matched
                # registered phrases (gaming mode commands, etc.) get
                # dispatched directly + skip the LLM path. No-op when
                # intent.enabled=False (default).
                trace.set_phase("intent")
                if self._maybe_dispatch_intent(user_text):
                    trace.tlog(
                        logger, "intent:dispatched",
                        text=user_text[:160],
                    )
                    self._last_response_finished_monotonic = time.monotonic()
                    if _addr_cfg.follow_up_enabled:
                        follow_up_until = (
                            self._last_response_finished_monotonic
                            + _addr_cfg.warm_mode_duration_seconds
                        )
                    else:
                        follow_up_until = None
                    trace.tlog(
                        logger, "loop:iteration_end",
                        via="intent", follow_up=bool(follow_up_until),
                    )
                    continue

                # Catalog 13 (evolution): intercept an explicit "evolve
                # now" / "evolution status" command BEFORE routing. A
                # strict matcher gates this so ordinary speech never trips
                # it; on no-match / disabled it returns False and the turn
                # proceeds to routing as usual.
                trace.set_phase("evolution")
                if self._maybe_handle_evolution_command(user_text):
                    self._last_response_finished_monotonic = time.monotonic()
                    if _addr_cfg.follow_up_enabled:
                        follow_up_until = (
                            self._last_response_finished_monotonic
                            + _addr_cfg.warm_mode_duration_seconds
                        )
                    else:
                        follow_up_until = None
                    trace.tlog(
                        logger, "loop:iteration_end",
                        via="evolution_command",
                        follow_up=bool(follow_up_until),
                    )
                    continue

                # Catalog 13 (evolution): record this addressed turn for
                # the self-improvement subsystem. The opportunity signals
                # in the utterance feed skill distillation; the prior
                # response's barge-in (if any) tunes the response
                # temperament; then maybe trigger an autonomous cycle
                # (rare, single-flight, on a daemon thread off the hot
                # path). Fail-open + microsecond cost when nothing fires.
                self._record_evolution_turn(user_text)

                # Capability routing (Phase 5): classify the utterance into
                # one of the routing kinds and let CapabilityVoiceController
                # dispatch. Coding kinds (CODE_TASK / CANCEL / progress /
                # adjustment / clarification) route through the existing
                # CodingTaskRunner. OpenClaw-bound kinds (browser / media /
                # messaging / file / shell / hybrid) get stub voice responses
                # in this Foundation phase. CONVERSATIONAL falls through to
                # the normal LLM path below.
                if self.coding_voice is not None:
                    trace.set_phase("routing")
                    from kenning.openclaw_routing import classify_routing
                    from kenning.openclaw_routing.intents import RoutingIntentKind
                    has_active = self.coding_voice.runner.has_active_task()
                    has_pending = self.coding_voice.has_pending_clarification()
                    routing_intent = classify_routing(
                        user_text,
                        has_active_coding_task=has_active,
                        has_pending_clarification=has_pending,
                    )
                    trace.tlog(
                        logger, "routing:classified",
                        kind=routing_intent.kind.value,
                        conf=float(routing_intent.confidence or 0.0),
                        source=routing_intent.source,
                        reason=routing_intent.reason,
                        has_active_task=has_active,
                        has_pending_clarification=has_pending,
                    )
                    # openclaw-clawhub T12: intercept "log a concern /
                    # flag that response" BEFORE the capability path. The
                    # handler needs _last_response_text (the PRIOR turn's
                    # response), still intact here, to anchor the filed
                    # Report. A strict regex matcher gates this so normal
                    # queries never trip it; on no-match it returns False
                    # and the turn proceeds to routing as usual.
                    if self._maybe_handle_report_concern(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="report_concern",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # B3: "run the calculator" / "launch the server" -- run or
                    # launch a finished sandbox program. Strict matcher +
                    # project resolution -> ordinary utterances fall through.
                    if self._maybe_handle_run_program(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="run_program",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Spotify playback control ("play despacito", "skip",
                    # "pause the music", "what's playing", "turn it up").
                    # AFTER run/launch so "play the calculator" (a
                    # sandbox program) wins; strict matcher -> ordinary
                    # chatter falls through.
                    if self._maybe_handle_spotify(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="spotify",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Production-hardening #4: "scrap it" -- cancel the
                    # running coding task AND revert its recorded edits.
                    # Strict matcher -> ordinary utterances (and bare
                    # "cancel") fall through.
                    if self._maybe_handle_scrap_command(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="scrap",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Anticheat-safe mode toggle: "enable anticheat
                    # mode" hard-blocks every desktop-interaction
                    # surface (input/capture/windows/clipboard/...)
                    # while keeping voice + the team relay alive.
                    if self._maybe_handle_anticheat_toggle(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="anticheat_toggle",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Relay mute toggle: "mute the team chat" / "you can
                    # talk to my team again" -- session-scoped streaming
                    # safety switch. Checked BEFORE the relay handler.
                    if self._maybe_handle_relay_toggle(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="relay_toggle",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Voice relay: "tell my teammates X" -- rephrase to a
                    # direct second-person line and speak it on the
                    # configured secondary output device (VoiceMeeter
                    # strip -> mic bus) so the game voice chat hears
                    # Kenning. Strict matcher -> "tell me ..." and normal
                    # utterances fall through.
                    if self._maybe_handle_relay_speech(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            # Relay turns hold the window open LONGER
                            # (relay_speech.follow_up_seconds) so a whole
                            # in-game conversation flows without
                            # re-waking; relay matches inside the window
                            # bypass the addressing gate entirely.
                            _relay_ext = float(getattr(
                                self, "_relay_follow_up_seconds", 0.0,
                            ) or 0.0)
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + max(
                                    _addr_cfg.warm_mode_duration_seconds,
                                    _relay_ext,
                                )
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="relay_speech",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Settings panel: "pull up your settings" spawns the
                    # detached control-panel GUI; "close the settings"
                    # kills it. Strict matcher -> ordinary utterances
                    # fall through.
                    if self._maybe_handle_settings_gui(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="settings_gui",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Catalog 12 (felo-search T3): intercept an explicit
                    # "research X in depth" / "deep dive on X" request and
                    # run a bounded DeepResearchLoop, then synthesize +
                    # speak the answer. Strict matcher -> normal search /
                    # conversational queries never trip it; on no-match /
                    # disabled / search-unavailable it returns False and the
                    # turn proceeds to routing as usual.
                    if self._maybe_handle_deep_research(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="deep_research",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Deep-memory recall: intercept an explicit "recall
                    # everything we discussed about X" / "dig deep into your
                    # memory about X" request and run a bounded DeepMemoryLoop
                    # (iterative RAG), then synthesize + speak. Strict matcher
                    # -> normal recall questions stay on the fast RAG path; on
                    # no-match / memory-unavailable it returns False and the
                    # turn proceeds to routing as usual.
                    if self._maybe_handle_deep_recall(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="deep_recall",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Deep code exploration: "search the codebase for X" /
                    # "where is Y defined" -> bounded DeepExplorationLoop
                    # (iterative ripgrep over the project source), then speak
                    # where the matches are. Strict matcher -> coding tasks +
                    # web / memory requests fall through.
                    if self._maybe_handle_code_exploration(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="code_exploration",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Conversation-history recall: "what did I say earlier
                    # about X?" / "what did you tell me about Y?" answered from
                    # the in-memory dual-history store (the verbatim turn is
                    # spoken back). Strict matcher -> normal questions fall
                    # through to routing. Needs no LLM/Qdrant.
                    if self._maybe_handle_history_recall(user_text):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="history_recall",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # Gaming/anticheat refusal gate. MUST run before the
                    # capability dispatches below (OPEN_LAST_SOURCE /
                    # NAVIGATE_TO_SITE open a browser at 5253/5272;
                    # handle_capability_intent drives desktop/browser/etc.).
                    # While a protected game runs, every such capability is
                    # refused in character and never dispatched. Relay/Spotify/
                    # identity/conversation/toggles were already handled above.
                    if self._maybe_refuse_capability_in_gaming(routing_intent):
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="gaming_refusal",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # 2026-05-22 OPEN_LAST_SOURCE: intercept here because
                    # the handler needs orchestrator-local state
                    # (_last_search_payload + _last_response_text) that
                    # the capability controller doesn't have access to.
                    if routing_intent.kind == RoutingIntentKind.OPEN_LAST_SOURCE:
                        self._handle_open_last_source(routing_intent)
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="open_last_source",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    # 2026-05-22 NAVIGATE_TO_SITE: same intercept
                    # pattern -- handler hits SearxNG + scores domains
                    # + opens via webbrowser / AppLauncher.
                    if routing_intent.kind == RoutingIntentKind.NAVIGATE_TO_SITE:
                        self._handle_navigate_to_site(routing_intent)
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="navigate_to_site",
                            follow_up=bool(follow_up_until),
                        )
                        continue
                    capability_response = self.coding_voice.handle_capability_intent(routing_intent)
                    if capability_response is not None:
                        trace.tlog(
                            logger, "routing:capability_response",
                            kind=routing_intent.kind.value,
                            chars=len(capability_response.text or ""),
                            preview=(capability_response.text or "")[:160],
                        )
                        self._handle_capability_response(
                            capability_response, routing_intent,
                        )
                        self._last_response_finished_monotonic = time.monotonic()
                        if _addr_cfg.follow_up_enabled:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + _addr_cfg.warm_mode_duration_seconds
                            )
                        else:
                            follow_up_until = None
                        trace.tlog(
                            logger, "loop:iteration_end",
                            via="capability", follow_up=bool(follow_up_until),
                        )
                        continue
                    trace.tlog(
                        logger, "routing:fallthrough_to_llm",
                        kind=routing_intent.kind.value,
                    )

                # SEMANTIC COMMAND ROUTER (additive fallback layer) -- the LAST
                # deterministic attempt before the conversational LLM. Everything
                # above (exact relay/Spotify/identity matchers + capability
                # dispatch) has already missed; route the NORMALIZED text by
                # similarity to curated exemplars. A confident TEAM CALLOUT is
                # forced through the relay path; a confident IDENTITY through the
                # greeting; conversational / ambiguous / low-confidence ABSTAINS
                # to the LLM path below -- UNCHANGED. Fail-safe: any router error
                # is swallowed so the normal LLM path is never disrupted.
                _router_consumed = False
                try:
                    from kenning.audio.command_router import get_command_router
                    _cr = get_command_router()
                    if _cr is not None:
                        _rd = _cr.route(user_text)
                        trace.tlog(
                            logger, "router:decision",
                            family=_rd.family or "abstain",
                            abstained=_rd.abstained,
                            conf=round(_rd.confidence, 3),
                            margin=round(_rd.margin, 3),
                            reason=_rd.reason,
                        )
                        if not _rd.abstained and _rd.family == "team_callout":
                            _router_consumed = self._maybe_handle_relay_speech(
                                user_text, force=True)
                        elif not _rd.abstained and _rd.family == "identity":
                            _router_consumed = self._maybe_handle_relay_speech(
                                "introduce yourself")
                        elif not _rd.abstained and _rd.family == "desktop_refuse":
                            # A desktop / automation request the capability
                            # classifier missed. While a protected game is
                            # running, refuse IN CHARACTER (anticheat); otherwise
                            # let it fall through to the normal path.
                            _ac = False
                            try:
                                from kenning.safety.anticheat import anticheat_active
                                _ac = bool(anticheat_active())
                            except Exception:                       # noqa: BLE001
                                _ac = False
                            if _ac:
                                self._speak(
                                    "Not while you are in the game. I touch "
                                    "nothing on this machine but your voice and "
                                    "your team's ears.")
                                _router_consumed = True
                except Exception as e:                               # noqa: BLE001
                    logger.debug("semantic router skipped: %s", e)
                if _router_consumed:
                    self._last_response_finished_monotonic = time.monotonic()
                    if _addr_cfg.follow_up_enabled:
                        follow_up_until = (
                            self._last_response_finished_monotonic
                            + _addr_cfg.warm_mode_duration_seconds
                        )
                    else:
                        follow_up_until = None
                    trace.tlog(
                        logger, "loop:iteration_end",
                        via="semantic_router",
                        follow_up=bool(follow_up_until),
                    )
                    continue

                trace.set_phase("respond")
                # Catalog 09 batch G wiring: thread the classified
                # intent kind through to _respond so the LLM can pick
                # a per-intent condenser before generating. Default
                # ``None`` when the coding_voice (and thus the routing
                # classifier) isn't wired -- legacy fixed-condenser
                # behavior preserved.
                _intent_kind: Optional[str] = None
                try:
                    _intent_kind = (
                        routing_intent.kind.value
                        if self.coding_voice is not None
                        else None
                    )
                except Exception:
                    _intent_kind = None
                self._respond(user_text, routing_intent_kind=_intent_kind)
                self._last_response_finished_monotonic = time.monotonic()
                if _addr_cfg.follow_up_enabled:
                    follow_up_until = (
                        self._last_response_finished_monotonic
                        + _addr_cfg.warm_mode_duration_seconds
                    )
                    print(
                        f"  (still listening for ~{int(_addr_cfg.warm_mode_duration_seconds)} s -- "
                        f"keep talking or stay silent to drop back to wake-word mode)"
                    )
                else:
                    follow_up_until = None
                trace.tlog(
                    logger, "loop:iteration_end",
                    via="respond", follow_up=bool(follow_up_until),
                )
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()

    # --- phase: wake ---------------------------------------------------------

    def _wait_for_wake_word(self) -> bool:
        """Block until wake word fires. Returns False if shutdown was requested."""
        self.audio.drain()
        self.wake.reset()
        self.ring.clear()
        while not self._shutdown.is_set():
            chunk = self.audio.get_chunk(timeout=0.5)
            if chunk is None:
                # Idle heartbeat (~every 0.5s, LLM/TTS guaranteed idle):
                # apply any settings-panel config reload + runtime
                # actions HERE so GUI changes take effect live (hot)
                # without waiting for the next turn, and safely (no
                # generation in flight). Both are mtime/offset-gated and
                # fail-open.
                self._maybe_reload_config()
                self._drain_gui_actions()
                continue
            self.ring.write(chunk)
            if self.wake.process(chunk):
                return True
        return False

    # --- phase: capture ------------------------------------------------------

    def _capture_utterance(self) -> np.ndarray:
        """Record from now until VAD reports end-of-speech (or timeout).

        Adaptive end-of-turn (2026-05-11): once speech has been active
        for ``vad.long_utterance_threshold_seconds`` the VAD silence
        requirement is bumped to ``vad.long_utterance_silence_duration_ms``
        for the remainder of the capture. This keeps short utterances
        snappy but gives long technical descriptions room to breathe
        through thinking pauses without prematurely closing.

        Smart Turn V3 confirmation (2026-05-12): when
        ``vad.smart_turn.enabled`` is True and the model file is on
        disk, the VAD silence baseline drops to
        ``smart_turn.fast_path_silence_duration_ms`` (typically 500 ms
        vs the legacy 1200 ms). On the first SPEECH_END, the captured
        audio is run through the model. Verdict ``complete`` returns
        immediately; verdict ``incomplete`` extends the capture by
        ``incomplete_extension_ms`` and bumps the VAD silence
        requirement to the legacy backstop value (so the second
        SPEECH_END comes from genuine VAD silence). Long utterances
        beyond ``smart_turn.window_seconds`` of speech bypass the
        model -- the adaptive long-utterance backstop already handles
        those at the VAD layer.
        """
        self.vad.reset()
        # 2026-05-19 Tracks 1c-1e: best-effort cancel of any in-flight
        # background summarizer pass. The user is about to (or has just
        # started to) speak; we want the GPU free for the foreground
        # path. The summarizer's cancel flag is read between sub-calls,
        # so this is a hint -- a mid-LLM-call summarizer still has to
        # complete the in-flight create_chat_completion before
        # foreground LLM can serialise behind it. No-op when the
        # summarizer is disabled.
        self._cancel_background_summarizer()
        # 2026-05-16 latency pass 2: drop any stale speculative STT
        # result from a prior turn (e.g. an empty-utterance turn that
        # never called _collect_speculative_stt). Without this, a stale
        # transcript could leak into this turn's main-loop user_text.
        self._reset_speculative_stt_state()
        # 2026-05-18 latency pass 3 (Phase 1): kick off the PortAudio
        # device open NOW, before any speech is captured. The ~50 ms
        # open cost overlaps with the entire speech-plus-silence-wait
        # window (typically 1-30 s) instead of only the post-capture
        # tail. After Phase 4 of the prior pass collapsed Whisper to
        # zero foreground time, the legacy "kick off after capture"
        # placement no longer had enough overlap to complete before
        # the first TTS write -- speak_stream was falling back to a
        # fresh open. Idempotent at the engine layer (prepare_output_stream
        # no-ops when a stream is already cached); fail-open at every
        # level. Re-armed on every capture so the next turn benefits
        # too (speak_stream consumes-and-clears the cache per turn).
        self._kick_off_tts_preopen()
        # Pre-roll: take the COLD slice (short) from the ring so the
        # wake-word tail does not bleed into the transcript as a "Tron" prefix.
        # PER-WAKE-WORD (2026-06-14): the active word can override the cold
        # pre-roll -- "ultron" ends in a hard, transcribable "-tron" so it runs
        # SHORTER than the audio default. Falls back to _cold_pre_roll_seconds.
        cold_pre_roll_s = self._cold_pre_roll_seconds
        try:
            _aw = getattr(getattr(self, "wake", None), "active_word", None)
            if _aw:
                from kenning.config import get_config as _gc
                _per = getattr(_gc().wake_word, "cold_pre_roll", {}) or {}
                if _aw in _per:
                    cold_pre_roll_s = float(_per[_aw])
        except Exception:                                            # noqa: BLE001
            pass
        cold_pre_roll_samples = int(cold_pre_roll_s * settings.SAMPLE_RATE)
        chunks: list[np.ndarray] = [self.ring.snapshot(cold_pre_roll_samples)]
        # 2026-05-22: streaming STT integration. When the engine supports
        # live partials (Moonshine v2 streaming variants), kick off the
        # session NOW and feed the COLD pre-roll. The capture loop below
        # will feed each subsequent chunk; on capture-end we call
        # ``stop_stream`` to finalize and the engine stashes the result
        # so the post-capture ``transcribe(buffer)`` returns it instantly.
        # Fail-open at every level -- streaming bugs degrade to the
        # legacy one-shot path, never to silence.
        streaming_active = self._maybe_start_stt_stream()
        if streaming_active:
            for c in chunks:
                self._maybe_feed_stt_chunk(c)
        speech_seen = False
        speech_start_samples = 0
        long_utterance_bump_applied = False
        elapsed_samples = 0
        max_samples = int(self._max_utterance_seconds * settings.SAMPLE_RATE)
        # Allow up to MIN_SILENCE * 2 of leading silence before bailing.
        silence_grace = int(2.0 * settings.SAMPLE_RATE)
        leading_silence = 0
        long_threshold_samples = int(
            self._long_utterance_threshold_seconds * settings.SAMPLE_RATE
        )
        # Smart Turn V3 capture state. ``smart_turn_used`` latches
        # after one inference call so we don't re-check on each
        # subsequent SPEECH_END within the same capture. The anchor
        # is the elapsed_samples value when the incomplete verdict
        # was returned; user-resumes-speech cancels the timeout by
        # setting the anchor back to zero.
        #
        # 2026-05-16 latency pass 2: ``smart_turn_medium_anchor``
        # tracks the medium-confidence-complete grace window. When the
        # model returns a verdict in the gradient-fire "medium" band
        # at the (shortened) fast-path checkpoint, we wait
        # ``medium_grace_ms`` more silence before trusting the verdict.
        # On grace elapse: break (submit). User-resumes-speech also
        # cancels by setting the anchor to zero.
        smart_turn_used = False
        smart_turn_incomplete_anchor = 0
        smart_turn_medium_anchor = 0
        extension_samples = int(
            self._smart_turn_incomplete_extension_ms / 1000.0 * settings.SAMPLE_RATE
        )
        medium_grace_samples = int(
            self._smart_turn_medium_grace_ms / 1000.0 * settings.SAMPLE_RATE
        )

        # 2026-05-16 latency pass 2: speculative STT state. We kick off
        # Whisper in the background once we see a brief run of
        # consecutive silence frames after speech -- typically ~32 ms
        # at the 16 ms blocksize. By the time the full fast-path
        # silence baseline elapses (~300 ms), Whisper (~78 ms) has
        # finished and the transcript is ready for the main loop.
        # Re-armable on user-resumes-speech.
        consecutive_silence_chunks = 0
        speculative_kicked = False
        # 2 chunks (~32 ms at 16 ms blocksize, ~64 ms at 32 ms) is
        # conservative enough to avoid reacting to single-block dips
        # (breaths, "umm" pauses) while still kicking off well within
        # the fast-path silence baseline.
        speculative_silence_kickoff_chunks = 2

        while not self._shutdown.is_set() and elapsed_samples < max_samples:
            chunk = self.audio.get_chunk(timeout=0.5)
            if chunk is None:
                continue
            chunks.append(chunk)
            elapsed_samples += chunk.shape[0]
            if streaming_active:
                self._maybe_feed_stt_chunk(chunk)

            result = self.vad.process(chunk)
            if result.event == SpeechEvent.SPEECH_START:
                if not speech_seen:
                    speech_seen = True
                    speech_start_samples = elapsed_samples
                # Smart-turn previously said "incomplete" / "medium" and
                # we were in a grace window. User has resumed speaking
                # -- cancel both timeouts; the next SPEECH_END (at the
                # bumped VAD silence threshold for the incomplete path,
                # or the fast-path for the medium path) is the real end.
                smart_turn_incomplete_anchor = 0
                smart_turn_medium_anchor = 0
                # 2026-05-16 latency pass 2: speculative STT became
                # stale (user resumed before SPEECH_END). Invalidate
                # the in-flight result so the foreground STT runs
                # fresh on the now-larger audio. Allow re-arm on the
                # next silence period.
                if speculative_kicked:
                    self._invalidate_speculative_stt()
                    speculative_kicked = False
                consecutive_silence_chunks = 0
            elif result.event == SpeechEvent.SPEECH_END and speech_seen:
                speech_samples = elapsed_samples - speech_start_samples
                if (
                    not smart_turn_used
                    and self._smart_turn_should_check(
                        speech_seen=speech_seen,
                        speech_samples=speech_samples,
                    )
                ):
                    captured = np.concatenate(chunks).astype(
                        np.float32, copy=False,
                    )
                    verdict = self._run_smart_turn(captured)
                    smart_turn_used = True
                    # 2026-05-16 latency pass 2: gradient-fire bands.
                    band = self._classify_smart_turn_verdict(verdict)
                    if band == "undecided":
                        # Inference failed; trust VAD's verdict at
                        # the fast-path baseline.
                        break
                    if band == "early_complete":
                        logger.info(
                            "Smart Turn V3: early-complete (prob=%.3f, "
                            "%.1f ms) -- submitting at fast-path baseline",
                            verdict.probability, verdict.latency_ms,
                        )
                        break
                    if band == "medium_complete":
                        # Wait an extra ``medium_grace_ms`` of silence
                        # before trusting the verdict. The user may
                        # resume speaking; if not, the medium-grace
                        # timeout below will accept end-of-turn.
                        smart_turn_medium_anchor = elapsed_samples
                        logger.info(
                            "Smart Turn V3: medium-complete (prob=%.3f, "
                            "%.1f ms) -- waiting %d ms more before "
                            "trusting verdict",
                            verdict.probability,
                            verdict.latency_ms,
                            self._smart_turn_medium_grace_ms,
                        )
                    else:
                        # band == "incomplete": user trailed off
                        # mid-thought. Enter the extension window and
                        # bump VAD silence to the legacy backstop so
                        # the next SPEECH_END is at the slow rate.
                        smart_turn_incomplete_anchor = elapsed_samples
                        self.vad.set_min_silence_duration_ms(
                            self._long_utterance_silence_duration_ms
                        )
                        logger.info(
                            "Smart Turn V3: incomplete (prob=%.3f, %.1f ms) "
                            "-- extending capture by up to %d ms; VAD "
                            "silence requirement raised to %d ms",
                            verdict.probability,
                            verdict.latency_ms,
                            self._smart_turn_incomplete_extension_ms,
                            self._long_utterance_silence_duration_ms,
                        )
                else:
                    # Legacy path: trust VAD, or smart-turn already
                    # used / utterance too long for smart-turn window.
                    break

            # 2026-05-16 latency pass 2: speculative STT during the
            # silence-accumulation phase. After SPEECH_START we track
            # consecutive silence chunks; once we cross the kickoff
            # threshold (~32 ms at 16 ms blocksize) we have strong
            # evidence the user has stopped speaking. Kick off Whisper
            # on the audio captured so far on a background thread so
            # by the time the full fast-path baseline (~300 ms) elapses
            # and Smart Turn V3 confirms end-of-turn, the transcript is
            # ready. Fail-open: the kick-off is idempotent and skips
            # itself if already in flight; SPEECH_START during the
            # silence run invalidates the in-flight result above.
            if speech_seen:
                if result.probability < self.vad.threshold:
                    consecutive_silence_chunks += 1
                else:
                    consecutive_silence_chunks = 0
                if (
                    not speculative_kicked
                    and consecutive_silence_chunks >= speculative_silence_kickoff_chunks
                ):
                    speculative_kicked = True
                    audio_so_far = np.concatenate(chunks).astype(
                        np.float32, copy=False,
                    )
                    self._kick_off_speculative_stt(audio_so_far)

            # Once we've been speaking longer than the threshold,
            # extend the silence requirement so a thinking pause
            # doesn't end the capture mid-thought.
            if (
                speech_seen
                and not long_utterance_bump_applied
                and self._long_utterance_threshold_seconds > 0.0
                and (elapsed_samples - speech_start_samples) >= long_threshold_samples
            ):
                self.vad.set_min_silence_duration_ms(
                    self._long_utterance_silence_duration_ms
                )
                long_utterance_bump_applied = True
                logger.info(
                    "Adaptive VAD: speech %.1fs long, silence requirement raised to %d ms",
                    (elapsed_samples - speech_start_samples) / settings.SAMPLE_RATE,
                    self._long_utterance_silence_duration_ms,
                )

            # 2026-05-16 latency pass 2: Smart Turn V3 medium-grace
            # timeout. The model returned "medium-confidence complete"
            # at the fast-path checkpoint; we waited the additional
            # ``medium_grace_ms`` of silence (matching the prior 500 ms
            # baseline). User did not resume speaking, so trust the
            # medium verdict and submit.
            if (
                smart_turn_used
                and smart_turn_medium_anchor > 0
                and (elapsed_samples - smart_turn_medium_anchor) >= medium_grace_samples
            ):
                logger.info(
                    "Smart Turn V3 medium-grace elapsed (%d ms); "
                    "accepting end-of-turn",
                    self._smart_turn_medium_grace_ms,
                )
                break

            # Smart Turn V3 extension timeout: if the model said
            # "incomplete" but the user never actually resumed
            # speaking, accept end-of-turn after the configured
            # extension window. Without this, the orchestrator would
            # hang until max_utterance_seconds whenever smart-turn
            # was wrong about an incomplete verdict.
            if (
                smart_turn_used
                and smart_turn_incomplete_anchor > 0
                and (elapsed_samples - smart_turn_incomplete_anchor) >= extension_samples
            ):
                logger.info(
                    "Smart Turn V3 extension timeout: no resumed speech in "
                    "%d ms; accepting end-of-turn",
                    self._smart_turn_incomplete_extension_ms,
                )
                break

            if not speech_seen:
                leading_silence += chunk.shape[0]
                if leading_silence >= silence_grace:
                    # Stop streaming on the early-bail path so the
                    # next capture starts with a clean session.
                    if streaming_active:
                        self._maybe_stop_stt_stream()
                    return np.zeros(0, dtype=np.float32)

        # Finalize streaming -- the engine stashes the final text so
        # the orchestrator's downstream ``transcribe(buffer)`` call
        # returns it instantly without re-running the model.
        if streaming_active:
            self._maybe_stop_stt_stream()
        return np.concatenate(chunks).astype(np.float32, copy=False)

    # --- phase: follow-up listening -----------------------------------------

    def _follow_up_listen(self, deadline: float) -> Union[str, np.ndarray]:
        """Wait for either the wake word or a VAD-bounded utterance.

        Returns one of:
        - ``_FU_TIMEOUT`` when the deadline elapses without either firing
        - ``_FU_WAKE`` when the wake word fires (orchestrator should re-arm
          for a fresh wake-gated capture)
        - an ``np.ndarray`` containing the captured utterance audio when VAD
          reports SPEECH_END

        Smart Turn V3 confirmation (2026-05-12) mirrors the COLD-path
        behaviour in :meth:`_capture_utterance`: when enabled + model
        available, an early SPEECH_END is confirmed by the model;
        ``incomplete`` extends the capture by ``incomplete_extension_ms``
        and bumps the VAD silence requirement to the legacy backstop.
        """
        self.audio.drain()
        self.wake.reset()
        self.vad.reset()
        # 2026-05-19 Tracks 1c-1e: mirror of _capture_utterance --
        # best-effort cancel the in-flight summarizer so the foreground
        # follow-up path has the GPU.
        self._cancel_background_summarizer()
        # 2026-05-16 latency pass 2: drop any stale speculative STT
        # result from a prior turn (mirror of _capture_utterance).
        self._reset_speculative_stt_state()
        # 2026-05-18 latency pass 3 (Phase 1): kick off the PortAudio
        # device open now so the ~50 ms open cost overlaps the entire
        # follow-up listen window. Mirrors the COLD-path placement in
        # _capture_utterance. Idempotent / fail-open. See the prose in
        # _capture_utterance for the full rationale.
        self._kick_off_tts_preopen()
        # Don't clear the ring — we want pre-roll continuity from the moment
        # TTS finished.

        speech_started = False
        speech_chunks: list[np.ndarray] = []
        pre_roll: Optional[np.ndarray] = None
        speech_samples = 0
        max_samples = int(self._max_utterance_seconds * settings.SAMPLE_RATE)
        # Smart Turn V3 state -- mirrors _capture_utterance.
        smart_turn_used = False
        smart_turn_incomplete_anchor = 0
        smart_turn_medium_anchor = 0
        extension_samples = int(
            self._smart_turn_incomplete_extension_ms / 1000.0 * settings.SAMPLE_RATE
        )
        medium_grace_samples = int(
            self._smart_turn_medium_grace_ms / 1000.0 * settings.SAMPLE_RATE
        )
        # 2026-05-16 latency pass 2: speculative STT state mirrors
        # _capture_utterance. Kick off Whisper on first short run of
        # silence chunks; consume in main run() via
        # _collect_speculative_stt.
        consecutive_silence_chunks = 0
        speculative_kicked = False
        speculative_silence_kickoff_chunks = 2
        # 2026-06-12: streaming-STT lane on the WARM path. On streaming
        # engines (Moonshine medium-streaming-en) the speculative lane
        # above is a deliberate no-op (_kick_off_speculative_stt's
        # streaming race guard), which left follow-up turns paying a
        # full synchronous Moonshine re-transcribe (~700 ms live) in
        # run()'s foreground STT call. Mirror the COLD-path streaming
        # session here -- but start it at SPEECH_START (not
        # window-open) so the streamed audio is exactly pre_roll +
        # speech_chunks, matching the returned buffer, and no CPU is
        # burned transcribing room chatter for the whole warm window.
        streaming_active = False

        while not self._shutdown.is_set() and time.monotonic() < deadline:
            chunk = self.audio.get_chunk(timeout=0.1)
            if chunk is None:
                continue
            self.ring.write(chunk)

            # Wake word always wins — even if we're mid-utterance.
            if self.wake.process(chunk):
                if streaming_active:
                    # The captured audio is being dropped -- discard
                    # the partial transcript so it can't leak into the
                    # next capture's cache-hit path.
                    self._maybe_discard_stt_stream()
                return _FU_WAKE

            result = self.vad.process(chunk)

            if not speech_started:
                if result.event == SpeechEvent.SPEECH_START:
                    # WARM pre-roll: take the longer slice so the
                    # leading word isn't clipped. Silero VAD has
                    # ~100-200 ms detection latency on speech-start;
                    # without enough pre-roll we capture audio FROM
                    # the speech-start moment forward, missing the
                    # first phoneme(s) the user already produced.
                    warm_pre_roll_samples = int(
                        self._warm_pre_roll_seconds * settings.SAMPLE_RATE
                    )
                    pre_roll = self.ring.snapshot(warm_pre_roll_samples)
                    speech_chunks.append(chunk)
                    speech_started = True
                    speech_samples = chunk.shape[0]
                    # 2026-06-12: start the streaming-STT session at
                    # SPEECH_START and feed exactly what the returned
                    # buffer will contain (pre-roll + chunks).
                    streaming_active = self._maybe_start_stt_stream()
                    if streaming_active:
                        if pre_roll is not None and pre_roll.size:
                            self._maybe_feed_stt_chunk(pre_roll)
                        self._maybe_feed_stt_chunk(chunk)
                # else: still waiting for speech — keep ticking.
                continue

            speech_chunks.append(chunk)
            speech_samples += chunk.shape[0]
            if streaming_active:
                self._maybe_feed_stt_chunk(chunk)

            if result.event == SpeechEvent.SPEECH_START and smart_turn_used:
                # User resumed speaking after smart-turn said
                # incomplete or medium -- cancel both timeouts.
                smart_turn_incomplete_anchor = 0
                smart_turn_medium_anchor = 0

            # 2026-05-16 latency pass 2: invalidate any in-flight
            # speculative STT when speech resumes -- the captured
            # audio at speculative kick-off time is now a stale prefix
            # of the real utterance.
            if result.event == SpeechEvent.SPEECH_START and speculative_kicked:
                self._invalidate_speculative_stt()
                speculative_kicked = False
                consecutive_silence_chunks = 0

            if result.event == SpeechEvent.SPEECH_END:
                if (
                    not smart_turn_used
                    and self._smart_turn_should_check(
                        speech_seen=True,
                        speech_samples=speech_samples,
                    )
                ):
                    pieces = (
                        [pre_roll] if pre_roll is not None else []
                    ) + speech_chunks
                    captured = np.concatenate(pieces).astype(
                        np.float32, copy=False,
                    )
                    verdict = self._run_smart_turn(captured)
                    smart_turn_used = True
                    # 2026-05-16 latency pass 2: gradient-fire bands.
                    band = self._classify_smart_turn_verdict(verdict)
                    if band == "undecided":
                        if streaming_active:
                            self._maybe_stop_stt_stream()
                        return captured
                    if band == "early_complete":
                        logger.info(
                            "Smart Turn V3 (follow-up): early-complete "
                            "(prob=%.3f, %.1f ms)",
                            verdict.probability, verdict.latency_ms,
                        )
                        if streaming_active:
                            self._maybe_stop_stt_stream()
                        return captured
                    if band == "medium_complete":
                        # Wait the medium-grace window; if user
                        # doesn't resume speaking, the timeout below
                        # accepts end-of-turn and returns.
                        smart_turn_medium_anchor = speech_samples
                        logger.info(
                            "Smart Turn V3 (follow-up): medium-complete "
                            "(prob=%.3f, %.1f ms) -- waiting %d ms more",
                            verdict.probability,
                            verdict.latency_ms,
                            self._smart_turn_medium_grace_ms,
                        )
                    else:
                        # band == "incomplete": extend the capture.
                        smart_turn_incomplete_anchor = speech_samples
                        self.vad.set_min_silence_duration_ms(
                            self._long_utterance_silence_duration_ms
                        )
                        logger.info(
                            "Smart Turn V3 (follow-up): incomplete "
                            "(prob=%.3f, %.1f ms) -- extending up to %d ms",
                            verdict.probability,
                            verdict.latency_ms,
                            self._smart_turn_incomplete_extension_ms,
                        )
                else:
                    pieces = (
                        [pre_roll] if pre_roll is not None else []
                    ) + speech_chunks
                    if streaming_active:
                        self._maybe_stop_stt_stream()
                    return np.concatenate(pieces).astype(
                        np.float32, copy=False,
                    )

            # 2026-05-16 latency pass 2: speculative STT during the
            # silence-accumulation phase. Mirror of the
            # _capture_utterance logic: track consecutive silence
            # chunks while speech_started; kick off Whisper once we
            # cross the threshold.
            if speech_started:
                if result.probability < self.vad.threshold:
                    consecutive_silence_chunks += 1
                else:
                    consecutive_silence_chunks = 0
                if (
                    not speculative_kicked
                    and consecutive_silence_chunks >= speculative_silence_kickoff_chunks
                ):
                    speculative_kicked = True
                    pieces = (
                        [pre_roll] if pre_roll is not None else []
                    ) + speech_chunks
                    audio_so_far = np.concatenate(pieces).astype(
                        np.float32, copy=False,
                    )
                    self._kick_off_speculative_stt(audio_so_far)

            # 2026-05-16 latency pass 2: medium-grace timeout. Same
            # gradient-fire band that fired at the (shortened)
            # fast-path checkpoint; user did not resume speaking
            # during the grace window, so trust the medium verdict
            # and submit.
            if (
                smart_turn_used
                and smart_turn_medium_anchor > 0
                and (speech_samples - smart_turn_medium_anchor) >= medium_grace_samples
            ):
                logger.info(
                    "Smart Turn V3 (follow-up) medium-grace elapsed "
                    "(%d ms); accepting end-of-turn",
                    self._smart_turn_medium_grace_ms,
                )
                pieces = (
                    [pre_roll] if pre_roll is not None else []
                ) + speech_chunks
                if streaming_active:
                    self._maybe_stop_stt_stream()
                return np.concatenate(pieces).astype(
                    np.float32, copy=False,
                )

            # Smart-turn extension timeout: if "incomplete" was
            # returned and no speech-resume cancelled the anchor,
            # accept end-of-turn after the configured grace.
            if (
                smart_turn_used
                and smart_turn_incomplete_anchor > 0
                and (speech_samples - smart_turn_incomplete_anchor) >= extension_samples
            ):
                logger.info(
                    "Smart Turn V3 (follow-up) extension timeout: "
                    "accepting end-of-turn after %d ms of silence",
                    self._smart_turn_incomplete_extension_ms,
                )
                pieces = (
                    [pre_roll] if pre_roll is not None else []
                ) + speech_chunks
                if streaming_active:
                    self._maybe_stop_stt_stream()
                return np.concatenate(pieces).astype(
                    np.float32, copy=False,
                )

            if speech_samples >= max_samples:
                # Hard cap — return what we have, classifier can still gate it.
                pieces = ([pre_roll] if pre_roll is not None else []) + speech_chunks
                if streaming_active:
                    self._maybe_stop_stt_stream()
                return np.concatenate(pieces).astype(np.float32, copy=False)

        if streaming_active:
            # Deadline elapsed (or shutdown) mid-utterance: the audio
            # is being dropped, so discard the partial transcript too.
            self._maybe_discard_stt_stream()
        return _FU_TIMEOUT

    # --- coding pipeline glue -----------------------------------------------

    def _speak(self, text: str) -> None:
        """Synchronously speak a fixed string + print it. Used by the coding
        pipeline for progress narrations and completion announcements --
        the regular LLM streaming path uses ``speak_stream`` instead."""
        if not text:
            return
        print(f"  kenning: {text}")
        try:
            self.tts.speak(text)
        except Exception as e:
            logger.warning("speak failed: %s", e)

    def _handle_capability_response(
        self, response, routing_intent,
    ) -> None:
        """Speak the response, applying A4 pre-task confirmation when set.

        Default path (no ``pre_task_confirmation``): same as before --
        speak ``response.text``.

        A4 path: speak ``response.pre_task_confirmation`` first with
        barge-in detection. If barge-in fires, audit the abort and
        skip both the deferred dispatch AND the post-dispatch ``text``
        (replaced with a brief acknowledgement).
        """
        pre_text = getattr(response, "pre_task_confirmation", None)
        deferred = getattr(response, "deferred_dispatch", None)
        label = getattr(response, "pre_task_label", None)
        if pre_text and deferred is not None:
            window_s = float(
                getattr(settings, "CODING_PRE_TASK_BARGE_IN_WINDOW_S", 0.5)
            )
            barge_in = self._speak_with_barge_in_check(
                pre_text, post_check_window_s=window_s,
            )
            if barge_in:
                # Record the abort + speak a short cancellation. The
                # next utterance becomes the user's clarifying input.
                if self.coding_voice is not None:
                    try:
                        self.coding_voice.runner.record_pre_task_aborted(
                            label=label,
                            reason="barge_in",
                            intent_text=getattr(routing_intent, "raw_text", ""),
                        )
                    except Exception as e:
                        logger.debug("record_pre_task_aborted failed: %s", e)
                self._speak("Cancelled. What did you mean?")
                return
            # No barge-in -- run the dispatch.
            try:
                deferred()
            except Exception as e:
                logger.warning("A4 deferred dispatch raised: %s", e)
            self._speak(response.text)
            return
        # Default path.
        self._speak(response.text)

    def _speak_with_barge_in_check(
        self,
        text: str,
        *,
        post_check_window_s: float = 0.5,
    ) -> bool:
        """A4: speak ``text`` and report whether wake-word fired during
        the playback or for ``post_check_window_s`` after.

        Returns ``True`` when the wake-word detector saw a fire
        recently enough that we should treat it as a barge-in --
        meaning the caller (pre-task confirmation path) should DROP
        the deferred dispatch.

        TTS failure is treated as "no barge-in detected" -- we don't
        want a Piper hiccup to silently brick the coding pipeline. The
        caller still dispatches.
        """
        if not text:
            return False
        # Record the trigger timestamp BEFORE speaking so a wake fire
        # that happened during a previous utterance can't be confused
        # with one during this confirmation.
        before_ts = self.wake._last_trigger_ts  # noqa: SLF001
        print(f"  kenning: {text}")
        try:
            self.tts.speak(text)
        except Exception as e:
            logger.warning("pre-task speak failed: %s", e)
            return False
        # Wait briefly so the user has a window to fire the wake word
        # AFTER the TTS audio actually finishes playing back. Cheap.
        try:
            time.sleep(max(0.0, float(post_check_window_s)))
        except Exception:
            pass
        try:
            after_ts = self.wake._last_trigger_ts  # noqa: SLF001
        except Exception:
            return False
        # Barge-in iff the trigger timestamp advanced during/after the speak.
        if after_ts > before_ts and after_ts > 0:
            logger.info(
                "A4 barge-in detected during pre-task confirmation",
            )
            return True
        return False

    def _drain_evolution_command_failures(self) -> None:
        """Catalog 14 (T1): drain command/tool failures the coding runner
        queued during a task and feed each to the EvolutionService (which runs
        the failure detector + the repair-distillation feed). Fail-open + a
        zero-cost no-op when evolution or coding is disabled."""
        if self.evolution is None or self.coding_voice is None:
            return
        try:
            runner = getattr(self.coding_voice, "runner", None)
            drain = getattr(runner, "drain_command_failures", None)
            if drain is None:
                return
            for command, output, exit_code in drain():
                self.evolution.record_command_failure(command, output, exit_code=exit_code)
        except Exception as e:  # noqa: BLE001
            logger.debug("evolution command-failure drain failed: %s", e)

    def _drain_evolution_narrations(self) -> None:
        """Guardrail brake (#15+#65): speak any one-line narration the
        EvolutionService queued (currently the post-apply auto-revert
        notice -- "I rolled back my most recent self-improvement..."), so
        the user hears when a kept skill was withdrawn. Fail-open + a
        zero-cost no-op when evolution is disabled or nothing is queued."""
        evolution = getattr(self, "evolution", None)
        if evolution is None:
            return
        try:
            pop = getattr(evolution, "pop_pending_narration", None)
            if pop is None:
                return
            line = pop()
            if line:
                self._speak(line)
                self._last_response_finished_monotonic = time.monotonic()
        except Exception as e:  # noqa: BLE001
            logger.debug("evolution narration drain failed: %s", e)

    def _drain_coding_dialog_narrations(self) -> None:
        """Speak any dialog-appearance narration the coding runner's dialog
        auto-handler queued (catalog 08/09, default ON). The handler detects a
        native dialog mid-task and queues a voice-friendly line (e.g. "A 'Save
        As' dialog appeared in notepad.exe -- shall I confirm?"); this drains +
        speaks it so the user can respond (their spoken yes/no then routes via
        WINDOW_CLOSE_CONFIRMATION). Without this drain the auto-handler's
        narrations were queued but never surfaced. Fail-open + a zero-cost
        no-op when coding is disabled."""
        if self.coding_voice is None:
            return
        try:
            runner = getattr(self.coding_voice, "runner", None)
            pop = getattr(runner, "pop_dialog_narration", None)
            if pop is None:
                return
            while True:
                line = pop()
                if not line:
                    break
                self._speak(line)
                self._last_response_finished_monotonic = time.monotonic()
        except Exception as e:  # noqa: BLE001
            logger.debug("coding dialog-narration drain failed: %s", e)

    def _drain_coding_loop_alerts(self) -> None:
        """T1: speak any loop-detection heads-up the coding runner queued when
        a task's tool-call stream tripped a hard escalation (the same tool
        failing identically the circuit-breaker number of times). The runner
        narrates at most once per task; this drains + speaks each queued line.
        Fail-open + a zero-cost no-op when coding is disabled."""
        if self.coding_voice is None:
            return
        try:
            runner = getattr(self.coding_voice, "runner", None)
            pop = getattr(runner, "pop_loop_alert", None)
            if pop is None:
                return
            while True:
                alert = pop()
                if not alert:
                    break
                self._speak(alert)
                self._last_response_finished_monotonic = time.monotonic()
        except Exception as e:  # noqa: BLE001
            logger.debug("coding loop-alert drain failed: %s", e)

    def _announce_coding_completion_if_pending(self) -> None:
        """If a background coding task just finished, speak its summary
        before we go back to listening for the next utterance.

        Phase 4: also fires a proactive Telegram notification with the
        same summary text so the user gets the update on their phone
        when they're away from the desk. Fire-and-forget; failures
        log and never propagate."""
        if self.coding_voice is None:
            return
        try:
            narration = self.coding_voice.pending_completion()
        except Exception as e:
            logger.warning("coding_voice.pending_completion failed: %s", e)
            return
        if narration:
            self._speak(narration)
            self._last_response_finished_monotonic = time.monotonic()
            self._notify_coding_completion(narration)

    def _announce_pending_clarifications(self) -> None:
        """Speak any clarifications Claude is waiting on. Each prompt is
        spoken at most once -- the user's next utterance answers it.

        Phase 4: also fires a Telegram notification per clarification
        so a user away from the desk knows their attention is needed."""
        if self.coding_voice is None:
            return
        try:
            prompts = self.coding_voice.pending_clarifications()
        except Exception as e:
            logger.warning("coding_voice.pending_clarifications failed: %s", e)
            return
        for prompt in prompts:
            self._speak(prompt)
            self._last_response_finished_monotonic = time.monotonic()
            self._notify_coding_clarification(prompt)

    def _notify_coding_completion(self, summary: str) -> None:
        """Fire-and-forget Telegram notification for a coding-task
        completion. Bridge handles all gating + fail-open; this
        helper just bridges the sync orchestrator loop to the async
        notifier."""
        if self.openclaw_bridge is None:
            return
        try:
            self.openclaw_bridge.fire_and_forget(
                lambda: self.openclaw_bridge.notifications.notify_coding_task_completion(
                    summary,
                ),
            )
        except Exception as e:                                 # noqa: BLE001
            logger.warning(
                "coding-completion notification dispatch failed: %s", e,
            )

    def _notify_coding_clarification(self, prompt: str) -> None:
        """Fire-and-forget Telegram notification for a clarification
        request. See :meth:`_notify_coding_completion`."""
        if self.openclaw_bridge is None:
            return
        try:
            self.openclaw_bridge.fire_and_forget(
                lambda: self.openclaw_bridge.notifications.notify_coding_task_clarification(
                    prompt,
                ),
            )
        except Exception as e:                                 # noqa: BLE001
            logger.warning(
                "clarification notification dispatch failed: %s", e,
            )

    def _announce_pending_run_report(self) -> None:
        """B3: speak the result of a backgrounded sandbox program run
        ("run the calculator") once it finishes. Drained each voice-loop
        iteration; fail-open + no-op when coding is disabled or nothing is
        pending."""
        cv = getattr(self, "coding_voice", None)
        if cv is None:
            return
        try:
            report = cv.pop_run_report()
        except Exception as e:                                       # noqa: BLE001
            logger.debug("run-report drain failed: %s", e)
            return
        if report:
            self._speak(report)

    def _announce_pending_budget_warning(self) -> None:
        """Phase 7: surface token-budget warnings (80%) and halt notices
        (100%) raised by the runner. Spoken once per crossing."""
        if self.coding_voice is None:
            return
        try:
            warning = self.coding_voice.pending_budget_warning()
        except Exception as e:
            logger.warning("coding_voice.pending_budget_warning failed: %s", e)
            return
        if warning:
            self._speak(warning)
            self._last_response_finished_monotonic = time.monotonic()

    def _announce_pending_canonical_abort(self) -> None:
        """4B plan Item 7: surface canonical-path-monitor aborts.

        When the monitor cancels a coding session for going off the
        rails, the runner queues a one-line voice narration; this
        method speaks it once. Mirrors
        :meth:`_announce_pending_budget_warning`.
        """
        if self.coding_voice is None:
            return
        try:
            warning = self.coding_voice.pending_canonical_abort()
        except Exception as e:
            logger.warning("coding_voice.pending_canonical_abort failed: %s", e)
            return
        if warning:
            self._speak(warning)
            self._last_response_finished_monotonic = time.monotonic()

    def _announce_pending_anchor_narration(self) -> None:
        """E2 goal-anchor planning: surface per-anchor voice narration.

        The runner queues opening / warning / transition / completion
        lines as USAGE events advance the active anchor. This method
        polls + speaks once per top-of-loop iteration. No-op when
        goal-anchors are disabled (the pop returns ``None``).
        """
        if self.coding_voice is None:
            return
        try:
            narration = self.coding_voice.pending_anchor_narration()
        except Exception as e:
            logger.warning(
                "coding_voice.pending_anchor_narration failed: %s", e,
            )
            return
        if narration:
            self._speak(narration)
            self._last_response_finished_monotonic = time.monotonic()

    def _maybe_run_background_summarizer(self) -> None:
        """2026-05-19 Tracks 1c-1e voice-loop hook.

        Spawn a daemon thread that calls the background summarizer if:
        * the summarizer was constructed (flag on at start-up),
        * no previous summarizer thread is still running.

        The summarizer's own gating (idle threshold + cadence +
        min_turns + in_flight) decides whether the call actually
        performs an LLM round-trip on this attempt -- so it is safe to
        invoke from the top of every run-loop iteration. Most calls
        short-circuit cheaply inside ``maybe_summarize``.

        Wakes through the run loop are independent of the summarizer:
        the wake-word listener runs on its own audio thread and is not
        affected by an in-flight summary call. Foreground LLM
        contention is avoided in practice because the summarizer only
        proceeds after ``idle_threshold_seconds`` of quiet, which is
        well past the moment any prior foreground call finished. If
        the user does wake Kenning during a summary, the cancel flag
        is set (best-effort) and the foreground call effectively
        serialises behind the in-flight LLM call -- at most a ~1-2 s
        delay on the first response.

        Fail-open: any exception in thread launch is swallowed; the
        next iteration retries.
        """
        if self.background_summarizer is None:
            return
        with self._background_summarizer_lock:
            prev = self._background_summarizer_thread
            if prev is not None and prev.is_alive():
                return
            last_activity = self._last_response_finished_monotonic
            if last_activity <= 0.0:
                # No foreground turn has finished yet on this process.
                # Skip until at least one round-trip has happened so the
                # idle-threshold gate has a meaningful reference point.
                return

            def _run() -> None:
                try:
                    self.background_summarizer.maybe_summarize(
                        last_activity_monotonic=last_activity,
                    )
                except Exception as e:                        # noqa: BLE001
                    logger.warning(
                        "background summarizer maybe_summarize failed (%s); "
                        "next attempt will try again.", e,
                    )

            try:
                t = threading.Thread(
                    target=_run,
                    name="kenning-background-summarizer",
                    daemon=True,
                )
                t.start()
                self._background_summarizer_thread = t
            except Exception as e:                            # noqa: BLE001
                logger.warning(
                    "background summarizer thread launch failed (%s)", e,
                )

    def _cancel_background_summarizer(self) -> None:
        """Signal any in-flight summarizer call to abort ASAP.

        Called when the orchestrator pivots to capture or speak so the
        GPU is freed for the foreground path. Idempotent. No-op when
        the summarizer is disabled. Does NOT join the thread -- the
        summarizer's cancel flag is read between LLM sub-calls and at
        end-of-pass; an actively-streaming LLM call still has to
        complete before the lock-free transition happens (acceptable
        worst-case delay of ~1-2 s).
        """
        if self.background_summarizer is None:
            return
        try:
            self.background_summarizer.cancel()
        except Exception as e:                                # noqa: BLE001
            logger.warning(
                "background summarizer cancel failed (%s); ignoring.", e,
            )

    # --- phase: process ------------------------------------------------------

    def _resolve_cited_source(
        self, ordinal=None, referent: str = "",
    ):
        """Pick the source the user is asking to open.

        Strategy (priority order, first hit wins):

        1. Ordinal: when the user said "the first/second/third" or
           "number 2", return ``sources[ordinal-1]``. Out-of-range
           (e.g. "the fifth" when only 3 sources exist) falls
           through to the next strategy.
        2. Referent substring: when the user supplied a phrase
           ("NBC", "Boeing crash"), search source titles + domain
           roots for a case-insensitive substring match. Longest
           match wins.
        3. Embedding similarity (referent only): when the dense
           embedder is available, encode the referent + each source
           title and pick the best cosine match above 0.55.
        4. Cited-in-response: scan the LLM's last response text for
           publication names (existing behaviour for bare "show me
           that article" with no referent).
        5. Source[0] fallback: the first source in the list.

        Returns the chosen source or None when the payload is empty.
        """
        payload = self._last_search_payload
        if not payload or not payload.sources:
            return None
        sources = payload.sources

        # Strategy 1: ordinal.
        if ordinal is not None:
            if ordinal == -1:
                return sources[-1]
            if 1 <= ordinal <= len(sources):
                return sources[ordinal - 1]
            logger.info(
                "open_last_source: ordinal=%s out of range (have %d "
                "sources); falling through to referent / fallback.",
                ordinal, len(sources),
            )

        from urllib.parse import urlparse
        import re as _re

        # Build candidate strings (title segments, domain roots) per
        # source. Used by both the referent substring match AND the
        # cited-in-response fallback below.
        def _candidates_for(source) -> list:
            cands: list = []
            title = (getattr(source, "title", "") or "").strip()
            if title:
                cands.append(title)
                segments = _re.split(r"\s*[|\-–—·:]\s*", title)
                for seg in segments:
                    seg = seg.strip()
                    if seg and len(seg) >= 3 and seg != title:
                        cands.append(seg)
            url = getattr(source, "url", "") or ""
            try:
                host = urlparse(url).hostname or ""
            except Exception:                                       # noqa: BLE001
                host = ""
            if host:
                stripped = host.lower().replace("www.", "")
                root = stripped.split(".")[0]
                if root and len(root) >= 4:
                    cands.append(root)
                    for suffix in (
                        "news", "times", "post", "today", "daily",
                        "press", "tribune", "herald", "journal",
                        "magazine", "review",
                    ):
                        if root.endswith(suffix) and root != suffix:
                            cands.append(
                                f"{root[:-len(suffix)]} {suffix}"
                            )
            return cands

        # Strategy 2: referent substring match.
        if referent:
            ref_lc = referent.lower().strip()
            best = None
            best_len = 0
            for source in sources:
                for cand in _candidates_for(source):
                    cand_lc = cand.lower().strip()
                    if len(cand_lc) < 3:
                        continue
                    # Match in either direction: referent in candidate
                    # ("NBC" in "U.S. News | NBC News") OR candidate in
                    # referent ("nbcnews" in "the nbc news story").
                    if cand_lc in ref_lc or ref_lc in cand_lc:
                        match_len = min(len(cand_lc), len(ref_lc))
                        if match_len > best_len:
                            best = source
                            best_len = match_len
            if best is not None:
                return best

        # Strategy 3: embedding similarity (referent only, when the
        # dense embedder is available).
        if referent and self.memory is not None:
            chosen = self._embedding_pick_source(referent, sources)
            if chosen is not None:
                return chosen

        # Strategy 4: cited-in-response (fallback for bare "show me
        # that article" with no referent).
        response_lc = (self._last_response_text or "").lower()
        if not response_lc:
            return sources[0]

        best = None
        best_match_len = 0
        for source in sources:
            for cand in _candidates_for(source):
                cand_lc = cand.lower().strip()
                if len(cand_lc) < 3:
                    continue
                if cand_lc in response_lc and len(cand_lc) > best_match_len:
                    best = source
                    best_match_len = len(cand_lc)

        # Strategy 5: fallback to source [0].
        return best if best is not None else sources[0]

    def _embedding_pick_source(self, referent: str, sources):
        """Pick the source whose title best matches ``referent`` by
        dense-embedding cosine similarity.

        Returns the chosen source or None when the embedder is
        unavailable, similarities are too low, or any error occurs.
        Threshold: 0.55. Below that, callers should fall through to
        the next resolution strategy.
        """
        try:
            embedder = self.memory._embedder  # noqa: SLF001
        except Exception:                                           # noqa: BLE001
            return None
        if embedder is None:
            return None

        titles = [(getattr(s, "title", "") or "").strip() for s in sources]
        if not any(titles):
            return None
        try:
            import numpy as np
            doc_vecs = embedder.encode_dense(titles)
            query_vec = embedder.encode_query_dense(referent)
            # Cosine similarity (vectors are L2-normalized by bge-small,
            # but guard against future model swaps).
            def _norm(v):
                n = np.linalg.norm(v)
                return v / n if n > 0 else v
            doc_n = np.stack([_norm(v) for v in doc_vecs])
            q_n = _norm(query_vec)
            sims = doc_n @ q_n
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
        except Exception as e:                                      # noqa: BLE001
            logger.debug("embedding source pick failed: %s", e)
            return None

        if best_sim < 0.55:
            logger.info(
                "open_last_source: embedding best sim=%.3f below 0.55 "
                "threshold; falling through.", best_sim,
            )
            return None

        logger.info(
            "open_last_source: embedding pick idx=%d sim=%.3f title=%r",
            best_idx, best_sim, titles[best_idx][:80],
        )
        return sources[best_idx]

    def _handle_open_last_source(self, routing_intent) -> None:
        """Open the URL cited in the most recent search-augmented turn.

        Routes to :func:`webbrowser.open` by default. When the user
        specifies a monitor target ("on monitor 2"), routes through
        :func:`kenning.desktop.voice.handle_app_launch` so Chrome opens
        on the requested screen.

        Fail-open: missing payload, missing URL, browser-open failure
        all degrade to a spoken voice message; never raise.
        """
        intent = getattr(routing_intent, "open_last_source_intent", None)
        ordinal = getattr(intent, "ordinal", None) if intent else None
        referent = getattr(intent, "referent", "") if intent else ""
        chosen = self._resolve_cited_source(
            ordinal=ordinal, referent=referent,
        )
        if chosen is None:
            msg = "I don't have a recent article to open from our last exchange."
            self._handle_capability_response(
                _voice_text(msg), routing_intent,
            )
            return

        url = getattr(chosen, "url", "") or ""
        title = getattr(chosen, "title", "") or url
        if not url:
            msg = "I have a source but its URL is missing."
            self._handle_capability_response(
                _voice_text(msg), routing_intent,
            )
            return

        mon_idx = getattr(intent, "monitor_index", None) if intent else None
        mon_q = getattr(intent, "monitor_query", "") if intent else ""

        if mon_idx is not None or mon_q:
            try:
                from kenning.openclaw_routing.intents import AppLaunchIntent
                from kenning.desktop.voice import handle_app_launch

                al = AppLaunchIntent(
                    app_name="chrome",
                    url=url,
                    monitor_index=mon_idx,
                    monitor_query=mon_q,
                    fullscreen=False,
                    maximize=False,
                    raw_text=getattr(routing_intent, "raw_text", ""),
                )
                result = handle_app_launch(al)
                self._handle_capability_response(
                    _voice_text(result.voice_message or
                                     f"Opening {title}."),
                    routing_intent,
                )
                return
            except Exception as e:                                  # noqa: BLE001
                logger.warning("Monitor-targeted source open failed: %s; "
                               "falling back to default browser.", e)

        try:
            import webbrowser
            webbrowser.open(url, new=2)
            self._handle_capability_response(
                _voice_text(f"Opening {title}."),
                routing_intent,
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("webbrowser.open failed for %s: %s", url, e)
            self._handle_capability_response(
                _voice_text("I couldn't open the browser."),
                routing_intent,
            )

    def _handle_navigate_to_site(self, routing_intent) -> None:
        """Query SearxNG for the user's brand-named site, pick the
        best matching URL by domain heuristics, and open it.

        Strategy:
          1. Query SearxNG with ``{site_query} official website``
             (general category, top ~10 results).
          2. Score each result by: hostname-contains-brand-keyword
             (+30), exact-brand-domain-match like ``netflix.com``
             (+40), no subdomain like ``www.netflix.com`` (+10),
             rank inverse (+0..9). Penalize generic listing sites
             (wikipedia, reddit, ...).
          3. Open the best candidate via ``webbrowser.open`` (or
             the AppLauncher with Chrome when a monitor target was
             explicitly requested).

        Fail-open: empty search results, no acceptable candidates,
        or open-failure all degrade to a spoken voice message.
        """
        from urllib.parse import urlparse
        import re as _re

        intent = getattr(routing_intent, "navigate_to_site_intent", None)
        site_query = (
            getattr(intent, "site_query", "") if intent else ""
        ).strip()
        if not site_query:
            self._handle_capability_response(
                _voice_text("I didn't catch which site you wanted opened."),
                routing_intent,
            )
            return

        mon_idx = getattr(intent, "monitor_index", None) if intent else None
        mon_q = getattr(intent, "monitor_query", "") if intent else ""

        # Build a normalized brand keyword for hostname matching.
        # "HBO Max" -> "hbomax"; "Disney Plus" -> "disneyplus".
        brand_key = _re.sub(r"[^a-z0-9]", "", site_query.lower())

        # Query SearxNG (general category) for the official site.
        try:
            from kenning.web_search.searxng import SearxNGSearchClient
            client = SearxNGSearchClient()
            search_q = f"{site_query} official website"
            results = client.search(search_q, count=10)
        except Exception as e:                                      # noqa: BLE001
            logger.warning("NAVIGATE_TO_SITE search failed: %s", e)
            results = []

        if not results:
            # Fall back to a Google "I'm feeling lucky" style URL --
            # better than nothing when SearxNG returns nothing.
            url = (
                "https://www.google.com/search?btnI=1&q="
                + _re.sub(r"\s+", "+", site_query)
                + "+official+site"
            )
            title = site_query
            chosen_score = -1
        else:
            # Score each result.
            _PENALTY_HOSTS = {
                "wikipedia.org", "en.wikipedia.org", "reddit.com",
                "facebook.com", "twitter.com", "x.com",
                "linkedin.com", "youtube.com", "instagram.com",
                "tiktok.com", "amazon.com",
            }

            def _score(result, rank: int) -> int:
                url = getattr(result, "url", "") or ""
                try:
                    host = (urlparse(url).hostname or "").lower()
                except Exception:                                   # noqa: BLE001
                    return -1
                if not host:
                    return -1
                stripped = host.replace("www.", "")
                # Normalize host for keyword match: drop dots / dashes.
                host_norm = _re.sub(r"[^a-z0-9]", "", stripped)
                score = 0
                # Strongest signal: hostname's root matches the brand.
                root = stripped.split(".")[0]
                root_norm = _re.sub(r"[^a-z0-9]", "", root)
                if brand_key and root_norm == brand_key:
                    score += 40
                elif brand_key and brand_key in host_norm:
                    score += 30
                # Clean domain bonus (no leading subdomain).
                if stripped.count(".") == 1:
                    score += 10
                # Standard TLD bonus.
                tld = stripped.split(".")[-1]
                if tld in {"com", "net", "org", "io"}:
                    score += 3
                # Rank inverse: rank 0 -> +9, rank 9 -> +0.
                score += max(0, 9 - rank)
                # Penalize known aggregator / non-official sites
                # unless the user EXPLICITLY asked for them.
                if any(stripped.endswith(p) for p in _PENALTY_HOSTS):
                    if brand_key not in {"wikipedia", "reddit",
                                          "facebook", "twitter", "x",
                                          "linkedin", "youtube",
                                          "instagram", "tiktok",
                                          "amazon"}:
                        score -= 20
                return score

            scored = [
                (_score(r, i), i, r) for i, r in enumerate(results)
            ]
            scored.sort(key=lambda t: (-t[0], t[1]))
            chosen_score, _idx, chosen = scored[0]
            url = getattr(chosen, "url", "") or ""
            title = getattr(chosen, "title", "") or url

        logger.info(
            "navigate_to_site: query=%r -> %s (score=%s)",
            site_query, url, chosen_score,
        )

        if not url:
            self._handle_capability_response(
                _voice_text(
                    f"I couldn't find an official site for {site_query}."
                ),
                routing_intent,
            )
            return

        # Open on requested monitor (Chrome) OR default browser.
        if mon_idx is not None or mon_q:
            try:
                from kenning.openclaw_routing.intents import AppLaunchIntent
                from kenning.desktop.voice import handle_app_launch

                al = AppLaunchIntent(
                    app_name="chrome",
                    url=url,
                    monitor_index=mon_idx,
                    monitor_query=mon_q,
                    fullscreen=False,
                    maximize=False,
                    raw_text=getattr(routing_intent, "raw_text", ""),
                )
                result = handle_app_launch(al)
                msg = result.voice_message or f"Opening {site_query}."
                self._handle_capability_response(
                    _voice_text(msg), routing_intent,
                )
                return
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "Monitor-targeted navigate-to-site open failed: %s; "
                    "falling back to default browser.", e,
                )

        try:
            import webbrowser
            webbrowser.open(url, new=2)
            self._handle_capability_response(
                _voice_text(f"Opening {site_query}."),
                routing_intent,
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("webbrowser.open failed for %s: %s", url, e)
            self._handle_capability_response(
                _voice_text("I couldn't open the browser."),
                routing_intent,
            )

    def _respond(
        self,
        user_text: str,
        *,
        routing_intent_kind: Optional[str] = None,
    ) -> None:
        """Stream LLM tokens into TTS and watch for wake-word interruption.

        Phase 4: classifies the utterance through the web-search gate first.
        SEARCH -> speak an acknowledgment phrase, run the search workflow
        (Brave + Jina + LLM rank) in parallel with the ack TTS, then
        generate the final response with sources injected.
        NO_SEARCH / UNCERTAIN -> base path (unchanged from Phase 3).

        ``routing_intent_kind`` (catalog 09 batch G) is the string value
        of the :class:`RoutingIntentKind` that classified this turn (or
        ``None`` when routing was bypassed). When
        ``llm.history_compression.intent_adaptive`` is enabled the LLM
        engine reads it to pick a per-intent condenser before building
        the prompt. The legacy fixed pipeline ignores it entirely.
        """
        self._interrupt.clear()
        self._last_search_payload = None
        # openclaw-clawhub T15: per-turn telemetry timer + error flag.
        # Emitted in the finally so every turn is counted exactly once.
        turn_start = time.monotonic()
        turn_errored = False
        # 2026-05-22 OPEN_LAST_SOURCE: accumulate the spoken response so
        # the next-turn "show me that article" handler can match cited
        # publication names back to the source list.
        self._last_response_text = ""
        response_buf: list = []
        watcher: Optional[threading.Thread] = None
        if settings.BARGE_IN_ENABLED:
            watcher = threading.Thread(
                target=self._interrupt_watcher, daemon=True, name="wake-watcher"
            )
            watcher.start()
        else:
            logger.info("Barge-in wake watcher disabled")

        # Catalog 09 batch G wiring: thread the intent through to the
        # engine BEFORE building the response stream so _build_messages
        # can pick the right condenser. The clear-on-exit lives in the
        # finally below so a mid-stream exception still leaves the
        # engine in a clean state for the next turn.
        try:
            if self.llm is not None:
                self.llm.set_current_intent_kind(routing_intent_kind)
        except Exception as e:                                       # noqa: BLE001
            logger.debug("set_current_intent_kind failed: %s", e)

        # Catalog 13/14 (evolution): apply the learned response-temperament
        # hint PLUS (catalog 14 T3) a bounded "[Evolution: ...]" pending-queue
        # nudge for THIS turn, through the SAME set_temperament_hint seam.
        # Both inject into the SYSTEM prompt (NOT the user text, so the gate /
        # clock detectors are unaffected). Empty when the temperament is
        # balanced AND the queue is empty -> the prompt is byte-identical to
        # the pre-evolution path. Fail-open.
        try:
            if self.llm is not None and self.evolution is not None:
                self.llm.set_temperament_hint(self.evolution.pre_turn_system_hint())
        except Exception as e:                                       # noqa: BLE001
            logger.debug("set_temperament_hint failed: %s", e)

        try:
            print("  kenning: ", end="", flush=True)
            token_stream = self._build_response_stream(user_text)

            def gated():
                for token in token_stream:
                    if self._interrupt.is_set() or self._shutdown.is_set():
                        self.llm.cancel()
                        return
                    print(token, end="", flush=True)
                    response_buf.append(token)
                    yield token

            self.tts.speak_stream(gated())
            print()  # newline after streamed response
            self._last_response_text = "".join(response_buf)
            # Record the assistant response for "what did you say earlier?"
            # verbatim recall. Fail-open + a no-op when the store is absent.
            self._record_dialogue_turn("assistant", self._last_response_text)

            # 2026-05-22: enable_thinking=False is the active voice-path
            # default (saves 5-10 s TTFT on factual / math turns via the
            # Qwen3 /no_think marker). The trade-off is the loss of the
            # model's chain-of-thought block, which on some classes of
            # harder questions might regress accuracy. Sample
            # ``llm.enable_thinking_drift_sample_rate`` of these turns
            # and emit an observation pairing the user_text with the
            # final response, so offline review can spot regression
            # classes before they bite. Sampling + emit is fail-open --
            # any failure leaves the voice path untouched.
            try:
                self._maybe_emit_thinking_drift_sample(
                    user_text, self._last_response_text,
                )
            except Exception as e:                                  # noqa: BLE001
                logger.debug("thinking_drift_sample emit failed: %s", e)

            # Sources go to the transcript only -- no TTS read-out, since
            # citations interleaved with the spoken answer would clutter the
            # voice output. The user can scan the printed list to verify.
            if self._last_search_payload and self._last_search_payload.sources:
                # Catalog 12 (felo-search T4): surface the search strategy
                # (the reformulated queries fanned out) in the TRANSCRIPT
                # only -- never spoken, so spoken-reply concision is
                # untouched. Gated by web_search.expose_search_strategy
                # (default ON); _format_strategy_line self-suppresses when
                # only a single query was used.
                strat_qs = None
                try:
                    from kenning.config import get_config as _get_cfg
                    if _get_cfg().web_search.expose_search_strategy:
                        strat_qs = self._last_search_payload.queries
                except Exception:                                       # noqa: BLE001
                    strat_qs = None
                print(
                    f"  {format_sources_for_transcript(self._last_search_payload.sources, strategy_queries=strat_qs)}"
                )
        except Exception as e:
            turn_errored = True
            logger.exception("Response pipeline failed: %s", e)
            print(f"\n  [error] {e}")
        finally:
            # Catalog 13 (evolution): capture whether THIS response was
            # barged into, BEFORE we set the interrupt to release the
            # watcher. The next turn's recorder consumes it to nudge the
            # response temperament terser. Fail-open.
            try:
                self._last_turn_barged_in = self._interrupt.is_set()
            except Exception:                                        # noqa: BLE001
                pass
            self._interrupt.set()  # release watcher
            if watcher is not None:
                watcher.join(timeout=1.0)
            # Catalog 09 batch G wiring: clear the per-turn intent so a
            # subsequent direct (non-routed) generate_stream call -- e.g.
            # the speculative LLM thread, the model-swap pre-warm, or a
            # test fixture -- doesn't inherit stale state from this turn.
            try:
                if self.llm is not None:
                    self.llm.set_current_intent_kind(None)
            except Exception as e:                                   # noqa: BLE001
                logger.debug("set_current_intent_kind(None) failed: %s", e)
            # Catalog 13 (evolution): clear the per-turn temperament hint
            # for the same reason -- a direct generate_stream call must not
            # inherit this turn's tone directive.
            try:
                if self.llm is not None:
                    self.llm.set_temperament_hint("")
            except Exception as e:                                   # noqa: BLE001
                logger.debug("set_temperament_hint('') failed: %s", e)
            # openclaw-clawhub T15: emit the aggregate per-turn metric.
            # Fail-private (no-op unless opted in) + fail-open.
            self._emit_turn_telemetry(
                routing_intent_kind, turn_start, errored=turn_errored,
            )
            # Guardrail brake (#15+#65): feed this turn's response-side
            # observation (LLM TTFT + error flag) into the evolution
            # metrics ring. Fail-open + zero-cost when evolution is off.
            self._note_evolution_turn_metrics(turn_errored)

    def _maybe_emit_thinking_drift_sample(
        self, user_text: str, response_text: str,
    ) -> None:
        """Dice-roll a no-think turn into the observations file for review.

        Called from the end of :meth:`_respond` (after the response is
        fully streamed). When ``llm.enable_thinking_drift_sample_rate``
        is > 0 AND ``random.random()`` lands below it, emits one
        ``thinking_drift_sample`` observation with the user text +
        response. Offline reviewers can grep
        ``data/observations.jsonl`` to look for regression classes
        that the no-think default might be masking.

        Fail-open: any failure in the sampling or emit path is
        swallowed -- the voice loop must never be impacted by
        observation IO.
        """
        if not user_text or not response_text:
            return
        try:
            from kenning.config import get_config
            rate = float(
                get_config().llm.enable_thinking_drift_sample_rate,
            )
        except Exception:                                          # noqa: BLE001
            return
        if rate <= 0.0:
            return
        if random.random() >= rate:
            return
        observe_llm_thinking_drift_sample(
            user_text=user_text,
            response_text=response_text,
            user_message_len=len(user_text),
            response_message_len=len(response_text),
        )

    def _maybe_conversational_ack(self, user_text: str) -> Optional[str]:
        """Return a filler-ack phrase to prepend on the conversational
        path, or None if the gate suppresses it.

        2026-05-12 filler-ack: masks the ~2.5 s perceived gap between
        Whisper completing and the LLM's first TTS chunk on the no-
        search conversational branch. The web-search path already
        yields its own ack from :meth:`_search_augmented_tokens`; this
        helper covers the no-search branches only.

        Gate semantics live in
        :func:`kenning.conversational_ack.is_conversational_ack_eligible`;
        this method threads the orchestrator's pending-clarification
        state in so coding-task dialogues don't double-ack.
        """
        has_clar = False
        if self.coding_voice is not None:
            try:
                has_clar = bool(self.coding_voice.has_pending_clarification())
            except Exception as e:
                logger.debug(
                    "has_pending_clarification check failed (treating as False): %s",
                    e,
                )
        if not is_conversational_ack_eligible(
            user_text, has_pending_clarification=has_clar
        ):
            return None
        try:
            return self.conv_ack_source.next_phrase()
        except Exception as e:
            logger.warning("Conversational ack source failed: %s", e)
            return None

    def _kick_off_tts_preopen(self) -> Optional[threading.Thread]:
        """Start the TTS output-stream pre-open on a daemon thread.

        2026-05-15 latency: opening ``sd.OutputStream`` takes ~50 ms
        on Windows (PortAudio + WASAPI handshake). Doing it BEFORE
        Whisper STT (which runs ~80-150 ms) overlaps the cost so by
        the time the LLM yields its first token + ack, the stream
        is ready to write to.

        Fail-open at every level: missing engine method (legacy /
        unit-test fixture), exception in pre-open, or pool failure
        all leave the engine in its pre-existing state -- the live
        ``speak_stream`` path falls back to opening fresh.
        """
        tts = getattr(self, "tts", None)
        if tts is None:
            return None
        prep = getattr(tts, "prepare_output_stream", None)
        if not callable(prep):
            return None
        try:
            t = threading.Thread(
                target=prep, daemon=True, name="tts-stream-preopen",
            )
            t.start()
            return t
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "TTS stream pre-open kickoff failed (%s); live path will "
                "open fresh inside speak_stream.", e,
            )
            return None

    # ----- streaming STT integration (2026-05-22 -------------------------
    # Three helpers the capture loop calls to wire up live partial
    # transcription on engines that support it (Moonshine v2 streaming
    # arches). All helpers are fail-open: any error degrades to the
    # legacy one-shot path, never to silence. Duck-typed against the
    # engine via ``hasattr`` so non-streaming engines are unaffected.

    def _stt_streaming_enabled(self) -> bool:
        """Return True iff the engine supports streaming AND the config
        flag ``stt.moonshine_streaming_capture`` is on. Cached lazily;
        construction-time engine swaps re-check via the lazy import."""
        stt = getattr(self, "stt", None)
        if stt is None:
            return False
        if not getattr(stt, "supports_streaming", lambda: False)():
            return False
        try:
            from kenning.config import get_config
            return bool(getattr(
                get_config().stt, "moonshine_streaming_capture", True,
            ))
        except Exception:                                              # noqa: BLE001
            return True

    def _maybe_start_stt_stream(self) -> bool:
        """Begin a streaming STT session if the engine supports it.

        2026-05-22 update: also spawns a background worker that drains
        a chunk queue and feeds the engine -- the capture thread MUST
        NOT block on Moonshine's mid-stream ``update_transcription``
        calls (50-100 ms on CPU), or sounddevice's input buffer
        overflows and audio is dropped (the silent-turn-1 bug + the
        ``Audio status flag: input overflow`` warnings).

        Returns True iff streaming is now active and the capture loop
        should keep enqueueing chunks.
        """
        if not self._stt_streaming_enabled():
            return False
        try:
            self.stt.start_stream()
        except Exception as e:                                         # noqa: BLE001
            logger.warning(
                "Streaming STT start failed (%s); falling back to one-shot.",
                e,
            )
            return False

        # Background-worker pattern: the capture thread enqueues each
        # ~16 ms mic chunk, the worker drains the queue and calls the
        # engine's feed_audio (which internally triggers the C-side
        # update_transcription every ~200 ms). This keeps the capture
        # thread responsive even when the model is slow on a partial.
        import queue
        self._stt_stream_queue = queue.Queue(maxsize=512)
        self._stt_stream_sentinel = object()
        self._stt_stream_worker_started = True

        sentinel = self._stt_stream_sentinel
        q = self._stt_stream_queue
        stt = self.stt
        sample_rate = settings.SAMPLE_RATE

        def _worker() -> None:
            while True:
                try:
                    item = q.get(timeout=2.0)
                except Exception:
                    continue
                if item is sentinel:
                    return
                try:
                    feed = getattr(stt, "feed_audio", None)
                    if feed is not None:
                        feed(item, sample_rate=sample_rate)
                except Exception as e:                                # noqa: BLE001
                    logger.debug("Streaming STT worker feed failed: %s", e)

        t = threading.Thread(
            target=_worker, daemon=True, name="stt-stream-worker",
        )
        t.start()
        self._stt_stream_worker = t
        return True

    def _maybe_feed_stt_chunk(self, chunk: np.ndarray) -> None:
        """Enqueue a single audio chunk for the background worker.

        Non-blocking: if the queue is somehow full (shouldn't happen
        with a 512-slot buffer at 16 ms chunks = ~8 s of audio), the
        chunk is dropped silently rather than blocking the capture
        thread. The model just sees a brief gap; the buffered audio
        is still returned to the orchestrator for fallback transcribe.
        """
        q = getattr(self, "_stt_stream_queue", None)
        if q is None:
            return
        try:
            q.put_nowait(chunk)
        except Exception as e:                                         # noqa: BLE001
            logger.debug("Streaming STT queue full / put failed: %s", e)

    def _maybe_stop_stt_stream(self) -> Optional[str]:
        """Finalize the streaming session and return the final text.

        The text is also stashed inside the engine (Moonshine
        :attr:`_last_streaming_text`) so a subsequent
        ``self.stt.transcribe(buffer)`` call returns it instantly
        without re-running the model. Returns ``None`` on any error so
        the caller can fall through to the legacy path.

        2026-05-22: also signals the background feed worker to finish
        + joins it before calling ``stop_stream`` on the engine. This
        ensures all queued chunks are actually consumed by the model
        before we ask for the final transcript.
        """
        # Drain + finish the background worker BEFORE stopping the
        # engine -- otherwise we'd race against pending chunks.
        try:
            q = getattr(self, "_stt_stream_queue", None)
            worker = getattr(self, "_stt_stream_worker", None)
            sentinel = getattr(self, "_stt_stream_sentinel", None)
            if q is not None and sentinel is not None:
                try:
                    q.put(sentinel, timeout=2.0)
                except Exception:                                       # noqa: BLE001
                    pass
            if worker is not None and worker.is_alive():
                worker.join(timeout=4.0)
        except Exception as e:                                         # noqa: BLE001
            logger.debug("Streaming STT worker drain failed: %s", e)
        finally:
            self._stt_stream_queue = None
            self._stt_stream_worker = None
            self._stt_stream_sentinel = None

        try:
            stop = getattr(self.stt, "stop_stream", None)
            if stop is None:
                return None
            return stop()
        except Exception as e:                                         # noqa: BLE001
            logger.warning(
                "Streaming STT stop_stream failed: %s -- the main "
                "transcribe(buffer) call below will run fresh.", e,
            )
            return None

    def _maybe_discard_stt_stream(self) -> None:
        """Stop any active streaming-STT session and DISCARD its text.

        Abort-path counterpart of :meth:`_maybe_stop_stt_stream`: used
        when the captured audio itself is being dropped (wake-word
        fired during the follow-up window, or the window deadline
        elapsed mid-utterance), so the engine's stashed final text
        must not leak into the NEXT capture's ``transcribe(buffer)``
        cache-hit path. Fail-open at every layer.
        """
        self._maybe_stop_stt_stream()
        try:
            clear = getattr(
                getattr(self, "stt", None), "clear_stream_cache", None,
            )
            if callable(clear):
                clear()
        except Exception as e:                                         # noqa: BLE001
            logger.debug("STT stream-cache clear failed: %s", e)

    def _reset_speculative_stt_state(self) -> None:
        """Clear any leftover speculative STT state from a prior capture.

        Called at the start of :meth:`_capture_utterance` and
        :meth:`_follow_up_listen` so a stale result from the prior
        turn (e.g. an empty-utterance turn that never called
        :meth:`_collect_speculative_stt`) can't leak into the current
        turn's main-loop transcript.

        If a previous thread is still running, we let it finish in
        the background (its result is discarded by the reset).

        2026-05-18 latency pass 3 (Phase 2): also resets the chained
        speculative-classification slot so a prior turn's cached
        verdict / ack / RAG future doesn't leak into this turn.
        """
        with self._speculative_stt_lock:
            self._speculative_stt_thread = None
            self._speculative_stt_result = None
            self._speculative_stt_invalidated = False
            # Note: leave _speculative_stt_active alone -- a still-
            # running background thread will set it False on exit;
            # forcing it False here could race with the thread's
            # completion path and let two concurrent kick-offs slip
            # through. The kick-off path checks _active to no-op
            # safely on overlap.
        self._reset_speculative_classification_state()

    def _reset_speculative_classification_state(self) -> None:
        """Clear any leftover speculative classification state.

        Called from :meth:`_reset_speculative_stt_state` so the slots
        stay in lockstep. Best-effort: the existing classification's
        RAG future (if any) is canceled so the rolled-over pool doesn't
        keep retrieving for a turn that no longer cares. Defensive
        against partial test fixtures that bypass :meth:`__init__` and
        only set the STT slot.

        Phase 3 extension: also chains to the speculative LLM reset so
        all three speculation lanes (STT / classification / LLM) clear
        atomically at the top of each capture.
        """
        lock = getattr(self, "_speculative_classification_lock", None)
        if lock is None:
            # Still reset LLM if its slot is set (test fixtures wire
            # them independently).
            self._reset_speculative_llm_state()
            return
        with lock:
            state = getattr(self, "_speculative_classification", None)
            self._speculative_classification = None
            self._speculative_classification_invalidated = False
        if state is not None:
            rag_future = state.get("rag_future")
            if rag_future is not None:
                try:
                    rag_future.cancel()
                except Exception:
                    pass
        # Phase 3: chain to LLM reset.
        self._reset_speculative_llm_state()

    def _kick_off_speculative_stt(self, audio: np.ndarray) -> None:
        """Start Whisper STT in a background thread on the captured audio.

        2026-05-16 latency pass 2: when VAD has accumulated a brief run
        of consecutive silence frames after speech (~32 ms at the new
        16 ms blocksize), we have strong evidence the user has stopped
        speaking. We kick off Whisper transcription on the audio
        captured so far on a background daemon thread. By the time the
        full fast-path silence baseline elapses (~300 ms with Phase 3),
        Whisper (~78 ms) has finished and its result is consumable via
        :meth:`_collect_speculative_stt`. Net win: the foreground
        Whisper time disappears from the critical path.

        Idempotent: re-calling while an inference is already in flight
        is a no-op. Fail-open: thread-launch failures or transcription
        errors leave the speculative state empty; the caller falls back
        to the foreground STT path.

        2026-05-22 streaming-STT skip: when the active engine supports
        streaming AND streaming capture is enabled, this method is a
        no-op. The streaming engine's own listener already maintains a
        live partial transcript; kicking off ANOTHER snapshot
        transcribe would race the streaming session, read an empty
        partial too early, and cache that empty string as the
        speculative result -- which the main loop would then treat as
        the final transcript (silently dropping the turn). Skipping
        this path on streaming engines means the main loop instead
        reads the engine's stashed final text via
        ``self.stt.transcribe(speech)`` after capture ends.

        Args:
            audio: Float32 PCM at 16 kHz. The audio buffer accumulated
                so far. Whisper sees this snapshot; later silence
                appended to the live capture does not change the
                transcript (silence is silence).
        """
        # 2026-05-22: streaming-STT race guard. The streaming engine
        # itself produces partials -- speculative STT is both
        # redundant and harmful here (it would race the streaming
        # session and cache an empty partial).
        if self._stt_streaming_enabled():
            return
        with self._speculative_stt_lock:
            if self._speculative_stt_active:
                return
            self._speculative_stt_active = True
            self._speculative_stt_result = None
            self._speculative_stt_invalidated = False
        # Copy the audio so the background thread doesn't race with
        # the live capture's growing chunk list.
        audio_copy = audio.copy() if audio is not None else None

        def _run() -> None:
            try:
                text = self.stt.transcribe(audio_copy)
            except Exception as e:                                  # noqa: BLE001
                logger.warning("Speculative STT inference failed: %s", e)
                text = None
            with self._speculative_stt_lock:
                self._speculative_stt_result = text
                self._speculative_stt_active = False
                stt_invalidated = self._speculative_stt_invalidated
            # 2026-05-18 latency pass 3 (Phase 2): chain speculative
            # classification work onto this thread. Cheap (rule-path
            # gate + ack pick + RAG kick-off, ~5-10 ms total) so we
            # collapse it into the STT thread rather than spinning a
            # second one. Skipped on STT failure / invalidate / empty
            # transcript -- the main loop falls back to fresh work.
            if (
                text is not None
                and text.strip()
                and not stt_invalidated
            ):
                self._run_speculative_classification(text)

        try:
            t = threading.Thread(
                target=_run, daemon=True, name="speculative-stt",
            )
            t.start()
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "Speculative STT thread launch failed (%s); live path "
                "will run STT in the foreground.", e,
            )
            with self._speculative_stt_lock:
                self._speculative_stt_active = False
            return
        self._speculative_stt_thread = t

    def _run_speculative_classification(self, user_text: str) -> None:
        """Compute the rule-path gate verdict + ack phrase + RAG future
        for ``user_text`` and stash them in ``_speculative_classification``.

        2026-05-18 latency pass 3 (Phase 2). Runs synchronously on the
        speculative-STT thread (chained from :meth:`_kick_off_speculative_stt`'s
        ``_run`` after STT completes successfully). Stays on the same
        thread so we don't spin a second daemon; cumulative work is
        only ~5-10 ms (gate rule classifier + ack-pool next pick +
        thread-pool submit for RAG).

        Skips the LLM-preflight branch of the web gate: the speculative
        path uses ``classify_by_rules`` directly, which returns ``None``
        on UNCERTAIN. The main loop's fresh :meth:`web_gate.classify`
        call will run preflight when needed -- speculation only saves
        time on the rule-determined fast path, which is the common
        case for short conversational queries.

        Fail-open at every stage: any exception leaves the slot empty
        and the main loop falls back to the legacy fresh path. Defensive
        against partial test fixtures: bails when the classification
        lock isn't set up.
        """
        lock = getattr(self, "_speculative_classification_lock", None)
        if lock is None:
            return
        # Check whether we were invalidated between STT result storage
        # and now. If so, skip the work.
        with lock:
            if self._speculative_classification_invalidated:
                return

        verdict = None
        try:
            web_gate = getattr(self, "web_gate", None)
            if web_gate is not None:
                from kenning.web_search.gating import classify_by_rules
                verdict = classify_by_rules(user_text)
        except Exception as e:                                       # noqa: BLE001
            logger.debug(
                "Speculative gate (rule) failed: %s -- main loop will "
                "classify fresh.", e,
            )
            verdict = None

        ack_phrase = None
        try:
            ack_phrase = self._maybe_conversational_ack(user_text)
        except Exception as e:                                       # noqa: BLE001
            logger.debug(
                "Speculative ack pick failed: %s -- main loop will "
                "compute fresh.", e,
            )
            ack_phrase = None

        rag_future = None
        try:
            rag_future, _used_async = self._kick_off_rag_prefetch(user_text)
        except Exception as e:                                       # noqa: BLE001
            logger.debug(
                "Speculative RAG pre-fetch failed: %s -- main loop will "
                "retrieve serially.", e,
            )
            rag_future = None

        # Re-check invalidation before storing -- the user may have
        # resumed speaking while we ran the classification work.
        with lock:
            if self._speculative_classification_invalidated:
                if rag_future is not None:
                    try:
                        rag_future.cancel()
                    except Exception:
                        pass
                return
            self._speculative_classification = {
                "text": user_text,
                "gate_verdict": verdict,
                "ack_phrase": ack_phrase,
                "rag_future": rag_future,
            }

        # 2026-05-18 latency pass 3 (Phase 3): chain speculative LLM
        # generation when the rule-path verdict resolves to NO_SEARCH.
        # SEARCH path is skipped because the search-augmented prompt
        # body differs. UNCERTAIN (verdict==None) is skipped because
        # the main path will run the LLM preflight; speculating
        # would race / cost double.
        try:
            from kenning.web_search import GateDecision
            should_spec_llm = (
                verdict is not None
                and getattr(verdict, "decision", None) == GateDecision.NO_SEARCH
            )
        except Exception:                                            # noqa: BLE001
            should_spec_llm = False
        if should_spec_llm:
            try:
                self._kick_off_speculative_llm(
                    user_text, verdict, rag_future,
                )
            except Exception as e:                                   # noqa: BLE001
                logger.debug(
                    "Speculative LLM kickoff failed: %s -- main loop "
                    "will run fresh.", e,
                )

    def swap_stt_engine(self, name: str) -> bool:
        """Runtime swap between the primary and gaming STT engines.

        Returns True if the swap took effect, False if the requested
        engine isn't loaded (e.g., dual-STT not configured or the
        gaming engine failed to load at startup). Invalidates any in-
        flight speculative STT so a swap mid-utterance doesn't leak
        stale results.

        Args:
            name: Either the primary engine name (e.g., "parakeet") or
                the gaming engine name (e.g., "moonshine"). Unknown
                names log WARN and leave ``self.stt`` unchanged.
        """
        registry = getattr(self, "_stt_registry", None)
        if registry is None:
            logger.warning("swap_stt_engine: registry not initialised")
            return False
        if name == registry.active_name:
            return True
        # Invalidate before swapping so a transcript landing late from
        # the old engine doesn't get attributed to the new one.
        try:
            self._invalidate_speculative_stt()
        except Exception as e:                                # noqa: BLE001
            logger.debug("swap_stt_engine: invalidate skipped (%s)", e)
        prior_name = registry.active_name
        new_engine = registry.swap_to(name)
        if registry.active_name == name:
            self.stt = new_engine
            logger.info(
                "STT engine swapped %s -> %s", prior_name, name,
            )
            return True
        logger.warning(
            "swap_stt_engine(%r): swap declined; staying on %s",
            name, registry.active_name,
        )
        return False

    def _invalidate_speculative_stt(self) -> None:
        """Mark any in-flight speculative STT result as invalid.

        Called when VAD reports SPEECH_START after a speculative
        inference has been kicked off -- meaning the user resumed
        speaking before SPEECH_END, so the speculative audio buffer
        is now a STALE prefix of the real utterance. The thread
        continues running (we don't try to cancel CTranslate2; the
        wasted CPU is fine on a background daemon), but its result
        is discarded on collection.

        2026-05-18 latency pass 3 (Phase 2): also invalidates the
        chained classification slot. The STT thread checks
        ``_speculative_classification_invalidated`` before storing
        the classification result so a late-arriving invalidation
        still wins the race.
        """
        with self._speculative_stt_lock:
            self._speculative_stt_invalidated = True
        self._invalidate_speculative_classification()

    def _invalidate_speculative_classification(self) -> None:
        """Mark the speculative classification slot as invalid.

        2026-05-18 latency pass 3 (Phase 2). Best-effort: cancels the
        in-flight RAG future and stamps the invalidated flag so a
        later-arriving STT thread that hasn't yet stored its
        classification result drops it instead. Idempotent. Defensive
        against partial test fixtures.

        Phase 3 extension: also invalidates the speculative LLM slot
        so the three speculation lanes stay in lockstep on
        SPEECH_START.
        """
        lock = getattr(self, "_speculative_classification_lock", None)
        if lock is None:
            return
        with lock:
            state = getattr(self, "_speculative_classification", None)
            self._speculative_classification_invalidated = True
        if state is not None:
            rag_future = state.get("rag_future")
            if rag_future is not None:
                try:
                    rag_future.cancel()
                except Exception:
                    pass
        # Phase 3: chain to LLM invalidation.
        self._invalidate_speculative_llm()

    def _collect_speculative_classification(
        self, user_text: str,
    ) -> Optional[dict]:
        """Return the cached classification for ``user_text``, or None.

        2026-05-18 latency pass 3 (Phase 2). Returns the state dict
        with keys ``text``, ``gate_verdict``, ``ack_phrase``,
        ``rag_future`` when:

        * Classification was stored under this exact transcript.
        * The slot was not invalidated.

        Always clears the slot atomically so the caller takes
        ownership of the RAG future (preventing double-collect by a
        stale call). Returns None on miss; main loop falls back to
        the legacy fresh-classification path. Defensive against
        partial test fixtures.
        """
        lock = getattr(self, "_speculative_classification_lock", None)
        if lock is None:
            return None
        with lock:
            was_invalidated = self._speculative_classification_invalidated
            state = self._speculative_classification
            self._speculative_classification = None
            self._speculative_classification_invalidated = False
        if state is None:
            return None
        if was_invalidated or state.get("text") != user_text:
            # Invalidated mid-flight OR stale result from a prior turn:
            # cancel RAG and return None so the main loop falls back to
            # the legacy fresh path.
            rag_future = state.get("rag_future")
            if rag_future is not None:
                try:
                    rag_future.cancel()
                except Exception:
                    pass
            return None
        return state

    # ----- 2026-05-18 latency pass 3 (Phase 3): speculative LLM ---------

    def _kick_off_speculative_llm(
        self, user_text: str, verdict, rag_future,
    ) -> None:
        """Start LLM generation on a background daemon thread.

        Called from :meth:`_run_speculative_classification` after the
        rule-path gate verdict resolves to NO_SEARCH. The thread:

        1. Applies :func:`apply_uncertainty` to the verdict; aborts
           if the uncertainty layer upgrades NO_SEARCH -> SEARCH
           (the search-augmented prompt body is different).
        2. Applies :func:`apply_brevity_hint` to the augmented text.
        3. Resolves the RAG future (joining the in-flight retrieval).
        4. Calls :meth:`LLMEngine.generate_stream` with
           ``record_history=False`` so the speculation doesn't pollute
           history if invalidated. Buffers each token into a
           ``queue.Queue``. Accumulates the full response so the
           consumer can record history explicitly via
           :meth:`LLMEngine.record_completed_turn`.

        On SPEECH_START during silence wait, the orchestrator calls
        :meth:`_invalidate_speculative_llm` which sets the invalidated
        flag and signals :meth:`LLMEngine.cancel`. The iteration loop
        exits cleanly; the buffer gets a sentinel; no history record.

        Fail-open at every level: missing LLM / verdict / state /
        exceptions all leave the speculation silently inactive and
        the main loop falls back to the legacy fresh-call path.

        Args:
            user_text: The transcript the speculation is keyed on.
                Must match the main path's user_text for the
                speculation to be consumed.
            verdict: The gate verdict returned by
                ``classify_by_rules``. Must have ``decision==NO_SEARCH``.
            rag_future: The pre-fetched RAG retrieval future (from
                :meth:`_kick_off_rag_prefetch`). May be ``None``.
        """
        # Defensive against partial fixtures.
        lock = getattr(self, "_speculative_llm_lock", None)
        llm = getattr(self, "llm", None)
        if lock is None or llm is None or verdict is None:
            return
        with lock:
            if self._speculative_llm_active:
                return
            try:
                import queue as _queue
                self._speculative_llm_buffer = _queue.Queue()
            except Exception:
                return
            self._speculative_llm_active = True
            self._speculative_llm_text = user_text
            self._speculative_llm_response = None
            self._speculative_llm_completed = False
            self._speculative_llm_invalidated = False
        buffer = self._speculative_llm_buffer

        def _run() -> None:
            # Local imports keep cold-start cheap on test fixtures
            # that never construct a real orchestrator.
            from kenning.uncertainty import apply as apply_uncertainty
            from kenning.response_style import apply_brevity_hint
            from kenning.web_search import GateDecision

            accumulated: list = []
            completed = False
            try:
                # apply_uncertainty may upgrade NO_SEARCH -> SEARCH on
                # low-confidence + temporal patterns. The search path
                # uses a different prompt body, so abort speculation
                # in that branch.
                final_verdict, augmented = apply_uncertainty(
                    verdict, user_text,
                )
                if final_verdict.decision != GateDecision.NO_SEARCH:
                    return
                augmented = apply_brevity_hint(augmented)
                snippets = self._collect_rag_future(rag_future)
                stream = llm.generate_stream(
                    augmented,
                    gate_verdict=final_verdict,
                    precomputed_rag_snippets=snippets,
                    record_history=False,
                )
                for token in stream:
                    # Check invalidation before pushing each token.
                    # If invalidated, cancel the underlying stream so
                    # the iterator exits on its next chunk.
                    with lock:
                        if self._speculative_llm_invalidated:
                            try:
                                llm.cancel()
                            except Exception:
                                pass
                            return
                    accumulated.append(token)
                    buffer.put(token)
                completed = True
            except Exception as e:                                   # noqa: BLE001
                logger.warning(
                    "Speculative LLM failed (%s); main path will run "
                    "fresh.", e,
                )
            finally:
                # Sentinel always emitted so consumers don't hang.
                try:
                    buffer.put(None)
                except Exception:
                    pass
                with lock:
                    self._speculative_llm_response = "".join(accumulated)
                    self._speculative_llm_completed = completed
                    self._speculative_llm_active = False

        try:
            t = threading.Thread(
                target=_run, daemon=True, name="speculative-llm",
            )
            t.start()
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "Speculative LLM thread launch failed (%s); main path "
                "will run fresh.", e,
            )
            with lock:
                self._speculative_llm_active = False
            return
        self._speculative_llm_thread = t

    def _invalidate_speculative_llm(self) -> None:
        """Mark any in-flight speculative LLM as invalid.

        Sets the invalidated flag and signals :meth:`LLMEngine.cancel`
        so the streaming iterator exits at its next chunk. The
        speculation thread's ``finally`` block still emits the
        sentinel, so consumers waiting on the buffer don't hang.

        Idempotent / defensive against partial fixtures.
        """
        lock = getattr(self, "_speculative_llm_lock", None)
        if lock is None:
            return
        with lock:
            self._speculative_llm_invalidated = True
        llm = getattr(self, "llm", None)
        if llm is not None:
            try:
                llm.cancel()
            except Exception:
                pass

    def _collect_speculative_llm(self, user_text: str):
        """Return ``(iter, response_committer)`` for the speculation, or
        ``(None, None)`` on miss.

        On hit, the iterator yields tokens from the buffer until the
        sentinel arrives. The committer is a zero-arg function the
        caller invokes after consuming the iterator successfully -- it
        records the turn in history (so an unconsumed speculation
        leaves no orphan). On invalidate, the iterator exits early
        and the committer is a no-op.

        Returns ``(None, None)`` when:
        * Speculation was never started.
        * Speculation is keyed on a different transcript (stale).
        * Speculation was invalidated.

        The slot is cleared atomically on every call so a second
        attempt returns ``(None, None)``.

        2026-05-18 latency pass 3 (Phase 3). Defensive against
        partial fixtures.
        """
        lock = getattr(self, "_speculative_llm_lock", None)
        if lock is None:
            return None, None
        with lock:
            was_invalidated = self._speculative_llm_invalidated
            spec_text = self._speculative_llm_text
            buffer = self._speculative_llm_buffer
            thread = self._speculative_llm_thread
            # Clear slot atomically so a stale re-collect returns None.
            self._speculative_llm_text = None
            self._speculative_llm_buffer = None
            self._speculative_llm_thread = None
            self._speculative_llm_invalidated = False
        if was_invalidated or buffer is None or spec_text != user_text:
            # On mismatch / invalidate, drain the buffer (best-effort
            # so the producer thread doesn't pile up). Don't yield.
            return None, None

        def _drain():
            # Yield tokens until the sentinel. ``timeout`` is generous
            # because the producer might be still generating; we exit
            # promptly on EOF.
            while True:
                try:
                    token = buffer.get(timeout=15.0)
                except Exception:
                    return
                if token is None:
                    return
                yield token

        def _commit_history():
            """Record the consumed turn in history. No-op if the
            speculation didn't complete cleanly (e.g., cancel)."""
            if thread is not None and thread.is_alive():
                # Producer hasn't finished -- wait briefly so the
                # response/completed fields are populated.
                thread.join(timeout=1.0)
            with lock:
                response = self._speculative_llm_response
                completed = self._speculative_llm_completed
                self._speculative_llm_response = None
                self._speculative_llm_completed = False
            llm = getattr(self, "llm", None)
            if llm is None or not completed:
                return
            if not response or not response.strip():
                return
            try:
                llm.record_completed_turn(spec_text, response)
            except Exception as e:                                   # noqa: BLE001
                logger.warning(
                    "Speculative-LLM history record failed: %s", e,
                )

        return _drain(), _commit_history

    def _reset_speculative_llm_state(self) -> None:
        """Clear leftover speculative LLM state from a prior capture.

        Called from :meth:`_reset_speculative_stt_state` (via the
        classification reset) so the three speculation slots stay in
        lockstep. Best-effort: if a previous speculation thread is
        still running, we set the invalidated flag and cancel the
        LLM stream; the thread's ``finally`` block cleans up its
        own state.
        """
        lock = getattr(self, "_speculative_llm_lock", None)
        if lock is None:
            return
        with lock:
            in_flight = self._speculative_llm_active
            self._speculative_llm_text = None
            self._speculative_llm_buffer = None
            self._speculative_llm_thread = None
            self._speculative_llm_response = None
            self._speculative_llm_completed = False
            self._speculative_llm_invalidated = False
        if in_flight:
            llm = getattr(self, "llm", None)
            if llm is not None:
                try:
                    llm.cancel()
                except Exception:
                    pass

    def _collect_speculative_stt(
        self, *, timeout_s: float = 2.0,
    ) -> Optional[str]:
        """Wait for the in-flight speculative STT (if any) and return
        its result, or None when invalidated / failed / never kicked
        off / still running past ``timeout_s``.

        Always cleans up state so the next capture starts with a
        fresh speculative slot. Idempotent: a second call returns
        None.

        Args:
            timeout_s: Maximum time to wait for the background thread
                to finish. The expected Whisper runtime is ~78 ms;
                anything beyond 2 s indicates a CUDA stall or a hung
                transcription -- in which case we discard and let
                the caller run STT in the foreground.

        Returns:
            The transcript on success, or None on any failure /
            invalidation / timeout / not-started case.
        """
        thread = self._speculative_stt_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_s)))
        with self._speculative_stt_lock:
            if self._speculative_stt_invalidated:
                result: Optional[str] = None
            else:
                result = self._speculative_stt_result
            # Reset for the next capture even on early return.
            self._speculative_stt_thread = None
            self._speculative_stt_result = None
            self._speculative_stt_active = False
            self._speculative_stt_invalidated = False
        return result

    def _kick_off_rag_prefetch(self, user_text: str):
        """Start Qdrant RAG retrieval on a background thread.

        Returns ``(future, used_async)``. ``used_async`` is False when
        the multi-pass retrieval flag is on (in which case the LLM
        needs the gate_verdict's category list to drive its fan-out,
        so we can't pre-fetch single-pass) -- in that branch the
        future is ``None`` and the caller falls back to in-line
        retrieval inside the LLM call.

        2026-05-15 latency: pre-fetch overlaps the ~30-50 ms Qdrant
        round-trip with the ~5-150 ms web-gate classification. On
        rule-based gate turns (the common case) this is the bigger
        win because the gate finishes in microseconds and the LLM
        would otherwise pay the RAG cost serially.

        Defensive ``getattr`` reads on ``self.memory`` / ``self.llm``
        keep this method usable in unit-test fixtures that bypass
        :meth:`__init__` and only set the attributes their tests
        actually touch.
        """
        memory = getattr(self, "memory", None)
        llm = getattr(self, "llm", None)
        if memory is None or llm is None:
            return None, False
        retrieve_fn = getattr(llm, "retrieve_rag_snippets", None)
        if retrieve_fn is None:
            return None, False
        try:
            from kenning.config import get_config
            mem_cfg = get_config().memory
            multi_pass = bool(
                getattr(mem_cfg.retrieval, "multi_pass_enabled", False)
            )
        except Exception:
            multi_pass = False
        if multi_pass:
            # The multi-pass path keys off ``gate_verdict.context_categories``
            # which the LLM preflight populates inside ``web_gate.classify``.
            # Pre-fetching now (without a verdict) would silently downgrade
            # to single-pass. Skip the pre-fetch; LLM handles it serially.
            return None, False
        try:
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rag-prefetch",
            )
            future = pool.submit(retrieve_fn, user_text)
            # Daemon-shutdown semantics: we shut the pool down WITHOUT
            # waiting so a slow Qdrant doesn't pin the orchestrator.
            pool.shutdown(wait=False)
            return future, True
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "RAG pre-fetch kickoff failed (%s); LLM will retrieve serially.",
                e,
            )
            return None, False

    @staticmethod
    def _collect_rag_future(
        future, *, deadline_s: float = 5.0,
    ) -> Optional[list]:
        """Best-effort join on the RAG pre-fetch future.

        Returns the snippet list on success, ``None`` on timeout /
        exception (caller falls back to in-line retrieval inside the
        LLM call). The deadline is generous because the only failure
        mode worth blocking on is "Qdrant hung" -- 5 s comfortably
        covers a healthy retrieval (~30-50 ms typical) and lets a
        misbehaving store time out cleanly.
        """
        if future is None:
            return None
        try:
            return future.result(timeout=deadline_s)
        except Exception as e:                                       # noqa: BLE001
            logger.warning(
                "RAG pre-fetch result unavailable (%s); LLM will "
                "retrieve serially.", e,
            )
            return None

    def _build_response_stream(self, user_text: str):
        """Yield tokens for the response, applying the web-search gate.

        Returned generator handles three paths:
          * No gate or NO_SEARCH/UNCERTAIN -> filler-ack (if gate
            permits), then base ``llm.generate_stream``.
          * SEARCH with successful retrieval -> ack phrase, then a
            search-augmented prompt.
          * SEARCH with empty/failed retrieval -> ack phrase, then base
            generation (LLM at least apologizes accurately).

        2026-05-15 latency: kicks off RAG retrieval on a background
        thread BEFORE the web-gate call so the two costs overlap. The
        precomputed snippets are passed to ``generate_stream`` so the
        LLM doesn't pay the retrieval cost serially. Falls back to
        in-line retrieval when memory is disabled or multi-pass is on
        (the multi-pass path needs the verdict).

        2026-05-18 latency pass 3 (Phase 2): consume the speculative
        classification slot if it was populated during silence wait.
        On hit, the rule-path gate verdict + RAG future are reused
        instead of being recomputed -- saving the ~5 ms rule classify
        AND giving the RAG retrieval ~200-300 ms more overlap. On miss
        (slot empty / invalidated / verdict UNCERTAIN at speculation
        time), falls through to the legacy fresh-kick-off path.

        2026-05-19 round 4: bare "what time is it" / "what day is
        today" asks short-circuit to a local-clock reply (no gate, no
        LLM, no search). The computer has a clock; consulting NIST
        for the wall-clock time is absurd. Mixed-intent or richer
        time-related queries fall through to the LLM path.
        """
        from kenning import trace
        # 2026-05-19 round 4: local clock / date short-circuit.
        # The detector is strict -- only fires on bare time/date asks.
        try:
            from kenning.local_clock_reply import maybe_local_clock_reply
            clock_reply = maybe_local_clock_reply(user_text)
        except Exception as e:                                    # noqa: BLE001
            logger.debug("local clock reply check failed (%s)", e)
            clock_reply = None
        if clock_reply:
            trace.tlog(
                logger, "build_response:local_clock_short_circuit",
                user_text=user_text[:80], reply=clock_reply,
            )
            # Commit to memory as a normal assistant turn so follow-ups
            # ("how about the time in London?") see the prior context.
            try:
                if self.llm is not None:
                    self.llm.record_completed_turn(user_text, clock_reply)
            except Exception as e:                                # noqa: BLE001
                logger.debug("record clock-reply turn failed (%s)", e)
            yield clock_reply
            return

        # 2026-05-18 latency pass 3 (Phase 2): consume cached
        # speculative classification when available. The slot is
        # cleared atomically so the next turn starts fresh.
        spec_class = self._collect_speculative_classification(user_text)
        if spec_class is not None:
            rag_future = spec_class.get("rag_future")
            cached_verdict = spec_class.get("gate_verdict")
            trace.tlog(
                logger, "speculation:classification_hit",
                has_cached_verdict=cached_verdict is not None,
                has_rag_future=rag_future is not None,
            )
        else:
            # Kick off the RAG pre-fetch first so it overlaps everything
            # that follows. The future is consumed by the LLM call below
            # (or discarded on the search-augmented branch).
            rag_future, kicked = self._kick_off_rag_prefetch(user_text)
            cached_verdict = None
            trace.tlog(
                logger, "rag:prefetch_kickoff",
                kicked=kicked, has_future=rag_future is not None,
            )

        # 2026-05-22 -- semantic SEARCH override from the intent
        # recognizer. If a "needs fresh data" phrase matched earlier
        # in the turn, force the verdict to SEARCH (overriding any
        # cached preflight). Done AFTER speculation so the rule layer's
        # high-confidence rule verdicts still win when they fired.
        if getattr(self, "_next_turn_force_search", False):
            self._next_turn_force_search = False  # consume
            cached_verdict = GateVerdict(
                GateDecision.SEARCH, "high", "intent_recognizer",
                "freshness-intent matched; overriding preflight verdict",
                has_temporal_dependency=True,
            )
            trace.tlog(
                logger, "gate:intent_force_search",
                decision="SEARCH", source="intent_recognizer",
            )

        if self.web_gate is None or self.web_executor is None:
            trace.tlog(
                logger, "gate:no_web_gate_configured",
                next="conversational_llm",
            )
            ack = self._maybe_conversational_ack(user_text)
            if ack:
                trace.tlog(logger, "ack:conversational", text=ack)
                yield ack + " "
            snippets = self._collect_rag_future(rag_future)
            trace.tlog(
                logger, "rag:collected",
                snippet_count=len(snippets) if snippets else 0,
            )
            # 2026-05-20 round 7: store the BARE user_text in memory
            # (not the brevity-hinted prompt); see history_user_message
            # docstring on LLMEngine.generate_stream for the
            # contamination-loop rationale.
            yield from self.llm.generate_stream(
                apply_brevity_hint(user_text),
                precomputed_rag_snippets=snippets,
                history_user_message=user_text,
                # 2026-05-22 perf fix: retrieve against the BARE
                # user_text, not the brevity-hinted body. The hint
                # prefix bloats the query and slows cross-encoder
                # reranking.
                rag_query=user_text,
                # 2026-05-22 latency fix: voice path doesn't have the
                # budget for Qwen3.5's <think> reasoning block (5-10 s
                # of TTFT on math / factual questions). The
                # ``/no_think`` user-message marker keeps the model
                # producing visible output directly.
                enable_thinking=False,
            )
            return

        # If speculation already pinned the rule-path verdict, skip the
        # fresh classify call. The rule layer is deterministic on the
        # transcript so the speculative + main verdicts agree by
        # construction. ``cached_verdict is None`` covers both "no
        # speculation" and "speculation hit UNCERTAIN" -- in either
        # case we run the full classify (which may trigger LLM
        # preflight on the UNCERTAIN branch, paid serially as before).
        if cached_verdict is not None:
            verdict = cached_verdict
            trace.tlog(
                logger, "gate:cached_verdict_used",
                decision=verdict.decision.value, source=verdict.source,
            )
        elif self._barebones_skip_web_search():
            # Bare-bones gaming mode: skip the web-search preflight (an LLM
            # classification call) + executor entirely. Force NO_SEARCH so the
            # turn is a plain STT->LLM->TTS reply with no GPU/compute for search.
            verdict = GateVerdict(
                GateDecision.NO_SEARCH, "high", "gaming_mode",
                "gaming mode: web search skipped",
            )
        else:
            try:
                trace.set_phase("gate")
                t0 = time.monotonic()
                verdict = self.web_gate.classify(user_text)
                trace.tlog(
                    logger, "gate:classify_complete",
                    decision=verdict.decision.value,
                    confidence=verdict.confidence,
                    source=verdict.source,
                    reason=verdict.reason,
                    elapsed_ms=int((time.monotonic() - t0) * 1000),
                )
            except Exception as e:
                logger.warning("Web gate failed (%s) -- falling through to base", e)
                trace.tlog(logger, "gate:failure", error=str(e))
                ack = self._maybe_conversational_ack(user_text)
                if ack:
                    yield ack + " "
                snippets = self._collect_rag_future(rag_future)
                # Round 7: bare user_text recorded in memory.
                yield from self.llm.generate_stream(
                    apply_brevity_hint(user_text),
                    precomputed_rag_snippets=snippets,
                    history_user_message=user_text,
                    rag_query=user_text,           # 2026-05-22 perf
                    enable_thinking=False,         # 2026-05-22 latency
                )
                return

        # Phase 5: translate preflight uncertainty signals into behavior.
        # May upgrade NO_SEARCH -> SEARCH (low confidence + temporal), and
        # may prepend a short [Confidence: ...] addendum to the user text
        # so the LLM matches its tone to the actual confidence level.
        verdict_before = verdict.decision.value
        verdict, augmented_text = apply_uncertainty(verdict, user_text)
        if verdict.decision.value != verdict_before:
            trace.tlog(
                logger, "gate:uncertainty_upgrade",
                from_decision=verdict_before,
                to_decision=verdict.decision.value,
            )

        logger.info(
            "gate: %s (%s, %s) -- %s",
            verdict.decision.value, verdict.source, verdict.confidence,
            verdict.reason,
        )
        if verdict.decision != GateDecision.SEARCH:
            # 2026-05-12 filler-ack on conversational path (NO_SEARCH /
            # UNCERTAIN). Yields before brevity-hinted prompt + LLM
            # stream so the TTS pipeline starts speaking the ack
            # within ~200 ms of Whisper completing -- masks the
            # ~2.5 s perceived gap before the LLM's first token
            # synthesises. Gated against pending coding-clarifications
            # and short utterances so it doesn't fire on interjections.
            ack = self._maybe_conversational_ack(user_text)
            if ack:
                yield ack + " "

            # 2026-05-18 latency pass 3 (Phase 3): try to consume the
            # speculative LLM stream first. When speculation fired
            # during silence wait and finished or is still running with
            # buffered tokens, this saves the entire LLM TTFT (~63 ms)
            # plus partial decode time. On miss / invalidation, falls
            # through to the legacy fresh call.
            spec_iter, commit_history = self._collect_speculative_llm(
                user_text,
            )
            if spec_iter is not None:
                # 2026-05-22: count yields. When the speculative producer
                # thread crashes before emitting any token (e.g. a bug in
                # llama-cpp-python's PLD path), it still drops the sentinel
                # in ``finally``, so this iterator returns immediately with
                # zero yields. Without this fallback, the user got silence
                # because the warning "main path will run fresh" was never
                # actually wired to a fresh call. Now: if the spec produced
                # nothing, fall through to the legacy fresh-call path
                # below.
                yielded_any = False
                try:
                    for tok in spec_iter:
                        yielded_any = True
                        yield tok
                finally:
                    if commit_history is not None:
                        try:
                            commit_history()
                        except Exception as e:                       # noqa: BLE001
                            logger.warning(
                                "Speculative LLM history commit failed: %s",
                                e,
                            )
                if yielded_any:
                    return
                logger.warning(
                    "Speculative LLM yielded 0 tokens; running fresh "
                    "main-path LLM call so the turn isn't silent.",
                )

            # 2026-05-10 brevity reinforcement: prepend a 1-3-sentence
            # directive when the user's question is brief and isn't an
            # explicit ask for depth. Counters the 4B model's habit of
            # producing 4-paragraph essays in response to "What are
            # the Orcs in 40k?". The search path already carries its
            # own length directive in the augmented prompt so this is
            # only applied off the search branch. Pure-text addendum;
            # no SOUL.md / persona changes (voice-quality lock).
            augmented_text = apply_brevity_hint(augmented_text)
            # V1-gap A2: thread the verdict through so multi-pass
            # retrieval activates (when configured + categories present).
            snippets = self._collect_rag_future(rag_future)
            # 2026-05-20 round 7: store the BARE user_text in memory.
            # The augmented_text carries '[Confidence: ...]' / brevity-
            # hint markers that should NOT be persisted as the user's
            # turn -- doing so makes RAG retrieve them later as
            # "relevant earlier context", producing a contamination
            # loop. The LLM still receives the augmented text for
            # grounding.
            yield from self.llm.generate_stream(
                augmented_text,
                gate_verdict=verdict,
                precomputed_rag_snippets=snippets,
                history_user_message=user_text,
                # 2026-05-22 perf fix: retrieve uses the BARE user_text
                # (typically 10-50 chars) instead of the augmented body
                # (which carries brevity / confidence markers and is
                # 200+ chars). Drops cross-encoder reranking from
                # ~5-30 s to ~1-3 s on CPU.
                rag_query=user_text,
                # 2026-05-22 latency: skip Qwen3.5's <think> block on
                # the voice path so factual/math questions don't pay
                # 5-10 s of internal-reasoning TTFT.
                enable_thinking=False,
            )
            return

        # SEARCH branch: the search-augmented prompt sets the LLM up
        # with self-contained context (Brave + Jina sources). Drop the
        # pre-fetched RAG -- ``_search_augmented_tokens`` uses
        # ``suppress_memory_context=True`` implicitly by routing
        # through a different prompt body, and unrelated past chatter
        # would contaminate the search-only answer.
        if rag_future is not None:
            try:
                rag_future.cancel()
            except Exception:
                pass
        # 2026-05-20 round 7: pass the bare user_text through so the
        # search-augmented branch can record it in memory (not the
        # multi-thousand-char augmented prompt which was the
        # contamination root cause in the live 2026-05-20 session).
        yield from self._search_augmented_tokens(
            augmented_text, verdict, bare_user_text=user_text,
        )

    def _search_augmented_tokens(
        self,
        user_text: str,
        verdict,
        bare_user_text: Optional[str] = None,
    ):
        """Yield ack phrase + search-augmented LLM tokens.

        ``user_text`` is the augmented prompt body passed to the LLM
        (carries brevity / confidence markers + search instructions).
        ``bare_user_text`` is the original user utterance -- when
        provided, it is what gets persisted to conversation memory
        via ``history_user_message``. This prevents the multi-thousand-
        char augmented prompt body (with "[Confidence: ...]" markers
        and search-result instructions) from being stored and then
        retrieved by RAG as "relevant earlier context" -- the
        contamination loop observed in the 2026-05-20 live session.

        Order of operations (2026-05-09 refinement):
          1. Yield ack token FIRST so the TTS pipeline starts speaking
             "Verifying against the network." (or similar) immediately.
             User gets audible feedback before any network call leaves
             the box.
          2. Submit the Brave + Jina workflow to a worker thread.
          3. Wait for search to return.
          4. Yield LLM tokens that answer the user from the search
             sources. Conversational memory IS available to the LLM
             via the smart-retrieval path (cosine threshold +
             recency-weighted composite scoring in
             :meth:`ConversationMemory.retrieve`), so relevant prior
             context flows in (e.g. a follow-up troubleshooting query
             gets the original troubleshooting context) while
             unrelated past chatter is filtered out by the relevance
             threshold.

        The augmented prompt explicitly tells the LLM to use prior
        context only if it relates to THIS specific question --
        defence in depth on top of the relevance filter.
        """
        from concurrent.futures import ThreadPoolExecutor

        ack_phrase = self.ack_source.next_phrase() if self.ack_source else "Searching."
        # The ack ends with a period so the TTS pipeline flushes it as a
        # complete sentence immediately. We add a trailing space so it
        # blends into the streamed answer without an awkward double-period.
        ack = ack_phrase + " "

        # Yield the ack to the consumer BEFORE we submit the search job
        # so the user perceives feedback first, then network activity.
        # The ack travels through the TTS pipeline while we start search
        # on a worker thread; both are concurrent from the user's POV.
        yield ack

        # 2026-05-22 news-category routing: when the user asked a
        # news-style question, route the SearxNG query to news engines
        # (Bing News / Yahoo News / Reuters) instead of generic web
        # search. Without this, "what's the latest news" matched
        # Whatnot.com + Collins Dictionary because Bing-general ranks
        # on the word "what" first. Brave / DDG ignore the kwarg.
        try:
            from kenning.web_search.gating import _NEWS_QUERIES
            search_categories = "news" if _NEWS_QUERIES.search(user_text) else None
        except Exception:                                            # noqa: BLE001
            search_categories = None

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-search")
        try:
            search_future = pool.submit(
                self.web_executor.run,
                user_text,
                verdict.search_queries or [user_text],
                3,                            # top_n
                search_categories,            # categories
            )

            try:
                payload = search_future.result(timeout=20.0)
            except Exception as e:
                logger.warning("Search workflow failed: %s", e)
                payload = None

            if not payload or not payload.sources:
                # Search produced nothing actionable; let the base LLM answer
                # and acknowledge the gap. We don't re-run the gate.
                logger.info(
                    "Search returned no sources (%s); falling back to base LLM",
                    payload.notes if payload else "n/a",
                )
                fallback_query = (
                    f"{user_text}\n\n"
                    "(I attempted a web search but it returned no usable "
                    "results. Answer from your existing knowledge and be "
                    "explicit about uncertainty if relevant.)"
                )
                yield from self.llm.generate_stream(
                    fallback_query,
                    history_user_message=bare_user_text,
                    # 2026-05-22 perf fix: retrieve against the BARE
                    # user_text (typically 20-50 chars) instead of the
                    # augmented body. The cross-encoder reranker on
                    # CPU was taking 30+ seconds per turn evaluating
                    # 9k+ char augmented queries.
                    rag_query=bare_user_text,
                    enable_thinking=False,
                )
                return

            self._last_search_payload = payload
            sources_block = format_sources_for_prompt(payload.sources)

            # 2026-05-22 news multi-event directive: when the user
            # asked "what's the latest news" / "any news today" /
            # "what's happening" -- a digest of MULTIPLE distinct
            # stories is more useful than a single-event summary.
            # Detected via the same _NEWS_QUERIES regex the gate uses.
            try:
                from kenning.web_search.gating import _NEWS_QUERIES
                is_news_query = bool(_NEWS_QUERIES.search(user_text))
            except Exception:                                        # noqa: BLE001
                is_news_query = False

            if is_news_query:
                shape_directive = (
                    "This is a news / current-events query. Summarize "
                    "3-5 DISTINCT stories from the sources above, one "
                    "short sentence each. Don't dwell on a single "
                    "event -- give the user a quick scan of what's "
                    "happening across the snippets. Attribute each "
                    "story to its source (e.g. 'CNN reports...', "
                    "'per NBC News...'). If the snippets only "
                    "describe one event, say so and summarize it.\n\n"
                )
            else:
                shape_directive = ""

            augmented = (
                f"User question: {user_text}\n\n"
                f"Fresh information from web search:\n{sources_block}\n\n"
                + shape_directive +
                "Answer the user's current question using ONLY the "
                "facts present in the search snippets above. Do not "
                "invent specifics that aren't visible in the snippets. "
                "If the snippets are too thin to fully answer, say so "
                "plainly (e.g. \"the search didn't cover X\") and stop "
                "-- do not pad with general knowledge that the search "
                "did not surface.\n\n"
                "When you attribute a fact, attribute it to a source "
                "whose name actually appears in the snippets (titles "
                "or domain). Do NOT cite a publication you remember "
                "from training data if the snippet block doesn't show "
                "that publication. \"According to NIST...\" is fine "
                "if NIST is one of the listed sources; \"According to "
                "Britannica...\" is forbidden if Britannica isn't in "
                "the source list.\n\n"
                "If any prior conversation context is genuinely "
                "relevant to THIS specific question (e.g. a related "
                "troubleshooting thread the user is continuing), you "
                "may briefly tie the answer to it. Otherwise treat the "
                "question as standalone -- do NOT drag in unrelated "
                "topics from past turns. Stay in character. Be concise. "
                "End the response when you have answered the question."
            )
            yield from self.llm.generate_stream(
                augmented,
                history_user_message=bare_user_text,
                # 2026-05-22 perf fix: retrieve against the BARE
                # user_text (~26 chars on a typical question) instead
                # of the augmented body (~9000+ chars containing the
                # full search-result block + instruction footer). The
                # cross-encoder reranker on CPU was taking 30+ seconds
                # per turn evaluating the long augmented query against
                # 20 candidates; the bare query drops that to ~2-3 s.
                rag_query=bare_user_text,
                # 2026-05-22 latency: skip Qwen3.5's <think> block.
                # The web sources are self-contained; the model just
                # needs to phrase the answer, not reason about it.
                enable_thinking=False,
            )
        finally:
            pool.shutdown(wait=False)

    def _interrupt_watcher(self) -> None:
        """Run wake-word detection during TTS playback for barge-in."""
        # Brief grace so the watcher doesn't trigger on residual user audio.
        time.sleep(settings.BARGE_IN_GRACE_SECONDS)
        self.audio.drain()
        local_wake = self.wake  # share the model, single-threaded predict
        while not self._interrupt.is_set() and not self._shutdown.is_set():
            chunk = self.audio.get_chunk(timeout=0.1)
            if chunk is None:
                continue
            self.ring.write(chunk)
            try:
                if local_wake.process(chunk):
                    logger.info("Barge-in detected; interrupting response")
                    print("\n  [interrupted]")
                    self.tts.stop()
                    self.llm.cancel()
                    self._pending_capture.set()
                    self._interrupt.set()
                    return
            except Exception as e:
                logger.warning("Wake watcher error: %s", e)
                return
