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
import threading
import time
from enum import Enum
from typing import Optional, Union

import numpy as np

from config import settings
from ultron.addressing import AddressingClassifier, AddressingDecision
from ultron.audio import (
    AudioCapture,
    RingBuffer,
    VoiceActivityDetector,
    WakeWordDetector,
)
from ultron.audio.smart_turn import (
    SMART_TURN_SAMPLE_RATE,
    SmartTurnDetector,
    SmartTurnVerdict,
    build_detector_from_config,
)
from ultron.audio.vad import SpeechEvent
from ultron.llm import LLMEngine
from ultron.transcription import (
    DualSTTRegistry,
    WhisperEngine,
    make_dual_stt_engines,
    make_stt_engine,
)
from ultron.tts import RvcConverter, TextToSpeech
from ultron.utils.logging import get_logger
from ultron.coding import (
    CodingTaskRunner,
    CodingVoiceController,
    ProjectRegistry,
    ProjectResolver,
    UltronMCPServer,
)
from ultron.coding.coordinator import ConversationCoordinator
from ultron.coding.narration import StatusNarrator
from ultron.coding.voice import VoiceResponse as _VoiceResponse


def _voice_text(text: str) -> _VoiceResponse:
    """Wrap a plain string in a VoiceResponse so it can flow through
    :meth:`Orchestrator._handle_capability_response`."""
    return _VoiceResponse(text=text, handled=True)

from ultron.uncertainty import apply as apply_uncertainty
from ultron.conversational_ack import (
    ConversationalAckSource,
    is_conversational_ack_eligible,
)
from ultron.response_style import apply_brevity_hint
from ultron.safety.validator import (
    build_validator_from_config as _build_safety_validator_from_config,
    set_validator as _set_safety_validator,
)
from ultron.web_search import (
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

logger = get_logger("pipeline.orchestrator")


class State(Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    FOLLOW_UP_LISTENING = "follow_up"


# Sentinel values returned by :meth:`Orchestrator._follow_up_listen`.
_FU_TIMEOUT = "timeout"
_FU_WAKE = "wake"


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
            from ultron.config import get_config
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
        # via :func:`ultron.safety.get_validator`. Fail-open: if
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
        except Exception as e:
            self.safety_validator = None
            logger.warning(
                "safety validator construction failed (%s); call sites "
                "will see the permissive no-op validator", e,
            )
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
        self.memory = self._load_memory_if_enabled()
        # 2026-05-22 perf: warm the cross-encoder reranker shared
        # singleton at startup. Without this, the first turn that
        # exercises memory retrieval OR web-search snippet ranking
        # pays a ~2 s cold model load on the user's first interaction.
        # Loading at construct time amortises that cost into the
        # already-slow startup window. Fail-open: any error falls
        # back to the lazy-load path inside the reranker itself.
        try:
            from ultron.memory.reranker import get_shared_reranker
            t0 = time.monotonic()
            shared = get_shared_reranker()
            # Touch the model so the underlying sentence-transformers
            # ``CrossEncoder`` is loaded into memory + ONNX-compiled.
            shared._ensure_model()
            logger.info(
                "Cross-encoder reranker warmed in %.2fs",
                time.monotonic() - t0,
            )
        except Exception as e:                                          # noqa: BLE001
            logger.debug("Reranker warmup skipped (%s)", e)
        self.llm = LLMEngine(memory=self.memory)
        # 2026-05-10 voice swap: select TTS engine via ``tts.engine`` config.
        # ``"piper_rvc"`` (default) keeps the legacy Piper + RVC stack;
        # ``"xtts_v3"`` swaps in the XTTS v2 streaming + v3 Ultron filter
        # stack. The engines share the same ``speak`` / ``speak_stream``
        # / ``warmup`` / ``stop`` interface so the orchestrator's
        # downstream playback path (the producer-signaled lookahead in
        # speak_stream) doesn't change.
        self.rvc, self.tts = self._load_tts_engine()
        self.tts.warmup()
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
        # 2026-05-22 -- engine-agnostic intent recognizer. When
        # ``intent.enabled`` is True, matched utterances short-circuit
        # the LLM gating path; the dispatcher fires registered handlers
        # (e.g., gaming mode engage/disengage) directly. Default OFF so
        # operators opt in after deciding which phrases to register.
        self._intent_recognizer = self._init_intent_recognizer_if_enabled()
        # 2026-05-22 -- consumed by _build_response_stream. Set by the
        # intent dispatcher when a "needs fresh data" phrase matches;
        # forces the gate verdict to SEARCH (skipping preflight LLM).
        self._next_turn_force_search = False
        self._last_response_finished_monotonic: float = 0.0
        self._last_search_payload = None
        # 2026-05-22 OPEN_LAST_SOURCE: accumulated text of the most
        # recent assistant response. Used to match cited publication
        # names back to the source list when the user says "show me
        # that article".
        self._last_response_text: str = ""

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

    def _load_mcp_server_if_enabled(self):
        """Construct + start the MCP server (Phase 1+). Failures degrade
        silently -- the coding pipeline can run without MCP, just without
        the supervisor's clarification round-trip."""
        if not (settings.CODING_ENABLED and settings.CODING_MCP_ENABLED):
            return None
        try:
            # Phase 7: pass the per-session audit dir so SessionStore
            # auto-logs every state change to logs/sessions/<id>.jsonl.
            # A3 wiring: thread the live ConversationMemory through so
            # ``project.lookup_facts`` reads from Qdrant.
            server = UltronMCPServer(
                session_audit_dir=settings.CODING_SESSION_AUDIT_DIR,
                memory=self.memory,
            )
            server.start(ready_timeout_s=5.0)
            logger.info("MCP server listening at %s", server.sse_url)
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
        try:
            renderer = None
            try:
                from ultron.coding.templates import TemplateRenderer
                renderer = TemplateRenderer()
            except FileNotFoundError as e:
                logger.warning("Template renderer disabled (%s)", e)
            from ultron.coding.verification import Verifier
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
        """
        try:
            from ultron.desktop.vlm import build_vlm_from_config, set_vlm

            vlm = build_vlm_from_config(enabled=True, device="cpu")
            if vlm is not None:
                set_vlm(vlm)
                logger.info("VLM (moondream2) constructed -- lazy-loads on first use.")
        except Exception as e:                                    # noqa: BLE001
            logger.warning(
                "VLM construction skipped (%s) -- screen-context queries "
                "will fall back to text-only window/UIA context.", e,
            )

    def _init_intent_recognizer_if_enabled(self):
        """2026-05-22 -- construct the intent recognizer when enabled.

        Registers phrases from config + wires gaming-mode handlers.
        Lazy-loads the Gemma-300M embedding model on first use unless
        ``intent.warmup_on_init`` is True. Fail-open: any construction
        error logs WARN and returns None so the run loop skips the
        intent check.
        """
        try:
            from ultron.config import get_config
            cfg = get_config().intent
        except Exception as e:                                    # noqa: BLE001
            logger.warning("intent: config read failed (%s)", e)
            return None
        if not getattr(cfg, "enabled", False):
            return None
        try:
            from ultron.intent import (
                UltronIntentRecognizer, set_intent_recognizer,
            )
            recognizer = UltronIntentRecognizer(
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
        from ultron.config import get_config
        from ultron.openclaw_bridge import OpenClawBridge

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
        if not settings.CODING_ENABLED:
            return None
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
                from ultron.config import get_config as _get_cfg
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
                # the OpenClawDispatcher's plugin enable/disable path.
                gaming_mode_manager=self._load_gaming_mode_manager_if_enabled(),
                # 2026-05-22 supervisor stack -- None when disabled or
                # when construction failed (controller falls through
                # to legacy ProjectResolver path).
                supervisor_dispatch=supervisor_dispatch,
                project_index=project_index,
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
        from ultron.config import get_config

        project_index = None
        if cfg.index_enabled and embedder is not None:
            try:
                from ultron.coding.project_index import ProjectIndex
                project_index = ProjectIndex(embedder=embedder)
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "Supervisor: ProjectIndex construction failed (%s); "
                    "supervisor will run registry-only.", e,
                )
                project_index = None

        supervisor = None
        if cfg.decide_enabled:
            try:
                from ultron.coding.project_supervisor import ProjectSupervisor
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
                from ultron.coding.supervisor_dispatch import SupervisorDispatchController
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
                )
            except Exception as e:                                  # noqa: BLE001
                logger.warning(
                    "Supervisor: SupervisorDispatchController construction "
                    "failed (%s); supervisor will only emit decisions, not "
                    "dispatch.", e,
                )
                dispatch_controller = None

        return project_index, supervisor, dispatch_controller

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
        from ultron.config import get_config, resolve_path

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
            from ultron.openclaw_routing.gaming_mode import GamingModeManager

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
            gaming_llm_preset = (cfg.llm_preset or "").strip()
            # Cell to share the pre-engage preset between callbacks.
            llm_preset_before_engage: dict = {"value": None}

            # Stash the pre-engage STT engine so disengage knows what to restore to.
            stt_name_before_engage: dict = {"value": None}

            def _engage_extra():
                if gaming_llm_preset:
                    llm = getattr(self, "llm", None)
                    if llm is not None and hasattr(llm, "reload_for_preset"):
                        try:
                            current_preset = get_config().llm.preset
                            if current_preset != gaming_llm_preset:
                                ok, msg = llm.reload_for_preset(gaming_llm_preset)
                                if ok:
                                    llm_preset_before_engage["value"] = current_preset
                                    logger.info(
                                        "gaming engage: LLM swapped %s -> %s",
                                        current_preset, gaming_llm_preset,
                                    )
                                else:
                                    logger.warning(
                                        "gaming engage: LLM swap to %s failed (%s); "
                                        "keeping %s",
                                        gaming_llm_preset, msg, current_preset,
                                    )
                        except Exception as e:                    # noqa: BLE001
                            logger.warning(
                                "gaming engage: LLM swap skipped (%s)", e,
                            )
                # STT swap (Parakeet -> Moonshine on the typical setup).
                registry = getattr(self, "_stt_registry", None)
                if registry is not None and registry.has_gaming():
                    try:
                        prior_stt = registry.active_name
                        if self.swap_stt_engine(registry.gaming_name):
                            stt_name_before_engage["value"] = prior_stt
                            # Kill the Parakeet HTTP server to free its VRAM.
                            # Safe to call when Parakeet isn't running -- no-op.
                            try:
                                from ultron.transcription.parakeet_engine import (
                                    stop_parakeet_server,
                                )
                                if stop_parakeet_server():
                                    logger.info(
                                        "gaming engage: Parakeet server stopped "
                                        "(~700 MB VRAM freed)",
                                    )
                            except Exception as e:                # noqa: BLE001
                                logger.warning(
                                    "gaming engage: Parakeet server stop "
                                    "failed (%s); STT swap still in effect", e,
                                )
                    except Exception as e:                        # noqa: BLE001
                        logger.warning(
                            "gaming engage: STT swap skipped (%s)", e,
                        )
                tts = getattr(self, "tts", None)
                if tts is not None and hasattr(tts, "move_to_device"):
                    tts.move_to_device("cpu")
                try:
                    from ultron.desktop.vlm import get_vlm
                    vlm = get_vlm()
                    if vlm is not None and vlm.loaded:
                        vlm.unload()
                except Exception as e:                            # noqa: BLE001
                    logger.warning("gaming engage: VLM unload skipped (%s)", e)

            def _disengage_extra():
                tts = getattr(self, "tts", None)
                if tts is not None and hasattr(tts, "move_to_device"):
                    tts.move_to_device(tts_kokoro_default_device)
                # STT restore: stay on gaming engine while Parakeet
                # respawns in the background; swap back when ready.
                prior_stt = stt_name_before_engage["value"]
                registry = getattr(self, "_stt_registry", None)
                if prior_stt is not None and registry is not None:
                    if prior_stt == "parakeet":
                        # Spawn Parakeet server in background; swap pointer
                        # only when /healthz reports ready so transcribes
                        # in the interim still hit the gaming engine.
                        try:
                            from ultron.transcription.parakeet_engine import (
                                start_parakeet_server,
                            )

                            def _restore_when_ready():
                                try:
                                    start_parakeet_server(wait_for_ready=True)
                                    self.swap_stt_engine(prior_stt)
                                    logger.info(
                                        "gaming disengage: Parakeet ready; "
                                        "STT swapped back to %s", prior_stt,
                                    )
                                except Exception as e:            # noqa: BLE001
                                    logger.warning(
                                        "gaming disengage: Parakeet restore "
                                        "failed (%s); staying on gaming engine", e,
                                    )

                            threading.Thread(
                                target=_restore_when_ready,
                                daemon=True,
                                name="parakeet-restore",
                            ).start()
                        except Exception as e:                    # noqa: BLE001
                            logger.warning(
                                "gaming disengage: failed to spawn restore "
                                "thread (%s)", e,
                            )
                    else:
                        # Non-Parakeet primary: swap pointer immediately.
                        self.swap_stt_engine(prior_stt)
                    stt_name_before_engage["value"] = None
                prior_preset = llm_preset_before_engage["value"]
                if prior_preset is not None:
                    llm = getattr(self, "llm", None)
                    if llm is not None and hasattr(llm, "reload_for_preset"):
                        try:
                            ok, msg = llm.reload_for_preset(prior_preset)
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
                        except Exception as e:                    # noqa: BLE001
                            logger.warning(
                                "gaming disengage: LLM restore skipped (%s)", e,
                            )
                    llm_preset_before_engage["value"] = None

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
                "kokoro_engage_device=cpu, kokoro_disengage_device=%s, "
                "vlm_unload_on_engage=True, llm_preset=%s)",
                cfg.plugins_to_disable, cfg.toggle_docker,
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
        from ultron.config import get_config
        from ultron.web_search.provider_chain import SearchProviderChain
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
            from ultron.web_search.reader_chain import ReaderChain
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
            from ultron.config import PROJECT_ROOT
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

        from ultron.config import get_config, resolve_path
        addr_cfg = get_config().addressing
        return AddressingClassifier(
            rule_confidence_threshold=addr_cfg.rule_confidence_threshold,
            default_silent_on_uncertain=addr_cfg.default_uncertain_to_not_addressed,
            log_path=resolve_path(addr_cfg.log_path),
            zero_shot_model_name=addr_cfg.zero_shot_model,
            load_zero_shot_eagerly=addr_cfg.load_eagerly,
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
            from ultron.memory import ConversationMemory, HybridEmbedder
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
        try:
            from ultron.config import get_config
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
            from ultron.memory.background_summarizer import (
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

    @staticmethod
    def _load_rvc_if_enabled() -> RvcConverter | None:
        """Try to load RVC; warn and continue with plain Piper on failure."""
        if not settings.RVC_ENABLED:
            return None
        if not settings.RVC_MODEL_PATH.is_file():
            logger.warning(
                "RVC enabled but model missing at %s — falling back to plain Piper",
                settings.RVC_MODEL_PATH,
            )
            return None
        try:
            return RvcConverter()
        except Exception as e:
            logger.warning("RVC load failed (%s) — falling back to plain Piper", e)
            return None

    def _load_tts_engine(self):
        """Construct the configured TTS engine.

        Returns a ``(rvc_or_none, tts_engine)`` pair. The ``rvc``
        attribute is kept on Orchestrator for diagnostic purposes
        even though only the legacy engine uses it.

        Raises any engine-construction error -- TTS is not optional;
        the orchestrator can't run without a voice path.
        """
        from ultron.config import get_config, resolve_path
        try:
            engine_name = get_config().tts.engine
        except Exception:
            engine_name = "piper_rvc"

        if engine_name == "xtts_v3":
            from ultron.tts.xtts_v3 import XttsV3Speech
            logger.info("TTS engine: xtts_v3 (XTTS v2 streaming + v3 filter)")
            tts = XttsV3Speech()
            return None, tts
        if engine_name == "kokoro":
            # 2026-05-20 round 8: lightweight StyleTTS2 + ISTFTNet on
            # CPU. Stock voice (no v3 filter); ~330 MB on disk; zero
            # VRAM. Config wiring reads tts.kokoro.* so the operator
            # can swap voice / device / speed without code edits.
            from ultron.tts.kokoro_engine import KokoroSpeech
            kokoro_cfg = getattr(get_config().tts, "kokoro", None)
            kwargs = {}
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
                }
            logger.info(
                "TTS engine: kokoro (StyleTTS2 + ISTFTNet, voice=%s, device=%s)",
                kwargs.get("voice", "af_alloy"),
                kwargs.get("device", "cpu"),
            )
            tts = KokoroSpeech(**kwargs)
            return None, tts
        if engine_name == "piper_rvc":
            logger.info("TTS engine: piper_rvc (legacy Piper + RVC)")
            rvc = self._load_rvc_if_enabled()
            tts = TextToSpeech(rvc=rvc)
            return rvc, tts
        raise RuntimeError(
            f"Unknown tts.engine: {engine_name!r}. "
            f"Valid: 'piper_rvc' | 'xtts_v3' | 'kokoro'."
        )

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
            from ultron.tts.precomputed_ack import (
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
            # lets OpenClaw spawn Ultron's MCP across restarts.
            try:
                self.openclaw_bridge.shutdown()
            except Exception:
                pass

    # --- main loop -----------------------------------------------------------

    def run(self) -> None:
        """Block forever, processing wake events until shutdown."""
        from ultron.config import get_config
        from ultron import trace
        _addr_cfg = get_config().addressing
        self.audio.start()
        word = self.wake.active_word
        print(f"\n  Ultron is listening. Say '{word}' to wake.\n")
        if self.wake.using_fallback:
            print(
                f"  (Wake word currently fallback='{word}'. "
                f"Train a custom model for true 'ultron' detection — see README.)\n"
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
                # Coding-task completion push: if a background Claude Code
                # task just finished, announce it before we go back to
                # listening. This gives the unsolicited "Done. Created X
                # in Y..." narration the spec calls for.
                self._announce_coding_completion_if_pending()
                # Phase 2: surface any clarifications Claude is parked on.
                self._announce_pending_clarifications()
                # Phase 7: surface token-budget warnings + halt notices.
                self._announce_pending_budget_warning()
                # 4B plan Item 7: surface canonical-path-monitor aborts.
                self._announce_pending_canonical_abort()
                # E2 goal-anchor planning: surface anchor lifecycle
                # narration (opening / warning / transition / completion).
                self._announce_pending_anchor_narration()
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
                    from ultron.openclaw_routing import classify_routing
                    from ultron.openclaw_routing.intents import RoutingIntentKind
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

                trace.set_phase("respond")
                self._respond(user_text)
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
        # wake-word "Ultron" tail does not bleed into Whisper as a
        # "Tron" prefix. The full ring is sized for the larger WARM
        # slice; the COLD path explicitly limits how much it consumes.
        cold_pre_roll_samples = int(
            self._cold_pre_roll_seconds * settings.SAMPLE_RATE
        )
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

        while not self._shutdown.is_set() and time.monotonic() < deadline:
            chunk = self.audio.get_chunk(timeout=0.1)
            if chunk is None:
                continue
            self.ring.write(chunk)

            # Wake word always wins — even if we're mid-utterance.
            if self.wake.process(chunk):
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
                # else: still waiting for speech — keep ticking.
                continue

            speech_chunks.append(chunk)
            speech_samples += chunk.shape[0]

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
                        return captured
                    if band == "early_complete":
                        logger.info(
                            "Smart Turn V3 (follow-up): early-complete "
                            "(prob=%.3f, %.1f ms)",
                            verdict.probability, verdict.latency_ms,
                        )
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
                return np.concatenate(pieces).astype(
                    np.float32, copy=False,
                )

            if speech_samples >= max_samples:
                # Hard cap — return what we have, classifier can still gate it.
                pieces = ([pre_roll] if pre_roll is not None else []) + speech_chunks
                return np.concatenate(pieces).astype(np.float32, copy=False)

        return _FU_TIMEOUT

    # --- coding pipeline glue -----------------------------------------------

    def _speak(self, text: str) -> None:
        """Synchronously speak a fixed string + print it. Used by the coding
        pipeline for progress narrations and completion announcements --
        the regular LLM streaming path uses ``speak_stream`` instead."""
        if not text:
            return
        print(f"  ultron: {text}")
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
        print(f"  ultron: {text}")
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
        the user does wake Ultron during a summary, the cancel flag
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
                    name="ultron-background-summarizer",
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
        :func:`ultron.desktop.voice.handle_app_launch` so Chrome opens
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
                from ultron.openclaw_routing.intents import AppLaunchIntent
                from ultron.desktop.voice import handle_app_launch

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
            from ultron.web_search.searxng import SearxNGSearchClient
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
                from ultron.openclaw_routing.intents import AppLaunchIntent
                from ultron.desktop.voice import handle_app_launch

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

    def _respond(self, user_text: str) -> None:
        """Stream LLM tokens into TTS and watch for wake-word interruption.

        Phase 4: classifies the utterance through the web-search gate first.
        SEARCH -> speak an acknowledgment phrase, run the search workflow
        (Brave + Jina + LLM rank) in parallel with the ack TTS, then
        generate the final response with sources injected.
        NO_SEARCH / UNCERTAIN -> base path (unchanged from Phase 3).
        """
        self._interrupt.clear()
        self._last_search_payload = None
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

        try:
            print("  ultron: ", end="", flush=True)
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

            # Sources go to the transcript only -- no TTS read-out, since
            # citations interleaved with the spoken answer would clutter the
            # voice output. The user can scan the printed list to verify.
            if self._last_search_payload and self._last_search_payload.sources:
                print(f"  {format_sources_for_transcript(self._last_search_payload.sources)}")
        except Exception as e:
            logger.exception("Response pipeline failed: %s", e)
            print(f"\n  [error] {e}")
        finally:
            self._interrupt.set()  # release watcher
            if watcher is not None:
                watcher.join(timeout=1.0)

    def _maybe_conversational_ack(self, user_text: str) -> Optional[str]:
        """Return a filler-ack phrase to prepend on the conversational
        path, or None if the gate suppresses it.

        2026-05-12 filler-ack: masks the ~2.5 s perceived gap between
        Whisper completing and the LLM's first TTS chunk on the no-
        search conversational branch. The web-search path already
        yields its own ack from :meth:`_search_augmented_tokens`; this
        helper covers the no-search branches only.

        Gate semantics live in
        :func:`ultron.conversational_ack.is_conversational_ack_eligible`;
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
            from ultron.config import get_config
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
                from ultron.web_search.gating import classify_by_rules
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
            from ultron.web_search import GateDecision
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
            from ultron.uncertainty import apply as apply_uncertainty
            from ultron.response_style import apply_brevity_hint
            from ultron.web_search import GateDecision

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
            from ultron.config import get_config
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
        from ultron import trace
        # 2026-05-19 round 4: local clock / date short-circuit.
        # The detector is strict -- only fires on bare time/date asks.
        try:
            from ultron.local_clock_reply import maybe_local_clock_reply
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
            from ultron.web_search.gating import _NEWS_QUERIES
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
                from ultron.web_search.gating import _NEWS_QUERIES
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
