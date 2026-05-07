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
from ultron.audio.vad import SpeechEvent
from ultron.llm import LLMEngine
from ultron.transcription import WhisperEngine
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
from ultron.uncertainty import apply as apply_uncertainty
from ultron.web_search import (
    AcknowledgmentSource,
    BraveSearchClient,
    GateDecision,
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

    MAX_UTTERANCE_SECONDS = 15.0  # hard cap to avoid unbounded recording

    def __init__(self) -> None:
        self.audio = AudioCapture()
        self.ring = RingBuffer(
            int(settings.RING_BUFFER_SECONDS * settings.SAMPLE_RATE)
        )
        self.wake = WakeWordDetector()
        self.vad = VoiceActivityDetector()
        self.stt = WhisperEngine()
        self.memory = self._load_memory_if_enabled()
        self.llm = LLMEngine(memory=self.memory)
        self.rvc = self._load_rvc_if_enabled()
        self.tts = TextToSpeech(rvc=self.rvc)
        self.tts.warmup()
        self.addressing = self._load_addressing_classifier()
        self.web_gate, self.web_executor, self.ack_source = (
            self._load_web_search_if_enabled()
        )
        self.mcp_server = self._load_mcp_server_if_enabled()
        self.coding_coordinator = self._load_coding_coordinator_if_enabled()
        self.coding_voice = self._load_coding_voice_if_enabled()
        self._last_response_finished_monotonic: float = 0.0
        self._last_search_payload = None

        self._shutdown = threading.Event()
        self._interrupt = threading.Event()
        self._pending_capture = threading.Event()
        self._state: State = State.IDLE

    def _load_mcp_server_if_enabled(self):
        """Construct + start the MCP server (Phase 1+). Failures degrade
        silently -- the coding pipeline can run without MCP, just without
        the supervisor's clarification round-trip."""
        if not (settings.CODING_ENABLED and settings.CODING_MCP_ENABLED):
            return None
        try:
            server = UltronMCPServer()
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
            coordinator = ConversationCoordinator(
                store=self.mcp_server.store,
                llm=self.llm,
                renderer=renderer,
                verifier=verifier,
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
            controller = CodingVoiceController(
                runner=runner,
                registry=registry,
                resolver=resolver,
                sandbox_root=settings.CODING_SANDBOX_PATH,
                coordinator=self.coding_coordinator,
            )
            logger.info(
                "Coding voice ready (bridge=%s, sandbox=%s, coordinator=%s)",
                runner.bridge.name(), settings.CODING_SANDBOX_PATH,
                "on" if self.coding_coordinator is not None else "off",
            )
            return controller
        except Exception as e:
            logger.warning("Coding voice init failed (%s) -- disabled.", e)
            return None

    def _load_web_search_if_enabled(self):
        """Construct the web-search gate + executor if enabled.

        Returns ``(gate, executor, ack_source)`` triple. Any of them can be
        ``None`` if web search is disabled or the API key is missing -- the
        rest of the pipeline still works.
        """
        if not settings.WEB_SEARCH_ENABLED:
            return None, None, None
        if not settings.WEB_SEARCH_BRAVE_API_KEY:
            logger.warning(
                "WEB_SEARCH_ENABLED=True but ULTRON_BRAVE_API_KEY missing -- "
                "web search disabled."
            )
            return None, None, None
        try:
            brave = BraveSearchClient()
            jina = JinaReaderClient()
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
            executor = WebSearchExecutor(
                brave=brave, jina=jina, llm=self.llm, cache=cache,
            )
            ack = AcknowledgmentSource()
            return gate, executor, ack
        except Exception as e:
            logger.warning("Web search init failed (%s) -- disabled.", e)
            return None, None, None

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

        return AddressingClassifier(
            rule_confidence_threshold=settings.ADDRESSING_RULE_CONFIDENCE_THRESHOLD,
            default_silent_on_uncertain=settings.ADDRESSEE_DEFAULT_SILENT,
            log_path=settings.ADDRESSING_LOG_PATH,
            zero_shot_model_name=settings.ADDRESSING_ZERO_SHOT_MODEL,
            load_zero_shot_eagerly=settings.ADDRESSING_LOAD_EAGERLY,
            recent_turns_provider=recent_turns_provider,
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

    # --- main loop -----------------------------------------------------------

    def run(self) -> None:
        """Block forever, processing wake events until shutdown."""
        self.audio.start()
        word = self.wake.active_word
        print(f"\n  Ultron is listening. Say '{word}' to wake.\n")
        if self.wake.using_fallback:
            print(
                f"  (Wake word currently fallback='{word}'. "
                f"Train a custom model for true 'ultron' detection — see README.)\n"
            )
        if self.memory is not None:
            print(f"  Memory: {len(self.memory)} prior turns loaded.\n")

        # When the follow-up window is open this holds the deadline (monotonic
        # time). ``None`` means we're in plain wake-word-gated IDLE mode.
        follow_up_until: Optional[float] = None

        try:
            while not self._shutdown.is_set():
                # Coding-task completion push: if a background Claude Code
                # task just finished, announce it before we go back to
                # listening. This gives the unsolicited "Done. Created X
                # in Y..." narration the spec calls for.
                self._announce_coding_completion_if_pending()
                # Phase 2: surface any clarifications Claude is parked on.
                self._announce_pending_clarifications()

                speech: Optional[np.ndarray] = None
                came_from_follow_up = False

                if self._pending_capture.is_set():
                    # Barge-in or wake-during-follow-up → fresh wake-gated capture.
                    self._pending_capture.clear()
                    self._state = State.CAPTURING
                    print(f"  [{self._state.value}] capturing your request…")
                    speech = self._capture_utterance()
                    follow_up_until = None
                elif (
                    follow_up_until is not None
                    and settings.FOLLOW_UP_ENABLED
                    and time.monotonic() < follow_up_until
                ):
                    self._state = State.FOLLOW_UP_LISTENING
                    outcome = self._follow_up_listen(deadline=follow_up_until)
                    # outcome is either an ndarray (audio captured) or a
                    # sentinel string. Type-check first — comparing an ndarray
                    # to a string with `==` gives an element-wise array, which
                    # raises in a boolean context.
                    if isinstance(outcome, str):
                        if outcome == _FU_TIMEOUT:
                            print("  (follow-up window closed; waiting for wake word)")
                            follow_up_until = None
                            continue
                        if outcome == _FU_WAKE:
                            self._state = State.CAPTURING
                            print(f"  [{self._state.value}] capturing your request…")
                            speech = self._capture_utterance()
                            follow_up_until = None
                    else:
                        # Got a VAD-bounded utterance during follow-up.
                        speech = outcome
                        came_from_follow_up = True
                else:
                    self._state = State.IDLE
                    follow_up_until = None
                    if not self._wait_for_wake_word():
                        break
                    self._state = State.CAPTURING
                    print(f"  [{self._state.value}] capturing your request…")
                    speech = self._capture_utterance()

                if speech is None or speech.size == 0:
                    if not came_from_follow_up:
                        print("  (heard nothing; standing down)")
                    continue

                self._state = State.PROCESSING
                user_text = self.stt.transcribe(speech)
                if not user_text.strip():
                    if not came_from_follow_up:
                        print("  (no transcription; standing down)")
                    continue

                # In the follow-up window, gate every utterance through the
                # CPU-side addressing classifier. Don't reset the deadline on
                # rejected speech -- we measure FOLLOW_UP_TIMEOUT_SECONDS from
                # the *last response*, not from the last sound in the room.
                if came_from_follow_up:
                    seconds_since = (
                        time.monotonic() - self._last_response_finished_monotonic
                    )
                    verdict = self.addressing.classify(
                        user_text, seconds_since_response=seconds_since
                    )
                    if verdict.decision != AddressingDecision.ADDRESSED:
                        print(
                            f"  (heard: {user_text!r} -- not for me "
                            f"[{verdict.source}: {verdict.reason}])"
                        )
                        continue
                    print(f"  (follow-up) you: {user_text}")
                else:
                    print(f"  you: {user_text}")

                # Coding pipeline gets first crack: progress queries,
                # cancellations, and new-task-creation utterances are
                # handled here without ever touching the LLM. handle_utterance
                # returns None for anything not coding-related; we then fall
                # through to the normal LLM/search path.
                if self.coding_voice is not None:
                    coding_response = self.coding_voice.handle_utterance(user_text)
                    if coding_response is not None:
                        self._speak(coding_response.text)
                        self._last_response_finished_monotonic = time.monotonic()
                        if settings.FOLLOW_UP_ENABLED:
                            follow_up_until = (
                                self._last_response_finished_monotonic
                                + settings.FOLLOW_UP_TIMEOUT_SECONDS
                            )
                        else:
                            follow_up_until = None
                        continue

                self._respond(user_text)
                self._last_response_finished_monotonic = time.monotonic()
                if settings.FOLLOW_UP_ENABLED:
                    follow_up_until = (
                        self._last_response_finished_monotonic
                        + settings.FOLLOW_UP_TIMEOUT_SECONDS
                    )
                    print(
                        f"  (still listening for ~{int(settings.FOLLOW_UP_TIMEOUT_SECONDS)} s -- "
                        f"keep talking or stay silent to drop back to wake-word mode)"
                    )
                else:
                    follow_up_until = None
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
        """Record from now until VAD reports end-of-speech (or timeout)."""
        self.vad.reset()
        # Pre-roll: include the half-second before the wake word so an
        # immediately-following request isn't clipped.
        chunks: list[np.ndarray] = [self.ring.snapshot()]
        speech_seen = False
        elapsed_samples = 0
        max_samples = int(self.MAX_UTTERANCE_SECONDS * settings.SAMPLE_RATE)
        # Allow up to MIN_SILENCE * 2 of leading silence before bailing.
        silence_grace = int(2.0 * settings.SAMPLE_RATE)
        leading_silence = 0

        while not self._shutdown.is_set() and elapsed_samples < max_samples:
            chunk = self.audio.get_chunk(timeout=0.5)
            if chunk is None:
                continue
            chunks.append(chunk)
            elapsed_samples += chunk.shape[0]

            result = self.vad.process(chunk)
            if result.event == SpeechEvent.SPEECH_START:
                speech_seen = True
            elif result.event == SpeechEvent.SPEECH_END and speech_seen:
                break

            if not speech_seen:
                leading_silence += chunk.shape[0]
                if leading_silence >= silence_grace:
                    return np.zeros(0, dtype=np.float32)

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
        """
        self.audio.drain()
        self.wake.reset()
        self.vad.reset()
        # Don't clear the ring — we want pre-roll continuity from the moment
        # TTS finished.

        speech_started = False
        speech_chunks: list[np.ndarray] = []
        pre_roll: Optional[np.ndarray] = None
        speech_samples = 0
        max_samples = int(self.MAX_UTTERANCE_SECONDS * settings.SAMPLE_RATE)

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
                    pre_roll = self.ring.snapshot()
                    speech_chunks.append(chunk)
                    speech_started = True
                    speech_samples = chunk.shape[0]
                # else: still waiting for speech — keep ticking.
                continue

            speech_chunks.append(chunk)
            speech_samples += chunk.shape[0]

            if result.event == SpeechEvent.SPEECH_END:
                pieces = ([pre_roll] if pre_roll is not None else []) + speech_chunks
                return np.concatenate(pieces).astype(np.float32, copy=False)

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

    def _announce_coding_completion_if_pending(self) -> None:
        """If a background coding task just finished, speak its summary
        before we go back to listening for the next utterance."""
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

    def _announce_pending_clarifications(self) -> None:
        """Speak any clarifications Claude is waiting on. Each prompt is
        spoken at most once -- the user's next utterance answers it."""
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

    # --- phase: process ------------------------------------------------------

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
                    yield token

            self.tts.speak_stream(gated())
            print()  # newline after streamed response

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

    def _build_response_stream(self, user_text: str):
        """Yield tokens for the response, applying the web-search gate.

        Returned generator handles three paths:
          * No gate or NO_SEARCH/UNCERTAIN -> base ``llm.generate_stream``.
          * SEARCH with successful retrieval -> ack phrase, then a
            search-augmented prompt.
          * SEARCH with empty/failed retrieval -> ack phrase, then base
            generation (LLM at least apologizes accurately).
        """
        if self.web_gate is None or self.web_executor is None:
            yield from self.llm.generate_stream(user_text)
            return

        try:
            verdict = self.web_gate.classify(user_text)
        except Exception as e:
            logger.warning("Web gate failed (%s) -- falling through to base", e)
            yield from self.llm.generate_stream(user_text)
            return

        # Phase 5: translate preflight uncertainty signals into behavior.
        # May upgrade NO_SEARCH -> SEARCH (low confidence + temporal), and
        # may prepend a short [Confidence: ...] addendum to the user text
        # so the LLM matches its tone to the actual confidence level.
        verdict, augmented_text = apply_uncertainty(verdict, user_text)

        logger.info(
            "gate: %s (%s, %s) -- %s",
            verdict.decision.value, verdict.source, verdict.confidence,
            verdict.reason,
        )
        if verdict.decision != GateDecision.SEARCH:
            yield from self.llm.generate_stream(augmented_text)
            return

        yield from self._search_augmented_tokens(augmented_text, verdict)

    def _search_augmented_tokens(self, user_text: str, verdict):
        """Yield ack phrase + search-augmented LLM tokens.

        The search workflow (Brave + Jina + LLM rank) runs on a worker
        thread that's kicked off BEFORE we yield the ack -- TTS speaks the
        ack while the network calls are in flight, hiding the search
        latency behind the voice prompt.
        """
        from concurrent.futures import ThreadPoolExecutor

        ack_phrase = self.ack_source.next_phrase() if self.ack_source else "Searching."
        # The ack ends with a period so the TTS pipeline flushes it as a
        # complete sentence immediately. We add a trailing space so it
        # blends into the streamed answer without an awkward double-period.
        ack = ack_phrase + " "

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-search")
        try:
            search_future = pool.submit(
                self.web_executor.run,
                user_text,
                verdict.search_queries or [user_text],
            )
            yield ack

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
                yield from self.llm.generate_stream(fallback_query)
                return

            self._last_search_payload = payload
            sources_block = format_sources_for_prompt(payload.sources)
            augmented = (
                f"User question: {user_text}\n\n"
                f"Fresh information from web search:\n{sources_block}\n\n"
                "Answer the user's question using this information. Cite "
                "sources naturally in prose (e.g. \"According to NASA, ...\"). "
                "Stay in character. Be concise."
            )
            yield from self.llm.generate_stream(augmented)
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
