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

        self._shutdown = threading.Event()
        self._interrupt = threading.Event()
        self._pending_capture = threading.Event()
        self._state: State = State.IDLE

    @staticmethod
    def _load_memory_if_enabled():
        """Build a :class:`ConversationMemory` (with embedder) if enabled.

        Failures degrade gracefully: missing deps → memory disabled, missing
        embedder → memory persists turns but RAG retrieval returns empty.
        """
        if not settings.MEMORY_ENABLED:
            return None
        try:
            from ultron.memory import ConversationMemory
            from ultron.memory.embeddings import Embedder
        except Exception as e:
            logger.warning("Memory module import failed (%s) — disabling memory", e)
            return None

        embedder = None
        try:
            embedder = Embedder()
        except Exception as e:
            logger.warning(
                "Embedder load failed (%s) — memory will persist turns but "
                "RAG retrieval will be disabled.", e
            )

        try:
            return ConversationMemory(embedder=embedder)
        except Exception as e:
            logger.warning("ConversationMemory init failed (%s) — disabling memory", e)
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
                # addressee classifier. Don't reset the deadline on rejected
                # speech — we measure 30 s from the *last response*, not from
                # the last sound in the room.
                if came_from_follow_up:
                    if not self.llm.should_respond(user_text):
                        print(f"  (heard: {user_text!r} — not for me)")
                        continue
                    print(f"  (follow-up) you: {user_text}")
                else:
                    print(f"  you: {user_text}")

                self._respond(user_text)
                if settings.FOLLOW_UP_ENABLED:
                    follow_up_until = time.monotonic() + settings.FOLLOW_UP_TIMEOUT_SECONDS
                    print(
                        f"  (still listening for ~{int(settings.FOLLOW_UP_TIMEOUT_SECONDS)} s — "
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

    # --- phase: process ------------------------------------------------------

    def _respond(self, user_text: str) -> None:
        """Stream LLM tokens into TTS and watch for wake-word interruption."""
        self._interrupt.clear()
        watcher: Optional[threading.Thread] = None
        if settings.BARGE_IN_ENABLED:
            watcher = threading.Thread(
                target=self._interrupt_watcher, daemon=True, name="wake-watcher"
            )
            watcher.start()
        else:
            logger.info("Barge-in wake watcher disabled")

        try:
            token_stream = self.llm.generate_stream(user_text)
            print("  ultron: ", end="", flush=True)

            def gated():
                for token in token_stream:
                    if self._interrupt.is_set() or self._shutdown.is_set():
                        self.llm.cancel()
                        return
                    print(token, end="", flush=True)
                    yield token

            self.tts.speak_stream(gated())
            print()  # newline after streamed response
        except Exception as e:
            logger.exception("Response pipeline failed: %s", e)
            print(f"\n  [error] {e}")
        finally:
            self._interrupt.set()  # release watcher
            if watcher is not None:
                watcher.join(timeout=1.0)

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
