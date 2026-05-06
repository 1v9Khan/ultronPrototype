"""Main event loop.

The orchestrator owns every component and runs a small state machine:

    IDLE
      └─ wake word fires ──► CAPTURING
                                └─ VAD end-of-speech ──► PROCESSING
                                                             ├─ Whisper
                                                             ├─ LLM (streaming)
                                                             └─ TTS (streaming)
                                                                 │
                                                                 │ wake word fires
                                                                 ▼
                                                              CAPTURING (next turn)

Two threads matter:
- The audio thread (inside :class:`AudioCapture`) only enqueues chunks.
- The orchestrator's own thread does everything else.

During TTS playback an *interrupt watcher* thread also runs the wake-word
detector on incoming audio so the user can barge in.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Optional

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
        self.llm = LLMEngine()
        self.rvc = self._load_rvc_if_enabled()
        self.tts = TextToSpeech(rvc=self.rvc)

        self._shutdown = threading.Event()
        self._interrupt = threading.Event()
        self._pending_capture = threading.Event()
        self._state: State = State.IDLE

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

        try:
            while not self._shutdown.is_set():
                if self._pending_capture.is_set():
                    self._pending_capture.clear()
                else:
                    self._state = State.IDLE
                    if not self._wait_for_wake_word():
                        break

                self._state = State.CAPTURING
                print(f"  [{self._state.value}] capturing your request…")
                speech = self._capture_utterance()
                if speech.size == 0:
                    print("  (heard nothing; standing down)")
                    continue

                self._state = State.PROCESSING
                user_text = self.stt.transcribe(speech)
                if not user_text.strip():
                    print("  (no transcription; standing down)")
                    continue

                print(f"  you: {user_text}")
                self._respond(user_text)
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

    # --- phase: process ------------------------------------------------------

    def _respond(self, user_text: str) -> None:
        """Stream LLM tokens into TTS and watch for wake-word interruption."""
        self._interrupt.clear()
        watcher = threading.Thread(
            target=self._interrupt_watcher, daemon=True, name="wake-watcher"
        )
        watcher.start()

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
            watcher.join(timeout=1.0)

    def _interrupt_watcher(self) -> None:
        """Run wake-word detection during TTS playback for barge-in."""
        # Brief grace so the watcher doesn't trigger on residual user audio.
        time.sleep(0.3)
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
