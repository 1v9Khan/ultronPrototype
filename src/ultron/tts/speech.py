"""Piper TTS wrapper with sentence-level streaming and optional RVC conversion.

The streaming API takes an iterator of text fragments (usually LLM tokens)
and synthesizes once a sentence boundary is reached, so audio starts playing
before the LLM finishes generating. Synthesis runs on a worker thread; the
main thread can interrupt mid-stream by calling :meth:`stop`.

If an :class:`RvcConverter` is passed in, every synthesized sentence is run
through it before playback. This converts Piper's neutral voice to the
trained target (Ultron). RVC may output at a different sample rate than
Piper, so each clip carries its own ``(pcm, sample_rate)`` pair through the
queue.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import sounddevice as sd

from config import settings
from ultron.tts.rvc import RvcConverter
from ultron.utils.logging import get_logger

logger = get_logger("tts.speech")

# A clip is (pcm_int16, sample_rate). Sample rate may vary per clip when RVC
# is in the loop because it can up-rate output beyond Piper's native 22050.
Clip = Tuple[np.ndarray, int]


class TextToSpeech:
    """Piper TTS playback with synchronous and streaming modes.

    Args:
        voice_path: Path to a Piper ``.onnx`` voice file.
        config_path: Path to the matching ``.onnx.json`` config.
        sample_rate: Piper's native output rate (medium voices = 22050).
        flush_chars: Characters that flush a buffered fragment as a sentence.
        length_scale: Piper pacing; >1.0 slows the voice down.
        rvc: Optional :class:`RvcConverter`. When set, every Piper sentence
            is run through RVC before playback.
    """

    def __init__(
        self,
        voice_path: Path = settings.TTS_VOICE_PATH,
        config_path: Path = settings.TTS_VOICE_CONFIG_PATH,
        sample_rate: int = settings.TTS_OUTPUT_SAMPLE_RATE,
        flush_chars: str = settings.TTS_SENTENCE_FLUSH_CHARS,
        length_scale: float = settings.TTS_LENGTH_SCALE,
        rvc: Optional[RvcConverter] = None,
    ) -> None:
        from piper import PiperVoice

        if not Path(voice_path).is_file():
            raise FileNotFoundError(
                f"Piper voice not found at {voice_path}. "
                f"Run `python scripts/download_models.py` first."
            )

        self.voice_path = Path(voice_path)
        self.piper_sample_rate = sample_rate
        self.flush_chars = set(flush_chars)
        self.length_scale = length_scale
        self.rvc = rvc
        self.output_device = self._resolve_output_device(settings.AUDIO_OUTPUT_DEVICE)
        self._stop_event = threading.Event()
        self._playback_lock = threading.Lock()

        logger.info("Loading Piper voice: %s", voice_path)
        t0 = time.monotonic()
        try:
            self._voice = PiperVoice.load(str(voice_path), config_path=str(config_path))
        except TypeError:
            self._voice = PiperVoice.load(str(voice_path))
        logger.info(
            "Piper voice ready in %.2fs (length_scale=%.2f, rvc=%s)",
            time.monotonic() - t0,
            length_scale,
            "on" if rvc else "off",
        )
        if self.output_device is None:
            logger.info("TTS output device: system default")
        else:
            try:
                dev = sd.query_devices(self.output_device, "output")
                logger.info("TTS output device: %s (%s)", self.output_device, dev["name"])
            except Exception:
                logger.info("TTS output device: %s", self.output_device)

    def __enter__(self) -> "TextToSpeech":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # --- public API ----------------------------------------------------------

    def stop(self) -> None:
        """Interrupt any in-progress playback."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass

    def speak(self, text: str) -> None:
        """Synchronously synthesize and play ``text`` to completion."""
        if not text.strip():
            return
        self._stop_event.clear()
        clip = self._synthesize(text)
        if clip[0].size > 0 and not self._stop_event.is_set():
            self._play(clip)

    def speak_stream(self, fragments: Iterable[str]) -> None:
        """Consume token fragments and play sentence-by-sentence.

        A worker thread reads the token iterator, accumulates a sentence,
        synthesizes (and converts via RVC if configured), and pushes the
        resulting clip onto a playback queue. The calling thread drains
        the queue and plays each clip in order.
        """
        self._stop_event.clear()
        audio_q: queue.Queue[Optional[Clip]] = queue.Queue(maxsize=8)

        def synth_worker() -> None:
            buffer = []
            try:
                for frag in fragments:
                    if self._stop_event.is_set():
                        break
                    if not frag:
                        continue
                    buffer.append(frag)
                    if any(c in self.flush_chars for c in frag):
                        sentence = "".join(buffer).strip()
                        buffer.clear()
                        if sentence:
                            clip = self._synthesize(sentence)
                            if clip[0].size > 0:
                                audio_q.put(clip)
                tail = "".join(buffer).strip()
                if tail and not self._stop_event.is_set():
                    clip = self._synthesize(tail)
                    if clip[0].size > 0:
                        audio_q.put(clip)
            except Exception as e:
                logger.error("TTS worker error: %s", e)
            finally:
                audio_q.put(None)  # sentinel

        worker = threading.Thread(target=synth_worker, daemon=True)
        worker.start()

        while True:
            if self._stop_event.is_set():
                break
            try:
                clip = audio_q.get(timeout=10.0)
            except queue.Empty:
                logger.warning("TTS playback queue starved; aborting")
                break
            if clip is None:
                break
            self._play(clip)

        worker.join(timeout=2.0)

    # --- internals -----------------------------------------------------------

    def _synthesize(self, text: str) -> Clip:
        """Synthesize one sentence: Piper → optional RVC → (pcm, sample_rate)."""
        t0 = time.monotonic()
        pcm, sr = self._piper_synth(text)
        if pcm.size == 0:
            return pcm, sr

        if self.rvc is not None:
            try:
                pcm, sr = self.rvc.convert(pcm, sr)
            except Exception as e:
                logger.warning("RVC convert failed (using raw Piper): %s", e)

        logger.debug(
            "TTS pipeline: %d chars → %.2fs audio @ %d Hz in %.0fms",
            len(text),
            len(pcm) / max(sr, 1),
            sr,
            (time.monotonic() - t0) * 1000,
        )
        return pcm, sr

    def _piper_synth(self, text: str) -> Clip:
        """Run Piper alone and return ``(int16 pcm, sample_rate)``."""
        wav_buffer = io.BytesIO()
        try:
            with wave.open(wav_buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.piper_sample_rate)
                # length_scale is the modern Piper kwarg; older builds ignore
                # unknown kwargs gracefully, but we wrap defensively anyway.
                try:
                    self._voice.synthesize(
                        text, wav, length_scale=self.length_scale
                    )
                except TypeError:
                    self._voice.synthesize(text, wav)
        except Exception as e:
            logger.error("Piper synth failed for %r: %s", text[:60], e)
            return np.zeros(0, dtype=np.int16), self.piper_sample_rate

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            sr = wav.getframerate()
        pcm = np.frombuffer(frames, dtype=np.int16)
        return pcm, sr

    def _resolve_output_device(self, configured: Optional[str]) -> Optional[str | int]:
        """Resolve playback target to explicit output device or None for default."""
        if configured:
            return configured
        try:
            default_dev = sd.default.device
            if isinstance(default_dev, (list, tuple)) and len(default_dev) >= 2:
                return default_dev[1]
            if isinstance(default_dev, int):
                return default_dev
        except Exception:
            pass
        return None

    def _play(self, clip: Clip) -> None:
        pcm, sr = clip
        with self._playback_lock:
            if self._stop_event.is_set():
                return
            try:
                sd.play(
                    pcm,
                    samplerate=sr,
                    blocking=False,
                    device=self.output_device,
                )
                duration = len(pcm) / max(sr, 1)
                deadline = time.monotonic() + duration + 0.5
                while time.monotonic() < deadline:
                    if self._stop_event.is_set():
                        sd.stop()
                        return
                    try:
                        if not sd.get_stream().active:
                            return
                    except Exception:
                        return
                    time.sleep(0.02)
            except Exception as e:
                logger.warning("Playback error: %s", e)
