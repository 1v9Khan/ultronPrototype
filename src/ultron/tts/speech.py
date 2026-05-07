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
from ultron.audio.devices import describe_device, resolve_device
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
        self.output_device = resolve_device(settings.AUDIO_OUTPUT_DEVICE, "output")
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
        logger.info("TTS output device: %s", describe_device(self.output_device, "output"))

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

    def warmup(self, text: str = "System ready.") -> None:
        """Prime Piper + optional RVC so first real response starts faster."""
        if not text.strip():
            return
        t0 = time.monotonic()
        try:
            self._synthesize(text)
            logger.info("TTS warmup complete in %.0fms", (time.monotonic() - t0) * 1000)
        except Exception as e:
            logger.warning("TTS warmup skipped: %s", e)

    def speak_stream(self, fragments: Iterable[str]) -> None:
        """Consume token fragments and play sentence-by-sentence.

        A worker thread reads the token iterator, accumulates a sentence,
        synthesizes (and converts via RVC if configured), and pushes the
        resulting clip onto a playback queue. The calling thread holds a
        single :class:`sounddevice.OutputStream` open for the entire
        utterance, eliminating the per-sentence open/close clicks that
        Windows audio drivers produce, and applies brief edge fades + a
        pre-roll silence so the first sample doesn't pop.
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

        # Pull one clip up-front so we know the playback sample rate (RVC may
        # output at a different rate than Piper) before opening the stream.
        first_clip: Optional[Clip] = None
        try:
            first_clip = audio_q.get(timeout=10.0)
        except queue.Empty:
            logger.warning("TTS playback queue starved before first clip")
            worker.join(timeout=2.0)
            return
        if first_clip is None:
            worker.join(timeout=2.0)
            return

        sr = first_clip[1]
        block_frames = max(1, int(sr * 0.05))
        stream: Optional[sd.OutputStream] = None
        last_clip = first_clip
        try:
            with self._playback_lock:
                if self._stop_event.is_set():
                    return
                stream = sd.OutputStream(
                    samplerate=sr,
                    channels=2,
                    dtype="int16",
                    device=self.output_device,
                )
                stream.start()
                logger.info(
                    "TTS stream opened: %d Hz via %s",
                    sr,
                    describe_device(self.output_device, "output"),
                )

                # 50 ms silence pre-roll lets the audio device spin up before
                # real samples land — kills the cold-start pop.
                self._write_silence(stream, sr, 0.05)

                # One-clip lookahead so we know which clip is last in time
                # to fade out its tail before writing.
                clip = first_clip
                is_first = True
                while True:
                    try:
                        nxt = audio_q.get(timeout=10.0)
                    except queue.Empty:
                        logger.warning("TTS playback queue starved; aborting")
                        nxt = None
                    is_last = nxt is None

                    audio = self._stereo_pcm(clip[0])
                    # Tiny fade on every clip edge so the inter-clip silence
                    # gap doesn't introduce a click. Short enough (~4 ms) to
                    # be inaudible as volume modulation.
                    edge_ms = settings.TTS_EDGE_FADE_MS
                    if edge_ms > 0:
                        audio = self._apply_fade_in(audio, sr, ms=edge_ms)
                        audio = self._apply_fade_out(audio, sr, ms=edge_ms)

                    for start in range(0, audio.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(audio[start : start + block_frames])

                    if is_last:
                        # Brief silence tail for clean driver flush.
                        self._write_silence(stream, sr, 0.05)
                        break

                    # Inter-sentence pause.
                    pause_ms = settings.TTS_PAUSE_MS
                    if pause_ms > 0 and not self._stop_event.is_set():
                        self._write_silence(stream, sr, pause_ms / 1000.0)

                    if nxt[1] != sr:
                        # Sample-rate change between clips is rare. Reopen.
                        stream.stop()
                        stream.close()
                        sr = nxt[1]
                        block_frames = max(1, int(sr * 0.05))
                        stream = sd.OutputStream(
                            samplerate=sr,
                            channels=2,
                            dtype="int16",
                            device=self.output_device,
                        )
                        stream.start()
                        self._write_silence(stream, sr, 0.05)
                    clip = nxt
        except Exception as e:
            logger.warning("TTS streaming playback error: %s", e)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
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
                if hasattr(self._voice, "synthesize_wav"):
                    syn_config = self._synthesis_config()
                    self._voice.synthesize_wav(text, wav, syn_config=syn_config)
                else:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.piper_sample_rate)
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
        if pcm.size == 0:
            logger.warning("Piper produced no audio for %r", text[:60])
        return pcm, sr

    def _synthesis_config(self):
        """Build a Piper synthesis config when the installed API supports it."""
        try:
            from piper.config import SynthesisConfig

            return SynthesisConfig(length_scale=self.length_scale)
        except Exception:
            return None

    def _play(self, clip: Clip) -> None:
        pcm, sr = clip
        with self._playback_lock:
            if self._stop_event.is_set():
                return
            try:
                audio = self._stereo_pcm(pcm)
                duration = audio.shape[0] / max(sr, 1)
                logger.info(
                    "Playing TTS clip: %.2fs @ %d Hz via %s",
                    duration,
                    sr,
                    describe_device(self.output_device, "output"),
                )

                block_frames = max(1, int(sr * 0.05))
                with sd.OutputStream(
                    samplerate=sr,
                    channels=2,
                    dtype="int16",
                    device=self.output_device,
                ) as stream:
                    for start in range(0, audio.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(audio[start : start + block_frames])
            except Exception as e:
                logger.warning("Playback error: %s", e)

    @staticmethod
    def _stereo_pcm(pcm: np.ndarray) -> np.ndarray:
        """Return 2-channel int16 PCM for predictable headphone playback."""
        mono = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if mono.size == 0:
            return np.zeros((0, 2), dtype=np.int16)
        return np.column_stack((mono, mono)).astype(np.int16, copy=False)

    @staticmethod
    def _apply_fade_in(audio: np.ndarray, sr: int, ms: float = 4.0) -> np.ndarray:
        """Linear ramp from 0 to full amplitude across the first ``ms``."""
        n = audio.shape[0]
        if n == 0:
            return audio
        fade = min(n, max(1, int(sr * ms / 1000.0)))
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32).reshape(-1, 1)
        out = audio.copy()
        out[:fade] = (out[:fade].astype(np.float32) * ramp).astype(np.int16)
        return out

    @staticmethod
    def _apply_fade_out(audio: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
        """Linear ramp from full amplitude to 0 across the last ``ms``."""
        n = audio.shape[0]
        if n == 0:
            return audio
        fade = min(n, max(1, int(sr * ms / 1000.0)))
        ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32).reshape(-1, 1)
        out = audio.copy()
        out[-fade:] = (out[-fade:].astype(np.float32) * ramp).astype(np.int16)
        return out

    @staticmethod
    def _write_silence(stream: sd.OutputStream, sr: int, duration_s: float) -> None:
        """Write a brief block of zeros to ``stream`` (driver underrun guard)."""
        n = max(0, int(sr * duration_s))
        if n == 0:
            return
        silence = np.zeros((n, 2), dtype=np.int16)
        try:
            stream.write(silence)
        except Exception as e:
            logger.debug("Silence write failed (likely closing stream): %s", e)
