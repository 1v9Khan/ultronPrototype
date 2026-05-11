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
from typing import Callable, Iterable, NamedTuple, Optional, Tuple

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


class ClipItem(NamedTuple):
    """A queued audio clip with a producer-signaled "is this the last?" hint.

    Pushed by synth workers onto the playback queues (``piper_q`` and
    ``audio_q``). The producer-signaled lookahead pattern lets the
    playback path play each clip IMMEDIATELY on receipt without first
    blocking on the next clip to determine "is this last?". The old
    play-after-peek pattern would delay the first clip (commonly the
    web-search ack phrase) until the next clip arrived OR a 10 s
    timeout fired -- which made the ack arrive AFTER the response
    instead of before it.

    Conventions:

    * ``is_known_last=False`` (default): producer doesn't know if this
      is the last. Playback plays it, then waits for the next item or
      end-of-stream sentinel (``None``).
    * ``is_known_last=True``: producer knows this is the final clip.
      Playback plays it with the standard tail silence and exits
      without waiting for a next item.
    * ``None`` sentinel pushed onto the queue marks end-of-stream.
      The most recently received clip is treated as final; tail
      silence is written and playback exits.
    """

    audio: np.ndarray
    sample_rate: int
    is_known_last: bool = False


# How long playback / RVC stage waits for the next clip from upstream
# before giving up. The previous 10 s value timed out during long web
# searches (3 Brave + 2 Jina = ~10 s wall, plus LLM TTFT) which killed
# audio playback mid-response. 60 s is generous enough to span the
# longest realistic search + LLM stall without burning excess CPU on
# spinning waits.
_QUEUE_GET_TIMEOUT_SECONDS = 60.0


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

        Three optional latency optimisations gated by config (all default ON):

        1. ``tts.pipeline_parallel_enabled`` — split synthesis into two
           worker stages (Piper then RVC) connected by a bounded queue,
           so Piper N+1 runs in parallel with RVC N. Saves ~Piper time
           (~50 ms) per subsequent sentence on multi-sentence responses.
           When false, the legacy single-worker path runs.

        2. ``tts.speculative_stream_open_enabled`` — open the audio
           output stream at the expected RVC sample rate while the
           first sentence is still synthesising. Saves ~20-30 ms on
           first-sentence audio. The existing sample-rate-mismatch
           close-and-reopen path catches the rare case where actual
           output rate differs from the expected one.

        3. ``tts.output_low_latency_mode`` — pass ``latency='low'`` to
           ``sd.OutputStream`` so PortAudio asks the host API for the
           smallest acceptable buffer. Saves 30-100 ms of OS-level
           audio buffering.

        Voice character is unchanged: same Piper buffer feeds the same
        RVC; only the stage timing overlaps. Same fade-in/fade-out,
        same inter-sentence pause, same pre-roll silence.
        """
        self._stop_event.clear()
        from ultron.config import get_config

        try:
            tts_cfg = get_config().tts
            pipeline_parallel = tts_cfg.pipeline_parallel_enabled
            spec_open = tts_cfg.speculative_stream_open_enabled
            spec_sr = tts_cfg.speculative_stream_sample_rate
            low_latency = tts_cfg.output_low_latency_mode
        except Exception:
            # Defensive: tests may construct TextToSpeech without a
            # full config. Fall back to legacy behaviour.
            pipeline_parallel = False
            spec_open = False
            spec_sr = settings.TTS_OUTPUT_SAMPLE_RATE
            low_latency = False

        # Queue carries ClipItem (audio, sample_rate, is_known_last)
        # tuples; ``None`` is the end-of-stream sentinel. Playback uses
        # the producer-signaled is_known_last flag (or the sentinel) to
        # avoid the play-after-peek pattern that delayed first-clip
        # playback in the legacy implementation.
        audio_q: queue.Queue[Optional[ClipItem]] = queue.Queue(maxsize=8)
        # Track threads so we can join cleanly on every exit path.
        workers: list[threading.Thread] = []

        if pipeline_parallel and self.rvc is not None:
            # Two-stage pipeline: piper_worker -> piper_q -> rvc_worker
            # -> audio_q. The bounded piper_q (maxsize=2) gives natural
            # backpressure if RVC falls behind without ever growing
            # unbounded.
            piper_q: queue.Queue[Optional[ClipItem]] = queue.Queue(maxsize=2)

            def piper_worker() -> None:
                try:
                    self._run_synth_loop(
                        fragments=fragments,
                        push=lambda item: piper_q.put(item),
                        synth_fn=self._piper_synth_only,
                    )
                except Exception as e:
                    logger.error("TTS piper worker error: %s", e)
                finally:
                    # End-of-stream sentinel ALWAYS goes through, even
                    # on exception. Downstream waits indefinitely for
                    # this rather than timing out, so we must push it.
                    piper_q.put(None)

            def rvc_worker() -> None:
                try:
                    while not self._stop_event.is_set():
                        try:
                            piper_item = piper_q.get(
                                timeout=_QUEUE_GET_TIMEOUT_SECONDS
                            )
                        except queue.Empty:
                            # Generous timeout; only fires on a real
                            # producer hang (e.g. piper_worker dead
                            # without finally executing). Treat as
                            # end-of-stream so downstream wraps up.
                            logger.warning(
                                "TTS RVC stage waited %.0fs for piper "
                                "without a clip; treating as end-of-stream",
                                _QUEUE_GET_TIMEOUT_SECONDS,
                            )
                            break
                        if piper_item is None:
                            break
                        if piper_item.audio.size == 0:
                            continue
                        out_clip = self._apply_rvc(
                            (piper_item.audio, piper_item.sample_rate)
                        )
                        if out_clip[0].size > 0:
                            audio_q.put(
                                ClipItem(
                                    out_clip[0],
                                    out_clip[1],
                                    piper_item.is_known_last,
                                )
                            )
                except Exception as e:
                    logger.error("TTS rvc worker error: %s", e)
                finally:
                    audio_q.put(None)

            piper_t = threading.Thread(
                target=piper_worker, daemon=True, name="tts-piper"
            )
            rvc_t = threading.Thread(
                target=rvc_worker, daemon=True, name="tts-rvc"
            )
            piper_t.start()
            rvc_t.start()
            workers.extend([piper_t, rvc_t])
        else:
            # Legacy single-worker path: Piper + RVC happen serially
            # inside _synthesize. Used when pipeline_parallel is off or
            # RVC isn't wired.
            def synth_worker() -> None:
                try:
                    self._run_synth_loop(
                        fragments=fragments,
                        push=lambda item: audio_q.put(item),
                        synth_fn=self._synthesize,
                    )
                except Exception as e:
                    logger.error("TTS worker error: %s", e)
                finally:
                    audio_q.put(None)

            worker = threading.Thread(
                target=synth_worker, daemon=True, name="tts-synth"
            )
            worker.start()
            workers.append(worker)

        sr: int = spec_sr if (spec_open and self.rvc is not None) else (
            spec_sr if spec_open else settings.TTS_OUTPUT_SAMPLE_RATE
        )
        block_frames = max(1, int(sr * 0.05))
        stream: Optional[sd.OutputStream] = None
        first_clip: Optional[Clip] = None

        try:
            with self._playback_lock:
                if self._stop_event.is_set():
                    return

                if spec_open:
                    # Speculative open: stream is up while first sentence
                    # synthesises. The 50 ms pre-roll silence buys time
                    # for the device to settle before real samples land.
                    stream = self._open_output_stream(sr, low_latency)
                    stream.start()
                    self._write_silence(stream, sr, 0.05)
                    logger.info(
                        "TTS stream opened (speculative %s): %d Hz via %s",
                        "low-latency" if low_latency else "default",
                        sr,
                        describe_device(self.output_device, "output"),
                    )

                # Pull first clip; this blocks until first sentence is ready.
                try:
                    first_item = audio_q.get(
                        timeout=_QUEUE_GET_TIMEOUT_SECONDS
                    )
                except queue.Empty:
                    logger.warning("TTS playback queue starved before first clip")
                    return
                if first_item is None:
                    return

                actual_sr = first_item.sample_rate
                if not spec_open:
                    # Legacy: open the stream now that we know the rate.
                    sr = actual_sr
                    block_frames = max(1, int(sr * 0.05))
                    stream = self._open_output_stream(sr, low_latency)
                    stream.start()
                    self._write_silence(stream, sr, 0.05)
                    logger.info(
                        "TTS stream opened%s: %d Hz via %s",
                        " (low-latency)" if low_latency else "",
                        sr,
                        describe_device(self.output_device, "output"),
                    )
                elif actual_sr != sr:
                    # Speculative SR didn't match. Close and reopen at
                    # the actual rate. This is the rare-case fallback;
                    # configure ``speculative_stream_sample_rate`` to
                    # the right value to avoid it.
                    logger.info(
                        "TTS speculative SR %d != actual %d; reopening",
                        sr, actual_sr,
                    )
                    if stream is not None:
                        try:
                            stream.stop()
                            stream.close()
                        except Exception:
                            pass
                    sr = actual_sr
                    block_frames = max(1, int(sr * 0.05))
                    stream = self._open_output_stream(sr, low_latency)
                    stream.start()
                    self._write_silence(stream, sr, 0.05)

                # Producer-signaled lookahead: play each clip
                # IMMEDIATELY on receipt, then ask for the next. The
                # producer signals "is this the last?" either via the
                # ClipItem.is_known_last flag or by pushing the
                # end-of-stream sentinel (None). This is the ack-first
                # path: the web-search ack clip arrives, plays, THEN
                # the search begins -- instead of waiting for the
                # search result clip to determine "is the ack last?".
                item = first_item
                while True:
                    audio = self._stereo_pcm(item.audio)
                    # Tiny fade on every clip edge so the inter-clip
                    # silence gap doesn't introduce a click.
                    edge_ms = settings.TTS_EDGE_FADE_MS
                    if edge_ms > 0:
                        audio = self._apply_fade_in(audio, sr, ms=edge_ms)
                        audio = self._apply_fade_out(audio, sr, ms=edge_ms)

                    # Play the current clip BEFORE asking for next. The
                    # block-write loop respects the stop_event so a
                    # barge-in interrupt still terminates promptly.
                    for start in range(0, audio.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(audio[start : start + block_frames])

                    if item.is_known_last:
                        # Producer told us this was the final clip.
                        self._write_silence(stream, sr, 0.05)
                        break

                    # Inter-sentence pause comes before next clip.
                    pause_ms = settings.TTS_PAUSE_MS
                    if pause_ms > 0 and not self._stop_event.is_set():
                        self._write_silence(stream, sr, pause_ms / 1000.0)

                    # Now block for the next clip. Generous timeout so
                    # long generator stalls (web search, slow LLM TTFT)
                    # don't kill playback mid-stream.
                    try:
                        nxt = audio_q.get(
                            timeout=_QUEUE_GET_TIMEOUT_SECONDS
                        )
                    except queue.Empty:
                        # Real upstream hang. We already played the
                        # inter-sentence pause; finish with tail
                        # silence so the driver flushes cleanly.
                        logger.warning(
                            "TTS playback waited %.0fs for next clip "
                            "without one; treating as end-of-stream",
                            _QUEUE_GET_TIMEOUT_SECONDS,
                        )
                        self._write_silence(stream, sr, 0.05)
                        break

                    if nxt is None:
                        # End-of-stream sentinel: previous was final.
                        self._write_silence(stream, sr, 0.05)
                        break

                    if nxt.sample_rate != sr:
                        # Sample-rate change between clips is rare. Reopen.
                        stream.stop()
                        stream.close()
                        sr = nxt.sample_rate
                        block_frames = max(1, int(sr * 0.05))
                        stream = self._open_output_stream(sr, low_latency)
                        stream.start()
                        self._write_silence(stream, sr, 0.05)
                    item = nxt
        except Exception as e:
            logger.warning("TTS streaming playback error: %s", e)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            for w in workers:
                w.join(timeout=2.0)

    # --- streaming helpers (extracted for the parallel pipeline) -------------

    def _run_synth_loop(
        self,
        *,
        fragments: Iterable[str],
        push: Callable[[ClipItem], None],
        synth_fn: Callable[[str], Clip],
    ) -> None:
        """Walk ``fragments`` token-by-token, synthesise on flush chars,
        and push each completed clip via ``push`` as a :class:`ClipItem`.

        Used by both the legacy single-worker path and the piper
        worker in the parallel pipeline. Each push carries
        ``is_known_last=False``; the producer cannot know in advance
        whether the next fragment yields another sentence, since the
        upstream generator may still produce tokens. End-of-stream is
        signalled by the worker's ``finally`` block pushing ``None``
        onto the queue (handled by the worker, not this loop).

        Args:
            fragments: iterable of LLM token strings.
            push: callable invoked with each non-empty :class:`ClipItem`.
            synth_fn: ``str -> Clip`` synthesiser. Either ``_synthesize``
                (legacy: Piper + RVC fused) or ``_piper_synth_only``
                (parallel: Piper alone, RVC happens in a downstream stage).
        """
        buffer: list[str] = []
        for frag in fragments:
            if self._stop_event.is_set():
                break
            if not frag:
                continue

            # A single LLM token may contain a flush character followed
            # by the start of the next word (e.g. ". Th" where "is"
            # arrives in the next token). Naively appending the whole
            # token then flushing tears words in half. Walk char-by-
            # char: text up to+including the flush char closes the
            # current sentence; text after it opens the next.
            remaining = frag
            while remaining:
                flush_pos = next(
                    (i for i, c in enumerate(remaining) if c in self.flush_chars),
                    -1,
                )
                if flush_pos == -1:
                    buffer.append(remaining)
                    break
                buffer.append(remaining[: flush_pos + 1])
                sentence = "".join(buffer).strip()
                buffer.clear()
                remaining = remaining[flush_pos + 1 :]
                if sentence:
                    pcm, sr = synth_fn(sentence)
                    if pcm.size > 0:
                        push(ClipItem(pcm, sr, is_known_last=False))

        tail = "".join(buffer).strip()
        if tail and not self._stop_event.is_set():
            pcm, sr = synth_fn(tail)
            if pcm.size > 0:
                push(ClipItem(pcm, sr, is_known_last=False))

    def _piper_synth_only(self, text: str) -> Clip:
        """Synthesise text with Piper alone (no RVC).

        Used by the parallel pipeline's piper_worker. Returns
        ``(int16 pcm, piper_sample_rate)``. The downstream RVC stage
        consumes this and does the conversion separately.
        """
        return self._piper_synth(text)

    def _apply_rvc(self, piper_clip: Clip) -> Clip:
        """Convert a Piper clip via RVC, falling back to raw Piper on error.

        Mirrors the RVC-failure handling baked into ``_synthesize`` so
        the parallel pipeline preserves the same fail-soft contract.
        """
        if self.rvc is None:
            return piper_clip
        pcm, sr = piper_clip
        if pcm.size == 0:
            return piper_clip
        try:
            return self.rvc.convert(pcm, sr)
        except Exception as e:
            logger.warning("RVC convert failed (using raw Piper): %s", e)
            from ultron.errors import RVCConversionError
            from ultron.resilience import get_error_log
            get_error_log().record(
                RVCConversionError(
                    f"RVC convert failed: {e}",
                    context={"sample_rate": int(sr), "pcm_samples": int(pcm.size)},
                    recovery="fell back to raw Piper output (no Ultron filter)",
                ),
                dependency="rvc",
            )
            return piper_clip

    def _open_output_stream(
        self, sample_rate: int, low_latency: bool
    ) -> sd.OutputStream:
        """Open a stereo int16 OutputStream at ``sample_rate``.

        ``low_latency=True`` passes ``latency='low'`` to PortAudio so
        the host API uses its smallest acceptable buffer (typically
        30-50 ms on Windows WASAPI shared mode vs the 100-200 ms
        default). Safe to leave on; PortAudio falls back gracefully on
        platforms / devices that don't honour the hint.
        """
        kwargs: dict = {
            "samplerate": sample_rate,
            "channels": 2,
            "dtype": "int16",
            "device": self.output_device,
        }
        if low_latency:
            kwargs["latency"] = "low"
        return sd.OutputStream(**kwargs)

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
                # Late-binding imports to avoid circular bootstrap dependency.
                from ultron.errors import RVCConversionError
                from ultron.resilience import get_error_log
                get_error_log().record(
                    RVCConversionError(
                        f"RVC convert failed: {e}",
                        context={"sample_rate": int(sr), "pcm_samples": int(pcm.size)},
                        recovery="fell back to raw Piper output (no Ultron filter)",
                    ),
                    dependency="rvc",
                )

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
            from ultron.errors import PiperSynthesisError
            from ultron.resilience import get_error_log
            get_error_log().record(
                PiperSynthesisError(
                    f"Piper synth failed: {e}",
                    context={"text_preview": text[:60], "text_chars": len(text)},
                    recovery="returned silent clip; orchestrator falls back to terminal print",
                ),
                dependency="piper_tts",
            )
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
        # Read low-latency preference from config; default true for the
        # speak() path too. Falls back to False when config isn't built
        # (test scenarios that bypass the loader).
        try:
            from ultron.config import get_config
            low_latency = get_config().tts.output_low_latency_mode
        except Exception:
            low_latency = False
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
                with self._open_output_stream(sr, low_latency) as stream:
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
