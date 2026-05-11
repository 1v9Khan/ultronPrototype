"""XTTS v2 + v3 Ultron filter TTS engine (drop-in replacement for Piper+RVC).

Architecture:

    main venv (this module)            isolated XTTS venv
    --------------------               -------------------
    XttsV3Speech                <-->   xtts_server.py (FastAPI)
        speak_stream(...)              POST /synthesize
        _synthesize(text) ----HTTP---> XTTS streaming inference
                              <-PCM--  v3 filter (this venv)

The XTTS server runs as a subprocess in its own Python venv (the
``.venv-xtts`` next to the audio prep). HTTP keeps the venvs decoupled
because Coqui TTS's deps (transformers 4.x pinned, hydra 1.3, omegaconf
2.3) conflict with what the main Ultron venv needs (older omegaconf
that fairseq 0.12.2 wants for the legacy RVC path).

Latency:
- XTTS streaming TTFT (model only): ~234 ms (benchmarked 2026-05-10)
- Through HTTP: ~375 ms TTFB (60 ms of asyncio + threadpool overhead)
- v3 filter at runtime: ~10-30 ms per sentence
- Composite first-audio-byte: ~400 ms

This is competitive with the legacy Piper+RVC path (~313 ms TTS synth
median) at much higher voice quality.
"""

from __future__ import annotations

import io
import json
import logging
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional, Tuple

import numpy as np
import sounddevice as sd

from config import settings
from ultron.audio.devices import describe_device, resolve_device
from ultron.tts.ultron_filter import apply_filter as apply_ultron_filter
from ultron.utils.logging import get_logger

logger = get_logger("tts.xtts_v3")

# Re-export the same Clip / ClipItem contract that the legacy Piper
# pipeline uses. The orchestrator's playback path consumes ClipItem
# tuples, so we honour that contract verbatim.
Clip = Tuple[np.ndarray, int]


class ClipItem(NamedTuple):
    audio: np.ndarray
    sample_rate: int
    is_known_last: bool = False


# Same generous timeout as the Piper+RVC path's playback queue (matches
# ultron.tts.speech._QUEUE_GET_TIMEOUT_SECONDS so downstream playback
# behaviour is consistent).
_QUEUE_GET_TIMEOUT_SECONDS = 60.0

# How long to wait for the XTTS server's /healthz to come up. Cold
# loads hit ~25 s for model + warmup; we add headroom for slower disks
# and first-run model downloads.
_SERVER_STARTUP_TIMEOUT_S = 180.0
_SERVER_HEALTHZ_POLL_INTERVAL_S = 0.5


class XttsServerStartError(RuntimeError):
    """Raised when the XTTS server subprocess can't be started."""


class XttsSynthError(RuntimeError):
    """Raised when a synthesis HTTP call fails (caller decides whether
    to fall back to silent clip vs propagate)."""


def _find_free_port() -> int:
    """Bind to port 0 to let the OS assign a free port, then close."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class XttsV3Speech:
    """XTTS v2 streaming TTS with v3 Ultron post-filter.

    Drop-in replacement for ``ultron.tts.speech.TextToSpeech``. Same
    public surface (``speak``, ``speak_stream``, ``warmup``, ``stop``)
    so the orchestrator can swap engines via config without touching
    the playback path.
    """

    def __init__(
        self,
        *,
        server_python: Optional[Path] = None,
        server_script: Optional[Path] = None,
        reference_audio: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        flush_chars: str = settings.TTS_SENTENCE_FLUSH_CHARS,
        filter_preset: str = "v3_heavy",
        filter_tail_silence_ms: float = 200.0,
        speed: Optional[float] = None,
        rvc=None,  # accepted-but-ignored for legacy ctor compat
    ) -> None:
        # Resolve paths via config when not explicitly passed. Defaults
        # point at the layout established in the audio prep work.
        from ultron.config import get_config, resolve_path
        cfg = get_config()
        xtts_cfg = getattr(cfg.tts, "xtts_v3", None)

        if server_python is None:
            sp = (xtts_cfg.server_python if xtts_cfg else None) or \
                "ultronVoiceAudio/.venv-xtts/Scripts/python.exe"
            server_python = resolve_path(sp)
        if server_script is None:
            ss = (xtts_cfg.server_script if xtts_cfg else None) or \
                "ultronVoiceAudio/scripts/xtts_server.py"
            server_script = resolve_path(ss)
        if reference_audio is None:
            ra = (xtts_cfg.reference_audio if xtts_cfg else None) or \
                "ultronVoiceAudio/Ultron_vocals_mono_v1.wav"
            reference_audio = resolve_path(ra)

        if not Path(server_python).is_file():
            raise XttsServerStartError(
                f"XTTS server Python not found at {server_python}. "
                f"Did you create the .venv-xtts venv?"
            )
        if not Path(server_script).is_file():
            raise XttsServerStartError(
                f"XTTS server script not found at {server_script}."
            )
        if not Path(reference_audio).is_file():
            raise XttsServerStartError(
                f"XTTS reference audio not found at {reference_audio}."
            )

        self.server_python = Path(server_python)
        self.server_script = Path(server_script)
        self.reference_audio = Path(reference_audio)
        self.host = host
        self.port = int(port) if port is not None else _find_free_port()
        self.base_url = f"http://{self.host}:{self.port}"

        self.flush_chars = set(flush_chars)
        self.filter_preset = filter_preset
        self.filter_tail_silence_ms = float(filter_tail_silence_ms)
        # Cadence: passed to XTTS ``inference_stream(speed=...)`` on the
        # server side. Adjusts synthesis duration tokens; does NOT touch
        # the post-synthesis v3 filter chain.
        if speed is None:
            speed = float(xtts_cfg.speed) if xtts_cfg is not None else 1.0
        self._synth_speed = float(speed)
        # 2026-05-11 chunk-streaming investigation: was prototyped but
        # not shipped. Pedalboard's PitchShift (Rubber Band offline
        # mode) buffers ~25 000 samples internally with ``reset=False``,
        # which means streaming chunks through the v3_heavy chain
        # produces zero output until the buffer fills (and the buffered
        # audio can't be cleanly drained). Per-chunk ``reset=True``
        # works but produces ~125 % RMS divergence at chunk boundaries
        # -- audible artifacts. The v3 chain order is user-locked, so
        # moving PitchShift to the end (which would unblock streaming)
        # is out of scope. The audio is still streamed at the HTTP
        # level (server pushes PCM chunks as they're synthesised), but
        # the client accumulates the full sentence before filter
        # processing. See docs/codebase_structure.md for the
        # investigation notes.

        # Match the Piper path's output device + lock behaviour so the
        # orchestrator + barge-in handling stay uniform.
        self.output_device = resolve_device(settings.AUDIO_OUTPUT_DEVICE, "output")
        self._stop_event = threading.Event()
        self._playback_lock = threading.Lock()

        # Server lifecycle.
        self._server_proc: Optional[subprocess.Popen] = None
        self._sample_rate: int = 24000  # XTTS native; confirmed via /info after start
        self._start_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        """Spawn the XTTS server subprocess and wait for /healthz."""
        argv = [
            str(self.server_python),
            "-u",
            str(self.server_script),
            "--host", self.host,
            "--port", str(self.port),
            "--reference", str(self.reference_audio),
        ]
        logger.info(
            "Starting XTTS server (port=%d, ref=%s)",
            self.port,
            self.reference_audio.name,
        )
        try:
            # Inherit stderr to the parent so we see crashes; pipe
            # stdout to /dev/null since the server is verbose on
            # uvicorn startup.
            self._server_proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
            )
        except FileNotFoundError as e:
            raise XttsServerStartError(f"Failed to spawn XTTS server: {e}") from e

        # Poll /healthz until ready or timeout.
        deadline = time.monotonic() + _SERVER_STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._server_proc.poll() is not None:
                code = self._server_proc.returncode
                self._server_proc = None
                raise XttsServerStartError(
                    f"XTTS server exited during startup (code {code})."
                )
            try:
                req = urllib.request.Request(self.base_url + "/healthz")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("ok") and payload.get("speaker_cached"):
                        # Confirm sample rate via /info.
                        try:
                            with urllib.request.urlopen(
                                self.base_url + "/info", timeout=2.0
                            ) as ir:
                                info = json.loads(ir.read().decode("utf-8"))
                                self._sample_rate = int(info.get("sample_rate", 24000))
                        except Exception:
                            pass
                        logger.info(
                            "XTTS server ready in %.1fs (sample_rate=%d)",
                            _SERVER_STARTUP_TIMEOUT_S - (deadline - time.monotonic()),
                            self._sample_rate,
                        )
                        return
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                pass  # not ready yet
            time.sleep(_SERVER_HEALTHZ_POLL_INTERVAL_S)

        # Timeout
        self._stop_server_subprocess()
        raise XttsServerStartError(
            f"XTTS server did not become ready within {_SERVER_STARTUP_TIMEOUT_S}s."
        )

    def _stop_server_subprocess(self) -> None:
        """Best-effort: try graceful /shutdown, then SIGTERM/SIGKILL."""
        if self._server_proc is None:
            return
        try:
            req = urllib.request.Request(
                self.base_url + "/shutdown", method="POST"
            )
            urllib.request.urlopen(req, timeout=1.0).close()
        except Exception:
            pass
        try:
            self._server_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
        finally:
            self._server_proc = None

    def __enter__(self) -> "XttsV3Speech":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        self._stop_server_subprocess()

    # ------------------------------------------------------------------
    # Public API (mirrors TextToSpeech)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Interrupt any in-progress playback (signal only; doesn't stop the server)."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass

    def speak(self, text: str) -> None:
        """Synthesize + play ``text`` synchronously."""
        if not text.strip():
            return
        self._stop_event.clear()
        clip = self._synthesize(text)
        if clip[0].size > 0 and not self._stop_event.is_set():
            self._play(clip)

    def warmup(self, text: str = "Online.") -> None:
        """Touch the server with a tiny request so the first real
        utterance doesn't pay any cold-cache cost."""
        if not text.strip():
            return
        t0 = time.monotonic()
        try:
            self._synthesize(text)
            logger.info("XTTS warmup complete in %.0fms", (time.monotonic() - t0) * 1000)
        except Exception as e:
            logger.warning("XTTS warmup skipped: %s", e)

    def speak_stream(self, fragments: Iterable[str]) -> None:
        """Consume token fragments and play sentence-by-sentence.

        Same producer-signaled lookahead playback contract as
        :meth:`ultron.tts.speech.TextToSpeech.speak_stream` -- queues
        :class:`ClipItem` tuples onto an internal audio queue and
        plays each clip immediately on receipt without blocking on
        the next clip first.
        """
        self._stop_event.clear()

        try:
            from ultron.config import get_config
            tts_cfg = get_config().tts
            spec_open = tts_cfg.speculative_stream_open_enabled
            # 2026-05-11 SR-mismatch fix: ``tts.speculative_stream_sample_rate``
            # is tuned for the legacy Piper+RVC stack (48 kHz). The XTTS
            # engine produces 24 kHz natively. Reading the global field
            # here forced a close-and-reopen on every turn (50-100 ms
            # wasted) when xtts_v3 was active. The engine knows its own
            # native rate, so use it directly -- the legacy speech.py
            # path is unchanged and still uses the config field.
            spec_sr = self._sample_rate
            low_latency = tts_cfg.output_low_latency_mode
        except Exception:
            spec_open = False
            spec_sr = self._sample_rate
            low_latency = False

        audio_q: queue.Queue[Optional[ClipItem]] = queue.Queue(maxsize=8)
        workers: list[threading.Thread] = []

        def synth_worker() -> None:
            try:
                self._run_synth_loop(
                    fragments=fragments,
                    push=lambda item: audio_q.put(item),
                )
            except Exception as e:
                logger.error("XTTS synth worker error: %s", e)
            finally:
                audio_q.put(None)

        worker = threading.Thread(target=synth_worker, daemon=True, name="xtts-synth")
        worker.start()
        workers.append(worker)

        sr: int = spec_sr if spec_open else self._sample_rate
        block_frames = max(1, int(sr * 0.05))
        stream: Optional[sd.OutputStream] = None
        first_item: Optional[ClipItem] = None

        try:
            with self._playback_lock:
                if self._stop_event.is_set():
                    return

                if spec_open:
                    stream = self._open_output_stream(sr, low_latency)
                    stream.start()
                    self._write_silence(stream, sr, 0.05)

                try:
                    first_item = audio_q.get(timeout=_QUEUE_GET_TIMEOUT_SECONDS)
                except queue.Empty:
                    logger.warning("XTTS playback queue starved before first clip")
                    return
                if first_item is None:
                    return

                actual_sr = first_item.sample_rate
                if not spec_open:
                    sr = actual_sr
                    block_frames = max(1, int(sr * 0.05))
                    stream = self._open_output_stream(sr, low_latency)
                    stream.start()
                    self._write_silence(stream, sr, 0.05)
                elif actual_sr != sr:
                    logger.info("XTTS speculative SR %d != actual %d; reopening", sr, actual_sr)
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

                item = first_item
                while True:
                    audio = self._stereo_pcm(item.audio)
                    edge_ms = settings.TTS_EDGE_FADE_MS
                    if edge_ms > 0:
                        audio = self._apply_fade_in(audio, sr, ms=edge_ms)
                        audio = self._apply_fade_out(audio, sr, ms=edge_ms)

                    for start in range(0, audio.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(audio[start : start + block_frames])

                    if item.is_known_last:
                        self._write_silence(stream, sr, 0.05)
                        break

                    pause_ms = settings.TTS_PAUSE_MS
                    if pause_ms > 0 and not self._stop_event.is_set():
                        self._write_silence(stream, sr, pause_ms / 1000.0)

                    try:
                        nxt = audio_q.get(timeout=_QUEUE_GET_TIMEOUT_SECONDS)
                    except queue.Empty:
                        logger.warning(
                            "XTTS playback waited %.0fs without next clip; ending",
                            _QUEUE_GET_TIMEOUT_SECONDS,
                        )
                        self._write_silence(stream, sr, 0.05)
                        break

                    if nxt is None:
                        self._write_silence(stream, sr, 0.05)
                        break

                    if nxt.sample_rate != sr:
                        stream.stop()
                        stream.close()
                        sr = nxt.sample_rate
                        block_frames = max(1, int(sr * 0.05))
                        stream = self._open_output_stream(sr, low_latency)
                        stream.start()
                        self._write_silence(stream, sr, 0.05)
                    item = nxt
        except Exception as e:
            logger.warning("XTTS streaming playback error: %s", e)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            for w in workers:
                w.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_synth_loop(
        self,
        *,
        fragments: Iterable[str],
        push: Callable[[ClipItem], None],
    ) -> None:
        """Walk fragments, synth on flush chars, push ClipItems."""
        buffer: list[str] = []
        for frag in fragments:
            if self._stop_event.is_set():
                break
            if not frag:
                continue
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
                    pcm, sr = self._synthesize(sentence)
                    if pcm.size > 0:
                        push(ClipItem(pcm, sr, is_known_last=False))

        tail = "".join(buffer).strip()
        if tail and not self._stop_event.is_set():
            pcm, sr = self._synthesize(tail)
            if pcm.size > 0:
                push(ClipItem(pcm, sr, is_known_last=False))

    def _synthesize(self, text: str) -> Clip:
        """Synthesize one sentence: HTTP → assemble PCM → v3 filter → (pcm, sr)."""
        t0 = time.monotonic()
        try:
            pcm_i16 = self._http_synthesize(text)
        except Exception as e:
            logger.error("XTTS server synth failed for %r: %s", text[:60], e)
            from ultron.errors import PiperSynthesisError  # closest typed error
            from ultron.resilience import get_error_log
            get_error_log().record(
                PiperSynthesisError(
                    f"XTTS server synth failed: {e}",
                    context={"text_preview": text[:60], "text_chars": len(text)},
                    recovery="returned silent clip; orchestrator falls back to terminal print",
                ),
                dependency="xtts_server",
            )
            return np.zeros(0, dtype=np.int16), self._sample_rate

        if pcm_i16.size == 0:
            logger.warning("XTTS produced no audio for %r", text[:60])
            return pcm_i16, self._sample_rate

        # Apply v3 filter. Convert int16 -> float32 [-1, 1], filter,
        # convert back. The filter pads tail_silence_ms of trailing
        # zeros so reverb decay isn't clipped at the buffer end.
        pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
        try:
            filtered_f32 = apply_ultron_filter(
                pcm_f32,
                self._sample_rate,
                preset=self.filter_preset,
                tail_silence_ms=self.filter_tail_silence_ms,
            )
        except Exception as e:
            logger.warning("Ultron filter failed (using raw PCM): %s", e)
            filtered_f32 = pcm_f32

        # Convert back to int16 with clipping.
        np.clip(filtered_f32, -1.0, 1.0, out=filtered_f32)
        out_pcm = (filtered_f32 * 32767.0).astype(np.int16)
        logger.debug(
            "XTTS+v3: %d chars -> %.2fs audio @ %d Hz in %.0fms",
            len(text),
            out_pcm.size / max(self._sample_rate, 1),
            self._sample_rate,
            (time.monotonic() - t0) * 1000,
        )
        return out_pcm, self._sample_rate

    def _http_synthesize(self, text: str) -> np.ndarray:
        """POST /synthesize, accumulate streamed PCM, return int16 array."""
        body = json.dumps(
            {"text": text, "language": "en", "speed": self._synth_speed}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/synthesize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            sr_header = resp.headers.get("X-Sample-Rate")
            if sr_header:
                self._sample_rate = int(sr_header)
            chunks: list[bytes] = []
            while True:
                c = resp.read(8192)
                if not c:
                    break
                chunks.append(c)
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        raw = b"".join(chunks)
        return np.frombuffer(raw, dtype=np.int16).copy()


    def _play(self, clip: Clip) -> None:
        """Single-shot playback. Same shape as TextToSpeech._play."""
        pcm, sr = clip
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
                    "Playing XTTS+v3 clip: %.2fs @ %d Hz via %s",
                    duration, sr, describe_device(self.output_device, "output"),
                )
                block_frames = max(1, int(sr * 0.05))
                with self._open_output_stream(sr, low_latency) as stream:
                    for start in range(0, audio.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(audio[start : start + block_frames])
            except Exception as e:
                logger.warning("Playback error: %s", e)

    def _open_output_stream(self, sample_rate: int, low_latency: bool) -> sd.OutputStream:
        kwargs: dict = {
            "samplerate": sample_rate,
            "channels": 2,
            "dtype": "int16",
            "device": self.output_device,
        }
        if low_latency:
            kwargs["latency"] = "low"
        return sd.OutputStream(**kwargs)

    @staticmethod
    def _stereo_pcm(pcm: np.ndarray) -> np.ndarray:
        mono = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if mono.size == 0:
            return np.zeros((0, 2), dtype=np.int16)
        return np.column_stack((mono, mono)).astype(np.int16, copy=False)

    @staticmethod
    def _apply_fade_in(audio: np.ndarray, sr: int, ms: float = 4.0) -> np.ndarray:
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
        n = max(0, int(sr * duration_s))
        if n == 0:
            return
        silence = np.zeros((n, 2), dtype=np.int16)
        try:
            stream.write(silence)
        except Exception as e:
            logger.debug("Silence write failed (likely closing stream): %s", e)
