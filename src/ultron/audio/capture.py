"""Microphone capture.

A `sounddevice.InputStream` callback runs on a high-priority audio thread.
The callback's only job is to push the chunk onto a queue — anything heavier
risks underrun and dropouts. Consumers (VAD, wake word) pull from the queue
on their own threads.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from config import settings
from ultron.audio.devices import describe_device, resolve_device
from ultron.utils.logging import get_logger

logger = get_logger("audio.capture")


class AudioCaptureError(RuntimeError):
    """Raised when the input stream cannot be opened or recovered."""


class AudioCapture:
    """Continuous microphone capture into a thread-safe queue.

    Use as a context manager:

        with AudioCapture() as mic:
            chunk = mic.get_chunk(timeout=1.0)
    """

    def __init__(
        self,
        sample_rate: int = settings.SAMPLE_RATE,
        channels: int = settings.CHANNELS,
        blocksize: int = settings.BLOCKSIZE,
        device: Optional[str | int] = settings.AUDIO_DEVICE,
        max_queue_size: int = 256,
        input_gain_db: Optional[float] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.configured_device = device
        self.device: Optional[int] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._overrun_warned = False
        # 2026-05-09 audio-quality pass: pre-amp applied in the audio
        # callback. ``input_gain_db=None`` -> read from config (allows
        # tests to construct AudioCapture without a config singleton).
        if input_gain_db is None:
            try:
                from ultron.config import get_config
                input_gain_db = float(getattr(get_config().audio, "input_gain_db", 0.0))
            except Exception:
                input_gain_db = 0.0
        self.input_gain_db = float(input_gain_db)
        # Linear multiplier; cached so the audio thread doesn't recompute.
        # 0 dB -> 1.0 (no-op fast path).
        self._gain_linear = 1.0 if self.input_gain_db == 0.0 else float(
            10.0 ** (self.input_gain_db / 20.0)
        )

    # --- context manager -----------------------------------------------------

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Open the input stream and begin capturing."""
        with self._lock:
            if self._stream is not None:
                return
            try:
                self.device = resolve_device(self.configured_device, "input")
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=self.blocksize,
                    dtype=settings.DTYPE,
                    device=self.device,
                    callback=self._callback,
                )
                self._stream.start()
            except Exception as e:
                self._stream = None
                raise AudioCaptureError(f"Failed to open input stream: {e}") from e
            logger.info(
                "Audio capture started: %d Hz, %d ch, blocksize=%d, device=%s",
                self.sample_rate,
                self.channels,
                self.blocksize,
                describe_device(self.device, "input"),
            )

    def stop(self) -> None:
        """Stop and close the input stream. Safe to call twice."""
        with self._lock:
            if self._stream is None:
                return
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error closing audio stream: %s", e)
            finally:
                self._stream = None
            logger.info("Audio capture stopped")

    # --- consumer API --------------------------------------------------------

    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Pop the next captured chunk, or return None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard any pending chunks. Useful right before re-arming wake word."""
        with self._queue.mutex:
            self._queue.queue.clear()

    def qsize(self) -> int:
        return self._queue.qsize()

    # --- audio thread callback ----------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        """Runs on the audio thread. Must not block."""
        if status:
            # input_overflow / input_underflow / etc.
            if not self._overrun_warned:
                logger.warning("Audio status flag: %s", status)
                self._overrun_warned = True

        # Copy because sounddevice reuses the buffer.
        chunk = indata[:, 0].copy() if self.channels == 1 else indata.copy()
        # Apply pre-amp gain (audio-quality pass). Fast path skips the
        # multiply when gain is 0 dB. Float audio is in [-1, 1]; we
        # clip to that range to prevent wraparound on int16 conversion
        # downstream.
        if self._gain_linear != 1.0:
            chunk = chunk * self._gain_linear
            if self._gain_linear > 1.0:
                # Hard-clip to prevent distortion from over-gain. Soft
                # limiting would be smoother but adds a small per-block
                # CPU cost on the audio thread; clipping is one numpy
                # call and is acceptable for a single-mic prototype.
                np.clip(chunk, -1.0, 1.0, out=chunk)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest to make room — better than blocking the audio thread.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except queue.Empty:
                pass
