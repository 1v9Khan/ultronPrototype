"""InjectableCapture -- a drop-in replacement for AudioCapture that feeds audio
from the test harness instead of the microphone, so battery-command WAVs traverse
the EXACT live capture path (wake word -> pre-roll ring -> VAD -> whisper) just as
if spoken into the mic. Runtime capture.py is UNTOUCHED: the harness swaps
``orchestrator.audio`` with this before ``run()``.

Behaviour: a silent "mic" by default (returns real-time-paced silence frames so
the wake loop stays alive and never trips the capture-stall watchdog), until the
harness ``feed_pcm(...)`` a command's audio -- those frames are then served in
order (still real-time paced), the wake word fires mid-stream, and the trailing
silence lets the VAD close the utterance. Format matches the mic exactly:
float32 mono [-1,1], 16 kHz, 256-sample blocks.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from kenning.audio.capture import AudioCapture


class InjectableCapture(AudioCapture):
    def __init__(self, *args, realtime: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pending: deque = deque()
        self._feed_lock = threading.Lock()
        self._realtime = realtime
        self._frame_s = self.blocksize / float(self.sample_rate)
        self._injected_frames = 0
        self._silence = np.zeros(self.blocksize, dtype=np.float32)

    # -- lifecycle: NO microphone --------------------------------------------
    def start(self) -> None:  # noqa: D401
        return  # never open a real input stream

    def stop(self) -> None:
        return

    # -- harness feed ---------------------------------------------------------
    def feed_pcm(self, pcm: np.ndarray) -> None:
        """Enqueue a command's PCM (float32 [-1,1] OR int16) @ 16 kHz mono.
        Split into exact blocksize frames (zero-padding the last)."""
        a = np.asarray(pcm).reshape(-1)
        if a.dtype == np.int16:
            a = a.astype(np.float32) / 32768.0
        else:
            a = a.astype(np.float32)
        frames = []
        n = self.blocksize
        for i in range(0, len(a), n):
            blk = a[i:i + n]
            if len(blk) < n:
                blk = np.concatenate([blk, np.zeros(n - len(blk), dtype=np.float32)])
            frames.append(blk)
        with self._feed_lock:
            self._pending.extend(frames)
            self._injected_frames += len(frames)

    def pending(self) -> int:
        with self._feed_lock:
            return len(self._pending)

    def drain(self) -> None:
        # Only clear the legacy queue; NEVER drop the harness's pending feed
        # (the orchestrator drains at each wait-for-wake entry, and we must keep
        # a just-fed command intact).
        with self._queue.mutex:
            self._queue.queue.clear()

    # -- consumer API: serve fed frames, else paced silence ------------------
    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        with self._feed_lock:
            frame = self._pending.popleft() if self._pending else None
        if self._realtime:
            time.sleep(self._frame_s)        # ~16 ms -- real mic cadence
        return frame if frame is not None else self._silence.copy()
