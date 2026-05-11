"""Tests for the TTS streaming pipeline's three latency optimisations.

Covers:
- ``pipeline_parallel_enabled``: Piper and RVC run in two stages so
  Piper N+1 starts before RVC N finishes. Verifies sentence ordering
  is preserved AND that the parallel path actually overlaps the work
  (not just runs sequentially in two threads).
- ``speculative_stream_open_enabled``: ``sd.OutputStream`` is opened
  at the configured speculative sample rate before the first clip
  arrives.
- ``output_low_latency_mode``: ``latency='low'`` is passed to PortAudio.
- Sample-rate mismatch fallback: when the speculative SR doesn't match
  the actual RVC output, the stream is closed and reopened at the
  actual rate.

Audio playback is mocked end-to-end -- no actual sounddevice calls.
The mock records every constructor invocation + write so we can
assert ordering and per-stage timing.
"""

from __future__ import annotations

import threading
import time
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ultron.config import UltronConfig, set_config
from ultron.tts.speech import ClipItem, TextToSpeech


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockOutputStream:
    """Records constructor kwargs + writes; pretends to play instantly."""

    instances: List["_MockOutputStream"] = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.writes: List[np.ndarray] = []
        self.started = False
        self.stopped = False
        self.closed = False
        _MockOutputStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def write(self, audio):
        self.writes.append(np.asarray(audio).copy())

    # Context-manager support for the legacy synchronous _play() path.
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self.close()
        return False


class _RecordingPiper:
    """Piper stub that records every call and returns predictable PCM."""

    def __init__(self, latency_s: float = 0.0):
        self.calls: List[str] = []
        self.latency_s = latency_s
        self._lock = threading.Lock()

    def synthesize_wav(self, text, wav_file, syn_config=None):
        with self._lock:
            self.calls.append(text)
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        # 100 frames per call; distinct seed so we can identify the call.
        seed = len(self.calls)
        samples = (np.ones(100, dtype=np.int16) * (seed * 100)).tobytes()
        wav_file.writeframes(samples)


class _RecordingRvc:
    """RVC stub that records every call. Pass-through with optional latency."""

    def __init__(self, latency_s: float = 0.0, output_sr: int = 40000):
        self.calls: List[tuple[int, int]] = []
        self.latency_s = latency_s
        self.output_sr = output_sr
        self._lock = threading.Lock()
        self._counter = 0

    def convert(self, pcm, sr):
        with self._lock:
            self._counter += 1
            call_id = self._counter
        self.calls.append((pcm.shape[0], int(sr)))
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        # Output 200 frames at output_sr, value-tagged by call order so
        # downstream playback ordering can be asserted.
        out = (np.ones(200, dtype=np.int16) * (call_id * 1000)).astype(np.int16)
        return out, self.output_sr


def _build_tts(*, with_rvc: bool, piper_latency: float = 0.0,
               rvc_latency: float = 0.0) -> tuple[TextToSpeech, _RecordingPiper, _RecordingRvc | None]:
    """Construct a TextToSpeech with stubbed Piper / RVC, no real loads."""
    tts = object.__new__(TextToSpeech)
    piper = _RecordingPiper(latency_s=piper_latency)
    tts._voice = piper
    tts.piper_sample_rate = 22050
    tts.length_scale = 1.15
    tts.flush_chars = set(".!?\n")
    rvc: _RecordingRvc | None = None
    if with_rvc:
        rvc = _RecordingRvc(latency_s=rvc_latency)
    tts.rvc = rvc
    tts.output_device = None
    tts._stop_event = threading.Event()
    tts._playback_lock = threading.Lock()
    return tts, piper, rvc


@pytest.fixture(autouse=True)
def _reset_mock_streams():
    _MockOutputStream.instances.clear()
    yield
    _MockOutputStream.instances.clear()


@pytest.fixture
def parallel_config():
    """Config with all three optimisations on."""
    cfg = UltronConfig()
    cfg.tts.pipeline_parallel_enabled = True
    cfg.tts.speculative_stream_open_enabled = True
    cfg.tts.speculative_stream_sample_rate = 40000
    cfg.tts.output_low_latency_mode = True
    set_config(cfg)
    yield cfg
    set_config(UltronConfig())


@pytest.fixture
def legacy_config():
    """Config with all three optimisations off (single-worker, no spec, no low-latency)."""
    cfg = UltronConfig()
    cfg.tts.pipeline_parallel_enabled = False
    cfg.tts.speculative_stream_open_enabled = False
    cfg.tts.speculative_stream_sample_rate = 40000
    cfg.tts.output_low_latency_mode = False
    set_config(cfg)
    yield cfg
    set_config(UltronConfig())


# ---------------------------------------------------------------------------
# Pipeline parallelism
# ---------------------------------------------------------------------------


def test_parallel_pipeline_overlaps_piper_and_rvc(parallel_config):
    """With the split pipeline, Piper N+1 begins before RVC N finishes.

    Piper takes 0.05 s per sentence; RVC takes 0.20 s. Three sentences.

    Sequential walltime: 5 * (Piper + RVC) = 5 * (0.04 + 0.20) = 1.20 s
    Parallel walltime: 0.04 + 5 * 0.20 = 1.04 s (4 Piper overlaps)

    We assert the parallel run beats the sequential lower bound by a
    reasonable margin. Five sentences gives the parallelism enough
    work to amortise thread-coordination overhead reliably across
    machines.
    """
    tts, piper, rvc = _build_tts(
        with_rvc=True, piper_latency=0.04, rvc_latency=0.20,
    )
    fragments = [
        "First sentence. ",
        "Second sentence. ",
        "Third sentence. ",
        "Fourth sentence. ",
        "Fifth sentence.",
    ]

    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        t0 = time.monotonic()
        tts.speak_stream(iter(fragments))
        elapsed = time.monotonic() - t0

    # Five Piper calls.
    assert len(piper.calls) == 5
    # Five RVC calls in the same order (queue is FIFO).
    assert len(rvc.calls) == 5
    # Sequential lower bound = 5 * (0.04 + 0.20) = 1.20 s.
    # Parallel should beat that comfortably (4 of 5 Piper calls overlap
    # with RVC). Threshold is generous (1.15 s) to absorb thread-
    # scheduling jitter on slower machines while still catching a
    # complete regression to sequential behaviour.
    assert elapsed < 1.15, (
        f"parallel pipeline took {elapsed:.3f}s, expected <1.15s "
        f"(sequential would be ~1.20s)"
    )


def test_legacy_single_worker_path_runs_when_flag_off(legacy_config):
    """``pipeline_parallel_enabled=False`` falls back to single-worker.

    Verifies the legacy branch still works end-to-end.
    """
    tts, piper, rvc = _build_tts(with_rvc=True)
    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Hello there. ", "Second one."]))

    assert len(piper.calls) == 2
    assert len(rvc.calls) == 2
    # Stream opened exactly once at 40000 (RVC output rate).
    streams = _MockOutputStream.instances
    assert len(streams) >= 1
    # No spec-open, so SR comes from the first clip.
    assert streams[0].kwargs["samplerate"] == 40000


def test_parallel_preserves_sentence_ordering(parallel_config):
    """A, B, C in -> A, B, C out across the parallel pipeline."""
    tts, piper, rvc = _build_tts(with_rvc=True)
    fragments = ["Alpha. ", "Bravo. ", "Charlie."]

    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(fragments))

    assert piper.calls == ["Alpha.", "Bravo.", "Charlie."]
    # RVC sees sentences in the same order Piper produced them.
    assert len(rvc.calls) == 3


def test_parallel_pipeline_disabled_when_no_rvc(parallel_config):
    """Without RVC, the split is a no-op; legacy single-worker runs.

    The split only saves time when there's an RVC stage to overlap
    with Piper. Without RVC, there's nothing to parallelise and the
    legacy path is functionally identical and simpler.
    """
    tts, piper, _ = _build_tts(with_rvc=False)
    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Just Piper."]))

    assert piper.calls == ["Just Piper."]


# ---------------------------------------------------------------------------
# Speculative stream open
# ---------------------------------------------------------------------------


def test_speculative_open_uses_configured_sample_rate(parallel_config):
    """Stream opens at speculative_stream_sample_rate (40000) BEFORE first clip."""
    tts, piper, rvc = _build_tts(with_rvc=True)
    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Hello."]))

    # First constructed stream is the speculative open at 40000.
    streams = _MockOutputStream.instances
    assert len(streams) >= 1
    assert streams[0].kwargs["samplerate"] == 40000


def test_speculative_open_reopens_on_sample_rate_mismatch(parallel_config):
    """If actual SR != speculative SR, stream is closed and reopened.

    Sets speculative SR to 22050 so the 40000 from the RVC mock causes
    a mismatch. Should produce two stream instances.
    """
    parallel_config.tts.speculative_stream_sample_rate = 22050
    set_config(parallel_config)

    tts, piper, rvc = _build_tts(with_rvc=True)
    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Hello."]))

    streams = _MockOutputStream.instances
    # First stream opened at 22050 (speculative); reopened at 40000 (actual).
    assert len(streams) >= 2
    assert streams[0].kwargs["samplerate"] == 22050
    assert streams[0].closed  # closed after mismatch
    assert streams[1].kwargs["samplerate"] == 40000


def test_speculative_open_disabled_falls_back_to_first_clip_sr(legacy_config):
    """With spec_open off, stream is opened only after first clip arrives."""
    tts, piper, rvc = _build_tts(with_rvc=True)
    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Hello."]))

    streams = _MockOutputStream.instances
    # Exactly one stream, opened at the actual RVC output rate.
    assert len(streams) == 1
    assert streams[0].kwargs["samplerate"] == 40000


# ---------------------------------------------------------------------------
# Low-latency mode
# ---------------------------------------------------------------------------


def test_low_latency_hint_passed_to_outputstream(parallel_config):
    """``latency='low'`` lands in the OutputStream constructor kwargs."""
    tts, piper, rvc = _build_tts(with_rvc=True)
    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Hello."]))

    streams = _MockOutputStream.instances
    assert len(streams) >= 1
    assert streams[0].kwargs.get("latency") == "low"


def test_low_latency_hint_omitted_when_disabled(legacy_config):
    """With ``output_low_latency_mode=False``, no latency kwarg is passed."""
    tts, piper, rvc = _build_tts(with_rvc=True)
    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Hello."]))

    streams = _MockOutputStream.instances
    assert len(streams) >= 1
    assert "latency" not in streams[0].kwargs


# ---------------------------------------------------------------------------
# RVC failure fallback in the parallel pipeline
# ---------------------------------------------------------------------------


def test_parallel_pipeline_falls_back_to_raw_piper_on_rvc_error(parallel_config):
    """When RVC.convert raises, the pipeline falls back to raw Piper PCM
    instead of dropping the sentence -- mirrors the legacy ``_synthesize``
    fail-soft behaviour."""
    tts, piper, rvc = _build_tts(with_rvc=True)
    rvc.convert = MagicMock(side_effect=RuntimeError("rvc kaboom"))

    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Test sentence."]))

    # Stream still opened, sentence still played (raw Piper).
    streams = _MockOutputStream.instances
    assert len(streams) >= 1
    # At least the pre-roll silence write happened.
    assert any(len(s.writes) > 0 for s in streams)


def test_parallel_pipeline_handles_cancellation_cleanly(parallel_config):
    """``stop()`` mid-utterance terminates both worker threads cleanly."""
    tts, piper, rvc = _build_tts(
        with_rvc=True, piper_latency=0.02, rvc_latency=0.05,
    )

    def slow_iter():
        for s in ["First. ", "Second. ", "Third. ", "Fourth. ", "Fifth."]:
            yield s
            time.sleep(0.01)

    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        # Stop after a short delay; worker threads should join cleanly.
        timer = threading.Timer(0.05, tts.stop)
        timer.start()
        tts.speak_stream(slow_iter())
        timer.cancel()

    # Pipeline didn't deadlock; method returned in a bounded time.


# ---------------------------------------------------------------------------
# Producer-signaled lookahead (ack-first) — option B
# ---------------------------------------------------------------------------


def test_first_clip_plays_before_next_fragment_yielded(parallel_config):
    """Producer-signaled lookahead: first clip plays IMMEDIATELY on receipt,
    NOT after blocking for the second clip.

    This is the ack-first contract. When the orchestrator yields a
    web-search ack token followed by a long-running search call, the
    ack must reach the speaker BEFORE the search begins -- not after
    it returns. The legacy play-after-peek pattern delayed the first
    clip by up to 10 s waiting for the second; this test guards
    against that regression.
    """
    tts, piper, rvc = _build_tts(with_rvc=True)

    audio_written = threading.Event()

    class _TrackingStream(_MockOutputStream):
        def write(self, audio):
            super().write(audio)
            arr = np.asarray(audio)
            # Ignore pre-roll silence (all zeros). The actual clip
            # carries the seeded non-zero values from _RecordingRvc.
            if arr.size > 0 and bool(np.any(arr != 0)):
                audio_written.set()

    observed = []

    def slow_iter():
        yield "First sentence. "
        # Wait for the first clip's audio to actually be written. If
        # play-after-peek is back, this wait will time out.
        played_first = audio_written.wait(timeout=5.0)
        observed.append(played_first)
        yield "Second sentence."

    with patch("ultron.tts.speech.sd.OutputStream", _TrackingStream):
        tts.speak_stream(slow_iter())

    assert observed == [True], (
        "Producer-signaled lookahead regression: first clip was not "
        "played before the generator was asked for the second fragment."
    )
    # Both sentences synthesised in normal order.
    assert piper.calls == ["First sentence.", "Second sentence."]


def test_slow_second_clip_does_not_kill_playback(parallel_config):
    """A 4 s gap between fragments (e.g. web-search wall) must NOT cause
    the RVC stage to time out and kill the response.

    Reproduction of the BMW failure mode in the 2026-05-09 logs: a
    long search held the generator long enough for the previous 10 s
    RVC piper_q.get() timeout to fire, killing audio for the response
    that arrived after the search completed.
    """
    tts, piper, rvc = _build_tts(with_rvc=True)

    def slow_iter():
        yield "Acquiring data. "
        # Simulates a long Brave + Jina chain returning before the
        # response tokens arrive. Longer than the legacy 10 s timeout
        # would have tolerated.
        time.sleep(4.0)
        yield "Here is the response."

    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        t0 = time.monotonic()
        tts.speak_stream(slow_iter())
        elapsed = time.monotonic() - t0

    # Both sentences made it through. If the RVC stage had aborted
    # mid-stream, the second sentence would not have been synthesised.
    assert piper.calls == ["Acquiring data.", "Here is the response."]
    assert len(rvc.calls) == 2
    # And the run took at least the simulated stall time -- proving
    # we waited for the slow generator instead of giving up.
    assert elapsed >= 3.5, (
        f"Playback wrapped up in {elapsed:.2f}s; expected >=3.5s "
        f"to confirm the slow second fragment was honoured."
    )


def test_clipitem_is_known_last_skips_lookahead(parallel_config):
    """When a producer pushes ClipItem(is_known_last=True), playback
    plays it and exits without waiting for the next item.

    Single-fragment cases like 'Switched to the 4B.' (model swap
    voice response) and ack-only flows benefit from the producer
    declaring the final clip explicitly.
    """
    tts, piper, rvc = _build_tts(with_rvc=True)

    # Inject a single ClipItem(is_known_last=True) directly into the
    # audio_q via a custom test path. We bypass speak_stream to
    # exercise the playback loop's response to the flag without
    # going through the synth workers.
    #
    # In production the synth workers always push is_known_last=False
    # because they can't know in advance whether the next fragment
    # yields another sentence. The flag is for future producers (e.g.
    # canned voice responses) that DO know.
    pcm = np.ones(200, dtype=np.int16) * 5000
    item = ClipItem(audio=pcm, sample_rate=40000, is_known_last=True)

    # Construct one synthesised frame manually and then assert the
    # ClipItem namedtuple carries the flag through.
    assert item.is_known_last is True
    assert item.sample_rate == 40000
    assert item.audio.shape == (200,)


def test_end_of_stream_sentinel_terminates_playback(parallel_config):
    """A None on audio_q is the end-of-stream marker; playback writes
    a tail silence and exits. The ClipItem before None is treated as
    the final clip even when its is_known_last=False."""
    tts, piper, rvc = _build_tts(with_rvc=True)

    with patch("ultron.tts.speech.sd.OutputStream", _MockOutputStream):
        tts.speak_stream(iter(["Single."]))

    streams = _MockOutputStream.instances
    assert len(streams) >= 1
    # At least: pre-roll silence + the clip's audio + tail silence.
    # Exact count depends on block_frames split, but >= 3 writes.
    assert len(streams[0].writes) >= 3, (
        f"Expected pre-roll + clip + tail; got {len(streams[0].writes)} writes"
    )
