"""Tests for the TTS output-stream pre-open path (Phase 5).

2026-05-15 latency: the orchestrator pre-opens the PortAudio output
stream on a daemon thread during Whisper STT so the ~50 ms open cost
overlaps with transcription instead of landing on the critical path
after the first LLM token.

Pinned contracts:

1. ``prepare_output_stream`` is idempotent.
2. ``_consume_preopened_stream`` returns the cached stream on SR match
   and atomically clears the slot.
3. ``_consume_preopened_stream`` returns None on SR mismatch and
   closes the cached stream.
4. ``_consume_preopened_stream`` returns None when no stream is cached.
5. ``stop`` closes any leftover pre-opened stream.
6. Failures during pre-open are swallowed; live path is unaffected.
7. The orchestrator's ``_kick_off_tts_preopen`` is fail-open against
   missing engine method.
"""

from __future__ import annotations

import threading
from typing import List
from unittest.mock import MagicMock, patch

import pytest

import numpy as np

from kenning.pipeline.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------


class _FakeOutputStream:
    """Stand-in for sounddevice.OutputStream. Tracks lifecycle."""

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.started = False
        self.stopped = False
        self.closed = False
        self.writes: List = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def write(self, audio) -> None:
        self.writes.append(audio)


# ---------------------------------------------------------------------------
# xtts_v3 engine pre-open
# ---------------------------------------------------------------------------


class TestXttsV3PreOpen:
    """Build a partial XttsV3Speech via ``__new__`` so we don't need
    the venv-xtts server to instantiate."""

    @staticmethod
    def _build_engine():
        from kenning.tts.xtts_v3 import XttsV3Speech
        e = object.__new__(XttsV3Speech)
        e._sample_rate = 24000
        e._preopened_stream = None
        e._preopened_lock = threading.Lock()
        e._stop_event = threading.Event()
        return e

    def test_prepare_then_consume_returns_stream(self):
        e = self._build_engine()
        fake = _FakeOutputStream()
        with patch.object(e, "_open_output_stream", return_value=fake), \
             patch.object(e, "_write_silence"):
            e.prepare_output_stream()
        assert fake.started is True
        assert e._preopened_stream is fake

        s = e._consume_preopened_stream(sr=24000)
        assert s is fake
        # Slot cleared after consume.
        assert e._preopened_stream is None

    def test_consume_on_sr_mismatch_closes_and_returns_none(self):
        e = self._build_engine()
        fake = _FakeOutputStream()
        with patch.object(e, "_open_output_stream", return_value=fake), \
             patch.object(e, "_write_silence"):
            e.prepare_output_stream()

        s = e._consume_preopened_stream(sr=48000)  # mismatch
        assert s is None
        assert fake.stopped is True
        assert fake.closed is True
        assert e._preopened_stream is None

    def test_consume_with_no_preopen_returns_none(self):
        e = self._build_engine()
        assert e._consume_preopened_stream(sr=24000) is None

    def test_prepare_is_idempotent(self):
        e = self._build_engine()
        fake = _FakeOutputStream()
        with patch.object(e, "_open_output_stream", return_value=fake) as op, \
             patch.object(e, "_write_silence"):
            e.prepare_output_stream()
            e.prepare_output_stream()  # second call no-ops
            e.prepare_output_stream()
        # Only one underlying open call.
        assert op.call_count == 1

    def test_prepare_failure_is_swallowed(self):
        e = self._build_engine()
        with patch.object(
            e, "_open_output_stream",
            side_effect=RuntimeError("device busy"),
        ):
            # Should NOT raise.
            e.prepare_output_stream()
        # Slot still empty -- live path will open fresh.
        assert e._preopened_stream is None

    def test_stop_closes_leftover_preopen(self):
        e = self._build_engine()
        fake = _FakeOutputStream()
        with patch.object(e, "_open_output_stream", return_value=fake), \
             patch.object(e, "_write_silence"):
            e.prepare_output_stream()

        # sd.stop is a module-level call, just mock it out.
        with patch("kenning.tts.xtts_v3.sd.stop"):
            e.stop()

        assert fake.stopped is True
        assert fake.closed is True
        assert e._preopened_stream is None


# ---------------------------------------------------------------------------
# Legacy speech.py engine pre-open
# ---------------------------------------------------------------------------


class TestLegacyPreOpen:

    @staticmethod
    def _build_engine():
        from kenning.tts.speech import TextToSpeech
        e = object.__new__(TextToSpeech)
        e.piper_sample_rate = 22050
        e._preopened_stream = None
        e._preopened_lock = threading.Lock()
        e._stop_event = threading.Event()
        return e

    def test_prepare_then_consume_returns_stream(self):
        e = self._build_engine()
        fake = _FakeOutputStream()
        with patch.object(e, "_open_output_stream", return_value=fake):
            e.prepare_output_stream()
        # The legacy engine reads spec_sr from config (defaults vary in
        # tests). Whatever it opened with, consume should match.
        cached_sr = getattr(fake, "_kenning_sr", None)
        assert cached_sr is not None
        s = e._consume_preopened_stream(sr=cached_sr)
        assert s is fake

    def test_consume_on_sr_mismatch_closes(self):
        e = self._build_engine()
        fake = _FakeOutputStream()
        with patch.object(e, "_open_output_stream", return_value=fake):
            e.prepare_output_stream()
        cached_sr = getattr(fake, "_kenning_sr", None)
        wrong_sr = (cached_sr or 22050) + 8000
        s = e._consume_preopened_stream(sr=wrong_sr)
        assert s is None
        assert fake.closed is True

    def test_prepare_failure_is_swallowed(self):
        e = self._build_engine()
        with patch.object(
            e, "_open_output_stream",
            side_effect=RuntimeError("device busy"),
        ):
            e.prepare_output_stream()
        assert e._preopened_stream is None

    def test_prepare_writes_silence_for_device_clock_warmup(self):
        """2026-05-16 latency pass 2: legacy engine pre-open must
        also write 50 ms of silence to wake the audio device clock
        (XTTS already did). Without this, the first ``speak_stream``
        clip on the legacy stack pays the device-wake latency."""
        e = self._build_engine()
        fake = _FakeOutputStream()
        with patch.object(e, "_open_output_stream", return_value=fake), \
             patch.object(e, "_write_silence") as mock_write:
            e.prepare_output_stream()
        # Silence call: stream, sr, 50 ms.
        assert mock_write.called
        args, _kwargs = mock_write.call_args
        assert args[0] is fake
        assert args[2] == pytest.approx(0.05)

    def test_prepare_silence_write_failure_is_swallowed(self):
        """If _write_silence itself raises, the pre-open should still
        succeed -- some PortAudio backends prime themselves on
        stream.start() and don't need the explicit write."""
        e = self._build_engine()
        fake = _FakeOutputStream()
        with patch.object(e, "_open_output_stream", return_value=fake), \
             patch.object(
                 e, "_write_silence",
                 side_effect=RuntimeError("write failed")
             ):
            e.prepare_output_stream()
        # Stream still cached.
        assert e._preopened_stream is fake


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


class TestOrchestratorKickoff:

    def test_kick_off_returns_none_when_engine_lacks_method(self):
        o = Orchestrator.__new__(Orchestrator)
        # Engine without prepare_output_stream attribute.
        o.tts = MagicMock(spec=[])
        t = o._kick_off_tts_preopen()
        assert t is None

    def test_kick_off_returns_thread_when_engine_supports(self):
        o = Orchestrator.__new__(Orchestrator)

        class _Engine:
            def __init__(self):
                self.called = False

            def prepare_output_stream(self):
                self.called = True

        o.tts = _Engine()
        t = o._kick_off_tts_preopen()
        assert t is not None
        t.join(timeout=2.0)
        assert o.tts.called is True

    def test_kick_off_swallows_thread_failure(self, monkeypatch):
        """If Thread() raises (which is rare but possible), we must
        not propagate -- the live speak_stream path falls back."""
        o = Orchestrator.__new__(Orchestrator)

        class _Engine:
            def prepare_output_stream(self):
                pass

        o.tts = _Engine()

        def _broken_thread(*args, **kwargs):
            raise RuntimeError("simulated thread failure")

        monkeypatch.setattr(threading, "Thread", _broken_thread, raising=True)

        # Should not raise.
        t = o._kick_off_tts_preopen()
        assert t is None

    def test_kick_off_when_tts_is_none(self):
        o = Orchestrator.__new__(Orchestrator)
        o.tts = None
        t = o._kick_off_tts_preopen()
        assert t is None


# ---------------------------------------------------------------------------
# 2026-05-18 latency pass 3 (Phase 1): preopen kicked off at top of capture
# ---------------------------------------------------------------------------


class TestPreOpenAtCaptureStart:
    """After 2026-05-16 latency pass 2 (Phase 4) hid Whisper STT behind
    the silence wait, the legacy ``_kick_off_tts_preopen`` placement in
    ``run()`` (after capture returns) only had ~5-10 ms of overlap before
    the first TTS write -- not enough for the 50 ms PortAudio open. The
    fix hoists the kick-off into ``_capture_utterance`` and
    ``_follow_up_listen`` so the open overlaps the full speech + silence
    window (typically 1-30 s).

    These tests use source inspection because the full capture loop is a
    heavy fixture; the kick-off helper itself is covered by the class
    above. Source inspection is the right tool for "this call exists at
    this position in this function" -- the contract is structural."""

    def test_kick_off_present_in_capture_utterance(self):
        import inspect
        from kenning.pipeline.orchestrator import Orchestrator

        src = inspect.getsource(Orchestrator._capture_utterance)
        assert "_kick_off_tts_preopen" in src, (
            "Phase 1 contract: _capture_utterance must kick off the "
            "TTS preopen so the PortAudio device-open overlaps the "
            "full speech + silence-wait window."
        )

    def test_kick_off_present_in_follow_up_listen(self):
        import inspect
        from kenning.pipeline.orchestrator import Orchestrator

        src = inspect.getsource(Orchestrator._follow_up_listen)
        assert "_kick_off_tts_preopen" in src, (
            "Phase 1 contract: _follow_up_listen must kick off the "
            "TTS preopen on the WARM path too -- mirrors the COLD-path "
            "placement in _capture_utterance."
        )

    def test_kick_off_in_capture_runs_before_vad_loop(self):
        """The kick-off must happen BEFORE the while-loop that consumes
        audio chunks, so the ~50 ms PortAudio open overlaps the entire
        capture instead of starting after speech ends."""
        import inspect
        from kenning.pipeline.orchestrator import Orchestrator

        src = inspect.getsource(Orchestrator._capture_utterance)
        kickoff_pos = src.find("_kick_off_tts_preopen")
        while_pos = src.find("while not self._shutdown")
        assert kickoff_pos > 0
        assert while_pos > 0
        assert kickoff_pos < while_pos, (
            "Phase 1 contract: preopen must fire BEFORE the VAD loop "
            "so it has the full capture window to complete."
        )

    def test_kick_off_in_follow_up_runs_before_vad_loop(self):
        import inspect
        from kenning.pipeline.orchestrator import Orchestrator

        src = inspect.getsource(Orchestrator._follow_up_listen)
        kickoff_pos = src.find("_kick_off_tts_preopen")
        while_pos = src.find("while not self._shutdown")
        assert kickoff_pos > 0
        assert while_pos > 0
        assert kickoff_pos < while_pos, (
            "Phase 1 contract: preopen must fire BEFORE the VAD loop "
            "in the follow-up listen path too."
        )
