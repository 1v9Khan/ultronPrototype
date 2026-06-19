"""Tests for the speculative-STT path on Orchestrator.

2026-05-16 latency pass 2: the orchestrator kicks off Whisper STT in a
background thread as soon as VAD reports a short run of consecutive
silence chunks (~32 ms at the new 16 ms blocksize). By the time the
fast-path silence baseline (~300 ms) elapses and Smart Turn V3 confirms
end-of-turn, Whisper (~78 ms) has finished and the transcript is
consumable from the main run() loop without paying the full
foreground Whisper latency.

These tests cover the kick-off / collect / invalidate / reset helpers
on Orchestrator in isolation. The orchestrator itself is constructed
via ``object.__new__`` so we don't load any models -- the helpers
are tested as pure state machines + thread coordination.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub Orchestrator
# ---------------------------------------------------------------------------


def _stub_orchestrator(stt_result: Optional[str] = "hello world",
                       stt_delay_s: float = 0.0,
                       stt_raises: Optional[Exception] = None):
    """Build a partial Orchestrator with the speculative-STT helpers
    attached. The stub ``stt`` returns ``stt_result`` (or raises) after
    ``stt_delay_s`` so tests can exercise the thread-coordination
    paths."""
    from kenning.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    o._speculative_stt_lock = threading.Lock()
    o._speculative_stt_thread = None
    o._speculative_stt_result = None
    o._speculative_stt_active = False
    o._speculative_stt_invalidated = False

    def _transcribe(audio):
        if stt_delay_s > 0:
            time.sleep(stt_delay_s)
        if stt_raises is not None:
            raise stt_raises
        return stt_result

    o.stt = SimpleNamespace(transcribe=_transcribe)
    return o


def _silence(seconds: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


# ---------------------------------------------------------------------------
# Kick-off contract
# ---------------------------------------------------------------------------


def test_kick_off_starts_background_thread():
    """Successful kick-off marks state active and stores a thread
    handle that can be joined."""
    o = _stub_orchestrator(stt_result="hi", stt_delay_s=0.05)
    o._kick_off_speculative_stt(_silence(0.5))
    # Thread handle should exist and be alive (or just completed).
    assert o._speculative_stt_thread is not None
    result = o._collect_speculative_stt(timeout_s=2.0)
    assert result == "hi"


def test_kick_off_is_idempotent_while_in_flight():
    """Re-calling kick-off while a thread is still running is a no-op
    -- the existing inference completes; the second request is
    silently dropped."""
    o = _stub_orchestrator(stt_result="first", stt_delay_s=0.15)
    o._kick_off_speculative_stt(_silence(0.5))
    # Second call -- should not launch a new thread.
    first_thread = o._speculative_stt_thread
    o._kick_off_speculative_stt(_silence(0.5))
    assert o._speculative_stt_thread is first_thread
    result = o._collect_speculative_stt(timeout_s=2.0)
    assert result == "first"


def test_kick_off_skips_when_thread_launch_fails(monkeypatch):
    """If threading.Thread() raises during construction, state must
    end up clean (not stuck active) and no leak -- the caller falls
    back to the foreground STT path."""
    o = _stub_orchestrator()

    original_thread = threading.Thread

    def _failing_thread(*args, **kwargs):
        raise RuntimeError("thread launch failed")

    monkeypatch.setattr(threading, "Thread", _failing_thread)
    o._kick_off_speculative_stt(_silence(0.5))
    # State must be back to inactive so the next kick-off can proceed.
    assert o._speculative_stt_active is False
    # Restore for any later test scaffolding.
    monkeypatch.setattr(threading, "Thread", original_thread)


# ---------------------------------------------------------------------------
# Collect contract
# ---------------------------------------------------------------------------


def test_collect_returns_none_when_no_kickoff():
    """No kick-off ever happened -> collect returns None and the
    state stays clean."""
    o = _stub_orchestrator()
    assert o._collect_speculative_stt() is None


def test_collect_waits_for_in_flight_thread():
    """Collect must join the thread (up to timeout) so the result
    is fully populated before returning."""
    o = _stub_orchestrator(stt_result="done", stt_delay_s=0.10)
    o._kick_off_speculative_stt(_silence(0.5))
    # Immediately collect -- should block until the thread finishes.
    result = o._collect_speculative_stt(timeout_s=2.0)
    assert result == "done"


def test_collect_resets_state_for_next_capture():
    """After collect, internal state is reset so the next kick-off
    starts fresh -- no stale result leak."""
    o = _stub_orchestrator(stt_result="first")
    o._kick_off_speculative_stt(_silence(0.5))
    first = o._collect_speculative_stt(timeout_s=2.0)
    assert first == "first"
    assert o._speculative_stt_result is None
    assert o._speculative_stt_active is False
    assert o._speculative_stt_thread is None


def test_collect_returns_none_on_transcription_exception():
    """If Whisper raises mid-call, the background thread swallows
    the exception and stores None; collect returns None."""
    o = _stub_orchestrator(stt_raises=RuntimeError("CUDA OOM"))
    o._kick_off_speculative_stt(_silence(0.5))
    assert o._collect_speculative_stt(timeout_s=2.0) is None


def test_collect_returns_none_when_thread_hangs_past_timeout():
    """If the background thread is still running past the timeout,
    collect returns None. The caller falls back to foreground STT."""
    o = _stub_orchestrator(stt_result="late", stt_delay_s=1.0)
    o._kick_off_speculative_stt(_silence(0.5))
    # Very short timeout -- thread won't finish.
    result = o._collect_speculative_stt(timeout_s=0.05)
    assert result is None
    # Cleanup: let the thread finish so the test runner doesn't leak it.
    if o._speculative_stt_thread is None:
        pass


# ---------------------------------------------------------------------------
# Invalidate contract
# ---------------------------------------------------------------------------


def test_invalidate_causes_collect_to_return_none():
    """User resumed speaking before SPEECH_END -> invalidate the
    in-flight speculative result. Collect returns None even though
    the thread completed successfully."""
    o = _stub_orchestrator(stt_result="stale", stt_delay_s=0.05)
    o._kick_off_speculative_stt(_silence(0.5))
    o._invalidate_speculative_stt()
    # Wait for the thread to actually finish before collect.
    time.sleep(0.15)
    result = o._collect_speculative_stt(timeout_s=2.0)
    assert result is None


def test_invalidate_then_kick_off_again_after_collect():
    """After invalidation + collect, the state must be clean enough
    to support a fresh kick-off in the next silence period."""
    o = _stub_orchestrator(stt_result="run1", stt_delay_s=0.05)
    o._kick_off_speculative_stt(_silence(0.5))
    o._invalidate_speculative_stt()
    time.sleep(0.15)
    assert o._collect_speculative_stt(timeout_s=2.0) is None
    # Swap the stub result to verify it's a fresh run.
    o.stt.transcribe = lambda audio: "run2"
    o._kick_off_speculative_stt(_silence(0.5))
    assert o._collect_speculative_stt(timeout_s=2.0) == "run2"


# ---------------------------------------------------------------------------
# Reset contract
# ---------------------------------------------------------------------------


def test_reset_clears_stale_result_without_killing_thread():
    """_reset_speculative_stt_state at the start of a capture must
    clear any stale result so it can't leak into the new turn's
    transcript."""
    o = _stub_orchestrator(stt_result="ancient")
    o._kick_off_speculative_stt(_silence(0.5))
    # Let the background thread complete and populate the result.
    time.sleep(0.05)
    while o._speculative_stt_active:
        time.sleep(0.005)
    # Now reset (simulating the start of a new capture).
    o._reset_speculative_stt_state()
    assert o._speculative_stt_result is None
    assert o._speculative_stt_invalidated is False
    assert o._speculative_stt_thread is None


# ---------------------------------------------------------------------------
# Audio buffer is snapshotted, not aliased
# ---------------------------------------------------------------------------


def test_kick_off_copies_audio_to_avoid_race():
    """The background thread reads its OWN copy of the audio buffer
    so the live capture can keep growing its chunk list without
    racing with the in-flight inference."""
    received_audio = []

    o = _stub_orchestrator()
    # Custom stt.transcribe that records what was passed in.
    def _capture(audio):
        received_audio.append(audio)
        return "ok"
    o.stt.transcribe = _capture

    original = _silence(0.5)
    o._kick_off_speculative_stt(original)
    # Mutate the original BEFORE the thread finishes (race window).
    original[:] = 0.99
    o._collect_speculative_stt(timeout_s=2.0)
    # Thread should have seen the original zero-buffer, not the mutated one.
    assert len(received_audio) == 1
    seen = received_audio[0]
    assert np.allclose(seen, 0.0)


# ---------------------------------------------------------------------------
# Capture-loop invalidation: mid-utterance pause must NOT commit a stale
# partial (2026-06-18 truncation fix).
#
# The speculative kickoff fires after ~32 ms of silence -- FAR below the
# SPEECH_END (MIN_SILENCE) baseline. Previously the in-flight result was only
# invalidated on a VAD SPEECH_START event, which only happens after a full
# SPEECH_END. So a natural mid-utterance micro-pause (32-~300 ms) kicked off
# speculation on the pre-pause LEAD but never invalidated it -- _collect_*
# then committed that lead as the final transcript, dropping everything the
# user said after the pause ("the raw isn't picking up the whole thing").
# The fix invalidates + re-arms whenever speech RESUMES after a kickoff,
# regardless of whether a SPEECH_START event fired.
# ---------------------------------------------------------------------------


def _capture_orch(monkeypatch, vad_script, *, transcribe="ok"):
    """Partial Orchestrator wired to drive ``_capture_utterance`` through a
    scripted VAD. ``vad_script`` is a list of (SpeechEvent, probability), one
    per chunk, consumed lock-step by ``vad.process``/``audio.get_chunk``; the
    loop ends on the SPEECH_END entry (smart-turn is stubbed off, so the VAD
    decides end-of-turn). Heavy helpers (wake-strip, smart-turn, streaming) are
    stubbed so only the speculative-STT silence/resume logic is exercised."""
    from kenning.pipeline import orchestrator as orch_mod
    from kenning.pipeline.orchestrator import Orchestrator
    from kenning.audio.vad import VadResult

    o = object.__new__(Orchestrator)
    o._speculative_stt_lock = threading.Lock()
    o._speculative_stt_thread = None
    o._speculative_stt_result = None
    o._speculative_stt_active = False
    o._speculative_stt_invalidated = False
    o._shutdown = threading.Event()

    o.stt = SimpleNamespace(transcribe=lambda audio: transcribe)
    o.ring = SimpleNamespace(
        snapshot=lambda n: np.zeros(max(0, int(n)), dtype=np.float32))
    o.wake = SimpleNamespace(active_word=None)

    # Config knobs chosen to keep the loop on the plain VAD path.
    o._cold_pre_roll_seconds = 0.0
    o._max_utterance_seconds = 30.0
    o._long_utterance_threshold_seconds = 0.0      # disable the long-utterance bump
    o._long_utterance_silence_duration_ms = 1200
    o._smart_turn_incomplete_extension_ms = 0
    o._smart_turn_medium_grace_ms = 0

    # No-op heavy helpers.
    o._cancel_background_summarizer = lambda: None
    o._kick_off_tts_preopen = lambda: None
    o._maybe_start_stt_stream = lambda: False
    o._maybe_feed_stt_chunk = lambda c: None
    o._stt_streaming_enabled = lambda: False
    o._smart_turn_should_check = lambda **k: False  # VAD decides end-of-turn
    o._strip_wake_audio = lambda buf, *a, **k: buf  # avoid Silero segmentation
    monkeypatch.setattr(orch_mod, "_trim_wake_from_capture",
                        lambda audio, *a, **k: audio)

    state = {"i": 0}

    def _process(chunk):
        ev, prob = vad_script[min(state["i"], len(vad_script) - 1)]
        state["i"] += 1
        return VadResult(event=ev, is_speech=prob >= 0.5, probability=prob)

    o.vad = SimpleNamespace(
        threshold=0.5, reset=lambda: None, process=_process,
        set_min_silence_duration_ms=lambda ms: None,
    )

    chunk = np.zeros(256, dtype=np.float32)
    served = {"n": 0}

    def _get_chunk(timeout=0.5):
        if served["n"] >= len(vad_script):
            o._shutdown.set()  # belt-and-braces if SPEECH_END didn't break
            return None
        served["n"] += 1
        return chunk

    o.audio = SimpleNamespace(get_chunk=_get_chunk)
    return o


def _count_invalidations(o):
    n = {"calls": 0}
    orig = o._invalidate_speculative_stt

    def _counting():
        n["calls"] += 1
        return orig()

    o._invalidate_speculative_stt = _counting
    return n


def test_capture_invalidates_speculative_on_midpause_resume(monkeypatch):
    """A mid-utterance pause kicks off speculation; resumed speech WITHOUT a
    SPEECH_END/START cycle must invalidate the stale partial."""
    from kenning.audio.vad import SpeechEvent as E

    o = _capture_orch(monkeypatch, [
        (E.SPEECH_START, 1.0), (E.NONE, 1.0),
        (E.NONE, 0.0), (E.NONE, 0.0),   # >=2 silence chunks -> speculative kickoff
        (E.NONE, 1.0), (E.NONE, 1.0),   # speech RESUMES (no SPEECH_START event)
        (E.NONE, 0.0), (E.NONE, 0.0),   # final trailing silence
        (E.SPEECH_END, 0.0),            # smart-turn off -> VAD ends the capture
    ])
    n = _count_invalidations(o)
    o._capture_utterance()
    o._collect_speculative_stt(timeout_s=2.0)  # join/cleanup bg threads
    assert n["calls"] >= 1, (
        "resumed speech after a speculative kickoff must invalidate the stale "
        "pre-pause partial -- otherwise the committed transcript drops "
        "everything said after the pause")


def test_capture_keeps_speculative_without_resume(monkeypatch):
    """A single trailing-silence kickoff with NO resumed speech must not be
    invalidated -- the speculative latency win is preserved for the common
    case."""
    from kenning.audio.vad import SpeechEvent as E

    o = _capture_orch(monkeypatch, [
        (E.SPEECH_START, 1.0), (E.NONE, 1.0), (E.NONE, 1.0),
        (E.NONE, 0.0), (E.NONE, 0.0),   # trailing silence -> kickoff, NO resume
        (E.SPEECH_END, 0.0),
    ])
    n = _count_invalidations(o)
    o._capture_utterance()
    o._collect_speculative_stt(timeout_s=2.0)
    assert n["calls"] == 0, (
        "a trailing-silence kickoff with no resumed speech must NOT be "
        "invalidated -- the speculative latency win must be preserved")
