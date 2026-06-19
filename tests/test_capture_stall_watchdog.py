"""Capture-stall watchdog regression tests (2026-06-18).

A healthy mic input stream delivers a chunk every ~16 ms (a quiet room still
streams SILENCE chunks), so ``audio.get_chunk(timeout=0.5)`` only returns None
when the stream has STOPPED producing callbacks -- a USB-overrun / CPU-starvation
stall after a heavy in-process turn. Without recovery this leaves Ultron
intermittently DEAF (the next wake word never fires -- the live "one works, the
next won't" symptom). ``_wait_for_wake_word`` now counts consecutive timeouts and
calls ``_restart_capture_stream`` (stop+start) after ~1s so the wake pipeline
self-heals -- with ZERO added delay in the healthy (no-stall) case.

These drive the real ``_wait_for_wake_word`` loop with a scripted mic, so the
counting + restart trigger + recovery are exercised end-to-end.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np


def _wake_orch(chunk_script):
    """Partial Orchestrator wired to drive ``_wait_for_wake_word`` with a scripted
    mic. ``chunk_script`` is consumed one item per loop iteration: ``None`` is a
    get_chunk TIMEOUT (a stall tick); an ndarray is a delivered chunk. The wake
    fires on a chunk whose first sample == 1.0. Returns (orch, calls) where calls
    counts the AudioCapture stop/start/drain (stop+start are ONLY called by the
    watchdog's _restart_capture_stream)."""
    from kenning.pipeline.orchestrator import Orchestrator

    o = object.__new__(Orchestrator)
    o._shutdown = threading.Event()
    calls = {"stop": 0, "start": 0, "drain": 0}

    it = iter(chunk_script)

    def _get_chunk(timeout=0.5):
        try:
            return next(it)
        except StopIteration:
            o._shutdown.set()   # end the loop if the script is exhausted
            return None

    o.audio = SimpleNamespace(
        get_chunk=_get_chunk,
        drain=lambda: calls.__setitem__("drain", calls["drain"] + 1),
        stop=lambda: calls.__setitem__("stop", calls["stop"] + 1),
        start=lambda: calls.__setitem__("start", calls["start"] + 1),
    )
    o.wake = SimpleNamespace(
        reset=lambda: None,
        process=lambda chunk: bool(
            chunk is not None and len(chunk) and float(chunk[0]) == 1.0),
    )
    o.ring = SimpleNamespace(clear=lambda: None, write=lambda c: None)
    o._drain_gui_actions = lambda: None
    o._maybe_reload_config = lambda: None
    o._maybe_recover_embedding = lambda: None
    return o, calls


_WAKE = np.ones(256, dtype=np.float32)        # first sample 1.0 -> wake fires
_SILENCE = np.zeros(256, dtype=np.float32)     # delivered silence -> no wake


def test_stall_restarts_stream_then_recovers():
    """Two consecutive get_chunk timeouts (a stalled mic) trigger a stream
    restart, after which a delivered wake chunk fires normally."""
    o, calls = _wake_orch([None, None, _WAKE])
    assert o._wait_for_wake_word() is True
    assert calls["stop"] == 1, "a ~1s stall must restart (stop) the input stream"
    assert calls["start"] == 1, "the input stream must be re-started after a stall"


def test_no_stall_no_restart():
    """A live stream (silence then a wake chunk, no timeouts) must NOT restart --
    the watchdog adds zero overhead in the healthy case."""
    o, calls = _wake_orch([_SILENCE, _SILENCE, _WAKE])
    assert o._wait_for_wake_word() is True
    assert calls["stop"] == 0, "no stall -> must never restart the stream"
    assert calls["start"] == 0


def test_single_timeout_does_not_restart():
    """A single isolated timeout (below the consecutive threshold) must NOT
    restart -- avoids false restarts on a one-off scheduling hiccup."""
    o, calls = _wake_orch([None, _SILENCE, None, _WAKE])
    assert o._wait_for_wake_word() is True
    assert calls["stop"] == 0, "an isolated timeout must not trip the watchdog"
    assert calls["start"] == 0
