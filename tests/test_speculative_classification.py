"""Tests for the speculative-classification path on Orchestrator.

2026-05-18 latency pass 3 (Phase 2): once :meth:`_kick_off_speculative_stt`'s
background thread produces a transcript, the same thread chains the
rule-path web-gate, ack-phrase pick, and RAG pre-fetch. The result is
stored in ``_speculative_classification`` keyed by the transcript and
consumed by :meth:`_build_response_stream` -- saving the ~5 ms rule
classify on cache-hit AND giving the RAG retrieval ~200-300 ms more
overlap with the silence wait.

These tests cover the helpers in isolation. The orchestrator is
constructed via ``object.__new__`` so no models are loaded.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Stub Orchestrator
# ---------------------------------------------------------------------------


def _stub_orchestrator(
    *,
    web_gate=None,
    web_executor=None,
    memory=None,
    llm=None,
    rag_snippets=None,
    ack_text: Optional[str] = "Mm.",
):
    """Build a partial Orchestrator with the classification helpers and
    the minimum attributes ``_run_speculative_classification`` reads.

    ``web_gate`` is wired through ``self.web_gate`` -- present to satisfy
    the ``classify_by_rules`` call site. The real classifier is invoked
    from the helper (we don't mock the global function); use distinct
    transcripts in tests to drive deterministic SEARCH / NO_SEARCH /
    None outcomes.
    """
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    o._speculative_classification_lock = threading.Lock()
    o._speculative_classification = None
    o._speculative_classification_invalidated = False
    o.web_gate = web_gate
    o.web_executor = web_executor
    o.memory = memory
    o.llm = llm
    # ConversationalAckSource is a small shuffled-cycle helper; stub it
    # with a fixed return so tests don't depend on the shuffle seed.

    class _StubAckSource:
        def __init__(self, phrase):
            self._phrase = phrase

        def next_phrase(self) -> str:
            return self._phrase

    o.conv_ack_source = _StubAckSource(ack_text or "Mm.")
    o.coding_voice = None
    return o


# ---------------------------------------------------------------------------
# _run_speculative_classification
# ---------------------------------------------------------------------------


def test_classification_stores_result_for_text():
    """Happy path: a non-empty transcript runs through the classifier
    helpers and produces a populated ``_speculative_classification`` dict."""
    o = _stub_orchestrator(web_gate=SimpleNamespace())  # presence-only
    o._run_speculative_classification("What is the weather today?")

    state = o._speculative_classification
    assert state is not None
    assert state["text"] == "What is the weather today?"
    # "weather today" should hit the time-sensitive SEARCH rule.
    assert state["gate_verdict"] is not None
    # ack_phrase: long-utterance triggers the stub ack source.
    assert state["ack_phrase"] == "Mm."


def test_classification_skipped_on_invalidated():
    """If invalidation arrives before the helper runs, the slot is
    left empty so a later-arriving result doesn't claim a stale lane."""
    o = _stub_orchestrator(web_gate=SimpleNamespace())
    o._speculative_classification_invalidated = True
    o._run_speculative_classification("hello")
    assert o._speculative_classification is None


def test_classification_skipped_on_invalidated_mid_work():
    """Invalidation between the gate call and the store should also
    drop the result -- the second invalidation check guards the race."""
    o = _stub_orchestrator(web_gate=SimpleNamespace())
    original_ack = o._maybe_conversational_ack

    def _ack_then_invalidate(text):
        # Simulate user resuming speech mid-classification.
        o._speculative_classification_invalidated = True
        return original_ack(text)

    o._maybe_conversational_ack = _ack_then_invalidate
    o._run_speculative_classification("What is the weather today?")
    assert o._speculative_classification is None


def test_classification_handles_missing_web_gate():
    """No web_gate attribute -> verdict stays None; ack + RAG still run."""
    o = _stub_orchestrator(web_gate=None)
    o._run_speculative_classification("hello world this is a long question")
    state = o._speculative_classification
    assert state is not None
    assert state["gate_verdict"] is None


def test_classification_handles_ack_exception(monkeypatch):
    """Ack helper failure must be swallowed; the rest of the
    classification still populates."""
    o = _stub_orchestrator(web_gate=SimpleNamespace())

    def _boom(text):
        raise RuntimeError("ack pool broken")

    o._maybe_conversational_ack = _boom
    o._run_speculative_classification("What is the weather today?")
    state = o._speculative_classification
    assert state is not None
    assert state["ack_phrase"] is None


def test_classification_handles_rag_kickoff_exception(monkeypatch):
    """RAG kick-off failure must be swallowed; rag_future stays None."""
    o = _stub_orchestrator(web_gate=SimpleNamespace())

    def _boom(text):
        raise RuntimeError("pool broken")

    o._kick_off_rag_prefetch = _boom
    o._run_speculative_classification("What is the weather today?")
    state = o._speculative_classification
    assert state is not None
    assert state["rag_future"] is None


# ---------------------------------------------------------------------------
# _invalidate_speculative_classification
# ---------------------------------------------------------------------------


def test_invalidate_sets_flag_and_cancels_rag():
    """Invalidate marks the slot AND cancels the RAG future so the
    rolled-over thread doesn't keep retrieving."""
    o = _stub_orchestrator(web_gate=SimpleNamespace())
    canceled = []

    class _FakeFuture:
        def cancel(self):
            canceled.append(True)

    o._speculative_classification = {
        "text": "hi",
        "gate_verdict": None,
        "ack_phrase": None,
        "rag_future": _FakeFuture(),
    }
    o._invalidate_speculative_classification()
    assert o._speculative_classification_invalidated is True
    assert canceled == [True]


def test_invalidate_is_idempotent_with_no_state():
    """No state slot populated -> invalidate sets the flag and
    returns cleanly."""
    o = _stub_orchestrator()
    o._invalidate_speculative_classification()
    assert o._speculative_classification_invalidated is True


def test_invalidate_swallows_cancel_exception():
    """RAG future.cancel() raising must not propagate."""
    o = _stub_orchestrator()

    class _BadFuture:
        def cancel(self):
            raise RuntimeError("can't cancel")

    o._speculative_classification = {
        "text": "hi",
        "gate_verdict": None,
        "ack_phrase": None,
        "rag_future": _BadFuture(),
    }
    o._invalidate_speculative_classification()  # must not raise


def test_invalidate_propagates_to_stt_invalidate():
    """The STT invalidate must also invalidate the classification
    so both slots stay in lockstep when SPEECH_START fires."""
    o = _stub_orchestrator()
    # Wire up the STT lock so _invalidate_speculative_stt can run.
    o._speculative_stt_lock = threading.Lock()
    o._speculative_stt_invalidated = False
    o._invalidate_speculative_stt()
    assert o._speculative_classification_invalidated is True


# ---------------------------------------------------------------------------
# _collect_speculative_classification
# ---------------------------------------------------------------------------


def test_collect_returns_none_when_empty():
    """Nothing was ever stored -> collect returns None."""
    o = _stub_orchestrator()
    assert o._collect_speculative_classification("anything") is None


def test_collect_returns_state_on_text_match_and_clears_slot():
    """Slot was populated for this exact transcript -> returned and
    cleared atomically so a second collect returns None."""
    o = _stub_orchestrator()
    o._speculative_classification = {
        "text": "hi",
        "gate_verdict": None,
        "ack_phrase": "Mm.",
        "rag_future": None,
    }
    first = o._collect_speculative_classification("hi")
    assert first is not None
    assert first["ack_phrase"] == "Mm."
    # Slot cleared.
    assert o._speculative_classification is None
    # Second call returns None.
    assert o._collect_speculative_classification("hi") is None


def test_collect_returns_none_on_text_mismatch_and_cancels_rag():
    """A stored result for a different transcript is stale -- return
    None and cancel the RAG future."""
    o = _stub_orchestrator()
    canceled = []

    class _FakeFuture:
        def cancel(self):
            canceled.append(True)

    o._speculative_classification = {
        "text": "previous turn",
        "gate_verdict": None,
        "ack_phrase": None,
        "rag_future": _FakeFuture(),
    }
    result = o._collect_speculative_classification("current turn")
    assert result is None
    assert canceled == [True]


def test_collect_returns_none_when_invalidated():
    """Invalidated slot is dropped on collect; the flag clears so the
    next turn starts fresh."""
    o = _stub_orchestrator()
    o._speculative_classification = {
        "text": "hi",
        "gate_verdict": None,
        "ack_phrase": None,
        "rag_future": None,
    }
    o._speculative_classification_invalidated = True
    result = o._collect_speculative_classification("hi")
    assert result is None
    # Flag cleared for the next capture.
    assert o._speculative_classification_invalidated is False


def test_collect_defensive_on_missing_lock():
    """Fixture without the classification lock -> collect returns None
    rather than crashing."""
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    # Note: no _speculative_classification_lock attribute.
    result = o._collect_speculative_classification("anything")
    assert result is None


# ---------------------------------------------------------------------------
# _reset_speculative_classification_state
# ---------------------------------------------------------------------------


def test_reset_clears_state_and_cancels_rag():
    """Reset at the start of a capture must drop any rolled-over
    classification AND cancel its RAG future."""
    o = _stub_orchestrator()
    canceled = []

    class _FakeFuture:
        def cancel(self):
            canceled.append(True)

    o._speculative_classification = {
        "text": "stale",
        "gate_verdict": None,
        "ack_phrase": None,
        "rag_future": _FakeFuture(),
    }
    o._speculative_classification_invalidated = True
    o._reset_speculative_classification_state()
    assert o._speculative_classification is None
    assert o._speculative_classification_invalidated is False
    assert canceled == [True]


def test_reset_defensive_on_missing_lock():
    """Fixture without the lock -> reset returns cleanly."""
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    # No lock attribute.
    o._reset_speculative_classification_state()


# ---------------------------------------------------------------------------
# Chain from speculative STT
# ---------------------------------------------------------------------------


def test_stt_thread_chains_classification_on_success():
    """When speculative STT succeeds and the result is non-empty, the
    same daemon thread calls _run_speculative_classification."""
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    o._speculative_stt_lock = threading.Lock()
    o._speculative_stt_thread = None
    o._speculative_stt_result = None
    o._speculative_stt_active = False
    o._speculative_stt_invalidated = False
    o._speculative_classification_lock = threading.Lock()
    o._speculative_classification = None
    o._speculative_classification_invalidated = False

    o.stt = SimpleNamespace(transcribe=lambda audio: "Some user text")
    o.web_gate = None  # no rule check
    o.memory = None
    o.llm = None

    class _StubAckSource:
        def next_phrase(self):
            return "Mm."

    o.conv_ack_source = _StubAckSource()
    o.coding_voice = None

    import numpy as np
    o._kick_off_speculative_stt(np.zeros(16000, dtype=np.float32))
    # Wait for chain to complete.
    if o._speculative_stt_thread is not None:
        o._speculative_stt_thread.join(timeout=2.0)
    # Classification should now have been populated.
    state = o._speculative_classification
    assert state is not None
    assert state["text"] == "Some user text"


def test_stt_thread_skips_classification_on_empty_transcript():
    """Whisper returning empty / whitespace transcript means there was
    no real speech -- don't speculate downstream."""
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    o._speculative_stt_lock = threading.Lock()
    o._speculative_stt_thread = None
    o._speculative_stt_result = None
    o._speculative_stt_active = False
    o._speculative_stt_invalidated = False
    o._speculative_classification_lock = threading.Lock()
    o._speculative_classification = None
    o._speculative_classification_invalidated = False

    o.stt = SimpleNamespace(transcribe=lambda audio: "   ")
    o.web_gate = None
    o.memory = None
    o.llm = None

    class _StubAckSource:
        def next_phrase(self):
            return "Mm."

    o.conv_ack_source = _StubAckSource()
    o.coding_voice = None

    import numpy as np
    o._kick_off_speculative_stt(np.zeros(16000, dtype=np.float32))
    if o._speculative_stt_thread is not None:
        o._speculative_stt_thread.join(timeout=2.0)
    assert o._speculative_classification is None


def test_stt_thread_skips_classification_when_invalidated():
    """If STT was invalidated before the thread finishes, the
    chained classification must not run -- the user has resumed
    speaking and the audio buffer is now stale."""
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    o._speculative_stt_lock = threading.Lock()
    o._speculative_stt_thread = None
    o._speculative_stt_result = None
    o._speculative_stt_active = False
    o._speculative_stt_invalidated = False
    o._speculative_classification_lock = threading.Lock()
    o._speculative_classification = None
    o._speculative_classification_invalidated = False

    def _slow_transcribe(audio):
        # Invalidate during STT
        time.sleep(0.05)
        return "this should not classify"

    o.stt = SimpleNamespace(transcribe=_slow_transcribe)
    o.web_gate = None
    o.memory = None
    o.llm = None

    class _StubAckSource:
        def next_phrase(self):
            return "Mm."

    o.conv_ack_source = _StubAckSource()
    o.coding_voice = None

    import numpy as np
    o._kick_off_speculative_stt(np.zeros(16000, dtype=np.float32))
    # Invalidate while STT is mid-run.
    o._invalidate_speculative_stt()
    if o._speculative_stt_thread is not None:
        o._speculative_stt_thread.join(timeout=2.0)
    assert o._speculative_classification is None


# ---------------------------------------------------------------------------
# Reset propagation from STT reset
# ---------------------------------------------------------------------------


def test_stt_reset_also_clears_classification():
    """The combined STT+classification reset called at the top of
    _capture_utterance / _follow_up_listen must clear BOTH slots
    so a prior turn's state doesn't leak."""
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    o._speculative_stt_lock = threading.Lock()
    o._speculative_stt_thread = None
    o._speculative_stt_result = "stale"
    o._speculative_stt_active = False
    o._speculative_stt_invalidated = True
    o._speculative_classification_lock = threading.Lock()
    o._speculative_classification = {
        "text": "stale",
        "gate_verdict": None,
        "ack_phrase": None,
        "rag_future": None,
    }
    o._speculative_classification_invalidated = True

    o._reset_speculative_stt_state()
    assert o._speculative_stt_result is None
    assert o._speculative_classification is None
    assert o._speculative_classification_invalidated is False
