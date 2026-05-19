"""Tests for the speculative-LLM path on Orchestrator (Phase 3 of
2026-05-18 latency pass 3).

When the speculative-classification thread settles a rule-path NO_SEARCH
verdict during the silence wait, the same daemon thread kicks off
``llm.generate_stream(record_history=False)`` on the speculative
transcript. Tokens accumulate into a ``queue.Queue``; the response-stream
consumer drains the queue (in lieu of a fresh LLM call) and explicitly
commits the turn to history once the response has been emitted to TTS.

These tests cover the helpers in isolation. The orchestrator is built
via ``object.__new__`` so no models are loaded. The stub LLM mimics
the surface ``LLMEngine`` exposes to the speculative path:
``generate_stream``, ``cancel``, ``record_completed_turn``.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Stub LLM + orchestrator
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimum LLMEngine surface for speculative-LLM tests."""

    def __init__(self, tokens=None, raise_in_stream=None, per_token_delay_s=0.0):
        self.tokens = list(tokens) if tokens is not None else ["Hello", " ", "world."]
        self.raise_in_stream = raise_in_stream
        self.per_token_delay_s = per_token_delay_s
        self.cancel_count = 0
        self.canceled = threading.Event()
        self.recorded_turns = []
        self.calls = []

    def generate_stream(
        self,
        user_message,
        *,
        gate_verdict=None,
        precomputed_rag_snippets=None,
        record_history=True,
        **kwargs,
    ):
        self.calls.append({
            "user_message": user_message,
            "gate_verdict": gate_verdict,
            "precomputed_rag_snippets": precomputed_rag_snippets,
            "record_history": record_history,
        })
        for tok in self.tokens:
            if self.canceled.is_set():
                return
            if self.per_token_delay_s > 0:
                time.sleep(self.per_token_delay_s)
            yield tok
        if self.raise_in_stream is not None:
            raise self.raise_in_stream

    def cancel(self):
        self.cancel_count += 1
        self.canceled.set()

    def record_completed_turn(self, user_message, response):
        self.recorded_turns.append((user_message, response))


def _stub_orchestrator(llm=None, web_gate_present=True, ack_text="Mm."):
    """Build a partial Orchestrator wired with the Phase 3 speculation
    slots and a stub LLM."""
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    # Phase 2 + 3 slots.
    o._speculative_stt_lock = threading.Lock()
    o._speculative_stt_thread = None
    o._speculative_stt_result = None
    o._speculative_stt_active = False
    o._speculative_stt_invalidated = False
    o._speculative_classification_lock = threading.Lock()
    o._speculative_classification = None
    o._speculative_classification_invalidated = False
    o._speculative_llm_lock = threading.Lock()
    o._speculative_llm_thread = None
    o._speculative_llm_buffer = None
    o._speculative_llm_text = None
    o._speculative_llm_response = None
    o._speculative_llm_completed = False
    o._speculative_llm_active = False
    o._speculative_llm_invalidated = False
    o.llm = llm if llm is not None else _StubLLM()
    o.web_gate = SimpleNamespace() if web_gate_present else None
    o.memory = None
    o.coding_voice = None

    class _StubAckSource:
        def __init__(self, phrase):
            self._phrase = phrase

        def next_phrase(self):
            return self._phrase

    o.conv_ack_source = _StubAckSource(ack_text)
    return o


def _no_search_verdict():
    """Build a minimal verdict object that satisfies the speculative-LLM
    NO_SEARCH gate condition."""
    from ultron.web_search import GateDecision
    return SimpleNamespace(
        decision=GateDecision.NO_SEARCH,
        confidence=0.99,
        source="rule",
        reason="test-no-search",
        search_queries=[],
        knowledge_confidence=0.99,
        knowledge_source="weights",
        has_temporal_dependency=False,
        context_categories=[],
        memory_search_queries=[],
    )


def _search_verdict():
    from ultron.web_search import GateDecision
    return SimpleNamespace(
        decision=GateDecision.SEARCH,
        confidence=0.99,
        source="rule",
        reason="test-search",
        search_queries=["query"],
        knowledge_confidence=0.5,
        knowledge_source="web_search_needed",
        has_temporal_dependency=True,
        context_categories=[],
        memory_search_queries=[],
    )


# ---------------------------------------------------------------------------
# LLMEngine.record_history kwarg + record_completed_turn
# ---------------------------------------------------------------------------


class TestLLMEngineHistoryDefer:
    """The Phase 3 contract: ``generate_stream(record_history=False)``
    must skip the auto-record at end of stream; ``record_completed_turn``
    is the public commit hook."""

    @staticmethod
    def _build_engine(*, record_turn_log: list):
        from ultron.llm import inference

        e = object.__new__(inference.LLMEngine)
        e._runtime = "in_process"
        e._memory = None
        e._history = []
        e._cancel = threading.Event()
        e._explicit_system_prompt = "you are ultron"
        e._persona_loader = None
        e._static_system_prompt = "you are ultron"
        e.system_prompt = "you are ultron"
        e.history_turns = 4
        e._logged_initial_persona = True

        # Patch _record_turn so we can observe calls.
        def _record(user_message, assistant_message):
            record_turn_log.append((user_message, assistant_message))

        e._record_turn = _record

        # Stub _build_messages to a no-op (we don't exercise prompt building).
        def _build_messages(user_message, **kwargs):
            return [{"role": "user", "content": user_message}]

        e._build_messages = _build_messages

        # Stub _apply_no_think_marker -- pass-through.
        e._apply_no_think_marker = staticmethod(lambda m, t: m)

        # Stub _chat_completion_kwargs -- minimal.
        e._chat_completion_kwargs = staticmethod(
            lambda c, t, *, stream: {"stream": stream},
        )

        # Stub _llm.create_chat_completion to emit a fixed token list.
        class _Llama:
            def create_chat_completion(self, messages, **kwargs):
                if not kwargs.get("stream"):
                    return {"choices": [{"message": {"content": "ok"}}]}

                def _gen():
                    for tok in ("a", "b", "c"):
                        yield {"choices": [{"delta": {"content": tok}}]}

                return _gen()

        e._llm = _Llama()
        return e

    def test_record_history_true_records_turn(self):
        log = []
        e = self._build_engine(record_turn_log=log)
        result = list(e.generate_stream("hello"))
        # Stream consumed the canned tokens.
        assert "".join(result) == "abc"
        # Recorded once.
        assert log == [("hello", "abc")]

    def test_record_history_false_skips_auto_record(self):
        log = []
        e = self._build_engine(record_turn_log=log)
        result = list(e.generate_stream("hello", record_history=False))
        assert "".join(result) == "abc"
        # NOT recorded.
        assert log == []

    def test_record_completed_turn_records_explicitly(self):
        log = []
        e = self._build_engine(record_turn_log=log)
        e.record_completed_turn("hello", "abc")
        assert log == [("hello", "abc")]

    def test_record_completed_turn_skips_empty(self):
        log = []
        e = self._build_engine(record_turn_log=log)
        e.record_completed_turn("hello", "")
        e.record_completed_turn("hello", "   ")
        e.record_completed_turn("", "abc")
        assert log == []


# ---------------------------------------------------------------------------
# _kick_off_speculative_llm
# ---------------------------------------------------------------------------


def test_kick_off_starts_thread_and_buffers_tokens():
    """Successful kick-off accepts the verdict, runs the stub LLM, and
    fills the buffer with the canned tokens."""
    llm = _StubLLM(tokens=["hello", " ", "world"])
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    # Wait for the background thread to finish.
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)
    # Buffer should now hold three tokens + the None sentinel.
    drained = []
    buffer = o._speculative_llm_buffer
    # buffer may have been cleared by a prior collect; for this test we
    # didn't collect, so it should still be set.
    assert buffer is not None
    while True:
        tok = buffer.get(timeout=2.0)
        if tok is None:
            break
        drained.append(tok)
    assert drained == ["hello", " ", "world"]
    # Speculation completed normally.
    assert o._speculative_llm_completed is True
    assert o._speculative_llm_response == "hello world"


def test_kick_off_is_idempotent():
    """Second call while in flight is a no-op."""
    llm = _StubLLM(tokens=["x"], per_token_delay_s=0.05)
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    first_thread = o._speculative_llm_thread
    o._kick_off_speculative_llm("hi-other", _no_search_verdict(), None)
    assert o._speculative_llm_thread is first_thread
    if first_thread is not None:
        first_thread.join(timeout=2.0)


def test_kick_off_skips_when_llm_missing():
    """No LLM attribute -> kick-off no-ops."""
    o = _stub_orchestrator(llm=None)
    o.llm = None
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    assert o._speculative_llm_active is False
    assert o._speculative_llm_buffer is None


def test_kick_off_skips_when_verdict_none():
    """No verdict -> kick-off no-ops."""
    o = _stub_orchestrator()
    o._kick_off_speculative_llm("hi", None, None)
    assert o._speculative_llm_active is False
    assert o._speculative_llm_buffer is None


# ---------------------------------------------------------------------------
# _invalidate_speculative_llm
# ---------------------------------------------------------------------------


def test_invalidate_signals_cancel_and_sets_flag():
    """Invalidate sets the flag AND calls llm.cancel() so the stream
    iterator exits at its next chunk."""
    llm = _StubLLM(tokens=["x"] * 50, per_token_delay_s=0.005)
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    # Give the thread a moment to start streaming.
    time.sleep(0.01)
    o._invalidate_speculative_llm()
    # llm.cancel was called at least once.
    assert llm.cancel_count >= 1
    assert o._speculative_llm_invalidated is True
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)


def test_invalidate_idempotent_with_no_state():
    """Invalidate with no speculation in flight returns cleanly."""
    o = _stub_orchestrator()
    o._invalidate_speculative_llm()  # must not raise


def test_invalidate_defensive_on_missing_lock():
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    # No lock attribute.
    o._invalidate_speculative_llm()  # must not raise


# ---------------------------------------------------------------------------
# _collect_speculative_llm
# ---------------------------------------------------------------------------


def test_collect_returns_none_when_empty():
    """Nothing was ever kicked off -> collect returns (None, None)."""
    o = _stub_orchestrator()
    spec_iter, commit = o._collect_speculative_llm("hi")
    assert spec_iter is None
    assert commit is None


def test_collect_drains_buffer_and_commits_history_on_completion():
    """Speculation completed normally -> iterator yields all tokens and
    the commit hook records the turn."""
    llm = _StubLLM(tokens=["abc", "def"])
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)
    spec_iter, commit = o._collect_speculative_llm("hi")
    assert spec_iter is not None
    tokens = list(spec_iter)
    assert tokens == ["abc", "def"]
    # commit_history records.
    commit()
    assert llm.recorded_turns == [("hi", "abcdef")]


def test_collect_returns_none_on_text_mismatch():
    """Speculation for a different transcript -> collect returns
    (None, None) and clears the slot."""
    llm = _StubLLM(tokens=["x"])
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("first-text", _no_search_verdict(), None)
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)
    spec_iter, commit = o._collect_speculative_llm("different-text")
    assert spec_iter is None
    assert commit is None
    # State cleared.
    assert o._speculative_llm_buffer is None


def test_collect_returns_none_when_invalidated():
    """Invalidated speculation is not returned for consumption."""
    llm = _StubLLM(tokens=["x"], per_token_delay_s=0.005)
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    o._invalidate_speculative_llm()
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)
    spec_iter, commit = o._collect_speculative_llm("hi")
    assert spec_iter is None
    assert commit is None


def test_commit_history_is_noop_on_incomplete_speculation():
    """If speculation was canceled mid-stream, commit_history must not
    record an orphan turn."""
    # Build a speculation that gets canceled mid-stream.
    cancel_event = threading.Event()

    class _CancelableLLM(_StubLLM):
        def generate_stream(self, user_message, **kwargs):
            for tok in ("a", "b", "c", "d", "e"):
                if self.canceled.is_set():
                    return
                yield tok
                if tok == "b":
                    cancel_event.set()
                    # Give the invalidator time to call cancel().
                    time.sleep(0.05)

    llm = _CancelableLLM()
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    # Wait for the producer to yield "b" then invalidate.
    cancel_event.wait(timeout=2.0)
    o._invalidate_speculative_llm()
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)
    spec_iter, commit = o._collect_speculative_llm("hi")
    # Invalidated -> no iterator.
    assert spec_iter is None
    assert commit is None
    # And no turn was recorded.
    assert llm.recorded_turns == []


def test_collect_defensive_on_missing_lock():
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    spec_iter, commit = o._collect_speculative_llm("anything")
    assert spec_iter is None
    assert commit is None


# ---------------------------------------------------------------------------
# _reset_speculative_llm_state
# ---------------------------------------------------------------------------


def test_reset_clears_state_and_cancels_in_flight():
    """Reset at the top of capture must drop any rolled-over LLM
    speculation AND cancel its stream."""
    llm = _StubLLM(tokens=["x"] * 30, per_token_delay_s=0.005)
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    time.sleep(0.01)  # let producer start
    o._reset_speculative_llm_state()
    # Cancel signaled.
    assert llm.cancel_count >= 1
    # State cleared.
    assert o._speculative_llm_text is None
    assert o._speculative_llm_buffer is None
    if o._speculative_llm_thread is not None:
        # Let the producer finish.
        o._speculative_llm_thread.join(timeout=2.0)


def test_reset_defensive_on_missing_lock():
    from ultron.pipeline.orchestrator import Orchestrator
    o = object.__new__(Orchestrator)
    o._reset_speculative_llm_state()  # must not raise


# ---------------------------------------------------------------------------
# Cross-lane invalidation
# ---------------------------------------------------------------------------


def test_invalidate_classification_also_invalidates_llm():
    """Phase 3 wiring: the classification invalidate must propagate to
    the LLM lane so all three speculation slots stay in lockstep on
    SPEECH_START."""
    llm = _StubLLM(tokens=["x"] * 30, per_token_delay_s=0.005)
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    time.sleep(0.01)
    o._invalidate_speculative_classification()
    assert o._speculative_llm_invalidated is True
    assert llm.cancel_count >= 1
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)


def test_invalidate_stt_also_invalidates_llm():
    """And the STT invalidate must propagate all the way down to LLM."""
    llm = _StubLLM(tokens=["x"] * 30, per_token_delay_s=0.005)
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    time.sleep(0.01)
    o._invalidate_speculative_stt()
    assert o._speculative_llm_invalidated is True
    assert llm.cancel_count >= 1
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)


def test_reset_classification_also_resets_llm():
    """Reset must propagate so all three slots clear atomically."""
    llm = _StubLLM(tokens=["x"], per_token_delay_s=0.0)
    o = _stub_orchestrator(llm=llm)
    o._kick_off_speculative_llm("hi", _no_search_verdict(), None)
    if o._speculative_llm_thread is not None:
        o._speculative_llm_thread.join(timeout=2.0)
    o._reset_speculative_classification_state()
    assert o._speculative_llm_text is None
    assert o._speculative_llm_buffer is None


# ---------------------------------------------------------------------------
# Chain from _run_speculative_classification (verdict-gated kick-off)
# ---------------------------------------------------------------------------


def test_classification_kicks_off_llm_on_no_search():
    """When the rule-path verdict is NO_SEARCH, the chained
    classification thread fires the speculative LLM."""
    from ultron.web_search.gating import GateDecision
    from ultron.web_search import gating as gating_mod

    llm = _StubLLM(tokens=["weather"])
    o = _stub_orchestrator(llm=llm)
    # Stub classify_by_rules to return NO_SEARCH deterministically.
    original = gating_mod.classify_by_rules

    def _stub_rules(text):
        return _no_search_verdict()

    gating_mod.classify_by_rules = _stub_rules
    try:
        o._run_speculative_classification("hi")
        # Wait briefly for the LLM thread to register.
        if o._speculative_llm_thread is not None:
            o._speculative_llm_thread.join(timeout=2.0)
        # The stub LLM was called.
        assert len(llm.calls) == 1
        assert llm.calls[0]["record_history"] is False
    finally:
        gating_mod.classify_by_rules = original


def test_classification_skips_llm_on_search_verdict():
    """When the rule-path verdict is SEARCH, no LLM speculation fires."""
    from ultron.web_search import gating as gating_mod

    llm = _StubLLM(tokens=["x"])
    o = _stub_orchestrator(llm=llm)
    original = gating_mod.classify_by_rules

    def _stub_rules(text):
        return _search_verdict()

    gating_mod.classify_by_rules = _stub_rules
    try:
        o._run_speculative_classification("what's the weather today")
        # No LLM call.
        assert llm.calls == []
    finally:
        gating_mod.classify_by_rules = original


def test_classification_skips_llm_on_uncertain_verdict():
    """When classify_by_rules returns None (UNCERTAIN), no LLM
    speculation fires -- the main path will run the preflight."""
    from ultron.web_search import gating as gating_mod

    llm = _StubLLM(tokens=["x"])
    o = _stub_orchestrator(llm=llm)
    original = gating_mod.classify_by_rules

    def _stub_rules(text):
        return None

    gating_mod.classify_by_rules = _stub_rules
    try:
        o._run_speculative_classification("ambiguous")
        assert llm.calls == []
    finally:
        gating_mod.classify_by_rules = original
