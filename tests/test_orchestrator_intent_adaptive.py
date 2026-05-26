"""Orchestrator-level tests for the per-intent condenser wiring
(catalog 09 batch G).

The orchestrator threads the classified
:class:`RoutingIntentKind` through to the LLM engine via
:meth:`LLMEngine.set_current_intent_kind` immediately before
:meth:`generate_stream` and clears it back to ``None`` in the
``_respond`` finally block.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from ultron.pipeline.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal LLM stub tracking set_current_intent_kind calls."""

    def __init__(self) -> None:
        self.intent_kind_calls: List[Optional[str]] = []
        self._current_intent_kind: Optional[str] = None
        self.canceled = False

    def set_current_intent_kind(self, intent_kind):
        self.intent_kind_calls.append(intent_kind)
        self._current_intent_kind = intent_kind

    def get_current_intent_kind(self):
        return self._current_intent_kind

    def cancel(self):
        self.canceled = True


class _StubTTS:
    def __init__(self) -> None:
        self.spoken_streams = 0

    def speak_stream(self, gen):
        # Drain the generator to mirror the real TTS contract.
        for _ in gen:
            pass
        self.spoken_streams += 1


def _make_orch(*, llm=None, tts=None) -> Orchestrator:
    o = Orchestrator.__new__(Orchestrator)
    o.llm = llm
    o.tts = tts
    o.memory = None
    o.web_gate = None
    o.web_executor = None
    o.coding_voice = None
    o._interrupt = __import__("threading").Event()
    o._shutdown = __import__("threading").Event()
    o._last_search_payload = None
    o._last_response_text = ""
    o._last_response_finished_monotonic = 0.0
    o._speculative_llm_active = False
    o._speculative_llm_invalidated = False
    o._speculative_llm_response = ""
    o._speculative_llm_completed = False
    o._next_turn_force_search = False
    # Stub out the response stream so we don't actually invoke the
    # LLM -- the wiring under test is the setter/clearer, not the
    # downstream generation.
    return o


# ---------------------------------------------------------------------------
# Setter wiring
# ---------------------------------------------------------------------------


def test_respond_sets_intent_kind_before_streaming(monkeypatch):
    llm = _StubLLM()
    tts = _StubTTS()
    o = _make_orch(llm=llm, tts=tts)

    # Stub _build_response_stream so it yields a single token without
    # touching the real generation path.
    def _stub_stream(user_text):
        # At this point the intent kind should already be set.
        assert llm.get_current_intent_kind() == "factual"
        yield "hello"

    monkeypatch.setattr(o, "_build_response_stream", _stub_stream)
    monkeypatch.setattr(
        "ultron.pipeline.orchestrator.settings.BARGE_IN_ENABLED",
        False,
    )

    o._respond("ping", routing_intent_kind="factual")

    # First call: "factual" set on entry.
    assert llm.intent_kind_calls[0] == "factual"


def test_respond_clears_intent_kind_after_streaming(monkeypatch):
    llm = _StubLLM()
    tts = _StubTTS()
    o = _make_orch(llm=llm, tts=tts)

    def _stub_stream(user_text):
        yield "hello"

    monkeypatch.setattr(o, "_build_response_stream", _stub_stream)
    monkeypatch.setattr(
        "ultron.pipeline.orchestrator.settings.BARGE_IN_ENABLED",
        False,
    )

    o._respond("ping", routing_intent_kind="code_task")

    # First set to code_task, last set to None.
    assert llm.intent_kind_calls[0] == "code_task"
    assert llm.intent_kind_calls[-1] is None
    assert llm.get_current_intent_kind() is None


def test_respond_clears_intent_kind_after_exception(monkeypatch):
    """Exception in the response stream MUST still trigger the clear
    (the finally block runs)."""
    llm = _StubLLM()
    tts = _StubTTS()
    o = _make_orch(llm=llm, tts=tts)

    def _stub_stream(user_text):
        raise RuntimeError("simulated streaming failure")
        yield  # unreachable

    monkeypatch.setattr(o, "_build_response_stream", _stub_stream)
    monkeypatch.setattr(
        "ultron.pipeline.orchestrator.settings.BARGE_IN_ENABLED",
        False,
    )

    o._respond("ping", routing_intent_kind="factual")
    assert llm.get_current_intent_kind() is None
    assert llm.intent_kind_calls[-1] is None


def test_respond_with_none_intent_kind_still_clears(monkeypatch):
    """When the caller doesn't pass an intent kind (or passes None),
    we still set it to None on entry and clear to None on exit. The
    LLM treats ``None`` as 'use the default fallback condenser'."""
    llm = _StubLLM()
    tts = _StubTTS()
    o = _make_orch(llm=llm, tts=tts)

    def _stub_stream(user_text):
        yield "x"

    monkeypatch.setattr(o, "_build_response_stream", _stub_stream)
    monkeypatch.setattr(
        "ultron.pipeline.orchestrator.settings.BARGE_IN_ENABLED",
        False,
    )

    o._respond("ping")  # no routing_intent_kind
    assert llm.intent_kind_calls[0] is None
    assert llm.intent_kind_calls[-1] is None


def test_respond_setter_failure_does_not_break_streaming(monkeypatch):
    """If the LLM stub raises from set_current_intent_kind, the
    response pipeline degrades gracefully (the per-intent feature is
    optional)."""
    tts = _StubTTS()

    class _BoomLLM:
        def set_current_intent_kind(self, intent_kind):
            raise RuntimeError("simulated setter failure")
        def cancel(self):
            pass

    o = _make_orch(llm=_BoomLLM(), tts=tts)

    streamed_tokens = []

    def _stub_stream(user_text):
        streamed_tokens.append(user_text)
        yield "x"

    monkeypatch.setattr(o, "_build_response_stream", _stub_stream)
    monkeypatch.setattr(
        "ultron.pipeline.orchestrator.settings.BARGE_IN_ENABLED",
        False,
    )

    o._respond("ping", routing_intent_kind="factual")
    # The stream still ran despite the setter failure.
    assert streamed_tokens == ["ping"]


def test_respond_no_llm_does_not_crash(monkeypatch):
    """Orchestrator with llm=None (rare but possible in test
    fixtures) must not crash on the setter call."""
    tts = _StubTTS()
    o = _make_orch(llm=None, tts=tts)

    def _stub_stream(user_text):
        yield "x"

    monkeypatch.setattr(o, "_build_response_stream", _stub_stream)
    monkeypatch.setattr(
        "ultron.pipeline.orchestrator.settings.BARGE_IN_ENABLED",
        False,
    )

    o._respond("ping", routing_intent_kind="factual")
    # No crash -- streaming completed.
