"""Tests for deep-memory recall: the strict matcher (must NOT hijack the
fast RAG path) + the orchestrator short-circuit handler."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from ultron.memory.deep_recall import DeepRecallMatch, match_deep_recall


# ---------------------------------------------------------------------------
# Matcher precision -- the load-bearing safety property is that NORMAL recall
# questions stay on the fast single-pass RAG path (return None here).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "recall everything we discussed about the database schema",
        "dig deep into your memory about my preferences",
        "thoroughly recall what I told you about the project deadline",
        "what's everything you remember about my car situation",
        "exhaustively recall everything we talked about regarding the budget",
        "do a deep recall on what we discussed about the API design",
    ],
)
def test_matcher_fires_on_explicit_deep_recall(text):
    m = match_deep_recall(text)
    assert m is not None, f"should match: {text!r}"
    assert m.topic  # a non-empty topic was extracted
    assert m.raw_text == text


@pytest.mark.parametrize(
    "text",
    [
        # No depth marker -> fast RAG path (the critical non-hijack cases).
        "what do you remember about my car",
        "did I mention the dentist appointment",
        "remind me what we discussed yesterday",
        "do you recall my sister's name",
        # Depth but no memory referent -> not a memory command.
        "explain quantum computing in depth",
        "tell me everything about black holes",
        # Web research shape -> defers to match_deep_research.
        "research the latest news on AI thoroughly",
        "search the web in depth for python tutorials",
        # Trivial / empty.
        "hello there",
        "",
        "   ",
    ],
)
def test_matcher_does_not_fire_on_fast_path_or_web(text):
    assert match_deep_recall(text) is None, f"should NOT match: {text!r}"


def test_matcher_extracts_topic_after_about():
    m = match_deep_recall("recall everything we discussed about the database schema")
    assert m is not None
    assert m.topic == "the database schema"


# ---------------------------------------------------------------------------
# Orchestrator handler -- via __new__ stub (no real voice stack).
# ---------------------------------------------------------------------------


class _FakeInnerLLM:
    def create_chat_completion(self, **_kwargs):
        # Empty content -> decompose / gap-fill fail open to verbatim search.
        return {"choices": [{"message": {"content": ""}}]}


class _FakeLLM:
    def __init__(self):
        self._llm = _FakeInnerLLM()
        self.last_augmented = None

    def generate_stream(self, augmented, **_kwargs):
        self.last_augmented = augmented
        return iter(["From memory: ", "we discussed the schema on Monday."])

    def cancel(self):
        pass


class _FakeMemory:
    def __init__(self, turns):
        self._turns = turns
        self.calls = []

    def retrieve(self, query, k=3):
        self.calls.append((query, k))
        return list(self._turns)


class _FakeTTS:
    def __init__(self):
        self.streamed = []

    def speak_stream(self, gen):
        self.streamed.extend(list(gen))


def _make_orch(monkeypatch, *, memory, llm):
    from ultron.pipeline import orchestrator as orch_mod
    from ultron.pipeline.orchestrator import Orchestrator

    monkeypatch.setattr(orch_mod.settings, "BARGE_IN_ENABLED", False, raising=False)
    o = Orchestrator.__new__(Orchestrator)
    o.memory = memory
    o.llm = llm
    o.tts = _FakeTTS()
    o._interrupt = threading.Event()
    o._shutdown = threading.Event()
    o._last_response_text = ""
    o._spoken = []
    o._speak = lambda text: o._spoken.append(text)
    return o


def test_handler_recalls_and_synthesizes(monkeypatch):
    turn = SimpleNamespace(
        id="t1", role="user",
        content="we discussed the database schema on Monday",
    )
    llm = _FakeLLM()
    mem = _FakeMemory([turn])
    o = _make_orch(monkeypatch, memory=mem, llm=llm)

    handled = o._maybe_handle_deep_recall(
        "recall everything we discussed about the database schema"
    )

    assert handled is True
    assert any("dig through" in s for s in o._spoken)  # ack
    assert mem.calls  # memory was queried
    assert "we discussed the schema on Monday." in "".join(o.tts.streamed)
    # The recalled turn content reached the synthesis prompt.
    assert "database schema on Monday" in (llm.last_augmented or "")


def test_handler_speaks_nothing_found_when_memory_empty(monkeypatch):
    llm = _FakeLLM()
    mem = _FakeMemory([])  # nothing recalled
    o = _make_orch(monkeypatch, memory=mem, llm=llm)

    handled = o._maybe_handle_deep_recall(
        "dig deep into your memory about the budget"
    )

    assert handled is True
    assert any("don't have anything" in s for s in o._spoken)
    assert o.tts.streamed == []  # no synthesis when nothing recalled


def test_handler_falls_through_on_no_match(monkeypatch):
    o = _make_orch(monkeypatch, memory=_FakeMemory([]), llm=_FakeLLM())
    assert o._maybe_handle_deep_recall("hello there") is False


def test_handler_falls_through_without_memory(monkeypatch):
    o = _make_orch(monkeypatch, memory=None, llm=_FakeLLM())
    assert o._maybe_handle_deep_recall(
        "recall everything we discussed about X"
    ) is False
