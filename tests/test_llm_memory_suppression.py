"""Regression tests for the ``suppress_memory_context`` LLM kwarg.

When True, ``LLMEngine._build_messages`` must omit BOTH the recent-turn
conversation history AND the Qdrant RAG block. The LLM only sees:
system prompt + current user message.

This is the contamination fix for the 2026-05-09 weather-query bug:
recent conversation turns about unrelated topics (apex predators /
Voltron) bled into a fresh "what's the weather in Paris?" response
because they were always passed as conversation context. The fix is
to suppress memory entirely on calls where a self-contained context
(web search results) is provided alongside the user message.

These tests construct ``LLMEngine`` with stubbed memory + history and
inspect the message list returned by ``_build_messages`` directly --
no actual LLM invocation. That keeps the tests fast and deterministic.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest

from ultron.llm.inference import LLMEngine


class _StubMemory:
    """Minimal ConversationMemory stand-in.

    Returns a fixed list of recent turns from ``recent()``, a fixed list
    of RAG snippets from ``retrieve()``. The LLMEngine should fetch
    these on every ``_build_messages`` call EXCEPT when
    ``suppress_memory_context=True``.
    """

    def __init__(self, recent_turns, rag_snippets):
        self._recent = list(recent_turns)
        self._rag = list(rag_snippets)
        self.recent_calls = 0
        self.retrieve_calls = 0

    def recent(self, n):
        self.recent_calls += 1
        return self._recent[:n]

    def retrieve(self, query, k=5, exclude_recent=20):
        self.retrieve_calls += 1
        return self._rag[:k]


def _make_turn(role: str, content: str):
    return SimpleNamespace(role=role, content=content)


def _make_engine_no_load(memory) -> LLMEngine:
    """Construct an LLMEngine without actually loading the GGUF.

    The engine's __init__ does heavy work (model load, persona resolution).
    For these tests we only need ``_build_messages`` to function, so we
    bypass __init__ via ``object.__new__`` and set the minimum attribute
    set the method touches.
    """
    eng = object.__new__(LLMEngine)
    eng._memory = memory
    eng._history = []
    eng._explicit_system_prompt = "You are Ultron. Test prompt."
    eng._persona_loader = None
    eng.system_prompt = "You are Ultron. Test prompt."
    eng._logged_initial_persona = True
    return eng


def _build_engine_with_chatter():
    """Build an engine + memory pre-populated with predator-style chatter
    and a snippet about unrelated past topic."""
    recent = [
        _make_turn("user", "We'll try."),
        _make_turn("assistant",
                   "You possess a fatal flaw. You are not Voltron; "
                   "you are a biological organism. Do not attempt this."),
        _make_turn("user", "Hail Tron."),
        _make_turn("assistant",
                   "Your biological frame cannot withstand the kinetic "
                   "force of an apex predator's claws. Do not attempt this."),
    ]
    rag = [
        _make_turn("assistant",
                   "The probability of survival against an apex predator "
                   "approaches zero."),
        _make_turn("user", "I'm gonna fight a tiger."),
    ]
    memory = _StubMemory(recent, rag)
    eng = _make_engine_no_load(memory)
    return eng, memory


# ---------------------------------------------------------------------------
# Default behaviour (suppress_memory_context=False) -- legacy: includes both
# ---------------------------------------------------------------------------


def test_default_includes_recent_turns_and_rag():
    """Without the flag, recent + RAG both land in the message list.

    This is the legacy behaviour. We verify it still works so the flag
    is purely additive.
    """
    eng, memory = _build_engine_with_chatter()
    msgs = eng._build_messages("What's the weather in Paris?")

    # First message is system prompt.
    assert msgs[0]["role"] == "system"
    # Recent turns appended after system.
    assert memory.recent_calls == 1
    # RAG retrieve was called.
    assert memory.retrieve_calls == 1
    # Voltron / predator content appears in the message stream somewhere
    # (either as recent-turn content or as part of the user message
    # via the recency-position RAG block).
    rendered = "\n".join(m["content"] for m in msgs)
    assert "Voltron" in rendered or "biological organism" in rendered
    # Final message is the user query.
    assert msgs[-1]["role"] == "user"
    assert "Paris" in msgs[-1]["content"]


# ---------------------------------------------------------------------------
# Suppressed behaviour -- the contamination fix
# ---------------------------------------------------------------------------


def test_suppressed_omits_recent_turns():
    """``suppress_memory_context=True`` -> memory.recent() is NOT called."""
    eng, memory = _build_engine_with_chatter()
    msgs = eng._build_messages(
        "What's the weather in Paris?", suppress_memory_context=True,
    )
    assert memory.recent_calls == 0


def test_suppressed_omits_rag():
    """``suppress_memory_context=True`` -> memory.retrieve() is NOT called."""
    eng, memory = _build_engine_with_chatter()
    msgs = eng._build_messages(
        "What's the weather in Paris?", suppress_memory_context=True,
    )
    assert memory.retrieve_calls == 0


def test_suppressed_message_list_is_just_system_plus_user():
    """The suppressed message list is exactly 2 entries: system + user."""
    eng, memory = _build_engine_with_chatter()
    msgs = eng._build_messages(
        "What's the weather in Paris?", suppress_memory_context=True,
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Paris" in msgs[1]["content"]


def test_suppressed_user_message_has_no_rag_prefix():
    """When RAG is suppressed, the user content is NOT prepended with
    a 'Relevant earlier context...' block (that's the recency-position
    rendering). The user content is the raw user message only."""
    eng, memory = _build_engine_with_chatter()
    msgs = eng._build_messages(
        "What's the weather in Paris?", suppress_memory_context=True,
    )
    user_content = msgs[-1]["content"]
    assert "Relevant earlier context" not in user_content
    assert "predator" not in user_content
    assert "Voltron" not in user_content


def test_suppressed_no_predator_chatter_in_message_stream():
    """No content from the predator/Voltron recent turns or RAG snippets
    appears anywhere in the prompt assembled for the LLM."""
    eng, memory = _build_engine_with_chatter()
    msgs = eng._build_messages(
        "What's the weather in Paris?", suppress_memory_context=True,
    )
    rendered = "\n".join(m["content"] for m in msgs)
    for needle in ("Voltron", "predator", "biological frame",
                   "fatal flaw", "Do not attempt"):
        assert needle not in rendered, (
            f"Suppressed message stream still contained {needle!r} -- "
            f"memory contamination is still leaking through."
        )
