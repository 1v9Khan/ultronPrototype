"""Regression tests for the ``precomputed_rag_snippets`` LLM kwarg.

2026-05-15 latency: the orchestrator pre-fetches RAG snippets on a
background thread in parallel with the web-gate call, then passes
them to ``LLMEngine.generate`` / ``generate_stream`` via the new
``precomputed_rag_snippets`` kwarg. The LLM uses them as-is and
skips the internal ``_retrieve_rag_snippets`` call.

These tests pin three contracts:

1. **Precomputed snippets land in the message list.** When the kwarg
   is passed, the resulting message body contains the provided
   snippets and does NOT call ``memory.retrieve``.
2. **Empty precomputed list is honored.** Passing ``[]`` produces a
   no-RAG message (not "fall back to retrieve").
3. **None means legacy behaviour.** When the kwarg is absent /
   ``None``, the engine retrieves via memory as it always did.
4. **Public ``retrieve_rag_snippets`` exists and proxies the private
   variant.** The orchestrator calls this to pre-fetch.
5. **suppress_memory_context wins over precomputed.** When both are
   set, suppression takes priority (the search-augmented branch
   needs to drop memory entirely).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

from ultron.llm.inference import LLMEngine


# ---------------------------------------------------------------------------
# Helpers (mirror the test_llm_memory_suppression.py stubbing pattern).
# ---------------------------------------------------------------------------


class _StubMemory:
    """Minimal ConversationMemory stand-in. Tracks call counts so we
    can assert the engine bypasses ``retrieve`` when precomputed is set."""

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
    """Mimic the MemoryTurn shape ``_format_rag_block`` consumes."""
    return SimpleNamespace(role=role, content=content)


def _make_engine_no_load(memory) -> LLMEngine:
    """Construct an LLMEngine without actually loading the GGUF."""
    eng = object.__new__(LLMEngine)
    eng._memory = memory
    eng._history = []
    eng._explicit_system_prompt = "You are Ultron. Test prompt."
    eng._persona_loader = None
    eng._static_system_prompt = "You are Ultron. Test prompt."
    eng.system_prompt = "You are Ultron. Test prompt."
    eng._logged_initial_persona = True
    return eng


def _build_engine() -> "tuple[LLMEngine, _StubMemory]":
    # NB: distinct vocabularies so the compression layer (default-ON
    # for the rag surface) doesn't dedup tokens and break the
    # substring assertions.
    recent = [
        _make_turn("user", "earlier user inquiry"),
        _make_turn("assistant", "earlier assistant reply"),
    ]
    rag = [
        _make_turn("assistant", "Apollo orbital mechanics"),
        _make_turn("user", "Saturn rocket fuel composition"),
    ]
    memory = _StubMemory(recent, rag)
    return _make_engine_no_load(memory), memory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_precomputed_snippets_appear_in_message_body():
    eng, memory = _build_engine()
    custom_snippets = [
        _make_turn("assistant", "precomputed snippet ONE"),
        _make_turn("user", "precomputed snippet TWO"),
    ]

    msgs = eng._build_messages(
        "What's up?", precomputed_rag_snippets=custom_snippets,
    )

    # User message holds the RAG block prepended to the question.
    user_msg = msgs[-1]
    assert user_msg["role"] == "user"
    assert "precomputed snippet ONE" in user_msg["content"]
    assert "precomputed snippet TWO" in user_msg["content"]
    # The stored snippets were NOT retrieved.
    assert "Apollo orbital mechanics" not in user_msg["content"]
    assert "Saturn rocket fuel composition" not in user_msg["content"]


def test_precomputed_skips_internal_retrieve_call():
    eng, memory = _build_engine()
    custom_snippets = [_make_turn("assistant", "S1")]

    eng._build_messages(
        "hi", precomputed_rag_snippets=custom_snippets,
    )

    # The whole point: the LLM didn't pay the retrieval cost.
    assert memory.retrieve_calls == 0


def test_precomputed_empty_list_means_no_rag():
    """Passing [] = caller fetched and got nothing. Don't retry."""
    eng, memory = _build_engine()

    msgs = eng._build_messages("hi", precomputed_rag_snippets=[])

    user_msg = msgs[-1]
    # No RAG content from the stored memory snippets:
    assert "Apollo orbital mechanics" not in user_msg["content"]
    # And no fallback retrieval:
    assert memory.retrieve_calls == 0


def test_none_falls_back_to_internal_retrieve():
    """When the kwarg is not passed, legacy behaviour: retrieve internally."""
    eng, memory = _build_engine()

    msgs = eng._build_messages("hi")  # no precomputed

    user_msg = msgs[-1]
    # Stored snippets DO land in the message:
    assert "Apollo orbital mechanics" in user_msg["content"]
    # And retrieve WAS called once:
    assert memory.retrieve_calls == 1


def test_suppress_wins_over_precomputed():
    """suppress_memory_context=True drops both recent AND RAG --
    even if precomputed snippets are passed. The search-augmented
    branch needs the LLM to ignore prior conversation entirely."""
    eng, memory = _build_engine()
    custom_snippets = [_make_turn("assistant", "should NOT appear")]

    msgs = eng._build_messages(
        "hi",
        precomputed_rag_snippets=custom_snippets,
        suppress_memory_context=True,
    )

    # System + user, nothing else.
    assert len(msgs) == 2
    user_msg = msgs[-1]
    assert "should NOT appear" not in user_msg["content"]
    assert "Apollo" not in user_msg["content"]
    assert memory.retrieve_calls == 0
    assert memory.recent_calls == 0


def test_retrieve_rag_snippets_public_proxies_private():
    """The public method exists and calls through to the private one."""
    eng, memory = _build_engine()

    # Spy on the underlying memory retrieve.
    snippets = eng.retrieve_rag_snippets("anything")

    assert memory.retrieve_calls == 1
    assert isinstance(snippets, list)
    # Returned the stored items.
    assert snippets[0].content == "Apollo orbital mechanics"


def test_retrieve_rag_snippets_returns_empty_when_no_memory():
    """When memory is None, the public method returns [] not None,
    so callers can pass the result straight to generate_stream."""
    eng = _make_engine_no_load(memory=None)

    out = eng.retrieve_rag_snippets("anything")
    assert out == []


def test_precomputed_preserves_recent_history():
    """Recent-turn history is independent of the RAG snippets --
    pre-fetching RAG must not drop the conversation history."""
    eng, memory = _build_engine()
    custom_snippets = [_make_turn("assistant", "S1")]

    msgs = eng._build_messages(
        "hi", precomputed_rag_snippets=custom_snippets,
    )

    # System + recent (2 turns) + user = 4 messages.
    assert len(msgs) == 4
    assert memory.recent_calls == 1


def test_precomputed_with_gate_verdict_is_compatible():
    """gate_verdict and precomputed can both be set -- the verdict
    is normally used for multi-pass routing, but with precomputed
    we skip that path. No exception."""
    eng, memory = _build_engine()

    fake_verdict = SimpleNamespace(
        context_categories=["weather"],
        memory_search_queries=["paris weather"],
    )
    custom_snippets = [_make_turn("assistant", "S1")]

    msgs = eng._build_messages(
        "hi",
        gate_verdict=fake_verdict,
        precomputed_rag_snippets=custom_snippets,
    )

    # Precomputed wins; no internal retrieve.
    assert memory.retrieve_calls == 0
    user_msg = msgs[-1]
    assert "S1" in user_msg["content"]
