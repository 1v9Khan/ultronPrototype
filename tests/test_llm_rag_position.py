"""4B optimization plan Stage G — RAG injection position tests.

Verifies that ``LLMEngine._build_messages`` places retrieved Qdrant
memories at the position dictated by ``llm.rag.position``:

- ``"system"`` (legacy): folded into the leading system message.
- ``"recency"`` (new default): prepended to the final user message.

Mocks the ConversationMemory so we can drive deterministic snippets in
without standing up Qdrant or loading the embedder.
"""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch

from ultron.llm.inference import LLMEngine


# Stand-in for memory.MemoryTurn — only the fields _format_rag_block reads.
_FakeTurn = namedtuple("_FakeTurn", ["role", "content"])


class _MemCfg:
    rag_top_k = 3
    rag_exclude_recent = 20
    recent_turns = 0


class _LLMRagCfg:
    def __init__(self, position: str = "recency"):
        self.position = position


class _CfgWithMemory:
    def __init__(self, position: str = "recency"):
        self.memory = _MemCfg()

        class _LLM:
            def __init__(self):
                self.rag = _LLMRagCfg(position=position)
        self.llm = _LLM()


def _make_engine(snippets: list, position: str) -> LLMEngine:
    eng = LLMEngine.__new__(LLMEngine)
    eng._runtime = "in_process"
    eng._llm = MagicMock()
    eng._cancel = __import__("threading").Event()
    eng._history = __import__("collections").deque()

    mem = MagicMock()
    mem.retrieve.return_value = snippets
    mem.recent.return_value = []
    eng._memory = mem

    eng._explicit_system_prompt = "PERSONA"
    eng._persona_loader = None
    eng._logged_initial_persona = True
    return eng


def test_recency_position_prepends_rag_to_user_message() -> None:
    snippets = [
        _FakeTurn("user", "Bob asked about the project deadline last Tuesday."),
        _FakeTurn("assistant", "I told him May 30."),
    ]
    eng = _make_engine(snippets, "recency")

    with patch("ultron.llm.inference.get_config", return_value=_CfgWithMemory("recency")):
        msgs = eng._build_messages("when's the deadline?")

    # System message: persona only, NO RAG content
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "PERSONA"
    assert "Relevant earlier context" not in msgs[0]["content"]

    # Last message is user, with RAG prepended
    last = msgs[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("Relevant earlier context from prior conversations:")
    assert "Bob asked about the project deadline" in last["content"]
    assert "I told him May 30" in last["content"]
    assert last["content"].endswith("when's the deadline?")


def test_system_position_folds_rag_into_system_message() -> None:
    """Legacy mode preserved for back-compat / rollback."""
    snippets = [
        _FakeTurn("user", "Old context entry."),
    ]
    eng = _make_engine(snippets, "system")

    with patch("ultron.llm.inference.get_config", return_value=_CfgWithMemory("system")):
        msgs = eng._build_messages("hello")

    # System message contains both persona and RAG content
    assert msgs[0]["role"] == "system"
    assert "PERSONA" in msgs[0]["content"]
    assert "Relevant earlier context" in msgs[0]["content"]
    assert "Old context entry" in msgs[0]["content"]

    # User message has the raw query, no RAG prefix
    last = msgs[-1]
    assert last["role"] == "user"
    assert last["content"] == "hello"


def test_no_snippets_means_no_rag_block_anywhere() -> None:
    """When retrieve() returns nothing, neither path injects a block —
    the user message is the raw query, the system message is just the
    persona."""
    eng = _make_engine([], "recency")

    with patch("ultron.llm.inference.get_config", return_value=_CfgWithMemory("recency")):
        msgs = eng._build_messages("hello")

    assert msgs[0]["content"] == "PERSONA"
    last = msgs[-1]
    assert last["role"] == "user"
    assert last["content"] == "hello"


def test_retrieve_failure_falls_back_to_no_rag() -> None:
    """memory.retrieve raising must not break message construction —
    the fallback is "no RAG injection this turn" with a warning log."""
    eng = _make_engine([], "recency")
    eng._memory.retrieve.side_effect = RuntimeError("qdrant down")

    with patch("ultron.llm.inference.get_config", return_value=_CfgWithMemory("recency")):
        msgs = eng._build_messages("hello")

    assert msgs[0]["content"] == "PERSONA"
    assert msgs[-1]["content"] == "hello"


def test_format_rag_block_empty_returns_empty_string() -> None:
    """Helper invariant — falsy result enables simple truthiness checks
    in the caller."""
    assert LLMEngine._format_rag_block([]) == ""


def test_format_rag_block_renders_role_and_content() -> None:
    snippets = [
        _FakeTurn("user", "alpha"),
        _FakeTurn("assistant", "beta"),
    ]
    block = LLMEngine._format_rag_block(snippets)
    assert "Relevant earlier context from prior conversations:" in block
    assert "- user: alpha" in block
    assert "- assistant: beta" in block


def test_recency_position_does_not_affect_history() -> None:
    """History (recent turns) must still sit between system + user
    regardless of RAG position. Snippets at recency must NOT be
    duplicated into the history."""
    snippets = [_FakeTurn("user", "rag-context")]
    eng = _make_engine(snippets, "recency")
    eng._memory.recent.return_value = [
        _FakeTurn("user", "earlier-q"),
        _FakeTurn("assistant", "earlier-a"),
    ]

    cfg = _CfgWithMemory("recency")
    cfg.memory.recent_turns = 5

    with patch("ultron.llm.inference.get_config", return_value=cfg):
        msgs = eng._build_messages("now")

    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "earlier-q"
    assert msgs[2]["content"] == "earlier-a"
    # RAG only in the final user message
    assert "rag-context" in msgs[-1]["content"]
    assert "rag-context" not in msgs[1]["content"]


# ---------------------------------------------------------------------------
# V1-gap A2: gate_verdict pass-through into multi-pass retrieval.
# ---------------------------------------------------------------------------


def test_build_messages_default_uses_single_pass_retrieve():
    """No gate_verdict -> the engine still calls memory.retrieve(...) (the
    legacy single-pass path), even when the memory exposes
    retrieve_for_query."""
    snippets = [_FakeTurn("user", "snippet")]
    eng = _make_engine(snippets, "recency")
    eng._memory.retrieve_for_query = MagicMock(return_value=snippets)

    with patch("ultron.llm.inference.get_config", return_value=_CfgWithMemory("recency")):
        eng._build_messages("query")

    eng._memory.retrieve.assert_called_once()
    eng._memory.retrieve_for_query.assert_not_called()


def test_build_messages_with_verdict_uses_retrieve_for_query():
    """When a gate_verdict is passed, the engine routes through
    retrieve_for_query so the multi-pass fan-out activates."""
    snippets = [_FakeTurn("user", "snippet")]
    eng = _make_engine(snippets, "recency")
    eng._memory.retrieve_for_query = MagicMock(return_value=snippets)

    class _Verdict:
        context_categories = ["category A"]
        memory_search_queries = []

    with patch("ultron.llm.inference.get_config", return_value=_CfgWithMemory("recency")):
        eng._build_messages("query", gate_verdict=_Verdict())

    eng._memory.retrieve_for_query.assert_called_once()
    args, kwargs = eng._memory.retrieve_for_query.call_args
    assert args[0] == "query"
    assert isinstance(args[1], _Verdict)
    eng._memory.retrieve.assert_not_called()


def test_build_messages_falls_back_when_retrieve_for_query_missing():
    """If memory doesn't expose retrieve_for_query (e.g., a stub from a
    test that predates A2), the engine still works via retrieve."""
    snippets = [_FakeTurn("user", "s")]
    eng = _make_engine(snippets, "recency")

    # Remove retrieve_for_query if Mock auto-added it.
    if hasattr(eng._memory, "retrieve_for_query"):
        del eng._memory.retrieve_for_query

    class _Verdict:
        context_categories = []
        memory_search_queries = []

    with patch("ultron.llm.inference.get_config", return_value=_CfgWithMemory("recency")):
        eng._build_messages("query", gate_verdict=_Verdict())

    eng._memory.retrieve.assert_called_once()
