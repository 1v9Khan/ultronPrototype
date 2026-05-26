"""Tests for the per-intent condenser selection wiring inside
:meth:`LLMEngine._build_messages` (catalog 09 batch G).

The orchestrator calls :meth:`LLMEngine.set_current_intent_kind` with
the string value of a :class:`RoutingIntentKind` before invoking
:meth:`generate_stream`. When
``llm.history_compression.intent_adaptive`` is True the build path
picks a per-intent condenser (NoOp / Recent / Amortized /
LLMSummarizing) and reshapes the recent-history block BEFORE the
existing closed-window / last-N processors run.

The default ``intent_adaptive=False`` preserves the legacy fixed
pipeline byte-for-byte so the voice-path TTFT baseline is unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ultron.llm.inference import LLMEngine


class _StubMemory:
    """Minimal ConversationMemory stand-in."""

    def __init__(self, recent_turns, rag_snippets):
        self._recent = list(recent_turns)
        self._rag = list(rag_snippets)

    def recent(self, n):
        return self._recent[:n]

    def retrieve(self, query, k=5, exclude_recent=20):
        return self._rag[:k]


def _make_turn(role: str, content: str):
    return SimpleNamespace(role=role, content=content)


def _make_engine_no_load(memory) -> LLMEngine:
    """Construct an LLMEngine without loading the GGUF."""
    eng = object.__new__(LLMEngine)
    eng._memory = memory
    eng._history = []
    eng._explicit_system_prompt = "You are Ultron. Test prompt."
    eng._persona_loader = None
    eng.system_prompt = "You are Ultron. Test prompt."
    eng._logged_initial_persona = True
    eng._current_intent_kind = None
    return eng


def _build_engine_with_history():
    """Build an engine + memory with 6 recent turns."""
    recent = [
        _make_turn("user", f"user msg {i}: " + ("x" * 60))
        for i in range(3)
    ] + [
        _make_turn("assistant", f"assistant reply {i}: " + ("y" * 60))
        for i in range(3)
    ]
    rag = []
    memory = _StubMemory(recent, rag)
    return _make_engine_no_load(memory), memory


# ---------------------------------------------------------------------------
# set_current_intent_kind / get_current_intent_kind
# ---------------------------------------------------------------------------


def test_set_intent_kind_sets_and_clears():
    eng, _ = _build_engine_with_history()
    assert eng.get_current_intent_kind() is None
    eng.set_current_intent_kind("conversational")
    assert eng.get_current_intent_kind() == "conversational"
    eng.set_current_intent_kind(None)
    assert eng.get_current_intent_kind() is None


def test_set_intent_kind_accepts_arbitrary_strings():
    eng, _ = _build_engine_with_history()
    eng.set_current_intent_kind("code_task")
    assert eng.get_current_intent_kind() == "code_task"
    eng.set_current_intent_kind("totally_unknown_intent_xyz")
    assert eng.get_current_intent_kind() == "totally_unknown_intent_xyz"


# ---------------------------------------------------------------------------
# Default OFF: build_messages unchanged when intent_adaptive=False
# ---------------------------------------------------------------------------


def test_intent_adaptive_off_does_not_invoke_selector(monkeypatch):
    """With ``intent_adaptive=False`` the selector is never called -- the
    legacy fixed pipeline runs unchanged."""
    eng, _ = _build_engine_with_history()
    eng.set_current_intent_kind("code_task")

    from ultron.config import get_config
    cfg = get_config().llm.history_compression
    monkeypatch.setattr(cfg, "intent_adaptive", False)
    monkeypatch.setattr(cfg, "enabled", True)

    call_log = []

    def _spy(*args, **kwargs):
        call_log.append((args, kwargs))
        from ultron.llm.condensers.recent import RecentCondenser
        return RecentCondenser()

    monkeypatch.setattr(
        "ultron.llm.condensers.factory.select_condenser_for_intent",
        _spy,
    )

    eng._build_messages("What is the weather in Paris today?")
    assert call_log == []


# ---------------------------------------------------------------------------
# Default ON: selector is invoked with the current intent kind
# ---------------------------------------------------------------------------


def test_intent_adaptive_on_invokes_selector_with_intent(monkeypatch):
    eng, _ = _build_engine_with_history()
    eng.set_current_intent_kind("conversational")

    from ultron.config import get_config
    cfg = get_config().llm.history_compression
    monkeypatch.setattr(cfg, "intent_adaptive", True)
    monkeypatch.setattr(cfg, "enabled", True)

    seen = []

    def _spy(intent, **kwargs):
        seen.append(intent)
        from ultron.llm.condensers.noop import NoOpCondenser
        return NoOpCondenser()

    monkeypatch.setattr(
        "ultron.llm.condensers.factory.select_condenser_for_intent",
        _spy,
    )

    eng._build_messages("What is the weather in Paris today?")
    assert seen == ["conversational"]


def test_intent_adaptive_on_with_none_intent_calls_selector_with_none(
    monkeypatch,
):
    """If the orchestrator forgot to set an intent kind (None), the
    selector is STILL called -- with None -- so it can pick its own
    sensible default."""
    eng, _ = _build_engine_with_history()
    assert eng.get_current_intent_kind() is None

    from ultron.config import get_config
    cfg = get_config().llm.history_compression
    monkeypatch.setattr(cfg, "intent_adaptive", True)
    monkeypatch.setattr(cfg, "enabled", True)

    seen = []

    def _spy(intent, **kwargs):
        seen.append(intent)
        from ultron.llm.condensers.recent import RecentCondenser
        return RecentCondenser()

    monkeypatch.setattr(
        "ultron.llm.condensers.factory.select_condenser_for_intent",
        _spy,
    )

    eng._build_messages("What is the weather in Paris today?")
    assert seen == [None]


# ---------------------------------------------------------------------------
# Condenser result is applied to the history block
# ---------------------------------------------------------------------------


def test_intent_adaptive_applies_condenser_output(monkeypatch):
    """The condenser's output ``turns`` REPLACE the raw recent history
    that gets emitted into the message list."""
    eng, _ = _build_engine_with_history()
    eng.set_current_intent_kind("factual")

    from ultron.config import get_config
    cfg = get_config().llm.history_compression
    monkeypatch.setattr(cfg, "intent_adaptive", True)
    monkeypatch.setattr(cfg, "enabled", True)
    # Disable the downstream closed-window / last-N processors so we
    # see the condenser output directly.
    monkeypatch.setattr(cfg, "closed_window_enabled", False)
    monkeypatch.setattr(cfg, "last_n_enabled", False)

    class _OneTurnCondenser:
        def condense(self, turns_in, *, context=None):
            from ultron.llm.condensers.base import CondenseResult
            return CondenseResult(
                turns=(("user", "CONDENSED REPLACEMENT"),),
                dropped_turn_count=max(0, len(turns_in) - 1),
            )

    def _factory(intent, **kwargs):
        return _OneTurnCondenser()

    monkeypatch.setattr(
        "ultron.llm.condensers.factory.select_condenser_for_intent",
        _factory,
    )

    msgs = eng._build_messages("What is the weather in Paris today?")
    rendered = "\n".join(m["content"] for m in msgs)
    assert "CONDENSED REPLACEMENT" in rendered
    # Original raw history strings should NOT appear.
    assert "user msg 0" not in rendered
    assert "assistant reply 0" not in rendered


# ---------------------------------------------------------------------------
# Fail-open: a condenser exception leaves the raw history flowing through
# ---------------------------------------------------------------------------


def test_intent_adaptive_fail_open_on_condenser_exception(monkeypatch):
    eng, _ = _build_engine_with_history()
    eng.set_current_intent_kind("coding")

    from ultron.config import get_config
    cfg = get_config().llm.history_compression
    monkeypatch.setattr(cfg, "intent_adaptive", True)
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "closed_window_enabled", False)
    monkeypatch.setattr(cfg, "last_n_enabled", False)

    class _BoomCondenser:
        def condense(self, turns_in, *, context=None):
            raise RuntimeError("simulated condenser failure")

    monkeypatch.setattr(
        "ultron.llm.condensers.factory.select_condenser_for_intent",
        lambda intent, **kw: _BoomCondenser(),
    )

    # The whole _build_messages MUST NOT raise -- the legacy raw history
    # block must flow through unchanged.
    msgs = eng._build_messages("What is the weather in Paris today?")
    rendered = "\n".join(m["content"] for m in msgs)
    # Raw history strings still present because we fell open.
    assert "user msg 0" in rendered


def test_intent_adaptive_fail_open_on_factory_exception(monkeypatch):
    eng, _ = _build_engine_with_history()
    eng.set_current_intent_kind("coding")

    from ultron.config import get_config
    cfg = get_config().llm.history_compression
    monkeypatch.setattr(cfg, "intent_adaptive", True)
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "closed_window_enabled", False)
    monkeypatch.setattr(cfg, "last_n_enabled", False)

    def _boom(intent, **kw):
        raise RuntimeError("simulated factory failure")

    monkeypatch.setattr(
        "ultron.llm.condensers.factory.select_condenser_for_intent",
        _boom,
    )

    msgs = eng._build_messages("What is the weather in Paris today?")
    rendered = "\n".join(m["content"] for m in msgs)
    assert "user msg 0" in rendered


# ---------------------------------------------------------------------------
# Condenser result with error -> output ignored
# ---------------------------------------------------------------------------


def test_intent_adaptive_skips_result_with_error(monkeypatch):
    """A CondenseResult with ``error != None`` (``ok=False``) is
    discarded -- raw history flows through unchanged."""
    eng, _ = _build_engine_with_history()
    eng.set_current_intent_kind("factual")

    from ultron.config import get_config
    cfg = get_config().llm.history_compression
    monkeypatch.setattr(cfg, "intent_adaptive", True)
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "closed_window_enabled", False)
    monkeypatch.setattr(cfg, "last_n_enabled", False)

    class _ErroringCondenser:
        def condense(self, turns_in, *, context=None):
            from ultron.llm.condensers.base import CondenseResult
            return CondenseResult(
                turns=(("user", "REPLACEMENT (but errored)"),),
                error="non-fatal partial failure",
            )

    monkeypatch.setattr(
        "ultron.llm.condensers.factory.select_condenser_for_intent",
        lambda intent, **kw: _ErroringCondenser(),
    )

    msgs = eng._build_messages("What is the weather in Paris today?")
    rendered = "\n".join(m["content"] for m in msgs)
    assert "REPLACEMENT (but errored)" not in rendered
    assert "user msg 0" in rendered


# ---------------------------------------------------------------------------
# Real selector + real RecentCondenser end-to-end smoke
# ---------------------------------------------------------------------------


def test_intent_adaptive_real_recent_condenser_trims_history(monkeypatch):
    """End-to-end with the real selector + RecentCondenser: enough
    recent turns get trimmed to fit RecentCondenser's defaults."""
    eng = object.__new__(LLMEngine)
    eng._memory = None  # use _history directly
    eng._history = [
        (role, "msg " + str(i))
        for i, role in enumerate(
            ["user", "assistant"] * 30,
        )
    ]
    eng._explicit_system_prompt = "sys"
    eng._persona_loader = None
    eng.system_prompt = "sys"
    eng._logged_initial_persona = True
    eng._current_intent_kind = "factual"

    from ultron.config import get_config
    cfg = get_config().llm.history_compression
    monkeypatch.setattr(cfg, "intent_adaptive", True)
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "closed_window_enabled", False)
    monkeypatch.setattr(cfg, "last_n_enabled", False)

    msgs = eng._build_messages("Explain the difference between TCP and UDP.")
    # RecentCondenser default keeps a window; we should see at most
    # (system + recent_turns_kept + user) = small fixed envelope.
    # The exact count depends on RecentCondenser defaults; we just
    # verify the message list is bounded smaller than the raw 60 turns.
    assert len(msgs) < 60 + 2  # +2 for system+user
