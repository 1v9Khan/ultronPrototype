"""Orchestrator-level tests for the parallel RAG pre-fetch (Phase 2).

2026-05-15 latency: the orchestrator kicks off RAG retrieval on a
background thread BEFORE the web-gate call so the two costs overlap.
Pre-fetched snippets are passed to ``LLMEngine.generate_stream`` via
the ``precomputed_rag_snippets`` kwarg.

Tests below construct an Orchestrator via ``__new__`` and inject
stubs so the unit suite stays fast (no Whisper / LLM / TTS load).
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from ultron.pipeline.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubMemory:
    """Truthy memory; the prefetch path checks ``self.memory is None``
    and bails when so."""

    def __init__(self) -> None:
        pass


class _StubLLM:
    def __init__(self) -> None:
        self.retrieve_calls: List[str] = []
        self.generate_stream_calls: List[dict] = []
        self._next_snippets: List = []

    def retrieve_rag_snippets(self, user_message: str, *, gate_verdict=None):
        self.retrieve_calls.append(user_message)
        return list(self._next_snippets)

    def generate_stream(self, prompt: str, **kwargs):
        self.generate_stream_calls.append({"prompt": prompt, **kwargs})
        # Yield a single dummy token so the generator drains.
        yield "ok"


def _make_orch(
    *,
    memory: Optional[object] = None,
    llm: Optional[_StubLLM] = None,
    web_gate=None,
    web_executor=None,
) -> Orchestrator:
    o = Orchestrator.__new__(Orchestrator)
    o.memory = memory
    o.llm = llm or _StubLLM()
    o.web_gate = web_gate
    o.web_executor = web_executor
    # Required by _maybe_conversational_ack.
    o.coding_voice = None
    o.conv_ack_source = MagicMock()
    o.conv_ack_source.next_phrase.return_value = "Mm."
    return o


def _set_multi_pass(cfg_value: bool, monkeypatch):
    """Patch get_config to control multi_pass_enabled."""
    fake_cfg = MagicMock()
    fake_cfg.memory.retrieval.multi_pass_enabled = cfg_value
    monkeypatch.setattr(
        "ultron.config.get_config", lambda: fake_cfg, raising=True,
    )


# ---------------------------------------------------------------------------
# _kick_off_rag_prefetch
# ---------------------------------------------------------------------------


class TestKickOffPrefetch:
    def test_returns_none_when_memory_disabled(self):
        orch = _make_orch(memory=None)
        future, used_async = orch._kick_off_rag_prefetch("hi")
        assert future is None
        assert used_async is False

    def test_skips_when_multi_pass_enabled(self, monkeypatch):
        """Multi-pass needs the gate_verdict's context_categories;
        pre-fetching single-pass now would silently downgrade. So
        the helper bails when the flag is on."""
        _set_multi_pass(True, monkeypatch)
        orch = _make_orch(memory=_StubMemory())
        future, used_async = orch._kick_off_rag_prefetch("hi")
        assert future is None
        assert used_async is False

    def test_kicks_off_when_single_pass(self, monkeypatch):
        _set_multi_pass(False, monkeypatch)
        llm = _StubLLM()
        llm._next_snippets = [{"role": "assistant", "content": "S1"}]
        orch = _make_orch(memory=_StubMemory(), llm=llm)

        future, used_async = orch._kick_off_rag_prefetch("hi")

        assert future is not None
        assert used_async is True
        # The future eventually resolves to the stub's snippets.
        result = future.result(timeout=2.0)
        assert result == [{"role": "assistant", "content": "S1"}]
        # And the LLM was asked to retrieve once.
        assert llm.retrieve_calls == ["hi"]

    def test_kickoff_failure_is_swallowed(self, monkeypatch):
        """If ThreadPoolExecutor construction raises, we degrade
        silently to no pre-fetch."""
        _set_multi_pass(False, monkeypatch)
        orch = _make_orch(memory=_StubMemory())

        # Break ThreadPoolExecutor for this call.
        def _broken_executor(*args, **kwargs):
            raise RuntimeError("simulated executor failure")

        monkeypatch.setattr(
            "concurrent.futures.ThreadPoolExecutor",
            _broken_executor,
            raising=True,
        )

        future, used_async = orch._kick_off_rag_prefetch("hi")
        assert future is None
        assert used_async is False


# ---------------------------------------------------------------------------
# _collect_rag_future
# ---------------------------------------------------------------------------


class TestCollectFuture:
    def test_none_future_returns_none(self):
        out = Orchestrator._collect_rag_future(None)
        assert out is None

    def test_completed_future_returns_value(self):
        f: Future = Future()
        f.set_result(["snippet"])
        out = Orchestrator._collect_rag_future(f)
        assert out == ["snippet"]

    def test_exception_returns_none(self):
        f: Future = Future()
        f.set_exception(RuntimeError("qdrant unreachable"))
        out = Orchestrator._collect_rag_future(f)
        assert out is None

    def test_empty_list_returns_empty_list(self):
        """[] is a valid result (caller fetched and got nothing).
        Must be distinguishable from None (which means "didn't
        try")."""
        f: Future = Future()
        f.set_result([])
        out = Orchestrator._collect_rag_future(f)
        assert out == []


# ---------------------------------------------------------------------------
# _build_response_stream integration
# ---------------------------------------------------------------------------


class TestBuildResponseStreamPrefetch:
    def test_prefetch_kicks_off_then_passes_to_llm(self, monkeypatch):
        """When the gate is disabled (None) we still run the no-gate
        branch and the LLM receives the precomputed snippets."""
        _set_multi_pass(False, monkeypatch)
        llm = _StubLLM()
        llm._next_snippets = [{"role": "user", "content": "M1"}]
        orch = _make_orch(memory=_StubMemory(), llm=llm)

        tokens = list(orch._build_response_stream("hello there"))

        # LLM was called with our pre-fetched snippets.
        assert len(llm.generate_stream_calls) == 1
        call = llm.generate_stream_calls[0]
        assert call.get("precomputed_rag_snippets") == [
            {"role": "user", "content": "M1"},
        ]
        # The dummy LLM yielded "ok".
        assert "ok" in tokens
        # Retrieval was via the public method.
        assert llm.retrieve_calls == ["hello there"]

    def test_no_memory_skips_prefetch(self, monkeypatch):
        """Memory is None -> no pre-fetch; LLM gets precomputed=None."""
        _set_multi_pass(False, monkeypatch)
        llm = _StubLLM()
        orch = _make_orch(memory=None, llm=llm)

        list(orch._build_response_stream("hello"))

        assert llm.retrieve_calls == []
        call = llm.generate_stream_calls[0]
        # The kwarg is set but resolves to None (Future was None,
        # _collect_rag_future returned None).
        assert call.get("precomputed_rag_snippets") is None

    def test_multi_pass_skips_prefetch_passes_none(self, monkeypatch):
        """Multi-pass enabled -> no pre-fetch; LLM falls back to its
        internal retrieve via gate_verdict=verdict."""
        _set_multi_pass(True, monkeypatch)
        llm = _StubLLM()
        orch = _make_orch(memory=_StubMemory(), llm=llm)

        list(orch._build_response_stream("hello"))

        assert llm.retrieve_calls == []  # prefetch skipped
        call = llm.generate_stream_calls[0]
        # No precomputed snippets passed.
        assert call.get("precomputed_rag_snippets") is None
