"""Tests for the new web-search snippet ranker dispatch.

The frontier 2026-05-21 pass added three ranker options:
- ``cross_encoder`` (default): bge-reranker-v2-m3, ~20-50 ms/10-snip on CPU.
- ``llm`` (legacy): local Qwen JSON-emit ranking, ~500-1500 ms.
- ``none``: take provider order, ~0 ms.

These tests verify the dispatch logic without loading the actual
cross-encoder model (mocked).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ultron.config import WebSearchConfig, UltronConfig
from ultron.web_search.brave import SearchResult


def _result(rank, title, snippet="", url=None):
    return SearchResult(
        url=url or f"https://test/{rank}",
        title=title,
        snippet=snippet,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_default_ranker_is_cross_encoder():
    """2026-05-21 default flipped from llm-only to cross_encoder."""
    cfg = WebSearchConfig()
    assert cfg.ranker == "cross_encoder"


def test_ranker_validates_literal():
    """Only valid values accepted."""
    WebSearchConfig(ranker="llm")
    WebSearchConfig(ranker="none")
    with pytest.raises(Exception):                                       # noqa: PT011
        WebSearchConfig(ranker="bogus")


def test_full_config_ranker_round_trip():
    cfg = UltronConfig.model_validate({
        "web_search": {"ranker": "llm"}
    })
    assert cfg.web_search.ranker == "llm"


# ---------------------------------------------------------------------------
# Dispatch behaviour
# ---------------------------------------------------------------------------


def test_dispatch_none_takes_provider_order(monkeypatch):
    """``ranker: none`` returns results in provider order, no compute."""
    from ultron.web_search import search as search_module

    # Force config to "none"
    cfg = MagicMock()
    cfg.web_search.ranker = "none"
    monkeypatch.setattr(search_module, "get_config", lambda: cfg)

    results = [_result(i, f"t{i}", f"s{i}") for i in range(8)]
    out = search_module._rank_snippets(None, "query", results, top_n=3)
    assert [r.rank for r in out] == [0, 1, 2]


def test_dispatch_cross_encoder_when_default(monkeypatch):
    """``ranker: cross_encoder`` (default) goes through the cross-encoder
    path. We mock the inner ``_rank_snippets_cross_encoder`` and verify
    it's called."""
    from ultron.web_search import search as search_module

    cfg = MagicMock()
    cfg.web_search.ranker = "cross_encoder"
    monkeypatch.setattr(search_module, "get_config", lambda: cfg)

    called = {"cross_encoder": 0, "llm": 0}

    def fake_ce(query, results, top_n=3):
        called["cross_encoder"] += 1
        return results[:top_n]

    def fake_llm(llm, query, results, top_n=3):
        called["llm"] += 1
        return results[:top_n]

    monkeypatch.setattr(
        search_module, "_rank_snippets_cross_encoder", fake_ce,
    )
    monkeypatch.setattr(
        search_module, "_rank_snippets_llm", fake_llm,
    )

    results = [_result(i, f"t{i}") for i in range(8)]
    search_module._rank_snippets(None, "query", results, top_n=3)
    assert called["cross_encoder"] == 1
    assert called["llm"] == 0


def test_dispatch_llm_path(monkeypatch):
    """``ranker: llm`` routes to the legacy LLM ranker."""
    from ultron.web_search import search as search_module

    cfg = MagicMock()
    cfg.web_search.ranker = "llm"
    monkeypatch.setattr(search_module, "get_config", lambda: cfg)

    called = {"cross_encoder": 0, "llm": 0}

    def fake_ce(query, results, top_n=3):
        called["cross_encoder"] += 1
        return results[:top_n]

    def fake_llm(llm, query, results, top_n=3):
        called["llm"] += 1
        return results[:top_n]

    monkeypatch.setattr(
        search_module, "_rank_snippets_cross_encoder", fake_ce,
    )
    monkeypatch.setattr(
        search_module, "_rank_snippets_llm", fake_llm,
    )

    results = [_result(i, f"t{i}") for i in range(8)]
    search_module._rank_snippets(MagicMock(), "query", results, top_n=3)
    assert called["llm"] == 1
    assert called["cross_encoder"] == 0


def test_dispatch_short_lists_skip_ranking():
    """When results <= top_n, all dispatch paths short-circuit
    to ``results[:top_n]`` without invoking any model."""
    from ultron.web_search.search import _rank_snippets

    results = [_result(i, f"t{i}") for i in range(3)]
    out = _rank_snippets(None, "query", results, top_n=3)
    assert out == results


def test_dispatch_empty_results_short_circuits():
    from ultron.web_search.search import _rank_snippets
    assert _rank_snippets(None, "query", [], top_n=3) == []


def test_dispatch_config_failure_falls_back_to_cross_encoder(monkeypatch):
    """If ``get_config()`` raises (test fixture, etc.), dispatch
    defaults to cross_encoder rather than crashing."""
    from ultron.web_search import search as search_module

    def boom():
        raise RuntimeError("simulated config failure")
    monkeypatch.setattr(search_module, "get_config", boom)

    called = {"cross_encoder": 0}

    def fake_ce(query, results, top_n=3):
        called["cross_encoder"] += 1
        return results[:top_n]

    monkeypatch.setattr(
        search_module, "_rank_snippets_cross_encoder", fake_ce,
    )

    results = [_result(i, f"t{i}") for i in range(8)]
    search_module._rank_snippets(None, "query", results, top_n=3)
    assert called["cross_encoder"] == 1


# ---------------------------------------------------------------------------
# Cross-encoder path behaviour
# ---------------------------------------------------------------------------


def test_cross_encoder_returns_provider_order_when_reranker_unavailable(monkeypatch):
    """If ``_get_cross_encoder`` returns None (load failed), the
    cross-encoder ranker returns ``results[:top_n]`` in provider
    order without crashing."""
    from ultron.web_search import search as search_module

    monkeypatch.setattr(search_module, "_get_cross_encoder", lambda: None)
    results = [_result(i, f"t{i}") for i in range(8)]
    out = search_module._rank_snippets_cross_encoder("q", results, top_n=3)
    assert [r.rank for r in out] == [0, 1, 2]


def test_cross_encoder_uses_scores_to_reorder(monkeypatch):
    """Mock the underlying model.predict to return scores in a
    deliberate order; verify the ranker returns candidates sorted
    by descending score."""
    from ultron.web_search import search as search_module

    # Fake reranker that returns deterministic scores
    fake_model = MagicMock()
    # 5 candidates, scores deliberately out of input order
    fake_model.predict.return_value = [0.1, 0.9, 0.3, 0.7, 0.5]
    fake_reranker = MagicMock()
    fake_reranker._model = fake_model
    fake_reranker._ensure_model = MagicMock(return_value=True)

    monkeypatch.setattr(search_module, "_get_cross_encoder",
                        lambda: fake_reranker)

    results = [_result(i, f"t{i}", f"snip {i}") for i in range(5)]
    out = search_module._rank_snippets_cross_encoder("q", results, top_n=3)
    # Highest scores: idx 1 (0.9), idx 3 (0.7), idx 4 (0.5)
    assert [r.rank for r in out] == [1, 3, 4]


def test_cross_encoder_predict_failure_falls_back(monkeypatch):
    """If model.predict raises, return results[:top_n] in provider order."""
    from ultron.web_search import search as search_module

    fake_model = MagicMock()
    fake_model.predict.side_effect = RuntimeError("simulated predict crash")
    fake_reranker = MagicMock()
    fake_reranker._model = fake_model
    fake_reranker._ensure_model = MagicMock(return_value=True)

    monkeypatch.setattr(search_module, "_get_cross_encoder",
                        lambda: fake_reranker)

    results = [_result(i, f"t{i}", f"snip {i}") for i in range(5)]
    out = search_module._rank_snippets_cross_encoder("q", results, top_n=3)
    assert [r.rank for r in out] == [0, 1, 2]


def test_get_cross_encoder_caches_instance(monkeypatch):
    """Multiple calls to ``_get_cross_encoder`` return the same
    instance (cached) so we don't pay model-load cost repeatedly."""
    from ultron.web_search import search as search_module

    # Reset the cache
    search_module._CROSS_ENCODER_CACHE = None

    construct_count = {"n": 0}

    class FakeReranker:
        def __init__(self):
            construct_count["n"] += 1

    monkeypatch.setattr(
        "ultron.memory.reranker.CrossEncoderReranker", FakeReranker,
    )

    r1 = search_module._get_cross_encoder()
    r2 = search_module._get_cross_encoder()
    r3 = search_module._get_cross_encoder()
    assert r1 is r2 is r3
    assert construct_count["n"] == 1


def test_get_cross_encoder_caches_failure(monkeypatch):
    """If construction fails once, the cache stores a sentinel so
    we don't retry on every search query."""
    from ultron.web_search import search as search_module

    search_module._CROSS_ENCODER_CACHE = None
    construct_count = {"n": 0}

    def boom():
        construct_count["n"] += 1
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(
        "ultron.memory.reranker.CrossEncoderReranker", boom,
    )

    assert search_module._get_cross_encoder() is None
    assert search_module._get_cross_encoder() is None
    assert search_module._get_cross_encoder() is None
    assert construct_count["n"] == 1  # only attempted once
