"""Cross-encoder reranker tests (frontier item 2, 2026-05-21).

Unit-level tests for :class:`ultron.memory.reranker.CrossEncoderReranker`
+ integration touch-points on :class:`ConversationMemory._apply_reranker`.

No real model download happens here -- ``sentence_transformers.CrossEncoder``
is mocked. The tests verify:
- Config schema accepts the new fields with safe defaults.
- Reranker constructs lazily; first ``rerank`` triggers load.
- ``rerank`` returns top-k ordered by cross-encoder score (highest first).
- Empty query / empty candidates / top_k<=0 returns pre-rerank order.
- Model load failure is fail-open (pre-rerank order returned, no raise).
- ``predict`` failure on a specific batch is fail-open.
- ``rerank_with_scores`` carries score + pre-rerank index correctly.
- ``ConversationMemory._apply_reranker`` lazily constructs the reranker
  and fail-opens if construction fails.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ultron.config import MemoryRerankingConfig, MemoryConfig, UltronConfig


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_reranking_config_defaults():
    """2026-05-21 frontier search pass: default flipped to True
    now that the cross-encoder is also used for web-search ranking.
    Model loads once per process; memory.retrieve() pays ~265 ms
    per call in exchange for better RAG quality."""
    cfg = MemoryRerankingConfig()
    assert cfg.enabled is True
    assert cfg.model == "BAAI/bge-reranker-v2-m3"
    assert cfg.device == "cpu"
    assert cfg.max_length == 512
    assert cfg.candidate_count == 20


def test_reranking_config_validates_ranges():
    with pytest.raises(Exception):                                       # noqa: PT011
        MemoryRerankingConfig(max_length=10)        # below 64
    with pytest.raises(Exception):                                       # noqa: PT011
        MemoryRerankingConfig(max_length=3000)      # above 2048
    with pytest.raises(Exception):                                       # noqa: PT011
        MemoryRerankingConfig(candidate_count=0)
    with pytest.raises(Exception):                                       # noqa: PT011
        MemoryRerankingConfig(candidate_count=101)


def test_memory_config_includes_reranking():
    """2026-05-21: default flipped to True alongside the web-search
    cross-encoder rollout (shared model)."""
    cfg = MemoryConfig()
    assert hasattr(cfg, "reranking")
    assert cfg.reranking.enabled is True


def test_full_config_round_trip_enables_reranking():
    cfg = UltronConfig.model_validate({
        "memory": {
            "reranking": {
                "enabled": True,
                "candidate_count": 30,
                "device": "cpu",
            }
        }
    })
    assert cfg.memory.reranking.enabled is True
    assert cfg.memory.reranking.candidate_count == 30


# ---------------------------------------------------------------------------
# CrossEncoderReranker behaviour
# ---------------------------------------------------------------------------


def _mock_turn(turn_id: int, content: str):
    """Build a stand-in for a MemoryTurn carrying just the fields the
    reranker reads."""
    return SimpleNamespace(id=turn_id, content=content)


def _stub_cross_encoder_with_scores(scores):
    """Return a MagicMock CrossEncoder whose .predict returns ``scores``."""
    mock_ce_instance = MagicMock()
    mock_ce_instance.predict.return_value = list(scores)
    mock_ce_cls = MagicMock()
    mock_ce_cls.return_value = mock_ce_instance
    return mock_ce_cls, mock_ce_instance


def test_rerank_empty_query_returns_prefix(monkeypatch):
    """Empty / whitespace-only query falls back to pre-rerank order
    (no model load, no predict call)."""
    from ultron.memory.reranker import CrossEncoderReranker

    cands = [_mock_turn(i, f"text {i}") for i in range(5)]
    rr = CrossEncoderReranker()
    out = rr.rerank("", cands, 3)
    assert [c.id for c in out] == [0, 1, 2]
    out2 = rr.rerank("   ", cands, 2)
    assert [c.id for c in out2] == [0, 1]


def test_rerank_empty_candidates_returns_empty():
    from ultron.memory.reranker import CrossEncoderReranker

    rr = CrossEncoderReranker()
    assert rr.rerank("query", [], 5) == []


def test_rerank_top_k_zero_returns_empty():
    from ultron.memory.reranker import CrossEncoderReranker

    cands = [_mock_turn(i, f"text {i}") for i in range(5)]
    rr = CrossEncoderReranker()
    assert rr.rerank("query", cands, 0) == []


def test_rerank_orders_by_score_desc(monkeypatch):
    """Highest cross-encoder score wins position 0."""
    from ultron.memory import reranker as reranker_module

    cands = [
        _mock_turn(0, "irrelevant text"),
        _mock_turn(1, "highly relevant text"),
        _mock_turn(2, "kind-of relevant text"),
    ]
    # Scores deliberately out-of-order so the reranker has work to do.
    mock_ce_cls, mock_ce = _stub_cross_encoder_with_scores([0.1, 0.9, 0.5])
    monkeypatch.setattr(
        reranker_module, "CrossEncoder", mock_ce_cls, raising=False
    )
    # Also patch the lazy import inside _ensure_model.
    import sys
    fake_st_mod = SimpleNamespace(CrossEncoder=mock_ce_cls)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    rr = reranker_module.CrossEncoderReranker()
    out = rr.rerank("query", cands, 3)
    assert [c.id for c in out] == [1, 2, 0]

    # predict was called once with the (query, content) pairs.
    args, _ = mock_ce.predict.call_args
    pairs = args[0]
    assert pairs == [
        ("query", "irrelevant text"),
        ("query", "highly relevant text"),
        ("query", "kind-of relevant text"),
    ]


def test_rerank_with_scores_carries_indices(monkeypatch):
    """``rerank_with_scores`` preserves the original (pre-rerank) index
    so the caller can debug which item was bumped from where."""
    from ultron.memory import reranker as reranker_module

    cands = [
        _mock_turn(10, "A"),
        _mock_turn(20, "B"),
        _mock_turn(30, "C"),
    ]
    mock_ce_cls, _ = _stub_cross_encoder_with_scores([0.5, 0.9, 0.7])
    import sys
    fake_st_mod = SimpleNamespace(CrossEncoder=mock_ce_cls)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    rr = reranker_module.CrossEncoderReranker()
    out = rr.rerank_with_scores("q", cands, 3)
    # Order: B (0.9) > C (0.7) > A (0.5)
    assert [r.turn.id for r in out] == [20, 30, 10]
    assert [r.pre_rerank_index for r in out] == [1, 2, 0]
    assert pytest.approx(out[0].score, rel=1e-6) == 0.9


def test_rerank_truncates_to_top_k(monkeypatch):
    """``top_k`` smaller than candidate count drops the tail."""
    from ultron.memory import reranker as reranker_module

    cands = [_mock_turn(i, f"text {i}") for i in range(5)]
    mock_ce_cls, _ = _stub_cross_encoder_with_scores([0.1, 0.9, 0.5, 0.3, 0.7])
    import sys
    fake_st_mod = SimpleNamespace(CrossEncoder=mock_ce_cls)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    rr = reranker_module.CrossEncoderReranker()
    out = rr.rerank("q", cands, 2)
    # Order: 1 (0.9) > 4 (0.7) > 2 (0.5) > 3 (0.3) > 0 (0.1)
    # Top-2: [1, 4]
    assert [c.id for c in out] == [1, 4]


def test_rerank_model_load_failure_is_fail_open(monkeypatch, caplog):
    """If CrossEncoder construction raises, the reranker logs WARN
    and returns the pre-rerank order. NEVER raises."""
    from ultron.memory.reranker import CrossEncoderReranker
    import sys

    def boom(*_a, **_kw):
        raise RuntimeError("simulated load failure")

    fake_st_mod = SimpleNamespace(CrossEncoder=boom)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    cands = [_mock_turn(i, f"t{i}") for i in range(3)]
    rr = CrossEncoderReranker()
    out = rr.rerank("query", cands, 5)
    # Pre-rerank order preserved, NOT raised.
    assert [c.id for c in out] == [0, 1, 2]


def test_rerank_predict_failure_is_fail_open(monkeypatch):
    """If predict raises mid-call, return pre-rerank order."""
    from ultron.memory.reranker import CrossEncoderReranker
    import sys

    bad_ce = MagicMock()
    bad_ce.predict.side_effect = RuntimeError("simulated predict failure")
    fake_st_mod = SimpleNamespace(CrossEncoder=MagicMock(return_value=bad_ce))
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    cands = [_mock_turn(i, f"t{i}") for i in range(3)]
    rr = CrossEncoderReranker()
    out = rr.rerank("query", cands, 5)
    assert [c.id for c in out] == [0, 1, 2]


def test_rerank_lazy_load(monkeypatch):
    """Construction does NOT load the model; first ``rerank`` does."""
    from ultron.memory import reranker as reranker_module
    import sys

    load_count = {"n": 0}

    class LazyMockCE:
        def __init__(self, *a, **kw):
            load_count["n"] += 1
            self._scores = [0.5]

        def predict(self, pairs, **kw):
            return [0.5] * len(pairs)

    fake_st_mod = SimpleNamespace(CrossEncoder=LazyMockCE)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    rr = reranker_module.CrossEncoderReranker()
    assert load_count["n"] == 0  # not loaded at construction
    rr.rerank("q", [_mock_turn(1, "x")], 1)
    assert load_count["n"] == 1  # loaded on first call
    rr.rerank("q", [_mock_turn(2, "y")], 1)
    assert load_count["n"] == 1  # cached on second call


def test_rerank_eager_loads_at_construction(monkeypatch):
    """``eager=True`` at construction triggers an immediate load."""
    from ultron.memory import reranker as reranker_module
    import sys

    load_count = {"n": 0}

    class LazyMockCE:
        def __init__(self, *a, **kw):
            load_count["n"] += 1

        def predict(self, *a, **kw):
            return [0.0]

    fake_st_mod = SimpleNamespace(CrossEncoder=LazyMockCE)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)

    reranker_module.CrossEncoderReranker(eager=True)
    assert load_count["n"] == 1


# ---------------------------------------------------------------------------
# ConversationMemory._apply_reranker (the integration touch point)
# ---------------------------------------------------------------------------


def test_apply_reranker_empty_candidates_returns_empty():
    """``_apply_reranker`` short-circuits on empty candidates."""
    from ultron.memory.qdrant_store import ConversationMemory
    cm = object.__new__(ConversationMemory)
    cm._reranker = None
    assert cm._apply_reranker("q", [], 5) == []


def test_apply_reranker_top_k_zero_returns_empty():
    from ultron.memory.qdrant_store import ConversationMemory
    cm = object.__new__(ConversationMemory)
    cm._reranker = None
    cands = [_mock_turn(i, f"t{i}") for i in range(3)]
    assert cm._apply_reranker("q", cands, 0) == []


def test_apply_reranker_construction_failure_is_fail_open(monkeypatch):
    """If the shared reranker factory raises, we return the pre-rerank
    order, not crash."""
    from ultron.memory.qdrant_store import ConversationMemory
    from ultron.memory import reranker as reranker_module

    # 2026-05-22: _apply_reranker now routes through the module-level
    # ``get_shared_reranker`` singleton; patch THAT.
    def boom():
        raise RuntimeError("simulated construction failure")

    monkeypatch.setattr(
        reranker_module, "get_shared_reranker", boom, raising=True,
    )
    reranker_module.reset_shared_reranker()

    cm = object.__new__(ConversationMemory)
    cm._reranker = None
    cands = [_mock_turn(i, f"t{i}") for i in range(4)]
    out = cm._apply_reranker("query", cands, 2)
    assert [c.id for c in out] == [0, 1]


def test_apply_reranker_caches_instance(monkeypatch):
    """``_apply_reranker`` fetches the shared reranker once and caches
    it on ``self._reranker``."""
    from ultron.memory.qdrant_store import ConversationMemory
    from ultron.memory import reranker as reranker_module

    construct_count = {"n": 0}

    class FakeReranker:
        def __init__(self, *a, **kw):
            construct_count["n"] += 1

        def rerank(self, q, c, k):
            return list(c)[:k]

    # 2026-05-22: patch the shared-factory function so the per-instance
    # cache test still validates "construct once, reuse forever".
    fake = FakeReranker()  # one fake instance shared across calls
    monkeypatch.setattr(
        reranker_module, "get_shared_reranker", lambda: fake, raising=True,
    )
    reranker_module.reset_shared_reranker()

    cm = object.__new__(ConversationMemory)
    cm._reranker = None
    cands = [_mock_turn(i, f"t{i}") for i in range(3)]
    cm._apply_reranker("q1", cands, 2)
    cm._apply_reranker("q2", cands, 2)
    assert construct_count["n"] == 1
