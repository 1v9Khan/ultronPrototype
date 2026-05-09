"""Tests for the 2026-05-09 nuanced-retrieval pass.

Two new behaviours land in :class:`ConversationMemory.retrieve`:

1. **Cosine relevance threshold** -- candidates whose dense-vector
   cosine similarity to the query is below
   ``memory.rag_min_relevance`` are dropped from the result set
   entirely (not just downranked). This is the fix for
   "apex-predator chatter contaminating a Paris-weather query".

2. **Composite scoring with recency** -- candidates that pass the
   threshold are ranked by ``cosine_sim + RRF + recency_boost`` so
   recent-and-relevant ranks ahead of old-and-relevant.

These tests mock the Qdrant client + embedder so we can hand-craft
the cosine relationships and verify the filter / sort behaviour
without spinning up a real vector store.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ultron.config import UltronConfig, set_config
from ultron.memory.qdrant_store import (
    ConversationMemory,
    _payload_to_turn,
)


def _make_qdrant_point(
    *,
    turn_id: int,
    role: str,
    content: str,
    rrf_score: float,
    dense_vector: list,
    ts: float = 0.0,
):
    """Build a fake Qdrant ScoredPoint return value for query_points()."""
    return SimpleNamespace(
        score=rrf_score,
        payload={
            "turn_id": turn_id,
            "role": role,
            "content": content,
            "ts": ts,
            "session_id": "test",
        },
        vector={"dense": list(dense_vector)},
    )


def _build_memory_with_mock_client(
    *, points, query_dense, next_id=100,
):
    """Construct a ConversationMemory whose Qdrant client returns the
    given canned points. Bypasses __init__ so we don't load the real
    embedder / open the Qdrant store on disk.
    """
    mem = object.__new__(ConversationMemory)
    mem._next_id = next_id
    import threading
    mem._lock = threading.Lock()
    mem._recent = []

    # Mock embedder: query_dense returns a fixed vector; sparse stub.
    mem._embedder = MagicMock()
    mem._embedder.encode_query_dense.return_value = np.array(
        query_dense, dtype=np.float32,
    )
    sparse = SimpleNamespace(indices=[1], values=[0.1])
    mem._embedder.encode_query_sparse.return_value = sparse

    # Mock Qdrant client: query_points returns the canned points.
    response = SimpleNamespace(points=list(points))
    mem._client = MagicMock()
    mem._client.query_points.return_value = response
    return mem


@pytest.fixture
def relevance_config():
    """Activate the relevance filter at threshold 0.4."""
    cfg = UltronConfig()
    cfg.memory.rag_min_relevance = 0.4
    cfg.memory.rag_top_k = 5
    cfg.memory.rag_exclude_recent = 20
    set_config(cfg)
    yield cfg
    set_config(UltronConfig())


@pytest.fixture
def threshold_disabled():
    """rag_min_relevance=0 -> legacy fast path (no filter, no sort)."""
    cfg = UltronConfig()
    cfg.memory.rag_min_relevance = 0.0
    cfg.memory.rag_top_k = 5
    cfg.memory.rag_exclude_recent = 20
    set_config(cfg)
    yield cfg
    set_config(UltronConfig())


# ---------------------------------------------------------------------------
# Threshold filtering
# ---------------------------------------------------------------------------


def test_retrieve_drops_below_threshold(relevance_config):
    """Cosine < threshold -> candidate dropped, not just downranked."""
    # Query vector points along axis 0.
    query_dense = [1.0, 0.0, 0.0]
    # Two candidates: one aligned (cosine 1.0), one orthogonal (cosine 0.0).
    points = [
        _make_qdrant_point(
            turn_id=1, role="user", content="aligned topic",
            rrf_score=0.05, dense_vector=[1.0, 0.0, 0.0],
        ),
        _make_qdrant_point(
            turn_id=2, role="assistant",
            content="apex predator chatter (orthogonal)",
            rrf_score=0.04, dense_vector=[0.0, 1.0, 0.0],
        ),
    ]
    mem = _build_memory_with_mock_client(
        points=points, query_dense=query_dense,
    )
    hits = mem.retrieve("aligned query", k=5, exclude_recent=10)
    contents = [h.content for h in hits]
    assert "aligned topic" in contents
    assert "apex predator chatter (orthogonal)" not in contents


def test_retrieve_returns_empty_when_nothing_clears_threshold(relevance_config):
    """If every candidate is orthogonal to query, retrieve returns []."""
    query_dense = [1.0, 0.0, 0.0]
    points = [
        _make_qdrant_point(
            turn_id=1, role="user", content="off-topic A",
            rrf_score=0.05, dense_vector=[0.0, 1.0, 0.0],
        ),
        _make_qdrant_point(
            turn_id=2, role="assistant", content="off-topic B",
            rrf_score=0.04, dense_vector=[0.0, 0.0, 1.0],
        ),
    ]
    mem = _build_memory_with_mock_client(
        points=points, query_dense=query_dense,
    )
    hits = mem.retrieve("query about topic X", k=5, exclude_recent=10)
    assert hits == []


def test_retrieve_keeps_high_relevance_old_turn(relevance_config):
    """Old turns that ARE highly relevant still get retrieved.

    Mirrors the user's "month-old similar troubleshooting" use case.
    """
    import time
    query_dense = [1.0, 0.0, 0.0]
    one_month_ago = time.time() - (30 * 86400)
    points = [
        _make_qdrant_point(
            turn_id=1, role="user",
            content="similar problem from a month ago",
            rrf_score=0.04, dense_vector=[0.95, 0.05, 0.0],
            ts=one_month_ago,
        ),
    ]
    mem = _build_memory_with_mock_client(
        points=points, query_dense=query_dense,
    )
    hits = mem.retrieve("current similar problem", k=5, exclude_recent=10)
    assert len(hits) == 1
    assert "month ago" in hits[0].content


# ---------------------------------------------------------------------------
# Recency-weighted ordering
# ---------------------------------------------------------------------------


def test_retrieve_prefers_recent_over_old_when_both_relevant(relevance_config):
    """Two equally-relevant candidates -- recent ranks first."""
    import time
    query_dense = [1.0, 0.0, 0.0]
    now = time.time()
    points = [
        _make_qdrant_point(
            turn_id=10, role="user", content="old relevant",
            rrf_score=0.05, dense_vector=[1.0, 0.0, 0.0],
            ts=now - (60 * 86400),  # 60 days old
        ),
        _make_qdrant_point(
            turn_id=11, role="user", content="recent relevant",
            rrf_score=0.05, dense_vector=[1.0, 0.0, 0.0],
            ts=now - (1 * 3600),    # 1 hour ago
        ),
    ]
    mem = _build_memory_with_mock_client(
        points=points, query_dense=query_dense,
    )
    hits = mem.retrieve("relevant query", k=2, exclude_recent=5)
    assert [h.content for h in hits] == ["recent relevant", "old relevant"]


# ---------------------------------------------------------------------------
# Legacy fast path
# ---------------------------------------------------------------------------


def test_retrieve_legacy_fast_path_when_threshold_zero(threshold_disabled):
    """rag_min_relevance=0 -> RRF order returned as-is, no filtering.

    Backwards-compatibility check: setting the threshold to 0 must
    reproduce the pre-2026-05-09 single-pass behaviour exactly.
    """
    query_dense = [1.0, 0.0, 0.0]
    # Even though one candidate is orthogonal, with threshold=0 it
    # comes through.
    points = [
        _make_qdrant_point(
            turn_id=1, role="user", content="orthogonal hit",
            rrf_score=0.05, dense_vector=[0.0, 1.0, 0.0],
        ),
    ]
    mem = _build_memory_with_mock_client(
        points=points, query_dense=query_dense,
    )
    hits = mem.retrieve("query", k=5, exclude_recent=5)
    assert len(hits) == 1
    assert hits[0].content == "orthogonal hit"


def test_retrieve_caps_at_k(relevance_config):
    """Threshold filter doesn't break the top-K cap."""
    query_dense = [1.0, 0.0, 0.0]
    # 5 above threshold + 1 below.
    points = [
        _make_qdrant_point(
            turn_id=i, role="user", content=f"hit_{i}",
            rrf_score=0.05 - (0.001 * i),
            dense_vector=[1.0, 0.05 * i, 0.0],
        )
        for i in range(5)
    ] + [
        _make_qdrant_point(
            turn_id=99, role="user", content="orthogonal",
            rrf_score=0.04, dense_vector=[0.0, 1.0, 0.0],
        ),
    ]
    mem = _build_memory_with_mock_client(
        points=points, query_dense=query_dense,
    )
    hits = mem.retrieve("query", k=3, exclude_recent=5)
    assert len(hits) == 3
    assert all("orthogonal" not in h.content for h in hits)


# ---------------------------------------------------------------------------
# History-cap in LLM message assembly
# ---------------------------------------------------------------------------


def test_history_turns_for_llm_caps_recent_feed():
    """``memory.history_turns_for_llm`` caps how many recent turns are
    appended to the LLM message list. Independent of cache size
    (recent_turns)."""
    from ultron.llm.inference import LLMEngine

    cfg = UltronConfig()
    cfg.memory.recent_turns = 20
    cfg.memory.history_turns_for_llm = 4
    set_config(cfg)
    try:
        eng = object.__new__(LLMEngine)
        eng._explicit_system_prompt = "system"
        eng._persona_loader = None
        eng.system_prompt = "system"
        eng._logged_initial_persona = True
        eng._history = []

        # Fake memory: recent() returns whatever count is asked for.
        recent_calls = {"asked_for": None}

        class _FakeMem:
            def recent(self, n):
                recent_calls["asked_for"] = n
                return [
                    SimpleNamespace(role="user", content=f"u{i}")
                    for i in range(n)
                ]

            def retrieve(self, *a, **k):
                return []

        eng._memory = _FakeMem()

        eng._build_messages("hello")

        # We asked memory.recent for the capped number, not 20.
        assert recent_calls["asked_for"] == 4
    finally:
        set_config(UltronConfig())


def test_history_turns_for_llm_respects_cache_size_floor():
    """When the cache size is smaller than the cap, we use the cache size."""
    from ultron.llm.inference import LLMEngine

    cfg = UltronConfig()
    cfg.memory.recent_turns = 2
    cfg.memory.history_turns_for_llm = 10
    set_config(cfg)
    try:
        eng = object.__new__(LLMEngine)
        eng._explicit_system_prompt = "system"
        eng._persona_loader = None
        eng.system_prompt = "system"
        eng._logged_initial_persona = True
        eng._history = []

        recent_calls = {"asked_for": None}

        class _FakeMem:
            def recent(self, n):
                recent_calls["asked_for"] = n
                return []

            def retrieve(self, *a, **k):
                return []

        eng._memory = _FakeMem()

        eng._build_messages("hello")
        # Capped at min(history_turns_for_llm=10, recent_turns=2) -> 2.
        assert recent_calls["asked_for"] == 2
    finally:
        set_config(UltronConfig())
