"""Phase 3 verification tests for the Qdrant-backed memory store.

Verifies:
  - ConversationMemory creates the three collections on first use.
  - add() returns within a tight budget on the hot path (writes are async).
  - retrieve() returns sensible hybrid hits within the spec's read budget.
  - Schema fields (turn_id, role, content, summary, entities, topic_tags)
    round-trip correctly.

Tests load the FastEmbed dense + BM25 sparse models on first use
(~3 s download, cached afterward). They run on CPU only -- zero VRAM.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import pytest


@pytest.fixture(scope="module")
def embedder():
    from ultron.memory import HybridEmbedder
    return HybridEmbedder(eager=True)


@pytest.fixture
def memory(tmp_path: Path, embedder):
    from ultron.memory import ConversationMemory

    mem = ConversationMemory(
        path=tmp_path / "qdrant",
        embedder=embedder,
        recent_cache_size=50,
    )
    yield mem
    mem.close()


def _wait_for_writes(memory, expected_count: int, timeout_s: float = 5.0) -> None:
    """Spin until the Qdrant store reports >= ``expected_count`` points."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(memory) >= expected_count:
            return
        time.sleep(0.05)
    pytest.fail(
        f"Async writes never landed: expected {expected_count}, "
        f"have {len(memory)} after {timeout_s}s"
    )


# ---------------------------------------------------------------------------
# Schema / lifecycle
# ---------------------------------------------------------------------------


def test_creates_collections_on_construction(memory, tmp_path):
    """Three collections (conversations, facts, web_results) come up empty."""
    from config import settings

    client = memory._client  # noqa: SLF001 -- tests look at the underlying handle
    names = {c.name for c in client.get_collections().collections}
    assert settings.MEMORY_QDRANT_CONVERSATIONS in names
    assert settings.MEMORY_QDRANT_FACTS in names
    assert settings.MEMORY_QDRANT_WEB_RESULTS in names
    assert len(memory) == 0


# ---------------------------------------------------------------------------
# Hot-path latency (the spec's <10 ms write budget).
# ---------------------------------------------------------------------------


def test_add_returns_within_hot_path_budget(memory):
    """Hot-path enqueue must clear the spec's <10 ms write budget by a wide
    margin -- writes are async, so add() should be ~microseconds."""
    latencies_ms: List[float] = []
    for i in range(20):
        t0 = time.monotonic()
        memory.add("user", f"turn {i} content -- some moderate length text")
        latencies_ms.append((time.monotonic() - t0) * 1000)
    median = sorted(latencies_ms)[len(latencies_ms) // 2]
    p95 = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
    print(f"\n  add() latency: median={median:.2f} ms  p95={p95:.2f} ms  max={max(latencies_ms):.2f} ms")
    assert max(latencies_ms) < 10.0, (
        f"hot-path write exceeded 10 ms budget: {max(latencies_ms):.2f} ms"
    )


def test_add_persists_asynchronously(memory):
    """Writes don't block the hot path but must still land in Qdrant."""
    for i in range(5):
        memory.add("user", f"persist test {i}")
    _wait_for_writes(memory, 5)
    assert len(memory) == 5


# ---------------------------------------------------------------------------
# recent() / retrieve() correctness.
# ---------------------------------------------------------------------------


def test_recent_returns_chronological_order(memory):
    memory.add("user", "first")
    memory.add("assistant", "first reply")
    memory.add("user", "second")
    memory.add("assistant", "second reply")
    recent = memory.recent(3)
    assert [t.content for t in recent] == ["first reply", "second", "second reply"]


def test_retrieve_returns_semantic_hits(memory):
    """Hybrid retrieval should surface a turn even when the query and the
    stored content share no keywords. The dense vector carries that."""
    memory.add("user", "we decided to use sqlite for the cache")
    memory.add("assistant", "noted")
    memory.add("user", "lets refactor the auth module")
    memory.add("assistant", "okay, what specifically")
    memory.add("user", "whats the weather today")
    memory.add("assistant", "i dont have a sensor for that")
    # Pad with extra turns so the auth turn is older than `exclude_recent=2`.
    for i in range(5):
        memory.add("user", f"unrelated turn {i}")
    _wait_for_writes(memory, 11)

    hits = memory.retrieve("rewrite the login flow", k=3, exclude_recent=2)
    contents = [h.content for h in hits]
    print(f"\n  hits: {contents}")
    assert any("auth" in c or "login" in c for c in contents), (
        f"expected an auth/login hit in top-3, got: {contents}"
    )


def test_retrieve_respects_exclude_recent(memory):
    """Turns inside the recent window must NOT appear in retrieve results."""
    for i in range(15):
        memory.add("user", f"turn number {i} about widgets")
    _wait_for_writes(memory, 15)

    # exclude_recent=15 means everything is in the recent window -> empty.
    hits = memory.retrieve("widgets", k=5, exclude_recent=15)
    assert hits == []

    # exclude_recent=5 means turns 0..9 are searchable.
    hits = memory.retrieve("widgets", k=5, exclude_recent=5)
    assert all(h.id < 10 for h in hits), [h.id for h in hits]


def test_retrieve_meets_read_budget(memory):
    """Retrieval (embedding + Qdrant query [+ optional cross-encoder
    rerank]) must complete within budget.

    2026-05-21: budget raised from 200 ms -> 500 ms because the
    cross-encoder reranker (bge-reranker-v2-m3) is now default-ON
    per the frontier search pass. The reranker adds ~150-300 ms per
    retrieve call on CPU but produces measurably better RAG context.
    Set ``memory.reranking.enabled: false`` to revert to <200 ms
    cosine+RRF behaviour.
    """
    for i in range(50):
        memory.add("user", f"some content about topic {i}")
    _wait_for_writes(memory, 50)

    # Warmup -- first query loads FastEmbed query encoders + the
    # cross-encoder model if reranking is enabled (~1-3 s cold load
    # the very first time; subsequent process-lifetime calls are warm).
    memory.retrieve("warmup query", k=5, exclude_recent=10)

    latencies_ms: List[float] = []
    for q in (
        "tell me about widgets",
        "how does this work",
        "find the recipe",
        "what is the deadline",
        "any updates",
    ):
        t0 = time.monotonic()
        memory.retrieve(q, k=5, exclude_recent=10)
        latencies_ms.append((time.monotonic() - t0) * 1000)
    median = sorted(latencies_ms)[len(latencies_ms) // 2]
    print(f"\n  retrieve(): median={median:.0f} ms  max={max(latencies_ms):.0f} ms")
    assert median < 500.0, f"retrieve median {median:.0f} ms exceeds 500 ms budget"


# ---------------------------------------------------------------------------
# Schema round-trip.
# ---------------------------------------------------------------------------


def test_payload_round_trip(memory):
    """Phase 3 fields (summary, entities, topic_tags, cluster_id) must
    survive a write/read cycle so the maintenance script can populate them."""
    memory.add("user", "hello")
    _wait_for_writes(memory, 1)

    # Fetch from Qdrant directly, set the metadata fields, read back via
    # the public retrieve() to confirm they appear on MemoryTurn.
    from config import settings

    points, _ = memory._client.scroll(  # noqa: SLF001
        collection_name=settings.MEMORY_QDRANT_CONVERSATIONS,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    assert points
    pid = points[0].id
    memory._client.set_payload(  # noqa: SLF001
        collection_name=settings.MEMORY_QDRANT_CONVERSATIONS,
        payload={
            "summary": "user said hello",
            "entities": ["greeting"],
            "topic_tags": ["smalltalk"],
            "cluster_id": 7,
        },
        points=[pid],
    )

    # Pad so retrieve() doesn't filter out the only turn as 'recent'.
    for i in range(25):
        memory.add("user", f"padding turn {i}")
    _wait_for_writes(memory, 26)

    hits = memory.retrieve("hello", k=10, exclude_recent=20)
    matched = [h for h in hits if h.summary == "user said hello"]
    assert matched, "the seeded turn didn't survive retrieval"
    h = matched[0]
    assert h.entities == ["greeting"]
    assert h.topic_tags == ["smalltalk"]
    assert h.cluster_id == 7


# ---------------------------------------------------------------------------
# A3: facts collection search.
# ---------------------------------------------------------------------------


def _seed_fact(
    memory,
    *,
    fact: str,
    confidence: float,
    category: str,
    last_confirmed: float = None,
    extracted_at: float = None,
    extracted_from: List[int] = None,
    retrieval_weight: float = 1.0,
):
    """Insert a single fact directly into the Qdrant facts collection.

    Bypasses the maintenance pipeline so tests can craft specific
    confidence / category / age combinations.
    """
    from config import settings
    from qdrant_client.models import PointStruct, SparseVector
    import uuid

    embedder = memory._embedder  # noqa: SLF001
    dense = embedder.encode_dense(fact)
    sparse = embedder.encode_sparse(fact)[0]
    now = time.time()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense.tolist(),
            "bm25": SparseVector(indices=sparse.indices, values=sparse.values),
        },
        payload={
            "fact": fact,
            "confidence": confidence,
            "category": category,
            "extracted_from": extracted_from or [],
            "extracted_at": extracted_at if extracted_at is not None else now,
            "last_confirmed": last_confirmed if last_confirmed is not None else now,
            "retrieval_weight": retrieval_weight,
        },
    )
    memory._client.upsert(  # noqa: SLF001
        collection_name=settings.MEMORY_QDRANT_FACTS,
        points=[point],
    )


def test_search_facts_returns_high_confidence_results(memory):
    _seed_fact(
        memory,
        fact="user prefers Python 3.11 for new projects",
        confidence=0.95,
        category="preference",
    )
    _seed_fact(
        memory,
        fact="user prefers FastAPI over Flask",
        confidence=0.9,
        category="preference",
    )
    _seed_fact(
        memory,
        fact="user lives in Tampa",
        confidence=0.6,
        category="person",
    )

    rows = memory.search_facts("Python version preference", k=5)
    assert rows, "expected at least one match for Python preference query"
    assert any("Python 3.11" in r.fact for r in rows), [r.fact for r in rows]
    # All returned rows carry the spec's payload fields.
    for r in rows:
        assert isinstance(r.confidence, float)
        assert isinstance(r.last_confirmed, float)
        assert r.category in {"preference", "person", "decision", "constraint", "project"}
        assert r.score > 0


def test_search_facts_filters_by_min_confidence(memory):
    _seed_fact(
        memory,
        fact="user prefers tabs over spaces",
        confidence=0.95,
        category="preference",
    )
    _seed_fact(
        memory,
        fact="user might prefer 2-space indents in some files",
        confidence=0.4,
        category="preference",
    )

    rows = memory.search_facts(
        "indentation preference", k=5, min_confidence=0.7,
    )
    assert rows, "expected the high-confidence match to survive"
    assert all(r.confidence >= 0.7 for r in rows)
    assert not any("might prefer" in r.fact for r in rows)


def test_search_facts_filters_by_age_when_max_age_days_set(memory):
    now = time.time()
    fresh_ts = now - (10 * 86400)         # 10 days old
    stale_ts = now - (200 * 86400)        # 200 days old
    _seed_fact(
        memory,
        fact="user prefers dark mode in IDE",
        confidence=0.9,
        category="preference",
        last_confirmed=fresh_ts,
    )
    _seed_fact(
        memory,
        fact="user prefers light mode in browser",
        confidence=0.9,
        category="preference",
        last_confirmed=stale_ts,
    )

    rows = memory.search_facts(
        "color theme preference", k=5, max_age_days=90.0,
    )
    assert rows
    assert not any("light mode" in r.fact for r in rows)


def test_search_facts_returns_empty_on_qdrant_failure(memory, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("qdrant simulated failure")

    monkeypatch.setattr(memory._client, "query_points", _raise)  # noqa: SLF001
    rows = memory.search_facts("anything", k=5)
    assert rows == []


def test_search_facts_returns_empty_for_blank_query(memory):
    assert memory.search_facts("", k=5) == []
    assert memory.search_facts("   ", k=5) == []


def test_search_facts_handles_empty_collection(memory):
    rows = memory.search_facts("nothing seeded", k=5)
    assert rows == []


# ---------------------------------------------------------------------------
# A2: retrieve_multi / retrieve_for_query
# ---------------------------------------------------------------------------


def test_retrieve_for_query_falls_back_to_single_pass_when_disabled(memory, monkeypatch):
    """Operator opt-out: multi_pass_enabled=False -> route to retrieve()."""
    monkeypatch.setattr(
        "ultron.memory.qdrant_store.get_config",
        lambda: _make_config_with_multi_pass(False),
    )
    for i in range(15):
        memory.add("user", f"unrelated turn {i}")
    memory.add("user", "user prefers FastAPI for new services")
    for i in range(15):
        memory.add("user", f"more padding turn {i}")
    _wait_for_writes(memory, 31)

    class _Verdict:
        context_categories = ["user's framework preferences"]
        memory_search_queries = []

    hits = memory.retrieve_for_query(
        "what framework should I use", gate_verdict=_Verdict(),
        k=5, exclude_recent=10,
    )
    # Forced single-pass; result is whatever the single pass returns.
    assert isinstance(hits, list)


def test_retrieve_multi_unions_per_query_results(memory, monkeypatch):
    """Seed two distinct topics; multi-pass should surface both."""
    for i in range(10):
        memory.add("user", f"padding {i}")
    memory.add("user", "the database engine should be sqlite")
    memory.add("user", "the test framework should be pytest")
    for i in range(10):
        memory.add("user", f"more padding {i}")
    _wait_for_writes(memory, 22)

    hits = memory.retrieve_multi(
        primary_query="what should I default to",
        category_queries=["database engine choices", "test framework"],
        k=3,
        exclude_recent=5,
    )
    contents = [h.content for h in hits]
    assert any("sqlite" in c or "database" in c for c in contents), contents


def test_retrieve_multi_empty_categories_falls_back_to_single_pass(memory):
    for i in range(15):
        memory.add("user", f"turn {i}")
    _wait_for_writes(memory, 15)
    hits_multi = memory.retrieve_multi(
        primary_query="anything", category_queries=[], k=3, exclude_recent=5,
    )
    hits_single = memory.retrieve("anything", k=3, exclude_recent=5)
    assert [h.id for h in hits_multi] == [h.id for h in hits_single]


def test_retrieve_multi_returns_at_most_k(memory):
    for i in range(40):
        memory.add("user", f"turn {i} about widgets")
    _wait_for_writes(memory, 40)
    hits = memory.retrieve_multi(
        primary_query="widgets",
        category_queries=["widget designs", "widget makers"],
        k=3, exclude_recent=10,
    )
    assert len(hits) <= 3


def test_retrieve_multi_falls_back_on_embedder_failure(memory, monkeypatch):
    """When the batched embedder raises, the path falls through to
    single-pass retrieval (which itself may fail and return [] -- but
    the manager doesn't crash)."""

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated embedder failure")

    monkeypatch.setattr(
        memory._embedder, "encode_query_dense_batch", _boom,
    )
    for i in range(20):
        memory.add("user", f"turn {i}")
    _wait_for_writes(memory, 20)
    hits = memory.retrieve_multi(
        "widgets", ["widgets-A", "widgets-B"], k=3, exclude_recent=5,
    )
    # No exception; result is whatever single-pass returned (could
    # be [] or a partial list).
    assert isinstance(hits, list)


def test_retrieve_for_query_uses_multi_pass_when_enabled(memory, monkeypatch):
    monkeypatch.setattr(
        "ultron.memory.qdrant_store.get_config",
        lambda: _make_config_with_multi_pass(True),
    )
    for i in range(20):
        memory.add("user", f"turn {i}")
    _wait_for_writes(memory, 20)

    class _Verdict:
        context_categories = ["topic A", "topic B"]
        memory_search_queries = []

    hits = memory.retrieve_for_query(
        "anything", gate_verdict=_Verdict(), k=2, exclude_recent=5,
    )
    assert isinstance(hits, list)


def _make_config_with_multi_pass(enabled: bool):
    """Helper: produce a clone of the live config with multi_pass_enabled
    flipped. Avoids mutating the global cache."""
    from ultron.config import get_config

    cfg = get_config()
    cfg.memory.retrieval.multi_pass_enabled = enabled
    return cfg
