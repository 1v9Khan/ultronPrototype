"""V1-gap A2: ranking helper unit tests.

Pure-function tests; no Qdrant, no embedder. The helpers operate on
plain Python lists so we can assert exact values for tight cases.
"""

from __future__ import annotations

import math
import time

import pytest

from ultron.memory.ranking import (
    CandidateScore,
    RankingWeights,
    compute_composite_score,
    compute_recency_boost,
    compute_redundancy_penalty,
    compute_surprise_score,
    cosine_similarity,
    select_top_k,
)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_orthogonal_vectors_returns_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_identical_vectors_returns_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_opposite_vectors_returns_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_mismatched_length_returns_zero():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_similarity_none_inputs_return_zero():
    assert cosine_similarity(None, [1.0]) == 0.0
    assert cosine_similarity([1.0], None) == 0.0


# ---------------------------------------------------------------------------
# compute_recency_boost
# ---------------------------------------------------------------------------


def test_recency_boost_now_is_full():
    now = 1_000_000.0
    assert compute_recency_boost(now, half_life_days=7.0, now=now) == pytest.approx(1.0)


def test_recency_boost_half_life_old_is_half():
    now = 1_000_000.0
    one_week_ago = now - 7 * 86400.0
    assert compute_recency_boost(
        one_week_ago, half_life_days=7.0, now=now,
    ) == pytest.approx(0.5)


def test_recency_boost_two_half_lives_old_is_quarter():
    now = 1_000_000.0
    two_weeks_ago = now - 14 * 86400.0
    assert compute_recency_boost(
        two_weeks_ago, half_life_days=7.0, now=now,
    ) == pytest.approx(0.25)


def test_recency_boost_zero_timestamp_returns_zero():
    assert compute_recency_boost(0.0, half_life_days=7.0) == 0.0


def test_recency_boost_future_timestamp_clamps_to_one():
    """Clock skew shouldn't produce >1.0 boosts."""
    now = 1_000_000.0
    future = now + 86400.0
    assert compute_recency_boost(
        future, half_life_days=7.0, now=now,
    ) == 1.0


def test_recency_boost_invalid_half_life_returns_zero():
    assert compute_recency_boost(1_000_000.0, half_life_days=0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_surprise_score
# ---------------------------------------------------------------------------


def test_surprise_score_zero_when_primary_dominates():
    """High primary similarity -> already on-topic -> no surprise bonus."""
    same = [1.0, 0.0]
    score = compute_surprise_score(
        candidate_dense=same, primary_dense=same, category_score=0.7,
    )
    # primary_sim = 1.0; category_score = 0.7; max(0, 0.7 - 1.0) = 0
    assert score == 0.0


def test_surprise_score_positive_when_category_high_primary_low():
    candidate = [1.0, 0.0]
    primary = [0.0, 1.0]  # orthogonal -> primary_sim 0
    score = compute_surprise_score(
        candidate_dense=candidate, primary_dense=primary,
        category_score=0.85,
    )
    # primary_sim = 0; category_score = 0.85; max(0, 0.85) = 0.85
    assert score == pytest.approx(0.85)


def test_surprise_score_zero_with_missing_vectors():
    score = compute_surprise_score(
        candidate_dense=None, primary_dense=[1.0],
        category_score=0.85,
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# compute_redundancy_penalty
# ---------------------------------------------------------------------------


def test_redundancy_penalty_zero_for_first_pick():
    assert compute_redundancy_penalty([1.0, 0.0], picked=[]) == 0.0


def test_redundancy_penalty_high_for_duplicate():
    cand = [1.0, 0.0]
    picked = [CandidateScore(
        candidate_id="x", payload={}, rrf_score=1.0,
        dense=[1.0, 0.0],
    )]
    assert compute_redundancy_penalty(cand, picked) == pytest.approx(1.0)


def test_redundancy_penalty_uses_max_over_picked():
    cand = [1.0, 0.0]
    picked = [
        CandidateScore("a", {}, rrf_score=1.0, dense=[0.0, 1.0]),  # 0 sim
        CandidateScore("b", {}, rrf_score=1.0, dense=[1.0, 0.0]),  # 1 sim
    ]
    assert compute_redundancy_penalty(cand, picked) == pytest.approx(1.0)


def test_redundancy_penalty_skips_picked_without_dense():
    cand = [1.0, 0.0]
    picked = [
        CandidateScore("a", {}, rrf_score=1.0, dense=None),
        CandidateScore("b", {}, rrf_score=1.0, dense=[0.5, 0.5]),
    ]
    pen = compute_redundancy_penalty(cand, picked)
    assert 0 < pen < 1


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------


def test_composite_score_combines_components():
    weights = RankingWeights(
        rrf_weight=1.0, recency_weight=0.0,
        surprise_weight=0.0, redundancy_weight=0.0,
    )
    cand = CandidateScore(
        candidate_id="x", payload={"ts": 0.0},
        rrf_score=0.5, dense=[1.0, 0.0],
        category_similarity=0.5,
    )
    score = compute_composite_score(
        cand, weights=weights, primary_dense=[1.0, 0.0], picked=[],
    )
    assert score == pytest.approx(0.5)


def test_composite_score_recency_adds_when_recent():
    now = time.time()
    weights = RankingWeights(
        rrf_weight=1.0, recency_weight=1.0,
        recency_half_life_days=7.0,
        surprise_weight=0.0, redundancy_weight=0.0,
    )
    cand = CandidateScore(
        "x", {"ts": now}, rrf_score=0.5, dense=[1.0, 0.0],
    )
    # rrf=0.5, recency=1.0, surprise=0, redundancy=0 -> 1.5
    score = compute_composite_score(
        cand, weights=weights, primary_dense=[1.0, 0.0],
        picked=[], now=now,
    )
    assert score == pytest.approx(1.5)


def test_composite_score_redundancy_penalises():
    weights = RankingWeights(
        rrf_weight=1.0, recency_weight=0.0,
        surprise_weight=0.0, redundancy_weight=1.0,
    )
    cand = CandidateScore(
        "x", {"ts": 0.0}, rrf_score=1.0, dense=[1.0, 0.0],
    )
    duplicate = CandidateScore(
        "y", {}, rrf_score=1.0, dense=[1.0, 0.0],
    )
    # rrf=1.0, redundancy=1.0 -> 0.0
    score = compute_composite_score(
        cand, weights=weights, primary_dense=None,
        picked=[duplicate],
    )
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# select_top_k
# ---------------------------------------------------------------------------


def test_select_top_k_picks_highest_first():
    weights = RankingWeights(
        rrf_weight=1.0, recency_weight=0.0,
        surprise_weight=0.0, redundancy_weight=0.0,
    )
    candidates = [
        CandidateScore("low", {"ts": 0}, rrf_score=0.1, dense=[1.0, 0.0]),
        CandidateScore("high", {"ts": 0}, rrf_score=0.9, dense=[0.0, 1.0]),
        CandidateScore("mid", {"ts": 0}, rrf_score=0.5, dense=[0.5, 0.5]),
    ]
    picked = select_top_k(candidates, k=2, weights=weights)
    assert [p.candidate_id for p in picked] == ["high", "mid"]


def test_select_top_k_diverges_from_pure_rrf_when_redundancy_active():
    """A high-RRF near-duplicate of an already-picked candidate should
    be passed over for a more diverse alternative."""
    weights = RankingWeights(
        rrf_weight=1.0, recency_weight=0.0,
        surprise_weight=0.0, redundancy_weight=2.0,  # strong penalty
    )
    candidates = [
        CandidateScore("first",     {"ts": 0}, rrf_score=1.0, dense=[1.0, 0.0]),
        CandidateScore("duplicate", {"ts": 0}, rrf_score=0.95, dense=[1.0, 0.0]),
        CandidateScore("diverse",   {"ts": 0}, rrf_score=0.6, dense=[0.0, 1.0]),
    ]
    picked = select_top_k(candidates, k=2, weights=weights)
    assert picked[0].candidate_id == "first"
    # redundancy penalty pushes "duplicate" below "diverse".
    assert picked[1].candidate_id == "diverse"


def test_select_top_k_returns_at_most_k():
    weights = RankingWeights()
    candidates = [
        CandidateScore(str(i), {"ts": 0}, rrf_score=1.0,
                       dense=[float(i), 0.0])
        for i in range(20)
    ]
    picked = select_top_k(candidates, k=5, weights=weights)
    assert len(picked) == 5


def test_select_top_k_empty_input():
    assert select_top_k([], k=5, weights=RankingWeights()) == []


def test_select_top_k_zero_k():
    candidates = [
        CandidateScore("x", {}, rrf_score=1.0, dense=[1.0]),
    ]
    assert select_top_k(candidates, k=0, weights=RankingWeights()) == []


def test_select_top_k_populates_composite_score():
    weights = RankingWeights()
    candidates = [
        CandidateScore("x", {"ts": 0}, rrf_score=0.5, dense=[1.0, 0.0]),
    ]
    picked = select_top_k(candidates, k=1, weights=weights)
    assert picked[0].composite_score > 0  # set during selection
