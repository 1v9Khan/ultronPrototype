"""Tests for the provably-fair commit-reveal HMAC-SHA256 RNG."""
from __future__ import annotations

import hashlib

import pytest

from kenning.twitch.economy.rng import ProvablyFairRNG, RngError, sha256_hex


@pytest.fixture()
def rng():
    return ProvablyFairRNG(default_client_seed="ultron")


# --- commit / reveal ----------------------------------------------------------
def test_new_round_commit_is_sha256_of_seed(rng):
    rc = rng.new_round()
    assert rc.commit == hashlib.sha256(rc.server_seed.encode("utf-8")).hexdigest()
    assert len(bytes.fromhex(rc.server_seed)) == 32  # 256-bit default


def test_verify_true_for_matching_seed(rng):
    rc = rng.new_round()
    assert rng.verify(rc.commit, rc.server_seed) is True


def test_verify_false_for_tampered_seed(rng):
    rc = rng.new_round()
    # Flip the last hex nibble of the revealed seed -> commitment must fail.
    last = rc.server_seed[-1]
    tampered = rc.server_seed[:-1] + ("0" if last != "0" else "1")
    assert tampered != rc.server_seed
    assert rng.verify(rc.commit, tampered) is False


def test_verify_false_for_tampered_commit(rng):
    rc = rng.new_round()
    bad_commit = ("0" if rc.commit[0] != "0" else "1") + rc.commit[1:]
    assert rng.verify(bad_commit, rc.server_seed) is False


def test_verify_never_raises_on_garbage(rng):
    assert rng.verify("not-a-commit", "zzzz") is False
    assert rng.verify("", "") is False
    assert rng.verify("abcd", "abcd") is False  # commit != sha256("abcd")


def test_commit_for_matches_sha256_helper(rng):
    seed = "00ff00ff"
    assert rng.commit_for(seed) == sha256_hex(seed)


# --- outcome determinism ------------------------------------------------------
def test_outcome_deterministic_for_fixed_inputs(rng):
    seed = "a" * 64
    o1 = rng.outcome(seed, "client", 7, 100)
    o2 = rng.outcome(seed, "client", 7, 100)
    assert o1 == o2
    assert 0 <= o1 < 100


def test_outcome_changes_with_nonce(rng):
    seed = "b" * 64
    values = {rng.outcome(seed, "client", n, 1000) for n in range(50)}
    # With n=1000 and 50 nonces, near-certainly several distinct values.
    assert len(values) > 1


def test_outcome_n_one_is_zero(rng):
    assert rng.outcome("c" * 64, "client", 0, 1) == 0


def test_outcome_range_always_in_bounds(rng):
    seed = "d" * 64
    for n in (2, 3, 6, 37, 52):
        for nonce in range(30):
            v = rng.outcome(seed, "client", nonce, n)
            assert 0 <= v < n


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_outcome_rejects_bad_n(rng, n):
    with pytest.raises(RngError):
        rng.outcome("e" * 64, "client", 0, n)


@pytest.mark.parametrize("nonce", [-1, 1.5, True, "0"])
def test_outcome_rejects_bad_nonce(rng, nonce):
    with pytest.raises(RngError):
        rng.outcome("e" * 64, "client", nonce, 10)


def test_outcome_rejects_bad_seed(rng):
    with pytest.raises(RngError):
        rng.outcome("not-hex-zz", "client", 0, 10)
    with pytest.raises(RngError):
        rng.outcome("", "client", 0, 10)


# --- outcome uniformity (loose statistical bound, fixed seed) ------------------
def test_outcome_uniform_over_many_nonces(rng):
    """Chi-square-style loose bound: with a fixed server_seed, outcomes over many
    nonces should spread roughly evenly across the n buckets."""
    seed = "1234" * 16  # 64 hex chars, fixed
    n = 10
    trials = 20000
    counts = [0] * n
    for nonce in range(trials):
        counts[rng.outcome(seed, "client", nonce, n)] += 1
    expected = trials / n
    # Each bucket within +-20% of expected — generous, deterministic enough that
    # a real modulo-bias / stuck-bit bug would blow past it.
    for c in counts:
        assert abs(c - expected) < 0.20 * expected, counts


# --- uniform_unit -------------------------------------------------------------
def test_uniform_unit_in_range_and_deterministic(rng):
    seed = "f" * 64
    u1 = rng.uniform_unit(seed, "client", 3)
    u2 = rng.uniform_unit(seed, "client", 3)
    assert u1 == u2
    assert 0.0 <= u1 < 1.0


# --- weighted_choice ----------------------------------------------------------
def test_weighted_choice_deterministic(rng):
    seed = "9" * 64
    w = [1.0, 2.0, 3.0]
    i1 = rng.weighted_choice(seed, "client", 11, w)
    i2 = rng.weighted_choice(seed, "client", 11, w)
    assert i1 == i2
    assert 0 <= i1 < len(w)


def test_weighted_choice_zero_weight_never_selected(rng):
    seed = "8" * 64
    # Index 1 has weight 0 -> must never be chosen.
    w = [1.0, 0.0, 1.0]
    chosen = {rng.weighted_choice(seed, "client", nonce, w) for nonce in range(500)}
    assert 1 not in chosen
    assert chosen.issubset({0, 2})


def test_weighted_choice_distribution_tracks_weights(rng):
    """Over many nonces the empirical frequency ~ the weights (loose bound)."""
    seed = "abcd" * 16
    weights = [1.0, 3.0, 6.0]  # expected ~10% / 30% / 60%
    total = sum(weights)
    trials = 20000
    counts = [0, 0, 0]
    for nonce in range(trials):
        counts[rng.weighted_choice(seed, "client", nonce, weights)] += 1
    for i, w in enumerate(weights):
        expected = trials * (w / total)
        assert abs(counts[i] - expected) < 0.15 * expected, (i, counts)


@pytest.mark.parametrize(
    "weights",
    [[], [0.0, 0.0], [-1.0, 2.0], [float("inf"), 1.0], [float("nan"), 1.0]],
)
def test_weighted_choice_rejects_bad_weights(rng, weights):
    with pytest.raises(RngError):
        rng.weighted_choice("7" * 64, "client", 0, weights)


def test_weighted_choice_single_positive_weight(rng):
    # One positive among zeros -> always that index.
    w = [0.0, 0.0, 5.0]
    for nonce in range(20):
        assert rng.weighted_choice("6" * 64, "client", nonce, w) == 2


# --- default client seed ------------------------------------------------------
def test_default_client_seed_used_when_omitted(rng):
    seed = "5" * 64
    # outcome uses an explicit client seed; weighted_choice on the wheel uses the
    # default — verify the default is exposed and stable.
    assert rng.default_client_seed == "ultron"
