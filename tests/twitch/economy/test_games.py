"""Tests for provably-fair games: SpinTheWheel + Slots."""
from __future__ import annotations

import pytest

from kenning.twitch.economy.games import (
    LOSE_ALL,
    GameError,
    Slots,
    SpinTheWheel,
    WheelSegment,
)
from kenning.twitch.economy.rng import ProvablyFairRNG


@pytest.fixture()
def rng():
    return ProvablyFairRNG(default_client_seed="ultron")


FIXED_SEED = "1234abcd" * 8  # 64 hex chars


# --- wheel: basic + determinism -----------------------------------------------
def test_wheel_spin_deterministic(rng):
    wheel = SpinTheWheel(
        [
            WheelSegment("nothing", 1.0, 0),
            WheelSegment("small", 2.0, 50),
            WheelSegment("jackpot", 1.0, 500),
        ],
        rng=rng,
    )
    r1 = wheel.spin(FIXED_SEED, "client", 5)
    r2 = wheel.spin(FIXED_SEED, "client", 5)
    assert r1.index == r2.index
    assert r1.target_angle == r2.target_angle
    assert r1.segment.label == r2.segment.label


def test_wheel_target_angle_within_chosen_arc(rng):
    wheel = SpinTheWheel(
        [
            WheelSegment("a", 1.0, 0),
            WheelSegment("b", 2.0, 0),
            WheelSegment("c", 3.0, 0),
            WheelSegment("d", 4.0, 0),
        ],
        rng=rng,
    )
    for nonce in range(200):
        res = wheel.spin(FIXED_SEED, "client", nonce)
        # The chosen target angle lands inside the chosen segment's arc.
        assert wheel.angle_in_chosen_arc(res), (
            nonce, res.index, res.target_angle, res.arc_start, res.arc_span,
        )
        # And strictly within [start, start+span).
        assert res.arc_start <= res.target_angle < res.arc_start + res.arc_span


def test_wheel_arc_spans_proportional_to_weight(rng):
    wheel = SpinTheWheel(
        [WheelSegment("a", 1.0, 0), WheelSegment("b", 3.0, 0)],
        rng=rng,
    )
    # Spin many nonces; segment b (weight 3) should win ~3x as often as a.
    counts = {0: 0, 1: 0}
    for nonce in range(4000):
        counts[wheel.spin(FIXED_SEED, "client", nonce).index] += 1
    ratio = counts[1] / max(counts[0], 1)
    assert 2.4 < ratio < 3.6, counts


def test_wheel_provenance_commit_matches(rng):
    wheel = SpinTheWheel([WheelSegment("a", 1.0, 0)], rng=rng)
    res = wheel.spin(FIXED_SEED, "client", 0)
    prov = res.provenance
    assert prov.game == "spin_the_wheel"
    assert prov.server_seed == FIXED_SEED
    # The published commit re-verifies against the revealed seed.
    assert rng.verify(prov.commit, prov.server_seed) is True


# --- wheel: LOSE ALL gating (AT-4) --------------------------------------------
def test_lose_all_off_by_default_never_selected(rng):
    wheel = SpinTheWheel(
        [
            WheelSegment("safe", 1.0, 10),
            WheelSegment("RUIN", 1.0, LOSE_ALL),  # huge weight, still inert
        ],
        rng=rng,
        # allow_lose_all defaults False
    )
    assert wheel.allow_lose_all is False
    chosen_labels = set()
    for nonce in range(1000):
        res = wheel.spin(FIXED_SEED, "client", nonce)
        chosen_labels.add(res.segment.label)
        assert res.segment.consequence != LOSE_ALL
    # The LOSE_ALL segment is never landed on while gated off.
    assert "RUIN" not in chosen_labels
    assert chosen_labels == {"safe"}


def test_lose_all_with_huge_weight_still_inert_when_gated_off(rng):
    wheel = SpinTheWheel(
        [
            WheelSegment("safe", 1.0, 10),
            WheelSegment("RUIN", 1000.0, LOSE_ALL),
        ],
        rng=rng,
    )
    for nonce in range(500):
        assert wheel.spin(FIXED_SEED, "client", nonce).segment.label == "safe"


def test_lose_all_selectable_when_explicitly_enabled(rng):
    wheel = SpinTheWheel(
        [
            WheelSegment("safe", 1.0, 10),
            WheelSegment("RUIN", 1.0, LOSE_ALL),
        ],
        rng=rng,
        allow_lose_all=True,
    )
    assert wheel.allow_lose_all is True
    seen_ruin = any(
        wheel.spin(FIXED_SEED, "client", nonce).segment.consequence == LOSE_ALL
        for nonce in range(1000)
    )
    assert seen_ruin, "with allow_lose_all=True the LOSE_ALL segment must be reachable"


def test_wheel_all_lose_all_gated_off_is_an_error(rng):
    # If the ONLY segments are LOSE_ALL and they're gated off, nothing is
    # selectable -> construction must fail rather than silently spin forever.
    with pytest.raises(GameError):
        SpinTheWheel(
            [WheelSegment("RUIN", 1.0, LOSE_ALL)],
            rng=rng,
            allow_lose_all=False,
        )


# --- wheel: validation --------------------------------------------------------
def test_wheel_rejects_empty_segments(rng):
    with pytest.raises(GameError):
        SpinTheWheel([], rng=rng)


def test_wheel_rejects_bad_weight(rng):
    with pytest.raises(GameError):
        SpinTheWheel([WheelSegment("a", -1.0, 0)], rng=rng)
    with pytest.raises(GameError):
        SpinTheWheel([WheelSegment("a", float("inf"), 0)], rng=rng)


def test_wheel_rejects_all_zero_weight(rng):
    with pytest.raises(GameError):
        SpinTheWheel(
            [WheelSegment("a", 0.0, 0), WheelSegment("b", 0.0, 0)],
            rng=rng,
        )


def test_wheel_uses_default_client_seed_when_omitted(rng):
    wheel = SpinTheWheel(
        [WheelSegment("a", 1.0, 0), WheelSegment("b", 1.0, 0)],
        rng=rng,
    )
    res = wheel.spin(FIXED_SEED, nonce=3)
    assert res.provenance.client_seed == "ultron"


# --- slots --------------------------------------------------------------------
def test_slots_deterministic(rng):
    slots = Slots(["cherry", "lemon", "bar", "seven"], reels=3, rng=rng)
    r1 = slots.pull(FIXED_SEED, "client", 9)
    r2 = slots.pull(FIXED_SEED, "client", 9)
    assert r1.reels == r2.reels
    assert r1.indices == r2.indices
    assert r1.is_win == r2.is_win


def test_slots_reel_count_and_symbol_membership(rng):
    symbols = ["cherry", "lemon", "bar"]
    slots = Slots(symbols, reels=3, rng=rng)
    for nonce in range(100):
        res = slots.pull(FIXED_SEED, "client", nonce)
        assert len(res.reels) == 3
        assert len(res.indices) == 3
        for s in res.reels:
            assert s in symbols


def test_slots_win_detection_consistent(rng):
    symbols = ["a", "b"]
    slots = Slots(symbols, reels=3, rng=rng)
    saw_win = False
    saw_loss = False
    for nonce in range(200):
        res = slots.pull(FIXED_SEED, "client", nonce)
        all_same = len(set(res.reels)) == 1
        assert res.is_win == all_same
        if res.is_win:
            assert res.win_symbol == res.reels[0]
            saw_win = True
        else:
            assert res.win_symbol is None
            saw_loss = True
    # With 2 symbols / 3 reels both a win (p=1/4) and a loss must appear.
    assert saw_win and saw_loss


def test_slots_provenance_verifies(rng):
    slots = Slots(["a", "b", "c"], reels=3, rng=rng)
    res = slots.pull(FIXED_SEED, "client", 1)
    assert res.provenance.game == "slots"
    assert rng.verify(res.provenance.commit, res.provenance.server_seed) is True


def test_slots_rejects_bad_config(rng):
    with pytest.raises(GameError):
        Slots(["only-one"], rng=rng)  # needs >= 2 symbols
    with pytest.raises(GameError):
        Slots(["a", "b"], reels=0, rng=rng)
    with pytest.raises(GameError):
        Slots(["a", ""], rng=rng)  # empty symbol


def test_slots_default_client_seed(rng):
    slots = Slots(["a", "b"], rng=rng)
    res = slots.pull(FIXED_SEED, nonce=2)
    assert res.provenance.client_seed == "ultron"


# --- end-to-end provably-fair golden ------------------------------------------
def test_provably_fair_golden_round(rng):
    """A full round: commit before, reveal after, re-derive the same outcome."""
    rc = rng.new_round()
    wheel = SpinTheWheel(
        [
            WheelSegment("x", 1.0, 0),
            WheelSegment("y", 1.0, 100),
            WheelSegment("z", 1.0, 250),
        ],
        rng=rng,
    )
    # Server decides the result BEFORE any animation, using the secret seed.
    result = wheel.spin(rc.server_seed, "viewer-chosen-seed", 42)

    # Reveal: the commit published before the round verifies against the seed.
    assert rng.verify(rc.commit, rc.server_seed)
    # Anyone can re-derive the identical winning index from the revealed seed.
    redo = wheel.spin(rc.server_seed, "viewer-chosen-seed", 42)
    assert redo.index == result.index
    assert redo.target_angle == result.target_angle
