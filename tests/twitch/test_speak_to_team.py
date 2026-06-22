"""S13 speak-to-team vetting — exact allowlist only, safety-screened, no free text."""
from __future__ import annotations

import re

from kenning.twitch.speak_to_team import SPEAK_TO_TEAM_ALLOWLIST, vet_redeem


def test_allowlisted_phrase_is_allowed_and_returns_canonical() -> None:
    v = vet_redeem("good luck have fun")
    assert v.allowed and v.phrase == "good luck have fun"


def test_obfuscated_allowlisted_phrase_matches() -> None:
    assert vet_redeem("G L H F").allowed          # spaces removed -> 'glhf'
    v = vet_redeem("G.G")                          # dots removed -> 'gg'
    assert v.allowed and v.phrase == "gg"          # returns the canonical allowlisted phrase, not the raw input
    assert vet_redeem("  Good  Luck  Have  Fun  ").allowed


def test_free_text_is_rejected() -> None:
    assert not vet_redeem("hey team you are all doing great keep it up").allowed
    assert not vet_redeem("good luck have fun and also rush b").allowed  # near-match w/ extra = reject


def test_tactical_content_is_rejected() -> None:
    for tactical in ["rush b now", "enemy on a main", "rotate to site", "they have no smokes", "push mid"]:
        assert not vet_redeem(tactical).allowed, tactical


def test_slur_input_is_blocked_by_safety() -> None:
    assert not vet_redeem("gg you faggot").allowed
    assert not vet_redeem("n1gg3r").allowed


def test_empty_and_garbage_fail_closed() -> None:
    assert not vet_redeem("").allowed
    assert not vet_redeem("   ").allowed
    assert not vet_redeem("\x00\x01").allowed


def test_allowlist_contains_no_tactical_words() -> None:
    """Meta-guard: the allowlist must stay LOW-TACTICAL (a valid-but-false tactical
    callout could throw a ranked round)."""
    tactical = re.compile(
        r"\b(rush|rotate|push|flank|enemy|enemies|smoke|smokes|flash|site|plant|defuse|"
        r"spike|mid|heaven|hell|peek|bombsite)\b", re.IGNORECASE)
    offenders = [p for p in SPEAK_TO_TEAM_ALLOWLIST if tactical.search(p)]
    assert not offenders, f"tactical phrases in the speak-to-team allowlist: {offenders}"
