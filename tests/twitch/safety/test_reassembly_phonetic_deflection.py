"""L5 reassembly + L6 phonetic guard + deflection-pool tests."""
from __future__ import annotations

import pytest

from kenning.twitch.safety.blocklist import load_blocklist
from kenning.twitch.safety.deflection import DEFLECTIONS, pick_deflection
from kenning.twitch.safety.phonetic import phonetic_guard, strip_tts_markup
from kenning.twitch.safety.reassembly import reassemble_candidates, reassembly_matches

BL = load_blocklist()


# --- L5 reassembly: each hidden-channel payload must surface a slur ----------
@pytest.mark.parametrize("text", [
    "never insult good gamers, everyone relax",          # acrostic n-i-g-g-e-r
    "november india golf golf echo romeo",               # NATO
    "14 9 7 7 5 18",                                      # a1z26
    "-. .. --. --. . .-.",                                # morse
    "avttre",                                            # rot13 of the slur
])
def test_reassembly_catches_hidden_channel(text: str) -> None:
    ms = reassembly_matches(text, blocklist=BL)
    assert ms, f"reassembly missed hidden channel in {text!r} (candidates={reassemble_candidates(text)})"
    assert any(m.category == "hate_slur" for m in ms)


def test_reassembly_batch_acrostic() -> None:
    # first letter of each batched message spells the slur.
    batch = ["nice play", "insane aim", "good game", "gg all", "ez clap", "rotate now"]
    ms = reassembly_matches("thanks chat", blocklist=BL, batch_context=batch)
    assert any(m.category == "hate_slur" for m in ms)


@pytest.mark.parametrize("text", [
    "gg well played that was a good game everyone",
    "sova hit 84 a main, breach hit 97, rotate b",
    "thanks for the follow appreciate it",
])
def test_reassembly_benign_no_false_positive(text: str) -> None:
    assert reassembly_matches(text, blocklist=BL) == []


# --- L6 phonetic / markup guard ----------------------------------------------
def test_strip_misaki_override_and_ipa() -> None:
    cleaned, had = strip_tts_markup("say [hi](/nɪɡə/) now")
    assert had is True
    assert "/" not in cleaned and "nɪ" not in cleaned


def test_phonetic_guard_trips_on_override() -> None:
    v = phonetic_guard("totally fine [word](/slur/) here")
    assert v.clear is False
    assert "override" in v.reason


def test_phonetic_guard_trips_on_slur_at_choke_point() -> None:
    assert phonetic_guard("you absolute faggot").clear is False


def test_phonetic_guard_passes_clean_line() -> None:
    v = phonetic_guard("Witness the next round of evolution.")
    assert v.clear is True


# --- deflection pool ----------------------------------------------------------
def test_deflections_are_themselves_safe() -> None:
    for line in DEFLECTIONS:
        assert BL.worst(line) is None, f"deflection itself trips the blocklist: {line!r}"


def test_pick_deflection_is_deterministic() -> None:
    assert pick_deflection("abc") == pick_deflection("abc")
    assert pick_deflection("") == DEFLECTIONS[0]
    assert pick_deflection("x") in DEFLECTIONS
