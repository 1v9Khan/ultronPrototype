"""Valorant STT correction: fixes mis-heard agent names + terms, leaves clean
callout text untouched."""
from __future__ import annotations

import pytest

from kenning.audio._stt_correct import correct_callout_stt as fix


@pytest.mark.parametrize("garbled,expected", [
    ("tell our team Silva has sold.", "tell our team Sova has ult."),
    ("tell my team our jet has", "tell my team our Jett has"),
    ("we have killed Royal.", "we have killed Reyna."),
    ("their neon has old", "their Neon has ult"),
    ("enemy Cipher main", "enemy Cypher main"),
    ("Race has ult", "Raze has ult"),
    ("Felix on A", "Phoenix on A"),
    ("Royal pushing B", "Reyna pushing B"),
    ("their kayo knife", "their KAY/O knife"),
    ("vise turret down", "Vyse turret down"),
    ("brim ulting", "Brimstone ulting"),
])
def test_corrects_mishears(garbled, expected):
    assert fix(garbled) == expected


@pytest.mark.parametrize("garbled,expected", [
    # "ult" recovered ONLY when the grammar makes it a callout
    ("their neon has old", "their Neon has ult"),
    ("Sova has sold", "Sova has ult"),
    ("she got alt", "she got ult"),
    ("they popped vault", "they popped ult"),
    ("old is up", "ult is up"),
    ("Killjoy has her old", "Killjoy has her ult"),
])
def test_ult_recovered_in_context(garbled, expected):
    assert fix(garbled) == expected


@pytest.mark.parametrize("literal", [
    "fall back to old position",
    "the old smoke",
    "hold the old angle",
])
def test_literal_old_is_not_turned_into_ult(literal):
    assert fix(literal) == literal


@pytest.mark.parametrize("garbled,expected", [
    ("enemy Royal on site a", "enemy Reyna on A site"),
    ("Cipher on amen", "Cypher on A main"),
    ("diffuse the spike", "defuse the spike"),
    ("molotov on B", "molly on B"),
])
def test_terms_and_sites(garbled, expected):
    assert fix(garbled) == expected


@pytest.mark.parametrize("clean", [
    "tell my team Sova has ult",
    "tell my team rotate to A",
    "tell my team I have B site",
    "tell my team Jett ulted",
    "tell my team Reyna is pushing",
    "play some daft punk",          # not a callout -> unchanged
    "set the volume to forty",
])
def test_leaves_clean_text_unchanged(clean):
    assert fix(clean) == clean


def test_empty_and_idempotent():
    assert fix("") == ""
    once = fix("tell our team Silva has sold")
    assert fix(once) == once        # correcting corrected text is a no-op
