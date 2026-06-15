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
])
def test_corrects_mishears(garbled, expected):
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
