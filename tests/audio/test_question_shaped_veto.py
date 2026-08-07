"""Pins ``question_shaped`` -- the force-relay question veto (2026-07-23).

Live battery failure: the semantic router scored clean questions as
team_callout and ``force=True`` bypassed the strict matcher's question
rejection, speaking the question VERBATIM on the team mic:

    turn=10  "do you think they're gonna go A?"  -> relayed verbatim
    turn=28  "do you think I die here?"          -> relayed verbatim
    turn=29  "do you think they rotated"         -> relayed as "They rotated."

The veto runs at BOTH force sites (router team_callout branch + turbo
RELAY_TO_TEAM backstop). Explicit relay phrasings never reach those sites
(the scaffold strip + strict matcher consume them first), so a leading
interrogative or trailing "?" is safe to treat as ask-Ultron.
"""

from __future__ import annotations

import pytest

from kenning.audio.command_normalizer import question_shaped


# The battery's real misrouted questions (corrected ground truth) + shape kin.
QUESTIONS = [
    "do you think they're gonna go A?",
    "do you think I die here?",
    "do you think they rotated",          # no "?" -- opener alone must catch it
    "do you think ill win this game",
    "do you think I'll have a good team this game",
    "should I push mid?",
    "am I going to win this round?",
    "am I washed up?",
    "am I cooked?",
    "what site are they gonna go to?",
    "where are they going",
    "why do I keep getting the same map?",
    "will they rotate",
    "can I clutch this",
    "they're going A?",                   # trailing "?" alone must catch it
]

# Real callouts / statements that MUST stay relayable.
CALLOUTS = [
    "rotate B",
    "two on A",
    "Jett A main",
    "tell my team rotate B",
    "they're pushing mid",
    "Sage backsite",
    "I push mid",
    "watch the flank",
    "he's one shot",
    "planting A",
    "going B, smoke mid",
]


@pytest.mark.parametrize("text", QUESTIONS)
def test_questions_are_vetoed(text: str) -> None:
    assert question_shaped(text), text


@pytest.mark.parametrize("text", CALLOUTS)
def test_callouts_are_not_vetoed(text: str) -> None:
    assert not question_shaped(text), text


def test_degenerate_inputs_fail_open() -> None:
    assert not question_shaped("")
    assert not question_shaped("   ")
    assert not question_shaped(None)  # type: ignore[arg-type]
