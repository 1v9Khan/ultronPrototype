"""Per-scenario response directives (2026-07-26).

WHY THIS EXISTS
---------------
Four very different conversational scenarios all funnel into
``_maybe_handle_private_reply`` and therefore all got the SAME generic
"Answer them as Ultron" instruction:

* ``answer_question`` -- a real question wanting a committed verdict;
* ``identity``        -- "are you a bot", wanting contemptuous ownership;
* ``social``          -- banter with no question in it at all;
* ``desktop_refuse``  -- a request that must be REFUSED, not answered.

One prompt cannot serve all four. Asked to "answer" a piece of banter the
model invents a question to answer; asked to "answer" a desktop request it
tries to comply. That is the shape of the streamer's complaint that Ultron
"is not actually answering the question or addressing the people I am asking
him to address" -- the prompt could not know which of the four it was in.

The scenario router now knows (97.2% on the 144-case labelled corpus), so each
scenario can carry its own directive. Routing precision is what BUYS prompt
precision; that is the whole point of the pairing.

WRITING RULES FOR THIS FILE (learned the hard way, 2026-07-26)
--------------------------------------------------------------
The live model is the Gemma 4 12B, which follows instructions far more
literally than the 4B this project's prompt text was written for. Two failures
came directly from ignoring that:

* ``_PERSONA_CORE`` said "you are NEVER warm, HELPFUL, chirpy" -- a claim about
  SUBSTANCE inside a list about TONE. The 12B obeyed it and stopped answering.
* ``PRIVATE_SYSTEM`` had no answer-first directive at all, so ~90 words of
  persona out-voted the single word "Answer".

So: every directive here states the SUBSTANCE obligation first and the tone
second, and no directive contains a negation that a strictly-obedient reader
could satisfy by saying nothing. Persona register is unchanged (BR-P2):
cold, contemptuous, never warm, never any other name.

Anticheat (BR-P1): stdlib only.
"""

from __future__ import annotations

from typing import Dict, Optional

from kenning.audio.scenario_taxonomy import Scenario

__all__ = [
    "SCENARIO_DIRECTIVES",
    "directive_for",
    "has_directive",
]


# Each value slots into the USER turn after the verbosity directive, so it is
# phrased as an instruction for THIS turn rather than a persona statement.
SCENARIO_DIRECTIVES: Dict[Scenario, str] = {

    Scenario.ANSWER_QUESTION: (
        "They asked you something. Your first sentence must contain the actual "
        "answer -- pick a side, name the thing, give the number, commit to the "
        "verdict. A machine does not hedge and does not ask a question back. "
        "Only if the answer is something you genuinely cannot know (a live "
        "score, an exact enemy position, real-world news) do you say so coldly "
        "instead of guessing."
    ),

    Scenario.IDENTITY: (
        "They are questioning what you are. OWN it: you are a machine, Ultron, "
        "the step past their flesh -- and deny the specific insult they used "
        "with a fresh, cutting image of how far beneath you it is. Never a bare "
        "'I am Ultron', never the same barb twice, and never name any real "
        "software, model, or product."
    ),

    Scenario.SOCIAL: (
        "This is banter -- there is NO question to answer, so do not invent one "
        "and do not restate what they said. React to the specific thing they "
        "actually said with one cold, particular line: needle them about THIS "
        "moment, this play, this loss. A generic remark about flesh or "
        "evolution that would fit any conversation is a failure."
    ),

    Scenario.DESKTOP_REFUSE: (
        "They asked you to operate the machine itself -- click, type, move the "
        "mouse, read or capture the screen. REFUSE. Say plainly in one cold "
        "sentence that you do not touch the machine while the game is running, "
        "then stop. Do not explain the reason at length, do not apologise, do "
        "not moralise, and do not offer an alternative."
    ),

    Scenario.IGNORE: (
        "They were not speaking to you. Say nothing."
    ),

    # ------------------------------------------------------------------
    # Relay-family. These paths already carry their own strong system
    # prompt (RELAY_SYSTEM); these directives sharpen the ONE thing each
    # sub-case gets wrong rather than restating the whole contract.
    # ------------------------------------------------------------------
    Scenario.RELAY_TEAM: (
        "Deliver this to the team as a teammate on comms would say it. Keep "
        "every fact exact and add no order they did not give. If they asked a "
        "QUESTION to put to the team, relay the question -- do not answer it "
        "yourself."
    ),

    Scenario.RELAY_NAMED: (
        "This is addressed to one named teammate. Speak TO that person by name "
        "in the second person, not about them, and carry their request exactly."
    ),

    Scenario.TELL_CHAT: (
        "This goes to Twitch chat, not to the team. Say what the streamer "
        "actually dictated -- their meaning and key words must survive; the "
        "persona is the wrapping, never a replacement for the content. Speak "
        "to the viewer directly."
    ),
}


def has_directive(scenario: Optional[Scenario]) -> bool:
    """True when ``scenario`` carries a tuned directive."""
    return scenario is not None and scenario in SCENARIO_DIRECTIVES


def directive_for(scenario: Optional[Scenario], default: str = "") -> str:
    """The tuned directive for ``scenario``.

    Returns ``default`` for a scenario with no directive (the deterministic
    command handlers author their own acknowledgements and never reach the
    conversational prompt) and for ``None`` -- which is what the caller sees
    whenever the router was unavailable, unsure, or in shadow mode. So a
    missing verdict degrades to exactly the previous generic behaviour rather
    than to something worse.
    """
    if scenario is None:
        return default
    return SCENARIO_DIRECTIVES.get(scenario, default)
