"""Pins the relay fact-guard's INVENTION check (2026-07-23).

Live battery (turn=35): the force-relay path fed a NON-tactical payload
("Izumi, I'm fine.") to the generic LLM relay prompt and the 4B produced
"Sage backsite, Sova heaven, Cypher CT." -- fully fabricated comms, spoken
on the team mic. ``_output_keeps_facts`` waived validation whenever the
PAYLOAD had no fact tokens ("not tactical -> not the validator's job"), so
invented agents/locations in the OUTPUT sailed through. Now a non-tactical
payload whose output carries agents/locations fails the check and the relay
abstains to the literal payload.
"""

from __future__ import annotations

import pytest

from kenning.audio.relay_speech import _output_keeps_facts


@pytest.mark.parametrize("payload,line", [
    # THE live hallucination: banter in, fabricated tactical comms out.
    ("Izumi, I'm fine.", "Sage backsite, Sova heaven, Cypher CT."),
    # Any conjured agent/location from a non-tactical payload is invention.
    ("good round", "Sova heaven, push now."),
    ("I'm fine.", "Jett is lurking A main."),
])
def test_nontactical_payload_with_invented_facts_fails(payload, line):
    assert not _output_keeps_facts(payload, line)


@pytest.mark.parametrize("payload,line", [
    # Persona-flavored non-tactical lines with NO fact tokens still pass.
    ("Izumi, I'm fine.", "Izumi is fine. Flesh endures."),
    ("nice try", "Nice try. Their extinction is merely delayed."),
    ("hello", "Hello."),
    # Tactical payloads keep their existing preservation semantics.
    ("rotate B", "Rotating B."),
    ("two on A", "Two on A. A flaw."),
])
def test_faithful_lines_still_pass(payload, line):
    assert _output_keeps_facts(payload, line)
