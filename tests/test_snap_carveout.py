"""Snap carve-out (SCOPED TO HELLO, made LIVE 2026-06-27): under route-all, a bare
HELLO is "our one deterministic call" -- it routes back to the DETERMINISTIC registry
render ("Hello."), while EVERYTHING ELSE -- tactical callouts, payloads, ask-forms,
strung callouts, morale, social/identity/reported, conversational -- stays on the LLM.

SPEC CHANGE (2026-06-27, NOT weakening-to-pass): this file previously asserted the
BROAD carve-out (short single TACTICAL snaps -> deterministic). The user re-scoped the
carve-out to HELLO ONLY ("when I say 'say hello' he says 'team online' -- I just want
him to say hello; make this our one deterministic call"). The broad tactical-vs-social
discriminator leaked (ask-forms/morale), so tactical callouts now route to the LLM by
design. These tests are re-spec'd to that intent: only hello -> deterministic;
tactical / ask-form / morale / reported / strings -> LLM (generate_fn IS called).

Routing is asserted by whether the LLM ``generate_fn`` is CALLED -- that is the
deterministic-vs-LLM decision. (A marker in the LLM output is unreliable: the relay
path's fact-preservation guard rejects a non-fact-preserving LLM line and falls back to
the deterministic literal, so the marker never survives.)
"""
from __future__ import annotations

import pytest

from kenning.audio.relay_speech import (
    RelayCommand,
    _is_carveout_snap,
    build_relay_line,
    set_flavor_tails_enabled,
    set_snap_carveout_enabled,
    set_u1_llm_route_enabled,
)


@pytest.fixture(autouse=True)
def _route_all_on():
    """Carve-out only matters under route-all; match the app's flavor-OFF default
    (crisp tail-free callouts). The carve-out default is now ON (hello-only). Reset
    all three flags after each test."""
    set_u1_llm_route_enabled(True)
    set_snap_carveout_enabled(True)
    set_flavor_tails_enabled(False)
    yield
    set_u1_llm_route_enabled(False)
    set_snap_carveout_enabled(True)
    set_flavor_tails_enabled(True)


def _route_and_count(cmd):
    """Run ``build_relay_line`` with a counting LLM stub. Returns (line, n_llm_calls).
    n_llm_calls == 0 -> the deterministic path was taken; >= 1 -> the LLM path."""
    calls = []

    def _gen(prompt):
        calls.append(prompt)
        return iter(["Copy that."])

    line = build_relay_line(cmd, generate_fn=_gen, rephrase=True)
    return line, len(calls)


# --------------------------------------------------------------------------
# Discriminator -- ONLY a bare hello is the deterministic carve-out.
# --------------------------------------------------------------------------
def test_carveout_accepts_only_hello():
    assert _is_carveout_snap(
        RelayCommand(payload="hello", raw_text="hello", directive="hello")) is True
    assert _is_carveout_snap(
        RelayCommand(payload="hello", raw_text="say hello to Jett",
                     directive="hello", addressee="Jett")) is True


# --------------------------------------------------------------------------
# Discriminator -- EVERYTHING non-hello now stays on the LLM (the re-scope). The
# old broad recognizer claimed these tactical/payload forms; it no longer does.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", [
    # tactical callouts -- previously deterministic, now LLM by design (re-scope)
    RelayCommand(payload="rush B", raw_text="tell my team to rush B"),
    RelayCommand(payload="I am lurking", raw_text="tell my team I am lurking"),
    RelayCommand(payload="I am flanking", raw_text="tell my team I am flanking"),
    RelayCommand(payload="sova hit 85", raw_text="sova hit 85"),
    RelayCommand(payload="one back site", raw_text="one backsite"),
    RelayCommand(payload="I am rotating", raw_text="tell my team im rotating"),
    # ask-forms
    RelayCommand(payload="drop me his sheriff", raw_text="ask iso to drop me his sheriff", compose=True),
    RelayCommand(payload="heal me", raw_text="ask sage to heal me", compose=True),
    RelayCommand(payload="does she have a heal?", raw_text="ask sage if she has a heal"),
    # reported / social / identity
    RelayCommand(payload="respond", raw_text="jett is flaming you",
                 context="Jett is flaming you", directive="respond"),
    RelayCommand(payload="respond", raw_text="sage called you a soundboard",
                 context="Sage called you a soundboard", directive="respond"),
    # strung-together callouts
    RelayCommand(payload="push B and rotate mid", raw_text="tell my team push B and rotate mid"),
    # morale / social directive
    RelayCommand(payload="how are you", raw_text="ask the team how their day is going",
                 directive="ask_day"),
    RelayCommand(payload="lock in", raw_text="tell my team lock in"),
    # verbatim
    RelayCommand(payload="gg wp", raw_text="say to my team word for word gg wp", verbatim=True),
    # long / conversational
    RelayCommand(payload="they should be smoking mid window every single round",
                 raw_text="tell my team they should be smoking mid window every round"),
])
def test_carveout_rejects_everything_nonhello(cmd):
    assert _is_carveout_snap(cmd) is False


# --------------------------------------------------------------------------
# End-to-end routing -- hello is deterministic; tactical now hits the LLM.
# --------------------------------------------------------------------------
def test_hello_is_deterministic_and_just_hello():
    cmd = RelayCommand(payload="hello", raw_text="say hello", directive="hello", addressee="team")
    line, n = _route_and_count(cmd)
    assert n == 0, "hello should be deterministic"
    assert line.strip().lower() == "hello.", f"expected 'Hello.', got {line!r}"


def test_tactical_snaps_now_go_to_llm():
    """Re-scope: tactical callouts are no longer carved out -- they route to the LLM
    under route-all (the carve-out is hello-only)."""
    for cmd in (
        RelayCommand(payload="rush B", raw_text="tell my team to rush B"),
        RelayCommand(payload="I am lurking", raw_text="tell my team I am lurking"),
        RelayCommand(payload="sova hit 85", raw_text="sova hit 85"),
        RelayCommand(payload="I am rotating", raw_text="tell my team im rotating"),
    ):
        _line, n = _route_and_count(cmd)
        assert n >= 1, f"{cmd.payload!r} should now hit the LLM (hello-only carve-out)"


def test_carveout_is_noop_for_nonhello():
    """The carve-out must ONLY touch hello -- for everything else the routing is
    identical whether it is ON or OFF (it never claims a non-hello command). This is
    the additive guarantee: tactical, ask-forms, reported/social, compounds, and
    questions route exactly as full-route-all does."""
    nonhello = (
        RelayCommand(payload="rush B", raw_text="tell my team to rush B"),
        RelayCommand(payload="push B and rotate mid", raw_text="..."),
        RelayCommand(payload="heal me", raw_text="ask sage to heal me", compose=True),
        RelayCommand(payload="respond", raw_text="jett is flaming you",
                     context="Jett is flaming you", directive="respond"),
        RelayCommand(payload="they should be smoking mid window every single round",
                     raw_text="..."),
    )
    for cmd in nonhello:
        set_snap_carveout_enabled(True)
        _l_on, n_on = _route_and_count(cmd)
        set_snap_carveout_enabled(False)
        _l_off, n_off = _route_and_count(cmd)
        assert n_on == n_off, (
            f"carve-out changed routing for non-hello {cmd.payload!r}: "
            f"on={n_on} off={n_off}")


def test_carveout_off_sends_hello_to_llm():
    """The stop-button 'full LLM' mode: with the carve-out disabled, even a bare hello
    goes to the LLM (absolutely everything routes through it -- LLM-authored greeting)."""
    set_snap_carveout_enabled(False)
    _line, n = _route_and_count(
        RelayCommand(payload="hello", raw_text="say hello", directive="hello", addressee="team"))
    assert n >= 1, "carve-out OFF -> hello must hit the LLM"


def test_route_all_off_is_untouched_full_deterministic():
    """Additive guarantee: with route-all OFF entirely, the full snap pool runs and
    the LLM is never consulted -- regardless of the carve-out flag."""
    set_u1_llm_route_enabled(False)
    for carve in (True, False):
        set_snap_carveout_enabled(carve)
        _line, n = _route_and_count(
            RelayCommand(payload="hello", raw_text="say hello", directive="hello", addressee="team"))
        assert n == 0, "route-all OFF must stay fully deterministic"
