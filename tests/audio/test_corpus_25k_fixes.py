"""Root-cause fixes from the 25,000-case corpus audit (2026-06-18).

Each fix targets a DETERMINISTIC-layer bug (normalization / matching) so the
relay routes correctly WITHOUT relying on the embedding relay-intent gate as a
safety net. Tests pin the fix + guard against regression.
"""
from __future__ import annotations

from kenning.audio.command_normalizer import (
    _NARRATION_MUSING_RE,
    _strip_scaffold,
    recover_relay_lead,
)
from kenning.audio.relay_speech import _payload_has_content


# ---------------------------------------------------------------------------
# F1: "let my team/squad/teammates know <imperative>" (no "that") must REFRAME
# to "tell my team <X>", not drop the lead. The bug: the wrapper remainder
# "drop spike on me" matched _HAS_RELAY_LEAD (on the ambiguous tactical verb
# "drop"), so the reframe used it as-is and the relay was MISSED.
# ---------------------------------------------------------------------------


def test_f1_wrapper_reframes_tactical_payload_to_team_relay():
    assert _strip_scaffold("let my team know drop spike on me") == \
        "tell my team drop spike on me"
    assert _strip_scaffold("let my squad know give me a rifle") == \
        "tell my team give me a rifle"
    assert _strip_scaffold("let the team know give up mid this round") == \
        "tell my team give up mid this round"
    assert _strip_scaffold("let my teammates know drop molly on the choke") == \
        "tell my team drop molly on the choke"
    assert _strip_scaffold("let the squad know share credits with the team") == \
        "tell my team share credits with the team"


def test_f1_group_addressed_remainder_stays_as_is():
    # "call out X" is a genuine relay verb -> used as-is (not double-prepended).
    assert _strip_scaffold("let the team know call out the flank") \
        .startswith("call out") or _strip_scaffold(
        "let the team know call out the flank").startswith("tell my team call out")
    # a remainder that already addresses a group keeps its lead (no double tell).
    out = _strip_scaffold("let my team know drop the whole team a smoke")
    assert out.count("tell my team") <= 1


def test_f1_does_not_touch_plain_relays():
    # A normal "tell my team X" is unaffected by the reframe path.
    assert _strip_scaffold("tell my team rotate to A") == "tell my team rotate to A"
    # No wrapper -> bare tactical imperative is left alone here (scaffold no-op).
    assert _strip_scaffold("drop spike on me") == "drop spike on me"


# ---------------------------------------------------------------------------
# F2: a trailing single-letter SITE callout (A/B/C) after a position cue is real
# content -- "they are A" was dropped because "a" is the junk article (B/C
# already passed since they aren't junk words).
# ---------------------------------------------------------------------------


def test_f2_site_letter_position_callouts_are_content():
    assert _payload_has_content("they are A")
    assert _payload_has_content("rotate to A")
    assert _payload_has_content("push to A")
    assert _payload_has_content("one A")
    assert _payload_has_content("they are B")   # already worked; stays valid
    assert _payload_has_content("they are C")


def test_f2_still_rejects_genuine_junk_fragments():
    # The all-junk gate must still drop clipped fragments.
    assert not _payload_has_content("that the")
    assert not _payload_has_content("of them")
    assert not _payload_has_content("about")
    assert not _payload_has_content("a")        # bare article, single word
    # an article "a" NOT trailing a position cue is not rescued
    assert not _payload_has_content("they are the")


# ---------------------------------------------------------------------------
# F5: musing / past-recount / general-statement framings that merely MENTION
# telling the team must NOT be canonicalized/recovered into a live relay. These
# are deterministic (the narration-musing gate short-circuits before the
# embedding relay-intent gate), so the system no longer relies on the embedder.
# ---------------------------------------------------------------------------

_F5_FALSE_RELAYS = [
    "I told my team to slow push and they just ran in and got wiped",   # recount
    "I told my squad to play passive and everyone peeked aggressive",   # recount
    "part of me wants to ask my team to stack A but I think B",         # musing
    "one side of me says tell my team to play passive",                 # musing
    "one of my biggest weaknesses is not telling my team to plant",     # general
    "one of these days my team will tell itself to eco",               # general
    "the meta right now is to tell your team to take mid",              # general
    "great controllers tell their team where they're smoking",         # general
    "there's no one to tell my team to anchor B",                      # general
    "chat: should I ask my team to stack A or spread it out",          # chat addr
    "processing out loud here: do I ask my team to save",              # think-aloud
]

_F5_REAL_RELAYS = [
    "tell my team rotate to B",
    "told my team rotate to A",            # bare STT-mishear of "tell" -> relay
    "tell the squad to save",
    "tell my team I told them to push and they did",  # "I told" in the PAYLOAD
]


def test_f5_musing_and_recounts_do_not_recover_a_relay_lead():
    for t in _F5_FALSE_RELAYS:
        assert recover_relay_lead(t) == t, (
            f"musing/recount wrongly recovered to a relay: {t!r} -> "
            f"{recover_relay_lead(t)!r}"
        )


def test_f5_real_relays_still_recover_or_keep_their_lead():
    for t in _F5_REAL_RELAYS:
        out = recover_relay_lead(t)
        assert out.lower().lstrip().startswith(("tell", "told")), (
            f"legit relay lost its lead: {t!r} -> {out!r}")


def test_f5_musing_gate_spares_real_self_status():
    # First-person self-status callouts must NEVER be gated as musing.
    for t in ("I'm planting", "I died", "I'm low", "I need a drop",
              "I got one", "I have spike"):
        assert not _NARRATION_MUSING_RE.match(t), t


# ---------------------------------------------------------------------------
# F3: a verbatim/context relay whose team noun is a reported SUBJECT
# ("my teammate is flaming me, tell them to calm down ...") lost its "tell them"
# directive because _strip_scaffold treated "my teammate is ..." as an outer
# relay frame and stripped the nested verb. The context clause must not enable
# that strip, so the real directive survives.
# ---------------------------------------------------------------------------


def test_f3_context_subject_clause_keeps_the_relay_directive():
    # The "tell them ..." directive must survive (not be stripped).
    out = _strip_scaffold("my teammate is flaming me, tell them to calm down")
    assert "tell them to calm down" in out or "calm down" in out
    assert "tell them" in out  # nested directive preserved, not deleted
    out2 = _strip_scaffold("the squad keeps dying, tell them to slow down")
    assert "tell them" in out2


def test_f3_genuine_doubled_team_lead_still_strips():
    # "my team, tell them rotate" (team as ADDRESS, not subject) should still
    # collapse the doubled relay verb (no copula after the team noun).
    out = _strip_scaffold("my team tell them rotate to B")
    assert out.lower().startswith("tell my team") or "rotate to B" in out


# ---------------------------------------------------------------------------
# F4: "ask <agent> about <topic>" is a topic-question relay to that agent. It
# was MISSED because the named-ask payload prefix only accepted bare question
# words (if/whether/why/how/...), not "about".
# ---------------------------------------------------------------------------


def test_f4_ask_agent_about_topic_relays():
    from kenning.audio.command_normalizer import normalize_command
    from kenning.audio.relay_speech import match_relay_command
    for t, agent in [
        ("ask my cypher about his cage near window", "Cypher"),
        ("ask Sage about the retake plan", "Sage"),
        ("ask my sova about the dart timing", "Sova"),
    ]:
        cmd = match_relay_command(normalize_command(t))
        assert cmd is not None, t
        assert cmd.addressee == agent, (t, cmd.addressee)
        assert cmd.payload.startswith("about"), (t, cmd.payload)
    # existing question-word asks must STILL work.
    cmd = match_relay_command(normalize_command("ask Sage if she has her ult"))
    assert cmd is not None and cmd.addressee == "Sage"
