"""match_tell_chat — the voice→Twitch-chat tell grammar (spec 12, 2026-07-09).

Pins the two accepted forms (tagged "tell <name> in chat <msg>", broadcast
"tell chat <msg>"), the message cleaning, and — critically — the DISJOINTNESS
contract: no team-relay or teammate-social form may ever match (R7), because
this matcher runs BEFORE _maybe_handle_relay_speech in the dispatch cascades.
"""
from __future__ import annotations

import pytest

from kenning.audio.relay_speech import TellChatCommand, match_tell_chat


# ---------------------------------------------------------------- broadcast
@pytest.mark.parametrize(
    "text,msg",
    [
        ("tell chat brb", "brb"),
        ("tell the chat hello everyone", "hello everyone"),
        ("Ultron, tell chat I'll be back in five", "I'll be back in five"),
        ("say to chat gg", "gg"),
        ("say to the twitch chat good game", "good game"),
        ("post in chat the discord link is below", "the discord link is below"),
        ("put in the chat we go again", "we go again"),
        ("tell everyone in chat thanks for the raid", "thanks for the raid"),
        ("tell everybody in the chat one more game", "one more game"),
        ("tell them in chat gg", "gg"),
        # demonstrative "that" is KEPT (only reported-speech "that" drops)
        ("please tell chat that was my last round", "that was my last round"),
    ],
)
def test_broadcast_forms(text: str, msg: str) -> None:
    cmd = match_tell_chat(text)
    assert cmd == TellChatCommand(name=None, message=msg)


# ------------------------------------------------- wake homophones + leads
# Review 2026-07-09 P1: a mis-heard wake or a politeness scaffold must match
# HERE on the raw transcript — the normalizer strips/reframes these leads and
# the leftover "tell chat X" group form would be transmitted to the TEAM mic.
@pytest.mark.parametrize(
    "text,name,msg",
    [
        ("altron tell chat brb", None, "brb"),
        ("Voltron, tell chat one sec", None, "one sec"),
        ("ultra tell bob in chat hi", "bob", "hi"),
        ("ron, tell chat starting soon", None, "starting soon"),
        ("hey ultron tell chat we won", None, "we won"),
        ("could you tell chat brb", None, "brb"),
        ("can you please tell dragon slayer in chat nice one",
         "dragon slayer", "nice one"),
        ("i need you to tell chat gg", None, "gg"),
        ("go ahead and tell chat thanks all", None, "thanks all"),
        ("make sure you tell bob in chat i saw it", "bob", "i saw it"),
        ("ultron, would you tell chat five more minutes", None,
         "five more minutes"),
    ],
)
def test_wake_homophones_and_politeness_leads(text, name, msg) -> None:
    cmd = match_tell_chat(text)
    assert cmd == TellChatCommand(name=name, message=msg)


# ------------------------------------------------------------------- tagged
@pytest.mark.parametrize(
    "text,name,msg",
    [
        ("tell shroud in chat thanks for the sub", "shroud", "thanks for the sub"),
        ("Ultron tell dragon slayer in chat nice one", "dragon slayer", "nice one"),
        ("message bob in the chat that I saw it", "bob", "I saw it"),
        ("reply to jay dee in chat yes exactly", "jay dee", "yes exactly"),
        ("notify timmy on chat hurry up", "timmy", "hurry up"),
        ("write to mods in chat check the queue", "mods", "check the queue"),
        ("tell xx sniper xx in twitch chat you rock", "xx sniper xx", "you rock"),
        ("ultron, can you tell casey in chat welcome back", "casey", "welcome back"),
        ("inform ricky on the chat he is muted", "ricky", "he is muted"),
    ],
)
def test_tagged_forms(text: str, name: str, msg: str) -> None:
    cmd = match_tell_chat(text)
    assert cmd == TellChatCommand(name=name, message=msg)


def test_name_split_lands_on_first_in_chat() -> None:
    cmd = match_tell_chat("tell bob in chat see you in chat tomorrow")
    assert cmd == TellChatCommand(name="bob", message="see you in chat tomorrow")


# ------------------------------------------- delimiter STT-mishear tolerance
# Live 2026-07-10: Whisper rendered "tell 1v9khan IN CHAT hi" as
# "Tell 1v9con and chat hi." -> the strict "in chat" delimiter missed and the
# command fell through to the LLM (no chat post). The delimiter now absorbs
# the observed mishear family; the fuzzy roster match handles the name.
@pytest.mark.parametrize(
    "text,name,msg",
    [
        ("Tell 1v9con and chat hi.", "1v9con", "hi."),      # the EXACT live line
        ("tell bob an chat hello", "bob", "hello"),
        ("tell bob en chat one sec", "bob", "one sec"),
        ("tell bob into chat see you", "bob", "see you"),
        ("tell bob in chad hi", "bob", "hi"),               # "chat" mis-heard
        ("say hi to bob and chat", "bob", "hi"),
    ],
)
def test_delimiter_mishears_still_match(text, name, msg) -> None:
    assert match_tell_chat(text) == TellChatCommand(name=name, message=msg)


def test_delimiter_mishear_greet_verb_carries_flag() -> None:
    # greet-verb + mishear delimiter: same match, now with greet=True
    # (2026-07-23 -- the greet/welcome forms are also spoken aloud).
    assert match_tell_chat("greet bob and chat") == TellChatCommand(
        name="bob", message="hi", greet=True)


def test_delimiter_mishears_broadcast_and_disjointness() -> None:
    # broadcast group form tolerates the delimiter mishear too
    assert match_tell_chat("tell everyone and chat gg") == TellChatCommand(
        name=None, message="gg")
    # 2026-07-23 contract change: "tell <person> <msg>" with no chat word is
    # now the low-confidence BARE form (roster-gated in the handler), so
    # "tell chad hi" matches as a bare tell instead of falling through at
    # the matcher level -- the handler still routes it to the teammate-social
    # path whenever "chad" isn't a confident chat-roster hit.
    _chad = match_tell_chat("tell chad hi")
    assert _chad is not None and _chad.bare is True and _chad.name == "chad"
    # group names still reject through the widened delimiter
    assert match_tell_chat("tell my team and chat the plan") is None
    # multi-person casual tells match bare on the FIRST name (roster-gated).
    _bj = match_tell_chat("tell bob and jane the plan")
    assert _bj is not None and _bj.bare is True and _bj.name == "bob"


# ------------------------------------------------- greeting-before-name forms
# Review 2026-07-09: the natural inverse phrasing ("say hi to <name> in chat")
# puts the greeting BEFORE the name — the streamer's reported failing case.
@pytest.mark.parametrize(
    "text,name,msg,greet",
    [
        ("say hi to bob in chat", "bob", "hi", False),
        ("Ultron, say hi to dragon slayer in chat", "dragon slayer", "hi",
         False),
        ("say hello to timmy in chat", "timmy", "hello", False),
        ("say hey to jay dee in the chat", "jay dee", "hey", False),
        ("could you say what's up to ricky in chat", "ricky", "what's up",
         False),
        # "say WELCOME to ..." is a welcome ask -> greet=True (also spoken,
        # 2026-07-23 user direction).
        ("say welcome to newbie in chat", "newbie", "welcome", True),
        ("say hi to bob in chat and thanks for the follow", "bob",
         "hi and thanks for the follow", False),
        # greet / welcome verbs synthesize a greeting (always greet=True)
        ("greet casey in chat", "casey", "hi", True),
        ("Ultron greet dragon slayer in the chat", "dragon slayer", "hi",
         True),
        ("welcome timmy to chat", "timmy", "welcome", True),
        ("welcome ricky to the chat", "ricky", "welcome", True),
        ("welcome bob aboard in chat", "bob", "welcome", True),
    ],
)
def test_greeting_before_name_forms(text, name, msg, greet) -> None:
    assert match_tell_chat(text) == TellChatCommand(
        name=name, message=msg, greet=greet)


@pytest.mark.parametrize(
    "text,msg,greet",
    [
        ("say hi to everyone in chat", "hi", False),
        ("say hello to everybody in chat", "hello", False),
        ("greet everyone in chat", "hi", True),
        ("welcome all to the chat", "welcome", True),
    ],
)
def test_greeting_to_whole_audience_broadcasts(text, msg, greet) -> None:
    assert match_tell_chat(text) == TellChatCommand(
        name=None, message=msg, greet=greet)


def test_greeting_to_team_falls_through() -> None:
    # "say hi to my team in chat" is a team reference -> not a chat tag
    assert match_tell_chat("say hi to my team in chat") is None
    assert match_tell_chat("greet the squad in chat") is None


# --------------------------------------------------------- message cleaning
def test_leading_that_is_dropped_and_whitespace_collapsed() -> None:
    cmd = match_tell_chat("tell   bob   in chat   that   you   are   right")
    assert cmd == TellChatCommand(name="bob", message="you are right")


def test_demonstrative_that_is_kept() -> None:
    cmd = match_tell_chat("tell chat that was insane")
    assert cmd == TellChatCommand(name=None, message="that was insane")
    cmd = match_tell_chat("tell bob in chat that is the plan")
    assert cmd == TellChatCommand(name="bob", message="that is the plan")


def test_message_is_length_capped() -> None:
    long = "tell bob in chat " + "x" * 1000
    cmd = match_tell_chat(long)
    assert cmd is not None
    assert len(cmd.message) == 400


def test_control_characters_are_stripped() -> None:
    cmd = match_tell_chat("tell chat hi\x00\x07 there")
    assert cmd == TellChatCommand(name=None, message="hi there")


# ------------------------------------------------- disjointness (R7) + None
@pytest.mark.parametrize(
    "text",
    [
        # Team-relay leads must NEVER match (they belong to match_relay_command).
        "tell my team rotate B",
        "tell my team two garage",
        "tell the squad push A",
        "tell my teammates in chat the plan",       # group word in the name slot
        "tell the squad in chat hi",                # group word in the name slot
        "tell my team in chat the plan",            # "my ..." name reject
        "say to the guys we win this",
        "tell 'em to rotate",
        # Teammate-social relay forms must fall through — AGENT names in the
        # tell slot belong to the team relay, never the chat tell (the
        # 2026-07-23 bare form explicitly rejects them).
        "tell jett nice shot",
        "tell sage nice job",
        # Bare pronouns in the name slot fall through.
        "tell him in chat hello",
        "tell her in chat hello",
        # Function words in the bare name slot fall through.
        "tell the team to rotate",
        "tell someone to smoke mid",
        # Incomplete — no message (incl. the bare form's degenerate
        # channel-phrase leftover).
        "tell chat",
        "tell bob in chat",
        "tell chat   ",
        # Ordinary speech.
        "we should chat in a bit",
        "tell me about pandas",
        "what did chat say",
        "",
    ],
)
def test_falls_through(text: str) -> None:
    assert match_tell_chat(text) is None


# --------------------------------------------- verbless + STT verb mishears
# Live 2026-07-10 (second round): the wake strip swallowed the VERB entirely
# ("Saltwater bottle in chat, hello...") and Whisper heard "tell" as "I'll"
# with a sentence break after "chat". Both exact lines pinned.
def test_live_line_verbless_no_verb() -> None:
    cmd = match_tell_chat(
        "Saltwater bottle in chat, hello, welcome to the stream, sorry for the delay.")
    assert cmd == TellChatCommand(
        name="Saltwater bottle",
        message="hello, welcome to the stream, sorry for the delay.",
        verbless=True)


def test_live_line_ill_verb_and_sentence_break() -> None:
    cmd = match_tell_chat(
        "I'll saltwater bottle in the chat. Hi, welcome to the stream. "
        "Sorry for the delay.")
    assert cmd == TellChatCommand(
        name="saltwater bottle",
        message="Hi, welcome to the stream. Sorry for the delay.",
        verbless=False)                        # "I'll" counts as the verb


def test_verb_mishears_till_and_sentence_punct_broadcast() -> None:
    assert match_tell_chat("till bob in chat hi") == TellChatCommand(
        name="bob", message="hi")
    assert match_tell_chat("tell chat. hello there") == TellChatCommand(
        name=None, message="hello there")


def test_verbless_rejects_audience_group_and_pronoun_names() -> None:
    # commentary about the audience must NOT broadcast or tag
    assert match_tell_chat("everyone in chat is nice") is None
    assert match_tell_chat("chat in chat gg") is None
    assert match_tell_chat("my team in chat hello") is None
    assert match_tell_chat("them in chat gg") is None


def test_verbless_conversational_shape_matches_low_confidence() -> None:
    """'I posted in chat earlier' MATCHES the verbless form by design — the
    HANDLER's confidence gate (verbless + no roster match -> fall through to
    conversation) is what keeps it out of chat. Pinned here so the contract
    is explicit."""
    cmd = match_tell_chat("I posted in chat earlier")
    assert cmd is not None and cmd.verbless is True
    assert cmd.name == "I posted"


def test_none_input_is_safe() -> None:
    assert match_tell_chat(None) is None  # type: ignore[arg-type]


# --------------------------------------------- 2026-07-23: BARE tell form
# Live battery ground truth: "Ultron tell Izumi I am fine" was FORCE-relayed
# to the team as hallucinated comms and "Tell IceMapple he's wrong." was
# answered ABOUT the person on the desktop — both are chat tells with no
# "in chat" delimiter. The bare form matches them LOW-CONFIDENCE (the
# handler consumes only on a confident roster hit).
@pytest.mark.parametrize(
    "text,name,msg",
    [
        ("Ultron tell Izumi I am fine", "Izumi", "I am fine"),
        ("Tell IceMapple he's wrong.", "IceMapple", "he's wrong."),
        ("tell 1v9con good catch", "1v9con", "good catch"),
        ("Ultron, tell dragonslayer the raid is at nine", "dragonslayer",
         "the raid is at nine"),
    ],
)
def test_bare_tell_matches_low_confidence(text, name, msg) -> None:
    cmd = match_tell_chat(text)
    assert cmd == TellChatCommand(name=name, message=msg, bare=True)


def test_bare_tell_rejects_agents_and_team_words() -> None:
    # Valorant agents belong to the TEAM relay.
    assert match_tell_chat("tell Sage plant the spike") is None
    assert match_tell_chat("tell kay o to flash") is None
    # Team-possessives / group words never match bare.
    assert match_tell_chat("tell my team rotate B") is None
    assert match_tell_chat("tell everyone hi") is None


def test_greet_forms_carry_the_greet_flag() -> None:
    # The greet/welcome-verb form is ALSO spoken aloud by the handler
    # (user direction 2026-07-23) — the flag is its trigger.
    cmd = match_tell_chat("Welcome 1v9con to the chat.")
    assert cmd is not None and cmd.greet is True and cmd.name == "1v9con"
    cmd = match_tell_chat("greet casey in chat")
    assert cmd is not None and cmd.greet is True
    # The dictation forms never carry it.
    cmd = match_tell_chat("tell bob in chat hello")
    assert cmd is not None and cmd.greet is False


# ------------------------------------------- 2026-07-23: compose faithfulness
def test_compose_guard_requires_content_words() -> None:
    from kenning.audio.relay_speech import tell_chat_compose_ok

    # The dictated content must survive the in-character rewrite.
    assert tell_chat_compose_ok(
        "he's wrong", "You are wrong, IceMapple. The math does not lie.")
    assert tell_chat_compose_ok(
        "I don't have to go to bed",
        "The streamer's bed can wait. He answers to no clock.")
    # A composition that drops the content entirely fails -> literal fallback.
    assert not tell_chat_compose_ok(
        "I don't have to go to bed", "Flesh is weak. I endure.")
    assert not tell_chat_compose_ok("he's wrong", "")
    # No content words (bare greetings) -> any non-empty composition stands.
    assert tell_chat_compose_ok("hi", "The machine acknowledges you.")


def test_config_defaults_exist() -> None:
    """The spec-12 chat config fields ship with the intended defaults."""
    from kenning.config import TwitchChatConfig

    cfg = TwitchChatConfig()
    assert cfg.tell_chat_enabled is True
    assert cfg.tell_chat_match_floor == 60
    assert "{name}" in cfg.tell_chat_template
    assert "{message}" in cfg.tell_chat_template
    assert "{message}" in cfg.tell_chat_broadcast_template
    assert cfg.first_time_welcome_enabled is True
    assert "{name}" in cfg.first_time_welcome_text
    assert "{delay}" in cfg.first_time_welcome_text
    assert "{name}" in cfg.first_time_welcome_text_no_delay
    assert cfg.first_time_welcome_max_per_minute == 4
    assert cfg.stream_delay_seconds == 40


# ---------------------------------------------------------------------------
# LIVE BUG 2026-07-26: trailing sentence punctuation defeated the
# "say hi to <name> in chat" greeting form. Whisper puts a period on the end of
# essentially every transcript, so this form was effectively NEVER reached in
# production -- the turns fell through to the relay matcher and were parroted
# verbatim onto the TEAM MIC:
#   raw='Say hello to Izumi in the chat.' -> route='relay_llm' channel='team_mic'
# which is the "he isn't addressing the people I ask him to address" report.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,name,message", [
    ("Say hello to Izumi in the chat.", "Izumi", "hello"),
    ("Say hi to Izumi in the chat!", "Izumi", "hi"),
    ("say hi to izumi in chat.", "izumi", "hi"),
    ("Ultron, say hello to Izumi in the chat.", "Izumi", "hello"),
    ("say hey to Kappa123 in the chat?", "Kappa123", "hey"),
])
def test_greet_to_survives_trailing_punctuation(text, name, message):
    cmd = match_tell_chat(text)
    assert cmd is not None, f"trailing punctuation defeated the match: {text!r}"
    assert cmd.name == name
    assert cmd.message.startswith(message)


def test_greet_to_still_keeps_the_extra_clause():
    cmd = match_tell_chat("say hi to izumi in chat, welcome them")
    assert cmd is not None
    assert cmd.name == "izumi"
    assert "welcome them" in cmd.message


@pytest.mark.parametrize("text", [
    "say hi to my team",
    "tell my team to push A",
    "say hello to the enemy team in mid",
    "Say hi to Izumi.",          # no chat marker at all -- must NOT be a chat tell
])
def test_greet_to_does_not_over_capture(text):
    assert match_tell_chat(text) is None


# ---------------------------------------------------------------------------
# BARE greeting resolved against known chatters (2026-07-26). The other half of
# the live "say hi to izumi" failure: with no "in chat" marker the utterance
# says nothing about viewer-vs-teammate, so it fell to the relay matcher and
# was parroted onto the TEAM MIC. data/twitch/welcomed.db already knew Izumi.
# ---------------------------------------------------------------------------

_FAKE_CHATTERS = {"izumiikiryo", "icemapple14", "saltwaterbottle"}


def _resolver(name):
    from kenning.audio.chatter_names import resolve_chatter
    return resolve_chatter(name, chatters=_FAKE_CHATTERS)


@pytest.mark.parametrize("text,name,message", [
    ("Say hi to Izumi.", "Izumi", "hi"),
    ("say hello to IceMapple", "IceMapple", "hello"),
    ("Say hi to Izumi, welcome back", "Izumi", "hi"),
])
def test_bare_greeting_resolves_a_known_chatter(text, name, message):
    cmd = match_tell_chat(text, chatter_resolver=_resolver)
    assert cmd is not None, f"known chatter not resolved: {text!r}"
    assert cmd.name == name
    assert cmd.message.startswith(message)


@pytest.mark.parametrize("text", [
    "say hi to my team",       # the team, not a viewer
    "say hi to Sage",          # a Valorant agent -- must stay a team callout
    "say hi to Sova",
    "say hi to Kevin",         # not a known chatter
    "say hi to them",
])
def test_bare_greeting_refuses_everything_else(text):
    assert match_tell_chat(text, chatter_resolver=_resolver) is None


@pytest.mark.parametrize("text", [
    "Say hi to Izumi.", "say hello to IceMapple",
])
def test_bare_greeting_needs_the_resolver(text):
    """With no resolver the behaviour must be byte-identical to before."""
    assert match_tell_chat(text) is None
