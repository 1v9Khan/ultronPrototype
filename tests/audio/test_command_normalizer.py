"""Pins the pre-routing command normalizer: callouts route, vocab is corrected,
and conversational / Spotify text is NEVER over-corrected ("zero mistakes")."""

import pytest

from kenning.audio.command_normalizer import normalize_command
from kenning.audio.relay_speech import match_relay_command, _GREET_RE


def _routes_relay(text: str) -> bool:
    return match_relay_command(normalize_command(text)) is not None


def _is_greet(text: str) -> bool:
    return bool(_GREET_RE.match(normalize_command(text)))


# --- Callouts that MUST reach the relay (clipped leads + blends + vocab) -----
CALLOUTS = [
    "my team there's a Jett A main",
    "there's an enemy on jet main",
    "a jet on a main",
    "My team, there's a Jet A main",
    "It's a chamber holding long",
    "my team two enemies be main",
    "my team their neon has ult",
    "I hope my team Silva has his ult",
    "My team Jet ulted",
    "Call out a ray zombie",
    "my team I'm flanking through mid",
    "my team I'm planting",
    "my team good game",
    "my team to watch the flank",
    "my team were going to win",
    "tell my team Omen is lurking",
    "Tell my team their soba has old",       # phonetic Sova + ult
    "their cipher is in heaven",
    "warn my team killjoy turret on B",
]


@pytest.mark.parametrize("text", CALLOUTS)
def test_callouts_route_to_relay(text):
    assert _routes_relay(text), f"should relay: {text!r} -> {normalize_command(text)!r}"


# --- Conversational / Spotify / identity: must NOT be grabbed by relay -------
# NB: "<teammate> asked about <topic>" is intentionally NOT here -- a reported
# question now routes to Ultron's in-character ANSWER path (see
# test_reported_topic_question_routes_to_answer). "Tell me about X" (the USER
# asking Ultron directly) stays conversational.
NOT_RELAY = [
    "Tell me about Tony Stark",
    "what do you think of the enemy team",
    "are we going to win",
    "who are you",
    "what time is it",
    "play some Daft Punk",
    "pause the music",
    "turn it up",
    "skip this song",
    "what song is this",
    "set the volume to 40",
    "explain the spike timer",
    "thank you",
]


@pytest.mark.parametrize("text", NOT_RELAY)
def test_non_callouts_not_relayed(text):
    # Greetings are allowed to match the greet path; the rest must NOT relay.
    if _is_greet(text):
        return
    assert not _routes_relay(text), (
        f"should NOT relay: {text!r} -> {normalize_command(text)!r}")


# --- Reported questions route to the in-character ANSWER path (Marvel/topic/
# identity asked by a teammate) -- NOT a literal callout of the question --------
def test_reported_topic_question_routes_to_answer():
    from kenning.audio.relay_speech import match_relay_command
    for utt in [
        "Jett asked about Tony Stark",
        "my teammate asked about Black Widow",
        "my teammate is wondering about Iron Man",
        "Reyna asked how far the moon is",
        "team asked if you are a streamer",
    ]:
        cmd = match_relay_command(normalize_command(utt))
        assert cmd is not None, utt
        assert cmd.compose and cmd.context, (utt, cmd)   # answer, not a callout
        assert not cmd.verbatim, utt
    # Marvel names survive normalization (no "Iron Man" -> "Iron main").
    assert "Iron Man" in normalize_command("my teammate is wondering about Iron Man")


# --- Vocab correction: the canonical term appears in the normalized output ---
VOCAB = [
    ("tell my team silva has ult", "Sova"),
    ("tell my team jet is pushing", "Jett"),
    ("tell my team cipher in heaven", "Cypher"),
    ("tell my team race ulted", "Raze"),
    ("tell my team their royal is low", "Reyna"),
    ("tell my team Arsova has ult", "our Sova"),
    ("call out a ray zombie", "Raze on B"),
    ("tell my team brimstoan smoked A", "Brimstone"),   # phonetic/fuzzy
    ("tell my team vipor wall is up", "Viper"),          # phonetic/fuzzy
    ("tell my team two enemies be main", "B main"),
]


@pytest.mark.parametrize("text,expected", VOCAB)
def test_vocab_corrected(text, expected):
    out = normalize_command(text)
    assert expected in out, f"{text!r} -> {out!r} (missing {expected!r})"


# --- ZERO MISTAKES: conversational text must be returned VERBATIM ------------
NO_OVERCORRECT = [
    "Tell me about Tony Stark",
    "what do you think of the enemy team",
    "play some Daft Punk",
    "pause the music",
    "are we going to win",
    "explain the spike timer",
]


@pytest.mark.parametrize("text", NO_OVERCORRECT)
def test_no_overcorrection_on_conversational(text):
    # Stripping leading filler is allowed, but no Valorant agent/term should be
    # injected and no "tell my team" lead added.
    out = normalize_command(text)
    assert not out.lower().startswith("tell my team"), f"{text!r} -> {out!r}"
    assert out == text, f"conversational altered: {text!r} -> {out!r}"


def test_empty_and_noise():
    assert normalize_command("") == ""
    assert normalize_command("   ") == "   "
    # single-word noise should not become a relay
    assert not _routes_relay("me")


# --- Bare greetings: NEVER snapped to a location or relayed (live bug: "Hello."
# -> "tell my team hell." -> broadcast "No hell." to the team) ----------------
BARE_GREETINGS = [
    "Hello.", "hello", "hi", "hey", "hey there", "yo", "yo Ultron",
    "hiya", "howdy", "sup", "what's up", "good morning",
]


@pytest.mark.parametrize("text", BARE_GREETINGS)
def test_bare_greeting_not_mangled_or_relayed(text):
    out = normalize_command(text)
    # preserved verbatim: never rewritten into a relay, never corrupted into the
    # location "hell" (the live "Hello." -> "tell my team hell." -> "No hell." bug)
    assert out == text.strip(), f"{text!r} altered -> {out!r}"
    assert not out.lower().startswith("tell my team"), f"{text!r} -> {out!r}"
    assert not _routes_relay(text), f"{text!r} -> {out!r}"


# --- "I want my team to X" -> "tell my team X" (no doubled lead, payload kept) -
def test_want_my_team_extracts_directive():
    out = normalize_command("I want my team to rotate to B")
    assert _routes_relay("I want my team to rotate to B")
    assert out.lower().count("my team") == 1, f"doubled addressee: {out!r}"
    assert "to b" in out.lower(), f"site lost: {out!r}"


def test_want_my_team_variants_relay():
    for t in [
        "I want my team to fall back",
        "I need my team to save",
        "I wanna tell my team to push B",
        "I want the squad to rotate",
    ]:
        out = normalize_command(t)
        assert out.lower().startswith("tell my team "), f"{t!r} -> {out!r}"
        assert out.lower().count("my team") == 1, f"doubled: {t!r} -> {out!r}"


# --- Site letter at the end of a movement order: "to be" -> "to B" -----------
def test_site_letter_at_end_of_movement():
    assert "to B" in normalize_command("tell my team to rotate to be")
    assert "to B" in normalize_command("my team push to be")
    assert "to C" in normalize_command("tell my team to rotate to see")


# --- Audit 2026-06-15: verbatim family, possessive, STT mishears -------------
def _verbatim(text: str):
    return match_relay_command(normalize_command(text))


def test_verbatim_family_relays_exactly():
    # All of the user's "Verbatim" forms must relay with verbatim=True (no tail).
    for t in [
        "repeat to my team watermelon",
        "say to my team mic check one two",
        "say exactly to my team testing testing",
        "repeat to the team spike is down",
        "tell my team word for word rotating now",
        "Pete to my team watermelon",          # STT repeat->Pete
        "Heat to the team spike is down",       # STT repeat->Heat
    ]:
        cmd = _verbatim(t)
        assert cmd is not None and getattr(cmd, "verbatim", False), (
            f"should be verbatim: {t!r} -> {normalize_command(t)!r}")


def test_say_content_to_team_is_not_verbatim():
    # "say <content> to my team" (addressee NOT immediately after say) rephrases.
    cmd = _verbatim("say we are rotating to my team")
    assert cmd is not None and not getattr(cmd, "verbatim", True)


def test_team_possessive_strips_lead():
    out = normalize_command("my team's cypher cage on A")
    assert "team's" not in out.lower(), out
    assert _routes_relay("my team's cypher cage on A")


def test_stt_agent_and_count_mishears():
    assert "Sova" in normalize_command("Silver has his ult")
    # "three" (a count) must NOT be corrupted to the location "tree"
    assert "tree" not in normalize_command("my team three pushing B").lower()
    assert "three" in normalize_command("my team three pushing B").lower()
    # "won mid" -> count "one mid"; the location "tree" is preserved
    assert "one mid" in normalize_command("my team won mid").lower()
    assert "tree" in normalize_command("tell my team split through tree").lower()


def test_hey_agent_blend_dropped():
    # "hey Sage" blends to "Hellsage"; the glued hell-prefix is dropped.
    out = normalize_command("Hellsage nice job")
    assert "hell" not in out.lower(), out
    assert "Sage" in out
