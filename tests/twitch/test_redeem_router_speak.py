"""Speak-redeem tests for the channel-point REDEEM ROUTER (2026-06-26).

The two NON-game "speak" redeems let a VIEWER make Ultron speak their own typed
message via TTS after a MANDATORY Llama-Guard safety screen. Fully offline: the
guard is a mocked classify fn, the speak callbacks are list appenders, and the
router is driven with an injected drain_fn returning canned redeem dicts. No live
sidecar / model / network.

Asserts:
  * SAFE text on the "say" title -> say_speak_fn called ONCE with the framed text;
    team_speak_fn NOT called; the framed line names the viewer.
  * SAFE text on the "team" title -> team_speak_fn called ONCE; say NOT called.
  * UNSAFE text -> NEITHER speak callback called; the optional blocked-chat note
    is posted; the guard NEVER reveals what tripped.
  * guard UNREACHABLE / raises -> FAIL-CLOSED (no speak).
  * no guard configured -> FAIL-CLOSED (no speak).
  * idempotent on redemption_id: the same id twice -> spoken once.
  * the team title with NO team callback wired -> not spoken (chat->team boundary).
  * length is bounded + control chars stripped before TTS (sanitize unit tests).
  * a speak redeem is NOT a game (no overlay redeem_result, no ledger move).
"""
from __future__ import annotations

from kenning.twitch.economy.rng import ProvablyFairRNG
from kenning.twitch.redeem_router import (
    SPEAK_SAY,
    SPEAK_TEAM,
    RedeemRouter,
    frame_speak_line,
    sanitize_speak_text,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _redeem(rid: str, title: str, *, login: str = "viewer1", user_input: str = "") -> dict:
    return {
        "type": "redeem",
        "redemption_id": rid,
        "reward_id": "rw-" + rid,
        "reward_title": title,
        "user_input": user_input,
        "chatter_login": login,
        "chatter_name": login.capitalize(),
        "chatter_user_id": "uid-" + rid,
        "status": "unfulfilled",
    }


def _seeded_rng() -> ProvablyFairRNG:
    return ProvablyFairRNG(default_client_seed="ultron")


class _GuardResult:
    """Mimics kenning.twitch.safety.validator.GuardResult enough for the router."""

    def __init__(self, unsafe: bool, category: str = "") -> None:
        self.unsafe = unsafe
        self.category = category
        self.score = 0.95 if unsafe else 0.05


def _safe_guard(_text: str) -> _GuardResult:
    return _GuardResult(unsafe=False)


def _unsafe_guard(_text: str) -> _GuardResult:
    return _GuardResult(unsafe=True, category="S10")


_SAY_TITLE = "ultron says"
_TEAM_TITLE = "ultron tells my team"
_SPEAK_MAP = {_SAY_TITLE: SPEAK_SAY, _TEAM_TITLE: SPEAK_TEAM}


def _router(*, drain, guard=_safe_guard, say=None, team=None, blocked=None,
            max_chars: int = 200, ledger=None, overlay=None):
    return RedeemRouter(
        drain_fn=drain,
        rng=_seeded_rng(),
        ledger=ledger,
        overlay_emit=overlay,
        speak_reward_map=_SPEAK_MAP,
        guard_classify_fn=guard,
        say_speak_fn=say,
        team_speak_fn=team,
        blocked_chat_fn=blocked,
        speak_max_chars=max_chars,
    )


# --------------------------------------------------------------------------- #
# sanitize_speak_text (pure)
# --------------------------------------------------------------------------- #
def test_sanitize_strips_control_chars_and_collapses_whitespace() -> None:
    # Control chars (NUL, BEL, ESC) are DROPPED; tab/newline become spaces and
    # whitespace runs collapse. Printable residue (e.g. the "[31m" left after an
    # ESC is stripped from an ANSI sequence) is ORDINARY text and is kept -- the
    # sanitizer strips control CHARS, it is not an ANSI-sequence parser.
    raw = "hello\x00\x07 \t\n  world\x1b[31m!"
    out = sanitize_speak_text(raw, max_chars=200)
    assert out == "hello world[31m!"
    # No control chars survive.
    assert all(ord(c) >= 0x20 and ord(c) != 0x7F for c in out)


def test_sanitize_caps_length_at_word_boundary() -> None:
    raw = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    out = sanitize_speak_text(raw, max_chars=20)
    assert len(out) <= 20
    # Trimmed back to a word boundary -> no trailing partial word / no trailing space.
    assert not out.endswith(" ")
    assert out.split()[-1] in raw.split()


def test_sanitize_caps_even_a_single_huge_token() -> None:
    out = sanitize_speak_text("x" * 1000, max_chars=50)
    assert len(out) == 50


def test_sanitize_empty_and_whitespace_only() -> None:
    assert sanitize_speak_text("", max_chars=200) == ""
    assert sanitize_speak_text("   \t\n  ", max_chars=200) == ""
    assert sanitize_speak_text(None, max_chars=200) == ""


def test_sanitize_respects_hard_max_over_config() -> None:
    # A misconfigured huge cap is still clamped by the module hard ceiling (500).
    out = sanitize_speak_text("y " * 2000, max_chars=100000)
    assert len(out) <= 500


# --------------------------------------------------------------------------- #
# frame_speak_line
# --------------------------------------------------------------------------- #
def test_frame_say_names_the_viewer() -> None:
    line = frame_speak_line("alice", "good luck tonight", to_team=False)
    assert "alice" in line.lower()
    assert "good luck tonight" in line
    assert "says" in line.lower()


def test_frame_team_names_viewer_no_relay_prefix() -> None:
    # 2026-06-26: the team variant DROPPED the leading "Relaying from chat."
    # prefix; by default (SAY-NAME ON) it still names the viewer.
    from kenning.twitch.redeem_router import set_say_name_enabled
    set_say_name_enabled(True)
    line = frame_speak_line("bob", "you got this", to_team=True)
    assert "relaying from chat" not in line.lower()
    assert "bob" in line.lower()
    assert "says" in line.lower()
    assert "you got this" in line


def test_frame_team_say_name_off_speaks_bare_message() -> None:
    # SAY-NAME OFF -> the team line is the bare message (no viewer prefix).
    from kenning.twitch.redeem_router import set_say_name_enabled
    try:
        set_say_name_enabled(False)
        line = frame_speak_line("bob", "you got this", to_team=True)
        assert line == "you got this"
        assert "bob" not in line.lower()
    finally:
        set_say_name_enabled(True)


def test_frame_empty_body_returns_empty() -> None:
    assert frame_speak_line("alice", "", to_team=False) == ""


# --------------------------------------------------------------------------- #
# say redeem
# --------------------------------------------------------------------------- #
def test_say_redeem_safe_text_speaks_once_on_say_bus_only() -> None:
    say: list[str] = []
    team: list[str] = []
    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, login="alice",
                               user_input="good luck have fun")],
        say=say.append, team=team.append,
    )
    outcomes = router.tick()
    assert len(say) == 1
    assert team == []  # never the team mic
    assert "alice" in say[0].lower()
    assert "good luck have fun" in say[0]
    # The outcome dict reports it spoke on the say bus.
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o["type"] == "redeem_speak" and o["bus"] == "say" and o["spoken"] is True


def test_say_redeem_emits_unified_speech_card() -> None:
    overlay: list[dict] = []
    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, login="alice", user_input="hi chat")],
        say=lambda _l: None, overlay=overlay.append,
    )
    router.tick()
    # A spoken speak redeem now renders the SAME bottom-left card style as the
    # games — a "speech" card carrying the viewer + the (framed) spoken line.
    from kenning.twitch.overlay.server import validate_event
    assert len(overlay) == 1
    card = overlay[0]
    assert card["type"] == "speech" and card["bus"] == "say"
    assert "alice" in card["viewer"]
    assert "hi chat" in card["text"]
    validate_event(card)   # must pass the real overlay validator


def test_blocked_speak_redeem_emits_no_card() -> None:
    # A guard-blocked speak never spoke -> no card (only spoken speaks render).
    overlay: list[dict] = []
    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, login="alice", user_input="bad")],
        guard=_unsafe_guard, say=lambda _l: None, overlay=overlay.append,
    )
    router.tick()
    assert overlay == []


# --------------------------------------------------------------------------- #
# team redeem
# --------------------------------------------------------------------------- #
def test_team_redeem_safe_text_speaks_on_team_bus_only() -> None:
    say: list[str] = []
    team: list[str] = []
    router = _router(
        drain=lambda: [_redeem("r1", _TEAM_TITLE, login="carol",
                               user_input="nice round team")],
        say=say.append, team=team.append,
    )
    router.tick()
    assert len(team) == 1
    assert say == []
    assert "nice round team" in team[0]
    assert "carol" in team[0].lower()


def test_team_redeem_without_team_callback_is_not_spoken() -> None:
    # The chat->team boundary: when the streamer hasn't opted the team redeem in,
    # the orchestrator wires NO team_speak_fn -> the team title is refused.
    say: list[str] = []
    router = _router(
        drain=lambda: [_redeem("r1", _TEAM_TITLE, user_input="rush B")],
        say=say.append, team=None,
    )
    outcomes = router.tick()
    assert say == []
    assert outcomes[0]["spoken"] is False
    assert "not enabled" in outcomes[0]["reason"]


# --------------------------------------------------------------------------- #
# guard (the mandatory safety screen) — fail-CLOSED everywhere
# --------------------------------------------------------------------------- #
def test_unsafe_text_is_not_spoken_and_posts_blocked_note() -> None:
    say: list[str] = []
    blocked: list[str] = []
    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, login="troll",
                               user_input="<something hateful>")],
        guard=_unsafe_guard, say=say.append, blocked=blocked.append,
    )
    outcomes = router.tick()
    assert say == []                       # NOT spoken
    assert len(blocked) == 1               # a brief chat note was posted
    # The note never reveals the category (anti-probe).
    assert "S10" not in blocked[0]
    assert outcomes[0]["spoken"] is False


def test_guard_raises_fails_closed() -> None:
    say: list[str] = []

    def boom_guard(_text: str):
        raise RuntimeError("guard sidecar down")

    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, user_input="hello")],
        guard=boom_guard, say=say.append,
    )
    outcomes = router.tick()
    assert say == []                       # fail-CLOSED on guard error
    assert outcomes[0]["spoken"] is False
    assert "fail-closed" in outcomes[0]["reason"]


def test_no_guard_configured_fails_closed() -> None:
    say: list[str] = []
    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, user_input="hello")],
        guard=None, say=say.append,
    )
    router.tick()
    assert say == []                       # no guard -> never speak


# --------------------------------------------------------------------------- #
# sanitize is applied before TTS through the router
# --------------------------------------------------------------------------- #
def test_router_sanitizes_and_bounds_before_speaking() -> None:
    say: list[str] = []
    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, login="al",
                               user_input="ha\x00ha " + ("z" * 999))],
        say=say.append, max_chars=40,
    )
    router.tick()
    assert len(say) == 1
    spoken = say[0]
    # No control chars reached TTS.
    assert "\x00" not in spoken
    # The body (after the "al says: " frame) is capped to <= max_chars.
    body = spoken.split("says:", 1)[1].strip()
    assert len(body) <= 40


def test_empty_user_input_is_not_spoken() -> None:
    say: list[str] = []
    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, user_input="   ")],
        say=say.append,
    )
    outcomes = router.tick()
    assert say == []
    assert outcomes[0]["spoken"] is False
    assert outcomes[0]["reason"] == "empty input"


# --------------------------------------------------------------------------- #
# idempotency (EventSub replay safety)
# --------------------------------------------------------------------------- #
def test_speak_redeem_is_idempotent_on_redemption_id() -> None:
    say: list[str] = []
    batches = [
        [_redeem("dup", _SAY_TITLE, user_input="hi"),
         _redeem("dup", _SAY_TITLE, user_input="hi")],   # dup within a batch
        [_redeem("dup", _SAY_TITLE, user_input="hi")],    # dup across ticks
    ]

    def drain() -> list[dict]:
        return batches.pop(0) if batches else []

    router = _router(drain=drain, say=say.append)
    router.tick()
    router.tick()
    assert len(say) == 1                   # spoken exactly once despite 3 events


def test_speak_failure_does_not_break_tick() -> None:
    def boom_say(_line: str) -> None:
        raise RuntimeError("tts down")

    router = _router(
        drain=lambda: [_redeem("r1", _SAY_TITLE, user_input="hello")],
        say=boom_say,
    )
    outcomes = router.tick()               # must not raise
    assert outcomes[0]["spoken"] is False
    assert "speak error" in outcomes[0]["reason"]


# --------------------------------------------------------------------------- #
# feature-OFF parity: a router with NO speak map ignores the titles
# --------------------------------------------------------------------------- #
def test_no_speak_map_treats_title_as_non_game_redeem() -> None:
    # Without a speak map (feature OFF), "ultron says" is just an unknown reward:
    # not a game, so the unified card style emits NO card (the old generic alert
    # banner was retired). No speak, no crash.
    overlay: list[dict] = []
    router = RedeemRouter(
        drain_fn=lambda: [_redeem("r1", _SAY_TITLE, login="zoe")],
        rng=_seeded_rng(),
        overlay_emit=overlay.append,
    )
    outcomes = router.tick()
    assert outcomes == []                  # not a game
    assert overlay == []                   # no card for a non-game redeem
