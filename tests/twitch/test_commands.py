"""Tests for the closed-grammar chat-command parser (S10/S12).

Drives :func:`kenning.twitch.commands.parse_command` with synthetic
:class:`ChatEvent` objects — no network, no creds, no models. Every recognised
command parses to its kind with the right typed args; bad amounts are rejected;
non-prefixed lines return None; unknown ``!tokens`` map to UNKNOWN; ``is_mod`` is
resolved from EventSub badges only.
"""
from __future__ import annotations

import pytest

from kenning.twitch.clients.eventsub import ChatEvent
from kenning.twitch.commands import (
    ALL_SENTINEL,
    MAX_AMOUNT,
    Command,
    CommandKind,
    parse_command,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_event(
    text: str,
    *,
    badges: list[dict] | None = None,
    user_id: str = "u123",
    login: str = "alice",
) -> ChatEvent:
    """Construct a minimal ChatEvent carrying ``text`` + optional badges."""
    return ChatEvent(
        broadcaster_user_id="b1",
        chatter_user_id=user_id,
        chatter_login=login,
        chatter_name=login.capitalize(),
        text=text,
        badges=list(badges or []),
        message_id="m1",
    )


MOD_BADGE = [{"set_id": "moderator", "id": "1", "info": ""}]
BROADCASTER_BADGE = [{"set_id": "broadcaster", "id": "1", "info": ""}]
SUB_BADGE = [{"set_id": "subscriber", "id": "12", "info": "12"}]


# --------------------------------------------------------------------------- #
# each command parses with correct kind/args
# --------------------------------------------------------------------------- #
def test_points_parses():
    cmd = parse_command(make_event("!points"))
    assert cmd is not None
    assert cmd.kind is CommandKind.POINTS
    assert cmd.args == {}
    assert cmd.user_id == "u123"
    assert cmd.user_login == "alice"
    assert cmd.raw == "!points"


def test_wheel_parses_no_args():
    cmd = parse_command(make_event("!wheel"))
    assert cmd is not None
    assert cmd.kind is CommandKind.WHEEL
    assert cmd.args == {}


def test_trivia_parses_no_args():
    cmd = parse_command(make_event("!trivia"))
    assert cmd is not None and cmd.kind is CommandKind.TRIVIA


def test_leaderboard_parses():
    cmd = parse_command(make_event("!leaderboard"))
    assert cmd is not None and cmd.kind is CommandKind.LEADERBOARD


def test_help_parses():
    cmd = parse_command(make_event("!help"))
    assert cmd is not None and cmd.kind is CommandKind.HELP


def test_gamble_amount_parses():
    cmd = parse_command(make_event("!gamble 50"))
    assert cmd is not None
    assert cmd.kind is CommandKind.GAMBLE
    assert cmd.args == {"amount": 50}


def test_slots_amount_parses():
    cmd = parse_command(make_event("!slots 250"))
    assert cmd is not None
    assert cmd.kind is CommandKind.SLOTS
    assert cmd.args["amount"] == 250


def test_heist_amount_parses():
    cmd = parse_command(make_event("!heist 1000"))
    assert cmd is not None
    assert cmd.kind is CommandKind.HEIST
    assert cmd.args["amount"] == 1000


# --------------------------------------------------------------------------- #
# '!gamble all' special sentinel
# --------------------------------------------------------------------------- #
def test_gamble_all_special():
    cmd = parse_command(make_event("!gamble all"))
    assert cmd is not None
    assert cmd.kind is CommandKind.GAMBLE
    assert cmd.args == {"amount": ALL_SENTINEL}


def test_gamble_all_is_case_insensitive():
    cmd = parse_command(make_event("!gamble ALL"))
    assert cmd is not None and cmd.args["amount"] == ALL_SENTINEL


def test_heist_all_allowed():
    cmd = parse_command(make_event("!heist all"))
    assert cmd is not None and cmd.args["amount"] == ALL_SENTINEL


def test_give_all_rejected():
    # !give does NOT permit 'all' — must be a concrete integer.
    cmd = parse_command(make_event("!give @bob all"))
    assert cmd is not None
    assert cmd.kind is CommandKind.GIVE
    assert "amount" not in cmd.args
    assert "error" in cmd.args
    assert cmd.args["target"] == "bob"


# --------------------------------------------------------------------------- #
# negative / overflow / non-numeric amount rejected
# --------------------------------------------------------------------------- #
def test_negative_amount_rejected():
    cmd = parse_command(make_event("!gamble -5"))
    assert cmd is not None
    assert cmd.kind is CommandKind.GAMBLE
    assert "amount" not in cmd.args
    assert "error" in cmd.args


def test_zero_amount_rejected():
    cmd = parse_command(make_event("!gamble 0"))
    assert cmd is not None
    assert "amount" not in cmd.args
    assert "error" in cmd.args


def test_overflow_amount_rejected():
    huge = str(MAX_AMOUNT + 1)
    cmd = parse_command(make_event(f"!gamble {huge}"))
    assert cmd is not None
    assert "amount" not in cmd.args
    assert "error" in cmd.args


def test_giant_number_rejected():
    cmd = parse_command(make_event("!slots 99999999999999999999999999999"))
    assert cmd is not None
    assert "amount" not in cmd.args
    assert "error" in cmd.args


def test_non_numeric_amount_rejected():
    cmd = parse_command(make_event("!gamble lots"))
    assert cmd is not None
    assert "amount" not in cmd.args
    assert "error" in cmd.args


def test_decimal_amount_rejected():
    cmd = parse_command(make_event("!gamble 5.5"))
    assert cmd is not None
    assert "amount" not in cmd.args
    assert "error" in cmd.args


def test_max_amount_boundary_allowed():
    cmd = parse_command(make_event(f"!gamble {MAX_AMOUNT}"))
    assert cmd is not None
    assert cmd.args == {"amount": MAX_AMOUNT}


def test_gamble_missing_amount_rejected():
    cmd = parse_command(make_event("!gamble"))
    assert cmd is not None
    assert cmd.kind is CommandKind.GAMBLE
    assert "amount" not in cmd.args
    assert "error" in cmd.args


# --------------------------------------------------------------------------- #
# !duel @user <amount> extracts target + amount
# --------------------------------------------------------------------------- #
def test_duel_extracts_target_and_amount():
    cmd = parse_command(make_event("!duel @bob 50"))
    assert cmd is not None
    assert cmd.kind is CommandKind.DUEL
    assert cmd.args == {"target": "bob", "amount": 50}


def test_duel_target_without_at_sign():
    cmd = parse_command(make_event("!duel bob 50"))
    assert cmd is not None
    assert cmd.args["target"] == "bob"
    assert cmd.args["amount"] == 50


def test_duel_target_lowercased():
    cmd = parse_command(make_event("!duel @BoB 50"))
    assert cmd is not None and cmd.args["target"] == "bob"


def test_duel_all_amount():
    cmd = parse_command(make_event("!duel @bob all"))
    assert cmd is not None
    assert cmd.args["target"] == "bob"
    assert cmd.args["amount"] == ALL_SENTINEL


def test_duel_missing_amount_rejected():
    cmd = parse_command(make_event("!duel @bob"))
    assert cmd is not None
    assert cmd.kind is CommandKind.DUEL
    assert "amount" not in cmd.args
    assert "error" in cmd.args


def test_duel_invalid_target_rejected():
    cmd = parse_command(make_event("!duel @@@ 50"))
    assert cmd is not None
    assert cmd.kind is CommandKind.DUEL
    assert "target" not in cmd.args
    assert "error" in cmd.args


def test_give_extracts_target_and_amount():
    cmd = parse_command(make_event("!give @carol 30"))
    assert cmd is not None
    assert cmd.kind is CommandKind.GIVE
    assert cmd.args == {"target": "carol", "amount": 30}


def test_give_negative_amount_rejected():
    cmd = parse_command(make_event("!give @carol -10"))
    assert cmd is not None
    assert "amount" not in cmd.args
    assert "error" in cmd.args
    assert cmd.args["target"] == "carol"


# --------------------------------------------------------------------------- #
# is_mod resolved from badges (never from text)
# --------------------------------------------------------------------------- #
def test_is_mod_from_moderator_badge():
    cmd = parse_command(make_event("!points", badges=MOD_BADGE))
    assert cmd is not None and cmd.is_mod is True


def test_is_mod_from_broadcaster_badge():
    cmd = parse_command(make_event("!points", badges=BROADCASTER_BADGE))
    assert cmd is not None and cmd.is_mod is True


def test_is_mod_false_for_subscriber():
    cmd = parse_command(make_event("!points", badges=SUB_BADGE))
    assert cmd is not None and cmd.is_mod is False


def test_is_mod_false_with_no_badges():
    cmd = parse_command(make_event("!points", badges=[]))
    assert cmd is not None and cmd.is_mod is False


def test_is_mod_not_spoofable_from_text():
    # A chatter literally typing "moderator" in the body gets no authority.
    cmd = parse_command(make_event("!points moderator broadcaster"))
    assert cmd is not None and cmd.is_mod is False


def test_is_mod_malformed_badges_fail_safe():
    # Garbage badge structures never grant mod.
    cmd = parse_command(make_event("!points", badges=[{"id": "x"}, "junk", 42]))
    assert cmd is not None and cmd.is_mod is False


# --------------------------------------------------------------------------- #
# non-prefixed -> None
# --------------------------------------------------------------------------- #
def test_non_prefixed_returns_none():
    assert parse_command(make_event("hello team how is everyone")) is None


def test_word_containing_bang_midline_returns_none():
    assert parse_command(make_event("that was great!points")) is None


def test_empty_text_returns_none():
    assert parse_command(make_event("")) is None


def test_bare_prefix_returns_none():
    assert parse_command(make_event("!")) is None


def test_prefix_with_only_whitespace_returns_none():
    assert parse_command(make_event("!   ")) is None


def test_none_event_returns_none():
    assert parse_command(None) is None


# --------------------------------------------------------------------------- #
# unknown '!foo' -> UNKNOWN
# --------------------------------------------------------------------------- #
def test_unknown_command_maps_to_unknown():
    cmd = parse_command(make_event("!foo bar baz"))
    assert cmd is not None
    assert cmd.kind is CommandKind.UNKNOWN
    assert cmd.args.get("command") == "foo"


def test_unknown_command_still_resolves_mod():
    cmd = parse_command(make_event("!nope", badges=MOD_BADGE))
    assert cmd is not None
    assert cmd.kind is CommandKind.UNKNOWN
    assert cmd.is_mod is True


# --------------------------------------------------------------------------- #
# grammar robustness: case, whitespace, custom prefix, aliases
# --------------------------------------------------------------------------- #
def test_command_word_case_insensitive():
    cmd = parse_command(make_event("!POINTS"))
    assert cmd is not None and cmd.kind is CommandKind.POINTS


def test_extra_whitespace_tolerated():
    cmd = parse_command(make_event("!gamble    75"))
    assert cmd is not None and cmd.args == {"amount": 75}


def test_leading_whitespace_before_prefix():
    cmd = parse_command(make_event("   !points"))
    assert cmd is not None and cmd.kind is CommandKind.POINTS


def test_custom_prefix():
    cmd = parse_command(make_event("?points"), prefix="?")
    assert cmd is not None and cmd.kind is CommandKind.POINTS


def test_custom_prefix_does_not_match_default():
    # With a custom prefix, the default '!' line is ordinary chat.
    assert parse_command(make_event("!points"), prefix="?") is None


def test_balance_alias_maps_to_points():
    cmd = parse_command(make_event("!balance"))
    assert cmd is not None and cmd.kind is CommandKind.POINTS


def test_top_alias_maps_to_leaderboard():
    cmd = parse_command(make_event("!top"))
    assert cmd is not None and cmd.kind is CommandKind.LEADERBOARD


def test_returns_command_instance():
    cmd = parse_command(make_event("!points"))
    assert isinstance(cmd, Command)


def test_extra_trailing_tokens_ignored_for_bet():
    # '!gamble 50 extra junk' still parses the amount; extra tokens are ignored
    # by the closed grammar (no free text reaches a model).
    cmd = parse_command(make_event("!gamble 50 lol gg"))
    assert cmd is not None and cmd.args == {"amount": 50}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
