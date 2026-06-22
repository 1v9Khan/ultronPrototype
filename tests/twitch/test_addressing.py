"""Tests for S10a — semantic chat addressing (deterministic-first, FAIL-CLOSED).

Fully OFFLINE: real :class:`ChatEvent` instances (the committed dataclass) + a
deterministic in-memory ``embed_fn`` mock for the residual path. No network, no
creds, no models. Asserts the cost-asymmetric contract: ambiguity / errors / a
bare unaddressed line all fail CLOSED to IGNORE, and resolution is by the
immutable user_id / login so a spoofed display name is ignored.
"""
from __future__ import annotations

import math

import pytest

from kenning.twitch.addressing import (
    AddressVerdict,
    ChatAddress,
    classify_chat,
)
from kenning.twitch.clients.eventsub import ChatEvent

# --- canonical identities under test ----------------------------------------- #
BOT_LOGIN = "ultronbot"
BOT_UID = "999000111"
STREAMER_LOGIN = "thestreamer"
STREAMER_UID = "555000222"

# A third, innocent viewer.
OTHER_LOGIN = "randomviewer"
OTHER_UID = "777000333"


def _classify(event: ChatEvent, *, embed_fn=None) -> AddressVerdict:
    return classify_chat(
        event,
        bot_login=BOT_LOGIN,
        bot_user_id=BOT_UID,
        streamer_login=STREAMER_LOGIN,
        streamer_user_id=STREAMER_UID,
        embed_fn=embed_fn,
    )


def _event(text, *, fragments=None, reply_parent_user_id=None,
           chatter_user_id="100", chatter_login="chatterjoe",
           chatter_name="ChatterJoe") -> ChatEvent:
    return ChatEvent(
        broadcaster_user_id=STREAMER_UID,
        chatter_user_id=chatter_user_id,
        chatter_login=chatter_login,
        chatter_name=chatter_name,
        text=text,
        fragments=fragments or [],
        reply_parent_user_id=reply_parent_user_id,
    )


def _mention_fragment(*, user_id, user_login, user_name=None, text=None):
    """A typed Twitch ``mention`` fragment (nested ``mention`` object form)."""
    return {
        "type": "mention",
        "text": text if text is not None else f"@{user_login}",
        "mention": {
            "user_id": user_id,
            "user_login": user_login,
            "user_name": user_name if user_name is not None else user_login,
        },
    }


# --------------------------------------------------------------------------- #
# 1. reply-to-bot -> TO_ULTRON (immutable parent_user_id)
# --------------------------------------------------------------------------- #
def test_reply_to_bot_is_to_ultron():
    ev = _event("yeah but is that true though", reply_parent_user_id=BOT_UID)
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON
    assert v.confidence >= 0.95
    assert "reply" in v.reason


def test_reply_to_someone_else_is_not_to_ultron_via_reply():
    # Reply to another user, no @bot, bare banter -> not TO_ULTRON by the reply rule.
    ev = _event("lol yeah", reply_parent_user_id=OTHER_UID)
    v = _classify(ev)
    assert v.address == ChatAddress.IGNORE


# --------------------------------------------------------------------------- #
# 2. @bot -> TO_ULTRON (both by typed fragment user_id and by raw @login)
# --------------------------------------------------------------------------- #
def test_at_bot_fragment_user_id_is_to_ultron():
    frag = _mention_fragment(user_id=BOT_UID, user_login=BOT_LOGIN)
    ev = _event(f"@{BOT_LOGIN} what is the score", fragments=[frag])
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON
    assert "user_id" in v.reason


def test_at_bot_raw_login_only_is_to_ultron():
    # No typed fragments at all — only the raw '@login' in the body. Login is
    # immutable, so this still resolves to the bot.
    ev = _event(f"@{BOT_LOGIN.upper()} you there?")  # case-insensitive
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON


# --------------------------------------------------------------------------- #
# 3. @otheruser -> TO_OTHER
# --------------------------------------------------------------------------- #
def test_at_other_user_is_to_other():
    frag = _mention_fragment(user_id=OTHER_UID, user_login=OTHER_LOGIN)
    ev = _event(f"@{OTHER_LOGIN} nice play man", fragments=[frag])
    v = _classify(ev)
    assert v.address == ChatAddress.TO_OTHER


def test_raw_at_other_user_is_to_other():
    ev = _event("@someguy123 you dropped that round")
    v = _classify(ev)
    assert v.address == ChatAddress.TO_OTHER


# --------------------------------------------------------------------------- #
# 4. streamer @ -> TO_STREAMER
# --------------------------------------------------------------------------- #
def test_at_streamer_fragment_is_to_streamer():
    frag = _mention_fragment(user_id=STREAMER_UID, user_login=STREAMER_LOGIN)
    ev = _event(f"@{STREAMER_LOGIN} great stream today", fragments=[frag])
    v = _classify(ev)
    assert v.address == ChatAddress.TO_STREAMER


def test_at_streamer_raw_login_is_to_streamer():
    ev = _event(f"yo @{STREAMER_LOGIN} clutch that round")
    v = _classify(ev)
    assert v.address == ChatAddress.TO_STREAMER


# --------------------------------------------------------------------------- #
# 5. '!command' -> COMMAND  (even when it also @mentions someone)
# --------------------------------------------------------------------------- #
def test_bang_prefix_is_command():
    ev = _event("!points")
    v = _classify(ev)
    assert v.address == ChatAddress.COMMAND
    assert v.confidence >= 0.95


def test_bang_command_wins_over_mention():
    # A '!' command takes precedence over a trailing @bot mention.
    frag = _mention_fragment(user_id=BOT_UID, user_login=BOT_LOGIN)
    ev = _event(f"!gamble 100 @{BOT_LOGIN}", fragments=[frag])
    v = _classify(ev)
    assert v.address == ChatAddress.COMMAND


# --------------------------------------------------------------------------- #
# 6. bare chatter -> IGNORE (fail-closed, no embedder)
# --------------------------------------------------------------------------- #
def test_bare_chatter_ignores_fail_closed():
    ev = _event("that was such a sick flick honestly")
    v = _classify(ev)
    assert v.address == ChatAddress.IGNORE
    assert "fail-closed" in v.reason


def test_empty_text_ignores():
    ev = _event("    ")
    v = _classify(ev)
    assert v.address == ChatAddress.IGNORE


# --------------------------------------------------------------------------- #
# 7. leading 'ultron' token -> TO_ULTRON
# --------------------------------------------------------------------------- #
def test_leading_ultron_token_is_to_ultron():
    ev = _event("ultron are you real")
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON
    assert "leading" in v.reason


def test_leading_ultron_variant_is_to_ultron():
    # Common ASR/typo variant chat uses.
    ev = _event("hey altron what do you think")
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON


def test_leading_bot_login_token_is_to_ultron():
    # Custom-named bot, leading bare login token (no '@').
    ev = _event(f"{BOT_LOGIN} whats the round count")
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON


def test_third_person_ultron_mention_is_not_leading():
    # "ultron is broken" reads as ABOUT the bot, not TO it. No leading-address
    # boost (the regex anchors a name FOLLOWED by address-y content is fine, but a
    # bare third-person statement still leads with 'ultron' -> we accept that as
    # addressed only via the leading rule; assert the mid-sentence case ignores).
    ev = _event("i think the ultron bot is kinda broken lol")
    v = _classify(ev)
    # 'ultron' is mid-sentence, no '@', no leading token -> fail-closed IGNORE.
    assert v.address == ChatAddress.IGNORE


# --------------------------------------------------------------------------- #
# 8. spoofed display name is ignored (resolution uses immutable user_id / login)
# --------------------------------------------------------------------------- #
def test_spoofed_display_name_does_not_resolve_to_bot():
    # A troll sets their DISPLAY name to "Ultron" but their login/user_id are
    # their own. An @mention of THEM must be TO_OTHER, never TO_ULTRON.
    frag = _mention_fragment(
        user_id=OTHER_UID, user_login=OTHER_LOGIN, user_name="Ultron",
    )
    ev = _event(
        f"@{OTHER_LOGIN} gg",
        fragments=[frag],
        chatter_name="Ultron",  # spoofed display name on the chatter too
    )
    v = _classify(ev)
    assert v.address == ChatAddress.TO_OTHER


def test_spoofed_chatter_name_does_not_self_address():
    # The CHATTER's display name is "ultron" but the line is bare banter; the
    # spoofable name must not turn this into TO_ULTRON.
    ev = _event("ggs everyone good games", chatter_name="ultron")
    v = _classify(ev)
    assert v.address == ChatAddress.IGNORE


# --------------------------------------------------------------------------- #
# 9. residual embed path — both directions (margin honored)
# --------------------------------------------------------------------------- #
def _direction_embed_fn():
    """A deterministic mock embedder.

    Lines that look like they're addressed to the bot embed near unit vector
    ``[1, 0]``; banter embeds near ``[0, 1]``. The to-Ultron exemplar cloud lands
    near ``[1,0]`` and the not-cloud near ``[0,1]``, so a to-Ultron query gets a
    positive margin and a banter query a negative one — exercising both residual
    branches deterministically with no real model.
    """
    pos_markers = ("you", "your", "answer", "respond", "think", "real",
                   "opinion", "funny", "joke", "watching", "understand", "would")
    neg_markers = ("gg", "lol", "stream", "game", "clutch", "poggers", "same",
                   "lagging", "hello", "insane", "love", "time")

    def embed(text: str):
        t = (text or "").lower()
        pos = sum(t.count(m) for m in pos_markers)
        neg = sum(t.count(m) for m in neg_markers)
        # Bias toward [1,0] when pos-leaning, [0,1] when neg-leaning. Always a
        # nonzero, finite 2-vector.
        x = 1.0 + 2.0 * pos
        y = 1.0 + 2.0 * neg
        norm = math.sqrt(x * x + y * y)
        return [x / norm, y / norm]

    return embed


def test_residual_to_ultron_direction():
    ev = _event("do you actually understand what we say and would you respond")
    v = _classify(ev, embed_fn=_direction_embed_fn())
    assert v.address == ChatAddress.TO_ULTRON
    assert "residual" in v.reason


def test_residual_not_to_ultron_direction_ignores():
    ev = _event("gg that clutch was insane lol same poggers")
    v = _classify(ev, embed_fn=_direction_embed_fn())
    assert v.address == ChatAddress.IGNORE
    assert "residual" in v.reason


def test_residual_without_embedder_ignores():
    # Same to-Ultron-leaning line, but NO embedder supplied -> fail-closed IGNORE.
    ev = _event("do you actually understand what we say")
    v = _classify(ev, embed_fn=None)
    assert v.address == ChatAddress.IGNORE
    assert "fail-closed" in v.reason


# --------------------------------------------------------------------------- #
# 10. fail-closed robustness — embedder raising / returning garbage
# --------------------------------------------------------------------------- #
def test_residual_embedder_raises_fails_closed():
    def boom(_text):
        raise RuntimeError("embedder sidecar down")

    ev = _event("do you understand what we are saying right now")
    v = _classify(ev, embed_fn=boom)
    assert v.address == ChatAddress.IGNORE
    assert "fail-closed" in v.reason


def test_residual_embedder_returns_empty_fails_closed():
    ev = _event("would you answer my question please")
    v = _classify(ev, embed_fn=lambda _t: [])
    assert v.address == ChatAddress.IGNORE


def test_residual_embedder_returns_nan_fails_closed():
    ev = _event("would you answer my question please")
    v = _classify(ev, embed_fn=lambda _t: [float("nan"), 1.0])
    assert v.address == ChatAddress.IGNORE


# --------------------------------------------------------------------------- #
# precedence + parsing edge cases
# --------------------------------------------------------------------------- #
def test_reply_to_bot_beats_at_other_user():
    # Reply parent is the bot AND the body @mentions another user -> reply wins.
    frag = _mention_fragment(user_id=OTHER_UID, user_login=OTHER_LOGIN)
    ev = _event(f"@{OTHER_LOGIN} yeah", reply_parent_user_id=BOT_UID, fragments=[frag])
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON


def test_at_bot_beats_at_other_when_both_mentioned():
    # Body @mentions both the bot and another viewer -> bot resolution wins.
    frags = [
        _mention_fragment(user_id=OTHER_UID, user_login=OTHER_LOGIN),
        _mention_fragment(user_id=BOT_UID, user_login=BOT_LOGIN),
    ]
    ev = _event(f"@{OTHER_LOGIN} and @{BOT_LOGIN} settle this", fragments=frags)
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON


def test_from_eventsub_mention_shape_round_trips():
    # Build a real EventSub-shaped payload through ChatEvent.from_eventsub so the
    # fragment-parsing contract is exercised end to end.
    payload = {
        "event": {
            "broadcaster_user_id": STREAMER_UID,
            "chatter_user_id": "100",
            "chatter_user_login": "chatterjoe",
            "chatter_user_name": "ChatterJoe",
            "message_id": "abc-123",
            "message": {
                "text": f"@{BOT_LOGIN} are you watching this",
                "message_type": "text",
                "fragments": [
                    {"type": "text", "text": ""},
                    _mention_fragment(user_id=BOT_UID, user_login=BOT_LOGIN),
                    {"type": "text", "text": " are you watching this"},
                ],
            },
        }
    }
    ev = ChatEvent.from_eventsub(payload)
    assert ev is not None
    v = _classify(ev)
    assert v.address == ChatAddress.TO_ULTRON


def test_garbage_fragments_do_not_crash_fail_closed():
    # Malformed fragments (not dicts / missing keys) must not raise — fail-closed.
    ev = _event(
        "just chatting here",
        fragments=[None, 42, {"type": "mention"}, {"no_type": True}],
    )
    v = _classify(ev)
    assert v.address == ChatAddress.IGNORE


def test_non_chatevent_object_fails_closed():
    # A wholly unexpected object (no .text / .fragments) must IGNORE, never raise.
    class Weird:
        pass

    v = classify_chat(
        Weird(),
        bot_login=BOT_LOGIN,
        bot_user_id=BOT_UID,
        streamer_login=STREAMER_LOGIN,
        streamer_user_id=STREAMER_UID,
    )
    assert v.address == ChatAddress.IGNORE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
