"""Tests for the bot chat-SEND client + the periodic commands-panel text.

Fully offline: the Helix transport is injected (no network); the panel builder is
pure. Covers the send happy-path, a Twitch ``is_sent=false`` drop, a missing token,
empty text, a non-2xx, the 500-char trim, and the panel text with/without a guide
URL.
"""
from __future__ import annotations

import json
import types

from kenning.twitch.clients.chat_send import MAX_MESSAGE_CHARS, ChatSendClient
from kenning.twitch.panel import (
    MAX_CHAT_CHARS,
    append_cooldown_hint,
    build_commands_panel_text,
    cooldown_hint_suffix,
    run_interval_poster,
)


def _ok_transport(record=None):
    def t(method, url, headers, body):
        if record is not None:
            record.append((method, url, dict(headers), json.loads(body)))
        return 200, json.dumps({"data": [{"is_sent": True}]}).encode()
    return t


def test_chat_send_posts_and_reports_sent():
    rec = []
    c = ChatSendClient("cid", get_token=lambda: "tok", transport=_ok_transport(rec))
    assert c.send("B1", "U1", "hello chat") is True
    method, url, headers, body = rec[0]
    assert method == "POST" and url.endswith("/chat/messages")
    assert body == {"broadcaster_id": "B1", "sender_id": "U1", "message": "hello chat"}
    assert headers["Authorization"] == "Bearer tok" and headers["Client-Id"] == "cid"


def test_chat_send_dropped_by_twitch_returns_false():
    def t(method, url, headers, body):
        return 200, json.dumps({"data": [{"is_sent": False, "drop_reason": {"code": "x"}}]}).encode()
    c = ChatSendClient("cid", get_token=lambda: "tok", transport=t)
    assert c.send("B1", "U1", "x") is False


def test_chat_send_requires_token_and_nonempty_text():
    c = ChatSendClient("cid", get_token=lambda: "", transport=_ok_transport())
    assert c.send("B1", "U1", "x") is False                    # no token
    c2 = ChatSendClient("cid", get_token=lambda: "tok", transport=_ok_transport())
    assert c2.send("B1", "U1", "   ") is False                  # empty text
    assert c2.send("", "U1", "hi") is False                     # no broadcaster id


def test_chat_send_non_2xx_is_false():
    c = ChatSendClient("cid", get_token=lambda: "tok",
                       transport=lambda *a: (401, b'{"message":"unauthorized"}'))
    assert c.send("B1", "U1", "x") is False


def test_chat_send_transport_raise_is_false():
    def boom(*a):
        raise RuntimeError("dns")
    c = ChatSendClient("cid", get_token=lambda: "tok", transport=boom)
    assert c.send("B1", "U1", "x") is False


def test_chat_send_trims_to_500():
    captured = {}

    def t(method, url, headers, body):
        captured["msg"] = json.loads(body)["message"]
        return 200, b'{"data":[{"is_sent":true}]}'
    c = ChatSendClient("cid", get_token=lambda: "tok", transport=t)
    c.send("B1", "U1", "x" * 600)
    assert len(captured["msg"]) == MAX_MESSAGE_CHARS == 500


def test_panel_text_without_url():
    t = build_commands_panel_text(types.SimpleNamespace(commands_panel_doc_url=""))
    # The current condensed panel advertises the live games + the Credits currency
    # (the !gamble/!points line is handled by StreamElements, so it's not listed).
    assert "!slots" in t and "!help" in t and "!heist" in t
    assert "Credits" in t
    assert "Full guide" not in t and len(t) <= MAX_CHAT_CHARS


def test_panel_text_with_url_and_length_cap():
    t = build_commands_panel_text(types.SimpleNamespace(commands_panel_doc_url="https://docs.example/g"))
    assert "Full guide" in t and "https://docs.example/g" in t
    assert len(t) <= MAX_CHAT_CHARS


# --------------------------------------------------------------------------- #
# run_interval_poster — the talk-to-Ultron hint poster (injected clock)
# --------------------------------------------------------------------------- #
class _FakeClock:
    """Counts sleep() ticks and stops the poster after enough virtual seconds so a
    test never blocks (1s slices, like the real loop)."""

    def __init__(self, stop_after_s):
        self.t = 0.0
        self._stop_after = stop_after_s

    def sleep(self, dt):
        self.t += dt

    def should_stop(self):
        return self.t >= self._stop_after


def test_interval_poster_fires_on_interval_with_offset():
    posts = []
    # offset 5s, interval 10s, stop at 28s -> posts at t=5, 15, 25 (3 posts).
    clock = _FakeClock(stop_after_s=28.0)
    run_interval_poster(
        lambda: "💬 hint", posts.append,
        interval_s=10.0, should_stop=clock.should_stop,
        sleep_fn=clock.sleep, first_offset_s=5.0,
    )
    assert posts == ["💬 hint", "💬 hint", "💬 hint"]


def test_interval_poster_empty_text_posts_nothing():
    posts = []
    clock = _FakeClock(stop_after_s=25.0)
    run_interval_poster(
        lambda: "", posts.append,
        interval_s=10.0, should_stop=clock.should_stop,
        sleep_fn=clock.sleep, first_offset_s=5.0,
    )
    assert posts == []   # an empty build never posts


def test_interval_poster_swallows_post_errors():
    calls = {"n": 0}

    def boom(_text):
        calls["n"] += 1
        raise RuntimeError("sidecar down")

    clock = _FakeClock(stop_after_s=28.0)
    # Must NOT raise even though every post raises; the loop survives.
    run_interval_poster(
        lambda: "x", boom,
        interval_s=10.0, should_stop=clock.should_stop,
        sleep_fn=clock.sleep, first_offset_s=5.0,
    )
    assert calls["n"] == 3   # tried each scheduled post, none crashed the loop


def test_cooldown_hint_suffix_minutes_and_seconds():
    # whole minutes -> "(N minute cooldown)"; the user's 2-minute default.
    assert cooldown_hint_suffix(120) == "(2 minute cooldown)"
    assert cooldown_hint_suffix(60) == "(1 minute cooldown)"
    # non-whole-minute -> seconds.
    assert cooldown_hint_suffix(30) == "(30 second cooldown)"
    assert cooldown_hint_suffix(90) == "(90 second cooldown)"
    # off / invalid -> empty.
    assert cooldown_hint_suffix(0) == ""
    assert cooldown_hint_suffix(-5) == ""


def test_append_cooldown_hint_is_idempotent():
    base = "Talk to Ultron!"
    once = append_cooldown_hint(base, 120)
    assert once == "Talk to Ultron! (2 minute cooldown)"
    # appending again does not duplicate the suffix.
    assert append_cooldown_hint(once, 120) == once
    # cooldown off -> unchanged.
    assert append_cooldown_hint(base, 0) == base


def test_reply_cooldown_config_default():
    from kenning.config import TwitchChatConfig
    assert TwitchChatConfig().reply_cooldown_seconds == 120


def test_talk_hint_config_defaults():
    from kenning.config import TwitchChatConfig
    c = TwitchChatConfig()
    assert c.talk_hint_enabled is True
    assert c.talk_hint_interval_minutes == 10
    assert c.talk_hint_text == (
        '💬 Just type "Ultron" followed by a statement or question '
        "and he will talk to you!"
    )
    # commands panel interval default bumped to 15.
    assert c.commands_panel_interval_minutes == 15
