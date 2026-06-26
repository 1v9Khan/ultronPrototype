"""Tests for the Twitch dev TEST PANEL window.

These run HEADLESS-SAFE and deterministically on any box: no REAL Tk window is
built. Matching ``tests/twitch/test_moderation_gui.py``, the window's daemon UI
thread is either forced off via the ``KENNING_TEST_PANEL_HEADLESS`` env flag (the
no-op / fail-open path) or the validation + emit logic is driven directly with a
stubbed ``_vars`` dict so the ``available=True`` code path runs without ever
creating a Tk root.

Two contracts under test:
  1. Import, construction and every public method (``show`` / ``hide`` /
     ``toggle`` / ``close``) NEVER raise -- whether or not a display exists.
  2. Activating each control invokes ``on_test`` with the CORRECT action string +
     parsed params; input validation no-ops (empty message / blank login / bad
     bet) suppress the callback; a missing / throwing ``on_test`` never escapes.
"""

from __future__ import annotations

import time

import pytest

import kenning.twitch.test_panel as tp
from kenning.twitch.test_panel import TestPanel, make_test_panel


@pytest.fixture(autouse=True)
def _force_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the fail-open / no-window path for every test so no real Tk root is
    created -- identical behavior with or without a display, and no leaked Tk
    thread into the rest of the sweep."""
    monkeypatch.setenv(tp._HEADLESS_ENV, "1")
    monkeypatch.setenv(tp._MOD_HEADLESS_ENV, "1")


def _settle() -> None:
    time.sleep(0.01)


def _panel_with_recorder(monkeypatch: pytest.MonkeyPatch):
    """Build a panel whose ``available`` is forced True (UI loop neutered) and a
    recorder list capturing every ``on_test`` call as ``(action, params)``.

    The control methods read :attr:`_vars`; tests assign plain string values
    there (``_read_var`` accepts a plain value or a tk StringVar)."""
    monkeypatch.setattr(TestPanel, "_probe_tk_available",
                        staticmethod(lambda: True))
    monkeypatch.setattr(TestPanel, "_ui_loop", lambda self: None)
    calls: list[tuple[str, dict]] = []
    panel = TestPanel(lambda action, **params: calls.append((action, params)))
    assert panel.available is True
    return panel, calls


# ---------------------------------------------------------------------------
# Contract 1: never-raise + fail-open
# ---------------------------------------------------------------------------

def test_import_and_construct_never_raise() -> None:
    panel = TestPanel(lambda *a, **k: None)
    assert isinstance(panel.available, bool)
    assert panel.available is False  # forced headless
    assert panel.shown is False
    panel.close()


def test_public_methods_are_noops_when_headless() -> None:
    panel = TestPanel(lambda *a, **k: None)
    panel.show()
    panel.toggle()
    panel.hide()
    _settle()
    assert panel.shown is False
    panel.close()
    assert panel.shown is False


def test_factory_always_returns_panel() -> None:
    panel = make_test_panel(lambda *a, **k: None)
    assert isinstance(panel, TestPanel)
    assert panel.available is False
    panel.show()
    panel.hide()
    panel.close()


def test_factory_tolerates_non_callable() -> None:
    panel = make_test_panel(None)  # type: ignore[arg-type]
    assert panel._on_test is None
    # Emitting with no callback is a logged no-op, never raises.
    panel._emit("redeem_wheel")
    panel.close()


def test_available_path_show_hide_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend a display exists, but neuter the real Tk loop so no root is built.
    monkeypatch.setattr(TestPanel, "_probe_tk_available",
                        staticmethod(lambda: True))
    monkeypatch.setattr(TestPanel, "_ui_loop", lambda self: None)
    panel = TestPanel(lambda *a, **k: None)
    assert panel.available is True
    panel.show()      # starts the (stubbed) UI thread + enqueues a raise
    panel.toggle()
    panel.hide()
    panel._drain_requests()  # the stubbed loop never drains; do it here
    panel.close()


# ---------------------------------------------------------------------------
# Contract 2: each control fires on_test with the right action + params
# ---------------------------------------------------------------------------

def test_speak_say(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["say_message"] = "  hello chat  "
    assert panel._emit_speak(tp.ACT_SPEAK_SAY) is True
    assert calls == [("speak_say", {"message": "hello chat"})]


def test_speak_team(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["team_message"] = "rush B now"
    assert panel._emit_speak(tp.ACT_SPEAK_TEAM) is True
    assert calls == [("speak_team", {"message": "rush B now"})]


def test_speak_empty_message_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["say_message"] = "   "
    assert panel._emit_speak(tp.ACT_SPEAK_SAY) is False
    panel._vars["team_message"] = ""
    assert panel._emit_speak(tp.ACT_SPEAK_TEAM) is False
    assert calls == []


def test_raid(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["raid_login"] = "@CoolStreamer"
    panel._vars["raid_viewers"] = "150"
    assert panel._emit_raid() is True
    assert calls == [("raid", {"login": "coolstreamer", "viewers": 150})]


def test_raid_blank_login_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["raid_login"] = "  "
    panel._vars["raid_viewers"] = "10"
    assert panel._emit_raid() is False
    assert calls == []


def test_raid_blank_viewers_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["raid_login"] = "raider"
    panel._vars["raid_viewers"] = ""  # blank -> default
    assert panel._emit_raid() is True
    action, params = calls[0]
    assert action == "raid"
    assert params == {"login": "raider", "viewers": panel._default_viewers}


def test_raid_negative_viewers_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["raid_login"] = "raider"
    panel._vars["raid_viewers"] = "-5"
    assert panel._emit_raid() is True
    assert calls[0][1]["viewers"] == 0


def test_chat_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["slots_bet"] = "250"
    assert panel._emit_chat_bet(tp.ACT_CHAT_SLOTS, "slots") is True
    assert calls == [("chat_slots", {"command": "slots", "bet": 250})]


def test_chat_heist(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["heist_bet"] = "300"
    assert panel._emit_chat_bet(tp.ACT_CHAT_HEIST, "heist") is True
    assert calls == [("chat_heist", {"command": "heist", "bet": 300})]


def test_chat_bet_blank_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["slots_bet"] = ""  # blank -> default bet
    assert panel._emit_chat_bet(tp.ACT_CHAT_SLOTS, "slots") is True
    assert calls[0][1]["bet"] == panel._default_bet


def test_chat_bet_nonpositive_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["slots_bet"] = "0"
    assert panel._emit_chat_bet(tp.ACT_CHAT_SLOTS, "slots") is False
    panel._vars["heist_bet"] = "-10"
    assert panel._emit_chat_bet(tp.ACT_CHAT_HEIST, "heist") is False
    assert calls == []


def test_chat_duel(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["duel_target"] = "@Rival"
    panel._vars["duel_bet"] = "500"
    assert panel._emit_chat_duel() is True
    assert calls == [
        ("chat_duel", {"command": "duel", "target": "rival", "bet": 500})
    ]


def test_chat_duel_no_target_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["duel_target"] = ""
    panel._vars["duel_bet"] = "100"
    assert panel._emit_chat_duel() is False
    assert calls == []


def test_chat_duel_bad_bet_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["duel_target"] = "rival"
    panel._vars["duel_bet"] = "0"
    assert panel._emit_chat_duel() is False
    assert calls == []


def test_chat_give(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["give_target"] = "@Friend"
    panel._vars["give_amount"] = "42"
    assert panel._emit_chat_give() is True
    assert calls == [
        ("chat_give", {"command": "give", "target": "friend", "amount": 42})
    ]


def test_chat_give_no_target_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["give_target"] = "  "
    panel._vars["give_amount"] = "42"
    assert panel._emit_chat_give() is False
    assert calls == []


def test_chat_give_bad_amount_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["give_target"] = "friend"
    panel._vars["give_amount"] = "-1"
    assert panel._emit_chat_give() is False
    assert calls == []


@pytest.mark.parametrize(
    "action,command",
    [
        (tp.ACT_CHAT_WHEEL, "wheel"),
        (tp.ACT_CHAT_LEADERBOARD, "leaderboard"),
        (tp.ACT_CHAT_TRIVIA, "trivia"),
        (tp.ACT_CHAT_RAFFLE, "raffle"),
        (tp.ACT_CHAT_ULTRON, "ultron"),
        (tp.ACT_CHAT_HELP, "help"),
    ],
)
def test_chat_simple_commands(monkeypatch: pytest.MonkeyPatch,
                              action: str, command: str) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    assert panel._emit_chat_simple(action, command) is True
    assert calls == [(action, {"command": command})]


@pytest.mark.parametrize(
    "action",
    [tp.ACT_REDEEM_WHEEL, tp.ACT_REDEEM_SLOTS,
     tp.ACT_REDEEM_HEIST, tp.ACT_REDEEM_DUEL],
)
def test_redeem_games(monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    assert panel._emit_redeem(action) is True
    assert calls == [(action, {})]  # no params


def test_chat_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["reply_message"] = "  what agent should I play?  "
    assert panel._emit_chat_reply() is True
    assert calls == [("chat_reply", {"message": "what agent should I play?"})]


def test_chat_reply_empty_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    panel._vars["reply_message"] = ""
    assert panel._emit_chat_reply() is False
    assert calls == []


@pytest.mark.parametrize(
    "action",
    [tp.ACT_AUTO_TRIVIA, tp.ACT_COMMANDS_PANEL, tp.ACT_TALK_HINT],
)
def test_extras(monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    panel, calls = _panel_with_recorder(monkeypatch)
    assert panel._emit_extra(action) is True
    assert calls == [(action, {})]


# ---------------------------------------------------------------------------
# Emit-level robustness
# ---------------------------------------------------------------------------

def test_emit_with_no_callback_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TestPanel, "_probe_tk_available",
                        staticmethod(lambda: True))
    monkeypatch.setattr(TestPanel, "_ui_loop", lambda self: None)
    panel = TestPanel(None)  # type: ignore[arg-type]
    assert panel._on_test is None
    # Every emit path must tolerate a missing callback without raising.
    panel._vars["say_message"] = "hi"
    assert panel._emit_speak(tp.ACT_SPEAK_SAY) is True  # validated + 'fired'
    panel._vars["raid_login"] = "r"
    assert panel._emit_raid() is True
    assert panel._emit_redeem(tp.ACT_REDEEM_WHEEL) is True


def test_emit_swallows_throwing_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TestPanel, "_probe_tk_available",
                        staticmethod(lambda: True))
    monkeypatch.setattr(TestPanel, "_ui_loop", lambda self: None)

    def _boom(*_a, **_k):
        raise RuntimeError("kaboom")

    panel = TestPanel(_boom)
    panel._vars["say_message"] = "hi"
    # The throwing backend must never propagate out of a click handler.
    assert panel._emit_speak(tp.ACT_SPEAK_SAY) is True
    panel._emit(tp.ACT_REDEEM_SLOTS)  # direct emit also swallows


def test_read_int_parses_embedded_number(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, _ = _panel_with_recorder(monkeypatch)
    panel._vars["k"] = "20 viewers"
    assert panel._read_int("k", default=99) == 20
    panel._vars["k"] = "junk"
    assert panel._read_int("k", default=99) == 99
    panel._vars["k"] = ""
    assert panel._read_int("k", default=7) == 7


def test_read_login_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, _ = _panel_with_recorder(monkeypatch)
    panel._vars["k"] = "  @MixedCase  "
    assert panel._read_login("k") == "mixedcase"
    panel._vars["k"] = ""
    assert panel._read_login("k") == ""


def test_read_var_with_tk_stringvar_like(monkeypatch: pytest.MonkeyPatch) -> None:
    panel, _ = _panel_with_recorder(monkeypatch)

    class _FakeVar:
        def get(self) -> str:
            return "from-getter"

    panel._vars["k"] = _FakeVar()
    assert panel._read_var("k") == "from-getter"
    # A getter that throws degrades to "".
    class _BadVar:
        def get(self):
            raise RuntimeError("no")

    panel._vars["k"] = _BadVar()
    assert panel._read_var("k") == ""


def test_label_for_covers_every_action() -> None:
    actions = [
        tp.ACT_SPEAK_SAY, tp.ACT_SPEAK_TEAM, tp.ACT_RAID,
        tp.ACT_CHAT_SLOTS, tp.ACT_CHAT_WHEEL, tp.ACT_CHAT_HEIST,
        tp.ACT_CHAT_DUEL, tp.ACT_CHAT_GIVE, tp.ACT_CHAT_LEADERBOARD,
        tp.ACT_CHAT_TRIVIA, tp.ACT_CHAT_RAFFLE, tp.ACT_CHAT_ULTRON,
        tp.ACT_CHAT_HELP, tp.ACT_REDEEM_WHEEL, tp.ACT_REDEEM_SLOTS,
        tp.ACT_REDEEM_HEIST, tp.ACT_REDEEM_DUEL, tp.ACT_CHAT_REPLY,
        tp.ACT_AUTO_TRIVIA, tp.ACT_COMMANDS_PANEL, tp.ACT_TALK_HINT,
    ]
    for a in actions:
        label = TestPanel._label_for(a)
        assert isinstance(label, str)
        assert label.strip()  # every action maps to a non-empty human label
