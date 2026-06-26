"""Tests for the Twitch mod-action confirmation window.

These run HEADLESS-SAFE and deterministically on any box: no REAL Tk window is
built. Matching the rest of the suite (e.g. ``tests/audio/test_log_viewer.py``,
which never spins up the real overlay), the window's daemon UI thread is either
forced off via the ``KENNING_MOD_GUI_HEADLESS`` env flag (the no-op / fail-open
path) or stubbed so the ``available=True`` code path runs without ever creating
a Tk root. Rapidly creating + destroying many real Tk interpreters in one
process is fragile on Windows ('Tcl_AsyncDelete: ... wrong thread') and is NOT
what this contract is about.

The contract under test: import, construction and every public method
(``prompt`` / ``update_match`` / ``hide`` / ``close``) NEVER raise -- whether or
not a display exists.
"""

from __future__ import annotations

import time

import pytest

import kenning.twitch.moderation_gui as mod_gui
from kenning.twitch.moderation_gui import (
    ModerationConfirmGUI,
    ModerationControlPanel,
    make_control_panel,
)


@pytest.fixture(autouse=True)
def _force_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the fail-open / no-window path for every test so no real Tk root is
    created. This makes the suite pass identically whether or not a display
    exists, and never leaks a Tk thread into the rest of the sweep."""
    monkeypatch.setenv(mod_gui._HEADLESS_ENV, "1")


def _settle() -> None:
    # Even in the headless path nothing blocks, but a tiny settle keeps the
    # tests structurally identical to the with-display flow.
    time.sleep(0.01)


def test_import_and_construct_never_raise() -> None:
    gui = ModerationConfirmGUI()
    assert isinstance(gui.available, bool)
    assert gui.available is False  # forced headless
    assert gui.shown is False
    gui.close()


def test_prompt_never_raises() -> None:
    gui = ModerationConfirmGUI()
    results: list[str] = []
    gui.prompt(
        "TIMEOUT 10m",
        "cool_viewer_42",
        ["cool_viewer", "kool_viewer_42", "coolviewer42"],
        lambda r: results.append(r),
    )
    _settle()
    gui.close()


def test_update_match_never_raises() -> None:
    gui = ModerationConfirmGUI()
    gui.prompt("BAN", "spammer", ["spammer1", "spammer_x"], lambda _r: None)
    _settle()
    gui.update_match("spammer_x", ["spammerx", "the_spammer"])
    _settle()
    gui.close()


def test_hide_never_raises() -> None:
    gui = ModerationConfirmGUI()
    gui.hide()  # before any prompt
    gui.prompt("UNBAN", "redeemed_user", [], lambda _r: None)
    _settle()
    gui.hide()
    _settle()
    gui.close()


def test_all_action_headers_accepted() -> None:
    gui = ModerationConfirmGUI()
    for action in ("TIMEOUT 10m", "BAN", "UNBAN", "UNTIMEOUT",
                   "DELETE LAST MSG"):
        gui.prompt(action, "someuser", ["alt1", "alt2"], lambda _r: None)
        _settle()
    gui.close()


def test_methods_tolerate_none_and_empty_inputs() -> None:
    gui = ModerationConfirmGUI()
    gui.prompt("", "", [], lambda _r: None)
    _settle()
    gui.update_match("", [])
    _settle()
    # A non-callable on_result must still be tolerated (stored as no callback).
    gui.prompt("BAN", "x", ["y"], None)  # type: ignore[arg-type]
    _settle()
    gui.hide()
    gui.close()


def test_double_construction_is_independent() -> None:
    a = ModerationConfirmGUI(width=300, height=200)
    b = ModerationConfirmGUI(width=420, height=320)
    a.prompt("BAN", "u1", ["a"], lambda _r: None)
    b.prompt("UNBAN", "u2", ["b"], lambda _r: None)
    _settle()
    a.close()
    b.close()


def test_headless_methods_are_true_noops() -> None:
    # Forced headless: no UI thread is ever started, regardless of the calls.
    gui = ModerationConfirmGUI()
    gui.prompt("BAN", "u", ["a", "b"], lambda _r: None)
    gui.update_match("u2", ["c"])
    gui.hide()
    _settle()
    assert gui.shown is False
    gui.close()
    assert gui.shown is False


# ---------------------------------------------------------------------------
# available=True code path -- exercised WITHOUT a real Tk root by stubbing the
# UI thread target. Proves prompt/update_match/hide/_fire never raise when a
# window WOULD be built, while staying deterministic + headless-safe.
# ---------------------------------------------------------------------------


def test_available_path_methods_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend a display exists, but neuter the real Tk loop so no root is built.
    monkeypatch.setattr(ModerationConfirmGUI, "_probe_tk_available",
                        staticmethod(lambda: True))
    monkeypatch.setattr(ModerationConfirmGUI, "_ui_loop", lambda self: None)

    gui = ModerationConfirmGUI()
    assert gui.available is True
    got: list[str] = []
    # prompt() starts the (stubbed) UI thread and enqueues a render request.
    gui.prompt("TIMEOUT 10m", "matched_user",
               ["alt_a", "alt_b"], lambda r: got.append(r))
    gui.update_match("matched_user2", ["alt_c"])
    gui.hide()
    # Drain the queued requests on THIS thread (the stubbed loop never does);
    # the render path no-ops gracefully because no widgets were built.
    gui._drain_requests()
    gui.close()


def test_fire_emits_result_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # The click handler's contract: emit exactly once, suppress a re-fire, and
    # never raise even if the callback throws.
    monkeypatch.setattr(ModerationConfirmGUI, "_probe_tk_available",
                        staticmethod(lambda: True))
    monkeypatch.setattr(ModerationConfirmGUI, "_ui_loop", lambda self: None)

    gui = ModerationConfirmGUI()
    got: list[str] = []
    gui._on_result = lambda r: got.append(r)
    gui._result_sent = False
    gui._fire("yes")
    gui._fire("no")  # suppressed -- a result was already sent
    assert got == ["yes"]

    # An invalid token is ignored.
    gui._result_sent = False
    got.clear()
    gui._fire("bogus")
    assert got == []

    # A throwing callback must not propagate out of the click handler.
    def _boom(_r: str) -> None:
        raise RuntimeError("kaboom")

    gui._on_result = _boom
    gui._result_sent = False
    gui._fire("cancel")  # must not raise
    gui.close()


# ===========================================================================
# ModerationControlPanel -- the click-to-moderate sidebar.
#
# Like the confirm-window tests above, these run HEADLESS-SAFE: no REAL Tk root
# is built. The autouse ``_force_headless`` fixture forces the fail-open path,
# and the command-emit / validation LOGIC is driven DIRECTLY (we populate the
# panel's ``_vars`` dict and call the pure ``_emit_*`` handlers), proving each
# control invokes ``on_command`` with the right action / user / seconds /
# enabled -- without ever creating a Tk widget or simulating a real click.
# ===========================================================================


def _record_panel(**kwargs):
    """Build a headless panel whose on_command appends every call to a list."""
    calls: list[tuple] = []

    def _on_command(action: str, *, user: str = "", seconds: int = 0,
                    enabled: bool = True) -> None:
        calls.append((action, user, seconds, enabled))

    panel = ModerationControlPanel(_on_command, **kwargs)
    return panel, calls


def test_panel_import_and_construct_never_raise() -> None:
    panel, _calls = _record_panel()
    assert isinstance(panel.available, bool)
    assert panel.available is False  # forced headless
    assert panel.shown is False
    panel.close()


def test_panel_factory_returns_inert_when_headless() -> None:
    panel = make_control_panel(lambda *a, **k: None)
    assert panel.available is False
    assert panel.shown is False
    # Every public method is a safe no-op.
    panel.show()
    panel.hide()
    panel.toggle()
    panel.close()
    assert panel.shown is False


def test_panel_show_hide_toggle_close_never_raise() -> None:
    panel, _calls = _record_panel()
    panel.show()
    panel.hide()
    panel.toggle()
    panel.close()
    panel.close()  # idempotent


def test_ban_emits_action_with_user() -> None:
    panel, calls = _record_panel()
    panel._vars["ban_user"] = "spammer42"
    assert panel._emit_user_action(mod_gui._ACT_BAN) is True
    assert calls == [("ban", "spammer42", 0, True)]
    panel.close()


def test_unban_untimeout_delete_emit_user_only() -> None:
    panel, calls = _record_panel()
    panel._vars["unban_user"] = "redeemed"
    panel._vars["untimeout_user"] = "freed"
    panel._vars["delete_message_user"] = "noisy"
    assert panel._emit_user_action(mod_gui._ACT_UNBAN) is True
    assert panel._emit_user_action(mod_gui._ACT_UNTIMEOUT) is True
    assert panel._emit_user_action(mod_gui._ACT_DELETE) is True
    assert calls == [
        ("unban", "redeemed", 0, True),
        ("untimeout", "freed", 0, True),
        ("delete_message", "noisy", 0, True),
    ]
    panel.close()


def test_timeout_parses_seconds_field() -> None:
    panel, calls = _record_panel()
    panel._vars["timeout_user"] = "rude_guy"
    panel._vars["timeout_seconds"] = "300"
    assert panel._emit_user_action(mod_gui._ACT_TIMEOUT, default_seconds=600) is True
    assert calls == [("timeout", "rude_guy", 300, True)]
    panel.close()


def test_timeout_blank_seconds_uses_default() -> None:
    panel, calls = _record_panel()
    panel._vars["timeout_user"] = "rude_guy"
    panel._vars["timeout_seconds"] = ""  # blank -> the wired default
    assert panel._emit_user_action(mod_gui._ACT_TIMEOUT, default_seconds=600) is True
    assert calls == [("timeout", "rude_guy", 600, True)]
    panel.close()


def test_timeout_extracts_leading_int_from_noisy_field() -> None:
    # A field like "600s" / "10 min" still parses the leading integer.
    panel, calls = _record_panel()
    panel._vars["timeout_user"] = "x"
    panel._vars["timeout_seconds"] = "120 sec"
    assert panel._emit_user_action(mod_gui._ACT_TIMEOUT, default_seconds=600) is True
    assert calls == [("timeout", "x", 120, True)]
    panel.close()


def test_timeout_nonpositive_seconds_is_noop() -> None:
    panel, calls = _record_panel()
    panel._vars["timeout_user"] = "x"
    panel._vars["timeout_seconds"] = "0"
    assert panel._emit_user_action(mod_gui._ACT_TIMEOUT, default_seconds=600) is False
    assert calls == []  # validation no-op: nothing fired
    panel.close()


def test_empty_username_is_validation_noop() -> None:
    panel, calls = _record_panel()
    # No username set at all.
    assert panel._emit_user_action(mod_gui._ACT_BAN) is False
    # Whitespace-only username.
    panel._vars["timeout_user"] = "   "
    assert panel._emit_user_action(mod_gui._ACT_TIMEOUT) is False
    assert calls == []  # nothing fired
    panel.close()


def test_clear_chat_emits_channel_action() -> None:
    panel, calls = _record_panel()
    assert panel._emit_channel_action(mod_gui._ACT_CLEAR_CHAT, enabled=True) is True
    assert calls == [("clear_chat", "", 0, True)]
    panel.close()


def test_slow_mode_on_carries_seconds_off_does_not() -> None:
    panel, calls = _record_panel()
    panel._vars["slow_seconds"] = "45"
    assert panel._emit_channel_action(mod_gui._ACT_SLOW, enabled=True) is True
    assert panel._emit_channel_action(mod_gui._ACT_SLOW, enabled=False) is True
    assert calls == [
        ("slow_mode", "", 45, True),
        ("slow_mode", "", 0, False),
    ]
    panel.close()


def test_slow_mode_blank_field_uses_default() -> None:
    panel, calls = _record_panel(default_slow_seconds=30)
    panel._vars["slow_seconds"] = ""
    assert panel._emit_channel_action(mod_gui._ACT_SLOW, enabled=True) is True
    assert calls == [("slow_mode", "", 30, True)]
    panel.close()


def test_followers_only_carries_minutes_as_seconds_value() -> None:
    panel, calls = _record_panel()
    panel._vars["followers_minutes"] = "10"
    assert panel._emit_channel_action(mod_gui._ACT_FOLLOWERS, enabled=True) is True
    # The numeric value (a MINUTE count) rides on the ``seconds`` slot; the
    # orchestrator maps it to follower_mode_duration (minutes).
    assert calls == [("followers_only", "", 10, True)]
    panel.close()


def test_followers_only_off_ignores_field() -> None:
    panel, calls = _record_panel()
    panel._vars["followers_minutes"] = "10"
    assert panel._emit_channel_action(mod_gui._ACT_FOLLOWERS, enabled=False) is True
    assert calls == [("followers_only", "", 0, False)]
    panel.close()


def test_boolean_only_toggles_emit_enabled() -> None:
    panel, calls = _record_panel()
    for action in (mod_gui._ACT_SUBSCRIBERS, mod_gui._ACT_EMOTE, mod_gui._ACT_UNIQUE):
        assert panel._emit_channel_action(action, enabled=True) is True
        assert panel._emit_channel_action(action, enabled=False) is True
    assert calls == [
        ("subscribers_only", "", 0, True),
        ("subscribers_only", "", 0, False),
        ("emote_only", "", 0, True),
        ("emote_only", "", 0, False),
        ("unique_chat", "", 0, True),
        ("unique_chat", "", 0, False),
    ]
    panel.close()


def test_every_action_token_is_reachable() -> None:
    # Sanity: the user-targeted + channel-wide action sets cover exactly the
    # commands the panel exposes, and each fires its expected token.
    panel, calls = _record_panel()
    for action in mod_gui._USER_TARGETED_ACTIONS:
        panel._vars[f"{action}_user"] = "someone"
        if action == mod_gui._ACT_TIMEOUT:
            panel._vars[f"{action}_seconds"] = "60"
        assert panel._emit_user_action(action, default_seconds=600) is True
    fired_user = {c[0] for c in calls}
    assert fired_user == set(mod_gui._USER_TARGETED_ACTIONS)

    calls.clear()
    for action in mod_gui._CHANNEL_ACTIONS:
        assert panel._emit_channel_action(action, enabled=True) is True
    fired_channel = {c[0] for c in calls}
    assert fired_channel == set(mod_gui._CHANNEL_ACTIONS)
    panel.close()


def test_missing_on_command_never_raises() -> None:
    # A panel built with a non-callable on_command stores no callback and every
    # emit path is a safe no-op (fail-open).
    panel = ModerationControlPanel(None)  # type: ignore[arg-type]
    panel._vars["ban_user"] = "x"
    # Returns True (validation passed) but the missing callback is just dropped.
    assert panel._emit_user_action(mod_gui._ACT_BAN) is True
    assert panel._emit_channel_action(mod_gui._ACT_CLEAR_CHAT, enabled=True) is True
    panel.close()


def test_throwing_on_command_never_propagates() -> None:
    def _boom(action: str, *, user: str = "", seconds: int = 0,
             enabled: bool = True) -> None:
        raise RuntimeError("backend kaboom")

    panel = ModerationControlPanel(_boom)
    panel._vars["ban_user"] = "x"
    # The throwing backend must not propagate out of the emit handler.
    assert panel._emit_user_action(mod_gui._ACT_BAN) is True  # fired, swallowed
    assert panel._emit_channel_action(mod_gui._ACT_CLEAR_CHAT, enabled=True) is True
    panel.close()


def test_panel_can_own_confirm_popup() -> None:
    # with_confirm=True composes (but does not entangle) a ModerationConfirmGUI.
    panel, _calls = _record_panel(with_confirm=True)
    assert isinstance(panel.confirm, ModerationConfirmGUI)
    # The composed popup is also headless/fail-open and never raises.
    panel.confirm.prompt("BAN", "u", ["a"], lambda _r: None)
    panel.close()  # tears down both


def test_panel_confirm_can_be_injected() -> None:
    cfm = ModerationConfirmGUI()
    panel, _calls = _record_panel(confirm=cfm)
    assert panel.confirm is cfm
    panel.close()


def test_panel_without_confirm_has_none() -> None:
    panel, _calls = _record_panel()
    assert panel.confirm is None
    panel.close()


def test_panel_headless_methods_are_true_noops() -> None:
    panel, _calls = _record_panel()
    panel.show()
    panel.toggle()
    panel.hide()
    _settle()
    assert panel.shown is False
    panel.close()
    assert panel.shown is False


# ---------------------------------------------------------------------------
# available=True path -- exercised WITHOUT a real Tk root by stubbing the UI
# thread target, mirroring the confirm-window tests above. Proves
# show/hide/toggle never raise when a window WOULD be built.
# ---------------------------------------------------------------------------


def test_panel_available_path_methods_never_raise(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ModerationControlPanel, "_probe_tk_available",
                        staticmethod(lambda: True))
    monkeypatch.setattr(ModerationControlPanel, "_ui_loop", lambda self: None)

    panel, calls = _record_panel()
    assert panel.available is True
    panel.show()
    panel.toggle()
    panel.hide()
    # Drain the queued window requests on THIS thread (the stubbed loop never
    # does); they no-op gracefully because no root was built.
    panel._drain_requests()
    # The emit logic still works on the available path.
    panel._vars["ban_user"] = "x"
    assert panel._emit_user_action(mod_gui._ACT_BAN) is True
    assert calls == [("ban", "x", 0, True)]
    panel.close()
