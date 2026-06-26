"""Orchestrator wiring for the Twitch clickable panels (2026-06-26).

Two panels are built by ``Orchestrator._start_twitch_panels`` and wired to the
live backends:

  * the MODERATION CONTROL PANEL -> ``_twitch_panel_mod_command`` maps a clicked
    action to the SAME write-sidecar moderation path the voice path uses
    (prepare -> auto-confirm for user actions; chat_settings for channel modes),
    and surfaces the fuzzy-username CONFIRM popup on an ambiguous match;
  * the dev TEST PANEL -> ``_twitch_test_inject`` fires each live pipeline DIRECTLY
    with a SYNTHETIC event (via the routers' new ``inject`` seam / the speak fns).

These are exercised as UNBOUND methods on a tiny fake ``self`` (no boot, no model,
no Tk) -- mirroring tests/twitch/test_team_speak_audio_routing.py. The matchers and
the summon handlers are tested too.
"""
from __future__ import annotations

import pytest

from kenning.pipeline.orchestrator import Orchestrator
from kenning.twitch.moderation_gui import match_moderation_panel_command
from kenning.twitch.test_panel import match_test_panel_command


# --------------------------------------------------------------------------- #
# Voice matchers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "open the moderation panel",
    "show me the moderation panel",
    "pull up the mod panel",
    "show the moderation control panel",
])
def test_moderation_panel_open_phrasings(text):
    assert match_moderation_panel_command(text) == "open"


@pytest.mark.parametrize("text", [
    "close the moderation panel",
    "hide the mod panel",
])
def test_moderation_panel_close_phrasings(text):
    assert match_moderation_panel_command(text) == "close"


@pytest.mark.parametrize("text", [
    "what's on the moderation panel",     # question
    "rotate to a site",                   # unrelated
    "ban that guy",                       # a mod command, not the panel
    "the moderation panel is really useful for streamers and i love it a lot",  # too long
])
def test_moderation_panel_non_matches(text):
    assert match_moderation_panel_command(text) is None


@pytest.mark.parametrize("text", [
    "show me the test panel",
    "open the test panel",
    "pull up the dev panel",
])
def test_test_panel_open_phrasings(text):
    assert match_test_panel_command(text) == "open"


@pytest.mark.parametrize("text", [
    "close the test panel",
    "hide the dev panel",
])
def test_test_panel_close_phrasings(text):
    assert match_test_panel_command(text) == "close"


@pytest.mark.parametrize("text", [
    "is the test panel open",
    "rush b with the smokes",
])
def test_test_panel_non_matches(text):
    assert match_test_panel_command(text) is None


# --------------------------------------------------------------------------- #
# Moderation control panel -> on_command mapping
# --------------------------------------------------------------------------- #
class _FakeRemote:
    """Records prepare/confirm/chat_settings calls; returns a prepared token."""

    def __init__(self, *, prepare_result=None):
        self.prepares = []
        self.confirms = []
        self.chat_settings_calls = []
        self._prepare_result = prepare_result or {"ok": True, "token": "T1",
                                                  "action": "ban", "target": "alice"}

    def prepare(self, text):
        self.prepares.append(text)
        return dict(self._prepare_result)

    def confirm(self, token):
        self.confirms.append(token)
        return {"ok": True}

    def chat_settings(self, text):
        self.chat_settings_calls.append(text)
        return {"ok": True, "readback": "Done."}


def _make_orch():
    """A real Orchestrator instance WITHOUT __init__ so its bound methods are
    available, but no boot / model / audio. Tests set only the attrs they need."""
    return Orchestrator.__new__(Orchestrator)


def _ModOrch(remote, *, confirm=None):
    orch = _make_orch()
    orch._twitch_mod_remote = remote
    orch._twitch_mod_confirm = confirm
    return orch


def _mod_cmd(orch, action, **kw):
    orch._twitch_panel_mod_command(action, **kw)


def test_panel_ban_prepares_and_auto_confirms():
    remote = _FakeRemote(prepare_result={"ok": True, "token": "TK", "action": "ban",
                                         "target": "alice"})
    orch = _ModOrch(remote)
    _mod_cmd(orch, "ban", user="alice")
    assert remote.prepares == ["ban alice"]
    assert remote.confirms == ["TK"]          # a click IS the confirmation


def test_panel_timeout_builds_seconds_text():
    remote = _FakeRemote()
    orch = _ModOrch(remote)
    _mod_cmd(orch, "timeout", user="bob", seconds=120)
    assert remote.prepares == ["timeout bob for 120 seconds"]
    assert remote.confirms  # auto-confirmed


def test_panel_channel_mode_uses_chat_settings_not_prepare():
    remote = _FakeRemote()
    orch = _ModOrch(remote)
    _mod_cmd(orch, "slow_mode", seconds=30, enabled=True)
    assert remote.chat_settings_calls == ["slow mode 30 seconds"]
    assert remote.prepares == []              # channel modes never two-phase


def test_panel_followers_minutes_mapping():
    remote = _FakeRemote()
    orch = _ModOrch(remote)
    _mod_cmd(orch, "followers_only", seconds=5, enabled=True)
    assert remote.chat_settings_calls == ["followers only 5 minutes"]


def test_panel_clear_chat():
    remote = _FakeRemote()
    orch = _ModOrch(remote)
    _mod_cmd(orch, "clear_chat", enabled=True)
    assert remote.chat_settings_calls == ["clear chat"]


def test_panel_no_remote_is_a_safe_noop():
    orch = _ModOrch(None)
    # Must not raise.
    _mod_cmd(orch, "ban", user="alice")


def test_panel_ambiguous_surfaces_confirm_popup():
    # prepare() reports an ambiguous fuzzy match -> the confirm popup is prompted,
    # and YES re-prepares against the chosen login then confirms.
    class _AmbiguousRemote(_FakeRemote):
        def __init__(self):
            super().__init__()
            self._first = True

        def prepare(self, text):
            self.prepares.append(text)
            if self._first:
                self._first = False
                return {"ok": False, "reason_blocked": "ambiguous",
                        "action": "ban",
                        "candidates": [{"login": "alice99"}, {"login": "alicia"}]}
            return {"ok": True, "token": "TK2", "action": "ban", "target": "alice99"}

    class _Confirm:
        available = True

        def __init__(self):
            self.prompts = []

        def prompt(self, action, best, alts, on_result):
            self.prompts.append((action, best, list(alts)))
            on_result("yes")            # streamer clicks YES on the best match

    remote = _AmbiguousRemote()
    confirm = _Confirm()
    orch = _ModOrch(remote, confirm=confirm)
    _mod_cmd(orch, "ban", user="alic")
    assert confirm.prompts and confirm.prompts[0][1] == "alice99"
    # YES -> re-prepared against the exact login + confirmed.
    assert "ban alice99" in remote.prepares
    assert remote.confirms == ["TK2"]


# --------------------------------------------------------------------------- #
# Test panel -> on_test synthetic-event injection
# --------------------------------------------------------------------------- #
class _Recorder:
    def __init__(self):
        self.injected = []

    def inject(self, ev):
        self.injected.append(ev)


def _TestOrch():
    orch = _make_orch()
    orch._twitch_redeem_router = _Recorder()
    orch._twitch_chat_game_router = _Recorder()
    orch._twitch_raid_handler = _Recorder()
    orch.spoke = []
    orch.team = []
    # The speak/team seams the dispatch calls (bound here, not the real ones).
    orch._twitch_speak_and_post = orch.spoke.append
    orch._twitch_team_speak = lambda framed, rs: orch.team.append(framed)
    return orch


def _inject(orch, action, **kw):
    orch._twitch_test_inject(action, **kw)


def test_test_panel_speak_say_uses_speak_and_post():
    orch = _TestOrch()
    _inject(orch, "speak_say", message="hello chat")
    orch._twitch_test_speak_thread.join(timeout=2)   # speak runs on a daemon thread
    assert orch.spoke and "hello chat" in orch.spoke[0]


def test_test_panel_speak_team_frames_and_team_speaks():
    orch = _TestOrch()
    _inject(orch, "speak_team", message="rotate A")
    orch._twitch_test_speak_thread.join(timeout=2)   # team-speak runs on a daemon thread
    assert orch.team and "rotate A" in orch.team[0]


def test_test_panel_raid_injects_event():
    orch = _TestOrch()
    _inject(orch, "raid", login="raider1", viewers=50)
    ev = orch._twitch_raid_handler.injected[0]
    assert ev["type"] == "raid" and ev["from_login"] == "raider1" and ev["viewers"] == 50


def test_test_panel_chat_slots_injects_command():
    orch = _TestOrch()
    _inject(orch, "chat_slots", command="slots", bet=250)
    ev = orch._twitch_chat_game_router.injected[0]
    assert ev["type"] == "chat" and ev["text"] == "!slots 250"


def test_test_panel_chat_duel_injects_target_and_bet():
    orch = _TestOrch()
    _inject(orch, "chat_duel", command="duel", target="rival", bet=100)
    ev = orch._twitch_chat_game_router.injected[0]
    assert ev["text"] == "!duel @rival 100"


def test_test_panel_chat_give_injects_target_and_amount():
    orch = _TestOrch()
    _inject(orch, "chat_give", command="give", target="friend", amount=300)
    ev = orch._twitch_chat_game_router.injected[0]
    assert ev["text"] == "!give @friend 300"


def test_test_panel_simple_chat_command():
    orch = _TestOrch()
    _inject(orch, "chat_wheel", command="wheel")
    ev = orch._twitch_chat_game_router.injected[0]
    assert ev["text"] == "!wheel"


def test_test_panel_redeem_injects_reward_title():
    orch = _TestOrch()
    _inject(orch, "redeem_slots")
    ev = orch._twitch_redeem_router.injected[0]
    assert ev["type"] == "redeem" and ev["reward_title"] == "Slots"


def test_test_panel_redeem_wheel_title():
    orch = _TestOrch()
    _inject(orch, "redeem_wheel")
    ev = orch._twitch_redeem_router.injected[0]
    assert ev["reward_title"] == "Spin the Wheel"


def test_test_panel_inject_is_fail_open_with_no_routers():
    class _Bare:
        pass
    # No router attributes at all -> must not raise.
    Orchestrator._twitch_test_inject(_Bare(), "chat_slots", command="slots", bet=10)
    Orchestrator._twitch_test_inject(_Bare(), "redeem_slots")
    Orchestrator._twitch_test_inject(_Bare(), "raid", login="x", viewers=1)


# --------------------------------------------------------------------------- #
# Voice summon handlers
# --------------------------------------------------------------------------- #
class _SummonOrch:
    def __init__(self, mod_panel=None, test_panel=None):
        self._twitch_mod_panel = mod_panel
        self._twitch_test_panel = test_panel
        self.said = []

    def _speak(self, text):
        self.said.append(text)


class _PanelStub:
    def __init__(self):
        self.shown = 0
        self.hidden = 0

    def show(self):
        self.shown += 1

    def hide(self):
        self.hidden += 1


def test_summon_moderation_panel_open_close():
    panel = _PanelStub()
    orch = _SummonOrch(mod_panel=panel)
    assert Orchestrator._maybe_handle_moderation_panel(orch, "open the moderation panel") is True
    assert panel.shown == 1
    assert Orchestrator._maybe_handle_moderation_panel(orch, "close the moderation panel") is True
    assert panel.hidden == 1
    # non-match falls through.
    assert Orchestrator._maybe_handle_moderation_panel(orch, "rush b") is False


def test_summon_test_panel_open_close():
    panel = _PanelStub()
    orch = _SummonOrch(test_panel=panel)
    assert Orchestrator._maybe_handle_test_panel(orch, "show me the test panel") is True
    assert panel.shown == 1
    assert Orchestrator._maybe_handle_test_panel(orch, "hide the test panel") is True
    assert panel.hidden == 1


def test_summon_handles_missing_panel():
    orch = _SummonOrch(mod_panel=None)
    # Consumes the command (True) + speaks an unavailable line.
    assert Orchestrator._maybe_handle_moderation_panel(orch, "open the moderation panel") is True
    assert orch.said and "available" in orch.said[0].lower()
