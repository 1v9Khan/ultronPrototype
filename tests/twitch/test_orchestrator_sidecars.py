"""Orchestrator Twitch sidecar auto-spawn / reap / overlay wiring.

Uses a BARE ``Orchestrator.__new__`` instance (no heavy init) + monkeypatched
``subprocess.Popen`` / ``kill_process_tree`` so nothing real is launched. Asserts
the spawn env contract, idempotency, reaping, and the in-process overlay lifecycle.
"""
from __future__ import annotations

import pathlib
import subprocess

from kenning.config import TwitchConfig
from kenning.pipeline.orchestrator import Orchestrator

_ROOT = pathlib.Path(__file__).resolve().parents[2]


class _FakeProc:
    _n = 1000

    def __init__(self, cmd=None) -> None:
        _FakeProc._n += 1
        self.pid = _FakeProc._n
        self.cmd = cmd


def _patch_popen(monkeypatch):
    calls: list = []

    def fake_popen(cmd, **kw):
        calls.append((cmd, kw))
        return _FakeProc(cmd)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def test_spawns_only_read_by_default(monkeypatch) -> None:
    monkeypatch.chdir(_ROOT)  # so scripts/ resolves via os.path.abspath
    calls = _patch_popen(monkeypatch)
    orch = Orchestrator.__new__(Orchestrator)
    cfg = TwitchConfig(
        enabled=True,
        auth={"client_id": "abc", "broadcaster_login": "1v9khan", "bot_login": "ultron_kenning"},
        moderation={"voice_commands_enabled": False},  # isolate to just the read sidecar
    )
    orch._start_twitch_sidecars(cfg)

    assert len(calls) == 1                       # read only (no guard model, no write)
    cmd, kw = calls[0]
    assert cmd[1].endswith("twitch_read_sidecar.py")
    assert cmd[2] == "8773"
    assert kw["env"]["KENNING_TWITCH_PARENT_PID"]
    assert kw["env"]["KENNING_TWITCH_CLIENT_ID"] == "abc"
    assert kw["env"]["KENNING_TWITCH_BROADCASTER_LOGIN"] == "1v9khan"
    assert "cwd" in kw
    assert orch._twitch_sidecar_procs[0][0] == "twitch_read"

    # idempotent: a second call does NOT re-spawn
    orch._start_twitch_sidecars(cfg)
    assert len(calls) == 1


def test_spawns_guard_when_model_present(monkeypatch) -> None:
    monkeypatch.chdir(_ROOT)
    calls = _patch_popen(monkeypatch)
    orch = Orchestrator.__new__(Orchestrator)
    # point guard_model_path at a real existing file so os.path.exists passes
    guard_file = str(_ROOT / "scripts" / "twitch_guard_sidecar.py")
    cfg = TwitchConfig(enabled=True, safety={"guard_model_path": guard_file})
    orch._start_twitch_sidecars(cfg)

    roles = [r for r, _ in orch._twitch_sidecar_procs]
    assert roles == ["twitch_read", "twitch_guard"]
    guard_call = next(c for c in calls if c[0][1].endswith("twitch_guard_sidecar.py"))
    assert guard_call[0][2] == "8774"
    assert guard_call[1]["env"]["KENNING_TWITCH_GUARD_MODEL"] == guard_file


def test_kill_reaps_all_tracked_sidecars(monkeypatch) -> None:
    from kenning.subprocess import kill_tree

    killed: list = []

    class _Res:
        total_killed = 1

    monkeypatch.setattr(kill_tree, "kill_process_tree",
                        lambda pid, **kw: (killed.append(pid), _Res())[1])
    orch = Orchestrator.__new__(Orchestrator)
    orch._zombie_killer = None
    orch._twitch_sidecar_procs = [("twitch_read", _FakeProc()), ("twitch_guard", _FakeProc())]
    orch._kill_twitch_sidecars()

    assert len(killed) == 2
    assert orch._twitch_sidecar_procs == []
    # reaping again is a clean no-op
    orch._kill_twitch_sidecars()
    assert len(killed) == 2


def test_overlay_starts_and_stops_when_enabled() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    cfg = TwitchConfig(enabled=True, overlay={"enabled": True, "port": 0})
    orch._start_twitch_overlay(cfg)
    server = orch._twitch_overlay_server
    assert server is not None
    url = server.url()
    assert url.startswith("http://127.0.0.1:") and "token=" in url
    orch._stop_twitch_overlay()
    assert orch._twitch_overlay_server is None


def test_overlay_noop_when_disabled() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    orch._start_twitch_overlay(TwitchConfig(enabled=True))  # overlay.enabled defaults False
    assert getattr(orch, "_twitch_overlay_server", None) is None


def test_hook_wires_sidecars_overlay_and_shutdown() -> None:
    import inspect

    hook = inspect.getsource(Orchestrator._start_twitch_chat_mode)
    assert "_start_twitch_sidecars(tcfg)" in hook
    assert "_start_twitch_overlay(tcfg)" in hook
    assert "RedeemRouter" in hook and "make_redeem_drain_fn" in hook  # redeem loop wired
    cls = inspect.getsource(Orchestrator)
    assert "_kill_twitch_sidecars()" in cls
    assert "_stop_twitch_overlay()" in cls
    # moderation dispatch is wired into BOTH the full + lean command paths
    run_src = inspect.getsource(Orchestrator.run)
    assert run_src.count("_maybe_handle_twitch_moderation(user_text)") == 2


# --- voice-moderation two-phase handler (injected fake remote) --------------- #
class _FakeRemote:
    def __init__(self, prep) -> None:
        self._prep = prep
        self.confirmed: list = []
        self.cancelled: list = []

    def prepare(self, text):
        return dict(self._prep)

    def confirm(self, token):
        self.confirmed.append(token)
        return {"ok": True}

    def cancel(self, token):
        self.cancelled.append(token)


def _mod_orch(remote):
    orch = Orchestrator.__new__(Orchestrator)
    orch._twitch_mod_remote = remote
    orch._twitch_mod_pending = None
    spoken: list = []
    orch._speak = lambda s, *a, **k: spoken.append(s)  # type: ignore[assignment]
    return orch, spoken


def test_moderation_handler_noop_without_remote() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    orch._twitch_mod_remote = None
    assert orch._maybe_handle_twitch_moderation("ban xqc") is False


def test_moderation_handler_not_a_command_falls_through() -> None:
    orch, spoken = _mod_orch(_FakeRemote({"not_a_command": True, "ok": False}))
    assert orch._maybe_handle_twitch_moderation("nice clutch jett") is False
    assert spoken == []


def test_moderation_sidecar_error_falls_through() -> None:
    """A sidecar HTTP error dict must NOT trigger 'I cannot do that' — fall through."""
    orch, spoken = _mod_orch(_FakeRemote({"ok": False, "error": "http_500"}))
    assert orch._maybe_handle_twitch_moderation("say hello") is False
    assert spoken == []


def test_moderation_sidecar_unavailable_falls_through() -> None:
    """A transport-failure response (error=unavailable) must fall through silently."""
    orch, spoken = _mod_orch(_FakeRemote({"ok": False, "error": "unavailable"}))
    assert orch._maybe_handle_twitch_moderation("relay hello to the team") is False
    assert spoken == []


def test_moderation_two_phase_confirm_yes() -> None:
    remote = _FakeRemote({"ok": True, "token": "tok123", "readback": "Ban viewer xqc. Confirm?",
                          "action": "ban", "target": "xqc"})
    orch, spoken = _mod_orch(remote)
    # phase 1: command -> readback + pending set
    assert orch._maybe_handle_twitch_moderation("ban xqc") is True
    assert orch._twitch_mod_pending and orch._twitch_mod_pending["token"] == "tok123"
    assert spoken[-1] == "Ban viewer xqc. Confirm?"
    # phase 2: "yes" -> confirm + cleared
    assert orch._maybe_handle_twitch_moderation("yes") is True
    assert remote.confirmed == ["tok123"]
    assert orch._twitch_mod_pending is None
    assert "Banned xqc" in spoken[-1]


def test_moderation_two_phase_cancel_no() -> None:
    remote = _FakeRemote({"ok": True, "token": "t2", "readback": "Timeout viewer foo. Confirm?",
                          "action": "timeout", "target": "foo"})
    orch, spoken = _mod_orch(remote)
    orch._maybe_handle_twitch_moderation("timeout foo for 10 minutes")
    assert orch._maybe_handle_twitch_moderation("no") is True
    assert remote.cancelled == ["t2"]
    assert orch._twitch_mod_pending is None
    assert "Stood down" in spoken[-1]


def test_moderation_blocked_command_speaks_reason() -> None:
    orch, spoken = _mod_orch(_FakeRemote(
        {"ok": False, "reason_blocked": "protected", "target": "the_broadcaster"}))
    assert orch._maybe_handle_twitch_moderation("ban the_broadcaster") is True
    assert orch._twitch_mod_pending is None
    assert "will not action" in spoken[-1]


def test_moderation_decision_classifier() -> None:
    assert Orchestrator._twitch_mod_decision("yes") == "yes"
    assert Orchestrator._twitch_mod_decision("Do it.") == "yes"
    assert Orchestrator._twitch_mod_decision("cancel") == "no"
    assert Orchestrator._twitch_mod_decision("never mind") == "no"
    assert Orchestrator._twitch_mod_decision("what time is it") == "other"
