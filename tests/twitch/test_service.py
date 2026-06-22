"""S10 ChatModeService — toggle, tick, guard-gating, drain fail-safe."""
from __future__ import annotations

from kenning.twitch.clients.eventsub import ChatEvent
from kenning.twitch.service import ChatModeService, make_read_drain_fn


class _Cfg:
    read_sidecar_endpoint = "http://127.0.0.1:8773"

    class auth:
        bot_login = "ultronbot"
        broadcaster_login = "streamer"

    class chat:
        batch_max_messages = 10
        reply_max_chars = 240

    class safety:
        guard_required = True
        guard_endpoint = "http://127.0.0.1:8774"


class Guard:
    def __init__(self, h=True, c=True):
        self._h, self._c = h, c

    def health(self):
        return self._h

    def canary(self):
        return self._c

    def classify(self, text, *, exchange=""):
        from kenning.twitch.safety.validator import GuardResult
        return GuardResult(unsafe=False, score=0.0)


def _ev(text):
    return ChatEvent(broadcaster_user_id="b", chatter_user_id="u1", chatter_login="bob",
                     chatter_name="Bob", text=text)


def _svc(*, drain, guard=Guard(), spoken=None):
    return ChatModeService(
        _Cfg, llm_fn=lambda s, u: "Acknowledged CHATTER_1.",
        orchestrator_speak=lambda t: (spoken if spoken is not None else []).append(t),
        drain_fn=drain, guard_client=guard,
    )


def test_toggle_on_is_guard_gated_then_active() -> None:
    s_bad = _svc(drain=lambda: [], guard=Guard(h=False))
    assert s_bad.set_chat_mode(True)[0] is False and not s_bad.active
    s = _svc(drain=lambda: [])
    assert s.set_chat_mode(True)[0] is True and s.active
    s.set_chat_mode(False)
    assert not s.active


def test_tick_processes_when_active() -> None:
    spoken = []
    s = _svc(drain=lambda: [_ev("@ultronbot are you real")], spoken=spoken)
    s.set_chat_mode(True)
    r = s.tick()
    assert r is not None and r.spoke and spoken


def test_tick_when_off_is_none() -> None:
    s = _svc(drain=lambda: [_ev("hi")])
    assert s.tick() is None  # not enabled


def test_tick_fail_closed_on_drain_error() -> None:
    def boom():
        raise RuntimeError("read sidecar down")

    s = _svc(drain=boom)
    s.set_chat_mode(True)
    assert s.tick() is None and s.active  # lockdown, no crash


def test_make_read_drain_fn_is_failsafe_when_sidecar_down() -> None:
    # nothing listening on this port -> empty batch, never raises.
    drain = make_read_drain_fn("http://127.0.0.1:59597", timeout=0.3)
    assert drain() == []


def test_sync_and_tick_reconciles_to_flag() -> None:
    spoken = []
    s = _svc(drain=lambda: [_ev("@ultronbot hi")], spoken=spoken)
    # want_on True -> enables (guard ok) + ticks
    r = s.sync_and_tick(True)
    assert s.active and r is not None and r.spoke
    # want_on False -> disables + no tick
    assert s.sync_and_tick(False) is None and not s.active
