"""S10 ChatModeRuntime — guard-gated enable, tick loop, fail-closed, flagged review."""
from __future__ import annotations

from dataclasses import dataclass

from kenning.twitch.pipeline import ChatReplyPipeline
from kenning.twitch.runtime import ChatModeRuntime, ChatModeState
from kenning.twitch.safety.validator import build_chat_validator


@dataclass
class Ev:
    text: str
    chatter_user_id: str = "u1"
    chatter_login: str = "bob"
    chatter_name: str = "Bob"


class Guard:
    def __init__(self, h=True, c=True):
        self._h, self._c = h, c

    def health(self):
        return self._h

    def canary(self):
        return self._c


ALL = lambda e: True  # noqa: E731
IDENT = lambda evs: list(evs)  # noqa: E731


def _runtime(*, drain, guard=Guard(), guard_required=True, reply="Acknowledged.", on_flagged=None):
    spoken = []
    pipe = ChatReplyPipeline(validator=build_chat_validator(audit_path=None),
                             speak_fn=lambda t, provenance=None: spoken.append(t))
    rt = ChatModeRuntime(
        pipeline=pipe, drain_fn=drain, is_reply_target=ALL, select_fn=IDENT,
        reply_fn=lambda s: reply, guard_client=guard, guard_required=guard_required,
        on_flagged=on_flagged,
    )
    return rt, spoken


def test_enable_requires_healthy_guard() -> None:
    rt, _ = _runtime(drain=lambda: [])
    rt2, _ = _runtime(drain=lambda: [], guard=None)
    assert rt2.enable()[0] is False and not rt2.active and rt2.state == ChatModeState.OFF
    rt3, _ = _runtime(drain=lambda: [], guard=Guard(h=True, c=False))  # canary fails
    assert rt3.enable()[0] is False
    assert rt.enable()[0] is True and rt.active and rt.state == ChatModeState.READY


def test_enable_without_guard_when_not_required() -> None:
    rt, _ = _runtime(drain=lambda: [], guard=None, guard_required=False)
    assert rt.enable()[0] is True and rt.active


def test_tick_off_does_nothing() -> None:
    drained = []
    rt, _ = _runtime(drain=lambda: drained.append(1) or [])
    assert rt.tick() is None
    assert drained == []  # drain never called while OFF


def test_tick_processes_and_speaks() -> None:
    rt, spoken = _runtime(drain=lambda: [Ev("nice clutch jett")])
    rt.enable()
    r = rt.tick()
    # @-tagged to the viewer it answers (2026-06-26).
    assert r is not None and r.spoke == "@Bob Acknowledged."
    assert spoken == ["@Bob Acknowledged."]


def test_tick_empty_buffer_returns_none_stays_ready() -> None:
    rt, _ = _runtime(drain=lambda: [])
    rt.enable()
    assert rt.tick() is None and rt.state == ChatModeState.READY


def test_tick_drain_error_is_fail_closed_lockdown() -> None:
    def boom():
        raise RuntimeError("read sidecar down")

    rt, spoken = _runtime(drain=boom)
    rt.enable()
    assert rt.tick() is None and rt.state == ChatModeState.LOCKDOWN
    assert spoken == []
    assert rt.active  # lockdown auto-recovers on the next clean tick


def test_flagged_messages_surface_to_review_and_are_not_spoken() -> None:
    seen = []
    rt, spoken = _runtime(drain=lambda: [Ev("you faggot", chatter_user_id="bad")],
                          on_flagged=seen.append)
    rt.enable()
    rt.tick()
    assert any(f.user_id == "bad" for f in seen)
    assert spoken == []  # unsafe inbound is never spoken


def test_disable_returns_to_off() -> None:
    rt, _ = _runtime(drain=lambda: [])
    rt.enable()
    rt.disable()
    assert rt.state == ChatModeState.OFF and not rt.active
