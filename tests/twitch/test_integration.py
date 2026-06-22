"""S10 integration factory — the wired chat-mode runtime end-to-end (mocked deps)."""
from __future__ import annotations

from kenning.audio.provenance import Provenance
from kenning.twitch.clients.eventsub import ChatEvent
from kenning.twitch.integration import build_chat_mode_runtime, make_stream_speak_fn


def Ev(text, *, uid="u1", login="bob", name="Bob"):
    # Real ChatEvent (the reply module isinstance-filters to it; the read sidecar
    # yields real ChatEvents in production).
    return ChatEvent(broadcaster_user_id="b", chatter_user_id=uid, chatter_login=login,
                     chatter_name=name, text=text)


class _Cfg:
    class auth:
        bot_login = "ultronbot"
        broadcaster_login = "streamer"

    class chat:
        batch_max_messages = 10
        reply_max_chars = 240

    class safety:
        guard_required = False


def _build(drain, spoken):
    return build_chat_mode_runtime(
        _Cfg,
        llm_fn=lambda system, user: "Acknowledged CHATTER_1.",
        speak_fn=lambda t, provenance=None: spoken.append((t, provenance)),
        drain_fn=drain,
        guard_client=None,            # guard_required=False -> enable allowed
        embed_fn=None,
    )


def test_addressed_message_is_answered_and_detokenized() -> None:
    spoken = []
    rt = _build(lambda: [Ev("@ultronbot are you real", name="Bob")], spoken)
    assert rt.enable()[0] is True
    r = rt.tick()
    assert r is not None and r.spoke
    # CHATTER_1 was de-tokenized back to the real display name by reply.generate_reply
    assert "CHATTER_1" not in r.spoke and "Bob" in r.spoke
    assert spoken and spoken[0][1] == Provenance.TWITCH_CHAT


def test_unaddressed_message_is_ignored() -> None:
    spoken = []
    rt = _build(lambda: [Ev("hello everyone in chat today")], spoken)
    rt.enable()
    r = rt.tick()
    assert r is None or r.spoke is None
    assert spoken == []


def test_make_stream_speak_fn_refuses_non_chat_provenance() -> None:
    calls = []
    speak = make_stream_speak_fn(lambda t: calls.append(t))
    speak("hi", provenance=Provenance.TWITCH_CHAT)
    assert calls == ["hi"]
    speak("leak", provenance=Provenance.LOCAL_VOICE)   # must be refused
    assert calls == ["hi"]


def test_enable_is_guard_gated_when_required() -> None:
    # flip guard_required on -> with no guard client, enable must fail-CLOSED.
    class Cfg2(_Cfg):
        class safety:
            guard_required = True

    rt = build_chat_mode_runtime(
        Cfg2, llm_fn=lambda s, u: "x", speak_fn=lambda t, provenance=None: None,
        drain_fn=lambda: [], guard_client=None,
    )
    assert rt.enable()[0] is False and not rt.active
