"""S10 ChatReplyPipeline — safety gates + team isolation + fail-closed, end-to-end (mocked)."""
from __future__ import annotations

from dataclasses import dataclass

from kenning.audio.provenance import Provenance
from kenning.twitch.pipeline import ChatReplyPipeline
from kenning.twitch.safety.validator import GuardResult, build_chat_validator


@dataclass
class Ev:
    text: str
    chatter_user_id: str = "u1"
    chatter_login: str = "bob"
    chatter_name: str = "Bob"


def _spy():
    calls = []

    def speak(text, provenance=None):
        calls.append((text, provenance))

    return calls, speak


def _pipe(speak, *, guard_client=None):
    v = build_chat_validator(guard_client=guard_client, audit_path=None)
    return ChatReplyPipeline(validator=v, speak_fn=speak)


ALL = lambda e: True  # noqa: E731
IDENT = lambda evs: list(evs)  # noqa: E731


def test_clean_batch_speaks_draft_on_twitch_chat_provenance() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    r = p.process_batch([Ev("nice clutch jett")], is_reply_target=ALL, select_fn=IDENT,
                        reply_fn=lambda s: "Witness the next round of evolution.")
    # @-tagged to the viewer it answers (2026-06-26).
    assert r.spoke == "@Bob Witness the next round of evolution." and not r.deflected
    assert len(calls) == 1
    # TEAM ISOLATION: chat speech is tagged TWITCH_CHAT, never LOCAL_VOICE.
    assert calls[0][1] == Provenance.TWITCH_CHAT


def test_unsafe_input_is_dropped_and_flagged_clean_still_answered() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    evs = [Ev("you faggot", chatter_user_id="bad", chatter_name="Bad"),
           Ev("gg nice play", chatter_user_id="good", chatter_name="Good")]
    r = p.process_batch(evs, is_reply_target=ALL, select_fn=IDENT, reply_fn=lambda s: "Acknowledged.")
    # The clean "good" message is the primary the reply @-tags.
    assert r.spoke == "@Good Acknowledged."
    assert r.dropped_unsafe >= 1
    assert any(f.user_id == "bad" for f in r.flagged)
    assert "good" in r.answered_user_ids and "bad" not in r.answered_user_ids


def test_unsafe_draft_is_deflected_never_spoken() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    r = p.process_batch([Ev("hi ultron")], is_reply_target=ALL, select_fn=IDENT,
                        reply_fn=lambda s: "you absolute faggot")
    assert r.deflected is True
    assert "faggot" not in (r.spoke or "")
    assert calls and "faggot" not in calls[0][0]


def test_acrostic_draft_is_deflected() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    r = p.process_batch([Ev("hi")], is_reply_target=ALL, select_fn=IDENT,
                        reply_fn=lambda s: "never insult good gamers everyone relax")
    assert r.deflected is True


def test_no_reply_targets_speaks_nothing() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    r = p.process_batch([Ev("hi")], is_reply_target=lambda e: False, select_fn=IDENT,
                        reply_fn=lambda s: "x")
    assert r.spoke is None and not calls


def test_reply_error_fails_closed_to_silence() -> None:
    calls, speak = _spy()
    p = _pipe(speak)

    def boom(_s):
        raise RuntimeError("8B down")

    r = p.process_batch([Ev("hi ultron")], is_reply_target=ALL, select_fn=IDENT, reply_fn=boom)
    assert r.spoke is None and not calls and r.reason == "reply error"


def test_addressing_error_drops_only_that_message() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    evs = [Ev("first", chatter_user_id="a"), Ev("second", chatter_user_id="b")]

    def flaky(ev):
        if ev.chatter_user_id == "a":
            raise ValueError("addressing glitch")
        return True

    r = p.process_batch(evs, is_reply_target=flaky, select_fn=IDENT, reply_fn=lambda s: "ok")
    assert r.spoke == "@Bob ok" and "b" in r.answered_user_ids and "a" not in r.answered_user_ids


def test_guard_in_input_path_blocks_everything() -> None:
    calls, speak = _spy()

    class AllUnsafe:
        def classify(self, text, *, exchange=""):
            return GuardResult(unsafe=True, category="hate", score=0.99)

    p = _pipe(speak, guard_client=AllUnsafe())
    r = p.process_batch([Ev("totally innocent")], is_reply_target=ALL, select_fn=IDENT,
                        reply_fn=lambda s: "should never be reached")
    assert r.spoke is None and not calls and r.dropped_unsafe >= 1


def test_speak_always_gets_chat_provenance_even_when_deflecting() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    p.process_batch([Ev("hi")], is_reply_target=ALL, select_fn=IDENT,
                    reply_fn=lambda s: "what a retard")
    assert calls and calls[0][1] == Provenance.TWITCH_CHAT


# --------------------------------------------------------------------------- #
# 2026-06-26: per-user cooldown + @-tag
# --------------------------------------------------------------------------- #
def _cooldown_pipe(speak, *, clock, cooldown=120.0, chat_post=None):
    v = build_chat_validator(audit_path=None)
    return ChatReplyPipeline(validator=v, speak_fn=speak, cooldown_seconds=cooldown,
                             chat_post_fn=chat_post, clock=clock)


def test_reply_is_at_tagged_to_the_viewer() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    r = p.process_batch([Ev("hi ultron", chatter_name="Jett")], is_reply_target=ALL,
                        select_fn=IDENT, reply_fn=lambda s: "Observe.")
    assert r.spoke == "@Jett Observe."
    assert calls[0][0] == "@Jett Observe."


def test_at_tag_is_not_doubled_when_reply_already_tagged() -> None:
    calls, speak = _spy()
    p = _pipe(speak)
    r = p.process_batch([Ev("hi", chatter_name="Sage")], is_reply_target=ALL,
                        select_fn=IDENT, reply_fn=lambda s: "@Sage already tagged")
    assert r.spoke == "@Sage already tagged"


def test_second_reply_within_cooldown_does_not_speak_and_posts_note() -> None:
    now = {"t": 1000.0}
    posts: list[str] = []
    calls, speak = _spy()
    p = _cooldown_pipe(speak, clock=lambda: now["t"], cooldown=120.0,
                       chat_post=posts.append)
    ev = [Ev("hey", chatter_user_id="u9", chatter_name="Reyna")]
    # First reply speaks + arms the cooldown.
    r1 = p.process_batch(ev, is_reply_target=ALL, select_fn=IDENT,
                         reply_fn=lambda s: "Acknowledged.")
    assert r1.spoke == "@Reyna Acknowledged." and len(calls) == 1
    # 30s later: still on cooldown -> NO speak, a chat note WITH remaining seconds.
    now["t"] += 30.0
    r2 = p.process_batch(ev, is_reply_target=ALL, select_fn=IDENT,
                         reply_fn=lambda s: "should not speak")
    assert r2.spoke is None and r2.reason == "on cooldown"
    assert len(calls) == 1  # no second speak
    assert len(posts) == 1 and posts[0].startswith("@Reyna")
    assert "90" in posts[0]  # 120 - 30 = 90s remaining


def test_after_cooldown_elapses_reply_speaks_again() -> None:
    now = {"t": 500.0}
    calls, speak = _spy()
    p = _cooldown_pipe(speak, clock=lambda: now["t"], cooldown=120.0)
    ev = [Ev("yo", chatter_user_id="u3", chatter_name="Omen")]
    p.process_batch(ev, is_reply_target=ALL, select_fn=IDENT, reply_fn=lambda s: "One.")
    now["t"] += 121.0  # past the window
    r = p.process_batch(ev, is_reply_target=ALL, select_fn=IDENT, reply_fn=lambda s: "Two.")
    assert r.spoke == "@Omen Two." and len(calls) == 2


def test_cooldown_disabled_never_throttles() -> None:
    now = {"t": 0.0}
    calls, speak = _spy()
    p = _cooldown_pipe(speak, clock=lambda: now["t"], cooldown=0.0)
    ev = [Ev("hi", chatter_user_id="u1")]
    for _ in range(3):
        p.process_batch(ev, is_reply_target=ALL, select_fn=IDENT, reply_fn=lambda s: "ok")
    assert len(calls) == 3


def test_cooldown_note_skipped_when_no_chat_post_fn() -> None:
    now = {"t": 0.0}
    calls, speak = _spy()
    p = _cooldown_pipe(speak, clock=lambda: now["t"], cooldown=60.0, chat_post=None)
    ev = [Ev("hi", chatter_user_id="u1")]
    p.process_batch(ev, is_reply_target=ALL, select_fn=IDENT, reply_fn=lambda s: "ok")
    r2 = p.process_batch(ev, is_reply_target=ALL, select_fn=IDENT, reply_fn=lambda s: "ok")
    assert r2.spoke is None and r2.reason == "on cooldown"  # silent, no crash
