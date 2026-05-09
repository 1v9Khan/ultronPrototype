"""A4 unit tests: Orchestrator's pre-task confirmation + barge-in flow.

These tests don't load the full voice stack -- they construct an
Orchestrator instance via ``__new__`` and inject only the attributes
the methods under test touch. That keeps the suite fast and CI-friendly
while still exercising the production code paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from ultron.coding.voice import VoiceResponse
from ultron.pipeline.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubTTS:
    def __init__(self, raise_after: int = -1):
        self.spoken: List[str] = []
        self._raise_after = raise_after

    def speak(self, text: str) -> None:
        if 0 <= self._raise_after <= len(self.spoken):
            raise RuntimeError("simulated piper failure")
        self.spoken.append(text)


class _StubWake:
    def __init__(self) -> None:
        self._last_trigger_ts = 0.0

    def fire(self, *, ts: Optional[float] = None) -> None:
        self._last_trigger_ts = ts if ts is not None else time.monotonic()


@dataclass
class _StubRunner:
    aborted: List[dict] = field(default_factory=list)

    def record_pre_task_aborted(
        self, *, label: Optional[str], reason: str, intent_text: str = "",
    ) -> None:
        self.aborted.append({
            "label": label, "reason": reason, "intent_text": intent_text,
        })


@dataclass
class _StubCodingVoice:
    runner: _StubRunner


@dataclass
class _StubRoutingIntent:
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orch(*, tts=None, wake=None, coding_voice=None) -> Orchestrator:
    """Build a partially-initialised Orchestrator just for unit tests."""
    o = Orchestrator.__new__(Orchestrator)
    o.tts = tts or _StubTTS()
    o.wake = wake or _StubWake()
    o.coding_voice = coding_voice
    return o


# ---------------------------------------------------------------------------
# _speak_with_barge_in_check
# ---------------------------------------------------------------------------


def test_speak_with_barge_in_no_wake_returns_false():
    orch = _make_orch()
    barge_in = orch._speak_with_barge_in_check(  # noqa: SLF001
        "I'll have Claude Code do something. Going ahead.",
        post_check_window_s=0.0,
    )
    assert barge_in is False
    assert orch.tts.spoken == [
        "I'll have Claude Code do something. Going ahead.",
    ]


def test_speak_with_barge_in_detects_wake_during_speech(monkeypatch):
    """A wake fire whose timestamp lands AFTER the speak call completes
    must be reported as a barge-in."""
    wake = _StubWake()
    tts = _StubTTS()
    orch = _make_orch(tts=tts, wake=wake)

    # Patch tts.speak to fire the wake mid-speech.
    real_speak = tts.speak

    def _speak_then_fire(text: str) -> None:
        real_speak(text)
        wake.fire()  # advances _last_trigger_ts past `before_ts`

    tts.speak = _speak_then_fire
    barge_in = orch._speak_with_barge_in_check(  # noqa: SLF001
        "test confirmation", post_check_window_s=0.0,
    )
    assert barge_in is True


def test_speak_with_barge_in_ignores_pre_existing_wake():
    """A wake that fired BEFORE we started speaking must not be treated
    as a barge-in -- the user's interrupt must overlap the playback."""
    wake = _StubWake()
    wake.fire()  # set timestamp before _speak_with_barge_in_check is called
    orch = _make_orch(wake=wake)
    barge_in = orch._speak_with_barge_in_check(  # noqa: SLF001
        "test", post_check_window_s=0.0,
    )
    assert barge_in is False


def test_speak_with_barge_in_tts_failure_returns_false():
    """A TTS error must NOT be misread as a barge-in -- we want the
    coding pipeline to keep going on a Piper hiccup."""
    tts = _StubTTS(raise_after=0)
    orch = _make_orch(tts=tts)
    barge_in = orch._speak_with_barge_in_check(  # noqa: SLF001
        "test", post_check_window_s=0.0,
    )
    assert barge_in is False


def test_speak_with_barge_in_empty_text_returns_false():
    orch = _make_orch()
    assert orch._speak_with_barge_in_check("") is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# _handle_capability_response: legacy (no pre-task confirmation) path
# ---------------------------------------------------------------------------


def test_handle_capability_response_legacy_speaks_text():
    orch = _make_orch()
    response = VoiceResponse(text="Working on it.")
    orch._handle_capability_response(response, _StubRoutingIntent())  # noqa: SLF001
    assert orch.tts.spoken == ["Working on it."]


# ---------------------------------------------------------------------------
# _handle_capability_response: A4 dispatch path
# ---------------------------------------------------------------------------


def test_handle_capability_response_a4_dispatches_when_no_barge_in():
    runner = _StubRunner()
    coding_voice = _StubCodingVoice(runner=runner)
    orch = _make_orch(coding_voice=coding_voice)

    dispatched = {"called": False}

    def _dispatch():
        dispatched["called"] = True

    response = VoiceResponse(
        text="Working on calculator.",
        pre_task_confirmation="I'll have Claude Code add subtract on the calculator project. Going ahead.",
        deferred_dispatch=_dispatch,
        pre_task_label="calculator",
    )
    orch._handle_capability_response(response, _StubRoutingIntent())  # noqa: SLF001

    assert dispatched["called"] is True
    assert orch.tts.spoken == [
        "I'll have Claude Code add subtract on the calculator project. Going ahead.",
        "Working on calculator.",
    ]
    assert runner.aborted == []  # no abort logged


def test_handle_capability_response_a4_skips_dispatch_on_barge_in(monkeypatch):
    runner = _StubRunner()
    coding_voice = _StubCodingVoice(runner=runner)
    wake = _StubWake()
    tts = _StubTTS()
    orch = _make_orch(tts=tts, wake=wake, coding_voice=coding_voice)

    # Inject a wake fire during the pre-task speak.
    real_speak = tts.speak

    def _speak_then_fire(text: str) -> None:
        real_speak(text)
        if "Going ahead" in text:
            wake.fire()

    tts.speak = _speak_then_fire

    dispatched = {"called": False}

    def _dispatch():
        dispatched["called"] = True

    response = VoiceResponse(
        text="Working on calculator.",
        pre_task_confirmation="I'll have Claude Code add subtract on the calculator project. Going ahead.",
        deferred_dispatch=_dispatch,
        pre_task_label="calculator",
    )
    routing_intent = _StubRoutingIntent(raw_text="add subtract to calculator")
    orch._handle_capability_response(response, routing_intent)  # noqa: SLF001

    # Dispatch did NOT run.
    assert dispatched["called"] is False
    # The post-dispatch ``text`` was NOT spoken; user heard the cancellation.
    assert orch.tts.spoken == [
        "I'll have Claude Code add subtract on the calculator project. Going ahead.",
        "Cancelled. What did you mean?",
    ]
    # Audit recorded.
    assert runner.aborted == [{
        "label": "calculator",
        "reason": "barge_in",
        "intent_text": "add subtract to calculator",
    }]


def test_handle_capability_response_a4_audit_failure_does_not_crash():
    """A raising audit-log call must not bubble up into the orchestrator."""

    class _BoomRunner:
        def record_pre_task_aborted(self, **kwargs):
            raise RuntimeError("simulated audit failure")

    coding_voice = _StubCodingVoice(runner=_BoomRunner())
    wake = _StubWake()
    tts = _StubTTS()
    orch = _make_orch(tts=tts, wake=wake, coding_voice=coding_voice)
    real_speak = tts.speak

    def _speak_then_fire(text: str) -> None:
        real_speak(text)
        wake.fire()

    tts.speak = _speak_then_fire
    response = VoiceResponse(
        text="...",
        pre_task_confirmation="confirm",
        deferred_dispatch=lambda: None,
        pre_task_label="x",
    )
    # Must not raise.
    orch._handle_capability_response(response, _StubRoutingIntent())  # noqa: SLF001
    # User still heard the cancellation.
    assert any("Cancelled" in t for t in orch.tts.spoken)


def test_handle_capability_response_a4_dispatch_failure_logged_not_raised():
    """A raising deferred_dispatch must not crash the voice loop; the
    user still hears the post-dispatch text (best-effort)."""
    runner = _StubRunner()
    coding_voice = _StubCodingVoice(runner=runner)
    orch = _make_orch(coding_voice=coding_voice)

    def _boom():
        raise RuntimeError("simulated dispatch failure")

    response = VoiceResponse(
        text="Working on it.",
        pre_task_confirmation="confirm",
        deferred_dispatch=_boom,
        pre_task_label="x",
    )
    # Must not raise.
    orch._handle_capability_response(response, _StubRoutingIntent())  # noqa: SLF001
    assert orch.tts.spoken == ["confirm", "Working on it."]
