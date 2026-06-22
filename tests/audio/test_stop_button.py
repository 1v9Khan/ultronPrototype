"""Tests for the tiny clickable STOP window's logic layers.

No real Tk window is created here (matching the waveform sweep): the voice
matcher, the click->callback wiring, the fail-open contracts, and the
orchestrator show/hide dispatch are all exercised hermetically.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kenning.audio.stop_button import (
    StopButtonOverlay,
    match_stop_button_command,
)
from kenning.pipeline.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# match_stop_button_command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "show the stop button",
        "show me the stop button",
        "pull up the stop button",
        "bring up the stop button",
        "open the stop button",
        "give me the stop button",
        "summon the stop button",
        "put up the stop button",
        "show the panic button",
        "show the kill switch",
        "show the stop panel",
        "Show The Stop Button.",
        "please show the stop button",
        # STT mangles "show me the stop button" into these imperatives -- a
        # mangled leading verb must still summon it (the live bug 2026-06-16).
        "Hit the stop button.",
        "Call me the stop button.",
        "tap the stop button",
        "press the stop button",
        # bare / trailing-politeness noun phrase
        "stop button",
        "the stop button please",
        "gimme the kill switch",
    ],
)
def test_open_phrasings_match(text: str) -> None:
    assert match_stop_button_command(text) == "open"


@pytest.mark.parametrize(
    "text",
    [
        "hide the stop button",
        "close the stop button",
        "dismiss the stop button",
        "get rid of the stop button",
        "take down the stop button",
        "put away the stop button",
        "hide the panic button",
        "hide the kill switch",
        "please hide the stop button",
    ],
)
def test_close_phrasings_match(text: str) -> None:
    assert match_stop_button_command(text) == "close"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "stop",
        "stop the music",
        "what does the stop button do",
        "tell my team to stop pushing",
        "i hit the stop button earlier",
        "where is the stop button",
        "show me the enemy",
        "push the button",
    ],
)
def test_non_matching_fall_through(text: str) -> None:
    assert match_stop_button_command(text) is None


# ---------------------------------------------------------------------------
# StopButtonOverlay -- construction + click + fail-open (no window built)
# ---------------------------------------------------------------------------


def test_not_shown_before_show() -> None:
    ov = StopButtonOverlay(on_stop=lambda: None)
    assert ov.shown is False


def test_dimensions_are_clamped() -> None:
    ov = StopButtonOverlay(on_stop=lambda: None, width=1, bar_height=-5,
                           button_height=1)
    assert ov._width >= 72
    assert ov._bar_h >= 0
    assert ov._btn_h >= 20


def test_fire_calls_callback() -> None:
    hits = []
    ov = StopButtonOverlay(on_stop=lambda: hits.append(1))
    ov._fire()
    assert hits == [1]


def test_fire_is_fail_open() -> None:
    def boom() -> None:
        raise RuntimeError("kaboom")

    ov = StopButtonOverlay(on_stop=boom)
    # A throwing callback must never propagate out of the click handler.
    ov._fire()


def test_hide_and_close_are_noops_when_never_shown() -> None:
    ov = StopButtonOverlay(on_stop=lambda: None)
    ov.hide()
    ov.close()
    assert ov.shown is False


# ---------------------------------------------------------------------------
# Orchestrator._maybe_handle_stop_button -- dispatch logic (unbound call so no
# full orchestrator is constructed)
# ---------------------------------------------------------------------------


class _FakeButton:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def show(self) -> None:
        self.calls.append("show")

    def hide(self) -> None:
        self.calls.append("hide")


def _fake_orch(button):
    said: list[str] = []
    fake = SimpleNamespace(
        _stop_button=button,
        _speak=lambda t: said.append(t),
    )
    return fake, said


def test_dispatch_open_shows_and_speaks() -> None:
    btn = _FakeButton()
    fake, said = _fake_orch(btn)
    handled = Orchestrator._maybe_handle_stop_button(fake, "show the stop button")
    assert handled is True
    assert btn.calls == ["show"]
    assert said and "up" in said[0].lower()


def test_dispatch_close_hides_and_speaks() -> None:
    btn = _FakeButton()
    fake, said = _fake_orch(btn)
    handled = Orchestrator._maybe_handle_stop_button(fake, "hide the stop button")
    assert handled is True
    assert btn.calls == ["hide"]
    assert said and "hidden" in said[0].lower()


def test_dispatch_non_match_returns_false() -> None:
    btn = _FakeButton()
    fake, said = _fake_orch(btn)
    handled = Orchestrator._maybe_handle_stop_button(fake, "tell my team to rotate")
    assert handled is False
    assert btn.calls == []
    assert said == []


def test_dispatch_handles_missing_button() -> None:
    fake, said = _fake_orch(None)
    handled = Orchestrator._maybe_handle_stop_button(fake, "show the stop button")
    # Still consumes the command (True) and speaks a graceful unavailable line.
    assert handled is True
    assert said and "available" in said[0].lower()


# ---------------------------------------------------------------------------
# FLAG button (2026-06-20) -- last-turn review logging
# ---------------------------------------------------------------------------


def test_overlay_accepts_flag_callback() -> None:
    hits = []
    ov = StopButtonOverlay(on_stop=lambda: None,
                           on_flag=lambda: hits.append(1),
                           flag_height=30, flag_label="FLAG IT")
    assert ov._on_flag is not None
    assert ov._flag_h == 30
    assert ov._flag_label == "FLAG IT"
    ov._on_flag()
    assert hits == [1]


def test_overlay_flag_defaults_and_clamp() -> None:
    ov = StopButtonOverlay(on_stop=lambda: None)
    assert ov._on_flag is None            # absent by default
    assert ov._flag_h == 26               # default height
    assert ov._flag_label == "FLAG LAST"
    ov2 = StopButtonOverlay(on_stop=lambda: None, flag_height=-5, flag_label="")
    assert ov2._flag_h == 0               # clamped to >= 0
    assert ov2._flag_label == "FLAG LAST"  # empty -> default


def test_config_flag_defaults() -> None:
    from kenning.config import StopButtonConfig
    c = StopButtonConfig()
    assert c.flag_height == 26
    assert c.flag_label == "FLAG LAST"


def test_stop_button_flag_logs_last_turn(tmp_path, monkeypatch) -> None:
    import json
    import pathlib
    import time
    import kenning.config as kc
    monkeypatch.setattr(kc, "resolve_path",
                        lambda p: tmp_path / pathlib.Path(p).name)
    o = Orchestrator.__new__(Orchestrator)
    o._current_raw_stt = "their sova ulted B"
    o._current_raw_stt_monotonic = time.monotonic()
    o._last_response_text = "Their Sova ult is up. Play wide."
    o._last_response_finished_monotonic = time.monotonic()
    o._last_scenario = None
    o._stop_button_flag()
    rec = json.loads(
        (tmp_path / "flagged_turns.jsonl").read_text("utf-8").strip())
    assert rec["last_heard"] == "their sova ulted B"
    assert rec["last_response"] == "Their Sova ult is up. Play wide."
    assert rec["flag"] == "user_flagged_turn"
    assert "flagged_at" in rec and "seconds_since_response" in rec


def test_stop_button_flag_appends(tmp_path, monkeypatch) -> None:
    import pathlib
    import time
    import kenning.config as kc
    monkeypatch.setattr(kc, "resolve_path",
                        lambda p: tmp_path / pathlib.Path(p).name)
    o = Orchestrator.__new__(Orchestrator)
    o._current_raw_stt = "x"
    o._current_raw_stt_monotonic = time.monotonic()
    o._last_response_text = "y"
    o._last_response_finished_monotonic = time.monotonic()
    o._last_scenario = None
    o._stop_button_flag()
    o._stop_button_flag()
    lines = (tmp_path / "flagged_turns.jsonl").read_text(
        "utf-8").strip().splitlines()
    assert len(lines) == 2          # appends, never overwrites


def test_stop_button_flag_fail_open(monkeypatch) -> None:
    # A broken resolve_path (or missing attrs) must never raise out of the click.
    import kenning.config as kc

    def _boom(_p):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(kc, "resolve_path", _boom)
    o = Orchestrator.__new__(Orchestrator)
    o._stop_button_flag()           # missing attrs + broken path: must not raise


def test_construction_wires_on_flag() -> None:
    import inspect
    src = inspect.getsource(Orchestrator.__init__)
    assert "on_flag=self._stop_button_flag" in src
    assert "flag_height=_sb.flag_height" in src


# --- always-listening robustness (2026-06-21): command buried in filler + statement guard ---


@pytest.mark.parametrize("text", [
    # always-listening captures the command amid surrounding speech -- the clause
    # scan must still summon it (the live bug: it fell through to PRIVATE_REPLY).
    "Oh, oh, oh, Ultron, Ultron started talking. Show me the stop button.",
    "wait hold on, show me the stop button",
    "okay umm, pull up the stop button please",
])
def test_open_within_noisy_always_listening_capture(text: str) -> None:
    assert match_stop_button_command(text) == "open"


@pytest.mark.parametrize("text", [
    "the stop button interface is not working",
    "the stop button isn't responding",
    "my kill switch is broken",
])
def test_button_complaint_does_not_summon(text: str) -> None:
    # A STATEMENT about the button (a complaint / narration) must NOT summon it.
    assert match_stop_button_command(text) is None
