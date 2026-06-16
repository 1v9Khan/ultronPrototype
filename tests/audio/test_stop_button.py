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
