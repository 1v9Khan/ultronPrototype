"""Tests for :mod:`ultron.utils.spinner`."""

from __future__ import annotations

import io
import time

import pytest

from ultron.utils.spinner import (
    DEFAULT_FIRST_FRAME_DELAY_SECONDS,
    DEFAULT_UPDATE_INTERVAL_SECONDS,
    Spinner,
    TRACK_LENGTH,
    WaitingSpinner,
)


def test_track_length_constant():
    assert TRACK_LENGTH == 10


def test_default_first_frame_delay():
    assert DEFAULT_FIRST_FRAME_DELAY_SECONDS == 0.5


def test_default_update_interval():
    assert DEFAULT_UPDATE_INTERVAL_SECONDS == 0.1


def test_constructor_succeeds():
    s = Spinner("test", stream=io.StringIO())
    assert s._message == "test"
    assert s._frames  # frames were built


def test_first_frame_delay_blocks_visible_output():
    """Within the delay window, step() should write nothing."""
    sink = io.StringIO()
    s = Spinner(
        "msg",
        stream=sink,
        first_frame_delay=10.0,  # huge delay; we won't pass it
    )
    s.start()
    s.step()
    s.step()
    s.step()
    # No output: still in first-frame delay.
    assert sink.getvalue() == ""


def test_step_renders_after_first_frame_delay():
    sink = io.StringIO()
    s = Spinner(
        "msg",
        stream=sink,
        first_frame_delay=0.0,
        update_interval=0.0,
    )
    s.step()
    out = sink.getvalue()
    assert "msg" in out
    # A bounce frame contains at least one of the spinner chars.
    assert "█" in out or "#" in out


def test_update_interval_throttles():
    """Within update_interval seconds we should NOT advance the visible frame."""
    sink = io.StringIO()
    s = Spinner(
        stream=sink,
        first_frame_delay=0.0,
        update_interval=10.0,  # huge interval
    )
    s.step()
    first = sink.getvalue()
    s.step()  # should be throttled
    s.step()
    second = sink.getvalue()
    # Nothing additional written because the interval blocked it.
    assert second == first


def test_force_ascii_uses_ascii_chars():
    sink = io.StringIO()
    s = Spinner(
        "x",
        stream=sink,
        first_frame_delay=0.0,
        update_interval=0.0,
        force_ascii=True,
    )
    s.step()
    out = sink.getvalue()
    assert "#" in out
    assert "█" not in out


def test_frames_built_for_full_bounce():
    """Forward 10 frames + backward 8 frames = 18 frames total."""
    s = Spinner(stream=io.StringIO())
    assert len(s._frames) == 18


def test_continuity_class_var_updates_after_step():
    Spinner.reset_continuity()
    assert Spinner.last_frame_index == 0
    sink = io.StringIO()
    s = Spinner(
        stream=sink,
        first_frame_delay=0.0,
        update_interval=0.0,
    )
    s.step()
    assert Spinner.last_frame_index == 1
    s.step()
    assert Spinner.last_frame_index == 2


def test_continuity_carries_across_instances():
    Spinner.reset_continuity()
    sink1 = io.StringIO()
    s1 = Spinner(stream=sink1, first_frame_delay=0.0, update_interval=0.0)
    s1.step()
    s1.step()
    s1.step()
    saved_idx = Spinner.last_frame_index
    # New instance picks up where the last left off.
    sink2 = io.StringIO()
    s2 = Spinner(stream=sink2, first_frame_delay=0.0, update_interval=0.0)
    assert s2._frame_index == saved_idx


def test_reset_continuity_clears_state():
    Spinner.last_frame_index = 7
    Spinner.reset_continuity()
    assert Spinner.last_frame_index == 0


def test_context_manager_clears_on_exit():
    sink = io.StringIO()
    with Spinner("loading", stream=sink, first_frame_delay=0.0, update_interval=0.0) as s:
        s.step()
        before_exit = sink.getvalue()
        assert "loading" in before_exit
    # After exit, the line has been cleared (carriage return + spaces written).
    assert "\r" in sink.getvalue()


def test_end_is_safe_before_any_step():
    """Calling end() before any visible step must not crash."""
    s = Spinner(stream=io.StringIO())
    s.end()  # no exception
    s.end()  # idempotent-ish


def test_waiting_spinner_lifecycle():
    sink = io.StringIO()
    w = WaitingSpinner(
        "loading",
        stream=sink,
        tick_interval=0.05,
        first_frame_delay=0.0,
    )
    w.start()
    time.sleep(0.2)
    w.stop(timeout=1)
    out = sink.getvalue()
    assert "loading" in out


def test_waiting_spinner_context_manager():
    sink = io.StringIO()
    with WaitingSpinner(
        "ctx",
        stream=sink,
        tick_interval=0.05,
        first_frame_delay=0.0,
    ):
        time.sleep(0.15)
    # After stop, last write should be the clear-line sequence.
    assert "\r" in sink.getvalue()


def test_waiting_spinner_start_is_idempotent():
    w = WaitingSpinner(stream=io.StringIO())
    w.start()
    first_thread = w._thread
    w.start()
    assert w._thread is first_thread
    w.stop()
