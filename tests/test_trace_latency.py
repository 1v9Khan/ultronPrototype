"""Pins the per-turn latency marks (2026-07-26).

Built because there was NO per-stage latency instrumentation: the only
metric was a single ``RESPONSE LATENCY`` line on the relay path, so the
conversational path (the one running the 12B) was entirely unmeasured and
every optimisation would have been a guess.

The load-bearing property is CROSS-THREAD accumulation: speculative STT and
speculative LLM run on background daemons that inherit the turn id via
``restore()`` but get a fresh thread-local store. Marks therefore key off the
turn id in a process-global map -- a thread-local map would drop exactly the
stages worth measuring.
"""

from __future__ import annotations

import threading

import pytest

from kenning import trace


@pytest.fixture(autouse=True)
def _clean_turn():
    trace.set_turn(None)
    trace.set_phase(None)
    yield
    tid = trace.get_turn()
    if tid is not None:
        trace.reset_latency(tid)
    trace.set_turn(None)
    trace.set_phase(None)


def test_no_turn_is_a_noop() -> None:
    """Start-up logs have no turn; marking must not raise or accumulate."""
    trace.set_turn(None)
    trace.mark("stt")
    assert trace.latency_stages() == []
    assert trace.latency_line() == ""


def test_stages_report_in_order_with_deltas() -> None:
    trace.set_turn(9001)
    trace.mark_at("turn_close", 100.0)
    trace.mark_at("stt", 100.5)
    trace.mark_at("route", 100.6)
    trace.mark_at("first_audio", 101.2)

    stages = trace.latency_stages()
    assert [s[0] for s in stages] == ["turn_close", "stt", "route", "first_audio"]
    # since_first_ms is cumulative from the first mark
    assert stages[0][1] == pytest.approx(0.0)
    assert stages[-1][1] == pytest.approx(1200.0)
    # delta_ms is the gap from the previous mark -- this is what finds the
    # expensive stage.
    assert stages[1][2] == pytest.approx(500.0)
    assert stages[2][2] == pytest.approx(100.0)
    assert stages[3][2] == pytest.approx(600.0)


def test_line_is_sorted_slowest_first() -> None:
    trace.set_turn(9002)
    trace.mark_at("turn_close", 0.0)
    trace.mark_at("cheap", 0.01)
    trace.mark_at("expensive", 1.01)
    trace.mark_at("middling", 1.31)

    line = trace.latency_line()
    assert line.startswith("total=1310ms | ")
    # slowest stage first, and the anchor mark is not itself a cost
    assert line.index("expensive=") < line.index("middling=") < line.index("cheap=")
    assert "turn_close=" not in line


def test_line_empty_below_two_marks() -> None:
    trace.set_turn(9003)
    trace.mark("only_one")
    assert trace.latency_line() == ""


def test_background_thread_marks_land_on_the_same_turn() -> None:
    """THE contract: a daemon that restored the parent's trace state must
    contribute to the parent's latency report."""
    trace.set_turn(9004)
    trace.mark_at("turn_close", 0.0)
    state = trace.snapshot()

    def worker() -> None:
        trace.restore(state)          # what the speculative threads do
        trace.mark_at("speculative_stt", 0.4)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    trace.mark_at("first_audio", 1.0)
    names = [s[0] for s in trace.latency_stages()]
    assert names == ["turn_close", "speculative_stt", "first_audio"], (
        "background-thread mark was lost -- the map is not keyed by turn id")


def test_reset_clears_only_that_turn() -> None:
    trace.set_turn(9005)
    trace.mark("a")
    trace.mark("b")
    trace.set_turn(9006)
    trace.mark("c")
    trace.mark("d")

    trace.reset_latency(9005)
    assert trace.latency_stages(9005) == []
    assert len(trace.latency_stages(9006)) == 2


def test_map_is_bounded() -> None:
    """A long stream must not leak one entry per turn forever."""
    for tid in range(20000, 20000 + trace._MARKS_MAX_TURNS * 3):
        trace.set_turn(tid)
        trace.mark("x")
    with trace._marks_lock:
        assert len(trace._marks) <= trace._MARKS_MAX_TURNS
    for tid in range(20000, 20000 + trace._MARKS_MAX_TURNS * 3):
        trace.reset_latency(tid)
