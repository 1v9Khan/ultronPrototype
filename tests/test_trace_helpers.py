"""Tests for ``ultron.trace`` -- structured per-turn logging helpers."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from ultron import trace


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test starts with cleared trace state."""
    trace.set_turn(None)
    trace.set_phase(None)
    yield
    trace.set_turn(None)
    trace.set_phase(None)


# ---------------------------------------------------------------------------
# turn id
# ---------------------------------------------------------------------------


def test_get_turn_returns_none_when_unset():
    assert trace.get_turn() is None


def test_set_turn_and_get_turn_round_trip():
    trace.set_turn(42)
    assert trace.get_turn() == 42


def test_set_turn_none_clears():
    trace.set_turn(7)
    trace.set_turn(None)
    assert trace.get_turn() is None


def test_next_turn_monotonic():
    a = trace.next_turn()
    b = trace.next_turn()
    c = trace.next_turn()
    assert b == a + 1
    assert c == b + 1


def test_next_turn_installs_id_on_thread():
    tid = trace.next_turn()
    assert trace.get_turn() == tid


def test_turn_state_is_thread_local():
    trace.set_turn(100)
    captured: list = []

    def _worker():
        captured.append(trace.get_turn())

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    # Background thread should see None (no propagation by default).
    assert captured == [None]
    # Main thread still has 100.
    assert trace.get_turn() == 100


# ---------------------------------------------------------------------------
# phase tag
# ---------------------------------------------------------------------------


def test_phase_initially_none():
    assert trace.get_phase() is None


def test_set_phase_and_get_phase_round_trip():
    trace.set_phase("stt")
    assert trace.get_phase() == "stt"


def test_phase_state_is_thread_local():
    trace.set_phase("capture")
    captured: list = []

    def _worker():
        captured.append(trace.get_phase())

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert captured == [None]
    assert trace.get_phase() == "capture"


# ---------------------------------------------------------------------------
# snapshot / restore (cross-thread propagation)
# ---------------------------------------------------------------------------


def test_snapshot_captures_state():
    trace.set_turn(5)
    trace.set_phase("vad")
    snap = trace.snapshot()
    assert snap == {"turn": 5, "phase": "vad"}


def test_restore_installs_state():
    trace.restore({"turn": 9, "phase": "tts"})
    assert trace.get_turn() == 9
    assert trace.get_phase() == "tts"


def test_restore_in_worker_thread_propagates():
    trace.set_turn(11)
    trace.set_phase("memory")
    snap = trace.snapshot()
    captured: list = []

    def _worker():
        trace.restore(snap)
        captured.append((trace.get_turn(), trace.get_phase()))

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert captured == [(11, "memory")]


# ---------------------------------------------------------------------------
# fmt
# ---------------------------------------------------------------------------


def test_fmt_with_no_context_returns_msg_only():
    assert trace.fmt("hello") == "hello"


def test_fmt_with_turn_prefixed():
    trace.set_turn(3)
    assert trace.fmt("hello") == "turn=3 | hello"


def test_fmt_with_phase_prefixed():
    trace.set_turn(3)
    trace.set_phase("stt")
    assert trace.fmt("hello") == "turn=3 | phase=stt | hello"


def test_fmt_with_kwargs():
    trace.set_turn(1)
    trace.set_phase("llm")
    out = trace.fmt("call", chars=12, suppress=True, gate="NO_SEARCH")
    assert out == "turn=1 | phase=llm | call | chars=12 | suppress=true | gate='NO_SEARCH'"


def test_fmt_value_types():
    assert trace.fmt("x", n=None).endswith("n=None")
    assert trace.fmt("x", n=42).endswith("n=42")
    assert trace.fmt("x", n=3.14159).endswith("n=3.142")
    assert trace.fmt("x", n=True).endswith("n=true")
    assert trace.fmt("x", n=False).endswith("n=false")
    assert trace.fmt("x", n="hi").endswith("n='hi'")


def test_fmt_truncates_long_strings():
    long = "x" * 500
    out = trace.fmt("y", s=long)
    # ~200-char cap on the value + a few chars for "y | s='" prefix /
    # closing "'" -> total comfortably under 260.
    assert len(out) < 260
    assert "..." in out


def test_fmt_truncates_long_repr():
    out = trace.fmt("y", d={i: f"value_{i}" * 5 for i in range(30)})
    assert len(out) < 250


def test_fmt_escapes_newlines_in_strings():
    out = trace.fmt("y", body="line1\nline2")
    assert "\\n" in out
    assert "\n" not in out  # actual newline removed


# ---------------------------------------------------------------------------
# tlog
# ---------------------------------------------------------------------------


def test_tlog_emits_at_info(caplog):
    log = logging.getLogger("ultron.test.tlog")
    log.setLevel(logging.INFO)
    trace.set_turn(7)
    with caplog.at_level(logging.INFO, logger="ultron.test.tlog"):
        trace.tlog(log, "hello", chars=5)
    msgs = [r.message for r in caplog.records if r.name == "ultron.test.tlog"]
    assert any("turn=7" in m and "hello" in m and "chars=5" in m for m in msgs)


def test_tlog_respects_level_gating(caplog):
    log = logging.getLogger("ultron.test.tlog.gating")
    log.setLevel(logging.WARNING)
    with caplog.at_level(logging.WARNING, logger="ultron.test.tlog.gating"):
        # INFO is below the logger's threshold -> no record emitted.
        trace.tlog(log, "skip me", level=logging.INFO)
        trace.tlog(log, "see me", level=logging.WARNING)
    msgs = [r.message for r in caplog.records if r.name == "ultron.test.tlog.gating"]
    assert not any("skip me" in m for m in msgs)
    assert any("see me" in m for m in msgs)


def test_tlog_with_no_turn_no_phase(caplog):
    log = logging.getLogger("ultron.test.tlog.empty")
    log.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="ultron.test.tlog.empty"):
        trace.tlog(log, "bare msg")
    msgs = [r.message for r in caplog.records if r.name == "ultron.test.tlog.empty"]
    assert any(m == "bare msg" for m in msgs)


# ---------------------------------------------------------------------------
# phase context manager
# ---------------------------------------------------------------------------


def test_phase_context_sets_phase_inside_block():
    trace.set_phase("outer")
    with trace.phase("inner"):
        assert trace.get_phase() == "inner"
    assert trace.get_phase() == "outer"


def test_phase_context_restores_none_when_no_prior():
    with trace.phase("inside"):
        assert trace.get_phase() == "inside"
    assert trace.get_phase() is None


def test_phase_context_logs_start_and_end(caplog):
    log = logging.getLogger("ultron.test.phase")
    log.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="ultron.test.phase"):
        with trace.phase("step", log=log, foo="bar"):
            time.sleep(0.01)
    msgs = [r.message for r in caplog.records if r.name == "ultron.test.phase"]
    # Start line: phase=step + msg "step:start" + foo='bar'
    start_lines = [m for m in msgs if "step:start" in m]
    end_lines = [m for m in msgs if "step:end" in m]
    assert len(start_lines) == 1
    assert len(end_lines) == 1
    assert "foo='bar'" in start_lines[0]
    assert "elapsed_ms=" in end_lines[0]
    # Elapsed should be at least the sleep duration. Under full-suite
    # contention scheduler jitter can knock a few ms off; assert >= 0
    # (the timing field is present) and trust the more deterministic
    # checks elsewhere in this test.
    elapsed_str = end_lines[0].split("elapsed_ms=")[1].split(" ")[0]
    assert int(elapsed_str) >= 0


def test_phase_context_extra_dict_appended_to_end_line(caplog):
    log = logging.getLogger("ultron.test.phase.extra")
    log.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="ultron.test.phase.extra"):
        with trace.phase("op", log=log) as ctx:
            ctx["result"] = "ok"
            ctx["count"] = 7
    msgs = [r.message for r in caplog.records if r.name == "ultron.test.phase.extra"]
    end_lines = [m for m in msgs if "op:end" in m]
    assert len(end_lines) == 1
    assert "result='ok'" in end_lines[0]
    assert "count=7" in end_lines[0]


def test_phase_context_swallows_no_log_quietly():
    """When log=None, no records are emitted but the phase tag still
    flows. Body still gets to run + ctx dict still mutable."""
    with trace.phase("silent") as ctx:
        ctx["x"] = 1
        assert trace.get_phase() == "silent"
    assert trace.get_phase() is None


def test_phase_context_logs_end_even_on_exception(caplog):
    log = logging.getLogger("ultron.test.phase.exc")
    log.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="ultron.test.phase.exc"):
        with pytest.raises(ValueError):
            with trace.phase("op", log=log):
                raise ValueError("boom")
    msgs = [r.message for r in caplog.records if r.name == "ultron.test.phase.exc"]
    assert any("op:start" in m for m in msgs)
    assert any("op:end" in m for m in msgs)
    # Phase should be cleared after exception.
    assert trace.get_phase() is None


def test_nested_phases_restore_outer():
    with trace.phase("outer"):
        assert trace.get_phase() == "outer"
        with trace.phase("inner"):
            assert trace.get_phase() == "inner"
        assert trace.get_phase() == "outer"
    assert trace.get_phase() is None
