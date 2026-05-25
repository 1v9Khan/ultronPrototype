"""Tests for ultron.streaming.window."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.streaming import window as w


class _Clock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# is_compiling_output
# ---------------------------------------------------------------------------

class TestIsCompiling:
    def test_matches_common_markers(self) -> None:
        for marker in ("Compiling foo", "BUILDING images", "Bundling app"):
            assert w.is_compiling_output(marker) is True

    def test_empty_returns_false(self) -> None:
        assert w.is_compiling_output("") is False
        assert w.is_compiling_output(None) is False  # type: ignore[arg-type]

    def test_unrelated_line(self) -> None:
        assert w.is_compiling_output("hello world") is False


# ---------------------------------------------------------------------------
# Basic feed + flush
# ---------------------------------------------------------------------------

class TestFeed:
    def test_feed_below_budget_does_not_flush(self) -> None:
        flushed: list[str] = []
        writer = w.WindowedOutputWriter(
            on_flush=flushed.append, line_budget=10, byte_budget=10_000,
            debounce_ms=100, clock=_Clock(),
        )
        writer.feed_line("hello")
        assert flushed == []

    def test_feed_at_line_budget_flushes(self) -> None:
        flushed: list[str] = []
        writer = w.WindowedOutputWriter(
            on_flush=flushed.append, line_budget=2, byte_budget=10_000,
            debounce_ms=100, clock=_Clock(),
        )
        writer.feed_line("a")
        assert flushed == []
        writer.feed_line("b")
        assert flushed == ["a\nb"]

    def test_feed_at_byte_budget_flushes(self) -> None:
        flushed: list[str] = []
        writer = w.WindowedOutputWriter(
            on_flush=flushed.append, line_budget=100, byte_budget=10,
            debounce_ms=100, clock=_Clock(),
        )
        writer.feed_line("abcdefghij")  # 10 chars + 1 newline = 11 bytes
        assert flushed and "abcdef" in flushed[0]

    def test_flush_forces_emit(self) -> None:
        flushed: list[str] = []
        writer = w.WindowedOutputWriter(
            on_flush=flushed.append, line_budget=10, debounce_ms=1000, clock=_Clock(),
        )
        writer.feed_line("only")
        assert writer.flush() is True
        assert flushed == ["only"]
        assert writer.flush() is False  # empty buffer; no-op

    def test_maybe_flush_respects_debounce(self) -> None:
        flushed: list[str] = []
        clock = _Clock()
        writer = w.WindowedOutputWriter(
            on_flush=flushed.append, line_budget=10, debounce_ms=100, clock=clock,
        )
        writer.feed_line("x")
        # Immediately calling maybe_flush should fire (last_flush_at=0).
        assert writer.maybe_flush() is True
        writer.feed_line("y")
        # Without advancing the clock, the debounce window should block.
        assert writer.maybe_flush() is False
        clock.advance(0.2)
        assert writer.maybe_flush() is True

    def test_none_line_ignored(self) -> None:
        writer = w.WindowedOutputWriter(line_budget=2, debounce_ms=100, clock=_Clock())
        assert writer.feed_line(None) is False  # type: ignore[arg-type]
        assert writer.total_lines() == 0

    def test_crlf_stripped(self) -> None:
        flushed: list[str] = []
        writer = w.WindowedOutputWriter(
            on_flush=flushed.append, line_budget=1, debounce_ms=100, clock=_Clock(),
        )
        writer.feed_line("hello\r\n")
        assert flushed == ["hello"]


# ---------------------------------------------------------------------------
# Overflow / spillover
# ---------------------------------------------------------------------------

class TestOverflow:
    def test_spillover_kicks_in_at_threshold(self, tmp_path: Path) -> None:
        writer = w.WindowedOutputWriter(
            line_budget=10000, byte_budget=10_000_000,
            spill_line_threshold=5, spill_byte_threshold=10_000_000,
            overflow_dir=tmp_path / "of",
            clock=_Clock(),
        )
        for i in range(10):
            writer.feed_line(f"line-{i}")
        assert writer.spilled() is True
        snap = writer.snapshot()
        assert snap.overflow_path is not None
        assert snap.overflow_path.parent == (tmp_path / "of")

    def test_snapshot_renders_with_marker(self, tmp_path: Path) -> None:
        writer = w.WindowedOutputWriter(
            line_budget=10000, byte_budget=10_000_000,
            spill_line_threshold=3, head_tail_lines=2,
            overflow_dir=tmp_path / "of", clock=_Clock(),
        )
        for i in range(6):
            writer.feed_line(f"line-{i}")
        writer.flush()
        rendered = writer.snapshot().render()
        assert "elided" in rendered

    def test_total_counters_grow(self) -> None:
        writer = w.WindowedOutputWriter(line_budget=10, debounce_ms=100, clock=_Clock())
        for _ in range(5):
            writer.feed_line("x")
        assert writer.total_lines() == 5
        assert writer.total_bytes() > 0

    def test_close_emits_remaining(self, tmp_path: Path) -> None:
        flushed: list[str] = []
        writer = w.WindowedOutputWriter(
            on_flush=flushed.append, line_budget=10, clock=_Clock(),
        )
        writer.feed_line("only")
        writer.close()
        assert flushed == ["only"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_line_budget(self) -> None:
        with pytest.raises(ValueError):
            w.WindowedOutputWriter(line_budget=0)

    def test_invalid_byte_budget(self) -> None:
        with pytest.raises(ValueError):
            w.WindowedOutputWriter(byte_budget=0)

    def test_invalid_head_tail_lines(self) -> None:
        with pytest.raises(ValueError):
            w.WindowedOutputWriter(head_tail_lines=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_defaults_match_documented_values(self) -> None:
        assert w.DEFAULT_LINE_BUDGET == 20
        assert w.DEFAULT_BYTE_BUDGET == 2 * 1024
        assert w.DEFAULT_DEBOUNCE_MS == 100
        assert w.DEFAULT_HEAD_TAIL_LINES == 100
        assert w.DEFAULT_SPILL_LINE_THRESHOLD == 1000
        assert w.DEFAULT_SPILL_BYTE_THRESHOLD == 512 * 1024

    def test_compiling_markers_lowercase(self) -> None:
        for m in w.COMPILING_MARKERS:
            assert m == m.lower()
