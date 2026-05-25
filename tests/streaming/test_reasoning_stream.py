"""Tests for ultron.streaming.reasoning_stream."""

from __future__ import annotations

import pytest

from ultron.streaming import reasoning_stream as rs


class _Clock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# Reasoning chunks accumulate then finalise
# ---------------------------------------------------------------------------

class TestDemultiplex:
    def test_reasoning_then_text_finalises_block(self) -> None:
        chunks: list[rs.ReasoningChunkEvent] = []
        finalised: list[rs.ReasoningFinalisedEvent] = []
        text: list[str] = []
        demux = rs.ReasoningDemultiplexer(
            on_reasoning_chunk=chunks.append,
            on_reasoning_finalised=finalised.append,
            on_text_chunk=text.append,
            clock=_Clock(),
        )
        demux.feed_reasoning("first thought", signature="sig-1")
        demux.feed_reasoning("second thought")
        demux.feed_text("here is the answer")
        assert text == ["here is the answer"]
        assert len(chunks) == 2
        assert len(finalised) == 1
        block = finalised[0]
        assert block.full_text == "first thoughtsecond thought"
        assert block.chunk_count == 2
        assert block.signature == "sig-1"

    def test_no_pending_text_emits_only_text(self) -> None:
        text: list[str] = []
        finalised: list[rs.ReasoningFinalisedEvent] = []
        demux = rs.ReasoningDemultiplexer(
            on_reasoning_finalised=finalised.append,
            on_text_chunk=text.append,
            clock=_Clock(),
        )
        demux.feed_text("hello")
        assert text == ["hello"]
        assert finalised == []

    def test_drop_reasoning_suppresses(self) -> None:
        chunks: list[rs.ReasoningChunkEvent] = []
        finalised: list[rs.ReasoningFinalisedEvent] = []
        text: list[str] = []
        demux = rs.ReasoningDemultiplexer(
            on_reasoning_chunk=chunks.append,
            on_reasoning_finalised=finalised.append,
            on_text_chunk=text.append,
            drop_reasoning=True,
            clock=_Clock(),
        )
        demux.feed_reasoning("ignored thought")
        demux.feed_text("answer")
        assert chunks == []
        assert finalised == []  # nothing pending → no finalisation
        assert text == ["answer"]

    def test_set_drop_reasoning_dynamic(self) -> None:
        text: list[str] = []
        chunks: list[rs.ReasoningChunkEvent] = []
        demux = rs.ReasoningDemultiplexer(
            on_reasoning_chunk=chunks.append,
            on_text_chunk=text.append,
            clock=_Clock(),
        )
        demux.feed_reasoning("first")
        demux.set_drop_reasoning(True)
        demux.feed_reasoning("dropped")
        demux.feed_text("done")
        assert len(chunks) == 1  # only the first reasoning chunk

    def test_finalise_method_returns_event(self) -> None:
        finalised: list[rs.ReasoningFinalisedEvent] = []
        demux = rs.ReasoningDemultiplexer(
            on_reasoning_finalised=finalised.append,
            clock=_Clock(),
        )
        demux.feed_reasoning("a")
        event = demux.finalise()
        assert event is not None
        assert event.full_text == "a"
        assert len(finalised) == 1

    def test_finalise_without_pending_returns_none(self) -> None:
        demux = rs.ReasoningDemultiplexer(clock=_Clock())
        assert demux.finalise() is None

    def test_callback_exception_does_not_propagate(self) -> None:
        def boom(_event: rs.ReasoningChunkEvent) -> None:
            raise RuntimeError()
        demux = rs.ReasoningDemultiplexer(
            on_reasoning_chunk=boom,
            clock=_Clock(),
        )
        # Should not raise.
        demux.feed_reasoning("anything")
        assert demux.has_pending_reasoning() is True

    def test_block_count_increments(self) -> None:
        demux = rs.ReasoningDemultiplexer(clock=_Clock())
        demux.feed_reasoning("a")
        demux.feed_text("b")
        demux.feed_reasoning("c")
        demux.feed_text("d")
        assert demux.reasoning_blocks_finalised() == 2

    def test_has_pending_reasoning(self) -> None:
        demux = rs.ReasoningDemultiplexer(clock=_Clock())
        assert demux.has_pending_reasoning() is False
        demux.feed_reasoning("a")
        assert demux.has_pending_reasoning() is True
        demux.feed_text("b")
        assert demux.has_pending_reasoning() is False
