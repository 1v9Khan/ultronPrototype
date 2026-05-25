"""Tests for ultron.streaming.coordinator."""

from __future__ import annotations

import pytest

from ultron.streaming import coordinator as co


# ---------------------------------------------------------------------------
# Basic stream iteration
# ---------------------------------------------------------------------------

class TestStreamIteration:
    def test_next_chunk_returns_items(self) -> None:
        stream = co.StreamCoordinator([1, 2, 3])
        assert stream.state() is co.StreamState.IDLE
        assert stream.next_chunk() == 1
        assert stream.state() is co.StreamState.STREAMING
        assert stream.next_chunk() == 2
        assert stream.next_chunk() == 3
        assert stream.next_chunk() is None
        assert stream.state() is co.StreamState.COMPLETE

    def test_iterate_helper(self) -> None:
        stream = co.StreamCoordinator(["a", "b", "c"])
        out = list(stream.iterate())
        assert out == ["a", "b", "c"]
        assert stream.state() is co.StreamState.COMPLETE
        assert stream.chunks_emitted() == 3

    def test_wait_for_completion_after_iteration(self) -> None:
        stream = co.StreamCoordinator([1])
        list(stream.iterate())
        assert stream.wait_for_completion(timeout=1.0) is True

    def test_stop_marks_cancelled(self) -> None:
        stream = co.StreamCoordinator([1, 2, 3])
        stream.stop()
        assert stream.next_chunk() is None
        assert stream.state() is co.StreamState.CANCELLED


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_on_usage_fires_for_usage_chunks(self) -> None:
        seen: list[object] = []
        stream = co.StreamCoordinator(
            [
                co.StreamChunk(kind="text", content="hello"),
                co.StreamChunk(kind="usage", content={"in": 10, "out": 5}),
            ],
            on_usage=seen.append,
        )
        list(stream.iterate())
        assert len(seen) == 1
        assert getattr(seen[0], "kind") == "usage"

    def test_on_usage_handles_dict_kind(self) -> None:
        seen: list[object] = []
        stream = co.StreamCoordinator(
            [{"type": "usage", "tokens": 10}, {"type": "text"}],
            on_usage=seen.append,
        )
        list(stream.iterate())
        assert len(seen) == 1
        assert seen[0]["type"] == "usage"  # type: ignore[index]

    def test_on_state_change_fires_through_lifecycle(self) -> None:
        states: list[co.StreamState] = []
        stream = co.StreamCoordinator(
            [1, 2],
            on_state_change=states.append,
        )
        list(stream.iterate())
        # At minimum the transitions IDLE -> STREAMING -> COMPLETE should fire.
        assert co.StreamState.STREAMING in states
        assert co.StreamState.COMPLETE in states

    def test_callback_exception_swallowed(self) -> None:
        def boom(_chunk: object) -> None:
            raise RuntimeError("boom")
        stream = co.StreamCoordinator(
            [co.StreamChunk(kind="usage", content={})],
            on_usage=boom,
        )
        # Should not raise.
        list(stream.iterate())


# ---------------------------------------------------------------------------
# Retry status
# ---------------------------------------------------------------------------

class TestRetryStatus:
    def test_publish_retry_attempt_transitions_state(self) -> None:
        records: list[co.RetryStatus] = []
        stream = co.StreamCoordinator([], on_retry=records.append)
        status = stream.publish_retry_attempt(
            attempt=1, max_attempts=3, delay_seconds=2.0,
            error_snippet="429 rate limit",
        )
        assert status.attempt == 1
        assert status.delay_seconds == 2.0
        assert stream.state() is co.StreamState.RETRYING
        assert stream.last_retry_status() is not None
        assert len(records) == 1

    def test_clear_retry_status_returns_to_streaming(self) -> None:
        stream = co.StreamCoordinator([1])
        # First, get it into STREAMING via a chunk pull.
        stream.next_chunk()
        stream.publish_retry_attempt(
            attempt=1, max_attempts=3, delay_seconds=1.0,
        )
        assert stream.state() is co.StreamState.RETRYING
        stream.clear_retry_status()
        assert stream.state() is co.StreamState.STREAMING
        assert stream.last_retry_status() is None

    def test_retry_status_clamps_error_snippet(self) -> None:
        stream = co.StreamCoordinator([])
        status = stream.publish_retry_attempt(
            attempt=1, max_attempts=2, delay_seconds=0.0,
            error_snippet="x" * 500,
        )
        assert len(status.error_snippet) <= 200

    def test_retry_status_negative_delay_floors_to_zero(self) -> None:
        stream = co.StreamCoordinator([])
        status = stream.publish_retry_attempt(
            attempt=1, max_attempts=2, delay_seconds=-5.0,
        )
        assert status.delay_seconds == 0.0


# ---------------------------------------------------------------------------
# Source iterator failure
# ---------------------------------------------------------------------------

class TestSourceFailure:
    def test_iter_raises_marks_failed(self) -> None:
        class BadSource:
            def __iter__(self):
                raise RuntimeError("init failure")
        stream = co.StreamCoordinator(BadSource())
        assert stream.next_chunk() is None
        assert stream.state() is co.StreamState.FAILED
        assert stream.last_error() is not None

    def test_iteration_raises_marks_failed(self) -> None:
        def gen():
            yield 1
            raise RuntimeError("midstream")
        stream = co.StreamCoordinator(gen())
        # First call returns 1, second raises and transitions to FAILED.
        assert stream.next_chunk() == 1
        assert stream.next_chunk() is None
        assert stream.state() is co.StreamState.FAILED
