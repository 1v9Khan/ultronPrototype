"""Tests for ultron.observations.safe_capture."""

from __future__ import annotations

import asyncio

import pytest

from ultron.observations import safe_capture as sc


@pytest.fixture(autouse=True)
def _reset() -> None:
    sc.reset_safe_capture_stats()
    yield
    sc.reset_safe_capture_stats()


# ---------------------------------------------------------------------------
# safe_capture (sync)
# ---------------------------------------------------------------------------

class TestSafeCapture:
    def test_returns_value_on_success(self) -> None:
        assert sc.safe_capture(lambda: 42) == 42

    def test_forwards_args_and_kwargs(self) -> None:
        def add(a: int, b: int, *, c: int = 0) -> int:
            return a + b + c
        assert sc.safe_capture(add, 1, 2, c=3) == 6

    def test_swallows_exception_and_returns_fallback(self) -> None:
        def boom() -> int:
            raise RuntimeError("nope")
        assert sc.safe_capture(boom, fallback=-1) == -1

    def test_swallows_exception_default_fallback(self) -> None:
        def boom() -> None:
            raise ValueError("x")
        assert sc.safe_capture(boom) is None

    def test_stats_counts_success(self) -> None:
        sc.safe_capture(lambda: 1)
        sc.safe_capture(lambda: 2)
        stats = sc.safe_capture_stats()
        assert stats.total_calls == 2
        assert stats.success_calls == 2
        assert stats.failure_calls == 0

    def test_stats_counts_failure(self) -> None:
        def boom() -> None:
            raise RuntimeError("fail")
        sc.safe_capture(boom, error_context="memory.write")
        stats = sc.safe_capture_stats()
        assert stats.total_calls == 1
        assert stats.failure_calls == 1
        assert stats.last_failure_message is not None
        assert "RuntimeError" in stats.last_failure_message
        assert stats.per_context_failures.get("memory.write") == 1

    def test_per_context_failures_accumulate(self) -> None:
        def boom() -> None:
            raise ValueError("x")
        for _ in range(3):
            sc.safe_capture(boom, error_context="bus.publish")
        stats = sc.safe_capture_stats()
        assert stats.per_context_failures["bus.publish"] == 3

    def test_log_traceback_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        def boom() -> None:
            raise ValueError("fail-with-traceback")
        with caplog.at_level("WARNING", logger="ultron.observations.safe_capture"):
            sc.safe_capture(boom, error_context="test", log_traceback=True)
        # The log line should mention ValueError.
        assert any("ValueError" in rec.getMessage() for rec in caplog.records)

    def test_stats_snapshot_is_a_copy(self) -> None:
        sc.safe_capture(lambda: 1)
        snapshot = sc.safe_capture_stats()
        snapshot.total_calls = 999
        again = sc.safe_capture_stats()
        assert again.total_calls == 1


# ---------------------------------------------------------------------------
# safe_capture_async
# ---------------------------------------------------------------------------

class TestSafeCaptureAsync:
    def test_returns_value_on_success(self) -> None:
        async def f() -> int:
            return 10

        async def main() -> int:
            return await sc.safe_capture_async(f)
        assert asyncio.run(main()) == 10

    def test_forwards_args(self) -> None:
        async def f(a: int, b: int) -> int:
            return a * b

        async def main() -> int:
            return await sc.safe_capture_async(f, 3, 4)
        assert asyncio.run(main()) == 12

    def test_swallows_exception(self) -> None:
        async def f() -> int:
            raise RuntimeError("nope")

        async def main() -> int:
            return await sc.safe_capture_async(f, fallback=-1)
        assert asyncio.run(main()) == -1

    def test_cancelled_error_propagates(self) -> None:
        async def f() -> int:
            raise asyncio.CancelledError()

        async def main() -> int:
            return await sc.safe_capture_async(f)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(main())

    def test_synchronous_callable_also_accepted(self) -> None:
        def sync_fn() -> int:
            return 1

        async def main() -> int:
            return await sc.safe_capture_async(sync_fn)
        assert asyncio.run(main()) == 1


# ---------------------------------------------------------------------------
# safe_capture_decorator
# ---------------------------------------------------------------------------

class TestSafeCaptureDecorator:
    def test_sync_decorator_swallows(self) -> None:
        @sc.safe_capture_decorator(error_context="x", fallback=-1)
        def f() -> int:
            raise RuntimeError()
        assert f() == -1

    def test_sync_decorator_returns_value(self) -> None:
        @sc.safe_capture_decorator(error_context="x")
        def f() -> int:
            return 5
        assert f() == 5

    def test_async_decorator_swallows(self) -> None:
        @sc.safe_capture_decorator(error_context="x", fallback=-1)
        async def f() -> int:
            raise RuntimeError()

        async def main() -> int:
            return await f()
        assert asyncio.run(main()) == -1

    def test_async_decorator_returns_value(self) -> None:
        @sc.safe_capture_decorator(error_context="x")
        async def f() -> int:
            return 9

        async def main() -> int:
            return await f()
        assert asyncio.run(main()) == 9

    def test_decorator_preserves_name(self) -> None:
        @sc.safe_capture_decorator(error_context="x")
        def my_function() -> int:
            return 1
        assert my_function.__name__ == "my_function"
