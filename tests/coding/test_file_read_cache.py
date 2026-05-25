"""Tests for ultron.coding.file_read_cache."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ultron.coding import file_read_cache as frc


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Ensure the module-level registry is clean per test."""
    frc.reset_file_read_cache_registry()
    yield
    frc.reset_file_read_cache_registry()


# ---------------------------------------------------------------------------
# Basic cache behaviour
# ---------------------------------------------------------------------------

class TestBasic:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        file = tmp_path / "a.txt"
        file.write_text("hello", encoding="utf-8")
        assert cache.maybe_serve_from_cache(file) is None

    def test_record_then_serve(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        file = tmp_path / "a.txt"
        file.write_text("hello", encoding="utf-8")
        cache.record_read(file, "hello")
        hit = cache.maybe_serve_from_cache(file)
        assert hit is not None
        assert hit.content == "hello"
        assert hit.read_count == 2  # record_read=1, cache hit increments to 2
        assert "served from per-session cache" in hit.notice

    def test_repeated_hits_increment_count(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        file = tmp_path / "a.txt"
        file.write_text("x", encoding="utf-8")
        cache.record_read(file, "x")
        for expected in (2, 3, 4):
            hit = cache.maybe_serve_from_cache(file)
            assert hit is not None
            assert hit.read_count == expected

    def test_mtime_change_invalidates(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        file = tmp_path / "a.txt"
        file.write_text("v1", encoding="utf-8")
        cache.record_read(file, "v1")
        # Bump mtime by writing fresh content.
        time.sleep(0.05)
        file.write_text("v2", encoding="utf-8")
        os.utime(file, None)
        assert cache.maybe_serve_from_cache(file) is None

    def test_invalidate_returns_true_when_present(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        file = tmp_path / "a.txt"
        file.write_text("x", encoding="utf-8")
        cache.record_read(file, "x")
        assert cache.invalidate(file) is True
        assert cache.maybe_serve_from_cache(file) is None

    def test_invalidate_returns_false_when_absent(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        assert cache.invalidate(tmp_path / "never.txt") is False

    def test_clear_drops_everything(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        for name in ("a.txt", "b.txt"):
            f = tmp_path / name
            f.write_text("x", encoding="utf-8")
            cache.record_read(f, "x")
        cache.clear()
        assert len(cache) == 0

    def test_len_reports_entry_count(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        assert len(cache) == 0
        f = tmp_path / "a.txt"
        f.write_text("x", encoding="utf-8")
        cache.record_read(f, "x")
        assert len(cache) == 1

    def test_record_read_overwrites_existing_content(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        f = tmp_path / "a.txt"
        f.write_text("old", encoding="utf-8")
        cache.record_read(f, "old")
        f.write_text("new", encoding="utf-8")
        os.utime(f, None)
        cache.record_read(f, "new")
        hit = cache.maybe_serve_from_cache(f)
        assert hit is not None and hit.content == "new"

    def test_record_read_missing_file_no_op(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        # File does not exist; record_read should silently fall through.
        cache.record_read(tmp_path / "ghost.txt", "ghost")
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Capacity / eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_max_entries_evicts_lowest_count(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1", max_entries=2)
        files = []
        for name in ("a.txt", "b.txt", "c.txt"):
            f = tmp_path / name
            f.write_text(name, encoding="utf-8")
            files.append(f)
            cache.record_read(f, name)
        # Once we exceed max_entries, the lowest read_count gets dropped.
        assert len(cache) == 2

    def test_high_count_survives_eviction(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1", max_entries=2)
        a = tmp_path / "a.txt"
        a.write_text("a", encoding="utf-8")
        cache.record_read(a, "a")
        # Promote a's read_count via repeated reads.
        for _ in range(5):
            cache.maybe_serve_from_cache(a)
        b = tmp_path / "b.txt"
        b.write_text("b", encoding="utf-8")
        cache.record_read(b, "b")
        c = tmp_path / "c.txt"
        c.write_text("c", encoding="utf-8")
        cache.record_read(c, "c")
        # b should be the eviction victim because it has the smallest count.
        assert cache.maybe_serve_from_cache(a) is not None
        assert cache.maybe_serve_from_cache(c) is not None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_counts_entries_and_reads(self, tmp_path: Path) -> None:
        cache = frc.FileReadCache(session_id="s1")
        for name in ("a.txt", "b.txt"):
            f = tmp_path / name
            f.write_text(name, encoding="utf-8")
            cache.record_read(f, name)
        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["total_reads"] >= 2


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_returns_same_instance_for_same_session(self) -> None:
        a = frc.get_file_read_cache("alpha")
        b = frc.get_file_read_cache("alpha")
        assert a is b

    def test_get_returns_distinct_instances_per_session(self) -> None:
        a = frc.get_file_read_cache("alpha")
        b = frc.get_file_read_cache("beta")
        assert a is not b

    def test_max_entries_applied_on_first_creation(self) -> None:
        a = frc.get_file_read_cache("alpha", max_entries=5)
        assert a._max_entries == 5
        b = frc.get_file_read_cache("alpha", max_entries=99)
        # Ignored on second call; the first creation wins.
        assert b._max_entries == 5

    def test_reset_clears_registry(self) -> None:
        frc.get_file_read_cache("alpha")
        frc.reset_file_read_cache_registry()
        new = frc.get_file_read_cache("alpha")
        assert new is not None  # but a fresh instance


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_does_not_corrupt(self, tmp_path: Path) -> None:
        import threading
        cache = frc.FileReadCache(session_id="s1")
        f = tmp_path / "a.txt"
        f.write_text("x", encoding="utf-8")

        def worker() -> None:
            for _ in range(10):
                cache.record_read(f, "x")
                cache.maybe_serve_from_cache(f)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        # Cache should still have exactly one entry.
        assert len(cache) == 1
