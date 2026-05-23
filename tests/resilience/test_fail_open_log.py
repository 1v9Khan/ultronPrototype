"""Tests for ultron.resilience.fail_open_log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultron.resilience import fail_open_log


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Each test starts with a clean counter state."""
    fail_open_log.reset_for_testing()
    yield
    fail_open_log.reset_for_testing()


# ---------------------------------------------------------------------------
# record + session_counts
# ---------------------------------------------------------------------------


def test_record_increments_counter() -> None:
    fail_open_log.record("bus_slow_subscriber")
    assert fail_open_log.session_counts() == {"bus_slow_subscriber": 1}


def test_record_accumulates_per_category() -> None:
    for _ in range(3):
        fail_open_log.record("reranker_load_fail")
    fail_open_log.record("bus_slow_subscriber")
    assert fail_open_log.session_counts() == {
        "reranker_load_fail": 3,
        "bus_slow_subscriber": 1,
    }


def test_record_with_reason_does_not_aggregate() -> None:
    """Reason metadata is ignored for now; counter is per-category."""
    fail_open_log.record("bus_slow_subscriber", reason="turn.started")
    fail_open_log.record("bus_slow_subscriber", reason="gate.verdict")
    assert fail_open_log.session_counts() == {"bus_slow_subscriber": 2}


def test_record_unknown_category_still_works() -> None:
    """Open-ended: caller can introduce a new category at any time."""
    fail_open_log.record("brand_new_category")
    assert fail_open_log.session_counts() == {"brand_new_category": 1}


def test_record_never_raises_on_internal_error(monkeypatch) -> None:
    """Counter must not break the wrapping subsystem."""
    # Force the lock to raise by replacing it briefly.
    class _BrokenLock:
        def __enter__(self):
            raise RuntimeError("lock is broken")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(fail_open_log, "_LOCK", _BrokenLock())
    # Must not raise.
    fail_open_log.record("anything")


# ---------------------------------------------------------------------------
# configure + flush_to_disk + previous_session_counts
# ---------------------------------------------------------------------------


def test_configure_resets_counts(tmp_path: Path) -> None:
    log = tmp_path / "fail_open.jsonl"
    fail_open_log.record("bus_slow_subscriber")
    assert fail_open_log.session_counts()

    fail_open_log.configure(log)
    assert fail_open_log.session_counts() == {}


def test_flush_to_disk_writes_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "fail_open.jsonl"
    fail_open_log.configure(log)
    fail_open_log.record("bus_slow_subscriber")
    fail_open_log.record("reranker_load_fail")
    fail_open_log.record("reranker_load_fail")

    fail_open_log.flush_to_disk()

    assert log.exists()
    lines = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["counts"] == {
        "bus_slow_subscriber": 1,
        "reranker_load_fail": 2,
    }
    assert "session_start" in lines[0]
    assert "session_end" in lines[0]
    assert lines[0]["session_end"] >= lines[0]["session_start"]


def test_flush_to_disk_creates_parent_dir(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "dir" / "fail_open.jsonl"
    fail_open_log.configure(log)
    fail_open_log.record("bus_slow_subscriber")

    fail_open_log.flush_to_disk()

    assert log.exists()


def test_flush_to_disk_skips_when_no_path() -> None:
    """configure(None) disables disk persistence; flush is no-op."""
    fail_open_log.configure(None)
    fail_open_log.record("bus_slow_subscriber")

    # Must not raise; nothing on disk.
    fail_open_log.flush_to_disk()


def test_flush_to_disk_skips_empty_session(tmp_path: Path) -> None:
    log = tmp_path / "fail_open.jsonl"
    fail_open_log.configure(log)
    # No records; flush should not create the file.
    fail_open_log.flush_to_disk()
    assert not log.exists()


def test_flush_appends_across_calls(tmp_path: Path) -> None:
    log = tmp_path / "fail_open.jsonl"
    fail_open_log.configure(log)
    fail_open_log.record("category_a")
    fail_open_log.flush_to_disk()

    fail_open_log.configure(log)
    fail_open_log.record("category_b")
    fail_open_log.record("category_b")
    fail_open_log.flush_to_disk()

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["counts"] == {"category_a": 1}
    assert json.loads(lines[1])["counts"] == {"category_b": 2}


def test_previous_session_counts_reads_last(tmp_path: Path) -> None:
    log = tmp_path / "fail_open.jsonl"
    fail_open_log.configure(log)
    fail_open_log.record("category_a")
    fail_open_log.flush_to_disk()
    fail_open_log.configure(log)
    fail_open_log.record("category_b")
    fail_open_log.record("category_b")
    fail_open_log.flush_to_disk()

    fail_open_log.configure(log)
    # The most recent entry is category_b.
    counts = fail_open_log.previous_session_counts()
    assert counts == {"category_b": 2}


def test_previous_session_counts_explicit_path(tmp_path: Path) -> None:
    """When called with an explicit path, ignore the configured path."""
    log = tmp_path / "explicit.jsonl"
    log.write_text(
        json.dumps({"counts": {"x": 5}, "session_start": 0, "session_end": 1}) + "\n",
        encoding="utf-8",
    )
    counts = fail_open_log.previous_session_counts(log)
    assert counts == {"x": 5}


def test_previous_session_counts_missing_file_returns_none(tmp_path: Path) -> None:
    counts = fail_open_log.previous_session_counts(tmp_path / "absent.jsonl")
    assert counts is None


def test_previous_session_counts_empty_file_returns_none(tmp_path: Path) -> None:
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    counts = fail_open_log.previous_session_counts(log)
    assert counts is None


def test_previous_session_counts_blank_lines_only_returns_none(tmp_path: Path) -> None:
    log = tmp_path / "blanks.jsonl"
    log.write_text("\n\n   \n", encoding="utf-8")
    counts = fail_open_log.previous_session_counts(log)
    assert counts is None


def test_previous_session_counts_malformed_returns_none(tmp_path: Path) -> None:
    log = tmp_path / "bad.jsonl"
    log.write_text("this is not json\n", encoding="utf-8")
    counts = fail_open_log.previous_session_counts(log)
    assert counts is None


def test_previous_session_counts_no_counts_field_returns_none(tmp_path: Path) -> None:
    log = tmp_path / "no_counts.jsonl"
    log.write_text(json.dumps({"session_start": 0}) + "\n", encoding="utf-8")
    counts = fail_open_log.previous_session_counts(log)
    assert counts is None


# ---------------------------------------------------------------------------
# render_summary
# ---------------------------------------------------------------------------


def test_render_summary_empty_returns_friendly_string() -> None:
    assert fail_open_log.render_summary({}) == "no fail-open events recorded"
    assert fail_open_log.render_summary(None) == "no fail-open events recorded"


def test_render_summary_sorts_categories() -> None:
    rendered = fail_open_log.render_summary(
        {"z_cat": 1, "a_cat": 5, "m_cat": 3},
    )
    assert rendered == "a_cat=5, m_cat=3, z_cat=1"


def test_render_summary_single_category() -> None:
    assert fail_open_log.render_summary({"bus_slow_subscriber": 3}) == "bus_slow_subscriber=3"


# ---------------------------------------------------------------------------
# KNOWN_CATEGORIES sanity
# ---------------------------------------------------------------------------


def test_known_categories_are_unique_strings() -> None:
    cats = fail_open_log.KNOWN_CATEGORIES
    assert len(cats) == len(set(cats))
    assert all(isinstance(c, str) and c for c in cats)


def test_known_categories_include_bus_slow_subscriber() -> None:
    """The bus watchdog's category must be in the catalog."""
    assert "bus_slow_subscriber" in fail_open_log.KNOWN_CATEGORIES
