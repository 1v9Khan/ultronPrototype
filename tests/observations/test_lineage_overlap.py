"""Tests for :mod:`ultron.observations.lineage_overlap`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultron.observations import (
    LineageOverlap,
    ObservationWriter,
    compute_lineage_overlap,
    emit_lineage_usage_rows,
)


# ---------------------------------------------------------------------------
# compute_lineage_overlap
# ---------------------------------------------------------------------------


def test_empty_response_marks_nothing_used() -> None:
    overlaps = compute_lineage_overlap("", {"a": "some memory content", "b": ""})
    assert all(o.used is False for o in overlaps)
    assert [o.lineage_id for o in overlaps] == ["a", "b"]


def test_strong_word_overlap_marks_used() -> None:
    response = (
        "The Eiffel Tower is in Paris, the capital of France."
    )
    memory_contents = {
        "matches": "We discussed the Eiffel Tower in Paris earlier today.",
        "unrelated": "Recipe for sourdough bread starter.",
    }
    overlaps = compute_lineage_overlap(response, memory_contents)
    by_id = {o.lineage_id: o for o in overlaps}
    assert by_id["matches"].used is True
    # Words shared: eiffel, tower, in, paris, the (some may be filtered
    # by tokeniser; >=3 is the floor).
    assert by_id["matches"].word_overlap >= 3
    assert by_id["unrelated"].used is False


def test_long_literal_substring_marks_used_even_when_words_few() -> None:
    response = (
        "I remember the codename was 'PROJECT_LIGHTNING_FORK' as you said."
    )
    memory_contents = {
        "quotation": "Codename: PROJECT_LIGHTNING_FORK",
    }
    overlaps = compute_lineage_overlap(response, memory_contents)
    assert overlaps[0].used is True
    assert len(overlaps[0].longest_substring) >= 12


def test_thresholds_are_tunable() -> None:
    response = "alpha beta"
    memory_contents = {"a": "alpha gamma delta epsilon"}
    overlaps = compute_lineage_overlap(
        response, memory_contents, min_word_overlap=10, min_substring_chars=999,
    )
    # Single shared word "alpha" is below the bumped threshold.
    assert overlaps[0].used is False


def test_overlap_preserves_input_order() -> None:
    response = "x"
    memory_contents = {"c": "x", "a": "y", "b": "z"}
    overlaps = compute_lineage_overlap(response, memory_contents)
    assert [o.lineage_id for o in overlaps] == ["c", "a", "b"]


def test_overlap_as_dict_carries_diagnostics() -> None:
    o = LineageOverlap(
        lineage_id="m_1",
        used=True,
        word_overlap=4,
        longest_substring="example substring",
    )
    payload = o.as_dict()
    assert payload["lineage_id"] == "m_1"
    assert payload["used"] is True
    assert payload["word_overlap"] == 4
    assert payload["longest_substring_len"] == len("example substring")
    assert payload["longest_substring_preview"] == "example substring"


def test_longest_substring_preview_truncated() -> None:
    long = "x" * 200
    o = LineageOverlap(
        lineage_id="m", used=True, word_overlap=0, longest_substring=long,
    )
    assert len(o.as_dict()["longest_substring_preview"]) == 60


# ---------------------------------------------------------------------------
# emit_lineage_usage_rows
# ---------------------------------------------------------------------------


def test_emit_writes_one_row_per_overlap(tmp_path: Path) -> None:
    writer = ObservationWriter(tmp_path / "lineage.jsonl", enabled=True)
    overlaps = [
        LineageOverlap("a", used=True, word_overlap=4, longest_substring=""),
        LineageOverlap("b", used=False, word_overlap=0, longest_substring=""),
    ]
    summary = emit_lineage_usage_rows("parent_xyz", overlaps, writer=writer)
    assert summary.emitted == 2
    assert summary.used_count == 1
    assert summary.failed == 0

    rows = [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(r["event_type"] == "lineage_usage" for r in rows)
    assert all(r["parent_event_id"] == "parent_xyz" for r in rows)
    by_id = {r["extra"]["lineage_id"]: r for r in rows}
    assert by_id["a"]["outcome"] == "success"
    assert by_id["a"]["extra"]["used"] is True
    assert by_id["b"]["outcome"] == "unknown_yet"
    assert by_id["b"]["extra"]["used"] is False


def test_emit_handles_disabled_writer(tmp_path: Path) -> None:
    writer = ObservationWriter(tmp_path / "noop.jsonl", enabled=False)
    overlaps = [LineageOverlap("a", used=True, word_overlap=4, longest_substring="")]
    summary = emit_lineage_usage_rows("p", overlaps, writer=writer)
    # Disabled writer returns False from emit; summary records as failure.
    assert summary.emitted == 0
    assert summary.failed == 1
    # used_count counts the OVERLAP used flag, not the emit result, so
    # it still increments.
    assert summary.used_count == 1


def test_emit_empty_overlap_list_is_noop(tmp_path: Path) -> None:
    writer = ObservationWriter(tmp_path / "out.jsonl", enabled=True)
    summary = emit_lineage_usage_rows("p", [], writer=writer)
    assert summary.emitted == 0
    assert summary.used_count == 0
    assert summary.failed == 0
    assert not writer.path.exists()
