"""Tests for the eval harness (``scripts/eval_harness.py``).

The harness is intentionally classifier-only, so these tests run against
the real routing/addressing/web-gate classifiers; no model loads.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = PROJECT_ROOT / "scripts" / "eval_harness.py"

spec = importlib.util.spec_from_file_location("ultron_eval_harness", HARNESS_PATH)
assert spec is not None and spec.loader is not None
eval_harness = importlib.util.module_from_spec(spec)
sys.modules.setdefault(spec.name, eval_harness)
spec.loader.exec_module(eval_harness)


# ---------------------------------------------------------------------------
# parse_corpus_row + load_corpus
# ---------------------------------------------------------------------------


def test_parse_corpus_row_minimal_required_fields() -> None:
    row = eval_harness.parse_corpus_row(
        {"id": "x", "utterance": "hello"}
    )
    assert row.id == "x"
    assert row.utterance == "hello"
    assert row.expected_routing_kind is None
    assert row.expected_addressing is None
    assert row.expected_web_gate is None
    assert row.has_active_coding_task is False
    assert row.has_pending_clarification is False
    assert row.seconds_since_response == 0.0
    assert row.tags == ()
    assert row.notes == ""


def test_parse_corpus_row_full_payload_normalises() -> None:
    row = eval_harness.parse_corpus_row(
        {
            "id": "full",
            "utterance": "open chrome on monitor 2",
            "expected_routing_kind": "app_launch",
            "expected_addressing": "ADDRESSED",
            "expected_web_gate": "NO_SEARCH",
            "has_active_coding_task": True,
            "has_pending_clarification": True,
            "seconds_since_response": 3.5,
            "tags": ["a", "b"],
            "notes": "explanatory text",
        }
    )
    assert row.expected_routing_kind == "app_launch"
    assert row.expected_addressing == "ADDRESSED"
    assert row.expected_web_gate == "NO_SEARCH"
    assert row.has_active_coding_task is True
    assert row.has_pending_clarification is True
    assert row.seconds_since_response == 3.5
    assert row.tags == ("a", "b")
    assert row.notes == "explanatory text"


@pytest.mark.parametrize(
    "payload, error_marker",
    [
        ({}, "missing string 'id'"),
        ({"id": ""}, "missing string 'id'"),
        ({"id": 7, "utterance": "x"}, "missing string 'id'"),
        ({"id": "x"}, "missing string 'utterance'"),
        ({"id": "x", "utterance": 5}, "missing string 'utterance'"),
        ({"id": "x", "utterance": "y", "tags": "not_a_list"}, "'tags' must be a list"),
    ],
)
def test_parse_corpus_row_rejects_invalid(payload, error_marker) -> None:
    with pytest.raises(ValueError, match=error_marker):
        eval_harness.parse_corpus_row(payload)


def test_parse_corpus_row_empty_expected_strings_normalised_to_none() -> None:
    row = eval_harness.parse_corpus_row(
        {
            "id": "x",
            "utterance": "y",
            "expected_routing_kind": "",
            "expected_addressing": None,
        }
    )
    assert row.expected_routing_kind is None
    assert row.expected_addressing is None


def test_load_corpus_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(
        "\n"
        "# leading comment\n"
        '{"id": "a", "utterance": "x"}\n'
        "\n"
        "# trailing comment\n"
        '{"id": "b", "utterance": "y"}\n',
        encoding="utf-8",
    )
    rows = eval_harness.load_corpus(corpus)
    assert [r.id for r in rows] == ["a", "b"]


def test_load_corpus_rejects_duplicate_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "dup.jsonl"
    corpus.write_text(
        '{"id": "a", "utterance": "x"}\n'
        '{"id": "a", "utterance": "y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        eval_harness.load_corpus(corpus)


def test_load_corpus_reports_line_number_on_bad_json(tmp_path: Path) -> None:
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text(
        '{"id": "a", "utterance": "x"}\n'
        "this is not json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSON"):
        eval_harness.load_corpus(corpus)


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


def _row(id: str, tags: tuple[str, ...] = ()) -> eval_harness.CorpusRow:
    return eval_harness.CorpusRow(
        id=id,
        utterance="x",
        expected_routing_kind=None,
        expected_addressing=None,
        expected_web_gate=None,
        has_active_coding_task=False,
        has_pending_clarification=False,
        seconds_since_response=0.0,
        tags=tags,
        notes="",
    )


def test_filter_rows_no_tag_returns_all() -> None:
    rows = [_row("a", ("x",)), _row("b", ("y",))]
    filtered = eval_harness.filter_rows(rows, tag=None)
    assert [r.id for r in filtered] == ["a", "b"]


def test_filter_rows_with_tag_filters() -> None:
    rows = [_row("a", ("x", "y")), _row("b", ("y",)), _row("c", ("z",))]
    filtered = eval_harness.filter_rows(rows, tag="y")
    assert [r.id for r in filtered] == ["a", "b"]


# ---------------------------------------------------------------------------
# Dimension scorers (real classifiers, no models)
# ---------------------------------------------------------------------------


def test_score_routing_marks_skipped_when_no_label() -> None:
    rows = [_row("unlabeled")]
    score = eval_harness.score_routing(rows)
    assert score.total == 0
    assert score.skipped == 1


def test_score_routing_counts_correct_and_failures() -> None:
    rows = [
        eval_harness.parse_corpus_row(
            {
                "id": "ok",
                "utterance": "tell me a joke",
                "expected_routing_kind": "conversational",
            }
        ),
        eval_harness.parse_corpus_row(
            {
                "id": "wrong",
                "utterance": "tell me a joke",
                "expected_routing_kind": "code_task",
            }
        ),
    ]
    score = eval_harness.score_routing(rows)
    assert score.total == 2
    assert score.correct == 1
    assert len(score.failures) == 1
    assert score.failures[0]["id"] == "wrong"
    assert score.failures[0]["expected"] == "code_task"
    assert score.failures[0]["actual"] == "conversational"


def test_score_addressing_handles_none_rule_hit() -> None:
    rows = [
        eval_harness.parse_corpus_row(
            {
                "id": "none_expected",
                "utterance": "the quick brown fox jumps",
                "expected_addressing": "NONE",
            }
        ),
        eval_harness.parse_corpus_row(
            {
                "id": "addressed",
                "utterance": "ultron, open chrome",
                "expected_addressing": "ADDRESSED",
            }
        ),
    ]
    score = eval_harness.score_addressing(rows)
    assert score.total == 2
    assert score.correct == 2


def test_score_web_gate_recognises_rule_decisions() -> None:
    rows = [
        eval_harness.parse_corpus_row(
            {
                "id": "search_yes",
                "utterance": "current weather in belmont",
                "expected_web_gate": "SEARCH",
            }
        ),
        eval_harness.parse_corpus_row(
            {
                "id": "personal_no",
                "utterance": "what did i say earlier today",
                "expected_web_gate": "NO_SEARCH",
            }
        ),
    ]
    score = eval_harness.score_web_gate(rows)
    assert score.total == 2
    assert score.correct == 2


# ---------------------------------------------------------------------------
# build_report + gates
# ---------------------------------------------------------------------------


def test_build_report_aggregates_dimensions_and_gates() -> None:
    rows = [
        eval_harness.parse_corpus_row(
            {
                "id": "r1",
                "utterance": "tell me a joke",
                "expected_routing_kind": "conversational",
            }
        ),
    ]
    report = eval_harness.build_report(
        rows,
        ("routing",),
        gates={"routing_kind_accuracy": 0.5},
        elapsed_seconds=0.001,
    )
    assert report["corpus_size"] == 1
    assert report["dimensions_run"] == ["routing"]
    assert report["overall_pass"] is True
    assert "routing" in report["results"]
    assert "routing_kind_accuracy" in report["gates"]
    assert report["gates"]["routing_kind_accuracy"]["passed"] is True


def test_build_report_fails_overall_when_gate_below_threshold() -> None:
    rows = [
        eval_harness.parse_corpus_row(
            {
                "id": "r1",
                "utterance": "tell me a joke",
                "expected_routing_kind": "code_task",
            }
        ),
    ]
    report = eval_harness.build_report(
        rows, ("routing",), gates={"routing_kind_accuracy": 0.95}
    )
    assert report["overall_pass"] is False
    assert report["gates"]["routing_kind_accuracy"]["passed"] is False


def test_build_report_empty_scored_set_passes_gate() -> None:
    # Row has no expected_routing_kind so routing is skipped.
    rows = [
        eval_harness.parse_corpus_row({"id": "r1", "utterance": "tell me a joke"})
    ]
    report = eval_harness.build_report(rows, ("routing",))
    assert report["gates"]["routing_kind_accuracy"]["passed"] is True


def test_build_report_unknown_dimension_silently_skipped() -> None:
    rows = [_row("a")]
    report = eval_harness.build_report(rows, ("not_a_real_dimension",))
    assert report["results"] == {}
    assert report["overall_pass"] is True


# ---------------------------------------------------------------------------
# format_console_summary + write_report
# ---------------------------------------------------------------------------


def test_format_console_summary_includes_dimensions_and_gates() -> None:
    rows = [
        eval_harness.parse_corpus_row(
            {
                "id": "r1",
                "utterance": "tell me a joke",
                "expected_routing_kind": "conversational",
            }
        )
    ]
    report = eval_harness.build_report(rows, ("routing",))
    summary = eval_harness.format_console_summary(report)
    assert "routing" in summary
    assert "overall:" in summary
    assert "routing_kind_accuracy" in summary


def test_format_console_summary_verbose_lists_failures() -> None:
    rows = [
        eval_harness.parse_corpus_row(
            {
                "id": "r1",
                "utterance": "tell me a joke",
                "expected_routing_kind": "code_task",
            }
        )
    ]
    report = eval_harness.build_report(rows, ("routing",))
    summary = eval_harness.format_console_summary(report, verbose=True)
    assert "[r1]" in summary
    assert "tell me a joke" in summary


def test_write_report_creates_parent_and_round_trips(tmp_path: Path) -> None:
    report = {"hello": "world"}
    target = tmp_path / "deep" / "report.json"
    written = eval_harness.write_report(report, target)
    assert written == target
    assert json.loads(target.read_text(encoding="utf-8")) == report


# ---------------------------------------------------------------------------
# Real corpus baseline
# ---------------------------------------------------------------------------


def test_real_corpus_loads_and_has_diverse_labels() -> None:
    corpus_path = PROJECT_ROOT / "tests" / "eval" / "corpus.jsonl"
    rows = eval_harness.load_corpus(corpus_path)
    assert len(rows) >= 50
    routing_labels = {r.expected_routing_kind for r in rows if r.expected_routing_kind}
    addressing_labels = {r.expected_addressing for r in rows if r.expected_addressing}
    web_gate_labels = {r.expected_web_gate for r in rows if r.expected_web_gate}
    # Sanity checks: corpus covers at least the obvious axes.
    assert {"conversational", "code_task", "progress_query", "app_launch"} <= routing_labels
    assert {"ADDRESSED", "NOT_ADDRESSED"} <= addressing_labels
    assert {"SEARCH", "NO_SEARCH"} <= web_gate_labels


def test_real_corpus_meets_default_gates() -> None:
    """The shipped corpus is the regression baseline.

    If a classifier change drops one of the per-dimension accuracies below
    the default gate, the harness will fail and this test will fail with
    it -- forcing the change-author to either fix the classifier or
    update the corpus + gate threshold deliberately.
    """
    corpus_path = PROJECT_ROOT / "tests" / "eval" / "corpus.jsonl"
    rows = eval_harness.load_corpus(corpus_path)
    report = eval_harness.build_report(
        rows, eval_harness.KNOWN_DIMENSIONS
    )
    assert report["overall_pass"] is True, (
        "real corpus baseline failed at least one gate; "
        f"summary:\n{eval_harness.format_console_summary(report, verbose=True)}"
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_smoke_returns_zero_when_corpus_passes(tmp_path: Path) -> None:
    corpus = tmp_path / "tiny.jsonl"
    corpus.write_text(
        '{"id": "ok", "utterance": "tell me a joke", "expected_routing_kind": "conversational"}\n',
        encoding="utf-8",
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = eval_harness.main(
            [
                "--corpus", str(corpus),
                "--dimensions", "routing",
                "--no-write",
            ]
        )
    out = buf.getvalue()
    assert rc == 0
    assert "overall: PASS" in out


def test_cli_smoke_returns_one_on_gate_failure(tmp_path: Path) -> None:
    corpus = tmp_path / "fail.jsonl"
    corpus.write_text(
        '{"id": "wrong", "utterance": "tell me a joke", "expected_routing_kind": "code_task"}\n',
        encoding="utf-8",
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = eval_harness.main(
            [
                "--corpus", str(corpus),
                "--dimensions", "routing",
                "--no-write",
            ]
        )
    out = buf.getvalue()
    assert rc == 1
    assert "overall: FAIL" in out


def test_cli_smoke_writes_report_to_path(tmp_path: Path) -> None:
    corpus = tmp_path / "ok.jsonl"
    corpus.write_text(
        '{"id": "ok", "utterance": "tell me a joke", "expected_routing_kind": "conversational"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = eval_harness.main(
            [
                "--corpus", str(corpus),
                "--dimensions", "routing",
                "--output", str(output),
                "--quiet",
            ]
        )
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["corpus_size"] == 1
    assert payload["overall_pass"] is True


def test_cli_smoke_missing_corpus_returns_two(tmp_path: Path) -> None:
    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        rc = eval_harness.main(
            [
                "--corpus", str(tmp_path / "missing.jsonl"),
                "--dimensions", "routing",
                "--no-write",
            ]
        )
    assert rc == 2
    assert "corpus not found" in buf_err.getvalue()


def test_cli_smoke_rejects_unknown_dimension(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(
        '{"id": "x", "utterance": "y"}\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc_info:
        eval_harness.main(
            [
                "--corpus", str(corpus),
                "--dimensions", "not_a_dimension",
                "--no-write",
            ]
        )
    assert exc_info.value.code != 0


def test_cli_filter_tag_narrows_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "tagged.jsonl"
    corpus.write_text(
        '{"id": "kept", "utterance": "tell me a joke", "expected_routing_kind": "conversational", "tags": ["keepme"]}\n'
        '{"id": "skipped", "utterance": "tell me a joke", "expected_routing_kind": "code_task", "tags": ["dropme"]}\n',
        encoding="utf-8",
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = eval_harness.main(
            [
                "--corpus", str(corpus),
                "--dimensions", "routing",
                "--filter-tag", "keepme",
                "--no-write",
            ]
        )
    # Only the kept row scored -> 1/1 correct -> PASS
    assert rc == 0
    assert "overall: PASS" in buf.getvalue()
