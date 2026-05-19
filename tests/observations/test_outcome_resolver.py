"""Tests for :class:`OutcomeResolver`."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest

from ultron.observations import (
    Observation,
    ObservationWriter,
    OutcomeResolver,
    Resolution,
    ResolverSummary,
    get_observation_writer,
    resolve_outcomes,
    set_observation_writer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(epoch: float) -> str:
    """Format an epoch second as an ISO 8601 UTC timestamp."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _write_row(path: Path, **fields: object) -> None:
    """Append one observation row to ``path``.

    Most fields default-to-None; tests supply the ones they care about.
    """
    payload = {
        "event_id": fields.get("event_id"),
        "timestamp": fields.get("timestamp", _iso(0.0)),
        "subsystem": fields.get("subsystem", ""),
        "event_type": fields.get("event_type", ""),
        "parent_event_id": fields.get("parent_event_id"),
        "intent_kind": fields.get("intent_kind"),
        "outcome": fields.get("outcome"),
        "latency_ms": fields.get("latency_ms"),
        "tokens_used": fields.get("tokens_used"),
        "lineage_ids": fields.get("lineage_ids", []),
        "payload_ref": fields.get("payload_ref"),
        "extra": fields.get("extra", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


@pytest.fixture
def capturing_writer(tmp_path: Path) -> Iterator[ObservationWriter]:
    target = tmp_path / "resolutions_out.jsonl"
    writer = ObservationWriter(target, enabled=True)
    previous = get_observation_writer()
    set_observation_writer(writer)
    yield writer
    set_observation_writer(previous)


# ---------------------------------------------------------------------------
# LLM events resolve from own fields
# ---------------------------------------------------------------------------


def test_resolves_llm_stream_completed_to_success(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_a",
        timestamp=_iso(0.0),
        subsystem="llm",
        event_type="generate_stream",
        outcome="unknown_yet",
        extra={"completed": True, "canceled": False, "response_chars": 42},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.resolved_now == 1
    assert summary.resolutions[0].outcome == "success"
    rows = _read_resolution_rows(capturing_writer.path)
    assert len(rows) == 1
    assert rows[0]["parent_event_id"] == "ev_a"
    assert rows[0]["outcome"] == "success"
    assert rows[0]["event_type"] == "outcome_resolution"


def test_resolves_llm_stream_canceled_to_corrected(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_c",
        timestamp=_iso(0.0),
        subsystem="llm",
        event_type="generate_stream",
        outcome="unknown_yet",
        extra={"completed": False, "canceled": True},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.resolutions[0].outcome == "corrected"


def test_resolves_llm_stream_truncated_to_failed(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_t",
        timestamp=_iso(0.0),
        subsystem="llm",
        event_type="generate_stream",
        outcome="unknown_yet",
        extra={"completed": False, "canceled": False},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.resolutions[0].outcome == "failed"


def test_resolves_llm_generate_to_success(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_g",
        timestamp=_iso(0.0),
        subsystem="llm",
        event_type="generate",
        outcome="unknown_yet",
        extra={"streamed": False, "response_chars": 100},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.resolutions[0].outcome == "success"


# ---------------------------------------------------------------------------
# Cross-event correction signals
# ---------------------------------------------------------------------------


def test_routing_event_with_subsequent_cancel_resolves_to_corrected(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_route_1",
        timestamp=_iso(0.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="conversational",
        outcome="unknown_yet",
        extra={"confidence": 0.6},
    )
    _write_row(
        obs_path,
        event_id="ev_route_2",
        timestamp=_iso(5.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="cancel",
        outcome="unknown_yet",
        extra={"confidence": 0.95},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        window_seconds=30.0,
        now_provider=lambda: 1000.0,
    )
    # The conversational event resolves to corrected; the CANCEL itself
    # also resolves (to success, since nothing later corrects it).
    by_id = {r.parent_event_id: r for r in summary.resolutions}
    assert by_id["ev_route_1"].outcome == "corrected"
    assert "CANCEL" in by_id["ev_route_1"].reason
    assert by_id["ev_route_2"].outcome == "success"


def test_routing_event_with_subsequent_not_addressed_resolves_to_corrected(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_route",
        timestamp=_iso(0.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="conversational",
        outcome="unknown_yet",
        extra={},
    )
    _write_row(
        obs_path,
        event_id="ev_addr",
        timestamp=_iso(10.0),
        subsystem="addressing",
        event_type="classify_addressing",
        outcome="unknown_yet",
        extra={"decision": "NOT_ADDRESSED", "classifier_source": "rule"},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    by_id = {r.parent_event_id: r for r in summary.resolutions}
    assert by_id["ev_route"].outcome == "corrected"


def test_routing_event_with_no_signal_resolves_to_success(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_route",
        timestamp=_iso(0.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="conversational",
        outcome="unknown_yet",
        extra={},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.resolutions[0].outcome == "success"


def test_correction_signal_outside_window_is_ignored(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_route",
        timestamp=_iso(0.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="conversational",
        outcome="unknown_yet",
        extra={},
    )
    # CANCEL arrives 120 s later -- outside the 30 s window.
    _write_row(
        obs_path,
        event_id="ev_cancel",
        timestamp=_iso(120.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="cancel",
        outcome="unknown_yet",
        extra={},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        window_seconds=30.0,
        now_provider=lambda: 1000.0,
    )
    by_id = {r.parent_event_id: r for r in summary.resolutions}
    assert by_id["ev_route"].outcome == "success"


# ---------------------------------------------------------------------------
# Deferral + skip semantics
# ---------------------------------------------------------------------------


def test_events_younger_than_min_age_are_deferred(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_young",
        timestamp=_iso(990.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="conversational",
        outcome="unknown_yet",
        extra={},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=30.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.resolved_now == 0
    assert summary.deferred_age == 1
    assert summary.resolutions == []


def test_already_resolved_events_are_skipped(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_a",
        timestamp=_iso(0.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="conversational",
        outcome="unknown_yet",
        extra={},
    )
    _write_row(
        obs_path,
        event_id="res_1",
        timestamp=_iso(5.0),
        subsystem="observations",
        event_type="outcome_resolution",
        parent_event_id="ev_a",
        outcome="success",
        extra={"reason": "previous run"},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.already_resolved == 1
    assert summary.resolved_now == 0


def test_events_with_terminal_outcome_field_are_skipped(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_done",
        timestamp=_iso(0.0),
        subsystem="routing",
        event_type="classify_routing",
        intent_kind="conversational",
        outcome="success",  # already terminal
        extra={},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.resolved_now == 0


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_file_returns_empty_summary(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    summary = resolve_outcomes(
        observations_path=tmp_path / "nope.jsonl",
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.scanned == 0
    assert summary.resolutions == []


def test_malformed_lines_are_skipped(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    obs_path.write_text(
        "not valid json\n"
        + json.dumps({
            "event_id": "ev_ok",
            "timestamp": _iso(0.0),
            "subsystem": "llm",
            "event_type": "generate",
            "outcome": "unknown_yet",
            "extra": {},
        })
        + "\n",
        encoding="utf-8",
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    assert summary.scanned == 1
    assert summary.resolutions[0].outcome == "success"


def test_summary_as_dict_counts_by_outcome(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_ok",
        timestamp=_iso(0.0),
        subsystem="llm",
        event_type="generate_stream",
        outcome="unknown_yet",
        extra={"completed": True, "canceled": False},
    )
    _write_row(
        obs_path,
        event_id="ev_corr",
        timestamp=_iso(1.0),
        subsystem="llm",
        event_type="generate_stream",
        outcome="unknown_yet",
        extra={"completed": False, "canceled": True},
    )
    _write_row(
        obs_path,
        event_id="ev_fail",
        timestamp=_iso(2.0),
        subsystem="llm",
        event_type="generate_stream",
        outcome="unknown_yet",
        extra={"completed": False, "canceled": False},
    )
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    payload = summary.as_dict()
    assert payload["scanned"] == 3
    assert payload["resolved_now"] == 3
    assert payload["by_outcome"] == {"success": 1, "corrected": 1, "failed": 1}


def test_emit_failures_counted(tmp_path: Path) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_ok",
        timestamp=_iso(0.0),
        subsystem="llm",
        event_type="generate",
        outcome="unknown_yet",
        extra={},
    )
    failing_writer = ObservationWriter(tmp_path / "out.jsonl", enabled=False)
    summary = resolve_outcomes(
        observations_path=obs_path,
        writer=failing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    # Resolution was computed but emit returned False (disabled writer).
    assert summary.resolved_now == 1
    assert summary.emitted_failures == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_resolution_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_resolution_to_observation_carries_parent_and_reason() -> None:
    r = Resolution(parent_event_id="abc", outcome="success", reason="why")
    obs = r.to_observation()
    assert obs.event_type == "outcome_resolution"
    assert obs.outcome == "success"
    assert obs.parent_event_id == "abc"
    assert obs.extra["reason"] == "why"


def test_resolver_class_is_reusable(
    tmp_path: Path, capturing_writer: ObservationWriter
) -> None:
    obs_path = tmp_path / "obs.jsonl"
    _write_row(
        obs_path,
        event_id="ev_1",
        timestamp=_iso(0.0),
        subsystem="llm",
        event_type="generate",
        outcome="unknown_yet",
        extra={},
    )
    resolver = OutcomeResolver(
        observations_path=obs_path,
        writer=capturing_writer,
        min_age_seconds=0.0,
        now_provider=lambda: 1000.0,
    )
    first = resolver.resolve()
    assert first.resolved_now == 1
    # The capturing_writer wrote the resolution row to its own file, not
    # back into obs.jsonl, so re-running the resolver against obs_path
    # would re-resolve. That's intentional for tests; in production
    # both files are usually the same path.
    second = resolver.resolve()
    assert second.resolved_now == 1  # repeats since separate files
