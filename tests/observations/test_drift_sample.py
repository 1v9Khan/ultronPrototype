"""Tests for observe_llm_thinking_drift_sample.

The drift sample is the offline-review surface for the
``enable_thinking=False`` voice-path default. It records the
user_text + final response on a sampled subset so a regression
class can be spotted before it hits a user.

The sampling itself (dice roll against
``llm.enable_thinking_drift_sample_rate``) lives in the orchestrator;
this file just verifies the helper itself emits a well-formed row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from ultron.observations import (
    ObservationWriter,
    get_observation_writer,
    observe_llm_thinking_drift_sample,
    set_observation_writer,
)


@pytest.fixture
def capturing_writer(tmp_path: Path) -> Iterator[ObservationWriter]:
    target = tmp_path / "obs.jsonl"
    writer = ObservationWriter(target, enabled=True)
    previous = get_observation_writer()
    set_observation_writer(writer)
    yield writer
    set_observation_writer(previous)


def _read_rows(writer: ObservationWriter) -> list[dict]:
    if not writer.path.exists():
        return []
    return [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_emits_one_row_with_texts(capturing_writer: ObservationWriter) -> None:
    event_id = observe_llm_thinking_drift_sample(
        user_text="what is 7 times 8",
        response_text="56",
        user_message_len=19,
        response_message_len=2,
    )
    assert event_id is not None

    rows = _read_rows(capturing_writer)
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "llm"
    assert row["event_type"] == "thinking_drift_sample"
    assert row["outcome"] == "unknown_yet"
    extra = row["extra"]
    assert extra["user_text"] == "what is 7 times 8"
    assert extra["response_text"] == "56"
    assert extra["enable_thinking"] is False
    assert extra["user_message_len"] == 19
    assert extra["response_message_len"] == 2


def test_emit_with_parent_event_id(capturing_writer: ObservationWriter) -> None:
    """parent_event_id chains the drift sample to the originating llm_call row."""
    event_id = observe_llm_thinking_drift_sample(
        user_text="ping",
        response_text="pong",
        user_message_len=4,
        response_message_len=4,
        parent_event_id="abc123def456",
    )
    rows = _read_rows(capturing_writer)
    assert rows[0]["parent_event_id"] == "abc123def456"


def test_extra_metadata_merged(capturing_writer: ObservationWriter) -> None:
    """Caller-supplied extra fields are merged into the row's extra dict."""
    observe_llm_thinking_drift_sample(
        user_text="x",
        response_text="y",
        user_message_len=1,
        response_message_len=1,
        extra={"intent_kind": "conversational", "turn_id": 42},
    )
    extra = _read_rows(capturing_writer)[0]["extra"]
    assert extra["intent_kind"] == "conversational"
    assert extra["turn_id"] == 42
    # Helper-provided fields still present.
    assert extra["enable_thinking"] is False


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_long_user_text_truncated(capturing_writer: ObservationWriter) -> None:
    long_text = "u" * 5000
    observe_llm_thinking_drift_sample(
        user_text=long_text,
        response_text="r",
        user_message_len=len(long_text),
        response_message_len=1,
    )
    rows = _read_rows(capturing_writer)
    saved = rows[0]["extra"]["user_text"]
    assert len(saved) < len(long_text)
    assert saved.endswith("... <truncated>")
    # The original length is preserved in the *_message_len field
    # so reviewers can see how much was elided.
    assert rows[0]["extra"]["user_message_len"] == 5000


def test_long_response_text_truncated(capturing_writer: ObservationWriter) -> None:
    long_response = "r" * 10000
    observe_llm_thinking_drift_sample(
        user_text="hi",
        response_text=long_response,
        user_message_len=2,
        response_message_len=len(long_response),
    )
    rows = _read_rows(capturing_writer)
    saved = rows[0]["extra"]["response_text"]
    assert saved.endswith("... <truncated>")
    assert rows[0]["extra"]["response_message_len"] == 10000


def test_exactly_cap_length_not_truncated(capturing_writer: ObservationWriter) -> None:
    """Text exactly at the cap survives intact."""
    at_cap = "a" * 4000  # The helper's internal cap.
    observe_llm_thinking_drift_sample(
        user_text=at_cap,
        response_text="ok",
        user_message_len=4000,
        response_message_len=2,
    )
    saved = _read_rows(capturing_writer)[0]["extra"]["user_text"]
    assert saved == at_cap
    assert "<truncated>" not in saved


def test_none_user_text_treated_as_empty(capturing_writer: ObservationWriter) -> None:
    """Defensive: the truncate helper accepts None without crashing."""
    observe_llm_thinking_drift_sample(
        user_text=None,  # type: ignore[arg-type]
        response_text="hello",
        user_message_len=0,
        response_message_len=5,
    )
    saved = _read_rows(capturing_writer)[0]["extra"]["user_text"]
    assert saved == ""
