"""Tests for the canonical observation schema."""

from __future__ import annotations

import json
import re

import pytest

from ultron.observations.schema import (
    KNOWN_OUTCOMES,
    KNOWN_SUBSYSTEMS,
    Observation,
    new_event_id,
)


def test_new_event_id_shape_and_uniqueness() -> None:
    ids = {new_event_id() for _ in range(100)}
    assert len(ids) == 100
    for event_id in ids:
        assert re.fullmatch(r"[0-9a-f]{16}", event_id), event_id


def test_observation_create_fills_defaults() -> None:
    obs = Observation.create(subsystem="routing", event_type="classifier_verdict")
    assert obs.subsystem == "routing"
    assert obs.event_type == "classifier_verdict"
    # Auto-filled defaults:
    assert re.fullmatch(r"[0-9a-f]{16}", obs.event_id)
    assert obs.timestamp.endswith("+00:00") or obs.timestamp.endswith("Z")
    # Unpopulated optionals:
    assert obs.parent_event_id is None
    assert obs.intent_kind is None
    assert obs.outcome is None
    assert obs.latency_ms is None
    assert obs.tokens_used is None
    assert obs.lineage_ids == ()
    assert obs.payload_ref is None
    assert obs.extra == {}


def test_observation_create_explicit_event_id_and_timestamp() -> None:
    obs = Observation.create(
        subsystem="memory",
        event_type="retrieval",
        event_id="deadbeefcafebabe",
        timestamp="2026-05-18T12:00:00+00:00",
    )
    assert obs.event_id == "deadbeefcafebabe"
    assert obs.timestamp == "2026-05-18T12:00:00+00:00"


def test_observation_to_dict_round_trips_through_json() -> None:
    obs = Observation.create(
        subsystem="llm",
        event_type="generate_stream",
        intent_kind="conversational",
        outcome="unknown_yet",
        latency_ms=63.0,
        tokens_used=128,
        lineage_ids=("mem_a", "mem_b"),
        payload_ref="logs/mcp_calls.jsonl#L42",
        extra={"prompt_chars": 320, "nested": {"k": 1}},
        parent_event_id="cafebabedeadbeef",
    )
    payload = obs.to_dict()
    # Lineage ids are a list in JSON, tuples don't survive json.dumps.
    assert payload["lineage_ids"] == ["mem_a", "mem_b"]
    blob = json.dumps(payload)
    reloaded = json.loads(blob)
    assert reloaded == payload


def test_observation_create_normalises_lineage_iterable() -> None:
    obs = Observation.create(
        subsystem="memory",
        event_type="retrieval",
        lineage_ids=("mem_a", "mem_b", "mem_c"),
    )
    assert obs.lineage_ids == ("mem_a", "mem_b", "mem_c")


def test_observation_create_copies_extra_mapping() -> None:
    src = {"x": 1}
    obs = Observation.create(
        subsystem="routing", event_type="verdict", extra=src
    )
    # Mutating the caller's dict must not bleed into the observation.
    src["y"] = 2
    assert obs.extra == {"x": 1}


def test_known_subsystems_and_outcomes_are_frozen_sets() -> None:
    assert isinstance(KNOWN_SUBSYSTEMS, frozenset)
    assert isinstance(KNOWN_OUTCOMES, frozenset)
    # The KNOWN_* sets are convention only -- writers can still emit
    # custom values. We just want to be sure the set survives import.
    assert "routing" in KNOWN_SUBSYSTEMS
    assert "unknown_yet" in KNOWN_OUTCOMES


def test_observation_is_frozen() -> None:
    obs = Observation.create(subsystem="routing", event_type="v")
    with pytest.raises(Exception):  # frozen dataclass -> FrozenInstanceError
        obs.subsystem = "memory"  # type: ignore[misc]
