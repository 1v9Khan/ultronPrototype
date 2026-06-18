"""Tests for the observation integration helpers + the wired call sites.

Each test installs a real :class:`ObservationWriter` pointed at a
tmp_path file so we can inspect the emitted rows. The session-scoped
autouse fixture in :mod:`tests.conftest` disables the singleton by
default; these tests override that for the duration of each test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from kenning.observations import (
    Observation,
    ObservationWriter,
    get_observation_writer,
    observe_addressing_verdict,
    observe_llm_call,
    observe_retrieval,
    observe_routing_verdict,
    set_observation_writer,
)


@pytest.fixture
def capturing_writer(tmp_path: Path) -> Iterator[ObservationWriter]:
    """Real writer pointed at a temp file; readable via the fixture."""
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
# Helper-level emit
# ---------------------------------------------------------------------------


def test_observe_routing_verdict_emits_one_row(capturing_writer: ObservationWriter) -> None:
    event_id = observe_routing_verdict(
        utterance="tell me a joke",
        intent_kind="conversational",
        confidence=1.0,
        source="default",
        reason="fallback",
        latency_ms=1.2,
    )
    assert event_id is not None
    rows = _read_rows(capturing_writer)
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "routing"
    assert row["event_type"] == "classify_routing"
    assert row["intent_kind"] == "conversational"
    assert row["latency_ms"] == pytest.approx(1.2)
    assert row["outcome"] == "unknown_yet"
    assert row["extra"]["utterance_len"] == len("tell me a joke")
    assert row["extra"]["confidence"] == 1.0
    assert row["extra"]["source"] == "default"
    assert row["extra"]["reason"] == "fallback"


def test_observe_addressing_verdict_emits_one_row(capturing_writer: ObservationWriter) -> None:
    event_id = observe_addressing_verdict(
        utterance="kenning, open chrome",
        decision="ADDRESSED",
        confidence=0.95,
        reason="direct address by name",
        seconds_since_response=0.0,
        source="rule",
        latency_ms=0.5,
    )
    assert event_id is not None
    rows = _read_rows(capturing_writer)
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "addressing"
    assert row["event_type"] == "classify_addressing"
    assert row["extra"]["decision"] == "ADDRESSED"
    # 2026-06-18: the fused scorer is the primary path; it still emits the
    # observation via _log. ("rule" was the pre-fusion source label.)
    assert row["extra"]["classifier_source"] in ("fusion", "rule")


def test_observe_retrieval_records_lineage_ids(capturing_writer: ObservationWriter) -> None:
    event_id = observe_retrieval(
        query="what did we discuss last",
        lineage_ids=("mem_1", "mem_2", "mem_3"),
        k=5,
        latency_ms=12.3,
        collection="conversations",
    )
    assert event_id is not None
    rows = _read_rows(capturing_writer)
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "memory"
    assert row["lineage_ids"] == ["mem_1", "mem_2", "mem_3"]
    assert row["extra"]["k"] == 5
    assert row["extra"]["result_count"] == 3
    assert row["extra"]["collection"] == "conversations"


def test_observe_llm_call_emits_tokens(capturing_writer: ObservationWriter) -> None:
    event_id = observe_llm_call(
        event_type="generate",
        user_message_len=42,
        tokens_used=128,
        latency_ms=63.0,
        streamed=False,
        enable_thinking=False,
    )
    assert event_id is not None
    rows = _read_rows(capturing_writer)
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "llm"
    assert row["event_type"] == "generate"
    assert row["tokens_used"] == 128
    assert row["latency_ms"] == 63.0
    assert row["extra"]["user_message_len"] == 42
    assert row["extra"]["streamed"] is False
    assert row["extra"]["enable_thinking"] is False


def test_helpers_return_none_when_writer_disabled(tmp_path: Path) -> None:
    disabled = ObservationWriter(tmp_path / "obs.jsonl", enabled=False)
    previous = get_observation_writer()
    set_observation_writer(disabled)
    try:
        assert (
            observe_routing_verdict(
                utterance="x", intent_kind="conversational",
                confidence=1.0, source="default", reason="r", latency_ms=0.0,
            ) is None
        )
        assert (
            observe_addressing_verdict(
                utterance="x", decision="ADDRESSED", confidence=0.9,
                reason="r", seconds_since_response=0.0, source="rule", latency_ms=0.0,
            ) is None
        )
        assert (
            observe_retrieval(
                query="x", lineage_ids=(), k=1, latency_ms=0.0,
            ) is None
        )
        assert (
            observe_llm_call(
                event_type="generate", user_message_len=0,
                tokens_used=None, latency_ms=0.0, streamed=False,
            ) is None
        )
    finally:
        set_observation_writer(previous)


# ---------------------------------------------------------------------------
# Wired call sites
# ---------------------------------------------------------------------------


def test_classify_routing_emits_observation(capturing_writer: ObservationWriter) -> None:
    from kenning.openclaw_routing.classifier import classify_routing

    intent = classify_routing("tell me a joke")
    assert intent.kind.value == "conversational"

    rows = _read_rows(capturing_writer)
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "routing"
    assert row["event_type"] == "classify_routing"
    assert row["intent_kind"] == "conversational"
    assert row["latency_ms"] is not None and row["latency_ms"] >= 0.0


def test_classify_routing_emits_intent_kind_for_different_inputs(
    capturing_writer: ObservationWriter,
) -> None:
    from kenning.openclaw_routing.classifier import classify_routing

    classify_routing("kenning, write me a python script")
    classify_routing("how is the project going", has_active_coding_task=True)
    classify_routing("open youtube on my second monitor")

    rows = _read_rows(capturing_writer)
    kinds = [row["intent_kind"] for row in rows]
    assert "code_task" in kinds
    assert "progress_query" in kinds
    assert "app_launch" in kinds


def test_addressing_classifier_emits_observation(
    capturing_writer: ObservationWriter, monkeypatch
) -> None:
    """Even with the rule-classifier path (no zero-shot load), the emit
    should fire via ``AddressingClassifier._log``."""
    from kenning.addressing.classifier import AddressingClassifier
    from kenning.addressing.zero_shot import ZeroShotAddresseeModel

    # Don't load the zero-shot model -- not needed for a rule hit.
    monkeypatch.setattr(
        ZeroShotAddresseeModel, "_ensure_loaded", lambda self: None
    )
    clf = AddressingClassifier(load_zero_shot_eagerly=False, log_path=None)
    verdict = clf.classify("kenning, what time is it")
    assert verdict.decision.value == "ADDRESSED"

    rows = _read_rows(capturing_writer)
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "addressing"
    assert row["extra"]["decision"] == "ADDRESSED"
    # 2026-06-18: the fused scorer is the primary path; it still emits the
    # observation via _log. ("rule" was the pre-fusion source label.)
    assert row["extra"]["classifier_source"] in ("fusion", "rule")


def test_conversation_memory_retrieve_emits_empty_query_observation(
    capturing_writer: ObservationWriter,
) -> None:
    """Empty query returns []; we still emit an observation row with
    result_count=0 so the eval harness can see the call shape."""
    # Use a minimal stub of ConversationMemory: we only exercise the
    # public retrieve() wrapper, which short-circuits on empty input
    # before any Qdrant code runs.
    from kenning.memory.qdrant_store import ConversationMemory

    # Construct without going through __init__ to avoid needing Qdrant.
    mem = ConversationMemory.__new__(ConversationMemory)
    result = mem.retrieve("")
    assert result == []

    rows = _read_rows(capturing_writer)
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "memory"
    assert row["event_type"] == "retrieve"
    assert row["extra"]["result_count"] == 0
    assert row["lineage_ids"] == []
