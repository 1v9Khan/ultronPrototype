"""Tests for the canonical event catalog (ultron.bus.events)."""

from __future__ import annotations

from ultron.bus import BUS_EVENT_CATALOG
from ultron.bus.event import BusEvent


def test_catalog_is_non_empty() -> None:
    assert len(BUS_EVENT_CATALOG) >= 15


def test_catalog_entries_are_bus_events() -> None:
    for entry in BUS_EVENT_CATALOG:
        assert isinstance(entry, BusEvent)


def test_catalog_types_unique() -> None:
    types = [e.type for e in BUS_EVENT_CATALOG]
    assert len(types) == len(set(types)), (
        f"duplicate event types in catalog: {types}"
    )


def test_catalog_types_are_dotted_strings() -> None:
    for e in BUS_EVENT_CATALOG:
        assert isinstance(e.type, str)
        assert "." in e.type, (
            f"event type {e.type!r} should be dotted (e.g. 'turn.started')"
        )
        assert e.type == e.type.lower(), (
            f"event type {e.type!r} should be lowercase"
        )


def test_catalog_descriptions_non_empty() -> None:
    for e in BUS_EVENT_CATALOG:
        assert e.description, (
            f"event {e.type!r} has empty description -- "
            "every catalog entry needs a human-readable description"
        )


def test_catalog_imports_named() -> None:
    """The catalog entries listed in __init__.py must all be importable
    by their canonical names."""
    from ultron.bus import (
        CodingFileChangedEvent,
        GamingDisengagedEvent,
        GamingEngagedEvent,
        GateVerdictEvent,
        LLMStreamCompleteEvent,
        LLMStreamTokenEvent,
        MemoryRetrievedEvent,
        ProjectDigestGeneratedEvent,
        ProjectIndexedEvent,
        RoutingClassifiedEvent,
        STTTranscribedEvent,
        SafetyViolatedEvent,
        SupervisorDecidedEvent,
        TTSPlayedEvent,
        TurnCompletedEvent,
        TurnStartedEvent,
        VRAMReclaimedEvent,
    )

    for evt in (
        TurnStartedEvent, TurnCompletedEvent, STTTranscribedEvent,
        RoutingClassifiedEvent, GateVerdictEvent, MemoryRetrievedEvent,
        LLMStreamTokenEvent, LLMStreamCompleteEvent, TTSPlayedEvent,
        CodingFileChangedEvent, ProjectIndexedEvent,
        ProjectDigestGeneratedEvent, SupervisorDecidedEvent,
        SafetyViolatedEvent, GamingEngagedEvent, GamingDisengagedEvent,
        VRAMReclaimedEvent,
    ):
        assert evt in BUS_EVENT_CATALOG
