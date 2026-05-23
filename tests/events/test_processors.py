"""Tests for the built-in callback processors."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from ultron.events.callbacks import (
    CallbackRegistry,
    CallbackResult,
    CallbackResultStatus,
    CallbackStatus,
    RegisteredCallback,
)
from ultron.events.models import StoredEvent
from ultron.events.processors import (
    ChannelGuardProcessor,
    CountingCallbackProcessor,
    LoggingCallbackProcessor,
    MemoryWriteProcessor,
    SkillActivatorProcessor,
    ThresholdSnapshotProcessor,
    build_default_processors,
)


def _event(session_id="sess", kind="K", **payload):
    return StoredEvent.make(session_id, kind, payload=payload)


def _registered(processor, session_id=None, event_kind=None) -> RegisteredCallback:
    registry = CallbackRegistry()
    return registry.register(processor, session_id=session_id, event_kind=event_kind)


# -- LoggingCallbackProcessor --


def test_logging_processor_returns_success(caplog):
    caplog.set_level(logging.INFO)
    processor = LoggingCallbackProcessor()
    callback = _registered(processor)
    result = processor(_event(text="hello"), callback)
    assert result.status == CallbackResultStatus.SUCCESS
    assert any("callback fire" in rec.message for rec in caplog.records)


def test_logging_processor_excludes_payload_by_default(caplog):
    caplog.set_level(logging.INFO)
    processor = LoggingCallbackProcessor(include_payload=False)
    processor(_event(secret_value="should not appear"), _registered(processor))
    joined = " ".join(rec.message for rec in caplog.records)
    assert "secret_value" not in joined


def test_logging_processor_includes_payload_when_set(caplog):
    caplog.set_level(logging.INFO)
    processor = LoggingCallbackProcessor(include_payload=True)
    processor(_event(text="visible"), _registered(processor))
    joined = " ".join(rec.message for rec in caplog.records)
    assert "visible" in joined


# -- CountingCallbackProcessor --


def test_counting_processor_increments_per_kind():
    processor = CountingCallbackProcessor()
    cb = _registered(processor)
    processor(_event(kind="A"), cb)
    processor(_event(kind="A"), cb)
    processor(_event(kind="B"), cb)
    assert processor.counts == {"A": 2, "B": 1}


def test_counting_processor_returns_count_in_extra():
    processor = CountingCallbackProcessor()
    cb = _registered(processor)
    result = processor(_event(), cb)
    assert result.status == CallbackResultStatus.SUCCESS
    assert result.extra["count"] == 1


# -- ThresholdSnapshotProcessor --


def test_threshold_processor_fires_at_threshold():
    fired: list[StoredEvent] = []

    def snapshot(event, callback):
        fired.append(event)
        return "snapped"

    processor = ThresholdSnapshotProcessor(snapshot_fn=snapshot, threshold=3)
    cb = _registered(processor)

    r1 = processor(_event(), cb)
    r2 = processor(_event(), cb)
    r3 = processor(_event(), cb)

    assert r1.status == CallbackResultStatus.SKIPPED
    assert r2.status == CallbackResultStatus.SKIPPED
    assert r3.status == CallbackResultStatus.SUCCESS
    assert r3.deactivate is True
    assert r3.detail == "snapped"
    assert len(fired) == 1


def test_threshold_processor_per_session_counter():
    processor = ThresholdSnapshotProcessor(
        snapshot_fn=lambda e, c: "ok", threshold=2
    )
    cb = _registered(processor)

    # Session A fires once -> skip.
    assert processor(_event(session_id="A"), cb).status == CallbackResultStatus.SKIPPED
    # Session B fires once -> skip.
    assert processor(_event(session_id="B"), cb).status == CallbackResultStatus.SKIPPED
    # Session A's second fire -> success.
    assert processor(_event(session_id="A"), cb).status == CallbackResultStatus.SUCCESS


def test_threshold_processor_can_repeat_when_deactivate_disabled():
    processor = ThresholdSnapshotProcessor(
        snapshot_fn=lambda e, c: "ok",
        threshold=1,
        deactivate_after_fire=False,
    )
    cb = _registered(processor)
    r1 = processor(_event(), cb)
    r2 = processor(_event(), cb)
    assert r1.deactivate is False
    assert r2.deactivate is False


def test_threshold_processor_label_override():
    processor = ThresholdSnapshotProcessor(
        snapshot_fn=lambda e, c: "ok",
        label_override="my_snapshot",
    )
    assert processor.label == "my_snapshot"


# -- MemoryWriteProcessor --


def test_memory_write_processor_calls_writer():
    captured: list[tuple[str, str]] = []
    processor = MemoryWriteProcessor(writer=lambda r, c: captured.append((r, c)))
    cb = _registered(processor)
    result = processor(_event(text="hello there"), cb)
    assert result.status == CallbackResultStatus.SUCCESS
    assert captured == [("system", "hello there")]


def test_memory_write_processor_skips_when_no_content():
    captured: list[tuple[str, str]] = []
    processor = MemoryWriteProcessor(writer=lambda r, c: captured.append((r, c)))
    cb = _registered(processor)
    result = processor(_event(), cb)
    assert result.status == CallbackResultStatus.SKIPPED
    assert captured == []


def test_memory_write_processor_role_mapping():
    captured: list[tuple[str, str]] = []
    processor = MemoryWriteProcessor(
        writer=lambda r, c: captured.append((r, c)),
        role_for_kind={"UserMessage": "user"},
    )
    cb = _registered(processor)
    processor(_event(kind="UserMessage", text="hi"), cb)
    assert captured == [("user", "hi")]


def test_memory_write_processor_handles_writer_exception():
    def _broken(role, content):
        raise RuntimeError("writer failure")

    processor = MemoryWriteProcessor(writer=_broken)
    cb = _registered(processor)
    result = processor(_event(text="anything"), cb)
    assert result.status == CallbackResultStatus.ERROR
    assert "writer failure" in (result.detail or "")


def test_memory_write_processor_custom_content_field():
    captured: list[tuple[str, str]] = []
    processor = MemoryWriteProcessor(
        writer=lambda r, c: captured.append((r, c)),
        content_field="body",
    )
    cb = _registered(processor)
    processor(_event(body="custom field"), cb)
    assert captured == [("system", "custom field")]


# -- ChannelGuardProcessor --


def test_channel_guard_detects_secret_pattern():
    processor = ChannelGuardProcessor()
    cb = _registered(processor)
    result = processor(
        _event(text="api_key=sk-1234567890abcdefABCDEF"), cb
    )
    assert result is not None
    assert result.status == CallbackResultStatus.ERROR
    assert "secret-like" in (result.detail or "")


def test_channel_guard_passes_clean_payloads():
    processor = ChannelGuardProcessor()
    cb = _registered(processor)
    result = processor(_event(text="just a normal sentence"), cb)
    assert result is None


def test_channel_guard_detects_password_assignment():
    processor = ChannelGuardProcessor()
    cb = _registered(processor)
    result = processor(_event(text='password = "supersecretpassword123"'), cb)
    assert result is not None
    assert result.status == CallbackResultStatus.ERROR


# -- SkillActivatorProcessor --


def test_skill_activator_calls_function():
    fired: list[StoredEvent] = []
    processor = SkillActivatorProcessor(
        activator_fn=lambda e, c: fired.append(e) or "activated"
    )
    cb = _registered(processor)
    result = processor(_event(topic="finance"), cb)
    assert result.status == CallbackResultStatus.SUCCESS
    assert result.detail == "activated"
    assert len(fired) == 1


def test_skill_activator_swallows_exception():
    def _bad(event, callback):
        raise RuntimeError("nope")

    processor = SkillActivatorProcessor(activator_fn=_bad)
    cb = _registered(processor)
    result = processor(_event(), cb)
    assert result.status == CallbackResultStatus.ERROR
    assert "nope" in (result.detail or "")


# -- build_default_processors --


def test_build_default_processors_returns_safe_set():
    defaults = build_default_processors()
    assert len(defaults) == 2
    assert any(isinstance(p, LoggingCallbackProcessor) for p in defaults)
    assert any(isinstance(p, CountingCallbackProcessor) for p in defaults)


# -- Integration: registry + processor --


def test_registry_executes_threshold_processor_e2e():
    """End-to-end: register a threshold processor, fire events, observe lifecycle."""
    fired: list[StoredEvent] = []
    processor = ThresholdSnapshotProcessor(
        snapshot_fn=lambda e, c: fired.append(e) or "ok",
        threshold=2,
    )
    registry = CallbackRegistry()
    row = registry.register(processor)

    # First event: skip.
    results = registry.execute_for_event(_event())
    assert results[0].status == CallbackResultStatus.SKIPPED

    # Second event: success + deactivation.
    results = registry.execute_for_event(_event())
    assert results[0].status == CallbackResultStatus.SUCCESS
    assert results[0].deactivate is True

    # After deactivation: no more fires.
    fetched = registry.get(row.id)
    assert fetched is not None
    assert fetched.status == CallbackStatus.DISABLED

    results = registry.execute_for_event(_event())
    assert results == []  # nothing matched (we disabled the only callback)
    assert len(fired) == 1
