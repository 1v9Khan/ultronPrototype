"""Tests for the event callback registry."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ultron.events.callbacks import (
    CallbackProcessor,
    CallbackRegistry,
    CallbackResult,
    CallbackResultStatus,
    CallbackStatus,
    FunctionProcessor,
    get_callback_registry,
    reset_callback_registry_for_testing,
    set_callback_registry,
)
from ultron.events.models import StoredEvent


@pytest.fixture(autouse=True)
def _isolate_singleton():
    reset_callback_registry_for_testing()
    yield
    reset_callback_registry_for_testing()


class _RecorderProcessor(CallbackProcessor):
    """Test helper: captures invocations + returns a custom result."""

    def __init__(self, *, result_status: CallbackResultStatus = CallbackResultStatus.SUCCESS,
                 deactivate: bool = False, raise_on_call: bool = False):
        self.calls: list[StoredEvent] = []
        self._result_status = result_status
        self._deactivate = deactivate
        self._raise = raise_on_call

    def __call__(self, event, callback):
        self.calls.append(event)
        if self._raise:
            raise RuntimeError("boom")
        return CallbackResult(
            status=self._result_status,
            callback_id=callback.id,
            event_id=event.id,
            session_id=event.session_id,
            deactivate=self._deactivate,
        )


def _event(session_id="sess", kind="K", **payload):
    return StoredEvent.make(session_id, kind, payload=payload)


def test_register_returns_active_callback():
    registry = CallbackRegistry()
    processor = _RecorderProcessor()
    row = registry.register(processor)
    assert row.status == CallbackStatus.ACTIVE
    assert row.processor is processor
    assert row.id
    assert row.label == "_RecorderProcessor"


def test_register_rejects_non_processor_type():
    registry = CallbackRegistry()
    with pytest.raises(TypeError):
        registry.register(lambda event, callback: None)  # type: ignore[arg-type]


def test_register_with_filters_records_them():
    registry = CallbackRegistry()
    row = registry.register(
        _RecorderProcessor(),
        session_id="sess",
        event_kind="K",
        label="custom",
    )
    assert row.session_id == "sess"
    assert row.event_kind == "K"
    assert row.label == "custom"


def test_register_duplicate_id_raises():
    registry = CallbackRegistry()
    registry.register(_RecorderProcessor(), callback_id="abc")
    with pytest.raises(ValueError):
        registry.register(_RecorderProcessor(), callback_id="abc")


def test_unregister_removes_callback():
    registry = CallbackRegistry()
    row = registry.register(_RecorderProcessor())
    assert registry.unregister(row.id) is True
    assert registry.get(row.id) is None
    assert registry.unregister(row.id) is False  # already gone


def test_set_status_toggles_active():
    registry = CallbackRegistry()
    row = registry.register(_RecorderProcessor())
    registry.set_status(row.id, CallbackStatus.DISABLED)
    fetched = registry.get(row.id)
    assert fetched is not None
    assert fetched.status == CallbackStatus.DISABLED
    assert fetched.is_active is False


def test_set_status_missing_returns_false():
    registry = CallbackRegistry()
    assert registry.set_status("nonexistent", CallbackStatus.DISABLED) is False


def test_list_callbacks_default_active_only():
    registry = CallbackRegistry()
    active = registry.register(_RecorderProcessor())
    disabled = registry.register(_RecorderProcessor())
    registry.set_status(disabled.id, CallbackStatus.DISABLED)
    listing = registry.list_callbacks()
    ids = [cb.id for cb in listing]
    assert active.id in ids
    assert disabled.id not in ids


def test_list_callbacks_filters_session_and_kind():
    registry = CallbackRegistry()
    target = registry.register(_RecorderProcessor(), session_id="s1", event_kind="K1")
    registry.register(_RecorderProcessor(), session_id="s2", event_kind="K2")
    assert [cb.id for cb in registry.list_callbacks(session_id="s1")] == [target.id]
    assert [cb.id for cb in registry.list_callbacks(event_kind="K1")] == [target.id]


def test_list_callbacks_include_disabled_optionally():
    registry = CallbackRegistry()
    active = registry.register(_RecorderProcessor())
    disabled = registry.register(_RecorderProcessor())
    registry.set_status(disabled.id, CallbackStatus.DISABLED)
    assert len(registry.list_callbacks(include_disabled=True)) == 2


def test_execute_fires_matching_callback():
    registry = CallbackRegistry()
    recorder = _RecorderProcessor()
    registry.register(recorder)
    event = _event()
    results = registry.execute_for_event(event)
    assert len(results) == 1
    assert results[0].status == CallbackResultStatus.SUCCESS
    assert recorder.calls == [event]


def test_execute_session_filter_blocks_mismatch():
    registry = CallbackRegistry()
    recorder = _RecorderProcessor()
    registry.register(recorder, session_id="other_session")
    results = registry.execute_for_event(_event(session_id="sess"))
    assert results == []
    assert recorder.calls == []


def test_execute_kind_filter_blocks_mismatch():
    registry = CallbackRegistry()
    recorder = _RecorderProcessor()
    registry.register(recorder, event_kind="K_target")
    results = registry.execute_for_event(_event(kind="K_other"))
    assert results == []


def test_execute_disabled_callback_skipped():
    registry = CallbackRegistry()
    recorder = _RecorderProcessor()
    row = registry.register(recorder)
    registry.set_status(row.id, CallbackStatus.DISABLED)
    results = registry.execute_for_event(_event())
    assert results == []
    assert recorder.calls == []


def test_execute_swallows_processor_exception():
    registry = CallbackRegistry()
    recorder = _RecorderProcessor(raise_on_call=True)
    registry.register(recorder)
    results = registry.execute_for_event(_event())
    assert len(results) == 1
    assert results[0].status == CallbackResultStatus.ERROR
    assert "boom" in (results[0].detail or "")
    assert registry.errors == 1


def test_execute_none_return_becomes_skipped():
    class _NoneProcessor(CallbackProcessor):
        def __call__(self, event, callback):
            return None

    registry = CallbackRegistry()
    registry.register(_NoneProcessor())
    results = registry.execute_for_event(_event())
    assert len(results) == 1
    assert results[0].status == CallbackResultStatus.SKIPPED


def test_execute_deactivate_flag_disables_callback():
    registry = CallbackRegistry()
    recorder = _RecorderProcessor(deactivate=True)
    row = registry.register(recorder)
    results = registry.execute_for_event(_event())
    assert results[0].deactivate is True
    fetched = registry.get(row.id)
    assert fetched is not None
    assert fetched.status == CallbackStatus.DISABLED


def test_execute_error_result_does_not_deactivate():
    registry = CallbackRegistry()
    recorder = _RecorderProcessor(raise_on_call=True)
    row = registry.register(recorder)
    registry.execute_for_event(_event())
    # Errors should NOT auto-disable; only explicit deactivate=True does.
    fetched = registry.get(row.id)
    assert fetched is not None
    assert fetched.status == CallbackStatus.ACTIVE


def test_execute_fires_multiple_callbacks_in_creation_order():
    registry = CallbackRegistry()
    rec1 = _RecorderProcessor()
    rec2 = _RecorderProcessor()
    row1 = registry.register(rec1)
    # Sleep to guarantee a different created_at on coarse clocks.
    time.sleep(0.001)
    row2 = registry.register(rec2)
    results = registry.execute_for_event(_event())
    assert [r.callback_id for r in results] == [row1.id, row2.id]


def test_dispatched_counter_increments():
    registry = CallbackRegistry()
    registry.register(_RecorderProcessor())
    registry.execute_for_event(_event())
    registry.execute_for_event(_event())
    assert registry.dispatched == 2


def test_clear_drops_callbacks_and_counters():
    registry = CallbackRegistry()
    registry.register(_RecorderProcessor())
    registry.execute_for_event(_event())
    registry.clear()
    assert registry.list_callbacks() == []
    assert registry.dispatched == 0


def test_function_processor_returns_success_for_truthy():
    registry = CallbackRegistry()
    captures: list[StoredEvent] = []

    def _fn(event, callback):
        captures.append(event)
        return "ok"

    registry.register(FunctionProcessor(func=_fn, func_label="my_func"))
    results = registry.execute_for_event(_event())
    assert results[0].status == CallbackResultStatus.SUCCESS
    assert results[0].detail == "ok"
    assert captures


def test_function_processor_skips_on_none():
    registry = CallbackRegistry()
    registry.register(FunctionProcessor(func=lambda e, c: None))
    results = registry.execute_for_event(_event())
    assert results[0].status == CallbackResultStatus.SKIPPED


def test_function_processor_passes_through_result_object():
    registry = CallbackRegistry()

    def _fn(event, callback):
        return CallbackResult(
            status=CallbackResultStatus.SUCCESS,
            callback_id=callback.id,
            event_id=event.id,
            session_id=event.session_id,
            detail="custom",
            deactivate=True,
        )

    registry.register(FunctionProcessor(func=_fn))
    results = registry.execute_for_event(_event())
    assert results[0].detail == "custom"
    assert results[0].deactivate is True


def test_persistence_writes_jsonl(tmp_path: Path):
    target = tmp_path / "callbacks.jsonl"
    registry = CallbackRegistry(persistence_path=target)
    row = registry.register(
        _RecorderProcessor(),
        session_id="sess",
        event_kind="K",
        label="my_cb",
    )
    assert target.exists()
    content = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    import json
    parsed = json.loads(content[0])
    assert parsed["id"] == row.id
    assert parsed["session_id"] == "sess"
    assert parsed["event_kind"] == "K"
    assert parsed["label"] == "my_cb"
    assert parsed["processor"] == "_RecorderProcessor"


def test_persistence_updates_on_status_change(tmp_path: Path):
    target = tmp_path / "callbacks.jsonl"
    registry = CallbackRegistry(persistence_path=target)
    row = registry.register(_RecorderProcessor())
    registry.set_status(row.id, CallbackStatus.DISABLED)
    import json
    parsed = json.loads(target.read_text(encoding="utf-8").strip())
    assert parsed["status"] == "disabled"


def test_persistence_updates_on_unregister(tmp_path: Path):
    target = tmp_path / "callbacks.jsonl"
    registry = CallbackRegistry(persistence_path=target)
    row = registry.register(_RecorderProcessor())
    registry.unregister(row.id)
    assert target.read_text(encoding="utf-8") in ("", "\n")


def test_persistence_failure_is_swallowed(tmp_path: Path, monkeypatch):
    # Make the parent directory a file so mkdir fails.
    blocker = tmp_path / "blocker"
    blocker.write_text("nope", encoding="utf-8")
    target = blocker / "callbacks.jsonl"
    registry = CallbackRegistry(persistence_path=target)
    # Should not raise even though persistence fails.
    row = registry.register(_RecorderProcessor())
    assert registry.get(row.id) is not None


def test_load_metadata_round_trip(tmp_path: Path):
    target = tmp_path / "callbacks.jsonl"
    registry = CallbackRegistry(persistence_path=target)
    row = registry.register(_RecorderProcessor(), session_id="s", event_kind="K")
    fresh = CallbackRegistry(persistence_path=target)
    metadata = fresh.load_metadata_from_disk()
    assert len(metadata) == 1
    assert metadata[0]["id"] == row.id


def test_load_metadata_missing_file_returns_empty(tmp_path: Path):
    target = tmp_path / "callbacks.jsonl"
    registry = CallbackRegistry(persistence_path=target)
    assert registry.load_metadata_from_disk() == []


def test_load_metadata_skips_malformed_lines(tmp_path: Path):
    target = tmp_path / "callbacks.jsonl"
    target.write_text("garbage\n{\"id\": \"a\"}\n", encoding="utf-8")
    registry = CallbackRegistry(persistence_path=target)
    metadata = registry.load_metadata_from_disk()
    assert len(metadata) == 1
    assert metadata[0]["id"] == "a"


def test_set_and_get_singleton():
    registry = CallbackRegistry()
    set_callback_registry(registry)
    assert get_callback_registry() is registry


def test_reset_singleton_clears():
    set_callback_registry(CallbackRegistry())
    reset_callback_registry_for_testing()
    assert get_callback_registry() is None


def test_slow_callback_warning_emitted(caplog):
    import logging
    caplog.set_level(logging.WARNING)

    class _SlowProcessor(CallbackProcessor):
        def __call__(self, event, callback):
            time.sleep(0.04)
            return CallbackResult(
                status=CallbackResultStatus.SUCCESS,
                callback_id=callback.id,
                event_id=event.id,
                session_id=event.session_id,
            )

    registry = CallbackRegistry(slow_callback_warn_ms=10.0)
    registry.register(_SlowProcessor())
    registry.execute_for_event(_event())
    # 40 ms wait > 10 ms threshold -> WARN.
    assert any("slow" in rec.message for rec in caplog.records)


def test_callback_id_format_is_hex_uuid():
    registry = CallbackRegistry()
    row = registry.register(_RecorderProcessor())
    assert len(row.id) == 32
    assert all(c in "0123456789abcdef" for c in row.id)
