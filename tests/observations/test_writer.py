"""Tests for the observation writer + singleton accessors."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ultron.observations import (
    Observation,
    ObservationWriter,
    emit_observation,
    get_observation_writer,
    new_event_id,
    set_observation_writer,
)
from ultron.observations import writer as writer_mod


@pytest.fixture(autouse=True)
def isolate_singleton() -> None:
    """Each test gets a clean singleton slot."""
    set_observation_writer(None)
    yield
    set_observation_writer(None)


# ---------------------------------------------------------------------------
# ObservationWriter happy path
# ---------------------------------------------------------------------------


def test_emit_writes_one_jsonl_row(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    w = ObservationWriter(path)
    obs = Observation.create(subsystem="routing", event_type="verdict")
    assert w.emit(obs) is True
    contents = path.read_text(encoding="utf-8").splitlines()
    assert len(contents) == 1
    payload = json.loads(contents[0])
    assert payload["subsystem"] == "routing"
    assert payload["event_type"] == "verdict"
    assert payload["event_id"] == obs.event_id


def test_emit_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "observations.jsonl"
    w = ObservationWriter(path)
    obs = Observation.create(subsystem="memory", event_type="retrieval")
    assert w.emit(obs) is True
    assert path.exists()


def test_emit_appends_subsequent_rows(tmp_path: Path) -> None:
    path = tmp_path / "o.jsonl"
    w = ObservationWriter(path)
    for _ in range(5):
        w.emit(Observation.create(subsystem="routing", event_type="v"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    # Each line is independently valid JSON.
    for line in lines:
        json.loads(line)


def test_emit_disabled_writer_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "o.jsonl"
    w = ObservationWriter(path, enabled=False)
    assert w.emit(Observation.create(subsystem="routing", event_type="v")) is False
    assert not path.exists()


def test_set_enabled_toggle_flips_writes(tmp_path: Path) -> None:
    path = tmp_path / "o.jsonl"
    w = ObservationWriter(path)
    w.set_enabled(False)
    assert w.emit(Observation.create(subsystem="routing", event_type="v")) is False
    w.set_enabled(True)
    assert w.emit(Observation.create(subsystem="routing", event_type="v")) is True


# ---------------------------------------------------------------------------
# Failure modes (never raise)
# ---------------------------------------------------------------------------


def test_emit_handles_serialize_failure(tmp_path: Path, caplog) -> None:
    """A non-JSON-serializable extra value must not raise."""
    path = tmp_path / "o.jsonl"
    w = ObservationWriter(path)
    # Pass a non-serialisable object inside extra. Object instances of
    # arbitrary classes aren't JSON-serialisable by default.

    class _Bomb:
        pass

    obs = Observation.create(
        subsystem="routing",
        event_type="v",
        extra={"bad": _Bomb()},
    )
    with caplog.at_level("WARNING", logger="ultron.observations"):
        result = w.emit(obs)
    assert result is False
    assert w.dropped == 1
    assert any("failed to serialize" in rec.message for rec in caplog.records)


def test_emit_handles_io_failure(tmp_path: Path, caplog, monkeypatch) -> None:
    """An OSError mid-write must not raise."""
    path = tmp_path / "o.jsonl"
    w = ObservationWriter(path)

    original_open = Path.open

    def boom(self, *args, **kwargs):
        if str(self) == str(path):
            raise OSError("disk full simulation")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", boom)
    with caplog.at_level("WARNING", logger="ultron.observations"):
        result = w.emit(Observation.create(subsystem="routing", event_type="v"))
    assert result is False
    assert w.dropped == 1
    assert any("failed to append" in rec.message for rec in caplog.records)


def test_emit_warn_once_per_failure_kind(tmp_path: Path, caplog, monkeypatch) -> None:
    """Repeated identical failures emit only one WARN line."""
    path = tmp_path / "o.jsonl"
    w = ObservationWriter(path)

    def boom(self, *args, **kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(Path, "open", boom)
    with caplog.at_level("WARNING", logger="ultron.observations"):
        for _ in range(10):
            w.emit(Observation.create(subsystem="routing", event_type="v"))
    warn_lines = [r for r in caplog.records if "failed to append" in r.message]
    assert len(warn_lines) == 1
    assert w.dropped == 10


def test_reset_warning_state_re_arms_warnings(tmp_path: Path, caplog, monkeypatch) -> None:
    path = tmp_path / "o.jsonl"
    w = ObservationWriter(path)

    def boom(self, *args, **kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(Path, "open", boom)
    with caplog.at_level("WARNING", logger="ultron.observations"):
        w.emit(Observation.create(subsystem="routing", event_type="v"))
        w.reset_warning_state()
        w.emit(Observation.create(subsystem="routing", event_type="v"))
    warn_lines = [r for r in caplog.records if "failed to append" in r.message]
    assert len(warn_lines) == 2


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_emits_do_not_corrupt(tmp_path: Path) -> None:
    """100 writers x 50 emits each -> exactly 5000 lines, all valid JSON."""
    path = tmp_path / "o.jsonl"
    w = ObservationWriter(path)
    n_threads = 100
    per_thread = 50

    def worker() -> None:
        for _ in range(per_thread):
            w.emit(
                Observation.create(
                    subsystem="routing",
                    event_type="v",
                    event_id=new_event_id(),
                )
            )

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * per_thread
    seen_ids = set()
    for line in lines:
        payload = json.loads(line)
        seen_ids.add(payload["event_id"])
    assert len(seen_ids) == n_threads * per_thread


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------


def test_get_singleton_constructs_default_writer() -> None:
    w = get_observation_writer()
    assert isinstance(w, ObservationWriter)
    # Subsequent calls return the same instance.
    assert get_observation_writer() is w


def test_set_singleton_replaces(tmp_path: Path) -> None:
    custom = ObservationWriter(tmp_path / "custom.jsonl")
    set_observation_writer(custom)
    assert get_observation_writer() is custom


def test_set_singleton_none_re_constructs_default() -> None:
    first = get_observation_writer()
    set_observation_writer(None)
    second = get_observation_writer()
    assert second is not first
    assert isinstance(second, ObservationWriter)


def test_emit_observation_routes_through_singleton(tmp_path: Path) -> None:
    custom = ObservationWriter(tmp_path / "from_singleton.jsonl")
    set_observation_writer(custom)
    obs = Observation.create(subsystem="memory", event_type="retrieval")
    assert emit_observation(obs) is True
    lines = (tmp_path / "from_singleton.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_default_path_constant_is_under_data_dir() -> None:
    # Light sanity that we haven't accidentally re-pointed the default path.
    assert writer_mod._DEFAULT_PATH == Path("data") / "observations.jsonl"
