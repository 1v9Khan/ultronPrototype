"""Tests for the idempotent file installer (T8 from the OpenHands catalog)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.install import (
    DEFAULT_MARKER,
    DEFAULT_PRESERVE_SUFFIX,
    InstallAction,
    InstallLogEntry,
    InstallLogWriter,
    InstallResult,
    install_with_marker,
    set_install_log_writer,
)


@pytest.fixture(autouse=True)
def _isolate_install_log(monkeypatch, tmp_path):
    """Redirect the install audit log to a temp file per test."""

    writer = InstallLogWriter(tmp_path / "install_log.jsonl")
    monkeypatch.setattr(
        "ultron.install.idempotent._DEFAULT_WRITER",
        writer,
    )
    yield writer


def test_install_to_missing_target_writes_file(tmp_path: Path):
    target = tmp_path / "hooks" / "pre-push"
    content = f"#!/bin/sh\n{DEFAULT_MARKER}\necho hi\n"

    result = install_with_marker(target, content)

    assert result.ok is True
    assert result.action == InstallAction.INSTALLED
    assert result.bytes_written == len(content.encode())
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content
    assert result.changed_disk is True


def test_install_skipped_when_marker_already_present(tmp_path: Path):
    target = tmp_path / "pre-commit"
    initial = f"#!/bin/sh\n{DEFAULT_MARKER}\nold content\n"
    target.write_text(initial, encoding="utf-8")

    new_content = f"#!/bin/sh\n{DEFAULT_MARKER}\nnew content\n"
    result = install_with_marker(target, new_content)

    assert result.action == InstallAction.SKIPPED_ALREADY_MARKED
    assert result.already_present is True
    assert result.ok is True
    assert result.changed_disk is False
    # The file should NOT have been touched.
    assert target.read_text(encoding="utf-8") == initial


def test_install_preserves_existing_unmarked_file(tmp_path: Path):
    target = tmp_path / "pre-commit"
    target.write_text("#!/bin/sh\nuser had their own\n", encoding="utf-8")
    preserve = target.with_suffix(target.suffix + DEFAULT_PRESERVE_SUFFIX)

    new_content = f"#!/bin/sh\n{DEFAULT_MARKER}\nultron version\n"
    result = install_with_marker(
        target,
        new_content,
        preserve_existing_as=preserve,
    )

    assert result.action == InstallAction.MOVED_THEN_INSTALLED
    assert result.moved_to == preserve
    assert preserve.read_text(encoding="utf-8") == "#!/bin/sh\nuser had their own\n"
    assert target.read_text(encoding="utf-8") == new_content


def test_install_replaces_unmarked_when_explicit_replace_true(tmp_path: Path):
    target = tmp_path / "config.json"
    target.write_text('{"existing": true}', encoding="utf-8")

    content = f"// {DEFAULT_MARKER}\n{{}}\n"
    result = install_with_marker(target, content, replace_unmarked=True)

    assert result.action == InstallAction.REPLACED_UNMARKED
    assert target.read_text(encoding="utf-8") == content


def test_install_refuses_unmarked_existing_without_explicit_consent(tmp_path: Path):
    target = tmp_path / "user_file.txt"
    target.write_text("user-owned content", encoding="utf-8")

    content = f"{DEFAULT_MARKER}\nultron content"
    result = install_with_marker(target, content)

    assert result.ok is False
    assert result.error is not None
    assert result.action == InstallAction.SKIPPED_ALREADY_MARKED
    # User content untouched on refusal.
    assert target.read_text(encoding="utf-8") == "user-owned content"


def test_dry_run_returns_dry_run_action_with_no_disk_write(tmp_path: Path):
    target = tmp_path / "dryrun.txt"

    content = f"{DEFAULT_MARKER}\nwould-have-been-installed"
    result = install_with_marker(target, content, dry_run=True)

    assert result.action == InstallAction.DRY_RUN
    assert result.changed_disk is False
    assert not target.exists()


def test_dry_run_against_existing_marked_still_skip(tmp_path: Path):
    target = tmp_path / "marked.txt"
    target.write_text(f"{DEFAULT_MARKER}\nold", encoding="utf-8")
    result = install_with_marker(target, f"{DEFAULT_MARKER}\nnew", dry_run=True)
    assert result.action == InstallAction.SKIPPED_ALREADY_MARKED
    assert result.already_present is True
    assert target.read_text(encoding="utf-8") == f"{DEFAULT_MARKER}\nold"


def test_custom_marker_used_for_detection(tmp_path: Path):
    target = tmp_path / "special.sh"
    custom = "# ULTRON-CUSTOM-MARKER-42"
    target.write_text(f"{custom}\nexisting", encoding="utf-8")

    new_content = f"{custom}\nnewer"
    result = install_with_marker(target, new_content, marker=custom)
    assert result.action == InstallAction.SKIPPED_ALREADY_MARKED


def test_warning_logged_when_content_lacks_marker(tmp_path: Path, caplog):
    import logging

    caplog.set_level(logging.WARNING)

    target = tmp_path / "no_marker_in_content.txt"
    # Deliberately omit the marker from content.
    result = install_with_marker(target, "no marker present")

    # The install still happens because there was no existing file.
    assert result.action == InstallAction.INSTALLED
    # But the warning was emitted so the next call won't recognise it.
    assert any("does not contain marker" in rec.message for rec in caplog.records)
    assert result.extra.get("marker_in_content") == "false"


def test_install_creates_parent_directories(tmp_path: Path):
    target = tmp_path / "deep" / "nested" / "path" / "out.txt"
    content = f"{DEFAULT_MARKER}\nbody"
    result = install_with_marker(target, content)
    assert result.action == InstallAction.INSTALLED
    assert target.exists()


def test_install_extra_metadata_preserved(tmp_path: Path):
    target = tmp_path / "extra.txt"
    extra = {"installer": "pre-push", "version": "1.0"}
    result = install_with_marker(
        target,
        f"{DEFAULT_MARKER}\nbody",
        extra=extra,
    )
    assert result.extra["installer"] == "pre-push"
    assert result.extra["version"] == "1.0"


def test_install_audit_log_writes_entry(tmp_path: Path, _isolate_install_log):
    target = tmp_path / "audit_target.txt"
    install_with_marker(target, f"{DEFAULT_MARKER}\nx")

    log_path = _isolate_install_log.path
    assert log_path is not None
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    import json

    row = json.loads(lines[0])
    assert row["target_path"] == str(target)
    assert row["action"] == "installed"
    assert row["bytes_written"] > 0
    assert row["error"] is None


def test_install_audit_log_records_skip(tmp_path: Path, _isolate_install_log):
    target = tmp_path / "marker_present.txt"
    target.write_text(f"{DEFAULT_MARKER}\nold", encoding="utf-8")
    install_with_marker(target, f"{DEFAULT_MARKER}\nnew")

    rows = _read_audit_rows(_isolate_install_log.path)
    assert len(rows) == 1
    assert rows[0]["action"] == "skipped_already_marked"


def test_install_audit_log_records_move_and_install(tmp_path: Path, _isolate_install_log):
    target = tmp_path / "needs_preserve.txt"
    target.write_text("legacy", encoding="utf-8")
    preserve = target.with_suffix(target.suffix + ".local")
    install_with_marker(
        target,
        f"{DEFAULT_MARKER}\nx",
        preserve_existing_as=preserve,
    )

    rows = _read_audit_rows(_isolate_install_log.path)
    assert len(rows) == 1
    assert rows[0]["action"] == "moved_then_installed"
    assert rows[0]["moved_to"] == str(preserve)


def test_install_audit_log_failure_path_swallowed(tmp_path: Path, monkeypatch):
    # Pointing the writer at a directory that can't be created (file in the way)
    # produces an OSError that should be caught + logged but not propagate.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    writer = InstallLogWriter(blocker / "child" / "log.jsonl")
    monkeypatch.setattr(
        "ultron.install.idempotent._DEFAULT_WRITER",
        writer,
    )

    target = tmp_path / "ok.txt"
    # Should not raise even though the audit log cannot be persisted.
    result = install_with_marker(target, f"{DEFAULT_MARKER}\nx")
    assert result.action == InstallAction.INSTALLED


def test_set_install_log_writer_swap(tmp_path: Path):
    log = tmp_path / "swapped.jsonl"
    set_install_log_writer(InstallLogWriter(log))

    target = tmp_path / "after_swap.txt"
    install_with_marker(target, f"{DEFAULT_MARKER}\nrow")

    rows = _read_audit_rows(log)
    assert len(rows) == 1
    assert rows[0]["target_path"] == str(target)

    # Reset to the test fixture default so the autouse fixture restores cleanly.
    set_install_log_writer(None)


def test_install_result_frozen(tmp_path: Path):
    target = tmp_path / "frozen.txt"
    result = install_with_marker(target, f"{DEFAULT_MARKER}\nx")
    with pytest.raises(Exception):
        result.action = InstallAction.DRY_RUN  # type: ignore[misc]


def test_install_unreadable_existing_recorded_as_error(tmp_path: Path, monkeypatch):
    target = tmp_path / "unreadable.txt"
    target.write_text("anything", encoding="utf-8")

    def _read_text(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise PermissionError("no")

    monkeypatch.setattr(Path, "read_text", _read_text)

    result = install_with_marker(target, f"{DEFAULT_MARKER}\nnew")
    assert result.ok is False
    assert result.error is not None
    assert "could not read" in result.error.lower()


def test_install_string_paths_accepted(tmp_path: Path):
    target = tmp_path / "str_path.txt"
    result = install_with_marker(str(target), f"{DEFAULT_MARKER}\nbody")
    assert result.action == InstallAction.INSTALLED
    assert target.exists()


def test_install_log_entry_shape():
    entry = InstallLogEntry(
        timestamp=1.0,
        target_path="/tmp/x",
        action="installed",
        marker=DEFAULT_MARKER,
        moved_to=None,
        bytes_written=42,
        error=None,
        extra={},
    )
    assert entry.timestamp == 1.0
    assert entry.action == "installed"
    assert entry.bytes_written == 42


def _read_audit_rows(path: Path | None) -> list[dict]:
    import json

    if path is None or not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
