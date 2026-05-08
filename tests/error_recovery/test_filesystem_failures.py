"""FilesystemError wrappers across the coding cluster.

Validates: when audit logs, project registry, or session-audit writes
hit OSError, the system degrades silently but a FilesystemError record
lands in errors.jsonl with the path + recovery action.

We drive real OSErrors by placing a directory at the path the writer
expects to open for append/write. Avoids mocking Path.open globally,
which would also block the error log's own write.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding.audit import SessionAuditWriter
from ultron.coding.projects import ProjectRegistry, Project


# ---------------------------------------------------------------------------
# SessionAuditWriter
# ---------------------------------------------------------------------------


def test_session_audit_mkdir_failure_logs_and_disables(
    errors_log, read_errors, tmp_path,
):
    """If the configured log_dir can't be created, the writer disables
    itself silently AND records the failure once."""
    # Point at a path under a file (mkdir under a file fails on every OS).
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("not a directory", encoding="utf-8")
    bad_dir = blocker / "sessions"

    writer = SessionAuditWriter(bad_dir)

    assert writer.log_dir is None  # disabled gracefully
    records = read_errors()
    fs_records = [r for r in records if r["error_type"] == "FilesystemError"]
    assert fs_records, f"expected FilesystemError; got {records!r}"
    rec = fs_records[0]
    assert rec["dependency"] == "filesystem"
    assert "log_dir mkdir failed" in rec["message"]
    assert "disabled" in rec["recovery"]


def test_session_audit_write_oserror_logs(errors_log, read_errors, tmp_path):
    """Place a directory at the session log file path so opening it
    for append fails with OSError. The writer must swallow it but log
    a FilesystemError record."""
    log_dir = tmp_path / "sessions"
    writer = SessionAuditWriter(log_dir)
    # Block the per-session file by making it a directory rather than file.
    target = log_dir / "session-123.jsonl"
    target.mkdir(parents=True, exist_ok=True)

    writer.write("session-123", "test_event", payload="x")

    records = read_errors()
    fs_records = [r for r in records if r["error_type"] == "FilesystemError"]
    assert fs_records, f"expected FilesystemError; got {records!r}"
    rec = fs_records[0]
    assert rec["dependency"] == "filesystem"
    assert "session-audit write failed" in rec["message"]
    assert rec["context"]["session_id"] == "session-123"
    assert rec["context"]["event"] == "test_event"


# ---------------------------------------------------------------------------
# ProjectRegistry
# ---------------------------------------------------------------------------


def test_registry_load_corrupt_json_logs_filesystem_error(
    errors_log, read_errors, tmp_path,
):
    """Corrupt registry file -> empty in-memory state + FilesystemError record."""
    registry_path = tmp_path / "projects.json"
    registry_path.write_text("{ this is not valid json", encoding="utf-8")

    registry = ProjectRegistry(registry_path)

    assert registry.list() == []
    records = read_errors()
    fs_records = [r for r in records if r["error_type"] == "FilesystemError"]
    assert fs_records, f"expected FilesystemError; got {records!r}"
    rec = fs_records[0]
    assert rec["dependency"] == "filesystem"
    assert "unreadable" in rec["message"]
    assert rec["context"]["underlying"] == "JSONDecodeError"
    assert "empty registry" in rec["recovery"]


def test_registry_save_oserror_logs_and_reraises(
    errors_log, read_errors, tmp_path,
):
    """A save failure must still raise (callers can catch) but the
    failure is recorded with path + recovery. We block the .tmp write
    by placing a directory at the .tmp path."""
    registry_path = tmp_path / "projects.json"
    registry = ProjectRegistry(registry_path)
    # Block the atomic-rename .tmp file by making it a directory.
    tmp_target = registry_path.with_suffix(registry_path.suffix + ".tmp")
    tmp_target.mkdir(parents=True, exist_ok=True)

    project = Project(name="demo", path=str(tmp_path / "demo"))

    with pytest.raises(OSError):
        registry.add(project)

    records = read_errors()
    fs_records = [r for r in records if r["error_type"] == "FilesystemError"]
    assert fs_records, f"expected FilesystemError; got {records!r}"
    rec = fs_records[0]
    assert rec["dependency"] == "filesystem"
    assert "registry write failed" in rec["message"]
    assert "in-memory registry retained" in rec["recovery"]


# ---------------------------------------------------------------------------
# CodingTaskRunner audit write
# ---------------------------------------------------------------------------


def test_coding_task_audit_write_oserror_logs_once(
    errors_log, read_errors, tmp_path, monkeypatch,
):
    """The runner's _log_record must skip on OSError but log a single
    FilesystemError to errors.jsonl (not one per failed write).

    Block the audit log file by pre-creating a directory at its path
    so every open-for-append fails with OSError.
    """
    # Reset the per-process dedup flag so this test sees the first-failure log.
    monkeypatch.setattr(
        "ultron.coding.runner._AUDIT_WRITE_FAILURE_LOGGED", False,
    )

    from ultron.coding.runner import CodingTaskRunner
    from ultron.coding.bridge import CodingBridge

    class _StubBridge(CodingBridge):
        def name(self) -> str:
            return "stub"

        def submit(self, request):  # pragma: no cover - not used here
            raise NotImplementedError

    log_path = tmp_path / "coding_tasks.jsonl"
    log_path.mkdir(parents=True, exist_ok=True)  # block writes

    runner = CodingTaskRunner(
        bridge=_StubBridge(),
        log_path=log_path,
    )

    runner._log_record({"a": 1})
    runner._log_record({"a": 2})  # subsequent failures - should NOT log again
    runner._log_record({"a": 3})

    records = read_errors()
    fs_records = [r for r in records if r["error_type"] == "FilesystemError"]
    assert len(fs_records) == 1, (
        f"expected exactly one FilesystemError (first-failure dedup); "
        f"got {len(fs_records)}: {records!r}"
    )
    rec = fs_records[0]
    assert rec["dependency"] == "filesystem"
    assert "coding tasks audit-log write failed" in rec["message"]
