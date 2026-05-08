"""MCP-server lifecycle and runtime failure modes.

Validates: bind failures and startup timeouts surface as MCPServerError
(both raised and logged); invoking an MCP tool with no active session
also produces a typed error; audit-log write failures degrade silently
but record a FilesystemError.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

os.environ.setdefault("ULTRON_CODING_MCP_ALLOW_ANY_ROOT", "1")

from ultron.coding.mcp_server import _AuditLog, UltronMCPServer  # noqa: E402
from ultron.errors import MCPServerError  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Lifecycle: bind failure
# ---------------------------------------------------------------------------


def test_bind_failure_raises_mcp_server_error_and_logs(
    errors_log, read_errors, tmp_path,
):
    """Two servers fight for the same port; the second hits OSError on
    bind. The original socket error is wrapped as MCPServerError and
    logged to errors.jsonl with dependency=mcp_server."""
    port = _free_port()

    first = UltronMCPServer(
        host="127.0.0.1", port=port, sse_path="/sse",
        log_path=tmp_path / "audit-1.jsonl",
        clarification_timeout_s=2.0,
    )
    first.start(ready_timeout_s=5.0)

    try:
        second = UltronMCPServer(
            host="127.0.0.1", port=port, sse_path="/sse",
            log_path=tmp_path / "audit-2.jsonl",
            clarification_timeout_s=2.0,
        )
        with pytest.raises(MCPServerError) as exc_info:
            second.start(ready_timeout_s=3.0)

        # The original OSError chains in via __cause__.
        assert exc_info.value.__cause__ is not None
        assert "bind failed" in exc_info.value.message

        records = read_errors()
        rec = next(r for r in records if r["error_type"] == "MCPServerError")
        assert rec["dependency"] == "mcp_server"
        assert rec["context"]["port"] == port
        assert "bind succeeds" in rec["recovery"]
    finally:
        first.stop(timeout_s=3.0)


# ---------------------------------------------------------------------------
# Tool call: no active session
# ---------------------------------------------------------------------------


def test_no_active_session_raises_typed_error_and_logs(
    errors_log, read_errors, tmp_path,
):
    """Calling a Claude-side tool helper before any session exists must
    produce an MCPServerError (not a bare RuntimeError) and log it."""
    server = UltronMCPServer(
        host="127.0.0.1", port=_free_port(), sse_path="/sse",
        log_path=tmp_path / "audit.jsonl",
        clarification_timeout_s=2.0,
    )

    with pytest.raises(MCPServerError) as exc_info:
        server._claude_active_session()

    assert "no session is active" in exc_info.value.message
    assert exc_info.value.context["active_session_count"] == 0

    records = read_errors()
    rec = next(r for r in records if r["error_type"] == "MCPServerError")
    assert rec["dependency"] == "mcp_server"
    assert "no session is active" in rec["message"]


# ---------------------------------------------------------------------------
# Audit-log write: filesystem failure
# ---------------------------------------------------------------------------


def test_audit_write_oserror_logs_filesystem_error(
    errors_log, read_errors, tmp_path,
):
    """Make the audit log path point at a directory we can't write to
    after construction (collide with an existing dir of the same name).
    The write must not raise but should record a FilesystemError."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    audit = _AuditLog(blocked)  # path is a directory, not a file
    audit.write(kind="test_event", payload="hello")

    records = read_errors()
    fs_records = [r for r in records if r["error_type"] == "FilesystemError"]
    assert fs_records, f"expected a FilesystemError; got {records!r}"
    rec = fs_records[0]
    assert rec["dependency"] == "filesystem"
    assert "audit-log write failed" in rec["message"]
    assert "system continues" in rec["recovery"]
