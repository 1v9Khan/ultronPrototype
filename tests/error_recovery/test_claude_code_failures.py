"""Claude Code subprocess failure modes: launch failure, timeout,
nonzero exit, stream-json error events. Plus the API-error pattern
detector that decides between ClaudeCodeError and AnthropicAPIError.

Validates: failures land in errors.jsonl with the right shape and
dependency tag; the AnthropicAPIError vs ClaudeCodeError split is
driven by the stream-json error text.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ultron.coding import direct_bridge as bridge_mod
from ultron.coding.bridge import TaskRequest
from ultron.coding.direct_bridge import (
    DirectClaudeCodeBridge,
    DirectTaskHandle,
    _looks_like_anthropic_api_error,
)


# ---------------------------------------------------------------------------
# Pattern detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "rate_limit_exceeded: requests per minute",
    "API rate limit reached",
    "the model is overloaded right now",
    "invalid_api_key",
    "Authentication_Error: bad token",
    "anthropic returned 529",
    "got HTTP 529 Overloaded",
])
def test_pattern_detector_recognizes_api_errors(text):
    assert _looks_like_anthropic_api_error(text) is True


@pytest.mark.parametrize("text", [
    "",
    "subprocess died",
    "file not found: /tmp/x",
    "JSONDecodeError: line 1 column 1",
    "permission denied opening config",
])
def test_pattern_detector_rejects_non_api_errors(text):
    assert _looks_like_anthropic_api_error(text) is False


# ---------------------------------------------------------------------------
# Stream-json error events: drive _handle_stream_event directly
# ---------------------------------------------------------------------------


def _make_minimal_handle(tmp_path) -> DirectTaskHandle:
    """Build a DirectTaskHandle without spawning a subprocess.

    Uses ``__new__`` to skip ``__init__`` (which calls ``_launch``).
    Populates only the fields ``_handle_stream_event`` reads.
    """
    h = DirectTaskHandle.__new__(DirectTaskHandle)
    h._task_id = "test-task-id"
    h._listeners = []
    h._listeners_lock = threading.Lock()
    from ultron.coding.bridge import TaskState, _StateMutex
    h._state = _StateMutex(TaskState(
        label="test", task_prompt="x", cwd=tmp_path, started_at=time.time(),
    ))
    h._request = TaskRequest(
        task_prompt="x", cwd=tmp_path, model="haiku", label="test",
    )
    return h


def test_stream_json_api_error_logs_anthropic_api_error(
    errors_log, read_errors, tmp_path,
):
    h = _make_minimal_handle(tmp_path)
    h._handle_stream_event({
        "type": "error",
        "error": "rate_limit_exceeded: too many requests",
    })

    records = read_errors()
    assert len(records) == 1
    rec = records[0]
    assert rec["error_type"] == "AnthropicAPIError"
    assert rec["dependency"] == "anthropic_api"
    assert "rate_limit" in rec["context"]["snippet"]
    assert "user notified" in rec["recovery"]


def test_stream_json_generic_error_logs_claude_code_error(
    errors_log, read_errors, tmp_path,
):
    h = _make_minimal_handle(tmp_path)
    h._handle_stream_event({
        "type": "error",
        "error": "subprocess crashed in tool handler",
    })

    records = read_errors()
    assert len(records) == 1
    rec = records[0]
    assert rec["error_type"] == "ClaudeCodeError"
    assert rec["dependency"] == "claude_code"
    assert "subprocess crashed" in rec["context"]["snippet"]


def test_stream_json_message_field_used_when_error_missing(
    errors_log, read_errors, tmp_path,
):
    """The handler accepts ``message`` as the alt key when ``error``
    is absent."""
    h = _make_minimal_handle(tmp_path)
    h._handle_stream_event({
        "type": "error",
        "message": "overloaded by upstream",
    })

    records = read_errors()
    assert len(records) == 1
    assert records[0]["error_type"] == "AnthropicAPIError"


# ---------------------------------------------------------------------------
# Launch failure: patch subprocess.Popen to raise
# ---------------------------------------------------------------------------


def test_launch_failure_logs_claude_code_error(errors_log, read_errors, tmp_path):
    bridge = DirectClaudeCodeBridge(claude_cli=sys.executable)

    request = TaskRequest(
        task_prompt="hello",
        cwd=tmp_path,
        model="haiku",
        label="launch-fail-test",
        timeout_s=2.0,
    )

    with patch("subprocess.Popen", side_effect=OSError("simulated launch failure")):
        handle = bridge.submit(request)
        result = handle.wait(timeout=5.0)

    assert result.success is False
    assert result.exit_status == -1

    records = read_errors()
    launch_records = [r for r in records if r["error_type"] == "ClaudeCodeError"]
    assert launch_records, f"expected a ClaudeCodeError; got {records!r}"
    rec = launch_records[0]
    assert rec["dependency"] == "claude_code"
    assert "failed to launch" in rec["message"]
    assert rec["context"]["task_id"] == handle.task_id()
    assert rec["context"]["label"] == "launch-fail-test"
    assert "task aborted" in rec["recovery"]


# ---------------------------------------------------------------------------
# Subprocess timeout: real subprocess, short timeout
# ---------------------------------------------------------------------------


def test_subprocess_timeout_logs_claude_code_error(
    errors_log, read_errors, tmp_path,
):
    """Run a fake claude that sleeps; bridge times it out."""
    fake_claude = tmp_path / "fake_claude_sleeper.py"
    fake_claude.write_text(
        "import time, sys\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    bridge = DirectClaudeCodeBridge(claude_cli=sys.executable)
    # Inject the script as the first argument so python runs it.
    request = TaskRequest(
        task_prompt=str(fake_claude),
        cwd=tmp_path,
        model="haiku",
        label="timeout-test",
        timeout_s=0.5,
    )

    with patch.object(
        DirectClaudeCodeBridge, "_build_argv",
        return_value=[sys.executable, str(fake_claude)],
    ):
        handle = bridge.submit(request)
        result = handle.wait(timeout=15.0)

    assert result.success is False
    records = read_errors()
    timeout_records = [
        r for r in records
        if r["error_type"] == "ClaudeCodeError"
        and "timeout" in r["message"]
    ]
    assert timeout_records, f"expected a timeout ClaudeCodeError; got {records!r}"
    rec = timeout_records[0]
    assert rec["dependency"] == "claude_code"
    assert rec["context"]["timeout_s"] == 0.5
    assert "cancelled" in rec["recovery"]


# ---------------------------------------------------------------------------
# Nonzero exit: real subprocess, exits 1 quickly
# ---------------------------------------------------------------------------


def test_nonzero_exit_logs_claude_code_error(
    errors_log, read_errors, tmp_path,
):
    fake_claude = tmp_path / "fake_claude_exit1.py"
    fake_claude.write_text(
        "import sys\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )

    bridge = DirectClaudeCodeBridge(claude_cli=sys.executable)
    request = TaskRequest(
        task_prompt="x",
        cwd=tmp_path,
        model="haiku",
        label="exit1-test",
        timeout_s=10.0,
    )

    with patch.object(
        DirectClaudeCodeBridge, "_build_argv",
        return_value=[sys.executable, str(fake_claude)],
    ):
        handle = bridge.submit(request)
        result = handle.wait(timeout=15.0)

    assert result.success is False
    assert result.exit_status == 1

    records = read_errors()
    exit_records = [
        r for r in records
        if r["error_type"] == "ClaudeCodeError"
        and "exited nonzero" in r["message"]
    ]
    assert exit_records, f"expected a nonzero-exit ClaudeCodeError; got {records!r}"
    rec = exit_records[0]
    assert rec["dependency"] == "claude_code"
    assert rec["context"]["exit_status"] == 1
    assert rec["context"]["label"] == "exit1-test"
