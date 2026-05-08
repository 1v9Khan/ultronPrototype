"""Smoke tests for the Phase 6 ScriptedClaudeBridge.

Verifies the mock bridge's basic behavior before we build scenarios on
top of it: emits events, updates session state via MCP store, runs
declare_complete through the handler, blocks on clarifications.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ultron.coding.bridge import EventKind, FileChangeKind, TaskRequest
from ultron.coding.mcp_server import UltronMCPServer
from ultron.coding.session import SessionStatus

from tests.coding.mock_bridge import ClaudeScript, ScriptedClaudeBridge


@pytest.fixture
def server():
    """A bare MCP server with no SSE listener (mock bridge calls into it
    directly)."""
    s = UltronMCPServer(host="127.0.0.1", port=0)  # port=0 means we won't start
    yield s
    # No stop() call -- we never started.


def test_mock_bridge_runs_simple_script_to_completion(server, tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    session = server.create_session(
        project_root=project, initial_prompt="hello world",
    )
    server.store.transition(session.session_id, SessionStatus.EXECUTING)

    script = (
        ClaudeScript()
        .progress("scaffolding", "set up", ["main.py"])
        .write_file("main.py", "print('hi')")
        .declare_complete(
            summary="hello world script done",
            files_created=["main.py"],
        )
    )
    bridge = ScriptedClaudeBridge(server, script, session_id=session.session_id)

    handle = bridge.submit(TaskRequest(
        task_prompt="hello", cwd=project, model="haiku",
        timeout_s=10.0, label="smoke",
    ))
    result = handle.wait(timeout=10.0)
    assert result is not None
    s = server.get_session_state(session.session_id)
    assert s.current_stage == "scaffolding"
    assert any(f.path == "main.py" for f in s.files_created)
    assert s.completion_claim is not None
    # No declare_complete handler wired -> session stays at VERIFYING
    # (the Phase 1 fallback). The handle still finishes; the session
    # status reflects "claim recorded; verification pending".
    assert s.status in (
        SessionStatus.VERIFYING, SessionStatus.COMPLETE,
    )


def test_mock_bridge_emits_events_to_listener(server, tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    session = server.create_session(
        project_root=project, initial_prompt="hi",
    )
    server.store.transition(session.session_id, SessionStatus.EXECUTING)
    script = (
        ClaudeScript()
        .progress("step1", "doing thing", ["a.py"])
        .write_file("a.py", "x = 1")
        .declare_complete(summary="done", files_created=["a.py"])
    )
    bridge = ScriptedClaudeBridge(server, script, session_id=session.session_id)

    captured = []
    handle = bridge.submit(TaskRequest(
        task_prompt="hi", cwd=project, model="haiku",
        timeout_s=10.0, label="smoke",
    ))
    handle.add_listener(captured.append)
    handle.wait(timeout=10.0)

    # First listener event might be missed since we attached after submit;
    # require at least the FILE_CHANGE.
    kinds = [e.kind for e in captured]
    assert EventKind.FILE_CHANGE in kinds


def test_mock_bridge_clarify_drives_responder(server, tmp_path: Path):
    """When a clarification responder is wired, .clarify() round-trips
    through it and the answer is recorded."""
    project = tmp_path / "p"
    project.mkdir()
    session = server.create_session(
        project_root=project, initial_prompt="hi",
    )
    server.store.transition(session.session_id, SessionStatus.EXECUTING)

    async def _responder(session_id, request, sess):
        return f"answer for: {request.question}"
    server.set_clarification_responder(_responder)

    answer_holder = {"value": None}
    def _capture(answer: str) -> None:
        answer_holder["value"] = answer

    script = (
        ClaudeScript()
        .clarify("what language?", on_answer=_capture)
        .declare_complete(summary="ok", files_created=[])
    )
    bridge = ScriptedClaudeBridge(server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="hi", cwd=project, model="haiku",
        timeout_s=10.0, label="clarify",
    ))
    handle.wait(timeout=10.0)

    assert answer_holder["value"] is not None
    assert "what language?" in answer_holder["value"]


def test_mock_bridge_declare_complete_runs_handler(server, tmp_path: Path):
    """When a declare_complete handler is wired, declare_complete dispatches
    to it and the response drives the bridge's terminal state."""
    project = tmp_path / "p"
    project.mkdir()
    session = server.create_session(
        project_root=project, initial_prompt="hi",
    )
    server.store.transition(session.session_id, SessionStatus.EXECUTING)

    handler_called = {"value": False}
    async def _handler(session_id):
        handler_called["value"] = True
        # Mark the session COMPLETE so the bridge ends successfully.
        server.store.transition(session_id, SessionStatus.COMPLETE)
        return "done"
    server.set_declare_complete_handler(_handler)

    script = ClaudeScript().declare_complete(summary="hi", files_created=[])
    bridge = ScriptedClaudeBridge(server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="hi", cwd=project, model="haiku",
        timeout_s=10.0, label="dc",
    ))
    result = handle.wait(timeout=10.0)
    assert handler_called["value"] is True
    assert result.success
    assert server.get_session_state(session.session_id).status == SessionStatus.COMPLETE


def test_mock_bridge_cancel_terminates_script(server, tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    session = server.create_session(
        project_root=project, initial_prompt="hi",
    )
    server.store.transition(session.session_id, SessionStatus.EXECUTING)
    script = (
        ClaudeScript()
        .progress("step1", "starting", [])
        .sleep(0.5)            # simulates work; cancel between sleep and next step
        .progress("step2", "more work", [])
        .declare_complete(summary="never reached", files_created=[])
    )
    bridge = ScriptedClaudeBridge(server, script, session_id=session.session_id)
    handle = bridge.submit(TaskRequest(
        task_prompt="hi", cwd=project, model="haiku",
        timeout_s=10.0, label="cancel",
    ))
    # Let the first step run, then cancel.
    time.sleep(0.05)
    handle.cancel()
    result = handle.wait(timeout=5.0)
    assert result is not None
    assert not result.success
    assert "cancel" in (result.error or "").lower() or "cancel" in (result.summary or "").lower()
