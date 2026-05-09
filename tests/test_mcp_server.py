"""Phase 1: UltronMCPServer lifecycle + Qwen-side tools + live SSE roundtrip.

Three test groups:

  - Lifecycle / Qwen-side: in-process. Fast.
  - .mcp.json helpers: filesystem-only.
  - Live SSE roundtrip: starts a real uvicorn server on a free port,
    connects with the official MCP SSE client, calls each Claude-side
    tool, validates the responses. No real Claude Code subprocess --
    that's covered by the e2e test in Phase 1f.

Tests pin ``ULTRON_CODING_MCP_ALLOW_ANY_ROOT=1`` so they can use tmp_path
as the project_root without the sandbox check kicking in.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

os.environ["ULTRON_CODING_MCP_ALLOW_ANY_ROOT"] = "1"

from ultron.coding.mcp_server import (  # noqa: E402
    UltronMCPServer,
    remove_mcp_config,
    write_mcp_config,
)
from ultron.coding.session import (  # noqa: E402
    ClarificationRequest,
    ProjectSession,
    SessionStatus,
)


def _free_port() -> int:
    """Find a free port for the test server. Avoids collisions when
    multiple tests run in parallel or in succession."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_server_constructs_without_starting(tmp_path: Path):
    """Importing + constructing must not bind the socket. The runner can
    delay start() until the orchestrator main loop actually runs."""
    server = UltronMCPServer(
        host="127.0.0.1",
        port=_free_port(),
        log_path=tmp_path / "mcp.jsonl",
    )
    assert not server.is_running()


def test_server_start_and_stop(tmp_path: Path):
    server = UltronMCPServer(
        host="127.0.0.1",
        port=_free_port(),
        log_path=tmp_path / "mcp.jsonl",
    )
    server.start(ready_timeout_s=5.0)
    try:
        assert server.is_running()
    finally:
        server.stop(timeout_s=5.0)
    # Port should be releasable -- we can grab it again. Add a small grace
    # for OS-level TIME_WAIT on some platforms.
    deadline = time.monotonic() + 3.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", server.port))
            break
        except OSError as e:
            last_error = e
            time.sleep(0.2)
    else:  # nobreak
        raise AssertionError(f"port stayed bound after stop(): {last_error}")


def test_server_double_start_is_idempotent(tmp_path: Path):
    server = UltronMCPServer(
        host="127.0.0.1",
        port=_free_port(),
        log_path=tmp_path / "mcp.jsonl",
    )
    server.start(ready_timeout_s=5.0)
    try:
        server.start(ready_timeout_s=5.0)  # no-op
        assert server.is_running()
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Qwen-side API (in-process)
# ---------------------------------------------------------------------------


def _server(tmp_path: Path) -> UltronMCPServer:
    return UltronMCPServer(
        host="127.0.0.1",
        port=_free_port(),
        log_path=tmp_path / "mcp.jsonl",
    )


def test_create_session_returns_planning_state(tmp_path: Path):
    server = _server(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    session = server.create_session(
        project_root=project, initial_prompt="build it", mode="edit",
    )
    assert session.status == SessionStatus.PLANNING
    assert session.user_intent == "build it"
    assert session.mode == "edit"
    assert session.session_id


def test_create_session_rejects_non_directory_project_root_for_edit(tmp_path: Path):
    server = _server(tmp_path)
    with pytest.raises(FileNotFoundError):
        server.create_session(
            project_root=tmp_path / "does_not_exist", initial_prompt="x",
            mode="edit",
        )


def test_create_session_rejects_file_at_project_root(tmp_path: Path):
    server = _server(tmp_path)
    bad = tmp_path / "not_a_dir.txt"
    bad.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        server.create_session(
            project_root=bad, initial_prompt="x", mode="new",
        )


def test_get_session_state_and_list_active(tmp_path: Path):
    server = _server(tmp_path)
    p1 = tmp_path / "p1"; p1.mkdir()
    p2 = tmp_path / "p2"; p2.mkdir()
    s1 = server.create_session(project_root=p1, initial_prompt="x")
    s2 = server.create_session(project_root=p2, initial_prompt="y")
    fetched = server.get_session_state(s1.session_id)
    assert fetched.session_id == s1.session_id

    active = set(server.list_active())
    assert active == {s1.session_id, s2.session_id}


def test_send_followup_validates_kind_and_records_adjustment(tmp_path: Path):
    server = _server(tmp_path)
    project = tmp_path / "p"; project.mkdir()
    session = server.create_session(project_root=project, initial_prompt="x")

    with pytest.raises(ValueError):
        server.send_followup(session.session_id, "hi", kind="not_a_kind")  # type: ignore[arg-type]

    server.send_followup(session.session_id, "switch to postgres", kind="adjustment")
    s = server.get_session_state(session.session_id)
    assert len(s.user_adjustments) == 1
    assert s.user_adjustments[0].text == "switch to postgres"


def test_terminate_session_removes_from_active(tmp_path: Path):
    server = _server(tmp_path)
    project = tmp_path / "p"; project.mkdir()
    session = server.create_session(project_root=project, initial_prompt="x")
    assert server.list_active() == [session.session_id]

    server.terminate_session(session.session_id, reason="user cancelled")
    assert server.list_active() == []
    assert server.get_session_state(session.session_id).status == SessionStatus.TERMINATED


def test_lookup_facts_returns_empty_when_no_memory_wired(tmp_path: Path):
    """Back-compat: tests that bypass the orchestrator must still see []."""
    server = _server(tmp_path)
    assert server.lookup_facts("anything") == []


def test_lookup_facts_audit_entry_marks_no_memory(tmp_path: Path):
    """The no-memory branch logs ``source=no_memory_wired`` so we can grep
    audit traces for stub fires."""
    log_path = tmp_path / "mcp.jsonl"
    server = UltronMCPServer(
        host="127.0.0.1",
        port=_free_port(),
        log_path=log_path,
    )
    server.lookup_facts("anything")
    line = log_path.read_text(encoding="utf-8").splitlines()[-1]
    record = json.loads(line)
    assert record["tool"] == "project.lookup_facts"
    assert record["source"] == "no_memory_wired"
    assert record["result_count"] == 0


def test_lookup_facts_calls_memory_search_facts(tmp_path: Path):
    """When memory IS wired, lookup_facts proxies to search_facts and
    returns dict-shaped rows."""

    class _StubMemory:
        def __init__(self):
            self.calls = []

        def search_facts(self, query, *, k, min_confidence, max_age_days):
            self.calls.append({
                "query": query, "k": k,
                "min_confidence": min_confidence,
                "max_age_days": max_age_days,
            })
            from ultron.memory.qdrant_store import FactRow
            return [
                FactRow(
                    fact="user prefers FastAPI",
                    confidence=0.92,
                    last_confirmed=time.time(),
                    category="preference",
                    score=0.95,
                    extracted_at=time.time(),
                    extracted_from=[1, 2],
                )
            ]

    stub = _StubMemory()
    server = UltronMCPServer(
        host="127.0.0.1",
        port=_free_port(),
        log_path=tmp_path / "mcp.jsonl",
        memory=stub,
    )
    rows = server.lookup_facts("Python framework")
    assert len(stub.calls) == 1
    assert stub.calls[0]["query"] == "Python framework"
    assert len(rows) == 1
    assert rows[0]["fact"] == "user prefers FastAPI"
    assert rows[0]["confidence"] == 0.92
    assert rows[0]["category"] == "preference"
    assert rows[0]["score"] == 0.95


def test_lookup_facts_swallows_search_facts_exception(tmp_path: Path):
    """A failure inside memory.search_facts must not propagate."""

    class _BoomMemory:
        def search_facts(self, *args, **kwargs):
            raise RuntimeError("simulated qdrant failure")

    server = UltronMCPServer(
        host="127.0.0.1",
        port=_free_port(),
        log_path=tmp_path / "mcp.jsonl",
        memory=_BoomMemory(),
    )
    assert server.lookup_facts("anything") == []


def test_lookup_facts_overrides_threshold_kwargs(tmp_path: Path):
    """Custom k / min_confidence / max_age_days flow through to search_facts."""

    class _CapturingMemory:
        def __init__(self):
            self.kwargs = None

        def search_facts(self, query, **kwargs):
            self.kwargs = {"query": query, **kwargs}
            return []

    cap = _CapturingMemory()
    server = UltronMCPServer(
        host="127.0.0.1",
        port=_free_port(),
        log_path=tmp_path / "mcp.jsonl",
        memory=cap,
    )
    server.lookup_facts(
        "Python", k=2, min_confidence=0.5, max_age_days=30.0,
    )
    assert cap.kwargs == {
        "query": "Python", "k": 2,
        "min_confidence": 0.5, "max_age_days": 30.0,
    }


def test_read_file_tree_returns_relative_paths_with_sizes(tmp_path: Path):
    server = _server(tmp_path)
    (tmp_path / "a.py").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("world", encoding="utf-8")
    # Skipped dir:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref", encoding="utf-8")

    out = server.read_file_tree(tmp_path)
    paths = {f["path"].replace("\\", "/"): f["size_bytes"] for f in out["files"]}
    assert paths == {"a.py": 5, "sub/b.py": 5}


# ---------------------------------------------------------------------------
# .mcp.json writer
# ---------------------------------------------------------------------------


def test_write_and_remove_mcp_config(tmp_path: Path):
    project = tmp_path / "p"
    project.mkdir()
    target = write_mcp_config(project, sse_url="http://127.0.0.1:99999/sse")
    assert target == project / ".mcp.json"
    assert target.is_file()
    config = json.loads(target.read_text(encoding="utf-8"))
    assert "ultron_coding" in config["mcpServers"]
    assert config["mcpServers"]["ultron_coding"]["url"] == "http://127.0.0.1:99999/sse"
    remove_mcp_config(project)
    assert not target.exists()


# ---------------------------------------------------------------------------
# Live SSE roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_roundtrip_lists_tools(tmp_path: Path):
    """Spin up a real server, connect with the SDK SSE client, list tools.
    Verifies the four Claude-facing tools are exposed with the expected
    names."""
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    server = _server(tmp_path)
    project = tmp_path / "p"; project.mkdir()
    server.create_session(project_root=project, initial_prompt="hi")
    server.start(ready_timeout_s=5.0)
    try:
        async with sse_client(server.sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {
                    "report_progress",
                    "request_clarification",
                    "declare_complete",
                    "report_test_results",
                } <= names
    finally:
        server.stop(timeout_s=5.0)


@pytest.mark.asyncio
async def test_sse_report_progress_records_state(tmp_path: Path):
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    server = _server(tmp_path)
    project = tmp_path / "p"; project.mkdir()
    session = server.create_session(project_root=project, initial_prompt="hi")
    server.start(ready_timeout_s=5.0)
    try:
        async with sse_client(server.sse_url) as (read, write):
            async with ClientSession(read, write) as cs:
                await cs.initialize()
                result = await cs.call_tool(
                    "report_progress",
                    {
                        "stage": "scaffolding",
                        "summary": "Created project skeleton",
                        "files_touched": ["main.py", "tests/test_main.py"],
                    },
                )
                assert not result.isError, result.content
        s = server.get_session_state(session.session_id)
        assert s.current_stage == "scaffolding"
        assert {f.path for f in s.files_created} == {"main.py", "tests/test_main.py"}
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_sse_request_clarification_blocks_until_resolved(tmp_path: Path):
    """The crucial piece: Claude's request_clarification call must NOT
    return until Qwen calls respond_to_clarification."""
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    server = _server(tmp_path)
    project = tmp_path / "p"; project.mkdir()
    session = server.create_session(project_root=project, initial_prompt="hi")
    server.start(ready_timeout_s=5.0)
    try:
        async with sse_client(server.sse_url) as (read, write):
            async with ClientSession(read, write) as cs:
                await cs.initialize()

                async def respond_after_delay() -> None:
                    # Wait until the request is actually pending (visible
                    # in session state), then resolve it from the Qwen
                    # side. ~150 ms is a reasonable upper bound.
                    deadline = asyncio.get_running_loop().time() + 5.0
                    while True:
                        s = server.get_session_state(session.session_id)
                        if s.pending_clarification is not None:
                            request_id = s.pending_clarification.request_id
                            await asyncio.sleep(0.15)
                            ok = server.respond_to_clarification(
                                request_id, "use sqlite", decision_path="test",
                            )
                            assert ok, "respond_to_clarification reported no waiter"
                            return
                        if asyncio.get_running_loop().time() > deadline:
                            raise AssertionError("clarification never reached pending state")
                        await asyncio.sleep(0.02)

                resp_task = asyncio.create_task(respond_after_delay())

                t0 = time.monotonic()
                result = await asyncio.wait_for(
                    cs.call_tool(
                        "request_clarification",
                        {
                            "question": "SQLite or Postgres?",
                            "options": ["sqlite", "postgres"],
                            "urgency": "blocking",
                        },
                    ),
                    timeout=10.0,
                )
                elapsed = time.monotonic() - t0
                await resp_task
                assert not result.isError, result.content
                # The tool returns the answer string in text content blocks.
                text_blocks = [c for c in result.content if c.type == "text"]
                assert text_blocks
                assert "sqlite" in text_blocks[0].text.lower()
                # And the call really did wait for the supervisor.
                assert elapsed >= 0.1, f"call returned too quickly: {elapsed*1000:.0f} ms"
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_sse_declare_complete_records_claim(tmp_path: Path):
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    server = _server(tmp_path)
    project = tmp_path / "p"; project.mkdir()
    session = server.create_session(project_root=project, initial_prompt="hi")
    # Move to executing so the verifying-transition is legal.
    server.store.transition(session.session_id, SessionStatus.EXECUTING)
    server.start(ready_timeout_s=5.0)
    try:
        async with sse_client(server.sse_url) as (read, write):
            async with ClientSession(read, write) as cs:
                await cs.initialize()
                result = await cs.call_tool(
                    "declare_complete",
                    {
                        "summary": "all done",
                        "entry_point": "main.py",
                        "run_command": "python main.py",
                        "files_created": ["main.py", "tests/test_main.py"],
                        "files_modified": [],
                    },
                )
                assert not result.isError, result.content
        s = server.get_session_state(session.session_id)
        assert s.status == SessionStatus.VERIFYING
        assert s.completion_claim is not None
        assert s.completion_claim.entry_point == "main.py"
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_sse_report_test_results_updates_session(tmp_path: Path):
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    server = _server(tmp_path)
    project = tmp_path / "p"; project.mkdir()
    session = server.create_session(project_root=project, initial_prompt="hi")
    server.start(ready_timeout_s=5.0)
    try:
        async with sse_client(server.sse_url) as (read, write):
            async with ClientSession(read, write) as cs:
                await cs.initialize()
                result = await cs.call_tool(
                    "report_test_results",
                    {
                        "passing": 5,
                        "failing": 0,
                        "skipped": 1,
                        "details": "all green",
                    },
                )
                assert not result.isError, result.content
        s = server.get_session_state(session.session_id)
        assert s.test_status.passing == 5
        assert s.test_status.failing == 0
        assert s.test_status.skipped == 1
    finally:
        server.stop()
