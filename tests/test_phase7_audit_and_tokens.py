"""Phase 7 — per-session audit log + token usage tracking.

Covers:
  * SessionAuditWriter writes JSONL + handles missing dir gracefully.
  * SessionStore auto-logs every state-affecting method to the per-session log.
  * Coordinator mirrors clarification + verification events.
  * Tokens flow: bridge USAGE event -> runner listener -> store.record_tokens.
  * Budget warnings fire at 80% and halt fires at 100%.
  * send_followup refuses once the budget is halted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

from ultron.coding.audit import SessionAuditWriter
from ultron.coding.bridge import EventKind, TaskEvent, TaskRequest
from ultron.coding.coordinator import ConversationCoordinator
from ultron.coding.runner import CodingTaskRunner
from ultron.coding.session import (
    ClarificationRequest,
    SessionStatus,
    SessionStore,
)
from ultron.coding.verification import Verifier

from tests.coding.mock_bridge import ClaudeScript, ScriptedClaudeBridge

os.environ.setdefault("ULTRON_CODING_MCP_ALLOW_ANY_ROOT", "1")


# ---------------------------------------------------------------------------
# SessionAuditWriter
# ---------------------------------------------------------------------------


def test_audit_writer_writes_jsonl(tmp_path: Path):
    log_dir = tmp_path / "sessions"
    writer = SessionAuditWriter(log_dir=log_dir)
    writer.write("abc123", "test_event", foo="bar", n=42)
    path = log_dir / "abc123.jsonl"
    assert path.is_file()
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["session_id"] == "abc123"
    assert record["event"] == "test_event"
    assert record["foo"] == "bar"
    assert record["n"] == 42
    assert "ts" in record


def test_audit_writer_disabled_when_log_dir_none(tmp_path: Path):
    writer = SessionAuditWriter(log_dir=None)
    # Should not raise, should not create files.
    writer.write("abc", "event")
    # No files anywhere.
    assert not list(tmp_path.iterdir())


def test_audit_writer_appends_multiple_records(tmp_path: Path):
    log_dir = tmp_path / "sessions"
    writer = SessionAuditWriter(log_dir=log_dir)
    writer.write("s1", "event_a")
    writer.write("s1", "event_b")
    writer.write("s2", "event_c")
    s1_lines = (log_dir / "s1.jsonl").read_text(encoding="utf-8").splitlines()
    s2_lines = (log_dir / "s2.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(s1_lines) == 2
    assert len(s2_lines) == 1
    assert json.loads(s1_lines[0])["event"] == "event_a"
    assert json.loads(s1_lines[1])["event"] == "event_b"


# ---------------------------------------------------------------------------
# SessionStore auto-logging
# ---------------------------------------------------------------------------


def _read_events(log_dir: Path, session_id: str) -> List[dict]:
    path = log_dir / f"{session_id}.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_store_auto_logs_every_state_change(tmp_path: Path):
    log_dir = tmp_path / "sessions"
    store = SessionStore(audit_writer=SessionAuditWriter(log_dir=log_dir))
    s = store.create(
        project_root=tmp_path / "p", user_intent="hello",
    )
    store.transition(s.session_id, SessionStatus.EXECUTING)
    store.record_stage(
        s.session_id, stage="step1", summary="did stuff",
        files_touched=["a.py"],
    )
    store.set_pending_clarification(s.session_id, ClarificationRequest(
        request_id="r1", question="what?", urgency="blocking",
    ))
    store.resolve_clarification(s.session_id, "answer-text", "rule_answer")
    store.record_test_results(
        s.session_id, passing=2, failing=0, skipped=0, details="ok",
    )
    store.record_adjustment(s.session_id, "tweak it", rendered_prompt="render")

    events = _read_events(log_dir, s.session_id)
    event_types = [e["event"] for e in events]
    assert event_types == [
        "session_created",
        "transition",
        "stage_recorded",
        "clarification_asked",
        "clarification_resolved",
        "test_results",
        "adjustment_recorded",
    ]


def test_store_logs_completion_claim(tmp_path: Path):
    from ultron.coding.session import CompletionClaim

    log_dir = tmp_path / "sessions"
    store = SessionStore(audit_writer=SessionAuditWriter(log_dir=log_dir))
    s = store.create(project_root=tmp_path / "p", user_intent="hi")
    store.record_completion_claim(s.session_id, CompletionClaim(
        summary="done", entry_point="main.py",
        files_created=["main.py"],
    ))
    events = _read_events(log_dir, s.session_id)
    types = [e["event"] for e in events]
    assert "completion_claimed" in types
    completion_event = next(e for e in events if e["event"] == "completion_claimed")
    assert completion_event["entry_point"] == "main.py"
    assert "main.py" in completion_event["files_created"]


def test_store_with_no_audit_writer_works_normally(tmp_path: Path):
    """Backward compat: existing tests construct SessionStore() with no
    audit writer. All methods must work without error."""
    store = SessionStore()
    s = store.create(project_root=tmp_path / "p", user_intent="hi")
    store.transition(s.session_id, SessionStatus.EXECUTING)
    store.record_stage(
        s.session_id, stage="x", summary="y", files_touched=[],
    )
    # No exception, no files written anywhere.
    assert s.status == SessionStatus.EXECUTING


# ---------------------------------------------------------------------------
# Coordinator mirrors clarification + verification into per-session log
# ---------------------------------------------------------------------------


def test_coordinator_clarification_decision_lands_in_session_log(tmp_path: Path):
    import asyncio

    log_dir = tmp_path / "sessions"
    store = SessionStore(audit_writer=SessionAuditWriter(log_dir=log_dir))
    coordinator = ConversationCoordinator(
        store=store, llm=None,
        log_path=tmp_path / "clarifications.jsonl",
    )
    s = store.create(project_root=tmp_path / "p", user_intent="hello")
    store.transition(s.session_id, SessionStatus.EXECUTING)
    store.set_pending_clarification(s.session_id, ClarificationRequest(
        request_id="r1",
        question="What test framework should I use?",
    ))
    store.transition(s.session_id, SessionStatus.AWAITING_CLARIFICATION)
    request = store.get(s.session_id).pending_clarification

    # Rule-answer fast path: no LLM needed.
    answer = asyncio.run(coordinator.decide_clarification(
        s.session_id, request, store.get(s.session_id),
    ))
    assert "pytest" in answer.lower() or answer  # at minimum non-empty

    events = _read_events(log_dir, s.session_id)
    types = [e["event"] for e in events]
    assert "clarification_decided" in types
    decision = next(e for e in events if e["event"] == "clarification_decided")
    assert decision["request_id"] == "r1"
    assert "rule_answer" in decision["decision_path"]


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


def test_record_tokens_increments_session_total(tmp_path: Path):
    store = SessionStore()
    s = store.create(project_root=tmp_path / "p", user_intent="hi")
    total = store.record_tokens(
        s.session_id, input_tokens=100, output_tokens=50,
        cache_creation_tokens=20, cache_read_tokens=200,
    )
    assert total == 170  # input + output + cache_creation; cache_read excluded
    s2 = store.get(s.session_id)
    assert s2.tokens_used == 170
    assert s2.tokens_input_total == 100
    assert s2.tokens_output_total == 50
    assert s2.tokens_cache_creation_total == 20
    assert s2.tokens_cache_read_total == 200

    # Subsequent calls accumulate.
    total2 = store.record_tokens(
        s.session_id, input_tokens=10, output_tokens=5,
    )
    assert total2 == 185


def test_record_tokens_logs_to_audit(tmp_path: Path):
    log_dir = tmp_path / "sessions"
    store = SessionStore(audit_writer=SessionAuditWriter(log_dir=log_dir))
    s = store.create(project_root=tmp_path / "p", user_intent="hi")
    store.record_tokens(s.session_id, input_tokens=10, output_tokens=5)
    events = _read_events(log_dir, s.session_id)
    types = [e["event"] for e in events]
    assert "tokens_recorded" in types
    e = next(e for e in events if e["event"] == "tokens_recorded")
    assert e["delta"] == 15
    assert e["total"] == 15


# ---------------------------------------------------------------------------
# Bridge -> runner -> store integration
# ---------------------------------------------------------------------------


def test_runner_forwards_usage_events_to_session(tmp_path: Path):
    """ScriptedClaudeBridge emits USAGE events via .tokens(). The runner's
    listener forwards them to store.record_tokens, which updates the
    session's tokens_used."""
    from ultron.coding.mcp_server import UltronMCPServer

    server = UltronMCPServer(host="127.0.0.1", port=0)
    s = server.create_session(
        project_root=tmp_path / "p", initial_prompt="hi",
    )
    (tmp_path / "p").mkdir()
    server.store.transition(s.session_id, SessionStatus.EXECUTING)

    runner = CodingTaskRunner(
        bridge=ScriptedClaudeBridge(server, ClaudeScript(), session_id=s.session_id),
        log_path=tmp_path / "audit.jsonl",
        store=server.store,
    )
    runner.bind_session(s.session_id)

    # Build a script with token reports + a final declare_complete.
    script = (
        ClaudeScript()
        .progress("step1", "ok", [])
        .tokens(input=200, output=100)
        .progress("step2", "ok", [])
        .tokens(input=300, output=150)
        .declare_complete(summary="ok", files_created=[])
    )
    runner.bridge = ScriptedClaudeBridge(server, script, session_id=s.session_id)

    handle = runner.start_task(TaskRequest(
        task_prompt="hi", cwd=tmp_path / "p", model="haiku",
        timeout_s=10.0, label="tok-test",
    ))
    handle.wait(timeout=10.0)

    s_after = server.get_session_state(s.session_id)
    # 200 + 100 + 300 + 150 = 750 (no cache)
    assert s_after.tokens_used == 750


# ---------------------------------------------------------------------------
# Budget warning + halt
# ---------------------------------------------------------------------------


def test_runner_queues_warning_when_budget_threshold_crossed(tmp_path: Path, monkeypatch):
    from config import settings as _settings
    from ultron.coding.mcp_server import UltronMCPServer

    monkeypatch.setattr(_settings, "CODING_TOKEN_BUDGET_PER_SESSION", 1000)
    monkeypatch.setattr(_settings, "CODING_TOKEN_WARNING_THRESHOLD", 0.8)

    server = UltronMCPServer(host="127.0.0.1", port=0)
    s = server.create_session(
        project_root=tmp_path / "p", initial_prompt="hi",
    )
    (tmp_path / "p").mkdir()
    server.store.transition(s.session_id, SessionStatus.EXECUTING)

    runner = CodingTaskRunner(
        bridge=ScriptedClaudeBridge(server, ClaudeScript(), session_id=s.session_id),
        log_path=tmp_path / "audit.jsonl",
        store=server.store,
    )
    runner.bind_session(s.session_id)

    # 850 tokens crosses the 80% threshold (= 800).
    script = (
        ClaudeScript()
        .tokens(input=500, output=350)
        .declare_complete(summary="ok", files_created=[])
    )
    runner.bridge = ScriptedClaudeBridge(server, script, session_id=s.session_id)
    handle = runner.start_task(TaskRequest(
        task_prompt="hi", cwd=tmp_path / "p", model="haiku",
        timeout_s=10.0, label="warn-test",
    ))
    handle.wait(timeout=10.0)

    warning = runner.pop_budget_warning()
    assert warning is not None
    assert "%" in warning  # mentions the percentage
    # Halted not yet (we're at 85%, not 100%).
    s_after = server.get_session_state(s.session_id)
    assert s_after.budget_warning_emitted
    assert not s_after.budget_halted


def test_runner_halts_at_100_percent_budget(tmp_path: Path, monkeypatch):
    from config import settings as _settings
    from ultron.coding.mcp_server import UltronMCPServer

    monkeypatch.setattr(_settings, "CODING_TOKEN_BUDGET_PER_SESSION", 500)
    monkeypatch.setattr(_settings, "CODING_TOKEN_WARNING_THRESHOLD", 0.8)

    server = UltronMCPServer(host="127.0.0.1", port=0)
    s = server.create_session(
        project_root=tmp_path / "p", initial_prompt="hi",
    )
    (tmp_path / "p").mkdir()
    server.store.transition(s.session_id, SessionStatus.EXECUTING)

    runner = CodingTaskRunner(
        bridge=ScriptedClaudeBridge(server, ClaudeScript(), session_id=s.session_id),
        log_path=tmp_path / "audit.jsonl",
        store=server.store,
    )
    runner.bind_session(s.session_id)

    # 600 tokens > 500-token budget.
    script = (
        ClaudeScript()
        .tokens(input=600, output=0)
        .declare_complete(summary="ok", files_created=[])
    )
    runner.bridge = ScriptedClaudeBridge(server, script, session_id=s.session_id)
    handle = runner.start_task(TaskRequest(
        task_prompt="hi", cwd=tmp_path / "p", model="haiku",
        timeout_s=10.0, label="halt-test",
    ))
    handle.wait(timeout=10.0)

    s_after = server.get_session_state(s.session_id)
    assert s_after.budget_halted

    # send_followup refuses now.
    result = runner.send_followup("more work please", kind="adjustment")
    assert result is None


def test_runner_logs_initial_prompt_to_session_log(tmp_path: Path):
    """Phase 7 spec: the per-session log captures every prompt sent to
    Claude. start_task -> claude_prompt_sent (kind=initial)."""
    from ultron.coding.mcp_server import UltronMCPServer

    log_dir = tmp_path / "sessions"
    server = UltronMCPServer(host="127.0.0.1", port=0)
    # Wire in our own audit writer so the per-session log lands here.
    server.session_audit = SessionAuditWriter(log_dir=log_dir)
    server.store._audit = server.session_audit
    s = server.create_session(
        project_root=tmp_path / "p", initial_prompt="build a hello world script",
    )
    (tmp_path / "p").mkdir()
    server.store.transition(s.session_id, SessionStatus.EXECUTING)

    runner = CodingTaskRunner(
        bridge=ScriptedClaudeBridge(server, ClaudeScript(), session_id=s.session_id),
        log_path=tmp_path / "audit.jsonl",
        store=server.store,
    )
    runner.bind_session(s.session_id)

    script = ClaudeScript().declare_complete(summary="ok", files_created=[])
    runner.bridge = ScriptedClaudeBridge(server, script, session_id=s.session_id)
    runner.start_task(TaskRequest(
        task_prompt="please write hello world", cwd=tmp_path / "p",
        model="haiku", timeout_s=10.0, label="prompt-log",
    )).wait(timeout=10.0)

    events = _read_events(log_dir, s.session_id)
    types = [e["event"] for e in events]
    assert "claude_prompt_sent" in types, types
    prompt_events = [e for e in events if e["event"] == "claude_prompt_sent"]
    initial = next(e for e in prompt_events if e["kind"] == "initial")
    assert initial["prompt"] == "please write hello world"
    assert initial["model"] == "haiku"
    assert initial["label"] == "prompt-log"


def test_runner_logs_followup_prompts_to_session_log(tmp_path: Path):
    """Each send_followup also lands in the per-session log with its kind."""
    from ultron.coding.mcp_server import UltronMCPServer

    log_dir = tmp_path / "sessions"
    server = UltronMCPServer(host="127.0.0.1", port=0)
    server.session_audit = SessionAuditWriter(log_dir=log_dir)
    server.store._audit = server.session_audit
    s = server.create_session(
        project_root=tmp_path / "p", initial_prompt="hi",
    )
    (tmp_path / "p").mkdir()
    server.store.transition(s.session_id, SessionStatus.EXECUTING)

    runner = CodingTaskRunner(
        bridge=ScriptedClaudeBridge(server, ClaudeScript(), session_id=s.session_id),
        log_path=tmp_path / "audit.jsonl",
        store=server.store,
    )
    runner.bind_session(s.session_id)

    # Run an initial task that finishes.
    script1 = ClaudeScript().declare_complete(summary="ok", files_created=[])
    runner.bridge = ScriptedClaudeBridge(server, script1, session_id=s.session_id)
    runner.start_task(TaskRequest(
        task_prompt="initial work", cwd=tmp_path / "p",
        model="haiku", timeout_s=10.0, label="initial",
    )).wait(timeout=10.0)
    # Now a follow-up. The runner needs another scripted bridge for the resume
    # path; reusing the same bridge instance is fine, swap the script.
    runner.bridge = ScriptedClaudeBridge(
        server,
        ClaudeScript().declare_complete(summary="ok-2", files_created=[]),
        session_id=s.session_id,
    )
    runner.send_followup("Try a different approach", kind="adjustment")

    events = _read_events(log_dir, s.session_id)
    prompt_events = [e for e in events if e["event"] == "claude_prompt_sent"]
    kinds = sorted(e["kind"] for e in prompt_events)
    assert "initial" in kinds
    assert "followup_adjustment" in kinds
    followup = next(e for e in prompt_events if e["kind"] == "followup_adjustment")
    assert followup["prompt"] == "Try a different approach"


def test_runner_pop_budget_warning_consumes_once(tmp_path: Path, monkeypatch):
    from config import settings as _settings
    from ultron.coding.mcp_server import UltronMCPServer

    monkeypatch.setattr(_settings, "CODING_TOKEN_BUDGET_PER_SESSION", 1000)

    server = UltronMCPServer(host="127.0.0.1", port=0)
    s = server.create_session(
        project_root=tmp_path / "p", initial_prompt="hi",
    )
    (tmp_path / "p").mkdir()
    server.store.transition(s.session_id, SessionStatus.EXECUTING)

    runner = CodingTaskRunner(
        bridge=ScriptedClaudeBridge(server, ClaudeScript(), session_id=s.session_id),
        log_path=tmp_path / "audit.jsonl",
        store=server.store,
    )
    runner.bind_session(s.session_id)
    script = (
        ClaudeScript()
        .tokens(input=900, output=0)
        .declare_complete(summary="ok", files_created=[])
    )
    runner.bridge = ScriptedClaudeBridge(server, script, session_id=s.session_id)
    runner.start_task(TaskRequest(
        task_prompt="hi", cwd=tmp_path / "p", model="haiku",
        timeout_s=10.0, label="consume-test",
    )).wait(timeout=10.0)

    first = runner.pop_budget_warning()
    second = runner.pop_budget_warning()
    assert first is not None
    assert second is None  # consumed
