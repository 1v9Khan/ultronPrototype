"""Tests for ``ultron.openclaw_bridge.mcp_tools``.

Each tool's impl function is testable without standing up a real
MCP server — they're plain Python functions that read/write disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ultron.openclaw_bridge import mcp_tools


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch):
    """Redirect mcp_tools' alert log + session audit dir into tmp_path
    so tests don't touch the real project artifacts."""
    alert_log_path = tmp_path / "logs" / "heartbeat_alerts.jsonl"
    session_dir = tmp_path / "logs" / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    alert_log_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        mcp_tools, "_load_alert_log_path",
        lambda: alert_log_path,
    )
    monkeypatch.setattr(
        mcp_tools, "_load_session_audit_dir",
        lambda: session_dir,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# get_heartbeat_alerts_impl
# ---------------------------------------------------------------------------


def test_alerts_empty_when_no_log(isolated_paths: Path):
    payload = mcp_tools.get_heartbeat_alerts_impl()
    assert payload["count"] == 0
    assert payload["alerts"] == []


def test_alerts_returns_recorded(isolated_paths: Path):
    log = mcp_tools._alert_log()
    log.record("disk filling up", source="disk", severity="warn")
    log.record("stuck session", source="coding-queue", severity="info")
    payload = mcp_tools.get_heartbeat_alerts_impl()
    assert payload["count"] == 2
    bodies = [a["text"] for a in payload["alerts"]]
    assert "disk filling up" in bodies
    assert "stuck session" in bodies


def test_alerts_only_unacknowledged_default(isolated_paths: Path):
    log = mcp_tools._alert_log()
    a = log.record("first")
    log.record("second")
    log.acknowledge(a.alert_id)
    payload = mcp_tools.get_heartbeat_alerts_impl()
    assert payload["count"] == 1
    assert payload["alerts"][0]["text"] == "second"


def test_alerts_include_ack_when_flag_off(isolated_paths: Path):
    log = mcp_tools._alert_log()
    a = log.record("first")
    log.record("second")
    log.acknowledge(a.alert_id)
    payload = mcp_tools.get_heartbeat_alerts_impl(only_unacknowledged=False)
    assert payload["count"] == 2


def test_alerts_negative_input_rejected(isolated_paths: Path):
    payload = mcp_tools.get_heartbeat_alerts_impl(since_seconds_ago=-1)
    assert "error" in payload


def test_alerts_limit_enforced(isolated_paths: Path):
    log = mcp_tools._alert_log()
    for i in range(10):
        log.record(f"alert {i}")
    payload = mcp_tools.get_heartbeat_alerts_impl(limit=3)
    assert payload["count"] == 3


# ---------------------------------------------------------------------------
# acknowledge_alert_impl
# ---------------------------------------------------------------------------


def test_acknowledge_marks_alert(isolated_paths: Path):
    log = mcp_tools._alert_log()
    a = log.record("hi")
    result = mcp_tools.acknowledge_alert_impl(a.alert_id)
    assert result["acknowledged"] is True
    assert result["alert_id"] == a.alert_id


def test_acknowledge_unknown_id(isolated_paths: Path):
    result = mcp_tools.acknowledge_alert_impl("nonexistent")
    assert result["acknowledged"] is False
    assert "unknown" in result["reason"].lower()


def test_acknowledge_rejects_empty(isolated_paths: Path):
    result = mcp_tools.acknowledge_alert_impl("")
    assert result["acknowledged"] is False


# ---------------------------------------------------------------------------
# list_active_coding_sessions_impl
# ---------------------------------------------------------------------------


def test_active_sessions_empty(isolated_paths: Path):
    payload = mcp_tools.list_active_coding_sessions_impl()
    assert payload == {"count": 0, "sessions": []}


def test_active_sessions_returns_recent(isolated_paths: Path):
    audit_dir = mcp_tools._load_session_audit_dir()
    session_path = audit_dir / "sess-abc.jsonl"
    session_path.write_text(
        json.dumps({
            "event": "started",
            "stage": "executing",
            "project_root": "/proj/foo",
            "user_intent": "build me a Flask app",
        }) + "\n",
        encoding="utf-8",
    )
    payload = mcp_tools.list_active_coding_sessions_impl()
    assert payload["count"] == 1
    s = payload["sessions"][0]
    assert s["session_id"] == "sess-abc"
    assert s["last_event"] == "started"
    assert s["stage"] == "executing"


def test_active_sessions_skips_completed(isolated_paths: Path):
    audit_dir = mcp_tools._load_session_audit_dir()
    (audit_dir / "sess-done.jsonl").write_text(
        json.dumps({"event": "started"}) + "\n" +
        json.dumps({"event": "complete"}) + "\n",
        encoding="utf-8",
    )
    payload = mcp_tools.list_active_coding_sessions_impl()
    assert payload["count"] == 0


def test_active_sessions_skips_old(isolated_paths: Path):
    import os
    audit_dir = mcp_tools._load_session_audit_dir()
    p = audit_dir / "sess-old.jsonl"
    p.write_text(
        json.dumps({"event": "started"}) + "\n",
        encoding="utf-8",
    )
    # Backdate file mtime by 48 hours.
    backdated = p.stat().st_atime - 48 * 3600
    os.utime(p, (backdated, backdated))
    payload = mcp_tools.list_active_coding_sessions_impl(max_age_hours=24)
    assert payload["count"] == 0


# ---------------------------------------------------------------------------
# run_maintenance_impl
# ---------------------------------------------------------------------------


def test_run_maintenance_rejects_unknown_task():
    payload = mcp_tools.run_maintenance_impl(scope=["banana"])
    assert payload["status"] == "error"
    assert "banana" in payload["error"]


def test_run_maintenance_subprocess_success(monkeypatch, tmp_path: Path):
    """Mock subprocess.run to return a clean JSON payload."""
    fake_payload = {"status": "ok", "summary": {"decay_stale_facts": 0}}
    fake_proc = MagicMock(returncode=0, stdout=json.dumps(fake_payload), stderr="")
    monkeypatch.setattr(
        "ultron.openclaw_bridge.mcp_tools.subprocess.run",
        lambda *a, **kw: fake_proc,
    )
    # Force the script existence check to pass (it does in normal layout).
    monkeypatch.setattr(
        "ultron.openclaw_bridge.mcp_tools._project_root",
        lambda: tmp_path,
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_maintenance_for_cron.py").touch()
    result = mcp_tools.run_maintenance_impl(scope=["decay_stale_facts"])
    assert result["status"] == "ok"
    assert result["summary"]["decay_stale_facts"] == 0


def test_run_maintenance_subprocess_init_error(monkeypatch, tmp_path: Path):
    fake_proc = MagicMock(returncode=2, stdout="", stderr="qdrant unreachable")
    monkeypatch.setattr(
        "ultron.openclaw_bridge.mcp_tools.subprocess.run",
        lambda *a, **kw: fake_proc,
    )
    monkeypatch.setattr(
        "ultron.openclaw_bridge.mcp_tools._project_root",
        lambda: tmp_path,
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_maintenance_for_cron.py").touch()
    result = mcp_tools.run_maintenance_impl()
    assert result["status"] == "error"
    assert "qdrant" in result["stderr"].lower()


def test_run_maintenance_missing_wrapper(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "ultron.openclaw_bridge.mcp_tools._project_root",
        lambda: tmp_path,
    )
    result = mcp_tools.run_maintenance_impl()
    assert result["status"] == "error"
    assert "missing" in result["error"].lower()


# ---------------------------------------------------------------------------
# get_recent_voice_alerts_impl
# ---------------------------------------------------------------------------


def test_voice_alerts_renders_lines(isolated_paths: Path):
    log = mcp_tools._alert_log()
    log.record("disk filling", source="disk", severity="warn")
    log.record("session stuck", source="coding-queue", severity="info")
    result = mcp_tools.get_recent_voice_alerts_impl()
    assert result["count"] == 2
    assert len(result["lines"]) == 2
    # Lines start with severity prefix.
    assert result["lines"][0].startswith(("info:", "warn:", "error:"))


def test_voice_alerts_limit_clamped(isolated_paths: Path):
    log = mcp_tools._alert_log()
    for i in range(10):
        log.record(f"alert {i}")
    result = mcp_tools.get_recent_voice_alerts_impl(limit=3)
    assert result["count"] == 3
    assert len(result["lines"]) == 3


# ---------------------------------------------------------------------------
# build_server smoke test
# ---------------------------------------------------------------------------


def test_build_server_registers_all_tools():
    """Ensure all five tools are present in the FastMCP instance."""
    import asyncio

    server = mcp_tools.build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "get_heartbeat_alerts",
        "acknowledge_alert",
        "run_maintenance",
        "list_active_coding_sessions",
        "get_recent_voice_alerts",
    }
