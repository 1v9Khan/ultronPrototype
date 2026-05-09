"""Tests for ``ultron.openclaw_bridge.system_status.SystemStatusReporter``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ultron.openclaw_bridge.heartbeat_alerts import HeartbeatAlertLog
from ultron.openclaw_bridge.system_status import SystemStatusReporter
from ultron.openclaw_routing.intents import SystemStatusIntent


@pytest.fixture
def alert_log(tmp_path: Path) -> HeartbeatAlertLog:
    return HeartbeatAlertLog(tmp_path / "alerts.jsonl", retention_days=30)


def _patch_sessions(sessions):
    """Patch the session-listing impl so the reporter sees a curated list."""
    return patch(
        "ultron.openclaw_bridge.system_status.list_active_coding_sessions_impl",
        return_value={"count": len(sessions), "sessions": sessions},
    )


# ---------------------------------------------------------------------------
# focus="alerts"
# ---------------------------------------------------------------------------


def test_alerts_focus_no_alerts(alert_log: HeartbeatAlertLog):
    r = SystemStatusReporter(alert_log)
    report = r.report(SystemStatusIntent(focus="alerts"))
    assert report.focus == "alerts"
    assert report.alerts == ()
    assert "no pending" in report.voice_message.lower()


def test_alerts_focus_single_alert(alert_log: HeartbeatAlertLog):
    alert_log.record("disk filling")
    r = SystemStatusReporter(alert_log)
    report = r.report(SystemStatusIntent(focus="alerts"))
    assert len(report.alerts) == 1
    assert "one alert" in report.voice_message.lower()
    assert "disk filling" in report.voice_message.lower()


def test_alerts_focus_multiple_alerts(alert_log: HeartbeatAlertLog):
    alert_log.record("first")
    alert_log.record("second")
    alert_log.record("third")
    r = SystemStatusReporter(alert_log, max_alerts=3)
    report = r.report(SystemStatusIntent(focus="alerts"))
    assert len(report.alerts) == 3
    assert "3 pending alerts" in report.voice_message.lower()
    # All three present in the message.
    assert "first" in report.voice_message
    assert "second" in report.voice_message
    assert "third" in report.voice_message


def test_alerts_focus_capped_at_max(alert_log: HeartbeatAlertLog):
    """Reporter respects max_alerts even when more exist."""
    for i in range(10):
        alert_log.record(f"alert {i}")
    r = SystemStatusReporter(alert_log, max_alerts=2)
    report = r.report(SystemStatusIntent(focus="alerts"))
    assert len(report.alerts) == 2


# ---------------------------------------------------------------------------
# focus="projects"
# ---------------------------------------------------------------------------


def test_projects_focus_none(alert_log: HeartbeatAlertLog):
    with _patch_sessions([]):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="projects"))
    assert report.focus == "projects"
    assert "nothing active" in report.voice_message.lower()


def test_projects_focus_one(alert_log: HeartbeatAlertLog):
    sessions = [{
        "session_id": "abc",
        "project_root": "/p/myproject",
        "stage": "executing",
    }]
    with _patch_sessions(sessions):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="projects"))
    assert "one active" in report.voice_message.lower()
    assert "myproject" in report.voice_message
    assert "executing" in report.voice_message


def test_projects_focus_multiple(alert_log: HeartbeatAlertLog):
    sessions = [
        {"session_id": "a", "project_root": "/p/foo", "stage": "executing"},
        {"session_id": "b", "project_root": "/p/bar", "stage": "verifying"},
    ]
    with _patch_sessions(sessions):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="projects"))
    assert "2 active sessions" in report.voice_message.lower()
    assert "foo" in report.voice_message
    assert "bar" in report.voice_message


# ---------------------------------------------------------------------------
# focus="all"
# ---------------------------------------------------------------------------


def test_all_focus_quiet(alert_log: HeartbeatAlertLog):
    with _patch_sessions([]):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="all"))
    assert "quiet" in report.voice_message.lower()


def test_all_focus_alerts_only(alert_log: HeartbeatAlertLog):
    alert_log.record("hi")
    with _patch_sessions([]):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="all"))
    msg_lower = report.voice_message.lower()
    assert "no active sessions" in msg_lower
    assert "alert" in msg_lower


def test_all_focus_sessions_only(alert_log: HeartbeatAlertLog):
    sessions = [{"session_id": "a", "project_root": "/p/foo", "stage": "executing"}]
    with _patch_sessions(sessions):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="all"))
    msg_lower = report.voice_message.lower()
    assert "foo" in msg_lower
    assert "no alerts" in msg_lower


def test_all_focus_combined(alert_log: HeartbeatAlertLog):
    alert_log.record("disk filling")
    sessions = [{"session_id": "a", "project_root": "/p/foo", "stage": "executing"}]
    with _patch_sessions(sessions):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="all"))
    msg_lower = report.voice_message.lower()
    assert "foo" in msg_lower
    assert "disk filling" in msg_lower


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_unknown_focus_treated_as_all(alert_log: HeartbeatAlertLog):
    """Empty / null focus normalises to "all"."""
    with _patch_sessions([]):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus=""))
    assert report.focus == "all"
    assert "quiet" in report.voice_message.lower() or "no active" in report.voice_message.lower()


def test_alert_text_truncation(alert_log: HeartbeatAlertLog):
    long = "x" * 500
    alert_log.record(long)
    r = SystemStatusReporter(alert_log)
    report = r.report(SystemStatusIntent(focus="alerts"))
    # Sanitiser caps to 160 chars + ellipsis.
    assert len(report.voice_message) < 250


def test_alert_log_read_failure_degrades(alert_log: HeartbeatAlertLog):
    """Disk read failures yield a graceful "no info" message."""
    with patch.object(
        alert_log, "get_alerts",
        side_effect=RuntimeError("fs error"),
    ):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="alerts"))
    # Reporter swallowed the exception; it sees zero alerts.
    assert report.alerts == ()
    assert "no pending" in report.voice_message.lower()


def test_session_listing_failure_degrades(alert_log: HeartbeatAlertLog):
    with patch(
        "ultron.openclaw_bridge.system_status.list_active_coding_sessions_impl",
        side_effect=RuntimeError("oops"),
    ):
        r = SystemStatusReporter(alert_log)
        report = r.report(SystemStatusIntent(focus="projects"))
    assert report.active_sessions == ()
    assert "nothing active" in report.voice_message.lower()
