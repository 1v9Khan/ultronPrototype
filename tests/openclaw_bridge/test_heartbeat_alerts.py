"""Tests for ``ultron.openclaw_bridge.heartbeat_alerts``."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from ultron.openclaw_bridge.heartbeat_alerts import (
    HeartbeatAlert,
    HeartbeatAlertLog,
)


@pytest.fixture
def alert_log(tmp_path: Path) -> HeartbeatAlertLog:
    return HeartbeatAlertLog(
        tmp_path / "logs" / "heartbeat.jsonl",
        retention_days=30,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_invalid_retention(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        HeartbeatAlertLog(tmp_path / "x", retention_days=0)


def test_creates_log_lazily(alert_log: HeartbeatAlertLog) -> None:
    assert not alert_log.path.exists()
    assert alert_log.get_alerts() == []
    # No file created on read.
    assert not alert_log.path.exists()


# ---------------------------------------------------------------------------
# record + read
# ---------------------------------------------------------------------------


def test_record_creates_file(alert_log: HeartbeatAlertLog) -> None:
    alert = alert_log.record("disk filling up", source="disk", severity="warn")
    assert alert.alert_id
    assert alert.text == "disk filling up"
    assert alert.severity == "warn"
    assert alert.acknowledged is False
    assert alert_log.path.exists()


def test_record_appends_subsequent(alert_log: HeartbeatAlertLog) -> None:
    a = alert_log.record("first", source="s")
    b = alert_log.record("second", source="s")
    assert a.alert_id != b.alert_id
    alerts = alert_log.get_alerts()
    assert len(alerts) == 2
    # Most-recent first.
    assert alerts[0].text == "second"
    assert alerts[1].text == "first"


def test_record_rejects_empty_text(alert_log: HeartbeatAlertLog) -> None:
    with pytest.raises(ValueError):
        alert_log.record("   ")


def test_record_rejects_invalid_severity(alert_log: HeartbeatAlertLog) -> None:
    with pytest.raises(ValueError):
        alert_log.record("hi", severity="critical")


def test_record_preserves_metadata(alert_log: HeartbeatAlertLog) -> None:
    alert = alert_log.record(
        "stuck", metadata={"session_id": "abc-123", "stage": "verify"},
    )
    fresh = alert_log.get_alerts()[0]
    assert fresh.metadata == {"session_id": "abc-123", "stage": "verify"}


# ---------------------------------------------------------------------------
# get_alerts filtering
# ---------------------------------------------------------------------------


def test_get_alerts_since_filter(alert_log: HeartbeatAlertLog) -> None:
    a = alert_log.record("old", source="s")
    # Force an obviously-older timestamp by editing the file directly.
    line = alert_log.path.read_text(encoding="utf-8")
    data = json.loads(line)
    data["timestamp"] = time.time() - 86400
    alert_log.path.write_text(
        json.dumps(data) + "\n", encoding="utf-8",
    )
    b = alert_log.record("new", source="s")
    recent = alert_log.get_alerts(since=time.time() - 60)
    assert {x.alert_id for x in recent} == {b.alert_id}


def test_get_alerts_only_unacknowledged(alert_log: HeartbeatAlertLog) -> None:
    a = alert_log.record("a")
    b = alert_log.record("b")
    alert_log.acknowledge(a.alert_id)
    open_alerts = alert_log.get_alerts(only_unacknowledged=True)
    assert {x.alert_id for x in open_alerts} == {b.alert_id}


def test_get_alerts_limit(alert_log: HeartbeatAlertLog) -> None:
    for i in range(5):
        alert_log.record(f"alert {i}")
    top2 = alert_log.get_alerts(limit=2)
    assert len(top2) == 2
    assert top2[0].text == "alert 4"
    assert top2[1].text == "alert 3"


# ---------------------------------------------------------------------------
# acknowledge
# ---------------------------------------------------------------------------


def test_acknowledge_updates_alert(alert_log: HeartbeatAlertLog) -> None:
    a = alert_log.record("hi")
    assert alert_log.acknowledge(a.alert_id) is True
    fresh = alert_log.get_alerts()[0]
    assert fresh.acknowledged is True
    assert fresh.acknowledged_at is not None


def test_acknowledge_returns_false_for_unknown_id(alert_log: HeartbeatAlertLog) -> None:
    alert_log.record("hi")
    assert alert_log.acknowledge("no-such-id") is False


def test_acknowledge_returns_false_when_already_acknowledged(
    alert_log: HeartbeatAlertLog,
) -> None:
    a = alert_log.record("hi")
    alert_log.acknowledge(a.alert_id)
    assert alert_log.acknowledge(a.alert_id) is False


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def test_prune_removes_old_entries(tmp_path: Path) -> None:
    log = HeartbeatAlertLog(tmp_path / "h.jsonl", retention_days=1)
    a = log.record("recent")
    # Manually edit the file to make `a` old.
    raw = log.path.read_text(encoding="utf-8")
    data = json.loads(raw)
    data["timestamp"] = time.time() - 2 * 86400
    log.path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    b = log.record("new")
    removed = log.prune()
    assert removed == 1
    remaining = log.get_alerts()
    assert {x.alert_id for x in remaining} == {b.alert_id}


def test_prune_no_op_on_fresh_log(alert_log: HeartbeatAlertLog) -> None:
    alert_log.record("hi")
    assert alert_log.prune() == 0


# ---------------------------------------------------------------------------
# Tolerates malformed JSONL
# ---------------------------------------------------------------------------


def test_get_alerts_skips_malformed_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "h.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        '{"alert_id":"abc","text":"good","source":"s","severity":"info","timestamp":100.0}\n'
        'this is not json\n'
        '{"alert_id":"def","text":"good2","source":"s","severity":"info","timestamp":200.0}\n',
        encoding="utf-8",
    )
    log = HeartbeatAlertLog(log_path, retention_days=30)
    alerts = log.get_alerts()
    assert len(alerts) == 2                                 # malformed line skipped


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_record(alert_log: HeartbeatAlertLog) -> None:
    """20 threads each record 10 alerts; lock serialises all 200."""

    def worker() -> None:
        for _ in range(10):
            alert_log.record("hello")

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    alerts = alert_log.get_alerts()
    assert len(alerts) == 200
