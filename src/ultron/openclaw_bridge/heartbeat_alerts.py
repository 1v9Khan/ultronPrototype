"""Heartbeat alert persistence (Phase 5).

The OpenClaw-side heartbeat agent (configured in
``~/.openclaw/openclaw.json`` under ``agents[].heartbeat``) raises
alerts when its checklist surfaces something the user should see.
This module records those alerts locally so:

1. A voice query like "what alerts did you flag?" can pull recent
   entries from the log.
2. The agent itself can read the log on the next tick (via the
   Ultron MCP tool ``ultron.get_heartbeat_alerts``) to avoid
   re-surfacing already-acknowledged items.
3. Auditing — every alert has a timestamp, source, severity, and
   acknowledgment state.

Storage is JSONL (one alert per line) under
``heartbeat.alert_log_path``. Append-only on record; updates
(acknowledgments) rewrite the file atomically. A read-time
threading lock serialises concurrent access from the orchestrator
thread, the MCP server thread, and any background heartbeat
delivery threads.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.heartbeat_alerts")


@dataclass
class HeartbeatAlert:
    """One alert raised by a heartbeat tick.

    Attributes:
        alert_id: stable UUID4 hex; used by ``acknowledge``.
        text: the alert message body, in Ultron's voice.
        source: free-form tag identifying which checklist item raised
            the alert (e.g. ``"coding-queue"``, ``"disk"``,
            ``"addressing-anomaly"``).
        severity: ``"info" | "warn" | "error"``. Controls Telegram
            delivery cadence and voice-query phrasing.
        timestamp: epoch seconds at record time.
        acknowledged_at: epoch seconds when the user acknowledged,
            or ``None`` when unacknowledged.
        metadata: optional structured detail; never logged to user.
    """

    alert_id: str
    text: str
    source: str = "heartbeat"
    severity: str = "info"
    timestamp: float = field(default_factory=lambda: time.time())
    acknowledged_at: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def acknowledged(self) -> bool:
        return self.acknowledged_at is not None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "HeartbeatAlert":
        data = json.loads(line)
        # Tolerate legacy entries missing newer fields.
        return cls(
            alert_id=data["alert_id"],
            text=data["text"],
            source=data.get("source", "heartbeat"),
            severity=data.get("severity", "info"),
            timestamp=float(data.get("timestamp", time.time())),
            acknowledged_at=(
                float(data["acknowledged_at"])
                if data.get("acknowledged_at") is not None
                else None
            ),
            metadata=data.get("metadata", {}) or {},
        )


_VALID_SEVERITIES = ("info", "warn", "error")


class HeartbeatAlertLog:
    """JSONL-backed alert log with thread-safe read/append/update.

    Args:
        path: file path. Created lazily on first record. Parent dirs
            auto-created.
        retention_days: alerts older than this are dropped on
            ``prune()``. Pruning is on-demand, not automatic.
    """

    def __init__(
        self,
        path: Path,
        *,
        retention_days: int = 30,
    ) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be >= 1")
        self._path = Path(path)
        self._retention_days = retention_days
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def record(
        self,
        text: str,
        *,
        source: str = "heartbeat",
        severity: str = "info",
        metadata: Optional[dict] = None,
    ) -> HeartbeatAlert:
        """Append a new alert. Returns the recorded alert.

        Validates ``severity`` and trims whitespace from ``text``.
        Atomic append via lock — concurrent writers serialise.
        """
        if severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {_VALID_SEVERITIES}, got {severity!r}"
            )
        text = (text or "").strip()
        if not text:
            raise ValueError("alert text must be non-empty")
        alert = HeartbeatAlert(
            alert_id=uuid.uuid4().hex,
            text=text,
            source=source,
            severity=severity,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._ensure_parent()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(alert.to_jsonl() + "\n")
        logger.info(
            "heartbeat alert recorded (id=%s, source=%s, severity=%s)",
            alert.alert_id, alert.source, alert.severity,
        )
        return alert

    def get_alerts(
        self,
        *,
        since: Optional[float] = None,
        only_unacknowledged: bool = False,
        limit: Optional[int] = None,
    ) -> List[HeartbeatAlert]:
        """Read alerts, optionally filtered by recency / ack state.

        Args:
            since: epoch seconds. Only alerts with
                ``timestamp >= since`` are returned. ``None`` =
                no time filter.
            only_unacknowledged: when True, drop alerts where
                ``acknowledged_at`` is set.
            limit: keep only the N most recent matching alerts
                after filtering. ``None`` = no limit.

        Result ordering is most-recent-first. The log is append-only
        (acknowledgments rewrite the file in original order), so file
        order = insertion order; reversing it yields newest-first
        without a timestamp tie-break headache.
        """
        alerts = self._read_all()
        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]
        if only_unacknowledged:
            alerts = [a for a in alerts if not a.acknowledged]
        alerts.reverse()                                  # newest-first
        if limit is not None and limit >= 0:
            alerts = alerts[:limit]
        return alerts

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged. Returns True iff a matching
        unacknowledged alert was found and updated; False otherwise
        (already acknowledged, or unknown id, or log empty).

        Atomic rewrite of the whole file under the lock — small files
        (<10k alerts after retention) make this acceptable; pruning
        keeps the size bounded.
        """
        with self._lock:
            alerts = self._read_all_unlocked()
            updated = False
            for alert in alerts:
                if alert.alert_id == alert_id and alert.acknowledged_at is None:
                    alert.acknowledged_at = time.time()
                    updated = True
                    break
            if not updated:
                return False
            self._write_all_unlocked(alerts)
        logger.info("heartbeat alert acknowledged (id=%s)", alert_id)
        return True

    def prune(self) -> int:
        """Drop alerts older than ``retention_days``. Returns the
        number of alerts removed."""
        cutoff = time.time() - (self._retention_days * 86400)
        with self._lock:
            alerts = self._read_all_unlocked()
            keep = [a for a in alerts if a.timestamp >= cutoff]
            removed = len(alerts) - len(keep)
            if removed > 0:
                self._write_all_unlocked(keep)
        if removed > 0:
            logger.info("pruned %d expired heartbeat alerts", removed)
        return removed

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> List[HeartbeatAlert]:
        with self._lock:
            return self._read_all_unlocked()

    def _read_all_unlocked(self) -> List[HeartbeatAlert]:
        if not self._path.exists():
            return []
        out: List[HeartbeatAlert] = []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(HeartbeatAlert.from_jsonl(line))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(
                            "skipping malformed alert at %s:%d (%s)",
                            self._path, line_no, e,
                        )
        except OSError as e:
            logger.warning("could not read alert log %s (%s)", self._path, e)
            return []
        return out

    def _write_all_unlocked(self, alerts: List[HeartbeatAlert]) -> None:
        """Atomic full-file rewrite via temp + replace."""
        self._ensure_parent()
        fd, tmp_path = tempfile.mkstemp(
            prefix=self._path.name + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for alert in alerts:
                    fh.write(alert.to_jsonl() + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
            raise


__all__ = [
    "HeartbeatAlert",
    "HeartbeatAlertLog",
]
