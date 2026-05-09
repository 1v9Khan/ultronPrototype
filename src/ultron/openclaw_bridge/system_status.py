"""System-status voice handler (Phase 13 finish).

Resolves SYSTEM_STATUS routing intents — voice queries like "what
alerts did you flag?" or "what is Ultron working on?" — by reading
the heartbeat alert log and the active coding session listing
directly from disk.

This handler is **Ultron-side only** — it does NOT call OpenClaw.
The same data the stdio MCP tools (in
:mod:`ultron.openclaw_bridge.mcp_tools`) expose to OpenClaw agents
is exposed locally here for the voice path. Both paths read the
same on-disk artifacts so a status report from voice and a Telegram
heartbeat alert tell the user the same story.

Output is a short voice narration in Ultron's character: brief,
unhurried, never apologetic. The narrator deliberately stays at
3–4 sentences for "all" queries and ≤2 for focused queries so the
TTS pipeline doesn't run on for several seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ultron.openclaw_bridge.heartbeat_alerts import (
    HeartbeatAlert,
    HeartbeatAlertLog,
)
from ultron.openclaw_bridge.mcp_tools import (
    list_active_coding_sessions_impl,
)
from ultron.openclaw_routing.intents import SystemStatusIntent
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.system_status")


@dataclass(frozen=True)
class SystemStatusReport:
    """Structured report; the reporter renders it to voice text."""

    focus: str                                      # "alerts" | "projects" | "all"
    alerts: tuple                                   # tuple of HeartbeatAlert (most-recent-first)
    active_sessions: tuple                          # tuple of dict
    voice_message: str
    spoken_focus: str = ""                          # what the user actually heard a heading for


class SystemStatusReporter:
    """Reads disk artifacts to produce a voice-friendly status report.

    Args:
        alert_log: configured :class:`HeartbeatAlertLog`. Required —
            constructing one ad-hoc would diverge from the bridge's
            shared instance.
        max_alerts: cap how many alerts to mention (default 3 — voice
            narration shouldn't enumerate more than a handful).
        max_sessions: same for sessions.
        recent_alert_window_days: only consider alerts from the last
            N days (default 7).
    """

    def __init__(
        self,
        alert_log: HeartbeatAlertLog,
        *,
        max_alerts: int = 3,
        max_sessions: int = 3,
        recent_alert_window_days: int = 7,
    ) -> None:
        self._alert_log = alert_log
        self._max_alerts = max(1, max_alerts)
        self._max_sessions = max(1, max_sessions)
        self._window_seconds = recent_alert_window_days * 86400

    def report(self, intent: SystemStatusIntent) -> SystemStatusReport:
        """Build a :class:`SystemStatusReport` from the configured
        focus. Never raises — disk read failures degrade to "no
        information available" voice messages."""
        focus = (intent.focus or "all").lower()
        alerts: List[HeartbeatAlert] = []
        sessions: List[dict] = []

        if focus in ("alerts", "all"):
            alerts = self._read_alerts()
        if focus in ("projects", "all"):
            sessions = self._read_active_sessions()

        voice = self._render_voice_message(focus, alerts, sessions)
        return SystemStatusReport(
            focus=focus,
            alerts=tuple(alerts),
            active_sessions=tuple(sessions),
            voice_message=voice,
            spoken_focus=focus,
        )

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _read_alerts(self) -> List[HeartbeatAlert]:
        try:
            cutoff = time.time() - self._window_seconds
            return self._alert_log.get_alerts(
                since=cutoff,
                only_unacknowledged=True,
                limit=self._max_alerts,
            )
        except Exception as exc:                            # noqa: BLE001
            logger.warning("alert log read failed: %s", exc)
            return []

    def _read_active_sessions(self) -> List[dict]:
        try:
            payload = list_active_coding_sessions_impl(
                max_age_hours=24,
            )
            sessions = payload.get("sessions") or []
            return list(sessions)[: self._max_sessions]
        except Exception as exc:                            # noqa: BLE001
            logger.warning("active-session listing failed: %s", exc)
            return []

    def _render_voice_message(
        self,
        focus: str,
        alerts: List[HeartbeatAlert],
        sessions: List[dict],
    ) -> str:
        """Build a brief in-character voice message."""
        if focus == "alerts":
            return self._render_alerts_only(alerts)
        if focus == "projects":
            return self._render_projects_only(sessions)
        # focus == "all"
        return self._render_combined(alerts, sessions)

    def _render_alerts_only(self, alerts: List[HeartbeatAlert]) -> str:
        if not alerts:
            return "No pending alerts."
        if len(alerts) == 1:
            a = alerts[0]
            return f"One alert. {self._sanitize(a.text)}"
        bits = ", ".join(self._sanitize(a.text) for a in alerts)
        return f"{len(alerts)} pending alerts. {bits}."

    def _render_projects_only(self, sessions: List[dict]) -> str:
        if not sessions:
            return "Nothing active."
        if len(sessions) == 1:
            s = sessions[0]
            label = self._session_label(s)
            return f"One active session: {label}."
        labels = ", ".join(self._session_label(s) for s in sessions)
        return f"{len(sessions)} active sessions: {labels}."

    def _render_combined(
        self,
        alerts: List[HeartbeatAlert],
        sessions: List[dict],
    ) -> str:
        if not alerts and not sessions:
            return "Quiet. No alerts, nothing active."
        parts: List[str] = []
        if sessions:
            if len(sessions) == 1:
                parts.append(
                    f"One active session: {self._session_label(sessions[0])}."
                )
            else:
                labels = ", ".join(
                    self._session_label(s) for s in sessions
                )
                parts.append(
                    f"{len(sessions)} active sessions: {labels}."
                )
        else:
            parts.append("No active sessions.")
        if alerts:
            if len(alerts) == 1:
                parts.append(f"One alert: {self._sanitize(alerts[0].text)}")
            else:
                first = self._sanitize(alerts[0].text)
                parts.append(
                    f"{len(alerts)} alerts pending. Most recent: {first}"
                )
        else:
            parts.append("No alerts.")
        return " ".join(parts)

    @staticmethod
    def _session_label(session: dict) -> str:
        """Produce a short human-readable label for one session
        ("project root last segment, stage status")."""
        root = session.get("project_root") or ""
        if root:
            label_root = Path(root).name or root
        else:
            label_root = session.get("session_id", "unknown")[:8]
        stage = session.get("stage")
        if stage:
            return f"{label_root} ({stage})"
        return label_root

    @staticmethod
    def _sanitize(text: str) -> str:
        """Trim trailing punctuation and clamp to a sentence-friendly
        length so concatenation doesn't produce double-periods or
        runaway sentences."""
        if not text:
            return ""
        cleaned = text.strip().rstrip(".")
        if len(cleaned) > 160:
            cleaned = cleaned[:157].rstrip() + "..."
        return cleaned


__all__ = [
    "SystemStatusReport",
    "SystemStatusReporter",
]
