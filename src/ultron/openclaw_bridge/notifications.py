"""Proactive notifications to remote channels (Phase 4).

When a coding task completes, a heartbeat alert fires, or a standing
order produces output, Ultron pings the user's phone via Telegram so
they hear about it even when away from the desk. This dispatcher is
the single seam for those proactive pings.

Architecture: NotificationDispatcher is constructed once by the
orchestrator and held alongside :class:`OpenClawBridge`. The
orchestrator's existing voice-path callers (CodingTaskRunner
completion, etc.) hand off to one of the ``notify_*`` methods. Each
method:

1. Checks the master ``notifications.<channel>.enabled`` flag.
2. Checks the per-event ``notify_on.<event>`` flag.
3. Resolves the target user id from env var (or fallback config).
4. Calls ``OpenClawClient.send_message(...)``.
5. Logs the outcome and returns a boolean.

Fail-open at every step: missing config, missing env var, missing
client, or transport failure all return ``False`` without raising.
The voice pipeline never blocks on a notification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ultron.config import NotificationsConfig
from ultron.openclaw_bridge.client import OpenClawClient, SendMessageResult
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.notifications")


@dataclass(frozen=True)
class NotificationResult:
    """Outcome of one ``notify_*`` call.

    Attributes:
        sent: True iff the message reached the channel (per the
            client's :class:`SendMessageResult`).
        channel: which channel was used (e.g. ``"telegram"``).
        skipped_reason: when ``sent=False``, why we didn't send
            (master disabled, per-event disabled, no target id, no
            client, transport failure).
    """

    sent: bool
    channel: str = ""
    target: str = ""
    skipped_reason: Optional[str] = None
    raw: Optional[SendMessageResult] = None


# Sentinel: when none of the per-event flags apply we still skip.
_DEFAULT_TIMEOUT_S = 10.0


class NotificationDispatcher:
    """Single seam for outbound proactive notifications.

    Args:
        client: shared :class:`OpenClawClient`. ``None`` is permitted —
            the dispatcher logs and returns ``sent=False`` on every
            call. Construction never raises.
        config: :class:`NotificationsConfig` from the loaded config.
        timeout_s: per-call timeout for ``client.send_message``.
    """

    def __init__(
        self,
        client: Optional[OpenClawClient],
        config: NotificationsConfig,
        *,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._client = client
        self._config = config
        self._timeout_s = timeout_s

    @property
    def telegram_enabled(self) -> bool:
        return bool(self._client) and self._config.telegram.enabled

    # -------------------------------------------------------------------
    # Public surface — one method per event class
    # -------------------------------------------------------------------

    async def notify_coding_task_completion(
        self, summary: str,
    ) -> NotificationResult:
        """Fire-and-forget notification: a coding task finished.

        Caller passes the user-facing completion narration so the
        Telegram message reads identically to what the user would
        have heard had they been at the desk.
        """
        return await self._dispatch_telegram(
            event="coding_task_completion",
            text=summary,
        )

    async def notify_coding_task_clarification(
        self, question: str,
    ) -> NotificationResult:
        """Coding session is paused waiting for the user."""
        return await self._dispatch_telegram(
            event="coding_task_clarification_needed",
            text=question,
        )

    async def notify_heartbeat_alert(self, alert_text: str) -> NotificationResult:
        """A heartbeat tick raised something the user should see."""
        return await self._dispatch_telegram(
            event="heartbeat_alerts",
            text=alert_text,
        )

    async def notify_standing_order_output(
        self, summary: str,
    ) -> NotificationResult:
        """A standing-order program produced output (weekly review,
        coding-watcher status, etc.)."""
        return await self._dispatch_telegram(
            event="standing_order_outputs",
            text=summary,
        )

    async def notify_search_results_async(
        self, summary: str,
    ) -> NotificationResult:
        """A web search took too long for the voice path; deliver the
        result asynchronously instead. Opt-in; off by default."""
        return await self._dispatch_telegram(
            event="search_results_async",
            text=summary,
        )

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    async def _dispatch_telegram(
        self,
        *,
        event: str,
        text: str,
    ) -> NotificationResult:
        """Resolve config + send. Fail-open at every step."""
        cfg = self._config.telegram
        if self._client is None:
            return NotificationResult(
                sent=False, channel="telegram",
                skipped_reason="bridge client unavailable",
            )
        if not cfg.enabled:
            return NotificationResult(
                sent=False, channel="telegram",
                skipped_reason="telegram disabled in config",
            )
        if not getattr(cfg.notify_on, event, False):
            return NotificationResult(
                sent=False, channel="telegram",
                skipped_reason=f"notify_on.{event} is false",
            )
        target = self._resolve_user_id()
        if not target:
            logger.warning(
                "telegram notification skipped: %s/%s missing in env "
                "and no fallback_user_id set",
                cfg.user_id_env, "fallback_user_id",
            )
            return NotificationResult(
                sent=False, channel="telegram",
                skipped_reason="no target user id resolved",
            )
        if not text or not text.strip():
            return NotificationResult(
                sent=False, channel="telegram", target=target,
                skipped_reason="empty text",
            )
        try:
            result = await self._client.send_message(
                "telegram", target, text,
                timeout_s=self._timeout_s,
            )
        except Exception as exc:                            # noqa: BLE001
            logger.warning(
                "telegram notification (%s) raised: %s",
                event, exc,
            )
            return NotificationResult(
                sent=False, channel="telegram", target=target,
                skipped_reason=f"transport error: {exc}",
            )
        if result.delivered:
            logger.info(
                "telegram notification (%s) delivered to %s "
                "(message_id=%s)",
                event, target, result.message_id or "-",
            )
            return NotificationResult(
                sent=True, channel="telegram", target=target,
                raw=result,
            )
        logger.warning(
            "telegram notification (%s) failed: %s",
            event, result.error or "(no detail)",
        )
        return NotificationResult(
            sent=False, channel="telegram", target=target,
            skipped_reason=result.error or "client returned delivered=False",
            raw=result,
        )

    def _resolve_user_id(self) -> str:
        cfg = self._config.telegram
        env_value = os.environ.get(cfg.user_id_env, "").strip()
        if env_value:
            return env_value
        if cfg.fallback_user_id:
            return str(cfg.fallback_user_id).strip()
        return ""


__all__ = [
    "NotificationDispatcher",
    "NotificationResult",
]
