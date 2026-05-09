"""Tests for ``ultron.openclaw_bridge.notifications.NotificationDispatcher``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from ultron.config import (
    NotificationsConfig,
    TelegramNotificationsConfig,
    TelegramNotifyOnConfig,
)
from ultron.openclaw_bridge.client import SendMessageResult
from ultron.openclaw_bridge.notifications import (
    NotificationDispatcher,
    NotificationResult,
)


def _make_cfg(
    *,
    enabled: bool = True,
    fallback_user_id: str | None = "TEST_USER",
    coding_completion: bool = True,
    coding_clarification: bool = True,
    heartbeat: bool = True,
    standing: bool = True,
    search_async: bool = False,
) -> NotificationsConfig:
    return NotificationsConfig(
        telegram=TelegramNotificationsConfig(
            enabled=enabled,
            user_id_env="UNSET_ENV_FOR_TEST_NOTIF",
            fallback_user_id=fallback_user_id,
            notify_on=TelegramNotifyOnConfig(
                coding_task_completion=coding_completion,
                coding_task_clarification_needed=coding_clarification,
                heartbeat_alerts=heartbeat,
                standing_order_outputs=standing,
                search_results_async=search_async,
            ),
        ),
    )


def _make_send_ok(message_id: str = "msg-1") -> SendMessageResult:
    return SendMessageResult(
        delivered=True, channel="telegram", target="TEST_USER",
        message_id=message_id,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UNSET_ENV_FOR_TEST_NOTIF", raising=False)
    yield


@pytest.fixture
def fake_client() -> Any:
    client = AsyncMock()
    client.send_message = AsyncMock(return_value=_make_send_ok())
    return client


# ---------------------------------------------------------------------------
# Construction + property
# ---------------------------------------------------------------------------


def test_telegram_enabled_property_reflects_config(fake_client: Any) -> None:
    cfg = _make_cfg(enabled=True)
    d = NotificationDispatcher(fake_client, cfg)
    assert d.telegram_enabled is True

    cfg_off = _make_cfg(enabled=False)
    d_off = NotificationDispatcher(fake_client, cfg_off)
    assert d_off.telegram_enabled is False

    # No client → not enabled even if config says so.
    d_no_client = NotificationDispatcher(None, _make_cfg(enabled=True))
    assert d_no_client.telegram_enabled is False


# ---------------------------------------------------------------------------
# notify_coding_task_completion
# ---------------------------------------------------------------------------


async def test_completion_sends_when_enabled(fake_client: Any) -> None:
    d = NotificationDispatcher(fake_client, _make_cfg(enabled=True))
    result = await d.notify_coding_task_completion("done!")
    assert isinstance(result, NotificationResult)
    assert result.sent is True
    assert result.channel == "telegram"
    assert result.target == "TEST_USER"
    fake_client.send_message.assert_awaited_once()
    args, kwargs = fake_client.send_message.call_args
    assert args[0] == "telegram"                      # channel
    assert args[1] == "TEST_USER"                     # target
    assert args[2] == "done!"                         # text


async def test_completion_skipped_when_master_off(fake_client: Any) -> None:
    d = NotificationDispatcher(fake_client, _make_cfg(enabled=False))
    result = await d.notify_coding_task_completion("done!")
    assert result.sent is False
    assert "telegram disabled" in (result.skipped_reason or "")
    fake_client.send_message.assert_not_called()


async def test_completion_skipped_when_per_event_off(fake_client: Any) -> None:
    d = NotificationDispatcher(
        fake_client, _make_cfg(enabled=True, coding_completion=False),
    )
    result = await d.notify_coding_task_completion("done!")
    assert result.sent is False
    assert "coding_task_completion" in (result.skipped_reason or "")
    fake_client.send_message.assert_not_called()


async def test_completion_skipped_when_no_recipient(fake_client: Any) -> None:
    d = NotificationDispatcher(
        fake_client, _make_cfg(enabled=True, fallback_user_id=None),
    )
    result = await d.notify_coding_task_completion("done!")
    assert result.sent is False
    assert "no target user id" in (result.skipped_reason or "")


async def test_completion_skipped_when_text_empty(fake_client: Any) -> None:
    d = NotificationDispatcher(fake_client, _make_cfg(enabled=True))
    result = await d.notify_coding_task_completion("   ")
    assert result.sent is False
    assert "empty text" in (result.skipped_reason or "")
    fake_client.send_message.assert_not_called()


async def test_completion_skipped_when_no_client() -> None:
    d = NotificationDispatcher(None, _make_cfg(enabled=True))
    result = await d.notify_coding_task_completion("done!")
    assert result.sent is False
    assert "client unavailable" in (result.skipped_reason or "")


async def test_completion_handles_transport_error() -> None:
    client = AsyncMock()
    client.send_message = AsyncMock(side_effect=RuntimeError("boom"))
    d = NotificationDispatcher(client, _make_cfg(enabled=True))
    result = await d.notify_coding_task_completion("done!")
    assert result.sent is False
    assert "transport error" in (result.skipped_reason or "")


async def test_completion_handles_undelivered_response() -> None:
    client = AsyncMock()
    client.send_message = AsyncMock(return_value=SendMessageResult(
        delivered=False, channel="telegram", target="TEST_USER",
        error="rate limited",
    ))
    d = NotificationDispatcher(client, _make_cfg(enabled=True))
    result = await d.notify_coding_task_completion("done!")
    assert result.sent is False
    assert "rate limited" in (result.skipped_reason or "")


# ---------------------------------------------------------------------------
# Other event methods (smoke-only — they share the same dispatch path)
# ---------------------------------------------------------------------------


async def test_clarification_dispatch(fake_client: Any) -> None:
    d = NotificationDispatcher(fake_client, _make_cfg(enabled=True))
    result = await d.notify_coding_task_clarification("which framework?")
    assert result.sent is True


async def test_heartbeat_dispatch(fake_client: Any) -> None:
    d = NotificationDispatcher(fake_client, _make_cfg(enabled=True))
    result = await d.notify_heartbeat_alert("disk filling up")
    assert result.sent is True


async def test_standing_order_dispatch(fake_client: Any) -> None:
    d = NotificationDispatcher(fake_client, _make_cfg(enabled=True))
    result = await d.notify_standing_order_output("weekly review attached")
    assert result.sent is True


async def test_search_async_off_by_default(fake_client: Any) -> None:
    """search_results_async defaults to off; the call should skip."""
    d = NotificationDispatcher(fake_client, _make_cfg(enabled=True))
    result = await d.notify_search_results_async("results...")
    assert result.sent is False
    assert "search_results_async" in (result.skipped_reason or "")


async def test_search_async_when_explicitly_enabled(fake_client: Any) -> None:
    d = NotificationDispatcher(
        fake_client, _make_cfg(enabled=True, search_async=True),
    )
    result = await d.notify_search_results_async("results...")
    assert result.sent is True


# ---------------------------------------------------------------------------
# Recipient resolution
# ---------------------------------------------------------------------------


async def test_recipient_from_env_var_takes_precedence(
    fake_client: Any, monkeypatch,
) -> None:
    monkeypatch.setenv("UNSET_ENV_FOR_TEST_NOTIF", "ENV_USER")
    d = NotificationDispatcher(
        fake_client, _make_cfg(enabled=True, fallback_user_id="FALLBACK"),
    )
    result = await d.notify_coding_task_completion("done!")
    assert result.sent is True
    assert result.target == "ENV_USER"


async def test_recipient_falls_back_when_env_unset(fake_client: Any) -> None:
    d = NotificationDispatcher(
        fake_client, _make_cfg(enabled=True, fallback_user_id="FALLBACK"),
    )
    result = await d.notify_coding_task_completion("done!")
    assert result.sent is True
    assert result.target == "FALLBACK"
