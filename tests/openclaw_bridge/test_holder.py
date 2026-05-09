"""Tests for ``ultron.openclaw_bridge.holder.OpenClawBridge``.

The holder is the orchestrator's entry point to the bridge. Tests
cover construction (with and without a discoverable CLI), startup
behaviour when the Gateway is reachable vs. unreachable, the retry
thread, and shutdown idempotency.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ultron.openclaw_bridge.holder import OpenClawBridge
from ultron.openclaw_bridge.mcp_registration import RegistrationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    enabled: bool = True,
    cli_path: str | None = None,
    mcp_command: str | None = None,
    workspace: str | None = None,
    voice_handoff: bool = False,
) -> SimpleNamespace:
    bridge = SimpleNamespace(
        cli_path=cli_path,
        cli_timeout_seconds=30.0,
        mcp_server_name="ultron-mcp",
        mcp_server_command=mcp_command,
        mcp_server_args=[],
        retry_registration_interval_seconds=0.05,
        workspace_dir=workspace,
        workspace_lock_timeout_seconds=2.0,
        inbound_voice_handoff_enabled=voice_handoff,
        inbound_voice_handoff_prefix="[voice]",
        tool_invocation_timeout_seconds=30.0,
        message_send_timeout_seconds=10.0,
    )
    return SimpleNamespace(
        enabled=enabled,
        gateway_url=None,
        auth_token_env="OPENCLAW_AUTH_TOKEN",
        health_check_timeout_seconds=30.0,
        health_check_interval_seconds=60.0,
        fail_open=True,
        required_agent_id="ultron-main",
        bridge=bridge,
    )


@pytest.fixture
def fake_cli(tmp_path: Path) -> Path:
    p = tmp_path / "openclaw_fake.cmd"
    p.write_text("# fake")
    return p


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_returns_none_when_disabled(tmp_path: Path) -> None:
    cfg = _make_cfg(enabled=False, workspace=str(tmp_path))
    assert OpenClawBridge.from_config(cfg) is None


def test_from_config_builds_components(fake_cli: Path, tmp_path: Path) -> None:
    cfg = _make_cfg(
        cli_path=str(fake_cli),
        workspace=str(tmp_path / "ws"),
        mcp_command=None,
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    assert bridge.client is not None
    assert bridge.workspace.workspace_dir == tmp_path / "ws"
    assert bridge.events.enabled is False                       # default
    assert bridge.registrar is not None                         # client present, but no command set
    assert bridge.registrar.is_configured is False


def test_from_config_tolerates_missing_cli(tmp_path: Path) -> None:
    cfg = _make_cfg(
        cli_path=str(tmp_path / "does_not_exist"),
        workspace=str(tmp_path / "ws"),
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    # CLI not found → client and registrar are None.
    assert bridge.client is None
    assert bridge.registrar is None
    # Workspace and events still constructed.
    assert bridge.workspace is not None
    assert bridge.events is not None


def test_from_config_enables_voice_handoff_when_flagged(
    fake_cli: Path, tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        cli_path=str(fake_cli),
        workspace=str(tmp_path / "ws"),
        voice_handoff=True,
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge.events.enabled is True


# ---------------------------------------------------------------------------
# start() — Gateway reachable
# ---------------------------------------------------------------------------


def test_start_calls_register_when_reachable(
    fake_cli: Path, tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        cli_path=str(fake_cli),
        workspace=str(tmp_path / "ws"),
        mcp_command="some-stdio-cmd",
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None

    # Force lifecycle.is_reachable -> True; force register -> success.
    bridge.lifecycle.is_reachable = MagicMock(return_value=True)             # type: ignore[method-assign]
    register_mock = AsyncMock(
        return_value=RegistrationResult(registered=True, name="ultron-mcp"),
    )
    bridge.registrar.register = register_mock                                # type: ignore[assignment]

    bridge.start()
    assert register_mock.await_count == 1
    # No retry thread should have been launched.
    assert bridge._retry_thread is None
    bridge.shutdown()


def test_start_schedules_retry_when_unreachable(
    fake_cli: Path, tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        cli_path=str(fake_cli),
        workspace=str(tmp_path / "ws"),
        mcp_command="cmd",
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    bridge.lifecycle.is_reachable = MagicMock(return_value=False)            # type: ignore[method-assign]

    # Make the registrar's register() raise so the retry loop keeps running
    # — but start it then immediately stop to verify the thread launches
    # and exits cleanly on shutdown.
    bridge.registrar.register = AsyncMock(                                   # type: ignore[assignment]
        return_value=RegistrationResult(
            registered=False, name="ultron-mcp", error="down",
        ),
    )

    bridge.start()
    assert bridge._retry_thread is not None
    assert bridge._retry_thread.is_alive()

    # Stop should kill the retry thread within the join timeout.
    bridge.shutdown()
    assert not bridge._retry_thread.is_alive()


def test_start_skips_registration_when_no_mcp_command(
    fake_cli: Path, tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        cli_path=str(fake_cli),
        workspace=str(tmp_path / "ws"),
        mcp_command=None,
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    bridge.lifecycle.is_reachable = MagicMock(return_value=True)             # type: ignore[method-assign]

    register_mock = AsyncMock()
    if bridge.registrar is not None:
        bridge.registrar.register = register_mock                            # type: ignore[assignment]
    bridge.start()
    register_mock.assert_not_called()
    bridge.shutdown()


def test_start_idempotent(fake_cli: Path, tmp_path: Path) -> None:
    cfg = _make_cfg(
        cli_path=str(fake_cli), workspace=str(tmp_path / "ws"),
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    bridge.lifecycle.is_reachable = MagicMock(return_value=True)             # type: ignore[method-assign]
    bridge.start()
    bridge.start()                                                            # second call is no-op
    bridge.shutdown()


def test_shutdown_idempotent(fake_cli: Path, tmp_path: Path) -> None:
    cfg = _make_cfg(
        cli_path=str(fake_cli), workspace=str(tmp_path / "ws"),
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    bridge.shutdown()
    bridge.shutdown()                                                        # second call is no-op


# ---------------------------------------------------------------------------
# Phase 4 — notifications + fire_and_forget
# ---------------------------------------------------------------------------


def test_bridge_includes_notifications_dispatcher(
    fake_cli: Path, tmp_path: Path,
) -> None:
    """from_config wires a NotificationDispatcher onto the bridge."""
    cfg = _make_cfg(cli_path=str(fake_cli), workspace=str(tmp_path / "ws"))
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    assert bridge.notifications is not None
    # Default notifications_cfg has telegram.enabled=False, so .telegram_enabled
    # depends on both the client being present and the master flag.
    assert bridge.notifications.telegram_enabled is False


def test_bridge_notifications_uses_provided_config(
    fake_cli: Path, tmp_path: Path,
) -> None:
    from ultron.config import (
        NotificationsConfig,
        TelegramNotificationsConfig,
        TelegramNotifyOnConfig,
    )
    cfg = _make_cfg(cli_path=str(fake_cli), workspace=str(tmp_path / "ws"))
    notif_cfg = NotificationsConfig(
        telegram=TelegramNotificationsConfig(
            enabled=True,
            user_id_env="UNSET",
            fallback_user_id="42",
            notify_on=TelegramNotifyOnConfig(),
        ),
    )
    bridge = OpenClawBridge.from_config(cfg, notifications_cfg=notif_cfg)
    assert bridge is not None
    assert bridge.notifications.telegram_enabled is True


def test_fire_and_forget_runs_coroutine(
    fake_cli: Path, tmp_path: Path,
) -> None:
    """The helper schedules the coroutine on a daemon thread; we
    verify it runs by waiting on a threading.Event the coroutine sets."""
    import threading

    cfg = _make_cfg(cli_path=str(fake_cli), workspace=str(tmp_path / "ws"))
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    completed = threading.Event()

    async def _coro() -> None:
        completed.set()

    bridge.fire_and_forget(_coro)
    assert completed.wait(timeout=2.0), "fire_and_forget did not run the coroutine"
    bridge.shutdown()


def test_fire_and_forget_swallows_exceptions(
    fake_cli: Path, tmp_path: Path,
) -> None:
    """Coroutine raising should NOT propagate — the helper logs and
    returns. We verify by submitting many failing coroutines and
    confirming the bridge stays usable."""
    import threading

    cfg = _make_cfg(cli_path=str(fake_cli), workspace=str(tmp_path / "ws"))
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    completed = threading.Event()

    async def _bad() -> None:
        raise RuntimeError("boom")

    async def _good() -> None:
        completed.set()

    for _ in range(3):
        bridge.fire_and_forget(_bad)
    bridge.fire_and_forget(_good)
    assert completed.wait(timeout=2.0)
    bridge.shutdown()


# ---------------------------------------------------------------------------
# Phase 5 — heartbeat alert log + record_heartbeat_alert
# ---------------------------------------------------------------------------


def test_bridge_includes_heartbeat_alert_log(
    fake_cli: Path, tmp_path: Path,
) -> None:
    from ultron.config import HeartbeatConfig

    cfg = _make_cfg(cli_path=str(fake_cli), workspace=str(tmp_path / "ws"))
    hb_cfg = HeartbeatConfig(alert_log_path=str(tmp_path / "h.jsonl"))
    bridge = OpenClawBridge.from_config(cfg, heartbeat_cfg=hb_cfg)
    assert bridge is not None
    assert bridge.heartbeat_alerts is not None
    assert bridge.heartbeat_alerts.get_alerts() == []


def test_record_heartbeat_alert_writes_to_log(
    fake_cli: Path, tmp_path: Path,
) -> None:
    from ultron.config import HeartbeatConfig

    cfg = _make_cfg(cli_path=str(fake_cli), workspace=str(tmp_path / "ws"))
    hb_cfg = HeartbeatConfig(
        alert_log_path=str(tmp_path / "h.jsonl"),
        auto_notify_telegram=False,                           # avoid spawning a thread
    )
    bridge = OpenClawBridge.from_config(cfg, heartbeat_cfg=hb_cfg)
    assert bridge is not None
    alert = bridge.record_heartbeat_alert(
        "disk filling up", source="disk", severity="warn",
    )
    assert alert.text == "disk filling up"
    stored = bridge.heartbeat_alerts.get_alerts()
    assert len(stored) == 1
    assert stored[0].alert_id == alert.alert_id


def test_record_heartbeat_alert_fires_notification_when_enabled(
    fake_cli: Path, tmp_path: Path,
) -> None:
    """auto_notify_telegram=True + Telegram master enabled → the
    bridge schedules a Telegram dispatch via fire_and_forget. We
    verify the dispatcher's notify method was invoked."""
    import threading
    from ultron.config import (
        HeartbeatConfig,
        NotificationsConfig,
        TelegramNotificationsConfig,
        TelegramNotifyOnConfig,
    )

    cfg = _make_cfg(cli_path=str(fake_cli), workspace=str(tmp_path / "ws"))
    notif_cfg = NotificationsConfig(
        telegram=TelegramNotificationsConfig(
            enabled=True,
            user_id_env="UNSET_HEARTBEAT_TEST",
            fallback_user_id="42",
            notify_on=TelegramNotifyOnConfig(heartbeat_alerts=True),
        ),
    )
    hb_cfg = HeartbeatConfig(
        alert_log_path=str(tmp_path / "h.jsonl"),
        auto_notify_telegram=True,
    )

    bridge = OpenClawBridge.from_config(
        cfg, notifications_cfg=notif_cfg, heartbeat_cfg=hb_cfg,
    )
    assert bridge is not None

    # Replace the dispatcher's underlying notify call with one we can wait on.
    notified = threading.Event()
    captured: list[str] = []

    async def fake_notify(text: str):
        captured.append(text)
        notified.set()
        from ultron.openclaw_bridge.notifications import NotificationResult
        return NotificationResult(sent=True, channel="telegram", target="42")

    bridge.notifications.notify_heartbeat_alert = fake_notify  # type: ignore[assignment]

    bridge.record_heartbeat_alert("test alert text")
    assert notified.wait(timeout=2.0)
    assert captured == ["test alert text"]


def test_record_heartbeat_alert_no_notify_when_flag_off(
    fake_cli: Path, tmp_path: Path,
) -> None:
    from ultron.config import HeartbeatConfig

    cfg = _make_cfg(cli_path=str(fake_cli), workspace=str(tmp_path / "ws"))
    hb_cfg = HeartbeatConfig(
        alert_log_path=str(tmp_path / "h.jsonl"),
        auto_notify_telegram=False,                          # disabled
    )
    bridge = OpenClawBridge.from_config(cfg, heartbeat_cfg=hb_cfg)
    assert bridge is not None
    notify_calls: list[str] = []

    async def fake_notify(text: str):
        notify_calls.append(text)
        from ultron.openclaw_bridge.notifications import NotificationResult
        return NotificationResult(sent=True, channel="telegram", target="x")

    bridge.notifications.notify_heartbeat_alert = fake_notify  # type: ignore[assignment]

    bridge.record_heartbeat_alert("don't push me")
    # Wait a beat to confirm no notification spawns.
    import time as _t
    _t.sleep(0.2)
    assert notify_calls == []
    # Alert still recorded to the log though.
    assert len(bridge.heartbeat_alerts.get_alerts()) == 1


# ---------------------------------------------------------------------------
# Phase 13 — mcp_server_command auto-resolve
# ---------------------------------------------------------------------------


def test_resolve_mcp_command_explicit_passes_through() -> None:
    """An explicit string + args list pass through verbatim."""
    cmd, args = OpenClawBridge._resolve_mcp_command(
        "/usr/bin/python", ["/path/to/server.py", "--stdio"],
    )
    assert cmd == "/usr/bin/python"
    assert args == ["/path/to/server.py", "--stdio"]


def test_resolve_mcp_command_none_disables() -> None:
    """``None`` is the explicit-disable sentinel."""
    cmd, args = OpenClawBridge._resolve_mcp_command(None, [])
    assert cmd is None
    assert args == []


def test_resolve_mcp_command_auto_finds_canonical() -> None:
    """``"auto"`` resolves to the canonical entry script + interpreter."""
    cmd, args = OpenClawBridge._resolve_mcp_command("auto", [])
    assert cmd is not None
    assert cmd.endswith(("python.exe", "python"))
    # First arg is the entry script path.
    assert args
    entry = args[0]
    assert entry.endswith("run_ultron_mcp_for_openclaw.py")
    assert Path(entry).exists()
    assert "--stdio" in args


def test_resolve_mcp_command_auto_falls_back_when_script_missing(monkeypatch) -> None:
    """If the entry script doesn't exist (unusual layout), auto
    returns (None, []) so the registrar disables itself."""
    fake_root = Path("/nonexistent/place")
    import ultron.openclaw_bridge.holder as holder_mod
    monkeypatch.setattr(
        "ultron.config.PROJECT_ROOT", fake_root,
    )
    cmd, args = holder_mod.OpenClawBridge._resolve_mcp_command("auto", [])
    assert cmd is None
    assert args == []


def test_bridge_from_config_with_auto_command(fake_cli: Path, tmp_path: Path) -> None:
    """End-to-end: from_config with mcp_server_command='auto' produces
    a registrar with is_configured=True."""
    cfg = _make_cfg(
        cli_path=str(fake_cli),
        workspace=str(tmp_path / "ws"),
        mcp_command="auto",                                  # NEW: auto path
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    assert bridge.registrar is not None
    assert bridge.registrar.is_configured is True


def test_bridge_from_config_with_none_command_disables(fake_cli: Path, tmp_path: Path) -> None:
    """from_config with mcp_server_command=None still disables."""
    cfg = _make_cfg(
        cli_path=str(fake_cli),
        workspace=str(tmp_path / "ws"),
        mcp_command=None,                                    # explicit disable
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    assert bridge.registrar is not None
    assert bridge.registrar.is_configured is False
