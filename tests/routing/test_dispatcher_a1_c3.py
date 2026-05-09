"""V1-gap A1 / C3: dispatcher handlers for gaming mode + desktop / window.

These tests exercise the dispatcher handlers in isolation (no
:class:`AutomationTaskRunner` wrapper). We use stub managers / clients
so we can assert on the resulting :class:`DispatchResult` voice
messages without spinning up OpenClaw.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from ultron.openclaw_routing.dispatcher import OpenClawDispatcher
from ultron.openclaw_routing.gaming_mode import GamingModeStatus
from ultron.openclaw_routing.intents import (
    DesktopIntent,
    GamingModeIntent,
    WindowIntent,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubReport:
    status: GamingModeStatus
    action: str
    plugin_states: list = field(default_factory=list)
    docker_acted: bool = False
    note: str = ""

    @property
    def all_plugin_actions_succeeded(self) -> bool:
        return all(getattr(p, "success", True) for p in self.plugin_states)


@dataclass
class _StubPluginState:
    plugin_id: str
    success: bool = True
    error: Optional[str] = None


class _StubGamingModeManager:
    def __init__(self) -> None:
        self.calls = []
        self._status = GamingModeStatus.IDLE
        self.engage_report = _StubReport(
            status=GamingModeStatus.ENGAGED, action="engage",
            plugin_states=[
                _StubPluginState(plugin_id="desktop-control"),
                _StubPluginState(plugin_id="windows-control"),
            ],
        )
        self.disengage_report = _StubReport(
            status=GamingModeStatus.IDLE, action="disengage",
            plugin_states=[
                _StubPluginState(plugin_id="desktop-control"),
                _StubPluginState(plugin_id="windows-control"),
            ],
        )

    async def engage(self):
        self.calls.append("engage")
        self._status = GamingModeStatus.ENGAGED
        return self.engage_report

    async def disengage(self):
        self.calls.append("disengage")
        self._status = GamingModeStatus.IDLE
        return self.disengage_report

    def status(self) -> GamingModeStatus:
        return self._status


@dataclass
class _ToolResult:
    success: bool
    error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class _StubClient:
    def __init__(self, *, scripted: Optional[Dict[str, _ToolResult]] = None) -> None:
        self.calls = []
        self._scripted = scripted or {}

    async def invoke_tool(self, tool_name, params, *, agent_id=None, timeout_s=None):
        self.calls.append({
            "tool_name": tool_name, "params": params,
            "agent_id": agent_id, "timeout_s": timeout_s,
        })
        return self._scripted.get(tool_name, _ToolResult(success=True, payload={}))


def _make_dispatcher(*, gaming_mode_manager=None, bridge=None):
    """Build a dispatcher with the V1-gap config sections enabled."""
    from ultron.config import get_config

    cfg = get_config()
    # Flip dispatcher-side flags ON for the test.
    cfg.desktop.enabled = True
    cfg.window_control.enabled = True
    return OpenClawDispatcher(
        config=cfg,
        gaming_mode_manager=gaming_mode_manager,
        bridge=bridge,
    )


# ---------------------------------------------------------------------------
# A1: handle_gaming_mode
# ---------------------------------------------------------------------------


def test_handle_gaming_mode_engage_succeeds():
    mgr = _StubGamingModeManager()
    dispatcher = _make_dispatcher(gaming_mode_manager=mgr)
    intent = GamingModeIntent(action="engage", trigger_phrase="gaming mode")
    result = asyncio.run(dispatcher.handle_gaming_mode(intent))
    assert result.success is True
    assert "shutting down desktop control" in result.voice_message.lower()
    assert mgr.calls == ["engage"]


def test_handle_gaming_mode_disengage_succeeds():
    mgr = _StubGamingModeManager()
    asyncio.run(mgr.engage())
    mgr.calls.clear()
    dispatcher = _make_dispatcher(gaming_mode_manager=mgr)
    intent = GamingModeIntent(action="disengage")
    result = asyncio.run(dispatcher.handle_gaming_mode(intent))
    assert result.success is True
    assert "full control restored" in result.voice_message.lower()
    assert mgr.calls == ["disengage"]


def test_handle_gaming_mode_status_engaged():
    mgr = _StubGamingModeManager()
    asyncio.run(mgr.engage())
    dispatcher = _make_dispatcher(gaming_mode_manager=mgr)
    result = asyncio.run(
        dispatcher.handle_gaming_mode(GamingModeIntent(action="status")),
    )
    assert "gaming mode is on" in result.voice_message.lower()


def test_handle_gaming_mode_status_idle():
    mgr = _StubGamingModeManager()
    dispatcher = _make_dispatcher(gaming_mode_manager=mgr)
    result = asyncio.run(
        dispatcher.handle_gaming_mode(GamingModeIntent(action="status")),
    )
    assert "gaming mode is off" in result.voice_message.lower()


def test_handle_gaming_mode_no_manager_returns_clear_error():
    dispatcher = _make_dispatcher(gaming_mode_manager=None)
    result = asyncio.run(
        dispatcher.handle_gaming_mode(GamingModeIntent(action="engage")),
    )
    assert result.success is False
    assert "isn't ready" in result.voice_message.lower()
    assert "openclaw" in result.voice_message.lower()


def test_handle_gaming_mode_unknown_action():
    mgr = _StubGamingModeManager()
    dispatcher = _make_dispatcher(gaming_mode_manager=mgr)
    result = asyncio.run(
        dispatcher.handle_gaming_mode(GamingModeIntent(action="zoom")),
    )
    assert result.success is False
    assert "don't know how" in result.voice_message.lower()


def test_handle_gaming_mode_engage_partial_failure_voice():
    mgr = _StubGamingModeManager()
    mgr.engage_report.plugin_states = [
        _StubPluginState(plugin_id="desktop-control", success=True),
        _StubPluginState(
            plugin_id="windows-control", success=False, error="not installed",
        ),
    ]
    dispatcher = _make_dispatcher(gaming_mode_manager=mgr)
    result = asyncio.run(
        dispatcher.handle_gaming_mode(GamingModeIntent(action="engage")),
    )
    assert result.success is False
    assert "with errors" in result.voice_message.lower()


# ---------------------------------------------------------------------------
# C3: handle_desktop_automation
# ---------------------------------------------------------------------------


def test_handle_desktop_screenshot_success():
    client = _StubClient(scripted={
        "desktop_screenshot": _ToolResult(
            success=True,
            payload={"path": r"C:\Users\test\screenshot.png"},
        ),
    })
    bridge = SimpleNamespace(client=client)
    dispatcher = _make_dispatcher(bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_desktop_automation(
            DesktopIntent(action="screenshot"),
        ),
    )
    assert result.success is True
    assert "screenshot captured" in result.voice_message.lower()
    assert client.calls[0]["tool_name"] == "desktop_screenshot"


def test_handle_desktop_list_windows_returns_count():
    client = _StubClient(scripted={
        "desktop_list_windows": _ToolResult(
            success=True,
            payload={"windows": [
                {"title": "Chrome", "handle": "h1", "app_name": "chrome.exe"},
                {"title": "VS Code", "handle": "h2", "app_name": "code.exe"},
            ]},
        ),
    })
    bridge = SimpleNamespace(client=client)
    dispatcher = _make_dispatcher(bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_desktop_automation(
            DesktopIntent(action="list_windows"),
        ),
    )
    assert result.success is True
    assert "2 windows open" in result.voice_message.lower()


def test_handle_desktop_find_window_with_target():
    client = _StubClient(scripted={
        "desktop_find_window": _ToolResult(
            success=True,
            payload={
                "handle": "abc", "title": "Cursor",
                "app_name": "cursor.exe",
            },
        ),
    })
    bridge = SimpleNamespace(client=client)
    dispatcher = _make_dispatcher(bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_desktop_automation(
            DesktopIntent(action="find_window", target="cursor"),
        ),
    )
    assert result.success is True
    assert "found" in result.voice_message.lower()


def test_handle_desktop_no_bridge_returns_stub():
    """When bridge is None, dispatcher returns the not-reachable stub."""
    dispatcher = _make_dispatcher(bridge=None)
    result = asyncio.run(
        dispatcher.handle_desktop_automation(
            DesktopIntent(action="screenshot"),
        ),
    )
    assert result.success is False
    assert "isn't reachable" in result.voice_message.lower()


def test_handle_desktop_blocked_when_gaming_mode_engaged():
    """Engaging gaming mode disables the underlying plugin -- we
    short-circuit with a clear voice message instead of attempting
    the call."""
    mgr = _StubGamingModeManager()
    asyncio.run(mgr.engage())
    client = _StubClient()
    bridge = SimpleNamespace(client=client)
    dispatcher = _make_dispatcher(gaming_mode_manager=mgr, bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_desktop_automation(
            DesktopIntent(action="screenshot"),
        ),
    )
    assert result.success is False
    assert "gaming mode is on" in result.voice_message.lower()
    # Tool was NOT called.
    assert client.calls == []


def test_handle_desktop_unknown_action():
    bridge = SimpleNamespace(client=_StubClient())
    dispatcher = _make_dispatcher(bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_desktop_automation(
            DesktopIntent(action="frobnicate"),
        ),
    )
    assert result.success is False


# ---------------------------------------------------------------------------
# C3: handle_window_automation
# ---------------------------------------------------------------------------


def test_handle_window_focus_success():
    client = _StubClient(scripted={
        "windows_focus_window": _ToolResult(success=True),
    })
    bridge = SimpleNamespace(client=client)
    dispatcher = _make_dispatcher(bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_window_automation(
            WindowIntent(action="focus", query="chrome"),
        ),
    )
    assert result.success is True
    assert "focused chrome" in result.voice_message.lower()


def test_handle_window_type_success():
    client = _StubClient(scripted={
        "windows_type_text": _ToolResult(success=True),
    })
    bridge = SimpleNamespace(client=client)
    dispatcher = _make_dispatcher(bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_window_automation(
            WindowIntent(action="type", query="search box", value="hello"),
        ),
    )
    assert result.success is True
    assert "typed" in result.voice_message.lower()


def test_handle_window_click_success():
    client = _StubClient(scripted={
        "windows_click_element": _ToolResult(success=True),
    })
    bridge = SimpleNamespace(client=client)
    dispatcher = _make_dispatcher(bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_window_automation(
            WindowIntent(action="click", ref="submit_button"),
        ),
    )
    assert result.success is True
    assert "clicked" in result.voice_message.lower()


def test_handle_window_blocked_during_gaming_mode():
    mgr = _StubGamingModeManager()
    asyncio.run(mgr.engage())
    bridge = SimpleNamespace(client=_StubClient())
    dispatcher = _make_dispatcher(gaming_mode_manager=mgr, bridge=bridge)
    result = asyncio.run(
        dispatcher.handle_window_automation(
            WindowIntent(action="focus", query="chrome"),
        ),
    )
    assert result.success is False
    assert "gaming mode is on" in result.voice_message.lower()
