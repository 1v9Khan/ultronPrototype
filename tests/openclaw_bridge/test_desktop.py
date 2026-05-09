"""V1-gap C3: DesktopTool + WindowControlTool unit tests.

Both wrap :meth:`OpenClawClient.invoke_tool`. We use a stub client that
records calls and returns scripted results so we can verify each
primitive shapes its arguments correctly without spawning OpenClaw.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from ultron.errors import OpenClawToolError
from ultron.openclaw_bridge.desktop import (
    DesktopScreenshotResult,
    DesktopTool,
    FindWindowResult,
    ListWindowsResult,
    WindowControlTool,
    WindowEntry,
)


@dataclass
class _ToolResult:
    success: bool
    error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class _StubClient:
    def __init__(
        self,
        *,
        scripted: Optional[Dict[str, _ToolResult]] = None,
        raise_for: Optional[Dict[str, Exception]] = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._scripted = scripted or {}
        self._raise_for = raise_for or {}

    async def invoke_tool(self, tool_name, params, *, agent_id=None, timeout_s=None):
        self.calls.append({
            "tool_name": tool_name, "params": dict(params),
            "agent_id": agent_id, "timeout_s": timeout_s,
        })
        if tool_name in self._raise_for:
            raise self._raise_for[tool_name]
        return self._scripted.get(tool_name, _ToolResult(success=True, payload={}))


# ---------------------------------------------------------------------------
# DesktopTool
# ---------------------------------------------------------------------------


def test_screenshot_calls_correct_tool():
    client = _StubClient(scripted={
        "desktop_screenshot": _ToolResult(
            success=True,
            payload={"path": "/tmp/screenshot.png"},
        ),
    })
    tool = DesktopTool(client)
    result = asyncio.run(tool.screenshot())
    assert isinstance(result, DesktopScreenshotResult)
    assert result.success is True
    assert result.image_path == "/tmp/screenshot.png"
    assert client.calls[0]["tool_name"] == "desktop_screenshot"


def test_screenshot_with_target_passes_param():
    client = _StubClient(scripted={
        "desktop_screenshot": _ToolResult(success=True, payload={}),
    })
    tool = DesktopTool(client)
    asyncio.run(tool.screenshot(target="active_window"))
    assert client.calls[0]["params"] == {"target": "active_window"}


def test_screenshot_decodes_base64_payload():
    raw = b"fake png bytes"
    client = _StubClient(scripted={
        "desktop_screenshot": _ToolResult(
            success=True,
            payload={
                "image_base64": base64.b64encode(raw).decode("ascii"),
            },
        ),
    })
    tool = DesktopTool(client)
    result = asyncio.run(tool.screenshot())
    assert result.image_bytes == raw


def test_screenshot_handles_tool_error():
    client = _StubClient(
        raise_for={"desktop_screenshot": OpenClawToolError("plugin not loaded")},
    )
    tool = DesktopTool(client)
    result = asyncio.run(tool.screenshot())
    assert result.success is False
    assert "plugin not loaded" in (result.error or "")


def test_list_windows_parses_payload():
    client = _StubClient(scripted={
        "desktop_list_windows": _ToolResult(
            success=True,
            payload={"windows": [
                {"title": "Chrome", "handle": "h1", "app_name": "chrome"},
                "Slack",  # bare-string entry
            ]},
        ),
    })
    tool = DesktopTool(client)
    result = asyncio.run(tool.list_windows())
    assert isinstance(result, ListWindowsResult)
    assert result.success is True
    assert len(result.windows) == 2
    assert result.windows[0].title == "Chrome"
    assert result.windows[1].title == "Slack"


def test_list_windows_handles_failure():
    client = _StubClient(scripted={
        "desktop_list_windows": _ToolResult(
            success=False, error="plugin not enabled",
        ),
    })
    tool = DesktopTool(client)
    result = asyncio.run(tool.list_windows())
    assert result.success is False


def test_find_window_returns_payload():
    client = _StubClient(scripted={
        "desktop_find_window": _ToolResult(
            success=True,
            payload={
                "handle": "abc123", "title": "Cursor",
                "app_name": "cursor.exe",
            },
        ),
    })
    tool = DesktopTool(client)
    result = asyncio.run(tool.find_window("cursor"))
    assert isinstance(result, FindWindowResult)
    assert result.success is True
    assert result.handle == "abc123"
    assert client.calls[0]["params"] == {"query": "cursor"}


def test_find_window_rejects_blank_query():
    tool = DesktopTool(_StubClient())
    result = asyncio.run(tool.find_window(""))
    assert result.success is False
    assert "empty" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# WindowControlTool
# ---------------------------------------------------------------------------


def test_focus_calls_focus_tool():
    client = _StubClient(scripted={
        "windows_focus_window": _ToolResult(success=True),
    })
    tool = WindowControlTool(client)
    result = asyncio.run(tool.focus("chrome"))
    assert result.success is True
    assert client.calls[0]["tool_name"] == "windows_focus_window"
    assert client.calls[0]["params"] == {"query": "chrome"}


def test_click_calls_click_tool():
    client = _StubClient(scripted={
        "windows_click_element": _ToolResult(success=True),
    })
    tool = WindowControlTool(client)
    result = asyncio.run(tool.click("submit_btn"))
    assert result.success is True
    assert client.calls[0]["tool_name"] == "windows_click_element"
    assert client.calls[0]["params"] == {"ref": "submit_btn"}


def test_type_text_calls_type_tool():
    client = _StubClient(scripted={
        "windows_type_text": _ToolResult(success=True),
    })
    tool = WindowControlTool(client)
    result = asyncio.run(tool.type_text("search_box", "hello world"))
    assert result.success is True
    assert client.calls[0]["params"] == {
        "ref": "search_box", "value": "hello world",
    }


def test_focus_rejects_blank_query():
    tool = WindowControlTool(_StubClient())
    result = asyncio.run(tool.focus(""))
    assert result.success is False


def test_click_rejects_blank_ref():
    tool = WindowControlTool(_StubClient())
    result = asyncio.run(tool.click(""))
    assert result.success is False


def test_type_handles_tool_error():
    client = _StubClient(
        raise_for={"windows_type_text": OpenClawToolError("not loaded")},
    )
    tool = WindowControlTool(client)
    result = asyncio.run(tool.type_text("box", "text"))
    assert result.success is False
    assert "not loaded" in (result.error or "")


def test_unexpected_exception_returns_structured_failure():
    client = _StubClient(
        raise_for={"windows_focus_window": RuntimeError("transport bad")},
    )
    tool = WindowControlTool(client)
    result = asyncio.run(tool.focus("chrome"))
    assert result.success is False
    assert "transport bad" in (result.error or "")
