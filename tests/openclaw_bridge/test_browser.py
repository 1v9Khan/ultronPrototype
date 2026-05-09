"""Tests for ``ultron.openclaw_bridge.browser.BrowserTool``."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ultron.errors import OpenClawToolError
from ultron.openclaw_bridge.browser import (
    ActionResult,
    BrowserTool,
    NavigateResult,
    PageTextResult,
    ScreenshotResult,
    Snapshot,
)
from ultron.openclaw_bridge.client import ToolInvocationResult


def _make_tool_result(success: bool, text: str = "", error: str = "") -> ToolInvocationResult:
    return ToolInvocationResult(
        success=success, tool_name="browser",
        text=text, error=(error or None),
    )


@pytest.fixture
def fake_client() -> Any:
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value=_make_tool_result(True, "ok"))
    return client


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_requires_client() -> None:
    with pytest.raises(ValueError):
        BrowserTool(None)                                 # type: ignore[arg-type]


def test_default_agent_id(fake_client: Any) -> None:
    tool = BrowserTool(fake_client)
    assert tool._agent_id == "ultron-main"                  # noqa: SLF001
    assert tool._default_timeout_s == 90.0                   # noqa: SLF001


# ---------------------------------------------------------------------------
# navigate()
# ---------------------------------------------------------------------------


async def test_navigate_returns_success(fake_client: Any) -> None:
    fake_client.invoke_tool = AsyncMock(return_value=_make_tool_result(
        True, "Title: Hacker News\nLoaded.",
    ))
    tool = BrowserTool(fake_client)
    result = await tool.navigate("https://news.ycombinator.com")
    assert isinstance(result, NavigateResult)
    assert result.success is True
    assert result.url == "https://news.ycombinator.com"
    assert result.title == "Hacker News"
    fake_client.invoke_tool.assert_awaited_once()
    args, kwargs = fake_client.invoke_tool.call_args
    assert args[0] == "browser"
    assert args[1]["action"] == "navigate"
    assert args[1]["url"] == "https://news.ycombinator.com"


async def test_navigate_rejects_empty_url(fake_client: Any) -> None:
    tool = BrowserTool(fake_client)
    result = await tool.navigate("   ")
    assert result.success is False
    assert "empty" in (result.error or "").lower()
    fake_client.invoke_tool.assert_not_called()


async def test_navigate_handles_tool_error(fake_client: Any) -> None:
    fake_client.invoke_tool = AsyncMock(side_effect=OpenClawToolError("unavailable"))
    tool = BrowserTool(fake_client)
    result = await tool.navigate("https://example.com")
    assert result.success is False
    assert "unavailable" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------


async def test_snapshot_default_mode_extracts_refs(fake_client: Any) -> None:
    fake_client.invoke_tool = AsyncMock(return_value=_make_tool_result(
        True, "[e1] Search box\n[e2] Submit button\n[e3] Help link",
    ))
    tool = BrowserTool(fake_client)
    snap = await tool.snapshot()
    assert isinstance(snap, Snapshot)
    assert snap.success is True
    assert snap.mode == "ai"
    assert snap.refs == {
        "e1": "Search box",
        "e2": "Submit button",
        "e3": "Help link",
    }


async def test_snapshot_aria_mode_no_refs(fake_client: Any) -> None:
    fake_client.invoke_tool = AsyncMock(return_value=_make_tool_result(
        True, "<a11y tree>",
    ))
    tool = BrowserTool(fake_client)
    snap = await tool.snapshot(mode="aria")
    assert snap.success is True
    assert snap.mode == "aria"
    assert snap.refs == {}


async def test_snapshot_rejects_invalid_mode(fake_client: Any) -> None:
    tool = BrowserTool(fake_client)
    with pytest.raises(ValueError):
        await tool.snapshot(mode="dom")                   # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# click() / type_text()
# ---------------------------------------------------------------------------


async def test_click_passes_ref(fake_client: Any) -> None:
    tool = BrowserTool(fake_client)
    result = await tool.click("e1")
    assert isinstance(result, ActionResult)
    assert result.success is True
    args, _ = fake_client.invoke_tool.call_args
    assert args[1]["action"] == "click"
    assert args[1]["ref"] == "e1"


async def test_click_rejects_empty_ref(fake_client: Any) -> None:
    tool = BrowserTool(fake_client)
    result = await tool.click("")
    assert result.success is False
    assert "empty" in (result.error or "").lower()
    fake_client.invoke_tool.assert_not_called()


async def test_type_text_passes_ref_and_text(fake_client: Any) -> None:
    tool = BrowserTool(fake_client)
    result = await tool.type_text("e2", "hello world")
    assert result.success is True
    args, _ = fake_client.invoke_tool.call_args
    assert args[1]["action"] == "type"
    assert args[1]["ref"] == "e2"
    assert args[1]["text"] == "hello world"


async def test_type_text_rejects_empty(fake_client: Any) -> None:
    tool = BrowserTool(fake_client)
    r1 = await tool.type_text("", "hi")
    r2 = await tool.type_text("e1", "")
    assert r1.success is False and r2.success is False


# ---------------------------------------------------------------------------
# screenshot()
# ---------------------------------------------------------------------------


async def test_screenshot_decodes_base64(fake_client: Any) -> None:
    payload = base64.b64encode(b"\x89PNG\r\n").decode()
    fake_client.invoke_tool = AsyncMock(return_value=_make_tool_result(
        True, f"Captured: data:image/png;base64,{payload}",
    ))
    tool = BrowserTool(fake_client)
    result = await tool.screenshot()
    assert isinstance(result, ScreenshotResult)
    assert result.success is True
    assert result.image_bytes == b"\x89PNG\r\n"


async def test_screenshot_no_base64_returns_none_bytes(fake_client: Any) -> None:
    fake_client.invoke_tool = AsyncMock(return_value=_make_tool_result(
        True, "Saved to /tmp/shot.png",
    ))
    tool = BrowserTool(fake_client)
    result = await tool.screenshot()
    assert result.success is True
    assert result.image_bytes is None
    assert "/tmp/shot.png" in result.text


# ---------------------------------------------------------------------------
# get_page_text()
# ---------------------------------------------------------------------------


async def test_get_page_text(fake_client: Any) -> None:
    fake_client.invoke_tool = AsyncMock(return_value=_make_tool_result(
        True, "Page content here.",
    ))
    tool = BrowserTool(fake_client)
    result = await tool.get_page_text()
    assert isinstance(result, PageTextResult)
    assert result.success is True
    assert result.text == "Page content here."


# ---------------------------------------------------------------------------
# Title extraction edge cases
# ---------------------------------------------------------------------------


def test_title_extraction_handles_quoted() -> None:
    text = 'Title: "News"\nrest'
    assert BrowserTool._extract_title(text) == "News"


def test_title_extraction_returns_none_when_missing() -> None:
    assert BrowserTool._extract_title("nothing relevant") is None


def test_title_extraction_empty_text() -> None:
    assert BrowserTool._extract_title("") is None


# ---------------------------------------------------------------------------
# Ref extraction edge cases
# ---------------------------------------------------------------------------


def test_ref_extraction_skips_non_bracketed() -> None:
    text = "header text\n[e1] hi\nfooter\n[e2] hello"
    refs = BrowserTool._extract_refs(text)
    assert refs == {"e1": "hi", "e2": "hello"}


def test_ref_extraction_handles_empty() -> None:
    assert BrowserTool._extract_refs("") == {}
