"""Desktop / windows control tool wrappers (V1-spec gap C3).

Thin facades over :meth:`OpenClawClient.invoke_tool` that mirror the
shape of :class:`BrowserTool` (Phase 6) -- one wrapper class per
plugin, one method per primitive. Each method assembles a structured
prompt asking the OpenClaw ``ultron-main`` agent to invoke the
underlying tool with specific parameters; the wrapper unpacks the
agent response into a typed dataclass.

Critical contract: all methods translate :class:`OpenClawToolError`
(plugin not installed / not enabled) into structured failures rather
than raising. The dispatcher then surfaces the failure as a clear
voice message to the user.

Tool slugs are configurable via ``config.desktop`` and
``config.window_control`` so the integration tracks slug renames in
the OpenClaw plugin marketplace without code changes.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ultron.errors import OpenClawToolError
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.desktop")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesktopScreenshotResult:
    success: bool
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    target: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class WindowEntry:
    title: str
    handle: str = ""
    app_name: str = ""


@dataclass(frozen=True)
class ListWindowsResult:
    success: bool
    windows: List[WindowEntry] = field(default_factory=list)
    error: Optional[str] = None


@dataclass(frozen=True)
class FindWindowResult:
    success: bool
    handle: str = ""
    title: str = ""
    app_name: str = ""
    error: Optional[str] = None


@dataclass(frozen=True)
class WindowActionResult:
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# DesktopTool (V1-gap C3)
# ---------------------------------------------------------------------------


class DesktopTool:
    """Wrapper over the OpenClaw ``desktop-control`` plugin.

    Args:
        client: live :class:`OpenClawClient`. Required -- no offline mode.
        agent_id: which OpenClaw agent issues the tool call.
        screenshot_tool: tool slug for screenshot capture.
        list_windows_tool: tool slug for enumerate-windows.
        find_window_tool: tool slug for window lookup by title.
    """

    def __init__(
        self,
        client: Any,
        *,
        agent_id: str = "ultron-main",
        screenshot_tool: str = "desktop_screenshot",
        list_windows_tool: str = "desktop_list_windows",
        find_window_tool: str = "desktop_find_window",
    ) -> None:
        self.client = client
        self.agent_id = agent_id
        self.screenshot_tool = screenshot_tool
        self.list_windows_tool = list_windows_tool
        self.find_window_tool = find_window_tool

    async def screenshot(
        self,
        target: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
    ) -> DesktopScreenshotResult:
        """Capture a screenshot of the full screen or a named window."""
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('desktop_screenshot')
        params: Dict[str, Any] = {}
        if target:
            params["target"] = target
        try:
            result = await self.client.invoke_tool(
                self.screenshot_tool, params,
                agent_id=self.agent_id, timeout_s=timeout_s,
            )
        except OpenClawToolError as e:
            return DesktopScreenshotResult(
                success=False, target=target,
                error=str(e)[:300] or "tool unavailable",
            )
        except Exception as e:                                       # noqa: BLE001
            logger.warning("desktop screenshot raised: %s", e)
            return DesktopScreenshotResult(
                success=False, target=target,
                error=str(e)[:300] or "unknown error",
            )
        if not getattr(result, "success", False):
            return DesktopScreenshotResult(
                success=False, target=target,
                error=getattr(result, "error", None) or "screenshot failed",
            )
        # Image data may arrive as a path, base64 in payload, or both.
        payload = getattr(result, "raw", None) or {}
        image_path = payload.get("path") or payload.get("image_path")
        image_bytes = None
        b64 = payload.get("image_base64") or payload.get("image_b64")
        if isinstance(b64, str) and b64:
            try:
                image_bytes = base64.b64decode(b64)
            except (ValueError, TypeError) as e:
                logger.debug("screenshot base64 decode failed: %s", e)
        return DesktopScreenshotResult(
            success=True, image_path=image_path, image_bytes=image_bytes,
            target=target,
        )

    async def list_windows(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> ListWindowsResult:
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('desktop_list_windows')
        try:
            result = await self.client.invoke_tool(
                self.list_windows_tool, {}, agent_id=self.agent_id,
                timeout_s=timeout_s,
            )
        except OpenClawToolError as e:
            return ListWindowsResult(success=False, error=str(e)[:300])
        except Exception as e:                                       # noqa: BLE001
            logger.warning("desktop list_windows raised: %s", e)
            return ListWindowsResult(success=False, error=str(e)[:300])
        if not getattr(result, "success", False):
            return ListWindowsResult(
                success=False,
                error=getattr(result, "error", None) or "list_windows failed",
            )
        payload = getattr(result, "raw", None) or {}
        raw = payload.get("windows") or []
        rows: List[WindowEntry] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    rows.append(WindowEntry(
                        title=str(item.get("title", "")),
                        handle=str(item.get("handle", "")),
                        app_name=str(item.get("app_name", "")),
                    ))
                elif isinstance(item, str):
                    rows.append(WindowEntry(title=item))
        return ListWindowsResult(success=True, windows=rows)

    async def find_window(
        self,
        query: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> FindWindowResult:
        # Anticheat-safe mode: hard-blocked while the user is in game.
        from ultron.safety.anticheat import guard as _anticheat_guard
        _anticheat_guard('desktop_find_window')
        if not (query or "").strip():
            return FindWindowResult(
                success=False, error="empty window query",
            )
        try:
            result = await self.client.invoke_tool(
                self.find_window_tool, {"query": query.strip()},
                agent_id=self.agent_id, timeout_s=timeout_s,
            )
        except OpenClawToolError as e:
            return FindWindowResult(success=False, error=str(e)[:300])
        except Exception as e:                                       # noqa: BLE001
            logger.warning("desktop find_window raised: %s", e)
            return FindWindowResult(success=False, error=str(e)[:300])
        if not getattr(result, "success", False):
            return FindWindowResult(
                success=False,
                error=getattr(result, "error", None) or "no matching window",
            )
        payload = getattr(result, "raw", None) or {}
        return FindWindowResult(
            success=True,
            handle=str(payload.get("handle", "")),
            title=str(payload.get("title", "")),
            app_name=str(payload.get("app_name", "")),
        )


# ---------------------------------------------------------------------------
# WindowControlTool (V1-gap C3)
# ---------------------------------------------------------------------------


class WindowControlTool:
    """Wrapper over the OpenClaw ``windows-control`` plugin (UI Automation).

    Methods:
        - :meth:`focus(query)` -- bring a window to the foreground.
        - :meth:`click(ref)` -- click a UIA reference.
        - :meth:`type_text(ref, text)` -- type into a UIA reference.
    """

    def __init__(
        self,
        client: Any,
        *,
        agent_id: str = "ultron-main",
        focus_tool: str = "windows_focus_window",
        click_tool: str = "windows_click_element",
        type_tool: str = "windows_type_text",
    ) -> None:
        self.client = client
        self.agent_id = agent_id
        self.focus_tool = focus_tool
        self.click_tool = click_tool
        self.type_tool = type_tool

    async def focus(
        self,
        query: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> WindowActionResult:
        if not (query or "").strip():
            return WindowActionResult(
                success=False, error="empty window query",
            )
        return await self._invoke(
            self.focus_tool, {"query": query.strip()}, timeout_s=timeout_s,
        )

    async def click(
        self,
        ref: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> WindowActionResult:
        if not (ref or "").strip():
            return WindowActionResult(
                success=False, error="empty UIA reference",
            )
        return await self._invoke(
            self.click_tool, {"ref": ref.strip()}, timeout_s=timeout_s,
        )

    async def type_text(
        self,
        ref: str,
        value: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> WindowActionResult:
        if not (ref or "").strip():
            return WindowActionResult(
                success=False, error="empty UIA reference",
            )
        return await self._invoke(
            self.type_tool,
            {"ref": ref.strip(), "value": value},
            timeout_s=timeout_s,
        )

    async def _invoke(
        self,
        tool_name: str,
        params: Dict[str, Any],
        *,
        timeout_s: Optional[float],
    ) -> WindowActionResult:
        try:
            result = await self.client.invoke_tool(
                tool_name, params, agent_id=self.agent_id,
                timeout_s=timeout_s,
            )
        except OpenClawToolError as e:
            return WindowActionResult(
                success=False, error=str(e)[:300] or "tool unavailable",
            )
        except Exception as e:                                       # noqa: BLE001
            logger.warning("windows-control %s raised: %s", tool_name, e)
            return WindowActionResult(
                success=False, error=str(e)[:300] or "unknown error",
            )
        if not getattr(result, "success", False):
            return WindowActionResult(
                success=False,
                error=getattr(result, "error", None) or f"{tool_name} failed",
            )
        return WindowActionResult(success=True)


__all__ = [
    "DesktopTool",
    "WindowControlTool",
    "DesktopScreenshotResult",
    "ListWindowsResult",
    "FindWindowResult",
    "WindowActionResult",
    "WindowEntry",
]
