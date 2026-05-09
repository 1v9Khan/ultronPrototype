"""Browser tool wrapper (Phase 6).

Thin facade over :meth:`OpenClawClient.invoke_tool` that exposes the
browser primitives Ultron's dispatcher / future skills want to call.
Each method assembles a structured prompt asking the OpenClaw
``ultron-main`` agent to use the browser tool with specific
parameters; the agent runs the tool turn and the wrapper unpacks the
result into a typed dataclass.

Why a wrapper at all? :meth:`OpenClawClient.invoke_tool` returns a
generic :class:`ToolInvocationResult` with the agent's free-form
text. Browser flows have their own shape — navigate returns a URL,
snapshot returns refs the next click can target, screenshot returns
bytes. A wrapper that knows the per-method semantics keeps callers
from re-parsing free text.

Phase 6 ships the wrapper and rewrites
:meth:`OpenClawDispatcher.handle_browser` to use it. Multi-step
flows (login → navigate → fill form) stay on the OpenClaw side via
the agent's reasoning; the wrapper is for one-shot operations
fired by Ultron's intent dispatch.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from ultron.errors import OpenClawToolError
from ultron.openclaw_bridge.client import OpenClawClient, ToolInvocationResult
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.browser")


SnapshotMode = Literal["ai", "aria"]


@dataclass(frozen=True)
class NavigateResult:
    """Outcome of a navigate call."""

    success: bool
    url: str = ""
    title: Optional[str] = None
    text: str = ""                                       # full agent response
    error: Optional[str] = None


@dataclass(frozen=True)
class Snapshot:
    """Snapshot of the current page. ``mode='ai'`` returns refs the
    next click/type call can use as a target; ``mode='aria'`` returns
    the raw accessibility tree."""

    success: bool
    mode: SnapshotMode = "ai"
    text: str = ""                                       # full snapshot text
    refs: Dict[str, str] = field(default_factory=dict)   # ref-id → label
    error: Optional[str] = None


@dataclass(frozen=True)
class ActionResult:
    """Outcome of click / type / scroll."""

    success: bool
    action: str = ""
    text: str = ""
    error: Optional[str] = None


@dataclass(frozen=True)
class ScreenshotResult:
    """Outcome of a screenshot call. ``image_bytes`` is None when the
    underlying agent didn't return a base64 payload (some prompts
    receive only a path / URL reference instead)."""

    success: bool
    image_bytes: Optional[bytes] = None
    text: str = ""
    error: Optional[str] = None


@dataclass(frozen=True)
class PageTextResult:
    """Outcome of a get_page_text call."""

    success: bool
    text: str = ""
    error: Optional[str] = None


class BrowserTool:
    """Browser primitives via OpenClaw agent.

    Args:
        client: a constructed :class:`OpenClawClient`. Required —
            without a live client, callers must short-circuit before
            constructing the wrapper.
        agent_id: which OpenClaw agent runs the browser tool turns.
            Defaults to ``ultron-main`` (the user-facing persona);
            tests / background workflows can override.
        default_timeout_s: per-call timeout. Browser actions can take
            seconds to a minute or more; default is generous.
    """

    def __init__(
        self,
        client: OpenClawClient,
        *,
        agent_id: str = "ultron-main",
        default_timeout_s: float = 90.0,
    ) -> None:
        if client is None:
            raise ValueError("BrowserTool requires a non-None OpenClawClient")
        self._client = client
        self._agent_id = agent_id
        self._default_timeout_s = default_timeout_s

    # -------------------------------------------------------------------
    # Public surface
    # -------------------------------------------------------------------

    async def navigate(
        self,
        url: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> NavigateResult:
        """Open ``url`` in the active browser tab. Returns a
        :class:`NavigateResult` describing the outcome."""
        url = (url or "").strip()
        if not url:
            return NavigateResult(success=False, error="empty url")
        result = await self._invoke(
            {"action": "navigate", "url": url},
            timeout_s=timeout_s,
        )
        if not result.success:
            return NavigateResult(success=False, url=url, error=result.error)
        return NavigateResult(
            success=True, url=url, text=result.text,
            title=self._extract_title(result.text),
        )

    async def snapshot(
        self,
        mode: SnapshotMode = "ai",
        *,
        timeout_s: Optional[float] = None,
    ) -> Snapshot:
        """Capture a snapshot of the current page.

        ``mode='ai'`` returns labelled refs (e.g. ``e12``) suitable
        for follow-up :meth:`click` / :meth:`type_text` calls.
        ``mode='aria'`` returns the accessibility tree as text.
        """
        if mode not in ("ai", "aria"):
            raise ValueError(f"mode must be 'ai' or 'aria', got {mode!r}")
        result = await self._invoke(
            {"action": "snapshot", "mode": mode},
            timeout_s=timeout_s,
        )
        if not result.success:
            return Snapshot(success=False, mode=mode, error=result.error)
        return Snapshot(
            success=True, mode=mode, text=result.text,
            refs=self._extract_refs(result.text) if mode == "ai" else {},
        )

    async def click(
        self,
        ref: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> ActionResult:
        """Click an element identified by ``ref`` (from a prior
        ``ai``-mode snapshot)."""
        ref = (ref or "").strip()
        if not ref:
            return ActionResult(
                success=False, action="click", error="empty ref",
            )
        result = await self._invoke(
            {"action": "click", "ref": ref},
            timeout_s=timeout_s,
        )
        return ActionResult(
            success=result.success, action="click",
            text=result.text, error=result.error,
        )

    async def type_text(
        self,
        ref: str,
        text: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> ActionResult:
        """Type ``text`` into the element identified by ``ref``."""
        ref = (ref or "").strip()
        if not ref:
            return ActionResult(
                success=False, action="type", error="empty ref",
            )
        if not text:
            return ActionResult(
                success=False, action="type", error="empty text",
            )
        result = await self._invoke(
            {"action": "type", "ref": ref, "text": text},
            timeout_s=timeout_s,
        )
        return ActionResult(
            success=result.success, action="type",
            text=result.text, error=result.error,
        )

    async def screenshot(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> ScreenshotResult:
        """Capture a screenshot of the active tab.

        The agent typically returns either a base64-encoded image
        body or a file path/URL reference. We decode base64 when
        present; otherwise the text reference is forwarded so the
        caller can fetch the file separately.
        """
        result = await self._invoke(
            {"action": "screenshot"},
            timeout_s=timeout_s,
        )
        if not result.success:
            return ScreenshotResult(success=False, error=result.error)
        image_bytes = self._extract_base64_image(result.text)
        return ScreenshotResult(
            success=True, image_bytes=image_bytes, text=result.text,
        )

    async def get_page_text(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> PageTextResult:
        """Return the rendered text of the current page."""
        result = await self._invoke(
            {"action": "get_page_text"},
            timeout_s=timeout_s,
        )
        if not result.success:
            return PageTextResult(success=False, error=result.error)
        return PageTextResult(success=True, text=result.text)

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    async def _invoke(
        self,
        params: Dict[str, Any],
        *,
        timeout_s: Optional[float] = None,
    ) -> ToolInvocationResult:
        try:
            return await self._client.invoke_tool(
                "browser", params,
                agent_id=self._agent_id,
                timeout_s=timeout_s if timeout_s is not None else self._default_timeout_s,
            )
        except OpenClawToolError as e:
            logger.warning(
                "browser tool reported unavailable: %s", e,
            )
            return ToolInvocationResult(
                success=False, tool_name="browser",
                error=str(e),
            )

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        """Best-effort title extraction from the agent's free-form
        navigate response. Looks for ``Title: <text>`` or
        ``"<text>"`` patterns; returns None when neither matches."""
        for line in (text or "").splitlines():
            line = line.strip()
            if line.lower().startswith("title:"):
                return line.split(":", 1)[1].strip().strip('"').strip()
        return None

    @staticmethod
    def _extract_refs(snapshot_text: str) -> Dict[str, str]:
        """Best-effort extraction of ``ai``-mode refs from snapshot
        text. Looks for ``[refId] label`` style lines and returns
        the mapping. Tolerates absence — callers fall back to passing
        the raw text to a follow-up agent turn."""
        refs: Dict[str, str] = {}
        for raw in (snapshot_text or "").splitlines():
            line = raw.strip()
            if not line.startswith("["):
                continue
            close = line.find("]")
            if close < 1:
                continue
            ref_id = line[1:close].strip()
            label = line[close + 1:].strip()
            if ref_id:
                refs[ref_id] = label
        return refs

    @staticmethod
    def _extract_base64_image(text: str) -> Optional[bytes]:
        """Pull the first base64 PNG/JPEG payload out of the agent's
        text. Tolerates ``data:image/...;base64,...`` and bare
        base64. Returns None when nothing recognisable is present."""
        if not text:
            return None
        # data: URI form
        marker = "base64,"
        idx = text.find(marker)
        if idx >= 0:
            payload = text[idx + len(marker):]
            payload = payload.split()[0] if payload else ""
            try:
                return base64.b64decode(payload, validate=True)
            except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
                return None
        return None


__all__ = [
    "ActionResult",
    "BrowserTool",
    "NavigateResult",
    "PageTextResult",
    "ScreenshotResult",
    "Snapshot",
    "SnapshotMode",
]
