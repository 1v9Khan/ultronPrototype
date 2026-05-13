"""OpenClawDispatcher — dispatch surface for OpenClaw-bound intents.

In Foundation Phase 5 every dispatch method returned a stub
:class:`DispatchResult` indicating the capability wasn't yet integrated.
OpenClaw integration phases progressively wire each handler to a real
Gateway call:

- Phase 4 (Telegram): :meth:`handle_messaging` calls
  :meth:`OpenClawClient.send_message` when a bridge is wired.
- Phase 6 (Browser tool): :meth:`handle_browser` will use
  :meth:`OpenClawClient.invoke_tool`.
- Phase 12 (Media generation): :meth:`handle_media_generation`
  similarly.

Stubs remain in place for capabilities that haven't yet shipped and
for the case where ``openclaw.enabled=False`` (current default). The
voice phrase stays in Ultron's voice so the user gets a coherent
response, not a stack trace.

4B plan Item 8: each handle_* method runs a pre-flight block-and-revise
validator (if enabled and an LLM is wired) BEFORE the dispatch. When
the validator returns ``allow=False`` the dispatcher returns a
DispatchResult shaped like a stub but with the validator's reason —
so the user hears why the call was blocked rather than seeing it
silently dropped. Default OFF.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ultron.config import UltronConfig, get_config
from ultron.openclaw_routing.intents import (
    BrowserIntent,
    DesktopIntent,
    DispatchResult,
    FileOpIntent,
    GamingModeIntent,
    MediaGenIntent,
    MessagingIntent,
    ShellOpIntent,
    WindowIntent,
)
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_routing.dispatcher")


class OpenClawDispatcher:
    """Dispatcher for OpenClaw-dependent intents.

    Args:
        config: The full :class:`UltronConfig`. Read at construction so
            tests can inject a config without re-reading the YAML.
        llm: an optional LLMEngine-like object used by the block-and-
            revise validator (4B plan Item 8). When ``None``, the
            validator fails open — dispatch proceeds as before.

    All dispatch methods are async because real OpenClaw calls will be
    HTTP. The stubs return immediately.
    """

    def __init__(
        self,
        config: UltronConfig | None = None,
        *,
        llm: Optional[Any] = None,
        bridge: Optional[Any] = None,
        gaming_mode_manager: Optional[Any] = None,
    ) -> None:
        cfg = config if config is not None else get_config()
        self._cfg = cfg
        self.enabled = cfg.openclaw.enabled
        self.gateway_url = cfg.openclaw.gateway_url
        self.fail_open = cfg.openclaw.fail_open
        self.stub_responses_enabled = cfg.routing.stub_responses_enabled
        # 4B plan Item 8 — block-and-revise validator. None when no LLM
        # is wired; the validator itself fails open when unset, so we
        # also short-circuit here to skip the LLM-call cost entirely.
        self._llm = llm
        # Phase 4 — OpenClaw bridge. When set + `bridge.client` is non-None,
        # `handle_messaging` calls a real Gateway send instead of the
        # stub. Any other handler currently still returns the stub; later
        # phases (6/12/etc.) replace those.
        self._bridge = bridge
        # V1-gap A1 — gaming-mode manager. None disables the engage/
        # disengage flow with a clear voice message; otherwise the
        # dispatcher routes GAMING_MODE intents to it.
        self._gaming_mode_manager = gaming_mode_manager

    # --- per-capability dispatch surface -----------------------------------

    async def handle_browser(self, intent: BrowserIntent) -> DispatchResult:
        """Browser automation (open/click/fill/screenshot).

        Phase 6: when the bridge is wired and ``browser.enabled`` is
        true, dispatches via :class:`BrowserTool` (which fires an
        OpenClaw agent turn against the browser plugin). Falls back
        to the stub voice message in either of:

        - bridge absent / client missing,
        - ``browser.enabled: false`` in config (operator opt-out).

        The 4B plan Item 8 block-and-revise pre-flight still runs
        regardless of bridge state.
        """
        blocked = self._maybe_block(
            tool_name="browser",
            goal=intent.raw_text,
            tool_args={
                "action": intent.action, "url": intent.url,
                "target": intent.target, "value": intent.value,
            },
        )
        if blocked is not None:
            return blocked

        client = self._bridge.client if self._bridge is not None else None
        if client is not None and self._cfg.browser.enabled:
            return await self._browse_via_bridge(intent)

        return self._stub_response(
            capability="browser_automation",
            voice_message=(
                "I'd open that page for you, but the gateway isn't connected yet."
            ),
            metadata={"action": intent.action, "url": intent.url},
        )

    async def _browse_via_bridge(self, intent: BrowserIntent) -> DispatchResult:
        """Phase 6 live path. Maps the BrowserIntent's action to a
        BrowserTool primitive, executes it, and packages the result
        as a DispatchResult. Fail-open at every step."""
        from ultron.openclaw_bridge.browser import BrowserTool

        client = self._bridge.client                              # checked by caller
        cfg = self._cfg.browser
        agent_id = self._cfg.openclaw.required_agent_id or "ultron-main"
        tool = BrowserTool(client, agent_id=agent_id)
        action = (intent.action or "navigate").lower()

        try:
            voice = ""
            metadata: Dict[str, Any] = {
                "stub": False, "action": action, "url": intent.url,
            }
            if action in ("navigate", "open", "go"):
                if not intent.url:
                    return DispatchResult(
                        success=False,
                        voice_message="I need a URL to open.",
                        error="missing url",
                        metadata=metadata,
                    )
                result = await tool.navigate(
                    intent.url,
                    timeout_s=cfg.default_navigation_timeout_seconds,
                )
                if not result.success:
                    return DispatchResult(
                        success=False,
                        voice_message=(
                            "I couldn't load that page just now."
                        ),
                        error=result.error,
                        metadata=metadata,
                    )
                voice = (
                    f"Loaded {result.title}." if result.title
                    else f"Loaded {intent.url}."
                )
                return DispatchResult(
                    success=True, voice_message=voice, metadata=metadata,
                )
            if action == "screenshot":
                result = await tool.screenshot(
                    timeout_s=cfg.default_screenshot_timeout_seconds,
                )
                metadata["has_image"] = result.image_bytes is not None
                if not result.success:
                    return DispatchResult(
                        success=False,
                        voice_message="I couldn't capture the screen.",
                        error=result.error,
                        metadata=metadata,
                    )
                return DispatchResult(
                    success=True,
                    voice_message="Screenshot captured.",
                    metadata=metadata,
                )
            if action == "snapshot":
                snap = await tool.snapshot(
                    cfg.default_snapshot_mode,
                    timeout_s=cfg.default_action_timeout_seconds,
                )
                metadata["mode"] = snap.mode
                metadata["ref_count"] = len(snap.refs)
                if not snap.success:
                    return DispatchResult(
                        success=False,
                        voice_message="I couldn't read the page.",
                        error=snap.error,
                        metadata=metadata,
                    )
                return DispatchResult(
                    success=True,
                    voice_message=f"Snapshot taken ({len(snap.refs)} elements).",
                    metadata=metadata,
                )
            if action == "click":
                if not intent.target:
                    return DispatchResult(
                        success=False,
                        voice_message="I don't know what to click.",
                        error="missing target ref",
                        metadata=metadata,
                    )
                result = await tool.click(
                    intent.target,
                    timeout_s=cfg.default_action_timeout_seconds,
                )
                return DispatchResult(
                    success=result.success,
                    voice_message=(
                        "Clicked." if result.success
                        else "I couldn't click that."
                    ),
                    error=result.error,
                    metadata=metadata,
                )
            if action == "type":
                if not intent.target or intent.value is None:
                    return DispatchResult(
                        success=False,
                        voice_message=(
                            "I need both a field and the text to type."
                        ),
                        error="missing target ref or value",
                        metadata=metadata,
                    )
                result = await tool.type_text(
                    intent.target, intent.value,
                    timeout_s=cfg.default_action_timeout_seconds,
                )
                return DispatchResult(
                    success=result.success,
                    voice_message=(
                        "Typed." if result.success
                        else "I couldn't type that."
                    ),
                    error=result.error,
                    metadata=metadata,
                )
            # Unknown action — fall through with a clear error.
            return DispatchResult(
                success=False,
                voice_message=(
                    f"I don't know how to {action} a page."
                ),
                error=f"unknown action: {action}",
                metadata=metadata,
            )
        except Exception as e:                                     # noqa: BLE001
            logger.warning("browser dispatch raised: %s", e)
            return DispatchResult(
                success=False,
                voice_message=(
                    "Something went wrong on the browser side. "
                    "Try again in a moment."
                ),
                error=str(e),
                metadata={"stub": False, "action": action},
            )

    async def handle_media_generation(self, intent: MediaGenIntent) -> DispatchResult:
        """Image / video / music generation via OpenClaw provider plugins.

        Phase 12: when bridge wired AND ``media_generation.enabled``,
        dispatch via :meth:`OpenClawClient.invoke_tool` against the
        appropriate tool slug for the requested medium. Falls back to
        the stub voice message otherwise.
        """
        blocked = self._maybe_block(
            tool_name="media_generation",
            goal=intent.raw_text,
            tool_args={"medium": intent.medium, "description": intent.description},
        )
        if blocked is not None:
            return blocked

        client = self._bridge.client if self._bridge is not None else None
        if client is not None and self._cfg.media_generation.enabled:
            return await self._generate_via_bridge(intent)

        return self._stub_response(
            capability="media_generation",
            voice_message=(
                "I'd generate that for you, but the gateway isn't connected yet."
            ),
            metadata={"medium": intent.medium},
        )

    async def _generate_via_bridge(self, intent: MediaGenIntent) -> DispatchResult:
        """Phase 12 live path. Maps the medium to the configured tool
        slug, builds a parameter dict (description + optional provider
        override), and fires :meth:`OpenClawClient.invoke_tool`.
        Fail-open at every step."""
        client = self._bridge.client                              # checked by caller
        cfg = self._cfg.media_generation
        medium = (intent.medium or "image").lower()
        description = (intent.description or "").strip()
        if not description:
            return DispatchResult(
                success=False,
                voice_message=(
                    "I need a description for what to generate."
                ),
                error="missing description",
                metadata={"stub": False, "medium": medium},
            )

        if medium == "image":
            tool_name = cfg.image_tool
            provider = cfg.default_image_provider
        elif medium == "video":
            tool_name = cfg.video_tool
            provider = cfg.default_video_provider
        elif medium in ("music", "audio"):
            tool_name = cfg.music_tool
            provider = cfg.default_music_provider
            medium = "music"                                       # normalise
        else:
            return DispatchResult(
                success=False,
                voice_message=f"I don't know how to generate {medium}.",
                error=f"unknown medium: {medium}",
                metadata={"stub": False, "medium": medium},
            )

        params: Dict[str, Any] = {"prompt": description}
        if provider:
            params["provider"] = provider

        agent_id = self._cfg.openclaw.required_agent_id or "ultron-main"
        try:
            result = await client.invoke_tool(
                tool_name, params,
                agent_id=agent_id,
                timeout_s=cfg.default_timeout_seconds,
            )
        except Exception as e:                                     # noqa: BLE001
            logger.warning("media-gen dispatch raised: %s", e)
            return DispatchResult(
                success=False,
                voice_message=(
                    "Something went wrong on the generation side. "
                    "Try again in a moment."
                ),
                error=str(e),
                metadata={"stub": False, "medium": medium},
            )

        if not result.success:
            return DispatchResult(
                success=False,
                voice_message=(
                    f"I couldn't generate that {medium} just now."
                ),
                error=result.error,
                metadata={"stub": False, "medium": medium},
            )

        # Voice phrasing: voice queries get a confirmation that the
        # media was sent (Telegram delivery in the canonical setup);
        # text queries see inline delivery handled by OpenClaw.
        voice = (
            f"{medium.capitalize()} generated. I sent it to your phone."
            if cfg.delivery_voice == "telegram"
            else f"{medium.capitalize()} generated."
        )
        return DispatchResult(
            success=True,
            voice_message=voice,
            metadata={
                "stub": False,
                "medium": medium,
                "tool_name": tool_name,
                "provider": provider,
            },
        )

    async def handle_messaging(self, intent: MessagingIntent) -> DispatchResult:
        """Send a message via Telegram, push, email, etc.

        Phase 4: when an OpenClaw bridge with a live client is wired
        in, this method calls :meth:`OpenClawClient.send_message`
        directly. When the bridge is absent (``openclaw.enabled=false``
        or CLI not discoverable), falls back to the stub voice
        message so the user hears why nothing happened.
        """
        blocked = self._maybe_block(
            tool_name="messaging",
            goal=intent.raw_text,
            tool_args={
                "channel": intent.channel,
                "recipient": intent.recipient,
                "body_preview": (intent.body or "")[:120],
            },
        )
        if blocked is not None:
            return blocked

        # Bridge wired? Try real send.
        client = self._bridge.client if self._bridge is not None else None
        if client is not None:
            return await self._send_via_bridge(intent)

        return self._stub_response(
            capability="messaging",
            voice_message=(
                "I'd send that for you, but the gateway isn't connected yet."
            ),
            metadata={"channel": intent.channel},
        )

    async def _send_via_bridge(self, intent: MessagingIntent) -> DispatchResult:
        """Phase 4: real send-via-bridge path. Resolves recipient from
        the intent, then from notifications config (Telegram only),
        then fails with a clear voice message. Fail-open at every
        step — exceptions translate into voice messages, not crashes.
        """
        client = self._bridge.client                         # checked by caller
        channel = (intent.channel or "telegram").lower()
        recipient = self._resolve_recipient(intent, channel=channel)
        if not recipient:
            logger.info(
                "messaging dispatch failed: no recipient resolved "
                "(channel=%s, intent.recipient=%s)",
                channel, intent.recipient,
            )
            return DispatchResult(
                success=False,
                voice_message=(
                    "I don't know who to send that to. Set "
                    "TELEGRAM_USER_ID in your environment, or tell me "
                    "the recipient explicitly."
                ),
                error="no recipient resolved",
                metadata={"channel": channel, "stub": False},
            )
        body = (intent.body or "").strip()
        if not body:
            return DispatchResult(
                success=False,
                voice_message="I need a message body to send.",
                error="empty body",
                metadata={"channel": channel, "stub": False},
            )
        try:
            send_result = await client.send_message(channel, recipient, body)
        except Exception as e:                              # noqa: BLE001
            logger.warning("send_message raised: %s", e)
            return DispatchResult(
                success=False,
                voice_message=(
                    "I couldn't reach the gateway just now. "
                    "Try again in a moment."
                ),
                error=str(e),
                metadata={"channel": channel, "stub": False},
            )
        if send_result.delivered:
            preview = body if len(body) <= 60 else body[:57] + "..."
            logger.info(
                "messaging dispatch delivered to %s on %s "
                "(message_id=%s)",
                recipient, channel, send_result.message_id or "-",
            )
            return DispatchResult(
                success=True,
                voice_message=f"Sent: {preview}",
                metadata={
                    "channel": channel,
                    "stub": False,
                    "message_id": send_result.message_id,
                },
            )
        return DispatchResult(
            success=False,
            voice_message=(
                f"The {channel} channel returned an error. "
                f"{send_result.error or ''}".strip()
            ),
            error=send_result.error or "send failed",
            metadata={"channel": channel, "stub": False},
        )

    def _resolve_recipient(
        self,
        intent: MessagingIntent,
        *,
        channel: str,
    ) -> str:
        """Resolve a recipient id for the configured channel.

        Order: explicit ``intent.recipient`` → notifications config
        (Telegram-only currently) → empty string.
        """
        if intent.recipient and intent.recipient.strip():
            return intent.recipient.strip()
        if channel == "telegram":
            tcfg = self._cfg.notifications.telegram
            env_value = os.environ.get(tcfg.user_id_env, "").strip()
            if env_value:
                return env_value
            if tcfg.fallback_user_id:
                return str(tcfg.fallback_user_id).strip()
        return ""

    async def handle_file_operation(self, intent: FileOpIntent) -> DispatchResult:
        """Filesystem operations outside the project sandbox."""
        blocked = self._maybe_block(
            tool_name="file_operation",
            goal=intent.raw_text,
            tool_args={"operation": intent.operation, "path": intent.path},
        )
        if blocked is not None:
            return blocked
        return self._stub_response(
            capability="file_operations",
            voice_message=(
                "I can't reach files outside the project sandbox yet."
            ),
            metadata={"operation": intent.operation, "path": intent.path},
        )

    async def handle_shell_operation(self, intent: ShellOpIntent) -> DispatchResult:
        """Shell command execution via OpenClaw exec tool."""
        blocked = self._maybe_block(
            tool_name="shell_operation",
            goal=intent.raw_text,
            tool_args={"command_preview": (intent.command or "")[:120]},
        )
        if blocked is not None:
            return blocked
        return self._stub_response(
            capability="shell_operations",
            voice_message="I can't run shell commands yet.",
            metadata={"command_preview": intent.command[:60]},
        )

    # --- V1-gap A1: gaming mode --------------------------------------------

    async def handle_gaming_mode(self, intent: GamingModeIntent) -> DispatchResult:
        """Engage / disengage / status for gaming mode.

        ``engage`` -> shut down configured anticheat-sensitive plugins.
        ``disengage`` -> restore them.
        ``status`` -> report whether gaming mode is currently on.

        The block-and-revise validator runs as for other handlers so
        we never accidentally engage / disengage on a tool call that
        doesn't match the user's stated goal.
        """
        # Pre-flight gating still applies.
        blocked = self._maybe_block(
            tool_name="gaming_mode",
            goal=intent.raw_text,
            tool_args={
                "action": intent.action,
                "trigger_phrase": intent.trigger_phrase,
            },
        )
        if blocked is not None:
            return blocked

        if self._gaming_mode_manager is None:
            return DispatchResult(
                success=False,
                voice_message=(
                    "Gaming mode isn't ready -- the OpenClaw bridge "
                    "isn't reachable. Make sure OpenClaw is running."
                ),
                error="no gaming_mode_manager",
                metadata={"action": intent.action, "stub": True},
            )

        action = (intent.action or "").lower().strip()
        try:
            if action == "engage":
                report = await self._gaming_mode_manager.engage()
                if report.note == "already engaged":
                    voice = "Gaming mode is already on. Have fun."
                elif report.all_plugin_actions_succeeded:
                    voice = "Shutting down desktop control. Have fun."
                else:
                    voice = (
                        "Gaming mode engaged with errors -- some plugins "
                        "didn't disable cleanly. Check logs/gaming_mode.jsonl."
                    )
            elif action == "disengage":
                report = await self._gaming_mode_manager.disengage()
                if report.note == "already idle":
                    voice = "Gaming mode wasn't on."
                elif report.all_plugin_actions_succeeded:
                    voice = "Full control restored."
                else:
                    voice = (
                        "Tried to restore desktop control but some plugins "
                        "didn't come back cleanly. Check logs/gaming_mode.jsonl."
                    )
            elif action == "status":
                from ultron.openclaw_routing.gaming_mode import GamingModeStatus
                status = self._gaming_mode_manager.status()
                if status == GamingModeStatus.ENGAGED:
                    voice = "Gaming mode is on."
                elif status == GamingModeStatus.IDLE:
                    voice = "Gaming mode is off."
                else:
                    voice = "Gaming mode is in the middle of changing state."
                return DispatchResult(
                    success=True,
                    voice_message=voice,
                    metadata={"action": "status", "status": status.value},
                )
            else:
                return DispatchResult(
                    success=False,
                    voice_message=f"I don't know how to {action!r} gaming mode.",
                    error=f"unknown gaming-mode action: {action!r}",
                    metadata={"action": action},
                )
        except Exception as e:                                       # noqa: BLE001
            logger.warning("gaming_mode dispatch raised: %s", e)
            return DispatchResult(
                success=False,
                voice_message=(
                    "I tried to change gaming mode but something went "
                    "wrong on the OpenClaw side. Try again in a moment."
                ),
                error=str(e)[:300],
                metadata={"action": action},
            )
        return DispatchResult(
            success=report.all_plugin_actions_succeeded,
            voice_message=voice,
            metadata={
                "action": action,
                "status": report.status.value,
                "plugin_states": [
                    {"id": p.plugin_id, "ok": p.success, "error": p.error}
                    for p in report.plugin_states
                ],
                "docker_acted": report.docker_acted,
            },
        )

    # --- V1-gap C3: desktop / windows automation ---------------------------

    async def handle_desktop_automation(self, intent: DesktopIntent) -> DispatchResult:
        """Voice routing for the OpenClaw ``desktop-control`` plugin.

        When gaming mode is engaged, returns a short-circuit message so
        the user understands why their request can't run -- the plugins
        the desktop tool needs are intentionally disabled.
        """
        blocked = self._maybe_block(
            tool_name="desktop_automation",
            goal=intent.raw_text,
            tool_args={"action": intent.action, "target": intent.target},
        )
        if blocked is not None:
            return blocked

        gaming_block = self._block_for_gaming_mode("desktop control")
        if gaming_block is not None:
            return gaming_block

        client = self._bridge.client if self._bridge is not None else None
        cfg = getattr(self._cfg, "desktop", None)
        if client is None or cfg is None or not getattr(cfg, "enabled", False):
            return self._stub_response(
                capability="desktop_automation",
                voice_message=(
                    "Desktop control isn't reachable. Confirm OpenClaw "
                    "is running and the desktop-control plugin is "
                    "installed and enabled."
                ),
                metadata={"action": intent.action, "target": intent.target},
            )
        return await self._desktop_via_bridge(intent, cfg)

    async def _desktop_via_bridge(
        self, intent: DesktopIntent, cfg,
    ) -> DispatchResult:
        from ultron.openclaw_bridge.desktop import DesktopTool

        client = self._bridge.client                                   # checked
        agent_id = self._cfg.openclaw.required_agent_id or "ultron-main"
        tool = DesktopTool(
            client,
            agent_id=agent_id,
            screenshot_tool=cfg.tool_slug_screenshot,
            list_windows_tool=cfg.tool_slug_list_windows,
            find_window_tool=cfg.tool_slug_find_window,
        )
        action = (intent.action or "").lower()
        try:
            if action == "screenshot":
                result = await tool.screenshot(
                    intent.target, timeout_s=cfg.default_screenshot_timeout_seconds,
                )
                if not result.success:
                    return DispatchResult(
                        success=False,
                        voice_message="I couldn't capture the screen.",
                        error=result.error,
                        metadata={"action": "screenshot", "stub": False},
                    )
                where = result.image_path or "the configured location"
                voice = f"Screenshot captured -- saved to {where}." if result.image_path else "Screenshot captured."
                return DispatchResult(
                    success=True,
                    voice_message=voice,
                    metadata={
                        "action": "screenshot", "stub": False,
                        "image_path": result.image_path,
                    },
                )
            if action == "list_windows":
                result = await tool.list_windows(
                    timeout_s=cfg.default_action_timeout_seconds,
                )
                if not result.success:
                    return DispatchResult(
                        success=False,
                        voice_message="I couldn't list the open windows.",
                        error=result.error,
                        metadata={"action": "list_windows", "stub": False},
                    )
                count = len(result.windows)
                if count == 0:
                    voice = "No open windows found."
                elif count <= 3:
                    titles = ", ".join(w.title for w in result.windows if w.title)
                    voice = f"{count} windows open: {titles}." if titles else f"{count} windows open."
                else:
                    voice = f"{count} windows are open. The list is in the metadata."
                return DispatchResult(
                    success=True,
                    voice_message=voice,
                    metadata={
                        "action": "list_windows", "stub": False,
                        "count": count,
                        "windows": [
                            {"title": w.title, "handle": w.handle, "app": w.app_name}
                            for w in result.windows
                        ],
                    },
                )
            if action == "find_window":
                if not intent.target:
                    return DispatchResult(
                        success=False,
                        voice_message="I need a window name to look up.",
                        error="missing target",
                        metadata={"action": "find_window", "stub": False},
                    )
                result = await tool.find_window(
                    intent.target,
                    timeout_s=cfg.default_action_timeout_seconds,
                )
                if not result.success:
                    return DispatchResult(
                        success=False,
                        voice_message=f"I couldn't find a {intent.target} window.",
                        error=result.error,
                        metadata={"action": "find_window", "stub": False},
                    )
                voice = (
                    f"Found {result.title or intent.target}."
                    if result.title else "Window found."
                )
                return DispatchResult(
                    success=True,
                    voice_message=voice,
                    metadata={
                        "action": "find_window", "stub": False,
                        "handle": result.handle, "title": result.title,
                        "app": result.app_name,
                    },
                )
            return DispatchResult(
                success=False,
                voice_message=f"I don't know how to {action} the desktop.",
                error=f"unknown action: {action}",
                metadata={"action": action, "stub": False},
            )
        except Exception as e:                                       # noqa: BLE001
            logger.warning("desktop dispatch raised: %s", e)
            return DispatchResult(
                success=False,
                voice_message=(
                    "Something went wrong on the desktop side. "
                    "Try again in a moment."
                ),
                error=str(e)[:300],
                metadata={"action": action, "stub": False},
            )

    async def handle_window_automation(self, intent: WindowIntent) -> DispatchResult:
        """Voice routing for the OpenClaw ``windows-control`` plugin."""
        blocked = self._maybe_block(
            tool_name="window_automation",
            goal=intent.raw_text,
            tool_args={
                "action": intent.action, "query": intent.query,
                "ref": intent.ref, "value_preview": (intent.value or "")[:120],
            },
        )
        if blocked is not None:
            return blocked

        gaming_block = self._block_for_gaming_mode("window control")
        if gaming_block is not None:
            return gaming_block

        client = self._bridge.client if self._bridge is not None else None
        cfg = getattr(self._cfg, "window_control", None)
        if client is None or cfg is None or not getattr(cfg, "enabled", False):
            return self._stub_response(
                capability="window_automation",
                voice_message=(
                    "Window control isn't reachable. Confirm OpenClaw "
                    "is running and the windows-control plugin is "
                    "installed and enabled."
                ),
                metadata={"action": intent.action, "query": intent.query},
            )
        return await self._window_via_bridge(intent, cfg)

    async def _window_via_bridge(
        self, intent: WindowIntent, cfg,
    ) -> DispatchResult:
        from ultron.openclaw_bridge.desktop import WindowControlTool

        client = self._bridge.client                                   # checked
        agent_id = self._cfg.openclaw.required_agent_id or "ultron-main"
        tool = WindowControlTool(
            client,
            agent_id=agent_id,
            focus_tool=cfg.tool_slug_focus,
            click_tool=cfg.tool_slug_click,
            type_tool=cfg.tool_slug_type,
        )
        action = (intent.action or "").lower()
        try:
            if action == "focus":
                if not intent.query:
                    return DispatchResult(
                        success=False,
                        voice_message="I need a window name to focus.",
                        error="missing query",
                        metadata={"action": "focus", "stub": False},
                    )
                result = await tool.focus(
                    intent.query,
                    timeout_s=cfg.default_action_timeout_seconds,
                )
                voice = (
                    f"Focused {intent.query}." if result.success
                    else f"I couldn't focus the {intent.query} window."
                )
                return DispatchResult(
                    success=result.success, voice_message=voice,
                    error=result.error,
                    metadata={"action": "focus", "stub": False, "query": intent.query},
                )
            if action == "click":
                ref = intent.ref or intent.query or ""
                if not ref:
                    return DispatchResult(
                        success=False,
                        voice_message="I don't know what to click.",
                        error="missing ref",
                        metadata={"action": "click", "stub": False},
                    )
                result = await tool.click(
                    ref, timeout_s=cfg.default_action_timeout_seconds,
                )
                voice = "Clicked." if result.success else "I couldn't click that."
                return DispatchResult(
                    success=result.success, voice_message=voice,
                    error=result.error,
                    metadata={"action": "click", "stub": False, "ref": ref},
                )
            if action == "type":
                if not intent.query and not intent.ref:
                    return DispatchResult(
                        success=False,
                        voice_message=(
                            "I need a target window or field to type into."
                        ),
                        error="missing ref / query",
                        metadata={"action": "type", "stub": False},
                    )
                result = await tool.type_text(
                    intent.ref or intent.query, intent.value or "",
                    timeout_s=cfg.default_action_timeout_seconds,
                )
                voice = "Typed." if result.success else "I couldn't type that."
                return DispatchResult(
                    success=result.success, voice_message=voice,
                    error=result.error,
                    metadata={
                        "action": "type", "stub": False,
                        "target": intent.ref or intent.query,
                    },
                )
            return DispatchResult(
                success=False,
                voice_message=f"I don't know how to {action} a window.",
                error=f"unknown action: {action}",
                metadata={"action": action, "stub": False},
            )
        except Exception as e:                                       # noqa: BLE001
            logger.warning("window dispatch raised: %s", e)
            return DispatchResult(
                success=False,
                voice_message=(
                    "Something went wrong on the window-control side. "
                    "Try again in a moment."
                ),
                error=str(e)[:300],
                metadata={"action": action, "stub": False},
            )

    def _block_for_gaming_mode(self, capability_label: str) -> Optional[DispatchResult]:
        """Short-circuit a desktop / window action when gaming mode is on.

        Without this, the user would get a confusing "tool unavailable"
        error for plugins that ARE installed but were intentionally
        disabled by the gaming-mode manager. The voice message is in
        Ultron's voice and includes the recovery action.
        """
        if self._gaming_mode_manager is None:
            return None
        try:
            from ultron.openclaw_routing.gaming_mode import GamingModeStatus
            status = self._gaming_mode_manager.status()
        except Exception:
            return None
        if status != GamingModeStatus.ENGAGED:
            return None
        return DispatchResult(
            success=False,
            voice_message=(
                f"Gaming mode is on. {capability_label.capitalize()} is "
                f"disabled. Say 'gaming mode off' to restore it."
            ),
            error="gaming mode engaged",
            metadata={"gaming_mode": "engaged"},
        )

    # --- 4B plan Item 8: block-and-revise pre-flight ------------------------

    def _maybe_block(
        self,
        *,
        tool_name: str,
        goal: str,
        tool_args: Dict[str, Any],
    ) -> Optional[DispatchResult]:
        """Two-layer pre-flight check.

        Layer 1 (2026-05-12 -- new): the runtime tool-call validator
        (rule-based, paired with the abliterated default LLM). Hard
        block on any rule verdict that isn't ``ALLOW`` or ``LOG_ONLY``.
        Construction failures and rule exceptions fail closed inside
        the validator itself; this method's contract is "returns None
        when the call should proceed, returns a DispatchResult to
        short-circuit when blocked".

        Layer 2 (4B plan Item 8 -- existing): the LLM-based block-and-
        revise validator. Only runs when layer 1 allowed; provides a
        soft check that the tool call advances the user's stated goal.
        Requires an LLM to be wired; falls open when missing.

        Failure-safe: any exception inside either layer falls open
        (returns None), preserving the dispatch path's behaviour.
        """
        # ----- Layer 1: runtime tool-call validator (Category K et al.) -----
        runtime_block = self._runtime_safety_check(
            tool_name=tool_name, goal=goal, tool_args=tool_args,
        )
        if runtime_block is not None:
            return runtime_block

        # ----- Layer 2: LLM-based block-and-revise -----
        try:
            from ultron.openclaw_routing.block_and_revise import (
                ToolCallValidator, is_enabled,
            )
            if not is_enabled(self._cfg):
                return None
            if self._llm is None:
                return None
            validator = ToolCallValidator(self._llm)
            result = validator.validate(
                goal=goal or "(no stated goal)",
                tool_name=tool_name,
                tool_args=tool_args,
            )
        except Exception as e:
            logger.warning("block-and-revise check failed: %s", e)
            return None

        if result.allow:
            return None

        logger.info(
            "block-and-revise blocked %s call: %s",
            tool_name, result.reason,
        )
        return DispatchResult(
            success=False,
            voice_message=(
                f"I held off on that — {result.reason}"
            ),
            error="blocked by block-and-revise validator",
            metadata={
                "blocked": True,
                "tool_name": tool_name,
                "verdict": result.verdict,
                "reason": result.reason,
            },
        )

    def _runtime_safety_check(
        self,
        *,
        tool_name: str,
        goal: str,
        tool_args: Dict[str, Any],
    ) -> Optional[DispatchResult]:
        """Layer 1 of :meth:`_maybe_block`: runtime rule-based validator.

        Builds a :class:`RuleContext` from the dispatcher's arguments
        and runs every registered rule. Returns a short-circuit
        :class:`DispatchResult` when the aggregated verdict is anything
        other than ALLOW / LOG_ONLY. Returns None on ALLOW / LOG_ONLY
        (proceed) and on any validator-side exception (fail-open to
        preserve dispatch availability).

        Path arguments in ``tool_args`` are extracted and pre-
        canonicalised so each rule doesn't re-resolve. The keys we
        recognise: ``path``, ``paths``, ``file``, ``files``,
        ``target_path``, ``destination``. Add more here when new
        intent shapes are added.
        """
        try:
            from ultron.safety import (
                RuleContext, Verdict, get_validator,
            )
            from ultron.safety.path_resolver import (
                PathResolveError, get_path_resolver,
            )
        except Exception as e:
            logger.debug(
                "runtime safety validator unavailable (%s); falling open",
                e,
            )
            return None

        validator = get_validator()
        # The no-op validator's check() returns ALLOW; the production
        # validator returns rule-based verdicts. Either way the call
        # path below handles it.

        # Extract path-shaped arguments. Keep raw originals around so
        # rules can re-inspect if needed.
        candidate_paths: list = []
        resolver = get_path_resolver()
        for key in (
            "path", "paths", "file", "files",
            "target_path", "destination", "dest",
        ):
            v = tool_args.get(key)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, str) and item:
                        try:
                            candidate_paths.append(resolver.resolve(item))
                        except PathResolveError:
                            # An unresolvable path is suspicious. Let
                            # the rules decide -- pass through the raw
                            # string so they can flag it.
                            candidate_paths.append(item)
            elif isinstance(v, str) and v:
                try:
                    candidate_paths.append(resolver.resolve(v))
                except PathResolveError:
                    candidate_paths.append(v)

        ctx = RuleContext(
            tool_name=f"openclaw.{tool_name}",
            arguments=dict(tool_args),
            capability="openclaw_dispatcher",
            paths=tuple(candidate_paths),
            user_text=goal or "",
            has_pending_clarification=False,
        )
        try:
            verdict = validator.check(ctx)
        except Exception as e:
            logger.warning(
                "runtime safety validator crashed (%s); failing open "
                "for this call only", e,
            )
            return None

        if verdict.is_allowed:
            return None

        logger.info(
            "runtime safety validator blocked %s call: rule=%s reason=%s",
            tool_name, verdict.triggered_rule_id, verdict.reason,
        )
        return DispatchResult(
            success=False,
            voice_message=(
                verdict.user_message
                or f"I held off on that — {verdict.reason}"
            ),
            error="blocked by runtime safety validator",
            metadata={
                "blocked": True,
                "blocked_by": "safety_validator",
                "tool_name": tool_name,
                "verdict": verdict.verdict.value,
                "rule_id": verdict.triggered_rule_id,
                "reason": verdict.reason,
            },
        )

    # --- internal ----------------------------------------------------------

    def _stub_response(
        self,
        *,
        capability: str,
        voice_message: str,
        metadata: Dict[str, Any] | None = None,
    ) -> DispatchResult:
        """Build the canonical "not yet integrated" stub. The OpenClaw
        integration prompt replaces these per-capability handlers; the
        helper stays for tests and operator dry-runs."""
        meta = {"stub": True, "capability": capability}
        if metadata:
            meta.update(metadata)
        logger.info(
            "OpenClawDispatcher stub: capability=%s (gateway enabled=%s)",
            capability, self.enabled,
        )
        return DispatchResult(
            success=False,
            voice_message=voice_message,
            error=(
                f"{capability} not yet integrated; available after "
                f"OpenClaw integration phase"
            ),
            metadata=meta,
        )


__all__ = ["OpenClawDispatcher"]
