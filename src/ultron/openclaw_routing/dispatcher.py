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
    DispatchResult,
    FileOpIntent,
    MediaGenIntent,
    MessagingIntent,
    ShellOpIntent,
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

    # --- 4B plan Item 8: block-and-revise pre-flight ------------------------

    def _maybe_block(
        self,
        *,
        tool_name: str,
        goal: str,
        tool_args: Dict[str, Any],
    ) -> Optional[DispatchResult]:
        """Run the block-and-revise validator. Returns ``None`` when the
        call should proceed (validator allows OR feature is disabled OR
        no LLM is wired). Returns a ``DispatchResult`` to short-circuit
        when the validator BLOCKs.

        Failure-safe: any exception inside the validator path falls
        open (returns None), preserving the dispatch path's behaviour.
        """
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
