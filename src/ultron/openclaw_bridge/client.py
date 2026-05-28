"""OpenClaw Gateway client (Phase 3.1).

Async bridge from Ultron's orchestrator to OpenClaw, used when Ultron
wants OpenClaw to do something — send a Telegram message, run a tool
turn, trigger a heartbeat. Implemented with subprocess transport over
the ``openclaw`` CLI rather than HTTP because OpenClaw 2026.5.7 doesn't
expose ``/tools/invoke`` or ``/messages`` HTTP endpoints. The CLI is
the documented public interface and the only stable contract.

Critical contract (matches lifecycle.py): every public method returns a
typed result on success and a typed-error-bearing result OR raises a
typed ``OpenClawGatewayError``/``OpenClawAuthError``/``OpenClawToolError``
on failure. The voice path must keep working when the Gateway is down,
so callers wrap invocations in try/except and degrade gracefully.

Auth model: bearer token from ``~/.openclaw/openclaw.json``. Reads via
:class:`OpenClawLifecycle._read_token` semantics — never logged. The
token is forwarded to the CLI via ``--token`` only when an explicit
override is required; the CLI defaults to reading the same file.

Threading: methods are async and run subprocesses via
``asyncio.create_subprocess_exec``. Each call is fully isolated — the
client holds no per-call state. Cheap to construct; one instance can
service the whole orchestrator lifetime.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ultron.errors import OpenClawAuthError, OpenClawGatewayError, OpenClawToolError
from ultron.openclaw_bridge.lifecycle import _read_token  # token reader reused
from ultron.subprocess.kill_tree import kill_process_tree
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.client")


# Windows uses openclaw.cmd (npm shim); POSIX uses plain openclaw.
_DEFAULT_CLI_NAME = "openclaw.cmd" if sys.platform == "win32" else "openclaw"

# Common Windows install location — npm-global shim. Tried after PATH lookup.
_WINDOWS_DEFAULT_CLI = (
    Path(os.environ.get("APPDATA", ""))
    / "npm"
    / "openclaw.cmd"
) if sys.platform == "win32" else None


def discover_cli(override: Optional[str] = None) -> str:
    """Resolve the ``openclaw`` CLI executable.

    Order: explicit ``override`` → ``ULTRON_OPENCLAW_CLI`` env var →
    PATH lookup → Windows npm-global default. Raises
    :class:`OpenClawGatewayError` if none of those produce an existing
    file.
    """
    if override:
        p = Path(override)
        if p.exists():
            return str(p.resolve())
        raise OpenClawGatewayError(
            f"openclaw CLI override path does not exist: {override}",
            context={"cli_path": override},
        )
    env = os.getenv("ULTRON_OPENCLAW_CLI")
    if env:
        p = Path(env)
        if p.exists():
            return str(p.resolve())
        # Fall through — env var was wrong, keep trying.
        logger.warning(
            "ULTRON_OPENCLAW_CLI=%s does not exist; falling back to PATH",
            env,
        )
    found = shutil.which(_DEFAULT_CLI_NAME)
    if found:
        return found
    if _WINDOWS_DEFAULT_CLI is not None and _WINDOWS_DEFAULT_CLI.exists():
        return str(_WINDOWS_DEFAULT_CLI)
    raise OpenClawGatewayError(
        "openclaw CLI not found. Install it (npm i -g openclaw) or set "
        "ULTRON_OPENCLAW_CLI to the absolute path.",
        context={"searched_name": _DEFAULT_CLI_NAME},
    )


@dataclass(frozen=True)
class CliResult:
    """Raw outcome of one CLI invocation. Helper return type for
    :meth:`OpenClawClient._run_cli`; public methods return their own
    typed dataclasses."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float


@dataclass(frozen=True)
class SendMessageResult:
    """Outcome of ``OpenClawClient.send_message``."""

    delivered: bool
    channel: str
    target: str
    message_id: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class HeartbeatResult:
    """Outcome of ``OpenClawClient.trigger_heartbeat``."""

    triggered: bool
    final_text: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class AgentRunResult:
    """Outcome of ``OpenClawClient.run_agent``. Used as the underlying
    payload for tool invocations — OpenClaw tools fire as part of an
    agent turn, not via standalone API calls."""

    success: bool
    agent_id: str
    text: str = ""                                         # final user-visible text
    raw: Optional[Dict[str, Any]] = None
    duration_s: float = 0.0
    error: Optional[str] = None


@dataclass(frozen=True)
class PluginToggleResult:
    """Outcome of an enable_plugin / disable_plugin call (A1 / C3)."""

    plugin_id: str
    action: str  # "enable" | "disable"
    success: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class PluginInfo:
    """One row from ``openclaw plugins list --json`` (A1 / C3)."""

    plugin_id: str
    name: str
    enabled: bool
    version: str = ""


@dataclass(frozen=True)
class ToolInvocationResult:
    """Convenience wrapper around :class:`AgentRunResult` for callers
    that explicitly want a tool result (browser, image-gen, etc.)."""

    success: bool
    tool_name: str
    text: str = ""
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OpenClawClient:
    """Async client for OpenClaw Gateway via the ``openclaw`` CLI.

    Methods are fail-open in spirit but explicit in shape: they return
    success/failure dataclasses on completion and raise typed errors
    only when the CLI itself can't be invoked. Callers wrap
    invocations in try/except and degrade gracefully when ``error``
    is set or an exception bubbles up.

    Args:
        cli_path: explicit override for the ``openclaw`` executable.
            ``None`` triggers :func:`discover_cli` at construction.
        default_timeout_s: applied when a per-call timeout isn't given.
        config_path: location of OpenClaw's user config (used by the
            CLI to read the auth token; we never inject it manually).
        default_agent_id: default ``--agent`` for ``run_agent`` /
            ``invoke_tool`` when caller doesn't specify one.
        env: extra environment variables passed to subprocesses.
            Useful when tests need to override e.g. ``OPENCLAW_HOME``.
    """

    def __init__(
        self,
        cli_path: Optional[str] = None,
        *,
        default_timeout_s: float = 30.0,
        config_path: Optional[Path] = None,
        default_agent_id: str = "ultron-main",
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._cli_path = discover_cli(cli_path)
        self._default_timeout_s = default_timeout_s
        self._config_path = (
            Path(config_path)
            if config_path is not None
            else Path.home() / ".openclaw" / "openclaw.json"
        )
        self._default_agent_id = default_agent_id
        self._env_overrides: Dict[str, str] = dict(env or {})

    @property
    def cli_path(self) -> str:
        return self._cli_path

    @property
    def auth_token(self) -> Optional[str]:
        """Mirror of :class:`OpenClawLifecycle.auth_token`. Re-read on
        every access so a token rotation lands without restart. Never
        logged."""
        return _read_token(self._config_path)

    # -------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------

    async def health(self, timeout_s: Optional[float] = None) -> bool:
        """Return True iff the Gateway responds healthy.

        Wraps ``openclaw health --json``. Fail-open: any error returns
        False. Never raises.
        """
        try:
            result = await self._run_cli(
                ["health", "--json"],
                timeout_s=timeout_s if timeout_s is not None else 5.0,
            )
        except OpenClawGatewayError as e:
            logger.debug("health probe failed: %s", e)
            return False
        if result.returncode != 0:
            logger.debug(
                "health probe non-zero (%d): %s",
                result.returncode, result.stderr.strip()[:200],
            )
            return False
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return False
        # OpenClaw `health --json` shape: top-level ``ok`` boolean OR
        # ``status: "healthy"`` depending on version. Accept both.
        if isinstance(payload, dict):
            if payload.get("ok") is True:
                return True
            status = payload.get("status")
            if isinstance(status, str) and status.lower() in {"healthy", "ok", "running"}:
                return True
        return False

    # -------------------------------------------------------------------
    # Send message (Phase 4: Telegram et al.)
    # -------------------------------------------------------------------

    async def send_message(
        self,
        channel: str,
        target: str,
        text: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> SendMessageResult:
        """Send a message via a configured OpenClaw channel.

        Wraps ``openclaw message send --channel <channel> --target <target>
        --message <text> --json``. Used for proactive notifications
        (coding-task completion, search result delivery, heartbeat
        alerts, etc.) — Phase 4 wires the Telegram channel.

        Returns a :class:`SendMessageResult` whose ``delivered`` field
        signals success. On any failure, ``delivered=False`` and
        ``error`` carries a short human-readable reason. Callers
        degrade gracefully and never re-raise.
        """
        if not text.strip():
            return SendMessageResult(
                delivered=False, channel=channel, target=target,
                error="empty message text",
            )
        args = [
            "message", "send",
            "--channel", channel,
            "--target", target,
            "--message", text,
            "--json",
        ]
        try:
            result = await self._run_cli(args, timeout_s=timeout_s)
        except OpenClawGatewayError as e:
            return SendMessageResult(
                delivered=False, channel=channel, target=target,
                error=str(e),
            )
        if result.returncode != 0:
            err_text = result.stderr.strip()[:300] or result.stdout.strip()[:300]
            self._raise_for_auth_failure(err_text, context={
                "op": "send_message", "channel": channel,
            })
            return SendMessageResult(
                delivered=False, channel=channel, target=target,
                error=err_text or f"exit {result.returncode}",
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = None
        message_id = None
        if isinstance(payload, dict):
            message_id = (
                payload.get("messageId")
                or payload.get("message_id")
                or payload.get("id")
            )
        return SendMessageResult(
            delivered=True, channel=channel, target=target,
            message_id=str(message_id) if message_id else None,
            raw=payload if isinstance(payload, dict) else None,
        )

    # -------------------------------------------------------------------
    # Trigger heartbeat (Phase 5)
    # -------------------------------------------------------------------

    async def trigger_heartbeat(
        self,
        text: Optional[str] = None,
        *,
        mode: str = "now",
        expect_final: bool = False,
        timeout_s: Optional[float] = None,
    ) -> HeartbeatResult:
        """Enqueue a system event and (optionally) trigger a heartbeat.

        Wraps ``openclaw system event --text <text> --mode <mode>``. The
        ``mode`` field controls when the heartbeat fires: ``"now"`` runs
        immediately; ``"next-heartbeat"`` queues it for the next
        scheduled tick.

        ``expect_final=True`` waits for the agent's final response and
        returns it in :attr:`HeartbeatResult.final_text`. Useful for
        synchronous "ping the agent now and tell me what it said"
        flows; defaults False because the typical use case is
        fire-and-forget.
        """
        args = ["system", "event", "--mode", mode, "--json"]
        if text:
            args.extend(["--text", text])
        if expect_final:
            args.append("--expect-final")
        if timeout_s is not None:
            args.extend(["--timeout", str(int(timeout_s * 1000))])
        try:
            # Heartbeat with --expect-final can take longer than the
            # default; honour caller's timeout, with a bit of slack.
            cli_timeout = (
                timeout_s + 5.0 if timeout_s is not None else None
            )
            result = await self._run_cli(args, timeout_s=cli_timeout)
        except OpenClawGatewayError as e:
            return HeartbeatResult(triggered=False, error=str(e))
        if result.returncode != 0:
            err_text = result.stderr.strip()[:300] or result.stdout.strip()[:300]
            self._raise_for_auth_failure(err_text, context={
                "op": "trigger_heartbeat",
            })
            return HeartbeatResult(triggered=False, error=err_text)
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else None
        except json.JSONDecodeError:
            payload = None
        final_text = None
        if isinstance(payload, dict):
            for key in ("finalText", "final_text", "text", "reply"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    final_text = value
                    break
        return HeartbeatResult(
            triggered=True,
            final_text=final_text,
            raw=payload if isinstance(payload, dict) else None,
        )

    # -------------------------------------------------------------------
    # Run agent / invoke tool (Phase 6 et al.)
    # -------------------------------------------------------------------

    async def run_agent(
        self,
        message: str,
        *,
        agent_id: Optional[str] = None,
        thinking: Optional[str] = None,
        deliver: bool = False,
        reply_channel: Optional[str] = None,
        reply_to: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> AgentRunResult:
        """Run one agent turn via the Gateway.

        Wraps ``openclaw agent --agent <id> --message <text> --json``
        with optional delivery flags. Used as the underlying primitive
        for :meth:`invoke_tool` and any future hand-offs that need an
        agent turn (cron prompts, standing orders triggered out-of-band).

        ``deliver=True`` instructs OpenClaw to send the agent's reply
        back through ``reply_channel``/``reply_to`` (or the routed
        session channel if those are omitted). Defaults False because
        most orchestrator-side calls want to consume the agent's output
        in-process, not push it back out to a channel.
        """
        target_agent = agent_id or self._default_agent_id
        args: List[str] = [
            "agent",
            "--agent", target_agent,
            "--message", message,
            "--json",
        ]
        if thinking:
            args.extend(["--thinking", thinking])
        if deliver:
            args.append("--deliver")
            if reply_channel:
                args.extend(["--reply-channel", reply_channel])
            if reply_to:
                args.extend(["--reply-to", reply_to])
        try:
            result = await self._run_cli(args, timeout_s=timeout_s)
        except OpenClawGatewayError as e:
            return AgentRunResult(
                success=False, agent_id=target_agent, error=str(e),
            )
        if result.returncode != 0:
            err_text = result.stderr.strip()[:300] or result.stdout.strip()[:300]
            self._raise_for_auth_failure(err_text, context={
                "op": "run_agent", "agent_id": target_agent,
            })
            return AgentRunResult(
                success=False, agent_id=target_agent,
                duration_s=result.duration_s, error=err_text,
            )
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else None
        except json.JSONDecodeError:
            payload = None
        text = ""
        if isinstance(payload, dict):
            for key in ("text", "reply", "finalText", "final_text", "content"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
        return AgentRunResult(
            success=True, agent_id=target_agent,
            text=text, raw=payload if isinstance(payload, dict) else None,
            duration_s=result.duration_s,
        )

    async def invoke_tool(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        agent_id: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> ToolInvocationResult:
        """Ask an OpenClaw agent to use a specific tool.

        OpenClaw 2026.5.7 doesn't expose direct tool invocation via the
        public CLI — tools fire as part of an agent turn. This method
        constructs a structured prompt asking the configured agent to
        use ``tool_name`` with ``params``, runs the turn, and unpacks
        the result.

        ``params`` is rendered as ``key=value`` pairs in the prompt.
        Callers that need richer structure should call
        :meth:`run_agent` directly with a hand-crafted message.

        Raises :class:`OpenClawToolError` only when the Gateway returns
        a tool-layer failure (i.e. the agent ran but the tool errored).
        Transport failures bubble up as
        :class:`OpenClawGatewayError`.
        """
        prompt_parts = [f"Use the {tool_name} tool"]
        params = params or {}
        if params:
            rendered = ", ".join(f"{k}={v!r}" for k, v in params.items())
            prompt_parts.append(f"with parameters: {rendered}")
        prompt_parts.append(
            ". Return the tool's result text. If the tool is unavailable, "
            "say so explicitly and do not improvise an answer."
        )
        message = " ".join(prompt_parts)
        run = await self.run_agent(
            message,
            agent_id=agent_id,
            thinking="low",
            timeout_s=timeout_s,
        )
        if not run.success:
            return ToolInvocationResult(
                success=False, tool_name=tool_name,
                error=run.error,
            )
        # Heuristic: if the agent text starts with "tool unavailable" or
        # explicitly says the tool isn't installed, treat as a tool
        # failure (so callers can branch on it). Otherwise pass through.
        lowered = run.text.lower()
        if any(
            phrase in lowered
            for phrase in ("tool unavailable", "tool is not available", "no such tool")
        ):
            raise OpenClawToolError(
                f"OpenClaw tool '{tool_name}' returned an unavailable response",
                context={"tool_name": tool_name, "agent_text": run.text[:300]},
            )
        return ToolInvocationResult(
            success=True, tool_name=tool_name,
            text=run.text, raw=run.raw,
        )

    # -------------------------------------------------------------------
    # MCP configuration (stdio servers OpenClaw spawns on demand)
    # -------------------------------------------------------------------

    async def mcp_list(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """List configured MCP servers.

        Wraps ``openclaw mcp list``. Parses the human-readable output
        as a fallback because the CLI doesn't always emit JSON. Returns
        a mapping of server name → ``{"command", "args", "env"}``;
        empty dict when no servers are configured. Raises
        :class:`OpenClawGatewayError` on transport failure.
        """
        # Try JSON first if the CLI supports it; ``mcp show`` is stable
        # but ``mcp list`` historically outputs plain text. Gracefully
        # degrade to text parsing when needed.
        result = await self._run_cli(["mcp", "list"], timeout_s=timeout_s)
        if result.returncode != 0:
            err = result.stderr.strip()[:300]
            self._raise_for_auth_failure(err, context={"op": "mcp_list"})
            raise OpenClawGatewayError(
                f"openclaw mcp list failed: {err or 'exit ' + str(result.returncode)}",
                context={"returncode": result.returncode},
            )
        return self._parse_mcp_list(result.stdout)

    async def mcp_show(
        self,
        name: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the JSON record for one MCP server, or None if not
        configured. Wraps ``openclaw mcp show <name>``."""
        result = await self._run_cli(
            ["mcp", "show", name], timeout_s=timeout_s,
        )
        if result.returncode != 0:
            # `mcp show` exits non-zero when the entry is missing; treat
            # as None rather than raising.
            return None
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    async def mcp_set(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        *,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """Register an MCP server entry. Wraps ``openclaw mcp set``.

        ``command`` is the executable OpenClaw will spawn for stdio.
        It must be reachable from the Gateway's process environment —
        an absolute path is safest. Returns True on success; raises
        :class:`OpenClawGatewayError` on transport / config failures.
        Idempotent: re-running with the same payload silently succeeds.
        """
        if not name or not command:
            raise ValueError("mcp_set requires non-empty name and command")
        payload = {
            "command": command,
            "args": list(args or []),
            "env": dict(env or {}),
        }
        result = await self._run_cli(
            ["mcp", "set", name, json.dumps(payload)],
            timeout_s=timeout_s,
        )
        if result.returncode != 0:
            err = result.stderr.strip()[:300] or result.stdout.strip()[:300]
            self._raise_for_auth_failure(err, context={"op": "mcp_set"})
            raise OpenClawGatewayError(
                f"openclaw mcp set {name} failed: {err}",
                context={"name": name, "returncode": result.returncode},
            )
        return True

    async def mcp_unset(
        self,
        name: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """Remove an MCP server entry. Wraps ``openclaw mcp unset``.

        Returns True iff the unset succeeded. False when the entry
        wasn't configured (CLI returns non-zero but it's not an
        error from our perspective). Raises only on transport failure.
        """
        result = await self._run_cli(
            ["mcp", "unset", name], timeout_s=timeout_s,
        )
        if result.returncode == 0:
            return True
        err = result.stderr.strip().lower()
        # OpenClaw's CLI prints "no such" / "not configured" on missing
        # entries; treat those as no-op success rather than transport
        # failures so the caller can use this idempotently.
        if any(m in err for m in ("not configured", "no such", "not found")):
            return False
        self._raise_for_auth_failure(err, context={"op": "mcp_unset"})
        raise OpenClawGatewayError(
            f"openclaw mcp unset {name} failed: {err[:200]}",
            context={"name": name, "returncode": result.returncode},
        )

    # -------------------------------------------------------------------
    # Plugin management (A1 / C3 — gaming mode + desktop control)
    # -------------------------------------------------------------------

    async def enable_plugin(
        self,
        plugin_id: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> "PluginToggleResult":
        """Enable a plugin via ``openclaw plugins enable <id>``.

        Returns a :class:`PluginToggleResult` with ``success=True`` on
        successful CLI exit. Returns ``success=False`` with a populated
        ``error`` field for known failure modes (plugin not installed,
        plugin already in target state). Transport-level errors raise
        :class:`OpenClawGatewayError` so callers can distinguish
        infrastructural failure from operational outcomes.
        """
        return await self._toggle_plugin(plugin_id, "enable", timeout_s=timeout_s)

    async def disable_plugin(
        self,
        plugin_id: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> "PluginToggleResult":
        """Disable a plugin via ``openclaw plugins disable <id>``.

        Same semantics as :meth:`enable_plugin`. Used by the gaming-mode
        manager (A1) to take ``desktop-control`` / ``windows-control``
        offline before launching anticheat-protected games.
        """
        return await self._toggle_plugin(plugin_id, "disable", timeout_s=timeout_s)

    async def _toggle_plugin(
        self,
        plugin_id: str,
        action: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> "PluginToggleResult":
        if action not in ("enable", "disable"):
            raise ValueError(f"_toggle_plugin: invalid action {action!r}")
        if not plugin_id or not str(plugin_id).strip():
            raise ValueError("_toggle_plugin: plugin_id must be non-empty")
        result = await self._run_cli(
            ["plugins", action, plugin_id], timeout_s=timeout_s,
        )
        if result.returncode == 0:
            return PluginToggleResult(
                plugin_id=plugin_id, action=action, success=True,
            )
        err = result.stderr.strip()
        lowered = err.lower()
        # OpenClaw's CLI surfaces "not installed" / "unknown plugin"
        # cleanly on stderr; bubble those as structured failures so the
        # voice layer can say the right thing.
        if any(m in lowered for m in ("not installed", "unknown plugin", "no such")):
            return PluginToggleResult(
                plugin_id=plugin_id, action=action, success=False,
                error=f"plugin {plugin_id!r} is not installed on this OpenClaw",
            )
        # Auth failures escalate.
        self._raise_for_auth_failure(err, context={"op": f"plugins.{action}"})
        return PluginToggleResult(
            plugin_id=plugin_id, action=action, success=False,
            error=err[:300] or f"plugins {action} returned {result.returncode}",
        )

    async def list_plugins(
        self,
        *,
        enabled_only: bool = False,
        timeout_s: Optional[float] = None,
    ) -> List["PluginInfo"]:
        """List discovered plugins via ``openclaw plugins list --json``.

        Returns an empty list when the CLI fails or the output isn't
        parseable -- callers (gaming-mode status reporter) treat that
        as 'I don't know which plugins are installed'.
        """
        args = ["plugins", "list", "--json"]
        if enabled_only:
            args.append("--enabled")
        try:
            result = await self._run_cli(args, timeout_s=timeout_s)
        except OpenClawGatewayError as e:
            logger.warning("list_plugins transport failure: %s", e)
            return []
        if result.returncode != 0:
            logger.warning(
                "list_plugins returned %s: %s",
                result.returncode, result.stderr[:200],
            )
            return []
        text = result.stdout.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("list_plugins JSON parse failed: %s", e)
            return []
        rows: List[PluginInfo] = []
        seq = payload if isinstance(payload, list) else payload.get("plugins", [])
        for entry in (seq or []):
            if not isinstance(entry, dict):
                continue
            try:
                rows.append(PluginInfo(
                    plugin_id=str(entry.get("id") or entry.get("name") or ""),
                    name=str(entry.get("name") or ""),
                    enabled=bool(entry.get("enabled", entry.get("status") == "enabled")),
                    version=str(entry.get("version") or ""),
                ))
            except (TypeError, ValueError) as e:
                logger.debug("list_plugins skipping malformed row: %s", e)
                continue
        return rows

    @staticmethod
    def _parse_mcp_list(stdout: str) -> Dict[str, Dict[str, Any]]:
        """Best-effort parse of ``openclaw mcp list`` text output.

        Recognised forms:

        - JSON object with server names as keys.
        - Lines like ``<name>: <command> [args...]``.
        - The literal "No MCP servers configured ..." sentinel.
        """
        text = stdout.strip()
        if not text:
            return {}
        if "No MCP servers configured" in text:
            return {}
        # JSON path.
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            normalised: Dict[str, Dict[str, Any]] = {}
            for name, value in payload.items():
                if isinstance(value, dict):
                    normalised[str(name)] = value
            return normalised
        # Text path: lenient line-by-line parse. Each non-empty line
        # is treated as a server entry; we extract the leading token
        # before a colon as the name.
        result: Dict[str, Dict[str, Any]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                name, _, rest = line.partition(":")
                result[name.strip()] = {"raw": rest.strip()}
            else:
                # Lines without a colon — store as a name with no payload.
                result[line] = {}
        return result

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    async def _run_cli(
        self,
        args: Sequence[str],
        *,
        timeout_s: Optional[float] = None,
    ) -> CliResult:
        """Run the openclaw CLI with the given args, returning
        :class:`CliResult`. Raises :class:`OpenClawGatewayError` on
        execution failures (CLI not found, OS errors, timeout)."""
        cmd: List[str] = [self._cli_path, *args]
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        env = self._build_env()
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as e:
            raise OpenClawGatewayError(
                f"openclaw CLI not executable at {self._cli_path}: {e}",
                context={"cli_path": self._cli_path},
            ) from e
        except OSError as e:
            raise OpenClawGatewayError(
                f"openclaw CLI launch failed: {e}",
                context={"cli_path": self._cli_path},
            ) from e
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            # The CLI is typically a shim (``openclaw.cmd`` on Windows, an
            # npm wrapper on POSIX) that spawns the real interpreter as a
            # GRANDCHILD. ``proc.kill()`` signals only the immediate child,
            # orphaning that grandchild — which keeps the stdout/stderr
            # pipes open and wedges the event loop's subprocess transport
            # at teardown (observed as a hung full test sweep on Windows).
            # Reap the whole tree so no descendant survives to hold the
            # pipes. Never raise from the cleanup path.
            await self._reap_process_tree(proc)
            raise OpenClawGatewayError(
                f"openclaw CLI timed out after {timeout:.1f}s "
                f"({' '.join(args[:3])}...)",
                context={"timeout_s": timeout, "args": list(args[:3])},
            ) from e
        duration = loop.time() - start
        return CliResult(
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            duration_s=duration,
        )

    async def _reap_process_tree(
        self,
        proc: "asyncio.subprocess.Process",
    ) -> None:
        """Terminate ``proc`` and every descendant, then let asyncio reap
        its transport. Called from the timeout cleanup path; never raises.

        The tree is walked and killed via :func:`kill_process_tree` while
        the root is still alive — once the root exits, psutil can no longer
        reach grandchildren (a shim such as ``openclaw.cmd`` spawns the
        real interpreter as a grandchild, so killing only the root would
        orphan it). The synchronous kill runs in the default executor so
        the event loop stays free to drain the now-closing pipes.

        Args:
            proc: The subprocess whose tree should be reaped.
        """
        pid = proc.pid
        if pid and pid > 0:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: kill_process_tree(pid, grace_seconds=1.0),
                )
            except Exception:  # noqa: BLE001 — cleanup must never raise
                logger.debug(
                    "kill_process_tree failed for openclaw CLI pid %s",
                    pid, exc_info=True,
                )
        # asyncio still holds a transport for ``proc``; proc.kill() is now
        # usually a no-op (already reaped) and proc.wait() lets the
        # transport close so the pipes are released.
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 — process already gone is fine
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

    def _build_env(self) -> Dict[str, str]:
        """Process env for subprocesses. Inherit from parent then layer
        in caller-supplied overrides. We deliberately do NOT inject the
        bearer token — the CLI reads it from openclaw.json itself, so
        we'd just be paving the same path."""
        env: Dict[str, str] = dict(os.environ)
        env.update(self._env_overrides)
        return env

    @staticmethod
    def _raise_for_auth_failure(
        err_text: str,
        *,
        context: Dict[str, Any],
    ) -> None:
        """Inspect a CLI stderr/stdout snippet and raise
        :class:`OpenClawAuthError` if it looks like the Gateway rejected
        our credentials. Otherwise return — the caller will record the
        generic transport failure on its own."""
        lowered = err_text.lower()
        markers = (
            "401", "403", "unauthorized", "forbidden",
            "invalid token", "auth required", "authentication failed",
        )
        if any(m in lowered for m in markers):
            raise OpenClawAuthError(
                "OpenClaw Gateway rejected the bridge's credentials",
                context={**context, "stderr_snippet": err_text[:200]},
            )


__all__ = [
    "AgentRunResult",
    "CliResult",
    "HeartbeatResult",
    "OpenClawClient",
    "SendMessageResult",
    "ToolInvocationResult",
    "discover_cli",
]
