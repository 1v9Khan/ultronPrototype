"""OpenClaw bridge.

Glue between Kenning's orchestrator and OpenClaw's Gateway. The bridge
is consulted when:

- Kenning's orchestrator wants to call an OpenClaw tool (browser, image
  generation, messaging, etc.).
- Kenning starts up (registers Kenning MCP with the Gateway, loads
  persona files).
- OpenClaw forwards an inbound event Kenning should react to.

The voice pipeline does NOT touch the bridge. Voice queries flow
through the existing in-process pipeline (or a future HTTP-client
mode of the same llama-cpp-server) without consulting OpenClaw.

Public surface (Phase 3 complete):

- :class:`PersonaLoader` (Phase 1) — workspace persona files +
  composed system prompts in four modes.
- :class:`OpenClawLifecycle` (Phase 3 foundation) — health probes
  for the Gateway. Never raises.
- :class:`OpenClawClient` (Phase 3.1) — async client over the
  ``openclaw`` CLI. Methods: ``send_message``, ``trigger_heartbeat``,
  ``run_agent``, ``invoke_tool``, plus ``mcp_set/list/show/unset``.
- :class:`WorkspaceWriter` (Phase 3.3) — coordinated writes to the
  shared workspace (MEMORY.md, USER.md, daily files) with atomic
  rename + advisory lockfiles.
- :class:`KenningMcpRegistrar` (Phase 3.2) — idempotent MCP entry
  registration. Fail-open + background retry.
- :class:`OpenClawEventReceiver` (Phase 3.4) — gated-off scaffold
  for inbound voice handoff.
"""

# ALL public names resolve LAZILY (PEP 562, see __getattr__ below) so importing
# THIS PACKAGE -- or any submodule of it (e.g. the LLM imports
# kenning.openclaw_bridge.persona for the workspace system prompt, which triggers
# this __init__) -- does NOT eager-load the bridge RUNTIME (holder / client /
# mcp_registration / notifications / system_status) or the browser-automation
# stack into RAM. That keeps a LEAN GAMING BOOT's anticheat surface minimal. Each
# name loads on first ACCESS; the gated (non-gaming) coding + bridge paths still
# get them via `from kenning.openclaw_bridge import X`. browser is ALSO blocked at
# the loader by the anticheat import firewall while a protected game is running.
_LAZY = {
    "ActionResult": "browser", "BrowserTool": "browser",
    "NavigateResult": "browser", "PageTextResult": "browser",
    "ScreenshotResult": "browser", "Snapshot": "browser", "SnapshotMode": "browser",
    "AgentRunResult": "client", "CliResult": "client", "HeartbeatResult": "client",
    "OpenClawClient": "client", "SendMessageResult": "client",
    "ToolInvocationResult": "client", "discover_cli": "client",
    "IncomingMessage": "events", "OpenClawEventReceiver": "events",
    "VoiceHandoffHandler": "events",
    "HeartbeatAlert": "heartbeat_alerts", "HeartbeatAlertLog": "heartbeat_alerts",
    "OpenClawBridge": "holder",
    "OpenClawLifecycle": "lifecycle", "OpenClawStatus": "lifecycle",
    "RegistrationResult": "mcp_registration", "KenningMcpRegistrar": "mcp_registration",
    "NotificationDispatcher": "notifications", "NotificationResult": "notifications",
    "SystemStatusReport": "system_status", "SystemStatusReporter": "system_status",
    "PersonaBundle": "persona", "PersonaFile": "persona", "PersonaLoader": "persona",
    "PromptMode": "persona", "default_workspace_dir": "persona",
    "WorkspaceWriter": "workspace", "WriteResult": "workspace",
}

__all__ = [
    # Browser tool (Phase 6)
    "ActionResult",
    "BrowserTool",
    "NavigateResult",
    "PageTextResult",
    "ScreenshotResult",
    "Snapshot",
    "SnapshotMode",
    # Client (Phase 3.1)
    "AgentRunResult",
    "CliResult",
    "HeartbeatResult",
    "OpenClawClient",
    "SendMessageResult",
    "ToolInvocationResult",
    "discover_cli",
    # Events (Phase 3.4)
    "IncomingMessage",
    "OpenClawEventReceiver",
    "VoiceHandoffHandler",
    # Heartbeat alerts (Phase 5)
    "HeartbeatAlert",
    "HeartbeatAlertLog",
    # Holder (Phase 3.5)
    "OpenClawBridge",
    # Lifecycle (Phase 3 foundation)
    "OpenClawLifecycle",
    "OpenClawStatus",
    # MCP registration (Phase 3.2)
    "RegistrationResult",
    "KenningMcpRegistrar",
    # Notifications (Phase 4)
    "NotificationDispatcher",
    "NotificationResult",
    # System status (Phase 13)
    "SystemStatusReport",
    "SystemStatusReporter",
    # Persona (Phase 1)
    "PersonaBundle",
    "PersonaFile",
    "PersonaLoader",
    "PromptMode",
    "default_workspace_dir",
    # Workspace writer (Phase 3.3)
    "WorkspaceWriter",
    "WriteResult",
]


def __getattr__(name: str):
    # PEP 562: resolve every public name lazily from its submodule, so neither a
    # package import nor a submodule import (e.g. persona, for the LLM system
    # prompt) eager-loads the bridge runtime or the browser-automation stack.
    sub = _LAZY.get(name)
    if sub is not None:
        import importlib
        return getattr(importlib.import_module(f"kenning.openclaw_bridge.{sub}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
