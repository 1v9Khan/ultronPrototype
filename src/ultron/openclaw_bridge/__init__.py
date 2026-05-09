"""OpenClaw bridge.

Glue between Ultron's orchestrator and OpenClaw's Gateway. The bridge
is consulted when:

- Ultron's orchestrator wants to call an OpenClaw tool (browser, image
  generation, messaging, etc.).
- Ultron starts up (registers Ultron MCP with the Gateway, loads
  persona files).
- OpenClaw forwards an inbound event Ultron should react to.

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
- :class:`UltronMcpRegistrar` (Phase 3.2) — idempotent MCP entry
  registration. Fail-open + background retry.
- :class:`OpenClawEventReceiver` (Phase 3.4) — gated-off scaffold
  for inbound voice handoff.
"""

from ultron.openclaw_bridge.browser import (
    ActionResult,
    BrowserTool,
    NavigateResult,
    PageTextResult,
    ScreenshotResult,
    Snapshot,
    SnapshotMode,
)
from ultron.openclaw_bridge.client import (
    AgentRunResult,
    CliResult,
    HeartbeatResult,
    OpenClawClient,
    SendMessageResult,
    ToolInvocationResult,
    discover_cli,
)
from ultron.openclaw_bridge.events import (
    IncomingMessage,
    OpenClawEventReceiver,
    VoiceHandoffHandler,
)
from ultron.openclaw_bridge.heartbeat_alerts import (
    HeartbeatAlert,
    HeartbeatAlertLog,
)
from ultron.openclaw_bridge.holder import OpenClawBridge
from ultron.openclaw_bridge.lifecycle import (
    OpenClawLifecycle,
    OpenClawStatus,
)
from ultron.openclaw_bridge.mcp_registration import (
    RegistrationResult,
    UltronMcpRegistrar,
)
from ultron.openclaw_bridge.notifications import (
    NotificationDispatcher,
    NotificationResult,
)
from ultron.openclaw_bridge.system_status import (
    SystemStatusReport,
    SystemStatusReporter,
)
from ultron.openclaw_bridge.persona import (
    PersonaBundle,
    PersonaFile,
    PersonaLoader,
    PromptMode,
    default_workspace_dir,
)
from ultron.openclaw_bridge.workspace import (
    WorkspaceWriter,
    WriteResult,
)

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
    "UltronMcpRegistrar",
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
