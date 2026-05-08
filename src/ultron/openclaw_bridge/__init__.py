"""OpenClaw bridge.

Glue between Ultron's orchestrator and OpenClaw's Gateway. The bridge
is consulted when:
- Ultron's orchestrator wants to call an OpenClaw tool (browser, image
  generation, etc.).
- Ultron starts up (registers Ultron MCP with the Gateway, loads
  persona files).
- OpenClaw forwards an inbound event Ultron should react to.

The voice pipeline does NOT touch the bridge. Voice queries flow
through the existing in-process pipeline (or a future HTTP-client
mode of the same llama-cpp-server) without consulting OpenClaw.

Phase 1 ships the persona loader. Subsequent phases add the HTTP
client, lifecycle manager, MCP registrar, workspace writer, and
event receiver.
"""

from ultron.openclaw_bridge.persona import (
    PersonaBundle,
    PersonaFile,
    PersonaLoader,
    PromptMode,
    default_workspace_dir,
)

__all__ = [
    "PersonaBundle",
    "PersonaFile",
    "PersonaLoader",
    "PromptMode",
    "default_workspace_dir",
]
