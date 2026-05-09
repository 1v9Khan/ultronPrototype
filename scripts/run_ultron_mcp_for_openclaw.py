"""Stdio MCP entry point for OpenClaw integration (Phase 13 finish).

OpenClaw 2026.5.7 expects ``mcp set`` to register a stdio command.
This script is what gets registered. It boots a FastMCP instance
exposing Ultron's read-mostly tools (heartbeat alerts, maintenance,
coding session listing) and serves them on stdio until OpenClaw
closes the channel.

This process is ephemeral — OpenClaw spawns a fresh instance per
agent run that needs an Ultron tool. State lives entirely on disk
(JSONL alert log, per-session audit files), so cross-process
coordination is unnecessary.

Imports stay light: the heavy ultron components (torch, LLM,
embedder) are NOT loaded here. The maintenance tool subprocesses
out to ``run_maintenance_for_cron.py`` so its own process owns the
heavy load.

Register with OpenClaw:

    openclaw mcp set ultron-mcp '{
      "command": "C:\\\\STC\\\\ultronPrototype\\\\.venv\\\\Scripts\\\\python.exe",
      "args": ["<this script's absolute path>", "--stdio"],
      "env": {}
    }'

The ``OpenClawBridgeConfig.mcp_server_command`` field auto-resolves
to this script when the bridge starts up — see
:meth:`UltronMcpRegistrar.register`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project's ``src/`` importable so we can pull in
# ``ultron.openclaw_bridge.mcp_tools``. Done before any ``ultron.*``
# import.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdio",
        action="store_true",
        default=True,
        help="Use stdio transport (default; only mode supported today).",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print registered tool names + descriptions and exit.",
    )
    args = parser.parse_args()

    from ultron.openclaw_bridge.mcp_tools import build_server, run_stdio

    if args.list_tools:
        server = build_server()
        # FastMCP exposes tools via the internal manager. Use list_tools
        # which is the user-facing introspection method.
        import asyncio
        tools = asyncio.run(server.list_tools())
        for tool in tools:
            print(f"{tool.name}: {tool.description}")
        return 0

    run_stdio()
    return 0


if __name__ == "__main__":
    sys.exit(main())
