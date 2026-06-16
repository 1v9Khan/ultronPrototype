"""Stdio MCP entry point for OpenClaw integration (Phase 13 finish).

OpenClaw 2026.5.7 expects ``mcp set`` to register a stdio command.
This script is what gets registered. It boots a FastMCP instance
exposing Kenning's read-mostly tools (heartbeat alerts, maintenance,
coding session listing) and serves them on stdio until OpenClaw
closes the channel.

This process is ephemeral — OpenClaw spawns a fresh instance per
agent run that needs an Kenning tool. State lives entirely on disk
(JSONL alert log, per-session audit files), so cross-process
coordination is unnecessary.

Imports stay light: the heavy kenning components (torch, LLM,
embedder) are NOT loaded here. The maintenance tool subprocesses
out to ``run_maintenance_for_cron.py`` so its own process owns the
heavy load.

Register with OpenClaw:

    openclaw mcp set kenning-mcp '{
      "command": "C:\\\\STC\\\\ultronPrototype\\\\.venv\\\\Scripts\\\\python.exe",
      "args": ["<this script's absolute path>", "--stdio"],
      "env": {}
    }'

The ``OpenClawBridgeConfig.mcp_server_command`` field auto-resolves
to this script when the bridge starts up — see
:meth:`KenningMcpRegistrar.register`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project's ``src/`` importable so we can pull in
# ``kenning.openclaw_bridge.mcp_tools``. Done before any ``kenning.*``
# import.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))


def _refuse_if_gaming() -> bool:
    """True (caller should exit) when Ultron's anticheat-safe / gaming mode is
    active.

    This MCP server is spawned EXTERNALLY by OpenClaw, once per agent-run that
    needs a Kenning tool -- it is NOT part of Ultron's lean gaming boot. But a
    Kenning-labeled Python process appearing (and sub-spawning maintenance
    workers) DURING a kernel-anticheat match is exactly the process footprint
    the user requires absent. So we hard-refuse to start whenever anticheat-safe
    mode is active (the config pin is permanently on for this rig, so in
    practice the OpenClaw<->Kenning bridge is simply unavailable while the
    anticheat posture is engaged -- a deliberate safety trade). FAIL-CLOSED: if
    we cannot determine the state, refuse -- the hard rule is that nothing
    unverified runs while gaming.
    """
    try:
        from kenning.safety.anticheat import anticheat_active

        return bool(anticheat_active())
    except Exception as e:  # noqa: BLE001 - fail-closed on any uncertainty
        sys.stderr.write(
            f"kenning-mcp: could not verify anticheat state ({e}); refusing to "
            f"start (fail-closed).\n"
        )
        return True


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

    # Anticheat hygiene: while the user is in a kernel-anticheat game, refuse to
    # come up AT ALL -- exit BEFORE importing the tool modules, so not even this
    # process's imports touch RAM during a match. (--list-tools is a deliberate
    # registration/introspection action, exempt.)
    if not args.list_tools and _refuse_if_gaming():
        sys.stderr.write(
            "kenning-mcp: anticheat / gaming mode is active -- refusing to start "
            "the OpenClaw MCP server (nothing OpenClaw-related runs while in a "
            "match). Disable gaming_mode.anticheat_safe_mode to use it.\n"
        )
        return 0

    from kenning.openclaw_bridge.mcp_tools import build_server, run_stdio

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
