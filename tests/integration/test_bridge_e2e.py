"""End-to-end integration tests for the OpenClaw bridge (Phase 3.6).

These tests do NOT require a running OpenClaw Gateway. They use a
small Python script as a stand-in for the ``openclaw`` CLI so we can
exercise the full bridge path (subprocess spawn → stdout parse →
result dataclass) with no external dependencies.

The full live-Gateway end-to-end (real ``openclaw mcp set`` against a
running Gateway) is exercised in Phase 4 once we have a real channel
to use as a smoke target.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ultron.openclaw_bridge import (
    OpenClawBridge,
    OpenClawClient,
    SendMessageResult,
)
from ultron.openclaw_bridge.mcp_registration import RegistrationResult


# ---------------------------------------------------------------------------
# Fake-CLI fixture
# ---------------------------------------------------------------------------


def _write_stub_cli(target: Path) -> Path:
    """Write a small Python script that mimics the ``openclaw`` CLI's
    response shapes for the subset of subcommands we exercise."""
    script_body = """\
import json
import sys

args = sys.argv[1:]
if not args:
    sys.exit(2)

if args[0] == "health" and "--json" in args:
    print(json.dumps({"ok": True}))
    sys.exit(0)

if args[:2] == ["message", "send"]:
    out = {"messageId": "stub-msg-1", "delivered": True}
    print(json.dumps(out))
    sys.exit(0)

if args[:2] == ["system", "event"]:
    print(json.dumps({"finalText": "ok"}))
    sys.exit(0)

if args[:2] == ["mcp", "list"]:
    print("No MCP servers configured in stub.json.")
    sys.exit(0)

if args[:2] == ["mcp", "show"]:
    sys.exit(1)                                              # not configured

if args[:2] == ["mcp", "set"]:
    sys.exit(0)

if args[:2] == ["mcp", "unset"]:
    sys.exit(0)

# Fallback: dump args as JSON so callers see something parseable.
print(json.dumps({"args": args}))
sys.exit(0)
"""
    target.write_text(script_body, encoding="utf-8")
    return target


@pytest.fixture
def stub_cli_wrapper(tmp_path: Path) -> Path:
    """A platform-appropriate executable wrapper around the stub script.

    On Windows we generate a ``.cmd`` shim so subprocess can spawn it
    without an interpreter prefix. On POSIX we make a shebanged
    wrapper script.
    """
    script = _write_stub_cli(tmp_path / "stub_cli.py")
    if sys.platform == "win32":
        wrapper = tmp_path / "openclaw_stub.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )
    else:
        wrapper = tmp_path / "openclaw_stub"
        wrapper.write_text(
            f"#!/usr/bin/env bash\nexec {sys.executable} {script} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    return wrapper


def _make_cfg(
    *,
    enabled: bool = True,
    cli_path: str | None = None,
    workspace: str | None = None,
    mcp_command: str | None = None,
    voice_handoff: bool = False,
) -> SimpleNamespace:
    bridge = SimpleNamespace(
        cli_path=cli_path,
        cli_timeout_seconds=5.0,
        mcp_server_name="ultron-mcp-test",
        mcp_server_command=mcp_command,
        mcp_server_args=[],
        retry_registration_interval_seconds=0.05,
        workspace_dir=workspace,
        workspace_lock_timeout_seconds=2.0,
        inbound_voice_handoff_enabled=voice_handoff,
        inbound_voice_handoff_prefix="[voice]",
        tool_invocation_timeout_seconds=5.0,
        message_send_timeout_seconds=5.0,
    )
    return SimpleNamespace(
        enabled=enabled,
        gateway_url=None,
        auth_token_env="OPENCLAW_AUTH_TOKEN",
        health_check_timeout_seconds=5.0,
        health_check_interval_seconds=60.0,
        fail_open=True,
        required_agent_id="ultron-main",
        bridge=bridge,
    )


# ---------------------------------------------------------------------------
# End-to-end through the holder
# ---------------------------------------------------------------------------


async def test_health_through_real_subprocess(
    stub_cli_wrapper: Path, tmp_path: Path,
) -> None:
    """OpenClawClient → real subprocess → stub CLI → parsed result.

    The health probe gets a generous 20s timeout (vs the 5s default).
    Under a loaded full sweep the Windows ``.cmd`` → ``python`` double-hop
    can take several seconds to cold-start, and a tight budget made this
    test flaky: it tripped the probe timeout and (before the tree-reap
    fix in ``OpenClawClient._run_cli``) orphaned the grandchild
    interpreter, which stalled the whole sweep. 20s clears realistic
    cold-start latency while staying well under the 30s per-test deadline.
    """
    client = OpenClawClient(cli_path=str(stub_cli_wrapper), default_timeout_s=20.0)
    assert await client.health(timeout_s=20.0) is True


async def test_send_message_through_real_subprocess(
    stub_cli_wrapper: Path,
) -> None:
    client = OpenClawClient(cli_path=str(stub_cli_wrapper), default_timeout_s=5.0)
    result = await client.send_message("telegram", "@me", "hello")
    assert isinstance(result, SendMessageResult)
    assert result.delivered is True
    assert result.message_id == "stub-msg-1"


async def test_trigger_heartbeat_through_real_subprocess(
    stub_cli_wrapper: Path,
) -> None:
    client = OpenClawClient(cli_path=str(stub_cli_wrapper), default_timeout_s=5.0)
    result = await client.trigger_heartbeat("ping", mode="now")
    assert result.triggered is True
    assert result.final_text == "ok"


async def test_mcp_set_show_unset_round_trip(
    stub_cli_wrapper: Path,
) -> None:
    client = OpenClawClient(cli_path=str(stub_cli_wrapper), default_timeout_s=5.0)
    # set succeeds
    assert await client.mcp_set(
        "ultron-mcp", "some-cmd", args=["--stdio"], env={},
    ) is True
    # show returns None (stub treats it as not configured)
    assert await client.mcp_show("ultron-mcp") is None
    # unset succeeds
    assert await client.mcp_unset("ultron-mcp") is True


# ---------------------------------------------------------------------------
# Bridge holder lifecycle
# ---------------------------------------------------------------------------


def test_bridge_constructs_and_starts_with_unreachable_gateway(
    stub_cli_wrapper: Path, tmp_path: Path,
) -> None:
    """When Gateway is unreachable the bridge should still construct,
    start, and operate in degraded mode (retry thread launched if MCP
    command was configured)."""
    cfg = _make_cfg(
        cli_path=str(stub_cli_wrapper),
        workspace=str(tmp_path / "ws"),
        mcp_command=None,                                      # no command → no retry
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    bridge.lifecycle.is_reachable = MagicMock(return_value=False)            # type: ignore[method-assign]
    bridge.start()
    # No MCP command → no retry thread.
    assert bridge._retry_thread is None
    bridge.shutdown()


def test_bridge_reachable_path_runs_register(
    stub_cli_wrapper: Path, tmp_path: Path,
) -> None:
    """When Gateway is reachable AND MCP command is configured, the
    holder should call register exactly once."""
    cfg = _make_cfg(
        cli_path=str(stub_cli_wrapper),
        workspace=str(tmp_path / "ws"),
        mcp_command="real-stdio-cmd",
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None
    bridge.lifecycle.is_reachable = MagicMock(return_value=True)             # type: ignore[method-assign]

    register_mock = AsyncMock(
        return_value=RegistrationResult(registered=True, name="ultron-mcp-test"),
    )
    assert bridge.registrar is not None
    bridge.registrar.register = register_mock                                # type: ignore[assignment]

    bridge.start()
    assert register_mock.await_count == 1
    bridge.shutdown()


def test_bridge_workspace_writer_path_e2e(
    stub_cli_wrapper: Path, tmp_path: Path,
) -> None:
    """Verify the WorkspaceWriter wiring works end-to-end through the
    holder — entries land on disk under the configured workspace."""
    workspace = tmp_path / "ws"
    cfg = _make_cfg(
        cli_path=str(stub_cli_wrapper),
        workspace=str(workspace),
    )
    bridge = OpenClawBridge.from_config(cfg)
    assert bridge is not None

    async def _write() -> None:
        result = await bridge.workspace.write_memory_entry(
            "phase 3 e2e test", prefix_timestamp=False,
        )
        assert result.error is None
        assert result.bytes_written > 0

    asyncio.run(_write())

    memory_files = list((workspace / "memory").glob("*.md"))
    assert len(memory_files) == 1
    assert "phase 3 e2e test" in memory_files[0].read_text(encoding="utf-8")
