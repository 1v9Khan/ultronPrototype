"""Tests for ``ultron.openclaw_bridge.client.OpenClawClient``.

The natural seam is :meth:`OpenClawClient._run_cli` — every public
method funnels through it. Mocking that method lets us exercise the
parsing, error-translation, and CLI-arg construction without spawning
a real subprocess. One test exercises CLI discovery end-to-end with
a real (trivial) fake executable to cover the transport path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence
from unittest.mock import AsyncMock, patch

import pytest

from ultron.errors import (
    OpenClawAuthError,
    OpenClawGatewayError,
    OpenClawToolError,
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


# ---------------------------------------------------------------------------
# discover_cli
# ---------------------------------------------------------------------------


def test_discover_cli_uses_explicit_override(tmp_path: Path) -> None:
    fake = tmp_path / "openclaw_fake"
    fake.write_text("# fake")
    assert discover_cli(str(fake)) == str(fake.resolve())


def test_discover_cli_raises_when_override_missing(tmp_path: Path) -> None:
    with pytest.raises(OpenClawGatewayError):
        discover_cli(str(tmp_path / "does_not_exist"))


def test_discover_cli_falls_back_to_env(monkeypatch, tmp_path: Path) -> None:
    fake = tmp_path / "openclaw_env"
    fake.write_text("# fake")
    monkeypatch.setenv("ULTRON_OPENCLAW_CLI", str(fake))
    monkeypatch.setattr("ultron.openclaw_bridge.client.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "ultron.openclaw_bridge.client._WINDOWS_DEFAULT_CLI",
        Path("does/not/exist"),
    )
    assert discover_cli() == str(fake.resolve())


def test_discover_cli_raises_when_nothing_found(monkeypatch) -> None:
    monkeypatch.delenv("ULTRON_OPENCLAW_CLI", raising=False)
    monkeypatch.setattr("ultron.openclaw_bridge.client.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "ultron.openclaw_bridge.client._WINDOWS_DEFAULT_CLI",
        Path("does/not/exist"),
    )
    with pytest.raises(OpenClawGatewayError):
        discover_cli()


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_cli(tmp_path: Path) -> Path:
    p = tmp_path / "openclaw_fake.cmd"
    p.write_text("# fake")
    return p


@pytest.fixture
def client(fake_cli: Path) -> OpenClawClient:
    return OpenClawClient(cli_path=str(fake_cli), default_timeout_s=2.0)


def test_client_construction_resolves_cli(fake_cli: Path) -> None:
    c = OpenClawClient(cli_path=str(fake_cli))
    assert Path(c.cli_path) == fake_cli.resolve()


def test_client_construction_raises_when_cli_missing(tmp_path: Path) -> None:
    with pytest.raises(OpenClawGatewayError):
        OpenClawClient(cli_path=str(tmp_path / "missing"))


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------


async def test_health_returns_true_on_ok_payload(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(0, '{"ok": true}', "", 0.01))
    with patch.object(client, "_run_cli", fake):
        assert await client.health() is True


async def test_health_returns_true_on_status_healthy(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(0, '{"status":"healthy"}', "", 0.01))
    with patch.object(client, "_run_cli", fake):
        assert await client.health() is True


async def test_health_returns_false_on_nonzero_exit(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(1, "", "boom", 0.01))
    with patch.object(client, "_run_cli", fake):
        assert await client.health() is False


async def test_health_returns_false_on_malformed_json(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(0, "not-json", "", 0.01))
    with patch.object(client, "_run_cli", fake):
        assert await client.health() is False


async def test_health_returns_false_on_gateway_error(client: OpenClawClient) -> None:
    fake = AsyncMock(side_effect=OpenClawGatewayError("bang"))
    with patch.object(client, "_run_cli", fake):
        assert await client.health() is False


# ---------------------------------------------------------------------------
# send_message()
# ---------------------------------------------------------------------------


async def test_send_message_constructs_correct_args(client: OpenClawClient) -> None:
    captured: List[Sequence[str]] = []

    async def fake_run(args: Sequence[str], *, timeout_s: Optional[float] = None) -> CliResult:
        captured.append(list(args))
        return CliResult(0, '{"messageId": "abc-123"}', "", 0.01)

    with patch.object(client, "_run_cli", fake_run):
        result = await client.send_message("telegram", "@user", "hello there")

    assert isinstance(result, SendMessageResult)
    assert result.delivered is True
    assert result.message_id == "abc-123"
    assert result.channel == "telegram"
    # Verify CLI args.
    assert captured[0][:2] == ["message", "send"]
    assert "--channel" in captured[0]
    assert "telegram" in captured[0]
    assert "--target" in captured[0]
    assert "@user" in captured[0]
    assert "--message" in captured[0]
    assert "hello there" in captured[0]
    assert "--json" in captured[0]


async def test_send_message_rejects_empty_text(client: OpenClawClient) -> None:
    result = await client.send_message("telegram", "@user", "   ")
    assert result.delivered is False
    assert "empty" in (result.error or "").lower()


async def test_send_message_returns_error_on_nonzero(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(2, "", "boom", 0.01))
    with patch.object(client, "_run_cli", fake):
        result = await client.send_message("telegram", "@user", "hi")
    assert result.delivered is False
    assert "boom" in (result.error or "")


async def test_send_message_raises_auth_error_on_401(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(1, "", "401 Unauthorized", 0.01))
    with patch.object(client, "_run_cli", fake):
        with pytest.raises(OpenClawAuthError):
            await client.send_message("telegram", "@user", "hi")


async def test_send_message_returns_error_on_gateway_failure(client: OpenClawClient) -> None:
    fake = AsyncMock(side_effect=OpenClawGatewayError("CLI crashed"))
    with patch.object(client, "_run_cli", fake):
        result = await client.send_message("telegram", "@user", "hi")
    assert result.delivered is False
    assert "CLI crashed" in (result.error or "")


# ---------------------------------------------------------------------------
# trigger_heartbeat()
# ---------------------------------------------------------------------------


async def test_trigger_heartbeat_constructs_args(client: OpenClawClient) -> None:
    captured: List[Sequence[str]] = []

    async def fake_run(args, *, timeout_s=None):
        captured.append(list(args))
        return CliResult(0, '{"finalText":"ok"}', "", 0.01)

    with patch.object(client, "_run_cli", fake_run):
        result = await client.trigger_heartbeat(
            "manual ping", mode="now", expect_final=True,
        )
    assert result.triggered is True
    assert result.final_text == "ok"
    assert captured[0][:2] == ["system", "event"]
    assert "--mode" in captured[0] and "now" in captured[0]
    assert "--text" in captured[0] and "manual ping" in captured[0]
    assert "--expect-final" in captured[0]
    assert "--json" in captured[0]


async def test_trigger_heartbeat_omits_text_when_none(client: OpenClawClient) -> None:
    captured: List[Sequence[str]] = []

    async def fake_run(args, *, timeout_s=None):
        captured.append(list(args))
        return CliResult(0, "{}", "", 0.01)

    with patch.object(client, "_run_cli", fake_run):
        await client.trigger_heartbeat()
    assert "--text" not in captured[0]


async def test_trigger_heartbeat_returns_error(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(1, "", "gateway down", 0.01))
    with patch.object(client, "_run_cli", fake):
        result = await client.trigger_heartbeat("ping")
    assert result.triggered is False
    assert "gateway down" in (result.error or "")


# ---------------------------------------------------------------------------
# run_agent()
# ---------------------------------------------------------------------------


async def test_run_agent_uses_default_agent_id(client: OpenClawClient) -> None:
    captured: List[Sequence[str]] = []

    async def fake_run(args, *, timeout_s=None):
        captured.append(list(args))
        return CliResult(0, '{"text":"sure"}', "", 0.01)

    with patch.object(client, "_run_cli", fake_run):
        result = await client.run_agent("hello")
    assert result.success is True
    assert result.agent_id == "ultron-main"
    assert result.text == "sure"
    assert "--agent" in captured[0] and "ultron-main" in captured[0]


async def test_run_agent_with_deliver_flags(client: OpenClawClient) -> None:
    captured: List[Sequence[str]] = []

    async def fake_run(args, *, timeout_s=None):
        captured.append(list(args))
        return CliResult(0, "{}", "", 0.01)

    with patch.object(client, "_run_cli", fake_run):
        await client.run_agent(
            "ping",
            agent_id="ultron-test",
            thinking="low",
            deliver=True,
            reply_channel="telegram",
            reply_to="@user",
        )
    args = captured[0]
    assert "--thinking" in args and "low" in args
    assert "--deliver" in args
    assert "--reply-channel" in args and "telegram" in args
    assert "--reply-to" in args and "@user" in args


# ---------------------------------------------------------------------------
# invoke_tool()
# ---------------------------------------------------------------------------


async def test_invoke_tool_passes_through_agent_text(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(
        0, '{"text":"opened the page"}', "", 0.01,
    ))
    with patch.object(client, "_run_cli", fake):
        result = await client.invoke_tool(
            "browser",
            params={"url": "https://example.com"},
        )
    assert isinstance(result, ToolInvocationResult)
    assert result.success is True
    assert result.tool_name == "browser"
    assert "opened" in result.text


async def test_invoke_tool_raises_on_unavailable_response(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(
        0, '{"text":"That tool is not available."}', "", 0.01,
    ))
    with patch.object(client, "_run_cli", fake):
        with pytest.raises(OpenClawToolError):
            await client.invoke_tool("browser", params={"url": "x"})


async def test_invoke_tool_returns_failure_on_run_error(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(1, "", "boom", 0.01))
    with patch.object(client, "_run_cli", fake):
        result = await client.invoke_tool("browser", params={})
    assert result.success is False
    assert "boom" in (result.error or "")


# ---------------------------------------------------------------------------
# MCP config helpers
# ---------------------------------------------------------------------------


async def test_mcp_set_constructs_payload(client: OpenClawClient) -> None:
    captured: List[Sequence[str]] = []

    async def fake_run(args, *, timeout_s=None):
        captured.append(list(args))
        return CliResult(0, "ok", "", 0.01)

    with patch.object(client, "_run_cli", fake_run):
        ok = await client.mcp_set(
            "ultron-mcp",
            command=r"C:\path\to\proxy.exe",
            args=["--stdio"],
            env={"FOO": "bar"},
        )
    assert ok is True
    assert captured[0][:3] == ["mcp", "set", "ultron-mcp"]
    payload = json.loads(captured[0][3])
    assert payload["command"] == r"C:\path\to\proxy.exe"
    assert payload["args"] == ["--stdio"]
    assert payload["env"] == {"FOO": "bar"}


async def test_mcp_set_raises_on_failure(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(2, "", "config locked", 0.01))
    with patch.object(client, "_run_cli", fake):
        with pytest.raises(OpenClawGatewayError):
            await client.mcp_set("name", "cmd", args=[])


async def test_mcp_set_rejects_empty_inputs(client: OpenClawClient) -> None:
    with pytest.raises(ValueError):
        await client.mcp_set("", "cmd")
    with pytest.raises(ValueError):
        await client.mcp_set("name", "")


async def test_mcp_show_returns_dict_on_success(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(
        0, '{"command": "x", "args": ["-y"], "env": {}}', "", 0.01,
    ))
    with patch.object(client, "_run_cli", fake):
        payload = await client.mcp_show("ultron-mcp")
    assert payload == {"command": "x", "args": ["-y"], "env": {}}


async def test_mcp_show_returns_none_when_missing(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(1, "", "no such entry", 0.01))
    with patch.object(client, "_run_cli", fake):
        assert await client.mcp_show("missing") is None


async def test_mcp_unset_returns_true_on_success(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(0, "ok", "", 0.01))
    with patch.object(client, "_run_cli", fake):
        assert await client.mcp_unset("ultron-mcp") is True


async def test_mcp_unset_returns_false_when_missing(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(1, "", "Not configured", 0.01))
    with patch.object(client, "_run_cli", fake):
        assert await client.mcp_unset("ultron-mcp") is False


async def test_mcp_list_handles_no_servers(client: OpenClawClient) -> None:
    fake = AsyncMock(return_value=CliResult(
        0, "No MCP servers configured in foo.\n", "", 0.01,
    ))
    with patch.object(client, "_run_cli", fake):
        result = await client.mcp_list()
    assert result == {}


async def test_mcp_list_parses_json(client: OpenClawClient) -> None:
    payload = {"ultron-mcp": {"command": "x", "args": []}}
    fake = AsyncMock(return_value=CliResult(0, json.dumps(payload), "", 0.01))
    with patch.object(client, "_run_cli", fake):
        result = await client.mcp_list()
    assert result == payload


async def test_mcp_list_parses_text(client: OpenClawClient) -> None:
    text = "ultron-mcp: /path/to/proxy --stdio\nother: /bin/x"
    fake = AsyncMock(return_value=CliResult(0, text, "", 0.01))
    with patch.object(client, "_run_cli", fake):
        result = await client.mcp_list()
    assert "ultron-mcp" in result
    assert "other" in result
    assert "proxy" in result["ultron-mcp"]["raw"]


# ---------------------------------------------------------------------------
# Real-subprocess transport (covers _run_cli's spawn/timeout paths)
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_cli(tmp_path: Path) -> Path:
    """Tiny Python script that echoes JSON to stdout, used to exercise
    real subprocess execution from :meth:`OpenClawClient._run_cli`."""
    script = tmp_path / "fake_openclaw.py"
    script.write_text(
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "if args and args[0] == 'health':\n"
        "    print(json.dumps({'ok': True}))\n"
        "else:\n"
        "    print('OK', ' '.join(args))\n"
    )
    return script


async def test_run_cli_executes_real_subprocess(echo_cli: Path) -> None:
    # Construct a client whose CLI is `python <echo_cli>`. We use
    # patching to substitute the executable resolution.
    client = OpenClawClient.__new__(OpenClawClient)
    client._cli_path = sys.executable                                       # noqa: SLF001
    client._default_timeout_s = 5.0                                         # noqa: SLF001
    client._config_path = Path.home() / ".openclaw" / "openclaw.json"       # noqa: SLF001
    client._default_agent_id = "ultron-main"                                # noqa: SLF001
    client._env_overrides = {}                                              # noqa: SLF001
    result = await client._run_cli([str(echo_cli), "health"])
    assert result.returncode == 0
    assert "ok" in result.stdout.lower()
