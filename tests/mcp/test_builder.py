"""Tests for the T22 MCP server registry builder.

build_mcp_server_registry turns the operator's mcp.servers config into a live
McpServerRegistry with real lifecycle callables (env-sanitised stdio spawn +
process tracking + kill_process_tree reaper), gated behind mcp.enabled. The
lifecycle callables are injectable so these tests never spawn real processes.
"""

from __future__ import annotations

import pytest

from ultron.config import McpConfig, McpServerSpec
from ultron.mcp import McpServerState, reset_mcp_server_registry_for_testing
from ultron.mcp.builder import (
    _make_stdio_starter,
    build_mcp_server_registry,
    transport_from_spec,
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_mcp_server_registry_for_testing()
    yield
    reset_mcp_server_registry_for_testing()


# --- transport_from_spec ----------------------------------------------------


def test_transport_from_spec_stdio():
    t = transport_from_spec(
        McpServerSpec(id="s", transport="stdio", command="tool", args=["--x"]),
    )
    assert t.kind.value == "stdio"
    assert t.command == "tool"
    assert t.args == ("--x",)


@pytest.mark.parametrize("kind", ["http", "sse", "streamable_http"])
def test_transport_from_spec_http_family(kind):
    t = transport_from_spec(
        McpServerSpec(id="s", transport=kind, url="http://localhost:9000"),
    )
    assert t.kind.value == kind
    assert t.url == "http://localhost:9000"


def test_transport_from_spec_unknown_raises():
    with pytest.raises(ValueError):
        transport_from_spec(McpServerSpec(id="s", transport="carrier_pigeon"))


# --- build_mcp_server_registry ----------------------------------------------


def test_disabled_returns_none():
    cfg = McpConfig(enabled=False, servers=[McpServerSpec(id="x", command="echo")])
    assert build_mcp_server_registry(cfg) is None


def test_enabled_registers_servers():
    cfg = McpConfig(enabled=True, servers=[
        McpServerSpec(id="s1", transport="stdio", command="tool", args=["--x"]),
        McpServerSpec(id="s2", transport="http", url="http://localhost:9000"),
    ])
    reg = build_mcp_server_registry(cfg, starter=lambda h: 1, killer=lambda p: None)
    assert reg is not None
    refs = {r.server_id: r for r in reg.list_registered()}
    assert set(refs) == {"s1", "s2"}
    assert refs["s1"].transport_kind == "stdio"
    assert refs["s2"].transport_kind == "http"


def test_register_sanitises_dangerous_env(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/evil.so")
    cfg = McpConfig(enabled=True, servers=[
        McpServerSpec(id="s", transport="stdio", command="t", env={"SAFE": "1"}),
    ])
    reg = build_mcp_server_registry(cfg, starter=lambda h: 1, killer=lambda p: None)
    handle = reg.get("s")
    assert handle.transport.env.get("SAFE") == "1"
    assert "LD_PRELOAD" not in handle.transport.env  # dropped by sanitiser


def test_unknown_transport_spec_skipped():
    cfg = McpConfig(enabled=True, servers=[
        McpServerSpec(id="bad", transport="carrier_pigeon", command="t"),
        McpServerSpec(id="ok", transport="stdio", command="t"),
    ])
    reg = build_mcp_server_registry(cfg, starter=lambda h: 1, killer=lambda p: None)
    assert reg.get("bad") is None  # fail-open: skipped
    assert reg.get("ok") is not None


def test_start_invokes_starter_and_records_pid():
    cfg = McpConfig(enabled=True, servers=[McpServerSpec(id="s", command="t")])
    started: list[str] = []
    reg = build_mcp_server_registry(
        cfg, starter=lambda h: (started.append(h.server_id), 999)[1], killer=lambda p: None,
    )
    assert reg.start("s") == McpServerState.CONNECTED
    assert reg.get("s").pid == 999
    assert started == ["s"]


def test_stop_invokes_killer_for_stdio():
    killed: list[int] = []
    cfg = McpConfig(enabled=True, servers=[McpServerSpec(id="s", command="t")])
    reg = build_mcp_server_registry(
        cfg, starter=lambda h: 777, killer=lambda pid: killed.append(pid),
    )
    reg.start("s")
    assert reg.stop("s") is True
    assert killed == [777]


# --- stdio starter (fake Popen) ---------------------------------------------


def test_stdio_starter_spawns_with_injected_popen():
    from ultron.mcp.registry import McpServerHandle
    from ultron.mcp.transport import StdioMcpTransportConfig

    calls: dict = {}

    class _FakeProc:
        pid = 555

    def _fake_popen(argv, **kw):
        calls["argv"] = argv
        calls["env"] = kw.get("env")
        return _FakeProc()

    starter = _make_stdio_starter(popen=_fake_popen)
    handle = McpServerHandle(
        server_id="s",
        transport=StdioMcpTransportConfig(command="tool", args=("--flag",), env={"A": "1"}),
    )
    pid = starter(handle)
    assert pid == 555
    assert calls["argv"] == ["tool", "--flag"]
    assert calls["env"] == {"A": "1"}


def test_stdio_starter_returns_none_for_http_transport():
    from ultron.mcp.registry import McpServerHandle
    from ultron.mcp.transport import HttpMcpTransportConfig

    starter = _make_stdio_starter(popen=lambda *a, **k: None)
    handle = McpServerHandle(
        server_id="s", transport=HttpMcpTransportConfig(url="http://x"),
    )
    assert starter(handle) is None  # no child process for HTTP
