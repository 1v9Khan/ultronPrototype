"""Construct + wire the MCP server registry from config (T22).

Turns the operator's ``mcp.servers`` config into a live
:class:`~ultron.mcp.registry.McpServerRegistry` with real lifecycle callables:

* a stdio ``starter`` that spawns the server child with a SANITISED
  environment (dangerous vars dropped via :func:`filter_environment`),
  tracks it in the T12 process registry + the zombie killer (persistent),
  and returns the pid; and
* a ``killer`` that reaps the child process tree on stop / shutdown (T8
  :func:`kill_process_tree`).

The registry tracks lifecycle + state; the JSON-RPC protocol layer (tool
discovery + invocation) is provided by the optional ``mcp`` Python SDK and is
NOT required to register or manage servers. Default-OFF: :func:`build_mcp_server_registry`
returns ``None`` unless ``mcp.enabled`` is True, so callers no-op cleanly.
"""

from __future__ import annotations

import subprocess
from typing import Optional

from ultron.mcp.registry import (
    McpServerHandle,
    McpServerRegistry,
    set_mcp_server_registry,
)
from ultron.mcp.transport import (
    HttpMcpTransportConfig,
    McpTransportKind,
    SseMcpTransportConfig,
    StdioMcpTransportConfig,
    StreamableHttpMcpTransportConfig,
    TransportConfig,
)
from ultron.utils.logging import get_logger

logger = get_logger("mcp.builder")


def transport_from_spec(spec) -> TransportConfig:
    """Build the matching transport dataclass from a config ``McpServerSpec``.

    Raises ``ValueError`` on an unknown transport kind.
    """
    kind = str(getattr(spec, "transport", "stdio") or "stdio").lower()
    timeout = float(getattr(spec, "connection_timeout_seconds", 30.0))
    if kind == McpTransportKind.STDIO.value:
        return StdioMcpTransportConfig(
            command=getattr(spec, "command", "") or "",
            args=tuple(getattr(spec, "args", ()) or ()),
            cwd=getattr(spec, "cwd", None),
            env=dict(getattr(spec, "env", {}) or {}),
            allow_env=tuple(getattr(spec, "allow_env", ()) or ()),
            connection_timeout_seconds=timeout,
        )
    url = getattr(spec, "url", "") or ""
    headers = dict(getattr(spec, "headers", {}) or {})
    allow_headers = tuple(getattr(spec, "allow_headers", ()) or ())
    if kind == McpTransportKind.HTTP.value:
        return HttpMcpTransportConfig(
            url=url, headers=headers, allow_headers=allow_headers,
            connection_timeout_seconds=timeout,
        )
    if kind == McpTransportKind.SSE.value:
        return SseMcpTransportConfig(
            url=url, headers=headers, allow_headers=allow_headers,
            connection_timeout_seconds=timeout,
        )
    if kind == McpTransportKind.STREAMABLE_HTTP.value:
        return StreamableHttpMcpTransportConfig(
            url=url, headers=headers, allow_headers=allow_headers,
            connection_timeout_seconds=timeout,
        )
    raise ValueError(f"unknown MCP transport kind: {kind!r}")


def _make_stdio_starter(*, popen=subprocess.Popen):
    """Return a ``starter`` that spawns stdio MCP children with a sanitised env
    + T12/T8 process tracking. HTTP-family servers have no child to spawn here
    (the connection is URL-based; the JSON-RPC layer is the SDK's job) -> the
    starter returns ``None`` for them. ``popen`` is injectable for tests."""

    def _starter(handle: McpServerHandle) -> Optional[int]:
        transport = handle.transport
        if not isinstance(transport, StdioMcpTransportConfig):
            return None  # HTTP / SSE / streamable -> no child process here.
        if not transport.command:
            raise ValueError(
                f"MCP server {handle.server_id}: stdio transport needs a command"
            )
        argv = [transport.command, *transport.args]
        # transport.env was already sanitised at register() time.
        proc = popen(
            argv,
            cwd=transport.cwd,
            env=dict(transport.env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pid = proc.pid
        handle.metadata["process"] = proc
        try:
            from ultron.subprocess.process_registry import get_process_registry
            get_process_registry().register(
                f"mcp:{handle.server_id}",
                scope_key=handle.scope_key or "mcp",
                pid=pid,
                command=" ".join(argv)[:200],
                tags=("mcp_server", handle.server_id),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("process-registry register for MCP %s failed: %s",
                         handle.server_id, e)
        try:
            from ultron.subprocess.zombie_killer import get_zombie_killer
            get_zombie_killer().register(
                pid, f"mcp:{handle.server_id}", persistent=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("zombie-killer register for MCP %s failed: %s",
                         handle.server_id, e)
        logger.info("MCP server %s started (pid=%s)", handle.server_id, pid)
        return pid

    return _starter


def _make_killer():
    """Return a ``killer`` that reaps the child process tree (T8) + drops it
    from the zombie killer."""

    def _killer(pid: int) -> None:
        try:
            from ultron.subprocess.kill_tree import kill_process_tree
            kill_process_tree(pid)
        except Exception as e:  # noqa: BLE001
            logger.debug("kill_process_tree for MCP pid %s failed: %s", pid, e)
        try:
            from ultron.subprocess.zombie_killer import get_zombie_killer
            get_zombie_killer().unregister(pid)
        except Exception:  # noqa: BLE001
            pass

    return _killer


def build_mcp_server_registry(cfg=None, *, starter=None, killer=None) -> Optional[McpServerRegistry]:
    """Construct + populate the MCP server registry from config.

    Returns the registry with every configured server registered (transport
    sanitised) and the real stdio starter + ``kill_process_tree`` killer wired,
    or ``None`` when MCP is disabled (``mcp.enabled`` False) so callers no-op.
    Sets the module singleton so :func:`~ultron.mcp.get_mcp_server_registry`
    returns it. Fail-open: a bad server spec is logged + skipped, never raising.

    Args:
        cfg: the ``mcp`` config section; pulled from the global config when
            omitted.
        starter / killer: injectable lifecycle callables (tests pass fakes).
    """
    if cfg is None:
        try:
            from ultron.config import get_config
            cfg = get_config().mcp
        except Exception as e:  # noqa: BLE001
            logger.debug("mcp config unavailable: %s", e)
            return None
    if not getattr(cfg, "enabled", False):
        return None
    registry = McpServerRegistry(
        starter=starter if starter is not None else _make_stdio_starter(),
        killer=killer if killer is not None else _make_killer(),
    )
    for spec in getattr(cfg, "servers", []) or []:
        server_id = getattr(spec, "id", "") or ""
        try:
            transport = transport_from_spec(spec)
            registry.register(
                server_id,
                transport=transport,
                scope_key=getattr(spec, "scope_key", "") or "",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("MCP server spec %r skipped: %s", server_id or "?", e)
    set_mcp_server_registry(registry)
    return registry


__all__ = ["build_mcp_server_registry", "transport_from_spec"]
