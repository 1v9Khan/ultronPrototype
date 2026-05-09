"""Ultron MCP registration with the OpenClaw Gateway (Phase 3.2).

The integration spec assumed Ultron's MCP server runs as a stdio
subprocess that OpenClaw spawns on demand. Reality (as of OpenClaw
2026.5.7 and the existing Ultron build): :class:`UltronMCPServer` runs
**in-process** inside Ultron's orchestrator over Server-Sent Events
(SSE), and ``openclaw mcp set`` only accepts stdio command/args.

The reconciliation: the registrar is **config-driven** rather than
hard-coded. ``openclaw.bridge.mcp_server_command`` defaults to ``None``,
which keeps registration disabled (the registrar logs "deferred" and
the bridge moves on). When a stdio entrypoint exists — either a real
stdio MCP server or a thin stdio→SSE proxy script — the user sets
``mcp_server_command`` and the registrar wires it up.

Design contracts:

- **Idempotent.** Re-running ``register()`` with the same payload is a
  no-op (verified by reading ``openclaw mcp show`` first).
- **Fail-open.** Transport failures are logged at WARN; never raised
  out of the orchestrator's startup path. ``register()`` returns a
  :class:`RegistrationResult` instead of raising.
- **Background retry.** :meth:`schedule_retry` returns a coroutine
  callers can ``asyncio.create_task(...)`` to periodically re-attempt
  registration when the Gateway was down at startup.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from ultron.errors import (
    OpenClawAuthError,
    OpenClawGatewayError,
    OpenClawToolError,
)
from ultron.openclaw_bridge.client import OpenClawClient
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_bridge.mcp_registration")


@dataclass(frozen=True)
class RegistrationResult:
    """Outcome of one :meth:`UltronMcpRegistrar.register` attempt."""

    registered: bool
    name: str
    already_registered: bool = False                # True iff the entry was idempotently no-op
    skipped_reason: Optional[str] = None             # set when registration was deliberately deferred
    error: Optional[str] = None                      # set on transport/config failure


class UltronMcpRegistrar:
    """Registers Ultron's MCP server with OpenClaw at startup.

    Args:
        client: shared :class:`OpenClawClient` for CLI invocation.
        name: logical MCP entry name in OpenClaw's config.
        command: stdio entrypoint OpenClaw will spawn. ``None``
            disables registration (the registrar logs and skips —
            useful when Ultron's MCP is SSE-only and no proxy exists
            yet).
        args: extra args appended to ``command`` at spawn time.
        env: extra environment variables passed to the spawned MCP
            server. Caller is responsible for not putting secrets
            here — they end up in OpenClaw's config file.
    """

    def __init__(
        self,
        client: OpenClawClient,
        *,
        name: str = "ultron-mcp",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        self._client = client
        self._name = name
        self._command = command
        self._args: List[str] = list(args or [])
        self._env: Dict[str, str] = dict(env or {})
        self._lock = asyncio.Lock()                  # serialise concurrent register() calls

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_configured(self) -> bool:
        """True iff the registrar has a stdio command to register.

        When False, :meth:`register` is a deliberate no-op rather
        than a failure — the bridge stays operational without MCP
        registration."""
        return self._command is not None

    # -------------------------------------------------------------------
    # Public surface
    # -------------------------------------------------------------------

    async def register(self) -> RegistrationResult:
        """Register Ultron's MCP server with OpenClaw.

        Idempotent: when the entry already matches the configured
        command/args/env, the call is a no-op
        (``already_registered=True``). Fail-open: returns a result
        with ``error`` set on failure rather than raising. Concurrent
        callers serialise on an internal lock.
        """
        if not self.is_configured:
            logger.info(
                "MCP registration deferred (no stdio command configured for %r)",
                self._name,
            )
            return RegistrationResult(
                registered=False, name=self._name,
                skipped_reason="no stdio command configured",
            )
        async with self._lock:
            try:
                existing = await self._client.mcp_show(self._name)
            except OpenClawGatewayError as e:
                logger.warning(
                    "could not read existing MCP entry %r (%s); attempting set anyway",
                    self._name, e,
                )
                existing = None
            if self._matches_existing(existing):
                logger.info(
                    "MCP entry %r already matches desired payload (idempotent no-op)",
                    self._name,
                )
                return RegistrationResult(
                    registered=True, name=self._name,
                    already_registered=True,
                )
            try:
                await self._client.mcp_set(
                    self._name,
                    command=self._command,                   # type: ignore[arg-type]
                    args=self._args,
                    env=self._env,
                )
            except OpenClawAuthError as e:
                logger.warning(
                    "auth rejected during MCP registration (%s)", e,
                )
                return RegistrationResult(
                    registered=False, name=self._name,
                    error=f"auth rejected: {e}",
                )
            except OpenClawGatewayError as e:
                logger.warning("MCP registration failed (%s)", e)
                return RegistrationResult(
                    registered=False, name=self._name,
                    error=str(e),
                )
            logger.info(
                "registered MCP entry %r (command=%s, args=%s)",
                self._name, self._command, self._args,
            )
            return RegistrationResult(registered=True, name=self._name)

    async def verify_registered(self) -> bool:
        """Return True iff the configured entry is currently registered
        with OpenClaw and matches the configured payload."""
        if not self.is_configured:
            return False
        try:
            existing = await self._client.mcp_show(self._name)
        except OpenClawGatewayError as e:
            logger.debug(
                "verify_registered: mcp_show failed (%s) — treating as not registered",
                e,
            )
            return False
        return self._matches_existing(existing)

    async def unregister(self) -> bool:
        """Remove the configured entry from OpenClaw. Used during clean
        shutdown when the bridge wants to deregister cleanly. Returns
        True iff the entry was removed; False if it wasn't there to
        begin with. Fail-open: never raises."""
        async with self._lock:
            try:
                return await self._client.mcp_unset(self._name)
            except OpenClawGatewayError as e:
                logger.warning(
                    "MCP unregister failed for %r (%s); leaving entry as-is",
                    self._name, e,
                )
                return False

    def schedule_retry(
        self,
        *,
        interval_s: float,
        on_success: Optional[Callable[[RegistrationResult], Awaitable[None]]] = None,
        max_attempts: Optional[int] = None,
    ) -> Awaitable[None]:
        """Return a coroutine that periodically retries registration.

        Caller wraps with :func:`asyncio.create_task` to run it in the
        background. The coroutine exits cleanly when registration
        succeeds (calling ``on_success`` if provided) or when
        ``max_attempts`` is reached. The interval includes failed
        attempts only — successful no-op attempts also stop the loop.

        Use this when ``register()`` returned with ``error`` set at
        startup; the orchestrator schedules the retry and proceeds
        without blocking on the Gateway.
        """
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        return self._retry_loop(
            interval_s=interval_s,
            on_success=on_success,
            max_attempts=max_attempts,
        )

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    async def _retry_loop(
        self,
        *,
        interval_s: float,
        on_success: Optional[Callable[[RegistrationResult], Awaitable[None]]],
        max_attempts: Optional[int],
    ) -> None:
        attempt = 0
        while True:
            attempt += 1
            result = await self.register()
            if result.registered:
                logger.info(
                    "MCP registration succeeded on attempt %d", attempt,
                )
                if on_success is not None:
                    try:
                        await on_success(result)
                    except Exception as exc:                # noqa: BLE001
                        logger.warning(
                            "on_success callback raised (%s)", exc,
                        )
                return
            if max_attempts is not None and attempt >= max_attempts:
                logger.warning(
                    "MCP registration giving up after %d attempts: %s",
                    attempt, result.error or result.skipped_reason,
                )
                return
            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                logger.info(
                    "MCP registration retry loop cancelled at attempt %d",
                    attempt,
                )
                raise

    def _matches_existing(self, existing: Optional[Dict]) -> bool:
        """True iff ``existing`` (parsed from ``openclaw mcp show``)
        matches the registrar's configured payload exactly."""
        if not existing or self._command is None:
            return False
        existing_cmd = existing.get("command")
        existing_args = existing.get("args") or []
        existing_env = existing.get("env") or {}
        if existing_cmd != self._command:
            return False
        if list(existing_args) != self._args:
            return False
        if dict(existing_env) != self._env:
            return False
        return True


__all__ = [
    "RegistrationResult",
    "UltronMcpRegistrar",
]
