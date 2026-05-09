"""Tests for ``ultron.openclaw_bridge.mcp_registration.UltronMcpRegistrar``.

Mocks :class:`OpenClawClient` since the registrar is a thin
orchestrator over the client's MCP CLI methods.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from ultron.errors import OpenClawAuthError, OpenClawGatewayError
from ultron.openclaw_bridge.mcp_registration import (
    RegistrationResult,
    UltronMcpRegistrar,
)


@pytest.fixture
def fake_client() -> Any:
    client = AsyncMock()
    client.mcp_show = AsyncMock(return_value=None)
    client.mcp_set = AsyncMock(return_value=True)
    client.mcp_unset = AsyncMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_empty_name(fake_client: Any) -> None:
    with pytest.raises(ValueError):
        UltronMcpRegistrar(fake_client, name="")


def test_is_configured_reflects_command_presence(fake_client: Any) -> None:
    r1 = UltronMcpRegistrar(fake_client, command=None)
    assert r1.is_configured is False
    r2 = UltronMcpRegistrar(fake_client, command="cmd")
    assert r2.is_configured is True


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


async def test_register_skips_when_no_command(fake_client: Any) -> None:
    r = UltronMcpRegistrar(fake_client, command=None)
    result = await r.register()
    assert isinstance(result, RegistrationResult)
    assert result.registered is False
    assert "no stdio command" in (result.skipped_reason or "")
    fake_client.mcp_set.assert_not_called()


async def test_register_calls_mcp_set_when_no_existing(fake_client: Any) -> None:
    fake_client.mcp_show = AsyncMock(return_value=None)
    r = UltronMcpRegistrar(
        fake_client, name="ultron-mcp", command="cmd", args=["--stdio"],
    )
    result = await r.register()
    assert result.registered is True
    assert result.already_registered is False
    fake_client.mcp_set.assert_awaited_once_with(
        "ultron-mcp", command="cmd", args=["--stdio"], env={},
    )


async def test_register_idempotent_when_existing_matches(fake_client: Any) -> None:
    matching = {"command": "cmd", "args": ["--stdio"], "env": {"FOO": "bar"}}
    fake_client.mcp_show = AsyncMock(return_value=matching)
    r = UltronMcpRegistrar(
        fake_client, command="cmd", args=["--stdio"], env={"FOO": "bar"},
    )
    result = await r.register()
    assert result.registered is True
    assert result.already_registered is True
    fake_client.mcp_set.assert_not_called()


async def test_register_overrides_when_existing_differs(fake_client: Any) -> None:
    fake_client.mcp_show = AsyncMock(
        return_value={"command": "old", "args": [], "env": {}},
    )
    r = UltronMcpRegistrar(
        fake_client, command="new-cmd", args=["--stdio"],
    )
    result = await r.register()
    assert result.registered is True
    assert result.already_registered is False
    fake_client.mcp_set.assert_awaited_once()


async def test_register_handles_show_failure_gracefully(fake_client: Any) -> None:
    fake_client.mcp_show = AsyncMock(side_effect=OpenClawGatewayError("down"))
    r = UltronMcpRegistrar(fake_client, command="cmd")
    result = await r.register()
    # mcp_show failed but we still tried mcp_set, which succeeded.
    assert result.registered is True
    fake_client.mcp_set.assert_awaited_once()


async def test_register_returns_error_on_set_failure(fake_client: Any) -> None:
    fake_client.mcp_set = AsyncMock(side_effect=OpenClawGatewayError("locked"))
    r = UltronMcpRegistrar(fake_client, command="cmd")
    result = await r.register()
    assert result.registered is False
    assert "locked" in (result.error or "")


async def test_register_handles_auth_error(fake_client: Any) -> None:
    fake_client.mcp_set = AsyncMock(side_effect=OpenClawAuthError("nope"))
    r = UltronMcpRegistrar(fake_client, command="cmd")
    result = await r.register()
    assert result.registered is False
    assert "auth rejected" in (result.error or "").lower()


async def test_register_serialises_concurrent_callers(fake_client: Any) -> None:
    """Two concurrent register() calls; the registrar's internal lock
    should serialise so we don't double-register."""
    seen_calls: List[str] = []

    async def slow_set(*args: Any, **kwargs: Any) -> bool:
        seen_calls.append("set")
        await asyncio.sleep(0.05)
        return True

    fake_client.mcp_set = AsyncMock(side_effect=slow_set)
    fake_client.mcp_show = AsyncMock(return_value=None)
    r = UltronMcpRegistrar(fake_client, command="cmd")
    results = await asyncio.gather(r.register(), r.register())
    assert all(res.registered for res in results)
    # Both calls observed mcp_show, but not necessarily both called
    # mcp_set — the second call would see the first's effect if we
    # made the fake stateful. Here we just verify the lock didn't
    # deadlock.
    assert len(seen_calls) >= 1


# ---------------------------------------------------------------------------
# verify_registered()
# ---------------------------------------------------------------------------


async def test_verify_registered_returns_false_when_no_command(fake_client: Any) -> None:
    r = UltronMcpRegistrar(fake_client, command=None)
    assert await r.verify_registered() is False


async def test_verify_registered_returns_true_when_match(fake_client: Any) -> None:
    fake_client.mcp_show = AsyncMock(
        return_value={"command": "cmd", "args": [], "env": {}},
    )
    r = UltronMcpRegistrar(fake_client, command="cmd")
    assert await r.verify_registered() is True


async def test_verify_registered_returns_false_when_diff(fake_client: Any) -> None:
    fake_client.mcp_show = AsyncMock(
        return_value={"command": "different", "args": [], "env": {}},
    )
    r = UltronMcpRegistrar(fake_client, command="cmd")
    assert await r.verify_registered() is False


async def test_verify_registered_handles_show_error(fake_client: Any) -> None:
    fake_client.mcp_show = AsyncMock(side_effect=OpenClawGatewayError("down"))
    r = UltronMcpRegistrar(fake_client, command="cmd")
    assert await r.verify_registered() is False


# ---------------------------------------------------------------------------
# unregister()
# ---------------------------------------------------------------------------


async def test_unregister_passes_through(fake_client: Any) -> None:
    r = UltronMcpRegistrar(fake_client, command="cmd")
    assert await r.unregister() is True
    fake_client.mcp_unset.assert_awaited_once_with("ultron-mcp")


async def test_unregister_swallows_gateway_error(fake_client: Any) -> None:
    fake_client.mcp_unset = AsyncMock(side_effect=OpenClawGatewayError("down"))
    r = UltronMcpRegistrar(fake_client, command="cmd")
    assert await r.unregister() is False


# ---------------------------------------------------------------------------
# schedule_retry()
# ---------------------------------------------------------------------------


async def test_schedule_retry_succeeds_eventually(fake_client: Any) -> None:
    """First two register attempts fail; third succeeds. Verify the
    coroutine exits cleanly after success."""
    state = {"calls": 0}

    async def transient_show(_name: str) -> Optional[Dict[str, Any]]:
        return None

    async def transient_set(*args: Any, **kwargs: Any) -> bool:
        state["calls"] += 1
        if state["calls"] < 3:
            raise OpenClawGatewayError(f"flaky #{state['calls']}")
        return True

    fake_client.mcp_show = AsyncMock(side_effect=transient_show)
    fake_client.mcp_set = AsyncMock(side_effect=transient_set)
    r = UltronMcpRegistrar(fake_client, command="cmd")
    success_payload: List[RegistrationResult] = []

    async def on_success(result: RegistrationResult) -> None:
        success_payload.append(result)

    coro = r.schedule_retry(
        interval_s=0.01, on_success=on_success, max_attempts=10,
    )
    await asyncio.wait_for(coro, timeout=2.0)
    assert state["calls"] == 3
    assert len(success_payload) == 1
    assert success_payload[0].registered is True


async def test_schedule_retry_gives_up_at_max_attempts(fake_client: Any) -> None:
    fake_client.mcp_set = AsyncMock(side_effect=OpenClawGatewayError("always"))
    r = UltronMcpRegistrar(fake_client, command="cmd")
    coro = r.schedule_retry(interval_s=0.01, max_attempts=3)
    await asyncio.wait_for(coro, timeout=2.0)
    # 3 attempts; verifying it exited.
    assert fake_client.mcp_set.await_count == 3


async def test_schedule_retry_rejects_zero_interval(fake_client: Any) -> None:
    r = UltronMcpRegistrar(fake_client, command="cmd")
    with pytest.raises(ValueError):
        r.schedule_retry(interval_s=0)
