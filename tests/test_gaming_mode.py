"""V1-gap A1: GamingModeManager tests.

The manager wraps an :class:`OpenClawClient` and toggles plugins via
``plugins enable / disable``. These tests use a stub client that records
calls and returns scripted results so we can verify state transitions
without spinning up a real OpenClaw subprocess.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from ultron.openclaw_routing.gaming_mode import (
    GamingModeManager,
    GamingModeReport,
    GamingModeStatus,
)


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------


@dataclass
class _StubResult:
    plugin_id: str
    action: str
    success: bool = True
    error: Optional[str] = None


class _StubClient:
    def __init__(
        self,
        *,
        succeed_for: Optional[List[str]] = None,
        fail_for: Optional[Dict[str, str]] = None,
    ) -> None:
        self.calls: List[Dict[str, str]] = []
        self._succeed_for = succeed_for
        self._fail_for = fail_for or {}

    async def enable_plugin(self, plugin_id):
        self.calls.append({"plugin_id": plugin_id, "action": "enable"})
        if plugin_id in self._fail_for:
            return _StubResult(
                plugin_id=plugin_id, action="enable",
                success=False, error=self._fail_for[plugin_id],
            )
        return _StubResult(plugin_id=plugin_id, action="enable", success=True)

    async def disable_plugin(self, plugin_id):
        self.calls.append({"plugin_id": plugin_id, "action": "disable"})
        if plugin_id in self._fail_for:
            return _StubResult(
                plugin_id=plugin_id, action="disable",
                success=False, error=self._fail_for[plugin_id],
            )
        if self._succeed_for is not None and plugin_id not in self._succeed_for:
            return _StubResult(
                plugin_id=plugin_id, action="disable",
                success=False,
                error=f"plugin {plugin_id!r} is not installed",
            )
        return _StubResult(plugin_id=plugin_id, action="disable", success=True)


# ---------------------------------------------------------------------------
# Engage / disengage
# ---------------------------------------------------------------------------


def test_engage_disables_all_configured_plugins(tmp_path):
    client = _StubClient()
    mgr = GamingModeManager(
        client=client,
        plugins_to_disable=["desktop-control", "windows-control"],
        log_path=tmp_path / "gaming_mode.jsonl",
    )
    assert mgr.status() == GamingModeStatus.IDLE

    report = asyncio.run(mgr.engage())

    assert report.status == GamingModeStatus.ENGAGED
    assert report.action == "engage"
    assert report.all_plugin_actions_succeeded is True
    assert [c["plugin_id"] for c in client.calls] == [
        "desktop-control", "windows-control",
    ]
    assert all(c["action"] == "disable" for c in client.calls)
    assert mgr.status() == GamingModeStatus.ENGAGED


def test_disengage_enables_only_previously_disabled_plugins(tmp_path):
    client = _StubClient()
    mgr = GamingModeManager(
        client=client,
        plugins_to_disable=["desktop-control", "windows-control"],
        log_path=tmp_path / "gaming_mode.jsonl",
    )
    asyncio.run(mgr.engage())
    client.calls.clear()

    asyncio.run(mgr.disengage())
    assert [c["plugin_id"] for c in client.calls] == [
        "desktop-control", "windows-control",
    ]
    assert all(c["action"] == "enable" for c in client.calls)
    assert mgr.status() == GamingModeStatus.IDLE


def test_engage_when_already_engaged_is_idempotent():
    client = _StubClient()
    mgr = GamingModeManager(
        client=client, plugins_to_disable=["desktop-control"],
    )
    asyncio.run(mgr.engage())
    client.calls.clear()
    report = asyncio.run(mgr.engage())
    assert report.note == "already engaged"
    assert client.calls == []
    assert mgr.status() == GamingModeStatus.ENGAGED


def test_disengage_when_idle_is_idempotent():
    client = _StubClient()
    mgr = GamingModeManager(
        client=client, plugins_to_disable=["desktop-control"],
    )
    report = asyncio.run(mgr.disengage())
    assert report.note == "already idle"
    assert client.calls == []
    assert mgr.status() == GamingModeStatus.IDLE


def test_engage_proceeds_when_one_plugin_fails():
    client = _StubClient(
        fail_for={"windows-control": "plugin not installed"},
    )
    mgr = GamingModeManager(
        client=client,
        plugins_to_disable=["desktop-control", "windows-control"],
    )
    report = asyncio.run(mgr.engage())
    assert report.status == GamingModeStatus.ENGAGED
    assert not report.all_plugin_actions_succeeded
    # Both calls were attempted.
    assert len(client.calls) == 2


def test_engage_skips_disengage_for_failed_plugins():
    """If a plugin failed during engage, disengage shouldn't try to
    re-enable it (we never disabled it)."""
    client = _StubClient(
        fail_for={"windows-control": "plugin not installed"},
    )
    mgr = GamingModeManager(
        client=client,
        plugins_to_disable=["desktop-control", "windows-control"],
    )
    asyncio.run(mgr.engage())
    client.calls.clear()
    asyncio.run(mgr.disengage())
    # Only ``desktop-control`` is restored.
    assert [c["plugin_id"] for c in client.calls] == ["desktop-control"]


def test_engage_writes_log_row(tmp_path):
    log_path = tmp_path / "gaming_mode.jsonl"
    client = _StubClient()
    mgr = GamingModeManager(
        client=client,
        plugins_to_disable=["desktop-control"],
        log_path=log_path,
    )
    asyncio.run(mgr.engage())
    assert log_path.is_file()
    line = log_path.read_text(encoding="utf-8").splitlines()[-1]
    record = json.loads(line)
    assert record["action"] == "engage"
    assert record["status"] == "engaged"
    assert isinstance(record["plugin_states"], list)


def test_no_client_returns_clear_error():
    mgr = GamingModeManager(client=None, plugins_to_disable=["x"])
    report = asyncio.run(mgr.engage())
    assert report.status == GamingModeStatus.ENGAGED
    assert not report.all_plugin_actions_succeeded
    assert "no openclaw client" in (report.plugin_states[0].error or "").lower()


def test_engage_swallows_client_exception():
    class _BoomClient:
        async def disable_plugin(self, plugin_id):
            raise RuntimeError("simulated transport failure")

        async def enable_plugin(self, plugin_id):
            return None

    mgr = GamingModeManager(
        client=_BoomClient(), plugins_to_disable=["desktop-control"],
    )
    # Must not raise.
    report = asyncio.run(mgr.engage())
    assert report.plugin_states[0].success is False
    assert "simulated" in (report.plugin_states[0].error or "")


def test_status_reflects_state_machine():
    client = _StubClient()
    mgr = GamingModeManager(
        client=client, plugins_to_disable=["desktop-control"],
    )
    assert mgr.status() == GamingModeStatus.IDLE
    asyncio.run(mgr.engage())
    assert mgr.status() == GamingModeStatus.ENGAGED
    asyncio.run(mgr.disengage())
    assert mgr.status() == GamingModeStatus.IDLE
