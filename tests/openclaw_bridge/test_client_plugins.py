"""V1-gap A1: OpenClawClient.enable_plugin / disable_plugin / list_plugins.

These wrap the openclaw CLI's ``plugins`` subcommand. We test them by
patching :meth:`OpenClawClient._run_cli` so we can script CLI return
codes / stdout / stderr without spawning real subprocesses.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional

import pytest

from ultron.openclaw_bridge.client import (
    CliResult,
    OpenClawClient,
    PluginInfo,
    PluginToggleResult,
)


@pytest.fixture
def client(monkeypatch):
    """Build an OpenClawClient with a mocked CLI path so init succeeds."""
    monkeypatch.setattr(
        "ultron.openclaw_bridge.client.discover_cli",
        lambda override=None: "/fake/openclaw",
    )
    return OpenClawClient()


def _patch_run_cli(client, *, returncode: int, stdout: str = "", stderr: str = ""):
    captured = {"args": None}

    async def _stub(args, *, timeout_s=None):
        captured["args"] = list(args)
        return CliResult(
            returncode=returncode, stdout=stdout, stderr=stderr,
            duration_s=0.0,
        )

    client._run_cli = _stub  # type: ignore[assignment]
    return captured


# ---------------------------------------------------------------------------
# enable_plugin / disable_plugin
# ---------------------------------------------------------------------------


def test_enable_plugin_invokes_correct_cli(client):
    captured = _patch_run_cli(client, returncode=0)
    result = asyncio.run(client.enable_plugin("desktop-control"))
    assert isinstance(result, PluginToggleResult)
    assert result.success is True
    assert result.action == "enable"
    assert result.plugin_id == "desktop-control"
    assert captured["args"] == ["plugins", "enable", "desktop-control"]


def test_disable_plugin_invokes_correct_cli(client):
    captured = _patch_run_cli(client, returncode=0)
    result = asyncio.run(client.disable_plugin("windows-control"))
    assert result.success is True
    assert result.action == "disable"
    assert captured["args"] == ["plugins", "disable", "windows-control"]


def test_disable_plugin_not_installed_returns_structured_failure(client):
    _patch_run_cli(
        client, returncode=1,
        stderr="plugin 'windows-control' is not installed",
    )
    result = asyncio.run(client.disable_plugin("windows-control"))
    assert result.success is False
    assert "not installed" in (result.error or "").lower()


def test_enable_plugin_unknown_plugin_returns_structured_failure(client):
    _patch_run_cli(
        client, returncode=1,
        stderr="Unknown plugin 'foo'.",
    )
    result = asyncio.run(client.enable_plugin("foo"))
    assert result.success is False
    assert "not installed" in (result.error or "").lower()


def test_disable_plugin_other_failure_passes_error_text(client):
    _patch_run_cli(
        client, returncode=2,
        stderr="some other failure",
    )
    result = asyncio.run(client.disable_plugin("desktop-control"))
    assert result.success is False
    assert "some other failure" in (result.error or "")


def test_enable_plugin_empty_id_raises(client):
    with pytest.raises(ValueError):
        asyncio.run(client.enable_plugin(""))


def test_invalid_action_raises(client):
    with pytest.raises(ValueError):
        asyncio.run(client._toggle_plugin("x", "frobnicate"))


# ---------------------------------------------------------------------------
# list_plugins
# ---------------------------------------------------------------------------


def test_list_plugins_parses_json_array(client):
    payload = [
        {"id": "desktop-control", "name": "Desktop Control",
         "enabled": True, "version": "1.0"},
        {"id": "windows-control", "name": "Windows Control",
         "enabled": False, "version": "0.5"},
    ]
    _patch_run_cli(client, returncode=0, stdout=json.dumps(payload))
    rows = asyncio.run(client.list_plugins())
    assert len(rows) == 2
    assert rows[0].plugin_id == "desktop-control"
    assert rows[0].enabled is True
    assert rows[1].enabled is False


def test_list_plugins_parses_dict_with_plugins_key(client):
    payload = {"plugins": [
        {"id": "x", "name": "X", "enabled": True},
    ]}
    _patch_run_cli(client, returncode=0, stdout=json.dumps(payload))
    rows = asyncio.run(client.list_plugins())
    assert len(rows) == 1
    assert rows[0].plugin_id == "x"


def test_list_plugins_returns_empty_on_nonzero(client):
    _patch_run_cli(client, returncode=1, stderr="failed")
    rows = asyncio.run(client.list_plugins())
    assert rows == []


def test_list_plugins_returns_empty_on_unparseable_json(client):
    _patch_run_cli(client, returncode=0, stdout="not-json")
    rows = asyncio.run(client.list_plugins())
    assert rows == []


def test_list_plugins_returns_empty_on_blank_stdout(client):
    _patch_run_cli(client, returncode=0, stdout="")
    rows = asyncio.run(client.list_plugins())
    assert rows == []


def test_list_plugins_skips_malformed_rows(client):
    payload = [
        "not_a_dict",
        {"id": "x", "name": "X", "enabled": True},
        None,
    ]
    _patch_run_cli(client, returncode=0, stdout=json.dumps(payload))
    rows = asyncio.run(client.list_plugins())
    assert len(rows) == 1
    assert rows[0].plugin_id == "x"


def test_list_plugins_supports_enabled_only_flag(client):
    captured = _patch_run_cli(client, returncode=0, stdout="[]")
    asyncio.run(client.list_plugins(enabled_only=True))
    assert "--enabled" in captured["args"]
