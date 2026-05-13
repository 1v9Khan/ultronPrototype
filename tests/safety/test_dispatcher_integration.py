"""Integration tests for the OpenClaw dispatcher's safety wiring.

Confirms ``_runtime_safety_check`` short-circuits dispatches when
a rule fires + lets benign calls through.
"""

from __future__ import annotations

import asyncio

import pytest

from ultron.config import get_config
from ultron.openclaw_routing.dispatcher import OpenClawDispatcher
from ultron.openclaw_routing.intents import (
    FileOpIntent,
    ShellOpIntent,
)
from ultron.safety import set_validator
from ultron.safety.validator import build_validator_from_config


def setup_module(_module):
    # Set up the singleton with the production validator for these
    # tests. Tests that need a permissive validator override per-test.
    set_validator(build_validator_from_config())


def teardown_module(_module):
    set_validator(None)


def _run(coro):
    return asyncio.run(coro)


def test_dispatcher_blocks_file_op_on_protected_path():
    d = OpenClawDispatcher(config=get_config())
    intent = FileOpIntent(
        operation="write",
        path="config.yaml",
        raw_text="write to config",
    )
    result = _run(d.handle_file_operation(intent))
    assert not result.success
    assert "held off" in result.voice_message.lower() or "blocked" in result.voice_message.lower()
    # Verify the validator metadata is present.
    assert result.metadata.get("blocked_by") == "safety_validator"
    assert result.metadata.get("rule_id") == "K1"


def test_dispatcher_blocks_shell_on_dangerous_command():
    d = OpenClawDispatcher(config=get_config())
    intent = ShellOpIntent(
        command="rm -rf /etc",
        raw_text="delete etc",
    )
    result = _run(d.handle_shell_operation(intent))
    assert not result.success
    assert result.metadata.get("blocked_by") == "safety_validator"


def test_dispatcher_allows_benign_sandbox_write_stub():
    """Sandbox-internal file ops pass the runtime validator; the
    dispatcher then falls through to the legacy stub (since the
    OpenClaw bridge isn't wired in this test environment). The
    important thing is the validator didn't block."""
    d = OpenClawDispatcher(config=get_config())
    intent = FileOpIntent(
        operation="write",
        path="data/sandbox/myproj/main.py",
        raw_text="write main.py in the sandbox",
    )
    result = _run(d.handle_file_operation(intent))
    # The dispatcher still returns a stub (the OpenClaw integration
    # isn't live in this test); the key is that it did NOT block
    # via the safety validator.
    assert result.metadata.get("blocked_by") != "safety_validator"


def test_dispatcher_blocks_force_push_to_main():
    d = OpenClawDispatcher(config=get_config())
    intent = ShellOpIntent(
        command="git push --force origin main",
        raw_text="force push",
    )
    result = _run(d.handle_shell_operation(intent))
    assert not result.success
    assert result.metadata.get("blocked_by") == "safety_validator"
    assert result.metadata.get("rule_id") == "F1"
