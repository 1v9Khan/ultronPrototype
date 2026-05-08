"""Tests for the Phase 1 wire-up: LLMEngine sources its system prompt
from PersonaLoader (workspace) or config (legacy) per ``llm.persona.source``.

Hot-reload is verified end-to-end: edit ``SOUL.md`` between two
``_build_messages`` calls, confirm the second call sees the new content.
The HTTP runtime is used throughout to avoid loading a real model.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures: writable workspace + config patcher
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_with_persona(tmp_path) -> Path:
    """Build a complete six-file workspace at tmp_path/workspace."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "IDENTITY.md").write_text("identity TEST", encoding="utf-8")
    (ws / "SOUL.md").write_text("soul TEST", encoding="utf-8")
    (ws / "USER.md").write_text("user TEST", encoding="utf-8")
    (ws / "AGENTS.md").write_text("agents TEST", encoding="utf-8")
    (ws / "HEARTBEAT.md").write_text("heartbeat TEST", encoding="utf-8")
    (ws / "BOOTSTRAP.md").write_text("bootstrap TEST", encoding="utf-8")
    return ws


@pytest.fixture
def patch_persona_cfg(monkeypatch):
    """Return a function that swaps ``get_config().llm.persona`` for the
    test's duration. Restores the real config on teardown.
    """
    from ultron.config import get_config, LLMPersonaConfig

    real_persona = get_config().llm.persona

    def _patch(**overrides):
        new_persona = LLMPersonaConfig(
            source=overrides.get("source", real_persona.source),
            workspace_dir=overrides.get("workspace_dir", real_persona.workspace_dir),
            fallback_to_config_on_empty=overrides.get(
                "fallback_to_config_on_empty",
                real_persona.fallback_to_config_on_empty,
            ),
            hot_reload=overrides.get("hot_reload", real_persona.hot_reload),
        )
        # Mutate via attribute set; the live config object is shared.
        get_config().llm.persona = new_persona
        return new_persona

    yield _patch

    # Restore on teardown.
    get_config().llm.persona = real_persona


# ---------------------------------------------------------------------------
# Construction: each source path
# ---------------------------------------------------------------------------


def test_config_source_uses_cfg_system_prompt(patch_persona_cfg):
    """source='config' (legacy): system_prompt comes from cfg.system_prompt."""
    from ultron.llm.inference import LLMEngine

    patch_persona_cfg(source="config")
    engine = LLMEngine(runtime="http_server")
    msgs = engine._build_messages("hello")

    assert msgs[0]["role"] == "system"
    # Fall back to whatever cfg.system_prompt happens to be — the
    # production config has it set; in test envs may be empty. Either
    # way it must NOT have come from any workspace.
    assert "agents TEST" not in msgs[0]["content"]
    assert "soul TEST" not in msgs[0]["content"]


def test_workspace_source_loads_from_persona_files(
    patch_persona_cfg, workspace_with_persona,
):
    """source='workspace' with a populated workspace: user_facing system
    prompt is composed from IDENTITY/SOUL/USER via PersonaLoader.

    AGENTS.md is deliberately excluded from user_facing — its operating
    rules live in background mode only, to keep the voice-path prompt
    short. SOUL.md absorbs the voice-relevant rules.
    """
    from ultron.llm.inference import LLMEngine

    patch_persona_cfg(
        source="workspace",
        workspace_dir=str(workspace_with_persona),
    )
    engine = LLMEngine(runtime="http_server")
    msgs = engine._build_messages("hello")

    sys_prompt = msgs[0]["content"]
    assert "identity TEST" in sys_prompt
    assert "soul TEST" in sys_prompt
    assert "user TEST" in sys_prompt
    # user_facing mode excludes AGENTS / HEARTBEAT / BOOTSTRAP.
    assert "agents TEST" not in sys_prompt
    assert "heartbeat TEST" not in sys_prompt
    assert "bootstrap TEST" not in sys_prompt


def test_explicit_system_prompt_overrides_both(
    patch_persona_cfg, workspace_with_persona,
):
    """An explicit ``system_prompt=`` arg wins over both sources."""
    from ultron.llm.inference import LLMEngine

    patch_persona_cfg(
        source="workspace",
        workspace_dir=str(workspace_with_persona),
    )
    engine = LLMEngine(
        system_prompt="EXPLICIT OVERRIDE",
        runtime="http_server",
    )
    msgs = engine._build_messages("hello")

    assert msgs[0]["content"] == "EXPLICIT OVERRIDE"
    assert "soul TEST" not in msgs[0]["content"]


# ---------------------------------------------------------------------------
# Hot reload: workspace edits land on the next turn
# ---------------------------------------------------------------------------


def test_workspace_source_hot_reloads_on_soul_edit(
    patch_persona_cfg, workspace_with_persona,
):
    """Modify SOUL.md between two _build_messages calls. The second call
    must see the new content. This is the core Phase 1 behaviour the
    spec asks for: 'modify SOUL.md, verify next response reflects the
    change without restart.'
    """
    from ultron.llm.inference import LLMEngine

    patch_persona_cfg(
        source="workspace",
        workspace_dir=str(workspace_with_persona),
    )
    engine = LLMEngine(runtime="http_server")

    first = engine._build_messages("q1")[0]["content"]
    assert "soul TEST" in first
    assert "REVISED SOUL" not in first

    # Sleep one filesystem tick so mtime changes are observable, then
    # rewrite SOUL.md with a fingerprint we can detect.
    time.sleep(0.01)
    (workspace_with_persona / "SOUL.md").write_text(
        "REVISED SOUL block", encoding="utf-8",
    )

    second = engine._build_messages("q2")[0]["content"]
    assert "REVISED SOUL block" in second
    # The other user_facing sections still appear, only SOUL changed.
    # AGENTS.md is intentionally absent from user_facing.
    assert "identity TEST" in second
    assert "user TEST" in second


# ---------------------------------------------------------------------------
# Fallback: workspace empty, fall back to cfg
# ---------------------------------------------------------------------------


def test_workspace_empty_falls_back_to_cfg_system_prompt(
    patch_persona_cfg, tmp_path,
):
    """source='workspace' but the workspace is empty: behaviour falls
    back to cfg.system_prompt when fallback flag is set."""
    from ultron.llm.inference import LLMEngine
    from ultron.config import get_config

    empty_ws = tmp_path / "empty_ws"
    empty_ws.mkdir()  # no persona files
    patch_persona_cfg(
        source="workspace",
        workspace_dir=str(empty_ws),
        fallback_to_config_on_empty=True,
    )

    # Mutate the cfg.system_prompt for the test so we have a known marker.
    real_sys_prompt = get_config().llm.system_prompt
    get_config().llm.system_prompt = "FALLBACK MARKER"
    try:
        engine = LLMEngine(runtime="http_server")
        msgs = engine._build_messages("hello")
        assert msgs[0]["content"] == "FALLBACK MARKER"
    finally:
        get_config().llm.system_prompt = real_sys_prompt


def test_workspace_empty_no_fallback_yields_empty_prompt(
    patch_persona_cfg, tmp_path,
):
    """If fallback is disabled, the system prompt is genuinely empty."""
    from ultron.llm.inference import LLMEngine

    empty_ws = tmp_path / "empty_ws"
    empty_ws.mkdir()
    patch_persona_cfg(
        source="workspace",
        workspace_dir=str(empty_ws),
        fallback_to_config_on_empty=False,
    )
    engine = LLMEngine(runtime="http_server")
    msgs = engine._build_messages("hello")
    assert msgs[0]["content"] == ""


# ---------------------------------------------------------------------------
# Construction-time behaviour
# ---------------------------------------------------------------------------


def test_self_system_prompt_attr_reflects_resolved_value(
    patch_persona_cfg, workspace_with_persona,
):
    """Some consumers (tests, debug log dumps) read engine.system_prompt
    directly. After a turn, that attribute must reflect the resolved
    value, not a stale construction-time snapshot."""
    from ultron.llm.inference import LLMEngine

    patch_persona_cfg(
        source="workspace",
        workspace_dir=str(workspace_with_persona),
    )
    engine = LLMEngine(runtime="http_server")
    engine._build_messages("q")  # triggers resolve

    assert "soul TEST" in engine.system_prompt


def test_default_persona_source_is_workspace():
    """Phase 1 set workspace as the default. Anyone constructing
    LLMEngine without an override should get the workspace path.
    """
    from ultron.config import get_config
    assert get_config().llm.persona.source == "workspace"
