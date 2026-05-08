"""Tests for openclaw_bridge.persona.PersonaLoader."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ultron.openclaw_bridge.persona import (
    PersonaBundle,
    PersonaFile,
    PersonaLoader,
    default_workspace_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_full_set(workspace: Path) -> None:
    """Write all six persona files with distinguishable, non-empty content."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "IDENTITY.md").write_text("identity body", encoding="utf-8")
    (workspace / "SOUL.md").write_text("soul body", encoding="utf-8")
    (workspace / "USER.md").write_text("user body", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("agents body", encoding="utf-8")
    (workspace / "HEARTBEAT.md").write_text("heartbeat body", encoding="utf-8")
    (workspace / "BOOTSTRAP.md").write_text("bootstrap body", encoding="utf-8")


@pytest.fixture
def workspace(tmp_path) -> Path:
    ws = tmp_path / "workspace"
    _write_full_set(ws)
    return ws


# ---------------------------------------------------------------------------
# default_workspace_dir
# ---------------------------------------------------------------------------


def test_default_workspace_dir_uses_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("ULTRON_OPENCLAW_WORKSPACE", str(tmp_path / "custom"))
    assert default_workspace_dir() == tmp_path / "custom"


def test_default_workspace_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("ULTRON_OPENCLAW_WORKSPACE", raising=False)
    expected = Path.home() / ".openclaw" / "workspace"
    assert default_workspace_dir() == expected


# ---------------------------------------------------------------------------
# Loading + bundle shape
# ---------------------------------------------------------------------------


def test_load_returns_bundle_with_all_six_files(workspace):
    loader = PersonaLoader(workspace)
    bundle = loader.load()

    assert isinstance(bundle, PersonaBundle)
    assert bundle.workspace_dir == workspace
    assert set(bundle.files.keys()) == {
        "IDENTITY.md", "SOUL.md", "USER.md",
        "AGENTS.md", "HEARTBEAT.md", "BOOTSTRAP.md",
    }
    for f in bundle.files.values():
        assert isinstance(f, PersonaFile)
        assert f.size_bytes > 0
        assert not f.is_empty


def test_load_handles_missing_files_gracefully(tmp_path):
    """Three files present, three missing — loader returns empty content
    for the missing ones and logs a warning. No crash."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "SOUL.md").write_text("soul", encoding="utf-8")
    (ws / "IDENTITY.md").write_text("identity", encoding="utf-8")
    (ws / "AGENTS.md").write_text("agents", encoding="utf-8")

    loader = PersonaLoader(ws)
    bundle = loader.load()

    assert bundle.files["SOUL.md"].content == "soul"
    assert bundle.files["USER.md"].content == ""
    assert bundle.files["USER.md"].is_empty
    assert bundle.files["HEARTBEAT.md"].is_empty
    assert bundle.files["BOOTSTRAP.md"].is_empty


def test_load_handles_corrupt_bytes_with_replace_errors(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    # Mix valid UTF-8 with a stray invalid continuation byte.
    (ws / "SOUL.md").write_bytes(b"valid \xff bytes")
    for name in ("IDENTITY.md", "USER.md", "AGENTS.md", "HEARTBEAT.md", "BOOTSTRAP.md"):
        (ws / name).write_text("ok", encoding="utf-8")

    loader = PersonaLoader(ws)
    bundle = loader.load()

    soul = bundle.files["SOUL.md"]
    assert "valid" in soul.content
    assert "bytes" in soul.content
    # Replacement char (or similar) appears, but no exception.
    assert len(soul.content) > 0


def test_current_returns_none_before_first_load(workspace):
    loader = PersonaLoader(workspace)
    assert loader.current is None
    loader.load()
    assert loader.current is not None


# ---------------------------------------------------------------------------
# System-prompt composition
# ---------------------------------------------------------------------------


def test_user_facing_mode_default_composition(workspace):
    """Default mode (user_facing): IDENTITY → SOUL → USER.

    AGENTS.md is deliberately NOT in user_facing — operating rules
    bloat the voice-path system prompt and cost ~+200 ms TTFT.
    Voice-relevant rules (do-not-lecture, uncertainty handling) live
    in SOUL.md instead.
    """
    loader = PersonaLoader(workspace)
    prompt = loader.get_system_prompt()

    parts = prompt.split("\n\n")
    assert parts == [
        "identity body",
        "soul body",
        "user body",
    ]
    assert "agents body" not in prompt
    assert "heartbeat body" not in prompt
    assert "bootstrap body" not in prompt
    assert not prompt.startswith("You are an internal worker")


def test_background_mode_strips_character_keeps_agents(workspace):
    """background mode: AGENTS only, prefixed with internal-worker framing.

    No SOUL.md (character) leakage, no IDENTITY.md (character) leakage,
    no USER.md (personal) leakage. Just operating rules + the prefix.
    """
    loader = PersonaLoader(workspace)
    prompt = loader.get_system_prompt(mode="background")

    assert "soul body" not in prompt
    assert "identity body" not in prompt
    assert "user body" not in prompt
    assert "heartbeat body" not in prompt
    assert "bootstrap body" not in prompt
    assert "agents body" in prompt
    assert prompt.startswith("You are an internal worker")
    assert "NO_REPLY" in prompt  # explicit "do not emit NO_REPLY" guidance


def test_heartbeat_mode_includes_only_heartbeat_with_prefix(workspace):
    loader = PersonaLoader(workspace)
    prompt = loader.get_system_prompt(mode="heartbeat")

    assert "heartbeat body" in prompt
    assert "soul body" not in prompt
    assert "agents body" not in prompt
    assert "HEARTBEAT_OK" in prompt  # mode prefix mentions the OK token


def test_bootstrap_mode_includes_only_bootstrap_with_prefix(workspace):
    loader = PersonaLoader(workspace)
    prompt = loader.get_system_prompt(mode="bootstrap")

    assert "bootstrap body" in prompt
    assert "soul body" not in prompt
    assert "agents body" not in prompt


def test_unknown_mode_raises_value_error(workspace):
    loader = PersonaLoader(workspace)
    with pytest.raises(ValueError, match="unknown PersonaLoader mode"):
        loader.get_system_prompt(mode="nonexistent")  # type: ignore[arg-type]


def test_user_facing_skips_empty_soul_but_renders_other_sections(tmp_path):
    """Empty SOUL.md doesn't break user_facing — IDENTITY/USER still render.

    AGENTS.md is not in user_facing's file list, so it's never
    included regardless of content.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "IDENTITY.md").write_text("identity body", encoding="utf-8")
    (ws / "SOUL.md").write_text("   \n   ", encoding="utf-8")  # whitespace only
    (ws / "USER.md").write_text("user body", encoding="utf-8")
    (ws / "AGENTS.md").write_text("agents body", encoding="utf-8")
    (ws / "HEARTBEAT.md").write_text("", encoding="utf-8")
    (ws / "BOOTSTRAP.md").write_text("", encoding="utf-8")

    loader = PersonaLoader(ws)
    prompt = loader.get_system_prompt()

    assert "identity body" in prompt
    assert "user body" in prompt
    assert "agents body" not in prompt  # not in user_facing's mode
    # Soul should be skipped because content is whitespace-only.
    assert prompt.split("\n\n") == [
        "identity body", "user body",
    ]


def test_html_comment_only_files_are_treated_as_empty(tmp_path):
    """A file whose content is nothing but HTML comments (used for
    human-reader documentation) must not bloat the rendered prompt."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "IDENTITY.md").write_text("identity body", encoding="utf-8")
    (ws / "SOUL.md").write_text("soul body", encoding="utf-8")
    (ws / "USER.md").write_text(
        "<!-- auto-populated from facts; empty for now -->",
        encoding="utf-8",
    )
    (ws / "AGENTS.md").write_text("agents body", encoding="utf-8")
    (ws / "HEARTBEAT.md").write_text("", encoding="utf-8")
    (ws / "BOOTSTRAP.md").write_text("", encoding="utf-8")

    loader = PersonaLoader(ws)
    prompt = loader.get_system_prompt()

    assert "identity body" in prompt
    assert "soul body" in prompt
    # The HTML comment is stripped; USER.md effectively renders empty.
    assert "auto-populated" not in prompt
    assert "<!--" not in prompt


def test_background_mode_with_missing_agents_still_emits_prefix(tmp_path):
    """Even if AGENTS.md is missing, the background prefix anchors the
    worker so it doesn't fall back to character mode."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "IDENTITY.md").write_text("identity body", encoding="utf-8")
    (ws / "SOUL.md").write_text("soul body", encoding="utf-8")

    loader = PersonaLoader(ws)
    prompt = loader.get_system_prompt(mode="background")

    assert prompt.startswith("You are an internal worker")
    assert "soul body" not in prompt
    assert "identity body" not in prompt


def test_user_facing_mode_with_all_files_missing_returns_empty(tmp_path):
    ws = tmp_path / "empty"
    ws.mkdir()
    loader = PersonaLoader(ws)
    assert loader.get_system_prompt() == ""


# ---------------------------------------------------------------------------
# refresh_if_stale
# ---------------------------------------------------------------------------


def test_refresh_if_stale_reuses_cache_when_unchanged(workspace):
    loader = PersonaLoader(workspace)
    first = loader.load()
    second = loader.refresh_if_stale()
    assert first is second  # exact same object — cache hit


def test_refresh_if_stale_reloads_when_file_modified(workspace):
    loader = PersonaLoader(workspace)
    loader.load()
    first_prompt = loader.get_system_prompt()

    # Modify SOUL.md. Sleep one filesystem tick so mtime changes
    # observably even on filesystems with second-level resolution.
    time.sleep(0.01)
    (workspace / "SOUL.md").write_text("soul body REVISED", encoding="utf-8")
    # On Windows mtime resolution can be coarse; force a different size too.
    assert (workspace / "SOUL.md").read_text(encoding="utf-8") != "soul body"

    second_prompt = loader.get_system_prompt()
    assert "REVISED" in second_prompt
    assert second_prompt != first_prompt


def test_refresh_if_stale_reloads_when_first_call(tmp_path):
    """If never loaded, refresh_if_stale should call load() once."""
    ws = tmp_path / "ws"
    _write_full_set(ws)

    loader = PersonaLoader(ws)
    assert loader.current is None
    bundle = loader.refresh_if_stale()
    assert loader.current is bundle


# ---------------------------------------------------------------------------
# Bundle fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_across_loads_when_files_unchanged(workspace):
    loader = PersonaLoader(workspace)
    a = loader.load()
    b = loader.load()
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_when_a_file_changes(workspace):
    loader = PersonaLoader(workspace)
    a = loader.load()
    time.sleep(0.01)
    (workspace / "AGENTS.md").write_text("agents body REVISED", encoding="utf-8")
    b = loader.load()
    assert a.fingerprint != b.fingerprint
