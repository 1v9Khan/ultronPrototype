"""Tests for :mod:`ultron.intent.command_registry`."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ultron.intent.command_registry import (
    Command,
    CommandRegistry,
    DEFAULT_REGISTRY,
    command,
)


@pytest.fixture
def registry() -> CommandRegistry:
    return CommandRegistry()


# ---------------------------------------------------------------------------
# Command dataclass
# ---------------------------------------------------------------------------


def test_command_is_frozen():
    cmd = Command(name="x", description="x")
    with pytest.raises(Exception):
        cmd.name = "y"  # type: ignore[misc]


def test_command_defaults():
    cmd = Command(name="x", description="x")
    assert cmd.phrases == ()
    assert cmd.examples == ()
    assert cmd.tags == frozenset()
    # Calling the default noop handler returns None.
    assert cmd.handler() is None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_adds_command(registry: CommandRegistry):
    cmd = Command(name="alpha", description="The alpha command.")
    assert registry.register(cmd) is True
    assert registry.get("alpha") is cmd


def test_register_rejects_duplicate(registry: CommandRegistry):
    cmd1 = Command(name="alpha", description="first")
    cmd2 = Command(name="alpha", description="second")
    assert registry.register(cmd1) is True
    assert registry.register(cmd2) is False
    assert registry.get("alpha").description == "first"


def test_register_overwrite_replaces(registry: CommandRegistry):
    cmd1 = Command(name="alpha", description="first")
    cmd2 = Command(name="alpha", description="second")
    registry.register(cmd1)
    assert registry.register(cmd2, overwrite=True) is True
    assert registry.get("alpha").description == "second"


def test_register_from_dict(registry: CommandRegistry):
    assert registry.register_from_dict({
        "name": "engage_gaming_mode",
        "description": "Switch to gaming VRAM profile.",
        "phrases": ["engage gaming mode", "switch to gaming mode"],
        "examples": ["I'm about to play Valorant"],
        "tags": ["voice", "vram"],
    }) is True
    cmd = registry.get("engage_gaming_mode")
    assert cmd is not None
    assert "engage gaming mode" in cmd.phrases
    assert "voice" in cmd.tags


def test_register_from_dict_missing_name_returns_false(registry: CommandRegistry):
    assert registry.register_from_dict({"description": "no name"}) is False


def test_register_from_dict_empty_name_returns_false(registry: CommandRegistry):
    assert registry.register_from_dict({
        "name": "  ",
        "description": "blank name",
    }) is False


def test_register_from_dict_ignores_non_string_phrases(registry: CommandRegistry):
    """Coerce + filter — invalid entries are dropped, not crashes."""
    assert registry.register_from_dict({
        "name": "x",
        "description": "x",
        "phrases": ["valid", 123, None, "also valid"],
    }) is True
    cmd = registry.get("x")
    assert "valid" in cmd.phrases
    assert "also valid" in cmd.phrases
    assert len(cmd.phrases) == 2


def test_register_from_json_file(registry: CommandRegistry, tmp_path: Path):
    target = tmp_path / "commands.json"
    target.write_text(json.dumps([
        {"name": "a", "description": "alpha", "phrases": ["alpha phrase"]},
        {"name": "b", "description": "beta", "phrases": ["beta phrase"]},
    ]))
    count = registry.register_from_json_file(target)
    assert count == 2
    assert registry.has("a")
    assert registry.has("b")


def test_register_from_json_missing_file_returns_zero(registry: CommandRegistry, tmp_path: Path):
    assert registry.register_from_json_file(tmp_path / "missing.json") == 0


def test_register_from_json_malformed_returns_zero(registry: CommandRegistry, tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{")
    assert registry.register_from_json_file(bad) == 0


def test_register_from_json_non_array_returns_zero(registry: CommandRegistry, tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "an array"}))
    assert registry.register_from_json_file(bad) == 0


def test_unregister_removes(registry: CommandRegistry):
    registry.register(Command(name="x", description="x"))
    assert registry.unregister("x") is True
    assert registry.unregister("x") is False
    assert registry.get("x") is None


def test_clear_drops_all(registry: CommandRegistry):
    registry.register(Command(name="a", description="a"))
    registry.register(Command(name="b", description="b"))
    registry.clear()
    assert len(registry) == 0


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_has_and_contains(registry: CommandRegistry):
    registry.register(Command(name="x", description="x"))
    assert registry.has("x")
    assert "x" in registry
    assert "missing" not in registry


def test_contains_rejects_non_string(registry: CommandRegistry):
    registry.register(Command(name="x", description="x"))
    assert 42 not in registry  # type: ignore[operator]


def test_list_all_sorted(registry: CommandRegistry):
    registry.register(Command(name="bravo", description="b"))
    registry.register(Command(name="alpha", description="a"))
    registry.register(Command(name="charlie", description="c"))
    names = [c.name for c in registry.list_all()]
    assert names == ["alpha", "bravo", "charlie"]


def test_list_by_tag(registry: CommandRegistry):
    registry.register(Command(
        name="vc",
        description="voice command",
        tags=frozenset({"voice"}),
    ))
    registry.register(Command(
        name="cc",
        description="coding command",
        tags=frozenset({"coding"}),
    ))
    voice_only = registry.list_by_tag("voice")
    assert [c.name for c in voice_only] == ["vc"]


def test_len(registry: CommandRegistry):
    assert len(registry) == 0
    registry.register(Command(name="x", description="x"))
    assert len(registry) == 1


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_match_finds_substring(registry: CommandRegistry):
    registry.register(Command(
        name="gaming",
        description="gaming mode",
        phrases=("engage gaming mode", "switch to gaming"),
    ))
    cmd = registry.match("Hey, please engage gaming mode now.")
    assert cmd is not None
    assert cmd.name == "gaming"


def test_match_case_insensitive(registry: CommandRegistry):
    registry.register(Command(
        name="x",
        description="x",
        phrases=("WAKE WORD",),
    ))
    cmd = registry.match("the wake word is set")
    assert cmd is not None
    assert cmd.name == "x"


def test_match_no_match_returns_none(registry: CommandRegistry):
    registry.register(Command(
        name="x",
        description="x",
        phrases=("hello",),
    ))
    assert registry.match("totally unrelated text") is None


def test_match_empty_input(registry: CommandRegistry):
    registry.register(Command(name="x", description="x", phrases=("hi",)))
    assert registry.match("") is None
    assert registry.match("   ") is None


def test_match_skips_commands_without_phrases(registry: CommandRegistry):
    registry.register(Command(name="x", description="x"))
    assert registry.match("anything") is None


# ---------------------------------------------------------------------------
# Help rendering
# ---------------------------------------------------------------------------


def test_format_help_empty(registry: CommandRegistry):
    out = registry.format_help()
    assert "No commands registered" in out


def test_format_help_includes_name_and_description(registry: CommandRegistry):
    registry.register(Command(
        name="alpha",
        description="The alpha command.",
        phrases=("alpha phrase",),
    ))
    out = registry.format_help()
    assert "**alpha**" in out
    assert "The alpha command." in out
    assert "alpha phrase" in out


def test_format_help_tag_filter(registry: CommandRegistry):
    registry.register(Command(
        name="a",
        description="alpha",
        tags=frozenset({"voice"}),
    ))
    registry.register(Command(
        name="b",
        description="beta",
        tags=frozenset({"coding"}),
    ))
    out = registry.format_help(tag_filter="voice")
    assert "alpha" in out
    assert "beta" not in out


def test_format_help_include_examples(registry: CommandRegistry):
    registry.register(Command(
        name="x",
        description="x",
        examples=("Try this thing",),
    ))
    out = registry.format_help(include_examples=True)
    assert "Try this thing" in out


# ---------------------------------------------------------------------------
# @command decorator
# ---------------------------------------------------------------------------


def test_decorator_registers(registry: CommandRegistry):
    @command(
        name="alpha",
        description="The alpha decorator command.",
        phrases=["alpha phrase"],
        registry=registry,
    )
    def alpha_handler():
        return "alpha"

    cmd = registry.get("alpha")
    assert cmd is not None
    assert cmd.description == "The alpha decorator command."
    assert cmd.handler() == "alpha"


def test_decorator_falls_back_to_docstring(registry: CommandRegistry):
    @command(
        name="docstring_cmd",
        phrases=["doc cmd"],
        registry=registry,
    )
    def doc_handler():
        """A command whose description comes from the docstring."""
        return None

    cmd = registry.get("docstring_cmd")
    assert cmd is not None
    assert "docstring" in cmd.description


def test_decorator_default_registry():
    """Test isolation: register, then clean up."""
    @command(
        name="__test_only_decorator_default__",
        description="for testing",
        registry=None,  # explicitly use DEFAULT_REGISTRY
    )
    def fn():
        return "ok"

    try:
        cmd = DEFAULT_REGISTRY.get("__test_only_decorator_default__")
        assert cmd is not None
        assert cmd.handler() == "ok"
    finally:
        DEFAULT_REGISTRY.unregister("__test_only_decorator_default__")


def test_decorator_rejects_duplicate_silently(registry: CommandRegistry):
    @command(name="x", description="first", registry=registry)
    def first():
        return 1

    @command(name="x", description="second", registry=registry)
    def second():
        return 2

    # Decorator returns the original function regardless of whether
    # registration succeeded.
    assert second() == 2
    # The registry still has the first one.
    assert registry.get("x").description == "first"


# ---------------------------------------------------------------------------
# Thread safety smoke
# ---------------------------------------------------------------------------


def test_concurrent_registration_safe(registry: CommandRegistry):
    """Many threads registering different commands must not crash."""
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        try:
            for i in range(20):
                registry.register(Command(
                    name=f"thread_{idx}_cmd_{i}",
                    description=f"t{idx}c{i}",
                ))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert errors == []
    assert len(registry) == 4 * 20
