"""Tests for ultron.safety.ignore."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.safety import ignore as ig


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    ig.reset_ignore_controller_registry()
    yield
    ig.reset_ignore_controller_registry()


def _write_ignore(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Layer compilation + path checks
# ---------------------------------------------------------------------------

class TestLayers:
    def test_no_layers_means_no_ignore(self, tmp_path: Path) -> None:
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        verdict = ctl.check_path("src/a.py")
        assert verdict.ignored is False

    def test_workspace_layer_blocks(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "secrets/\n*.key\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        assert ctl.check_path(tmp_path / "secrets" / "api.txt").ignored is True
        assert ctl.check_path(tmp_path / "private.key").ignored is True
        assert ctl.check_path(tmp_path / "src" / "a.py").ignored is False

    def test_project_layer_blocks(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultron" / ".ultronignore", "data/\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        assert ctl.check_path(tmp_path / "data" / "x.csv").ignored is True
        assert ctl.check_path(tmp_path / "src" / "a.py").ignored is False

    def test_global_layer_blocks(self, tmp_path: Path) -> None:
        global_path = tmp_path / "global" / ".ultronignore"
        _write_ignore(global_path, "*.pem\n")
        ctl = ig.IgnoreController(tmp_path, global_path=global_path)
        assert ctl.check_path(tmp_path / "key.pem").ignored is True

    def test_matched_layer_and_pattern(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "private/\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        v = ctl.check_path(tmp_path / "private" / "x")
        assert v.ignored is True
        assert v.matched_layer == "workspace"
        assert "private" in v.matched_pattern

    def test_mtime_invalidation_drops_stale_cache(self, tmp_path: Path) -> None:
        ignore_file = tmp_path / ".ultronignore"
        _write_ignore(ignore_file, "a/\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        assert ctl.check_path(tmp_path / "a" / "x").ignored is True
        # Update the file to remove the rule.
        import time, os as _os
        time.sleep(0.05)
        ignore_file.write_text("", encoding="utf-8")
        _os.utime(ignore_file, None)
        assert ctl.check_path(tmp_path / "a" / "x").ignored is False

    def test_invalidate_forces_reread(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "a/\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        ctl.check_path(tmp_path / "a")
        ctl.invalidate()
        # Should not crash on the next check.
        assert ctl.check_path(tmp_path / "a" / "x").ignored is True

    def test_configured_files(self, tmp_path: Path) -> None:
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "g")
        files = ctl.configured_files()
        assert files["workspace"] is not None
        assert files["project"] is not None
        assert files["global"] == (tmp_path / "g").resolve()


# ---------------------------------------------------------------------------
# !include directive
# ---------------------------------------------------------------------------

class TestInclude:
    def test_include_concatenates_other_file(self, tmp_path: Path) -> None:
        base = tmp_path / "base.ignore"
        _write_ignore(base, "private/\n")
        _write_ignore(
            tmp_path / ".ultronignore", f"!include {base.name}\n*.key\n",
        )
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        assert ctl.check_path(tmp_path / "private" / "x").ignored is True
        assert ctl.check_path(tmp_path / "a.key").ignored is True

    def test_include_cycle_terminates(self, tmp_path: Path) -> None:
        a = tmp_path / ".ultronignore"
        b = tmp_path / "other"
        _write_ignore(a, "!include other\n*.key\n")
        _write_ignore(b, "!include .ultronignore\n*.pem\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        # Should not infinite-loop and should pick up patterns from both.
        assert ctl.check_path(tmp_path / "a.key").ignored is True
        assert ctl.check_path(tmp_path / "a.pem").ignored is True

    def test_include_missing_file_quiet(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "!include nonexistent\n*.bad\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        assert ctl.check_path(tmp_path / "x.bad").ignored is True


# ---------------------------------------------------------------------------
# filter_paths / is_path_allowed
# ---------------------------------------------------------------------------

class TestFilterPaths:
    def test_filter_paths_keeps_allowed(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "secrets/\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        all_paths = [
            tmp_path / "secrets" / "a.txt",
            tmp_path / "src" / "main.py",
            tmp_path / "secrets" / "b.txt",
            tmp_path / "README.md",
        ]
        kept = ctl.filter_paths(all_paths)
        assert any("main.py" in p for p in kept)
        assert any("README.md" in p for p in kept)
        assert all("secrets" not in p for p in kept)

    def test_is_path_allowed_inverse(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "*.key\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        assert ctl.is_path_allowed(tmp_path / "a.py") is True
        assert ctl.is_path_allowed(tmp_path / "a.key") is False


# ---------------------------------------------------------------------------
# validate_command
# ---------------------------------------------------------------------------

class TestValidateCommand:
    def test_empty_command_allowed(self, tmp_path: Path) -> None:
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        v = ctl.validate_command("")
        assert v.denied_path is None
        v2 = ctl.validate_command("   ")
        assert v2.denied_path is None

    def test_non_reading_command_passes(self, tmp_path: Path) -> None:
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        # echo is not in the read list.
        v = ctl.validate_command("echo hello world")
        assert v.denied_path is None
        assert v.program == "echo"

    def test_cat_with_blocked_path(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "*.key\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        v = ctl.validate_command("cat secrets.key")
        assert v.denied_path == "secrets.key"
        assert v.program == "cat"
        assert "workspace" in v.reason

    def test_cat_with_allowed_path(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "*.key\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        v = ctl.validate_command("cat README.md")
        assert v.denied_path is None

    def test_powershell_aliases_recognised(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "*.key\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        for cmd_form in ("gc secrets.key", "Get-Content secrets.key", "type secrets.key"):
            v = ctl.validate_command(cmd_form)
            assert v.denied_path == "secrets.key"

    def test_flags_skipped(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "*.key\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        # The flag `-n` should not be treated as a path even though it
        # could match a glob; the path arg behind it should be checked.
        v = ctl.validate_command("head -n 5 secrets.key")
        assert v.denied_path == "secrets.key"

    def test_malformed_command_no_crash(self, tmp_path: Path) -> None:
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        # Unclosed quote — shlex raises ValueError; controller catches.
        v = ctl.validate_command("cat 'unclosed")
        assert v.denied_path is None

    def test_exe_suffix_stripped(self, tmp_path: Path) -> None:
        _write_ignore(tmp_path / ".ultronignore", "*.key\n")
        ctl = ig.IgnoreController(tmp_path, global_path=tmp_path / "absent")
        v = ctl.validate_command("C:/bin/cat.exe secrets.key")
        assert v.program == "cat"
        assert v.denied_path == "secrets.key"


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_same_workspace_returns_same_instance(self, tmp_path: Path) -> None:
        a = ig.get_ignore_controller(tmp_path)
        b = ig.get_ignore_controller(tmp_path)
        assert a is b

    def test_different_workspaces_distinct(self, tmp_path: Path) -> None:
        a = ig.get_ignore_controller(tmp_path)
        b = ig.get_ignore_controller(tmp_path / "sub")
        assert a is not b

    def test_global_only(self) -> None:
        ctl = ig.get_ignore_controller(None)
        assert ctl is not None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_ignore_filename(self) -> None:
        assert ig.DEFAULT_IGNORE_FILENAME == ".ultronignore"

    def test_command_set_populated(self) -> None:
        assert "cat" in ig.COMMANDS_THAT_READ_FILES
        assert "get-content" in ig.COMMANDS_THAT_READ_FILES
        assert "sls" in ig.COMMANDS_THAT_READ_FILES

    def test_lock_glyph(self) -> None:
        assert ig.LOCK_GLYPH == "\U0001F512"
