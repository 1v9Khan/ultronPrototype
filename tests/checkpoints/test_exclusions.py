"""Tests for ultron.checkpoints.exclusions."""

from __future__ import annotations

from ultron.checkpoints import exclusions as ex


class TestDefaults:
    def test_default_list_covers_common_ignores(self) -> None:
        for pat in ("__pycache__/", "node_modules/", ".venv/", "*.png"):
            assert pat in ex.DEFAULT_CHECKPOINT_EXCLUSIONS

    def test_voice_baseline_list_covers_locked_files(self) -> None:
        for pat in (
            "SOUL.md",
            "ultronVoiceAudio/Ultron_vocals_mono_v1.wav",
            "models/kokoro/**",
            "models/piper/**",
            "models/rvc/**",
            "ultron_james_spader_mcu_6941/**",
        ):
            assert pat in ex.VOICE_BASELINE_PROTECTED_PATTERNS


class TestCompose:
    def test_default_render_contains_both_sections(self) -> None:
        body = ex.compose_gitignore()
        assert "DEFAULT_CHECKPOINT_EXCLUSIONS" in body
        assert "VOICE_BASELINE_PROTECTED_PATTERNS" in body
        assert "SOUL.md" in body
        assert "node_modules/" in body

    def test_extra_patterns_appended(self) -> None:
        body = ex.compose_gitignore(extra_patterns=["secrets/", "*.kdbx"])
        assert "# extra patterns" in body
        assert "secrets/" in body
        assert "*.kdbx" in body

    def test_voice_baseline_can_be_disabled(self) -> None:
        body = ex.compose_gitignore(include_voice_baseline=False)
        assert "SOUL.md" not in body
        assert "VOICE_BASELINE_PROTECTED_PATTERNS" not in body

    def test_defaults_can_be_disabled(self) -> None:
        body = ex.compose_gitignore(include_defaults=False)
        assert "DEFAULT_CHECKPOINT_EXCLUSIONS" not in body
        assert "node_modules/" not in body

    def test_lfs_patterns_extracted(self) -> None:
        gitattrs = (
            "# comments are ignored\n"
            "*.weights filter=lfs diff=lfs merge=lfs -text\n"
            "data/large.bin filter=lfs\n"
            "normal-line\n"
        )
        body = ex.compose_gitignore(workspace_gitattributes=gitattrs)
        assert "*.weights" in body
        assert "data/large.bin" in body
        assert "normal-line" not in body
