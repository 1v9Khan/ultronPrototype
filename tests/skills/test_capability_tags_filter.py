"""Tests for the capability_tags filter wiring in SkillRegistry
(catalog 07 T4).

Covers:

* ``_coerce_capability_tag_strings`` -- frontmatter value coercion.
* ``_skill_active_for_capability_tags`` -- per-skill predicate.
* ``SkillRegistry.matching_skills`` -- end-to-end filter against
  ``capability_tags`` frontmatter values from real on-disk skill
  files.
* ``maybe_get_skills_block`` -- the orchestrator-facing one-call
  helper threads ``vlm_loaded`` / ``has_internet`` correctly.

The capability_tags enum is imported through the registry module so
the tests pin the public contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.skills.capability_tags import CapabilityTag
from ultron.skills.models import Skill, SkillSource, SkillType
from ultron.skills.registry import (
    SkillRegistry,
    _SourceSpec,
    _coerce_capability_tag_strings,
    _skill_active_for_capability_tags,
    maybe_get_skills_block,
    reset_skill_registry_for_testing,
    set_skill_registry,
)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_skill_registry_for_testing()
    yield
    reset_skill_registry_for_testing()


# ---------------------------------------------------------------------------
# _coerce_capability_tag_strings
# ---------------------------------------------------------------------------


class TestCoerceCapabilityTagStrings:

    def test_none_returns_empty(self):
        assert _coerce_capability_tag_strings(None) == []

    def test_single_string(self):
        assert _coerce_capability_tag_strings("requires-vlm") == ["requires-vlm"]

    def test_strips_whitespace(self):
        assert _coerce_capability_tag_strings("  requires-vlm  ") == ["requires-vlm"]

    def test_empty_string_returns_empty(self):
        assert _coerce_capability_tag_strings("") == []
        assert _coerce_capability_tag_strings("   ") == []

    def test_list_of_strings(self):
        assert _coerce_capability_tag_strings(["requires-vlm", "requires-internet"]) == [
            "requires-vlm",
            "requires-internet",
        ]

    def test_tuple_of_strings(self):
        assert _coerce_capability_tag_strings(("requires-vlm",)) == ["requires-vlm"]

    def test_filters_none_entries(self):
        assert _coerce_capability_tag_strings(["requires-vlm", None, "requires-internet"]) == [
            "requires-vlm",
            "requires-internet",
        ]

    def test_stringifies_non_string_entries(self):
        # Frontmatter parsers occasionally hand us int/float for
        # unquoted YAML values. The function stringifies + filters.
        assert _coerce_capability_tag_strings([123, "real-tag"]) == ["123", "real-tag"]

    def test_unknown_shape_returns_empty(self):
        assert _coerce_capability_tag_strings(42) == []
        assert _coerce_capability_tag_strings({"key": "value"}) == []


# ---------------------------------------------------------------------------
# _skill_active_for_capability_tags
# ---------------------------------------------------------------------------


def _skill_with_tags(*tag_strings: str) -> Skill:
    """Build an always-on Skill with the given capability_tags."""

    return Skill(
        name="probe",
        content="body",
        trigger=None,
        source=SkillSource.PUBLIC,
        type=SkillType.ALWAYS_ON,
        extra={"capability_tags": list(tag_strings)},
    )


class TestSkillActiveForCapabilityTags:

    def test_no_tags_passes_every_check(self):
        skill = Skill(name="legacy", content="body")
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=True, vlm_loaded=False, has_internet=False,
        ) is True

    def test_empty_list_passes_every_check(self):
        skill = _skill_with_tags()
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=True, vlm_loaded=False, has_internet=False,
        ) is True

    def test_gaming_mode_drops_incompatible_tag(self):
        skill = _skill_with_tags(CapabilityTag.LATENCY_SENSITIVE.value)
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=True, vlm_loaded=True, has_internet=True,
        ) is False

    def test_gaming_mode_passes_compatible_tag(self):
        skill = _skill_with_tags(CapabilityTag.GAMING_MODE_SAFE.value)
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=True, vlm_loaded=True, has_internet=True,
        ) is True

    def test_standby_mode_does_not_filter_on_gaming_incompatible_tag(self):
        skill = _skill_with_tags(CapabilityTag.LATENCY_SENSITIVE.value)
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=True, has_internet=True,
        ) is True

    def test_vlm_not_loaded_drops_requires_vlm(self):
        skill = _skill_with_tags(CapabilityTag.REQUIRES_VLM.value)
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=False, has_internet=True,
        ) is False

    def test_vlm_loaded_passes_requires_vlm(self):
        skill = _skill_with_tags(CapabilityTag.REQUIRES_VLM.value)
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=True, has_internet=True,
        ) is True

    def test_no_internet_drops_requires_internet(self):
        skill = _skill_with_tags(CapabilityTag.REQUIRES_INTERNET.value)
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=True, has_internet=False,
        ) is False

    def test_has_internet_passes_requires_internet(self):
        skill = _skill_with_tags(CapabilityTag.REQUIRES_INTERNET.value)
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=True, has_internet=True,
        ) is True

    def test_unknown_tag_string_ignored(self):
        # Forward-compat: an upstream tag not yet in the enum is silently
        # ignored so the skill stays available.
        skill = _skill_with_tags("requires-future-mcp-server")
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=True, vlm_loaded=False, has_internet=False,
        ) is True

    def test_mix_of_known_and_unknown_tags(self):
        skill = _skill_with_tags(
            "requires-future-thing",
            CapabilityTag.REQUIRES_VLM.value,
        )
        # The VLM tag still gates correctly.
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=False, has_internet=True,
        ) is False
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=True, has_internet=True,
        ) is True

    def test_string_value_is_tolerated(self):
        skill = Skill(
            name="probe",
            content="body",
            extra={"capability_tags": CapabilityTag.REQUIRES_VLM.value},
        )
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=False, has_internet=True,
        ) is False

    def test_unknown_shape_falls_open(self):
        skill = Skill(
            name="probe", content="body",
            extra={"capability_tags": 42},
        )
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=True, vlm_loaded=False, has_internet=False,
        ) is True

    def test_module_import_failure_falls_open(self, monkeypatch):
        import sys as _sys
        # Force the lazy import to fail. The skill MUST stay active so
        # the legacy "no filter" behaviour is preserved when the
        # capability_tags module is broken.
        monkeypatch.setitem(_sys.modules, "ultron.skills.capability_tags", None)
        skill = _skill_with_tags(CapabilityTag.REQUIRES_VLM.value)
        assert _skill_active_for_capability_tags(
            skill, gaming_mode=False, vlm_loaded=False, has_internet=True,
        ) is True


# ---------------------------------------------------------------------------
# Registry-level end-to-end (real on-disk skill files)
# ---------------------------------------------------------------------------


class TestRegistryEndToEnd:

    def test_vlm_required_skill_filtered_when_unavailable(self, tmp_path: Path):
        # Always-on skill so we don't depend on user_text triggering.
        _write(
            tmp_path / "vision_helper.md",
            "---\n"
            "name: vision_helper\n"
            "capability_tags:\n"
            "  - requires-vlm\n"
            "---\n"
            "vision body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        matches = registry.matching_skills("anything", vlm_loaded=False)
        assert all(m.skill.name != "vision_helper" for m in matches)
        # Sanity: it DOES match when vlm_loaded=True.
        matches_with_vlm = registry.matching_skills("anything", vlm_loaded=True)
        assert any(m.skill.name == "vision_helper" for m in matches_with_vlm)

    def test_gaming_mode_filters_latency_sensitive(self, tmp_path: Path):
        _write(
            tmp_path / "snappy.md",
            "---\n"
            "name: snappy\n"
            "capability_tags:\n"
            "  - latency-sensitive\n"
            "---\n"
            "body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        matches = registry.matching_skills("hello", gaming_mode=True)
        assert all(m.skill.name != "snappy" for m in matches)

    def test_no_internet_filters_requires_internet(self, tmp_path: Path):
        _write(
            tmp_path / "search.md",
            "---\n"
            "name: search\n"
            "capability_tags:\n"
            "  - requires-internet\n"
            "---\n"
            "body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        matches = registry.matching_skills("hi", has_internet=False)
        assert all(m.skill.name != "search" for m in matches)

    def test_untagged_skill_unaffected_by_filters(self, tmp_path: Path):
        _write(
            tmp_path / "legacy.md",
            "---\nname: legacy\n---\nbody",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        # Even with the strictest capability context, an untagged
        # skill makes it through.
        matches = registry.matching_skills(
            "hi", gaming_mode=True, vlm_loaded=False, has_internet=False,
        )
        assert any(m.skill.name == "legacy" for m in matches)

    def test_gaming_mode_derived_from_mode_when_none(self, tmp_path: Path):
        _write(
            tmp_path / "snappy.md",
            "---\n"
            "name: snappy\n"
            "capability_tags:\n"
            "  - latency-sensitive\n"
            "---\n"
            "body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        # No explicit gaming_mode kwarg -- the registry should derive
        # gaming_mode=True from mode="gaming".
        matches = registry.matching_skills("hi", mode="gaming")
        assert all(m.skill.name != "snappy" for m in matches)
        # And the reverse: mode="standby" derives gaming_mode=False.
        matches_standby = registry.matching_skills("hi", mode="standby")
        assert any(m.skill.name == "snappy" for m in matches_standby)

    def test_explicit_gaming_mode_overrides_mode_derivation(self, tmp_path: Path):
        _write(
            tmp_path / "snappy.md",
            "---\n"
            "name: snappy\n"
            "capability_tags:\n"
            "  - latency-sensitive\n"
            "---\n"
            "body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        # mode is gaming but explicit gaming_mode=False says "don't
        # apply gaming filter". Use case: a test that wants to verify
        # mode-vs-tag interactions independently.
        matches = registry.matching_skills(
            "hi", mode="gaming", gaming_mode=False,
        )
        assert any(m.skill.name == "snappy" for m in matches)

    def test_combined_mode_and_capability_filter(self, tmp_path: Path):
        # Skill restricted by BOTH modes: [gaming] and
        # capability_tags: [latency-sensitive]. Both filters must
        # pass independently.
        _write(
            tmp_path / "gaming_burst.md",
            "---\n"
            "name: gaming_burst\n"
            "modes:\n"
            "  - gaming\n"
            "capability_tags:\n"
            "  - latency-sensitive\n"
            "---\n"
            "body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        # gaming mode keeps the skill mode-active but the capability
        # filter still drops it (latency-sensitive incompatible).
        matches = registry.matching_skills("hi", mode="gaming")
        assert all(m.skill.name != "gaming_burst" for m in matches)
        # standby mode drops it via the modes filter alone.
        matches_standby = registry.matching_skills("hi", mode="standby")
        assert all(m.skill.name != "gaming_burst" for m in matches_standby)


# ---------------------------------------------------------------------------
# maybe_get_skills_block forwarding
# ---------------------------------------------------------------------------


class TestMaybeGetSkillsBlockForwarding:

    def test_forwards_vlm_loaded(self, tmp_path: Path, monkeypatch):
        _write(
            tmp_path / "vision_helper.md",
            "---\n"
            "name: vision_helper\n"
            "capability_tags:\n"
            "  - requires-vlm\n"
            "---\n"
            "vision body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        set_skill_registry(registry)
        block_no_vlm = maybe_get_skills_block("hi", vlm_loaded=False)
        assert "vision_helper" not in block_no_vlm
        block_with_vlm = maybe_get_skills_block("hi", vlm_loaded=True)
        assert "vision_helper" in block_with_vlm

    def test_forwards_has_internet(self, tmp_path: Path):
        _write(
            tmp_path / "search.md",
            "---\n"
            "name: search\n"
            "capability_tags:\n"
            "  - requires-internet\n"
            "---\n"
            "body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        set_skill_registry(registry)
        offline = maybe_get_skills_block("hi", has_internet=False)
        assert "search" not in offline
        online = maybe_get_skills_block("hi", has_internet=True)
        assert "search" in online

    def test_forwards_gaming_mode(self, tmp_path: Path):
        _write(
            tmp_path / "snappy.md",
            "---\n"
            "name: snappy\n"
            "capability_tags:\n"
            "  - latency-sensitive\n"
            "---\n"
            "body",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        set_skill_registry(registry)
        in_game = maybe_get_skills_block("hi", gaming_mode=True)
        assert "snappy" not in in_game
        standby = maybe_get_skills_block("hi", gaming_mode=False)
        assert "snappy" in standby

    def test_default_kwargs_inject_everything(self, tmp_path: Path):
        _write(
            tmp_path / "core.md",
            "---\nname: core\n---\nbody",
        )
        registry = SkillRegistry(
            [_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)]
        )
        set_skill_registry(registry)
        # Default args: mode=standby, vlm_loaded=True, has_internet=True
        block = maybe_get_skills_block("hi")
        assert "core" in block

    def test_no_registry_returns_empty(self):
        block = maybe_get_skills_block("hi", vlm_loaded=False)
        assert block == ""

    def test_exception_returns_empty(self, monkeypatch):
        from ultron.skills import registry as registry_module

        def _boom():
            raise RuntimeError("registry broken")

        monkeypatch.setattr(registry_module, "get_skill_registry", _boom)
        # Caller never sees the exception; it gets an empty string.
        block = maybe_get_skills_block("hi")
        assert block == ""
