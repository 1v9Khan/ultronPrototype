"""Tests for the T5 mode filter in ``SkillRegistry.matching_skills``.

A skill's frontmatter ``modes: [gaming, standby]`` list scopes which
modes the skill is active in. Skills with no ``modes`` declaration
match every mode (legacy / unscoped). The orchestrator forwards the
current mode (``"gaming"`` when ``GamingModeManager`` is engaged,
``"standby"`` otherwise) so the registry can return a mode-filtered
catalogue.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ultron.skills.models import (
    KeywordTrigger,
    Skill,
    SkillSource,
    SkillType,
)
from ultron.skills.registry import (
    SkillRegistry,
    _skill_active_in_mode,
    format_skills_block,
    maybe_get_skills_block,
    set_skill_registry,
)


def _make_skill(
    name: str,
    *,
    modes=None,
    keywords=("ssh",),
    min_user_text_chars=0,
) -> Skill:
    extra: dict = {}
    if modes is not None:
        extra["modes"] = modes
    trigger = KeywordTrigger(keywords=tuple(keywords),
                             min_user_text_chars=min_user_text_chars)
    return Skill(
        name=name,
        content=f"# {name} body",
        trigger=trigger,
        source=SkillSource.PUBLIC,
        type=SkillType.KNOWLEDGE,
        description=None,
        path=Path(f"/tmp/{name}.md"),
        version=None,
        extra=extra,
    )


def test_skill_with_no_modes_matches_every_mode():
    skill = _make_skill("coding")
    assert _skill_active_in_mode(skill, "standby") is True
    assert _skill_active_in_mode(skill, "gaming") is True
    assert _skill_active_in_mode(skill, "anything") is True


def test_skill_with_modes_list_filters_correctly():
    skill = _make_skill("gaming_skill", modes=["gaming"])
    assert _skill_active_in_mode(skill, "gaming") is True
    assert _skill_active_in_mode(skill, "standby") is False


def test_skill_modes_case_insensitive():
    skill = _make_skill("mixed", modes=["GAMING", "standby"])
    assert _skill_active_in_mode(skill, "gaming") is True
    assert _skill_active_in_mode(skill, "Standby") is True


def test_skill_modes_string_value_also_works():
    """Frontmatter sometimes ships modes as a single string."""
    skill = _make_skill("gaming_only", modes="gaming")
    assert _skill_active_in_mode(skill, "gaming") is True
    assert _skill_active_in_mode(skill, "standby") is False


def test_unknown_modes_value_fails_open_active():
    skill = _make_skill("broken", modes=42)  # nonsense type
    # Fail-open: skill stays active rather than getting silently dropped.
    assert _skill_active_in_mode(skill, "standby") is True


def test_registry_matching_skills_filters_by_mode(tmp_path):
    coding = _make_skill("coding", modes=["standby"], keywords=("ssh",))
    gaming = _make_skill("gaming", modes=["gaming"], keywords=("ssh",))
    universal = _make_skill("ack", keywords=("ssh",))  # no modes declared

    # Build a registry directly without scanning disk; inject the cache.
    registry = SkillRegistry(sources=[])
    registry._cache = {
        coding.name.lower(): coding,
        gaming.name.lower(): gaming,
        universal.name.lower(): universal,
    }
    registry._loaded_at = 1.0  # skip reload-if-stale

    standby_matches = registry.matching_skills("please ssh in", mode="standby")
    gaming_matches = registry.matching_skills("please ssh in", mode="gaming")

    standby_names = {m.skill.name for m in standby_matches}
    gaming_names = {m.skill.name for m in gaming_matches}

    assert standby_names == {"coding", "ack"}  # gaming excluded
    assert gaming_names == {"gaming", "ack"}   # coding excluded


def test_maybe_get_skills_block_forwards_mode(tmp_path):
    coding = _make_skill("coding", modes=["standby"], keywords=("test",))
    gaming = _make_skill("gaming", modes=["gaming"], keywords=("test",))

    registry = SkillRegistry(sources=[])
    registry._cache = {
        coding.name.lower(): coding,
        gaming.name.lower(): gaming,
    }
    registry._loaded_at = 1.0
    set_skill_registry(registry)
    try:
        gaming_block = maybe_get_skills_block("please test", mode="gaming")
        standby_block = maybe_get_skills_block(
            "please test", mode="standby",
        )
    finally:
        set_skill_registry(None)

    assert "gaming" in gaming_block.lower()
    assert "coding" not in gaming_block.lower() or "# coding" not in gaming_block
    assert "coding" in standby_block.lower()
