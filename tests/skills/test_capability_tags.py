"""Tests for the T5 capability-tag namespace + filtering."""

from __future__ import annotations

import pytest

from ultron.skills.capability_tags import (
    GAMING_MODE_INCOMPATIBLE_TAGS,
    K_PROTECTED_TAGS,
    CapabilityTag,
    TaggedCapability,
    derive_capability_tags,
    filter_capabilities,
    is_gaming_mode_safe,
    is_voice_path_safe,
    needs_explicit_intent,
)


# ---------------------------------------------------------------------------
# derive_capability_tags


def test_derive_from_internet_signal() -> None:
    tags = derive_capability_tags(source="r = requests.get('https://x')")
    assert CapabilityTag.REQUIRES_INTERNET in tags


def test_derive_from_pyautogui_signal() -> None:
    tags = derive_capability_tags(source="pyautogui.click(100, 100)")
    assert CapabilityTag.REQUIRES_DESKTOP_INPUT in tags


def test_derive_from_screen_capture_signal() -> None:
    tags = derive_capability_tags(source="from mss import mss")
    assert CapabilityTag.REQUIRES_SCREEN_CAPTURE in tags


def test_derive_from_vlm_signal() -> None:
    tags = derive_capability_tags(source="vlm = get_vlm()")
    assert CapabilityTag.REQUIRES_VLM in tags


def test_derive_from_shell_signal() -> None:
    tags = derive_capability_tags(source="subprocess.run(['ls'])")
    assert CapabilityTag.EXECUTES_SHELL in tags


def test_derive_from_python_exec_signal() -> None:
    tags = derive_capability_tags(source="exec(compile('x=1', '<x>', 'exec'))")
    assert CapabilityTag.EXECUTES_PYTHON in tags


def test_derive_secret_env_read() -> None:
    tags = derive_capability_tags(
        source='token = os.environ.get("OPENAI_API_KEY")'
    )
    assert CapabilityTag.READS_SECRETS in tags


def test_derive_write_file_signal() -> None:
    tags = derive_capability_tags(source='open("out.txt", "w").write("x")')
    assert CapabilityTag.WRITES_FILES in tags


def test_derive_clean_source_no_tags() -> None:
    tags = derive_capability_tags(source="x = 1 + 2")
    assert tags == ()


def test_derive_from_manifest_explicit_tags() -> None:
    manifest = {
        "capabilityTags": ["requires-internet", "voice-only"],
    }
    tags = derive_capability_tags(source="", manifest=manifest)
    assert CapabilityTag.REQUIRES_INTERNET in tags
    assert CapabilityTag.VOICE_ONLY in tags


def test_derive_skips_unknown_manifest_tags() -> None:
    manifest = {
        "capabilityTags": ["requires-internet", "unknown-future-tag"],
    }
    tags = derive_capability_tags(source="", manifest=manifest)
    # Unknown tag silently ignored.
    assert CapabilityTag.REQUIRES_INTERNET in tags
    assert len(tags) == 1


def test_derive_from_manifest_requires_block() -> None:
    manifest = {
        "requires": {"browser": True, "desktop": True, "vlm": True},
    }
    tags = derive_capability_tags(source="", manifest=manifest)
    assert CapabilityTag.REQUIRES_BROWSER in tags
    assert CapabilityTag.REQUIRES_DESKTOP_INPUT in tags
    assert CapabilityTag.REQUIRES_VLM in tags


def test_derive_envvars_implies_api_key_and_secrets() -> None:
    manifest = {"envVars": [{"name": "FOO_TOKEN"}]}
    tags = derive_capability_tags(source="", manifest=manifest)
    assert CapabilityTag.REQUIRES_API_KEY in tags
    assert CapabilityTag.READS_SECRETS in tags


def test_derive_returns_sorted_tuple() -> None:
    tags = derive_capability_tags(
        source="r = requests.get('x')\npyautogui.click(0, 0)"
    )
    # Sorted by tag string value (case-fold).
    values = [t.value for t in tags]
    assert values == sorted(values)


def test_derive_dedupes_across_source_and_manifest() -> None:
    source = "r = requests.get('x')"
    manifest = {"requires": {"internet": True}}
    tags = derive_capability_tags(source=source, manifest=manifest)
    # Should appear once even though both sources contribute it.
    assert tags.count(CapabilityTag.REQUIRES_INTERNET) == 1


# ---------------------------------------------------------------------------
# TaggedCapability


def test_tagged_capability_has() -> None:
    cap = TaggedCapability(
        name="x",
        tags=(CapabilityTag.REQUIRES_INTERNET, CapabilityTag.LATENCY_TOLERANT),
    )
    assert cap.has(CapabilityTag.REQUIRES_INTERNET)
    assert not cap.has(CapabilityTag.REQUIRES_VLM)


def test_tagged_capability_any_of() -> None:
    cap = TaggedCapability(name="x", tags=(CapabilityTag.REQUIRES_INTERNET,))
    assert cap.any_of([CapabilityTag.REQUIRES_INTERNET, CapabilityTag.REQUIRES_VLM])
    assert not cap.any_of([CapabilityTag.REQUIRES_VLM])


def test_tagged_capability_all_of() -> None:
    cap = TaggedCapability(
        name="x",
        tags=(CapabilityTag.REQUIRES_INTERNET, CapabilityTag.LATENCY_TOLERANT),
    )
    assert cap.all_of([CapabilityTag.REQUIRES_INTERNET, CapabilityTag.LATENCY_TOLERANT])
    assert not cap.all_of([CapabilityTag.REQUIRES_INTERNET, CapabilityTag.REQUIRES_VLM])


# ---------------------------------------------------------------------------
# filter_capabilities


def _cap(name: str, *tags: CapabilityTag) -> TaggedCapability:
    return TaggedCapability(name=name, tags=tuple(tags))


def test_filter_require_passes() -> None:
    items = [
        _cap("a", CapabilityTag.REQUIRES_INTERNET),
        _cap("b", CapabilityTag.REQUIRES_VLM),
    ]
    out = filter_capabilities(items, require=[CapabilityTag.REQUIRES_INTERNET])
    assert [c.name for c in out] == ["a"]


def test_filter_exclude_drops() -> None:
    items = [
        _cap("a", CapabilityTag.REQUIRES_INTERNET),
        _cap("b", CapabilityTag.EXECUTES_SHELL),
    ]
    out = filter_capabilities(items, exclude=[CapabilityTag.EXECUTES_SHELL])
    assert [c.name for c in out] == ["a"]


def test_filter_gaming_mode_drops_unsafe() -> None:
    items = [
        _cap("light", CapabilityTag.TEXT_ONLY),
        _cap("vlm-tool", CapabilityTag.REQUIRES_VLM),
        _cap("foreground", CapabilityTag.EXECUTES_FOREGROUND_WINDOW_ACTIONS),
    ]
    out = filter_capabilities(items, gaming_mode=True)
    assert [c.name for c in out] == ["light"]


def test_filter_vlm_unloaded_drops_vlm_requiring() -> None:
    items = [
        _cap("a", CapabilityTag.REQUIRES_VLM),
        _cap("b", CapabilityTag.REQUIRES_INTERNET),
    ]
    out = filter_capabilities(items, vlm_loaded=False)
    assert [c.name for c in out] == ["b"]


def test_filter_no_internet_drops_internet_requiring() -> None:
    items = [
        _cap("a", CapabilityTag.REQUIRES_INTERNET),
        _cap("b", CapabilityTag.TEXT_ONLY),
    ]
    out = filter_capabilities(items, has_internet=False)
    assert [c.name for c in out] == ["b"]


def test_filter_preserves_order() -> None:
    items = [
        _cap("first"),
        _cap("second"),
        _cap("third"),
    ]
    out = filter_capabilities(items)
    assert [c.name for c in out] == ["first", "second", "third"]


def test_filter_all_constraints_combined() -> None:
    items = [
        _cap("voice-llm", CapabilityTag.REQUIRES_LLM, CapabilityTag.LATENCY_TOLERANT),
        _cap("dangerous", CapabilityTag.EXECUTES_SHELL, CapabilityTag.REQUIRES_INTERNET),
        _cap("safe-net", CapabilityTag.REQUIRES_INTERNET, CapabilityTag.LATENCY_TOLERANT),
    ]
    out = filter_capabilities(
        items,
        require=[CapabilityTag.LATENCY_TOLERANT],
        exclude=[CapabilityTag.EXECUTES_SHELL],
        gaming_mode=False,
    )
    assert {c.name for c in out} == {"voice-llm", "safe-net"}


# ---------------------------------------------------------------------------
# Predicates


def test_is_gaming_mode_safe_clean() -> None:
    assert is_gaming_mode_safe([CapabilityTag.TEXT_ONLY])


def test_is_gaming_mode_safe_unsafe() -> None:
    assert not is_gaming_mode_safe([CapabilityTag.REQUIRES_VLM])


def test_needs_explicit_intent_when_tagged() -> None:
    assert needs_explicit_intent([CapabilityTag.REQUIRES_EXPLICIT_INTENT])


def test_needs_explicit_intent_k_category() -> None:
    assert needs_explicit_intent([CapabilityTag.K_CATEGORY_TERRITORY])
    assert needs_explicit_intent([CapabilityTag.SELF_MODIFIES_TOOLKIT])


def test_needs_explicit_intent_safe_default() -> None:
    assert not needs_explicit_intent([CapabilityTag.TEXT_ONLY])


def test_is_voice_path_safe_default() -> None:
    """Currently latency-sensitive still counts as safe (pre-ack covers it)."""
    assert is_voice_path_safe([CapabilityTag.LATENCY_SENSITIVE])
    assert is_voice_path_safe([CapabilityTag.LATENCY_TOLERANT])
    assert is_voice_path_safe([])


# ---------------------------------------------------------------------------
# Constants


def test_gaming_mode_incompatible_tags_non_empty() -> None:
    assert len(GAMING_MODE_INCOMPATIBLE_TAGS) >= 3


def test_k_protected_tags_non_empty() -> None:
    assert CapabilityTag.SELF_MODIFIES_TOOLKIT in K_PROTECTED_TAGS
    assert CapabilityTag.K_CATEGORY_TERRITORY in K_PROTECTED_TAGS
