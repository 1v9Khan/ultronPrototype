"""Tests for the ``coding.supervisor.tier`` rollup field.

Mirrors :meth:`ultron.config.LLMConfig._apply_preset` semantics: setting
``tier`` fills in the per-phase flags the operator left unset; explicit
per-flag values always win.
"""

from __future__ import annotations

import pytest

from ultron.config import SUPERVISOR_TIERS, CodingSupervisorConfig


# ---------------------------------------------------------------------------
# Tier preset catalog
# ---------------------------------------------------------------------------


def test_supervisor_tiers_includes_four_canonical_levels() -> None:
    assert set(SUPERVISOR_TIERS.keys()) == {
        "off",
        "indexing_only",
        "deciding",
        "full",
    }


def test_tier_off_has_everything_disabled() -> None:
    t = SUPERVISOR_TIERS["off"]
    assert t["enabled"] is False
    assert t["digests_enabled"] is False
    assert t["index_enabled"] is False
    assert t["decide_enabled"] is False
    assert t["narrate_enabled"] is False
    assert t["enriched_context_enabled"] is False


def test_tier_indexing_only_enables_data_layer() -> None:
    t = SUPERVISOR_TIERS["indexing_only"]
    assert t["enabled"] is True
    assert t["digests_enabled"] is True
    assert t["index_enabled"] is True
    assert t["decide_enabled"] is False
    assert t["narrate_enabled"] is False
    assert t["enriched_context_enabled"] is False


def test_tier_deciding_adds_decision_layer() -> None:
    t = SUPERVISOR_TIERS["deciding"]
    assert t["enabled"] is True
    assert t["digests_enabled"] is True
    assert t["index_enabled"] is True
    assert t["decide_enabled"] is True
    assert t["narrate_enabled"] is False
    assert t["enriched_context_enabled"] is False


def test_tier_full_enables_everything() -> None:
    t = SUPERVISOR_TIERS["full"]
    assert t["enabled"] is True
    assert t["digests_enabled"] is True
    assert t["index_enabled"] is True
    assert t["decide_enabled"] is True
    assert t["narrate_enabled"] is True
    assert t["enriched_context_enabled"] is True


# ---------------------------------------------------------------------------
# Tier resolution on the pydantic model
# ---------------------------------------------------------------------------


def test_default_tier_is_off() -> None:
    cfg = CodingSupervisorConfig()
    assert cfg.tier == "off"
    assert cfg.enabled is False
    assert cfg.digests_enabled is False
    assert cfg.index_enabled is False
    assert cfg.decide_enabled is False
    assert cfg.narrate_enabled is False
    assert cfg.enriched_context_enabled is False


def test_tier_indexing_only_fills_flags() -> None:
    cfg = CodingSupervisorConfig(tier="indexing_only")
    assert cfg.enabled is True
    assert cfg.digests_enabled is True
    assert cfg.index_enabled is True
    assert cfg.decide_enabled is False
    assert cfg.narrate_enabled is False
    assert cfg.enriched_context_enabled is False


def test_tier_deciding_fills_flags() -> None:
    cfg = CodingSupervisorConfig(tier="deciding")
    assert cfg.enabled is True
    assert cfg.digests_enabled is True
    assert cfg.index_enabled is True
    assert cfg.decide_enabled is True
    assert cfg.narrate_enabled is False
    assert cfg.enriched_context_enabled is False


def test_tier_full_fills_flags() -> None:
    cfg = CodingSupervisorConfig(tier="full")
    assert cfg.enabled is True
    assert cfg.digests_enabled is True
    assert cfg.index_enabled is True
    assert cfg.decide_enabled is True
    assert cfg.narrate_enabled is True
    assert cfg.enriched_context_enabled is True


# ---------------------------------------------------------------------------
# Explicit per-flag overrides win over tier-derived defaults
# ---------------------------------------------------------------------------


def test_explicit_flag_overrides_tier_default() -> None:
    """tier='deciding' implies narrate_enabled=False, but explicit True wins."""
    cfg = CodingSupervisorConfig(tier="deciding", narrate_enabled=True)
    assert cfg.decide_enabled is True  # tier-derived
    assert cfg.narrate_enabled is True  # explicit override


def test_explicit_false_override_holds_against_tier_full() -> None:
    """tier='full' enables everything; explicit False on one flag holds."""
    cfg = CodingSupervisorConfig(tier="full", enriched_context_enabled=False)
    assert cfg.decide_enabled is True
    assert cfg.narrate_enabled is True
    assert cfg.enriched_context_enabled is False


def test_explicit_enabled_false_with_tier_indexing_only() -> None:
    """Operator who sets enabled=False explicitly stays disabled even at tier indexing_only."""
    cfg = CodingSupervisorConfig(tier="indexing_only", enabled=False)
    assert cfg.enabled is False
    # The other flags still pick up from the tier (operator only
    # overrode 'enabled'); the master switch will short-circuit
    # construction in the orchestrator regardless.
    assert cfg.digests_enabled is True


# ---------------------------------------------------------------------------
# Non-flag fields untouched by tier
# ---------------------------------------------------------------------------


def test_tier_does_not_touch_threshold_fields() -> None:
    """tier only fills the boolean phase flags; thresholds are independent."""
    cfg = CodingSupervisorConfig(tier="full")
    assert cfg.resolve_threshold == 0.75
    assert cfg.clarify_threshold == 0.55


def test_tier_does_not_touch_log_path() -> None:
    cfg = CodingSupervisorConfig(tier="full")
    assert cfg.decisions_log_path == "logs/supervisor_decisions.jsonl"


def test_tier_does_not_touch_digest_knobs() -> None:
    cfg = CodingSupervisorConfig(
        tier="full",
        digest_max_summary_chars=2000,
        digest_max_files_in_prompt=20,
    )
    assert cfg.digest_max_summary_chars == 2000
    assert cfg.digest_max_files_in_prompt == 20


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_unknown_tier_rejected() -> None:
    with pytest.raises(Exception):
        CodingSupervisorConfig(tier="bogus_tier")
