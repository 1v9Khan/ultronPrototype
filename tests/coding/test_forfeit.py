"""Tests for the forfeit primitive (catalog T8)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from ultron.coding.forfeit import (
    DEFAULT_MIN_ACTIONS,
    DEFAULT_MIN_RUNTIME_SECONDS,
    ForfeitController,
    ForfeitOutcome,
    ForfeitResult,
    ForfeitTier,
    get_forfeit_controller,
)
from ultron.coding.session_registry import (
    SessionRegistry,
    reset_session_registries_for_testing,
)


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    yield
    reset_session_registries_for_testing()


@pytest.fixture
def reg(tmp_path: Path) -> SessionRegistry:
    return SessionRegistry(session_id="forfeit-test", root=tmp_path)


@pytest.fixture
def ctrl(reg: SessionRegistry) -> ForfeitController:
    return ForfeitController(
        registry=reg,
        min_actions_before_forfeit=0,
        min_runtime_seconds=0,
    )


# ---------------------------------------------------------------------------
# Constants + construction
# ---------------------------------------------------------------------------


def test_constants_sane():
    assert DEFAULT_MIN_ACTIONS >= 0
    assert DEFAULT_MIN_RUNTIME_SECONDS >= 0


def test_invalid_min_actions_raises(reg: SessionRegistry):
    with pytest.raises(ValueError):
        ForfeitController(registry=reg, min_actions_before_forfeit=-1)


def test_invalid_min_runtime_raises(reg: SessionRegistry):
    with pytest.raises(ValueError):
        ForfeitController(registry=reg, min_runtime_seconds=-1)


def test_construction_seeds_started_at(reg: SessionRegistry):
    ForfeitController(
        registry=reg, min_actions_before_forfeit=0, min_runtime_seconds=0
    )
    assert "forfeit_session_started_at" in reg


# ---------------------------------------------------------------------------
# Action counter
# ---------------------------------------------------------------------------


def test_record_action_increments(ctrl: ForfeitController):
    assert ctrl.action_count() == 0
    ctrl.record_action()
    assert ctrl.action_count() == 1
    ctrl.record_action()
    assert ctrl.action_count() == 2


# ---------------------------------------------------------------------------
# Forfeit grants + denials
# ---------------------------------------------------------------------------


def test_forfeit_disabled_denies(reg: SessionRegistry):
    ctrl = ForfeitController(
        registry=reg,
        enabled=False,
        min_actions_before_forfeit=0,
        min_runtime_seconds=0,
    )
    result = ctrl.forfeit(reason="stuck")
    assert result.outcome == ForfeitOutcome.DENIED_DISABLED


def test_forfeit_too_early_denied_by_actions(reg: SessionRegistry):
    ctrl = ForfeitController(
        registry=reg,
        min_actions_before_forfeit=5,
        min_runtime_seconds=0,
    )
    result = ctrl.forfeit(reason="stuck")
    assert result.outcome == ForfeitOutcome.DENIED_TOO_EARLY


def test_forfeit_too_early_denied_by_runtime(reg: SessionRegistry):
    ctrl = ForfeitController(
        registry=reg,
        min_actions_before_forfeit=0,
        min_runtime_seconds=10_000,  # impossibly large
    )
    result = ctrl.forfeit(reason="stuck")
    assert result.outcome == ForfeitOutcome.DENIED_TOO_EARLY


def test_forfeit_granted_when_thresholds_met(ctrl: ForfeitController):
    result = ctrl.forfeit(reason="stuck in loop")
    assert result.outcome == ForfeitOutcome.GRANTED
    assert result.reason == "stuck in loop"


def test_forfeit_already_forfeited_returns_existing(ctrl: ForfeitController):
    first = ctrl.forfeit(reason="first reason")
    second = ctrl.forfeit(reason="second reason")
    assert first.outcome == ForfeitOutcome.GRANTED
    assert second.outcome == ForfeitOutcome.ALREADY_FORFEITED
    # The existing state's reason wins.
    assert second.reason == "first reason"


# ---------------------------------------------------------------------------
# Tier handlers
# ---------------------------------------------------------------------------


def test_safe_tier_default(ctrl: ForfeitController):
    result = ctrl.forfeit(reason="stuck")
    assert result.tier == ForfeitTier.SAFE
    assert result.files_reverted == []


def test_safe_tier_with_repo_root_runs_salvage(
    ctrl: ForfeitController, tmp_path: Path, reg: SessionRegistry
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.py").write_text("body", encoding="utf-8")
    result = ctrl.forfeit(reason="stuck", repo_root=repo)
    assert result.outcome == ForfeitOutcome.GRANTED
    # Salvage may produce a SalvageResult.
    assert result.salvage is not None


def test_followup_tier_invokes_writer(
    ctrl: ForfeitController, tmp_path: Path
):
    received: list[str] = []

    def writer(reason: str) -> None:
        received.append(reason)

    result = ctrl.forfeit(
        reason="couldn't find the bug",
        tier=ForfeitTier.FOLLOWUP,
        followup_writer=writer,
    )
    assert result.outcome == ForfeitOutcome.GRANTED
    assert received == ["couldn't find the bug"]


def test_followup_writer_exception_swallowed(ctrl: ForfeitController):
    def boom(_: str) -> None:
        raise RuntimeError("simulated")

    result = ctrl.forfeit(
        reason="stuck", tier=ForfeitTier.FOLLOWUP, followup_writer=boom
    )
    assert result.outcome == ForfeitOutcome.GRANTED


# ---------------------------------------------------------------------------
# Listener callback
# ---------------------------------------------------------------------------


def test_listener_fires_on_grant(ctrl: ForfeitController):
    seen: list[ForfeitResult] = []
    ctrl.forfeit(reason="stuck", listener=seen.append)
    assert len(seen) == 1
    assert seen[0].outcome == ForfeitOutcome.GRANTED


def test_listener_fires_on_denial(reg: SessionRegistry):
    ctrl = ForfeitController(
        registry=reg, enabled=False, min_actions_before_forfeit=0,
        min_runtime_seconds=0,
    )
    seen: list[ForfeitResult] = []
    ctrl.forfeit(reason="stuck", listener=seen.append)
    assert len(seen) == 1
    assert seen[0].outcome == ForfeitOutcome.DENIED_DISABLED


def test_listener_exception_swallowed(ctrl: ForfeitController):
    def boom(_: ForfeitResult) -> None:
        raise RuntimeError("boom")

    # Doesn't raise even though listener is broken.
    result = ctrl.forfeit(reason="stuck", listener=boom)
    assert result.outcome == ForfeitOutcome.GRANTED


# ---------------------------------------------------------------------------
# State inspection
# ---------------------------------------------------------------------------


def test_is_forfeited_false_initially(ctrl: ForfeitController):
    assert ctrl.is_forfeited() is False


def test_is_forfeited_true_after_grant(ctrl: ForfeitController):
    ctrl.forfeit(reason="stuck")
    assert ctrl.is_forfeited() is True


def test_current_state_carries_metadata(ctrl: ForfeitController):
    ctrl.record_action()
    ctrl.record_action()
    ctrl.forfeit(reason="dead end")
    state = ctrl.current_state()
    assert state["reason"] == "dead end"
    assert state["outcome"] == "granted"
    assert state["actions_at_forfeit"] == 2


def test_reset_clears_state(ctrl: ForfeitController):
    ctrl.record_action()
    ctrl.forfeit(reason="stuck")
    ctrl.reset()
    assert ctrl.is_forfeited() is False
    assert ctrl.action_count() == 0


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


def test_state_survives_controller_reconstruction(tmp_path: Path):
    reset_session_registries_for_testing()
    reg = SessionRegistry(session_id="persist", root=tmp_path)
    c1 = ForfeitController(
        registry=reg,
        min_actions_before_forfeit=0,
        min_runtime_seconds=0,
    )
    c1.forfeit(reason="stuck")

    reg_b = SessionRegistry(session_id="persist", root=tmp_path)
    c2 = ForfeitController(registry=reg_b)
    assert c2.is_forfeited() is True
    assert c2.current_state()["reason"] == "stuck"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_get_forfeit_controller_returns_instance(tmp_path: Path):
    reset_session_registries_for_testing()
    reg = SessionRegistry(session_id="factory", root=tmp_path)
    c = get_forfeit_controller(
        "factory",
        registry=reg,
        min_actions_before_forfeit=0,
        min_runtime_seconds=0,
    )
    assert isinstance(c, ForfeitController)
    assert c.registry is reg


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_forfeit_result_is_frozen(ctrl: ForfeitController):
    r = ctrl.forfeit(reason="x")
    with pytest.raises(Exception):
        r.outcome = ForfeitOutcome.DENIED_TOO_EARLY  # type: ignore[misc]


def test_enum_values_stable():
    assert ForfeitOutcome.GRANTED.value == "granted"
    assert ForfeitOutcome.DENIED_TOO_EARLY.value == "denied_too_early"
    assert ForfeitOutcome.DENIED_DISABLED.value == "denied_disabled"
    assert ForfeitOutcome.ALREADY_FORFEITED.value == "already_forfeited"
    assert ForfeitTier.SAFE.value == "safe"
    assert ForfeitTier.REVERT.value == "revert"
    assert ForfeitTier.FOLLOWUP.value == "followup"
