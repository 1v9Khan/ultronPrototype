"""Tests for ultron.safety.auto_approval."""

from __future__ import annotations

import pytest

from ultron.safety import auto_approval as aa


class _Clock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

class TestModes:
    def test_default_mode_is_always_ask(self) -> None:
        matrix = aa.AutoApprovalMatrix()
        assert matrix.mode_for("K1") is aa.AutoApprovalMode.ALWAYS_ASK

    def test_set_mode_overrides_default(self) -> None:
        matrix = aa.AutoApprovalMatrix()
        matrix.set_mode("K1", aa.AutoApprovalMode.ALLOW_ALL)
        assert matrix.mode_for("K1") is aa.AutoApprovalMode.ALLOW_ALL

    def test_string_mode_coerced(self) -> None:
        matrix = aa.AutoApprovalMatrix({"R": "allow_local"})
        assert matrix.mode_for("R") is aa.AutoApprovalMode.ALLOW_LOCAL

    def test_unknown_string_falls_back(self) -> None:
        matrix = aa.AutoApprovalMatrix({"R": "garbage"})
        assert matrix.mode_for("R") is aa.AutoApprovalMode.ALWAYS_ASK

    def test_configured_modes_snapshot(self) -> None:
        matrix = aa.AutoApprovalMatrix({"A": "allow_local", "B": "allow_all"})
        snap = matrix.configured_modes()
        assert snap == {
            "A": aa.AutoApprovalMode.ALLOW_LOCAL,
            "B": aa.AutoApprovalMode.ALLOW_ALL,
        }


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_always_ask_returns_ask_user(self) -> None:
        matrix = aa.AutoApprovalMatrix()
        out = matrix.evaluate("R", "/x")
        assert out.outcome is aa.AutoApprovalOutcome.ASK_USER
        assert out.mode is aa.AutoApprovalMode.ALWAYS_ASK
        assert out.warmed is False

    def test_allow_all_returns_allow(self) -> None:
        matrix = aa.AutoApprovalMatrix({"R": "allow_all"})
        out = matrix.evaluate("R", "/x")
        assert out.outcome is aa.AutoApprovalOutcome.ALLOW

    def test_allow_local_with_local_target(self) -> None:
        matrix = aa.AutoApprovalMatrix(
            {"R": "allow_local"},
            locality_probe=lambda _: True,
        )
        out = matrix.evaluate("R", "src/a.py")
        assert out.outcome is aa.AutoApprovalOutcome.ALLOW
        assert out.locality is True

    def test_allow_local_with_external_target_asks(self) -> None:
        matrix = aa.AutoApprovalMatrix(
            {"R": "allow_local"},
            locality_probe=lambda _: False,
        )
        out = matrix.evaluate("R", "/etc/passwd")
        assert out.outcome is aa.AutoApprovalOutcome.ASK_USER
        assert out.locality is False

    def test_allow_local_without_probe_asks(self) -> None:
        matrix = aa.AutoApprovalMatrix({"R": "allow_local"})
        out = matrix.evaluate("R", "src/a.py")
        assert out.outcome is aa.AutoApprovalOutcome.ASK_USER
        assert out.locality is None

    def test_allow_external_with_external_target(self) -> None:
        matrix = aa.AutoApprovalMatrix(
            {"R": "allow_external"},
            locality_probe=lambda _: False,
        )
        out = matrix.evaluate("R", "https://api.example.com")
        assert out.outcome is aa.AutoApprovalOutcome.ALLOW

    def test_allow_external_with_local_target_asks(self) -> None:
        matrix = aa.AutoApprovalMatrix(
            {"R": "allow_external"},
            locality_probe=lambda _: True,
        )
        out = matrix.evaluate("R", "src/a.py")
        assert out.outcome is aa.AutoApprovalOutcome.ASK_USER

    def test_locality_probe_exception_treated_as_unknown(self) -> None:
        def broken(_: str) -> bool:
            raise RuntimeError("boom")
        matrix = aa.AutoApprovalMatrix(
            {"R": "allow_local"},
            locality_probe=broken,
        )
        out = matrix.evaluate("R", "src/a.py")
        # Fail-open: treat as unknown locality → ASK_USER.
        assert out.outcome is aa.AutoApprovalOutcome.ASK_USER

    def test_yolo_mode_allows_everything(self) -> None:
        matrix = aa.AutoApprovalMatrix({"R": "always_ask"}, yolo_mode=True)
        out = matrix.evaluate("R", "/etc/passwd")
        assert out.outcome is aa.AutoApprovalOutcome.ALLOW
        assert "yolo" in out.reason

    def test_evaluate_many(self) -> None:
        matrix = aa.AutoApprovalMatrix({"A": "allow_all", "B": "always_ask"})
        outs = matrix.evaluate_many([("A", "x"), ("B", "y")])
        assert outs[0].outcome is aa.AutoApprovalOutcome.ALLOW
        assert outs[1].outcome is aa.AutoApprovalOutcome.ASK_USER


# ---------------------------------------------------------------------------
# Session warming
# ---------------------------------------------------------------------------

class TestWarming:
    def test_record_user_grant_below_threshold(self) -> None:
        matrix = aa.AutoApprovalMatrix(warming_threshold=3)
        assert matrix.record_user_grant("R", "/x") is False
        assert matrix.record_user_grant("R", "/x") is False
        assert matrix.record_user_grant("R", "/x") is True
        # Once warmed, subsequent evaluations auto-allow.
        out = matrix.evaluate("R", "/x")
        assert out.outcome is aa.AutoApprovalOutcome.ALLOW
        assert out.warmed is True

    def test_warming_resets_after_promotion(self) -> None:
        matrix = aa.AutoApprovalMatrix(warming_threshold=2)
        matrix.record_user_grant("R", "/x")
        matrix.record_user_grant("R", "/x")  # warmed
        # The internal counter resets to 0 after promotion, so a future
        # revoke can re-warm cleanly.
        assert matrix.evaluate("R", "/x").warmed is True

    def test_revoke_grant_drops_warmed(self) -> None:
        matrix = aa.AutoApprovalMatrix(warming_threshold=1)
        matrix.record_user_grant("R", "/x")
        assert matrix.evaluate("R", "/x").warmed is True
        assert matrix.revoke_user_grant("R", "/x") is True
        out = matrix.evaluate("R", "/x")
        # Without warming + with default mode, should be ASK_USER.
        assert out.outcome is aa.AutoApprovalOutcome.ASK_USER
        assert out.warmed is False

    def test_record_user_denial_clears_counter(self) -> None:
        matrix = aa.AutoApprovalMatrix(warming_threshold=3)
        matrix.record_user_grant("R", "/x")
        matrix.record_user_grant("R", "/x")
        matrix.record_user_denial("R", "/x")
        # Counter reset to 0; the third grant should NOT warm.
        assert matrix.record_user_grant("R", "/x") is False

    def test_warming_disabled_by_zero_threshold(self) -> None:
        matrix = aa.AutoApprovalMatrix(warming_threshold=0)
        for _ in range(10):
            assert matrix.record_user_grant("R", "/x") is False

    def test_warming_ttl_expires(self) -> None:
        clock = _Clock()
        matrix = aa.AutoApprovalMatrix(
            warming_threshold=1,
            warming_ttl_seconds=100.0,
            clock=clock,
        )
        matrix.record_user_grant("R", "/x")
        assert matrix.evaluate("R", "/x").warmed is True
        clock.advance(200.0)
        out = matrix.evaluate("R", "/x")
        assert out.warmed is False
        assert out.outcome is aa.AutoApprovalOutcome.ASK_USER

    def test_warmed_pairs_snapshot(self) -> None:
        matrix = aa.AutoApprovalMatrix(warming_threshold=1)
        matrix.record_user_grant("R", "/x")
        matrix.record_user_grant("R", "/y")
        pairs = set(matrix.warmed_pairs())
        assert ("R", "/x") in pairs
        assert ("R", "/y") in pairs

    def test_clear_session_drops_everything(self) -> None:
        matrix = aa.AutoApprovalMatrix(warming_threshold=1)
        matrix.record_user_grant("R", "/x")
        matrix.clear_session()
        assert matrix.warmed_pairs() == []
        assert matrix.evaluate("R", "/x").warmed is False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_mode(self) -> None:
        assert aa.DEFAULT_AUTO_APPROVAL_MODE is aa.AutoApprovalMode.ALWAYS_ASK

    def test_default_warming_threshold(self) -> None:
        assert aa.DEFAULT_WARMING_THRESHOLD == 5

    def test_default_warming_ttl(self) -> None:
        assert aa.DEFAULT_WARMING_TTL_SECONDS == 30 * 60.0
