"""Tests for ultron.agent_loop.loop_detection."""

from __future__ import annotations

import pytest

from ultron.agent_loop import loop_detection as ld


# ---------------------------------------------------------------------------
# tool_call_signature
# ---------------------------------------------------------------------------

class TestSignature:
    def test_no_parameters(self) -> None:
        sig = ld.tool_call_signature("read_file")
        assert sig == "read_file|{}"

    def test_parameters_sorted(self) -> None:
        sig_a = ld.tool_call_signature("tool", {"b": 1, "a": 2})
        sig_b = ld.tool_call_signature("tool", {"a": 2, "b": 1})
        assert sig_a == sig_b

    def test_strips_default_noise_keys(self) -> None:
        with_noise = ld.tool_call_signature(
            "tool", {"a": 1, "task_progress": "halfway"},
        )
        without_noise = ld.tool_call_signature("tool", {"a": 1})
        assert with_noise == without_noise

    def test_strips_extra_noise_keys(self) -> None:
        sig = ld.tool_call_signature(
            "tool", {"a": 1, "_my_extra": "x"}, noise_keys=("_my_extra",),
        )
        assert "_my_extra" not in sig

    def test_handles_nested_mapping(self) -> None:
        sig = ld.tool_call_signature("tool", {"nested": {"x": 1, "y": 2}})
        assert '"nested":{"x":1,"y":2}' in sig

    def test_handles_lists_recursively(self) -> None:
        sig = ld.tool_call_signature("tool", {"l": [1, 2, 3]})
        assert '"l":[1,2,3]' in sig

    def test_coerces_non_serialisable_values(self) -> None:
        class Custom:
            def __repr__(self) -> str:
                return "<Custom>"
        sig = ld.tool_call_signature("tool", {"c": Custom()})
        assert "<Custom>" in sig


# ---------------------------------------------------------------------------
# LoopDetector
# ---------------------------------------------------------------------------

class TestLoopDetector:
    def test_construction_rejects_low_thresholds(self) -> None:
        with pytest.raises(ValueError):
            ld.LoopDetector(soft_threshold=1, hard_threshold=2)
        with pytest.raises(ValueError):
            ld.LoopDetector(soft_threshold=2, hard_threshold=1)
        with pytest.raises(ValueError):
            ld.LoopDetector(soft_threshold=3, hard_threshold=3)

    def test_initial_state(self) -> None:
        det = ld.LoopDetector()
        assert det.consecutive_count == 0
        assert det.last_signature is None
        assert det.halted is False

    def test_distinct_signatures_dont_accumulate(self) -> None:
        det = ld.LoopDetector(soft_threshold=2, hard_threshold=3)
        a = det.observe("read_file", {"path": "a.py"})
        b = det.observe("read_file", {"path": "b.py"})
        assert a.count == 1
        assert b.count == 1
        assert a.soft_warning is None and a.hard_escalation is None

    def test_consecutive_same_increments(self) -> None:
        det = ld.LoopDetector(soft_threshold=2, hard_threshold=3)
        det.observe("read_file", {"path": "a.py"})
        v = det.observe("read_file", {"path": "a.py"})
        assert v.count == 2
        assert v.soft_warning is not None

    def test_hard_escalation_at_threshold(self) -> None:
        det = ld.LoopDetector(soft_threshold=2, hard_threshold=3)
        det.observe("read_file", {"path": "a.py"})
        det.observe("read_file", {"path": "a.py"})
        v = det.observe("read_file", {"path": "a.py"})
        assert v.hard_escalation is not None
        assert v.should_halt is True
        assert det.halted is True

    def test_halted_persists_across_distinct_observations(self) -> None:
        det = ld.LoopDetector(soft_threshold=2, hard_threshold=3)
        for _ in range(3):
            det.observe("a", {"x": 1})
        # Even after a distinct event, we stay halted.
        v = det.observe("b", {"y": 2})
        assert v.hard_escalation is not None

    def test_reset_clears_state(self) -> None:
        det = ld.LoopDetector(soft_threshold=2, hard_threshold=3)
        for _ in range(3):
            det.observe("a", {"x": 1})
        assert det.halted is True
        det.reset()
        assert det.halted is False
        assert det.consecutive_count == 0
        v = det.observe("a", {"x": 1})
        assert v.count == 1 and v.hard_escalation is None

    def test_explicit_signature_overrides_compute(self) -> None:
        det = ld.LoopDetector(soft_threshold=2, hard_threshold=4)
        det.observe("a", signature="X")
        v = det.observe("b", signature="X")
        assert v.count == 2

    def test_noise_keys_default_strip_progress(self) -> None:
        det = ld.LoopDetector(soft_threshold=2, hard_threshold=3)
        # Two "different" calls that differ only in noise_keys.
        det.observe("tool", {"a": 1, "task_progress": "alpha"})
        v = det.observe("tool", {"a": 1, "task_progress": "beta"})
        # Should count as repeated because noise was stripped.
        assert v.count == 2

    def test_count_resets_on_new_signature(self) -> None:
        det = ld.LoopDetector(soft_threshold=3, hard_threshold=5)
        det.observe("a")
        det.observe("a")
        v = det.observe("b")
        assert v.count == 1
        assert v.soft_warning is None

    def test_soft_warning_includes_signature_and_count(self) -> None:
        det = ld.LoopDetector(soft_threshold=2, hard_threshold=4)
        det.observe("a", {"path": "p"})
        v = det.observe("a", {"path": "p"})
        assert v.soft_warning is not None
        assert v.signature in v.soft_warning
        assert "2 times" in v.soft_warning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_thresholds(self) -> None:
        assert ld.DEFAULT_SOFT_THRESHOLD == 3
        assert ld.DEFAULT_HARD_THRESHOLD == 5

    def test_default_noise_keys_set(self) -> None:
        assert "task_progress" in ld.DEFAULT_NOISE_KEYS
        assert "request_id" in ld.DEFAULT_NOISE_KEYS
