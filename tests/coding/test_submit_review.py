"""Tests for the multi-stage submit review loop (catalog T7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.coding.session_registry import (
    SessionRegistry,
    reset_session_registries_for_testing,
)
from ultron.coding.submit_review import (
    DEFAULT_DOC_DRIFT_STAGE,
    DEFAULT_STAGES,
    DEFAULT_TESTS_STAGE,
    DEFAULT_VOICE_LOCKED_PATTERNS,
    DEFAULT_VOICE_LOCK_STAGE,
    ReviewStage,
    ReviewState,
    StageOutcome,
    StageResult,
    SubmitReviewLoop,
    build_submit_review_loop,
    detect_voice_lock_hits,
)


@pytest.fixture(autouse=True)
def _cleanup() -> None:
    yield
    reset_session_registries_for_testing()


@pytest.fixture
def reg(tmp_path: Path) -> SessionRegistry:
    return SessionRegistry(session_id="review-test", root=tmp_path)


# ---------------------------------------------------------------------------
# Default stages
# ---------------------------------------------------------------------------


def test_default_stages_have_three_entries():
    assert len(DEFAULT_STAGES) == 3


def test_default_stage_names_stable():
    names = [s.name for s in DEFAULT_STAGES]
    assert names == ["VOICE_LOCK", "TESTS", "DOC_DRIFT"]


def test_voice_lock_stage_required():
    assert DEFAULT_VOICE_LOCK_STAGE.required is True


def test_tests_stage_required():
    assert DEFAULT_TESTS_STAGE.required is True


def test_doc_drift_stage_required():
    assert DEFAULT_DOC_DRIFT_STAGE.required is True


def test_review_stage_frozen():
    s = ReviewStage(name="X", prompt_template="p")
    with pytest.raises(Exception):
        s.name = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# detect_voice_lock_hits
# ---------------------------------------------------------------------------


def test_detect_voice_lock_hits_finds_soul_md():
    hits = detect_voice_lock_hits(["src/ultron/x.py", "SOUL.md", "tests/x.py"])
    assert hits == ["SOUL.md"]


def test_detect_voice_lock_hits_finds_piper_models():
    hits = detect_voice_lock_hits(["models/piper/en_US-ryan-medium.onnx"])
    assert len(hits) == 1


def test_detect_voice_lock_hits_finds_rvc_voice_model():
    hits = detect_voice_lock_hits(["ultron_james_spader_mcu_6941/Ultron.pth"])
    assert len(hits) == 1


def test_detect_voice_lock_hits_skips_empty_strings():
    hits = detect_voice_lock_hits(["", "src/x.py", "  "])
    assert hits == []


def test_detect_voice_lock_hits_case_insensitive():
    hits = detect_voice_lock_hits(["soul.md", "Soul.MD"])
    assert len(hits) == 2


def test_detect_voice_lock_hits_no_false_positives():
    hits = detect_voice_lock_hits(
        ["src/ultron/coding/runner.py", "tests/test_x.py", "docs/architecture.md"]
    )
    assert hits == []


def test_default_patterns_non_empty():
    assert len(DEFAULT_VOICE_LOCKED_PATTERNS) > 0


# ---------------------------------------------------------------------------
# Loop construction
# ---------------------------------------------------------------------------


def test_empty_stages_rejected(reg: SessionRegistry):
    with pytest.raises(ValueError):
        SubmitReviewLoop(stages=(), registry=reg)


def test_duplicate_stage_names_deduped(reg: SessionRegistry):
    s1 = ReviewStage(name="X", prompt_template="p1")
    s2 = ReviewStage(name="X", prompt_template="p2")  # same name
    s3 = ReviewStage(name="Y", prompt_template="p3")
    loop = SubmitReviewLoop(stages=(s1, s2, s3), registry=reg)
    assert len(loop.stages) == 2


def test_build_factory_with_defaults(reg: SessionRegistry, tmp_path: Path):
    loop = build_submit_review_loop("session-a", registry=reg)
    assert len(loop.stages) == 3


def test_build_factory_with_extra_stages(reg: SessionRegistry):
    extra = ReviewStage(name="CUSTOM", prompt_template="c")
    loop = build_submit_review_loop(
        "session-a", extra_stages=[extra], registry=reg
    )
    assert len(loop.stages) == 4
    assert loop.stages[-1].name == "CUSTOM"


def test_build_factory_skip_defaults(reg: SessionRegistry):
    extra = ReviewStage(name="ONLY", prompt_template="o")
    loop = build_submit_review_loop(
        "session-a", extra_stages=[extra], skip_defaults=True, registry=reg
    )
    assert len(loop.stages) == 1
    assert loop.stages[0].name == "ONLY"


# ---------------------------------------------------------------------------
# Loop state machine
# ---------------------------------------------------------------------------


def test_current_stage_starts_at_first(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    cur = loop.current_stage()
    assert cur is not None
    assert cur.name == "VOICE_LOCK"


def test_current_prompt_substitutes_context(reg: SessionRegistry):
    loop = SubmitReviewLoop(
        stages=(ReviewStage(name="X", prompt_template="hello {name}"),),
        registry=reg,
    )
    out = loop.current_prompt(context={"name": "world"})
    assert "hello world" in out


def test_current_prompt_missing_keys_render_empty(reg: SessionRegistry):
    loop = SubmitReviewLoop(
        stages=(ReviewStage(name="X", prompt_template="a={a} b={b}"),),
        registry=reg,
    )
    out = loop.current_prompt(context={"a": "set"})
    assert "a=set" in out
    assert "b=" in out  # b is missing -> empty


def test_resolve_passed_advances_counter(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.resolve("VOICE_LOCK", StageOutcome.PASSED)
    cur = loop.current_stage()
    assert cur is not None
    assert cur.name == "TESTS"


def test_resolve_skipped_advances_counter(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.resolve("VOICE_LOCK", StageOutcome.SKIPPED, note="no voice files")
    assert loop.current_stage().name == "TESTS"


def test_resolve_failed_required_blocks(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.resolve("VOICE_LOCK", StageOutcome.FAILED, note="SOUL.md touched")
    assert loop.is_blocked() is True
    cur = loop.current_stage()
    # Blocked stage is still surfaced as "current" until resolved.
    assert cur is not None
    assert cur.name == "VOICE_LOCK"


def test_resolve_failed_optional_does_not_block(reg: SessionRegistry):
    s = ReviewStage(name="OPT", prompt_template="p", required=False)
    loop = SubmitReviewLoop(stages=(s,), registry=reg)
    loop.resolve("OPT", StageOutcome.FAILED)
    assert loop.is_blocked() is False
    # But it didn't advance either -- failed stages don't move the cursor.
    # However is_complete should report based on no-blocked + no-uncovered.
    # current_stage returns None when nothing is left to resolve.
    # In this case current_stage starts at 0 with 1 stage; FAILED keeps
    # counter at 0 but is_blocked() returns False because not required.
    # is_complete should be False because counter didn't advance.
    assert loop.is_complete() is False


def test_resolve_unknown_stage_raises(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    with pytest.raises(RuntimeError):
        loop.resolve("UNKNOWN", StageOutcome.PASSED)


def test_resolve_same_stage_twice_raises(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.resolve("VOICE_LOCK", StageOutcome.PASSED)
    with pytest.raises(RuntimeError):
        loop.resolve("VOICE_LOCK", StageOutcome.PASSED)


def test_resolve_after_force_complete_raises(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.force_complete(reason="testing")
    with pytest.raises(RuntimeError):
        loop.resolve("VOICE_LOCK", StageOutcome.PASSED)


def test_resolve_all_passed_means_complete(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    for s in DEFAULT_STAGES:
        loop.resolve(s.name, StageOutcome.PASSED)
    assert loop.is_complete() is True
    assert loop.current_stage() is None


# ---------------------------------------------------------------------------
# Force complete
# ---------------------------------------------------------------------------


def test_force_complete_marks_remaining_as_forced(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.resolve("VOICE_LOCK", StageOutcome.PASSED)
    loop.force_complete(reason="user said skip")
    history = loop.history()
    assert len(history) == 3
    outcomes = {h.name: h.outcome for h in history}
    assert outcomes["VOICE_LOCK"] == StageOutcome.PASSED
    assert outcomes["TESTS"] == StageOutcome.FORCED
    assert outcomes["DOC_DRIFT"] == StageOutcome.FORCED
    assert loop.is_complete() is True


def test_force_complete_persists_reason(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.force_complete(reason="explicit override")
    forced = [h for h in loop.history() if h.outcome == StageOutcome.FORCED]
    assert all(h.note == "explicit override" for h in forced)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_state_survives_loop_reconstruction(tmp_path: Path):
    reset_session_registries_for_testing()
    reg = SessionRegistry(session_id="persist", root=tmp_path)
    loop_a = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop_a.resolve("VOICE_LOCK", StageOutcome.PASSED, note="clean")
    loop_a.resolve("TESTS", StageOutcome.PASSED)

    # Simulate fresh process -- new loop instance on same registry.
    reg_b = SessionRegistry(session_id="persist", root=tmp_path)
    loop_b = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg_b)
    assert loop_b.current_stage().name == "DOC_DRIFT"
    history_names = [h.name for h in loop_b.history()]
    assert "VOICE_LOCK" in history_names
    assert "TESTS" in history_names


def test_reset_clears_state(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.resolve("VOICE_LOCK", StageOutcome.PASSED)
    loop.reset()
    assert loop.current_stage().name == "VOICE_LOCK"
    assert loop.history() == []


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------


def test_status_summary_starts_at_stage_1(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    s = loop.status_summary()
    assert "1/3" in s
    assert "VOICE_LOCK" in s


def test_status_summary_when_complete(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.force_complete(reason="test")
    assert "complete" in loop.status_summary().lower()


def test_status_summary_when_blocked(reg: SessionRegistry):
    loop = SubmitReviewLoop(stages=DEFAULT_STAGES, registry=reg)
    loop.resolve("VOICE_LOCK", StageOutcome.FAILED)
    assert "BLOCKED" in loop.status_summary()
