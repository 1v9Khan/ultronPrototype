"""Tests for the format-error requery loop (catalog T14)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from ultron.llm.requery import (
    DEFAULT_MAX_REQUERIES,
    RequeryAttempt,
    RequeryLoop,
    RequeryReason,
    RequeryResult,
    build_requery_history,
    validate_json,
    validate_non_empty,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_default_max_requeries_matches_swe_agent():
    assert DEFAULT_MAX_REQUERIES == 3


def test_requery_reason_values_stable():
    assert RequeryReason.FORMAT.value == "format"
    assert RequeryReason.BASH_SYNTAX.value == "bash_syntax"
    assert RequeryReason.BLOCKED_ACTION.value == "blocked_action"
    assert RequeryReason.CONTENT_POLICY.value == "content_policy"
    assert RequeryReason.EMPTY.value == "empty"


# ---------------------------------------------------------------------------
# build_requery_history
# ---------------------------------------------------------------------------


def test_build_history_includes_broken_and_error():
    base = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    out = build_requery_history(
        base,
        broken_output="bad",
        error_template="please fix",
    )
    assert len(out) == 4
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert out[2] == {"role": "assistant", "content": "bad"}
    assert out[3] == {"role": "user", "content": "please fix"}


def test_build_history_substitutes_template():
    out = build_requery_history(
        [], broken_output="x", error_template="reason: {r}", context={"r": "FORMAT"}
    )
    assert out[-1]["content"] == "reason: FORMAT"


def test_build_history_missing_keys_render_empty():
    out = build_requery_history(
        [], broken_output="x", error_template="a={a} b={b}", context={"a": "set"}
    )
    assert out[-1]["content"] == "a=set b="


def test_build_history_does_not_mutate_base():
    base = [{"role": "user", "content": "hi"}]
    base_id = id(base)
    out = build_requery_history(base, broken_output="x", error_template="t")
    assert id(out) != base_id
    assert len(base) == 1  # unchanged


def test_build_history_copies_message_dicts():
    base = [{"role": "user", "content": "hi"}]
    out = build_requery_history(base, broken_output="x", error_template="t")
    out[0]["content"] = "mutated"
    # Original base is unchanged because messages are shallow-copied.
    assert base[0]["content"] == "hi"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_validate_non_empty_pass():
    ok, reason, msg = validate_non_empty("hello")
    assert ok is True


def test_validate_non_empty_fail_on_empty():
    ok, reason, msg = validate_non_empty("")
    assert ok is False
    assert reason == RequeryReason.EMPTY


def test_validate_non_empty_fail_on_whitespace():
    ok, _, _ = validate_non_empty("   \n\t  ")
    assert ok is False


def test_validate_json_pass_on_object():
    ok, _, _ = validate_json('{"a": 1}')
    assert ok is True


def test_validate_json_pass_on_array():
    ok, _, _ = validate_json("[1, 2, 3]")
    assert ok is True


def test_validate_json_pass_on_fenced():
    ok, _, _ = validate_json('```json\n{"a": 1}\n```')
    assert ok is True


def test_validate_json_fail_on_garbage():
    ok, reason, msg = validate_json("not json at all")
    assert ok is False
    assert reason == RequeryReason.FORMAT


# ---------------------------------------------------------------------------
# RequeryLoop -- happy path
# ---------------------------------------------------------------------------


def test_loop_succeeds_on_first_attempt():
    loop = RequeryLoop(
        generate_fn=lambda _: "should not be called",
        validate_fn=validate_non_empty,
    )
    result = loop.run([], initial_output="already valid")
    assert result.succeeded is True
    assert result.final_output == "already valid"
    assert result.attempts == []


def test_loop_succeeds_on_second_attempt():
    """First output empty, second non-empty."""

    def gen(_h: Sequence[Mapping[str, Any]]) -> str:
        return "corrected"

    loop = RequeryLoop(generate_fn=gen, validate_fn=validate_non_empty)
    result = loop.run([{"role": "user", "content": "hi"}], initial_output="")
    assert result.succeeded is True
    assert result.final_output == "corrected"
    # One attempt: the initial empty one (now marked succeeded=True
    # because the corrected output retroactively wins).
    assert len(result.attempts) == 1


def test_loop_max_retries_reached():
    """All outputs fail validation; loop hits the cap."""

    def gen(_h: Sequence[Mapping[str, Any]]) -> str:
        return ""  # always empty

    loop = RequeryLoop(
        generate_fn=gen, validate_fn=validate_non_empty, max_retries=2
    )
    result = loop.run([], initial_output="")
    assert result.succeeded is False
    assert result.max_retries_reached is True
    assert len(result.attempts) == 3  # initial + 2 retries


def test_loop_generate_fn_exception_records_failure():
    def boom(_h: Sequence[Mapping[str, Any]]) -> str:
        raise RuntimeError("LLM is down")

    loop = RequeryLoop(
        generate_fn=boom, validate_fn=validate_non_empty, max_retries=3
    )
    result = loop.run([], initial_output="")
    assert result.succeeded is False
    # Records the failed initial attempt + a special exception attempt.
    reasons = [a.reason for a in result.attempts]
    assert RequeryReason.OTHER in reasons


def test_loop_validate_fn_exception_treated_as_failure():
    def bad_validate(_o: str) -> tuple[bool, RequeryReason, str]:
        raise RuntimeError("validator boom")

    captures: list[Sequence[Mapping[str, Any]]] = []

    def gen(h: Sequence[Mapping[str, Any]]) -> str:
        captures.append(h)
        return "still no good"

    loop = RequeryLoop(
        generate_fn=gen,
        validate_fn=bad_validate,
        max_retries=2,
    )
    result = loop.run([], initial_output="anything")
    assert result.succeeded is False
    # Validator exceptions classify as RequeryReason.OTHER.
    assert result.attempts[0].reason == RequeryReason.OTHER


def test_loop_max_retries_zero():
    """When max_retries=0, only the initial output is checked."""

    def gen(_h: Sequence[Mapping[str, Any]]) -> str:
        return "shouldn't fire"

    loop = RequeryLoop(
        generate_fn=gen, validate_fn=validate_non_empty, max_retries=0
    )
    result = loop.run([], initial_output="")
    assert result.succeeded is False
    assert result.max_retries_reached is True
    assert len(result.attempts) == 1


# ---------------------------------------------------------------------------
# Loop -- history doesn't pollute
# ---------------------------------------------------------------------------


def test_loop_base_messages_never_mutated():
    base = [{"role": "user", "content": "original"}]
    captures: list[list[Mapping[str, Any]]] = []

    def gen(h: Sequence[Mapping[str, Any]]) -> str:
        captures.append(list(h))
        return ""  # keeps failing

    loop = RequeryLoop(generate_fn=gen, validate_fn=validate_non_empty, max_retries=2)
    loop.run(base, initial_output="")
    # The caller's base messages must be unchanged.
    assert base == [{"role": "user", "content": "original"}]


def test_loop_temp_history_carries_broken_output():
    captures: list[list[Mapping[str, Any]]] = []

    def gen(h: Sequence[Mapping[str, Any]]) -> str:
        captures.append([dict(m) for m in h])
        return ""

    # Empty initial output forces a re-query so gen fires at least once.
    loop = RequeryLoop(generate_fn=gen, validate_fn=validate_non_empty, max_retries=1)
    loop.run([{"role": "user", "content": "u"}], initial_output="")
    # First (and only) gen call should see u + assistant="" + user=error.
    assert len(captures) == 1
    seq = captures[0]
    assert seq[-2]["role"] == "assistant"
    assert seq[-2]["content"] == ""


# ---------------------------------------------------------------------------
# Context per attempt
# ---------------------------------------------------------------------------


def test_context_per_attempt_overrides_default():
    contexts_seen: list[Mapping[str, Any]] = []

    def gen(h: Sequence[Mapping[str, Any]]) -> str:
        # Capture the rendered template content (the last user message).
        contexts_seen.append(dict(h[-1]))
        return ""

    def ctx(idx: int, err: str) -> Mapping[str, Any]:
        return {"r": f"custom_{idx}", "error": err}

    loop = RequeryLoop(
        generate_fn=gen,
        validate_fn=validate_non_empty,
        error_template="iter={r} err={error}",
        max_retries=1,
    )
    loop.run([], initial_output="", context_per_attempt=ctx)
    assert any("custom_0" in str(c["content"]) for c in contexts_seen)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_negative_max_retries_raises():
    with pytest.raises(ValueError):
        RequeryLoop(
            generate_fn=lambda _h: "",
            validate_fn=validate_non_empty,
            max_retries=-1,
        )


def test_requery_attempt_is_frozen():
    a = RequeryAttempt(
        attempt_index=0,
        reason=RequeryReason.FORMAT,
        error_message="x",
        broken_output="o",
        corrected_output=None,
        succeeded=False,
    )
    with pytest.raises(Exception):
        a.attempt_index = 1  # type: ignore[misc]
