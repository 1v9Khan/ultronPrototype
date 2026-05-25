"""Tests for the multi-tier context-window guard (T4)."""

from __future__ import annotations

import logging

import pytest

from ultron.llm import context_window_guard as cwg


# ----------------------------------------------------------------------
# _normalise_positive_int


def test_normalise_none_returns_none() -> None:
    assert cwg._normalise_positive_int(None) is None


def test_normalise_zero_returns_none() -> None:
    assert cwg._normalise_positive_int(0) is None


def test_normalise_negative_returns_none() -> None:
    assert cwg._normalise_positive_int(-1) is None


def test_normalise_string_returns_none() -> None:
    assert cwg._normalise_positive_int("8192") is None


def test_normalise_bool_returns_none() -> None:
    assert cwg._normalise_positive_int(True) is None
    assert cwg._normalise_positive_int(False) is None


def test_normalise_positive_int_passes_through() -> None:
    assert cwg._normalise_positive_int(8192) == 8192


def test_normalise_positive_float_rounds_down() -> None:
    assert cwg._normalise_positive_int(8192.7) == 8192


def test_normalise_inf_returns_none() -> None:
    assert cwg._normalise_positive_int(float("inf")) is None


def test_normalise_nan_returns_none() -> None:
    assert cwg._normalise_positive_int(float("nan")) is None


# ----------------------------------------------------------------------
# resolve_context_window_info — source-precedence


def test_resolve_prefers_caller_override() -> None:
    info = cwg.resolve_context_window_info(
        caller_override_tokens=10000,
        models_config_tokens=8192,
        default_tokens=4096,
    )
    assert info.tokens == 10000
    assert info.source == cwg.ContextWindowSource.CALLER_OVERRIDE


def test_resolve_models_config_when_no_override() -> None:
    info = cwg.resolve_context_window_info(
        models_config_tokens=8192,
        default_tokens=4096,
    )
    assert info.tokens == 8192
    assert info.source == cwg.ContextWindowSource.MODELS_CONFIG


def test_resolve_falls_back_to_default() -> None:
    info = cwg.resolve_context_window_info(default_tokens=4096)
    assert info.tokens == 4096
    assert info.source == cwg.ContextWindowSource.DEFAULT


def test_resolve_no_sources_unknown() -> None:
    info = cwg.resolve_context_window_info()
    assert info.tokens is None
    assert info.source == cwg.ContextWindowSource.UNKNOWN


def test_resolve_agent_cap_below_base_overrides() -> None:
    info = cwg.resolve_context_window_info(
        models_config_tokens=32768,
        agent_cap_tokens=8192,
    )
    assert info.tokens == 8192
    assert info.source == cwg.ContextWindowSource.AGENT_CAP
    assert info.reference_tokens == 32768


def test_resolve_agent_cap_above_base_no_change() -> None:
    info = cwg.resolve_context_window_info(
        models_config_tokens=8192,
        agent_cap_tokens=32768,
    )
    assert info.tokens == 8192
    assert info.source == cwg.ContextWindowSource.MODELS_CONFIG


# ----------------------------------------------------------------------
# resolve_thresholds


def test_resolve_thresholds_uses_floors_when_tokens_none() -> None:
    t = cwg.resolve_thresholds(None)
    assert t.hard_min_tokens == cwg.DEFAULT_HARD_MIN_TOKENS
    assert t.warn_below_tokens == cwg.DEFAULT_WARN_BELOW_TOKENS


def test_resolve_thresholds_scales_with_large_tokens() -> None:
    t = cwg.resolve_thresholds(200000)
    # 10% of 200k = 20k > absolute floor 4k
    assert t.hard_min_tokens == 20000
    # 20% of 200k = 40k > absolute floor 8k
    assert t.warn_below_tokens == 40000


def test_resolve_thresholds_uses_floor_when_tokens_small() -> None:
    t = cwg.resolve_thresholds(8192)
    # 10% of 8192 = 819 < floor 4000
    assert t.hard_min_tokens == cwg.DEFAULT_HARD_MIN_TOKENS
    # 20% of 8192 = 1638 < floor 8000
    assert t.warn_below_tokens == cwg.DEFAULT_WARN_BELOW_TOKENS


# ----------------------------------------------------------------------
# evaluate_context_window_guard


def test_evaluate_voice_baseline_passes() -> None:
    # qwen3.5-4b at n_ctx=8192 should clear both floors.
    result = cwg.evaluate_context_window_guard(models_config_tokens=8192)
    assert result.should_block is False
    assert result.should_warn is False


def test_evaluate_below_hard_min_blocks() -> None:
    result = cwg.evaluate_context_window_guard(models_config_tokens=2048)
    assert result.should_block is True
    assert "too small" in result.block_message
    assert "2048" in result.block_message


def test_evaluate_below_warn_but_above_block_warns() -> None:
    # 6000 > hard_min(4000), < warn(8000)
    result = cwg.evaluate_context_window_guard(models_config_tokens=6000)
    assert result.should_block is False
    assert result.should_warn is True
    assert "close to the floor" in result.warn_message


def test_evaluate_unresolved_blocks() -> None:
    result = cwg.evaluate_context_window_guard()
    assert result.should_block is True
    assert "No source" in result.block_message


def test_evaluate_agent_cap_winning_source_in_block_message() -> None:
    result = cwg.evaluate_context_window_guard(
        models_config_tokens=32768,
        agent_cap_tokens=1024,
    )
    assert result.should_block is True
    assert "agent cap" in result.block_message.lower() or "agent" in result.block_message


def test_evaluate_environment_hint_appears_in_block_message() -> None:
    result = cwg.evaluate_context_window_guard(
        models_config_tokens=1024,
        environment_hint="self-hosted llama-cpp-server",
    )
    assert "self-hosted" in result.block_message


def test_evaluate_warn_threshold_inclusive_exclusivity() -> None:
    # Exactly equal to warn threshold: should NOT warn.
    result = cwg.evaluate_context_window_guard(
        models_config_tokens=cwg.DEFAULT_WARN_BELOW_TOKENS,
    )
    assert result.should_warn is False


# ----------------------------------------------------------------------
# run_guard_or_raise


def test_run_guard_or_raise_raises_on_block() -> None:
    with pytest.raises(cwg.ContextWindowGuardError) as exc_info:
        cwg.run_guard_or_raise(models_config_tokens=1024)
    assert "too small" in str(exc_info.value)


def test_run_guard_or_raise_passes_on_healthy_budget() -> None:
    result = cwg.run_guard_or_raise(models_config_tokens=16384)
    assert result.should_block is False
    assert result.should_warn is False


def test_run_guard_or_raise_logs_warn(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="ultron.llm.context_window_guard"):
        cwg.run_guard_or_raise(models_config_tokens=6000)
    assert any("close to the floor" in record.message for record in caplog.records)


def test_run_guard_or_raise_returns_result_on_warn() -> None:
    result = cwg.run_guard_or_raise(models_config_tokens=6000)
    assert result.should_warn is True
    assert result.info.tokens == 6000


# ----------------------------------------------------------------------
# format_block_message / format_warn_message


def test_format_block_message_caller_override_source() -> None:
    info = cwg.ContextWindowInfo(tokens=1000, source=cwg.ContextWindowSource.CALLER_OVERRIDE)
    thresholds = cwg.ContextWindowThresholds(hard_min_tokens=4000, warn_below_tokens=8000)
    message = cwg.format_block_message(info, thresholds)
    assert "1000" in message
    assert "4000" in message
    assert "explicit override" in message.lower()


def test_format_block_message_unknown_source() -> None:
    info = cwg.ContextWindowInfo(tokens=None, source=cwg.ContextWindowSource.UNKNOWN)
    thresholds = cwg.ContextWindowThresholds(hard_min_tokens=4000, warn_below_tokens=8000)
    message = cwg.format_block_message(info, thresholds)
    assert "No source" in message
