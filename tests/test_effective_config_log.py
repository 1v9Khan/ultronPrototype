"""Tests for ultron.config.log_effective_config.

The function is a startup diagnostic, not a behavioural change. We
verify that (a) all the high-impact sections get logged, (b) ULTRON_*
env vars surface, (c) known-secret env vars are elided, and (d) the
function never raises -- it must remain fail-open through any
malformed config or partial section.
"""

from __future__ import annotations

import logging

from ultron.config import (
    CodingSupervisorConfig,
    UltronConfig,
    log_effective_config,
)


# ---------------------------------------------------------------------------
# Smoke + section coverage
# ---------------------------------------------------------------------------


def test_log_effective_config_emits_llm_section(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(cfg, env={})

    messages = "\n".join(r.message for r in caplog.records)
    assert "LLM preset=" in messages


def test_log_effective_config_emits_tts_section(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(cfg, env={})

    messages = "\n".join(r.message for r in caplog.records)
    assert "TTS engine=" in messages


def test_log_effective_config_emits_stt_section(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(cfg, env={})

    messages = "\n".join(r.message for r in caplog.records)
    assert "STT engine=" in messages


def test_log_effective_config_emits_memory_section(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(cfg, env={})

    messages = "\n".join(r.message for r in caplog.records)
    assert "MEMORY" in messages
    assert "reranking=" in messages


def test_log_effective_config_emits_supervisor_section(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(cfg, env={})

    messages = "\n".join(r.message for r in caplog.records)
    assert "SUPERVISOR tier=" in messages


def test_log_effective_config_emits_gaming_mode_section(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(cfg, env={})

    messages = "\n".join(r.message for r in caplog.records)
    assert "GAMING_MODE" in messages


# ---------------------------------------------------------------------------
# Env var surfacing
# ---------------------------------------------------------------------------


def test_no_ultron_env_vars_logged_when_absent(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(cfg, env={"PATH": "/usr/bin"})

    messages = "\n".join(r.message for r in caplog.records)
    assert "no ULTRON_* env vars set" in messages


def test_ultron_env_var_listed(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(
            cfg,
            env={"ULTRON_LLM_PRESET": "qwen3.5-9b"},
        )

    messages = "\n".join(r.message for r in caplog.records)
    assert "ULTRON_LLM_PRESET" in messages
    assert "qwen3.5-9b" in messages


def test_known_env_override_gets_note(caplog) -> None:
    """ULTRON_LLM_MODEL_PATH must surface with its load-bearing warning."""
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(
            cfg,
            env={"ULTRON_LLM_MODEL_PATH": "models/Qwen3.5-9B-Q4_K_M.gguf"},
        )

    messages = "\n".join(r.message for r in caplog.records)
    assert "ULTRON_LLM_MODEL_PATH" in messages
    # The note flags this as a known footgun.
    assert "silently overrides" in messages or "9B/4B" in messages


def test_unknown_ultron_env_var_still_logged(caplog) -> None:
    """A novel ULTRON_* env var that isn't in the catalog still surfaces."""
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(
            cfg,
            env={"ULTRON_FUTURE_THING": "yes"},
        )

    messages = "\n".join(r.message for r in caplog.records)
    assert "ULTRON_FUTURE_THING" in messages
    assert "override active" in messages


def test_brave_api_key_value_elided(caplog) -> None:
    """API keys must NEVER appear verbatim in log output."""
    cfg = UltronConfig()
    secret = "br-abcdef1234567890-secret-token"
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(
            cfg,
            env={"ULTRON_BRAVE_API_KEY": secret},
        )

    messages = "\n".join(r.message for r in caplog.records)
    assert "ULTRON_BRAVE_API_KEY" in messages
    assert "<set>" in messages
    assert secret not in messages


def test_empty_api_key_marked_empty(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(
            cfg,
            env={"ULTRON_BRAVE_API_KEY": ""},
        )

    messages = "\n".join(r.message for r in caplog.records)
    assert "<empty>" in messages


def test_non_ultron_env_vars_ignored(caplog) -> None:
    cfg = UltronConfig()
    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(
            cfg,
            env={
                "PATH": "/usr/bin",
                "USERNAME": "alice",
                "ULTRON_LLM_PRESET": "qwen3.5-4b",
            },
        )

    messages = "\n".join(r.message for r in caplog.records)
    # Non-ULTRON env vars must NOT appear.
    assert "USERNAME" not in messages
    assert "alice" not in messages
    # The ULTRON_* one must.
    assert "ULTRON_LLM_PRESET" in messages


# ---------------------------------------------------------------------------
# Supervisor tier surfaces correctly
# ---------------------------------------------------------------------------


def test_supervisor_tier_reflected_in_log(caplog) -> None:
    cfg = UltronConfig()
    cfg.coding.supervisor = CodingSupervisorConfig(tier="deciding")

    with caplog.at_level(logging.INFO, logger="ultron.config.effective"):
        log_effective_config(cfg, env={})

    messages = "\n".join(r.message for r in caplog.records)
    assert "tier='deciding'" in messages
    assert "decide=True" in messages
    assert "narrate=False" in messages


# ---------------------------------------------------------------------------
# Fail-open contract
# ---------------------------------------------------------------------------


def test_log_effective_config_never_raises_on_missing_cfg() -> None:
    """If get_config() fails, the function logs WARN and returns cleanly."""
    # Pass None and stub the loader by monkeypatching at module level
    # to simulate a config-load failure.
    from ultron.config import set_config

    class BadConfig:
        """Anything attribute access raises on -- simulates a corrupt cfg."""

        def __getattr__(self, name):
            raise RuntimeError(f"corrupt: {name}")

    # Don't actually install BadConfig globally; pass it directly.
    log_effective_config(BadConfig(), env={})  # must not raise


def test_log_effective_config_continues_after_section_failure(caplog) -> None:
    """A single broken section logs WARN but doesn't stop later sections."""

    class PartiallyBrokenConfig:
        """LLM ok; everything else raises."""

        def __init__(self) -> None:
            self.llm = UltronConfig().llm

        def __getattr__(self, name):
            raise RuntimeError(f"broken: {name}")

    with caplog.at_level(logging.WARNING, logger="ultron.config.effective"):
        log_effective_config(PartiallyBrokenConfig(), env={})

    messages = "\n".join(r.message for r in caplog.records)
    # WARN logged for each section that broke.
    assert "TTS section failed" in messages or "MEMORY section failed" in messages
