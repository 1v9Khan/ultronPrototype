"""Thinking-mode toggle (2026-06-19): a runtime switch for whether the relay path
may AUTHOR via the LLM.

OFF (the default) -> every "compose" command (the identity probes, the social
reactions flaming / cringe / shut-up / arguing, flame-enemy/agent, backhanded
praise, think-and-respond) snaps from its DETERMINISTIC pool instead of the 3B,
so it works on the flavor-ON path too -- instant and predictable. ON -> the
orchestrator passes rephrase=True and those commands go back to the LLM.

These assert the matcher + the state + that thinking-OFF needs no LLM, at the
stable public-API level (no Orchestrator, no model).
"""
from __future__ import annotations

import pytest

from kenning.audio.command_normalizer import normalize_command
from kenning.audio.relay_speech import (
    match_relay_command, build_relay_line, match_thinking_toggle,
    thinking_mode_enabled, set_thinking_mode_enabled, set_flavor_tails_enabled,
)


class TestThinkingToggleMatcher:
    @pytest.mark.parametrize("text", [
        "thinking mode on", "thinking on", "turn on thinking mode",
        "enable thinking mode", "ultron thinking mode on", "think mode on",
        "turn thinking mode back on", "reasoning on", "thinking mode back on",
        "allow thinking mode",
    ])
    def test_on_forms(self, text) -> None:
        assert match_thinking_toggle(text) is True

    @pytest.mark.parametrize("text", [
        "thinking mode off", "thinking off", "turn off thinking mode",
        "disable thinking mode", "stop thinking", "no thinking mode",
        "ultron thinking mode off", "think mode off", "reasoning off",
    ])
    def test_off_forms(self, text) -> None:
        assert match_thinking_toggle(text) is False

    @pytest.mark.parametrize("text", [
        # ordinary speech with "think"/"on"/"off" must NOT toggle
        "what do you think", "I think they are A", "let me think about it",
        "push B now", "Sage nice try", "tell my team I got this",
        "flavor off", "flavor on", "they are on site", "one is off angle",
    ])
    def test_negatives(self, text) -> None:
        assert match_thinking_toggle(text) is None


class TestThinkingModeState:
    def test_default_off(self) -> None:
        # the env default (KENNING_THINKING_MODE unset) is OFF
        assert thinking_mode_enabled() is False

    def test_round_trip(self) -> None:
        prev = thinking_mode_enabled()
        try:
            set_thinking_mode_enabled(True)
            assert thinking_mode_enabled() is True
            set_thinking_mode_enabled(False)
            assert thinking_mode_enabled() is False
        finally:
            set_thinking_mode_enabled(prev)


class TestThinkingOffIsDeterministic:
    """With thinking OFF the orchestrator passes rephrase=False, so every compose
    command must render a sensible line with NO llm -- in BOTH flavor states."""

    COMPOSE = [
        "Sage asked if you are a soundboard, respond",
        "the team asked if I am a soundboard, respond",
        "Sage asked if I am a voice changer, respond",
        "Sage asked if you are a streamer, respond",
        "Sage is flaming you",
        "Sage called you cringe",
        "the team is arguing",
        "Sage told you to shut up",
        "flame the enemy",
        "flame my Sage",
        "praise Sage",
    ]

    @pytest.mark.parametrize("text", COMPOSE)
    def test_snaps_without_llm(self, text) -> None:
        prev = thinking_mode_enabled()
        try:
            set_thinking_mode_enabled(False)
            for tails in (True, False):
                set_flavor_tails_enabled(tails)
                cmd = match_relay_command(normalize_command(text))
                assert cmd is not None, f"no relay match for {text!r}"
                # rephrase=False, llm=None == the thinking-OFF live path
                line = build_relay_line(cmd, None, rephrase=False)
                assert line and len(line) > 3, f"empty/short: {line!r}"
                assert "None" not in line, f"stray None in {line!r}"
        finally:
            set_flavor_tails_enabled(False)
            set_thinking_mode_enabled(prev)
