"""Tests for ultron.evolution.intent -- the strict voice-command matcher."""

from __future__ import annotations

import pytest

from ultron.evolution.intent import EvolutionCommandKind, match_evolution_command


@pytest.mark.parametrize(
    "text",
    [
        "run evolution",
        "run an evolution cycle",
        "evolve your skills",
        "evolve yourself now",
        "self improve",
        "self-improve now",
        "distill a new skill",
        "improve yourself now",
    ],
)
def test_run_cycle_commands(text):
    cmd = match_evolution_command(text)
    assert cmd is not None
    assert cmd.kind is EvolutionCommandKind.RUN_CYCLE


@pytest.mark.parametrize(
    "text",
    [
        "evolution status",
        "what's the evolution digest",
        "self improvement report",
        "how have you been evolving",
        "what skills have you learned",
    ],
)
def test_status_commands(text):
    cmd = match_evolution_command(text)
    assert cmd is not None
    assert cmd.kind is EvolutionCommandKind.STATUS


@pytest.mark.parametrize(
    "text",
    [
        "how do pokemon evolve",
        "what's the weather today",
        "open youtube on monitor two",
        "improve my code please",
        "",
        "tell me about evolution as a topic",
    ],
)
def test_non_commands_return_none(text):
    assert match_evolution_command(text) is None
