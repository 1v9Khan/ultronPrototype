"""Tests for the Ultron persona lock in _gaming_conversational_prompt (2026-06-23).

BR-P2: when u1_llm_route is enabled, _gaming_conversational_prompt must return
ULTRON_GAMING_PERSONA (never None), so the workspace "You are Kenning" persona
cannot leak through to the LLM on any conversational call.
"""
from __future__ import annotations

import inspect


def test_gaming_conversational_prompt_returns_gaming_persona_when_u1_route(monkeypatch):
    """u1_llm_route_enabled() True → must return ULTRON_GAMING_PERSONA, not None."""
    from kenning.pipeline.orchestrator import Orchestrator
    from kenning.audio.llm_prompts import ULTRON_GAMING_PERSONA

    orch = Orchestrator.__new__(Orchestrator)
    orch.llm = None

    monkeypatch.setattr(
        "kenning.openclaw_routing.gaming_mode.is_gaming_mode_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        "kenning.safety.testing_mode.is_testing_mode_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        "kenning.audio.relay_speech.u1_llm_route_enabled",
        lambda: True,
        raising=False,
    )

    result = orch._gaming_conversational_prompt()
    # 2026-07-24: the selector appends a per-turn variety suffix, so the
    # persona-lock proof is PREFIX identity (still Ultron, never "Kenning").
    assert result is not None and result.startswith(ULTRON_GAMING_PERSONA), (
        "u1_llm_route active but _gaming_conversational_prompt returned None — "
        "workspace Kenning persona would leak (BR-P2)"
    )


def test_gaming_conversational_prompt_returns_none_when_desktop(monkeypatch):
    """Desktop mode (no gaming, no u1_route, no gaming model) → returns None."""
    from kenning.pipeline.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.llm = None

    monkeypatch.setattr(
        "kenning.openclaw_routing.gaming_mode.is_gaming_mode_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        "kenning.safety.testing_mode.is_testing_mode_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        "kenning.audio.relay_speech.u1_llm_route_enabled",
        lambda: False,
        raising=False,
    )

    result = orch._gaming_conversational_prompt()
    assert result is None


def test_gaming_conversational_prompt_source_contains_u1_route_check():
    """Source-level guard: the method source must reference u1_llm_route_enabled."""
    from kenning.pipeline.orchestrator import Orchestrator
    src = inspect.getsource(Orchestrator._gaming_conversational_prompt)
    assert "u1_llm_route_enabled" in src, (
        "Persona lock removed from _gaming_conversational_prompt source"
    )
