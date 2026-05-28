"""Tests for the catalog-13 evolution temperament-hint wiring in the
LLM engine.

Exercises the seam between :meth:`LLMEngine.set_temperament_hint` and
:meth:`LLMEngine._build_messages` to make sure:

* With no hint set, the system prompt is byte-identical to the base.
* A non-empty hint is appended to the system prompt for that turn.
* The hint never touches the USER message (so the web-gate / clock
  detectors that read the raw utterance are unaffected).
* The setter clears cleanly on "" / None.

Engine-stub pattern (no llama-cpp load), mirroring
``tests/skills/test_orchestrator_wiring.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_engine_stub(system_prompt: str):
    from ultron.llm.inference import LLMEngine

    engine = LLMEngine.__new__(LLMEngine)
    engine._explicit_system_prompt = system_prompt  # type: ignore[attr-defined]
    engine._static_system_prompt = system_prompt  # type: ignore[attr-defined]
    engine._persona_loader = None  # type: ignore[attr-defined]
    engine._logged_initial_persona = True  # type: ignore[attr-defined]
    engine.system_prompt = system_prompt
    engine._history = []  # type: ignore[attr-defined]
    engine._memory = None  # type: ignore[attr-defined]
    engine._cfg = MagicMock()  # type: ignore[attr-defined]
    engine._cfg.history_turns_for_llm = 0
    engine._sampling_params_for_request = lambda *a, **kw: {}  # type: ignore[attr-defined]
    return engine


def _patch_cfg(monkeypatch):
    from ultron.llm import inference as inference_mod

    fake_cfg = MagicMock()
    fake_cfg.llm.rag.position = "recency"
    monkeypatch.setattr(inference_mod, "get_config", lambda: fake_cfg)


def test_no_hint_leaves_prompt_unchanged(monkeypatch):
    _patch_cfg(monkeypatch)
    engine = _make_engine_stub("BASE SYSTEM PROMPT")
    # Unset attribute path (getattr default) is exercised here.
    msgs = engine._build_messages("what time is it")
    system_msg = next(m for m in msgs if m["role"] == "system")
    assert system_msg["content"] == "BASE SYSTEM PROMPT"
    assert "[Tone:" not in system_msg["content"]


def test_hint_appended_to_system_prompt(monkeypatch):
    _patch_cfg(monkeypatch)
    engine = _make_engine_stub("BASE SYSTEM PROMPT")
    engine.set_temperament_hint("[Tone: keep it concise.]")
    msgs = engine._build_messages("tell me about the project")
    system_msg = next(m for m in msgs if m["role"] == "system")
    assert system_msg["content"].startswith("BASE SYSTEM PROMPT")
    assert "[Tone: keep it concise.]" in system_msg["content"]


def test_hint_does_not_touch_user_message(monkeypatch):
    _patch_cfg(monkeypatch)
    engine = _make_engine_stub("BASE SYSTEM PROMPT")
    engine.set_temperament_hint("[Tone: be thorough.]")
    msgs = engine._build_messages("what day is today")
    user_msg = next(m for m in msgs if m["role"] == "user")
    # The raw utterance is preserved verbatim -- this is what the
    # web-gate + local-clock detectors read.
    assert user_msg["content"] == "what day is today"
    assert "[Tone:" not in user_msg["content"]


def test_setter_clears_on_empty_and_none(monkeypatch):
    _patch_cfg(monkeypatch)
    engine = _make_engine_stub("BASE SYSTEM PROMPT")
    engine.set_temperament_hint("[Tone: be precise and flag any uncertainty.]")
    assert engine.get_temperament_hint().startswith("[Tone:")
    engine.set_temperament_hint("")
    assert engine.get_temperament_hint() == ""
    msgs = engine._build_messages("hello")
    system_msg = next(m for m in msgs if m["role"] == "system")
    assert system_msg["content"] == "BASE SYSTEM PROMPT"
    # None also clears.
    engine.set_temperament_hint("[Tone: keep it concise.]")
    engine.set_temperament_hint(None)
    assert engine.get_temperament_hint() == ""


def test_skills_block_and_tone_compose(monkeypatch, tmp_path):
    """Both injections can coexist without clobbering each other."""
    from ultron.skills.models import SkillSource
    from ultron.skills.registry import (
        SkillRegistry,
        _SourceSpec,  # type: ignore[attr-defined]
        reset_skill_registry_for_testing,
        set_skill_registry,
    )

    reset_skill_registry_for_testing()
    try:
        (tmp_path / "core.md").write_text(
            "---\nname: core\n---\nCore always-on body.", encoding="utf-8"
        )
        set_skill_registry(
            SkillRegistry([_SourceSpec(directory=tmp_path, source=SkillSource.PUBLIC)])
        )
        _patch_cfg(monkeypatch)
        engine = _make_engine_stub("BASE SYSTEM PROMPT")
        engine.set_temperament_hint("[Tone: keep it concise.]")
        msgs = engine._build_messages("anything")
        system_msg = next(m for m in msgs if m["role"] == "system")
        assert "[Skills: core]" in system_msg["content"]
        assert "[Tone: keep it concise.]" in system_msg["content"]
    finally:
        reset_skill_registry_for_testing()
