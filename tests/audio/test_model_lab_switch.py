"""Model-lab voice hot-swap tests (Ultron 0.1.1, 2026-06-19).

Two surfaces:

1. ``relay_speech.match_model_lab_switch`` -- the strict voice matcher that maps
   a "switch to <model>" / "load model N" utterance to a config.LLM_PRESETS
   preset name (or None). Must fire on every roster alias + every numbered
   fallback, and abstain on game callouts / banter / junk.

2. ``Orchestrator._maybe_handle_model_lab_switch`` -- the handler: acks, calls
   ``self.llm.reload_for_preset(preset)``, no-ops with an ack when already on the
   target preset, and is fail-open. Exercised with a mock ``llm`` that records
   the reload + patched ``get_config`` so no real model is loaded.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kenning.audio.relay_speech import (
    MODEL_LAB_ROSTER,
    MODEL_LAB_SPOKEN,
    match_model_lab_switch,
)
from kenning.config import LLM_PRESETS
from kenning.pipeline.orchestrator import Orchestrator


# Preset constants (keep the tests independent of roster ordering).
P_3B = "llama-3.2-3b-abliterated"
P_HERETIC = "heretic-4b"
P_JOSIE_4B = "josiefied-qwen3-4b"
P_HUIHUI = "huihui-qwen25-7b"
P_JOSIE_8B = "josiefied-qwen3-8b"


# ---------------------------------------------------------------------------
# Roster wiring sanity
# ---------------------------------------------------------------------------


def test_roster_presets_all_exist_in_llm_presets():
    for entry in MODEL_LAB_ROSTER:
        assert entry.preset in LLM_PRESETS, entry.preset


def test_new_presets_are_cpu_by_default():
    # Gaming mode keeps every lab preset on CPU unless the device switch moves
    # it; the new presets must declare gpu_layers=0.
    assert LLM_PRESETS[P_HERETIC]["gpu_layers"] == 0
    assert LLM_PRESETS[P_HUIHUI]["gpu_layers"] == 0


def test_spoken_map_covers_every_preset():
    for entry in MODEL_LAB_ROSTER:
        assert MODEL_LAB_SPOKEN[entry.preset] == entry.spoken


# ---------------------------------------------------------------------------
# Voice matcher -- named aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # 3B baseline
        ("switch to the 3B", P_3B),
        ("load the 3b", P_3B),
        ("use baseline", P_3B),
        ("switch to baseline model", P_3B),
        ("try the three b", P_3B),
        # heretic 4B (+ ASR mishears)
        ("switch to heretic", P_HERETIC),
        ("load heretic", P_HERETIC),
        ("try the heretic 4b", P_HERETIC),
        ("switch to heretical", P_HERETIC),  # mishear
        ("switch to her ethic", P_HERETIC),  # mishear
        ("use the heretic model", P_HERETIC),
        # josiefied 4B (+ mishears)
        ("switch to josiefied 4b", P_JOSIE_4B),
        ("load the small josiefied", P_JOSIE_4B),
        ("switch to josie fied four b", P_JOSIE_4B),  # spaced mishear
        ("use josephied 4b", P_JOSIE_4B),  # mishear
        # huihui Qwen2.5 7B (+ mishears)
        ("switch to huihui", P_HUIHUI),
        ("load the 7B", P_HUIHUI),
        ("switch to qwen 2.5", P_HUIHUI),
        ("try hway hway", P_HUIHUI),  # mishear
        ("switch to we we", P_HUIHUI),  # mishear
        ("switch to hoy hoy", P_HUIHUI),  # mishear
        ("use the seven b model", P_HUIHUI),
        # josiefied 8B
        ("switch to josiefied 8b", P_JOSIE_8B),
        ("load the big josiefied", P_JOSIE_8B),
        ("switch to the 8B", P_JOSIE_8B),
        ("try the eight b", P_JOSIE_8B),
    ],
)
def test_match_named_aliases(text, expected):
    assert match_model_lab_switch(text) == expected


# ---------------------------------------------------------------------------
# Voice matcher -- numbered fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("switch to model one", P_3B),
        ("load model two", P_HERETIC),
        ("switch to model three", P_JOSIE_4B),
        ("use model four", P_HUIHUI),
        ("try model five", P_JOSIE_8B),
        # digit forms + verb variants
        ("switch to model 1", P_3B),
        ("give me model 2", P_HERETIC),
        ("run model 4", P_HUIHUI),
        ("change to model five", P_JOSIE_8B),
    ],
)
def test_match_numbered_fallback(text, expected):
    assert match_model_lab_switch(text) == expected


def test_numbered_fallback_aligns_with_roster_order():
    # "model N" maps to the Nth roster entry by construction.
    words = ["one", "two", "three", "four", "five"]
    for i, word in enumerate(words):
        assert match_model_lab_switch(f"switch to model {word}") \
            == MODEL_LAB_ROSTER[i].preset


def test_model_six_out_of_range_is_none():
    # Only one..five are accepted; "model six" never matches.
    assert match_model_lab_switch("switch to model six") is None


# ---------------------------------------------------------------------------
# Voice matcher -- negatives (callouts / banter / junk / fail-open)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "two A main",
        "tell my team rotate",
        "heretic is such a cool name",       # alias present, no switch verb
        "i love the 7B players on the enemy team",
        "they have a 3b stack pushing",
        "model is washed",                   # "model" but no number/verb shape
        "switch agents to jett",             # switch verb, not a model alias
        "what model are you running",        # question, no switch verb lead
        "load into the next map",            # load verb, not a model
        "use the heretic main callout",      # alias mid-callout, trailing words
        "the eight b enemy is one shot",     # number+b inside a callout
    ],
)
def test_non_matches_return_none(text):
    assert match_model_lab_switch(text) is None


def test_fail_open_on_non_string():
    # A non-string must not raise -- fail-open returns None.
    assert match_model_lab_switch(None) is None  # type: ignore[arg-type]
    assert match_model_lab_switch(12345) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Orchestrator handler
# ---------------------------------------------------------------------------


def _make_handler_obj(current_preset: str):
    """A minimal object bound to the real handler with a mock llm + _speak."""
    obj = MagicMock()
    obj.llm = MagicMock()
    obj.llm.reload_for_preset.return_value = (True, "ok")
    obj.spoken = []
    obj._speak = lambda text: obj.spoken.append(text)
    return obj


def _call_handler(obj, text):
    # Bind the unbound class method to our stand-in object.
    return Orchestrator._maybe_handle_model_lab_switch(obj, text)


def test_handler_swaps_to_new_preset():
    obj = _make_handler_obj(current_preset=P_3B)
    with patch("kenning.config.get_config") as gc:
        gc.return_value.llm.preset = P_3B
        handled = _call_handler(obj, "switch to heretic")
    assert handled is True
    obj.llm.reload_for_preset.assert_called_once_with(P_HERETIC)
    # Acked BEFORE the reload + a success line after.
    assert any("Loading" in s for s in obj.spoken)
    assert any("heretic" in s.lower() for s in obj.spoken)


def test_handler_noop_when_already_on_preset():
    obj = _make_handler_obj(current_preset=P_HERETIC)
    with patch("kenning.config.get_config") as gc:
        gc.return_value.llm.preset = P_HERETIC
        handled = _call_handler(obj, "switch to heretic")
    assert handled is True
    obj.llm.reload_for_preset.assert_not_called()
    assert any("Already on" in s for s in obj.spoken)


def test_handler_non_match_returns_false():
    obj = _make_handler_obj(current_preset=P_3B)
    with patch("kenning.config.get_config") as gc:
        gc.return_value.llm.preset = P_3B
        handled = _call_handler(obj, "tell my team rotate")
    assert handled is False
    obj.llm.reload_for_preset.assert_not_called()
    assert obj.spoken == []


def test_handler_numbered_form_routes_to_handler():
    obj = _make_handler_obj(current_preset=P_3B)
    with patch("kenning.config.get_config") as gc:
        gc.return_value.llm.preset = P_3B
        handled = _call_handler(obj, "load model four")
    assert handled is True
    obj.llm.reload_for_preset.assert_called_once_with(P_HUIHUI)


def test_handler_fail_open_on_reload_error():
    obj = _make_handler_obj(current_preset=P_3B)
    obj.llm.reload_for_preset.side_effect = RuntimeError("boom")
    with patch("kenning.config.get_config") as gc:
        gc.return_value.llm.preset = P_3B
        handled = _call_handler(obj, "switch to heretic")
    # Matched -> consumes the turn (True) and speaks a graceful failure line.
    assert handled is True
    assert any("didn't work" in s.lower() for s in obj.spoken)


def test_handler_reload_failure_speaks_message():
    obj = _make_handler_obj(current_preset=P_3B)
    obj.llm.reload_for_preset.return_value = (False, "preset files missing")
    with patch("kenning.config.get_config") as gc:
        gc.return_value.llm.preset = P_3B
        handled = _call_handler(obj, "switch to heretic")
    assert handled is True
    assert any("Couldn't switch" in s for s in obj.spoken)
