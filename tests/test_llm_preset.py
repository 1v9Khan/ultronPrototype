"""Preset resolution tests (2026-05-14 update).

Verifies ``LLMConfig.preset`` behaviour:
- Default preset ``josiefied-qwen3-4b`` resolves to the
  Goekdeniz-Guelmez Josiefied + abliterated Qwen3-4B-v2 Q5_K_M
  (current default since 2026-05-14 -- VRAM relief on the 4070 Ti).
  The abliterated model removes content-level refusals; the runtime
  tool-call validator under ``src/ultron/safety/`` gates the actual
  capability surface. No paired draft so no speculative decoding for
  this preset.
- ``josiefied-qwen3-8b`` resolves to the 8B variant of the same
  Josiefied + abliterated lineage. Retained for swap-back when the
  user wants the bigger abliterated model (~5.85 GB on disk).
- ``qwen3.5-9b`` resolves to the 9B GGUF + n_ctx=8192, no draft.
  Retained for swap-back. Not abliterated.
- ``qwen3.5-4b`` resolves to the 4B GGUF + 0.8B draft + n_ctx=8192.
  Retained for swap-back / speculative decoding. Not abliterated.
- ``custom`` does not touch any field; raw user values pass through.
- Explicit user fields always win over preset defaults (mixed mode).

These tests construct ``LLMConfig`` directly and load YAML fragments
through ``load_config``, so they cover both call paths.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ultron.config import LLM_PRESETS, LLMConfig, load_config


def test_default_preset_is_josiefied_4b() -> None:
    """2026-05-14: default flipped to Josiefied + abliterated Qwen3-4B-v2
    Q5_K_M for VRAM relief. Same abliterated lineage as the 8B at
    ~half the footprint. n_ctx=6144 (vs 8192 on the larger presets)
    shaves another ~150 MB off the KV cache. The runtime tool-call
    validator (``src/ultron/safety/``) still gates the actual capability
    surface."""
    cfg = LLMConfig()
    assert cfg.preset == "josiefied-qwen3-4b"
    assert cfg.model_path == "models/Josiefied-Qwen3-4B-abliterated-v2.Q5_K_M.gguf"
    assert cfg.n_ctx == 6144
    # No paired draft model -- no abliterated 0.6B / 0.8B GGUF on HF.
    assert cfg.draft_model_path is None


def test_josiefied_8b_preset_still_available() -> None:
    """Josiefied 8B is retained for swap-back when the user wants the
    larger abliterated model. Same lineage, just bigger."""
    cfg = LLMConfig(preset="josiefied-qwen3-8b")
    assert cfg.preset == "josiefied-qwen3-8b"
    assert cfg.model_path == (
        "models/Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf"
    )
    assert cfg.n_ctx == 8192
    assert cfg.draft_model_path is None


def test_josiefied_4b_preset_resolves_paths_and_ctx() -> None:
    """Explicitly setting the new preset name resolves the same way.

    n_ctx=6144 (smaller than the other presets' 8192) trims another
    ~150 MB off the KV cache without affecting voice / screen-context
    typical use -- see comments in LLM_PRESETS for the budget math.
    """
    cfg = LLMConfig(preset="josiefied-qwen3-4b")
    assert cfg.preset == "josiefied-qwen3-4b"
    assert cfg.model_path == "models/Josiefied-Qwen3-4B-abliterated-v2.Q5_K_M.gguf"
    assert cfg.n_ctx == 6144
    assert cfg.draft_model_path is None


def test_legacy_9b_preset_still_available() -> None:
    """The 9B preset is retained for swap-back / A-B comparison."""
    cfg = LLMConfig(preset="qwen3.5-9b")
    assert cfg.preset == "qwen3.5-9b"
    assert cfg.model_path == "models/Qwen3.5-9B-Q4_K_M.gguf"
    assert cfg.n_ctx == 8192
    assert cfg.draft_model_path is None


def test_4b_preset_resolves_paths_and_ctx() -> None:
    cfg = LLMConfig(preset="qwen3.5-4b")
    assert cfg.preset == "qwen3.5-4b"
    assert cfg.model_path == "models/Qwen3.5-4B-Q4_K_M.gguf"
    assert cfg.n_ctx == 8192
    assert cfg.draft_model_path == "models/Qwen3.5-0.8B-Q4_K_M.gguf"


def test_custom_preset_passes_through_raw_fields() -> None:
    cfg = LLMConfig(
        preset="custom",
        model_path="models/some-other.gguf",
        n_ctx=4096,
        draft_model_path=None,
    )
    assert cfg.preset == "custom"
    assert cfg.model_path == "models/some-other.gguf"
    assert cfg.n_ctx == 4096
    assert cfg.draft_model_path is None


def test_explicit_model_path_overrides_4b_preset() -> None:
    """Mixed mode — preset gives n_ctx + draft, user pins model_path."""
    cfg = LLMConfig(preset="qwen3.5-4b", model_path="models/custom-4b.gguf")
    assert cfg.model_path == "models/custom-4b.gguf"  # user wins
    assert cfg.n_ctx == 8192  # preset still applies to non-overridden
    assert cfg.draft_model_path == "models/Qwen3.5-0.8B-Q4_K_M.gguf"


def test_explicit_n_ctx_overrides_preset() -> None:
    cfg = LLMConfig(preset="qwen3.5-4b", n_ctx=8192)
    assert cfg.n_ctx == 8192
    assert cfg.model_path == "models/Qwen3.5-4B-Q4_K_M.gguf"  # preset still applies


def test_explicit_draft_model_path_overrides_preset() -> None:
    cfg = LLMConfig(preset="qwen3.5-4b", draft_model_path=None)
    assert cfg.draft_model_path is None  # explicit None wins
    assert cfg.model_path == "models/Qwen3.5-4B-Q4_K_M.gguf"


def test_9b_preset_does_not_set_draft() -> None:
    """9B explicitly has draft_model_path=None — switching from 4B back to
    9B in the same process must not retain a stale draft path."""
    cfg = LLMConfig(preset="qwen3.5-9b")
    assert cfg.draft_model_path is None


def test_custom_preset_with_default_model_path_is_legal() -> None:
    """custom preset doesn't auto-resolve, but the field has a default
    ('models/Qwen3.5-9B-Q4_K_M.gguf'), so this is allowed."""
    cfg = LLMConfig(preset="custom")
    assert cfg.preset == "custom"
    assert cfg.model_path == "models/Qwen3.5-9B-Q4_K_M.gguf"


def test_preset_table_contents() -> None:
    """The preset table is the contract that the launcher, swap script,
    and 4B-plan docs depend on. Lock it down. ``custom`` is the schema-
    only sentinel and does NOT appear in LLM_PRESETS."""
    assert set(LLM_PRESETS.keys()) == {
        "qwen3.5-9b", "qwen3.5-4b",
        "josiefied-qwen3-8b", "josiefied-qwen3-4b",
    }
    nine = LLM_PRESETS["qwen3.5-9b"]
    four = LLM_PRESETS["qwen3.5-4b"]
    eight_jos = LLM_PRESETS["josiefied-qwen3-8b"]
    four_jos = LLM_PRESETS["josiefied-qwen3-4b"]
    assert nine["model_path"].endswith("Qwen3.5-9B-Q4_K_M.gguf")
    assert nine["draft_model_path"] is None
    assert nine["n_ctx"] == 8192
    assert four["model_path"].endswith("Qwen3.5-4B-Q4_K_M.gguf")
    assert four["draft_model_path"].endswith("Qwen3.5-0.8B-Q4_K_M.gguf")
    assert four["n_ctx"] == 8192
    assert eight_jos["model_path"].endswith(
        "Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf"
    )
    assert eight_jos["draft_model_path"] is None
    assert eight_jos["n_ctx"] == 8192
    assert four_jos["model_path"].endswith(
        "Josiefied-Qwen3-4B-abliterated-v2.Q5_K_M.gguf"
    )
    assert four_jos["draft_model_path"] is None
    # 2026-05-14: 4B abliterated uses n_ctx=6144 (smaller than the
    # other presets' 8192) to trim ~150 MB off the KV cache.
    assert four_jos["n_ctx"] == 6144


def test_yaml_load_with_4b_preset(tmp_path: Path) -> None:
    """End-to-end: YAML config with preset key loads cleanly through
    the real loader and resolves the same way."""
    yaml_text = """
version: "1.0"
llm:
  preset: "qwen3.5-4b"
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.llm.preset == "qwen3.5-4b"
    assert cfg.llm.model_path == "models/Qwen3.5-4B-Q4_K_M.gguf"
    assert cfg.llm.n_ctx == 8192
    assert cfg.llm.draft_model_path == "models/Qwen3.5-0.8B-Q4_K_M.gguf"


def test_yaml_load_default_preset_back_compat(tmp_path: Path) -> None:
    """A YAML config that does not specify ``preset`` falls back to the
    schema default. As of 2026-05-14 the schema default is the
    Josiefied + abliterated Qwen3-4B-v2 preset (was josiefied-qwen3-8b
    before that, qwen3.5-4b before that, qwen3.5-9b before the 4B plan).
    This test pins the schema default for documentation; the production
    ``config.yaml`` always spells the preset out explicitly."""
    yaml_text = """
version: "1.0"
llm:
  history_turns: 6
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.llm.preset == "josiefied-qwen3-4b"
    assert cfg.llm.model_path == (
        "models/Josiefied-Qwen3-4B-abliterated-v2.Q5_K_M.gguf"
    )
    # 4B abliterated preset uses n_ctx=6144 (KV-cache trim).
    # Test omits an explicit n_ctx so the preset default applies.
    assert cfg.llm.n_ctx == 6144
    assert cfg.llm.draft_model_path is None


def test_yaml_load_custom_preset_with_explicit_paths(tmp_path: Path) -> None:
    yaml_text = """
version: "1.0"
llm:
  preset: "custom"
  model_path: "models/something-bespoke.gguf"
  n_ctx: 4096
  draft_model_path: "models/tiny-draft.gguf"
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.llm.preset == "custom"
    assert cfg.llm.model_path == "models/something-bespoke.gguf"
    assert cfg.llm.n_ctx == 4096
    assert cfg.llm.draft_model_path == "models/tiny-draft.gguf"


def test_invalid_preset_rejected() -> None:
    with pytest.raises(Exception):  # pydantic ValidationError
        LLMConfig(preset="qwen2-7b")  # not in Literal
