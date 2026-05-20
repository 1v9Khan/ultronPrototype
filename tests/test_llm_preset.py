"""Preset resolution tests (2026-05-14 update).

Verifies ``LLMConfig.preset`` behaviour:
- Default preset ``josiefied-qwen3-4b`` resolves to the
  Goekdeniz-Guelmez Josiefied + abliterated Qwen3-4B-v2 Q4_K_M
  (current default since 2026-05-14 -- VRAM relief on the 4070 Ti.
  Started Q5_K_M same day; trimmed to Q4_K_M to fit alongside the
  user's ~4.7 GB of background GPU usage).
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
    assert cfg.model_path == "models/Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf"
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
    assert cfg.model_path == "models/Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf"
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
        # 2026-05-19 additions: candidates for daily-use + gaming-mode
        # swaps. GGUFs NOT yet on disk -- presets are paper-only until
        # download. swap_llm_preset.py refuses the swap if files are
        # absent.
        "gemma-3-4b-abliterated",
        "llama-3.2-3b-abliterated",
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
        "Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf"
    )
    assert four_jos["draft_model_path"] is None
    # 2026-05-14: 4B abliterated uses n_ctx=6144 (smaller than the
    # other presets' 8192) to trim ~150 MB off the KV cache.
    assert four_jos["n_ctx"] == 6144


def test_gemma_4b_abliterated_preset_resolves() -> None:
    """2026-05-19 daily-use candidate: Gemma 3 4B abliterated paired
    with the 1B draft for speculative decoding. n_ctx=4096 trims KV
    cache further (Gemma uses GQA so KV is already smaller than
    Qwen's at the same context length).

    Filename invariants: main GGUF from mradermacher uses dot
    separator (``...abliterated.Q4_K_M.gguf``); draft from bartowski
    uses hyphen (``...-Q4_K_M.gguf``). Both must match the filenames
    written by ``scripts/download_models.py`` so
    ``swap_llm_preset.py``'s GGUF-presence check passes after a
    fresh download.
    """
    cfg = LLMConfig(preset="gemma-3-4b-abliterated")
    assert cfg.preset == "gemma-3-4b-abliterated"
    assert cfg.model_path == "models/gemma-3-4b-it-abliterated.Q4_K_M.gguf"
    assert cfg.n_ctx == 4096
    assert cfg.draft_model_path == "models/google_gemma-3-1b-it-Q4_K_M.gguf"


def test_llama_3_2_3b_abliterated_preset_resolves() -> None:
    """2026-05-19 gaming-mode candidate: Llama 3.2 3B abliterated
    paired with the 1B draft. Smaller VRAM footprint than Qwen3-4B,
    naturally brief conversational tone. n_ctx=2048 because gaming
    channel utterances are short -- smaller KV cache frees memory
    for Valorant + OBS.

    Same dot/hyphen invariant as the Gemma preset above.
    """
    cfg = LLMConfig(preset="llama-3.2-3b-abliterated")
    assert cfg.preset == "llama-3.2-3b-abliterated"
    assert cfg.model_path == "models/Llama-3.2-3B-Instruct-abliterated.Q4_K_M.gguf"
    assert cfg.n_ctx == 2048
    assert cfg.draft_model_path == "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"


def test_new_presets_match_download_script_filenames() -> None:
    """Regression: every preset's model_path + draft_model_path must
    end with a filename that ``scripts/download_models.py`` actually
    writes. If the constants in the download script drift from the
    preset paths, ``swap_llm_preset.py`` refuses the swap with
    "preset files missing" -- this test catches the drift early.
    """
    import importlib
    spec = importlib.util.spec_from_file_location(
        "download_models",
        Path(__file__).resolve().parent.parent / "scripts" / "download_models.py",
    )
    # Skip the import side effects (it touches HF cache etc.) by
    # reading the source instead and grepping for the *_FILE constants.
    download_src = (
        Path(__file__).resolve().parent.parent / "scripts" / "download_models.py"
    ).read_text(encoding="utf-8")

    expected_pairs = [
        ("gemma-3-4b-abliterated", "model_path", "gemma-3-4b-it-abliterated.Q4_K_M.gguf"),
        ("gemma-3-4b-abliterated", "draft_model_path", "google_gemma-3-1b-it-Q4_K_M.gguf"),
        ("llama-3.2-3b-abliterated", "model_path", "Llama-3.2-3B-Instruct-abliterated.Q4_K_M.gguf"),
        ("llama-3.2-3b-abliterated", "draft_model_path", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ]
    for preset, key, expected_filename in expected_pairs:
        actual = LLM_PRESETS[preset][key]
        assert actual.endswith(expected_filename), (
            f"{preset}.{key} = {actual!r} does not end with "
            f"{expected_filename!r}; the download script writes that "
            f"filename so the preset path must match."
        )
        assert expected_filename in download_src, (
            f"download_models.py does not reference {expected_filename!r}; "
            f"the preset path will resolve to a missing file after a "
            f"fresh download."
        )


def test_new_presets_in_literal_type() -> None:
    """Both new presets accepted by the Literal validator. Invalid
    preset strings still reject (covered in test_invalid_preset_rejected)."""
    # No exception on either; failure would raise ValidationError.
    LLMConfig(preset="gemma-3-4b-abliterated")
    LLMConfig(preset="llama-3.2-3b-abliterated")


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
        "models/Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf"
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
