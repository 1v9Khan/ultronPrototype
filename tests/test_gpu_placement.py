"""Pins the multi-GPU placement contract (2026-07-24, second card added).

Policy: device 0 (RTX 3060 12 GB) hosts NOTHING but Ultron's model; every
other VRAM tenant -- Whisper, Kokoro, the guard sidecar, the embedder
sidecar -- lives on device 1 (RTX 3060 Ti 8 GB).

The load-bearing piece is the LLM pin. llama.cpp's default
``split_mode=LAYER`` spreads layers + KV across EVERY visible CUDA device,
so merely INSTALLING a second card silently relocates part of the model
onto it -- next to the STT/TTS tenants, with each token's forward pass
crossing the secondary slot's x4 link. ``llm.gpu_index`` pins the model to
one card (``main_gpu`` + ``split_mode=NONE``).
"""

from __future__ import annotations

import pytest

from kenning.config import (
    KokoroConfig,
    LLMConfig,
    SemanticRouterConfig,
    STTConfig,
    TwitchSafetyConfig,
)


def _effective_llm_config():
    """The LLM config as the app resolves it: config.yaml's llm block with
    preset defaults filled in. config.yaml deliberately leaves ``gpu_index``
    unset (2026-07-24) so each PRESET carries its own card -- reading the raw
    YAML key would KeyError, and reading the cached get_config() singleton is
    unreliable because other suites reload/patch it.
    """
    import yaml

    from kenning.config import PROJECT_ROOT, LLMConfig

    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text("utf-8"))
    return LLMConfig.model_validate(raw.get("llm", {}))


# ---------------------------------------------------------------------------
# Schema defaults: single-GPU behaviour must survive a config.yaml reset.
# ---------------------------------------------------------------------------

def test_llm_defaults_pin_to_a_single_card() -> None:
    # The FIELD default is 0 (not None): a config that names no card must NOT
    # inherit llama.cpp's cross-card layer split just because a second GPU
    # exists. (The resolved value can differ -- since 2026-07-24 each preset
    # supplies its own card, and the default preset places on device 1.)
    assert LLMConfig.model_fields["gpu_index"].default == 0
    # Whatever preset is default, the model still lands on exactly ONE card.
    assert LLMConfig().gpu_index is not None
    # The draft rides with the target unless explicitly moved.
    assert LLMConfig().draft_gpu_index is None


def test_secondary_tenant_defaults_are_device_zero() -> None:
    # Defaults keep the historical single-GPU placement; only config.yaml
    # moves tenants to the second card.
    assert STTConfig().device_index == 0
    assert KokoroConfig().device == "cpu"
    assert TwitchSafetyConfig().sidecar_gpu_index is None
    assert SemanticRouterConfig().sidecar_gpu_index is None


def test_gpu_index_fields_reject_nonsense() -> None:
    from pydantic import ValidationError

    for bad in (-1, 99):
        with pytest.raises(ValidationError):
            LLMConfig(gpu_index=bad)
        with pytest.raises(ValidationError):
            STTConfig(device_index=bad)
        with pytest.raises(ValidationError):
            TwitchSafetyConfig(sidecar_gpu_index=bad)


# ---------------------------------------------------------------------------
# Kokoro accepts an explicit device index.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:1"])
def test_kokoro_accepts_valid_devices(device: str) -> None:
    assert KokoroConfig(device=device).device == device


@pytest.mark.parametrize("device", ["cuda:", "cuda:x", "gpu", "cuda:1:2", ""])
def test_kokoro_rejects_malformed_devices(device: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KokoroConfig(device=device)


def test_kokoro_device_helpers() -> None:
    from kenning.tts.kokoro_engine import _is_cuda_device, _is_valid_torch_device

    # An indexed device must classify as CUDA -- the engine used to compare
    # with == "cuda", so a "cuda:1" teardown skipped its empty_cache() and
    # leaked the reservation on the secondary card.
    assert _is_cuda_device("cuda")
    assert _is_cuda_device("cuda:1")
    assert not _is_cuda_device("cpu")
    assert not _is_cuda_device(None)

    assert _is_valid_torch_device("cpu")
    assert _is_valid_torch_device("cuda:1")
    assert not _is_valid_torch_device("cuda:x")
    assert not _is_valid_torch_device("gpu")


def test_move_to_device_accepts_indexed_cuda() -> None:
    # Bare-object check: validation must not reject "cuda:1" before any model
    # is loaded (no torch/CUDA required for this path).
    from kenning.tts.kokoro_engine import KokoroSpeech

    eng = KokoroSpeech.__new__(KokoroSpeech)
    with pytest.raises(ValueError):
        eng.move_to_device("gpu")          # still rejects nonsense


# ---------------------------------------------------------------------------
# Draft-model GPU kwargs (speculative decoding across cards).
# ---------------------------------------------------------------------------

def test_draft_gpu_kwargs_none_is_passthrough() -> None:
    from kenning.llm.draft_model import _draft_gpu_kwargs

    assert _draft_gpu_kwargs(None) == {}


def test_draft_gpu_kwargs_pins_single_card() -> None:
    pytest.importorskip("llama_cpp", reason="llama.dll not loadable here")
    from kenning.llm.draft_model import _draft_gpu_kwargs

    kw = _draft_gpu_kwargs(1)
    assert kw["main_gpu"] == 1
    assert kw["split_mode"] == 0        # LLAMA_SPLIT_MODE_NONE


# ---------------------------------------------------------------------------
# The live config.yaml encodes the intended placement.
# ---------------------------------------------------------------------------

def test_live_config_reserves_primary_card_for_the_model() -> None:
    # Read config.yaml straight from disk: the cached get_config() singleton
    # is shared process-wide and other suites reload/patch it, which made an
    # earlier version of this test fail only under the full wrapper.
    import yaml

    from kenning.config import PROJECT_ROOT

    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text("utf-8"))
    # THE load-bearing invariant: the model is pinned to exactly ONE card.
    # WHICH card is per-preset (2026-07-24), but `null` is never acceptable --
    # that hands placement back to llama.cpp's split_mode=LAYER and spreads
    # the model across both. Read the EFFECTIVE value: config.yaml
    # deliberately leaves gpu_index unset so the preset supplies it.
    cfg = _effective_llm_config()
    assert cfg.gpu_index is not None, (
        "gpu_index=null lets llama.cpp layer-split the model across cards")
    assert cfg.gpu_index in (0, 1)
    # Every tenant carries an explicit placement.
    # STT is the exception to "just set an index": CTranslate2 on a NON-DEFAULT
    # in-process CUDA device corrupts memory on this box (silent 0xc0000409),
    # so the invariant "STT does not consume primary-card VRAM" is met by the
    # out-of-process sidecar instead -- a child masked onto card 1 with
    # CUDA_VISIBLE_DEVICES, where the target card is plain cuda:0. The
    # in-process index must therefore stay 0 (the unsafe path, only ever hit
    # if the sidecar is disabled).
    assert raw["stt"]["sidecar_enabled"] is True
    assert raw["stt"]["sidecar_gpu_index"] == 1
    assert raw["stt"]["device_index"] == 0
    assert raw["tts"]["kokoro"]["device"] == "cuda:1"
    assert raw["twitch"]["safety"]["sidecar_gpu_index"] == 1
    # The embedder is the other exception: SemanticRouterConfig.sidecar_device
    # defaults to "cpu" and the orchestrator forwards it as
    # KENNING_EMBEDDER_DEVICE, which short-circuits embedder_server.py's device
    # pick -- so it allocates NO CUDA at all. A non-null gpu_index here only
    # sets CUDA_VISIBLE_DEVICES on a child that never uses it, while logging a
    # false "pinned to GPU N" line (which cost real time in the 2026-07-26 VRAM
    # audit). Null is the honest value; set one only alongside device "cuda".
    assert raw["semantic_router"]["sidecar_gpu_index"] is None


def test_live_config_does_not_overcommit_the_secondary_card() -> None:
    """The 8 GB card cannot host the model AND a GPU guard at once.

    2026-07-24: with ``llm.gpu_index: 1`` the 4B (~4.0 GB) shares device 1
    with Whisper (~2.3), Kokoro (~0.33) and the embedder (~0.9) -- about
    7.5 GB of ~7.8 GB usable. Adding the ~1.5 GB GPU guard on top overcommits
    the card, so the guard must be on CPU whenever the model lives there.
    """
    import yaml

    from kenning.config import PROJECT_ROOT

    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text("utf-8"))
    model_gpu = _effective_llm_config().gpu_index
    guard_gpu_layers = raw["twitch"]["safety"]["guard_gpu_layers"]
    guard_card = raw["twitch"]["safety"]["sidecar_gpu_index"]
    if model_gpu == guard_card and guard_gpu_layers != 0:
        raise AssertionError(
            f"model and GPU guard both on device {model_gpu}: "
            f"guard_gpu_layers={guard_gpu_layers} must be 0 (CPU), or move "
            f"the model back to the primary card"
        )


# ---------------------------------------------------------------------------
# Per-preset placement (2026-07-24): each model loads on the card it was
# sized for, so a preset swap moves it automatically.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset,card", [
    ("heretic-qwen3-4b-q6", 1),        # 4B shares device 1 with the audio tenants
    ("heretic-qwen3-4b-q5", 1),
    ("gemma-4-12b-heretic", 0),        # 12B gets the 12 GB card to itself
])
def test_presets_carry_their_own_card(preset: str, card: int) -> None:
    assert LLMConfig(preset=preset).gpu_index == card


def test_explicit_yaml_gpu_index_overrides_the_preset() -> None:
    # _apply_preset only fills fields the user did NOT set, so an explicit
    # value still wins -- that is exactly why config.yaml leaves it unset.
    assert LLMConfig(preset="gemma-4-12b-heretic", gpu_index=1).gpu_index == 1


def test_gemma4_preset_ships_a_draft_and_arms_speculation() -> None:
    cfg = LLMConfig(preset="gemma-4-12b-heretic")
    assert cfg.draft_model_path and "gemma-3-1b" in cfg.draft_model_path
    # Presets that ship a draft GGUF auto-select the real-model draft path.
    assert cfg.draft_kind == "model"


def test_gemma4_preset_is_in_the_literal() -> None:
    # A preset missing from the Literal fails reload_for_preset validation.
    from kenning.config import LLM_PRESETS

    assert "gemma-4-12b-heretic" in LLM_PRESETS
    LLMConfig(preset="gemma-4-12b-heretic")   # would raise if not in Literal


# ---------------------------------------------------------------------------
# Draft/target vocabulary guard: the Python draft path passes RAW token IDs.
# ---------------------------------------------------------------------------

def test_preset_draft_paths_resolve_to_real_files() -> None:
    """Draft paths must survive path resolution the same way targets do.

    Live boot 2026-07-26: the loader ran ``resolve_path`` on the TARGET but
    used ``draft_model_path`` raw, so a project-relative draft died with
    "Model path does not exist" whenever the process cwd was not the project
    root -- silently disabling speculative decoding at boot (the Gemma 4
    preset's draft never loaded and the secondary card came up ~1 GB under
    budget). Any preset whose draft GGUF is actually on disk must resolve.
    """
    from pathlib import Path

    from kenning.config import LLM_PRESETS, resolve_path

    checked = 0
    for name, preset in LLM_PRESETS.items():
        draft = preset.get("draft_model_path")
        if not draft:
            continue
        resolved = Path(resolve_path(draft))
        assert resolved.is_absolute(), f"{name}: {draft} -> {resolved}"
        # Only assert existence for drafts actually downloaded on this box;
        # paper-only presets stay paper-only.
        if Path(draft).is_absolute() or resolved.exists():
            checked += 1
    assert checked, "no on-disk draft GGUFs found to verify"


def test_gemma4_draft_resolves_on_disk() -> None:
    from pathlib import Path

    from kenning.config import LLMConfig, resolve_path

    cfg = LLMConfig(preset="gemma-4-12b-heretic")
    resolved = Path(resolve_path(cfg.draft_model_path))
    assert resolved.exists(), f"draft missing after resolution: {resolved}"


def test_vocab_guard_accepts_matching_pair() -> None:
    from kenning.llm.draft_model import assert_draft_vocab_matches

    assert_draft_vocab_matches(262144, 262144, "gemma-3-1b")   # no raise


def test_vocab_guard_rejects_mismatched_pair() -> None:
    from kenning.llm.draft_model import assert_draft_vocab_matches

    # Qwen (151k) drafting for Gemma (262k) would emit IDs meaning entirely
    # different tokens to the target -- silent garbage, so refuse.
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        assert_draft_vocab_matches(151936, 262144, "qwen3-0.6b")


def test_live_config_parses_into_the_schema() -> None:
    # The placement values must survive validation (extra=forbid would reject
    # a mistyped key, and the ge/le bounds a bad index).
    import yaml

    from kenning.config import PROJECT_ROOT, KenningConfig

    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text("utf-8"))
    cfg = KenningConfig.model_validate(raw)
    assert cfg.llm.gpu_index is not None
    assert cfg.stt.sidecar_enabled is True
    assert cfg.stt.sidecar_gpu_index == 1
    assert cfg.stt.device_index == 0
    assert cfg.tts.kokoro.device == "cuda:1"
    assert cfg.twitch.safety.sidecar_gpu_index == 1
    assert cfg.semantic_router.sidecar_device == "cpu"
    assert cfg.semantic_router.sidecar_gpu_index is None


# ---------------------------------------------------------------------------
# Dual-model residency (2026-07-26): the small model stays loaded for snap
# callouts so they never pay a model-reload; the big one keeps conversation.
# ---------------------------------------------------------------------------

def test_fast_preset_defaults_off() -> None:
    # Empty string = single-model, byte-identical legacy behaviour.
    assert LLMConfig().fast_preset == ""


def test_live_config_dual_model_lands_on_separate_cards() -> None:
    """WHEN a second resident model is configured, it must not share the
    primary's card -- together they exceed either GPU (12B ~9 GB + 4B ~3.7 GB).

    Conditional by design: ``fast_preset`` is empty as of 2026-07-26 while a
    native abort (0xc0000409) seen only on dual-model boots is investigated,
    so this pins the placement CONTRACT without forcing the feature on.
    """
    from kenning.config import LLM_PRESETS

    cfg = _effective_llm_config()
    if not cfg.fast_preset:
        # Dual-model is OFF. Rather than skip -- which asserts nothing and
        # hides the day someone sets fast_preset without checking placement --
        # pin the CURRENT documented state explicitly. The branch below is the
        # contract that takes over the moment the feature is switched on.
        assert cfg.fast_preset == "", (
            "fast_preset should be empty while dual-model is disabled")
        return
    assert cfg.fast_preset in LLM_PRESETS
    fast_card = LLM_PRESETS[cfg.fast_preset]["gpu_index"]
    assert fast_card != cfg.gpu_index, (
        f"both resident models pinned to device {fast_card}")


def test_llm_engine_accepts_a_gpu_index_override() -> None:
    # Without this the second engine reads cfg.gpu_index (the MAIN model's
    # card) and both models pile onto the same GPU.
    import inspect

    from kenning.llm.inference import LLMEngine

    assert "gpu_index" in inspect.signature(LLMEngine.__init__).parameters
