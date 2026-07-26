"""Pre-fetch every model the prototype needs.

Run once after install:

    python scripts/download_models.py

The script is idempotent — re-running it just verifies presence and skips
anything already on disk. Network failures are reported per-asset; one
failure does not abort the others.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# On Windows + Python 3.11, sys.stdout defaults to cp1252 which can't encode ✓/✗.
# Force UTF-8 so the status glyphs print without UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# Make `config` importable when running this file directly.
# Importing `config.settings` also redirects the HF cache to a writable path
# if the user's HF_HOME points somewhere broken — see settings._ensure_writable_hf_cache.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Asset specs
# ---------------------------------------------------------------------------

# Default LLM. If you swap LLM_MODEL_PATH in settings.py, swap this too.
# unsloth republishes Qwen's GGUFs with a fuller quant ladder.
LLM_REPO = "unsloth/Qwen3.5-9B-GGUF"
LLM_FILE = "Qwen3.5-9B-Q4_K_M.gguf"

# 4B optimization plan Stage B — additional GGUFs for the 4B + 0.8B
# speculative-decoding setup. The 9B above stays in models/ for swap-back;
# these are downloaded alongside it. Quants are Q4_K_M to match the 9B
# baseline and keep the LLM_PRESETS table coherent. See
# docs/4b_optimization_plan.md.
LLM_4B_REPO = "unsloth/Qwen3.5-4B-GGUF"
LLM_4B_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
LLM_DRAFT_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
LLM_DRAFT_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"

# 2026-05-12 — Josiefied-Qwen3-8B-abliterated-v1 (Goekdeniz-Guelmez).
# Q5_K_M strikes the balance between quality and VRAM headroom on the
# 4070 Ti (~5.85 GB on disk; ~10 GB peak voice-path stack vs 11.5 GB cap).
# Quantised by mradermacher (community-trusted quantiser). Pairs with
# the runtime tool-call validator under src/kenning/safety/ — the model
# is abliterated (no content-level refusals) but the validator gates
# the actual capability surface. Retained for swap-back as of 2026-05-14.
LLM_JOSIEFIED_REPO = "mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF"
LLM_JOSIEFIED_FILE = "Josiefied-Qwen3-8B-abliterated-v1.Q5_K_M.gguf"

# 2026-05-14 — Josiefied-Qwen3-4B-abliterated-v2 (Goekdeniz-Guelmez).
# Base: Qwen/Qwen3-4B-Instruct-2507. Same abliterated + Josiefied
# fine-tune lineage as the 8B above at ~half the VRAM footprint.
# Quant choice (2026-05-14 second-pass): Q4_K_M (~2.6 GB on disk;
# ~3.0 GB VRAM loaded). Q5_K_M was the initial choice but the user's
# workstation has ~4.7 GB of background GPU usage from Chrome /
# Discord / EdgeWebView / NVIDIA Broadcast, leaving too little
# headroom. Q4_K_M trims ~500 MB at negligible quality impact.
# The Q5_K_M file is also fetched (below) as a swap-back option.
LLM_JOSIEFIED_4B_REPO = "mradermacher/Josiefied-Qwen3-4B-abliterated-v2-GGUF"
LLM_JOSIEFIED_4B_FILE = "Josiefied-Qwen3-4B-abliterated-v2.Q4_K_M.gguf"
LLM_JOSIEFIED_4B_Q5_FILE = "Josiefied-Qwen3-4B-abliterated-v2.Q5_K_M.gguf"

# 2026-05-19 Track 4 -- Gemma 3 4B abliterated (mradermacher quants of
# the Goekdeniz-Guelmez abliterated fine-tune over Google's
# gemma-3-4b-it). Designed as the candidate daily-use swap targeted
# at the verbosity miscalibration documented in the 2026-05-19 design
# pass (IFEval 90.2 vs Qwen3's pattern of over-explaining factual
# queries and under-delivering on procedural depth). Pairs with the
# Gemma 3 1B IT draft for speculative decoding -- same tokenizer so
# the 60-75% acceptance rate holds on conversational text.
LLM_GEMMA_3_4B_REPO = "mradermacher/gemma-3-4b-it-abliterated-GGUF"
LLM_GEMMA_3_4B_FILE = "gemma-3-4b-it-abliterated.Q4_K_M.gguf"
# bartowski's Gemma 3 1B GGUF repo carries the ``google_`` prefix in
# both the repo slug and the upload filenames (verified live
# 2026-05-19: the unprefixed ``bartowski/gemma-3-1b-it-GGUF`` is a
# 404). Keep both fields in sync with the preset's
# ``draft_model_path`` in :mod:`kenning.config` -- the
# regression-guard test in ``tests/test_llm_preset.py`` enforces this.
LLM_GEMMA_3_1B_REPO = "bartowski/google_gemma-3-1b-it-GGUF"
LLM_GEMMA_3_1B_FILE = "google_gemma-3-1b-it-Q4_K_M.gguf"

# 2026-05-19 Track 4 -- Llama 3.2 3B abliterated (mradermacher quants
# of Meta's Llama-3.2-3B-Instruct base with refusal vectors removed).
# Designed as the gaming-mode preset: smaller VRAM footprint, naturally
# brief conversational tone, weaker tool-call discipline (acceptable
# in gaming mode where OpenClaw orchestration is disabled). Paired
# with the Llama 3.2 1B Instruct draft for speculative decoding.
LLM_LLAMA_3_2_3B_REPO = "mradermacher/Llama-3.2-3B-Instruct-abliterated-GGUF"
LLM_LLAMA_3_2_3B_FILE = "Llama-3.2-3B-Instruct-abliterated.Q4_K_M.gguf"
LLM_LLAMA_3_2_1B_REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
LLM_LLAMA_3_2_1B_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Moondream2 -- 1.9B vision-language model for "explain what I'm looking at"
# voice flows. CPU-only on-demand inference (~5-8 s per query). Total ~3.5 GB
# of FP16 weights on first download; cached under HF cache thereafter. Custom
# inference code via trust_remote_code=True -- vikhyatk is the model author.
# Pulled in by transformers.AutoModelForCausalLM on first VLM query; the
# pre-fetch here populates the cache so the first user query is fast.
#
# 2026-05-14: pin to a stable revision. Initially set to 2025-06-21
# but that still hit the tokenizer.json compat error on the venv's
# tokenizers 0.19.1. ``2024-08-26`` is the older stable release that
# predates the tokenizer.json format change. Must match the revision
# pinned in src/kenning/desktop/vlm.py.
MOONDREAM_REPO = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2024-08-26"

# Kokoro TTS — 2026-05-20 round 8 default engine. StyleTTS2 +
# ISTFTNet, ~330 MB on disk, near-realtime on CPU. The ``kokoro``
# PyPI package's ``KPipeline`` downloads weights into the HF cache on
# first use (transformers / huggingface_hub cache). The pre-fetch here
# just constructs ``KPipeline(lang_code='a')`` so the cache is warm
# before the first user query -- otherwise the orchestrator would pay
# the download cost on its first synth call.
KOKORO_LANG_CODE = "a"  # American English. Match src/kenning/tts/kokoro_engine.py.

# Smart Turn V3 — semantic end-of-turn detector (BSD-2-Clause).
# 8 MB int8 ONNX; CPU inference ~12 ms. Runs AFTER Silero detects silence
# to confirm the user is actually done speaking (vs trailed off mid-
# thought). Lets us drop the baseline VAD silence requirement
# substantially while preserving safety on mid-sentence pauses.
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
# v3.1 is no longer on the Hub as a plain ``smart-turn-v3.1.onnx`` -- the
# Pipecat repo now ships ``v3.1-cpu`` / ``v3.1-gpu`` and ``v3.2-cpu`` /
# ``v3.2-gpu``. We use the latest CPU variant (the wrapper pins to
# ``CPUExecutionProvider`` for zero VRAM cost). I/O contract is identical
# to v3.1 (input ``[batch, 80, 800]`` float32, output sigmoid).
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"

# Piper voice files
PIPER_VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/ryan/medium/en_US-ryan-medium.onnx"
)
PIPER_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
)
RVC_SUPPORT_BASE_URL = (
    "https://huggingface.co/r3gm/sonitranslate_voice_models/resolve/main/"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ already present: {dest.name}")
        return
    print(f"  → downloading {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print(f"  ✓ saved {dest}")
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"  ✗ failed: {e}")


def _hf_download(repo_id: str, filename: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"  ✓ already present: {filename}")
        return
    print(f"  → downloading {filename} from {repo_id}")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest_dir),
        )
        # local_dir downloads include symlinks/blobs depending on hub version;
        # ensure the actual file lives at the expected path.
        if Path(path) != target and Path(path).exists():
            Path(path).replace(target)
        print(f"  ✓ saved {target}")
    except Exception as e:
        print(f"  ✗ failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _prefetch_kokoro() -> None:
    """Pre-fetch Kokoro StyleTTS2 weights into the HF cache.

    The ``kokoro`` PyPI package's ``KPipeline`` downloads
    ``hexgrad/Kokoro-82M`` lazily on first use. Constructing the
    pipeline once here warms the cache so the orchestrator's first
    synth call doesn't pay the ~330 MB download cost.

    Also creates ``models/kokoro/`` as a sanity-gate directory --
    :class:`kenning.tts.kokoro_engine.KokoroSpeech` checks for its
    existence before letting ``KPipeline`` load (so a missing-
    directory state surfaces as a clear ``KokoroEngineLoadError``).

    Fail-open: if the kokoro package isn't installed or the download
    can't proceed, we print the error and continue. The orchestrator
    will surface the same error on first synth call.
    """
    kokoro_dir = settings.MODELS_DIR / "kokoro"
    kokoro_dir.mkdir(parents=True, exist_ok=True)
    print(f"  → ensuring {kokoro_dir} exists (sanity-gate directory)")
    try:
        from kokoro import KPipeline

        # Constructing the pipeline triggers HF Hub downloads of the
        # acoustic + vocoder weights. ``lang_code='a'`` selects
        # American English; the engine module pins the same value.
        print(f"  → pulling kokoro weights (lang_code={KOKORO_LANG_CODE!r})")
        _ = KPipeline(lang_code=KOKORO_LANG_CODE, device="cpu")
        print("  ✓ kokoro pipeline cached")
    except ImportError as e:
        print(f"  ✗ kokoro package not installed: {e}")
        print("    Run: pip install kokoro")
    except Exception as e:                                        # noqa: BLE001
        print(f"  ✗ failed: {e}")


def _prefetch_moondream2() -> None:
    """Pre-fetch the moondream2 weights into the HF cache.

    Uses transformers.AutoTokenizer / AutoModelForCausalLM to trigger
    a normal HF download. The model lazy-loads on first VLM query in
    the running orchestrator; this just warms the cache so that first
    query doesn't pay the ~3.5 GB download cost.

    Pins ``revision=MOONDREAM_REVISION`` (currently ``2025-06-21``) to
    avoid the ``tokenizer.json``-mismatch error that comes from main
    being updated faster than the ``tokenizers`` library can keep up
    with. Must match the revision pinned in
    :mod:`src/kenning/desktop/vlm.py`.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(
            f"  → pulling tokenizer + weights from {MOONDREAM_REPO} "
            f"@ {MOONDREAM_REVISION}"
        )
        AutoTokenizer.from_pretrained(
            MOONDREAM_REPO,
            revision=MOONDREAM_REVISION,
            trust_remote_code=True,
        )
        AutoModelForCausalLM.from_pretrained(
            MOONDREAM_REPO,
            revision=MOONDREAM_REVISION,
            trust_remote_code=True,
        )
        print("  ✓ moondream2 cached")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ failed: {e}")


def main() -> int:
    print("\nKenning model setup")
    print("-" * 40)

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 2026-05-20 round 8 swap: current LLM default is qwen3.5-4b (stock
    # Qwen 3.5 4B + 0.8B speculative draft). The abliterated / Gemma /
    # Llama presets remain in this download script so a one-line
    # ``swap_llm_preset.py <name>`` re-fetches and re-activates them.
    print("\n[1/12] LLM (Qwen3.5-4B Q4_K_M) — current default")
    _hf_download(LLM_4B_REPO, LLM_4B_FILE, settings.MODELS_DIR)

    print("\n[2/12] LLM (Qwen3.5-0.8B Q4_K_M) — speculative-decoding draft for current default")
    _hf_download(LLM_DRAFT_REPO, LLM_DRAFT_FILE, settings.MODELS_DIR)

    print("\n[3/12] LLM (Josiefied-Qwen3-4B-abliterated-v2 Q4_K_M) — retained for swap-back / abliterated")
    _hf_download(LLM_JOSIEFIED_4B_REPO, LLM_JOSIEFIED_4B_FILE, settings.MODELS_DIR)

    print("\n[3b/12] LLM (Josiefied-Qwen3-4B-abliterated-v2 Q5_K_M) — retained for swap-back / quality A/B")
    _hf_download(LLM_JOSIEFIED_4B_REPO, LLM_JOSIEFIED_4B_Q5_FILE, settings.MODELS_DIR)

    print("\n[4/12] LLM (Josiefied-Qwen3-8B-abliterated-v1 Q5_K_M) — retained for swap-back")
    _hf_download(LLM_JOSIEFIED_REPO, LLM_JOSIEFIED_FILE, settings.MODELS_DIR)

    print("\n[5/12] LLM (Qwen3.5-9B Q4_K_M) — retained for swap-back / larger context")
    _hf_download(LLM_REPO, LLM_FILE, settings.MODELS_DIR)

    # 2026-05-19 Track 4 -- candidate swap presets. Optional; the
    # downloads only matter once the user sets ``llm.preset`` to one
    # of these via swap_llm_preset.py or the voice MODEL_SWITCH
    # intent. Setting OFFLINE_SKIP_OPTIONAL_LLMS=1 in the environment
    # skips these fetches (useful on a constrained connection).
    if not os.environ.get("OFFLINE_SKIP_OPTIONAL_LLMS"):
        print("\n[5a/12] LLM (Gemma 3 4B abliterated Q4_K_M) — candidate daily-use swap (Track 4)")
        _hf_download(LLM_GEMMA_3_4B_REPO, LLM_GEMMA_3_4B_FILE, settings.MODELS_DIR)
        print("\n[5b/12] LLM (Gemma 3 1B IT Q4_K_M) — speculative draft for the Gemma preset")
        _hf_download(LLM_GEMMA_3_1B_REPO, LLM_GEMMA_3_1B_FILE, settings.MODELS_DIR)
        print("\n[5c/12] LLM (Llama 3.2 3B abliterated Q4_K_M) — gaming-mode preset")
        _hf_download(LLM_LLAMA_3_2_3B_REPO, LLM_LLAMA_3_2_3B_FILE, settings.MODELS_DIR)
        print("\n[5d/12] LLM (Llama 3.2 1B Instruct Q4_K_M) — speculative draft for the Llama preset")
        _hf_download(LLM_LLAMA_3_2_1B_REPO, LLM_LLAMA_3_2_1B_FILE, settings.MODELS_DIR)
    else:
        print(
            "\n[5a-5d/12] skipping optional swap-preset GGUFs "
            "(OFFLINE_SKIP_OPTIONAL_LLMS=1)",
        )

    print("\n[6/12] Kokoro TTS (StyleTTS2 + ISTFTNet) — current default engine")
    _prefetch_kokoro()

    print("\n[7/12] Piper voice (en_US-ryan-medium) — legacy TTS fallback")
    _download(PIPER_VOICE_URL, settings.TTS_VOICE_PATH)
    _download(PIPER_CONFIG_URL, settings.TTS_VOICE_CONFIG_PATH)

    print("\n[8/12] faster-whisper (downloads on first transcription)")
    print("  → triggering pre-fetch…")
    try:
        from faster_whisper import WhisperModel

        WhisperModel(
            settings.WHISPER_MODEL,
            device="cpu",  # CPU just for download; runtime uses CUDA
            compute_type="int8",
        )
        print(f"  ✓ {settings.WHISPER_MODEL} cached")
    except Exception as e:
        print(f"  ✗ failed: {e}")

    print("\n[9/12] openWakeWord pretrained models (downloads on first use)")
    try:
        import openwakeword.utils as ow_utils

        ow_utils.download_models()
        print("  ✓ pretrained models cached")
    except Exception as e:
        print(f"  ✗ failed: {e}")

    print("\n[10/12] Smart Turn V3.2 (cpu) — semantic end-of-turn detector (~8.7 MB int8)")
    smart_turn_dir = settings.MODELS_DIR / "smart_turn"
    _hf_download(SMART_TURN_REPO, SMART_TURN_FILE, smart_turn_dir)

    print(f"\n[11/12] moondream2 @ {MOONDREAM_REVISION} — vision-language model (~3.5 GB FP16, CPU inference)")
    _prefetch_moondream2()

    print(
        "\n[12/13] Dense embedder (jinaai/jina-embeddings-v3) — "
        "frontier-enhancement Item 3 swap (1024 dim, MTEB ~65.5)"
    )
    try:
        from fastembed import TextEmbedding
        # First instantiation downloads + caches the ONNX weights into
        # the FastEmbed cache. ~570 MB.
        _ = TextEmbedding("jinaai/jina-embeddings-v3", threads=2)
        print("  ✓ jina-embeddings-v3 cached")
    except Exception as e:
        print(f"  ✗ failed: {e}")
        print("    Falling back to bge-small-en-v1.5 will keep working "
              "if you revert config.")

    print(
        "\n[12a/13] Cross-encoder reranker (bge-reranker-v2-m3) — "
        "RAG quality lift, default-OFF (frontier item 2, 2026-05-21)"
    )
    try:
        from sentence_transformers import CrossEncoder
        # The constructor downloads + caches the model in the HF cache.
        # We don't actually need to keep the instance around -- it just
        # warms the cache so the first runtime call doesn't pay the
        # ~1-3 s load + ~1.1 GB download.
        _ = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
        print("  ✓ bge-reranker-v2-m3 cached")
    except Exception as e:
        print(f"  ✗ failed: {e}")
        print("    OK to ignore if memory.reranking.enabled stays False")

    print("\n[13/13] RVC support models + voice-conversion model (legacy TTS fallback)")
    _download(RVC_SUPPORT_BASE_URL + "hubert_base.pt", settings.RVC_HUBERT_PATH)
    _download(RVC_SUPPORT_BASE_URL + "rmvpe.pt", settings.RVC_RMVPE_PATH)
    if settings.RVC_MODEL_PATH.is_file() and settings.RVC_INDEX_PATH.is_file():
        print(f"  ✓ found: {settings.RVC_MODEL_PATH.name}")
        print(f"  ✓ found: {settings.RVC_INDEX_PATH.name}")
    else:
        print(f"  ! RVC model not found at {settings.RVC_MODEL_DIR}")
        print(f"    Expected: {settings.RVC_MODEL_PATH.name}")
        print(f"    Expected: {settings.RVC_INDEX_PATH.name}")
        print("    Set RVC_ENABLED=False in config/settings.py to disable, "
              "or drop the .pth + .index files into that directory.")

    # 2026-05-21: actually CHECK whether the wake-word ONNX is present
    # rather than unconditionally printing the "train your own" message.
    # Previous behaviour printed the warning even when the file existed,
    # which was a misleading and annoying log line on every download
    # script run.
    if Path(settings.WAKE_WORD_MODEL_PATH).is_file():
        print(f"\n  ✓ Kenning wake-word found: {settings.WAKE_WORD_MODEL_PATH}\n")
    else:
        print("\nNote: the custom Kenning wake-word model is not auto-downloaded.")
        print("Train your own and place at:")
        print(f"  {settings.WAKE_WORD_MODEL_PATH}")
        print("Until then, the prototype falls back to "
              f"'{settings.WAKE_WORD_FALLBACK}'.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
