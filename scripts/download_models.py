"""Pre-fetch every model the prototype needs.

Run once after install:

    python scripts/download_models.py

The script is idempotent — re-running it just verifies presence and skips
anything already on disk. Network failures are reported per-asset; one
failure does not abort the others.
"""

from __future__ import annotations

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

# Piper voice files
PIPER_VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/ryan/medium/en_US-ryan-medium.onnx"
)
PIPER_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
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


def main() -> int:
    print("\nUltron model setup")
    print("-" * 40)

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] LLM (Qwen3.5-9B Q4_K_M)")
    _hf_download(LLM_REPO, LLM_FILE, settings.MODELS_DIR)

    print("\n[2/4] Piper voice (en_US-ryan-medium)")
    _download(PIPER_VOICE_URL, settings.TTS_VOICE_PATH)
    _download(PIPER_CONFIG_URL, settings.TTS_VOICE_CONFIG_PATH)

    print("\n[3/4] faster-whisper (downloads on first transcription)")
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

    print("\n[4/4] openWakeWord pretrained models (downloads on first use)")
    try:
        import openwakeword.utils as ow_utils

        ow_utils.download_models()
        print("  ✓ pretrained models cached")
    except Exception as e:
        print(f"  ✗ failed: {e}")

    print("\n[5/5] RVC voice-conversion model (user-provided)")
    if settings.RVC_MODEL_PATH.is_file() and settings.RVC_INDEX_PATH.is_file():
        print(f"  ✓ found: {settings.RVC_MODEL_PATH.name}")
        print(f"  ✓ found: {settings.RVC_INDEX_PATH.name}")
    else:
        print(f"  ! RVC model not found at {settings.RVC_MODEL_DIR}")
        print(f"    Expected: {settings.RVC_MODEL_PATH.name}")
        print(f"    Expected: {settings.RVC_INDEX_PATH.name}")
        print("    Set RVC_ENABLED=False in config/settings.py to disable, "
              "or drop the .pth + .index files into that directory.")

    print("\nNote: the custom Ultron wake-word model is not auto-downloaded.")
    print("Train your own and place at:")
    print(f"  {settings.WAKE_WORD_MODEL_PATH}")
    print("Until then, the prototype falls back to "
          f"'{settings.WAKE_WORD_FALLBACK}'.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
