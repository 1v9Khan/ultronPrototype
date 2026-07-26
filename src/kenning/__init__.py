"""Kenning — a local voice-first AI assistant."""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CUDA DEVICE ORDER (2026-07-24, second GPU installed). MUST run before the
# first CUDA call in this process -- hence the top of the package __init__.
#
# CUDA's default enumeration is FASTEST_FIRST, which does NOT match the
# PCI-bus order nvidia-smi prints. On this box that silently INVERTS the two
# cards: nvidia-smi calls the RTX 3060 12 GB "0" and the RTX 3060 Ti 8 GB
# "1", while CUDA (ranking the Ti faster) hands out the opposite indices.
# Every placement knob in the project -- llm.gpu_index, stt.device_index,
# tts.kokoro.device "cuda:N", and the CUDA_VISIBLE_DEVICES masks for the
# sidecars -- is interpreted by CUDA, so under the default order they all
# addressed the WRONG card (measured: a model pinned to "device 0" loaded
# onto the 8 GB Ti and OOM'd, while the 12 GB card sat idle).
#
# Pinning PCI_BUS_ID makes CUDA agree with nvidia-smi, so a config index
# means the card the operator sees in nvidia-smi. Also removes a silent
# failure mode: FASTEST_FIRST ordering can change when hardware changes.
# ``setdefault`` so an explicit environment override still wins.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


__version__ = "0.1.0"


def _register_cuda_dll_paths() -> None:
    """Make CUDA runtime DLLs discoverable by llama-cpp / ctranslate2 / etc.

    The CUDA-built llama-cpp wheel from abetlen needs ``cudart64_12.dll`` and
    ``cublas64_12.dll`` on the Windows DLL search path. PyTorch bundles those
    in ``torch/lib/`` but doesn't add them to the global path. We add them
    here, plus any ``nvidia-*-cu12`` site-packages dirs, so every CUDA-bound
    component in the project finds its libs without requiring the user to
    install the standalone CUDA Toolkit.
    """
    if sys.platform != "win32":
        return
    candidates = []
    try:
        import torch  # noqa: F401  — only used for path discovery
        candidates.append(Path(sys.modules["torch"].__file__).parent / "lib")
    except Exception:
        pass
    site_packages = Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages"
    if site_packages.is_dir():
        for child in site_packages.iterdir():
            if child.name.startswith("nvidia_") and child.is_dir():
                bin_dir = child / "bin"
                if bin_dir.is_dir():
                    candidates.append(bin_dir)
    for p in candidates:
        try:
            if p.is_dir():
                os.add_dll_directory(str(p))
        except (FileNotFoundError, OSError):
            pass


_register_cuda_dll_paths()
