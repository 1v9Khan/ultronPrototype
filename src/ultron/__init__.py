"""Ultron — a local voice-first AI assistant."""

import os
import sys
from pathlib import Path

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
