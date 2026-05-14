"""Quick VRAM check.

Prints current GPU memory usage on the configured GPU index, plus the
hard cap (11.5 GB; physical RTX 4070 Ti budget) and the soft target.
Flags warning / critical thresholds.

The soft target is preset-aware: it tracks the active LLM preset's
expected footprint so swapping models also moves the no-regression
line. The hard cap is unchanged — that's physics.

| Preset       | Soft target | Why |
|--------------|-------------|-----|
| qwen3.5-9b   | 9216 MB     | original Foundation-phase budget |
| qwen3.5-4b   | 6700 MB     | 9.2 GB - 2.5 GB savings from 4B + 0.8B |
| custom       | 9216 MB     | conservative fallback |

If the active config can't be loaded (e.g. running pre-install or the
file is broken), the script falls back to the conservative 9216 MB
target so it remains a useful debugging tool.

Usage:
    python scripts/check_vram.py             # one-shot snapshot
    python scripts/check_vram.py --watch     # refresh every 2 s
    python scripts/check_vram.py --watch 0.5 # refresh every 0.5 s
    python scripts/check_vram.py --target-preset qwen3.5-4b  # override
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


HARD_CAP_MB = 11500
WARN_FRACTION = 0.85   # warn at 85 % of hard cap

# 4B optimization plan — preset-aware soft target table.
# 2026-05-14: added the Josiefied 4B / 8B abliterated targets. The
# 4B abliterated lands near the same VRAM as plain qwen3.5-4b; the
# 8B abliterated lands near the 9B target.
TARGET_MB_BY_PRESET: dict[str, int] = {
    "qwen3.5-9b": 9216,
    "qwen3.5-4b": 6700,
    "josiefied-qwen3-8b": 9216,
    "josiefied-qwen3-4b": 6700,
}
DEFAULT_TARGET_MB = 9216  # conservative fallback when preset unknown


def _resolve_target_mb(override: Optional[str] = None) -> tuple[int, str]:
    """Return ``(target_mb, preset_label)`` for the active preset.

    Order:
      1. ``--target-preset`` CLI override.
      2. ``ULTRON_LLM_PRESET`` env var.
      3. ``config.yaml:llm.preset`` (loaded best-effort).
      4. ``DEFAULT_TARGET_MB`` fallback.
    """
    if override:
        return TARGET_MB_BY_PRESET.get(override, DEFAULT_TARGET_MB), override
    import os
    env_preset = os.environ.get("ULTRON_LLM_PRESET")
    if env_preset:
        return TARGET_MB_BY_PRESET.get(env_preset, DEFAULT_TARGET_MB), env_preset
    # Try the loaded config without raising if anything's broken.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from ultron.config import get_config  # noqa: WPS433
        preset = get_config().llm.preset
        return TARGET_MB_BY_PRESET.get(preset, DEFAULT_TARGET_MB), preset
    except Exception:
        return DEFAULT_TARGET_MB, "unknown"


# Computed lazily in main(); module-level constant kept for back-compat
# with anything importing TARGET_MB directly. The lazy resolver above is
# preferred for new code.
TARGET_MB = DEFAULT_TARGET_MB


def vram_used_mb(gpu_id: int = 0) -> Optional[int]:
    """Return current VRAM use on ``gpu_id``, or None if nvidia-smi
    isn't available."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            f"--id={gpu_id}",
        ], text=True, timeout=5).strip()
        return int(out)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def vram_total_mb(gpu_id: int = 0) -> Optional[int]:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
            f"--id={gpu_id}",
        ], text=True, timeout=5).strip()
        return int(out)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def gpu_name(gpu_id: int = 0) -> Optional[str]:
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
            f"--id={gpu_id}",
        ], text=True, timeout=5).strip()
        return out
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _format_line(used: int, total: Optional[int], target_mb: int, preset_label: str) -> str:
    parts = [f"{used} MB used"]
    if total:
        pct = 100.0 * used / total
        parts.append(f"of {total} MB total ({pct:.0f}%)")
    parts.append(f"target {target_mb} MB ({preset_label})")
    parts.append(f"cap {HARD_CAP_MB} MB")
    status = "OK"
    if used > HARD_CAP_MB:
        status = "CRITICAL — over hard cap"
    elif used > HARD_CAP_MB * WARN_FRACTION:
        status = f"WARN — over {int(WARN_FRACTION*100)}% of hard cap"
    elif used > target_mb:
        status = "above target (under cap)"
    parts.append(f"[{status}]")
    return " | ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VRAM snapshot")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    parser.add_argument(
        "--watch", nargs="?", type=float, const=2.0, default=None,
        help="Refresh every N seconds (default 2.0)",
    )
    parser.add_argument(
        "--target-preset", type=str, default=None,
        help=(
            "Force the soft target to a specific preset's value "
            "(qwen3.5-9b / qwen3.5-4b / custom). Default: read from "
            "ULTRON_LLM_PRESET env var or config.yaml."
        ),
    )
    args = parser.parse_args(argv)

    name = gpu_name(args.gpu)
    if name is None:
        print("nvidia-smi not available or no GPU at that index.", file=sys.stderr)
        return 1
    total = vram_total_mb(args.gpu)
    target_mb, preset_label = _resolve_target_mb(args.target_preset)
    print(f"GPU {args.gpu}: {name}" + (f"  (total {total} MB)" if total else ""))

    if args.watch is None:
        used = vram_used_mb(args.gpu)
        if used is None:
            return 1
        print(_format_line(used, total, target_mb, preset_label))
        return 0

    print(f"Watching every {args.watch}s. Ctrl+C to stop.")
    try:
        while True:
            used = vram_used_mb(args.gpu)
            if used is None:
                break
            print(_format_line(used, total, target_mb, preset_label))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
