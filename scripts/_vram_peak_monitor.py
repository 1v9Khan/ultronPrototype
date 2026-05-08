"""Tiny VRAM peak monitor.

Polls GPU 0 used-MB at 200ms cadence for ``--seconds`` and writes the
peak to stdout (one integer). Used to capture VRAM during a foreground
operation (e.g., a chat completion call) without spinning up the full
``check_vram.py`` print loop.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def query_used_mb() -> int:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--interval-ms", type=int, default=200)
    args = parser.parse_args()

    deadline = time.monotonic() + args.seconds
    interval_s = args.interval_ms / 1000.0
    peak = 0
    samples = 0
    while time.monotonic() < deadline:
        try:
            mb = query_used_mb()
        except Exception:
            mb = 0
        if mb > peak:
            peak = mb
        samples += 1
        time.sleep(interval_s)
    sys.stdout.write(f"{peak}\n")
    sys.stdout.write(f"samples={samples}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
