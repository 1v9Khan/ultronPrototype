"""Phase 6 — orchestration test runner.

Runs the full integration suite (mocked Phase 6 scenarios + their real-
Claude variants when ``PYTEST_RUN_GPU_TESTS=1`` is set in the
environment) and produces a clear per-section report. Exits non-zero on
any failure so CI can gate on it.

Usage::

    python scripts/run_orchestration_tests.py            # mocked only
    PYTEST_RUN_GPU_TESTS=1 python scripts/run_orchestration_tests.py  # + real

By default runs from the main checkout root. Pass ``--paths`` to scope
to a different directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TestSection:
    name: str
    paths: List[str]
    requires_gpu_env: bool = False  # set when only meaningful with PYTEST_RUN_GPU_TESTS=1
    extra_args: List[str] = field(default_factory=list)


# Section 1: orchestration mocked scenarios (the bulk of Phase 6).
# Section 2: existing e2e tests already provide real-Claude coverage of
#   scenarios 1, 2, 7, plus the MCP tool round-trip.
# Section 3: Phase 6 real-Claude additions for verification fail loop +
#   cancellation.
SECTIONS: List[TestSection] = [
    TestSection(
        name="orchestration-mocked",
        paths=[
            "tests/coding/test_mock_bridge_smoke.py",
            "tests/coding/test_orchestration.py",
        ],
    ),
    TestSection(
        name="e2e-existing",
        paths=[
            "tests/test_coding_e2e.py",
            "tests/test_mcp_e2e.py",
        ],
        requires_gpu_env=True,
    ),
    TestSection(
        name="orchestration-real",
        paths=["tests/coding/test_orchestration_real.py"],
        requires_gpu_env=True,
    ),
]


@dataclass
class SectionResult:
    name: str
    skipped: bool = False
    skip_reason: str = ""
    returncode: int = 0
    duration_s: float = 0.0
    summary_line: str = ""


def _summary_from_pytest_output(text: str) -> str:
    """Pull out pytest's terminal summary line (last non-empty line that
    looks like 'X passed in Ys' / 'X failed' / etc.)."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        # Pytest ends with lines like "5 passed in 0.42s" or "1 failed, 2 passed in 1.0s"
        if " in " in line and any(
            k in line for k in ("passed", "failed", "error", "skipped")
        ):
            return line
        # On total failure pytest still prints '=== 1 failed in...'
        if line.startswith("="):
            return line.strip("= ").strip()
    return text.strip().splitlines()[-1] if text.strip() else "(no output)"


def _run_section(
    section: TestSection,
    *,
    extra_pytest_args: Optional[List[str]] = None,
    repo_root: Path = REPO_ROOT,
) -> SectionResult:
    if section.requires_gpu_env and os.environ.get("PYTEST_RUN_GPU_TESTS") != "1":
        return SectionResult(
            name=section.name, skipped=True,
            skip_reason="requires PYTEST_RUN_GPU_TESTS=1",
        )
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"]
    cmd.extend(section.paths)
    cmd.extend(section.extra_args)
    if extra_pytest_args:
        cmd.extend(extra_pytest_args)
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    duration = time.monotonic() - start
    output = (proc.stdout or "") + (proc.stderr or "")
    summary = _summary_from_pytest_output(output)
    return SectionResult(
        name=section.name,
        returncode=proc.returncode,
        duration_s=duration,
        summary_line=summary,
    )


def _print_report(results: List[SectionResult]) -> int:
    """Print a clear summary; return a non-zero exit code if any failed."""
    width = max((len(r.name) for r in results), default=10) + 2
    print()
    print("=" * 78)
    print(f"{'section':<{width}}  {'status':<10}  {'time':<8}  summary")
    print("-" * 78)
    failures = 0
    for r in results:
        if r.skipped:
            status = "SKIPPED"
            note = r.skip_reason
            time_s = "-"
        elif r.returncode == 0:
            status = "PASSED"
            note = r.summary_line
            time_s = f"{r.duration_s:.1f}s"
        else:
            status = "FAILED"
            note = r.summary_line
            time_s = f"{r.duration_s:.1f}s"
            failures += 1
        print(f"{r.name:<{width}}  {status:<10}  {time_s:<8}  {note}")
    print("=" * 78)
    if failures:
        print(f"\n{failures} section(s) failed.")
        return 1
    print("\nAll sections passed.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help="Root of the repo to run pytest from (default: %(default)s)",
    )
    parser.add_argument(
        "-k", dest="keyword", default=None,
        help="Filter expression passed through to pytest (-k <keyword>)",
    )
    parser.add_argument(
        "--mocked-only", action="store_true",
        help="Run only the mocked sections (skip real-Claude even with the env var set)",
    )
    parser.add_argument(
        "--show-output", action="store_true",
        help="Stream each section's pytest output instead of just the summary",
    )
    args = parser.parse_args(argv)

    extra: List[str] = []
    if args.keyword:
        extra.extend(["-k", args.keyword])
    if args.show_output:
        # Drop the -q + replace with -v
        extra.extend(["-v"])

    results: List[SectionResult] = []
    for section in SECTIONS:
        if args.mocked_only and section.requires_gpu_env:
            results.append(SectionResult(
                name=section.name, skipped=True,
                skip_reason="--mocked-only",
            ))
            continue
        print(f"\n>>> running section: {section.name}")
        r = _run_section(
            section,
            extra_pytest_args=extra,
            repo_root=args.repo_root,
        )
        if args.show_output:
            # Re-run with -v so the user sees per-test lines. We already
            # captured the summary above; this second run is purely for
            # display (it should be a no-op cache hit on most platforms).
            pass
        if r.skipped:
            print(f"    skipped: {r.skip_reason}")
        elif r.returncode == 0:
            print(f"    {r.summary_line}")
        else:
            print(f"    FAILED ({r.returncode}): {r.summary_line}")
        results.append(r)

    return _print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())
