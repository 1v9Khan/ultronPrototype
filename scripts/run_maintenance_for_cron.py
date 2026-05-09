"""Cron-friendly maintenance wrapper (OpenClaw integration Phase 7).

Thin shim around :mod:`scripts.maintenance` for scheduled invocation.
Outputs a structured JSON report to stdout that callers (Windows
Task Scheduler, OpenClaw cron jobs, manual scripts) can pipe into a
notification / log / agent prompt.

Usage:

    # Run all tasks, machine-readable JSON report to stdout.
    python scripts/run_maintenance_for_cron.py --json

    # Run a subset.
    python scripts/run_maintenance_for_cron.py \\
        --task extract_facts \\
        --task cleanup_web_cache

    # Pretty single-line summary for human eyes.
    python scripts/run_maintenance_for_cron.py --pretty

What this is NOT: this is not a long-running daemon, not an MCP tool,
and not a substitute for the in-process Qdrant+LLM that the live
Ultron uses. It loads the LLM cold every run (~30 s), so prefer
running it from cron when the voice path is idle (3am for nightly
maintenance is the canonical slot).

Exit codes:
- 0: all tasks completed (some may have processed zero items).
- 1: at least one task raised an exception.
- 2: invocation error (bad args, missing dependencies).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))


def _import_maintenance():
    """Lazy-import to keep --help fast."""
    import scripts.maintenance as maintenance_mod                # type: ignore
    return maintenance_mod


def run(
    *,
    tasks: List[str] | None,
    json_output: bool,
    pretty: bool,
) -> int:
    """Run the chosen maintenance tasks; return exit code."""
    maintenance = _import_maintenance()
    chosen = tasks or list(maintenance._TASKS)                   # type: ignore[attr-defined]

    started = time.time()
    captured = StringIO()
    summary: Dict[str, int] = {}
    error_count = 0

    try:
        client = maintenance._open_qdrant()                     # type: ignore[attr-defined]
        conn = maintenance._ensure_meta_db()                    # type: ignore[attr-defined]
    except Exception as e:                                       # noqa: BLE001
        _emit({
            "status": "error",
            "stage": "init",
            "error": str(e),
            "tasks_attempted": [],
            "summary": {},
        }, json_output=json_output, pretty=pretty)
        return 2

    needs_llm = any(
        t in chosen
        for t in ("backfill_metadata", "extract_facts",
                  "cluster_conversations", "daily_summary")
    )
    needs_embedder = "extract_facts" in chosen

    llm = None
    embedder = None
    try:
        if needs_llm:
            with redirect_stdout(captured):
                llm = maintenance._load_llm()                     # type: ignore[attr-defined]
        if needs_embedder:
            with redirect_stdout(captured):
                embedder = maintenance._load_embedder()           # type: ignore[attr-defined]
    except Exception as e:                                       # noqa: BLE001
        _emit({
            "status": "error",
            "stage": "load",
            "error": str(e),
            "tasks_attempted": [],
            "summary": {},
            "stdout_capture": captured.getvalue()[:1000],
        }, json_output=json_output, pretty=pretty)
        try:
            client.close()
            conn.close()
        except Exception:
            pass
        return 2

    for task in chosen:
        try:
            with redirect_stdout(captured):
                if task == "backfill_metadata":
                    summary[task] = maintenance.run_backfill_metadata(llm, client)
                elif task == "extract_facts":
                    summary[task] = maintenance.run_extract_facts(
                        llm, client, conn, embedder,
                    )
                elif task == "cluster_conversations":
                    summary[task] = maintenance.run_cluster_conversations(llm, client)
                elif task == "daily_summary":
                    summary[task] = maintenance.run_daily_summary(llm, client)
                elif task == "decay_stale_facts":
                    summary[task] = maintenance.run_decay_stale_facts(client)
                elif task == "cleanup_web_cache":
                    summary[task] = maintenance.run_cleanup_web_cache(client)
                else:
                    summary[task] = -1
                    error_count += 1
        except Exception as e:                                   # noqa: BLE001
            summary[task] = -1
            error_count += 1
            captured.write(f"\nTASK FAILED: {task}: {e}\n")

    try:
        conn.close()
        client.close()
    except Exception:
        pass

    duration = time.time() - started
    payload: Dict[str, Any] = {
        "status": "ok" if error_count == 0 else "partial",
        "duration_seconds": round(duration, 1),
        "tasks_attempted": chosen,
        "summary": summary,
    }
    if json_output and captured.getvalue():
        payload["stdout_capture"] = captured.getvalue()[-2000:]
    _emit(payload, json_output=json_output, pretty=pretty)
    return 0 if error_count == 0 else 1


def _emit(
    payload: Dict[str, Any],
    *,
    json_output: bool,
    pretty: bool,
) -> None:
    """Write payload to stdout in the chosen format."""
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if pretty:
        # Single-line summary suitable for a chat / Telegram message.
        status = payload.get("status", "?")
        duration = payload.get("duration_seconds", 0)
        summary = payload.get("summary") or {}
        bits = [f"{k}={v}" for k, v in summary.items()]
        line = (
            f"Maintenance {status} in {duration:.0f}s: " + ", ".join(bits)
            if bits else f"Maintenance {status} in {duration:.0f}s (no tasks)"
        )
        print(line)
        if payload.get("status") == "error":
            print(f"  error: {payload.get('error')}")
        return
    # Default: human-readable multi-line.
    print(f"Status: {payload.get('status')}")
    print(f"Duration: {payload.get('duration_seconds')}s")
    print(f"Tasks attempted: {payload.get('tasks_attempted')}")
    print("Summary:")
    for k, v in (payload.get("summary") or {}).items():
        print(f"  {k:25s} {v}")
    if payload.get("error"):
        print(f"Error: {payload['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    maintenance = _import_maintenance()
    parser.add_argument(
        "--task",
        choices=list(maintenance._TASKS),                        # type: ignore[attr-defined]
        action="append",
        help="run a single task (can be repeated; default is all tasks)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit a JSON object on stdout",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="emit a single-line human summary on stdout (Telegram-friendly)",
    )
    args = parser.parse_args()
    if args.json and args.pretty:
        print("--json and --pretty are mutually exclusive", file=sys.stderr)
        return 2
    return run(
        tasks=args.task,
        json_output=args.json,
        pretty=args.pretty,
    )


if __name__ == "__main__":
    sys.exit(main())
