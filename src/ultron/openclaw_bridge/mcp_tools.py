"""Stdio MCP server exposing Ultron's data to OpenClaw agents.

OpenClaw 2026.5.7's ``mcp set`` accepts a stdio command + args. Each
agent run that needs an Ultron tool spawns this script as a fresh
process; the script reads from on-disk artifacts (heartbeat alert
log, session audit logs, maintenance subprocess) and returns
results over the MCP stdio protocol.

This is intentionally separate from :class:`UltronMCPServer` (in
``ultron.coding.mcp_server``) which serves the Claude Code
subprocess via SSE and lives in the orchestrator process. Sharing
that server across both consumers would require a stdio→SSE proxy
shim; running a small dedicated stdio server is simpler and keeps
the surfaces decoupled.

Tools registered:

- ``get_heartbeat_alerts`` — read recent heartbeat alerts.
- ``acknowledge_alert`` — mark a heartbeat alert seen.
- ``run_maintenance`` — kick the maintenance pipeline via
  ``scripts/run_maintenance_for_cron.py``.
- ``list_active_coding_sessions`` — list active sessions from
  per-session audit logs.
- ``get_recent_voice_alerts`` — convenience wrapper that returns
  the data the OpenClaw heartbeat agent needs to avoid
  re-surfacing acknowledged items.

All tools fail-open: missing files, malformed JSONL, subprocess
errors all translate to structured error payloads rather than
crashing the MCP server. Read-only tools never modify on-disk
state outside their explicit write paths (``acknowledge_alert``).

Process startup is ephemeral — OpenClaw spawns a new instance per
MCP call. Imports stay minimal to keep cold start under ~1 s
(no torch / no LLM model loading).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration discovery
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Resolve the project root from this module's location.

    The stdio entry script (``scripts/run_ultron_mcp_for_openclaw.py``)
    sets PROJECT_ROOT before importing this module, so the layout
    discovery is straightforward."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _resolve_path(relative: str, *, root: Optional[Path] = None) -> Path:
    """Resolve a path relative to the project root, returning an
    absolute Path. Used so the stdio process picks up the same
    paths the orchestrator uses (alert log, session audit dir, etc.)."""
    p = Path(relative)
    if p.is_absolute():
        return p.resolve()
    base = root if root is not None else _project_root()
    return (base / p).resolve()


def _load_alert_log_path() -> Path:
    """Discover the heartbeat alert log path from config.yaml.

    Falls back to the canonical default if config can't be read.
    Doing the discovery without importing :mod:`ultron.config`
    keeps cold-start light — config loading would pull in pydantic
    and traverse the full schema.
    """
    cfg = _project_root() / "config.yaml"
    default = _resolve_path("logs/heartbeat_alerts.jsonl")
    if not cfg.exists():
        return default
    try:
        import yaml                                           # local import
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return default
    raw_path = (
        ((data.get("heartbeat") or {}).get("alert_log_path"))
        or "logs/heartbeat_alerts.jsonl"
    )
    return _resolve_path(raw_path)


def _load_session_audit_dir() -> Path:
    """Discover the per-session audit dir.

    Currently hard-coded to ``logs/sessions`` — the path comes from
    ``settings.CODING_SESSION_AUDIT_DIR`` which lives in the legacy
    config shim. Reading that shim from a stdio process would pull
    the full ultron package; instead we use the well-known default."""
    return _resolve_path("logs/sessions")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _alert_log():
    """Lazy-import the alert log (avoids loading until needed)."""
    from ultron.openclaw_bridge.heartbeat_alerts import HeartbeatAlertLog  # noqa: E501
    return HeartbeatAlertLog(_load_alert_log_path(), retention_days=30)


def get_heartbeat_alerts_impl(
    *,
    since_seconds_ago: int = 86400,
    only_unacknowledged: bool = True,
    limit: int = 50,
) -> Dict[str, Any]:
    """Return recent heartbeat alerts as a JSON-serializable dict.

    Defaults to the last 24 hours of unacknowledged alerts, capped
    at 50. The OpenClaw heartbeat agent calls this on each tick to
    avoid re-surfacing items the user has already seen.
    """
    if since_seconds_ago < 0:
        return {"error": "since_seconds_ago must be non-negative"}
    if limit < 0 or limit > 1000:
        return {"error": "limit must be between 0 and 1000"}
    cutoff = time.time() - since_seconds_ago if since_seconds_ago else None
    log = _alert_log()
    alerts = log.get_alerts(
        since=cutoff,
        only_unacknowledged=bool(only_unacknowledged),
        limit=limit,
    )
    return {
        "count": len(alerts),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "text": a.text,
                "source": a.source,
                "severity": a.severity,
                "timestamp": a.timestamp,
                "acknowledged": a.acknowledged,
                "acknowledged_at": a.acknowledged_at,
            }
            for a in alerts
        ],
    }


def acknowledge_alert_impl(alert_id: str) -> Dict[str, Any]:
    """Mark a heartbeat alert as seen.

    Returns ``{"acknowledged": True, "alert_id": ...}`` on success,
    ``{"acknowledged": False, "reason": ...}`` when the id is unknown
    or already acknowledged.
    """
    if not alert_id or not isinstance(alert_id, str):
        return {"acknowledged": False, "reason": "alert_id must be non-empty"}
    log = _alert_log()
    ok = log.acknowledge(alert_id.strip())
    if ok:
        return {"acknowledged": True, "alert_id": alert_id}
    return {
        "acknowledged": False,
        "alert_id": alert_id,
        "reason": "unknown id or already acknowledged",
    }


def run_maintenance_impl(scope: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the maintenance pipeline via the cron-friendly wrapper.

    ``scope`` is a list of task names (``backfill_metadata``,
    ``extract_facts``, ``cluster_conversations``, ``daily_summary``,
    ``decay_stale_facts``, ``cleanup_web_cache``). ``None`` runs
    all tasks.

    Subprocesses ``scripts/run_maintenance_for_cron.py --json`` so
    the heavy imports (torch, LLM, embedder) load in a child process
    that can be torn down afterwards. The agent receives the JSON
    payload directly.
    """
    valid_tasks = {
        "backfill_metadata", "extract_facts", "cluster_conversations",
        "daily_summary", "decay_stale_facts", "cleanup_web_cache",
    }
    if scope is not None:
        bad = [t for t in scope if t not in valid_tasks]
        if bad:
            return {
                "status": "error",
                "error": f"unknown tasks: {bad}",
                "valid_tasks": sorted(valid_tasks),
            }

    project_root = _project_root()
    script = project_root / "scripts" / "run_maintenance_for_cron.py"
    if not script.exists():
        return {"status": "error", "error": f"wrapper script missing: {script}"}

    cmd = [sys.executable, str(script), "--json"]
    if scope:
        for task in scope:
            cmd.extend(["--task", task])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2400,                                       # 40 min hard cap
            cwd=str(project_root),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "maintenance subprocess timed out (40 min cap)",
        }
    except OSError as e:
        return {"status": "error", "error": f"subprocess failed to start: {e}"}

    if proc.returncode == 2:
        return {
            "status": "error",
            "error": "maintenance failed during init/load",
            "stderr": proc.stderr.strip()[:500],
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        # If stdout couldn't be parsed (rare — wrapper always emits
        # JSON when called with --json), salvage what we can.
        return {
            "status": "error" if proc.returncode != 0 else "partial",
            "error": "could not parse wrapper output as JSON",
            "raw_output": proc.stdout[-500:],
        }
    return payload


def list_active_coding_sessions_impl(
    *, max_age_hours: int = 24,
) -> Dict[str, Any]:
    """List recent coding sessions that haven't completed.

    Reads per-session audit logs at ``logs/sessions/<id>.jsonl``.
    A session is "active" when its most recent event is not a
    completion / cancellation / abort terminal event. ``max_age_hours``
    bounds how far back to look — default 24 h.
    """
    audit_dir = _load_session_audit_dir()
    if not audit_dir.exists():
        return {"count": 0, "sessions": []}
    cutoff_ts = time.time() - max_age_hours * 3600
    sessions: List[Dict[str, Any]] = []
    for log_path in sorted(audit_dir.glob("*.jsonl")):
        try:
            mtime = log_path.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff_ts:
            continue
        last_event = _read_last_event(log_path)
        if last_event is None:
            continue
        event_type = (last_event.get("event") or last_event.get("kind") or "").lower()
        # Treat the canonical terminal event names as "complete".
        terminal = {
            "complete", "completed", "complete_session",
            "declared_complete", "cancelled", "canceled",
            "aborted", "failed", "session_complete",
        }
        if event_type in terminal:
            continue
        sessions.append({
            "session_id": log_path.stem,
            "last_event": event_type or "unknown",
            "last_update_seconds_ago": round(time.time() - mtime, 1),
            "stage": last_event.get("stage") or last_event.get("status"),
            "project_root": last_event.get("project_root"),
            "user_intent": (last_event.get("user_intent") or "")[:200],
        })
    return {"count": len(sessions), "sessions": sessions}


def get_recent_voice_alerts_impl(*, limit: int = 5) -> Dict[str, Any]:
    """Convenience wrapper used by the voice intent handler.

    Returns the most recent unacknowledged alerts (up to ``limit``)
    in a shape that's easy to render in voice narration.
    """
    payload = get_heartbeat_alerts_impl(
        since_seconds_ago=7 * 86400,                          # last week
        only_unacknowledged=True,
        limit=max(1, min(limit, 20)),
    )
    if "error" in payload:
        return payload
    summary_lines = [
        f"{a['severity']}: {a['text']}"
        for a in payload.get("alerts", [])
    ]
    return {
        "count": payload.get("count", 0),
        "lines": summary_lines,
        "alerts": payload.get("alerts", []),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_last_event(path: Path) -> Optional[Dict[str, Any]]:
    """Read the last well-formed JSON line from a JSONL audit file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    last: Optional[Dict[str, Any]] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            last = obj
    return last


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------


def build_server():
    """Construct and return a configured :class:`FastMCP` instance.

    Tools register at construction time. The caller drives the
    transport — typically via :func:`run_stdio` for OpenClaw spawn
    integration.
    """
    from mcp.server.fastmcp import FastMCP                    # local import

    mcp = FastMCP(name="ultron-mcp", log_level="WARNING")

    @mcp.tool(
        name="get_heartbeat_alerts",
        description=(
            "Return recent heartbeat alerts. Defaults to the last 24h "
            "of unacknowledged items, max 50."
        ),
    )
    def get_heartbeat_alerts(
        since_seconds_ago: int = 86400,
        only_unacknowledged: bool = True,
        limit: int = 50,
    ) -> Dict[str, Any]:
        return get_heartbeat_alerts_impl(
            since_seconds_ago=since_seconds_ago,
            only_unacknowledged=only_unacknowledged,
            limit=limit,
        )

    @mcp.tool(
        name="acknowledge_alert",
        description=(
            "Mark a heartbeat alert as seen so it won't be surfaced "
            "on the next heartbeat tick."
        ),
    )
    def acknowledge_alert(alert_id: str) -> Dict[str, Any]:
        return acknowledge_alert_impl(alert_id)

    @mcp.tool(
        name="run_maintenance",
        description=(
            "Run Ultron's memory maintenance pipeline. Pass `scope` "
            "as a list of task names, or omit for all tasks. Long-"
            "running (minutes) — caller should not await."
        ),
    )
    def run_maintenance(scope: Optional[List[str]] = None) -> Dict[str, Any]:
        return run_maintenance_impl(scope=scope)

    @mcp.tool(
        name="list_active_coding_sessions",
        description=(
            "List recent coding sessions that haven't completed. "
            "Returns id, last event, project root, and a snippet of "
            "the user's original intent."
        ),
    )
    def list_active_coding_sessions(
        max_age_hours: int = 24,
    ) -> Dict[str, Any]:
        return list_active_coding_sessions_impl(max_age_hours=max_age_hours)

    @mcp.tool(
        name="get_recent_voice_alerts",
        description=(
            "Convenience: most recent unacknowledged alerts in a "
            "voice-narration-friendly shape."
        ),
    )
    def get_recent_voice_alerts(limit: int = 5) -> Dict[str, Any]:
        return get_recent_voice_alerts_impl(limit=limit)

    return mcp


def run_stdio() -> None:
    """Entry point: bring up the MCP server in stdio mode.

    Caller is the stdio entry script
    (``scripts/run_ultron_mcp_for_openclaw.py``). Blocks until
    OpenClaw's spawned client closes the channel.
    """
    import asyncio

    server = build_server()
    asyncio.run(server.run_stdio_async())


__all__ = [
    "acknowledge_alert_impl",
    "build_server",
    "get_heartbeat_alerts_impl",
    "get_recent_voice_alerts_impl",
    "list_active_coding_sessions_impl",
    "run_maintenance_impl",
    "run_stdio",
]
