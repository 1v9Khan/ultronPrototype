"""Stdio MCP server exposing Ultron's data to OpenClaw agents.

OpenClaw 2026.5.7's ``mcp set`` accepts a stdio command + args. Each
agent run that needs an Ultron tool spawns this script as a fresh
process; the script reads from on-disk artifacts (heartbeat alert
log, session audit logs, maintenance subprocess) and returns
results over the MCP stdio protocol.

This is intentionally separate from :class:`UltronMCPServer` (in
``ultron.coding.mcp_server``) which serves the AI coding agent
subprocess via SSE and lives in the orchestrator process. Sharing
that server across both consumers would require a stdio→SSE proxy
shim; running a small dedicated stdio server is simpler and keeps
the surfaces decoupled.

Tools registered:

System / heartbeat / coding:

- ``get_heartbeat_alerts`` — read recent heartbeat alerts.
- ``acknowledge_alert`` — mark a heartbeat alert seen.
- ``run_maintenance`` — kick the maintenance pipeline via
  ``scripts/run_maintenance_for_cron.py``.
- ``list_active_coding_sessions`` — list active sessions from
  per-session audit logs.
- ``get_recent_voice_alerts`` — convenience wrapper that returns
  the data the OpenClaw heartbeat agent needs to avoid
  re-surfacing acknowledged items.

Desktop automation (Phase 7 -- native primitives surfaced to
OpenClaw agents for multi-step task delegation):

- ``enumerate_monitors`` — list connected displays.
- ``list_windows`` — visible top-level windows with monitor index.
- ``take_screenshot`` — capture a monitor (returns base64 PNG +
  metadata; optionally also returns a VLM description).
- ``describe_screen`` — capture + VLM in one call (text only;
  smaller MCP payload than the full screenshot).
- ``get_screen_context`` — assembled "what the user is looking at"
  snapshot for agent reasoning.
- ``launch_app`` — spawn a registered app (Chrome / Cursor /
  Discord / etc.) on a chosen monitor.
- ``launch_chrome_url`` — open a URL in the user's default Chrome
  on a chosen monitor (reuses real session + cookies).
- ``open_image_search`` — Chrome → Google Images for a query
  (the "show me a picture of X" convenience).
- ``move_window_to_monitor`` — move an existing window to a
  monitor (semantic-find via title / process substring).
- ``focus_window`` — bring a window to the foreground.
- ``window_action`` — maximize / minimize / restore.
- ``click_uia`` — semantic click on a UI element by name /
  automation_id (Cap-3 safety gated).
- ``type_into_uia`` — semantic type into an edit element.
- ``get_window_text`` — collect visible UIA text from a window.
- ``mouse_click`` / ``mouse_move`` / ``type_text`` / ``press_hotkey`` /
  ``scroll`` — pixel-coordinate / keyboard primitives via pyautogui
  (validator + rate-limit gated).

All tools fail-open: missing files, malformed JSONL, subprocess
errors, missing pywin32 / mss / pywinauto / transformers all
translate to structured error payloads rather than crashing the
MCP server. Read-only tools never modify on-disk state outside
their explicit write paths.

Process startup is ephemeral — OpenClaw spawns a new instance per
MCP call. Heavy imports (pywin32, mss, pywinauto, transformers /
torch) are deferred to per-tool call sites so cold start stays
under ~1 s on the heartbeat / acknowledge / maintenance paths.
The desktop tools pay their own per-call import cost.
"""

from __future__ import annotations

import base64
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
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
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
# Desktop automation tool implementations (Phase 7)
#
# Every function lazy-imports ultron.desktop so cold start of the MCP
# server stays light on the heartbeat / coding / maintenance paths.
# pywin32 / mss / pywinauto / transformers all load on first desktop
# tool call only.
# ---------------------------------------------------------------------------


def enumerate_monitors_impl() -> Dict[str, Any]:
    """List connected monitors. Returns ``{"count": N, "monitors": [...]}``.

    Each monitor entry contains: index, name, x, y, width, height, and
    is_primary. Fail-open: returns ``{"count": 0, "monitors": []}`` when
    pywin32 is unavailable.
    """
    try:
        from ultron.desktop.monitors import enumerate_monitors
    except Exception as e:                                       # noqa: BLE001
        return {"count": 0, "monitors": [], "error": f"import failed: {e}"}
    mons = enumerate_monitors()
    return {
        "count": len(mons),
        "monitors": [
            {
                "index": m.index, "name": m.name,
                "x": m.x, "y": m.y, "width": m.width, "height": m.height,
                "work_x": m.work_x, "work_y": m.work_y,
                "work_width": m.work_width, "work_height": m.work_height,
                "is_primary": m.is_primary,
            }
            for m in mons
        ],
    }


def list_windows_impl(
    *,
    include_minimized: bool = False,
    include_invisible: bool = False,
    limit: int = 40,
) -> Dict[str, Any]:
    """List visible top-level windows. ``limit`` caps the response."""
    try:
        from ultron.desktop.windows import enumerate_windows
    except Exception as e:                                       # noqa: BLE001
        return {"count": 0, "windows": [], "error": f"import failed: {e}"}
    wins = enumerate_windows(
        include_minimized=bool(include_minimized),
        include_invisible=bool(include_invisible),
    )
    if limit > 0:
        wins = wins[:limit]
    return {
        "count": len(wins),
        "windows": [
            {
                "hwnd": w.hwnd, "title": w.title,
                "class_name": w.class_name,
                "process_name": w.process_name, "pid": w.pid,
                "rect": list(w.rect),
                "monitor_index": w.monitor_index,
                "is_minimized": w.is_minimized,
                "is_foreground": w.is_foreground,
            }
            for w in wins
        ],
    }


def take_screenshot_impl(
    *,
    monitor_index: Optional[int] = None,
    include_image: bool = True,
    include_description: bool = False,
) -> Dict[str, Any]:
    """Capture a monitor screenshot.

    Args:
        monitor_index: which monitor to capture; None = foreground monitor
            (or monitor 0 when no foreground).
        include_image: when True, embed the PNG as base64. Set False for
            metadata-only responses (much smaller payload).
        include_description: when True, also run the VLM on the capture
            and return the description text. Adds ~5-8 s.

    Returns ``{"success": bool, "monitor_index": int, "width": int,
    "height": int, ..., "image_base64": Optional[str], "description":
    Optional[str]}``.
    """
    try:
        from ultron.desktop.capture import get_screen_capture
        from ultron.desktop.monitors import enumerate_monitors
        from ultron.desktop.windows import get_foreground_window
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}

    # Resolve target monitor index.
    if monitor_index is None:
        fg = get_foreground_window()
        if fg is not None and fg.monitor_index is not None:
            target = fg.monitor_index
        else:
            target = 0
    else:
        target = int(monitor_index)

    mons = enumerate_monitors()
    if not mons:
        return {"success": False, "error": "no monitors detected"}
    if not (0 <= target < len(mons)):
        return {
            "success": False,
            "error": f"monitor_index {target} out of range (have {len(mons)})",
        }

    cap = get_screen_capture()
    shot = cap.capture_monitor(target)
    if shot is None:
        return {"success": False, "error": "capture failed"}

    payload: Dict[str, Any] = {
        "success": True,
        "monitor_index": shot.monitor_index,
        "width": shot.width,
        "height": shot.height,
        "timestamp": shot.timestamp,
    }
    if include_image:
        payload["image_base64"] = base64.b64encode(shot.image_bytes).decode("ascii")
        payload["image_bytes_length"] = len(shot.image_bytes)

    if include_description:
        try:
            from ultron.desktop.vlm import get_vlm
        except Exception as e:                                   # noqa: BLE001
            payload["description_error"] = f"VLM import failed: {e}"
        else:
            vlm = get_vlm()
            if vlm is None:
                payload["description_error"] = "VLM not configured"
            else:
                result = vlm.describe(shot.image_bytes)
                if result.success:
                    payload["description"] = result.description
                    payload["description_elapsed_ms"] = result.elapsed_ms
                else:
                    payload["description_error"] = result.error

    return payload


def describe_screen_impl(
    *,
    monitor_index: Optional[int] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture + VLM in one call. Returns text-only payload.

    Convenience for agents that only want the description (small
    response, no base64 image bytes). Equivalent to ``take_screenshot``
    with ``include_image=False, include_description=True``.
    """
    payload = take_screenshot_impl(
        monitor_index=monitor_index,
        include_image=False,
        include_description=True,
    )
    if not payload.get("success"):
        return payload
    # If a custom prompt is supplied, re-run with it (the bundled call
    # used the default prompt).
    if prompt:
        try:
            from ultron.desktop.capture import get_screen_capture
            from ultron.desktop.vlm import get_vlm
        except Exception as e:                                   # noqa: BLE001
            payload["description_error"] = f"VLM import failed: {e}"
            return payload
        vlm = get_vlm()
        if vlm is None:
            payload["description_error"] = "VLM not configured"
            return payload
        # We need the bytes again; re-capture is cheap.
        cap = get_screen_capture()
        shot = cap.capture_monitor(int(payload["monitor_index"]))
        if shot is None:
            payload["description_error"] = "recapture for prompt failed"
            return payload
        result = vlm.describe(shot.image_bytes, prompt=prompt)
        if result.success:
            payload["description"] = result.description
            payload["description_elapsed_ms"] = result.elapsed_ms
            payload.pop("description_error", None)
        else:
            payload["description_error"] = result.error
    return payload


def get_screen_context_impl(
    *,
    include_uia: bool = True,
    include_vlm: bool = False,
    window_list_cap: int = 12,
) -> Dict[str, Any]:
    """Assembled screen-context snapshot ready for agent reasoning.

    Returns the structured view the orchestrator uses for "explain
    what I'm looking at" -- foreground app + window list + visible
    UIA text + optional VLM description. Does NOT include the
    screenshot image bytes (use ``take_screenshot`` for that).
    """
    try:
        from ultron.desktop.screen_context import build_screen_context
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}

    snap = build_screen_context(
        capture=bool(include_vlm),                              # only capture if VLM will read it
        include_uia=bool(include_uia),
        include_vlm=bool(include_vlm),
        window_list_cap=int(window_list_cap),
    )

    fg = snap.foreground
    return {
        "success": True,
        "timestamp": snap.timestamp,
        "elapsed_ms": snap.elapsed_ms,
        "foreground": (
            None if fg is None else {
                "hwnd": fg.hwnd, "title": fg.title,
                "process_name": fg.process_name,
                "monitor_index": fg.monitor_index,
            }
        ),
        "monitors": [
            {"index": m.index, "is_primary": m.is_primary,
             "width": m.width, "height": m.height}
            for m in snap.monitors
        ],
        "windows": [
            {"title": w.title, "process_name": w.process_name,
             "monitor_index": w.monitor_index,
             "is_foreground": w.is_foreground}
            for w in snap.windows
        ],
        "ui_text": list(snap.ui_text),
        "vlm_description": snap.vlm_description,
        "render_for_llm": snap.render_for_llm(),
    }


def _resolve_monitor(monitor_index: Optional[int]):
    """Look up a :class:`Monitor` by index. Returns ``(monitor, error_dict)``.

    On success ``error_dict`` is None. On failure ``monitor`` is None
    and ``error_dict`` is a ready-to-return MCP payload.
    """
    if monitor_index is None:
        return None, None
    try:
        from ultron.desktop.monitors import enumerate_monitors
    except Exception as e:                                       # noqa: BLE001
        return None, {"success": False, "error": f"import failed: {e}"}
    mons = enumerate_monitors()
    if not (0 <= int(monitor_index) < len(mons)):
        return None, {
            "success": False,
            "error": f"monitor_index {monitor_index} out of range",
        }
    return mons[int(monitor_index)], None


def launch_app_impl(
    *,
    app_name: str,
    monitor_index: Optional[int] = None,
    fullscreen: bool = False,
    maximize: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Launch a registered app, optionally on a target monitor.

    Args:
        app_name: registry name or alias (``"chrome"``, ``"cursor"``,
            ``"discord"``, ``"vscode"``, etc.).
        monitor_index: when set, move the launched window to this monitor.
        fullscreen: fill the monitor as a regular window.
        maximize: ``ShowWindow(SW_MAXIMIZE)`` after placement.
        extra_args: appended to the launcher's args.
    """
    if not app_name or not isinstance(app_name, str):
        return {"success": False, "error": "app_name is required"}
    mon, err = _resolve_monitor(monitor_index)
    if err is not None:
        return err
    try:
        from ultron.desktop.launcher import get_app_launcher
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    launcher = get_app_launcher()
    result = launcher.launch_app(
        app_name=app_name,
        monitor=mon,
        extra_args=list(extra_args or []),
        fullscreen=bool(fullscreen),
        maximize=bool(maximize),
        wait_for_window=mon is not None,
    )
    return {
        "success": result.success,
        "app_name": result.app_name,
        "exe_path": str(result.exe_path) if result.exe_path else None,
        "pid": result.pid,
        "hwnd": result.hwnd,
        "monitor_index": result.monitor_index,
        "window_appeared": getattr(result, "window_appeared", None),
        "error": result.error,
    }


def launch_chrome_url_impl(
    *,
    url: str,
    monitor_index: Optional[int] = None,
    fullscreen: bool = False,
    maximize: bool = False,
    window_width: Optional[int] = None,
    window_height: Optional[int] = None,
) -> Dict[str, Any]:
    """Open a URL in the user's real Chrome (default profile, signed-in)."""
    if not url or not isinstance(url, str):
        return {"success": False, "error": "url is required"}
    mon, err = _resolve_monitor(monitor_index)
    if err is not None:
        return err
    try:
        from ultron.desktop.launcher import get_app_launcher
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    size = None
    if window_width and window_height:
        size = (int(window_width), int(window_height))
    launcher = get_app_launcher()
    result = launcher.launch_chrome(
        url=url, monitor=mon,
        fullscreen=bool(fullscreen),
        maximize=bool(maximize),
        window_size=size,
    )
    return {
        "success": result.success,
        "url": url,
        "pid": result.pid,
        "hwnd": result.hwnd,
        "monitor_index": result.monitor_index,
        "window_appeared": getattr(result, "window_appeared", None),
        "error": result.error,
    }


def open_image_search_impl(
    *,
    query: str,
    monitor_index: Optional[int] = None,
    small_window: bool = True,
) -> Dict[str, Any]:
    """Open Google Images for a query in a new Chrome window."""
    if not query or not isinstance(query, str) or not query.strip():
        return {"success": False, "error": "query is required"}
    mon, err = _resolve_monitor(monitor_index)
    if err is not None:
        return err
    try:
        from ultron.desktop.launcher import get_app_launcher
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    launcher = get_app_launcher()
    result = launcher.open_image_search(
        query=query, monitor=mon, small_window=bool(small_window),
    )
    return {
        "success": result.success,
        "query": query,
        "pid": result.pid,
        "hwnd": result.hwnd,
        "monitor_index": result.monitor_index,
        "window_appeared": getattr(result, "window_appeared", None),
        "error": result.error,
    }


def move_window_to_monitor_impl(
    *,
    window_query: str,
    monitor_index: int,
    fullscreen: bool = False,
    maximize: bool = False,
) -> Dict[str, Any]:
    """Move an existing window to a target monitor.

    ``window_query`` is a substring match on title or process name
    (same semantics as :func:`ultron.desktop.windows.find_window`).
    """
    if not window_query or not isinstance(window_query, str):
        return {"success": False, "error": "window_query is required"}
    mon, err = _resolve_monitor(monitor_index)
    if err is not None:
        return err
    if mon is None:
        return {"success": False, "error": "monitor_index is required"}
    try:
        from ultron.desktop.placement import move_window_to_monitor
        from ultron.desktop.windows import find_window
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    win = find_window(window_query)
    if win is None:
        return {
            "success": False,
            "error": f"no window matching {window_query!r}",
        }
    result = move_window_to_monitor(
        hwnd=win.hwnd, monitor=mon,
        fullscreen=bool(fullscreen), maximize=bool(maximize),
    )
    return {
        "success": result.success,
        "hwnd": result.hwnd,
        "window_title": win.title,
        "monitor_index": result.monitor_index,
        "error": result.error,
    }


def focus_window_impl(*, window_query: str) -> Dict[str, Any]:
    """Bring a window to the foreground."""
    if not window_query or not isinstance(window_query, str):
        return {"success": False, "error": "window_query is required"}
    try:
        from ultron.desktop.placement import focus_window
        from ultron.desktop.windows import find_window
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    win = find_window(window_query)
    if win is None:
        return {
            "success": False,
            "error": f"no window matching {window_query!r}",
        }
    result = focus_window(win.hwnd)
    return {
        "success": result.success,
        "hwnd": result.hwnd,
        "window_title": win.title,
        "error": result.error,
    }


def window_action_impl(
    *,
    window_query: str,
    action: str,
) -> Dict[str, Any]:
    """Maximize / minimize / restore an existing window.

    ``action`` is one of ``maximize`` / ``minimize`` / ``restore``.
    """
    if not window_query or not isinstance(window_query, str):
        return {"success": False, "error": "window_query is required"}
    action = (action or "").strip().lower()
    if action not in ("maximize", "minimize", "restore"):
        return {
            "success": False,
            "error": f"unknown action {action!r}; "
                     f"expected maximize/minimize/restore",
        }
    try:
        from ultron.desktop.placement import (
            maximize_window, minimize_window, restore_window,
        )
        from ultron.desktop.windows import find_window
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    win = find_window(window_query)
    if win is None:
        return {
            "success": False,
            "error": f"no window matching {window_query!r}",
        }
    fn = {
        "maximize": maximize_window,
        "minimize": minimize_window,
        "restore": restore_window,
    }[action]
    result = fn(win.hwnd)
    return {
        "success": result.success,
        "hwnd": result.hwnd,
        "window_title": win.title,
        "action": action,
        "error": result.error,
    }


def click_uia_impl(
    *,
    window_query: str,
    element_query: str,
    automation_id: Optional[str] = None,
    control_type: Optional[str] = None,
    exact: bool = False,
    user_text: str = "",
) -> Dict[str, Any]:
    """UIA semantic click: find a window, find an element by name /
    automation_id within it, invoke the click.

    Goes through the safety validator (Cap-3 action-verb-click rule
    flags ``Submit`` / ``Pay`` / ``Send Money`` / etc.).
    """
    if not window_query or not isinstance(window_query, str):
        return {"success": False, "error": "window_query is required"}
    if not element_query or not isinstance(element_query, str):
        return {"success": False, "error": "element_query is required"}
    try:
        from ultron.desktop.uia import click_element
        from ultron.desktop.windows import find_window
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    win = find_window(window_query)
    if win is None:
        return {
            "success": False,
            "error": f"no window matching {window_query!r}",
        }
    result = click_element(
        win, element_query,
        automation_id=automation_id,
        control_type=control_type,
        exact=bool(exact),
        user_text=user_text,
    )
    return {
        "success": result.success,
        "element_name": result.element_name,
        "window_title": win.title,
        "error": result.error,
    }


def type_into_uia_impl(
    *,
    window_query: str,
    element_query: str,
    text: str,
    automation_id: Optional[str] = None,
    control_type: Optional[str] = None,
    exact: bool = False,
    clear_first: bool = True,
    user_text: str = "",
) -> Dict[str, Any]:
    """UIA semantic type: find a window, find an edit element, type ``text``."""
    if not window_query or not isinstance(window_query, str):
        return {"success": False, "error": "window_query is required"}
    if not element_query or not isinstance(element_query, str):
        return {"success": False, "error": "element_query is required"}
    if not isinstance(text, str):
        return {"success": False, "error": "text must be a string"}
    try:
        from ultron.desktop.uia import type_text_into_element
        from ultron.desktop.windows import find_window
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    win = find_window(window_query)
    if win is None:
        return {
            "success": False,
            "error": f"no window matching {window_query!r}",
        }
    result = type_text_into_element(
        win, element_query, text,
        automation_id=automation_id,
        control_type=control_type,
        exact=bool(exact),
        clear_first=bool(clear_first),
        user_text=user_text,
    )
    return {
        "success": result.success,
        "element_name": result.element_name,
        "window_title": win.title,
        "error": result.error,
    }


def get_window_text_impl(*, window_query: str) -> Dict[str, Any]:
    """Collect visible UIA text strings from a window.

    Useful for "what does this dialog say" without spinning up the VLM.
    """
    if not window_query or not isinstance(window_query, str):
        return {"success": False, "error": "window_query is required"}
    try:
        from ultron.desktop.uia import collect_window_text
        from ultron.desktop.windows import find_window
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    win = find_window(window_query)
    if win is None:
        return {
            "success": False,
            "error": f"no window matching {window_query!r}",
        }
    text_lines = collect_window_text(win)
    return {
        "success": True,
        "window_title": win.title,
        "text_lines": list(text_lines),
        "count": len(text_lines),
    }


def mouse_click_impl(
    *,
    x: Optional[int] = None,
    y: Optional[int] = None,
    button: str = "left",
    clicks: int = 1,
    user_text: str = "",
) -> Dict[str, Any]:
    """Pixel-coordinate mouse click via pyautogui (validator-gated).

    When ``x``/``y`` are null, clicks at current cursor location.
    """
    try:
        from ultron.desktop.input_control import get_input_controller
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    ctrl = get_input_controller()
    result = ctrl.click(
        x=x, y=y, button=button, clicks=int(clicks),
        user_text=user_text,
    )
    return {
        "success": result.success,
        "action": "mouse_click",
        "error": result.error,
    }


def mouse_move_impl(
    *,
    x: int,
    y: int,
    duration_s: float = 0.1,
    smooth: bool = False,
    user_text: str = "",
) -> Dict[str, Any]:
    """Move the cursor to absolute (x, y) coordinates.

    Catalog 09 T7 (GREEN): when ``smooth=True`` with ``duration_s>0``
    the move uses pyautogui's quadratic ease-in/ease-out tween rather
    than a linear path.
    """
    try:
        from ultron.desktop.input_control import get_input_controller
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    ctrl = get_input_controller()
    result = ctrl.move_mouse(
        x=int(x), y=int(y), duration_s=float(duration_s),
        smooth=bool(smooth),
        user_text=user_text,
    )
    return {
        "success": result.success,
        "action": "mouse_move",
        "error": result.error,
    }


def type_text_impl(
    *,
    text: str,
    interval_s: float = 0.0,
    wpm: Optional[int] = None,
    user_text: str = "",
) -> Dict[str, Any]:
    """Type text at the current keyboard focus (validator-gated).

    Catalog 09 T3 (GREEN): optional ``wpm`` (positive integer)
    overrides ``interval_s`` with the standard 5-chars-per-word
    cadence formula. 60-80 WPM passes most JS form validators that
    reject ``interval_s=0`` instant input.
    """
    try:
        from ultron.desktop.input_control import get_input_controller
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    ctrl = get_input_controller()
    result = ctrl.type_text(
        text=text, interval_s=float(interval_s),
        wpm=int(wpm) if wpm is not None else None,
        user_text=user_text,
    )
    return {
        "success": result.success,
        "action": "type_text",
        "error": result.error,
    }


def press_hotkey_impl(
    *,
    keys: List[str],
    user_text: str = "",
) -> Dict[str, Any]:
    """Press a hotkey combination (``["ctrl", "s"]``, ``["alt", "tab"]``).

    Keys are pressed in order then released in reverse.
    """
    if not keys or not isinstance(keys, list):
        return {"success": False, "error": "keys must be a non-empty list"}
    try:
        from ultron.desktop.input_control import get_input_controller
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    ctrl = get_input_controller()
    result = ctrl.press_hotkey(*keys, user_text=user_text)
    return {
        "success": result.success,
        "action": "press_hotkey",
        "error": result.error,
    }


def scroll_impl(
    *,
    amount: int,
    direction: str = "vertical",
    x: Optional[int] = None,
    y: Optional[int] = None,
    user_text: str = "",
) -> Dict[str, Any]:
    """Scroll the wheel at ``(x, y)`` (or current cursor location).

    Catalog 09 T1 (YELLOW): ``direction="vertical"`` (default) maps to
    pyautogui.scroll; ``direction="horizontal"`` maps to
    pyautogui.hscroll. Closes the browser-content-extraction gap
    where catalog 08 T5 can read UIA text but can't scroll the page
    to load lazy content.
    """
    try:
        from ultron.desktop.input_control import get_input_controller
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    ctrl = get_input_controller()
    result = ctrl.scroll(
        amount=int(amount), direction=direction, x=x, y=y,
        user_text=user_text,
    )
    return {
        "success": result.success,
        "action": "scroll",
        "error": result.error,
    }


def find_image_on_screen_impl(
    *,
    template_path: str,
    confidence: float = 0.8,
    region_left: Optional[int] = None,
    region_top: Optional[int] = None,
    region_width: Optional[int] = None,
    region_height: Optional[int] = None,
) -> Dict[str, Any]:
    """Locate a saved template image on screen (catalog 09 T6, YELLOW).

    Cap-2 read-only observation. The returned coordinates are
    consumed by a downstream gated ``mouse_click`` / ``InputController``
    call; the template-matching step itself does not touch input.

    The ``region_*`` kwargs are exposed individually rather than as
    a tuple so the MCP transport (JSON over stdio) can carry them
    naturally. Pass all four to constrain the search, or none for a
    full-screen scan.
    """
    try:
        from ultron.desktop.capture import find_image_on_screen
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}

    region = None
    if (
        region_left is not None or region_top is not None
        or region_width is not None or region_height is not None
    ):
        if (
            region_left is None or region_top is None
            or region_width is None or region_height is None
        ):
            return {
                "success": False,
                "error": "all four region_* args must be set together",
            }
        region = (
            int(region_left), int(region_top),
            int(region_width), int(region_height),
        )

    match = find_image_on_screen(
        template_path,
        confidence=float(confidence),
        region=region,
    )
    if match is None:
        return {
            "success": False,
            "action": "find_image_on_screen",
            "error": "no match (template missing, opencv absent, or below threshold)",
        }
    return {
        "success": True,
        "action": "find_image_on_screen",
        "match": {
            "left": match.left,
            "top": match.top,
            "width": match.width,
            "height": match.height,
            "center_x": match.center_x,
            "center_y": match.center_y,
            "confidence": match.confidence,
        },
    }


def clipboard_read_impl(
    *,
    user_text: str = "",
) -> Dict[str, Any]:
    """Read the system clipboard text (catalog 09 T4, YELLOW).

    Cap-2 read: routes through the safety validator with
    ``tool_name=desktop.clipboard.read``; the returned text is
    recorded in the taint tracker so any subsequent outbound tool
    call carrying these exact bytes trips the exfil check.
    """
    try:
        from ultron.desktop.clipboard import get_clipboard_manager
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    mgr = get_clipboard_manager()
    result = mgr.read_text(user_text=user_text)
    out: Dict[str, Any] = {
        "success": result.success,
        "action": "clipboard_read",
        "error": result.error,
    }
    if result.success:
        out["text"] = result.text
        out["tainted"] = result.tainted
    return out


def clipboard_write_impl(
    *,
    text: str,
    user_text: str = "",
) -> Dict[str, Any]:
    """Write text to the system clipboard (catalog 09 T4, YELLOW).

    Cap-3 write: routes through the safety validator with the full
    payload (2 KB preview when very large) so payload-based rules
    can block. The written bytes are recorded in the taint tracker
    so the orchestrator can verify a downstream paste lands in the
    expected target.
    """
    if not isinstance(text, str):
        return {"success": False, "error": "text must be string"}
    try:
        from ultron.desktop.clipboard import get_clipboard_manager
    except Exception as e:                                       # noqa: BLE001
        return {"success": False, "error": f"import failed: {e}"}
    mgr = get_clipboard_manager()
    result = mgr.write_text(text, user_text=user_text)
    return {
        "success": result.success,
        "action": "clipboard_write",
        "error": result.error,
        "tainted": result.tainted,
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

    # --- Desktop automation tools (Phase 7) -----------------------------

    @mcp.tool(
        name="enumerate_monitors",
        description=(
            "List connected monitors with their virtual-screen "
            "coordinates and primary flag."
        ),
    )
    def enumerate_monitors() -> Dict[str, Any]:
        return enumerate_monitors_impl()

    @mcp.tool(
        name="list_windows",
        description=(
            "List visible top-level windows with title, process name, "
            "monitor index, and foreground state. Default skips "
            "minimized windows."
        ),
    )
    def list_windows(
        include_minimized: bool = False,
        include_invisible: bool = False,
        limit: int = 40,
    ) -> Dict[str, Any]:
        return list_windows_impl(
            include_minimized=include_minimized,
            include_invisible=include_invisible,
            limit=limit,
        )

    @mcp.tool(
        name="take_screenshot",
        description=(
            "Capture a monitor (or the foreground monitor when "
            "monitor_index is null). Returns base64 PNG + metadata. "
            "Set include_description=true to also run the VLM "
            "(adds ~5-8 s)."
        ),
    )
    def take_screenshot(
        monitor_index: Optional[int] = None,
        include_image: bool = True,
        include_description: bool = False,
    ) -> Dict[str, Any]:
        return take_screenshot_impl(
            monitor_index=monitor_index,
            include_image=include_image,
            include_description=include_description,
        )

    @mcp.tool(
        name="describe_screen",
        description=(
            "Capture + VLM in one call. Returns text description "
            "only (no image bytes). Use 'prompt' to override the "
            "default 'describe what's visible' prompt with a "
            "specific question."
        ),
    )
    def describe_screen(
        monitor_index: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        return describe_screen_impl(
            monitor_index=monitor_index, prompt=prompt,
        )

    @mcp.tool(
        name="get_screen_context",
        description=(
            "Assembled 'what the user is looking at' snapshot: "
            "foreground app + window list + visible UIA text + "
            "optional VLM description. Returns structured payload "
            "plus a render_for_llm string ready for prompt injection."
        ),
    )
    def get_screen_context(
        include_uia: bool = True,
        include_vlm: bool = False,
        window_list_cap: int = 12,
    ) -> Dict[str, Any]:
        return get_screen_context_impl(
            include_uia=include_uia,
            include_vlm=include_vlm,
            window_list_cap=window_list_cap,
        )

    @mcp.tool(
        name="launch_app",
        description=(
            "Launch a registered app (chrome / cursor / discord / "
            "vscode / edge / firefox / notepad / explorer / terminal "
            "/ spotify / slack / obs) with optional monitor "
            "targeting + fullscreen / maximize placement."
        ),
    )
    def launch_app(
        app_name: str,
        monitor_index: Optional[int] = None,
        fullscreen: bool = False,
        maximize: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return launch_app_impl(
            app_name=app_name,
            monitor_index=monitor_index,
            fullscreen=fullscreen,
            maximize=maximize,
            extra_args=extra_args,
        )

    @mcp.tool(
        name="launch_chrome_url",
        description=(
            "Open a URL in the user's real Chrome (default profile, "
            "signed-in sessions preserved). Use this for 'open "
            "YouTube on my second monitor' rather than the generic "
            "browser plugin which uses an isolated profile."
        ),
    )
    def launch_chrome_url(
        url: str,
        monitor_index: Optional[int] = None,
        fullscreen: bool = False,
        maximize: bool = False,
        window_width: Optional[int] = None,
        window_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        return launch_chrome_url_impl(
            url=url,
            monitor_index=monitor_index,
            fullscreen=fullscreen,
            maximize=maximize,
            window_width=window_width,
            window_height=window_height,
        )

    @mcp.tool(
        name="open_image_search",
        description=(
            "Open a Google Images search for a query in a new Chrome "
            "window. Convenience for 'show me a picture of X'."
        ),
    )
    def open_image_search(
        query: str,
        monitor_index: Optional[int] = None,
        small_window: bool = True,
    ) -> Dict[str, Any]:
        return open_image_search_impl(
            query=query, monitor_index=monitor_index,
            small_window=small_window,
        )

    @mcp.tool(
        name="move_window_to_monitor",
        description=(
            "Move an existing window to a target monitor. "
            "window_query is a substring match on title or process "
            "name (e.g. 'chrome', 'cursor', 'discord')."
        ),
    )
    def move_window_to_monitor(
        window_query: str,
        monitor_index: int,
        fullscreen: bool = False,
        maximize: bool = False,
    ) -> Dict[str, Any]:
        return move_window_to_monitor_impl(
            window_query=window_query,
            monitor_index=monitor_index,
            fullscreen=fullscreen,
            maximize=maximize,
        )

    # --- Extended desktop tools (window actions, UIA, input) -----------

    @mcp.tool(
        name="focus_window",
        description=(
            "Bring a window to the foreground. window_query is a "
            "substring match on title or process name."
        ),
    )
    def focus_window(window_query: str) -> Dict[str, Any]:
        return focus_window_impl(window_query=window_query)

    @mcp.tool(
        name="window_action",
        description=(
            "Maximize / minimize / restore a window. "
            "action must be one of 'maximize', 'minimize', 'restore'."
        ),
    )
    def window_action(window_query: str, action: str) -> Dict[str, Any]:
        return window_action_impl(window_query=window_query, action=action)

    @mcp.tool(
        name="click_uia",
        description=(
            "Click a UI element by name or automation_id within a "
            "window using UI Automation (semantic, not pixel-based). "
            "Subject to Cap-3 action-verb safety rule "
            "(Submit/Pay/Send/Transfer return NEEDS_EXPLICIT_INTENT)."
        ),
    )
    def click_uia(
        window_query: str,
        element_query: str,
        automation_id: Optional[str] = None,
        control_type: Optional[str] = None,
        exact: bool = False,
        user_text: str = "",
    ) -> Dict[str, Any]:
        return click_uia_impl(
            window_query=window_query,
            element_query=element_query,
            automation_id=automation_id,
            control_type=control_type,
            exact=exact,
            user_text=user_text,
        )

    @mcp.tool(
        name="type_into_uia",
        description=(
            "Type text into a UI element by name or automation_id "
            "via UI Automation. clear_first wipes existing content."
        ),
    )
    def type_into_uia(
        window_query: str,
        element_query: str,
        text: str,
        automation_id: Optional[str] = None,
        control_type: Optional[str] = None,
        exact: bool = False,
        clear_first: bool = True,
        user_text: str = "",
    ) -> Dict[str, Any]:
        return type_into_uia_impl(
            window_query=window_query,
            element_query=element_query,
            text=text,
            automation_id=automation_id,
            control_type=control_type,
            exact=exact,
            clear_first=clear_first,
            user_text=user_text,
        )

    @mcp.tool(
        name="get_window_text",
        description=(
            "Collect visible UI Automation text from a window. "
            "Useful for reading dialog content / form labels / status "
            "bars without spinning up the VLM."
        ),
    )
    def get_window_text(window_query: str) -> Dict[str, Any]:
        return get_window_text_impl(window_query=window_query)

    @mcp.tool(
        name="mouse_click",
        description=(
            "Pixel-coordinate mouse click via pyautogui (validator + "
            "rate-limit gated). When x/y are null, clicks at current "
            "cursor location. button is 'left'/'right'/'middle'."
        ),
    )
    def mouse_click(
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        clicks: int = 1,
        user_text: str = "",
    ) -> Dict[str, Any]:
        return mouse_click_impl(
            x=x, y=y, button=button, clicks=clicks, user_text=user_text,
        )

    @mcp.tool(
        name="mouse_move",
        description=(
            "Move the cursor to absolute (x, y) coordinates over "
            "duration_s seconds. When smooth=true with duration_s>0 the "
            "move uses a bezier ease-in/ease-out tween (catalog 09 T7) "
            "rather than the default linear path -- helps with gaming-"
            "mode anti-detection and demo narration."
        ),
    )
    def mouse_move(
        x: int,
        y: int,
        duration_s: float = 0.1,
        smooth: bool = False,
        user_text: str = "",
    ) -> Dict[str, Any]:
        return mouse_move_impl(
            x=x, y=y, duration_s=duration_s, smooth=smooth,
            user_text=user_text,
        )

    @mcp.tool(
        name="type_text",
        description=(
            "Type a string at the current keyboard focus. For semantic "
            "targeting use type_into_uia. Validator + rate-limit gated. "
            "Optional wpm (catalog 09 T3) overrides interval_s with a "
            "human-cadence delay -- 60-80 WPM passes most JS form "
            "validators that reject interval_s=0 instant input."
        ),
    )
    def type_text(
        text: str,
        interval_s: float = 0.0,
        wpm: Optional[int] = None,
        user_text: str = "",
    ) -> Dict[str, Any]:
        return type_text_impl(
            text=text, interval_s=interval_s, wpm=wpm,
            user_text=user_text,
        )

    @mcp.tool(
        name="press_hotkey",
        description=(
            "Press a hotkey combination (['ctrl', 's'], ['alt', 'tab'], "
            "['ctrl', 'shift', 't']). Keys pressed in order, released "
            "in reverse."
        ),
    )
    def press_hotkey(
        keys: List[str], user_text: str = "",
    ) -> Dict[str, Any]:
        return press_hotkey_impl(keys=keys, user_text=user_text)

    @mcp.tool(
        name="scroll",
        description=(
            "Scroll the mouse wheel at (x, y) or current cursor "
            "location. direction='vertical' (default) scrolls up "
            "(positive amount) or down (negative); direction='horizontal' "
            "(catalog 09 T1) scrolls left (positive) or right (negative). "
            "amount is in OS-specific scroll units (~120 per notch)."
        ),
    )
    def scroll(
        amount: int,
        direction: str = "vertical",
        x: Optional[int] = None,
        y: Optional[int] = None,
        user_text: str = "",
    ) -> Dict[str, Any]:
        return scroll_impl(
            amount=amount, direction=direction, x=x, y=y,
            user_text=user_text,
        )

    @mcp.tool(
        name="find_image_on_screen",
        description=(
            "Locate a saved template image on screen (catalog 09 T6). "
            "Returns center_x/center_y physical pixel coordinates of "
            "the best match plus the bounding rect; pass these to "
            "mouse_click to click the located element. Requires "
            "opencv-python; falls open (returns success=false) if "
            "opencv is missing OR no match meets the confidence "
            "threshold. Pass all four region_* args to constrain "
            "search to a sub-rectangle."
        ),
    )
    def find_image_on_screen(
        template_path: str,
        confidence: float = 0.8,
        region_left: Optional[int] = None,
        region_top: Optional[int] = None,
        region_width: Optional[int] = None,
        region_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        return find_image_on_screen_impl(
            template_path=template_path,
            confidence=confidence,
            region_left=region_left,
            region_top=region_top,
            region_width=region_width,
            region_height=region_height,
        )

    @mcp.tool(
        name="clipboard_read",
        description=(
            "Read the system clipboard text (catalog 09 T4). Cap-2 "
            "observation: returned bytes are taint-tracked so any "
            "outbound tool carrying them later trips the validator's "
            "exfil check. The clipboard can hold sensitive content "
            "(passwords, private keys, confidential snippets); use "
            "this only when the user explicitly asked to read."
        ),
    )
    def clipboard_read(user_text: str = "") -> Dict[str, Any]:
        return clipboard_read_impl(user_text=user_text)

    @mcp.tool(
        name="clipboard_write",
        description=(
            "Write text to the system clipboard (catalog 09 T4). Cap-3 "
            "mutation: validator sees the full payload (2 KB preview "
            "on very large content) so payload-based rules can block. "
            "Bytes are taint-tracked so the orchestrator can verify "
            "the downstream Ctrl+V lands in the expected target."
        ),
    )
    def clipboard_write(text: str, user_text: str = "") -> Dict[str, Any]:
        return clipboard_write_impl(text=text, user_text=user_text)

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
    # heartbeat / coding / maintenance
    "acknowledge_alert_impl",
    "build_server",
    "get_heartbeat_alerts_impl",
    "get_recent_voice_alerts_impl",
    "list_active_coding_sessions_impl",
    "run_maintenance_impl",
    "run_stdio",
    # desktop automation (Phase 7)
    "describe_screen_impl",
    "enumerate_monitors_impl",
    "get_screen_context_impl",
    "launch_app_impl",
    "launch_chrome_url_impl",
    "list_windows_impl",
    "move_window_to_monitor_impl",
    "open_image_search_impl",
    "take_screenshot_impl",
    # extended desktop tools (Phase 7 polish)
    "click_uia_impl",
    "clipboard_read_impl",
    "clipboard_write_impl",
    "find_image_on_screen_impl",
    "focus_window_impl",
    "get_window_text_impl",
    "mouse_click_impl",
    "mouse_move_impl",
    "press_hotkey_impl",
    "scroll_impl",
    "type_into_uia_impl",
    "type_text_impl",
    "window_action_impl",
]
