"""One-shot helper to append the Phase 0 OpenClaw integration baseline
record to baselines.json. Mirrors the schema specified in the OpenClaw
integration prompt (Phase 0 Section 2.4) with the partial-record fields
populated from autonomous probes; user-driven interactive measurements
remain None until the user runs the corresponding step.

Run once from this worktree's root:
    python scripts/_record_phase0_baseline.py
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

PATH = Path("baselines.json")
KEY = "phase_0_openclaw_integration"

phase0 = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "openclaw_integration_phase_0_partial",
    "note": (
        "Partial baseline: VRAM idle + OpenClaw inventory captured "
        "autonomously. Voice-query + OpenClaw-turn VRAM and first-token "
        "latency require user-approved interactive model loads."
    ),
    "vram_idle_mb": 2986,
    "vram_during_voice_query_mb": None,
    "vram_during_openclaw_turn_mb": None,
    "first_token_latency_ms_p50": None,
    "first_token_latency_ms_p95": None,
    "openclaw": {
        "version": "2026.5.7",
        "build": "eeef486",
        "config_file": r"C:\Users\alecf\.openclaw\openclaw.json",
        "workspace_dir": r"C:\Users\alecf\.openclaw\workspace",
        "gateway_url": "ws://127.0.0.1:18789",
        "gateway_running": False,
        "gateway_mode": "local",
        "channels_configured": [],
        "mcp_servers_configured": [],
        "models_configured": ["openai/gpt-5.5 (placeholder; no API key)"],
        "default_agent_id": "main",
        "default_agent_runtime": "OpenClaw Pi Default",
        "default_agent_model": "gpt-5.5",
        "heartbeat_default": "30m on agent 'main'",
        "plugins_loaded": 48,
        "openai_provider_present": True,
        "ollama_provider_present": True,
        "lmstudio_provider_present": True,
        "doctor_findings": [
            "No command owner configured (commands.ownerAllowFrom unset).",
            "1/1 recent sessions missing transcripts (history will appear to reset).",
            "Skills: 6 eligible, 46 missing requirements (mostly bins/env).",
            "Gateway not running.",
        ],
    },
    "llama_cpp": {
        "version": "0.3.22",
        "core_importable": True,
        "server_importable": False,
        "missing_extras": [
            "starlette_context",
            "pydantic-settings",
            "sse-starlette",
        ],
        "fix_command": (
            "C:/STC/ultronPrototype/.venv/Scripts/pip.exe install "
            "'llama-cpp-python[server]'"
        ),
    },
    "tests": {
        "passing": 699,
        "skipped": 15,
        "failed": 0,
        "note": (
            "From this worktree at HEAD = origin/main + Phase 4 deferred "
            "wrappers (still uncommitted)."
        ),
    },
}


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    data[KEY] = phase0
    PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"appended {KEY} to {PATH}")


if __name__ == "__main__":
    main()
