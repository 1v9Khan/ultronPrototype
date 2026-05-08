"""One-shot helper to merge Phase 0 OpenClaw integration measurements
into baselines.json after a measure_baseline.py overwrite.

Reads:
  - ``baselines.json``                    (fresh metadata/vram_mb/latency_ms)
  - ``baselines.json.pre-phase0-rerun-bak`` (preserves phase_foundation_start)

Writes:
  - ``baselines.json`` with both blocks plus a fresh
    ``phase_0_openclaw_integration`` populated from this turn's
    interactive verification.

Run from the worktree root:
    python scripts/_merge_phase0_baselines.py
"""

from __future__ import annotations

import datetime
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRESH = ROOT / "baselines.json"
BACKUP = ROOT / "baselines.json.pre-phase0-rerun-bak"


def main() -> None:
    fresh = json.loads(FRESH.read_text(encoding="utf-8"))
    backup = json.loads(BACKUP.read_text(encoding="utf-8"))

    # Compute first-token P50 / P95 from the fresh per-query latencies.
    ttft = [q["first_token_ms"] for q in fresh["latency_ms"]["per_query"]]
    p50_ttft = statistics.median(ttft)
    p95_ttft = sorted(ttft)[int(0.95 * (len(ttft) - 1))]

    # Restore phase_foundation_start.
    if "phase_foundation_start" in backup:
        fresh["phase_foundation_start"] = backup["phase_foundation_start"]

    # Fresh phase_0_openclaw_integration with real measurements.
    fresh["phase_0_openclaw_integration"] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "openclaw_integration_phase_0",
        "outcome": "partial_pass",
        "summary": (
            "llama-cpp-server reachable and serving inference; voice "
            "pipeline baseline matches Foundation phase. OpenClaw can "
            "reach the local LLM and runs inference (VRAM peak 9082 MB). "
            "Final response-format compatibility with Qwen3.5's <think> "
            "scaffolding when invoked through OpenClaw's agent runner is "
            "deferred to Phase 1 along with replacing stock workspace "
            "persona files."
        ),
        "vram_idle_mb": fresh["vram_mb"]["before_load"],
        "vram_during_voice_query_mb": fresh["vram_mb"]["peak_under_load"],
        "vram_during_openclaw_turn_mb": 9082,
        "vram_resident_model_only_mb": 9104,
        "first_token_latency_ms_p50": p50_ttft,
        "first_token_latency_ms_p95": p95_ttft,
        "llama_cpp_server_direct_wall_ms": 463,
        "llama_cpp_server_direct_prompt_tokens": 38,
        "llama_cpp_server_direct_completion_tokens": 10,
        "openclaw": {
            "version": "2026.5.7",
            "build": "eeef486",
            "config_file": r"C:\Users\alecf\.openclaw\openclaw.json",
            "workspace_dir": r"C:\Users\alecf\.openclaw\workspace",
            "gateway_url": "ws://127.0.0.1:18789",
            "gateway_mode": "local",
            "channels_configured": [],
            "mcp_servers_configured": [],
            "provider_configured": "litellm",
            "provider_baseUrl": "http://127.0.0.1:8765/v1",
            "provider_api": "openai-completions",
            "model_id": "litellm/qwen3.5-9b-local",
            "model_reasoning": True,
            "test_agent_id": "ultron-test",
            "test_agent_tools_profile": "messaging",
            "default_agent_id": "ultron-test",
            "plugins_enabled": ["litellm"],
            "config_backups": [
                "openclaw.json.pre-llamacpp-bak",
                "openclaw.json.pre-test-agent-bak",
                "openclaw.json.pre-litellm-bak",
            ],
        },
        "llama_cpp_server": {
            "host": "127.0.0.1",
            "port": 8765,
            "api_path_prefix": "/v1",
            "api_key": "local-ultron",
            "launcher": "scripts/start_llamacpp_server.py",
            "model": "models/Qwen3.5-9B-Q4_K_M.gguf",
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "flash_attn": True,
            "type_k_v": 8,
            "model_alias": "qwen3.5-9b-local",
        },
        "deferred_to_phase_1_or_later": [
            "Replace stock workspace persona files (SOUL.md, AGENTS.md, "
            "IDENTITY.md, USER.md, HEARTBEAT.md, BOOTSTRAP.md) with content "
            "migrated from config.yaml:llm.system_prompt — the stock "
            "OpenClaw boilerplate confuses Qwen3.5 about its role.",

            "OpenClaw response-format with Qwen3.5's <think>...</think> "
            "scaffolding produces empty visible content (model emits "
            "thinking only). Need to either (a) configure OpenClaw to "
            "treat the think block as reasoning_content and continue, "
            "or (b) override server-side chat_format to plain chatml "
            "(may shift voice character; verify first).",

            "OpenClaw Gateway became unstable after multiple failed agent "
            "runs (1006 abnormal closures). Need a clean Gateway restart "
            "procedure documented + maybe a watchdog. The user's Gateway "
            "should be Ctrl+C'd and restarted via gateway.cmd before "
            "Phase 1 work begins.",

            "llama-cpp-server crashed once during the OpenClaw retry "
            "storm (VRAM dropped to idle, /v1/models returned HTTP 000). "
            "Cause unconfirmed. Phase 1 should monitor the server for "
            "stability under OpenClaw's request patterns; consider "
            "wrapping the server in a supervisor (e.g. NSSM service) "
            "with auto-restart.",

            "OpenAI-compat client SDK in OpenClaw caps single-request "
            "timeout at 30 s (dist/client-DZ1aRkVL.js:257). With a full "
            "tool bundle that exceeds Qwen3.5-9B prefill time. Mitigated "
            "via tools.profile='messaging' on the test agent; should be "
            "preserved as the default for any agent that uses the local "
            "Qwen until prompt-budget work in a later phase.",

            "Voice pipeline currently still uses llama-cpp-python "
            "in-process (config.yaml: llm.provider='llama_cpp'). Migrating "
            "it to HTTP-client of llama-cpp-server is the actual sharing "
            "deliverable — Phase 0 only proved the server-side path "
            "works. Migration has its own latency-regression-test gate.",
        ],
        "test_results": {
            "ultron_test_suite": {
                "passed": 699,
                "skipped": 15,
                "failed": 0,
                "note": (
                    "Snapshot from previous turn before any Phase 0 "
                    "interactive work. Re-run as part of Phase 0 close-out."
                ),
            },
            "llama_cpp_server_direct": {
                "models_endpoint": "ok",
                "chat_completion_tiny_prompt_wall_ms": 463,
                "completion_text": "OPENCLAW-LLAMACPP-OK",
            },
            "openclaw_agent_test": {
                "outcome": "incomplete_terminal_response",
                "reached_server": True,
                "ran_inference": True,
                "vram_peak_mb": 9082,
                "blocker": (
                    "Empty visible content in agent response. "
                    "Documented as a Phase 1 deferred item."
                ),
            },
        },
    }

    FRESH.write_text(
        json.dumps(fresh, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"merged baselines.json — keys: {list(fresh.keys())}")


if __name__ == "__main__":
    main()
