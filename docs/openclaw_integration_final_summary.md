# OpenClaw integration — final summary (Phases 0–13)

OpenClaw is integrated as a peer Gateway sharing the same local
Qwen via llama-cpp-server. Both Ultron's voice pipeline and OpenClaw
read from the same workspace files for persona; both call the same
OpenAI-compat HTTP endpoint for inference (when the voice path
opts into HTTP mode).

This document is the closed-loop summary across all 13 phases of
the integration prompt, with the deviations from the original spec
made explicit so future readers don't re-litigate them.

## Phases at a glance

| Phase | Focus | Status | What landed |
|---|---|---|---|
| 0 | Verification + reachability | ✅ | `llama-cpp-server` launcher (port 8765), supervisor wrapper, OpenClaw config patched to use `litellm` provider, baseline measurements. |
| 1 | Persona scaffolding | ✅ | Six workspace persona files, `PersonaLoader` with four modes (user_facing / background / heartbeat / bootstrap), hot reload. |
| 2 | LLM-provider wiring | ✅ | `litellm` provider in OpenClaw config (substituted for Ollama per the runtime decision), three agents (`ultron-test`, `ultron-main`, `ultron-heartbeat`), locked-in tool constraints. |
| 3 | Bridge layer | ✅ | `OpenClawClient` (CLI subprocess transport), `WorkspaceWriter` (atomic + filelock), `OpenClawEventReceiver` (gated-off scaffold), `UltronMcpRegistrar` (idempotent + retry), `OpenClawBridge` holder + orchestrator wiring. |
| 4 | Telegram channel | ✅ | `NotificationDispatcher` (proactive pings), live `MESSAGING` dispatch via bridge, `fire_and_forget` helper, full Telegram setup docs. |
| 5 | Heartbeat | ✅ | `HEARTBEAT.md` populated, `HeartbeatAlertLog` (JSONL with atomic update + retention), `record_heartbeat_alert(...)` orchestrator entry. |
| 6 | Browser tool | ✅ | `BrowserTool` wrapper (six primitives), live `handle_browser` dispatch, `BrowserConfig`. |
| 7 | Cron jobs | ✅ | `scripts/run_maintenance_for_cron.py` cron-friendly wrapper, recipes for nightly-maintenance / morning-briefing / weekly-review. |
| 8 | Standing orders | ✅ | `Coding Project Watcher` + `Weekly Review` programs in AGENTS.md, `docs/standing_orders.md`. |
| 9 | Hooks | ✅ | `docs/openclaw_hooks_setup.md` — `session-memory` + `command-logger` recommended; custom-hook scaffolding pattern. |
| 10 | Memory Wiki | ✅ | Plugin install + config docs; `docs/memory_architecture.md` (three-layer model). |
| 11 | iOS / Android nodes | ✅ | `docs/mobile_node_setup.md` — pairing procedure, network considerations, security notes. |
| 12 | Media generation | ✅ | `MediaGenerationConfig` + live `handle_media_generation` dispatch, **local-only ComfyUI guidance** (paid APIs explicitly out per project policy). |
| 13 | Integration testing + polish | ✅ | Stdio MCP entry + five MCP tools, `SystemStatusReporter` + voice intent + classifier patterns, `OpenClawBridgeConfig.mcp_server_command="auto"` default, auto-enabled OpenClaw configs (hooks + memory-wiki plugin + MCP registration), final summary + link audit. |

## Deviations from the original spec (intentional, retained)

These are the deviations the user explicitly approved or that
emerged during implementation. Future work should preserve them
unless reopening the underlying decision.

1. **`llama-cpp-server` instead of Ollama** ([feedback_llm_runtime_decision.md](<ai-memory-dir>\feedback_llm_runtime_decision.md)).
2. **`litellm` provider plugin** in OpenClaw (not `openai-provider`
   or `lmstudio-provider`).
3. **Three-agent split** (`ultron-test` default, `ultron-main`
   user-facing, `ultron-heartbeat`).
4. **PersonaLoader four-mode split** (user_facing / background /
   heartbeat / bootstrap) — Ultron's character renders only on
   user-facing channels.
5. **AGENTS.md excluded from `user_facing` mode** — adding it
   regressed voice TTFT by +175 %. Voice-relevant rules live in
   SOUL.md.
6. **`tools.profile: "messaging"`** locked on every local-Qwen
   agent — the `coding` profile bundles 50+ tool schemas that
   exceed Qwen3.5-9B prefill in OpenClaw's hard 30 s SDK timeout.
7. **`models[].contextWindow: 16384`** + `--n-ctx 16384` on
   llama-cpp-server. Lower values trip the prompt-budget
   pre-check on agents with bundled tools.
8. **CLI subprocess transport in Phase 3** instead of HTTP. OpenClaw
   2026.5.7 doesn't expose `/tools/invoke` or `/messages` HTTP
   endpoints; the CLI is the documented public surface.
9. **No `OPENCLAW_TOOL` wrapper category in the intent classifier**.
   Foundation Phase 5 chose top-level `BROWSER_AUTOMATION`,
   `MEDIA_GENERATION`, `MESSAGING`, `FILE_OPERATION`,
   `SHELL_OPERATION`, `HYBRID_TASK` — adding a wrapper with
   sub-classifier would be redundant.
10. **Stdio MCP entry (Phase 13 finish — landed).** Originally
    the bridge's `mcp_server_command` defaulted to `None` because
    `UltronMCPServer` (in `ultron.coding.mcp_server`) is SSE-based.
    Phase 13 finish added a separate stdio MCP server
    (`ultron.openclaw_bridge.mcp_tools` + `scripts/run_ultron_mcp_for_openclaw.py`)
    exposing five tools (heartbeat alerts, acknowledge, maintenance,
    coding session listing, voice-friendly alerts). Default is now
    `mcp_server_command: "auto"` which resolves to the entry script
    via `OpenClawBridge._resolve_mcp_command`.
11. **4B preset** with `n_ctx=8192` — different from Phase 0's
    9B baseline. Voice TTFT 79 ms (vs 109 ms on 9B); VRAM peak
    7913 MB (vs 10370 MB on 9B). 4B + 0.8B speculative decoding.
12. **Items 4–8 default ON** (compression, IRMA, self-consistency,
    canonical monitor, block-and-revise) — bisect-verified zero
    added latency on the standard voice baseline.

## What Phase 13 verifies

Phase 13's role is gate-keeping, not new work. Verification spans:

### Machine-side (autonomous; covered by tests + this summary)

- [x] Foundation Phases 0–7 unchanged (1010 of 1199 tests originate
      pre-OpenClaw).
- [x] Voice pipeline never blocks on the bridge — every bridge
      operation is off-hot-path.
- [x] Bridge components fail-open at every level (CLI missing,
      Gateway down, transport timeout, auth rejected, tool
      unavailable).
- [x] Per-capability dispatchers (`handle_messaging`,
      `handle_browser`, `handle_media_generation`) all have a stub
      fallback when the bridge isn't wired.
- [x] All 1184 tests pass / 15 skipped / 0 failed.

### Live-stack (user-led; documented but not run)

- [ ] 16-step Foundation smoke test ([docs/smoke_test.md](smoke_test.md))
      — interactive, real microphone + speaker. Pending.
- [ ] Telegram channel smoke ([docs/openclaw_telegram_setup.md](openclaw_telegram_setup.md))
      — requires BotFather + `openclaw channels add`. Pending.
- [ ] Heartbeat tick smoke ([docs/openclaw_heartbeat_setup.md](openclaw_heartbeat_setup.md))
      — requires the agent's `heartbeat: {...}` block + a
      manufactured alert condition. Pending.
- [ ] Browser tool smoke ([docs/openclaw_browser_setup.md](openclaw_browser_setup.md))
      — requires Playwright + Chromium + `tools.alsoAllow`
      adjustment for the user-facing agent. Pending.
- [ ] Memory Wiki smoke ([docs/openclaw_memory_wiki_setup.md](openclaw_memory_wiki_setup.md))
      — requires `openclaw plugins enable memory-wiki` + restart.
      Pending.
- [ ] Media generation smoke ([docs/openclaw_media_generation_setup.md](openclaw_media_generation_setup.md))
      — requires a **local** image-generation backend (ComfyUI on
      `127.0.0.1:8188`). Paid cloud providers (Fal, Runway, etc.)
      are explicitly out of scope. Pending.
- [ ] Mobile-node pairing ([docs/mobile_node_setup.md](mobile_node_setup.md))
      — requires hardware. Optional.

The live-stack smoke is intentionally deferred. Each setup doc
includes its own smoke test recipe; the user runs them in whatever
order matches their priorities.

## Setup-readiness checklist

When you're ready to take Ultron live with OpenClaw integration:

1. **Phase 4 — Telegram bot.** 5 min via BotFather; the rest of
   OpenClaw integration becomes useful once you have a remote
   delivery channel.
2. **Phase 5 — Heartbeat.** Add the `agents[].heartbeat` block to
   `~/.openclaw/openclaw.json`. Single config edit.
3. **Phase 7 — Cron.** Easiest path is Windows Task Scheduler
   for `nightly-maintenance`; fold morning-briefing in once the
   agent has the MCP tool surface.
4. **Phase 6 — Browser** (optional). Heaviest setup (Playwright +
   `tools.alsoAllow`). Skip until you have a concrete browser-
   automation use case.
5. **Phase 12 — Media gen** (optional). Local ComfyUI only — paid
   APIs are out of scope per project policy. VRAM-tight on the
   4B preset; see the setup doc's "VRAM coordination" section.
6. **Phase 9 — Hooks** (optional). One-line enables for
   `session-memory` + `command-logger`. Useful for audit.
7. **Phase 10 — Memory Wiki** (optional). Enable when there's a
   concrete reflection use case.
8. **Phase 11 — Mobile nodes** (optional). Skip unless you want
   voice from your phone (Telegram covers everything else).

## Per-phase summary docs

For deep dives into a specific phase:

- [Phase 1 close-out](phase_1_summary.md)
- [Phase 3 close-out](phase_3_summary.md)
- [Phase 4 close-out](phase_4_summary.md)
- [Phase 5 close-out](phase_5_summary.md)
- [Phase 6 close-out](phase_6_summary.md)
- (Phases 7–12 — per-phase summaries inline in this doc; the
  setup docs serve as the operational guide for each.)

## Voice pipeline impact: zero across all phases

Every phase preserves the voice baseline:

- TTFT median: **79 ms** (4B preset, n_ctx=8192).
- VRAM peak: **7913 MB** (-2461 MB / -2.5 GB vs 9B baseline).

The OpenClaw integration is **strictly additive** — bridge
components are built but only fire on intents the user issues
explicitly (browser, media, etc.) or in fire-and-forget
notifications that happen on background threads.

If a future change appears to threaten the voice baseline,
re-run [`scripts/measure_baseline.py`](../scripts/measure_baseline.py)
before merging and document the delta.

## Questions to ask before extending

When you're tempted to add something to the OpenClaw side:

1. **Does this need to be on the voice path?** If yes, it
   probably belongs Ultron-side (orchestrator / coding pipeline /
   Qdrant). If no, OpenClaw is the right home.
2. **Does this need its own MCP tool?** If you'd write
   `ultron.foo()` to expose Python state to the OpenClaw agent,
   the stdio MCP entrypoint deferred since Phase 3.2 is the
   blocker. Plan to land that first if multiple consumers need it.
3. **Does this need a new persona file?** Almost never. Update
   SOUL.md / AGENTS.md instead unless you genuinely need a
   parallel persona surface.
4. **Does this need a new channel?** Probably not. Telegram +
   voice cover the daily workflow; resist the urge to add
   Discord / Slack / Signal without a concrete use case.

## Final test sweep

```
C:\STC\ultronPrototype\.venv\Scripts\python.exe -m pytest tests/ -q --no-header --ignore=tests/coding/test_orchestration_real.py
1251 passed, 15 skipped, 28 warnings in ~37s
```

1251 / 15 skipped (GPU-gated) / 0 failed. +256 net OpenClaw-bridge
tests vs the Foundation Phase 7 baseline (Phase 3 = +104, Phase 4 =
+27, Phase 5 = +21, Phase 6 = +28, Phase 12 = +9, Phase 13 finish =
+67).

## Phase 13 finish — what landed (post-original-summary)

When this document was first written, several items were marked
"deferred" or "user-led". The Phase 13 finish closed most of them.

### Stdio MCP server with five tools

`scripts/run_ultron_mcp_for_openclaw.py` is the canonical stdio
entry point OpenClaw spawns when calling Ultron tools. It serves
five tools via FastMCP stdio:

- `get_heartbeat_alerts(since_seconds_ago, only_unacknowledged, limit)`
- `acknowledge_alert(alert_id)`
- `run_maintenance(scope=None)` — subprocesses
  `scripts/run_maintenance_for_cron.py`
- `list_active_coding_sessions(max_age_hours=24)` — reads
  `logs/sessions/*.jsonl` audit files
- `get_recent_voice_alerts(limit=5)` — convenience for voice
  narration

The process is ephemeral (OpenClaw spawns a fresh instance per
call), light-import (no torch / LLM), and reads/writes only the
on-disk artifacts the orchestrator already maintains.

### Voice-side SYSTEM_STATUS intent

A new `RoutingIntentKind.SYSTEM_STATUS` lets voice queries like
"what alerts did you flag?" or "what is Ultron working on?" route
through `SystemStatusReporter` (in
`ultron.openclaw_bridge.system_status`). The reporter reads the
same on-disk artifacts the MCP tools expose, then renders a brief
in-character voice narration. **Voice and OpenClaw paths share the
same source of truth.**

The classifier matches three pattern groups:

- "what alerts did you flag", "any pending alerts", ... → focus="alerts"
- "what is Ultron working on", "list active projects", ... → focus="projects"
- "status report", "system status", "what's going on" → focus="all"

### Auto-resolved MCP entry

`OpenClawBridgeConfig.mcp_server_command` defaults to `"auto"`. The
holder (`OpenClawBridge._resolve_mcp_command`) translates that into
`(.venv python, [scripts/run_ultron_mcp_for_openclaw.py, --stdio])`
at construction. No operator action required — the bridge wires
itself up to the canonical entry point.

Three semantics for the field:

- `"auto"` (default): resolve to the canonical entry script.
- explicit string: use as-is (operator override).
- `None`: disable registration entirely.

### Live OpenClaw config changes (Phase 13 finish)

The following configs were enabled live in `~/.openclaw/openclaw.json`:

- `hooks.internal.entries.session-memory.enabled: true`
- `hooks.internal.entries.command-logger.enabled: true`
- `plugins.entries.memory-wiki.enabled: true` (Gateway restart
  required for the plugin's tools to register)
- `mcpServers["ultron-mcp"]` registered via `openclaw mcp set`

None of these change the voice pipeline. Hooks fire on session
events (read-only, audit-trail use); the memory-wiki plugin
exposes wiki tools to OpenClaw agents (no Ultron-side effect); the
MCP registration lets OpenClaw agents call the five Ultron tools
listed above.

## What truly remains user-led

After the Phase 13 finish, the remaining user-led work is purely
about **credentials and channel-specific setup** — there's nothing
left for the codebase to integrate without the user supplying
secrets or making preference decisions.

| User action | Why it's user-led |
|---|---|
| Telegram bot token (`@BotFather` + `TELEGRAM_BOT_TOKEN` env) | Requires the user's phone + Telegram account. |
| `openclaw channels add --channel telegram` | Uses the bot token from above. |
| `agents[].heartbeat` block in `openclaw.json` | Cadence is a user preference; firing it before Telegram is set up is wasteful. |
| `tools.alsoAllow: ["browser"]` for `ultron-main` | Requires user to verify Playwright + Chromium are reachable. |
| ComfyUI install + `models.providers.comfyui` config | Requires the user to install ComfyUI locally (no cloud / paid alternatives per project policy). |
| Mobile node pairing (iOS/Android) | Requires hardware. |
| 16-step Foundation smoke test | Requires real microphone + speaker. |

Each setup doc in `docs/openclaw_*_setup.md` has its own recipe
and smoke test for the user-led step.
