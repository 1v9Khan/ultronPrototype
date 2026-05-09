# OpenClaw Phase 3 close-out

Bridge layer is complete. Five new modules + orchestrator wiring +
104 new tests, zero regressions, voice baseline unchanged.

## What landed

| File | Role |
|---|---|
| `src/ultron/openclaw_bridge/client.py` | `OpenClawClient` — async client over the `openclaw` CLI. Public surface: `health`, `send_message`, `trigger_heartbeat`, `run_agent`, `invoke_tool`, `mcp_set/show/list/unset`. |
| `src/ultron/openclaw_bridge/workspace.py` | `WorkspaceWriter` — atomic writes (`os.replace`) + advisory lockfiles (`filelock`) for `MEMORY.md`, `USER.md`, daily memory files. |
| `src/ultron/openclaw_bridge/events.py` | `OpenClawEventReceiver` — gated-off scaffold for the `[voice]`-prefix inbound handoff. Pure prefix/payload helpers locked down by tests. |
| `src/ultron/openclaw_bridge/mcp_registration.py` | `UltronMcpRegistrar` — idempotent registration via `openclaw mcp set` with `schedule_retry()` for background recovery. |
| `src/ultron/openclaw_bridge/holder.py` | `OpenClawBridge` — orchestrator-owned holder. Probes the Gateway, runs MCP registration, launches a daemon retry thread when needed. |
| `src/ultron/pipeline/orchestrator.py` | `_load_openclaw_bridge_if_enabled()` factory + `shutdown()` cleanup hook. |
| `src/ultron/config.py` | `OpenClawBridgeConfig` sub-model on `OpenClawConfig`. |
| `config.yaml` | `openclaw.bridge` subsection with twelve fields; defaults preserve fail-open posture. |

## Deviations from the integration spec

Both intentional, documented in module docstrings.

1. **CLI subprocess transport, not HTTP.** OpenClaw 2026.5.7 doesn't
   expose `/tools/invoke` or `/messages` HTTP endpoints. The
   `openclaw` CLI is the documented public surface and the only
   stable contract. Client methods invoke the CLI via
   `asyncio.create_subprocess_exec`. The auth token is read from
   `~/.openclaw/openclaw.json` by the CLI itself; we never inject
   it into args or environment.

2. **No `OPENCLAW_TOOL` wrapper category.** Foundation Phase 5
   already chose the cleaner architecture: top-level
   `BROWSER_AUTOMATION`, `MEDIA_GENERATION`, `MESSAGING`,
   `FILE_OPERATION`, `SHELL_OPERATION`, `HYBRID_TASK` in
   `RoutingIntentKind`. Adding a wrapper with sub-classifier
   would be redundant — the existing kinds are the OpenClaw tool
   routes.

## Fail-open contract

The voice pipeline NEVER touches the bridge. Bridge calls fire only
when an OpenClaw-bound intent activates (Phases 4+). When the
Gateway is unreachable:

- Construction succeeds — components still build, `client=None`
  if the CLI is missing.
- `OpenClawBridge.start()` logs a clear WARN and proceeds; if a
  stdio MCP command is configured, a daemon retry thread is
  launched (`openclaw-mcp-retry`).
- `OpenClawClient` methods return result dataclasses with `error`
  set rather than raising. Auth failures (401/403) raise
  `OpenClawAuthError`; transport failures raise
  `OpenClawGatewayError`.
- Shutdown joins the retry thread (≤2 s) and stops the receiver.
  The MCP entry stays registered so OpenClaw can spawn Ultron's
  MCP across restarts.

## Verification

| Criterion | Status |
|---|---|
| All Phase 0–2 work intact | ✅ Persona files, llama-cpp-server config, three-agent split, `tools.profile: messaging`, lifecycle module unchanged. |
| `OpenClawClient` constructs and runs CLI subprocesses | ✅ 35 unit tests + 4 real-subprocess integration tests. |
| `WorkspaceWriter` serialises concurrent writers | ✅ 4 threads × 20 entries land 80 entries intact under `filelock`. |
| `OpenClawEventReceiver` prefix logic locked down | ✅ 15 tests covering enabled/disabled, prefix match cases, dispatch error swallowing. |
| `UltronMcpRegistrar` idempotent + fail-open | ✅ 16 tests covering match-existing skip, transient retry, give-up at max_attempts, auth handling. |
| `OpenClawBridge` holder lifecycle | ✅ 10 tests covering construction with/without CLI, start with reachable/unreachable Gateway, retry thread launch + clean shutdown. |
| Orchestrator startup with `openclaw.enabled=False` | ✅ Default. Bridge factory returns `None`; no behavior change. |
| Orchestrator startup with `openclaw.enabled=True` + Gateway down | ✅ Bridge constructs, logs WARN, launches retry thread (when MCP command configured), voice pipeline unaffected. |
| All existing tests pass | ✅ 1099 passed / 15 skipped / 0 failed (1114 collected). +104 vs Phase 3 start. |
| Voice baseline unchanged | ✅ Phase 3 work is entirely off-hot-path; no measurement gate fired. TTFT median 79 ms, VRAM peak 7913 MB unchanged. |
| Auth token never logged | ✅ `_read_token` reused from `lifecycle.py`; never passed via args/env to subprocess. |

## Locked-in constraints (re-verified)

The Phase 0 + Phase 1 constraints carry forward unchanged:

1. `tools.profile: "messaging"` on every local-Qwen agent.
2. Explicit `tools.deny: ["group:web", "group:fs", ...]` belt-and-braces.
3. `models[].contextWindow: 16384` matched by `--n-ctx 16384` on
   llama-cpp-server.
4. `models[].reasoning: true`.
5. `api: "openai-completions"` on the litellm provider.
6. Don't disable `tools.deny` for `ultron-main` — the messaging
   profile stays in place until Phase 6 wires browser/file tools
   explicitly.

## Phase 4 starting state

Phase 4 (Telegram channel) is the natural next chunk. The bridge
infrastructure is ready:

- `OpenClawClient.send_message(channel, target, text)` is wired and
  unit-tested. Phase 4 just needs the channel configured in
  OpenClaw and a known target user id.
- `MESSAGING` intent kind is already in `RoutingIntentKind` from
  Foundation Phase 5; the dispatcher's `handle_messaging` stub
  becomes the new wiring site.
- `notifications.telegram` config schema needs to be added so
  proactive notifications (coding-task completion, etc.) have a
  knob.

No Phase 3 work blocks Phase 4. Phase 4 also gives us the first
real measurement gate for bridge transport — until then, the
`send_message` path is exercised only by the stub-CLI integration
tests.
