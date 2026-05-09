# OpenClaw Phase 6 close-out — Browser tool

`BROWSER_AUTOMATION` intents now route through OpenClaw's bundled
browser plugin. The dispatcher dispatches to a thin `BrowserTool`
wrapper that maps each intent action (navigate / snapshot / click /
type / screenshot) to a structured agent prompt and unpacks the
result into a typed dataclass.

## What landed

| File | Role |
|---|---|
| `src/ultron/openclaw_bridge/browser.py` | `BrowserTool` wrapper with six primitives. Each method calls `OpenClawClient.invoke_tool("browser", {...})` with action-specific params. Result parsing is best-effort: title extraction from text, ref extraction from snapshot output, base64 decoding for screenshots. |
| `src/ultron/openclaw_routing/dispatcher.py` | `handle_browser` rewritten — bridge-wired live path mapping `BrowserIntent.action` to the appropriate `BrowserTool` method. Falls back to stub when bridge absent OR `browser.enabled: false`. |
| `src/ultron/config.py` | `BrowserConfig` schema (master enabled flag, snapshot mode, per-action timeouts, ack-phrase pool). |
| `config.yaml` | `browser:` section with sensible defaults. |
| `docs/openclaw_browser_setup.md` | OpenClaw-side setup (Playwright + Chromium), tool-deny-list adjustment for `ultron-main` to allow browser, smoke test, voice ack pattern explanation, troubleshooting + security notes. |

## Key design decisions

- **Each action → one agent turn.** Multi-step flows (login →
  navigate → fill form) stay on the OpenClaw side via the agent's
  reasoning, not orchestrated from Python. The wrapper is for
  discrete operations Ultron's intent dispatch fires.

- **Best-effort result parsing.** Free-form agent text is parsed
  with tolerant heuristics (Title: prefix, [refId] label lines,
  base64 markers). Parsing failure degrades to passing the raw
  text through; callers see a `success=True` result with empty
  structured fields rather than spurious failures.

- **Master `browser.enabled` flag.** Lets the operator suppress
  browser dispatch without disabling the entire bridge. Useful
  while debugging browser issues without breaking other
  capabilities.

- **No additional persistent state.** Browser tool calls are one-off;
  no session pooling, no tab tracking. OpenClaw owns the active
  browser instance.

## Defaults & posture

- `browser.enabled: true` — but only fires when the bridge is wired
  AND `openclaw.enabled: true`. With `openclaw.enabled: false` (the
  current default), the dispatcher returns the stub voice message
  exactly as in Phase 5.
- `default_navigation_timeout_seconds: 30.0`, action: 10.0,
  screenshot: 30.0. Conservative; can be tightened per workload.
- `acknowledgment_phrases` — four ack phrases, mirrors web-search
  pattern.

## Voice pipeline impact: zero

Browser dispatch fires only on `BROWSER_AUTOMATION` intents (which
the classifier produces only when the user explicitly asks Ultron
to drive the browser). When fired, the call is async and runs in a
background task; the orchestrator plays an ack phrase within ~200
ms and returns to the listening loop. No Phase 6 code runs on the
voice hot path.

VRAM and TTFT baselines unchanged. No measurement gate fired for
Phase 6.

## Tests

| Suite | Count | Notes |
|---|---|---|
| `tests/openclaw_bridge/test_browser.py` | 19 | All six `BrowserTool` methods, edge cases (empty url/ref/text), title + ref + base64 extraction, error translation from `OpenClawToolError`. |
| `tests/routing/test_dispatcher.py` (extended) | 9 new | Live browser path: navigate via bridge, fall back when no bridge, fall back when `browser.enabled=false`, missing-url rejection, screenshot, click without target, unknown-action voice message, transport exception. |
| **Total** | **+28** | All pass; existing tests unaffected. |

Full sweep: **1175 passed / 15 skipped / 0 failed** (1190 collected).

## What requires user-side action (not blocking)

- **OpenClaw browser plugin readiness.** `openclaw doctor` should
  flag missing Playwright / Chromium; user runs
  `npx playwright install chromium` once. Documented in
  `docs/openclaw_browser_setup.md`.
- **Tool-deny adjustment.** Phase 0 locked `tools.profile: messaging`
  + explicit `tools.deny` on every local-Qwen agent. To enable
  browser dispatch on `ultron-main`, the user adds
  `tools.alsoAllow: ["browser"]` to that agent's config and
  restarts the Gateway. Documented; not auto-applied.

## Phase 7 starting state

Phase 7 (Cron jobs) is largely about OpenClaw-side configuration:

- `openclaw cron add` for nightly maintenance + morning briefing.
- A new Ultron MCP tool `ultron.run_maintenance(scope)` that runs
  the existing `scripts/maintenance.py` operations.
- Wire alerts from cron output to existing `NotificationDispatcher.notify_standing_order_output`.

The MCP tool addition is meaningful only when an OpenClaw stdio
entrypoint exists (deferred since Phase 3.2). Phase 7 can land the
config recipe + maintenance helper, with the actual MCP tool added
when the stdio entrypoint comes in (or via a thin CLI wrapper that
the cron prompt invokes directly).

No Phase 6 work blocks Phase 7.
