# OpenClaw Phase 5 close-out — Heartbeat

Heartbeat machinery is in place: the alert log persists locally,
the bridge holder records alerts and fires Telegram pings,
`HEARTBEAT.md` carries the real checklist. Live agent activation
requires user-side OpenClaw config (`agents[].heartbeat` block) —
documented but not auto-applied.

## What landed

| File | Role |
|---|---|
| `~/.openclaw/workspace/HEARTBEAT.md` | Real heartbeat checklist (was a Phase 1 placeholder). Five tasks: coding-queue check, alert log review, disk health, addressing-anomaly check, and a default-OK fallback. Output format documented. |
| `src/ultron/openclaw_bridge/heartbeat_alerts.py` | `HeartbeatAlertLog` (JSONL-backed, atomic update via temp+replace, lock-protected), `HeartbeatAlert` dataclass with severity/ack-state/metadata. |
| `src/ultron/openclaw_bridge/holder.py` | `OpenClawBridge.heartbeat_alerts` field + `record_heartbeat_alert(text, source, severity, ...)` convenience that records and (when enabled) fires `NotificationDispatcher.notify_heartbeat_alert` via the existing `fire_and_forget` helper. |
| `src/ultron/config.py` | `HeartbeatConfig` (alert_log_path, alert_retention_days, auto_notify_telegram). Hung off `UltronConfig.heartbeat`. |
| `config.yaml` | New `heartbeat:` section with retention + auto-notify knobs. |
| `docs/openclaw_heartbeat_setup.md` | User-side setup procedure: editing `~/.openclaw/openclaw.json` to add the `agents[].heartbeat` block (cadence, target=telegram, isolatedSession/lightContext/skipWhenBusy, activeHours window). Smoke-test recipe. Cadence guidance + tasks-block sub-cadence example. |

## Defaults & posture

- `heartbeat.alert_log_path: "logs/heartbeat_alerts.jsonl"` (relative
  to project root via `resolve_path`).
- `heartbeat.alert_retention_days: 30`.
- `heartbeat.auto_notify_telegram: true` — but Telegram delivery
  also requires the master Phase 4 flags
  (`notifications.telegram.enabled`,
  `notifications.telegram.notify_on.heartbeat_alerts`). With the
  user-side bot setup deferred, the auto-notify is a no-op until
  Phase 4 is fully live.
- `record_heartbeat_alert` always records; the Telegram push is
  best-effort + non-blocking (daemon thread via `fire_and_forget`).

## Voice pipeline impact: zero

Phase 5 work is entirely off-hot-path:

- Alert log writes happen on a background thread (when fired by
  the Phase 4 notification path) or from the OpenClaw-side agent
  (which has no shared loop with the voice pipeline).
- Reads happen on demand from MCP tools / voice queries that don't
  fire on every utterance.
- No persistent threads were added in Phase 5 — the existing
  `openclaw-mcp-retry` and `openclaw-notify` daemon threads
  cover all the async needs.

VRAM and TTFT baselines unchanged. No measurement gate fired.

## Tests

| Suite | Count | Notes |
|---|---|---|
| `tests/openclaw_bridge/test_heartbeat_alerts.py` | 17 | Construction, record/read, severity validation, since/unack/limit filtering, acknowledgment, prune, malformed-line tolerance, 20-thread × 10-record concurrency. |
| `tests/openclaw_bridge/test_holder.py` (extended) | 4 new | Bridge has `heartbeat_alerts` field, `record_heartbeat_alert` writes to log, auto-notify fires `notify_heartbeat_alert` when enabled, no notification when flag off. |
| **Total** | **+21** | All pass; existing tests unaffected. |

Full sweep: **1147 passed / 15 skipped / 0 failed** (1162 collected).

## Deferred (intentional)

- **Ultron MCP tools** for `get_heartbeat_alerts(since)` and
  `acknowledge_alert(alert_id)`. These are meaningful only when an
  OpenClaw stdio MCP entrypoint exists, which is itself deferred
  (Phase 3.2 left `mcp_server_command=None`). When that lands,
  adding the tools is straightforward — the alert log API already
  matches the planned MCP shape. Until then, the orchestrator-side
  helpers (via `bridge.heartbeat_alerts`) cover all in-process
  needs.

- **Voice intent for alert query** ("what alerts did you flag?").
  Defer to Phase 8 (standing orders) where there's a natural
  query pattern.

- **OpenClaw-side `agents[].heartbeat` block edit.** Documented in
  `docs/openclaw_heartbeat_setup.md`; user runs this once Telegram
  is live (Phase 4 user-side) and they're ready for periodic
  agent ticks.

## Phase 6 starting state

Phase 6 (Browser tool) builds on the Phase 3 bridge:

- `BROWSER_AUTOMATION` intent kind already in
  `RoutingIntentKind` from Foundation Phase 5.
- `OpenClawDispatcher.handle_browser` is currently a stub; Phase 6
  rewrites it to use `bridge.client.invoke_tool("browser", ...)`
  similar to how `handle_messaging` was rewritten in Phase 4.
- Browser activity takes seconds-to-minutes; the existing
  `acknowledgment_phrases` pattern (used by web search) carries
  over with a small browser-specific phrase pool.
- `BrowserTool` wrapper class needs to be added with
  navigate/snapshot/click/type/screenshot helpers built on top of
  `OpenClawClient.invoke_tool`.

No Phase 5 work blocks Phase 6.
