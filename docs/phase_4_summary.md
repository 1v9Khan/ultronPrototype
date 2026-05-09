# OpenClaw Phase 4 close-out — Telegram channel

Telegram bot is the first remote channel for Ultron. Phase 4 lands
the Ultron-side wiring (notification dispatcher, live MESSAGING
dispatch through the bridge, config schema, docs). Live verification
requires a one-time user setup on the OpenClaw side (BotFather +
`openclaw channels add`).

## What landed

| File | Role |
|---|---|
| `docs/openclaw_telegram_setup.md` | Full bot-creation procedure (BotFather walkthrough), token handling, `openclaw configure`/`channels add` recipe, smoke test checklist, security + rotation notes. |
| `src/ultron/openclaw_bridge/notifications.py` | `NotificationDispatcher` — fire-and-forget proactive Telegram pings on coding-task completion / clarification / heartbeat / standing orders / async search. Per-event opt-ins; fail-open at every step. |
| `src/ultron/openclaw_bridge/holder.py` | `OpenClawBridge.notifications` field; `fire_and_forget(coro_factory)` helper for off-hot-path dispatch from the sync orchestrator loop. `from_config(openclaw_cfg, notifications_cfg=None)` accepts the new config. |
| `src/ultron/openclaw_routing/dispatcher.py` | `OpenClawDispatcher.handle_messaging` now calls `bridge.client.send_message` when a bridge is wired. Falls through to the existing stub voice message when the bridge is absent. Recipient resolution: explicit `intent.recipient` → `notifications.telegram.user_id_env` → `fallback_user_id` → clear voice error. |
| `src/ultron/coding/voice.py` | `CapabilityVoiceController` accepts an `openclaw_bridge` kwarg and threads it into the dispatcher when constructing the on-demand `AutomationTaskRunner`. |
| `src/ultron/pipeline/orchestrator.py` | `_announce_coding_completion_if_pending()` and `_announce_pending_clarifications()` now fire-and-forget a Telegram notification after the voice narration plays. Bridge is constructed before `coding_voice` so the controller can pass the bridge handle through. |
| `src/ultron/config.py` | `NotificationsConfig` + `TelegramNotificationsConfig` + `TelegramNotifyOnConfig`. Hung off the top-level `UltronConfig.notifications`. |
| `config.yaml` | `notifications:` section with master `enabled: false` (default-off until the user sets up the bot), per-event opt-in flags, env-var-based recipient resolution. |

## Defaults & posture

`notifications.telegram.enabled: false` is the default. The bridge
itself defaults to `openclaw.enabled: false` (Phase 3 default).
Every layer fails open:

1. `openclaw.enabled=False` → bridge factory returns `None`; voice
   path identical to pre-Phase-3 behavior.
2. `openclaw.enabled=True` but Gateway unreachable → bridge logs
   WARN, launches retry thread (if MCP command configured), voice
   path unaffected.
3. `notifications.telegram.enabled=False` → dispatcher returns
   `NotificationResult(sent=False, skipped_reason='telegram disabled in config')`;
   no Telegram round-trip.
4. Master `enabled=True` but per-event flag off → same skip,
   different reason.
5. No `TELEGRAM_USER_ID` in env and no `fallback_user_id` → skip
   with clear log line.
6. Transport raises → skip with `transport error` reason.

## Voice pipeline impact: zero

Phase 4 work is entirely off-hot-path. The voice pipeline does NOT
touch any of:

- `NotificationDispatcher` (fired AFTER `_speak()` returns; runs on
  a daemon thread via `fire_and_forget`).
- `OpenClawDispatcher.handle_messaging` (only runs when a MESSAGING
  intent is dispatched, never on the voice path).
- The bridge holder's startup tasks (constructed once, off-loop).

VRAM and TTFT baselines unchanged. No measurement gate fired.

## Tests

| Suite | Count | Notes |
|---|---|---|
| `tests/openclaw_bridge/test_notifications.py` | 15 | All notify_* methods, recipient resolution, master/per-event gating, transport-error fail-open. |
| `tests/openclaw_bridge/test_holder.py` (extended) | 4 new | `notifications` field present, custom config plumbing, `fire_and_forget` runs + swallows exceptions. |
| `tests/routing/test_dispatcher.py` (extended) | 8 new | Live messaging path: dispatches via bridge when wired, falls back to stub when bridge absent, recipient resolution paths, transport-failure voice messages, empty-body rejection. |
| **Total** | **+27** | All pass; existing tests unaffected. |

Full sweep: **1126 passed / 15 skipped / 0 failed** (1141 collected).

## What's user-side

Phase 4 cannot be fully verified without the user running the
BotFather flow and configuring the channel in OpenClaw. The
machine-side wiring is done; the channel side requires:

1. Create bot via `@BotFather` on Telegram.
2. Save token to `TELEGRAM_BOT_TOKEN` env var.
3. Save own Telegram user id to `TELEGRAM_USER_ID` env var.
4. Run `openclaw channels add --channel telegram --token "$env:TELEGRAM_BOT_TOKEN"`
   (or interactive `openclaw configure`).
5. Restart OpenClaw Gateway.
6. Verify with `openclaw channels list`.
7. Flip `notifications.telegram.enabled: true` in `config.yaml`.
8. Smoke test: send "hello" via Telegram, expect in-character
   reply. Trigger a coding task, expect Telegram notification on
   completion.

Procedure is documented in `docs/openclaw_telegram_setup.md`.

## Phase 5 starting state

Phase 5 (Heartbeat) builds on the Phase 4 Telegram channel:
heartbeat alerts ride the same `notify_heartbeat_alert` path. The
`OpenClawClient.trigger_heartbeat` method is already in place from
Phase 3. Phase 5 adds the OpenClaw-side `agents[].heartbeat` config
block, populates `HEARTBEAT.md` with the actual checklist, and
adds an Ultron MCP tool for querying recent alerts (so a voice
"what alerts did you flag?" works).

No Phase 4 work blocks Phase 5.
