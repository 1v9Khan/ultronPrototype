# Heartbeat setup

Phase 5 of the OpenClaw integration. Heartbeat is OpenClaw's
periodic background agent — it runs a tick every N minutes / hours
against a small checklist (defined in
`~/.openclaw/workspace/HEARTBEAT.md`) and surfaces anything that
needs the user's attention.

Until Phase 5 the `ultron-heartbeat` agent in OpenClaw was a
placeholder that always replied `HEARTBEAT_OK`. Phase 5 makes the
checklist real and wires the alert log so:

1. Alerts get persisted locally (`logs/heartbeat_alerts.jsonl`).
2. The agent reads recent alerts on the next tick to avoid
   re-surfacing already-acknowledged items.
3. Voice queries (later phases) can pull from the same log.

## What's done autonomously (Phase 5)

- `~/.openclaw/workspace/HEARTBEAT.md` — populated with the real
  checklist (was placeholder in Phase 1).
- `src/ultron/openclaw_bridge/heartbeat_alerts.py` —
  `HeartbeatAlertLog` (JSONL with atomic update + retention).
- `src/ultron/openclaw_bridge/holder.py` —
  `OpenClawBridge.heartbeat_alerts` field +
  `record_heartbeat_alert(text, source, severity, ...)` convenience
  that records and (optionally) fires a Telegram notification via
  `NotificationDispatcher.notify_heartbeat_alert`.
- `config.yaml` `heartbeat:` section with retention + auto-notify
  knobs.
- Telegram delivery path already in place from Phase 4 — the
  alert text rides through `notifications.telegram.notify_on.heartbeat_alerts`
  per-event gating.

## What requires user-side OpenClaw configuration

The `agents.list[].heartbeat` block in `~/.openclaw/openclaw.json`
controls when the heartbeat fires and where it delivers its output.
Run this only when:

- The Telegram channel is configured (Phase 4 user-side step).
- `TELEGRAM_USER_ID` and `USER_TIMEZONE` are set in your env.
- You're ready for the Gateway to start firing periodic agent turns
  against your local Qwen.

### Recipe

Edit `~/.openclaw/openclaw.json`. Find the `ultron-heartbeat` entry
in `agents.list[]` (already created in Phase 0). Add a `heartbeat`
sub-object:

```json5
{
  agents: {
    list: [
      // ...
      {
        id: "ultron-heartbeat",
        // ... existing fields stay ...
        heartbeat: {
          // Tick cadence. Start conservative.
          every: "1h",

          // Where alerts go. "telegram" requires the channel set up in Phase 4.
          target: "telegram",
          to: "${TELEGRAM_USER_ID}",

          // Each tick runs in a fresh session. Critical for token
          // efficiency — without this every tick re-loads the full
          // workspace + previous heartbeat outputs.
          isolatedSession: true,

          // Only HEARTBEAT.md is injected from bootstrap files.
          // SOUL.md still loads (the agent's systemPromptOverride is
          // a separate path, not affected by lightContext).
          lightContext: true,

          // Skip if Ollama / llama-cpp-server is busy serving a
          // voice query. Important for shared local model.
          skipWhenBusy: true,

          // Quiet hours. No heartbeat outside this window in your
          // local timezone.
          activeHours: {
            start: "09:00",
            end: "23:00",
            timezone: "${USER_TIMEZONE}"
          },

          // Don't deliver the literal "HEARTBEAT_OK" no-op response
          // to Telegram. Only deliver actual alerts.
          showOk: false,

          // Truncate long alerts. Telegram's hard limit is ~4096
          // chars; 300 keeps notifications glanceable.
          ackMaxChars: 300
        }
      }
    ]
  }
}
```

### Smoke test

Restart the Gateway, then trigger a tick manually:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" system event `
    --text "Manual heartbeat trigger" --mode now
```

Expected:

- If your checklist finds nothing to report, no Telegram message
  arrives (the `showOk: false` flag suppresses HEARTBEAT_OK).
- If you've manufactured a condition that should alert (e.g.
  start a coding task and let it complete just before triggering),
  the alert lands in Telegram within a few seconds.

Verify the alert log:

```powershell
Get-Content C:\STC\ultronPrototype\logs\heartbeat_alerts.jsonl | Select-Object -Last 5
```

You should see one JSON line per alert with `alert_id`, `text`,
`source`, `severity`, `timestamp`, and (initially) `acknowledged_at: null`.

## Configuration reference

`config.yaml` knobs:

```yaml
heartbeat:
  alert_log_path: "logs/heartbeat_alerts.jsonl"
  alert_retention_days: 30          # prune entries older than N days
  auto_notify_telegram: true        # record + fire Telegram notification
```

Per-event delivery gating still lives under `notifications.telegram.notify_on.heartbeat_alerts`.
Set that to `false` to record alerts to the log without sending
them to Telegram.

## Cadence guidance

`every: "1h"` is a reasonable default. Faster cadences burn more
local Qwen compute. If you find heartbeat is firing too often:

- Move `every` to `"3h"` or `"6h"` for non-urgent monitoring.
- Use the `tasks:` block in `HEARTBEAT.md` to define sub-cadences:
  high-frequency for coding-queue checks, low-frequency for disk
  health.

```markdown
# Inside HEARTBEAT.md

tasks:
  - name: coding-queue
    interval: 30m
    prompt: "Check active coding tasks. Surface completions or
             stuck clarifications."

  - name: disk-health
    interval: 6h
    prompt: "Check workspace + qdrant_data disk usage. Alert if
             > 85% full."

# Default per-tick behaviour (the rest of HEARTBEAT.md)
```

OpenClaw runs each `tasks:` entry only when its interval has
elapsed — keeps tick cost bounded.

## Troubleshooting

- **`HEARTBEAT_OK` keeps arriving in Telegram** — set
  `heartbeat.showOk: false` in the agent config. The Gateway
  default is `true`, which delivers the no-op response.

- **Heartbeat doesn't fire** — confirm Gateway is running
  (`openclaw status --json`). If Gateway is healthy but no ticks,
  check `activeHours` — heartbeat is silent outside the window.
  Run `openclaw cron list` to see if a competing cron is taking
  the tick slot.

- **VRAM spikes during a tick** — heartbeat uses Ollama / llama-cpp-server
  (same instance the voice path uses). With `isolatedSession: true`
  + `lightContext: true` each tick allocates ~1-2 GB transient
  VRAM. Verify peak stays under 11.5 GB; if not, drop `every` to
  a lower frequency or reduce HEARTBEAT.md size.

- **Voice queries get queued behind a tick** — `skipWhenBusy: true`
  prevents heartbeat from queuing during voice queries. The reverse
  isn't enforced; if a tick is already running and a voice query
  arrives, the voice query waits a few seconds for the tick to
  finish. Acceptable trade-off for the prototype.

- **Alert log keeps growing** — `alert_retention_days` (default 30)
  bounds the log size. Pruning is manual; call
  `bridge.heartbeat_alerts.prune()` from a script or wait for a
  later phase to add it to the maintenance cron.
