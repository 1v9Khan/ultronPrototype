# Cron jobs setup

Phase 7 of the OpenClaw integration. Cron handles recurring tasks
that don't need to be on the voice path: nightly memory maintenance,
morning briefings, weekly reviews.

OpenClaw 2026.5.7's `openclaw cron *` subcommands manage scheduled
agent turns. Phase 7 documents the recipe + adds a cron-friendly
maintenance wrapper (`scripts/run_maintenance_for_cron.py`).

## What's done autonomously (Phase 7)

- `scripts/run_maintenance_for_cron.py` — thin shim around
  `scripts/maintenance.py`. Outputs JSON or single-line Telegram-
  friendly summary; captures stdout from underlying tasks; returns
  exit code 0 (clean) / 1 (some task errored) / 2 (init failure).
- `docs/openclaw_cron_setup.md` (this file) — recipes for
  `nightly-maintenance` + `morning-briefing` + `weekly-review`.

## What requires user-side OpenClaw configuration

`openclaw cron add` writes the cron entry to OpenClaw's config and
the Gateway scheduler. Run these once Telegram is live (Phase 4
user-side) so the output has somewhere to deliver.

### Nightly maintenance (3 am)

Two ways to run nightly maintenance, depending on whether the agent
needs to talk to your local Qwen or not.

#### Option A: Direct script invocation via Windows Task Scheduler (recommended for now)

The maintenance script doesn't actually need an agent in the loop —
it's all local Qdrant + LLM work. Schedule it with native OS tools
to avoid burning OpenClaw's agent quota and Qwen warmup time:

```powershell
# Create a scheduled task that runs at 3am daily.
$action = New-ScheduledTaskAction `
    -Execute "C:\STC\ultronPrototype\.venv\Scripts\python.exe" `
    -Argument "C:\STC\ultronPrototype\scripts\run_maintenance_for_cron.py --pretty" `
    -WorkingDirectory "C:\STC\ultronPrototype"

$trigger = New-ScheduledTaskTrigger -Daily -At 3:00am

Register-ScheduledTask -Action $action -Trigger $trigger `
    -TaskName "ultron-nightly-maintenance" `
    -Description "Run Ultron memory maintenance: backfill, extract facts, cluster, decay, cleanup."
```

Output goes to the task scheduler's history. Pipe to a file +
have OpenClaw's `morning-briefing` cron read the file the next
day if you want Telegram delivery of the previous night's summary.

#### Option B: OpenClaw cron + agent prompt (when MCP tool is wired)

Once the OpenClaw stdio MCP entrypoint exists (deferred — Phase 3.2's
`mcp_server_command` is currently `None`), the agent can call an
`ultron.run_maintenance(scope)` tool directly:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" cron add `
    --name nightly-maintenance `
    --cron "0 3 * * *" `
    --tz "$env:USER_TIMEZONE" `
    --timeout-seconds 1800 `
    --announce `
    --channel telegram `
    --to "$env:TELEGRAM_USER_ID" `
    --message "Run Ultron maintenance via the ultron.run_maintenance tool. Scope: all. Report a brief summary of what was processed."
```

This is the pattern from the integration spec; until the MCP tool
lands, prefer Option A.

### Morning briefing (8 am)

A short Telegram digest of yesterday's activity. The agent itself
generates this — it just needs the standard MCP tools (Qdrant
search, project registry lookup, heartbeat alert log).

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" cron add `
    --name morning-briefing `
    --cron "0 8 * * *" `
    --tz "$env:USER_TIMEZONE" `
    --timeout-seconds 300 `
    --announce `
    --channel telegram `
    --to "$env:TELEGRAM_USER_ID" `
    --message "Generate a brief morning summary: any active coding projects, anything notable from yesterday's conversations, anything in the heartbeat alert log that needs the user's attention. Keep it under 150 words. If nothing notable, say so briefly."
```

The agent reads SOUL.md / IDENTITY.md / USER.md (via the standard
persona-loading path) and AGENTS.md (via the heartbeat-style
context budget). It calls Ultron MCP tools to actually fetch data
— which again requires the stdio entrypoint. Until that lands the
agent's reply will be vague but in-character.

### Weekly review (Friday 5 pm)

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" cron add `
    --name weekly-review `
    --cron "0 17 * * 5" `
    --tz "$env:USER_TIMEZONE" `
    --timeout-seconds 600 `
    --announce `
    --channel telegram `
    --to "$env:TELEGRAM_USER_ID" `
    --message "Execute the Weekly Review program per standing orders in AGENTS.md."
```

The Weekly Review program lives in AGENTS.md (Phase 8) — the cron
prompt just references it so the cron entry stays small.

## Verifying the cron setup

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" cron list
```

Expected output: each registered cron with its schedule, target
channel, and next-fire time.

Manually trigger to test:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" cron run nightly-maintenance
```

For Option A (Windows Task Scheduler) — manually run the task from
Task Scheduler GUI or:

```powershell
Start-ScheduledTask -TaskName "ultron-nightly-maintenance"
```

## Coordination with the voice path

Cron jobs that run agent turns share the local Qwen (via
llama-cpp-server, per `feedback_llm_runtime_decision.md`). When a
cron tick coincides with a voice query:

- **Voice queries take priority** because they're in-process.
  The agent turn waits in llama-cpp-server's queue.
- **Heartbeat ticks defer when busy** (`skipWhenBusy: true` in the
  agent's heartbeat config). Cron does NOT defer — it runs at its
  scheduled time.

For the 3 am nightly maintenance this is irrelevant (user is
asleep). For the 8 am briefing it might overlap with the user
starting the day; the 5-minute briefing won't significantly delay
voice queries even if they collide.

## Configuration knobs

There are no Ultron-side config knobs for cron — the schedule lives
entirely in OpenClaw. The Ultron-side helper (`run_maintenance_for_cron.py`)
takes its configuration from `config.yaml` via the existing
maintenance pipeline.

To change which maintenance tasks run nightly, edit the cron command's
`--message` to specify a subset, or run the wrapper script with
`--task` arguments and use the output for whatever delivery you
prefer.

## Troubleshooting

- **Cron doesn't fire** — check `openclaw cron list` shows it as
  registered. Confirm Gateway is running. `openclaw cron run <name>`
  triggers the cron immediately for testing.

- **Maintenance task fails** — `run_maintenance_for_cron.py` returns
  exit code 1 when at least one task errored. The JSON output
  (`--json`) includes per-task counts; -1 means "task raised an
  exception". Check the captured stdout for the underlying error.

- **VRAM spike during maintenance** — maintenance loads the LLM
  cold (~30 s) and runs it serially against accumulated turns. If
  this collides with voice path, voice queries queue. Schedule
  maintenance at 3am or another off-hours slot.

- **Maintenance run takes hours** — first run on a large store can
  be slow. Typical cadence: run after each working session, not
  weekly. The `data/maintenance.sqlite` metadata tracks last-
  processed turn IDs so subsequent runs are idempotent.
