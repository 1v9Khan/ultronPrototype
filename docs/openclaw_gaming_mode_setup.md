# Gaming Mode setup (V1-spec gap A1)

Voice-triggered, anticheat-safe shutdown of OpenClaw plugins before
launching a Vanguard / Easy Anti-Cheat / etc. protected game. Engaging
gaming mode runs ``openclaw plugins disable <id>`` for each configured
slug, optionally stops Docker Desktop, and logs the transition.
"Gaming mode off" reverses the cycle.

## Prerequisites

1. **OpenClaw CLI installed** at the canonical npm-global path
   (`%APPDATA%\npm\openclaw.cmd` on Windows). Already required by the
   broader OpenClaw integration -- verify with:

   ```
   openclaw --version
   ```

2. **Plugins to disable installed.** The default config disables
   `desktop-control` and `windows-control`. Both are ClawHub plugins.
   Install whichever you actually have:

   ```
   openclaw plugins install clawhub:desktop-control
   openclaw plugins install clawhub:windows-control
   ```

   Verify the installed slug names match the config defaults:

   ```
   openclaw plugins list --enabled
   ```

   If your install registers different slugs, edit
   `gaming_mode.plugins_to_disable` in [config.yaml](../config.yaml) to
   match.

## Enabling

In [config.yaml](../config.yaml):

```yaml
gaming_mode:
  enabled: true
  plugins_to_disable:
    - desktop-control
    - windows-control
  toggle_docker: false           # set true to also stop Docker Desktop
  log_path: "logs/gaming_mode.jsonl"
```

After flipping `enabled: true`, restart Ultron. The orchestrator builds
a `GamingModeManager` at startup that uses the existing OpenClaw bridge
client to issue plugin enable/disable calls.

## Voice triggers

| Phrase | Action |
|---|---|
| `"Ultron, gaming mode"` | engage |
| `"Ultron, gaming mode on"` | engage |
| `"I'm about to play Valorant"` | engage |
| `"I'm gonna play CS2"` | engage |
| `"Ultron, gaming mode off"` | disengage |
| `"I'm done playing"` | disengage |
| `"Ultron, full control restored"` | disengage |
| `"Are we in gaming mode?"` | status |

Engage voice response: `"Shutting down desktop control. Have fun."`.
Disengage: `"Full control restored."`. Status: `"Gaming mode is on."` /
`"Gaming mode is off."`.

## Failure modes

| Situation | Voice message |
|---|---|
| `gaming_mode.enabled: false` | `"Gaming mode isn't wired up. Enable gaming_mode.enabled in config to use it."` |
| OpenClaw bridge not connected | The startup log records `"gaming_mode.enabled=true but no OpenClaw client wired -- gaming mode disabled this session."`; the manager isn't constructed and the dispatcher uses the not-wired-up message above. |
| One or more configured plugins not installed | Engage proceeds best-effort. Voice response: `"Gaming mode engaged with errors -- some plugins didn't disable cleanly. Check logs/gaming_mode.jsonl."`. The other plugins still toggle. |
| OpenClaw CLI auth failure | Plugin toggle returns failure with the auth error in the metadata. |

## Coupling with desktop control (V1-gap C3)

When gaming mode is engaged, voice routing to `desktop-control` /
`windows-control` plugins (the C3 path) short-circuits with the message
`"Gaming mode is on. Desktop control is disabled. Say 'gaming mode
off' to restore it."`. This avoids the confusing "tool unavailable"
error you'd otherwise get from a deliberately-disabled plugin.

## Audit log

Each engage/disengage writes a JSONL row to
`logs/gaming_mode.jsonl`. Inspect with:

```
type logs\gaming_mode.jsonl
```

Each row carries `action` (engage/disengage), `status`, the per-plugin
result list, and Docker action info if applicable.

## Docker toggle (optional)

Set `toggle_docker: true` to also kill Docker Desktop on engage and
restart it on disengage. This frees ~6 GB of system RAM and removes
container-side processes that some anticheat systems flag.

The default executable path is `C:\Program Files\Docker\Docker\Docker
Desktop.exe`. Override via `docker_executable_path` in config.

## Rollback

Set `gaming_mode.enabled: false` in `config.yaml` -- the manager isn't
constructed and the dispatcher returns the not-wired-up message for any
gaming-mode voice trigger.
