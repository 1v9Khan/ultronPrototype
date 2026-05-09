# Telegram channel setup

Phase 4 of the OpenClaw integration. The Telegram bot is the first
remote channel for Ultron — it lets you text Ultron from your phone
and receive proactive notifications (coding-task completion,
heartbeat alerts, etc.) when you're away from the desk.

## What this gets you

- **Inbound text → in-character response.** Send "what did we work
  on yesterday?" via Telegram; OpenClaw runs an `ultron-main` agent
  turn against the local Qwen, calls Ultron's MCP tools for Qdrant
  retrieval, and replies in Ultron's voice.
- **Proactive notifications from Ultron.** Coding tasks complete,
  heartbeat alerts fire, weekly review summaries land — all sent to
  your phone via Telegram instead of (or in addition to) the
  speakers.
- **No voice involved by default.** Text in, text out. The
  voice-handoff escape hatch (`[voice]` prefix) is gated behind
  `openclaw.bridge.inbound_voice_handoff_enabled` and stays off
  until Phase 4+ explicitly opts in.

## What this does NOT do

- Does not modify the voice pipeline. Voice queries still flow
  through `Whisper → Qwen → Piper → RVC` exactly as before.
- Does not introduce new VRAM. Telegram is text-only on the
  OpenClaw side; the Qwen turn it triggers shares the same
  llama-cpp-server instance the voice path already uses.
- Does not add an OpenClaw-side TTS provider. Ultron's voice stays
  Piper + RVC; OpenClaw's ElevenLabs/Azure TTS providers are
  intentionally not enabled.

## Bot creation (one-time, on your phone)

1. Open Telegram, search for `@BotFather`, start a chat.
2. Send `/newbot`.
3. Pick a display name ("Ultron") and a username ending in `bot`
   (e.g. `your_ultron_bot`).
4. BotFather replies with an HTTP API token. **Copy it** — it
   looks like `123456789:ABCdef...`. Keep it private; this token
   gives full control of the bot.
5. Optionally send BotFather:
   - `/setdescription` to give the bot a description
   - `/setuserpic` for a profile picture
   - `/setprivacy` → `Disable` if you want the bot to read group
     messages (default `Enable` only sees commands)
6. Find your own Telegram user id: send `/start` to `@userinfobot`
   and read off the `Id:` field. You'll whitelist this id below
   so only you can talk to your bot.

## Storing credentials (machine-side)

Tokens go in environment variables, never in `config.yaml` or
`openclaw.json`. Put them in `.env` (or your shell profile):

```
TELEGRAM_BOT_TOKEN=123456789:ABCdef-the-token-from-BotFather
TELEGRAM_USER_ID=987654321
USER_TIMEZONE=America/New_York
```

`.env` is already on `.gitignore`; double-check you haven't
accidentally tracked it before committing anything.

## OpenClaw configuration

The cleanest way is the interactive helper — it knows the schema
and validates as you go:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" configure
# Pick: Channels → Add → Telegram → paste token → set allowed users
```

Or non-interactively:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" channels add `
    --channel telegram `
    --token "$env:TELEGRAM_BOT_TOKEN"
```

Then verify with:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" channels list
```

You should see `telegram` listed with status enabled.

If the helper writes the bot token directly into
`~/.openclaw/openclaw.json`, edit the file and replace the literal
token with `${TELEGRAM_BOT_TOKEN}` so OpenClaw resolves it from the
environment at runtime. (OpenClaw 2026.5.7 supports `${VAR}` syntax
in config string values.)

The whitelist (`allowedUsers` field) should contain only your own
Telegram user id. Anyone who messages the bot from a non-listed id
should be silently dropped by OpenClaw — verify by sending a test
message from a second account, if you have one.

## Smoke test

After Gateway restart:

```powershell
# Restart the Gateway so it picks up the new channel.
# (Run from the Gateway's terminal: Ctrl+C, then re-run gateway.cmd.)

& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" channels list
# expected: telegram listed, enabled=true
```

Send "Hello, Ultron." to your bot from your phone. Within a few
seconds you should get an in-character text reply (the OpenClaw
`ultron-main` agent runs a Qwen turn through the persona files and
posts the reply back to Telegram).

If the reply doesn't arrive:

1. Check Gateway logs: `tail -f ~/.openclaw/logs/gateway.log` (PowerShell: `Get-Content -Wait`).
2. Confirm `openclaw doctor` reports the channel healthy.
3. Confirm your Telegram id is in `allowedUsers`. Non-allowed users
   get silently dropped — there's no error feedback to the sender.

## Ultron-side wiring (already in place after Phase 4)

These knobs in `config.yaml` control how Ultron uses the channel
once the OpenClaw side is configured:

```yaml
notifications:
  telegram:
    enabled: true
    user_id_env: "TELEGRAM_USER_ID"     # whose phone to ping
    notify_on:
      coding_task_completion: true
      coding_task_clarification_needed: true
      heartbeat_alerts: true
      standing_order_outputs: true
      search_results_async: false       # opt-in; can be noisy
```

`enabled: false` (default) keeps notifications off even when
OpenClaw is up — useful if you want inbound text but no proactive
pings while you experiment.

## Inbound voice handoff (deliberately deferred)

The integration spec describes a "voice handoff" escape hatch where
prefixing a Telegram message with `[voice]` causes Ultron's
orchestrator to speak the response through Piper + RVC (assumes
someone is at the desk to hear). The receiver scaffolding is in
place (`openclaw_bridge.events.OpenClawEventReceiver`) but the
transport hookup is **not** wired in Phase 4 — `inbound_voice_handoff_enabled`
defaults to `false`.

To turn it on later (Phase 4+ or whenever a real need arises):

1. Set `openclaw.bridge.inbound_voice_handoff_enabled: true` in
   `config.yaml`.
2. Wire a webhook subscription or polling loop in
   `OpenClawEventReceiver.start()`.
3. Test that prefixed messages route to the orchestrator's voice
   pipeline and unprefixed messages stay on the OpenClaw side.

## Troubleshooting

- **"telegram: not configured"** in `openclaw channels list` —
  the token wasn't saved. Re-run `openclaw configure` and paste
  the token again, or hand-edit `~/.openclaw/openclaw.json` to
  add `channels.telegram.{enabled, botToken}`.
- **Replies arrive but sound generic, not in-character** — the
  agent isn't reading `SOUL.md`. Check that `agents.list[].id` is
  `ultron-main` (or whichever you configured) and its
  `systemPromptOverride` is the user-facing persona (set up in
  Phase 1).
- **Ultron voice pipeline gets slower when Telegram is active** —
  shouldn't happen, but if it does it's because the Qwen turn for
  Telegram is queueing on the same llama-cpp-server. Voice path
  re-prioritises automatically since it's in-process. If you see
  sustained regression, set `notifications.telegram.notify_on.search_results_async: false` to
  reduce Qwen load.
- **Auth token rotation** — OpenClaw's bearer token (separate from
  the Telegram bot token) rotates if you regenerate it. The bridge
  re-reads the token on every request via `OpenClawLifecycle._read_token`,
  so rotations land without restart. Just confirm Telegram still
  works after a rotation.

## Security notes

- The Telegram bot token gives full control of the bot. Treat it
  like an API key. Never paste it into code review snippets,
  screenshots, or commit messages.
- The whitelist (`allowedUsers`) is the only thing keeping random
  Telegram users from talking to your bot. Telegram bots are
  publicly searchable by username.
- If you suspect token compromise: send `/revoke` to BotFather to
  revoke the existing token and issue a new one, then update
  `TELEGRAM_BOT_TOKEN` in your environment.
