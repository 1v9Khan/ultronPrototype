# Mobile node setup (iOS / Android)

Phase 11 of the OpenClaw integration. Documents the procedure for
pairing a phone with the local Gateway. Pairing is **optional** and
**not required** for any other phase — Telegram (Phase 4) covers
"text Ultron from your phone" perfectly well without a dedicated
app.

When you'd actually want to install a mobile node:

- **Voice triggering from the phone.** The OpenClaw iOS app
  supports Voice Wake (the host machine's openWakeWord doesn't
  match the phone's audio). Lets you say "Ultron" into your
  phone and have the Gateway wake.
- **Camera input.** "Ultron, what is this?" with a photo from
  the phone.
- **Screen capture.** "Ultron, look at my screen" — the agent
  reads what's on the phone's display.
- **Voice notes** — record on the phone, transcribe on the
  Gateway, process the transcript.
- **Talk Mode (Android)** — continuous voice conversation with
  the Gateway.

If you don't want any of those, **skip this phase**. Telegram
covers the common case.

## Hardware check

Before installing, verify you have:

- An iPhone (iOS 16+) or Android device (Android 11+).
- LAN access from the phone to the Windows host running OpenClaw,
  OR a way to expose the Gateway to the phone (Tailscale, public
  URL, etc.).

If you don't have the hardware available right now, this doc is
the future-reference; come back when you do.

## iOS setup

### 1. Install the app

OpenClaw iOS is distributed via TestFlight for the prototype period.
Check OpenClaw's docs for the current invitation link:
https://docs.openclaw.ai/mobile/ios

The bundle id is typically `ai.openclaw.openclaw` — search the App
Store / TestFlight by that.

### 2. Pair with the Gateway

On the Windows host:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" devices pair
```

This prints a QR code (and a fallback alphanumeric pairing code).
Open the iOS app, go to Settings → Pair Gateway, scan the QR.

Alternatively the app has a manual pairing option that takes the
same alphanumeric code.

### 3. Enable Gateway LAN access

By default the Gateway binds to 127.0.0.1 (loopback only). For LAN
access, edit `~/.openclaw/openclaw.json` to set:

```json5
{
  gateway: {
    bindHost: "0.0.0.0",            // listen on all interfaces
    // ... existing fields stay (auth.token, mode, etc.) ...
  }
}
```

Restart the Gateway. **Important**: opening the Gateway to the LAN
means anyone on your home network can attempt to reach it. Auth
token is the only thing keeping them out — verify
`gateway.auth.mode: "token"` and that the token is set.

For Tailscale or VPN setups, see OpenClaw's networking docs.

### 4. Verify pairing

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" nodes list
```

Expected: the new iOS node listed with its hostname / device id.

### 5. Test round-trip

From the iOS app:

- Send a text message to Ultron. Verify in-character reply.
- Try Voice Wake (say "Ultron" — the app's wake-word trigger
  forwards to the Gateway).
- Take a photo and ask "what is this?" — agent describes it.
- Test screen capture: open something on the phone, ask
  "what's on my screen?".

### 6. iOS-specific notes

- **Background audio.** iOS limits background mic access. Voice
  Wake works while the app is foregrounded; in the background,
  manual mic activation is needed.
- **Push notifications.** OpenClaw can deliver Telegram-style
  notifications natively on iOS once paired — set
  `target: "openclaw-ios"` in the relevant agent / cron / heartbeat
  config instead of (or in addition to) `target: "telegram"`.

## Android setup

### 1. Install the app

OpenClaw Android: check https://docs.openclaw.ai/mobile/android for
the current install link. Sideload-friendly during the prototype
period.

### 2. Pair with the Gateway

Same `openclaw devices pair` flow as iOS — scan the QR or enter
the pairing code.

### 3. Android-specific limitations

- **Voice Wake disabled.** OpenClaw Android currently uses manual
  mic activation only. Talk Mode (continuous voice) IS available,
  but you have to start it explicitly.
- **Canvas / screen capture / camera** — supported per the Android
  permissions model. Each capability prompts on first use.

## Multi-node coordination

When multiple nodes are paired (Windows host + iOS, e.g.), the
Gateway routes incoming events based on `agents[].defaults` and
the channel that received the event:

- An iOS Voice Wake → routed to `ultron-main`, response speaks
  back to the iOS app.
- A Telegram message → routed to `ultron-main`, response goes
  back to Telegram.
- A heartbeat tick → fires through `agents[].heartbeat.target`
  (typically `telegram`).

Voice triggers from any source land in the same Qdrant
conversations collection — there's no per-node isolation. If you
want isolation, configure per-node agents (out of scope for
prototype).

## Security posture

- **Pair only with devices you control.** Pairing gives the device
  full read access to the workspace and can issue any tool call
  the agents are configured for.
- **Rotate the auth token after losing a paired device.** Revoke
  the old token via `openclaw devices revoke <node-id>` and pair
  again.
- **LAN-bind cautiously.** `bindHost: "0.0.0.0"` exposes the
  Gateway to your local network. Use Tailscale / WireGuard / a
  VPN if you don't trust everything on the LAN.

## Disabling a paired node

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" devices revoke <node-id>
```

The node-id comes from `openclaw nodes list`. Revoke removes the
device's auth without un-installing the app — re-pair to restore.

## Why this is optional

The integration prompt's stretch goal is "Ultron accessible from
anywhere via the phone app." The minimum useful version of that is
**Telegram-only** — Phase 4 already gives you in-character text
chat plus proactive notifications from your phone, without needing
to install OpenClaw's app.

If the prototype settles into a "I want voice from my phone too"
pattern, install the iOS app at that point. Until then, Telegram
covers the daily workflow.

## Status (Phase 11)

This document is the procedural reference. **No installation has
been performed** — Phase 11 is intentionally docs-only unless the
user indicates they have hardware to pair.

If you have a paired iOS or Android node and want to verify the
Ultron-side wiring works:

1. Trigger a coding-task completion notification.
2. Confirm it arrives in the OpenClaw mobile app (and Telegram, if
   both are wired).
3. The orchestrator's `NotificationDispatcher` doesn't need any
   changes for mobile delivery — OpenClaw's channel routing
   handles it transparently when the node is paired and configured
   as a delivery target.
