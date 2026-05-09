# Browser tool setup

Phase 6 of the OpenClaw integration. The browser tool drives a real
Chrome instance via OpenClaw's bundled Playwright plugin, letting
Ultron open pages, fill forms, click elements, and capture
screenshots in response to voice or Telegram queries.

The Ultron-side wiring (`BrowserTool` wrapper, dispatcher rewrite,
config) is fully in place after Phase 6. Live browser activity
requires OpenClaw's browser tool to be reachable from the Gateway —
which in turn requires Playwright + a Chromium install.

## What's done autonomously (Phase 6)

- `src/ultron/openclaw_bridge/browser.py` — `BrowserTool` wrapper
  with navigate / snapshot / click / type / screenshot /
  get_page_text primitives. Each method maps to a structured agent
  prompt and unpacks the result into a typed dataclass.
- `src/ultron/openclaw_routing/dispatcher.py` —
  `OpenClawDispatcher.handle_browser` rewritten to use the wrapper
  when a bridge is wired and `browser.enabled: true`. Falls back
  to the existing stub voice message otherwise.
- `BrowserConfig` schema with timeouts + ack phrase pool.
- `config.yaml` `browser:` section.

## What requires user-side OpenClaw configuration

OpenClaw's browser plugin needs Playwright + Chromium installed and
reachable from the Gateway. On a fresh install:

### Verify the plugin is loaded

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" plugins list `
    | Select-String -Pattern "browser|playwright"
```

Expected: at least one entry (typically `@openclaw/browser-tool`)
listed as enabled.

### Verify Playwright + Chromium

OpenClaw bundles Playwright; on first browser-tool use it asks
Playwright to download Chromium. To pre-fetch:

```powershell
# Run from the directory where OpenClaw stores its node_modules.
# OpenClaw's npm-global install is typically at:
#   C:\Users\<user>\AppData\Roaming\npm\node_modules\openclaw
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" doctor
```

`openclaw doctor` reports any missing browser dependencies. If it
flags Playwright missing, follow OpenClaw's guidance:

```powershell
# Inside the openclaw install dir (path varies by npm version)
npx playwright install chromium
```

### Smoke test

After the Gateway is restarted with the browser plugin reachable:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" agent --agent ultron-main `
    -m "Take a screenshot of news.ycombinator.com using the browser tool."
```

Expected: a JSON response that includes a screenshot reference (file
path or base64 payload). If the tool is missing, OpenClaw replies
"tool unavailable" or similar — `BrowserTool._invoke` translates
that to an `OpenClawToolError` and the dispatcher returns a clear
voice message ("I couldn't load that page just now").

## Tool-deny lists and the messaging profile

The locked-in constraint from Phase 0 is `tools.profile: messaging`
on every local-Qwen agent in `~/.openclaw/openclaw.json`. The
messaging profile does NOT include the browser tool by default.

To enable browser dispatch on `ultron-main` while keeping the
prompt-budget / 30 s OpenAI SDK timeout safety, layer browser into
the deny list's `alsoAllow` (per OpenClaw 2026.5.7 schema):

```json5
{
  agents: {
    list: [
      {
        id: "ultron-main",
        // ... existing fields stay ...
        tools: {
          profile: "messaging",
          deny: [
            "group:web", "group:fs", "group:runtime",
            "memory_search", "send"
            // browser is NOT in deny anymore
          ],
          alsoAllow: ["browser"]   // explicit additional allow
        }
      }
    ]
  }
}
```

Re-run Phase 0 verification after the change to confirm:

1. Voice TTFT (median ≤ 100 ms baseline) unchanged on routine
   queries that don't hit the browser.
2. The `ultron-test` agent's deny list is unchanged — the worker
   agent stays locked down.

## Voice response strategy

Browser actions take 2–60 seconds. The dispatcher uses the existing
acknowledgment-phrase pattern (configured under
`browser.acknowledgment_phrases` in `config.yaml`) — orchestrator
plays one ack phrase within ~200 ms of the intent firing, then runs
the browser action in a background task.

When the action completes, the dispatcher returns a short voice
message describing the outcome ("Loaded News.ycombinator.com.",
"Screenshot captured.", "I couldn't click that."). The orchestrator
speaks it as part of the normal completion narration loop —
exactly the same path used by coding-task completions today.

## Configuration reference

`config.yaml` knobs:

```yaml
browser:
  enabled: true                                # set false to suppress dispatch
  default_snapshot_mode: "ai"                  # 'ai' (refs) or 'aria' (a11y tree)
  default_navigation_timeout_seconds: 30.0
  default_action_timeout_seconds: 10.0
  default_screenshot_timeout_seconds: 30.0
  long_running_progress_threshold_seconds: 5.0
  acknowledgment_phrases:
    - "Pulling up that page now."
    - "Looking at it."
    - "Loading the site."
    - "Give me a moment to navigate."
```

`browser.enabled: false` keeps the entire bridge alive but suppresses
browser-specific dispatch (handler reverts to the stub voice
message). Useful for debugging or for skipping browser activity
during a noisy week.

## Troubleshooting

- **"I couldn't load that page just now."** — browser tool returned
  an error. Check the Gateway log (`tail -f ~/.openclaw/logs/gateway.log`)
  for Playwright stack traces. Common causes: Chromium missing,
  network error, page loaded a download instead of HTML.

- **"Tool unavailable" from the agent** — `BrowserTool._invoke`
  translates this to `OpenClawToolError`. Means the agent's tool
  registry doesn't include `browser`. Verify `tools.alsoAllow`
  in the agent config and restart the Gateway.

- **VRAM spikes during a browser turn** — browser is CPU + Chromium,
  not GPU. The agent turn that drives the browser uses the local
  Qwen via llama-cpp-server, which is the same instance the voice
  pipeline uses. Concurrent voice + browser may queue at the
  llama-cpp-server. Voice path takes priority since it's in-process;
  browser activity is async background.

- **Browser opens visibly on every call** — Playwright defaults to
  headless mode; if you're seeing a real Chromium window, the agent
  prompt or OpenClaw config is overriding. For prototype use this
  is fine; for production-style automation switch to headless.

## Security notes

- Treat browser dispatch as a remote-code-execution surface for the
  agent's prompt. The block-and-revise validator (4B plan Item 8)
  short-circuits browser dispatch when the validator decides the
  action doesn't advance the user's stated goal — keep
  `openclaw.block_and_revise.enabled: true` to take advantage.
- The browser tool can navigate to arbitrary URLs the agent
  decides on. Don't ask the agent to log into financial accounts;
  Foundation Phase 5 explicitly excludes those scenarios.
- Screenshots may capture sensitive content (open tabs, form
  fields). They get returned as base64 payload or file path —
  ensure your storage / chat history retention is consistent
  with what's in the screenshot.
