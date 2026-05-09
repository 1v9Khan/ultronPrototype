# Desktop / Window control setup (V1-spec gap C3)

Voice routing for the OpenClaw `desktop-control` and `windows-control`
plugins. Enables phrases like `"take a screenshot of the desktop"`,
`"focus the chrome window"`, `"type 'hello' into the search box"`.

## Prerequisites

1. **OpenClaw CLI installed** (already required by other phases).

2. **Both plugins installed.** They're community plugins on ClawHub:

   ```
   openclaw plugins install clawhub:desktop-control
   openclaw plugins install clawhub:windows-control
   openclaw plugins enable desktop-control
   openclaw plugins enable windows-control
   ```

   Verify each plugin exposes the expected tool slugs (the defaults
   below are derived from the spec; adjust your `config.yaml` if your
   install uses different names):

   ```
   openclaw plugins inspect desktop-control --json
   openclaw plugins inspect windows-control --json
   ```

   Look for tool entries matching:
   - desktop: `desktop_screenshot`, `desktop_list_windows`,
     `desktop_find_window`
   - windows: `windows_focus_window`, `windows_click_element`,
     `windows_type_text`

3. **OpenClaw bridge wired** -- Ultron's orchestrator already
   constructs the `OpenClawClient` when `openclaw.enabled: true`.

## Enabling

In [config.yaml](../config.yaml):

```yaml
desktop:
  enabled: true
  default_screenshot_timeout_seconds: 10.0
  default_action_timeout_seconds: 5.0
  plugin_slug: "desktop-control"
  tool_slug_screenshot: "desktop_screenshot"
  tool_slug_list_windows: "desktop_list_windows"
  tool_slug_find_window: "desktop_find_window"

window_control:
  enabled: true
  default_action_timeout_seconds: 5.0
  plugin_slug: "windows-control"
  tool_slug_focus: "windows_focus_window"
  tool_slug_click: "windows_click_element"
  tool_slug_type: "windows_type_text"
```

Restart Ultron. The classifier now routes:

| Voice phrase | Routes to | Plugin tool |
|---|---|---|
| "Take a screenshot of the desktop" | `DESKTOP_AUTOMATION` (screenshot) | `desktop_screenshot` |
| "Take a screenshot of the active window" | `DESKTOP_AUTOMATION` (screenshot, target=active_window) | `desktop_screenshot` |
| "List my open windows" | `DESKTOP_AUTOMATION` (list_windows) | `desktop_list_windows` |
| "Find the cursor window" | `DESKTOP_AUTOMATION` (find_window, target=cursor) | `desktop_find_window` |
| "Focus the chrome window" | `WINDOW_AUTOMATION` (focus, query=chrome) | `windows_focus_window` |
| "Click the submit button in the form window" | `WINDOW_AUTOMATION` (click) | `windows_click_element` |
| "Type 'hello world' into the search box" | `WINDOW_AUTOMATION` (type, value="hello world") | `windows_type_text` |

## Anticheat safety (couples with A1)

When gaming mode is engaged (see
[openclaw_gaming_mode_setup.md](openclaw_gaming_mode_setup.md)), the
desktop / window dispatchers short-circuit with the voice message
`"Gaming mode is on. Desktop control is disabled. Say 'gaming mode
off' to restore it."`. The dispatcher reads the gaming-mode manager's
status before attempting any tool call.

## Failure modes

| Situation | Voice message |
|---|---|
| `desktop.enabled: false` | `"Desktop control isn't wired up yet. Enable the desktop-control OpenClaw plugin and set desktop.enabled in config."` |
| Plugin not installed | The OpenClaw agent reports the tool as unavailable; voice surfaces it as `"I couldn't capture the screen."` etc. |
| Gaming mode engaged | The short-circuit message above. |
| Tool invocation timeout | Translated to `"Something went wrong on the desktop side. Try again in a moment."` |

## Browser vs desktop priority

A screenshot utterance with a URL in it (e.g., "take a screenshot of
github.com") routes to the browser tool, not the desktop tool. That
matches the user expectation -- "screenshot of X.com" implies the page,
"screenshot of the desktop" implies the screen.

## Verification

Once both plugins are enabled, exercise each path:

1. `"Take a screenshot of the desktop"` -> Ultron speaks `"Screenshot
   captured -- saved to <path>."`. The image lands wherever the
   `desktop_screenshot` tool writes it (configured in OpenClaw's
   plugin config).
2. `"List my open windows"` -> Ultron speaks the count + first few
   titles.
3. `"Focus the chrome window"` -> Ultron speaks `"Focused chrome."`.

If any of those fail with `"... isn't wired up yet"`, check both the
`desktop.enabled` / `window_control.enabled` flag AND the
`openclaw plugins list --enabled` output.

## Rollback

Set `desktop.enabled: false` and/or `window_control.enabled: false` in
`config.yaml`. The classifier still parses the utterances (so they
don't get pulled into another category), but the dispatcher returns
the "not wired up" stub instead of attempting tool calls.
