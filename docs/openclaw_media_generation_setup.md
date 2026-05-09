# Media generation setup (local-only)

Phase 12 of the OpenClaw integration. Routes voice / Telegram
prompts like "make me an image of a cat" through OpenClaw's media
generation provider plugins to a **local** image / video / music
model and delivers the result via Telegram for voice queries or
inline for Telegram queries.

The Ultron-side wiring is fully in place after Phase 12. Live media
generation requires a local image-generation backend such as
ComfyUI to be running on the same machine.

## Project policy: only Claude Code is paid

Ultron's stack is local-or-free everywhere except the supervised
coding subsystem (Claude Code, paid via the Anthropic API). That
constraint applies to media generation too:

- **Use:** ComfyUI (local Stable Diffusion / SDXL / similar), other
  local Stable Diffusion runtimes (Automatic1111 / sd.next /
  Diffusers via Python), local audio models (Bark, MusicGen if you
  want music).
- **Do NOT use:** Fal, Runway, Suno, OpenAI image / video, Google
  Imagen, or any other pay-per-use cloud API. These are explicitly
  out of scope.

If a future need surfaces a use case ComfyUI can't cover, raise it
with the user before adding any paid integration.

## What's done autonomously (Phase 12)

- `MediaGenerationConfig` schema (`enabled`, per-medium tool slugs,
  optional provider overrides, timeouts, ack-phrase pool).
- `OpenClawDispatcher.handle_media_generation` rewritten — bridge-
  wired path that maps the intent's `medium` to one of three tool
  slugs (`image_generate`, `video_generate`, `music_generate`) and
  fires `OpenClawClient.invoke_tool`.
- Voice ack phrases for the seconds-to-minutes generation latency.
- Tests covering the success / fallback / unknown-medium / exception
  paths.

## ComfyUI install + integration

ComfyUI is the canonical local image-generation backend. It uses
the GPU which competes with Qwen for VRAM — Ultron's 4B preset
peaks at 7913 MB on an 11.5 GB hard cap, so SDXL is too tight.
**Stable Diffusion 1.5 with FP16 + xformers fits**; SDXL or larger
needs you to swap Ultron to a smaller preset first.

### 1. Install ComfyUI

Follow the upstream README at
https://github.com/comfyanonymous/ComfyUI. On Windows + your RTX
4070 Ti, the recommended install is:

```powershell
# In a directory of your choosing — NOT inside the Ultron repo
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download a model checkpoint into `ComfyUI/models/checkpoints/`.
For the 4 GB headroom on your 4070 Ti, prefer SD 1.5 base or a
small fine-tune. SDXL works only when you have Qwen unloaded.

### 2. Run ComfyUI's API server

```powershell
.venv\Scripts\Activate.ps1
python main.py --listen 127.0.0.1 --port 8188
```

ComfyUI exposes an HTTP API on `127.0.0.1:8188` once running.
Verify with:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8188/system_stats" -UseBasicParsing | Select-Object -ExpandProperty Content
```

### 3. Configure OpenClaw to use ComfyUI

Update `~/.openclaw/openclaw.json`:

```json5
{
  models: {
    providers: {
      // ... existing litellm entry stays ...
      comfyui: {
        baseUrl: "http://127.0.0.1:8188",
        // Default workflow ComfyUI runs when an image_generate
        // tool call arrives. Generate this from ComfyUI's UI by
        // designing a workflow then "Save (API Format)".
        defaultWorkflow: "ultron-default.json"
      }
    }
  }
}
```

The exact field names depend on OpenClaw's ComfyUI provider plugin
version — verify with:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" plugins inspect comfyui-provider
```

If the OpenClaw ComfyUI provider plugin isn't bundled in your
install, install it via:

```powershell
& "C:\Users\alecf\AppData\Roaming\npm\openclaw.cmd" plugins install @openclaw/comfyui-provider
```

Restart Gateway after config changes.

### 4. Update Ultron's preset for the provider name

Set `default_image_provider: "comfyui"` in `config.yaml` so the
dispatcher passes the right provider slug:

```yaml
media_generation:
  enabled: true
  default_image_provider: "comfyui"
  # ... rest stays default ...
```

## VRAM coordination — critical

ComfyUI + SD 1.5 in fp16 with xformers takes ~3-4 GB VRAM. Ultron's
4B preset peak is 7913 MB. Combined: ~11 GB which is right at the
hard cap. Realistic options:

### Option A: Run ComfyUI only when Ultron is idle

Use OpenClaw's heartbeat `skipWhenBusy: true` to avoid concurrent
agent turns + image generation. Ultron's voice path stays
in-process; if you trigger image generation while a voice query is
mid-flight, Qwen will queue at llama-cpp-server and the image gen
will queue at ComfyUI. Both eventually complete, but the user
might hear ~20 s of silence.

### Option B: Swap Ultron to a smaller preset before generating

```
"Ultron, switch to the 4B" → currently active
"Ultron, generate an image of an astronaut on Mars"
  → If VRAM tight: voice ack mentions waiting for Qwen to unload
"Ultron, switch to the 9B"  ← only if you want 9B back later
```

### Option C: Skip image generation on this hardware

The 11.5 GB hard cap genuinely doesn't leave room for anything
substantial alongside Qwen 4B. If you want fast, reliable image
generation, run ComfyUI on a second machine reachable via LAN
(`baseUrl: "http://192.168.x.x:8188"` in the OpenClaw provider
config) — that's still local + free, just on different hardware.

## Configuration knobs

`config.yaml`:

```yaml
media_generation:
  enabled: true
  image_tool: "image_generate"
  video_tool: "video_generate"
  music_tool: "music_generate"
  default_image_provider: "comfyui"          # local-only canonical
  default_video_provider: null               # set when a local video gen lands
  default_music_provider: null               # set when a local music gen lands
  default_timeout_seconds: 120.0
  delivery_voice: "telegram"                 # voice query → Telegram delivery
  delivery_text: "inline"
  acknowledgment_phrases:
    - "Working on that. Should be a moment."
    - "Generating now."
    - "I'll send it when it's ready."
```

`enabled: false` keeps the dispatcher returning the stub voice
message — useful while you're setting up ComfyUI for the first
time so unintended generation requests don't trigger anything.

## Smoke test

After ComfyUI is running and OpenClaw is configured:

Via Telegram (or via voice when set up):

> "Make me an image of an astronaut on Mars."

Expected:

1. Within ~200 ms: ack phrase plays / appears.
2. After 5–60 s: image arrives. Voice query → image delivered to
   Telegram + voice confirmation; Telegram query → inline image.

If the ack phrase plays but no image arrives:

- Check ComfyUI's terminal for errors (model not found, OOM, etc.).
- Verify `openclaw doctor` lists the comfyui provider as healthy.
- Verify the default workflow JSON loads cleanly via ComfyUI's UI
  (Edit → Load workflow).

## Voice ack pattern

Like web-search and browser actions, media generation takes too
long for synchronous voice flow. The orchestrator plays an ack
phrase from `acknowledgment_phrases` within ~200 ms of the intent
firing, then the actual generation runs in a background task. When
complete, the dispatcher returns a short voice confirmation that
the orchestrator speaks via the normal completion narration loop.

## Video and music

Local **video** and **music** generation are open research areas
with much fewer turnkey local options than image gen. As of this
prototype, Ultron's `video_generate` and `music_generate` tool
slugs exist in the dispatcher but no canonical local provider is
recommended. Options when you actually need these:

- **Video:** AnimateDiff via ComfyUI (local, GPU-heavy), Stable
  Video Diffusion, Open-Sora. None are turnkey.
- **Music:** MusicGen (Meta, local via Audiocraft), Bark for
  speech, AudioCraft for full music.

Configure these in `models.providers.<slug>` once you've installed
the local backend — the dispatcher already routes based on the
intent's medium, no Ultron-side code change needed.

## Troubleshooting

- **"I couldn't generate that image just now."** — provider returned
  an error. Common causes: ComfyUI not running, model checkpoint
  missing, VRAM OOM (Ultron's Qwen + ComfyUI competing for the
  hard cap).

- **"I'd generate that for you, but the gateway isn't connected
  yet."** — bridge not wired or `media_generation.enabled: false`.
  Check `openclaw.enabled`, restart Ultron.

- **VRAM spikes when generating** — ComfyUI competes with Qwen.
  See "VRAM coordination" above. The block-and-revise validator
  (4B plan Item 8) will catch obviously-misdirected generation
  requests; trust it.

- **Generated images don't arrive in Telegram** — check
  `notifications.telegram.enabled` is true and
  `delivery_voice: "telegram"` (the default). Telegram side may
  also need `allowedUsers` to include the bot's own ID for the
  return path; verify `openclaw channels logs telegram`.

## Security posture

- Generation prompts go to your local ComfyUI install. Nothing
  leaves the machine (vs cloud providers where the prompt + result
  are logged by the vendor).
- Generated content can still be unpredictable. The block-and-revise
  validator (4B plan Item 8) intercepts media-gen dispatches like
  any other tool call; trust the validator's reasoning to suppress
  ambiguous requests.
- Telegram delivery means the generated content lives in your
  Telegram chat history. Treat it accordingly.
