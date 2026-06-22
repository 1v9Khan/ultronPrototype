# Ultron × Twitch — WAKE-UP GUIDE (what's built + what needs you)

Built autonomously overnight 2026-06-21 on branch `claude/affectionate-lehmann-c9fc0a`.
Read `00_STATUS.md` for the full slice log; `03_spec/` for the spec; `02_board/MASTER.md` for the design.

## TL;DR — what landed (all committed, all tested, anticheat-clean, flag-gated default-OFF)
**307 new tests pass.** Two pillars are complete:
1. **The entire layered safety architecture** (the part you most wanted "so robust I don't worry"):
   L1 deterministic normalize+blocklist (covert-channel/homoglyph/leet/spaced/reversed + phonetic+fuzzy,
   **FNR==0** attack corpus, 0 benign FP) · L5 reassembly (acrostic/NATO/cipher/morse, batch-aware) ·
   L6 TTS-markup guard · the **ChatSafetyValidator** arbiter (most-restrictive-wins, fail-CLOSED, hash-chained
   audit w/ PII redaction, danger-score bands) · **L3 guard-model** client+canary+enable-gate+GGUF sidecar ·
   **structural team-mic isolation** (provenance taint + orchestrator relay guard + a static wall proving chat
   code can't even *name* the team path). The abliterated model is treated as hostile throughout.
2. **All the standalone I/O + feature building blocks**, each unit-tested: stdlib RFC6455 **EventSub** client ·
   **OAuth** device-flow + rotation-safe token store · **economy** (SQLite-WAL idempotent ledger + provably-fair
   commit-reveal RNG + spin-the-wheel/slots) · **overlay** server (loopback SSE, per-session token, strict CSP,
   textContent-only/XSS-safe HTML) · **Helix moderation** (self-idempotent, roster resolve, self/mod/broadcaster
   guard, mass-action breaker) · **read-sidecar** skeleton (parent-death deadman).

**Regression-clean:** the only main-code change is the additive `twitch` config section + the additive
(default-LOCAL_VOICE) relay provenance guard. Full wrapper failure set ⊆ the frozen control. `main` runtime is
byte-identical with the flags OFF.

## NOT yet wired (the remaining integration — needs your stack/creds; precisely specced)
The components exist and are tested in isolation; the LIVE glue that boots the sidecars and routes a real chat
message end-to-end is **S10/S12/S13** in `03_spec/tasks_manifest.json` + `02_board/MASTER.md` (build order). It is
deferred because it can only be VERIFIED with the running 8B + a guard model + real Twitch creds (booting the heavy
stack is BR-P3-gated; I won't load it while I can't confirm nothing else runs). Specifically remaining:
- **S10** chat-reply pipeline wiring: addressing (deterministic-metadata-first + the EmbeddingGemma residual) →
  selection engine → datamarked/CHATTER_N prompt → the 8B draft → the L5/L6 output sandwich → BroadcastSink (stream
  bus only). Plus the chat-mode Stream-Deck toggle + the guard/read/overlay sidecar spawn at boot (gated on
  `chat_mode_can_enable()`).
- **S12** remaining games/redeems/channel-points CRUD + the optional Qwen2.5 helper + content-ops.
- **S13** the speak-to-team redeem (default-OFF, AT-4, hardest-gated — ship only if exact-match-deterministic).

## ✅ DONE FOR YOU TONIGHT (2026-06-21, live-validated)
- **Guard model downloaded + LIVE-VALIDATED:** `E:\UltronModels\Llama-Guard-3-1B.Q5_K_M.gguf` (1.09 GB, from
  `QuantFactory/Llama-Guard-3-1B-GGUF`). Booted `scripts/twitch_guard_sidecar.py` against it and confirmed end-to-end
  through the production `GuardModelClient`: **canary PASSES**, `chat_mode_can_enable()` → `(True, ...)`; slur→S10(Hate),
  dox→S7(Privacy), self-harm→S11, "build a bomb"→S1, exchange-mode unsafe draft→S1 — while benign Valorant chat
  ("gg nice clutch jett", "sova hit 84 a main rotate b") passes clean. (A real-model bug was found+fixed+committed:
  Llama-Guard needs its manual prompt via `create_completion`, not the chat template — commit `697ddc8`.)
- **VRAM SPIKE measured:** the guard (Q5) is **~1.46 GB** resident. Budget on your 12 GB card:
  8B ~7.1 + guard ~1.46 + Kokoro ~1.5 + (Whisper/EmbeddingGemma on CPU = 0) + ~1.0 Windows ≈ **11.1 GB / 12.3 GB**.
  It FITS with ~1.2 GB headroom but the model total (~10.06 GB) is right at the 10 GB design cap → **for margin,
  swap to `Llama-Guard-3-1B.Q4_K_M.gguf` (~1.1 GB, same repo)**. Full co-resident confirmation = boot the stack
  once during your hands-off test (the arithmetic + the measured 1.46 GB guard footprint already answer it).
- **Guard sidecar launch command** (runs from the MAIN venv — it already has CUDA llama-cpp-python; the sidecar is a
  SEPARATE process so this is anticheat-clean; a `.venv-twitch` is only needed later for the OPTIONAL transformer
  extras like Prompt-Guard-2/Detoxify, NOT for the guard GGUF):
  ```
  $env:PYTHONPATH="<wt>\src;<wt>"
  $env:KENNING_TWITCH_GUARD_MODEL="E:\UltronModels\Llama-Guard-3-1B.Q5_K_M.gguf"
  $env:KENNING_TWITCH_GUARD_FAMILY="llama-guard"
  .venv\Scripts\python.exe scripts\twitch_guard_sidecar.py
  ```
- **Setup script confirmed runnable** (`scripts/twitch_setup.py --help` works) — it just needs YOUR client id (below).
- Guard sidecar was stopped after validation (VRAM freed); nothing is left running.

## YOUR STEPS (in order)
### 1. Twitch app + OAuth (≈10 min)
- Create a Twitch application at https://dev.twitch.tv/console/apps (OAuth redirect can be `http://localhost`;
  enable **Device Code Grant**). Note the **Client ID** (public, goes in config; the secret is NOT needed —
  device-code flow).
- Use a **dedicated bot account** for chat sends (gives the Chat-Bot-Badge transparency Twitch wants).
- Run the setup CLI (built + confirmed runnable tonight) ONCE per identity — it does the device-code flow (it
  prints a code + a `twitch.tv/activate` URL; you approve in a browser) and stores tokens atomically + 0600 under
  `~/.kenning/` (gitignored). The least-privilege scope set is built-in per identity:
  ```
  # on YOUR (broadcaster) account — approve in the browser as yourself:
  .venv\Scripts\python.exe scripts\twitch_setup.py --client-id <YOUR_CLIENT_ID> --identity broadcaster
  # on the dedicated BOT account — log into Twitch as the bot, then approve:
  .venv\Scripts\python.exe scripts\twitch_setup.py --client-id <YOUR_CLIENT_ID> --identity bot
  ```
- Put the Client ID + logins in `config.yaml` under `twitch.auth` (the section exists, all OFF).

### 2. Guard model — ALREADY DONE (downloaded + validated). Optional extras only.
The guard GGUF is downloaded and proven working (see "DONE FOR YOU" above); the guard sidecar runs from the MAIN
venv (no install needed). You only need `.venv-twitch` if you later want the OPTIONAL model-backed extras
(Prompt-Guard-2 injection classifier, Detoxify, Presidio/GLiNER PII, panphon/misaki phoneme L6) — those are
enhancements layered ON TOP of the already-working deterministic + guard stack:
```
python -m venv .venv-twitch ; .venv-twitch\Scripts\activate
pip install transformers detoxify presidio-analyzer gliner panphon misaki sqlite-vec
```
(For more VRAM margin, swap the guard to `Llama-Guard-3-1B.Q4_K_M.gguf` — same QuantFactory repo, ~1.1 GB.)

### 3. VRAM spike — MEASURED (see "DONE FOR YOU"). One optional confirm.
Guard footprint = ~1.46 GB (Q5); the computed co-resident budget (≈11.1/12.3 GB) fits with ~1.2 GB headroom but
sits at the 10 GB model cap → Q4_K_M recommended for margin. To CONFIRM the full co-residence once: with the
gaming instance stopped (BR-P3), boot Ultron with the guard sidecar up and watch `nvidia-smi` while a chat batch
runs alongside a team callout — verify callouts are never starved. (The `/canary` already passed live.)

### 4. OBS / VoiceMeeter (you wire the physical routing; I never touch your scenes)
- Add a **Browser Source** pointed at the overlay URL the server prints (`OverlayServer.url()`), 127.0.0.1 + token.
- Route the chat/stream `BroadcastSink` to a VoiceMeeter/OBS device **separate from the B1 team-mic bus**.
- Set Twitch AutoMod to max + a **2–4 s Chat Delay** + Shield-Mode defaults (the safety stack's L0 outer net).

### 5. Policy decisions you own
- Whether chat-mode may run **during ranked** (`twitch.chat.allow_during_ranked`, default OFF — the strongest
  single mitigation; we can't read game state, so it's discretion + the inferred busy-estimator).
- Viewer-memory retention/COPPA; persona existential bounds; whether to SPIKE a non-abliterated reply head.

### 6. Live calibration (after a hands-off-stream test session)
Capture your chat + your let-through/timeout/ban decisions → calibrate the danger-score bands
(`twitch.safety.tau_*`) and (optionally) a Prompt-Guard-2 domain-LoRA + the Kokoro+Whisper L7 mondegreen pair.

## Test it yourself (no creds needed)
```
$env:PYTHONPATH="<wt>\src;<wt>"; $env:KENNING_ROUTER_WAIT_SECONDS="0"
.venv\Scripts\python.exe -m pytest tests/twitch/ -q -p no:cacheprovider     # 307 pass
.venv\Scripts\python.exe scripts/twitch_validate_blocklist.py               # blocklist gate
```
