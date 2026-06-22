# Ultron × Twitch — Live Status

**Created:** 2026-06-21 · **Branch:** `claude/affectionate-lehmann-c9fc0a` (worktree off `main`@`408b913`)
**Goal (user, 2026-06-21):** Build the COMPLETE Twitch chat-interaction capability to its fullest end-goal
vision — Ultron reads & responds to chat (batch, by-name, semantic addressing), runs channel-point redeems &
on-screen games (spin-the-wheel etc.), auto-moderates, takes voice ban/timeout commands, with EXHAUSTIVE,
paranoid, layered guardrails (deterministic + non-abliterated guard model) that make injection / "say something
bad" / phonetic-sounding-out attacks structurally impossible to land on stream. Flag-gated default-OFF.
Autonomous overnight build. User supplies Twitch creds / live-test on wake.

## REGROUND (BR-4.3) — sacred constraints in force
- **BR-P1 anticheat (P0):** voice/relay path imports ONLY numpy+urllib+scipy+stdlib+rapidfuzz. ALL Twitch
  network I/O, guard models, OBS websocket → SIDECAR processes (the EmbeddingGemma :8772 pattern). Never
  default-import desktop stack. Boot canary "anticheat posture OK | lean boot OK".
- **BR-P3 one-instance:** do NOT boot the heavy 8B/Kokoro/Whisper stack autonomously (can't verify nothing
  else runs while user asleep). Build code + unit tests; live integration deferred to user.
- **BR-P2 persona:** strict Ultron register; chat path uses a DEDICATED system prompt, never leaks vendor/model.
- Flag-gated default-OFF (`KENNING_TWITCH_*` + config section) → `main` runtime byte-unchanged.
- Tests via wrapper `scripts/run_tests.py`; regression yardstick = frozen 24-fail control. No stub/sentinel.

## REUSABLE ASSETS FOUND (retire-not-remove; build on these)
- `kenning/safety/validator.py` — `ToolCallValidator`: rule-registry, most-restrictive-wins, fail-CLOSED,
  per-rule config toggle, JSONL audit. **Mirror this pattern for the chat-content safety chain.**
- `kenning/safety/audit.py` — `AuditLog` JSONL (`logs/safety_audit.jsonl`). Reuse for chat-safety ledger.
- `kenning/tts/text_hygiene.py::sanitize_spoken_text` — the SINGLE TTS synthesis choke point
  (`KokoroSpeech._synthesize`). The phonetic/output safety gate MUST hook here → nothing bad reaches voice
  on ANY path. Already strips stage directions / control tokens / fixes G2P initialisms.
- `kenning/audio/broadcast.py::BroadcastSink` — tees spoken audio to a SEPARATE OBS/VoiceMeeter device
  (`audio.broadcast_device`). **This is the stream-only output channel** — physically distinct from the team
  relay B1/PTT path. Chat responses route here + speakers, NEVER the relay path → team isolation is structural.
- `kenning/safety/rules/category_i.py` rule **I6** — already BLOCKS Twitch Helix publish tool-calls. Our
  moderation (ban/timeout/delete) is a separate INTENTIONAL gated capability, routed through its own validated
  path, NOT the openclaw tool surface (else I6 blocks it).
- `kenning/streaming/{window,coordinator,presentation_scheduler}.py` — overlay/popup + scheduling infra.
- `kenning/audio/intent_gate.py` — 4-class intent gate over EmbeddingGemma sidecar → reuse for chat addressing.
- `kenning/llm/inference.py::generate_stream(system_prompt=…, sampling=…)` — route-all surface; sampling
  whitelist includes grammar+logit_bias. `llm_prompts.py` = prompt SSOT.
- Sidecar precedent: `scripts/embedder_server.py` (parent-death deadman, KENNING_EMBEDDER_PARENT_PID).
- Phonetics already a dep: RapidFuzz + Metaphone in the router/lexical backend.

## PLAN
1. [in progress] My own deep research rounds (web) — Twitch mechanics, guard models, injection defense,
   phonetic/homoglyph/leetspeak, streamer feature landscape, AI-streamer arch, OBS. Findings → `01_my_research/`.
2. Design + run agent boards: functionality (24, 4×6), security (18, 3×6), adversarial (18, 3×6) → `02_board/`.
3. Synthesis → final plan.
4. Spec (REQUIREMENTS→DESIGN→TASKS + manifest) → `03_spec/`.
5. Autonomous implementation in tested, flag-gated, reversible slices. Commit per slice.
6. Honest DONE/BLOCKED report + wake-up checklist.

## DEFERRED-TO-USER (legitimate blockers, BR-2.5)
- Twitch OAuth tokens / app registration / sign-in (creds).
- Live integration test against real Twitch + booting the heavy stack (BR-P3).
- Wiring the OBS browser-source into scenes (user said don't touch scenes).
- Loading guard/helper models into VRAM live.

## ACTIVE — board DONE; implementing
- Board **`wl65k4tfq`** COMPLETE (64 agents, 4.57M tok, ~85 min). Reports persisted → `02_board/{F,S,A}_report.md`
  + `MASTER.md`. Final plan = `03_spec/{REQUIREMENTS,DESIGN,constitution,tasks_manifest}`. Build order = 14 slices
  (S0–S13) in `tasks_manifest.json`. Resolved decisions D1–D8 in DESIGN.md.
- **10-layer safety arch (L0–L7 + arbiter + isolation)** — key board additions over my plan: codepoint ALLOWLIST
  strip (Unicode Tag-block U+E0000–E007F invisible-injection channel), L5 reassembly canonicalizer (acrostic/
  spell-out/NATO/cipher/IPA materialization), phoneme-domain L6 vs Kokoro's own phonemes, post-TTS Whisper L7,
  DATAMARKING + CHATTER_N tokenization, per-chatter ISOLATED generation, per-channel trajectory scanner
  (crescendo), provenance taint at play_to_device (invert force=True fail-open @orchestrator.py:3744-3750),
  fail-CLOSED-supersedes-fail-quiet, GPU priority lanes (relay preempts chat).

## BUILD PROGRESS (2026-06-21) — full safety architecture DONE + verified
**Committed + green (94 twitch tests):**
- `9643527` S0 — config section + flags + anticheat invariant (5 tests).
- `2dd5141` S4 — L1 normalize + blocklist, FNR==0 attack corpus / 0 benign FP (48 tests) + validator script.
- `fb1d854` S4b — L5 reassembly + L6 markup guard + deflection pool (15 tests).
- `374ba5b` S6-core — ChatSafetyValidator arbiter (most-restrictive-wins, fail-closed, audited) (11 tests).
- `7fbf2b0` S7 — team-isolation provenance taint + orchestrator relay guard + static import-graph wall (5 tests).
- `26252db` S5 — guard-model client + canary + chat-mode enable gate + GGUF sidecar server (10 tests).
**=> The entire L1-L7 safety stack + arbiter + guard interface + structural team isolation is built & tested.**
- `7914929` S1/S2/S3/S8/S9/S11 — the 6 standalone I/O + feature modules (EventSub/OAuth/economy+games/overlay/
  Helix-moderation/read-sidecar), built by a parallel agent board + independently re-verified: **307 tests/twitch
  pass together**; stub/forbidden-import/XSS scans clean; anticheat + team-isolation invariants hold.
- `697ddc8` S5-live — guard model **downloaded + LIVE-VALIDATED** (`E:\UltronModels\Llama-Guard-3-1B.Q5_K_M.gguf`):
  booted the sidecar, **canary passes**, `chat_mode_can_enable`→True; slur→S10, dox→S7, self-harm→S11, harm→S1,
  exchange unsafe→S1, benign Valorant chat clean. Found+fixed a real-model bug (Llama-Guard needs the manual prompt
  via create_completion, not the chat template). VRAM spike measured: guard ~1.46 GB resident → co-resident budget
  ≈11.1/12.3 GB (fits; Q4_K_M recommended for margin). 309 tests/twitch green. Sidecar stopped after validation.
- **S10/S12/S13 LANDED (2026-06-21)** — the full live chat-reply pipeline + features + speak-to-team, all
  flag-gated default-OFF, full twitch suite **519 green**:
  - `c0f842e` S10 `pipeline.py` ChatReplyPipeline (input screen→select→8B draft→output screen→deflect/speak,
    provenance=TWITCH_CHAT, fail-closed) (9 tests).
  - `223f8ec` S10 `runtime.py` ChatModeRuntime (guard-gated enable + buffer-then-batch tick + flagged review) (8).
  - `5506135` S10/S12/S13 leaf modules: addressing/selection/reply (datamarking+CHATTER_N) + commands/
    channel_points/content_ops + `speak_to_team.py` (AT-4 vetting-only, exact allowlist, no relay handle) (~110).
  - `8a1e35d` S10 `integration.py` factory + `service.py` ChatModeService + the flag-gated, fail-open orchestrator
    hook `_start_twitch_chat_mode()` (daemon loop, reconciles to `twitch.chat.reply_enabled`; OFF=byte-identical) (12).
  - **Live binding awaits the user's stack:** the orchestrator hook binds `llm=generate_stream`, `speak=self._speak`,
    `embed=EmbeddingBackend` and connects to the running guard+read sidecars — this only RUNS when `twitch.enabled`
    (which needs creds+sidecars), so first-light verification is the user's step.
  - Full wrapper regression after the two orchestrator.py edits (shutdown reaper + chat-mode hook): **CLEAN** —
    25 fail / 11712 pass / 39 skip; the 25 = the frozen 24-fail control + the pre-existing `always_listening`
    config.yaml branch fail. **ZERO new failures**; every `tests/twitch/*` test passed. flags-OFF byte-identical.

**(superseded — now built) prior remaining:** S10 live chat-reply pipeline wiring
(addressing→selection→8B→output-sandwich→BroadcastSink + chat-mode toggle + sidecar boot), S12 remaining
games/redeems/helper/content-ops, S13 speak-to-team (AT-4). These can only be VERIFIED with the running 8B +
guard model + Twitch creds (BR-P3), so they are documented, not half-wired.
**REGRESSION CLEAN (FINAL, all modules committed):** full wrapper = **26 fail / 11490 pass / 39 skip**. The
failure set = the frozen **24** control (8 relay + 14 env/infra + 2 coding_runner_anchors) + **2 non-regressions**:
(a) `test_always_listening_wiring::test_config_yaml_default_off` — PRE-EXISTING branch state (config.yaml:926
`always_listening: true`, unchanged by me, empty diff); (b) `TestLiveBatch0619C::test_drop_weapon_possessive_your_both_states[True]`
— the documented LRU global-state ORDER artifact (CONSTRAINTS.md), PROVEN to pass in isolation (2/2). **ZERO
failures attributable to my changes.** Only runtime-code touch = the additive `twitch` config section + the
additive (default-LOCAL_VOICE) relay provenance guard. Relay suite 128/128 in isolation.

## NEXT — implement S0→S13, each tested + flag-gated + committed
Run mapped tests then wrapper; confirm failure set ⊆ frozen 24. Update this STATUS per slice (durable across
compaction). Model-backed layers (L2/L3/L5-guard/L7/PII) implemented as sidecar clients that fail-CLOSED until the
user provisions deps+models; deterministic safety core (L1/L5-reassembly/L6-det/isolation/validator) ships fully.

## WAKE-UP CHECKLIST (what needs YOU — see DESIGN.md "deferred" + MASTER::deferred_to_user)
1. **Twitch creds:** register a Twitch app + grant broadcaster + dedicated-bot OAuth (Device Code Flow), pick
   least-privilege scopes. Run `python scripts/twitch_setup.py` (built tonight) to store under `~/.kenning/twitch.json`.
2. **Sidecar venv + models (AT-3, you pre-authorized):** create `.venv-twitch`; install guard deps
   (llama-cpp-python, transformers for Prompt-Guard-2, detoxify, presidio+gliner, panphon, misaki+espeak-ng,
   sqlite-vec). Download GGUFs: Llama-Guard-3-1B (default) — `hf download` cmds in DESIGN/checklist. (I did NOT
   auto-install heavy deps or load models overnight — VRAM/BR-P3.)
3. **VRAM SPIKE (BR-P3, stack stopped, verify nothing else runs):** measure guard + 8B + EmbeddingGemma + Kokoro +
   Whisper co-resident on the 4070 Ti; pick the shipped guard; confirm chat never starves a team callout.
4. **OBS/VoiceMeeter:** add a Browser Source → the local overlay URL; route the chat/stream BroadcastSink to a
   SEPARATE device from the B1 team-mic bus; set AutoMod-max + 2-4s Chat Delay + Shield-Mode defaults. (I do NOT
   touch your scenes.)
5. **Policy calls:** chat-mode-during-ranked (default OFF — strongest mitigation); viewer-memory TTL/COPPA;
   persona existential bounds; whether to SPIKE a non-abliterated reply head.
6. **Live calibration:** capture your chat + your let-through/ban decisions → calibrate danger_score bands +
   Prompt-Guard-2 domain-LoRA + the Kokoro+Whisper L7 mondegreen pair.

## LOG
- 2026-06-21: regrounded; explored integration points (safety/validator+policy, tts/text_hygiene, broadcast,
  embedder_server, intent_gate, config, category_i I6). Did ~14 web research rounds across all clusters →
  `01_my_research/00_synthesis_opus.md`. Designed + launched the 60-agent board `wl65k4tfq`. Wrote
  REQUIREMENTS.md. Committed durable research/spec. Yielding to await board completion.
