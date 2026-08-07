# Kenning — Vision Roadmap (long horizon)

> **Scope.** This is the multi-year *vision* list: what we want to add, how we would build it, what it
> touches, and roughly the order we would build it in. It is deliberately larger than any one milestone.
>
> **Relationship to the other planning docs.** `docs/ultron_1_0/` is the *active near-term direction* and
> wins on any conflict about what is being built **right now**. `docs/canon/` is the rule of record and
> wins on any conflict about *how* work is done. `docs/ultron_1_0/CONSTRAINTS.md` is sacred and wins on
> any conflict about what must never regress. This document is the horizon those three point at.
>
> **Nothing here is committed work.** Each item becomes real only when it gets a spec
> (`REQUIREMENTS` → `DESIGN` with ≥2 alternatives → `TASKS` + `tasks_manifest.json`) under
> `docs/ultron_1_0/` or a successor directory.
>
> **Sources.** Phases 1 and 3–8 derive from a recorded planning conversation covering inference
> efficiency (MoE, low-precision, speculative decoding), post-training methods, a WSL-migration question,
> the "make the system incapable of harm" permission-layer argument, and the Kenning-as-platform /
> Ultron-as-persona branding conclusion. **Phase 2 (game intelligence) derives from the streamer's own
> feature list**, not from that conversation, and is written as proposals to be confirmed. Where this
> document contradicts the conversation it is deliberate and says so — notably on MoE (an unfavourable
> trade at 12 GB), FP4 (a Blackwell feature, unavailable on Ada), and the WSL migration (see 4.1).

---

## How to read this

Each item carries the same eight fields so the list can be worked from directly:

| Field | Meaning |
|---|---|
| **What** | The thing being added, in one or two sentences. |
| **Why now** | Why it sits at this point in the order. |
| **How** | The concrete build approach, including alternatives considered. |
| **Components** | New files/modules to create, by path. |
| **Work** | The task breakdown, roughly in dependency order. |
| **Integrates with** | The existing subsystems it touches. |
| **Leverages** | What already exists that we do not rebuild. |
| **Risks / Done-when** | What could go wrong; the acceptance bar. |

**Effort scale.** `S` ≈ one focused session · `M` ≈ a few sessions · `L` ≈ a milestone · `XL` ≈ a campaign.

### Ordering rationale

The order follows the sequence the topics were raised, which also happens to be a defensible build order:

1. **Efficiency first** — everything downstream (bigger models, more personas, more concurrent
   subsystems) is paid for out of the same 12 GB card. Headroom bought early is spent many times.
2. **Game intelligence second** — this is the actual product and the reason the rest exists. It is
   also mostly GREEN-rung work that can proceed in parallel with everything below it.
3. **Post-training third** — once serving is cheap and measurable, the leverage moves to *quality per
   token*, which is a training problem, not a serving one. Game data also makes the eval sets real.
4. **Portability fourth** — a decision best made *after* the runtime is out-of-process, because
   out-of-process is most of what portability actually requires.
5. **The capability broker fifth** — the gate on every form of public distribution. Nothing ships to a
   third party before this exists. (Bring it forward if game APIs start touching the network broadly.)
6. **Kenning identity sixth** — the rename is cheap while the user base is one person and expensive
   after. It must land before any public surface.
7. **Platform surface seventh** — plugins, APIs, creator tools. Only safe on top of the broker.
8. **Productization last** — packaging, editions, support.

### Non-negotiables carried forward from `CONSTRAINTS.md`

Every item below is subordinate to these. They are restated because a roadmap that quietly violates them
is worse than no roadmap.

- **Anticheat (BR-P1, P0).** Ultron runs beside Valorant + Vanguard. The voice/relay path imports only
  `numpy + urllib + scipy + stdlib + rapidfuzz`. `kenning.desktop` and audio-diagnostics monitoring are
  never default-imported. The import firewall installs in `__main__` before the Orchestrator. Any new
  subsystem that would sit on the voice path must be audited against this before design, not after.
- **10 GB VRAM cap** on the RTX 4070 Ti (12 GB). Build the best quality that fits.
- **One instance at a time** (BR-P3) — shared port 8772, wake word, audio devices, PTT.
- **Flag-gated, default-OFF.** New behavior ships behind a flag that defaults to today's behavior, so
  `main` runtime is unchanged until the flag is deliberately flipped.
- **Retire, don't remove.** The proven deterministic relay pipeline is the golden path. Superseded code
  becomes a fallback, not a deletion.
- **No cloud, no telemetry, no paid APIs** in the local product by default.

---

## Phase 1 — Inference substrate: cost, speed, and headroom

The theme of the conversation that seeded this document: MoE, low-precision math, and speculative
decoding as the levers on cost-per-token. All three apply here, but the local single-GPU case reorders
their value. Speculative decoding is nearly free and already wired. Precision is a ladder we are partway
up. MoE is a model-selection decision, not an engineering one.

### 1.1 Finish speculative decoding — `M`

**What.** Turn on draft-model speculative decoding for the generation path, with a measured accept-rate
gate that disables it automatically when it stops paying.

**Why now.** It is the single largest already-paid-for win in the repo. `llm.draft_kind` defaults to
`"none"`; the wiring exists and is inert. The presets for `IQ3_XS`/`IQ4_XS` speculative decoding already
landed (`d4df64b`), and the Ultron 1.0 default reverted to 8B + spec decoding in `99caae1`.

**How.** Pair each serving model with a much smaller same-tokenizer draft (a 0.5B–1B Qwen for the Qwen
family). Draft *k* tokens, verify in one forward pass, accept the longest matching prefix. The win is
proportional to accept rate × draft cheapness; below roughly 40% acceptance it becomes a loss, so the
system must measure and self-disable rather than trust a config value.

Alternatives considered: **n-gram / prompt-lookup decoding** (no draft model, no extra VRAM — genuinely
attractive for the relay path, where callouts are highly repetitive and much of the output is copied from
the prompt); **Medusa/EAGLE-style heads** (better accept rates but requires training heads per model,
which belongs in Phase 3); **no speculation** (the current state). Prompt-lookup is the best first move
for relay specifically because it costs zero VRAM and the callout corpus is small and repetitive.

**Components.**
- `src/kenning/llm/draft_model.py` — exists; extend with accept-rate accounting.
- `src/kenning/llm/spec_decode_policy.py` — NEW. Rolling accept-rate window, auto-disable, per-pool
  policy (relay vs conversation vs chat-reply have very different repetitiveness).
- `src/kenning/llm/prompt_lookup.py` — NEW. N-gram draft from the prompt itself; no model, no VRAM.
- `scripts/bench_spec_decode.py` — NEW. Sweeps *k*, draft model, and pool; emits a scorecard.

**Work.** (1) Instrument accept rate per turn into `trace.py`. (2) Bench the existing draft presets on the
real corpora under `tests/data/`. (3) Implement prompt-lookup as a zero-VRAM alternative. (4) Add the
auto-disable policy. (5) Per-pool defaults. (6) Flag-gate, default OFF, then flip after live A/B.

**Integrates with.** `llm/inference.py` (`generate_stream`), `pipeline/orchestrator.py` speculative-relay
path, `trace.py` stage marks, `audio/relay_speech.py`.

**Leverages.** The existing draft wiring, the per-model preset refactor, the relay corpora, and the
speculative-relay work from the callout-latency pass (~370–620 ms callouts).

**Risks.** Draft VRAM competes with the 10 GB cap — prompt-lookup sidesteps this entirely.
llama-cpp-python version sensitivity (0.3.22 pinned; `flash_attn=True` needs non-F16 KV).
**Done-when.** Measured tokens/sec improvement on the relay and conversation pools with no golden-digest
change, and a demonstrated auto-disable on a pool where it does not pay.

### 1.2 The precision ladder — INT4/FP4 and what actually fits — `M`

**What.** A systematic, measured walk down the quantization ladder for each model role, plus adoption of
4-bit floating point formats when the runtime supports them on this hardware.

**Why now.** VRAM is the binding constraint on everything else in this document. Every gigabyte
recovered is a gigabyte available for a bigger router, a second persona, or a vision model.

**How.** Today the stack already spans the ladder ad hoc — `Q5_K_M` for the 8B, `Q6_K` for the Heretic 4B,
`IQ3_XS`/`IQ4_XS` presets, `q8_0` KV cache, and a recent STT move from `int8_float16` back to `float16`
because the accuracy cost landed on exactly the proper nouns and agent names the relay depends on. That
last change is the template for the whole workstream: **quantization is not uniformly good, and the right
level depends on what the model is being asked to preserve.**

FP4 (and NVFP4/MXFP4) is the frontier direction, but honestly: it is a Blackwell-class tensor-core
feature. On a 4070 Ti (Ada) the practical ladder is INT4/IQ-family weights with FP16 accumulation. The
work here is to *stop guessing* and build the measurement rig, so that when hardware or `llama.cpp`
support changes, the answer is a benchmark run rather than a rewrite.

Alternatives: per-model hand-tuning (status quo — works but does not generalize); AWQ/GPTQ-style
calibrated quantization (better quality per bit, needs a calibration set — a natural fit once Phase 3
gives us trace corpora); keeping everything at Q5+ (safe, wasteful).

**Components.**
- `scripts/quant_ladder.py` — NEW. For a given model role and eval set, sweep quantization levels and
  emit quality-vs-VRAM-vs-latency.
- `docs/vision/quant_ladder_results.md` — NEW. The living results table.
- `src/kenning/llm/vram_budget.py` — NEW. A declarative budget: each subsystem registers its ceiling, and
  boot fails loudly rather than thrashing when the sum exceeds the cap.

**Work.** (1) Define per-role eval sets (relay accuracy, scenario-router accuracy, chat-reply quality,
STT WER on the Valorant gazetteer). (2) Build the sweep harness. (3) Populate the results table.
(4) Implement the VRAM budget registry and wire boot-time enforcement. (5) Re-pick defaults from data.

**Integrates with.** `config.py` preset system, `llm/inference.py`, the gaming-mode VRAM reclaim path,
device-wide VRAM idle telemetry (`d2402c1`).

**Leverages.** The per-model preset refactor, the existing gaming-mode swap machinery, the VRAM telemetry
already logging idle creep, and the `float16` STT lesson as prior art.

**Risks.** Quantization damage is *sneaky* — it shows up on rare tokens (agent names, callsigns) that
aggregate metrics miss. The eval sets must be adversarial about proper nouns.
**Done-when.** Every model role has a documented, measured quantization choice with a stated tradeoff, and
boot enforces a declared VRAM budget.

### 1.3 Mixture-of-Experts models — evaluate, don't assume — `M`

**What.** Evaluate MoE checkpoints (Qwen3-MoE family, Mixtral-class, and successors) as the serving model,
and understand honestly whether MoE helps *this* deployment.

**Why now.** It was the opening question of the seeding conversation and deserves a real answer rather
than the industry-level one.

**How.** The industry framing — "only a few experts fire per token, so compute per token drops" — is true
for *datacenter* economics, where compute is the bottleneck and memory is abundant. **The local
single-GPU case often inverts this.** A sparse model must hold *all* experts in VRAM while only using a
fraction per token, so an MoE with 30B total / 3B active needs roughly 30B-worth of weights resident but
delivers roughly 3B-worth of speed. On a 12 GB card that trade is frequently *worse* than a dense model
of the same memory footprint, because the dense model uses every parameter it paid to store.

Where MoE does win locally: when offloading is acceptable (expert-level CPU offload with a hot-expert
cache), or when the quality-per-active-parameter is high enough that a heavily quantized MoE beats a
dense model at equal VRAM. Both are empirical questions.

So the work is an honest bake-off, not an adoption plan. This item may well conclude "dense wins here" —
that is a successful outcome, and it retires a recurring open question.

**Components.**
- `scripts/model_bakeoff.py` — NEW. Generalizes the existing pool-parity harness to any candidate model:
  accuracy per pool, tokens/sec, VRAM, cold-load time.
- `docs/vision/model_bakeoff_results.md` — NEW.

**Work.** (1) Generalize `scripts/_pool_parity_harness.py` into a model-agnostic bake-off. (2) Assemble
candidates across dense and MoE at matched VRAM. (3) Run against the relay, scenario-router, chat-reply,
and QA pools. (4) Record the verdict with numbers. (5) If MoE wins, add expert-offload support.

**Integrates with.** The preset system, `llm/inference.py`, the scenario router from PR #1.

**Leverages.** `scripts/_pool_parity_harness.py`, the Heretic-4B parity loop, the model-vs-accuracy-vs-latency
table already produced for the scenario router (Qwen3-4B heretic Q5 at 97.2% / 125 ms).

**Risks.** `llama.cpp` MoE support quality varies by architecture; Qwen3.5-9B is already hard-blocked by
`GGML_ASSERT` (#23347), which is a reminder that "supported" and "works" differ.
**Done-when.** A documented verdict with measurements, and the recurring question closed.

### 1.4 The model host as a service — `L`

**What.** Move model serving fully out-of-process behind a stable local API, so the voice loop talks to a
serving daemon rather than owning `llama_cpp` in its own address space.

**Why now.** This is the keystone of the phase. It independently enables: crash isolation, model hot-swap
without restarting the voice loop, multiple consumers (voice + chat + coding + future plugins) sharing one
resident model, the portability work in Phase 4, and the plugin surface in Phase 7.

**How.** PR #1 already proved the pattern under duress: `faster-whisper`/CTranslate2 with `device_index=1`
corrupted memory in-process and killed Ultron with no traceback; `scripts/stt_server.py` fixed it by
masking the child onto one GPU with `CUDA_VISIBLE_DEVICES` so the card is plain `cuda:0` there. Decode
went 7.3 → 27.7 tok/s once the card stopped thrashing. **Generalize that pattern from STT to the LLM.**

Design: a supervised child process holding the model, speaking a small local protocol (loopback HTTP with
SSE for token streaming, matching the existing `:8772` embedder sidecar convention). The client keeps the
same `generate_stream` signature so nothing above it changes. Crash → supervisor restarts → the voice loop
degrades to a spoken "model restarting" rather than dying. Model swap becomes "start the new child, drain,
cut over, stop the old" instead of an in-process reload.

Alternatives: keep in-process (lowest latency, current state, no isolation); `llama.cpp` server binary
(mature, but another dependency and less control over persona/preset plumbing); full multi-process actor
system (over-engineered for one user).

The cost is real and must be measured: an IPC hop per token stream. The mitigation is that streaming
already dominates — the first-token path is what matters, and a loopback socket adds well under a
millisecond.

**Components.**
- `scripts/llm_server.py` — NEW. The serving child, mirroring `scripts/stt_server.py`.
- `src/kenning/llm/client.py` — NEW. Drop-in client preserving `generate_stream`.
- `src/kenning/llm/supervisor.py` — NEW. Spawn/health/restart/backoff; reuses the sidecar lifecycle
  patterns and the zombie-killer registration.
- `src/kenning/serving/protocol.py` — NEW. Shared request/response/stream schema.

**Work.** (1) Define the protocol. (2) Build the server with the `CUDA_VISIBLE_DEVICES` masking lesson
applied. (3) Build the client behind the existing interface. (4) Supervisor + health + zombie-killer
registration. (5) Flag-gate (`in_process` | `served`) with `in_process` as default. (6) Latency A/B.
(7) Hot-swap. (8) Flip the default once proven.

**Integrates with.** `llm/inference.py`, the orchestrator's speculative paths, gaming-mode model swap,
`twitch` sidecars, the coding stack, `_kill_twitch_sidecars` / zombie-killer lifecycle, `PROJECT_ROOT`
path anchoring (the machine-move fix — sidecar paths must never resolve against cwd).

**Leverages.** `scripts/stt_server.py` and its proven masking approach, the embedder sidecar on `:8772`,
`twitch/sidecar_launch.py`, the held-port restart diagnostic (`5bdcad5`), and the sidecar self-reap fix
(`reap_stray_sidecars` must never kill an ancestor).

**Risks.** Added latency on the hot path; another process to supervise; port collisions (the repo already
has a held-port diagnostic for exactly this). Anticheat: the child is *compute*, not automation, and must
be audited to stay that way.
**Done-when.** Voice loop runs against the served model with latency within noise of in-process, survives
a forced model-process kill without dying, and hot-swaps presets without a restart.

### 1.5 KV-cache and prompt-prefix discipline — `S`

**What.** Make prompt construction deliberately prefix-stable so `llama.cpp`'s prefix cache prefills the
static part once per model load instead of per turn.

**Why now.** Cheap, compounding, and already validated: PR #1 put the scenario taxonomy in the system
prompt specifically so the prefix cache prefills it once and only the ~15-token utterance is new per turn.
That trick should be a rule, not an accident.

**How.** Audit every prompt assembler for prefix stability, then enforce it: static persona and taxonomy
first, slowly-varying context next, per-turn content last. Add a test that asserts the prefix hash is
stable across turns for each pool. Right-size `n_ctx` from measured prompt lengths rather than habit — PR
#1 sized the 12B to `n_ctx` 2048 / `n_ubatch` 128 from a real distribution (median 251, p90 810, max 1074
over 935 turns).

**Components.**
- `src/kenning/audio/prompt_prefix.py` — NEW. Prefix/suffix split contract plus hash accounting.
- `tests/audio/test_prompt_prefix_stability.py` — NEW.

**Work.** (1) Audit `llm_prompts.py`, `ultron_prompt.py`, `scenario_prompts.py`, `_ultron_social.py`,
`reply.py`. (2) Introduce the split contract. (3) Add stability tests. (4) Re-measure prompt-length
distributions per pool and right-size `n_ctx`/`n_ubatch`.

**Integrates with.** All prompt SSOTs, the per-pool prompt registries (`ANSWER_SYSTEM_FOR`,
`_SOCIAL_SYSTEM_FOR`), the scenario router.

**Leverages.** The prefix-cache work already done for the taxonomy; the measured prompt distribution.

**Risks.** Persona edits that look harmless can break prefix stability — hence the test.
**Done-when.** Prefix hash stability is test-enforced per pool, and `n_ctx` is justified by a measured
distribution for every model role.

### 1.6 Latency as a first-class, always-on measurement — `S`

**What.** Extend per-turn stage tracing into a continuous latency budget with regression alarms.

**Why now.** Every optimization above needs a scoreboard, and the callout budget (~370–620 ms) is a
product requirement, not a nicety.

**How.** PR #1 already added per-turn stage marks keyed by turn id — so background speculative threads
contribute correctly — reported at each loop-iteration end, with `first_audio` marked at Kokoro's first
PCM write. Build on that: persist per-stage percentiles to a rolling store, define a budget per stage, and
log loudly when a stage exceeds its budget. Add an offline report that diffs two runs.

**Components.**
- `src/kenning/latency_budget.py` — NEW. Per-stage budgets + breach detection.
- `scripts/latency_report.py` — NEW. Percentiles, run-vs-run diff.
- `logs/latency/` — rolling per-turn stage records.

**Work.** (1) Persist stage marks. (2) Define budgets from the current baseline. (3) Breach logging.
(4) Reporting/diff tool. (5) Wire into the boot canary.

**Integrates with.** `trace.py`, `latency_hygiene.py`, the orchestrator loop, the boot canary.
**Leverages.** PR #1's turn-id-keyed stage marks and `first_audio`; the existing latency roadmap in
`docs/latency_optimizations_V1.md`.
**Risks.** Measurement overhead on the hot path — keep it to integer timestamps and defer formatting.
**Done-when.** A budget breach is visible in the log within the turn it happens, and two runs can be
diffed on one command.

---

## Phase 2 — Game intelligence: the Valorant assistant

This is the product. Everything else on this list is infrastructure that serves it or packaging that
sells it. Ultron already *hears* the game (voice), *speaks into* it (team relay + PTT), and *broadcasts
around* it (Twitch + overlay). What it does not yet have is **knowledge of the game being played** — who
is in the lobby, how the last match went, what the meta is, who these people are, and whether one of them
is live on Twitch right now.

> **Provenance note.** The items in this phase come from the streamer's own feature list rather than from
> the seeding conversation. They are written as proposals with concrete build paths; each should be
> confirmed or corrected before it gets a spec.

### 2.0 The data-source risk ladder — read this before designing anything here — `S` (policy)

Every feature in this phase is a question about **where the data comes from**, and that question is an
anticheat question first and an engineering question second. Vanguard is a kernel host-integrity monitor
and the project's P0 rule (`BR-P1`) is absolute. So the ladder is defined once, here, and every item below
declares which rung it stands on.

| Rung | Description | Examples | Policy |
|---|---|---|---|
| **GREEN** | Web APIs about *past* matches, public esports data, and the streamer's own account. Zero contact with the running game or its process. | Riot Developer API, unofficial match-history APIs, vlr.gg, Liquipedia, Twitch Helix, patch notes | Allowed. Default-ON candidates. |
| **AMBER** | Reads *local* state belonging to the Riot client — the lockfile plus localhost HTTP endpoints — to learn about the **current** lobby. No memory access, no injection, no hooks. | The local client API used by rank-checkers to show party/enemy ranks in agent select | **Explicit user decision required. Default OFF. Never on the default import path.** See the honest risk note below. |
| **RED** | Anything that touches the game process or automates play. | Reading game memory, DLL injection, hooking, in-process overlays, input automation for gameplay, aim/trigger assistance | **Never.** Ban-class, violates `BR-P1`, and out of scope permanently. |

**The honest AMBER risk note.** The local-client-API approach is technically modest — it reads a file the
Riot client writes and makes HTTP calls to `127.0.0.1`. It does not read memory and does not inject. A
large ecosystem of rank-checker tools uses it. However: Riot has been explicit that surfacing pre-game
information about other players is something they can act against, and "widely tolerated" is not
"sanctioned." The engineering risk is low; the *policy* risk is real and is the streamer's call, not the
system's. Therefore: default OFF, a separate opt-in flag, lazy-imported inside its own gate behind the
import firewall, excluded from the lean gaming boot, and covered by the anticheat grep scanner so it can
never drift onto the voice path. If the answer is "not worth it," every other item in this phase still
works — they just operate on post-match data instead of live lobby data.

**Done-when.** The ladder is encoded as a config-level policy with per-source rung declarations, and
`tests/safety/test_anticheat.py` is extended to assert no AMBER/RED source is reachable from the voice
path or in gaming mode.

### 2.1 Match history and player statistics — `L`

**What.** Ultron knows how you actually played: per-match results, per-agent and per-map performance,
K/D/A, headshot rate, first-blood rate, econ rating, rank and RR movement over time — and can answer
questions about it in persona, out loud, without you leaving the game.

**Why now.** It is the foundation of the whole phase. Dossiers (2.2), coaching (2.6), stream cards (2.7)
and tilt detection (2.8) all read from the same match store. Build the store once.

**How.** GREEN rung throughout. Two source options, and the design should support both because their
availability differs:

- **Official Riot Developer API** — the correct long-term source. VAL endpoints exist
  (match, content, ranked, status). The practical catch is access: personal keys are short-lived and the
  richer VALORANT endpoints are gated behind an application/approval process. Plan for it, do not depend
  on it landing quickly.
- **Community match-history APIs** — the pragmatic near-term source, keyed by Riot ID (`name#tag`),
  returning match lists, per-round detail, and MMR/RR history. Rate-limited and unofficial, so they must
  sit behind the existing circuit-breaker pattern and degrade to "I do not have that yet" rather than
  failing loudly mid-game.

Poll after a match ends rather than during play — the natural trigger already exists, because the system
knows when the streamer stops issuing callouts and gaming mode disengages. Store normalized matches in
SQLite (the durable-store pattern already used for the welcomed-chatter store), and write a compact
natural-language match summary into Qdrant so the existing RAG path can answer fuzzy questions ("how do I
do on Ascent lately?") without bespoke query code.

Alternatives: OCR the in-game scoreboard (rejected — needs screen capture, which is anticheat-gated and
strictly worse than an API); manual entry (rejected — nobody will do it).

**Components.**
- `src/kenning/game/` — NEW package, the whole phase lives here.
- `game/riot_client.py` — official API client, circuit-broken, key-rotation aware.
- `game/match_source.py` — source abstraction so official/community/offline-fixture are interchangeable.
- `game/store.py` — SQLite schema: `matches`, `rounds`, `participants`, `rank_history`.
- `game/stats.py` — derived aggregates: per-agent, per-map, per-side, rolling form.
- `game/summarize.py` — match → short natural-language summary for Qdrant.
- `scripts/game/backfill_matches.py` — one-shot history import.
- `tests/game/` — with recorded API fixtures so tests never hit the network.

**Work.** (1) Store schema + migrations. (2) Source abstraction + fixtures. (3) Community client behind a
circuit breaker. (4) Official client + the access application. (5) Post-match poll trigger on gaming-mode
disengage. (6) Aggregates. (7) Qdrant summaries. (8) Voice scenarios for the top ~10 questions.

**Integrates with.** Gaming-mode engage/disengage, Qdrant memory, the scenario router (new scenarios),
`agent_kits.py` for agent naming, the capability broker once it exists (external HTTP = T3).

**Leverages.** The circuit-breaker + error-phrase machinery already wrapping Brave/Jina; the durable
SQLite store pattern; the RAG retrieval path; the existing agent vocabulary and STT gazetteer, which
already knows agent and map names.

**Risks.** Unofficial API terms and rate limits; Riot ID changes; region/shard handling. Key material must
follow the existing secrets rules — never committed, env or `~/.kenning/` only.
**Done-when.** After a real match, Ultron answers "how did I just do" and "what's my Ascent win rate" from
stored data, with the network path circuit-broken and tests running offline on fixtures.

### 2.2 The player dossier — remembering teammates and enemies — `L`

**What.** A persistent memory of the people you play with and against: who they were, what they played,
whether you won, how often you have crossed paths, and freeform notes — so Ultron can say "you have
queued with this one four times, you have never lost" or "this is the Chamber who dropped 30 on you last
Tuesday."

**Why now.** It is the single most distinctive feature on the list and the one most likely to make people
say "I want that." It also falls out almost free once 2.1 exists, because match participants are already
being stored.

**How.** GREEN rung when built from post-match data — the participant list of every match you played is
already in the store. Identity is the hard part: Riot IDs can change, and the same display name is not a
stable key. Use the account PUUID where the API provides it and treat the display name as mutable
metadata, so a rename does not fragment a dossier.

The dossier record accrues: encounters (match id, side, agent, outcome), aggregates (times allied, times
against, your win rate with/against them), and notes. Notes come from two places — explicit voice
("remember this Jett, they were smurfing") and derived observations (statistical outliers relative to the
lobby's rank). Store notes in Qdrant so recall can be semantic rather than exact-match.

If AMBER is enabled, dossiers become *pre-game* rather than post-game — Ultron recognizes a name in agent
select. If not, recognition happens at the post-match debrief, which is still genuinely useful.

**A real ethical question, flagged deliberately.** This is a local database about other people, built
without their knowledge, on a system that broadcasts to a live audience. Local storage is defensible;
*announcing* it on stream is a different act. The design should separate "Ultron knows" from "Ultron says
it out loud on stream," default the second to OFF, and never speak another player's statistics to chat
without an explicit toggle. This is worth deciding on purpose rather than discovering later.

**Components.**
- `game/dossier.py` — record model, identity resolution, merge-on-rename.
- `game/encounters.py` — derive encounters from stored matches.
- `game/notes.py` — voice-captured and derived notes; Qdrant-backed.
- `game/privacy.py` — what may be spoken, to which channel (private vs team vs stream).

**Work.** (1) Identity model on PUUID with display-name history. (2) Encounter derivation backfill.
(3) Aggregates. (4) Voice command to attach a note. (5) Semantic note recall. (6) Channel-aware privacy
gate. (7) Retention/forget command.

**Integrates with.** The match store, Qdrant, the relay/private-reply channel split (which already
distinguishes team mic from private reply from chat), the Twitch chat path.
**Leverages.** The channel routing already built for relay vs private vs chat; the Qdrant facts
collection; the fuzzy-name matching already used for chatter names in the tell-chat feature.
**Risks.** Identity fragmentation on rename; the privacy/ethics dimension above; storage growth.
**Done-when.** Ultron recognizes a repeat player from stored history and reports the relationship, with
stream-announcement default OFF and a working "forget this player" command.

### 2.3 Live match context — the AMBER tier — `M` (gated on a policy decision)

**What.** Knowing the *current* lobby: who is in it, their ranks, agent-select state, map, and score —
enabling pre-game commentary and in-match awareness rather than post-match reporting.

**Why now.** It is what makes the assistant feel present rather than retrospective. It is also the only
item in this phase that carries policy risk, which is why it is isolated behind its own decision.

**How.** AMBER rung. Read the Riot client lockfile for local credentials, then call the client's localhost
endpoints for pre-game and current-match state. Strictly read-only; no writes, no game process contact.

Everything about the implementation must make it impossible for this to leak onto the protected path:
its own config flag defaulting OFF, lazy import inside the gate, absent from the lean gaming boot, an
explicit entry in the anticheat scanner's expectations, and a boot-canary line stating whether it is
active. If the streamer decides against it, the module simply never loads and the rest of the phase is
unaffected.

**Alternatives.** Post-match only (GREEN, no risk, less magic — this is the fallback). Voice-driven entry,
where the streamer simply *says* the enemy comp and Ultron remembers it (GREEN, zero risk, surprisingly
workable given the relay already parses agent names).

**Components.** `game/live/lockfile.py`, `game/live/client_api.py`, `game/live/session.py`, plus a
dedicated `tests/game/test_live_gate.py` asserting the module is unreachable when the flag is OFF.
**Work.** (1) Policy decision — this blocks the rest. (2) Lockfile + local client reader. (3) Session
model. (4) Gate + firewall + canary + scanner entries. (5) Wire to dossiers and coaching.
**Integrates with.** 2.2 dossiers, 2.6 coaching, the import firewall, the boot canary, gaming mode.
**Leverages.** The gating pattern already proven for `kenning.desktop`; the boot canary; the anticheat
grep scanner.
**Risks.** The policy risk described in 2.0. Also client-API instability across patches.
**Done-when.** A decision is recorded. If yes: the gate is proven closed by test when OFF, the scanner
passes, and the canary reports its state.

### 2.4 Meta knowledge and pro-match awareness — `M`

**What.** Ultron knows the current patch, agent/map meta, tier lists, and what is happening in pro play —
upcoming matches, recent results, standings — and can talk about it unprompted or on request.

**Why now.** It is entirely GREEN, it reuses the web stack already built, and it makes Ultron a better
stream companion during downtime, which is when a co-host earns its keep.

**How.** Three sources, all through the existing search/reader cascade rather than new infrastructure:
patch notes from the official site (the Trafilatura → Jina reader chain already handles this shape);
pro-match data from community esports APIs and wikis; and aggregate meta statistics from public stat
sites. Cache aggressively — the existing web-results cache already distinguishes volatile from stable TTLs,
and patch notes are extremely stable while live scores are not.

Fold the result into the LLM context the same way search results already are, and let the existing
freshness-intent detection ("needs fresh data") route meta questions to a refresh automatically.

**Components.** `game/meta/patch_notes.py`, `game/meta/esports.py`, `game/meta/tierlist.py`,
`game/meta/cache.py` (thin wrapper on the existing web cache with game-specific TTLs).
**Work.** (1) Patch-note ingestion + change summarization. (2) Esports schedule/results client.
(3) Meta-stat ingestion. (4) TTL policy. (5) Prompt injection format. (6) Scenario-router entries.
(7) Optional proactive "match starting in 10 minutes" notice.
**Integrates with.** The web search gate and executor, the reader cascade, the results cache, the intent
recognizer's freshness phrases, the Twitch chat path for posting schedules.
**Leverages.** SearxNG → Brave → DuckDuckGo cascade, Trafilatura → Jina readers, the volatile/stable TTL
cache, the deep-research loop for "explain this patch" style questions.
**Risks.** Scraping fragility and terms-of-use on stat sites; prefer APIs where they exist and cache hard.
**Done-when.** Ultron correctly answers "what changed in the last patch" and "who is playing tonight"
from cached sources, with no network call on a repeat question inside TTL.

### 2.5 Finding streamers in your games — `M`

**What.** When someone in your lobby is live on Twitch, Ultron tells you — and optionally tells chat.

**Why now.** It is a genuinely delightful feature, it is nearly free given the Twitch integration already
built, and it creates real stream moments.

**How.** GREEN on the Twitch side entirely. Take the player list (post-match from 2.1, or pre-game if 2.3
is enabled), then resolve names to live channels. Two mechanisms, used together:

1. **Live-stream scan** — query Twitch for channels currently live playing VALORANT and match against the
   lobby's names. This is the cheap, high-precision direction: the live set is bounded, and a normalized
   name match against it is fast.
2. **Channel search** — for names that do not match the live set, a targeted channel lookup catches
   streamers whose Twitch handle differs from their Riot ID.

Name matching is the hard part and the repo already has the right tool: the fuzzy chatter-name resolution
built for tell-chat, which deliberately refuses agent names, short names, and ambiguous matches. Reuse
that discipline — a false positive here ("this player is streamer X") is worse than a miss, especially on
stream.

**Components.** `game/streamers/finder.py`, `game/streamers/namematch.py`,
`game/streamers/announce.py` (channel-aware: private voice vs stream).
**Work.** (1) Live-stream scan by game id. (2) Normalized name matcher with confidence floors. (3) Channel
search fallback. (4) Confidence gate — announce only above a threshold. (5) Announcement templates.
(6) Optional auto-shoutout with an explicit toggle.
**Integrates with.** The existing Twitch Helix client and token/refresh machinery, the raid/shoutout path
already built, the overlay for a "streamer in your game" card, the dossier.
**Leverages.** The Twitch API client with proactive OAuth refresh and 401 self-heal; the `/shoutout`
implementation; `chatter_names.py`-style fuzzy resolution with its refusal rules; the overlay card system.
**Risks.** False positives are the main failure mode — gate hard on confidence. Rate limits on search.
**Done-when.** A known live streamer in a real lobby is correctly identified and announced, and a
deliberately ambiguous name is correctly refused.

### 2.6 In-game coaching and the post-match debrief — `L`

**What.** Ultron uses everything above to actually help: agent and comp suggestions grounded in your own
performance, economy and round advice, callout coaching, and a spoken debrief after each match.

**Why now.** It converts data into value. Without it, 2.1–2.4 are a database with a voice.

**How.** GREEN — all of it can run on stored data plus voice context, with no live game contact. The
debrief is the anchor feature and the easiest: on gaming-mode disengage, pull the match, compare against
rolling form, and speak a short persona-correct summary of what stood out. This is a well-shaped LLM task
with verifiable inputs, which also makes it a good candidate for the reward work in the post-training
phase.

Coaching during play must respect the hard constraint that already governs the relay: **latency and
brevity beat completeness.** The existing callout budget (~370–620 ms) and sentence caps apply. Coaching
that arrives late or long is worse than silence, so it should be opt-in, rare, and interruptible.

Agent/comp suggestions combine your per-agent stats (2.1), the current meta (2.4), and the map — a small,
well-scoped recommendation problem that does not need a large model.

**Components.** `game/coach/debrief.py`, `game/coach/suggest.py`, `game/coach/economy.py`,
`game/coach/prompts.py` (per-scenario prompt registry, following the existing per-pool pattern).
**Work.** (1) Debrief on disengage. (2) Rolling-form comparison. (3) Persona-correct templates + tests.
(4) Agent/comp suggestion. (5) Opt-in in-match tips with a hard rate limit. (6) Scenario-router entries.
**Integrates with.** Gaming mode, the relay path and its caps, the scenario router, the flavor/verbosity
system, the persona lock.
**Leverages.** The per-pool prompt registries, the verbosity axes, the sentence/char caps and
`strip_prompt_echo`, the callout latency work, the agent kits.
**Risks.** Coaching that is generic is worse than none — ground every claim in the streamer's own numbers.
Persona drift when the content is analytical; pin with persona-lock tests.
**Done-when.** A post-match debrief speaks within seconds of match end, cites real numbers from the store,
and passes the persona lock.

### 2.7 Game intelligence on stream — `M`

**What.** Surface all of the above to the audience: overlay cards for live stats and rank, chat commands
(`!rank`, `!stats`, `!lastmatch`, `!record`), and automatic milestone callouts.

**Why now.** The stream stack is the most built-out part of the system and this is the cheapest way to
make the game intelligence visible to someone other than the streamer.

**How.** The overlay already renders compact unified cards with hi-DPI density and an operator `&scale=`
knob; adding stat cards is a template and a data feed, not new infrastructure. Chat commands slot into the
existing command router alongside `!song`/`!album`/trivia. Milestones (rank up, personal best, win streak)
fire from the match store on ingest and post through the existing pinboard/poster consolidation so they
never become a chat flood — a lesson already learned and fixed once.

**Components.** `game/stream/cards.py`, `game/stream/commands.py`, `game/stream/milestones.py`, plus
overlay templates.
**Work.** (1) Card templates. (2) Data feed to the overlay. (3) Chat commands with the existing cooldown
discipline. (4) Milestone detection + throttling. (5) Economy hooks if stats become redeemable content.
**Integrates with.** The overlay server, the Twitch chat command router, the economy/redeem system, the
pinboard, the relay-aware chat cooldown.
**Leverages.** The entire Twitch subsystem — overlay cards, command routing, cooldowns, pinboard,
StreamElements economy, the write sidecar.
**Risks.** Chat flooding (already solved once — reuse the pinboard pattern, do not reinvent it).
**Done-when.** `!rank` and `!lastmatch` answer from the store, a stat card renders in OBS, and a milestone
fires without flooding chat.

### 2.8 Performance and tilt tracking — `M`

**What.** Longitudinal awareness of how you are actually doing — form trends, session length, performance
decay across a session, time-of-day patterns — and the judgment to say something about it.

**Why now.** It is the highest-value thing a personal assistant can do that a stat site cannot, because it
requires *continuity* and *presence*. It also needs the most data, so it should start collecting early even
if the feature ships late.

**How.** GREEN. Derived entirely from the match store plus session timing the orchestrator already knows.
Detect within-session decay (performance in the last N matches versus the session's first N), consecutive
losses, and unusually long sessions. Then — carefully — say something.

The delivery is the hard part, not the detection. A cold machine intelligence observing that your aim has
degraded for three straight matches is either excellent or insufferable depending entirely on frequency
and tone. Rate-limit hard, make it private-channel by default (never team, never chat), and make it
switchable off in one command.

**Components.** `game/form/trends.py`, `game/form/session.py`, `game/form/nudge.py`.
**Work.** (1) Session model. (2) Trend detection with statistical floors so noise does not trigger it.
(3) Nudge policy — rate limits, channel, opt-out. (4) Persona-correct phrasing. (5) Long-horizon reporting
("your best month on Jett").
**Integrates with.** The match store, gaming mode session boundaries, the private-reply channel, the
verbosity system.
**Leverages.** Gaming-mode engage/disengage as natural session boundaries; the private-reply path; the
existing cooldown patterns.
**Risks.** Being annoying is the primary failure mode and it is a product risk, not a technical one.
Default conservative.
**Done-when.** A simulated declining session triggers exactly one private nudge, and an opt-out command
silences it permanently.

### 2.9 Beyond Valorant — the game abstraction — `M` (defer until 2.1–2.8 are real)

**What.** Factor the game-specific parts behind an interface so a second title becomes a data/adapter job
rather than a rewrite.

**Why now.** Deliberately *last* in this phase. Premature generalization here would slow down the thing
that actually matters. But the eventual shape is worth knowing while building 2.1–2.8, so the seams end
up in sensible places.

**How.** The natural seams: a match-source adapter, a game vocabulary (agents/maps/roles → champions/maps,
or whatever the next title calls them), a meta source, and a coaching prompt set. The relay, dossier,
store, streamer-finder and stream surfaces are all game-agnostic once those four are pluggable.

**Components.** `game/adapters/base.py`, `game/adapters/valorant/`, `game/vocab.py`.
**Work.** (1) Extract the interface from the working Valorant implementation. (2) Move vocabulary out of
the STT gazetteer into the adapter. (3) Prove it with a second title's read-only match history.
**Integrates with.** The STT gazetteer and routing vocabulary, `agent_kits.py`, the scenario taxonomy.
**Risks.** Doing this too early. The rule is: build one game properly first.
**Done-when.** A second title's match history imports and answers basic stat questions with no change to
the core.

---

## Phase 3 — Post-training: an Ultron that improves from its own traces

The seeding conversation landed on the right answer: pretraining is still next-token prediction, but the
frontier for *behavior* is post-training — outcome-based RL with verifiable rewards, combined with
supervised tuning and distillation. This phase makes that concrete for a local, single-user system, where
the enormous advantage is that **we own the entire trace of every turn** and a great many of our tasks
have *checkable* answers.

### 3.1 The evaluation harness comes first — `L`

**What.** A single, versioned evaluation suite that scores a candidate model or prompt change across
every pool the system actually serves, producing one comparable scorecard.

**Why now.** Nothing else in this phase is meaningful without it. Training without evaluation is
guessing with extra steps. This is also the prerequisite that makes Phase 1's bake-offs trustworthy.

**How.** The repo already has the raw materials scattered across purpose-built harnesses: the 25k-command
corpus audits, the ~239-command battery replayed through real dispatch, the full-pipeline-from-raw-audio
injection harness, the 144-case labelled routing corpus with 36 confusable and 15 negative cases from PR
#1, and `scenario_scorecard.py`. Unify them behind one runner and one report format.

Critically, include the *adversarial* sets that catch the failures this system actually has: proper-noun
and agent-name preservation, persona leakage ("never name a vendor/model"), relay-vs-private addressing,
prompt-echo, fact invention, and the question-shaped veto.

**Components.**
- `scripts/eval/run_suite.py` — NEW. One entry point, one scorecard.
- `src/kenning/eval/` — NEW package: `pools.py`, `metrics.py`, `report.py`, `regression.py`.
- `tests/data/eval/` — versioned eval sets per pool.
- `docs/vision/eval_scorecards/` — checked-in historical scorecards.

**Work.** (1) Inventory existing harnesses. (2) Define the scorecard schema. (3) Port each harness to it.
(4) Add adversarial sets. (5) Baseline the current build. (6) Wire into the CI workflow
(`.github/workflows/enforce.yml`).

**Integrates with.** `scripts/relay_test/`, `scripts/_pool_parity_harness.py`, the golden digest and
flavor lint, `scripts/run_tests.py`.
**Leverages.** Every corpus already built — this is mostly consolidation, not new measurement.
**Risks.** Eval sets rot; version them and treat a change as a deliberate re-bless like the golden digest.
**Done-when.** One command produces a scorecard comparable to any historical scorecard, and CI runs it.

### 3.2 Trace capture and dataset construction — `M`

**What.** Turn the system's own operation into a clean, labelled, privacy-respecting training corpus.

**Why now.** The data has to exist before any tuning can use it, and capture should start early so the
corpus is large when training begins.

**How.** Every turn already emits rich structure — raw STT, normalized text, routing decision, gate
verdict, retrieved memory, prompt, generated output, TTS timing — across `trace.py`, the typed event bus,
`logs/flagged_turns.jsonl` from the stop-window FLAG button, and the various audit JSONLs. Formalize this
into a single append-only trace record per turn with a stable schema, then build a curation tool that
turns traces into training pairs.

The **FLAG button is the highest-value signal in the system** and is currently under-exploited: it is a
human "that was wrong" label produced at zero friction during live play. Extend it into a small taxonomy
(wrong route / wrong content / wrong persona / too slow / should not have spoken) so flags become
directly usable as training and eval data.

Privacy is a first-class constraint: traces contain viewer names, chat content, and voice transcripts.
Local-only by default, with explicit redaction before anything is ever shared.

**Components.**
- `src/kenning/traces/record.py` — NEW. Canonical per-turn record schema.
- `src/kenning/traces/writer.py` — NEW. Append-only, rotation, background write.
- `src/kenning/traces/redact.py` — NEW. Name/PII redaction.
- `scripts/traces/curate.py` — NEW. Trace → training pair, with filters.
- `src/kenning/audio/flag_taxonomy.py` — NEW. Structured flag reasons.

**Work.** (1) Schema. (2) Writer with a background queue. (3) Extend FLAG to a reason taxonomy in the
stop window. (4) Redaction. (5) Curation CLI. (6) Retention policy.

**Integrates with.** `trace.py`, the bus, the stop-window GUI, `logs/flagged_turns.jsonl`, Qdrant memory.
**Leverages.** The FLAG button, the event bus, existing audit JSONLs, the write-queue pattern from
`ConversationMemory`.
**Risks.** Disk growth; PII. Both handled by rotation, retention, and redaction-by-default.
**Done-when.** A week of live play yields a curated, redacted dataset that the eval suite can consume.

### 3.3 Distillation — shrink the good model into the fast one — `L`

**What.** Use a large model (local 12B, or an offline larger one) as a teacher to fine-tune a small fast
student for the specific, narrow jobs the system does constantly.

**Why now.** It is the highest-leverage training technique for this deployment. Most turns are *narrow*:
classify a scenario, rewrite a callout in persona, produce a three-sentence chat reply. Narrow jobs
distill extremely well — a tuned 1–4B can match a general 12B on a narrow task at a fraction of the cost.

**How.** Per-job students rather than one general student. The scenario router is the ideal first target:
it is a closed-label classification problem, it already has a 144-case labelled corpus and a scorecard,
and PR #1 measured that a 4B gets 97.2% while a 1B gets 37.5%. **A distilled 1B that reaches 4B accuracy
at 94 ms is a large, measurable win** — and it frees the 4B's VRAM.

Second target: the relay rewriter, where persona consistency matters more than general capability.

Training runs offline on the same box while not gaming, or on rented GPU time. LoRA/QLoRA keeps this
tractable; adapters merge into GGUF for serving.

**Components.**
- `scripts/train/distill_router.py` — NEW.
- `scripts/train/build_pairs.py` — NEW. Teacher-generated + trace-mined pairs.
- `src/kenning/llm/adapters.py` — NEW. Adapter registry and per-job model selection.
- `docs/vision/distillation_runs.md` — NEW. Run log with results.

**Work.** (1) Pick the router as target one. (2) Generate teacher labels over a large unlabelled utterance
set. (3) Mine hard negatives from confusable cases and FLAG data. (4) Train LoRA. (5) Evaluate on the
held-out scorecard. (6) Convert to GGUF. (7) Ship behind a flag. (8) Repeat for the relay rewriter.

**Integrates with.** The scenario router, the eval suite, the preset system, the serving daemon.
**Leverages.** The routing corpus, the scorecard, the measured model comparison table, and the existing
Kokoro fine-tune tooling as prior art for local training runs.
**Risks.** Overfitting to the streamer's own phrasing — mitigate with held-out sets and negative cases.
Persona drift — pin with the persona-lock tests and golden digest.
**Done-when.** A distilled student matches or beats the current model on its pool's scorecard at lower
latency and VRAM, with the persona lock intact.

### 3.4 Outcome-based RL with verifiable rewards — `XL`

**What.** Apply RL where the reward is *checkable by a program*, not by a human or a learned reward model.

**Why now.** It is the frontier technique the conversation identified, and unusually it is *more*
applicable to this system than to a general chatbot, because a surprising share of Ultron's jobs have
mechanically verifiable outcomes.

**How.** Be honest about scope: full-scale RL post-training of a base model is not happening on one
4070 Ti. What *is* tractable is preference optimization (DPO/ORPO-family) and small-scale policy
improvement on narrow, verifiable tasks — which is where the value is anyway.

The verifiable rewards this system genuinely has:
- **Routing correctness** — the label is known; the router either picked it or did not.
- **Callout factuality** — a callout naming a site, agent, or count either matches the utterance's
  extractable facts or invents them. `test_relay_fact_invention.py` already encodes this idea.
- **Persona compliance** — vendor/model names, the words "assistant"/"AI"/"Kenning" in persona output, and
  register violations are all mechanically detectable. This is already a test.
- **Format compliance** — sentence caps, character caps, prompt-echo.
- **Latency** — measured, and a legitimate reward term.
- **Coding tasks** — the strongest case: tests pass or they do not. The `Verifier`'s six checks are
  already an outcome signal, and the corrective re-prompt loop is already a crude policy-improvement loop.

Build the reward functions first as *evaluators* (they are useful immediately, independent of training),
then wire them as reward signals.

**Components.**
- `src/kenning/eval/rewards/` — NEW: `routing.py`, `factuality.py`, `persona.py`, `format.py`,
  `latency.py`, `coding.py`.
- `scripts/train/preference_pairs.py` — NEW. Build chosen/rejected pairs from reward disagreement.
- `scripts/train/dpo_run.py` — NEW.
- `docs/vision/reward_design.md` — NEW. What each reward measures and its failure modes.

**Work.** (1) Implement each reward as a standalone evaluator with tests. (2) Add them to the scorecard.
(3) Mine preference pairs where two sampled generations differ in reward. (4) Run DPO on the relay
rewriter. (5) Evaluate. (6) Extend to chat-reply. (7) Consider online policy improvement for the coding
loop last, as it is the most complex.

**Integrates with.** The eval suite, the trace corpus, the coding `Verifier` and `ConversationCoordinator`,
the persona-lock tests.
**Leverages.** `test_persona_lock.py`, `test_relay_fact_invention.py`, `test_question_shaped_veto.py`,
the coding verifier's six checks, the escalation-threshold model switch.
**Risks.** **Reward hacking is the central danger** — a model optimized for "no invented facts" can learn
to say nothing. Every reward needs a paired anti-degenerate check, and the golden path must be protected
by the existing regression control set. Persona rewards can flatten variety; the repeat-degradation
incident is the cautionary tale.
**Done-when.** Reward functions ship as evaluators with tests, and at least one pool shows a measured
quality gain from preference training with no regression on the control set.

### 3.5 Continual, safe self-improvement — `L`

**What.** Close the loop: the system proposes its own improvements from observed failures, and a walled
process decides whether to adopt them.

**Why now.** The scaffolding already exists and is under-used. `src/kenning/evolution/` is a bounded,
data-only self-improvement subsystem — Tier-3-walled, zero network/shell/eval, with correction detection,
feature-request capture, command-failure capture, recurrence thresholds, guardrail monitoring over a
rolling metrics window, and an auto-revert on post-apply regression.

**How.** Feed the evolution subsystem from the Phase 3.2 trace corpus and the FLAG taxonomy rather than
only from in-session signals, and extend its proposal space from data-only to *prompt* and *threshold*
changes — still data, still no code execution, but higher leverage. Every proposal runs through the eval
suite before adoption, and the guardrail brake reverts regressions automatically.

**Components.**
- `src/kenning/evolution/proposal_eval.py` — NEW. Route every proposal through the eval suite.
- `src/kenning/evolution/prompt_proposals.py` — NEW. Prompt/threshold proposal types.
- `docs/vision/evolution_ledger.md` — NEW. Human-readable adoption history.

**Work.** (1) Wire the trace corpus in. (2) Add prompt/threshold proposal types. (3) Gate adoption on a
scorecard improvement. (4) Extend the guardrail re-check to scorecard deltas. (5) Human-readable ledger.

**Integrates with.** `evolution/`, the eval suite, the trace corpus, the safety validator.
**Leverages.** The entire existing evolution subsystem, its guardrail brake and auto-revert.
**Risks.** Autonomous writes to `src/` are AT-4/ST-RED by rule — this must stay data-only and walled.
**Done-when.** A proposal is generated from real traces, evaluated, adopted, and shown to improve a
scorecard — with a demonstrated auto-revert on an induced regression.

---

## Phase 4 — Portability: WSL, containers, and the honest verdict

### 4.1 The question, answered directly — `S` (analysis)

The original question — *"how feasible is the WSL migration without breaking any of its functionality, and
how much work would that be?"* — never got an answer. Here it is.

**A full migration of Ultron into WSL2 would break the golden path, and the effort is not justified by
what it buys.** Specifically:

| Subsystem | WSL2 status | Consequence |
|---|---|---|
| WASAPI low-latency output | Not available | The low-latency output path, live speaker mute, and the 3-way output routing are Windows-native. |
| VoiceMeeter B1/B2 buses | Windows-only | The entire team-relay/stream-bus architecture, TEAM BUS toggle, and device-swap approach do not exist in WSL. |
| Audio capture (`sounddevice`/MME) | Bridged only | Requires a PulseAudio/PipeWire bridge, adding latency and a failure mode to the most latency-critical path. |
| HID PTT dongle | `usbipd-win` attach | Possible but fragile, and it moves a proven anticheat boundary. |
| Valorant + Vanguard | Windows kernel | Cannot move. Anything that must sit beside the game stays on Windows by definition. |
| `kenning.desktop` (pywin32/UIA/mss) | Windows-only | The desktop-assistant half of the platform is Win32 by construction. |
| OBS + overlay | Windows-side | Browser source is fine over loopback, but the app is Windows. |
| CUDA | Works | The one clean part — WSL2 CUDA passthrough is genuinely good. |

Effort for a *full* migration: `XL`, and the deliverable would be a slower, more fragile system.

**The important realization: the two-PC split already delivered the actual goal.** Whatever isolation a
WSL migration was meant to buy — separating Ultron from the game, freeing the GPU from Valorant, keeping
the anticheat surface minimal — the machine move already achieved, with Ultron on `10.0.0.3`, the dongle
and Valorant on the game PC, relay audio over VBAN, and PTT over an authenticated UDP hop. That is a
better answer than WSL, and it is already live.

So the item is **not** "migrate to WSL". It is:

### 4.2 Split-plane architecture: Windows edge, portable core — `L`

**What.** Draw an explicit line between the *portable core* (routing, prompting, LLM serving, memory,
evaluation, plugins) and the *Windows edge* (audio devices, VoiceMeeter, HID, Win32 desktop control, OBS),
with a defined interface between them.

**Why now.** It is the honest version of the portability goal, it is a prerequisite for any Linux/cloud
deployment, and it is largely a by-product of Phase 1.4's out-of-process serving.

**How.** Define a device/platform abstraction layer. Today, platform assumptions are diffuse — Win32
device strings, VoiceMeeter names, and MME indices reach fairly deep. Introduce a `platform/` package with
an interface per concern (audio in, audio out, PTT, desktop control, overlay transport) and a Windows
implementation. A null/mock implementation then lets the core run and be *tested* on any OS, which is
immediately valuable for CI even if we never deploy on Linux.

**Components.**
- `src/kenning/platform/__init__.py` — NEW. Capability detection + implementation selection.
- `src/kenning/platform/audio_{in,out}.py`, `ptt.py`, `desktop.py`, `overlay.py` — NEW interfaces.
- `src/kenning/platform/windows/` — NEW. The current behavior, moved behind the interface.
- `src/kenning/platform/null/` — NEW. Headless/test implementation.

**Work.** (1) Inventory platform-specific call sites. (2) Define interfaces. (3) Move Windows code behind
them without behavior change, proven by the existing suite. (4) Null implementations. (5) Run the core
suite headless in CI. (6) Document the boundary.

**Integrates with.** `audio/`, `ptt/`, `desktop/`, `twitch/overlay/`, `config.py` device resolution.
**Leverages.** The two-PC split, which already proved a networked boundary works; the remote PTT protocol
as the template for a remote edge; `PROJECT_ROOT` anchoring.
**Risks.** A refactor of this size can damage the golden path — it must be behavior-preserving and proven
by the existing suite, in small reversible slices.
**Done-when.** The core suite runs green headless on a non-Windows CI runner with null platform
implementations, and Windows behavior is byte-identical.

### 4.3 Containerize the portable core — `M`

**What.** A container image for the portable core (serving, routing, memory, eval), with the Windows edge
outside it.

**Why now.** After 3.2, it is nearly free, and it makes reproducible evaluation, training, and eventual
cloud-optional services tractable.

**How.** CUDA base image, the core package, no audio or Win32. The voice edge talks to it over the same
local protocol used in Phase 1.4. This is also the natural artifact for renting GPU time for Phase 3
training runs.

**Components.** `docker/core.Dockerfile`, `docker/compose.yaml`, `docs/vision/container_runbook.md`.
**Work.** (1) Image. (2) Pin CUDA/torch/llama-cpp versions. (3) Volume-mount models. (4) Verify the eval
suite runs inside. (5) Document GPU passthrough on both Windows/WSL2 and Linux.
**Integrates with.** The serving daemon, the eval suite, training scripts.
**Risks.** Version drift between host driver and container CUDA. **Done-when.** The eval suite produces an
identical scorecard inside and outside the container.

---

## Phase 5 — The capability broker: the safety layer that unlocks distribution

This is the load-bearing item of the entire document. The framing from the seeding conversation is
exactly right and worth restating: **do not try to pick a "safe" model — make the system structurally
incapable of causing harm.** The model proposes; a permission layer decides. No arbitrary shell. Only
approved, typed capabilities. Confirmations for sensitive actions. Audit logs by default. The persona
layer may role-play freely; the tool layer stays locked down.

That separation is what makes an *Ultron-voiced* assistant safe to hand to someone else — and nothing in
Phases 6–8 can ship without it.

### 5.1 The capability broker core — `XL`

**What.** A single mandatory chokepoint through which every side-effecting action must pass, expressed as
typed capability requests rather than code or shell.

**Why now.** It gates all distribution. It is also cheaper to build now than after a plugin ecosystem
exists, because retrofitting a chokepoint under existing callers is far harder than routing new callers
through one.

**How.** Three separated layers:

1. **Proposal.** The model emits a typed capability request — a name plus validated arguments — never
   code, never a shell string. `{"capability": "obs.scene.create", "args": {...}}`.
2. **Policy.** The broker resolves the request against a policy: is this capability registered, enabled,
   permitted for this caller, within its rate limit, and what risk tier is it? Tiers mirror the
   already-proven agent contract: **T1** read-only → auto-allow; **T2** reversible local effect → allow +
   log; **T3** external-touching → confirm unless pre-authorized this session; **T4** irreversible →
   always confirm with the exact action shown.
3. **Execution.** Only the broker executes, through a registered handler with validated arguments,
   timeouts, and a recorded outcome.

The key property: **there is no path from model output to system effect that bypasses the broker.** The
model never gets a shell. Adding a capability is a deliberate act of registration with a declared tier.

Alternatives: allowlisted shell commands (weaker — argument injection is endless); sandboxed code
execution (more general, much larger attack surface, and hostile to the anticheat constraint); status quo
of direct calls (unshippable to third parties).

**Components.**
- `src/kenning/broker/registry.py` — NEW. Capability registration, schema, tier, handler.
- `src/kenning/broker/policy.py` — NEW. Tier resolution, allowlists, rate limits, session grants.
- `src/kenning/broker/broker.py` — NEW. The chokepoint: validate → authorize → confirm → execute → audit.
- `src/kenning/broker/schema.py` — NEW. Pydantic request/result types.
- `src/kenning/broker/confirm.py` — NEW. Confirmation transport (voice + stop-window + overlay).
- `src/kenning/broker/ledger.py` — NEW. Append-only, hash-chained audit log.
- `src/kenning/broker/capabilities/` — NEW. One module per capability family: `spotify.py`, `obs.py`,
  `twitch.py`, `desktop.py`, `files.py`, `system.py`.
- `docs/vision/capability_catalog.md` — NEW. Every capability, its tier, and its rationale.

**Work.** (1) Schema and registry. (2) Broker with tiering. (3) Ledger with hash chaining. (4) Confirmation
transport reusing the barge-in speak path. (5) Port Spotify as the pilot family — small, well-understood,
already has a clean action set. (6) Port Twitch moderation (a genuinely privileged surface). (7) Port
desktop control, still gated behind the anticheat rules and lazy-imported inside its gate. (8) Add a
"deny-by-default" test that fails if any capability lacks a tier. (9) Route the LLM's action proposals
through it. (10) Retire direct call sites, keeping them as fallbacks until proven.

**Integrates with.** `safety/` validator (which already has rules, sandbox roots, protected files,
approved outbound APIs, and an audit path — the broker generalizes it), `spotify/`, `twitch/moderation`,
`desktop/`, the coding stack, `evolution/`, the bus.

**Leverages.** A great deal already exists: the runtime safety validator with per-rule toggles and
protected paths; the AT-1..AT-4 tiering already proven in the agent operating contract; audit JSONLs;
the barge-in confirmation path; `config.py` Pydantic v2 with `extra=forbid` for argument validation.

**Risks.** A chokepoint on the hot path must not add latency — T1/T2 must be a dictionary lookup and a
log write, with no I/O. Incomplete migration is the real danger: a single un-migrated direct call site
defeats the guarantee, so the deny-by-default test is essential.
**Done-when.** No side-effecting subsystem can act except through the broker (enforced by a test that
greps for direct call sites), every capability has a declared tier, and the ledger records every action
with its decision path.

### 5.2 Prompt-injection resistance — `M`

**What.** Treat all external content — Twitch chat, web pages, file contents, tool output — as untrusted
data that can never become instructions.

**Why now.** The system already ingests hostile-by-default input: **Twitch chat is a live adversarial
channel** where viewers actively try to make the bot say things.

**How.** Defense in depth, not one filter. Structural datamarking of untrusted spans (already used in
`TWITCH_CHAT_SYSTEM`); a hard rule that untrusted content can never *originate* a capability request —
only the user's own voice can; capability requests derived from a turn containing untrusted content get
tier-escalated; and a persona-lock output filter as the last gate.

**Components.**
- `src/kenning/safety/untrusted.py` — NEW. Provenance tagging through the pipeline.
- `src/kenning/safety/injection_tests.py` + `tests/safety/test_injection_corpus.py` — NEW. An adversarial
  corpus grown from real chat.

**Work.** (1) Thread provenance through the turn record. (2) Enforce "untrusted cannot originate an
action" in the broker. (3) Tier escalation rule. (4) Build the adversarial corpus. (5) Add to CI.
**Integrates with.** The broker, `twitch/reply.py`, web search, the moderation layer.
**Leverages.** Existing datamarking, moderation, the ban-guard, and the persona-lock tests.
**Risks.** Over-blocking makes chat interaction feel dead — measure with the eval suite.
**Done-when.** The adversarial corpus passes in CI and no untrusted span can reach the broker as an
originator.

### 5.3 Blast-radius containment — `M`

**What.** Even permitted actions get bounded: filesystem scope, rate limits, spend limits, kill switches.

**How.** Sandbox roots already exist for the coding stack; generalize to every file capability. Add
per-capability rate limits and a global "stop everything" that the existing always-on "Ultron stop" path
can trigger. Any future spend-capable capability requires an explicit budget with a hard ceiling.

**Components.** `src/kenning/broker/limits.py`, `src/kenning/broker/killswitch.py`.
**Leverages.** `coding.sandbox_root`, the safety validator's protected files/dirs, the stop-window kill
switch, the circuit-breaker pattern already used for external calls.
**Done-when.** Every file/network capability has a declared scope and limit, and the kill switch halts
in-flight capability execution.

---

## Phase 6 — Kenning: platform identity and the persona system

The conclusion of the seeding conversation: **Kenning is the platform; Ultron is one persona under it.**
The architecture already reflects this — the package is `src/kenning/`, and the Ultron persona is a layer
(`audio/ultron_prompt.py`, `_ultron_social.py`, `_ultron_answer.py`, `_ultron_identity.py`, agent kits,
flavor library). The work is to make that structural truth explicit and swappable.

### 6.1 Name diligence — `S` (but blocking)

**What.** Trademark search, domain and handle acquisition, and a decision on the public name.

**Why now.** It blocks every public surface, and the advice to not fall in love with a name before
searching is correct. "Kenning" is a real word (the Old Norse poetic compound — *whale-road* for sea)
which is good for memorability and bad for exclusivity: descriptive/common words are harder to protect and
more likely already registered in software classes.

**How.** Search the relevant trademark classes (software/SaaS) in the target jurisdictions; check the
package-name namespaces that matter (PyPI, npm, GitHub org, Docker Hub); check domains and social handles.
Prepare a fallback: a coined or compound mark is far easier to own than a dictionary word. Decide whether
the public product name and the internal package name need to match — they do not. Internal
`src/kenning/` can persist regardless.

**Work.** (1) Class search. (2) Namespace availability sweep. (3) Shortlist with a coined fallback.
(4) Acquire domain + handles. (5) Record the decision.
**Risks.** Discovering a conflict *after* public launch is the expensive outcome this item exists to
prevent. **Done-when.** A decision is recorded with the search evidence behind it, and the namespaces are
held.

### 6.2 The persona system — `L`

**What.** Promote personas from hardcoded to a registered, swappable, user-selectable pack format.

**Why now.** Two independent reasons converge. Commercially, a Marvel-derived character cannot be the
product's public face. Architecturally, a persona registry is simply better design — it makes the persona
testable, versionable, and comparable, and it lets one engine serve a streamer's Ultron, a calm desktop
assistant, and a coach without forking prompts.

**How.** A persona pack is data plus optional assets: identity and register rules, per-pool system prompts
(the `ANSWER_SYSTEM_FOR` / `_SOCIAL_SYSTEM_FOR` registries already have the right shape), flavor/tail
libraries with their taxonomy, deterministic response pools, a voice reference for TTS, and a persona-lock
test spec declaring what must never appear in output. Ultron becomes the reference pack, extracted
without behavior change and pinned by the golden digest.

Crucially the **persona layer must have no authority** — it can change *how* something is said and never
*what may be done*. That is the broker's job. This separation is what allows a persona to role-play a
hostile machine intelligence safely.

**Components.**
- `src/kenning/persona/registry.py`, `pack.py`, `loader.py`, `lock.py` — NEW.
- `personas/ultron/` — NEW. The extracted reference pack.
- `personas/assistant/` — NEW. A neutral second pack, which proves the abstraction.
- `docs/vision/persona_pack_format.md` — NEW.

**Work.** (1) Define the pack format. (2) Extract Ultron behavior-identically, proven by golden digest and
persona-lock tests. (3) Loader + registry. (4) Build a neutral second pack. (5) Make selection a config
and runtime choice. (6) Per-persona voice selection. (7) Per-persona eval scorecards.

**Integrates with.** All prompt SSOTs, the flavor TailEntry library, `voice_lines.py` and the snap
registry, Kokoro voice selection, the golden digest.
**Leverages.** The per-pool prompt registries, the 1628-entry curated flavor library with its 16-situation
taxonomy and `AGENT_GENDER` handling, the golden digest and flavor lint as the safety net.
**Risks.** The flavor library is hand-curated and coherence-audited; extraction must not disturb it. The
golden digest is the guard.
**Done-when.** Two personas ship, switching is a config change, each has its own scorecard, and the Ultron
pack is byte-identical in behavior to today.

### 6.3 User-facing string audit and the rename — `M`

**What.** Separate internal identifiers from user-visible branding, then rename the public surface.

**How.** Internal package and module names stay (`src/kenning/`, `ultron_prompt.py`) — churning them buys
nothing and risks a lot. What changes is every user-visible string: README, GUI labels, overlay text, log
banners, wake word, installer, docs. Introduce a product-name constant and a persona-name constant so
"the product" and "the character" are never the same string. Note the wake word is a *user* setting
already (the custom OpenWakeWord ONNX), so persona rename and wake-word change are independent.

**Components.** `src/kenning/branding.py` (NEW), plus a lint that fails on hardcoded product names in
user-facing modules.
**Work.** (1) Inventory user-visible strings. (2) Constants. (3) Lint. (4) Update public docs/README.
(5) Decide the public repo name.
**Risks.** The persona must keep saying its persona name — the lint must distinguish persona output from
product chrome. **Done-when.** No user-visible surface hardcodes a product name, and the lint enforces it.

---

## Phase 7 — The platform surface: plugins, APIs, creator tools

Only safe on top of Phase 5. This is where "Kenning could hold multiple personalities, creator tools,
APIs, plugins, maybe cloud services" becomes buildable.

### 7.1 Plugin SDK and capability manifest — `L`

**What.** A third-party extension format: a plugin declares capabilities it provides and requests, and the
broker enforces the contract.

**How.** A plugin is a manifest plus a Python module (in-process, restricted) or an out-of-process worker
speaking the local protocol (preferred for anything untrusted — it inherits OS-level isolation). The
manifest declares provided capabilities with schemas and tiers, requested capabilities, resource limits,
and a version constraint. Install is an explicit, user-confirmed grant of exactly the requested
capabilities, shown as a readable list.

**Components.** `src/kenning/plugins/{manifest,loader,host,api}.py`, `docs/vision/plugin_sdk.md`,
`examples/plugins/`.
**Leverages.** The broker, the serving protocol, the sidecar supervision pattern, the bus.
**Risks.** In-process plugins can defeat isolation — default to out-of-process, and require explicit trust
for in-process.
**Done-when.** An example third-party plugin installs, is granted a capability set, and cannot exceed it.

### 7.2 Local API and client libraries — `M`

**What.** A documented local HTTP/WS API exposing turns, events, capabilities, and state.

**How.** Extend the existing loopback-sidecar convention into a stable, versioned, authenticated
(loopback token, like the overlay token) API. Expose the typed event bus as a WS stream — that alone
enables overlays, Stream Deck integration, mobile remotes, and dashboards without any of them touching
internals.

**Components.** `src/kenning/api/{server,routes,auth,events}.py`, `clients/python/`, `clients/js/`,
`docs/vision/api_reference.md`.
**Leverages.** The bus, the overlay token convention, the existing sidecar HTTP servers.
**Done-when.** The overlay and stop-window are themselves API clients — dogfooding proves the surface.

### 7.3 The creator tier — `L`

**What.** Productize the streaming stack that already exists into a coherent creator-facing feature set.

**Why.** This is the most *saleable* thing already built: an AI co-host with chat interaction, a points
economy, redeems, games, moderation assistance, song requests, and audio-reactive overlays. It exists and
runs live. What it lacks is setup UX, multi-tenancy, and documentation.

**How.** Onboarding wizard for Twitch OAuth and scopes (the device-flow re-auth already exists), a
configuration surface that is not YAML, an overlay theme system, per-broadcaster isolation of stores and
tokens, and a runbook. Then generalize beyond Twitch behind a platform interface.

**Components.** `src/kenning/creator/{onboarding,tenancy,themes}.py`,
`docs/vision/creator_setup_guide.md`.
**Leverages.** The entire Twitch subsystem, the overlay with hi-DPI density and the `&scale=` operator
knob, StreamElements economy integration, moderation, the pinboard, welcome/ban-guard, chat alerts.
**Risks.** Multi-tenancy touches token storage — a security-sensitive change that must go through the
broker and the secrets rules.
**Done-when.** A second broadcaster can be onboarded without editing YAML or sharing credentials.

### 7.4 The desktop assistant tier — `L`

**What.** Finish the original Kenning vision — the present, local assistant that controls the computer and
runs workflows — now safely, behind the broker.

**Why now.** This was the *first* project and remains the larger long-term market; it was deferred because
the desktop stack cannot be default-loaded beside Vanguard. The broker plus the platform split resolve
that: desktop capabilities register as high-tier, stay lazy-imported inside their gate, and are simply
absent in gaming mode.

**How.** Register the existing `desktop/` primitives as broker capabilities with honest tiers — window
placement is T2, a click that submits a form is T4. Add workflow recording and replay as a *capability
sequence*, which is inherently safer than generated code. Use the VLM click-preview gate for
visually-confirmed actions.

**Components.** `src/kenning/broker/capabilities/desktop.py`, `src/kenning/workflows/{record,replay}.py`,
`docs/vision/desktop_tier.md`.
**Leverages.** The whole `desktop/` package already built and catalog-ported — launcher, placement, UIA,
element click, dialog control, OCR, clipboard, browser automation, click preview, screen context, VLM.
**Risks.** Anticheat is absolute here: this tier must remain impossible to load while gaming, enforced by
the import firewall and the boot canary, not by convention.
**Done-when.** A recorded workflow replays through the broker with per-step tiering, and the anticheat
scanner confirms none of it is reachable in gaming mode.

### 7.5 Configuration as a product surface — `M`

**What.** Replace hand-edited YAML with a real settings experience, keeping YAML as the source of truth.

**Why.** `config.yaml` is enormous and expert-only. Every knob is documented in-line, which is excellent
for us and impossible for a customer.

**How.** Generate the settings UI from the Pydantic schema — `config.py` is already Pydantic v2 with
`extra=forbid` and rich field metadata, so the schema can drive a UI with grouping, validation, and help
text taken from the existing comments. Add profiles (gaming / streaming / desktop) and a diff view.

**Components.** `src/kenning/settings_ui/`, `scripts/gen_settings_schema.py`.
**Leverages.** The Pydantic schema, `validate_config.py`, the existing ephemeral GUI config overlay and
settings panel.
**Done-when.** Every config field is reachable and validated from the UI, with YAML still authoritative.

---

## Phase 8 — Productization

### 8.1 Packaging and install — `L`
Signed Windows installer; bundled or first-run-downloaded models with checksums; a preflight that checks
GPU, VRAM, driver, and audio devices and explains failures in plain language. Leverages the existing
launcher scripts and boot canary. **Done-when** a clean Windows machine reaches a working first turn
without touching a terminal.

### 8.2 Editions and licensing — `M`
Decide the boundary between a free local core and paid tiers. The defensible split given this
architecture: the local engine free/source-available; paid for the creator tier, persona packs, plugin
distribution, and support. Requires a license decision — the repo is currently MIT, which is generous and
hard to walk back for already-published code, so any change applies going forward and needs care.

### 8.3 Telemetry stance — `S`
Privacy-first is a *feature* here and should be stated loudly: no cloud, no telemetry, everything local.
If diagnostics are ever added they must be opt-in, local-first, and show exactly what would be sent.
This is consistent with the existing no-cloud/no-telemetry constraint and is a genuine differentiator.

### 8.4 Update channel — `M`
Signed updates with rollback, honoring the retire-don't-remove principle. The `E:\Ultron-0.1\` standalone
backup pattern — a known-good build that stays runnable while the dev build is under maintenance — is
already the right instinct and should become a product feature.

### 8.5 Cloud-optional services — `XL`
Only ever *optional*, never required: persona pack distribution, plugin registry, encrypted config sync,
and optionally hosted inference for users without a capable GPU. Each must degrade to fully-local
operation. This is the last thing to build and the easiest to build wrong.

---

## Cross-cutting workstreams

These run continuously rather than in a phase.

- **Testing and CI — `M`, ongoing.** Land the pending items already identified: ruff/mypy installation
  (ask-first), CI required checks, and `tests/test_map.txt` for test mapping. Fix the known
  full-suite-only failures — roughly 73 tests pass individually but fail together because something
  outside `tests/audio/` mutates a shared singleton. That is a real bug in test isolation and it is
  currently masking regressions. Also resolve the golden-digest divergence on `_stt_correct._PHONETIC_INDEX`
  introduced as a dependency-version artifact of the machine move.
- **Documentation — `S`, ongoing.** `docs/codebase_structure.md` is ~995 KB and cannot be loaded whole
  into any model context. It should be *split* into per-area maps with a small always-loadable index —
  which would also make it far more useful to a future contributor.
- **Security — ongoing.** Secrets stay out of the repo (already enforced); the ledger is append-only and
  hash-chained; dependencies get reviewed before addition (already ask-first).
- **Hardware ladder — `S`, analysis.** Record what changes at 16 GB / 24 GB / 32 GB of VRAM so the
  quantization and model decisions have a documented upgrade path.

---

## Sequencing summary

```
Phase 1  Inference substrate ──┬─> 1.4 serving daemon ─────┐
                               └─> 1.1/1.2/1.3/1.5/1.6     │
                                                           │
Phase 2  Game intelligence ────> 2.0 risk ladder (gates 2.3)
         THE PRODUCT             2.1 matches ─> 2.2 dossiers ─> 2.6 coaching ─> 2.8 form
                                 2.4 meta · 2.5 streamers · 2.7 stream surfaces · 2.9 abstraction
                                                           │
Phase 3  Post-training ────────> 3.1 eval suite (gates 1.x and 2.x quality claims)
                                 3.2 traces ─> 3.3 distill ─> 3.4 RL ─> 3.5 evolution
                                                           │
Phase 4  Portability ──────────> 4.2 platform split <──────┘ (needs 1.4)
                                 4.3 containers
                                                           │
Phase 5  Capability broker ────> 5.1 broker ─> 5.2 injection ─> 5.3 blast radius
                                     │  (BLOCKS all distribution; also tiers 2.x network calls)
Phase 6  Identity ─────────────> 6.1 trademark (blocking) ─> 6.2 personas ─> 6.3 rename
                                     │
Phase 7  Platform surface ─────> 7.1 plugins ─> 7.2 API ─> 7.3 creator ─> 7.4 desktop ─> 7.5 settings
                                     │
Phase 8  Productization ───────> 8.1 packaging ─> 8.2 editions ─> 8.4 updates ─> 8.5 cloud-optional
```

**The three hard gates:** the eval suite (3.1) gates every quality claim in Phases 1–3. The broker (5.1)
gates every form of distribution. The trademark decision (6.1) gates every public surface.

---

## Open questions

1. **Is the desktop-assistant tier or the creator tier the real product?** They imply different customers,
   different pricing, and different roadmap weight. The creator tier is far more built; the desktop tier
   is the original vision and the larger market.
2. **How much does the Ultron persona matter commercially?** If a neutral persona sells as well, 6.2 gets
   simpler. If the character *is* the appeal, the persona-pack format has to be genuinely good and the
   naming work becomes more delicate.
3. **Single-user local, or multi-user from the start?** Multi-tenancy is much cheaper to design in than to
   retrofit, but it is dead weight if the product stays personal.
4. **Does MoE change anything for us at 12 GB?** Answered by 1.3, and the answer may well be no.
5. **Is a Linux-native deployment ever a goal**, or is the portable core only for CI, containers, and
   training? This determines how much rigor 4.2 needs.
6. **What is the licensing intent?** The current MIT grant on published code cannot be retracted, so the
   edition strategy in 8.2 must be designed around that.
7. **How much autonomy should evolution (3.5) ever have?** Data-only is the current wall. Prompt and
   threshold proposals are a meaningful step past it and deserve an explicit decision.

8. **Is the AMBER rung (2.3, live lobby data) worth the policy risk?** This is a judgement call, not an
   engineering one, and it is the single decision that most changes how present the assistant feels.
9. **Should Ultron ever speak another player's history on stream?** Local dossiers (2.2) are defensible;
   broadcasting them is a different act. Decide deliberately rather than by default.
10. **Which game after Valorant, if any?** The answer changes how much abstraction 2.9 deserves — and
    whether it deserves any.
