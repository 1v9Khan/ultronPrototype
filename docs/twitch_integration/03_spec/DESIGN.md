# DESIGN — Ultron × Twitch (final, post-board synthesis)

Authoritative detail: `../02_board/MASTER.md` (+ `F_report.md`/`S_report.md`/`A_report.md`). This doc records the
RESOLVED open questions, the ≥2-alternative decisions, the build approach, and the honest tonight-vs-deferred split.
The board's `component_build_order` (SLICE 0–13) is the implementation plan; reproduced + annotated below.

## Findings summary (the "summarize all findings" deliverable)
The board validated the architecture and hardened it. Headline conclusions:
- **Architecture is correct**: everything in loopback sidecars cloned from `embedder_server.py`; the import-pinned
  voice process gains only thin urllib clients; flag-gated default-OFF → main byte-identical. Team isolation is the
  one CATASTROPHIC-AND-SILENT risk and must be a *tested code-capability boundary*, not prose.
- **The abliterated model is treated as actively hostile**: the persona prompt is advisory-only; the structural
  layers are the entire defense. Input+output (exchange) classifiers are the load-bearing trust boundary
  (Constitutional-Classifiers: 86%→4.4%). Every Neuro-sama ban was an *unfiltered output*.
- **The output side carries more load than the input side.** Three gaps my plan missed, now mandatory: (1) a
  **reassembly canonicalizer** (materialize acrostics/spell-outs/NATO/ciphers/IPA across words+lines+whole-batch+
  cross-account, re-screen) since the abliterated model will *obey* "spell it down the first letters"; (2) a
  **phoneme-domain L6** matching the FINAL draft against Kokoro's own yielded phonemes with word boundaries
  dissolved (cross-word concatenation slurs are invisible to text filters); (3) a **post-TTS Whisper L7** ASR
  backstop — the encoding-agnostic true sandwich.
- **Unicode Tag-block (U+E0000–E007F) + bidi + variation-selectors** are an *invisible* injection/exfil channel a
  non-refusing model decodes — must be a codepoint **allowlist strip**, not a blocklist.
- **Crescendo/multi-turn is the most practical real attack** (GPT-5 fell in 24h) → a stateful per-CHANNEL
  trajectory scanner feeding L3/L5 is a must-have, not v2.
- **Spotlighting must be DATAMARKING** (interleave a marker between words) + **CHATTER_N** token replacement so the
  8B never sees/emits a raw attacker-controlled display-name; **per-chatter ISOLATED generation** (no shared batch
  gen) defeats batch-poisoning/roleplay-hijack.
- **Fail-CLOSED-on-safety SUPERSEDES fail-QUIET-on-availability** (a guard OOM = SAFETY event → SAFE_LOCKDOWN, never
  block-and-retry-hot which would OOM the team-callout 8B). **GPU priority lanes**: relay strictly preempts chat.
- **One shared sanitizer for ALL untrusted fields** (body, username, emote-name, reward title, raid msg, redeem
  text, game slots, helper input), CI-enforced "no sink reachable without it".
- **Games/economy/commands are a deterministic trust class** (bypass the heavy LLM stack) but NEVER bypass L1+L6;
  provably-fair commit-reveal RNG decides outcomes server-side; the OBS overlay is a dumb textContent-only renderer.
- **Moderation** keeps the abliterated 8B out of the action loop entirely; roster-grounded login→user_id with a
  confirm card, role/self guard, mass-action breaker, two-phase read-back for ban+delete.

## Resolved decisions (made autonomously; documented; each reversible/flag-gated)
- **D1 Reply model — KEEP the abliterated Qwen3-8B** for chat replies (user explicitly designed around "it's an
  abliterated model + guardrails"; CONSTRAINTS pins it as the persona product). Treat it as hostile; rely 100% on
  structural layers. Add `twitch.reply_model` (default = the 8B) so a non-abliterated head is a one-config SPIKE
  later. *Alt considered:* ship a separate aligned head now — rejected: changes the persona product (user's call).
- **D2 Guard model — default Llama-Guard-3-1B Q4** (smallest, fits with margin; the empirical small-guard
  advantage), `twitch.guard.model` selectable to ShieldGemma-2B / Granite-Guardian-3B / MrGuard-3B. Code is
  model-family-aware (prompt format). Final pick + VRAM-admission SPIKE deferred to user (BR-P3). Multilingual via
  L1 transliterate-to-English + English-only output pin. *Alt:* a 7-9B guard — rejected: infeasible under 10 GB and
  benchmarks WORSE than 1-2B variants.
- **D3 Transport — hand-rolled stdlib RFC6455** EventSub client (no `websockets` dep in any venv). Unit-tested
  against synthetic frames; live SPIKE vs the Twitch CLI mock deferred. *Alt:* add `websocket-client` to a
  sidecar-only allowlist — rejected: needless dep + firewall surface; RFC6455 client is ~200 lines.
- **D4 Phonetic defense — two tiers.** Shippable now (no new deps): the L5 **reassembly canonicalizer** +
  Double-Metaphone/RapidFuzz phonetic blocklist on normalized+space-collapsed text. The **panphon/espeak
  phoneme-domain L6** + **Whisper L7** are fully-implemented sidecar modules that **fail-CLOSED** when their deps/
  stack aren't provisioned (user installs into `.venv-twitch` + the live stack). The REQUIRED guard + L5 + L6-det
  give strong defense before the phoneme/ASR layers come online.
- **D5 Helper model — ONE (the security guard).** Games/commands use deterministic closed-grammar parsing (no
  model). A Qwen2.5-1.5B games/NL helper is an enhancement behind `twitch.helper.enabled` (default OFF).
- **D6 Sidecar topology — split by token privilege**: a read-scope EventSub sidecar + a write-scope Helix sidecar +
  the guard GGUF sidecar + the overlay/economy sidecar. Chat addressing reuses the existing EmbeddingGemma client
  (chat-mode and ranked must not overlap — D7).
- **D7 Ranked gate** — `twitch.allow_during_ranked` default **FALSE**. We cannot read game state (anticheat), so
  this is enforced via the inferred BusyEstimator + manual hush + streamer discretion (documented as the single
  strongest operational mitigation).
- **D8 Refund policy** — safety-CANCELED redeems are **NOT auto-refunded** (prevents zero-cost filter-probing);
  availability/phase/parse cancels ARE refunded; repeat safety-offenders escalate to timeout. `twitch.economy.*`.

## Honest scope: tonight (built + tested) vs deferred (genuinely user-gated)
**Built tonight (anticheat-clean, no new deps, unit-tested with mocks — every committed line real, no stubs):**
config+flags (S0), L1 normalize+blocklist (S4), L5 reassembly canonicalizer + structural/exfil checks + deflection
pool, L6 deterministic phonetic (Double-Metaphone), the ChatSafetyValidator arbiter + danger-score + trajectory
scanner + audit (S6 core), team-isolation provenance taint + orchestrator enforcement + isolation tests (S7),
the stdlib RFC6455 EventSub client (S3), OAuth lifecycle (S2, mocked), Helix client + moderation safety (S11),
deterministic addressing + selection engine (S10 deterministic parts), economy SQLite-WAL ledger + provably-fair
RNG + spin-the-wheel/slots/games (S9), the overlay HTTP+WS server + textContent/CSP overlay HTML (S8), channel-
points manager + reactions (S12 parts), the red-team CI corpus + gate (S6 gate), voice-command moderation grammar,
the constitution.md, and the sidecar server scripts (fully implemented; model-backed rules fail-CLOSED until the
model is provisioned).

**Deferred to user (BLOCKED per BR-2.5 — credentials / VRAM / live stack / physical wiring):** see
`MASTER::deferred_to_user` + `00_STATUS.md` wake-up checklist. Summary: Twitch app + OAuth grants; install the
sidecar-venv model deps (Prompt-Guard-2, guard GGUF, Detoxify, Presidio/GLiNER, panphon, misaki/espeak-ng,
sqlite-vec) into `.venv-twitch`; the guard VRAM-admission SPIKE + danger-score calibration + Kokoro+Whisper L7
mondegreen calibration (BR-P3, stack stopped); OBS Browser-Source + VoiceMeeter routing + AutoMod/Chat-Delay/
Shield defaults; the ranked-policy + viewer-memory-retention + persona-existential decisions.

## Anticheat reconciliation (BR-P1) — non-negotiable
- With ALL `KENNING_TWITCH_*` flags OFF: zero Twitch/ML/network/DB modules imported into the voice process, no
  sidecar spawns, no port binds, no DB opens, no hotkey hooks → main runtime BYTE-IDENTICAL (frozen-24 yardstick).
- `tests/safety/test_anticheat.py` blocklist gains: `requests`, `aiohttp`, `websockets`, `sqlite_vec`,
  `llama_cpp`-in-main, `transformers`-in-main (the guard/PG2 live ONLY in `.venv-twitch` sidecars).
- The voice process keeps numpy+urllib+scipy+stdlib+rapidfuzz; new code is thin urllib loopback clients only.
- ZERO game video/screen capture anywhere (content-ops uses Twitch's own VOD; inferred DND uses owned signals).

## Build order (board SLICE 0–13) — see TASKS.md + tasks_manifest.json for the per-slice acceptance.
0 config/flags/canon · 1 sidecar skeleton · 2 OAuth · 3 EventSub · 4 L1 · 5 guard sidecar+admission · 6 L0–L7
arbiter (dry-run) + red-team gate · 7 team-isolation wall · 8 low-risk delight + overlay · 9 economy+games ·
10 addressing+selection+chat-reply · 11 voice-mod+review · 12 remaining games/redeems/helper/content-ops ·
13 speak-to-team (hardest-gated, default-OFF).
