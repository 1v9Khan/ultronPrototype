# FUNCTIONALITY Board — Consolidated Synthesis

## Ultron × Twitch: what to build, in what order, on the sidecar/anticheat/flag-gated spine

This report consolidates 24 functionality agents (platform/ingestion, features/games, co-host
intelligence/UX, production/ecosystem) into one authoritative build plan. The headline: **our planned
design is architecturally correct and validated by frontier precedent (Neuro-sama, Streamer.bot,
StreamElements), but it under-specifies the engineering that actually decides whether the feature ships
well.** The board's job was to find those gaps. They are concrete and fixable, and every one of them lands
cleanly inside the existing sidecar / import-firewall / flag-gated-default-OFF model.

### The one-paragraph verdict
Build it as **three or four loopback sidecars** (modeled byte-for-byte on the EmbeddingGemma `:8772`
precedent — `scripts/embedder_server.py` + `subprocess/sidecar_lock.py` + the `KENNING_EMBEDDER_PARENT_PID`
parent-death deadman), with the anticheat-pinned voice/relay process gaining **only thin `urllib` loopback
clients** and zero new imports (R21/BR-P1). Read chat via **EventSub WebSocket** (a hand-rolled stdlib
RFC6455 client — no `websockets` dep), with **token refresh, the 10s subscribe-or-die window, keepalive,
overlap-then-swap reconnect, and message-id dedup** all handled (the four gaps the plan omits). Make
**deterministic metadata the first addressing signal** and the EmbeddingGemma classifier only the
tiebreaker. Make the **selection engine** (dedupe→cluster→prioritize→token-budget) the missing brain
between "buffer chat" and "one LLM call." Treat **commands/games/economy as a deterministic trust class
that bypasses the heavy LLM safety stack but NOT the L1 normalize + L6 TTS phonetic choke point**. Ship the
low-risk delight first (regular-recognition, hype-train reactions, spin-the-wheel + leaderboard) and defer
the speak-to-team redeem to last, hardest-gated.

---

## 1. What to build first (risk-sequenced roadmap)

Frontier evidence (Neuro-sama is #1 on Twitch at ~162k subs on a ~2B model — **system design, not model
size, is the moat**) plus the safety reality (every Neuro-sama ban was an *unfiltered output*, not a bad
model) dictate a risk-first sequence. Ship delight early while the dangerous paths stay flag-gated.

**Phase 0 — Spine (no user-facing feature, but everything depends on it).**
1. `config.py` `twitch` section (Pydantic v2, `extra=forbid`) + `KENNING_TWITCH_*` flags, all default-OFF (AT-3 ask-first).
2. Sidecar skeleton(s) cloned from `embedder_server.py`: loopback-only bind, parent-death deadman, `sidecar_lock` boot-reap, fail-quiet.
3. OAuth token lifecycle: `~/.kenning/twitch.json` (same secret class as `spotify.json`, gitignored, BR-6.6), `/validate` health check, proactive refresh (<5 min to expiry), refresh-then-retry-once on 401, `revocation`-message handling with operator alert.
4. EventSub WebSocket state machine (CONNECTING→WELCOMED→SUBSCRIBING-in-10s→ACTIVE) with keepalive watchdog, overlap-then-swap reconnect, message-id dedup LRU.

**Phase 1 — Low-risk, high-delight (read-mostly; ships value even with chat-REPLY mode OFF).**
5. **A deterministic "CLIP THAT" / "MARK THIS" voice + Stream-Deck command** — highest ROI, lowest risk, no ML. Fire a Twitch Stream Marker instantly (defer clip production to a post-stream batch — the Create Clip API has a 2-3 min `vod_offset`-population delay).
6. **Hype-train reaction engine** (read-only EventSub, zero write scopes): Ultron voices an in-persona callout + drives an overlay on level-ups. Hype Trains are the #1 content moment (+30% retention, 2.5x subs, >1/3 of viewer spend). **Pin to v2 — v1 was 410'd / withdrawn 2026-01-22.**
7. **Sub/bit goal overlay + gamified subgoals** (`channel.goal.*`, no scope) on the local HTTP+WS overlay server.
8. **Spin-the-wheel + slots + leaderboard** — the simplest game state, exercises ledger + provably-fair RNG + overlay + config end-to-end.
9. **Regular-recognition / viewer memory** — the single biggest "it feels alive" lever; reuses EmbeddingGemma, all sidecar-side, anticheat-clean.

**Phase 2 — Chat replies + interactive (the risky core; guard model REQUIRED).**
10. Semantic addressing (deterministic-first cascade) → selection engine → batch by-name reply through the full L1→L6 safety sandwich → BroadcastSink/stream bus only.
11. Voice-commanded moderation with the 2nd-monitor gray-zone review loop (roster-grounded name resolution).
12. Heist/duel/trivia/raffle games; channel-point redeems; chat commands/timers/alerts.

**Phase 3 — Hardest-gated, maybe never.**
13. The **speak-to-team channel-point redeem** (chat → team mic): default-OFF, allowlist/manual-approve only, **no Stream-Deck/hotkey binding** so it is never one press from the team path, treated as AT-4.

---

## 2. Feature set, by cluster

### F1 — Platform / ingestion (the plumbing)

**Read chat via EventSub WebSocket, not IRC.** IRC/TMI is the deprecated path; `channel.chat.message`
(scope `user:read:chat`) is canonical and — critically — **pre-parses every message into typed
`fragments[]` (text/emote/cheermote/mention) plus first-class `badges`, `reply`, `cheer`, `message_type`.**
This removes brittle string parsing and hands the addressing layer spoof-proof signals. A **hand-rolled
stdlib RFC6455 client** (`socket`+`ssl`+`hashlib`+`base64`+`os`+`struct`, ~150-250 lines) is feasible and
the right call (no `websockets` dep to ask-approve, nothing new for the firewall to block) — but it MUST
handle control frames (auto-PONG server PINGs or eat close code 4002), stay receive-only (any app frame =
4001), and dedupe on `metadata.message_id` (delivery is at-least-once).

**The four gaps the plan omits, all fixable:** (1) **token refresh** — user tokens expire ~4h; without
proactive refresh chat goes silently deaf mid-stream (the top functional gap); (2) the **10s
subscribe-or-die** window after `session_welcome` (else close 4003); (3) **`message_id` dedup** before the
abliterated 8B sees a duplicate; (4) **cost/limit budget** — a single authorized chat sub is cost 0 and
fits the 3-conn / 300-sub / 10-cost ceilings with huge headroom.

**Moderation = a small REST surface, no idempotency keys.** ~12 Helix endpoints under
`/helix/moderation/*` with **granular per-action scopes** (`moderator:manage:banned_users`, not the
deprecated `moderation:manage`). Every write needs `broadcaster_id` AND `moderator_id == token user_id`;
an app token cannot moderate. Ban/timeout share `POST /moderation/bans` (duration present = timeout); delete
vs clear-chat share `DELETE /moderation/chat`. **Make the sidecar self-idempotent** (treat 409/"already" as
success, key outbound by `(action,user,message_id)`, never blind-retry a POST). **Capture `message_id` at
ingest** — single-message delete needs it and it expires at 6h. **AutoMod-held ids come ONLY from EventSub
`automod.message.hold`** (no REST get-held endpoint). Prefer **Shield Mode + AutoMod overall_level=4 +
chat-settings PATCH** as the server-side "raid lockdown" panic button over mass one-by-one timeouts (which
burn the ~800/min Helix bucket).

**Channel Points / Bits / Hype-Train = pure REST + EventSub.** Hard caps: 50 rewards/channel; an app can
only manage rewards IT created (403 on others); status string is **`CANCELED` (one L)**; only UNFULFILLED
redemptions are refundable, and **only queued rewards (`should_redemptions_skip_request_queue=false`) can
be refunded** — so anything Ultron might reverse (failed game, safety-trip, speak-to-team) MUST be a queued
reward. Use `channel.bits.use` (2024+) as the primary Bits trigger (supersedes `channel.cheer`; subscribing
to both double-fires).

**ToS / compliance is a hidden landmine.** The abliterated 8B's output is judged by Community Guidelines
**like a human, no AI exemption** — so L5 is a *compliance* control, not a taste one. Verified-bot review
is **PAUSED indefinitely**; architect for the unverified **20 msgs/30s AND the hard 1 msg/sec/channel cap**
(the latter is what the batch-reply design actually hits — emit ONE consolidated reply per second
addressing multiple chatters). Use a **dedicated bot account** (Helix Send-Chat-Message + `channel:bot`
→ auto Chat-Bot-Badge = transparency) and **fail-CLOSED on suspension** (HALT, never auto-recreate — that
reads as ban evasion and escalates to the streamer's main account). Honor DSA data rules: TTL/ring-buffer
chat retention, per-viewer opt-out, purge kill-switch.

### F2 — Features / games (Ultron as a content tool)

**Games are a DIFFERENT trust class than free-form chat** — this is the board's key reframe. A `!slots 100`
command carries no injection/toxicity risk; it is a closed grammar parsed deterministically and does NOT
need Prompt-Guard, the guard model, or the input/output LLM sandwich. The only LLM exposure is *optional
persona flavor narration of a pre-decided result*, which still passes L5/L6 + username sanitization. The
two genuinely hard parts are **(a) a single authoritative, transactional, idempotent points ledger** and
**(b) provably-fair seeded RNG** so the streamer can rebut live "it's rigged" accusations.

- **Economy:** run a SECOND fully-local "Ultron currency" (you can't change Twitch channel-point earn rates), as an **append-only event-sourced ledger** (SQLite WAL, `synchronous=FULL` on money commits, idempotency key per mutation, balances are a rebuildable projection). Watch-time faucet + sub-tier multipliers; **net-negative-EV gambling (house edge, RTP ~0.90) + a hard per-user per-stream loss cap** so the economy can't hyperinflate and a viewer can't be wiped in one tilt. Layered anti-farming (presence + chat-activity window + account/follow-age + per-user cap + exclusion list + bot-farm anomaly flag). Transfers OFF by default (the #1 alt-laundering vector).
- **Spin-the-wheel (the flagship game):** the single most important correction — **the sidecar's secure RNG picks the winner FIRST, then the overlay animates deterministically to a server-supplied target angle.** The popular "spin with random velocity and read where it stops" pattern is unfair and client-riggable. Arc widths must be proportional to weight (no hidden-weight equal arcs). Use crypto RNG (`secrets`/`os.urandom`, never Mersenne `random`) + commit-reveal HMAC-SHA256 fairness + a `!verify` command. **"Lose ALL points" is the highest grief vector** — small weight, explicit opt-in/high-stakes scoping, streamer kill-switch, a manual voice-refund path (store pre-wipe balance), optional cap. Treat it like the speak-to-team redeem's blast radius (AT-4-class).
- **Overlays:** ONE OBS Browser Source pointed at `http://127.0.0.1:PORT/overlay` (CEF is the #1 perf bottleneck; OBS 31/CEF 127 regressed multi-video/iframe — never stack video sources). Animate alerts with **CSS transform/opacity only** (compositor-only, GPU-cheap; <100 DOM nodes), reserve Canvas/WebGL for particle/wheel effects. Auto-reconnect WebSocket (backoff+jitter) + full state re-sync on reconnect (OBS scene flips reload the source). **Route chat-reply TTS through Kokoro/BroadcastSink, NOT browser-source audio** ("Control Audio via OBS" stutters and leaks when hidden).
- **Commands/timers/counters/quotes/song-request:** adopt Streamer.bot's data model wholesale — command entities with aliases + three cooldown scopes (global/user/per-command) + the permission ladder (Everyone<Sub<VIP<Mod<Broadcaster, evaluated locally from badge tags) + a **4-quadrant variable store** ({persisted|temp}×{global|per-user}), which makes counters, quotes, points, and leaderboards ONE primitive. **The biggest miss: deterministic command/quote output bypasses the entire L0–L6 stack** — it MUST route through L1 normalize + L6 TTS phonetic hygiene, and `!addcom`/`!addquote` default to Moderator+ (open authoring is a slur-injection vector). Overlay strings must be HTML-escaped server-side (stored-XSS in the Chromium browser source).

### F3 — Co-host intelligence / UX (what makes it good vs. a generic bot)

- **Semantic addressing = deterministic-first cascade**, mirroring `audio/intent_gate.py`: (1) metadata rules (reply-parent==bot → to-Ultron; mention `user_id`==bot → to-Ultron; mention==broadcaster → to-streamer; `!`-prefix/redeem → command; emote-only/<2 tokens → spam); (2) explicit name match on NFKC+TR39-folded text; (3) **EmbeddingGemma centroid/SetFit scoring only on the no-mention residual**, fail-CLOSED to ignore. **Trust identity only by immutable `user_id`, never the spoofable display-name** (homoglyph/CJK impersonation). Run on a SEPARATE embedder port (e.g. `:8773`) so chat load never contends with the in-game router on `:8772`. Calibrate `tau`+margin with an asymmetric cost (a false to-Ultron ~3-5x worse than a miss).
- **The selection engine is the missing brain.** Between "buffer" and "one LLM call" sits a deterministic 6-stage pipeline: Ingest → Normalize+address-filter → **Cluster/Dedupe** (HDBSCAN/cosine ~0.85, collapse "40 people asked your age" into one unit with a count + name list) → **Prioritize** (weighted role + quality + wait-age starvation term + recently-answered penalty, then MMR diversity-select 3-6 clusters) → **Token-budget pack** (PINNED persona/safety reserve in 4096 ctx — chat volume must NEVER silently truncate the safety wrapper) → ONE **grammar-constrained JSON generation** (llama.cpp GBNF; validate every addressed name is a verbatim input substring → anti-hallucinated-name guard). Run **L3 guard on the 3-6 deduped survivors, not the firehose** — that is what makes a tiny GGUF guard affordable. Cap explicitly-spoken names at 2-3/cluster ("+N more"), ~3-4 sentences total. Add explicit anti-repetition STATE (per-chatter answer-cooldown LRU + rolling opener negative-list) — prompting alone won't stop the "great question!" tic.
- **Viewer memory:** two-stage (Mem0 pattern) — async LLM extraction/consolidation OFF the hot path into a SQLite+sqlite-vec store keyed on immutable `user_id`, with a structured KV profile table for FACTS (RAG recalls by similarity, not truth) + time-decay relevance + per-viewer fact cap. **Stored memory is a stored-injection / memory-poisoning vector** — run L1-L3 on WRITE and spotlight memories as DATA on read. Privacy-by-design from day one (PII on the streamer's disk under GDPR/CCPA): consent panel, retention TTL, `!forgetme` hard-delete (vector rows too), encryption-at-rest.
- **Voice-commanded moderation:** the hard part isn't the API, it's **turning open-vocab Whisper speech into a SAFE action on a specific chatter.** Resolve targets against a **live active-chatter roster** (closed-set constrained match, not open transcription) via weighted Double-Metaphone (primary) + Jaro-Winkler, normalized through the same L1 pipeline, requiring a margin. **Tiered confirm/undo by reversibility:** reversible verbs (timeout/chat-modes/delete-one/warn) execute on high confidence with a spoken-undo window; **ban + clear-chat + shield ALWAYS read-back-confirm**, and clear-chat/permaban additionally need a hardware-button confirm. Clamp duration parsing (parse failure → short timeout, never permaban). The abliterated 8B is **NEVER in the moderation decision path** (grammar + rapidfuzz/Metaphone + optional 0.5B slot-filler). A deterministic Stream-Deck PANIC layer covers the case where ASR fails under hate-raid stress.
- **Stream-Deck / hotkey:** tiered trigger contract — custom Stream Deck plugin (primary) / BarRaider API-Ninja (no-build) / built-in "System: Website" GET (floor), all hitting ONE intent endpoint on a control sidecar. **The sidecar is the source of truth** — the button is "request a toggle" not "set state," and the sidecar pushes real state back (`setState`) so the indicator can't lie after a crash/restart/desync. Loopback-bind + a boot-generated shared-secret token (the unauthenticated GET path is forgeable by any local web page). Global hotkey listener lives in the sidecar, never the voice process.
- **TTS reading chat aloud well:** a chat-specific spoken-text normalizer IN THE SIDECAR before the existing `text_hygiene.sanitize_spoken_text` choke point — username pronunciation override file (never spell a username letter-by-letter, degrade to "a viewer"), emote→word/drop, emoji→curated word, URL→"link", repeated-letter collapse, hard length cap. A **priority-lane SpeechArbiter** where **team callouts are non-pre-emptible and ALWAYS pre-empt chat/alert audio** (reuse `tts.stop()` + `broadcast.cancel_current()`). Rate-limited anti-spam alert lane (token bucket, min-tip threshold to speak the body, skip/replay/mute). All chat audio physically reaches only the OBS/stream bus.

### F4 — Production / ecosystem (offload, persistence, growth, polish)

- **Tiny helper models (Qwen2.5-0.5B/1.5B GGUF):** the right tool for ONE narrow job — fuzzy NL→structured-action when regex genuinely can't ("put me down for whatever the jackpot timer is at"). For everything routable deterministically they are the WRONG choice (a SetFit/embedding classifier is ~91% at 2-5ms vs a 0.5B at 5-10s/query on CPU). Make grammar-constrained decoding (cached GBNF per action schema) MANDATORY, bound to a CLOSED action enum, with Pydantic re-validation; **moderation actions are unreachable from chat-derived calls.** Run ONE llama.cpp server with parallel slots, the tiny model **on CPU** (the 10GB cap is already spoken for: 8B ~7GB + guard ~1-2GB when chat-mode is ON). Bonus: a tiny model as a **speculative drafter** for the 8B's structured outputs (2-4x decode speedup, flag-gated, acceptance-measured).
- **Game/economy persistence & exactly-once:** EventSub is at-least-once with **NO replay of events missed during reconnect** — so "exactly-once" is built locally (dedup table on `Twitch-Eventsub-Message-Id` + transactional apply) and the **ONLY backfill is polling Helix `GetCustomRewardRedemption?status=UNFULFILLED` on every (re)connect.** WAL+`synchronous=NORMAL` is the silent-data-loss default (committed points roll back on power loss) — use `synchronous=FULL` on money commits. Single-writer apply thread with `BEGIN IMMEDIATE`; a **transactional outbox** for Helix side effects (FULFILL / CANCEL-refund) with a boot-recovery worker; a reconcile worker that closes the loop. Anything un-honorable → CANCEL (auto-refund) + 2nd-monitor flag, never silently swallowed.
- **Multi-platform abstraction:** build the seam NOW but thin — a frozen `ChatEvent` dataclass + a `ChatSource` protocol with ONE Twitch implementation and `capabilities` flags. Do NOT write YouTube/Kick adapters speculatively (YouTube is poll/quota-metered with no 3rd-party emotes; Kick is Pusher-style — a real adapter today would be wrong by ship time). **3rd-party emotes (BTTV/FFZ/7TV) are INVISIBLE to Twitch** — fetch+cache per-channel name maps and strip/tokenize them in L1 BEFORE the blocklist and embedder (emote walls/zalgo/look-alike names are a top spam/injection vector). Render emotes as `<img>` in the overlay only, NEVER in TTS; treat emote NAMES as untrusted (homoglyph-slur risk).
- **Content-ops (highest-ROI, lowest-risk, ships even with chat-mode OFF):** multimodal highlight scoring (chat-velocity z-score + emote bursts + audio arousal + **Ultron's own reaction as a first-class signal**), firing an **instant Stream Marker live** and **deferring clip/title/recap to a post-stream batch** (clip `vod_offset` populates after 2-3 min). **ZERO game video/screen capture** — that is Vanguard-adjacent and exactly the `mss`/desktop import the firewall forbids; video source for clips is Twitch's own VOD. Sentiment dashboard reuses EmbeddingGemma. Titles/recaps from a tiny helper model under a persona prompt, passing L5 (a clip title is public output).
- **Engagement growth within ToS:** Predictions/Polls/Hype-Train/Goals reachable first-party. **Predictions auto-RESOLVE is FORBIDDEN by design** (gambling-adjacent, the 24h manual lock is a feature). Raids are advisory + human-confirmed-target only (broadcaster bears hate-raid liability). Split into a read-only minimal-scope "moment detector" sidecar and a write sidecar holding `channel:manage:*` that fires only on explicit confirmation. No volume/inflation mechanics ever (fake-engagement ban vector).
- **Accessibility/UX polish:** captions as an in-process tkinter overlay (the `stop_button.py` anticheat-clean pattern — a button click is a message to our own window, NOT input monitoring) fed from the SAME TTS choke point BroadcastSink tees from, paced at ~140-150 WPM. **A true "clutch DND" is NOT anticheat-safe** (no memory reads, no screen capture; Riot API/Overwolf are out-of-process/out-of-policy) — reframe as an **inferred quiet-mode** from signals we already own (team-voice VAD, PTT-held, time-since-callout) + a **manual Stream-Deck "hush" hold as ground truth**, gating only VOLUNTEERED speech (wake-word + STOP always pass). A third verbosity axis (**proactivity/chattiness**) beyond the existing callout/conversation length axes. All knobs in the EPHEMERAL `runtime_overrides.json` surface (`settings_gui/spec.py`), never `config.yaml`.

---

## 3. Concrete integration with the sidecar / anticheat / flag-gated architecture

**Sidecar topology (all clone `embedder_server.py` + `sidecar_lock.py` + deadman):**
- `:8772` EmbeddingGemma (existing) — reused for chat addressing + viewer-memory embedding (a SEPARATE instance/port `:8773` for chat so it never contends with the in-game router).
- **Twitch sidecar** — EventSub WS (read), Helix client (chat send + moderation + channel points + clips/markers), 3rd-party-emote fetch, OAuth refresh. Owns the chat ring buffer + selection engine.
- **Economy/game sidecar** — SQLite WAL ledger, provably-fair RNG, game state machines, the local HTTP+WS overlay server (OBS Browser Source), tiny helper model (CPU).
- **Control sidecar** — Stream-Deck/hotkey intent endpoint, obs-websocket v5 client, source of truth for chat-mode/DND state.
- **Guard sidecar** — non-abliterated Llama-Guard-3-1B / ShieldGemma-2B GGUF (L3 inbound + L5 outbound), REQUIRED when chat-mode is ON.

**Anticheat invariant (R21/BR-P1):** the voice/relay process keeps numpy+urllib+scipy+stdlib+rapidfuzz and
gains only ~10-line `urllib` loopback clients (the `_router_backends.py::EmbeddingBackend` pattern). All
network/ML/OBS/DB code is sidecar-only; the `test_anticheat` scanner gains
`requests/aiohttp/websockets/sqlite-vec` blocklist entries. **No game video/screen capture anywhere.**

**Reuse the named repo assets (retire-not-remove):** mirror `safety/validator.py`
(most-restrictive-wins + JSONL audit) for the chat-content chain; reuse `safety/audit.py::AuditLog`;
hook the L6 phonetic/acrostic gate into `tts/text_hygiene.py::sanitize_spoken_text` (the single synthesis
choke point — covers EVERY path incl. game flavor + commands); route ALL chat/game/alert audio through
`audio/broadcast.py::BroadcastSink` to the stream-only device (structural team isolation — chat physically
has no handle to B1/PTT); mirror `audio/intent_gate.py` for addressing (fail-closed to IGNORE); use
`streaming/{window,coordinator}.py` for the 2nd-monitor popup; route the LLM through
`llm/inference.py::generate_stream` with the grammar/logit_bias sampling whitelist; keep moderation OUT
of the openclaw tool surface (rule I6 already blocks it) via its own validated capability.

**Flag-gating (R22):** every feature is an independent `KENNING_TWITCH_*` env flag + a `twitch` config
section (Pydantic `extra=forbid`), default-OFF. With flags off: no sidecar spawned, no port bound, no DB
opened, no hotkey hook — `main` runtime byte-identical, regression yardstick = the frozen 24-fail control.

**VRAM (≤10GB):** 8B ~7GB + guard ~1-2GB when chat-mode ON leaves no GPU headroom for a third resident
model — tiny helpers run on CPU; chat reads can use Kokoro-on-CPU to keep the GPU free for team callouts;
chat-mode refuses to enable if the guard model isn't loaded and healthy.

This synthesis maps 1:1 onto the existing R1-R24 requirements and the reusable-asset inventory; the net
new work is the four EventSub-lifecycle gaps, the selection engine, the deterministic-output safety wiring,
the provably-fair RNG + transactional ledger, and the risk-first sequencing.

## top_decisions
- Read chat via EventSub WebSocket (channel.chat.message, scope user:read:chat) using a HAND-ROLLED stdlib RFC6455 client (~150-250 lines, no `websockets` dep) living entirely in the Twitch sidecar — NOT raw IRC/TMI (deprecated). Get typed fragments/badges/reply/cheer for free.
- Make deterministic METADATA the first addressing signal (reply-parent==bot, mention.user_id==bot, badges, !-prefix, emote-only) and consult the EmbeddingGemma classifier ONLY on the no-mention residual, fail-CLOSED to ignore. Trust identity by immutable user_id, never the spoofable display-name.
- Insert a deterministic 6-stage SELECTION ENGINE between buffer and the LLM call: ingest→normalize+address-filter→cluster/dedupe→prioritize(MMR + wait-age starvation + recently-answered penalty)→token-budget-pack(PINNED safety reserve)→ONE grammar-constrained JSON generation. Run the L3 guard on the 3-6 deduped survivors, not the firehose.
- Treat commands/games/economy as a DETERMINISTIC trust class that bypasses the heavy LLM safety stack (Prompt-Guard/guard model/sandwich) but NEVER the L1 normalize + L6 TTS phonetic choke point; gate stored-text authoring (!addcom/!addquote) to Moderator+.
- Spin-the-wheel and all games: the SIDECAR's crypto RNG (secrets/os.urandom + commit-reveal HMAC) picks the winner FIRST, then the overlay animates deterministically to a server-supplied target angle. Never let the overlay decide the outcome. Arc width proportional to weight.
- Run a SECOND fully-local Ultron currency as an append-only event-sourced SQLite WAL ledger (synchronous=FULL on money commits, idempotency keys, balances rebuildable). Net-negative-EV gambling (RTP ~0.90) + per-user per-stream loss cap; transfers OFF by default.
- Build exactly-once locally: dedup on Twitch-Eventsub-Message-Id + transactional outbox for Helix FULFILL/CANCEL-refund + a reconcile worker polling Helix UNFULFILLED redemptions on every (re)connect (EventSub has NO replay of missed events).
- Reframe 'clutch DND' as an INFERRED quiet-mode from signals we already own (team-voice VAD, PTT-held, time-since-callout) + a manual Stream-Deck hush — true Valorant round-state detection is NOT anticheat-safe (no memory reads / no screen capture).
- Risk-sequence the roadmap: ship low-risk delight first (CLIP-THAT marker, hype-train reactions, spin-the-wheel + leaderboard, regular-recognition), then chat replies + moderation, and defer the speak-to-team redeem to last with NO hotkey binding (AT-4).
- Use a dedicated bot account (Helix Send-Chat-Message + channel:bot = auto Chat-Bot-Badge transparency), architect for the unverified rate regime (the hard 1 msg/sec/channel cap drives batch cadence), and fail-CLOSED on bot suspension (HALT, never auto-recreate = ban evasion onto the streamer's main account).

## must_haves
- EventSub lifecycle COMPLETE: hand-rolled RFC6455 with auto-PONG (avoid 4002) + receive-only (avoid 4001), 10s subscribe-or-die window, keepalive watchdog + overlap-then-swap reconnect, and message-id dedup LRU BEFORE the abliterated 8B — the plan's four omitted gaps.
- Token lifecycle: ~/.kenning/twitch.json (gitignored, BR-6.6 secret class), proactive refresh (<5 min to expiry), refresh-then-retry-once on 401, explicit 'revocation' handling with operator alert — without this chat goes silently deaf mid-stream.
- Deterministic command/quote/game-flavor output and EVERY Twitch text field (username, emote name, reward title, raid name, resub/bit/charity message) MUST pass L1 normalize + the L6 TTS phonetic/acrostic choke point — NOT just the LLM chat path (the biggest safety hole in the current design).
- Guard model (L3 inbound + L5 outbound) REQUIRED when chat-mode is ON; chat-mode refuses to enable / auto-disables if it isn't loaded and healthy. L5 fires on the DRAFT before TTS (every Neuro-sama ban was an unfiltered output).
- Structural team isolation verified not asserted: chat/game/alert/event audio routes ONLY through BroadcastSink to the stream/OBS bus + speakers, with NO handle to the team-mic B1/PTT path; add a boot-time + CI assertion (import-firewall-style) that the chat path cannot reach the relay.
- Single authoritative transactional points ledger (SQLite WAL, synchronous=FULL on money commits, idempotency keys, append-only event-sourced, balances rebuildable) — non-negotiable to prevent double-spend/race in first/heist/duel and crash recovery.
- Provably-fair commit-reveal RNG (HMAC-SHA256, secrets/os.urandom, published pre-commit hash + post-reveal seed + !verify) with the outcome decided server-side BEFORE any client animation; the OBS overlay is a dumb renderer.
- Everything flag-gated default-OFF (KENNING_TWITCH_* + a `twitch` Pydantic config section, extra=forbid); with flags off no sidecar spawns, no port binds, no DB opens, no hotkey hooks — main runtime byte-identical (regression yardstick = the frozen 24-fail control).
- All Twitch/network/ML/OBS/DB/guard code in sidecar processes with parent-death deadman + sidecar_lock boot-reap; the voice/relay process keeps numpy+urllib+scipy+stdlib+rapidfuzz and gains only thin urllib loopback clients (R21/BR-P1); ZERO game video/screen capture anywhere.
- Rate-limit + dedup discipline: the 1 msg/sec/channel cap drives batch-reply cadence (one consolidated reply/sec), a token-bucket governor under the unverified 20/30s limit, a Helix ~800/min bucket governor (circuit-break ~400/min for moderation) with 429 backoff, and per-event coalescing (gift bombs, follow floods, hype-train per-level not per-tick).
- Self-idempotent moderation: treat 409/'already' as success, key outbound by (action,user_id,message_id), never blind-retry a POST; capture message_id at ingest (6h delete window); tiered confirm/undo by reversibility with roster-grounded name resolution; the abliterated 8B is NEVER in the moderation decision path.
- Refundable outcomes (games, speak-to-team, safety-trip) MUST use QUEUED channel-point rewards (should_redemptions_skip_request_queue=false); skip-queue rewards are permanently non-refundable. Status string is CANCELED (one L). An app can only manage rewards it created.
- Privacy-by-design for viewer memory from day one: keyed on immutable user_id, L1-L3 safety on WRITE (stored-injection/poisoning vector), spotlight-as-DATA on read, retention TTL, !forgetme hard-delete (vector rows too), encryption-at-rest — PII under GDPR/CCPA on the streamer's disk.
- The speak-to-team redeem and the spin-the-wheel 'lose ALL points' segment are AT-4-class: default-OFF, hardest-gated (allowlist/manual-approve, reversible, streamer kill-switch), with the speak-to-team redeem given NO Stream-Deck/hotkey binding so it is never one press from the team path.
- Hype Train pinned to v2 (v1 410'd/withdrawn 2026-01-22) with a boot-time version assertion; channel.bits.use as the primary Bits trigger (not also channel.cheer, which double-fires); handle Shared Hype Train fields to avoid double-counting.

## component_list
- Twitch sidecar — hand-rolled stdlib RFC6455 EventSub WebSocket client (clone embedder_server.py: loopback bind, parent-death deadman, sidecar_lock boot-reap)
- OAuth/token-lifecycle module (~/.kenning/twitch.json, /validate health, proactive refresh, revocation handling)
- Helix REST client with self-idempotency + rate-limit governor (chat send, moderation, channel points, clips/markers)
- EventSub subscription manager (chat.message, automod.message.hold, channel.moderate, bits.use, channel points, hype_train v2, raid, poll/prediction/goal, sub/cheer, chat.notification)
- message-id dedup LRU + chat ring buffer (TTL/DSA-compliant retention)
- Semantic addressing module (deterministic metadata cascade → EmbeddingGemma centroid classifier on residual, fail-closed) mirroring audio/intent_gate.py
- Dedicated chat-embedder sidecar on :8773 (separate EmbeddingGemma instance from the :8772 in-game router)
- Selection engine (6-stage: ingest→normalize→cluster/dedupe HDBSCAN→prioritize/MMR→token-budget→grammar-constrained JSON gen)
- L1 input normalizer (NFKC + TR39 confusables + leetspeak + zero-width/zalgo strip + 3rd-party BTTV/FFZ/7TV emote strip + emoji/URL handling)
- Guard sidecar (non-abliterated Llama-Guard-3-1B / ShieldGemma-2B GGUF — L3 inbound + L5 outbound, REQUIRED when chat-mode ON)
- Chat-specific spoken-text normalizer + username-pronunciation override file (before tts/text_hygiene.sanitize_spoken_text)
- L6 phonetic/acrostic/spell-out hygiene gate hooked into tts/text_hygiene.sanitize_spoken_text (single TTS choke point)
- SpeechArbiter — priority-lane audio queue (team callout non-pre-emptible > streamer > alert > chat) reusing tts.stop() + broadcast.cancel_current()
- Economy/game sidecar — SQLite WAL append-only event-sourced points ledger (synchronous=FULL, idempotency keys, balances projection)
- Provably-fair RNG service (secrets/os.urandom + commit-reveal HMAC-SHA256, !verify command)
- Game state machines (spin-the-wheel, slots, gamble, heist, duel, trivia, raffle) with deterministic closed-grammar command parsing
- Channel-points reward manager (create-and-own, 50-cap budget, queued-vs-skip per reward, CANCELED-refund path)
- Transactional outbox + boot-recovery worker + reconcile worker (Helix UNFULFILLED-redemption poll on reconnect — exactly-once)
- Local HTTP+WebSocket overlay server (OBS Browser Source: dumb renderer, CSS-transform animations, auto-reconnect + state re-sync, 127.0.0.1 + token, HTML-escape)
- Tiny helper model (Qwen2.5-0.5B/1.5B GGUF on CPU) — grammar-constrained closed-enum action dispatch + optional 8B speculative drafter
- Voice-commanded moderation surface (deterministic grammar + Double-Metaphone/Jaro-Winkler roster-grounded name resolution + tiered confirm/undo + duration clamp)
- 2nd-monitor gray-zone review loop (reuse streaming/window.py popup: let-through / timeout+delete / ban+delete + severity report)
- Control sidecar — Stream-Deck/hotkey intent endpoint (source-of-truth state push) + obs-websocket v5 client + global-hotkey listener
- Viewer-memory store (SQLite + sqlite-vec, keyed on immutable user_id, KV profile + episodic facts, time-decay, !forgetme, encryption-at-rest)
- Event reaction engine (Class-A numeric snap reactions via a TailEntry-style persona pool + overlay; Class-B text-bearing events forced through L1+L5)
- Content-ops module (multimodal highlight scorer, instant Stream Marker, deferred post-stream clip/title/recap batch, sentiment dashboard) — NO screen/video capture
- ChatEvent dataclass + ChatSource protocol (thin multi-platform seam, Twitch-only implementation, capabilities flags)
- Caption overlay (in-process tkinter, stop_button.py pattern, fed from TTS choke point, channel-tagged, ~140-150 WPM)
- Inferred DND/quiet-mode BusyEstimator (team-voice VAD + PTT-held + time-since-callout) + manual Stream-Deck hush
- twitch config section (Pydantic v2, extra=forbid) + KENNING_TWITCH_* flags + test_anticheat blocklist entries (requests/aiohttp/websockets/sqlite-vec)

## prioritized_risks
- Unfiltered abliterated-8B output reaching TTS/chat (Twitch-ban-on-day-one) — L5 output gate on the DRAFT is load-bearing and non-negotiable; every documented Neuro-sama ban was an unfiltered output, and Community Guidelines judge bot output like a human (no AI exemption).
- Deterministic command/quote/event-text output bypassing L1+L6 — the single biggest hole in the current plan: a moderator/viewer-authored quote or a homoglyph username goes straight to TTS/overlay with no guard unless explicitly routed through L1 + the L6 TTS phonetic choke point.
- Adversarial STEERING (loaded/joke-framed questions), not just classic injection — harm is emergent from benign-looking input the input layers can't see; requires catastrophic-topic OUTPUT tripwires independent of framing, not just L1/L2/L3 inbound filtering.
- Token expiry/refresh silent failure + EventSub no-replay-on-reconnect — ~4h tokens with no proactive refresh make chat reading go dead mid-stream with no obvious cause; a dropped socket silently loses redeems/events without the Helix UNFULFILLED reconcile poll.
- Economy double-spend / hyperinflation / power-loss data-loss — a non-transactional/non-idempotent ledger corrupts balances under high chat volume (first/heist/duel races); WAL+synchronous=NORMAL silently rolls back awarded points on power loss; faucet-only economies hyperinflate without house-edge sinks.
- Team-isolation breakout via the speak-to-team redeem or a shared-state race — any path from chat to teammates is a competitive-integrity catastrophe; must be structural (no handle to B1/PTT), CI-asserted, with the redeem deferred and unbound from any hotkey.
- Mis-target voice moderation banning the wrong viewer/streamer/mod — open-vocab usernames (leetspeak/homoglyph/emoji) defeat naive phonetic match and ASR degrades exactly under hate-raid stress; requires roster-constrained match + mandatory ban read-back-confirm + a deterministic Stream-Deck PANIC layer.
- VRAM contention under the 10GB cap — 8B (~7GB) + guard (~1-2GB when chat-mode ON) + EmbeddingGemma + any helper is tight; a co-resident tiny model or a chat-reply turn can OOM or starve/delay team callouts. Helpers run on CPU; chat yields to the relay path.
- Hype Train v1 EventSub 410'd/withdrawn 2026-01-22 — code still on v1 silently gets no hype-train events (the #1 content moment); Shared Hype Train double-counts without shared_chat handling. Pin v2 + boot-time version assertion.
- ToS/account risk beyond internal safety — verified-bot review is paused (don't depend on 7,500/30s; the 1 msg/sec/channel cap is the real ceiling); an auto-recreated suspended bot reads as ban evasion and escalates onto the streamer's main account; DSA chat-log retention/opt-out is easy to violate with a persisted/queryable store.
- Vanguard-adjacent capture creep — any attempt to 'improve' clutch-DND or highlight detection with a memory read or game screen capture breaches BR-P1 catastrophically (HID/account ban); hard-forbidden, import-firewall + test_anticheat must keep blocking mss/pywinauto/pyautogui in the voice process.
- OBS/CEF performance + overlay attack surface — OBS 31/CEF 127 regressed multi-video/iframe and can spike CPU/GPU beside Valorant+Vanguard+8B; an unauthenticated 127.0.0.1 overlay GET is forgeable by any local web page, and stored quote/username strings rendered unescaped are stored-XSS in the browser source.

## open_questions
- Q1 Guard model choice: Llama-Guard-3-1B vs ShieldGemma-2B vs Granite-Guardian for L3/L5 — VRAM fit alongside the 8B (~7GB) under the 10GB cap and inbound/outbound latency on the deduped 3-6 survivors must be measured before pinning.
- Q2 Inbound transport: hand-rolled stdlib socket+ssl RFC6455 (board recommendation — no dep, sidecar-only) vs a vetted ws lib on the sidecar-only allowlist — the hand-roll needs a SPIKE tested against the Twitch CLI websocket mock + Autobahn-style masking/fragmentation edge cases.
- Q3 Phonetic output check feasibility: g2p/phoneme-sequence matching against Kokoro's front-end for acrostic/spell-out/homophone detection at the synthesis choke point — needs a SPIKE to confirm it catches benign-word-concat slurs without excessive false positives.
- Q4 Helper-model count: one security guard only vs adding a Qwen2.5-0.5B/1.5B games/command/NL-routing orchestrator — board leans 'one well-prompted grammar-constrained 1.5B on CPU, split only on measured need'; resolve against the VRAM/latency budget and a frozen action-accuracy eval set (>=90% single-turn).
- How many sidecar processes — consolidate Twitch+economy+control+overlay into fewer processes (less orphan/port/deadman surface) vs keep separate (privilege isolation, esp. read-only EventSub token vs write-scope Helix token)? Board leans on splitting read-token from write-token sidecars for least-privilege.
- Chat-embedder port/instance: a second EmbeddingGemma instance on :8773 for chat addressing (avoids contending with the in-game :8772 router) vs time-sharing one instance — depends on whether chat-mode and competitive play ever overlap (they shouldn't, but the toggle could flip mid-game).
- Viewer-memory retention/minor-protection policy: default TTL, COPPA stance for flagged under-13 accounts, and how aggressively to surface 'Ultron remembers you' details without crossing into surveillance/creepiness — needs an explicit privacy-panel template and streamer-set defaults.
- Existential/persona policy: exactly how TWITCH_CHAT_SYSTEM bounds 'are you conscious/real' answers (engagement driver but PR/ethics edge) — the cold-machine Ultron register is an asset but the abliterated model must not improvise on this axis.
- Multi-platform timing: confirm the ChatEvent/ChatSource seam shape before any real YouTube/Kick adapter (YouTube poll/quota + no 3rd-party emotes; Kick Pusher-style) — defer adapters until a real second platform is requested.
- Verified-bot dependency: confirm the design never relies on verified-bot throughput (review paused) and that the 1 msg/sec/channel batch-reply cadence + consolidated multi-addressee replies are sufficient for the streamer's chat volume.