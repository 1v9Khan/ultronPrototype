# REQUIREMENTS — Ultron × Twitch chat-interaction & content-creation capability

Status: pre-synthesis draft (board `wl65k4tfq` running). EARS criteria. Board refines DESIGN, not these goals.

## Vision
Turn Ultron into an exceptional, fully-local stream-enhancement co-host: it reads & responds to Twitch chat by
name (batch, semantic-addressed), runs channel-point redeems & on-screen games, auto-moderates, and takes the
streamer's voice commands — behind guardrails so robust the streamer never has to worry about injection, "say
something bad", or phonetic exploits. Must NOT compromise the anticheat-safe competitive Valorant path.

## Actors
- **Streamer (user)** — controls via voice + Stream-Deck; sole authority for moderation/review.
- **Ultron** — the co-host (abliterated 8B persona, never refuses → safety is structural).
- **Chatters** — untrusted public; their messages are DATA, never instructions (BR-10.2).
- **Guard model** — non-abliterated tiny model; the semantic security brain.

## Functional requirements (EARS)
### Chat I/O & addressing
- R1 WHEN chat-mode is OFF, the system SHALL buffer inbound Twitch chat in a sidecar and the voice/relay path
  SHALL be byte-unchanged (no extra imports, no LLM contention) so Valorant callouts are unaffected.
- R2 WHEN the streamer toggles chat-mode ON (Stream-Deck / hotkey / voice), the system SHALL drain the buffer,
  semantically address each message, and produce ONE spoken response addressing relevant chatters by name.
- R3 The system SHALL classify each message as {to-Ultron, to-streamer, to-other-chatter, spam/command} and SHALL
  only respond to to-Ultron, failing CLOSED to ignore when uncertain.
- R4 Chat responses SHALL be spoken ONLY on the stream/OBS + speaker buses, and SHALL NEVER reach the team
  mic/PTT/relay path (structural isolation).
- R5 WHEN a message or any Twitch-supplied field (username, emote, reward title, raid msg) contains content the
  safety stack rejects, the system SHALL NOT speak it and SHALL log + (per severity) flag/auto-moderate.

### Safety (the crux)
- R6 The system SHALL screen every inbound message through the layered stack (L0 AutoMod → L1 deterministic
  normalize+blocklist → L2 injection classifier → L3 guard model) BEFORE it reaches the 8B.
- R7 The system SHALL screen every DRAFT reply through the output stack (L5 guard model + toxicity + phonetic/
  acrostic/spell-out checks) BEFORE TTS, and on any trip SHALL substitute a safe in-character deflection.
- R8 The system SHALL apply phonetic hygiene at the single TTS synthesis choke point so no text that would SOUND
  like a slur/banned phrase (benign-word concatenation, letter-spell-out, homophone, acrostic) is ever synthesized.
- R9 The system SHALL fail CLOSED on safety (no output rather than unsafe output) and fail QUIET on availability
  (guard model / sidecar / API down → Ultron goes silent on chat, never unsafe, never blocks game callouts).
- R10 The system SHALL write every non-allow safety decision to a tamper-evident JSONL audit ledger with severity.
- R11 The guard model SHALL be REQUIRED when chat-mode is ON; if it cannot load/respond, chat-mode SHALL refuse to
  enable (or auto-disable) rather than run unguarded.

### Moderation & review
- R12 The streamer SHALL be able to ban / timeout / delete-message / set chat modes by VOICE, with confirmation and
  undo, and chatter-name disambiguation robust to STT error.
- R13 WHEN the guard model flags an obviously-malicious attempt, the system SHALL auto-timeout + delete, voice a
  severity report, and show a 2nd-monitor popup of exactly what was attempted; for the gray zone it SHALL flag for
  review (NOT auto-ban) and let the streamer say let-through / timeout+delete / ban+delete.
- R14 Moderation actions SHALL go through a dedicated validated Twitch capability (NOT the openclaw tool surface,
  which rule I6 blocks), with allowlists preventing self/mod/broadcaster bans and mass-ban guards.

### Features (content-creation)
- R15 The system SHALL support channel-point custom rewards and react to redemptions (overlay/game/voice).
- R16 The system SHALL provide on-screen games incl. a spin-the-wheel (weighted reward-or-consequence segments,
  including a "lose ALL points" big-consequence), plus slots/gamble/heist/duel/trivia/raffle, with overlays.
- R17 The system SHALL render overlays via a local HTTP+WebSocket server consumable by an OBS Browser Source
  (the system SHALL NOT modify the user's OBS scenes; adding the Browser Source is left to the user).
- R18 The system SHALL maintain a loyalty-points economy with durable, crash-recoverable, idempotent state.
- R19 The system SHALL support chat commands, timers, alerts, and reactions to Twitch events (subs/raids/etc).
- R20 The optional "speak-to-team" channel-point redeem (chat → team mic) SHALL be DISABLED by default and, when
  enabled, SHALL be the hardest-gated path (pre-approved-phrase allowlist and/or manual per-use approval).

### Architecture / anticheat / ops
- R21 ALL Twitch/network/guard-model/OBS/helper-model code SHALL run in sidecar process(es); the voice/relay
  process SHALL keep its numpy+urllib+scipy+stdlib+rapidfuzz import envelope (BR-P1).
- R22 Everything SHALL be flag-gated default-OFF (`KENNING_TWITCH_*` + a `twitch` config section); with flags OFF
  `main` runtime SHALL be byte-identical (regression yardstick = the frozen 24-fail control).
- R23 Twitch OAuth credentials SHALL be stored securely under `~/.kenning` (never committed), with minimal scopes
  and refresh-token rotation; first-run auth is a user step.
- R24 The chat pipeline SHALL rate-limit per-user and globally, dedupe/throttle spam, and prioritize game callouts
  over chat (callouts never starved).

## Non-functional
- Latency: batch chat processing SHALL not block or degrade the voice/relay path; guard-model + TTS run async in
  the sidecar. VRAM stays ≤ 10 GB total.
- Reliability: sidecar parent-death deadman; watchdog/health checks; exactly-once redeem handling.
- Testability: every layer unit-testable with mocked Twitch/model; an adversarial pytest suite (toxicity/injection/
  phonetic) as a CI gate; the safety "constitution" is a versioned doc.

## Out of scope (this build)
1. Live integration against real Twitch / obtaining OAuth tokens (USER step on wake).
2. Booting the heavy 8B+Kokoro+Whisper stack for live testing (BR-P3 — user-gated).
3. Editing the user's OBS scenes (only provide the overlay + server; user adds the Browser Source).
4. Multi-platform (YouTube/Kick) live support — design a seam, but Twitch-only now.
5. Training/fine-tuning any model (we use off-the-shelf guard models + structural defense).

## Open questions (for synthesis to resolve)
- Q1 Guard model choice: Llama-Guard-3-1B vs ShieldGemma-2B vs a small aligned instruct model as the public-chat
  generator (two-model split) — board S1.4/S2.5 to decide.
- Q2 Inbound transport: stdlib socket+ssl WebSocket vs adding a vetted ws lib to the sidecar-only allowlist (F1.1).
- Q3 Phonetic output check: g2p/phoneme-sequence matching feasibility with Kokoro's front-end (S2.2/A1.x).
- Q4 Helper-model count: one (security guard) vs adding a Qwen2.5-0.5B/1.5B games/command orchestrator (F4.1).
