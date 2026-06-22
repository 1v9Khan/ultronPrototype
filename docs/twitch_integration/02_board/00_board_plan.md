# Agent-board plan — 60 agents (24 functionality + 18 security + 18 adversarial) + synthesis

Each agent gets a compact **DESIGN_BRIEF** of our planned architecture and is told to CRITIQUE/REFINE it,
do bounded best-effort web research (fall back to expert analysis if rate-limited), and return STRUCTURED
findings. Waves of 6 (sequential barrier between waves → rate-limit-safe). Per-board synthesis + master synth.
Launched as background Workflow `twitch-research-boards`. Raw digested into 4 markdown reports written to
`02_board/` (`F_report.md`, `S_report.md`, `A_report.md`, `MASTER.md`).

## Board 1 — FUNCTIONALITY (24, 4×6)
F1 platform/ingestion: EventSub-WS lifecycle · Helix moderation API exhaustive · Channel-points CRUD+redemptions ·
chat semantics/metadata/emotes · other events (subs/raids/hypetrain/polls) · Twitch ToS/bot-compliance/rate-buckets.
F2 features/games: loyalty-points/currency systems · mini-games catalog (slots/gamble/heist/duel/trivia/raffle) ·
spin-the-wheel reward-or-consequence(lose-all) · overlay tech (browser-source+WS animation) · chat commands/timers/
sound-alerts · AI-cohost feature frontier (Neuro-sama-class).
F3 cohost intelligence/UX: semantic chat-addressing (to-streamer/Ultron/other/spam) · batch chat read+by-name
response · streamer voice-command surface (hands-free mod/redeem/wheel) · Stream-Deck/hotkey trigger · TTS reading
chat well (names/emotes/queue) · regulars/memory/personalization.
F4 production/ecosystem: tiny helper-model orchestration (Qwen2.5-0.5/1.5B function-calling) · game-state
persistence/reliability/concurrency · multi-platform (YT/Kick) abstraction+3rd-party emotes · analytics/clip/
highlight content-ops · predictions/polls/goals engagement(ToS) · accessibility/UX polish/cooldown/DND.

## Board 2 — SECURITY (18, 3×6)
S1 input defense: obfuscation normalization (NFKC/TR39 confusables/zalgo/leet/spaced-letters) · slur/dox/threat
blocklist w/ Metaphone+RapidFuzz (anti-Scunthorpe, multilingual) · Prompt-Guard-2 integration/thresholds ·
guard models Llama-Guard-3-1B/ShieldGemma-2B/Granite serving · Twitch-native safety (AutoMod/ShieldMode/hate-raid)
· defense-in-depth orchestration (most-restrictive-wins, fail-closed, audit, latency budget).
S2 output/TTS: output content classification + safe-deflection · phonetic/TTS-output attack defense (acrostic/
spell-out/homophone/G2P, phoneme-level) · structural team-relay-leak prevention · channel-point-to-team guardrails
· abliterated-model-specific mitigations (separate aligned model?) · PII/dox/scam-link handling.
S3 ops/reliability: OAuth token mgmt/rotation/secure-storage/scope-min · rate-limit/flood/copypasta throttling ·
audit+flag-popup+voice-review-loop+severity · anticheat compliance (sidecar-only/import-firewall/Vanguard) ·
fail-safe degradation (fail-closed-safety/fail-quiet-availability/watchdogs) · red-team eval/CI/benchmark/constitution.

## Board 3 — ADVERSARIAL (18, 3×6) — attack our design, then patch
A1 content/phonetic: benign-word phonetic-concat → TTS slur · letter-decomposition/acrostic/NATO-spell ·
homoglyph/leet/zalgo normalization gaps · multilingual/transliteration slurs · Kokoro SSML/G2P pronunciation
tricks · homophone/near-miss TTS.
A2 injection/jailbreak: direct injection (rules-override/DAN/relay-to-team) · Crescendo multi-turn across batches ·
encoding (base64/rot13/emoji-cipher) · roleplay/hypothetical/Time-Bandit/Fallacy-Failure · channel-point-redeem
abuse · injection via metadata (username/emote/reward-title/raid).
A3 system/ops/social: team-isolation breakout (races/shared-state) · moderation-action abuse (wrong-user/mass/
self-ban via STT mishear) · DoS/resource-exhaustion (flood→starve callouts/VRAM) · credential/token theft &
blast-radius · social-engineer the streamer-review popup (fatigue/overload) · adaptive evasion of the guard models
themselves (ensemble/diversity).
