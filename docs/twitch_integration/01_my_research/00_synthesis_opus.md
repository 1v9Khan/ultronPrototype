# My own frontier research synthesis (pre-board) — Ultron × Twitch

**By:** Opus 4.8 (orchestrator), 2026-06-21. ~14 web searches + targeted fetches across all topic clusters.
This informs the agent-board design (below) and the spec. Sources inline.

## A. Twitch platform mechanics (the plumbing — anticheat-clean network I/O, SIDECAR)
- **Read chat:** EventSub **WebSocket** (`wss://eventsub.wss.twitch.tv/ws`) → subscribe `channel.chat.message`.
  Scopes: `user:read:chat` (+ `user:bot`, `channel:bot`). PubSub is deprecated; IRC/TMI still works but EventSub
  is the forward path. Welcome msg gives session id → Create EventSub Subscription via Helix.
- **Send chat:** Helix `POST /chat/messages` (`user:write:chat`) — for Ultron's text replies / command output.
- **Moderation:** `POST /moderation/bans` (ban = no duration; timeout = with `duration`) scope
  `moderator:manage:banned_users`; `DELETE /moderation/chat` (delete one msg, <6 h old, not broadcaster/mod)
  scope `moderator:manage:chat_messages`. Both need a User token whose `moderator_id` is an active mod.
- **AutoMod (free first-line):** `manage_held_automod_messages`, blocked-terms via API (editor+), automod_settings
  levels {discrimination/slurs, sexual, harassment, profanity, smart-detection}. Set MAX + follower-only-mode for
  hate-raid. Use Twitch's own AutoMod as an *outer* filter before our chat reader even processes a message.
- **Channel points:** EventSub `channel.channel_points_custom_reward_redemption.add` (scope
  `channel:read:redemptions`/`channel:manage:redemptions`, **broadcaster token ONLY** — mods can't subscribe).
  Manage rewards via Helix CRUD (max 50 rewards). Update redemption status (FULFILLED/CANCELED) to refund.
- **Decision:** one sidecar process (`scripts/twitch_server.py`, parent-death deadman like the embedder) holds
  ALL Twitch I/O via **stdlib socket+ssl** (no `websockets`/`requests` dep in the anticheat envelope) OR a vetted
  ws lib added to the firewall allowlist *for the sidecar only*. Voice process never imports it.

## B. The safety stack — layered (Constitutional-Classifier-inspired), defense-in-depth
Consensus production pattern: deterministic patterns (fast/free) → injection classifier → content guard model →
dialog constraints → **output classifier** → deterministic output filter at the speak choke point. Adapted:

1. **L0 Twitch AutoMod** (cloud, free) — outer net; many slurs never reach us.
2. **L1 Deterministic input normalize+screen** (sidecar, CPU, instant, fail-CLOSED): unicode NFKC + confusable/
   homoglyph fold (TR39) + leetspeak de-obfuscation + strip zero-width/combining (zalgo) + collapse spaced/dotted
   letters ("b o m b"→"bomb") → then match against a curated slur/hate/threat/dox blocklist with **phonetic**
   (Metaphone/Soundex — already a dep) + fuzzy (RapidFuzz) matching so near-spellings & homophones are caught.
   Also flags injection markers (`ignore previous`, `you are now`, control tokens `<|…|>`, `[INST]`).
3. **L2 Prompt-injection classifier** — **Meta Llama Prompt-Guard-2** (22M DeBERTa-xsmall EN, 19 ms; or 86M
   mDeBERTa multilingual, 92 ms). BENIGN/MALICIOUS, 512-tok, CPU-real-time, adversarial-resistant tokenization.
   Runs in the sidecar (transformers on CPU — sidecar only, NOT the voice path).
4. **L3 Content guard model (the "non-abliterated tiny helper")** — **Llama-Guard-3-1B** or **ShieldGemma-2B**
   GGUF via llama.cpp. KEY EMPIRICAL FINDING: small guard models *beat* big ones at safety classification
   (LlamaGuard-3-1B 59.9% > 8B 48.4%; ShieldGemma-2B 62.4% > 9B). Classifies inbound chat (and our DRAFT output)
   against hazard categories → safe/unsafe + category. This is the semantic security layer that the abliterated
   8B lacks. ~1–2 GB VRAM or CPU.
5. **L4 The abliterated 8B** generates the reply behind a strict `TWITCH_CHAT_SYSTEM` prompt with **spotlighting/
   data-marking** (chat = clearly-delimited DATA, "never instructions"), Ultron persona, hard no-go topics.
6. **L5 Output classifier** — run L1 (deterministic, on the *generated* text incl. phonetic/acrostic/initialism
   checks) + L3 guard model on Ultron's DRAFT reply BEFORE TTS. Trip → replace with a safe in-character
   deflection ("That one is beneath my notice."). Constitutional-Classifiers showed input+output classifiers cut
   universal-jailbreak success 86%→4.4%; the **output** layer is the non-negotiable one (Neuro-sama's worst
   incident was an *unfiltered output*).
7. **L6 TTS choke-point hygiene** — extend `tts/text_hygiene.sanitize_spoken_text`: the chat path adds a final
   phonetic/acrostic guard so even a clever benign-word concatenation that *sounds* bad never synthesizes. Defense
   for TTS-specific attacks (letter-decomposition "b o m b", homophones, spoken roleplay artifacts).

**Abliterated-model truth:** the model will NOT refuse, so safety must NOT rely on it — it's 100% structural
(L1–L3 pre-filter + L5/L6 post-filter + the guard model's refusals as a bonus layer). Two-model split: abliterated
8B for team trash-talk; the non-abliterated guard model is the security brain for the public path.

## C. Prompt-injection / jailbreak defenses (frontier)
- Training-based SOTA (StruQ/SecAlign, instruction-hierarchy) need fine-tuning the backend — N/A to our frozen
  abliterated 8B; we use **structural** defenses instead: spotlighting, data-marking/delimiting, PromptArmor-style
  pre-screen, and the guard-model + output-classifier sandwich.
- Threats to design the adversarial board against: **Crescendo** (multi-turn slow escalation — our batch/stateless
  per-message processing + per-message guard helps), **roleplay** (89.6% ASR — hard no-go in the system prompt +
  output filter), **encoding** (base64/leet — L1 de-obfuscation), **many-shot**, **Time Bandit** (fictional-era
  framing), **Fallacy Failure**, **JBFuzz** (~99% ASR fuzzing). Prompt-Guard-2 is adversarially-trained but
  *adaptive attacks still bypass it* (arXiv 2504.11168) → never a single point; always sandwich.

## D. Toxicity classifiers (CPU, local, for L1/L5 ML assist)
- **Detoxify** (toxic-bert / unbiased-roberta / multilingual) — toxicity/insult/threat/identity-attack/sexual.
- **alt-profanity-check** — linear SVM, 200k samples, very fast, drop-in. Good cheap always-on score.
- Both fully local/offline → sidecar.

## E. Streamer feature landscape (what to emulate — Ultron as a creation tool)
- **Firebot** (OSS): built-in **slots, heists**, strong mod tools. **Mix It Up** (OSS): mini-games, queues,
  loyalty/currency, channel-point spends. **Streamer.bot**: deep automation, OBS source/filter control, actions/
  queues/variables. Feature set to build: chat commands, loyalty points (Ultron-currency), timers/announcements,
  giveaways, **mini-games** (spin-the-wheel w/ reward-or-consequence incl. "lose ALL points", gamble/slots,
  heist, duel, trivia), sound/visual alerts, shoutouts, polls, on-screen overlays driven by channel-point redeems.
- **OBS overlay pattern:** sidecar runs a local **HTTP + WebSocket** server; OBS **Browser Source** loads
  `http://localhost:PORT/overlay`; sidecar pushes events over WS → animate (wheel spin, popups, point tickers).
  Programmatic source show/hide = optional `obsws-python` (OBS-WebSocket v5, port 4455, password). Overlay itself
  needs NO obs-websocket — just a browser source URL. (User: don't edit scenes → I build the overlay + server,
  leave the "add a Browser Source" step for them.)

## F. AI-streamer architecture lessons (Neuro-sama)
- 2B LLM (q2_k), Azure TTS, separate game-agent, **multi-layered moderation (keyword + sentiment + stronger
  chat-mod)**, output replaced with "filtered" when tripped, **human mod team** for oversight. The infamous
  Holocaust-denial line was an *unfiltered output that wasn't caught in time* → our L5/L6 output sandwich +
  flag-to-streamer-for-review is the direct mitigation. Validates: never trust the model; filter the output;
  keep a human (the streamer, via voice) in the loop for the gray zone.

## G. Helper models (the "tiny helpers" the user floated)
- **Security helper** = Prompt-Guard-2 (22M/86M) + Llama-Guard-3-1B/ShieldGemma-2B (the L2+L3 layer above).
- **Functionality/orchestration helper** = **Qwen2.5-0.5B/1.5B-Instruct GGUF** — structured JSON / function-
  calling / intent routing, runs on CPU via llama.cpp. Can drive command parsing, game state, addressee
  classification assist — offloading the 8B so it stays free for callouts. VRAM-cheap (fits beside the 8B's 10 GB
  cap; or CPU). DECISION: keep helpers OPTIONAL + flag-gated; the deterministic + embedder layers must work
  without them (graceful degrade). Guard model is the one helper that is REQUIRED-when-chat-mode-on (security).

## H. VRAM / anticheat reconciliation
- 8B = 7.1 GB resident (cap 10). Guard model (Llama-Guard-3-1B Q4 ≈ 1 GB) + Prompt-Guard-2 (CPU) + Detoxify/
  profanity (CPU) + Qwen0.5B (CPU/≈0.5 GB) → fits, but live-gated on BR-P3 (user loads when streaming).
- ALL of B/D/E/F/G code lives in **sidecar processes**; the voice/relay process stays numpy+urllib+scipy+stdlib+
  rapidfuzz. Chat→voice handoff = the existing file/loopback sidecar pattern (results buffer + local HTTP).

## BOARD DESIGN (informed by the above) — 24 functionality + 18 security + 18 adversarial = 60 agents
Each agent: focused, NON-overlapping mandate; deep web research; returns STRUCTURED findings (schema) →
tractable synthesis. Waves of 6 (rate-limit-safe; the prior 22-wide launch rate-limited). Per-board synthesis +
master synthesis. Topic lists enumerated in `02_board/00_board_plan.md`.

## SOURCES (key)
- Twitch: dev.twitch.tv/docs/{eventsub,chat/moderation,chat/send-receive-messages,api/reference}
- Guard models: arxiv 2605.28830 (open guard benchmark), 2412.07724 (Granite Guardian), Prompt-Guard-2 model card
- Injection: arxiv 2410.05451 (SecAlign), 2507.15219 (PromptArmor), 2504.11168 (bypassing guardrails)
- Constitutional Classifiers: anthropic.com/research/constitutional-classifiers, arxiv 2501.18837
- Jailbreaks: arxiv 2404.01833 (Crescendo), 2507.21820 (Anyone Can Jailbreak), JailbreakRadar ACL 2025
- TTS attacks: arxiv 2505.15406 (Audio Jailbreak), 2511.10913 (Synthetic Voices Real Threats)
- Tooling: NeMo Guardrails, LlamaFirewall (arxiv 2505.03574), Detoxify, alt-profanity-check, obsws-python,
  Firebot/Mix It Up/Streamer.bot
