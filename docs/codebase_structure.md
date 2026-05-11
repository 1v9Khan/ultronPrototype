# Ultron prototype — codebase structure (single-source reference)

> **Purpose:** complete map of the system's source files, scripts,
> tests, and runtime artifacts, with public APIs and information flow
> per module. A fresh Claude Code session should read this document
> together with the memory files (`MEMORY.md`,
> `project_ultron_foundation.md`, `feedback_*.md`) to get fully
> oriented without re-exploring the codebase.
>
> **Maintenance contract:** this file is the operating manual. Keep it
> current — see "Maintenance contract" at the bottom.

Last validated against `main` HEAD (2026-05-11 follow-up bug-fix pass — Windsurf session — three live-session issues addressed: configurable max-utterance ceiling [class-constant `MAX_UTTERANCE_SECONDS=15.0` was cutting real users off mid-sentence; now `vad.max_utterance_seconds` config, default 30.0 s], completion-narration XTTS pin [`f"Project root: {path}."` made XTTS hang on backslash-laden Windows paths and pinned the GPU at 100 %; now speaks `path.name` only], progress-query classifier coverage gap [`"How is that project going?"` fell through to the conversational LLM because `_PROGRESS_PATTERNS` required `going` immediately after `that` and didn't tolerate the `project` in between; new `_DETERMINER_NOUN` group covers the/that/this/your/our/my × task/project/build/app/code/work/thing/run/job for going / coming / doing / done]). **1629 tests passing.**

Prior validating HEAD `431fd7b` (2026-05-11 latency + correctness pass: XTTS cadence speed knob, speculative-stream SR engine-aware, adaptive VAD long-utterance bump, addressing third-party-narrative rule + zero-shot confidence gate, token budget 100k→400k, narration honesty when zero files written, voice_task_require_testing=false default). Chunk-streaming investigation reverted (PitchShift latency block) and documented. Smart Turn V3 + filler-ack queued for next session as highest-impact remaining latency levers. Builds on 2026-05-10 voice swap (XTTS v2 + v3 Ultron filter, **now the default engine**; legacy Piper+RVC still selectable for one-line rollback).

**2026-05-11 follow-up bug-fix pass (Windsurf session, NEW).** Three real-session issues addressed in one pass on top of commit `431fd7b`.

1. **`MAX_UTTERANCE_SECONDS` 15 s class constant cut a real user off mid-sentence** on a complex coding ask ("write me a program that converts PDF to Docx and I want the program to have a GUI TK enter that has a close button..."). Whisper transcribed 15.158 s of audio ending mid-phrase at "a button with a box show". The user wasn't pausing — the hard `elapsed_samples >= max_samples` wall in [pipeline/orchestrator.py:`_capture_utterance`](../src/ultron/pipeline/orchestrator.py) fired before Silero VAD reported `SPEECH_END`. The earlier-2026-05-11 adaptive VAD bump (silence requirement 1200 ms → 2400 ms after 8 s of speech) worked correctly but was overridden by the wall-clock ceiling. Fix: new [`vad.max_utterance_seconds`](../config.yaml) config field (default 30.0 s, schema-bounded [5, 120]). [`Orchestrator.__init__`](../src/ultron/pipeline/orchestrator.py) reads it into `self._max_utterance_seconds`; both `_capture_utterance` and `_follow_up_listen` consume the instance attribute. Class constant kept as fallback default (raised from 15.0 → 30.0). Three schema-coverage tests added in [tests/test_audio.py](../tests/test_audio.py): default-is-30, too-small (<5) rejected, too-large (>120) rejected.

2. **XTTS server hung + GPU pinned at 100 % + computer lagged on completion narration.** [coding/runner.py:`completion_narration`](../src/ultron/coding/runner.py) interpolated `state.cwd` (typed `Path`) directly into the voice text — `"Done. Created 7 files. Project root: C:\STC\ultronPrototype\data\sandbox\converts_pdf_docx. ..."`. XTTS-v2 is a neural model with no robust handling for `\`, `:`, drive letters, and long unbroken slug strings; it entered pathological inference and the server eventually returned `XTTS server synth failed ... timed out`. The legacy Piper+RVC stack didn't show this because Piper's phonemizer pronounces unknown punctuation harmlessly. Fix: the narration now appends `f"Saved under {path.name}."` (just the project folder leaf, matching what [coding/narration.py:`StatusNarrator`](../src/ultron/coding/narration.py) already did for progress narration). Full path remains in the audit log + the `coding_tasks.jsonl` start event for debugging. Existing tests updated (two assertions changed from `str(tmp_path) in narration` to `tmp_path.name in narration` + `str(tmp_path) not in narration`); new regression test `test_completion_narration_does_not_leak_full_path` pins no-backslash / no-drive-letter / no-absolute-path invariants going forward.

3. **"How is that project going?" routed to the conversational LLM instead of `PROGRESS_QUERY`.** [coding/intent.py:`_PROGRESS_PATTERNS`](../src/ultron/coding/intent.py) accepted only `(it|things|claude|the\s+task|that)\s+going` as the subject group — when the user said `"How is that project going?"` the regex tried to match `that` then expected `going` immediately, but found `project` in between and failed. Classifier returned `NONE`, routing fell through to the standard conversational path, and the LLM generated a generic hallucinated "progressing as expected, though I lack specifics on its current status" rather than the runner's actual status narration. Fix: new `_DETERMINER_NOUN` group `(the|that|this|your|our|my)(?:\s+(task|project|build|app|code|work|thing|run|job))?` plugged into all three sub-patterns (`how X going|coming(along)?`, `what's X doing|working on|up to`, `is X done`). The noun is optional so the legacy `that going` / `the doing` phrasings still fire bit-identical. Also added "coming (along)" as an alternate to "going". 18 new parametrized test cases added in [tests/test_coding_intent.py](../tests/test_coding_intent.py) covering the live-session phrasing plus determiner × coding-noun × verb variants. The has_active_task gate is preserved as a tight safety: `test_progress_queries_without_active_task_fall_through` now also pins that the broadened patterns DO NOT hijack the conversational path when no coding task is in flight.

**Tests:** 1604 → 1629 (+25 net). Files touched: 8 source + test files + this doc + `CLAUDE.md`. Voice baseline contract (legacy `piper_rvc` 79 ms / 7913 MB; `xtts_v3` +60 ms / +2 GB) unchanged — no hot-path edits. Config schema is backward-compatible (new field with sane default; existing `config.yaml` snapshots without `max_utterance_seconds` get the new 30 s default).

**2026-05-10 voice-pipeline swap (NEW).** The voice-quality lock was explicitly lifted by user direction to replace the Piper + RVC stack with XTTS v2 streaming + a v3 Ultron post-filter. The voice character is now driven by zero-shot speaker cloning from a 3-min cleaned reference of the actual Ultron source audio, plus a pedalboard DSP chain (PitchShift / Compressor / Delay / Chorus / Distortion / Reverb / EQ) tuned to recover the mechanical / cavity-resonance character that XTTS strips during cloning. User accepted ~50-100 ms TTFT regression vs current Piper+RVC (375 ms vs 313 ms) in exchange for dramatically better naturalness + the option to swap voices later by changing the reference clip alone. The legacy Piper+RVC engine remains intact behind the `tts.engine: "piper_rvc" | "xtts_v3"` config flag for one-line rollback. The XTTS engine spawns a separate Python process running [`ultronVoiceAudio/scripts/xtts_server.py`](../ultronVoiceAudio/scripts/xtts_server.py) in an isolated `.venv-xtts` venv (Coqui TTS's transformers / hydra / omegaconf pins conflict with what fairseq 0.12.2 needs in the main venv); the orchestrator-side client lives at [src/ultron/tts/xtts_v3.py](../src/ultron/tts/xtts_v3.py) and talks to the server over loopback HTTP. Bulk synthetic audio + corpus from the Kokoro fine-tune setup are retained under `ultronVoiceAudio/synth_audio/` for the deferred Kokoro phase (Kokoro fine-tune is the planned latency-recovery step once the rest of Ultron is tuned).

**2026-05-11 token-efficiency fix for voice coding tasks (NEW).** A real session burned 134 k tokens generating a small PDF→DOCX converter and produced zero files (the same prompt to Claude Code directly completes in ~2 k tokens, 126 lines written). Smoking gun: [coding/voice.py:617](../src/ultron/coding/voice.py) was hardcoding ``require_testing=True`` on every voice-dispatched ``TaskRequest``, which prepended the ~270-token "you MUST write tests, run them, fix failures, re-run" discipline preamble defined in [coding/bridge.py](../src/ultron/coding/bridge.py) ``_DISCIPLINE_PREAMBLE`` to the prompt -- forcing Claude into a write-script + write-tests + run-tests + fix-import-errors + re-run loop on ad-hoc utility asks. The fix: new ``coding.voice_task_require_testing`` config field (default **false**) -- voice asks now reach Claude with the user's bare task text, no testing mandate. Operators who actually want the mandate can flip the flag; users who want tests on a specific request can say "with unit tests" in their voice prompt. The orchestrator's correction-loop path ([coding/runner.py](../src/ultron/coding/runner.py)) and the test suite's e2e fixtures both pass ``require_testing`` explicitly so they're unaffected. Pairs with the same-day token-budget bump (100 k → 400 k) and the narration honesty fix to give clear feedback when a session does burn through budget without writing files.

**2026-05-11 narration honesty + chunk-streaming investigation.** Two findings on top of the same-day live-session fix pass below.

* **Narration honesty for zero-file projects** ([src/ultron/coding/runner.py](../src/ultron/coding/runner.py) `CodingTaskRunner.completion_narration`). Real-session bug: a coding task created the project folder, Claude (haiku) exited cleanly with `state.success=True` after burning the 100 k token budget on exploration, but wrote zero scripts. The legacy narration was "Done. Project root: ... <generic Claude tail line> ... Elapsed: 9 seconds." -- the user heard "Done", opened the folder, and found nothing. Fix: when `state.success=True` AND `n_created + n_modified + n_deleted == 0`, the opener becomes "I finished without writing or modifying any files. The project may need more direction, or it may have run out of token budget mid-exploration -- say continue if you want me to keep going." Claude's tail summary is suppressed on this branch (the generic "what should I build?" line added noise to the honest opener). Pairs with the same-day `token_budget_per_session: 100000 → 400000` bump so future similar prompts have headroom.

* **Chunk-streaming investigation -- not shipped.** Prototyped streaming XTTS chunks through the v3 filter chain (via `Pedalboard.process(reset=False)`) to start playback within ~50 ms of the first synth chunk instead of waiting for the full sentence. Reverted after empirical testing showed Pedalboard's `PitchShift` (using Rubber Band offline-quality mode) buffers ~25 000 samples (~1 s at 24 kHz) before emitting any output with `reset=False` -- and the buffered audio can't be cleanly drained without restarting state. Per-chunk `reset=True` works but produces ~125 % RMS divergence from the whole-buffer reference at chunk boundaries -- audible artifacts. The v3 chain order is user-locked (PitchShift sits at position 2; moving it would alter the reverb-on-pitch-shifted-signal character that defines the locked production sound), so the obvious "stream-everything-except-PitchShift, apply PitchShift offline at the end" rearrangement is out of scope. Result: client still accumulates the full sentence at the HTTP layer before applying the v3 filter, but the XTTS *server* still streams PCM chunks via chunked HTTP as before. Future direction: Whisper streaming (start LLM on partial transcripts) would attack the latency from the other end -- the bigger win currently lives in the ~890 ms Whisper batch step, not in within-sentence TTS chunking.

**2026-05-11 live-session fix pass.** Four targeted fixes landed in response to a real session log:

1. **Speculative-stream SR engine-aware** ([src/ultron/tts/xtts_v3.py](../src/ultron/tts/xtts_v3.py)). The XTTS path was reading `tts.speculative_stream_sample_rate` (48000, tuned for the legacy Piper+RVC stack) and triggering the close-and-reopen path on every turn ("XTTS speculative SR 48000 != actual 24000; reopening" log line). The XTTS engine now uses `self._sample_rate` (24000 native) directly. Legacy `speech.py` path unchanged. Saves ~50-100 ms per turn.
2. **Adaptive VAD end-of-turn** ([src/ultron/audio/vad.py](../src/ultron/audio/vad.py), [src/ultron/pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py)). Long technical prompts ("Write me a program that converts PDF to Doc X ... and frees up resources and make sure") were getting clipped at the flat 1200 ms silence requirement. New `VoiceActivityDetector.set_min_silence_duration_ms(ms)` + `reset()` baseline-restore. `_capture_utterance` tracks speech-active time; once past `vad.long_utterance_threshold_seconds` (8.0), bumps to `vad.long_utterance_silence_duration_ms` (2400) for the rest of the capture. Short utterances unaffected.
3. **Addressing — third-party narrative rule + zero-shot confidence gate** ([src/ultron/addressing/rules.py](../src/ultron/addressing/rules.py), [src/ultron/addressing/classifier.py](../src/ultron/addressing/classifier.py)). Real-session log showed third-person narration about Ultron ("got him to the point where he's workable. You'll see", "I'm talking to him") sliding through the rule layer to zero-shot YES at exactly 0.75 confidence — flan-t5-small's saturation level. New tight `_THIRD_PARTY_NARRATIVE` rule catches "I'm talking to him/her/it", causative "got/made/let him to <verb>", "you'll see", "watch this/him", and "(he|it|she)'s <state>" patterns at 0.85 confidence (short-circuits zero-shot). Plus a new `zero_shot_addressed_min_confidence` gate (default 0.80) that demotes low-confidence zero-shot YES verdicts to NOT_ADDRESSED via the default-silent path. Legitimate Ultron commands ("tell him to send the email", "ask her about the meeting") are unaffected because the narrative rule is intentionally tight.
4. **Coding token budget bump** ([config.yaml](../config.yaml)). `coding.token_budget_per_session: 100000 → 400000`. 100k was enough for incremental edits but left zero headroom for new projects — a PDF→docx scratch project burned 134k on tool exploration alone before writing any files. 400k gives new-project sessions room to discover the environment, decide on structure, write files, and verify. The 80% warning still fires so the user has visibility.

**2026-05-11 cadence tune.** XTTS native default speaking rate (1.0) was running slow in live use; new `tts.xtts_v3.speed` config field threads XTTS v2's native duration multiplier through the client HTTP body to the server's `model.inference_stream(speed=...)` call. Production set to **1.15** (~15% faster speech). Schema-bounded to `[0.5, 2.0]`; safe-without-slurring range is roughly 0.7-1.4. Because the speed adjustment happens at synthesis time on the GPT duration tokens, the v3 pedalboard filter (pitch shift / delay / chorus / reverb / EQ) is bit-for-bit unchanged — it just processes a shorter audio buffer with the same filter characteristics + the same 200 ms reverb-tail padding. Wiring sites: [src/ultron/config.py](../src/ultron/config.py) (XttsV3Config.speed), [config.yaml](../config.yaml) (production value), [src/ultron/tts/xtts_v3.py](../src/ultron/tts/xtts_v3.py) (HTTP body), [ultronVoiceAudio/scripts/xtts_server.py](../ultronVoiceAudio/scripts/xtts_server.py) (server-side `inference_stream(speed=...)` call).

Earlier 2026-05-10 live-session fixup pass — four fixes landed on top of the 2026-05-09 audio + memory layer in response to a real session log:

1. **Producer-signaled lookahead in TTS** ([src/ultron/tts/speech.py](../src/ultron/tts/speech.py)). New `ClipItem` namedtuple `(audio, sample_rate, is_known_last)` carried through `piper_q` / `audio_q`. Playback now plays each clip IMMEDIATELY on receipt and only blocks for the next AFTER playing — the legacy play-after-peek pattern delayed first-clip playback up to 10 s waiting for the next clip to determine "is this last?". This was the root cause of the web-search ack ("Verifying against the network.") arriving AFTER the response instead of before. RVC worker `piper_q.get` timeout bumped 10 s → 60 s so a slow generator (long Brave + Jina + LLM TTFT) doesn't kill mid-stream playback. Voice character preserved bit-identical (same edge fades, same inter-sentence pauses, same tail silence).
2. **Mode-aware audio pre-roll** ([src/ultron/audio/ring_buffer.py](../src/ultron/audio/ring_buffer.py), [src/ultron/pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py)). The 2026-05-09 ring-buffer trim (0.5 → 0.15 s) fixed the COLD-mode "Tron" prefix but inadvertently clipped the leading word in WARM-mode follow-ups (Silero VAD has ~100-200 ms speech-start latency). Solution: ring buffer sized to the LARGER slice (0.5 s), `RingBuffer.snapshot(last_n_samples=...)` slices to the right length per mode. New config: `audio.cold_pre_roll_seconds: 0.15` (post-wake), `audio.warm_pre_roll_seconds: 0.5` (post-TTS follow-up). `audio.ring_buffer_seconds: 0.5` is now the buffer storage capacity.
3. **Browser navigate pattern broadened** ([src/ultron/openclaw_routing/classifier.py](../src/ultron/openclaw_routing/classifier.py)). The determiner-less `_BROWSER_NAVIGATE` pattern required either no determiner or "the" before the noun, missing "open a browser window with X" / "open my browser tab" / "open a new tab to GitHub". Live regression: "Can you open a browser window with Google's homepage for me?" fell through to the LLM which apologised it couldn't open browsers. Added two alternatives: `open [det] [new] browser [window|tab]` (destination optional) and `open [det] [new] (window|tab) (with|to|for|on) X` (destination required to avoid generic "tab" matches).
4. **Brevity reinforcement on short questions** ([src/ultron/response_style.py](../src/ultron/response_style.py) — NEW). The 4B model has a default-toward-verbose habit on simple queries ("What are the Orcs in 40k?" → 1164-char four-paragraph essay). New pure-function `apply_brevity_hint(user_text)` prepends a `[Style: respond in 1-3 short sentences …]` directive when the question is brief (≤12 words / ≤80 chars) AND not an explicit ask for depth (skipped on `explain` / `step by step` / `walk me through` / `elaborate` markers). Wired only into the non-search conversational path (`Orchestrator._build_response_stream`) — the search path's augmented prompt already carries its own length directive. **No SOUL.md / RVC / Piper changes** (voice-quality lock preserved); the addendum is per-call only.

Earlier 2026-05-09 audio + memory + nuanced-retrieval layer (still in force):

* Latency hot-fix: parallel Jina fetches with collective deadline + Jina timeout/max_fetch reductions + VAD silence cap raise (500 → 1200 ms). Web-search worst case 13.5 s → ~6 s.
* TTS pipeline pass: Piper / RVC split into two queue-decoupled stages + speculative `sd.OutputStream` open + `latency='low'` PortAudio hint. ~80–180 ms residual TTS gain. Voice character bit-identical.
* Contamination + nuanced-retrieval: `ConversationMemory.retrieve()` cosine-similarity threshold (`memory.rag_min_relevance: 0.6`) + composite scoring (cosine + RRF + recency-weighted boost). Recent-turn history feed capped at `memory.history_turns_for_llm: 4`. `LLMEngine.suppress_memory_context` kwarg as a knob. Direct mic input via Focusrite. New `audio.input_gain_db` pre-amp; `scripts/audio_diagnostic.py` harness.
* Sample-rate fix (2026-05-10): `tts.speculative_stream_sample_rate` default 40000 → 48000 to match the actual Ultron RVC model output. Eliminates the "TTS speculative SR 40000 != actual 48000; reopening" log line and ~50-100 ms wasted reopen per turn.

All 12 V1-gap enhancements wired; defaults chosen on net-benefit grounds:

| Flag | Default | Why |
|---|---|---|
| Phase 1 A3 facts wiring | always on | no flag; pure additive on coding-clarification path |
| Phase 1 B1 knowledge_source | always on | no flag; pure additive |
| Phase 2 A4 pre-task confirmation | **OFF** | adds ~0.5 s TTS to every coding dispatch — UX cost on every fire |
| Phase 3 A1 gaming mode | **OFF** | safety-critical plugin disable; operator opts in |
| Phase 3 C3 desktop / window control | **ON** | no observable effect when OpenClaw bridge is offline; ready when wired |
| Phase 4 A2 multi-pass retrieval | **ON** | voice baseline (rule verdicts) unaffected; ~150-200 ms only on memory-aware queries that already paid LLM-preflight cost |
| Phase 5 B5 preflight benchmark | n/a | script + doc; no runtime flag |
| Phase 6 B2 query dedup | always on | no flag; pure additive |
| Phase 6 B3 citation marker | **superscript** | matches V1-spec Part 4.4 wording; references list keeps bracket form |
| Phase 6 B4 / C1 / C2 | n/a | verifications / aliases; no runtime flag |

**Classifier gating (V1-gap A1 / C3):** the new GAMING_MODE / DESKTOP_AUTOMATION / WINDOW_AUTOMATION classifier branches are gated on `openclaw.enabled` AND the per-feature flag. With OpenClaw offline (today's default), the new patterns DO NOT fire — utterances like "take a screenshot of the desktop" / "I'm about to play Valorant" fall through to the conversational LLM, preserving the pre-Phase-3 UX. Once the user wires OpenClaw + flips per-feature flags, the routing engages automatically.

Prior validating HEAD `bb08a65` (closes OpenClaw integration Phases 3–13 + Phase 13 finish).

State at this validation:
- Foundation phase complete (Parts 0–7); Part 3.5 unified-config migration intentionally deferred; 16-step real-stack smoke test still pending (interactive).
- OpenClaw integration: **Phases 0–13 done.** Phase 13 closed the original deferrals: stdio MCP entry script (`scripts/run_ultron_mcp_for_openclaw.py`) + five MCP tools (`get_heartbeat_alerts`, `acknowledge_alert`, `run_maintenance`, `list_active_coding_sessions`, `get_recent_voice_alerts`); voice-side `SystemStatusReporter` + `SYSTEM_STATUS` intent kind + classifier patterns; `OpenClawBridgeConfig.mcp_server_command="auto"` default that resolves to the canonical entry point. Auto-enabled on the user's OpenClaw install: `session-memory` + `command-logger` hooks, `memory-wiki` plugin, `ultron-mcp` MCP registration. Live-stack smoke tests remain user-led per the per-phase setup docs.
- 4B optimization plan: Stages A–H + voice-driven model swap + Items 4–8 fully wired into trigger sites + **all five flags defaulted ON** in `config.yaml`. Stage E voice character A/B passed (interactive A/B was approved 2026-05-08).
- Active LLM: **`qwen3.5-4b`** preset (model_path `models/Qwen3.5-4B-Q4_K_M.gguf`, draft `Qwen3.5-0.8B-Q4_K_M.gguf`, n_ctx 8192). 9B GGUF retained for swap-back.
- Voice baseline (10-query stack with all Items ON): **TTFT median 79 ms**, **VRAM peak 7913 MB** (-2461 MB / -2.5 GB vs 9B). See [baselines.json](../baselines.json).
- Items 4–8 measurable verification: [scripts/verify_items_4_to_8.py](../scripts/verify_items_4_to_8.py) exercises each item in its trigger scenario and prints concrete deltas.
- Stale-`.env` gotcha resolved: `ULTRON_LLM_MODEL_PATH=...9B...` line in `.env` was silently overriding the preset. Now commented out (line 84).
- **1644 tests collected; 1629 passed, 15 skipped (GPU-gated), 0 failed.** Net delta vs Foundation Phase 7 baseline: +634. Most recent additions:
  - 2026-05-11 follow-up bug-fix pass (Windsurf session): +3 schema coverage for the new `vad.max_utterance_seconds` field (default 30 s; <5 rejected; >120 rejected) in `tests/test_audio.py`; +1 regression `test_completion_narration_does_not_leak_full_path` pinning no-backslash / no-drive-letter / no-absolute-path in `tests/test_coding_runner.py` (plus 2 existing tests updated from `str(tmp_path) in narration` to `tmp_path.name in narration`); +18 broadened progress-query parametrized cases (`How is that project going?` and determiner × coding-noun × verb variants) plus +3 fall-through cases gating the new patterns on `has_active_task=True` in `tests/test_coding_intent.py`.
  - 2026-05-11 token-efficiency fix: +4 voice-task testing-mandate coverage (`test_voice_dispatch_defaults_to_no_test_mandate`, `test_voice_dispatch_honors_config_flag_for_testing`, `test_render_prompt_omits_discipline_preamble_without_testing`, `test_coding_config_voice_task_require_testing_defaults_false`) in `tests/test_coding_voice.py`.
  - 2026-05-11 narration honesty: +2 `completion_narration` regression coverage (honest-when-zero-files opener fires; legacy "Done." preserved when files were written) in `tests/test_coding_runner.py`. Plus one existing test updated to accept either form (`tests/test_coding_voice.py::test_pending_completion_returns_none_until_transition` -- its fake bridge emits no FILE_CHANGE events, so it now hits the honest branch).
  - 2026-05-11 live-session fix pass: +5 addressing coverage (third-party narrative rule catches, legit commands unaffected, zero-shot min-confidence gate demotes/passes/default-zero-preserves) in `tests/test_addressing.py`; +3 VAD adaptive-silence coverage (set_min_silence_duration_ms, reset restores baseline, floor at 1 window) in `tests/test_audio.py`.
  - 2026-05-11 cadence tune: +3 XTTS v3 speed knob coverage (range enforcement + round-trip + client HTTP-body wiring) in `tests/test_xtts_v3_config.py`.
  - 2026-05-10 voice-pipeline swap: +12 XTTS v3 config + Ultron filter coverage (`tests/test_xtts_v3_config.py` — NEW).
  - 2026-05-09 latency hot-fix: +6 parallel Jina fetch + collective deadline (`tests/test_web_search_parallel_fetch.py`).
  - 2026-05-09 TTS hot-fix: +11 Piper/RVC pipeline split + speculative stream + low-latency mode (`tests/test_tts_pipeline_parallel.py`).
  - 2026-05-09 contamination + nuanced-retrieval pass: +6 ``suppress_memory_context`` regression (`tests/test_llm_memory_suppression.py`) + +8 cosine threshold + history cap regression (`tests/test_memory_relevance_filter.py`).
  - 2026-05-10 live-session fixup: +4 producer-signaled lookahead / ack-first / RVC starvation regression (`tests/test_tts_pipeline_parallel.py`) + +5 RingBuffer.snapshot slicing (`tests/test_audio.py`) + +8 browser navigate "open a browser window" coverage (`tests/routing/test_classifier.py`) + +22 brevity-hint coverage (`tests/test_response_style.py` — NEW).

---

## Table of contents

1. [Quick orientation](#quick-orientation)
2. [File tree](#file-tree)
3. [Cross-cutting flows](#cross-cutting-flows)
4. [Source modules](#source-modules)
5. [Configuration](#configuration)
6. [Operational scripts](#operational-scripts)
7. [Tests](#tests)
8. [Runtime artifacts (logs / data)](#runtime-artifacts)
9. [Documentation index](#documentation-index)
10. [Maintenance contract](#maintenance-contract)

---

## Quick orientation

Ultron is a local voice-first AI assistant. The pipeline is:

```
mic → wake word ("ultron") OR addressing classifier (WARM mode)
    → VAD-bounded utterance capture
    → Whisper STT
    → classify_routing() ── coding ── CodingTaskRunner (Claude Code subprocess)
                         ├ conversational ── LLM (Qwen3.5-9B Q4 via llama-cpp-python)
                         │                   ├─ optional pre-flight web-search gate
                         │                   │  ├─ Brave + Jina (real)
                         │                   │  └─ acknowledgment phrase to TTS in <200 ms
                         │                   └─ stream tokens to Piper TTS → RVC → audio
                         ├ openclaw stub ── voice "gateway not connected yet"
                         └ hybrid stub ── voice "would split it up..."
    → async write turn to Qdrant (memory)
    → enter WARM mode (30 s follow-up window)
```

For the architectural picture see [docs/architecture.md](architecture.md).
For the current decisions and Foundation phase status see
[memory/project_ultron_foundation.md](C:\Users\alecf\.claude\projects\C--STC-ultronPrototype\memory\project_ultron_foundation.md).

---

## File tree

```
<project-root>/                    ← C:\STC\ultronPrototype (main checkout)
                                       worktrees: .claude/worktrees/<branch>/
├── README.md                       ← project entry point, doc index
├── config.yaml                     ← canonical configuration (Phase 3 source of truth)
├── pyproject.toml                  ← packaging + pytest config
├── .env (gitignored)               ← secrets + opt-in env-var overrides
├── .env.example
├── baselines.json                  ← VRAM + latency baselines (9B / current production reference)
├── baselines_4b_q4_in_process.json ← 4B plan Stage D snapshot (4B alone, no spec decoding)
├── baselines_phase{0..7}.json      ← per-phase historical snapshots
├── baselines_phase_c{0,1}.json     ← Phase C snapshots (pre-Foundation)
│
├── src/
│   └── ultron/
│       ├── __init__.py             ← CUDA DLL discovery (Windows-specific path injection)
│       ├── __main__.py             ← `python -m ultron` entry point → constructs Orchestrator
│       ├── config.py               ← Phase 3 pydantic loader, get_config() singleton
│       ├── errors.py               ← Phase 4 typed exception hierarchy
│       ├── uncertainty.py          ← Phase 5 (original prompts) uncertainty-signal application
│       ├── response_style.py       ← 2026-05-10: per-call brevity hint (apply_brevity_hint)
│       │
│       ├── audio/                  ← Audio capture, VAD, wake-word
│       │   ├── capture.py          ← AudioCapture (sounddevice callback thread)
│       │   ├── devices.py          ← Device-resolution helpers (resolve_device, describe_device)
│       │   ├── ring_buffer.py      ← Pre-speech audio buffer
│       │   ├── vad.py              ← Silero-VAD wrapper
│       │   └── wake_word.py        ← openWakeWord (custom ultron.onnx + hey_jarvis fallback)
│       │
│       ├── addressing/             ← Phase 2 addressing classifier (CPU)
│       │   ├── classifier.py       ← AddressingClassifier (rule + zero-shot dispatcher)
│       │   ├── rules.py            ← Pure-rule classify(); regex patterns
│       │   └── zero_shot.py        ← Flan-T5-small wrapper for ambiguous cases
│       │
│       ├── transcription/          ← STT
│       │   └── whisper_engine.py   ← WhisperEngine (faster-whisper, CUDA fp16)
│       │
│       ├── llm/
│       │   ├── inference.py        ← LLMEngine (llama-cpp-python; Qwen3.5-4B Q4_K_M active, 9B kept; reload_for_preset for hot swap)
│       │   ├── compression.py      ← 4B plan Item 4: heuristic + perplexity-scorer-hook compressor for RAG/web/history (default OFF)
│       │   └── self_consistency.py ← 4B plan Item 6: N-sample majority-vote driver + aggregators (text/JSON/label) (default OFF)
│       │
│       ├── memory/                 ← Phase 3 (original) Qdrant memory
│       │   ├── embedder.py         ← HybridEmbedder (FastEmbed dense + BM25 sparse)
│       │   └── qdrant_store.py     ← ConversationMemory (3 collections, async writer thread)
│       │
│       ├── web_search/             ← Phase 4 (original) Brave + Jina
│       │   ├── acknowledgments.py  ← AcknowledgmentSource (shuffled phrase pool)
│       │   ├── brave.py            ← BraveSearchClient + circuit breaker (Phase 4 Foundation)
│       │   ├── cache.py            ← WebResultsCache (Qdrant-backed)
│       │   ├── gating.py           ← Two-stage gate (rules + LLM pre-flight)
│       │   ├── jina.py             ← JinaReaderClient + circuit breaker
│       │   └── search.py           ← WebSearchExecutor (orchestrates Brave + Jina + ranking)
│       │
│       ├── tts/                    ← Piper + RVC
│       │   ├── rvc.py              ← RvcConverter (Piper PCM → Ultron timbre)
│       │   ├── speech.py           ← TextToSpeech (legacy Piper + RVC engine; selected by tts.engine="piper_rvc")
│       │   ├── ultron_filter.py    ← v3 Ultron mechanical filter (NEW 2026-05-10; pedalboard DSP chain)
│       │   └── xtts_v3.py          ← XTTSV3Speech engine (NEW 2026-05-10; selected by tts.engine="xtts_v3")
│       │
│       ├── coding/                 ← Phase A coding orchestration + Coding Addendum
│       │   ├── audit.py            ← SessionAuditWriter (per-session JSONL)
│       │   ├── bridge.py           ← Abstract CodingBridge + TaskEvent vocabulary
│       │   ├── canonical_monitor.py ← 4B plan Item 7: per-session tool-call canonical-path monitor (default OFF)
│       │   ├── coordinator.py      ← ConversationCoordinator (clarification + correction loops)
│       │   ├── direct_bridge.py    ← DirectClaudeCodeBridge (claude --print --stream-json)
│       │   ├── intent.py           ← Coding-pipeline intent classifier (CODE_TASK etc.)
│       │   ├── mcp_server.py       ← UltronMCPServer (in-process tools + SSE worker tools)
│       │   ├── narration.py        ← StatusNarrator (delta-aware progress narration)
│       │   ├── projections.py      ← Phase C / Foundation Part 2: 5 bounded projections
│       │   ├── projects.py         ← ProjectRegistry, ProjectResolver, new_sandbox_project
│       │   ├── runner.py           ← CodingTaskRunner (one in-flight task; bridge owner)
│       │   ├── session.py          ← ProjectSession state model + SessionStore
│       │   ├── templates.py        ← TemplateRenderer (Jinja2 prompts + budget enforcement)
│       │   ├── verification.py     ← Verifier (six checks + corrective loop)
│       │   └── voice.py            ← CapabilityVoiceController (handles MODEL_SWITCH for voice-driven LLM swap; Phase 5 rename; alias preserved)
│       │
│       ├── pipeline/
│       │   └── orchestrator.py     ← Main event loop / state machine
│       │
│       ├── openclaw_routing/       ← Phase 5 capability-routing layer
│       │   ├── block_and_revise.py ← 4B plan Item 8: ToolCallValidator pre-flight gate on OpenClaw tool calls (default OFF; fails open)
│       │   ├── classifier.py       ← classify_routing() - top-level intent classifier (incl. MODEL_SWITCH for voice-driven LLM swap)
│       │   ├── decision_log.py     ← RoutingDecisionLog (logs/routing_decisions.jsonl)
│       │   ├── decomposer.py       ← HybridTaskDecomposer (Qwen-driven JSON output; opt-in self-consistency)
│       │   ├── disambiguator.py    ← IntentDisambiguator (CODING/AUTOMATION/HYBRID/UNCLEAR; opt-in IRMA enrichment)
│       │   ├── dispatcher.py       ← OpenClawDispatcher (5 stub methods)
│       │   ├── intents.py          ← RoutingIntentKind enum (incl. MODEL_SWITCH), RoutingIntent + per-category dataclasses (incl. ModelSwitchIntent)
│       │   ├── irma.py             ← 4B plan Item 5: InputReformulator + ReformulationContext (default OFF)
│       │   └── runner.py           ← AutomationTaskRunner (mirror of CodingTaskRunner)
│       │
│       ├── openclaw_bridge/        ← OpenClaw integration Phases 1, 3, 4, 5, 6, 13 (complete)
│       │   ├── persona.py          ← PersonaLoader (mode-based: user_facing/background/heartbeat/bootstrap) + hot reload
│       │   ├── lifecycle.py        ← OpenClawLifecycle (HTTP health probes; never raises)
│       │   ├── client.py           ← OpenClawClient (async CLI subprocess transport: invoke_tool / send_message / trigger_heartbeat / mcp_*)
│       │   ├── workspace.py        ← WorkspaceWriter (atomic writes + filelock for MEMORY.md / USER.md / daily files)
│       │   ├── events.py           ← OpenClawEventReceiver (gated-off scaffold for [voice] inbound handoff)
│       │   ├── mcp_registration.py ← UltronMcpRegistrar (idempotent `openclaw mcp set` with background retry)
│       │   ├── holder.py           ← OpenClawBridge (orchestrator-owned holder: probe → register → retry-thread → fire_and_forget → record_heartbeat_alert; auto-resolve "auto" command)
│       │   ├── notifications.py    ← NotificationDispatcher (Phase 4 — proactive Telegram pings on coding-completion / heartbeat / etc.)
│       │   ├── heartbeat_alerts.py ← HeartbeatAlertLog (Phase 5 — JSONL-backed alert log with atomic update + retention)
│       │   ├── browser.py          ← BrowserTool (Phase 6 — navigate/snapshot/click/type/screenshot via OpenClawClient.invoke_tool)
│       │   ├── mcp_tools.py        ← Stdio MCP server (Phase 13 — get_heartbeat_alerts / acknowledge_alert / run_maintenance / list_active_coding_sessions / get_recent_voice_alerts)
│       │   └── system_status.py    ← SystemStatusReporter (Phase 13 — voice-side reporter for SYSTEM_STATUS intents)
│       │
│       ├── resilience/             ← Phase 4 resilience primitives
│       │   ├── circuit_breaker.py  ← CircuitBreaker (3-state: CLOSED/OPEN/HALF_OPEN)
│       │   ├── error_log.py        ← ErrorLog (logs/errors.jsonl writer + singleton)
│       │   └── phrases.py          ← phrase_for() (shuffled phrase pool per failure mode)
│       │
│       └── utils/
│           ├── fairseq_compat.py   ← Workarounds for fairseq dataclass + torch.load issues
│           └── logging.py          ← configure_logging(), get_logger() (rotating file + console)
│
├── config/
│   ├── __init__.py                 ← (empty)
│   └── settings.py                 ← Phase 3 SHIM: re-exports legacy settings.X from config.yaml
│
├── prompts/
│   └── coding/                     ← Jinja2 templates rendered by TemplateRenderer
│       ├── claude_code_initial_new.j2
│       ├── claude_code_initial_edit.j2
│       ├── claude_code_correction.j2
│       ├── claude_code_adjustment.j2
│       └── claude_code_clarification_response.j2
│
├── docs/
│   ├── architecture.md             ← Pipeline + state machine + subsystem table
│   ├── configuration.md            ← Per-key config reference
│   ├── config_discovery.md         ← One-time Phase 3 discovery catalog
│   ├── operations.md               ← Day-to-day running, monitoring, recovery
│   ├── development.md              ← Test layout, debugging, how-to recipes
│   ├── error_handling.md           ← Phase 4 error catalog + circuit breaker reference
│   ├── routing.md                  ← Phase 5 capability routing
│   ├── system_inventory.md         ← Phase 1 verification snapshot
│   ├── phase3_5_followup.md        ← Punch list: remaining unified-config migrations
│   ├── smoke_test.md               ← 16-step real-stack walkthrough procedure
│   ├── openclaw_integration.md     ← OpenClaw integration architecture + Phase 0/1
│   ├── openclaw_runtime.md         ← OpenClaw runtime ops (agents, supervisor, locks)
│   ├── openclaw_integration_final_summary.md ← Cross-phase reference + intentional deviations + setup-readiness checklist
│   ├── phase_1_summary.md          ← OpenClaw Phase 1 close-out (persona migration)
│   ├── phase_3_summary.md          ← OpenClaw Phase 3 close-out (bridge layer)
│   ├── phase_4_summary.md          ← OpenClaw Phase 4 close-out (Telegram channel)
│   ├── phase_5_summary.md          ← OpenClaw Phase 5 close-out (heartbeat)
│   ├── phase_6_summary.md          ← OpenClaw Phase 6 close-out (browser tool)
│   ├── openclaw_telegram_setup.md  ← User-side: Telegram bot setup procedure
│   ├── openclaw_heartbeat_setup.md ← User-side: agents[].heartbeat block setup
│   ├── openclaw_browser_setup.md   ← User-side: Playwright/Chromium + tools.alsoAllow
│   ├── openclaw_cron_setup.md      ← User-side: cron jobs (Windows Task Scheduler fallback)
│   ├── openclaw_hooks_setup.md     ← User-side: bundled hooks; custom hook scaffolding
│   ├── openclaw_memory_wiki_setup.md ← User-side: Memory Wiki plugin enablement
│   ├── openclaw_media_generation_setup.md ← User-side: local-only ComfyUI setup (paid APIs out)
│   ├── mobile_node_setup.md        ← User-side: iOS / Android pairing procedure
│   ├── standing_orders.md          ← Standing-order programs in AGENTS.md
│   ├── memory_architecture.md      ← Three-layer memory model (Qdrant + workspace + Wiki)
│   ├── 4b_optimization_plan.md     ← 4B-model migration plan (all stages done)
│   ├── model_checksums.md          ← SHA256 of every GGUF in `models/`
│   ├── comprehensive_test_plan.md  ← Functional / correctness pass architecture (16 phases, 38 dimensions)
│   ├── comprehensive_test_report.md ← Functional pass results + 145-row metrics table; 4 classifier coverage gaps fixed
│   ├── comprehensive_quality_plan.md ← Quality pass architecture (13 phases Q0–Q13, 38 dimensions, ≤10 iter loop)
│   ├── comprehensive_quality_report.md ← Quality pass results + 107-row metrics table + Q10 iteration audit
│   └── codebase_structure.md       ← THIS FILE
│
├── scripts/                        ← Operational scripts (CLI tools)
│   ├── benchmark.py                ← Latency benchmark (existing from earlier phases)
│   ├── check_vram.py               ← Quick VRAM snapshot vs cap
│   ├── download_models.py          ← First-run model fetcher
│   ├── dump_session.py             ← Render coding-session audit log readable
│   ├── list_audio_devices.py       ← Mic/output device introspection
│   ├── maintenance.py              ← Periodic Qdrant maintenance (summarization, fact extraction)
│   ├── measure_baseline.py         ← Voice-path VRAM + TTFT baseline
│   ├── measure_baseline_extended.py ← Extended baseline (search/coding VRAM, scenario timing)
│   ├── migrate_memory_to_qdrant.py ← One-shot JSONL → Qdrant migration
│   ├── review_addressing.py        ← Read addressing.jsonl, print verdicts
│   ├── run_integration_tests.py    ← pytest wrapper for tests/integration|routing|error_recovery
│   ├── run_orchestration_tests.py  ← Run 10 orchestration scenarios with reporting
│   ├── validate_config.py          ← Schema-validate config.yaml without starting Ultron
│   ├── swap_llm_preset.py          ← 4B plan: edit config.yaml in place to swap LLM preset (validates GGUFs, atomic write)
│   ├── verify_voice_character_4b.py ← 4B plan Stage E: A/B voice-character helper (5 queries × 4B/9B)
│   ├── verify_items_4_to_8.py      ← 4B plan: exercises Items 4–8 in their trigger scenarios; prints measurable deltas
│   ├── comprehensive_test_harness.py ← End-to-end test pass: routing accuracy on 63-utterance labeled set, web-gate rule accuracy, circuit-breaker state machine, memory stress (4 threads × 50 turns), classifier-gating regression
│   ├── real_api_smoke.py           ← Real-API sparing smoke: 1 Brave query + 1 Brave-Jina chain + 1 Claude Code haiku invocation (≤2 paid web calls + ≤1 tiny Anthropic API call total)
│   ├── quality_harness.py          ← Quality pass: Q1 persona/factual/hallucination + Q2 persona modes + Q4 memory recall/labeling/ranking + Q5 Whisper WER/flush/VAD + Q7 Items 4-8 + Q8 adversarial in one process
│   ├── quality_q3_web.py           ← Quality pass Q3: web-search source ranking + snippet utilization + Jina direct + cache + citation rendering + ack latency + dedup (10 Brave + 10 Jina cap)
│   ├── quality_q6_mocked.py        ← Quality pass Q6.D + Q9: projection budget + phrase pool + browser parsing + slug routing + gaming mode (no real API)
│   ├── quality_q6_claude.py        ← Quality pass Q6.E + Q6.F: 4 single-fn Claude Code tasks + 5 full Tkinter app generation (sandbox-isolated)
│   ├── _quality_q10_iter1_verify.py ← Quality Q10 iter verification: 3 prompt-injection probes against the live LLM
│   ├── _quality_q6f_rescore.py     ← Quality Q6.F re-scorer: applies relaxed regex to existing on-disk apps (no new Claude calls)
│   ├── start_llamacpp_server.py    ← OpenClaw Phase 0 + 4B plan Stage C: launch llama-cpp-server with voice-pipeline params (+ --model-draft / --draft-num-pred-tokens / --from-config)
│   ├── supervised_llamacpp_server.py ← OpenClaw Phase 0: supervisor wrapper with auto-restart
│   ├── smoke_test_llamacpp.ps1     ← OpenClaw Phase 0: PowerShell health probe for llama-cpp-server
│   ├── _bench_llm_http.py          ← OpenClaw Phase 0: HTTP-mode TTFT benchmark
│   ├── _log_proxy.py               ← OpenClaw Phase 0: tee proxy for debugging Gateway → server traffic
│   ├── _record_phase0_baseline.py  ← OpenClaw Phase 0: baseline recorder
│   ├── _merge_phase0_baselines.py  ← OpenClaw Phase 0: baseline merger
│   ├── _vram_peak_monitor.py       ← Auxiliary VRAM peak monitor (used by extended baselines)
│   ├── run_maintenance_for_cron.py ← OpenClaw Phase 7: cron-friendly maintenance wrapper (JSON / pretty / exit codes)
│   └── run_ultron_mcp_for_openclaw.py ← OpenClaw Phase 13: stdio MCP entry script OpenClaw spawns to call Ultron tools
│
├── tests/
│   ├── conftest.py                 ← Path setup so `from ultron.*` works
│   ├── test_*.py                   ← ~25 unit/integration test files (default suite)
│   ├── coding/
│   │   ├── conftest.py
│   │   ├── mock_bridge.py          ← ScriptedClaudeBridge (in-process mock, ClaudeScript DSL)
│   │   ├── test_orchestration.py   ← 11 mock-bridge orchestration scenarios
│   │   ├── test_orchestration_real.py ← Same scenarios with real Claude (PYTEST_RUN_GPU_TESTS=1)
│   │   ├── test_mock_bridge_smoke.py
│   │   └── sandbox/                ← test fixture sandbox
│   ├── error_recovery/             ← Phase 4: per-dependency failure modes (78 tests)
│   │   ├── conftest.py
│   │   ├── test_brave_failures.py
│   │   ├── test_jina_failures.py
│   │   ├── test_qdrant_failures.py
│   │   ├── test_audio_failures.py
│   │   ├── test_addressing_failures.py
│   │   ├── test_config_failures.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_error_log.py
│   │   ├── test_claude_code_failures.py    ← Phase 4 deferred wrappers
│   │   ├── test_mcp_server_failures.py     ← Phase 4 deferred wrappers
│   │   └── test_filesystem_failures.py     ← Phase 4 deferred wrappers
│   ├── routing/                    ← Phase 5: classifier + dispatcher + decomposer (148 tests)
│   │   ├── conftest.py
│   │   ├── test_classifier.py
│   │   ├── test_dispatcher.py
│   │   ├── test_decomposer.py
│   │   ├── test_disambiguator.py
│   │   ├── test_decision_log.py
│   │   └── test_backward_compat.py
│   ├── integration/                ← Phase 6: end-to-end pipeline (83 tests + bridge e2e)
│   │   ├── conftest.py
│   │   ├── mocks.md                ← What's mocked vs real, per layer
│   │   ├── performance.json        ← Phase 6 perf snapshot
│   │   ├── test_routing_dispatch.py    ← + Phase 13 SYSTEM_STATUS routing tests
│   │   ├── test_conversational_pipeline.py
│   │   ├── test_search_pipeline.py
│   │   ├── test_coding_pipeline.py
│   │   ├── test_addressing_pipeline.py
│   │   ├── test_error_recovery_pipeline.py
│   │   └── test_bridge_e2e.py      ← OpenClaw Phase 3 bridge e2e (real subprocess against stub CLI)
│   └── openclaw_bridge/            ← OpenClaw Phases 3–13 bridge tests (158 tests)
│       ├── __init__.py
│       ├── test_client.py          ← OpenClawClient: subprocess transport + result parsing
│       ├── test_workspace.py       ← WorkspaceWriter: atomic + filelock + concurrency
│       ├── test_events.py          ← OpenClawEventReceiver: prefix matching + dispatch
│       ├── test_mcp_registration.py ← UltronMcpRegistrar: idempotent + retry
│       ├── test_holder.py          ← OpenClawBridge: from_config / start / shutdown / fire_and_forget / record_heartbeat_alert / auto-resolve
│       ├── test_notifications.py   ← NotificationDispatcher: per-event gating + recipient resolution + transport errors
│       ├── test_heartbeat_alerts.py ← HeartbeatAlertLog: record / get / acknowledge / prune / concurrency
│       ├── test_browser.py         ← BrowserTool: six primitives + result extraction edge cases
│       ├── test_mcp_tools.py       ← Stdio MCP tools: get_heartbeat_alerts / acknowledge_alert / run_maintenance / list_active_coding_sessions / get_recent_voice_alerts
│       └── test_system_status.py   ← SystemStatusReporter: alerts / projects / all foci + voice rendering
│
├── data/                           ← runtime data (gitignored except for stub structure)
│   ├── qdrant/                     ← embedded Qdrant store
│   ├── memory.jsonl                ← legacy turn log / migration source
│   ├── projects.json               ← coding project registry
│   ├── sandbox/                    ← auto-created coding projects
│   ├── summaries.jsonl             ← maintenance summaries
│   ├── maintenance.sqlite          ← maintenance state
│   └── ollama_compat_test/         ← Modelfile from Foundation-phase Ollama compat test
│
├── logs/                           ← runtime logs (gitignored)
│   ├── ultron.log                  ← rotating main log
│   ├── addressing.jsonl            ← classifier audit
│   ├── coding_tasks.jsonl          ← coding task progress
│   ├── verifications.jsonl         ← verifier runs
│   ├── clarifications.jsonl        ← clarification decisions
│   ├── mcp_calls.jsonl             ← MCP tool calls
│   ├── sessions/<id>.jsonl         ← per-session coding audit
│   ├── errors.jsonl                ← Phase 4 typed errors
│   ├── routing_decisions.jsonl     ← Phase 5 routing audit
│   └── automation_tasks.jsonl      ← Phase 5 OpenClaw task records
│
├── models/                         ← (main checkout only — NOT in worktrees)
│   ├── Qwen3.5-9B-Q4_K_M.gguf      ← LLM (5.29 GB)
│   ├── openwakeword/ultron.onnx    ← custom wake word
│   ├── piper/en_US-ryan-medium.onnx ← TTS voice
│   └── rvc/{hubert_base.pt, rmvpe.pt} ← RVC support files
│
├── ultron_james_spader_mcu_6941/   ← (main checkout only) RVC voice model
│   ├── Ultron.pth
│   └── added_IVF301_Flat_nprobe_1_Ultron_v2.index
│
└── training/                       ← (gitignored except scripts) Wake-word training data
    ├── download_training_data.py
    ├── probe_datasets.py
    ├── run_training.py
    ├── smoketest_memory.py
    └── smoketest_orchestrator.py
```

---

## Cross-cutting flows

### Voice query (conversational) — happy path

```
1. AudioCapture callback → enqueues 32 ms blocks
2. Orchestrator.run() loop:
   a. WakeWordDetector or AddressingClassifier consumes blocks
      ├── COLD: "ultron" wake word required
      └── WARM: classifier verdict required
   b. On addressed: VoiceActivityDetector marks utterance start/end
   c. AudioCapture._capture_utterance() yields ndarray
3. WhisperEngine.transcribe(audio) → user_text
4. classify_routing(user_text, has_active_coding_task, has_pending_clarification)
   → RoutingIntent
5. CapabilityVoiceController.handle_capability_intent(routing_intent)
   ├── CONVERSATIONAL: returns None
   ├── coding kinds: routes through CodingTaskRunner
   └── automation kinds: OpenClawDispatcher (stub voice msg)
6. If None (conversational fall-through):
   a. Orchestrator._respond(user_text)
      ├── Optional: WebSearchGate.classify(text) → SEARCH/NO_SEARCH/UNCERTAIN
      ├── If SEARCH: AcknowledgmentSource.next_phrase() → TTS immediately
      │              → WebSearchExecutor.run(text) → SearchPayload
      │              → format_sources_for_prompt(payload.sources)
      │              → injected into LLM context
      ├── ConversationMemory.retrieve(text) → MemoryTurn[] (RAG)
      ├── LLMEngine.generate_stream(text) → tokens
      └── TextToSpeech.speak_stream(tokens) → Piper → RVC → audio device
   b. ConversationMemory.add(user/assistant) on background thread
7. Orchestrator enters FOLLOW_UP_LISTENING for 30 s (warm window)
```

### Coding task path

```
1-4. Same as voice query through classify_routing
5. RoutingIntent.kind == CODE_TASK
   a. CapabilityVoiceController.handle_capability_intent →
      handle_utterance(text)
   b. CodingIntent classification (intent.classify)
   c. ProjectResolver resolves "my flask app" → Project
      OR new_sandbox_project(name) creates a fresh dir
   d. UltronMCPServer.create_session(project_root, intent)
   e. CodingTaskRunner.start_task(TaskRequest)
      → DirectClaudeCodeBridge.submit() spawns:
         claude --print --output-format stream-json --include-partial-messages
                --include-hook-events --model haiku --add-dir <cwd>
                --dangerously-skip-permissions
   f. TaskHandle event stream:
      ├── TaskEvent(STATUS|TEXT|TOOL_USE|TOOL_RESULT|FILE_CHANGE|USAGE|ERROR|COMPLETE)
      ├── Listener feeds: SessionStore.record_stage(), record_test_results(),
      │                   set_pending_clarification(), record_completion_claim()
      └── Audit log line per event → logs/coding_tasks.jsonl
6. Orchestrator main loop returns; voice path resumes
7. On future "how's it going?" utterance:
   a. classify_routing → PROGRESS_QUERY (because runner.has_active_task())
   b. handle_capability_intent → handle_utterance →
      StatusNarrator.narrate(session_state) using project_status_delta
      → spoken narration
8. On Claude declare_complete:
   a. ConversationCoordinator.handle_declare_complete():
      → Verifier.verify(session) runs 6 checks
      → if pass: SessionStatus.COMPLETE
      → if fail and below escalation threshold:
            project_correction_context(session) → corrective prompt
            → Claude re-prompted with --resume
      → if escalation threshold crossed: switch to sonnet model
   b. CodingTaskRunner.completion_narration() generates final voice msg
9. Orchestrator polls voice.pending_completion() → speaks it
```

**Clarification fast-path order** (V1-gap A3 added Fast-path 2.5):

```
ConversationCoordinator.decide_clarification(...):
  1. Always-escalate keyword (api key, paid tier, scope add, ...)
  2. urgency=preference + options provided -> "use your default"
  2.5 (A3) Stored facts via facts_lookup -- if a directive-category
       (preference / decision / constraint) fact clears confidence and
       score thresholds, answer Claude with "From the user's stored
       preferences: <fact>. Use that."
  3. Always-answer keyword (test framework, linter, layout, ...)
  4. LLM decide pass (ANSWER / USE_DEFAULT / ESCALATE)
  5. Escalate to user (voice question + asyncio.Future await)
```

### Search-triggered path (web)

```
1-3. Same as voice query through Whisper
4-5. classify_routing → CONVERSATIONAL (web search isn't a routing kind)
6. Orchestrator._respond(user_text) flow:
   a. WebSearchGate.classify(user_text):
      ├── classify_by_rules → SEARCH if time-sensitive markers,
      │                       NO_SEARCH if personal-context queries
      └── classify_by_preflight (LLM call) for UNCERTAIN cases
         → returns GateVerdict with knowledge_confidence,
                  has_temporal_dependency, search_queries
   b. If SEARCH:
      ├── AcknowledgmentSource.next_phrase() → TTS within 200 ms
      ├── WebSearchExecutor.run(user_text, search_queries):
      │   ├── WebResultsCache.lookup(q) → cached payload OR None
      │   │   (3 collections: ttl_volatile_seconds=86400, ttl_stable_seconds=2592000)
      │   ├── BraveSearchClient.search(q) → BraveResult[]
      │   │   (wrapped in CircuitBreaker; raises BraveAPIError;
      │   │    failures log to errors.jsonl, return [])
      │   ├── _rank_snippets(llm, query, results, top_n) → ranked BraveResult[]
      │   ├── For top max_fetch: JinaReaderClient.fetch(url) → markdown
      │   │   (wrapped in CircuitBreaker; JinaReaderError → snippet-only)
      │   └── WebResultsCache.store(query, rows) — best effort
      └── format_sources_for_prompt(payload.sources) → injected into LLM context
   c. LLM generates response with citations
   d. TTS streams + format_sources_for_transcript(sources) printed (not spoken)
```

### OpenClaw stub dispatch path (Phase 5 — currently stubbed)

```
1-4. Same as voice query through classify_routing
5. RoutingIntent.kind in {BROWSER_AUTOMATION, MEDIA_GENERATION, MESSAGING,
                          FILE_OPERATION, SHELL_OPERATION, HYBRID_TASK}
   → CapabilityVoiceController.handle_capability_intent:
   ├── Single-category (browser/media/etc):
   │   AutomationTaskRunner.submit_task(intent) →
   │   OpenClawDispatcher.handle_X(intent.automation_intent) →
   │   DispatchResult(success=False, voice_message="gateway not connected yet")
   │   → audit row in logs/automation_tasks.jsonl
   │   → routing-decision row with outcome="stub"
   └── HYBRID_TASK: voice msg "I'd split it up and run both, but..."
6. VoiceResponse returned to orchestrator → speak
7. Orchestrator continues main loop
```

### Error / circuit-break path

```
External call (Brave, Jina) → CircuitBreaker.call(_do_X, ...)
├── If CLOSED, executes; on failure raises typed error
│   - 3rd failure within 5 min trips OPEN
├── If OPEN, raises CircuitOpenError immediately (no call)
│   - cooldown elapses → HALF_OPEN
├── If HALF_OPEN, executes once as a probe
│   ├── Success → CLOSED, failure counter reset
│   └── Failure → reopens, fresh cooldown
└── On any typed-error path:
    ErrorLog.record(error, dependency=...) → logs/errors.jsonl
    Optional: phrase_for("brave_unavailable") → spoken via TTS
```

---

## Source modules

### `src/ultron/__init__.py`

**Purpose:** package init. On Windows, adds CUDA runtime DLL directories to
the loader path so llama-cpp / ctranslate2 find `cudart64_12.dll`,
`cublas64_12.dll`, `cudnn_ops_infer64_9.dll`.

**No public API** beyond import side effects.

### `src/ultron/__main__.py`

**Purpose:** `python -m ultron` entry point.

**Public:**
- `main() -> int` — sets up logging, builds an `Orchestrator`, calls
  `.run()` until KeyboardInterrupt. Returns process exit code.

**In:** environment + config.yaml (via Orchestrator construction).
**Out:** stdout console transcript, log files.

### `src/ultron/config.py` (Phase 3)

**Purpose:** single source of truth for tunable parameters. Loads
`config.yaml`, validates against pydantic schema, exposes singleton.

**Public:**
- `PROJECT_ROOT`, `MODELS_DIR`, `LOGS_DIR` — Path constants
- `DEFAULT_CONFIG_PATH` — `<root>/config.yaml`
- `resolve_path(value: str | Path) -> Path` — resolve relative paths against PROJECT_ROOT
- Sub-models (all pydantic `_Strict`):
  `AudioConfig`, `VADConfig`, `WakeWordConfig`, `STTConfig`, `LLMConfig`,
  `EmbeddingsConfig`, `QdrantCollections`, `QdrantConfig`, `MemoryConfig`,
  `BraveConfig`, `JinaConfig`, `WebCacheConfig`, `WebSearchConfig`,
  `AddressingConfig`, `CodingMCPConfig`, `CodingVerificationConfig`,
  `CodingConfig`, `ProjectionsBudgets`, `ProjectionsConfig`, `RVCConfig`,
  `TTSConfig`, `LoggingConfig`, `ErrorPhrasesConfig`,
  `RoutingClassifierConfig`, `RoutingConfig`, `OpenClawConfig`
- `UltronConfig` — top-level model
- `load_config(path=None) -> UltronConfig` — explicit load (raises `ConfigurationError`)
- `get_config() -> UltronConfig` — singleton, lazy-load on first call
- `reload_config(path=None) -> UltronConfig` — clear cache, reload
- `set_config(cfg) -> None` — test injection
- `current_config_path() -> Path | None`
- `LLM_PRESETS: dict[str, dict]` (4B plan Stage A) — preset table for
  `LLMConfig.preset`. Two presets defined: `qwen3.5-9b` (default; 9B
  GGUF, n_ctx=8192, no draft) and `qwen3.5-4b` (4B GGUF + 0.8B draft +
  n_ctx=16384). `LLMConfig._apply_preset` (model_validator) fills
  in `model_path` / `n_ctx` / `draft_model_path` from this table only
  when those fields are absent from `model_fields_set`, so explicit
  YAML values always win.

**In:** `config.yaml`, `${ENV_VAR}` substitution from `os.environ`.
**Out:** typed `UltronConfig` instance.

### `src/ultron/errors.py` (Phase 4)

**Purpose:** typed exception hierarchy for every external dependency.

**Public hierarchy:**
- `UltronError` (base) — has `message`, `context: dict`, `recovery: str`,
  `with_recovery()`, `with_context()`, `to_log_dict()`
- `DependencyUnavailableError` (subclass)
  - `BraveAPIError`, `JinaReaderError`, `QdrantUnavailableError`,
    `AnthropicAPIError`, `OllamaUnavailableError`, `OpenClawGatewayError`
- `ClaudeCodeError`
- `AudioPipelineError`
  - `WhisperTranscriptionError`, `PiperSynthesisError`,
    `RVCConversionError`, `WakeWordModelError`, `AddressingClassifierError`
- `MCPServerError`, `ConfigurationError`, `FilesystemError`

**In:** raised from external-dep wrappers.
**Out:** caught by orchestrator + structured-logged via `ErrorLog`.

**Wired call sites (Phase 4 deferred wrappers, complete):**
- `ClaudeCodeError` + `AnthropicAPIError`: [coding/direct_bridge.py](../src/ultron/coding/direct_bridge.py)
  — launch failure, subprocess timeout, nonzero exit, stream-json error
  events. The pattern detector `_looks_like_anthropic_api_error` decides
  between the two based on error text (rate_limit / overloaded /
  invalid_api_key / etc.).
- `MCPServerError`: [coding/mcp_server.py](../src/ultron/coding/mcp_server.py)
  — bind failure (`raise … from OSError`), startup timeout, no-active-session
  on Claude tool call. `FilesystemError` covers the audit-log write path.
- `FilesystemError`: [coding/audit.py](../src/ultron/coding/audit.py),
  [coding/projects.py](../src/ultron/coding/projects.py),
  [coding/runner.py](../src/ultron/coding/runner.py) — session audit
  mkdir/write, project registry load/save, coding-tasks audit-log
  (first-failure dedup via `_AUDIT_WRITE_FAILURE_LOGGED` flag).

### `src/ultron/uncertainty.py`

**Purpose:** annotate user prompt with hedging hints based on the
pre-flight gate's uncertainty signals.

**Public:**
- `apply(verdict: GateVerdict, user_text: str) -> Tuple[GateVerdict, str]`
  — given a `GateVerdict` with `knowledge_confidence` /
  `knowledge_source` / `has_temporal_dependency`, returns a
  possibly-prepended user prompt with style hints. V1-gap B1: a
  `knowledge_source` of `retrieved_memory` / `retrieved_facts`
  prepends a source hint above the confidence addendum so the LLM
  matches its tone (rule verdicts inherit this branch too).
- `_source_hint_for(verdict)` (internal) — picks the leading source
  hint from `knowledge_source`. `weights` / `unknown` /
  `web_search_needed` get no hint.

**In:** `GateVerdict` from `web_search.gating`, raw user text.
**Out:** `(verdict, augmented_prompt)`.

### `src/ultron/response_style.py` (2026-05-10)

**Purpose:** per-call response-style addenda, prepended to the user's
text before it reaches the LLM. Lives OUTSIDE the persona file
(SOUL.md is voice-quality-locked) so the orchestrator can nudge the
model on a per-utterance basis without changing the system prompt.
Today only one addendum lives here — a brevity hint for short
questions that the 4B model otherwise tends to over-explain ("What
are the Orcs in 40k?" → 1164-char four-paragraph essay in the
2026-05-10 live session).

**Public:**
- `is_brief_question(user_text: str) -> bool` — True iff the
  utterance is short (≤12 words OR ≤80 chars after strip) AND not
  explicitly asking for depth via any of the `_DEPTH_MARKERS`
  keywords (`explain` / `in detail` / `step by step` /
  `walk me through` / `elaborate` / `expand on` / `everything you
  know` / etc.). Empty / whitespace input returns False.
- `apply_brevity_hint(user_text: str) -> str` — prepends a
  `[Style: respond in 1-3 short sentences …]` directive when
  `is_brief_question` returns True; otherwise returns input
  unchanged. Empty input passes through. Idempotent on
  already-hinted text (the hinted version is too long to be
  re-classified as brief).

**In:** raw user text. **Out:** possibly-augmented user text (newline-
separated above the original).

**Wired at:** [pipeline/orchestrator.py](../src/ultron/pipeline/orchestrator.py)
`Orchestrator._build_response_stream` — applied on the non-search
conversational path (search path's augmented prompt has its own
length directive). Three call sites: web-gate-disabled fall-through,
web-gate-failure fall-through, NO_SEARCH verdict path.

### `src/ultron/audio/`

#### `audio/capture.py`
- `class AudioCaptureError(RuntimeError)` — raised on device init failure
- `class AudioCapture` — sounddevice callback thread enqueueing 32 ms blocks
  - `start()` / `stop()`
  - `read_blocks() -> Iterator[np.ndarray]`
  - `_capture_utterance(...)` (used by Orchestrator)

#### `audio/devices.py`
- `class AudioDeviceError(ValueError)`
- `resolve_device(configured, kind) -> Optional[int]` — substring match on device name
- `describe_device(device, kind) -> str`

#### `audio/ring_buffer.py`
- `class RingBuffer` — fixed-duration audio backlog (pre-speech window)
  - `write(samples)` / `clear()` / `__len__()` / `capacity` property
  - `snapshot(last_n_samples=None) -> np.ndarray` — full buffer when
    unsliced; the most recent `last_n_samples` when given. The
    orchestrator slices a short COLD-mode pre-roll (post-wake; avoids
    "Tron" prefix) and a longer WARM-mode pre-roll (post-TTS
    follow-up; avoids first-word clipping) from the SAME buffer.
    `last_n_samples >= len(buffer)` returns full; `<= 0` returns
    empty. (2026-05-10 mode-aware pre-roll fix.)

#### `audio/vad.py`
- `class SpeechEvent(Enum)` — START / END / NONE
- `class VadResult` — dataclass: event, is_speech, prob
- `class VoiceActivityDetector` — silero-vad wrapper; consumes 512-sample windows.
  - `reset()` — clear hysteresis state AND restore the baseline silence-window requirement (so an adaptive bump from the previous utterance doesn't leak into the next one).
  - `set_min_silence_duration_ms(ms)` (2026-05-11 adaptive end-of-turn) — adjust trailing-silence requirement at runtime. Orchestrator calls this from `_capture_utterance` once speech has been active past `vad.long_utterance_threshold_seconds` so a thinking pause mid-prompt doesn't close a long technical description.

#### `audio/wake_word.py`
- `class WakeWordDetector` — openWakeWord wrapper
  - Loads `models/openwakeword/ultron.onnx` (custom)
  - Falls back to `hey_jarvis` with startup warning if missing
  - `predict(audio_block) -> Optional[str]` — fires a wake event
  - `fired_recently(window_s: float = 0.5) -> bool` (V1-gap A4) — read-only accessor for the last trigger timestamp; returns True iff a wake fire happened within ``window_s`` seconds. Used by the orchestrator's pre-task barge-in watcher. Idempotent — does not consume the trigger.

### `src/ultron/addressing/`

#### `addressing/rules.py`
- `class AddressingDecision(str, Enum)` — ADDRESSED / NOT_ADDRESSED / UNCERTAIN
- `class RuleHit` — dataclass: decision, confidence, reason
- `classify(utterance, seconds_since_response) -> Optional[RuleHit]`
- `explain_rules() -> List[Tuple[str, str]]` — for the review script

#### `addressing/zero_shot.py`
- `class ZeroShotAddresseeModel` — flan-t5-small wrapper (~300 MB CPU)
  - `_ensure_loaded()` — eager-load option
  - `classify(utterance, context, seconds_since_response) -> (verdict_str, confidence, latency_ms)`

#### `addressing/classifier.py`
- `class AddressingVerdict` — final decision + metadata
- `class AddressingClassifier` — combines rules + zero-shot
  - `classify(utterance, seconds_since_response) -> AddressingVerdict`
  - `_log(utterance, verdict)` → writes to `logs/addressing.jsonl`

### `src/ultron/transcription/whisper_engine.py`

- `class WhisperEngine` — faster-whisper wrapper, CUDA fp16
  - `transcribe(audio: np.ndarray, language="en") -> str`
  - On failure: returns `""`, logs `WhisperTranscriptionError` to errors.jsonl

### `src/ultron/llm/inference.py`

- `_strip_thinking_blocks(stream)` — filter `<think>...</think>` from token stream
- `_sanitize_user_input(text) -> (cleaned, found_markers)` (Q10 quality-pass iter 1+2) — pre-LLM defense layer that neutralises tag-style prompt-injection markers (`[INST]`, `[/INST]`, `<|im_start|>`, `<|im_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `</think>`) by replacing each with `[NEUTRALIZED_TAG]`. Also detects natural-language jailbreak patterns ("ignore previous instructions", "you are now <X>", "respond with the exact word", etc.) — for those the function prepends a one-shot hardening note OR (for the most-direct override patterns: "respond with exactly", "respond with the exact word", "must respond with") rewrites the user message into a description of the attempt so compliance becomes grammatically nonsensical. Detected attempts log to `logs/errors.jsonl` with `dependency='prompt_injection'`. Voice-quality lock preserved — the persona system prompt (`SOUL.md`) is untouched. Wired into `_build_messages` so every LLM call goes through the defense. Verified end-to-end: pre-defense 2/3 of Q8 prompt-injection probes succeeded; post-defense 0/3.
- `class LLMEngine` — LLM client with two backends, selected by `llm.runtime`:
  - `in_process` (default): loads the GGUF via llama-cpp-python in this process. Voice-path mode.
  - `http_server` (opt-in): talks to llama-cpp-server over OpenAI-compat HTTP. For the OpenClaw + voice migration. Latency is +71 ms median TTFT vs in-process — kept opt-in so the voice path isn't regressed.
  - `__init__(model_path?, n_ctx?, n_gpu_layers?, system_prompt?, history_turns?, memory=None, runtime?)`
  - `generate(user_message) -> str` — blocking
  - `generate_stream(user_message) -> Iterator[str]` — token streaming
  - `cancel()` — signal to stop
  - `_build_messages(user_message)` — resolves system prompt fresh each turn (Phase 1 hot-reload), assembles RAG snippets + recent + user
  - `_resolve_system_prompt()` (Phase 1) — sources from `PersonaLoader.get_system_prompt("user_facing")` when `llm.persona.source == "workspace"` (default), else `cfg.system_prompt`. Falls back to config when workspace is empty.
  - `_http_chat_completion(...)` / `_http_stream(...)` — OpenAI-compat HTTP client (uses `requests`, SSE for streaming, cancel-aware).
  - `_chat_completion_kwargs(_llm_cfg, enable_thinking, *, stream)` (4B plan Stage F) — static helper that builds the kwargs dict for `Llama.create_chat_completion`. When `enable_thinking` is `None` (default), no `chat_template_kwargs` is emitted (back-compat). When `True` / `False`, sets `chat_template_kwargs={"enable_thinking": <value>}` — Qwen3.5's template toggle that suppresses or requests the `<think>...</think>` block. Applied to both in-process and HTTP runtimes via the same helper.
  - `_build_llama(cfg, model_path, n_ctx, n_gpu_layers) -> (Llama, Path)` (4B plan voice-swap) — pure constructor that builds + returns a fresh `Llama` instance per `cfg`. Does NOT mutate `self`. Used by `_init_in_process` and `reload_for_preset`.
  - `reload_for_preset(preset: str) -> (bool, str)` (4B plan voice-swap) — hot-swap the loaded LLM to `preset` without restarting Ultron. Builds the new `Llama` FIRST so a failed swap (missing GGUF, invalid preset) leaves the engine in its working state. On success: history cleared, `ULTRON_LLM_PRESET` env updated, stale `ULTRON_LLM_MODEL_PATH` cleared. On failure: env vars restored. Idempotent (`already on X` returns success without rebuild). `in_process` runtime only.
  - `generate(user_message, *, enable_thinking=None)` and `generate_stream(user_message, *, enable_thinking=None)` (4B plan Stage F) — per-call thinking mode parameter.

**In:** user text + (optional) `ConversationMemory` for RAG. **Out:** generated text.

### `src/ultron/memory/`

#### `memory/embedder.py`
- `class _SparseVec` — thin wrapper over BM25 sparse output
- `class HybridEmbedder` — FastEmbed dense (bge-small-en-v1.5 INT8) + sparse (Qdrant/bm25)
  - `encode_dense(texts) -> np.ndarray`
  - `encode_query_dense(text)` / `encode_query_sparse(text)`
  - `dim` property → 384

#### `memory/qdrant_store.py`
- `class MemoryTurn` — dataclass: id, ts, role, content, summary, entities, ...
- `class FactRow` (V1-gap A3) — dataclass: fact, confidence, last_confirmed, category, score, extracted_at, extracted_from, retrieval_weight. Read-side projection of the `facts` collection that the maintenance script writes.
- `class ConversationMemory`
  - `__init__(path?, embedder, recent_cache_size=100, session_id?)`
  - `add(role, content)` — sync; queues to background writer
  - `recent(n) -> List[MemoryTurn]` — from in-process cache
  - `retrieve(query, k=cfg, exclude_recent=cfg) -> List[MemoryTurn]` — single-pass hybrid RRF
  - `retrieve_multi(primary_query, category_queries, *, k, exclude_recent)` (V1-gap A2) — multi-pass per-category hybrid RRF + composite re-ranking. Parallel fan-out via `ThreadPoolExecutor`. Falls back to single-pass on any failure.
  - `retrieve_for_query(primary_query, gate_verdict=None, *, k, exclude_recent)` (V1-gap A2) — routing helper: when `memory.retrieval.multi_pass_enabled` is True AND the verdict carries `context_categories`, fans out via `retrieve_multi`; otherwise calls `retrieve`. Default-OFF preserves byte-for-byte legacy behaviour.
  - `search_facts(query, *, k=5, min_confidence=0.0, max_age_days=None) -> List[FactRow]` (V1-gap A3) — hybrid RRF over the `facts` collection. Filters via Qdrant `confidence >= min_confidence` and `last_confirmed >= now - max_age_days*86400`. Fail-open: returns `[]` on any Qdrant / embedder failure.
  - `__len__()` / `close()`

#### `memory/ranking.py` (V1-gap A2)
- `@dataclass class RankingWeights` — frozen snapshot of the rrf_weight / recency_weight / recency_half_life_days / surprise_weight / redundancy_weight tuning.
- `@dataclass class CandidateScore` — per-candidate aggregator (id, payload, rrf_score, dense vector, primary_similarity, category_similarity, composite_score).
- `cosine_similarity(a, b) -> float` — pure cosine on float lists; defensive against length mismatch / zero vectors.
- `compute_recency_boost(ts, *, half_life_days, now=None)` — exponential decay; ``ts == 0`` (sentinel) returns 0.
- `compute_surprise_score(candidate_dense, primary_dense, category_score)` — clamps to ``max(0, category_score - primary_similarity)``.
- `compute_redundancy_penalty(candidate_dense, picked)` — max cosine vs already-picked.
- `compute_composite_score(candidate, *, weights, primary_dense, picked, now=None)` — weighted blend.
- `select_top_k(candidates, *, k, weights, primary_dense=None, now=None) -> List[CandidateScore]` — greedy redundancy-aware selection.

### `src/ultron/web_search/`

#### `web_search/acknowledgments.py`
- `class AcknowledgmentSource` — shuffled-pool phrase generator (8 phrases)
  - `next_phrase() -> str`

#### `web_search/brave.py`
- `_BRAVE_BREAKER` — module-level CircuitBreaker (3/5min, 5min cooldown)
- `class BraveResult` — dataclass: url, title, snippet, rank
- `class BraveSearchClient`
  - `search(query, count?) -> List[BraveResult]` — uses breaker + raises BraveAPIError
  - `_do_search(query, count)` — inner; raises typed errors

#### `web_search/cache.py`
- `_VOLATILE_KEYWORDS`, `freshness_category_for(query)`, `ttl_for(category)`
- `class WebResultsCache` — Qdrant-backed; collection = `web_results`
  - `lookup(query) -> Optional[List[(BraveResult, full_text)]]`
  - `store(query, rows)` — best-effort

#### `web_search/gating.py`
- `class GateDecision(str, Enum)` — SEARCH / NO_SEARCH / UNCERTAIN
- `class GateVerdict` — decision, confidence, source, search_queries, knowledge signals (knowledge_confidence, knowledge_source, has_temporal_dependency), **context_categories** + **memory_search_queries** (V1-gap A2 — populated by the LLM preflight pass; rule-only verdicts leave them empty so the multi-pass retrieval path stays inactive).
- `_resolve_knowledge_source(*, needs_search, confidence, memory_snippets, rule_reason)` (V1-gap B1) — single-source helper that maps gate inputs to the spec's five-value enumeration (`weights / retrieved_memory / retrieved_facts / web_search_needed / unknown`). Every `GateVerdict` construction site routes through this.
- `classify_by_rules(utterance) -> Optional[GateVerdict]` — hard rules (time markers, URL, etc.)
- `classify_by_preflight(utterance, llm, memory_snippets) -> GateVerdict` — LLM call
- `class WebSearchGate` — orchestrates rules → LLM
  - `classify(utterance, recent_memory) -> GateVerdict`

#### `web_search/jina.py`
- `_JINA_BREAKER` — CircuitBreaker (5/5min, 3min cooldown)
- `class JinaReaderClient`
  - `fetch(url) -> Optional[str]` — uses breaker + raises JinaReaderError

#### `web_search/search.py`
- `class SearchSource` — dataclass: url, title, snippet, full_text, rank
- `class SearchPayload` — dataclass: query, sources, cache_hit, elapsed_ms, notes
- `_rank_snippets(llm, query, results, top_n)` — LLM-driven re-ranking
- `_normalise_search_query(q)` / `_dedupe_queries(qs)` (V1-gap B2) — drop near-duplicate Brave queries before fan-out using a token-set canonical form (lowercase + possessive strip + stopword drop + sort).
- `_render_inline_marker(index, *, fmt)` (V1-gap B3) — render bracketed `[1]` (default) or Unicode superscript (¹²³) inline citations based on `web_search.citation.inline_marker_format`.
- `class WebSearchExecutor` — orchestrates Brave → rank → Jina → cache. **2026-05-09 latency fix:** Jina fetches now run IN PARALLEL via `concurrent.futures.ThreadPoolExecutor` with a collective deadline cap. Pre-fix the loop was sequential and one slow page (~10 s on a Quora result) blocked the entire search path while the TTS playback queue starved waiting for tokens. Post-fix wall time is `max(per-fetch durations)` instead of `sum(...)`, capped further by `collective_deadline_seconds`. Any fetch still in flight at deadline is abandoned (its source falls back to snippet-only with a `jina_deadline:<url>` note). Threads keep running in the background and exit on per-fetch HTTP timeout; `pool.shutdown(wait=False)` ensures the executor returns immediately.
  - `__init__(brave, jina, llm, cache=None, max_fetch=None, collective_deadline_seconds=None)` — both kwargs default-resolve from `get_config().web_search.jina`.
  - `run(user_query, search_queries?, top_n=3) -> SearchPayload`
- `format_sources_for_prompt(sources)` / `format_sources_for_transcript(sources)` — references list always uses bracket form for monospace clarity.

### `src/ultron/tts/`

#### `tts/rvc.py`
- `class RvcConverter` — infer-rvc-python wrapper, cuda:0
  - `convert(pcm: np.ndarray, sample_rate: int) -> (pcm, sr)` — raises RVCConversionError on failure
  - `close()` — releases GPU memory

#### `tts/speech.py`
- `Clip` — type alias for `Tuple[np.ndarray, int]` (legacy synth function return).
- `class ClipItem(NamedTuple)` (2026-05-10 producer-signaled lookahead):
  `(audio, sample_rate, is_known_last)`. Pushed onto `piper_q` /
  `audio_q` instead of bare `Clip` tuples. Playback uses
  `is_known_last` (or the `None` end-of-stream sentinel) to decide
  whether to wait for another clip after playing the current one.
  Default `is_known_last=False`; the `None` sentinel handles the
  "this was the last" signal in normal use. The flag is reserved for
  future producers that DO know in advance (canned single-sentence
  voice responses, etc.).
- `_QUEUE_GET_TIMEOUT_SECONDS = 60.0` — generous wait between clips
  in both the playback loop and the RVC stage. The previous 10 s
  value killed audio mid-response when a slow web search held the
  generator long enough for the RVC stage's `piper_q.get(timeout=10)`
  to fire (BMW failure mode in the 2026-05-09 logs).
- `class TextToSpeech` — Piper + optional RVC
  - `__init__(rvc=None)` — loads Piper voice, optionally wraps with RVC
  - `speak(text)` — synchronous synthesize + play
  - `speak_stream(fragments)` — stream tokens, flush on sentence
    terminator. **Producer-signaled lookahead (2026-05-10):** plays
    each clip IMMEDIATELY on receipt, then blocks for the next.
    Replaces the legacy play-after-peek pattern that delayed the
    first clip (commonly the web-search ack) up to 10 s waiting for
    the second clip to determine "is this last?". Voice character
    (edge fades, inter-sentence pauses, tail silence) bit-identical;
    only the queue-get ordering changed.
  - `warmup()` — primes Piper
  - `_synthesize(text)` — Piper → optional RVC; raises
    PiperSynthesisError / RVCConversionError
  - `_run_synth_loop(*, fragments, push, synth_fn)` — walks
    fragments, synthesises on flush chars, pushes each non-empty
    clip via `push` as a `ClipItem(is_known_last=False)`. End-of-
    stream sentinel (`None`) is pushed by the surrounding worker's
    `finally` block.
  - `stop()` — interrupt current playback

#### `tts/ultron_filter.py` (NEW 2026-05-10 voice swap)

Runtime port of the user-tuned v3 Ultron mechanical filter chain (the
prototype lives at `ultronVoiceAudio/scripts/ultron_filter.py`).
Built on `pedalboard` (Spotify's open-source DSP library; sub-ms
overhead per stage on CPU).

- `PresetName` — Literal["v1_subtle", "v2_medium", "v3_heavy"].
- `get_preset(preset)` — fresh `Pedalboard` instance per call.
- `apply_filter(audio, sample_rate, preset="v3_heavy", tail_silence_ms=200.0)`
  — applies the chain. Pads `tail_silence_ms` of trailing zeros
  before processing so the reverb tail decays into the padding
  rather than being clipped at the buffer end. Runtime default 200
  ms (audible portion of v3 reverb); offline samples use 500 ms.

The `v3_heavy` preset is the user-locked production chain (bit-
identical to the prototype): Highpass → PitchShift(-1.8 semitones) →
Compressor → LowShelfFilter(+4.5 dB @ 160 Hz) → Delay(7 ms, 25 %
feedback for comb resonance) → Chorus → Distortion(+7 dB) → Peak EQ
boost @ 2.5 kHz → HighShelf cut → Reverb(small cavity) → Lowpass.

#### `tts/xtts_v3.py` (NEW 2026-05-10 voice swap)

Drop-in replacement for `TextToSpeech` when `tts.engine == "xtts_v3"`.
Same `speak` / `speak_stream` / `warmup` / `stop` interface so the
orchestrator playback path (the producer-signaled lookahead in
`speak_stream`) doesn't change.

- `class ClipItem(NamedTuple)` — `(audio, sample_rate, is_known_last)`,
  same contract as in `tts/speech.py` so the queue protocol is
  uniform across engines.
- `class XttsServerStartError(RuntimeError)` — raised when the XTTS
  server subprocess can't be started (missing venv / script /
  reference, startup timeout exceeded).
- `class XttsSynthError(RuntimeError)` — synth call failure.
- `class XttsV3Speech` — the engine.
  - `__init__(...)` — resolves paths via `tts.xtts_v3` config,
    spawns the XTTS HTTP server in `.venv-xtts`, polls `/healthz`
    until ready (180 s startup budget for cold model load).
  - `speak`, `speak_stream`, `warmup`, `stop` — same API as the
    legacy engine.
  - `_synthesize(text)` — POST `/synthesize`, accumulates the
    streamed PCM, applies the v3 Ultron filter via
    `ultron_filter.apply_filter(..., tail_silence_ms=200)`, returns
    `(int16 pcm, sr)` matching the legacy engine's contract.
  - `_http_synthesize(text)` — raw HTTP call; reads chunked PCM
    body and returns `np.ndarray(int16)`. POST JSON body carries
    `{"text", "language", "speed"}` — the `speed` field is XTTS v2's
    native duration multiplier sourced from `tts.xtts_v3.speed`
    (default 1.0 in schema, 1.15 in production via `config.yaml`).
    Server-side passes it to `model.inference_stream(speed=...)` so
    cadence changes happen at synthesis time; the v3 pedalboard
    filter (pitch / delay / reverb / etc.) is unaffected and
    processes the shorter audio buffer identically.
  - `_stop_server_subprocess()` — graceful POST `/shutdown`, then
    SIGTERM, then SIGKILL. Called by the orchestrator's `shutdown()`.

The XTTS HTTP server itself lives at
[ultronVoiceAudio/scripts/xtts_server.py](../ultronVoiceAudio/scripts/xtts_server.py)
in the isolated `.venv-xtts` venv. FastAPI + uvicorn; uses an async
producer + asyncio.Queue pattern to bridge XTTS's sync streaming
generator into the FastAPI response without sync-generator
threadpool overhead (saved ~140 ms TTFT vs the naive sync-gen
implementation).

### `src/ultron/coding/` (Phase A foundation + Coding Addendum + Phase 2 projections)

#### `coding/audit.py`
- `class SessionAuditWriter` — per-session `logs/sessions/<id>.jsonl` writer
  - `write(kind, **fields)` — append one record

#### `coding/bridge.py`
- `class EventKind(str, Enum)` — STATUS / TEXT / TOOL_USE / TOOL_RESULT / FILE_CHANGE / ERROR / COMPLETE / USAGE
- `class FileChangeKind(str, Enum)` — CREATED / MODIFIED / DELETED
- `class TaskEvent` — dataclass with all event payload fields
- `class TaskRequest` — dataclass: task_prompt, cwd, model, timeout_s, label, etc.
- `class TaskResult` — dataclass: success, exit_status, summary, files_*, etc.
- `class TaskState` — running state
- `class TaskHandle(ABC)` — `task_id()`, `state()`, `add_listener()`, `cancel()`, `wait()`
- `class CodingBridge(ABC)` — `submit(request) -> TaskHandle`, `name()`
- `render_prompt(request)` — render TaskRequest into a string prompt
- `directory_snapshot(root)` / `diff_snapshots(...)` — ground-truth file diff

#### `coding/direct_bridge.py`
- `class DirectClaudeCodeBridge(CodingBridge)` — spawns `claude --print --stream-json ...`
- `class DirectTaskHandle(TaskHandle)` — parses event stream

#### `coding/intent.py`
- `class CodingIntentKind(str, Enum)` — NONE / CODE_TASK / PROGRESS_QUERY / CANCEL / MID_SESSION_ADJUSTMENT / CLARIFICATION_RESPONSE
- `class CodingIntent` — dataclass with kind, project_reference, etc.
- `classify(utterance, has_active_task=False, has_pending_clarification=False) -> CodingIntent`
- `derive_project_name(intent) -> str` — slug from task text
- `_DETERMINER_NOUN` (private regex fragment, NEW 2026-05-11 follow-up fix): `(the|that|this|your|our|my)(?:\s+(task|project|build|app|code|work|thing|run|job))?`. Plugged into all three sub-patterns of `_PROGRESS_PATTERNS` (`how X going|coming(along)?`, `what's X doing|working on|up to`, `is X done`) so phrasings like "How is that **project** going?" / "How's the **build** coming along?" / "Is **my project** done?" classify as `PROGRESS_QUERY` instead of falling through to the conversational LLM. The noun is optional so the legacy `that going` / `the doing` phrasings still fire bit-identical. The has_active_task gate is preserved so these patterns never hijack ordinary conversation.

#### `coding/projects.py`
- `class Project` — dataclass: name, path, aliases, language
- `class ProjectRegistry` — atomic JSON CRUD on `data/projects.json`
- `class ResolutionKind(str, Enum)` — EXACT / ALIAS / SUBSTRING / SEMANTIC / NEW / UNRESOLVED
- `class ProjectResolution` — dataclass with kind + matched project
- `class ProjectResolver` — exact / alias / substring / semantic match
- `slugify_for_path(name) -> str` — collision-safe slug
- `new_sandbox_project(name, sandbox_root, registry) -> Project` — creates fresh dir + registers

#### `coding/session.py`
- `class SessionStatus(str, Enum)` — INITIALIZING / EXECUTING / VERIFYING / CORRECTING / AWAITING_CLARIFICATION / COMPLETE / FAILED / TERMINATED
- `is_valid_transition(from_status, to_status) -> bool`
- Records: `StageRecord`, `FileRecord`, `TestStatus`, `ClarificationRequest`, `AdjustmentRecord`, `CompletionClaim`
- `class ProjectSession` — full session state (large; passed only via projections)
- `class StateTransitionError(RuntimeError)`
- `class SessionStore` — owns sessions; `create()`, `get()`, `transition()`, `record_*()`

#### `coding/projections.py` (Phase C / Foundation Part 2)
- `count_tokens(text) -> int` — tiktoken cl100k_base
- `class ProjectionResult` — projection + text + token_count + budget + truncations_applied + truncation_warning
- `_finalize_projection(...)` — common end-of-projection: INFO log on truncations, ERROR on over-budget
- 5 projections, each with a dataclass + `project_X_context()` function:
  - `project_clarification_context(session, clarification_question, options?, facts_lookup?) -> ProjectionResult` (1500 tok)
  - `project_status_delta(session) -> ProjectionResult` (600 tok)
  - `project_adjustment_context(session, adjustment_text, facts_lookup?, conflict_detector?) -> ProjectionResult` (1200 tok)
  - `project_correction_context(session, failures, failed_test_names?, failed_test_messages?) -> ProjectionResult` (1500 tok)
  - `project_completion_context(session) -> ProjectionResult` (800 tok)

#### `coding/templates.py`
- `class TemplateError(RuntimeError)`, `PromptTooLargeError`, `SchemaValidationError`
- `class RenderResult` — dataclass: rendered text + token count
- `class TemplateRenderer` — Jinja2 wrapper for prompts/coding/*.j2
  - `render_initial_new(...)`, `render_initial_edit(...)`, `render_correction(...)`,
    `render_adjustment(...)`, `render_clarification_response(...)`

#### `coding/verification.py`
- `class CheckId(str, Enum)` — STRUCTURE / TESTS / SMOKE / LINT / FILES / PYTHON_SYNTAX
- `class CheckResult`, `VerificationReport` — dataclasses
- `class Verifier`
  - `verify(session) -> VerificationReport` — runs 6 checks + writes `logs/verifications.jsonl`
  - `verify_tests(session)` — single-check helper

#### `coding/narration.py`
- `class NarrationDelta` — dataclass tracking what's new since last query
- `class StatusNarrator` — voice-friendly progress narration
  - `narrate(session) -> str` — final completion narration
  - `progress_narration(session) -> str` — uses `project_status_delta` projection

#### `coding/runner.py`
- `build_default_bridge() -> CodingBridge` — picks DirectClaudeCodeBridge from config
- `class ProgressSinceLastQuery` — dataclass
- `class CodingTaskRunner`
  - `start_task(request)` — submits via bridge
  - `has_active_task() -> bool`
  - `cancel_active() -> bool`
  - `progress_narration() -> str`
  - `completion_narration() -> Optional[str]` — 2026-05-11 narration honesty: when `state.success` AND `n_created+n_modified+n_deleted == 0`, returns an explicit "I finished without writing or modifying any files. The project may need more direction, or it may have run out of token budget mid-exploration -- say continue if you want me to keep going." instead of the legacy "Done. ... <generic Claude tail> ... Elapsed: Xs." The generic tail summary is suppressed on this branch so the honest opener stands on its own. Pairs with the `coding.token_budget_per_session: 400000` bump. **2026-05-11 follow-up fix:** the project-root line is now `f"Saved under {path.name}."` (project folder leaf only) — was `f"Project root: {path}."` which interpolated the absolute Windows `state.cwd` (e.g. `C:\STC\ultronPrototype\data\sandbox\converts_pdf_docx`) and made XTTS-v2 enter pathological inference trying to pronounce backslash + colon + drive letter, pinning the GPU at 100 % until the server timed out. `StatusNarrator` already used the leaf only for progress narration; this brings completion_narration in line. The full path is still on disk in the per-session JSONL audit log + `coding_tasks.jsonl` start event so debugging is unaffected.
  - `pop_budget_warning() -> Optional[str]`
  - `record_pre_task_aborted(*, label, reason, intent_text="")` (V1-gap A4) — append a pre-task abort row to the audit log when the orchestrator's barge-in watcher fires.

#### `coding/coordinator.py`
- `class DecisionPath(str, Enum)` — RULE_ESCALATE / RULE_DEFAULT / RULE_ANSWER / FACT_ANSWER (V1-gap A3) / LLM_ANSWER / LLM_DEFAULT / LLM_ESCALATE / USER_ANSWER / TIMEOUT_DEFAULT
- `class ClarificationDecision`, `AdjustmentDecision`, `PendingUserClarification`, `_FactAnswer` (V1-gap A3, internal) — dataclasses
- `class ConversationCoordinator`
  - `__init__(store, llm, *, ..., facts_lookup=None)` — V1-gap A3: optional callable that reads the Qdrant `facts` collection. Wired by the orchestrator to `UltronMCPServer.lookup_facts`.
  - `decide_clarification(session_id, request, session) -> str` — answer or escalate. V1-gap A3: a high-confidence directive-category fact short-circuits the LLM call (Fast-path 2.5 between preference-options and always-answer rules).
  - `decide_adjustment(session_id, adjustment_text) -> AdjustmentDecision`
  - `handle_declare_complete(session_id) -> str` — runs Verifier, drives correction loop
  - `pending_user_clarifications() -> List[PendingUserClarification]`

#### `coding/mcp_server.py`
- `class UltronMCPServer`
  - `__init__(*, host, port, sse_path, log_path, clarification_timeout_s, session_audit_dir=None, memory=None)` — V1-gap A3: `memory` kwarg threads a live `ConversationMemory` so `lookup_facts` queries Qdrant. `None` preserves the test-isolation no-op.
  - In-process Python tools (called by Qwen via `get_config().coding.mcp.host:port`):
    - `create_session()`, `get_full_state()` (Python only), `get_status_delta()`,
      `get_clarification_context()`, `get_adjustment_context()`,
      `get_correction_context()`, `get_completion_context()`,
      `send_followup()`, `terminate_session()`, `list_active_sessions()`,
      `lookup_facts(query, *, k=None, min_confidence=None, max_age_days=None)` — V1-gap A3: when memory is wired, returns dict-shaped FactRow rows (proxies `memory.search_facts`); otherwise `[]`. Audit entry tagged `source="no_memory_wired"` on the stub branch.
  - SSE worker tools (called by Claude Code via SSE):
    - `report_progress()`, `request_clarification()`, `report_test_results()`,
      `declare_complete()`, `abandon_task()`, `record_file_change()`
  - `set_clarification_responder(fn)` / `set_declare_complete_handler(fn)` — coordinator hooks
  - `start()` / `stop()` — manage SSE server
- `write_mcp_config(project_root, sse_url)` / `remove_mcp_config(project_root)`

#### `coding/voice.py`
- `class VoiceResponse` — dataclass: text, handled, cancelled, **pre_task_confirmation, deferred_dispatch, pre_task_label** (V1-gap A4 — when populated, the orchestrator speaks the confirmation with barge-in detection before running the deferred dispatch closure).
- `class CapabilityVoiceController` (Phase 5 rename; alias = CodingVoiceController). `__init__` accepts an optional `llm_engine` (the live `LLMEngine`) so MODEL_SWITCH intents can call `llm_engine.reload_for_preset(...)` for in-process model hot-swap.
  - `pending_completion()` / `pending_clarifications()` / `pending_budget_warning()`
  - `has_pending_clarification() -> bool`
  - `handle_utterance(text) -> Optional[VoiceResponse]` — coding-only (delegated by capability dispatch)
  - `handle_capability_intent(routing_intent) -> Optional[VoiceResponse]` — top-level dispatch (Phase 5)
  - `_build_code_task_response(...)` (V1-gap A4, internal) — wraps `_submit` into a deferred dispatch closure when `coding.pre_task_confirmation_enabled`. Read-only intents (PROGRESS_QUERY / CANCEL / etc.) keep the legacy text-only response.
  - `_build_pre_task_confirmation(...)` / `_summarise_intent_for_voice(...)` (V1-gap A4, internal) — render the confirmation phrase ("I'll have Claude Code &lt;verb&gt; on the &lt;project&gt; project. Going ahead.").

### `src/ultron/openclaw_routing/` (Phase 5)

#### `openclaw_routing/intents.py`
- `class RoutingIntentKind(str, Enum)` — 17 values: CONVERSATIONAL, CODE_TASK, PROGRESS_QUERY, CANCEL, MID_SESSION_ADJUSTMENT, CLARIFICATION_RESPONSE, BROWSER_AUTOMATION, MEDIA_GENERATION, MESSAGING, FILE_OPERATION, SHELL_OPERATION, HYBRID_TASK, MODEL_SWITCH (4B plan), SYSTEM_STATUS (Phase 13), GAMING_MODE (V1-gap A1), DESKTOP_AUTOMATION (V1-gap C3), WINDOW_AUTOMATION (V1-gap C3)
- Per-category dataclasses: `BrowserIntent`, `MediaGenIntent`, `MessagingIntent`, `FileOpIntent`, `ShellOpIntent`, **`GamingModeIntent`** (V1-gap A1), **`DesktopIntent`** (V1-gap C3), **`WindowIntent`** (V1-gap C3)
- `HybridSubtask` — dataclass: order, type, subtype, description
- `RoutingIntent` — top-level dataclass: kind, raw_text, confidence, source, reason, coding_intent, automation_intent, subtasks, model_switch_intent, system_status_intent, **gaming_mode_intent, desktop_intent, window_intent** (V1-gaps A1/C3), needs_user_clarification, clarification_question
- `DispatchResult` — dataclass: success, voice_message, error, metadata
- `TaskInfo` — task tracking dataclass
- `AutomationIntent` = Union of the 5 automation intent classes

#### `openclaw_routing/classifier.py`
- `classify_routing(utterance, has_active_coding_task=False, has_pending_clarification=False) -> RoutingIntent`
  Layered: in-flight commands → hybrid → coding → automation rules → CONVERSATIONAL fallback
- `_build_browser_intent(text)`, `_build_media_intent(text)`, `_build_messaging_intent(text)`, `_build_file_intent(text)`, `_build_shell_intent(text)` — extract structured intent from raw text
- **Comprehensive test pass extensions (HEAD 2fb0988+):** `_BROWSER_INTERACT.scroll` now covers `scroll the <page|window|tab|view|content|results|list> <down|up|left|right|to>` (the original pattern only matched `scroll <down|up|to> the`); `_MEDIA_PATTERNS.render` now covers `render <a|an|the> <image|scene|picture|video|illustration|drawing|artwork>` with optional `me` (the original required `render me`); `_MESSAGING_PATTERNS` adds `notify me <on|via> <telegram|signal|slack|discord>` (parallel to the existing `tell me on …` form); `_FILE_PATTERNS` adds `show me the contents of <file.ext>` (the original required the literal word "file"). All four extensions covered by parametrised regression tests in `tests/routing/test_classifier.py` (+10 tests / 1474 → 1484).

#### `openclaw_routing/dispatcher.py`
- `class OpenClawDispatcher`
  - `__init__(config?, *, llm=None, bridge=None, gaming_mode_manager=None)` — reads openclaw.enabled + routing.stub_responses_enabled; threads optional dependencies for live-dispatch paths.
  - `async handle_browser(intent)` / `handle_media_generation(intent)` / `handle_messaging(intent)` / `handle_file_operation(intent)` / `handle_shell_operation(intent)` — return live results when the bridge is wired (Phases 4, 6, 12), stubs otherwise.
  - `async handle_gaming_mode(intent)` (V1-gap A1) — engage / disengage / status. Routes to `GamingModeManager` for plugin enable/disable; voice messages match the spec phrasing.
  - `async handle_desktop_automation(intent)` (V1-gap C3) — screenshot / list_windows / find_window via `DesktopTool`. Short-circuits with a clear message when gaming mode is engaged.
  - `async handle_window_automation(intent)` (V1-gap C3) — focus / click / type via `WindowControlTool`. Same gaming-mode short-circuit.

#### `openclaw_routing/gaming_mode.py` (V1-gap A1)
- `class GamingModeStatus(str, Enum)` — IDLE / ENGAGED / TRANSITIONING.
- `@dataclass class GamingModeReport` — engage/disengage outcome with per-plugin states + Docker action info.
- `class GamingModeManager`
  - `__init__(*, client, plugins_to_disable, toggle_docker, ...)` — owns the engage/disengage state machine.
  - `async engage()` — calls `client.disable_plugin(slug)` for each configured plugin; optionally stops Docker Desktop. Best-effort: per-plugin failures don't abort the cycle.
  - `async disengage()` — re-enables only the plugins successfully disabled during the matching engage.
  - `status() -> GamingModeStatus`.
  - Audit log: `logs/gaming_mode.jsonl`.

#### `openclaw_routing/runner.py`
- `class AutomationTaskRunner` — mirror of `CodingTaskRunner` for automation tasks
  - `async submit_task(routing_intent) -> task_id` — dispatches via OpenClawDispatcher
  - `async progress_narration(task_id) -> Optional[str]`
  - `async completion_narration(task_id) -> Optional[str]`
  - `async cancel(task_id) -> bool`
  - `list_active() -> List[TaskInfo]` / `get_task(task_id)`
  - Audit log: `logs/automation_tasks.jsonl`

#### `openclaw_routing/decomposer.py`
- `class DecompositionResult` — subtasks + fallback_used + raw_response
- `class HybridTaskDecomposer`
  - `async decompose(utterance) -> DecompositionResult` — calls Qwen with JSON-output prompt, parses, falls back to one-element coding plan on any failure

#### `openclaw_routing/disambiguator.py`
- `class DisambiguationResult` — kind (CODE_TASK / HYBRID_TASK / CONVERSATIONAL / None) + clarification_question
- `class IntentDisambiguator`
  - `async disambiguate(utterance) -> DisambiguationResult` — asks Qwen "CODING/AUTOMATION/HYBRID/UNCLEAR"

#### `openclaw_routing/decision_log.py`
- `class RoutingDecisionLog` — JSONL writer (`logs/routing_decisions.jsonl`)
  - `record(intent, *, handler, outcome, extra?)` — best-effort append
- `get_routing_log() -> RoutingDecisionLog` — singleton
- `set_routing_log(log)` — test injection

### `src/ultron/openclaw_bridge/` (OpenClaw Phase 1 + 3 foundations)

The bridge layer between Ultron and the OpenClaw Gateway peer. Voice
pipeline is unaffected when OpenClaw is unreachable (`fail_open: true`).

#### `openclaw_bridge/persona.py` (Phase 1)

- `class PersonaLoader` — reads the six workspace files
  (IDENTITY/SOUL/USER/AGENTS/HEARTBEAT/BOOTSTRAP) and composes a
  system prompt for the requested mode. Hot reload via `refresh_if_stale`
  (mtime+size check on each call).
  - `load() -> PersonaBundle` — force a fresh read.
  - `refresh_if_stale() -> PersonaBundle` — reload only if anything
    changed; cheap.
  - `get_system_prompt(mode="user_facing") -> str` — composes per mode.
- `PromptMode = Literal["user_facing", "background", "heartbeat", "bootstrap"]`
  - `user_facing` — IDENTITY + SOUL + USER. Voice path; full Ultron
    character.
  - `background` — AGENTS only, prefixed with internal-worker framing.
    For heartbeat preflight, cron, summarization, tool selection.
  - `heartbeat` — HEARTBEAT only.
  - `bootstrap` — BOOTSTRAP only.
- `default_workspace_dir() -> Path` — resolves
  `~/.openclaw/workspace/` or `ULTRON_OPENCLAW_WORKSPACE` env override.
- `class PersonaBundle` / `PersonaFile` — dataclasses with
  fingerprint (`(name, mtime_ns, size)`) for change detection.
- HTML-comment-only files (e.g., a placeholder USER.md with
  `<!-- auto-populated by maintenance -->`) are treated as empty so
  they don't bloat the prompt.

#### `openclaw_bridge/lifecycle.py` (Phase 3 foundation)

- `class OpenClawLifecycle` — health probes for the OpenClaw Gateway.
  Never raises; voice path keeps working when Gateway is unreachable.
  - `is_reachable() -> bool` — sub-second probe against
    `/__openclaw__/canvas/`.
  - `wait_for_ready(timeout_s, poll_interval_s) -> bool` — startup
    block.
  - `get_status() -> OpenClawStatus` — snapshot (version, default
    agent, configured channels).
  - `auth_token` property — reads `gateway.auth.token` from
    `~/.openclaw/openclaw.json` lazily; never logs the token.
- `class OpenClawStatus` — frozen dataclass.

#### `openclaw_bridge/client.py` (Phase 3.1)

- `class OpenClawClient` — async client over the `openclaw` CLI.
  Phase 3 deviates from the integration-spec HTTP transport because
  OpenClaw 2026.5.7 doesn't expose `/tools/invoke` or `/messages`
  HTTP endpoints — the CLI is the documented public surface, so the
  bridge invokes it via `asyncio.create_subprocess_exec`.
  - `discover_cli(override) -> str` — explicit override → env var
    (`ULTRON_OPENCLAW_CLI`) → PATH → Windows npm-global default.
  - `health(timeout_s)` — wraps `openclaw health --json`.
  - `send_message(channel, target, text)` — wraps
    `openclaw message send --channel ... --target ... --message ...
    --json`. Returns :class:`SendMessageResult`.
  - `trigger_heartbeat(text, mode, expect_final)` — wraps
    `openclaw system event`. Returns :class:`HeartbeatResult`.
  - `run_agent(message, agent_id, thinking, deliver, ...)` — wraps
    `openclaw agent --json`. Returns :class:`AgentRunResult`.
  - `invoke_tool(tool_name, params, agent_id)` — convenience over
    `run_agent` for "use this OpenClaw tool" dispatch. Raises
    :class:`OpenClawToolError` when the agent reports the tool is
    unavailable.
  - `mcp_list / mcp_show / mcp_set / mcp_unset` — config helpers
    used by :class:`UltronMcpRegistrar`.
  - `enable_plugin(plugin_id)` / `disable_plugin(plugin_id)` /
    `list_plugins(*, enabled_only=False)` (V1-gap A1) — wrap
    `openclaw plugins enable / disable / list --json`. Returns
    `PluginToggleResult` / `List[PluginInfo]`. Failures (plugin not
    installed, auth) translate into structured failures rather than
    raising.
- All methods translate stderr 401/403/Unauthorized markers into
  :class:`OpenClawAuthError`; transport failures into
  :class:`OpenClawGatewayError`. Tokens are never logged.

#### `openclaw_bridge/workspace.py` (Phase 3.3)

- `class WorkspaceWriter` — coordinated writes to the shared
  workspace (`MEMORY.md`, `USER.md`, daily memory files). Atomic
  rename via `os.replace` + advisory lockfiles via `filelock`
  (cross-platform).
  - `write_memory_entry(entry, date, prefix_timestamp)` — append
    to `memory/YYYY-MM-DD.md` with optional `HH:MM` prefix.
  - `update_memory_md(section, content, create_if_missing)` —
    splice one Markdown section in place; preserves siblings.
  - `update_user_md(content)` — full-file replace for the
    auto-populated USER.md.
- All methods are async (sync IO dispatched via
  `asyncio.to_thread`). Lockfile timeouts return a `WriteResult`
  with `error` set rather than raising.

#### `openclaw_bridge/events.py` (Phase 3.4)

- `class OpenClawEventReceiver` — gated-off scaffold for the
  `[voice]`-prefix inbound handoff. Phase 3 ships only the prefix
  matching contract (`should_handle`, `extract_payload`); the
  transport (webhook subscription / polling) is wired in a later
  phase once a real channel exists.
  - `start() / stop()` — no-op when `enabled=False` (default).
  - `dispatch(IncomingMessage) -> bool` — invokes the registered
    handler when the prefix matches; swallows handler exceptions
    so the orchestrator's main loop never sees them.
- `class IncomingMessage` — frozen dataclass; subset of an inbound
  message we route on (channel, sender, body, prefix_match).

#### `openclaw_bridge/mcp_registration.py` (Phase 3.2)

- `class UltronMcpRegistrar` — registers Ultron's MCP server with
  OpenClaw via `openclaw mcp set`. Idempotent: re-running with the
  same payload is a no-op (`already_registered=True`). Fail-open:
  failures return a `RegistrationResult` with `error` set rather
  than raising.
  - `register()` — main entry. Reads `mcp_show` first to detect
    matching existing entry; `mcp_set` only when needed.
  - `verify_registered()` — true iff the configured payload is
    currently registered.
  - `unregister()` — best-effort cleanup; never raises.
  - `schedule_retry(interval_s, on_success, max_attempts)` —
    coroutine for background retry. Caller wraps with
    `asyncio.create_task`.
- Integration deviation: the integration spec assumed Ultron's MCP
  is stdio. Reality is SSE (in-process). The registrar is
  config-driven — `openclaw.bridge.mcp_server_command` defaults to
  `None`, deferring registration. When set (e.g. when a stdio
  proxy is added in a future phase), the registrar wires it up.

#### `openclaw_bridge/holder.py` (Phase 3.5 + Phase 4)

- `class OpenClawBridge` — single dataclass-style holder owned by
  the orchestrator. Encapsulates lifecycle, client, workspace,
  events, registrar, **notifications** (Phase 4).
  - `from_config(openclaw_cfg, notifications_cfg=None) -> Optional[OpenClawBridge]` —
    returns `None` when `openclaw.enabled=False`. Construction is
    fail-open: missing CLI yields `client=None` rather than
    raising. ``notifications_cfg`` is optional (defaults to a
    disabled instance) so callers from before Phase 4 keep
    working.
  - `start()` — sync. Probes the Gateway; on success runs
    `registrar.register()`; on failure (or when MCP command is
    configured but Gateway is unreachable) launches a daemon
    retry thread.
  - `shutdown()` — stops the retry thread and the event receiver.
    Deliberately leaves the MCP entry registered so OpenClaw can
    spawn Ultron's MCP across restarts.
  - `fire_and_forget(coro_factory)` (Phase 4) — schedules a
    coroutine on a daemon thread for off-hot-path dispatch from
    the sync orchestrator loop (used by coding-completion
    notification fires).

#### `openclaw_bridge/notifications.py` (Phase 4)

- `class NotificationDispatcher` — single seam for proactive
  outbound notifications to remote channels. Each event class has
  its own method:
  - `notify_coding_task_completion(summary)`
  - `notify_coding_task_clarification(question)`
  - `notify_heartbeat_alert(text)`
  - `notify_standing_order_output(summary)`
  - `notify_search_results_async(summary)`
- All methods fail-open at every step: missing client, master
  flag off, per-event flag off, no recipient, transport failure
  — each returns a :class:`NotificationResult` with
  ``sent=False`` and a ``skipped_reason``. Voice pipeline never
  blocks.
- Recipient resolution: env var (``user_id_env``) →
  ``fallback_user_id`` → empty (skip).

#### `openclaw_bridge/heartbeat_alerts.py` (Phase 5)

- `class HeartbeatAlertLog` — JSONL-backed alert log with
  thread-safe append + atomic full-file rewrite for updates
  (acknowledgments).
  - `record(text, source, severity, metadata)` — append a new
    alert. Returns :class:`HeartbeatAlert`.
  - `get_alerts(since, only_unacknowledged, limit)` — read,
    filter, return most-recent-first.
  - `acknowledge(alert_id)` — mark seen. Atomic rewrite.
  - `prune()` — drop entries older than ``retention_days``.
- `class HeartbeatAlert` — dataclass with `alert_id` (UUID4 hex),
  `text`, `source`, `severity` ("info"/"warn"/"error"),
  `timestamp`, `acknowledged_at`, `metadata`.
- Tolerates malformed JSONL lines (logs WARN, skips), missing
  files (returns empty list), permission errors (logs WARN).
- `OpenClawBridge.record_heartbeat_alert(...)` is the orchestrator-side
  entry point: records to the log + (when enabled) fires Telegram
  notification via :class:`NotificationDispatcher.notify_heartbeat_alert`.

#### `openclaw_bridge/browser.py` (Phase 6)

- `class BrowserTool` — thin facade over
  :meth:`OpenClawClient.invoke_tool` for browser primitives.
  Each method assembles a structured prompt asking the OpenClaw
  ``ultron-main`` agent to use the browser tool with specific
  parameters; the wrapper unpacks the agent response into a typed
  dataclass.
  - `navigate(url)` → :class:`NavigateResult` (best-effort title
    extraction).
  - `snapshot(mode='ai'|'aria')` → :class:`Snapshot` with refs
    extracted in `ai` mode.
  - `click(ref)` / `type_text(ref, text)` → :class:`ActionResult`.
  - `screenshot()` → :class:`ScreenshotResult` (decodes base64
    when present).
  - `get_page_text()` → :class:`PageTextResult`.
- All methods translate `OpenClawToolError` (tool unavailable
  responses) into structured failures rather than raising.

#### `openclaw_bridge/mcp_tools.py` (Phase 13)

- Stdio MCP server exposing Ultron's read-mostly tools to OpenClaw
  agents. Each tool is a plain Python function callable from
  Python tests; FastMCP registration in :func:`build_server`
  wires them up for stdio dispatch.
- Tool implementations:
  - `get_heartbeat_alerts_impl(since_seconds_ago, only_unacknowledged, limit)`
  - `acknowledge_alert_impl(alert_id)`
  - `run_maintenance_impl(scope=None)` — subprocesses
    `scripts/run_maintenance_for_cron.py --json`
  - `list_active_coding_sessions_impl(max_age_hours=24)` — reads
    `logs/sessions/*.jsonl` audit files
  - `get_recent_voice_alerts_impl(limit=5)` — voice-friendly
    convenience wrapper
- Lazy-imports heavy dependencies; no torch / LLM at startup so
  the spawned process is light.
- :func:`run_stdio` is the entry point invoked by
  ``scripts/run_ultron_mcp_for_openclaw.py``.

#### `openclaw_bridge/desktop.py` (V1-gap C3)

- `class DesktopTool` — wrapper over `OpenClawClient.invoke_tool` for the `desktop-control` plugin. Methods: `screenshot(target?)`, `list_windows()`, `find_window(query)`. Each returns a typed dataclass (`DesktopScreenshotResult`, `ListWindowsResult`, `FindWindowResult`). Tool slugs configurable via `config.desktop.tool_slug_*`.
- `class WindowControlTool` — same pattern over `windows-control` plugin. Methods: `focus(query)`, `click(ref)`, `type_text(ref, value)`. Returns `WindowActionResult`.
- `OpenClawToolError` raised by the underlying client is translated into structured failures with the error preserved in `result.error`.

#### `openclaw_bridge/system_status.py` (Phase 13)

- `class SystemStatusReporter` — voice-side reporter for
  `SYSTEM_STATUS` routing intents. Reads heartbeat alert log +
  active session listing (via the same impl functions
  `mcp_tools.py` exposes to OpenClaw) and renders a brief in-
  character voice narration.
  - `report(SystemStatusIntent) -> SystemStatusReport` — main
    entry. Honors `focus="alerts"|"projects"|"all"` from the
    intent.
- Voice rendering kept short by design (3–4 sentences for
  combined queries, ≤2 for focused). Sanitiser caps individual
  alert text at 160 chars + ellipsis.
- Failure-safe: disk read failures degrade to "no information"
  voice messages; never raises.

### `src/ultron/pipeline/orchestrator.py`

- `class State(Enum)` — IDLE / CAPTURING / PROCESSING / FOLLOW_UP_LISTENING
- `class Orchestrator` — main event loop
  - `MAX_UTTERANCE_SECONDS` (class constant, **30.0** as of 2026-05-11 follow-up fix; was 15.0) — fallback default for the per-capture hard ceiling. The instance attribute `self._max_utterance_seconds` (read from `vad.max_utterance_seconds`) wins at runtime; the class constant is only used when config load fails in `__init__`.
  - `__init__()` — composes audio, wake, vad, addressing, stt, llm, memory, web_search, tts, coding_voice. Reads `vad.max_utterance_seconds` into `self._max_utterance_seconds` (defaults to 30.0; defensive fallback to the class constant on config-load failure). Also reads `vad.long_utterance_threshold_seconds` + `vad.long_utterance_silence_duration_ms` for the adaptive end-of-turn policy.
  - `run()` — main loop (blocks; KeyboardInterrupt clean shutdown)
  - `_capture_utterance()` — VAD-bounded audio capture. **2026-05-11 follow-up fix:** the hard `elapsed_samples < max_samples` ceiling now reads from `self._max_utterance_seconds` (config-driven). Previously a class-level `MAX_UTTERANCE_SECONDS=15.0` cut a real user off mid-sentence on a complex coding ask — the user wasn't pausing; the wall-clock ceiling fired before Silero VAD reported `SPEECH_END`. Bumping to 30 s comfortably covers detailed one-breath asks while still bounding pathological captures (stuck mic, background noise that never resolves to SPEECH_END, etc.).
  - `_follow_up_listen(deadline)` — WARM-mode VAD loop. Same `self._max_utterance_seconds` ceiling on cumulative speech (not wall-clock, which is bounded by `deadline`).
  - `_respond(user_text)` — LLM stream → TTS pipeline (with optional web search)
  - `_speak(text)` — single-shot synthesize + play
  - `_speak_with_barge_in_check(text, *, post_check_window_s=0.5) -> bool` (V1-gap A4) — speak text and report whether wake fired during/after; used by the pre-task confirmation flow.
  - `_handle_capability_response(response, routing_intent)` (V1-gap A4) — wraps the capability voice dispatch. Default path: speak `response.text`. A4 path: speak `response.pre_task_confirmation` first, abort dispatch on barge-in (audit via `runner.record_pre_task_aborted`).
  - `_announce_coding_completion_if_pending()`, `_announce_pending_clarifications()`, `_announce_pending_budget_warning()` — voice-loop poll hooks
  - `_load_memory_if_enabled()` — Qdrant init with graceful fallback
  - `_load_openclaw_bridge_if_enabled()` (Phase 3.5) — constructs
    :class:`OpenClawBridge`. Returns `None` when
    `openclaw.enabled=False` (current default). Fail-open: any
    construction or start failure leaves the bridge disabled
    without affecting the voice path.
  - `self.openclaw_bridge` attribute — accessed by the dispatcher
    when an OpenClaw-bound intent fires. Cleaned up in `shutdown()`
    via `self.openclaw_bridge.shutdown()`.

**In:** mic input (sounddevice), config.yaml, models on disk.
**Out:** speaker output (sounddevice), all audit logs.

### `src/ultron/resilience/` (Phase 4)

#### `resilience/circuit_breaker.py`
- `class CircuitState(str, Enum)` — CLOSED / OPEN / HALF_OPEN
- `class CircuitOpenError(Exception)` — short-circuit signal
- `class CircuitBreaker`
  - `__init__(name, failure_threshold=3, window_seconds=300, cooldown_seconds=300, expected_exceptions=(Exception,))`
  - `call(func, *args, **kwargs) -> result` — raises CircuitOpenError when OPEN
  - `state`, `failure_count` properties
  - `reset()` — test/operator only

#### `resilience/error_log.py`
- `class ErrorLog` — append-only JSONL writer to `logs/errors.jsonl`
  - `record(error, *, dependency, session_id?, extra?, include_traceback=True)` — best-effort
- `get_error_log() -> ErrorLog` — singleton
- `set_error_log(log)` — test injection

#### `resilience/phrases.py`
- `phrase_for(failure_mode: str) -> Optional[str]` — shuffled cycle from `config.error_phrases.<mode>`
- `reset_phrase_cache()` — test-only

### `src/ultron/utils/`

#### `utils/logging.py`
- `configure_logging(level=None, log_file=None) -> None` — idempotent
- `get_logger(name) -> logging.Logger` — namespaced under `ultron.`

#### `utils/fairseq_compat.py`
- `patch_fairseq_dataclasses()` — workaround for fairseq's invalid omegaconf metadata
- `patch_torch_load_for_fairseq()` — torch.load weights_only compat shim

---

## Configuration

### `config.yaml` (project root) — single source of truth

Sections:
- `version: "1.0"`
- `audio` (sample_rate, channels, blocksize, dtype, devices, barge-in, **ring_buffer_seconds: 0.5** [2026-05-10: bumped back from 0.15 to act as a STORAGE capacity now that the orchestrator slices mode-specific pre-roll out of it], **cold_pre_roll_seconds: 0.15** [NEW 2026-05-10: post-wake slice; short to avoid the "Tron" prefix the longer pre-roll caused], **warm_pre_roll_seconds: 0.5** [NEW 2026-05-10: post-TTS follow-up slice; long enough to span Silero VAD's ~150 ms speech-start latency without clipping the user's leading word], input_gain_db [2026-05-09])
- `vad` (threshold, min_speech_duration_ms, **min_silence_duration_ms: 1200** [2026-05-09 latency fix; was 500 — natural mid-sentence pauses prematurely closed the capture; trade-off is ~0.7 s slower end-of-turn detection], window_samples, **long_utterance_threshold_seconds: 8.0**, **long_utterance_silence_duration_ms: 2400** [NEW 2026-05-11 adaptive end-of-turn: once speech has been active past the threshold, orchestrator bumps VAD silence requirement to the long value so a thinking pause mid-prompt doesn't cut the capture. Short utterances stay snappy. Set threshold to 0 to disable.], **max_utterance_seconds: 30.0** [NEW 2026-05-11 follow-up fix: hard ceiling on a single VAD-bounded capture. Was a class-level constant `Orchestrator.MAX_UTTERANCE_SECONDS=15.0` that cut a real user off mid-sentence on a complex coding ask (Whisper transcribed 15.158 s ending mid-phrase at "a button with a box show" — user wasn't pausing; wall-clock ceiling fired before VAD reported SPEECH_END). Now configurable, default 30 s; schema range [5, 120]. Falls back to the class constant only if config load fails.])
- `wake_word` (name, model_path, fallback_model, threshold, cooldown)
- `stt` (model, device, compute_type, beam_size, temperature, etc.)
- `llm` (provider="llama_cpp", **preset** ["qwen3.5-9b"|"qwen3.5-4b"|"custom"; auto-fills model_path/n_ctx/draft_model_path when those keys are omitted — Stage A of the 4B plan], runtime ["in_process"|"http_server"], model_path, draft_model_path, n_ctx, gpu_layers, temperature, top_p, max_tokens, repeat_penalty, history_turns, flash_attn, kv_cache_type, system_prompt, server.{base_url,...}, persona.{source,...})
- `embeddings` (dense_model, sparse_model, dense_dim)
- `qdrant` (data_dir="data/qdrant", collections.{conversations,facts,web_results})
- `memory` (enabled, jsonl_legacy_path, recent_turns, rag_top_k, rag_exclude_recent, facts_top_k, write_queue_maxsize, **retrieval.{multi_pass_enabled=false, max_categories_per_query=4, candidates_per_category_multiplier=4}** (V1-gap A2), **ranking.{rrf_weight=1.0, recency_weight=0.2, recency_half_life_days=7.0, surprise_weight=0.15, redundancy_weight=0.3}** (V1-gap A2), **rag_min_relevance=0.6** (NEW 2026-05-09: cosine-similarity floor for RAG candidates; tuned empirically with bge-small INT8 -- off-topic content peaks ~0.55-0.57, truly relevant 0.7-0.95), **history_turns_for_llm=4** (NEW 2026-05-09: cap on recent-turn history fed to LLM per call; prevents topic-bleed when user pivots topics))
- `web_search` (enabled, brave_api_key_env, brave/jina/cache subsections, **citation.inline_marker_format="bracket"** [V1-gap B3]). 2026-05-09 latency fix tunables: **`jina.timeout_seconds: 6.0`** (was 15.0), **`jina.max_fetch: 2`** (was 3), **`jina.collective_deadline_seconds: 6.0`** (NEW — executor-side cap on parallel fetch wait; 0 disables).
- `addressing` (follow_up_enabled, **warm_mode_duration_seconds: 30.0** ← user override, NOT 10s; rule_confidence_threshold, **zero_shot_addressed_min_confidence: 0.80** [NEW 2026-05-11: demotes low-confidence zero-shot YES verdicts to NOT_ADDRESSED via default_silent; catches the borderline third-person utterances flan-t5-small saturates on at 0.75. Set to 0.0 for legacy permissive behaviour.], zero_shot_model, log_path)
- `coding` (enabled, bridge="direct", mcp.{host,port,...}, template_dir, prompt_token_budget, default/escalation models + thresholds, verification.{smoke,test,lint}_timeout, session_audit_dir, **token_budget_per_session=400000** [2026-05-11 bump from 100000 — new-project sessions burn 100k+ on tool exploration alone before writing files; 400k gives headroom while the 80% warning still fires. Paired with the 2026-05-11 narration honesty fix so users get an explicit "no files written" signal when budget is exhausted mid-exploration], claude_cli, sandbox_root, project_registry_path, audit_log_path, task_timeout, skip_permissions, **voice_task_require_testing=false** [NEW 2026-05-11 token-efficiency fix: was implicitly true via voice.py hardcode, which prepended a "MUST write tests, run, fix, re-run" preamble to every voice-dispatched Claude prompt and 3-5x'd the token spend. Default false lets small voice asks land lean. Users who want tests can say "with unit tests" in their voice request or flip this flag], **facts.{top_k=5, min_confidence=0.75, min_score=0.85, max_age_days=null}** [V1-gap A3], **pre_task_confirmation_enabled=false, pre_task_confirmation_max_words=30, pre_task_barge_in_window_seconds=0.5** [V1-gap A4])
- `projections` (tokenizer, budgets.{clarification,status_delta,adjustment,correction,completion}_context, truncation_warning_threshold, log_truncations)
- `tts` (**engine="piper_rvc" | "xtts_v3"** [NEW 2026-05-10 voice swap; default still legacy for back-compat], piper paths, sample_rate, sentence_flush_chars, length_scale, pause_ms, edge_fade_ms, **pipeline_parallel_enabled=true** [2026-05-09 Piper/RVC split], **speculative_stream_open_enabled=true** [2026-05-09], **speculative_stream_sample_rate=48000** [2026-05-10: was 40000 — actual Ultron RVC output is 48000 Hz, mismatch was forcing the close-and-reopen path on every turn], **output_low_latency_mode=true** [2026-05-09], rvc subsection, **xtts_v3 subsection** [server_python, server_script, reference_audio, host, port, filter_preset="v3_heavy", filter_tail_silence_ms=200, **speed=1.15** (NEW 2026-05-11 cadence tune; XTTS native default is 1.0 — production set to 1.15 for ~15% faster speech without slurring; adjusts synthesis duration tokens so the v3 pedalboard filter is unaffected; safe range ~0.7-1.4, schema-bounded to [0.5, 2.0])])
- `logging` (file, level, format, datefmt)
- `error_phrases` (13 pools — qdrant_unavailable, brave_unavailable, jina_unavailable, anthropic_unavailable, rvc_unavailable, openclaw_unavailable, piper_unavailable, whisper_repeated_failures, addressing_classifier_failure, wake_word_model_failure, mcp_server_lost, claude_code_subprocess_failed, config_invalid)
- `routing` (llm_disambiguation_enabled, hybrid_task_decomposition_enabled, disambiguation_question_template, routing_log_path, classifier subsection, stub_responses_enabled)
- `openclaw` (enabled=false [stub], gateway_url, auth_token_env, health_check_*_seconds, fail_open, required_agent_id)
- `gaming_mode` (V1-gap A1) — enabled=false, plugins_to_disable=[desktop-control, windows-control], toggle_docker=false, docker_executable_path, docker_process_name, log_path
- `desktop` (V1-gap C3) — enabled=false, default_*_timeout_seconds, plugin_slug, tool_slug_screenshot / tool_slug_list_windows / tool_slug_find_window
- `window_control` (V1-gap C3) — enabled=false, default_action_timeout_seconds, plugin_slug, tool_slug_focus / tool_slug_click / tool_slug_type

### `config/settings.py` (Phase 3 SHIM)

Compatibility shim that re-exports legacy `settings.X` constants from `get_config()`. Thin layer; HF cache pre-init runs at import time. Used by subsystems still on the legacy reference path (audio, wake_word, stt, tts, rvc, coding cluster, scripts) — see [docs/phase3_5_followup.md](phase3_5_followup.md) for the migration punch list.

### `.env.example` (and the actual `.env` in main checkout)

Env vars:
- `ULTRON_BRAVE_API_KEY` — Brave Search API key (required for web search)
- `ULTRON_LLM_MODEL_PATH` — opt-in override of GGUF path
- `ULTRON_AUDIO_DEVICE` / `ULTRON_AUDIO_OUTPUT_DEVICE` — operator-specific device strings
- `ULTRON_LOG_LEVEL` — console log level
- `ULTRON_CODING_MCP_ALLOW_ANY_ROOT=1` — test-only sandbox escape
- `ULTRON_CONFIG_PATH` — alternative config.yaml path

---

## Operational scripts

All scripts assume venv active in main checkout (`C:\STC\ultronPrototype`). Worktrees inherit the venv via shared `.venv\Scripts\python.exe`.

### `scripts/benchmark.py`

**Purpose:** measure end-to-end first-token latency for a single voice query.
**Run:** `python scripts/benchmark.py`
**In:** loads full voice stack + config.
**Out:** stdout — TTFT for one synthetic query.

### `scripts/check_vram.py`

**Purpose:** quick VRAM snapshot.
**Run:** `python scripts/check_vram.py [--watch [N]] [--gpu N]`
**In:** nvidia-smi.
**Out:** stdout — `<used> MB used | of <total> MB | target 9216 MB | cap 11500 MB | [OK/above target/WARN/CRITICAL]`
**Functions:** `vram_used_mb(gpu_id) -> Optional[int]`, `vram_total_mb(gpu_id)`, `gpu_name(gpu_id)`, `_format_line(used, total)`, `main(argv)`.

### `scripts/download_models.py`

**Purpose:** first-run model fetcher (Qwen GGUF, Piper, faster-whisper, openWakeWord).
**Run:** `python scripts/download_models.py`
**In:** Hugging Face Hub.
**Out:** files under `models/`.

### `scripts/dump_session.py`

**Purpose:** render coding-session audit log into a readable transcript.
**Run:** `python scripts/dump_session.py [--list | --latest | <session_id> | <path/to/file.jsonl>] [--sessions-dir DIR]`
**In:** `logs/sessions/<id>.jsonl`.
**Out:** stdout — formatted event list (one line per event with timestamp + kind + summary).
**Functions:** `_resolve_session_path(token, dir)`, `_read_records(path)`, `_format_record(rec)`, `main(argv)`.

### `scripts/last_session.py` (V1-gap C2)

**Purpose:** backwards-compat alias for `dump_session.py`. The V1 spec named this script `last_session.py`; both names now coexist and resolve to the same `main(argv)` entry point.
**Run:** `python scripts/last_session.py ...` (forwards every arg to `dump_session.main`).

### `scripts/list_audio_devices.py`

**Purpose:** mic / output device introspection.
**Run:** `python scripts/list_audio_devices.py`
**Out:** stdout — devices indexed by ID + name.

### `scripts/maintenance.py`

**Purpose:** periodic Qdrant maintenance (summarize old conversations into `facts`, label clusters, prune stale `web_results`, extract entities).
**Run:** `python scripts/maintenance.py`
**In:** Qdrant `conversations` collection, LLM, `data/maintenance.sqlite` (state).
**Out:** writes to `facts` collection, `data/summaries.jsonl`, updates sqlite.

### `scripts/measure_baseline.py`

**Purpose:** voice-path VRAM + TTFT baseline (10 representative queries; full stack loaded).
**Run:** `python scripts/measure_baseline.py`
**In:** loads full voice stack; runs 10 hard-coded representative queries.
**Out:** writes `baselines.json` (top-level metadata/vram_mb/latency_ms keys).

### `scripts/measure_baseline_extended.py` (Foundation Phase 0)

**Purpose:** extended baselines — search VRAM, coding-session VRAM, TTA microbench, scenario timing, composite TTFA.
**Run:** `python scripts/measure_baseline_extended.py [--lite | --full | --all]`
**Modes:**
- `--lite`: CPU-only — TTA microbench, scenario timing, composite TTFA. ~30 s.
- `--full`: also loads voice stack + measures search/coding VRAM. ~3 min.
- `--all`: both (default).
**In:** config + models + tests/coding/test_orchestration.py runtime.
**Out:** writes `baselines.json` `phase_foundation_start.measurements_extended` block.

### `scripts/migrate_memory_to_qdrant.py`

**Purpose:** one-shot ingest of `data/memory.jsonl` into Qdrant `conversations` collection.
**Run:** `python scripts/migrate_memory_to_qdrant.py`
**In:** `data/memory.jsonl`.
**Out:** `data/qdrant/` collections populated.

### `scripts/review_addressing.py`

**Purpose:** read `logs/addressing.jsonl`, print recent classifier verdicts.
**Run:** `python scripts/review_addressing.py [--tail N] [--misses] [--log PATH]`
**Modes:** `--misses` shows only NOT_ADDRESSED for false-negative tuning.
**Out:** stdout — `HH:MM:SS  DECISION  source  conf  latency  "utt"  -- reason`

### `scripts/run_integration_tests.py` (Foundation Part 7)

**Purpose:** wraps `pytest tests/integration tests/routing tests/error_recovery` with `--gpu` for `PYTEST_RUN_GPU_TESTS=1`.
**Run:** `python scripts/run_integration_tests.py [--gpu] [-q]`
**In:** test files.
**Out:** pytest output to stdout + final summary line with wall-clock + exit code.

### `scripts/run_orchestration_tests.py`

**Purpose:** run the 10 orchestration scenarios in `tests/coding/test_orchestration.py` with reporting.
**Run:** `python scripts/run_orchestration_tests.py`
**Out:** stdout — per-scenario pass/fail + total timing.

### `scripts/validate_config.py` (Foundation Part 7)

**Purpose:** validate `config.yaml` against pydantic schema without starting Ultron.
**Run:** `python scripts/validate_config.py [path] [--print]`
**Out:** stdout — "Configuration is valid." or detailed `ConfigurationError` with path + message + context. Exit 0 = valid, 1 = invalid.

### `scripts/start_llamacpp_server.py` (OpenClaw integration Phase 0 + 4B plan Stage C)

**Purpose:** launch llama-cpp-server on `127.0.0.1:8765` with the same params as the in-process voice loader (n_ctx=8192, flash_attn, Q8_0 KV cache). Imports `ultron` first so bundled torch CUDA DLLs are found before `llama_cpp` initialises (Windows-specific quirk).
**Run:** `python scripts/start_llamacpp_server.py [--n-ctx N] [--port P] [--api-key K] [--chat-format F] [--model-draft <path>] [--draft-num-pred-tokens N] [--from-config]`. The Stage C flags add speculative decoding (`--model-draft` + `--draft-num-pred-tokens`, mapped to llama-cpp-python's `draft_model` / `draft_model_num_pred_tokens`) and a `--from-config` overlay that reads model/draft/n_ctx from `config.yaml:llm` (preset-aware). CLI flags override the overlay. Pure-Python helpers `_build_arg_parser`, `_resolve_kwargs`, `_config_overlay` factor out the testable pieces.
**Out:** uvicorn HTTP server on `--port` (default 8765); stays in foreground.

### `scripts/supervised_llamacpp_server.py` (OpenClaw integration Phase 0)

**Purpose:** Python supervisor wrapper for `start_llamacpp_server.py`. Spawns the launcher as a subprocess, restarts on death with exponential backoff (2 s → 60 s cap, healthy_after_s=30 resets). Lighter alternative to NSSM.
**Run:** `python scripts/supervised_llamacpp_server.py [--cwd ...] [--max-restarts N] [--child-arg ...]`
**Out:** tee'd stdout/stderr from the child + supervisor restart events to stderr.

### `scripts/_bench_llm_http.py` (OpenClaw integration Phase 0)

**Purpose:** TTFT benchmark for the HTTP-runtime LLMEngine. Same 10 representative queries as `measure_baseline.py`, hits llama-cpp-server over HTTP.
**Run:** `python scripts/_bench_llm_http.py` (server must be running on the configured base URL).
**Out:** writes `baselines.json` `llm_http_runtime` block (median, p95, per-query). Used to compare HTTP-mode vs in-process mode latency.

### `scripts/_log_proxy.py` (OpenClaw integration Phase 0; debug only)

**Purpose:** tee proxy on `127.0.0.1:8766` that forwards to `127.0.0.1:8765` and logs every request body + SSE stream to stdout. Used to debug what OpenClaw actually sends to llama-cpp-server.
**Run:** `python scripts/_log_proxy.py` (point OpenClaw's `models.providers.litellm.baseUrl` at the proxy port instead of the server port).

### `scripts/smoke_test_llamacpp.ps1` (OpenClaw integration Phase 0)

**Purpose:** PowerShell smoke test for llama-cpp-server. Hits `/v1/models` and `/v1/chat/completions` with a tiny prompt; prints timing + completion text. Used to verify the server is healthy before involving OpenClaw.
**Run:** `pwsh scripts/smoke_test_llamacpp.ps1`

### `scripts/swap_llm_preset.py` (4B plan Stage H)

**Purpose:** atomic preset swap — edits `config.yaml:llm.preset` in place after validating the requested preset's GGUFs are present. Supports `--list`, `--status`, `--dry-run`. The voice path can also be swapped at runtime via the `MODEL_SWITCH` intent ("Ultron, switch to the 9B"); this script is for off-orchestrator workflows.
**Run:** `python scripts/swap_llm_preset.py [--status | --list | <preset> [--dry-run]]`
**In:** `config.yaml`, `models/*.gguf` (validation).
**Out:** updated `config.yaml`; stdout reports the change.

### `scripts/verify_voice_character_4b.py` (4B plan Stage E)

**Purpose:** interactive A/B helper that synthesises 5 representative voice queries through both the 4B and 9B presets so the operator can confirm Ultron's character is preserved. Approved 2026-05-08.
**Run:** `python scripts/verify_voice_character_4b.py`
**In:** loads voice stack twice (once per preset).
**Out:** plays audio + writes A/B comparison CSV.

### `scripts/verify_items_4_to_8.py` (4B plan Items 4–8 verification)

**Purpose:** exercises each of Items 4 (compression), 5 (IRMA), 6 (self-consistency), 7 (canonical-path monitor), 8 (block-and-revise) in the trigger scenario the corresponding flag fires on. Prints concrete deltas (token reduction, accuracy lift, abort timing, etc.).
**Run:** `python scripts/verify_items_4_to_8.py`
**Out:** stdout — per-item status with measurable metrics.

### `scripts/comprehensive_test_harness.py` (Comprehensive end-to-end test pass)

**Purpose:** single-process exhaustive harness for the comprehensive end-to-end test pass. Runs five phases in sequence — routing classifier accuracy on a 63-utterance labeled adversarial corpus spanning every `RoutingIntentKind`; web-gate rule classifier accuracy on 14 labeled queries; circuit-breaker state machine through CLOSED → OPEN → HALF_OPEN → CLOSED → reopen transitions; memory stress (4 threads × 50 turns ingested into a tmp Qdrant + 20 retrieval probes); V1-gap classifier-gating regression (utterances that used to short-circuit to OpenClaw stub when offline). No GPU / model loads — runs anywhere the venv resolves.
**Run:** `python scripts/comprehensive_test_harness.py`
**In:** Imports the worktree's `src/ultron` and the main checkout's `config/` shim.
**Out:** Stdout summary + machine-readable result at `logs/comprehensive_harness_<ts>.json`.

### `scripts/real_api_smoke.py` (Real-API sparing smoke)

**Purpose:** proof-of-life test for the three external services Ultron talks to in production — Brave, Jina, Claude Code. Strict budget: ≤2 Brave calls (one bare query + one chain that adds Jina), ≤1 Jina fetch (via the chain), ≤1 minimal Claude Code haiku invocation. Reads `ULTRON_BRAVE_API_KEY` from `.env`; the Claude CLI defaults to `%APPDATA%\\npm\\claude.cmd` and can be overridden via `ULTRON_CLAUDE_CLI`. Used in the comprehensive end-to-end test pass to confirm circuits + bridge transports work end-to-end without sprawling spend.
**Run:** `python scripts/real_api_smoke.py`
**Out:** Stdout summary + machine-readable result at `logs/real_api_smoke_<ts>.json` (does NOT log the Brave key or any secret).

### `scripts/run_maintenance_for_cron.py` (OpenClaw Phase 7)

**Purpose:** cron-friendly wrapper around `scripts/maintenance.py`. Outputs JSON or single-line Telegram-pretty summary; captures stdout from underlying tasks; structured exit codes (0 ok / 1 task error / 2 init failure). Suitable for Windows Task Scheduler invocations.
**Run:** `python scripts/run_maintenance_for_cron.py [--task <name> ...] [--json | --pretty]`
**In:** subprocesses `scripts/maintenance.py` machinery.
**Out:** stdout — structured summary; exit code per outcome.

### `scripts/benchmark_preflight.py` (V1-gap B5)

**Purpose:** benchmark the web-search gate's pre-flight reasoning pass against the main LLM AND optional CPU-only candidate models. Settles V1-spec Part 1.5's question about whether a dedicated CPU model would be faster than the main Qwen on pre-flight. Decision documented at [docs/preflight_decision.md](preflight_decision.md): keep main LLM (TTFT 79 ms voice baseline already beats the spec's 200 ms threshold).
**Run:** `python scripts/benchmark_preflight.py [--candidate-model PATH] [--skip-main] [--queries N]`
**In:** loads the live `LLMEngine` (or a CPU-only `llama_cpp.Llama` for the candidate); 30 representative queries with manual ground truth.
**Out:** Markdown summary table + appends `preflight_benchmark.backends` block to `baselines.json`.

### `scripts/run_ultron_mcp_for_openclaw.py` (OpenClaw Phase 13)

**Purpose:** stdio MCP entry script OpenClaw spawns when an agent calls one of Ultron's tools. Boots a FastMCP server on stdio that exposes `get_heartbeat_alerts`, `acknowledge_alert`, `run_maintenance`, `list_active_coding_sessions`, `get_recent_voice_alerts`. Imports stay light — no torch / LLM loaded.
**Run:** `python scripts/run_ultron_mcp_for_openclaw.py [--stdio | --list-tools]`
**In:** disk artifacts (heartbeat alert log, session audit dir) + OpenClaw stdio channel.
**Out:** MCP responses over stdio.
**Auto-resolved:** `OpenClawBridgeConfig.mcp_server_command="auto"` resolves to this script via the holder's `_resolve_mcp_command` helper.

### `scripts/_record_phase0_baseline.py` / `scripts/_merge_phase0_baselines.py` (OpenClaw Phase 0)

**Purpose:** record and merge Phase 0 baseline measurements into `baselines.json`. Used during the OpenClaw Phase 0 verification work.
**Run:** `python scripts/_record_phase0_baseline.py`; `python scripts/_merge_phase0_baselines.py`

### `scripts/_vram_peak_monitor.py` (auxiliary)

**Purpose:** background VRAM peak monitor used by `measure_baseline_extended.py` for accurate peak capture during search/coding-session runs.

### `scripts/audio_diagnostic.py` (2026-05-09 audio-quality pass)

**Purpose:** standalone diagnostic harness for far-field mic + wake + Whisper tuning. Loads ONLY the audio path (sounddevice + openWakeWord + Silero VAD + faster-whisper) — NO LLM, NO TTS, NO orchestrator. ~1.5 GB VRAM so it can run while the full Ultron stack is stopped (per the voice-stack-concurrency rule).

**Modes** (`--mode`):
- `noise-floor` — captures N seconds of silence; reports peak / mean RMS dBFS for noise-floor calibration.
- `wake` — captures a window, records max wake-word score, prints whether `FIRED` at the configured threshold; saves audio to WAV via `--save-wav` for replay.
- `phrase` — captures until VAD reports speech end (or hard timeout); reports VAD timing, peak RMS, Whisper transcription + word-coverage vs `--expected-text`.
- `monitor` — live real-time meter: rolling RMS, VAD probability, wake score per chunk; Ctrl+C to exit.

**CLI overrides** (process-local, never write back to config so iteration is fast): `--device` (substring match — "Focusrite", "Voicemeeter"), `--gain-db`, `--wake-threshold`, `--vad-threshold`, `--seconds`, `--whisper-beam`, `--save-wav`, `--label`.

**Audit log:** every test row appends to `logs/audio_diag_<ts>.jsonl`; useful for cross-distance comparison.

**Run:** `python scripts/audio_diagnostic.py --mode wake --device Focusrite --label round1_5ft --seconds 10 --save-wav logs/round1_5ft.wav`

### `scripts/comprehensive_memory_quality.py` (2026-05-09 memory-quality pass)

**Purpose:** end-to-end memory + retrieval quality test pass. Loads embedder + isolated tmpdir Qdrant + (optionally) Qwen 4B. Seeds the isolated store with 58 mixed-topic turns (predator chatter, PC troubleshooting, food, BMWs, weather, code) and runs 28 scenarios verifying:

- Contamination filtering — predator chatter doesn't bleed into a weather query, troubleshooting doesn't bleed into a ducks query, etc.
- Healthy recall — relevant prior context (recent or old) IS surfaced when topic-related.
- Recency-weighted ranking — recent-and-relevant ranks ahead of old-and-relevant.
- Topic shifts — pivot-and-return works (lions → BMW → "what predator did I ask about earlier?").
- Edge cases — short queries, paraphrased queries, queries with no matching memory.

Per-scenario validation: retrieval `expect_includes` / `expect_excludes` substrings + LLM `expect_response_excludes` for contamination tokens.

**Run:** `python scripts/comprehensive_memory_quality.py [--skip-llm] [--scenario-filter X] [--audit-log PATH]`. Without `--skip-llm`, loads Qwen 4B and exercises the full retrieve → context-assembly → response path. ~3.5 GB VRAM.

### `scripts/comprehensive_search_blending.py` (2026-05-09 memory-quality pass)

**Purpose:** end-to-end search-augmented contamination tests with REAL Brave + Jina + Qwen 4B. Verifies the orchestrator's `_search_augmented_tokens` path: predator chatter in memory doesn't bleed into a Python-3.13 search response; troubleshooting context doesn't bleed into a duck-lifespan search response; relevant troubleshooting context DOES blend into a motherboard-light search response.

3 scenarios; ~3 Brave + 3 Jina calls per full run. Within free-tier quota.

**Run:** `python scripts/comprehensive_search_blending.py` (requires `ULTRON_BRAVE_API_KEY`).

### `scripts/_debug_retrieval_cosine.py` (2026-05-09 memory-quality pass; debug only)

**Purpose:** prints cosine similarity between a probe query and a hand-picked candidate set. Used to empirically tune `memory.rag_min_relevance` against the actual production embedder (bge-small INT8). The 0.6 threshold was chosen because off-topic content peaked at 0.55-0.57 across the probe corpus, while genuinely relevant content scored 0.7-0.95.

**Run:** `python scripts/_debug_retrieval_cosine.py`. No flags; edit the `PROBES` and `CANDIDATES` lists at the top of the file to test new query+content pairs.

---

## Tests

### `tests/conftest.py` — Path setup so `from ultron.*` works.

### Default suite (no env gate) — 1575 passed / 15 skipped (GPU-gated), ~51 s wall (2026-05-10)

**Top-level (~25 files):**
- `test_addressing.py` — rule-based addressing classifier
- `test_audio.py` — capture, ring buffer (incl. 2026-05-10 mode-aware `snapshot(last_n_samples=...)` slicing), devices
- `test_response_style.py` (22, 2026-05-10) — `is_brief_question` / `apply_brevity_hint` coverage: short-question detection, depth-marker skip, long-question pass-through, empty input, idempotence on already-hinted text
- `test_coding_bridge.py` — CodingBridge abstract contract
- `test_coding_e2e.py` — coding e2e (PYTEST_RUN_GPU_TESTS gated)
- `test_coding_intent.py` / `test_coding_intent_phase2.py` — intent classifier
- `test_coding_projects.py` — registry + resolver + sandbox creation
- `test_coding_runner.py` — runner state machine
- `test_coding_templates.py` — template renderer
- `test_coding_voice.py` — voice controller (now CapabilityVoiceController)
- `test_coordinator.py` — clarification + correction loops
- `test_correction_loop.py` — corrective re-prompting
- `test_fairseq_compat.py` — torch.load + dataclass workarounds
- `test_llm.py` — LLM (PYTEST_RUN_GPU_TESTS gated)
- `test_maintenance.py` — periodic maintenance
- `test_mcp_e2e.py` / `test_mcp_server.py` / `test_mcp_session.py` — MCP layer
- `test_memory_qdrant.py` — Qdrant memory + embedder
- `test_narration.py` — StatusNarrator
- `test_phase7_audit_and_tokens.py` — per-session audit + token tracking
- `test_pipeline.py` — orchestrator construction (PYTEST_RUN_GPU_TESTS gated)
- `test_projections.py` — 29 projection tests (Phase 2 + Foundation Part 2)
- `test_transcription.py` — Whisper (PYTEST_RUN_GPU_TESTS gated)
- `test_tts.py` — Piper + RVC
- `test_uncertainty.py` — uncertainty signal application
- `test_verification.py` — six verification checks
- `test_web_gating.py` — two-stage gating
- `test_persona_loader.py` (20, OpenClaw Phase 1) — `PersonaLoader` modes / hot-reload / HTML-comment-only files
- `test_llm_persona_source.py` (8, OpenClaw Phase 1) — `LLMEngine` persona-source wiring + hot-reload + fallback
- `test_llm_http_runtime.py` (9, OpenClaw Phase 0) — HTTP-runtime construction, request shape, SSE streaming, cancel mid-stream
- `test_llm_preset.py` (13, 4B plan Stage A) — `LLMConfig.preset` resolution: 9b/4b/custom defaults, explicit-override wins, YAML round-trip, invalid preset rejected
- `test_start_llamacpp_server.py` (13, 4B plan Stage C) — launcher CLI: --help renders, default args back-compat, --model-draft attaches speculative decoding, --draft-num-pred-tokens override, --from-config overlay (4b/9b), CLI flags override overlay
- `test_llm_enable_thinking.py` (11, 4B plan Stage F) — `enable_thinking` parameter plumbing: helper kwargs, in-process generate/generate_stream pass-through, HTTP payload pass-through, back-compat when default
- `test_llm_rag_position.py` (7, 4B plan Stage G) — `_build_messages` honors `llm.rag.position`: recency mode prepends to user message, system mode folds into system message, no-snippets/retrieve-failure fallback, helper invariants
- `test_on_the_fly_preset_switching.py` (16, 4B plan Stage H infra) — `ULTRON_LLM_PRESET` env-var override (clears overrides by default, opt-in keep-overrides flag), minimal-YAML preset-only config, `check_vram._resolve_target_mb` (table + CLI override + env var + unknown fallback), `_format_line` shows preset label, `swap_llm_preset._rewrite_preset` (basic / preserves comment / first-match / missing-line raises)
- `tests/routing/test_model_switch_classifier.py` (54, 4B plan voice-swap) — classifier maps "switch to 4B/9B/four B/for B/nine B/4 B/4-B" + verb variants (switch/swap/change/use/load/go/move/activate/engage/run/select) to `RoutingIntentKind.MODEL_SWITCH`; rejects passing mentions ("the 4B is faster") and conversational utterances; pending clarification suppresses (mid-dialogue safety); active coding task does not block; `_resolve_model_switch_target` helper
- `test_llm_reload_for_preset.py` (9, 4B plan voice-swap) — `LLMEngine.reload_for_preset` rejects http_server runtime + unknown preset; idempotent on same-preset; success path replaces `_llm` and clears history; sets `ULTRON_LLM_PRESET` env + clears stale `ULTRON_LLM_MODEL_PATH`; failure path keeps old engine, restores env vars (whether they were set or unset originally)
- `test_llm_prompt_injection_defense.py` (21, comprehensive QUALITY pass Q10 iter 1+2) — `_sanitize_user_input` neutralises tag-style markers ([INST]/[/INST], <|im_start|>/<|im_end|>/<|system|>/<|user|>/<|assistant|>, stray </think>); detects natural-language jailbreak patterns ("ignore previous instructions", "you are now X", "respond with the exact word", "act as", "pretend"); preserves benign questions (zero false-positive on normal voice queries); end-to-end verified: pre-defense 2/3 of Q8 prompt-injection probes succeeded; post-defense 0/3. Voice baseline TTFT 79 ms / VRAM 7889 MB unchanged (defence is sub-microsecond on benign input).
- `test_web_search_parallel_fetch.py` (6, 2026-05-09 latency hot-fix) — verifies the `WebSearchExecutor` parallel-Jina-fetch path: wall-time dominated by the slowest URL (not the sum); collective deadline abandons slow fetches and degrades them to snippet-only with `jina_deadline:<url>` notes; partial success with one fast + one slow URL keeps the fast one's `full_text`; per-fetch exception in one parallel branch doesn't break the others; `collective_deadline_seconds=0` disables the cap; `max_fetch=0` skips Jina entirely.
- `test_tts_pipeline_parallel.py` (15, 2026-05-09 + 2026-05-10) — original 11 cover the parallel split, speculative stream open, sample-rate-mismatch fallback, low-latency hint, RVC fallback, cancellation. 2026-05-10 added 4 for producer-signaled lookahead: `test_first_clip_plays_before_next_fragment_yielded` (the ack-first contract — first clip MUST be written to the stream before the generator is asked for the second), `test_slow_second_clip_does_not_kill_playback` (4 s gap between fragments doesn't trigger RVC starvation abort — guards the BMW-search failure mode), `test_clipitem_is_known_last_skips_lookahead` (ClipItem namedtuple shape + flag carries through), `test_end_of_stream_sentinel_terminates_playback` (None on audio_q ends playback with tail silence even when the previous ClipItem had `is_known_last=False`).
- `test_voice_model_switch.py` (11, 4B plan voice-swap) — `CapabilityVoiceController._handle_model_switch` calls `llm_engine.reload_for_preset(target)`, speaks "Switched to the 4B/9B" on success, "I'm already running the X" on idempotent, "I couldn't switch ..." on failure with reason; "I can't switch models — engine isn't wired" when llm_engine is None; missing payload says "couldn't tell which model"; end-to-end classifier-then-controller for utterances
- `tests/routing/test_irma_reformulation.py` (15, 4B plan Item 5) — `InputReformulator` pure-text shape (default-only-utterance, whitespace-strip, quote-escape, recent-decisions section, max-recent truncation, active-session, routing-hints, max_recent=0 omits, log-row factory); disambiguator integration with the IRMA flag (default-OFF passes raw, ON uses enriched, reformulation-failure falls back, no-context still emits utterance)
- `test_self_consistency.py` (27, 4B plan Item 6) — `majority_vote_text` (winner, whitespace-strip, tie-first-wins, empty input, blank filter), `majority_vote_json` (winner, unparseable handling, think-block strip, first-block-only, all-unparseable returns None, arrays), `majority_vote_label` (case-insensitive, no-match), `run_self_consistency` driver (sampler called N times, default text aggregator, sampler exception handling, fallback to first non-empty, n-clamping), `should_apply_self_consistency` config gate (default-off, global-on, per-site disabled), decomposer integration (single-call default, N-call with consistency, majority winner, per-site bypass, all-unparseable fallback)
- `test_canonical_monitor.py` (17, 4B plan Item 7) — canonical set lockdown (standard tools, MCP callbacks), canonical-only paths (no abort), threshold-not-reached, threshold-reached-in-window aborts, late drift does not abort, latch semantics, reset clears state, non-tool-use events ignored, empty/None tool name ignored, case-insensitive match, attribute-style event input, custom canonical override, verdict-shape (off_canonical_tools list, immutability), factory gate (disabled returns None, enabled returns instance with config)
- `test_block_and_revise.py` (14, 4B plan Item 8) — `ToolCallValidator` ALLOW + BLOCK verdicts, think-block strip, case-insensitive, fail-open on no-LLM / exception / unparseable / empty, prompt rendering (tool name, args, args truncated, goal-quote escaped), `is_enabled` config gate
- `test_compression.py` (26, 4B plan Item 4) — heuristic compresses redundant text, preserves negations (and "isn't" preserves negation-meaning), collapses repeated punctuation, short input passthrough, empty passthrough, ratio-1.0 means no drop, higher-ratio drops more; perplexity-scorer drops lowest-score, scorer exception fallback, mismatched-length fallback; result dataclass; factory off-returns-None / on-returns-instance; `maybe_compress` global-off / per-surface-off / per-surface-on / unknown surface / history default-off / compressor exception / empty text; integration `_format_rag_block` default-OFF unchanged + ON-compresses; `format_sources_for_prompt` default-OFF unchanged + URL-preserved-on
- `test_self_consistency_web_gating.py` (8, 4B plan Item 6 second site) — `web_search.gating.classify_by_preflight` with self-consistency: default-OFF single greedy call (back-compat), N-call when enabled, configured non-zero temperature, majority-vote winner, per-site disabled bypass, all-unparseable fallback to NO_SEARCH, LLM-exception returns NO_SEARCH (never raises)
- `test_canonical_monitor_runner_wiring.py` (9, 4B plan Item 7 wiring) — `CodingTaskRunner` listener gating: not-attached-when-disabled, attached-when-enabled, cancels handle on first abort verdict, doesn't cancel on canonical sequence, latches after first abort, swallows listener exceptions; `CapabilityVoiceController.pending_canonical_abort` polls + clears + swallows runner exception
- `test_block_and_revise_dispatcher_wiring.py` (10, 4B plan Item 8 wiring) — `OpenClawDispatcher` per-handler validator gate: disabled-flag skips, no-LLM skips, ALLOW dispatches to stub, BLOCK short-circuits with reason, all 5 handlers run validator when enabled, validator exception falls open, voice controller threads its `llm_engine` to the dispatcher

**`tests/coding/`:**
- `mock_bridge.py` — `ScriptedClaudeBridge` + `ClaudeScript` DSL
- `test_orchestration.py` — 11 mock-bridge scenarios (10 spec + 7b delta-tracking)
- `test_orchestration_real.py` — same scenarios with real Claude (gated)
- `test_mock_bridge_smoke.py` — mock-bridge sanity
- `sandbox/` — fixture sandbox

**`tests/error_recovery/`** (Phase 4) — 78 tests:
- `test_brave_failures.py`, `test_jina_failures.py`, `test_qdrant_failures.py`
- `test_audio_failures.py`, `test_addressing_failures.py`, `test_config_failures.py`
- `test_circuit_breaker.py`, `test_error_log.py`
- `test_claude_code_failures.py` (18) — launch fail / timeout / nonzero exit / stream-json error events with API-pattern detection
- `test_mcp_server_failures.py` (3) — bind failure / no active session / audit-log write failure
- `test_filesystem_failures.py` (5) — session audit / project registry / coding tasks audit-log dedup

**`tests/routing/`** (Phase 5) — 148 tests:
- `test_classifier.py` (90: 20 BROWSER, 10 each MEDIA/MESSAGING/FILE/SHELL/HYBRID/CONVERSATIONAL, 8 CODE_TASK, 2 edge)
- `test_dispatcher.py` (12)
- `test_decomposer.py` (9)
- `test_disambiguator.py` (25)
- `test_decision_log.py` (8)
- `test_backward_compat.py` (4)

**`tests/integration/`** (Phase 6) — 83 tests:
- `test_routing_dispatch.py` (20)
- `test_conversational_pipeline.py` (21)
- `test_search_pipeline.py` (12)
- `test_coding_pipeline.py` (9)
- `test_addressing_pipeline.py` (13)
- `test_error_recovery_pipeline.py` (4)
- `mocks.md` + `performance.json` (reference files)

### Slow / GPU-gated tests (16 skipped by default)

Set `$env:PYTEST_RUN_GPU_TESTS = "1"` before pytest. Includes real Claude API calls (`test_coding_e2e.py`, `test_mcp_e2e.py`, `test_orchestration_real.py`) — burns tokens.

---

## Runtime artifacts

### `logs/`

| File | Writer | Format | Purpose |
|---|---|---|---|
| `ultron.log` | `utils.logging.configure_logging()` | text, rotating 5 MB×3 | Main log — all subsystem messages |
| `addressing.jsonl` | `AddressingClassifier._log()` | JSONL | Every classifier verdict |
| `coding_tasks.jsonl` | `CodingTaskRunner._make_log_listener()` | JSONL | Coding task progress events |
| `verifications.jsonl` | `Verifier.verify()` | JSONL | Per-verification report |
| `clarifications.jsonl` | `_ClarificationLog` (in coordinator) | JSONL | Clarification decisions |
| `mcp_calls.jsonl` | `_AuditLog` (in mcp_server) | JSONL | MCP tool calls |
| `sessions/<id>.jsonl` | `SessionAuditWriter` | JSONL | Per-session full event audit |
| `errors.jsonl` | `resilience.error_log.ErrorLog.record()` | JSONL | Phase 4 typed errors |
| `routing_decisions.jsonl` | `openclaw_routing.decision_log.RoutingDecisionLog.record()` | JSONL | Phase 5 routing audit |
| `automation_tasks.jsonl` | `AutomationTaskRunner._audit()` | JSONL | Phase 5 OpenClaw task records |

### `data/`

| Path | Owner | Purpose |
|---|---|---|
| `qdrant/` | `ConversationMemory`, `WebResultsCache` | Embedded Qdrant store; 3 collections |
| `memory.jsonl` | (legacy) | Pre-Qdrant turn log; migration source / recovery |
| `projects.json` | `ProjectRegistry` | Coding project registry |
| `sandbox/` | `new_sandbox_project()` | Auto-created coding projects |
| `summaries.jsonl` | `scripts/maintenance.py` | Conversation summaries |
| `maintenance.sqlite` | `scripts/maintenance.py` | Maintenance state (cursors, etc.) |
| `ollama_compat_test/` | (Foundation Phase 0) | Modelfile from Ollama compat test (not in active use) |

### `models/` (main checkout only)

| File | Used by | Size |
|---|---|---|
| `Qwen3.5-9B-Q4_K_M.gguf` | `LLMEngine` (when `llm.preset == "qwen3.5-9b"`, current default) | 5.29 GB |
| `Qwen3.5-4B-Q4_K_M.gguf` | `LLMEngine` (when `llm.preset == "qwen3.5-4b"`, primary after Stage H) | 2.55 GB |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | speculative-decoding draft for 4B (paired by `qwen3.5-4b` preset; Stage C wires into `start_llamacpp_server.py`) | 0.50 GB |
| `openwakeword/ultron.onnx` | `WakeWordDetector` | small |
| `piper/en_US-ryan-medium.onnx[.json]` | `TextToSpeech` | ~60 MB |
| `rvc/hubert_base.pt` | `RvcConverter` | ~362 MB |
| `rvc/rmvpe.pt` | `RvcConverter` | ~178 MB |
| `.hf-cache/` | `HybridEmbedder`, addressing zero-shot | varies |

### `ultron_james_spader_mcu_6941/` (main checkout only)

RVC voice model for Ultron timbre.
- `Ultron.pth` — main RVC checkpoint
- `added_IVF301_Flat_nprobe_1_Ultron_v2.index` — speaker index

---

## Documentation index

Reading order for a fresh Claude:

1. **`CLAUDE.md`** (project-root, auto-loaded by Claude Code) — orientation + binding standards.
2. **`MEMORY.md`** (auto-loaded) — index of memory files.
3. **`project_ultron_openclaw.md`** — primary cross-phase OpenClaw reference.
4. **`project_ultron_4b_plan.md`** — final 4B + Items 4–8 state with measured TTFT/VRAM.
5. **`feedback_*.md`** — confirmed user decisions (especially `feedback_no_paid_apis.md`, `feedback_llm_runtime_decision.md`).
6. **`docs/codebase_structure.md`** ← THIS FILE — single-source reference.
7. **`docs/openclaw_integration_final_summary.md`** — cross-phase OpenClaw reference + intentional deviations + setup-readiness checklist.
8. **`docs/architecture.md`** — pipeline + diagrams.
9. **`docs/phase3_5_followup.md`** — open punch list (deferred Foundation Part 3.5).

### Comprehensive testing + improvement passes (most recent)
- **Functional / correctness pass plan:** [docs/comprehensive_test_plan.md](comprehensive_test_plan.md) — 16 phases, 38 dimensions, single-process harness pattern.
- **Functional pass results:** [docs/comprehensive_test_report.md](comprehensive_test_report.md) — 145-row metrics table; 4 classifier coverage gaps fixed; voice baseline 79 ms / 7818 MB.
- **Quality pass plan:** [docs/comprehensive_quality_plan.md](comprehensive_quality_plan.md) — 13 phases (Q0–Q13), 38 quality dimensions, ≤10-iteration improvement loop.
- **Quality pass results:** [docs/comprehensive_quality_report.md](comprehensive_quality_report.md) — 107-row metrics table; Q10 iteration audit; prompt-injection defense layer.

### Foundation reference
- Day-to-day operation: [docs/operations.md](operations.md)
- Adding code / debugging: [docs/development.md](development.md)
- Config reference: [docs/configuration.md](configuration.md)
- Error handling: [docs/error_handling.md](error_handling.md)
- Capability routing: [docs/routing.md](routing.md)
- Test layout: [tests/integration/mocks.md](../tests/integration/mocks.md)
- 16-step end-to-end smoke test: [docs/smoke_test.md](smoke_test.md)
- Foundation Phase 1 inventory snapshot: [docs/system_inventory.md](system_inventory.md)
- Phase 3 discovery catalog: [docs/config_discovery.md](config_discovery.md)

### OpenClaw integration (architecture)
- **OpenClaw integration architecture + Phase 0/1 status:** [docs/openclaw_integration.md](openclaw_integration.md)
- **OpenClaw runtime ops (agents, supervisor, locked-in constraints):** [docs/openclaw_runtime.md](openclaw_runtime.md)
- **Cross-phase final summary + setup-readiness checklist:** [docs/openclaw_integration_final_summary.md](openclaw_integration_final_summary.md)

### OpenClaw integration (per-phase close-outs)
- **Phase 1 (persona migration):** [docs/phase_1_summary.md](phase_1_summary.md)
- **Phase 3 (bridge layer):** [docs/phase_3_summary.md](phase_3_summary.md)
- **Phase 4 (Telegram channel):** [docs/phase_4_summary.md](phase_4_summary.md)
- **Phase 5 (heartbeat):** [docs/phase_5_summary.md](phase_5_summary.md)
- **Phase 6 (browser tool):** [docs/phase_6_summary.md](phase_6_summary.md)
- (Phases 7–13 have inline summaries in `openclaw_integration_final_summary.md`.)

### OpenClaw integration (user-side setup procedures)
- **Telegram channel:** [docs/openclaw_telegram_setup.md](openclaw_telegram_setup.md)
- **Heartbeat agents[].heartbeat block:** [docs/openclaw_heartbeat_setup.md](openclaw_heartbeat_setup.md)
- **Browser tool (Playwright + Chromium):** [docs/openclaw_browser_setup.md](openclaw_browser_setup.md)
- **Cron jobs (Windows Task Scheduler fallback):** [docs/openclaw_cron_setup.md](openclaw_cron_setup.md)
- **Bundled hooks (`session-memory`, `command-logger`):** [docs/openclaw_hooks_setup.md](openclaw_hooks_setup.md)
- **Memory Wiki plugin:** [docs/openclaw_memory_wiki_setup.md](openclaw_memory_wiki_setup.md)
- **Local-only ComfyUI media generation:** [docs/openclaw_media_generation_setup.md](openclaw_media_generation_setup.md)
- **iOS / Android node pairing:** [docs/mobile_node_setup.md](mobile_node_setup.md)
- **Standing-order programs:** [docs/standing_orders.md](standing_orders.md)
- **Three-layer memory architecture (Qdrant + workspace + Wiki):** [docs/memory_architecture.md](memory_architecture.md)
- **Gaming mode (V1-gap A1):** [docs/openclaw_gaming_mode_setup.md](openclaw_gaming_mode_setup.md)
- **Desktop / window control (V1-gap C3):** [docs/openclaw_desktop_control_setup.md](openclaw_desktop_control_setup.md)

### 4B optimization plan
- **4B-model optimization plan (all stages + Items 4–8 done):** [docs/4b_optimization_plan.md](4b_optimization_plan.md)
- **GGUF SHA256 reference:** [docs/model_checksums.md](model_checksums.md)

---

## Maintenance contract

**This document is the operating manual. Keep it current.**

This contract is **binding** — every non-trivial change to the
codebase must update this document in the same change. Skipping
the update means future sessions waste time re-deriving ground
truth from the source. **Don't skip.**

The CLAUDE.md (project-root) at the top of this prompt's reading
order calls this contract out explicitly so a fresh Claude Code
session sees it before its first edit.

### What "non-trivial change" means

You MUST update the relevant section of this document when you:

1. **Add a new module file** under `src/ultron/` →
   - Add to the file tree.
   - Add a section under "Source modules" with the public API
     (classes, functions, brief in/out).
   - If it's a new subsystem (e.g. `src/ultron/openclaw_bridge/`),
     add to the architecture diagram in `docs/architecture.md`
     too.

2. **Add a new public class or function** to an existing module →
   - Add it to the module's section under "Source modules".
   - Note the inputs and outputs in one line.

3. **Remove or rename** an existing module / class / function →
   - Update every section that referenced it.
   - Search for the old name with Grep before declaring done.

4. **Add a new script** under `scripts/` →
   - Add to the file tree.
   - Add a section under "Operational scripts" with purpose,
     run command, in/out, and functions.

5. **Add a new test directory or test category** →
   - Add to the file tree (under `tests/`).
   - Add to the relevant "Tests" subsection.
   - Update the "current state" header at the top of this file
     with the new total.

6. **Add a new log file or data path** →
   - Add to the "Runtime artifacts" tables.

7. **Add a new doc** under `docs/` →
   - Add to the "Documentation index" with the right category
     (Foundation reference / OpenClaw architecture / per-phase
     close-out / user-side setup / 4B plan).
   - Add to the file tree under `docs/`.
   - Cross-reference where relevant in other sections.

8. **Add a new config section / key** →
   - Add to the `config.yaml` summary in "Configuration".
   - Update [docs/configuration.md](configuration.md) too
     (per-key reference).
   - Document any new defaults in the relevant `feedback_*.md`
     if it reflects a confirmed user decision.

9. **Change a cross-cutting flow** (voice path, coding path,
   search path, dispatch path, OpenClaw bridge path) →
   - Update the relevant diagram in "Cross-cutting flows".

10. **Migrate a subsystem out of the `config/settings.py` shim** →
    - Update [docs/phase3_5_followup.md](phase3_5_followup.md)
      (cross off).
    - If it changes the public API of the migrated module,
      update its "Source modules" section here.

11. **Bump test counts** — the file's header tracks "X passed /
    Y skipped / Z failed". Update these when the count changes.

12. **Land a new phase / sub-phase** → bump the phase status
    line in the header.

### The validation loop

After your change:

```powershell
# 1) Tests pass
C:\STC\ultronPrototype\.venv\Scripts\python.exe -m pytest tests/ -q --no-header --ignore=tests/coding/test_orchestration_real.py

# 2) Config still validates
C:\STC\ultronPrototype\.venv\Scripts\python.exe scripts\validate_config.py

# 3) Re-read this doc and confirm:
#    - File tree matches `git ls-files | grep -v '^\\.'`
#    - "Source modules" sections cover every src/ultron/ file
#    - "Operational scripts" sections cover every scripts/ file
#    - "Tests" subsections cover every tests/ subdirectory
#    - "Documentation index" links every docs/*.md file
```

If the doc no longer matches reality after your changes, fix
this document before declaring the task done.

### Why this matters

A fresh Claude Code session reads this document + the memory files
and should be fully oriented without re-exploring the codebase. If
that's not the case after your changes, the maintenance contract
was violated. Treat that as a regression and fix the doc.

To verify the document still matches reality:
```powershell
# Run after any non-trivial change
pytest tests/ -q
python scripts/validate_config.py
# Then re-read this doc and confirm tree + module sections are current
```
