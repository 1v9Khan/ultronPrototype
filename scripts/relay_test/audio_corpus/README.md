# Live audio-injection corpus protocol

A dedicated end-to-end test harness that exercises the **full** Ultron voice
pipeline from raw audio — wake word → pre-roll → audio-domain wake-drop →
Whisper STT → norm1/norm2 → semantic routing → tail selection → the **real** 3B
(for LLM-routed turns) → Kokoro TTS — by feeding synthesized command audio in
exactly as if it were spoken into the mic. Unlike the text-level corpus tracers
(`scripts/relay_test/trace_corpus_full.py`), this validates the *audio* stages:
transcription accuracy, full-command capture (no clipping), and that the wake
word never contaminates the transcript.

Nothing in runtime `src/` is touched: the mic is a drop-in swap of
`orchestrator.audio`, and response audio is captured via a class-level Kokoro
synth hook.

## Pipeline

1. **`gen_commands.py`** — per battery command (`scripts/relay_test/battery_cmds.txt`,
   each line `Ultron, <command>`), synthesize the **command body** in a STOCK
   Kokoro voice (`am_michael`, fast combat cadence ~1.18x) to simulate a real
   person, then splice a composite clip:
   `[0.5s silence] + [trained "Ultron" wake sample] + [gap] + [stock body] + [1.3s silence]`.
   The trained wake sample (`training/crosscheck_ultron/*.wav`, fires ~0.94) is
   prepended because the custom openWakeWord model scores stock-Kokoro "Ultron"
   at ~0.27 (< 0.65 threshold) and would never fire. The stock body then tests
   the pre-roll + audio-domain wake-drop + no-clipping + STT exactly as live.
   Output: `out/wav/*.wav` (injection source) + `out/mp3/*.mp3` (deliverable) +
   `out/manifest.json`.
2. **`run_corpus.py`** — boot the full `Orchestrator` in-process, swap
   `orch.audio = InjectableCapture`, enable testing mode, optionally move the 3B
   to GPU (`--gpu`), and drive each clip through the live `orch.run()` loop. Per
   command it captures the full per-stage trace (from `logs/usage_trace.jsonl`),
   saves the spoken response audio, and **re-transcribes the response with
   Whisper** to verify it is understandable speech. Output: a session-stamped
   `session_<stamp>/corpus_<stamp>.log.jsonl` + `responses/*.{wav,mp3}`.
   Real LLM calls are NOT skipped — LLM-routed turns hit the real 3B so the
   output can be audited.
3. **`render_review.py`** — render the session log into a scannable per-case
   `.review.txt` with auto-flags (TX transcription mismatch, RE response-not-
   understandable, WK wake leak, NR no-trace-row, LLM fell-to-3B) to support a
   by-hand note-per-case audit.
4. **`inject.py`** — `InjectableCapture(AudioCapture)`, the drop-in mic: `feed_pcm`
   chunks fed audio into 256-sample frames and `get_chunk` serves them then
   real-time-paced silence. Zero change to runtime `capture.py`.

## Usage

```bash
# 1. generate the composite command clips
python scripts/relay_test/audio_corpus/gen_commands.py [--limit N]

# 2. run the full pipeline over them (real 3B; --gpu for speed)
python scripts/relay_test/audio_corpus/run_corpus.py [--limit N] [--gpu]

# 3. render the by-hand review file
python scripts/relay_test/audio_corpus/render_review.py \
    scripts/relay_test/audio_corpus/session_<stamp>/corpus_<stamp>.log.jsonl
```

By-hand audit notes are written to a session-stamped file under
`logs/relay_test/_corpus_audit_notes_<stamp>.md` (matching the corpus stamp).

## Caveat — TTS simulation fidelity

Stock `am_michael @1.18x` garbles SHORT Valorant jargon (e.g. "jett A main" →
"GENTLEMEN", "sova" → "Silva"), so transcription-mismatch (TX) flags on short
callouts are heavily inflated by the synthetic voice and are mostly **not**
pipeline bugs — a real human voice transcribes them correctly. A faithful
short-callout transcription audit needs the user's real voice (or a higher-
fidelity / jargon-tuned TTS). The wake-firing, capture, wake-drop, routing,
snap, and tail stages are validated regardless.

## Artifacts (not committed)

`out/` (generated audio + manifest) and `session_*/` (corpus logs + response
audio) are reproducible run artifacts and are git-ignored. Only the four
protocol scripts + this README are tracked.
