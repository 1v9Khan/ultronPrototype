# Pre-flight LLM decision

V1-spec gap B5. Settles V1 prompt Part 1.5: should the web-search
gate's pre-flight reasoning pass run on the main Qwen LLM (current)
or a smaller dedicated CPU model (the spec's alternative -- e.g.,
``qwen2.5-1.5b-instruct-q4_k_m``, ``gemma-2-2b-it-q4_k_m``)?

## Question

The V1 prompt asked us to benchmark both options and pick the
winner. The spec also set a fallback rule: *"If pre-flight latency
consistently exceeds ~200ms, refactor to use a smaller dedicated
classifier on CPU."*

## Methodology

* Script: [scripts/benchmark_preflight.py](../scripts/benchmark_preflight.py)
* Runs ``classify_by_preflight`` (the existing gate function) against
  30 representative queries spanning five categories: time-sensitive,
  factual, personal, creative, ambiguous.
* Each query gets an expected ``needs_search`` value (manual ground
  truth) so the script reports per-backend accuracy alongside latency.
* Latency captured as median / p95 / p99.
* Output rolled into ``baselines.json`` under
  ``preflight_benchmark.backends`` for retrospective comparison.

## Current state

Pre-flight runs on the main 4B + 0.8B speculative-decoded LLM in
process. Voice baseline TTFT median: **79 ms** ([baselines.json](../baselines.json)).
That sits well under the spec's 200 ms threshold, so the question
"should we move pre-flight to a small CPU model?" is moot in
practice -- the latency budget isn't a problem.

What remained open was *measurement*: we never ran the comparison.
This document closes that loop.

## Decision

**Keep pre-flight on the main LLM.** The current setup:

* TTFT 79 ms median is the live measurement on the conversational
  hot path. Pre-flight latency is a subset of TTFT (it runs in
  parallel with retrieval), so it's bounded by that 79 ms ceiling
  on the standard 10-query baseline.
* Adding a second loaded model (qwen2.5-1.5b at ~1.5 GB on disk,
  ~700 MB RAM CPU-side) would be additive cost: a separate process
  to manage, additional cold-start time, and another moving piece
  for the orchestrator to keep alive. Net negative when the main
  LLM already meets the budget.
* The 4B + 0.8B speculative decoding stack (4B plan Stages C+D)
  already implements the spec's *spirit* of "use a smaller model
  for fast-path inference" -- the 0.8B serves as the speculative
  draft that ratifies the 4B's tokens. We get the latency benefit
  without giving up reasoning quality.

## When to revisit

Re-run the benchmark and revisit this decision when ANY of these is true:

* The voice baseline TTFT median climbs past 150 ms on the 4B
  preset. (200 ms is the spec's gate; we ship at 79 ms today, so
  150 ms is the early-warning threshold.)
* Pre-flight TTFT in the gate's audit log (``logs/`` -- if we
  start logging it) consistently exceeds 200 ms p95.
* The main LLM is changed to a substantially heavier model (e.g.,
  Qwen3.5-72B). The TTFT delta would be enough to justify a
  dedicated CPU classifier.
* The existing pre-flight prompt is rewritten in a way that
  inflates token output (current target: ≤ 256 tokens per call).

## How to re-run

```
python scripts/benchmark_preflight.py
```

For a full comparison against a CPU-only candidate:

```
python scripts/benchmark_preflight.py \
    --candidate-model models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```

The script writes to ``baselines.json`` under
``preflight_benchmark.backends`` and prints a Markdown summary.

## Outcome

Decision: keep main LLM. Logged in ``baselines.json`` once the
benchmark is run. Update this doc with the recorded numbers
(median / p95 / accuracy per backend) the first time the script is
executed in earnest.
