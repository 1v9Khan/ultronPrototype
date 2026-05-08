"""Phase 0 baseline measurement.

Loads the existing Ultron stack (Whisper + LLM + embedder + RVC + Piper) and
records VRAM and latency for 10 representative queries. Output: baselines.json
at the worktree root. Re-run after each phase to compare.

Notes:
- Imports point at the main checkout (`C:\\STC\\ultronPrototype`) so model paths
  resolve correctly. Run from anywhere.
- TTS is exercised through synth (Piper + RVC) but **not** played back --
  saves audio device + ~30 s and matches Phase 0 spec ("measure up to playback").
- A temp copy of `data/memory.jsonl` is used so the warmup turn doesn't
  pollute the user's real conversation history.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Windows console default is cp1252; reconfigure stdout so any stray Unicode
# in our output (or downstream library logs) doesn't crash on print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Path setup: import code from this script's own ``src/`` (the worktree it
# lives in) so a measurement always exercises the version of the code we're
# evaluating. Models still live in the main checkout, so we add main's repo
# root for ``config/`` shim + relative ``models/`` paths to resolve via cwd.
# Run with ``cwd=C:\STC\ultronPrototype`` so ``models/...`` is found.
# ---------------------------------------------------------------------------
WORKTREE_ROOT = Path(__file__).resolve().parent.parent
MAIN_REPO_PATH = Path(r"C:\STC\ultronPrototype")
sys.path.insert(0, str(MAIN_REPO_PATH))            # config/ shim
sys.path.insert(0, str(WORKTREE_ROOT / "src"))     # newest ultron code

# Output: baselines.json at the worktree root.
OUTPUT_PATH = WORKTREE_ROOT / "baselines.json"

REPRESENTATIVE_QUERIES = [
    "What is the boiling point of water?",
    "Walk me through how a transistor works.",
    "Who was Nikola Tesla?",
    "What's nineteen times forty-three?",
    "Explain what a hash table is.",
    "Are you afraid of death?",
    "What's a good book to read on a flight?",
    "What do you think about meditation?",
    "And what about the Mariana Trench?",
    "Tell me something interesting about black holes.",
]


def vram_used_mb() -> int:
    """Total VRAM currently used by all processes on GPU 0, in MB."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
    ).strip()
    return int(out)


def synthetic_audio(seconds: float = 2.5, sr: int = 16000) -> np.ndarray:
    """Fixed sine for repeatable Whisper timing. Content is meaningless;
    we only need a consistent input length so transcription time is comparable.
    """
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def first_sentence_of(stream, t0: float, flush_chars=set(".!?\n")):
    """Drain ``stream`` until the first sentence terminator. Cancels generation
    afterward so we don't wait for a full response.

    Returns ``(first_token_ms, sentence_complete_ms, sentence_text)``.
    """
    first_token_ms: Optional[float] = None
    sentence_complete_ms: Optional[float] = None
    parts: list[str] = []

    iterator = iter(stream)
    for token in iterator:
        if first_token_ms is None:
            first_token_ms = (time.monotonic() - t0) * 1000
        parts.append(token)
        if any(c in flush_chars for c in token):
            sentence_complete_ms = (time.monotonic() - t0) * 1000
            break

    if first_token_ms is None:
        first_token_ms = (time.monotonic() - t0) * 1000
    if sentence_complete_ms is None:
        sentence_complete_ms = first_token_ms

    return first_token_ms, sentence_complete_ms, "".join(parts).strip()


def main() -> int:
    print("=" * 60)
    print("Phase 0 baseline measurement")
    print("=" * 60)

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "main_repo": str(MAIN_REPO_PATH),
            "queries_count": len(REPRESENTATIVE_QUERIES),
            "queries": REPRESENTATIVE_QUERIES,
            "notes": (
                "TTS measured through synth (Piper + RVC) without playback. "
                "VRAM is total GPU 0 used (includes desktop apps); deltas "
                "are the meaningful figures."
            ),
        },
        "vram_mb": {},
        "latency_ms": {},
    }

    vram_idle = vram_used_mb()
    results["vram_mb"]["before_load"] = vram_idle
    print(f"\n[before load] GPU 0 VRAM in use: {vram_idle} MB")

    # Quiet logging during the run.
    import os
    os.environ["ULTRON_LOG_LEVEL"] = "WARNING"

    from ultron.utils.logging import configure_logging
    configure_logging(level="WARNING")

    print("\nLoading components...")

    from ultron.transcription import WhisperEngine
    t = time.monotonic()
    stt = WhisperEngine()
    print(f"  Whisper loaded in {time.monotonic() - t:.1f}s")
    vram_after_stt = vram_used_mb()
    results["vram_mb"]["after_whisper"] = vram_after_stt
    print(f"    VRAM: {vram_after_stt} MB (delta {vram_after_stt - vram_idle:+d})")

    # Phase 3+: the memory module is CPU-only (FastEmbed bge-small + BM25 +
    # embedded Qdrant) and lives off the COLD-mode hot path entirely --
    # writes are async on a writer thread, retrievals run in parallel with
    # LLM warmup per the parallelization spec. Baseline measures the
    # Whisper -> LLM -> TTS path that's actually on the critical path, so
    # we explicitly do NOT load memory here. ConversationMemory and the
    # embedder have their own dedicated tests (tests/test_memory_qdrant.py)
    # that cover write/read latency under their own budgets.
    print("  Memory module not loaded in baseline (off hot path; tested separately).")
    vram_after_emb = vram_after_stt
    results["vram_mb"]["after_embedder"] = vram_after_emb

    from ultron.llm import LLMEngine
    t = time.monotonic()
    llm = LLMEngine(memory=None)
    print(f"  LLM loaded in {time.monotonic() - t:.1f}s")
    vram_after_llm = vram_used_mb()
    results["vram_mb"]["after_llm"] = vram_after_llm
    print(f"    VRAM: {vram_after_llm} MB (delta {vram_after_llm - vram_after_emb:+d})")

    from ultron.tts import RvcConverter, TextToSpeech
    t = time.monotonic()
    rvc = RvcConverter()
    print(f"  RVC loaded in {time.monotonic() - t:.1f}s")
    vram_after_rvc = vram_used_mb()
    results["vram_mb"]["after_rvc"] = vram_after_rvc
    print(f"    VRAM: {vram_after_rvc} MB (delta {vram_after_rvc - vram_after_llm:+d})")

    tts = TextToSpeech(rvc=rvc)
    tts.warmup()
    vram_loaded = vram_used_mb()
    results["vram_mb"]["full_stack_loaded"] = vram_loaded
    print(
        f"\n[loaded] Full stack VRAM: {vram_loaded} MB "
        f"(delta from before-load:{vram_loaded - vram_idle:+d} MB)"
    )

    # ----- LLM warmup (first call has cold-cache overhead) -----
    # Cancel after first token so the warmup doesn't record a turn into the
    # temp memory. Without this, memory grows past MEMORY_RAG_EXCLUDE_RECENT
    # and retrieve() starts returning snippets, which triggers a separate
    # bug in _build_messages (second system message rejected by Qwen3
    # template). Phase 0 sidesteps that; Phase 3 will rewrite the path.
    print("\nWarming LLM...")
    warm_stream = llm.generate_stream("Say 'ready' and nothing else.")
    for _tok in warm_stream:
        llm.cancel()
        break
    for _ in warm_stream:
        pass

    # ----- Whisper baseline on synthetic 2.5 s sample -----
    print("\nMeasuring Whisper on 2.5s synthetic sample (5 reps)...")
    audio = synthetic_audio()
    stt.transcribe(audio)  # warmup
    whisper_ms: list[float] = []
    for _ in range(5):
        t = time.monotonic()
        stt.transcribe(audio)
        whisper_ms.append((time.monotonic() - t) * 1000)
    whisper_median = statistics.median(whisper_ms)
    results["latency_ms"]["whisper_2_5s_sample"] = {
        "min": min(whisper_ms),
        "median": whisper_median,
        "max": max(whisper_ms),
        "samples": whisper_ms,
    }
    print(f"  median={whisper_median:.0f}ms  range={min(whisper_ms):.0f}-{max(whisper_ms):.0f}ms")

    # ----- 10 representative queries -----
    print(f"\nMeasuring {len(REPRESENTATIVE_QUERIES)} representative queries...")
    per_query: list[dict] = []
    peak_vram = vram_loaded

    for i, query in enumerate(REPRESENTATIVE_QUERIES, 1):
        print(f"\n[{i}/{len(REPRESENTATIVE_QUERIES)}] {query}")

        t0 = time.monotonic()
        stream = llm.generate_stream(query)
        ttft_ms, sentence_done_ms, sentence = first_sentence_of(stream, t0)
        # Cancel and drain so generator's finally() runs cleanly.
        llm.cancel()
        for _ in stream:
            pass

        # Piper + RVC synth on the first sentence -- no playback.
        synth_text = sentence or query
        t_synth = time.monotonic()
        pcm, sr = tts._synthesize(synth_text)  # noqa: SLF001 -- internal API for benchmarking
        tts_synth_ms = (time.monotonic() - t_synth) * 1000

        # Composite estimate: Whisper baseline + LLM TTFT + first-sentence synth.
        ttfa_estimate_ms = whisper_median + ttft_ms + tts_synth_ms

        v = vram_used_mb()
        if v > peak_vram:
            peak_vram = v

        per_query.append({
            "query": query,
            "first_token_ms": ttft_ms,
            "first_sentence_complete_ms": sentence_done_ms,
            "first_sentence_text": sentence[:200],
            "tts_synth_first_sentence_ms": tts_synth_ms,
            "tts_pcm_samples": int(pcm.size),
            "tts_pcm_sr": int(sr),
            "ttfa_estimate_ms": ttfa_estimate_ms,
            "vram_mb_after": v,
        })
        print(
            f"   ttft={ttft_ms:.0f}ms  "
            f"first_sentence={sentence_done_ms:.0f}ms  "
            f"tts_synth={tts_synth_ms:.0f}ms  "
            f"ttfa~={ttfa_estimate_ms:.0f}ms  "
            f"vram={v}MB"
        )

    results["latency_ms"]["per_query"] = per_query
    results["vram_mb"]["peak_under_load"] = peak_vram

    # Aggregates
    ttft = [q["first_token_ms"] for q in per_query]
    synth = [q["tts_synth_first_sentence_ms"] for q in per_query]
    ttfa = [q["ttfa_estimate_ms"] for q in per_query]
    results["latency_ms"]["aggregate"] = {
        "ttft_ms": {"min": min(ttft), "median": statistics.median(ttft), "max": max(ttft)},
        "tts_synth_ms": {"min": min(synth), "median": statistics.median(synth), "max": max(synth)},
        "ttfa_estimate_ms": {"min": min(ttfa), "median": statistics.median(ttfa), "max": max(ttfa)},
    }

    # ----- Cleanup & save -----
    try:
        rvc.close()
    except Exception:
        pass
    try:
        tts.stop()
    except Exception:
        pass

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Saved ->{OUTPUT_PATH}")
    print("=" * 60)
    agg = results["latency_ms"]["aggregate"]
    print(
        f"\nVRAM: before_load={vram_idle} MB, full_loaded={vram_loaded} MB "
        f"(delta {vram_loaded - vram_idle:+d}), peak={peak_vram} MB"
    )
    print(
        f"Latency medians: whisper={whisper_median:.0f} ms  "
        f"ttft={agg['ttft_ms']['median']:.0f} ms  "
        f"tts_synth={agg['tts_synth_ms']['median']:.0f} ms  "
        f"ttfa~={agg['ttfa_estimate_ms']['median']:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
