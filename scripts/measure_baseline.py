"""Voice-stack baseline measurement (2026-05-22 rewrite for the current stack).

Loads whichever STT / LLM / TTS engines ``config.yaml`` is currently
configured for — via the production factories
:func:`ultron.transcription.make_stt_engine` and
:func:`ultron.tts.make_tts_engine` — and measures:

  - VRAM at every load checkpoint (idle / +STT / +LLM / +intent / +TTS).
  - STT transcription latency on a synthetic 2.5 s sample (5 reps).
  - LLM time-to-first-token + first-sentence completion across 10
    representative voice queries with ``enable_thinking=False``
    (the voice-path default — saves 5-10 s by skipping the Qwen3.5
    chain-of-thought block).
  - TTS first-sentence synth latency through the configured engine
    (Kokoro / XTTS / Piper). Synth only, no playback.
  - Composite TTFA estimate = STT median + LLM TTFT + TTS first
    sentence.

Output: ``baselines.json`` at the worktree root.

The script intentionally exercises the SAME factories the orchestrator
uses (``ultron.tts.make_tts_engine`` + ``ultron.transcription.make_stt_engine``
+ ``ultron.intent.UltronIntentRecognizer`` for warmup) so a single
``config.yaml`` flip moves both production and measurement in lock-step.

**Voice-stack concurrency:** running this script LOADS the voice
stack. Per ``feedback_voice_stack_concurrency.md`` confirm no other
Ultron process is running before invoking. The runner refuses to
start if it detects another Ultron Python process.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Windows console default is cp1252; reconfigure stdout so any stray
# Unicode in our output (or downstream library logs) doesn't crash on
# print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
WORKTREE_ROOT = Path(__file__).resolve().parent.parent
MAIN_REPO_PATH = Path(r"C:\STC\ultronPrototype")
sys.path.insert(0, str(MAIN_REPO_PATH))
sys.path.insert(0, str(WORKTREE_ROOT / "src"))

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


# ---------------------------------------------------------------------------
# VRAM probe
# ---------------------------------------------------------------------------


def vram_used_mb() -> int:
    """Total VRAM currently in use on GPU 0, in MB.

    Returns 0 when ``nvidia-smi`` is unavailable (CPU-only workstation).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Audio synth
# ---------------------------------------------------------------------------


def synthetic_audio(seconds: float = 2.5, sr: int = 16000) -> np.ndarray:
    """Fixed sine for repeatable STT timing. Content is meaningless;
    only audio length affects model-FLOP timing in any practical sense."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Stream consumer
# ---------------------------------------------------------------------------


def first_sentence_of(
    stream,
    t0: float,
    flush_chars=frozenset(".!?\n"),
) -> Tuple[float, float, str]:
    """Drain ``stream`` until the first sentence terminator. Cancels
    generation afterward so we don't wait for a full response.

    Returns ``(first_token_ms, sentence_complete_ms, sentence_text)``.
    """
    first_token_ms: Optional[float] = None
    sentence_complete_ms: Optional[float] = None
    parts: List[str] = []

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


# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------


def _refuse_if_orchestrator_running() -> Optional[int]:
    """Return another Ultron Python PID if one is detected, else None.

    The ``feedback_voice_stack_concurrency.md`` rule mandates an explicit
    user check before loading the voice stack. This is a runtime
    backstop: if the orchestrator (``python -m ultron``) is already
    running, the load here would steal its CUDA memory and crash. We
    refuse rather than corrupt the running session.
    """
    try:
        import psutil
    except ImportError:
        return None
    cur_pid = __import__("os").getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if proc.info["pid"] == cur_pid:
                continue
            joined = " ".join(cmdline)
            # Identify the orchestrator launch pattern.
            if "ultron" in joined and (
                "-m ultron" in joined or "ultron.__main__" in joined
            ):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


# ---------------------------------------------------------------------------
# Synthesis call (engine-agnostic)
# ---------------------------------------------------------------------------


def _synth_clip(tts_engine, text: str) -> Tuple[np.ndarray, int]:
    """Run a synthesis call against ``tts_engine`` and return ``(pcm, sr)``.

    KokoroSpeech, XttsV3Speech, and TextToSpeech all expose an
    ``_synthesize(text)`` returning ``(np.ndarray, sample_rate)``.
    Calling it directly avoids the playback latency we don't want
    to measure.
    """
    return tts_engine._synthesize(text)  # noqa: SLF001 — intentional benchmark hook


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-intent",
        action="store_true",
        help="Skip the UltronIntentRecognizer warmup (use when "
             "intent.enabled is false in the current config).",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Override the voice-path default (enable_thinking=False) "
             "and measure with the Qwen3.5 chain-of-thought block included.",
    )
    parser.add_argument(
        "--queries-count",
        type=int,
        default=len(REPRESENTATIVE_QUERIES),
        help="Limit the number of representative queries (default: all 10).",
    )
    parser.add_argument(
        "--allow-concurrent",
        action="store_true",
        help="Bypass the orchestrator-already-running check. ONLY use "
             "when you know no other Ultron process exists.",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Voice-stack baseline (current production engines)")
    print("=" * 60)

    other_pid = None if args.allow_concurrent else _refuse_if_orchestrator_running()
    if other_pid is not None:
        print(
            f"ERROR: another Ultron process (PID={other_pid}) is running. "
            f"Stop it first or re-run with --allow-concurrent.",
            file=sys.stderr,
        )
        return 2

    import os
    os.environ.setdefault("ULTRON_LOG_LEVEL", "WARNING")
    from ultron.utils.logging import configure_logging
    configure_logging(level="WARNING")

    from ultron.config import get_config

    cfg = get_config()
    stt_engine_name = getattr(cfg.stt, "engine", "?")
    tts_engine_name = getattr(cfg.tts, "engine", "?")
    intent_enabled = bool(getattr(getattr(cfg, "intent", None), "enabled", False))
    enable_thinking_kwarg = True if args.enable_thinking else False

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "main_repo": str(MAIN_REPO_PATH),
            "queries_count": min(args.queries_count, len(REPRESENTATIVE_QUERIES)),
            "queries": REPRESENTATIVE_QUERIES[: args.queries_count],
            "config": {
                "stt_engine": stt_engine_name,
                "tts_engine": tts_engine_name,
                "intent_enabled": intent_enabled,
                "llm_preset": getattr(cfg.llm, "preset", "?"),
                "llm_n_ctx": getattr(cfg.llm, "n_ctx", None),
                "enable_thinking": enable_thinking_kwarg,
            },
            "notes": (
                "Constructed via make_stt_engine + make_tts_engine "
                "factories so this measurement always reflects whatever "
                "engines config.yaml currently selects. TTS measured "
                "through engine._synthesize without playback. VRAM "
                "is total GPU 0 used (includes desktop apps); deltas "
                "are the meaningful figures."
            ),
        },
        "vram_mb": {},
        "latency_ms": {},
    }

    vram_idle = vram_used_mb()
    results["vram_mb"]["before_load"] = vram_idle
    print(f"\n[before load] GPU 0 VRAM in use: {vram_idle} MB")
    print(
        f"  config: stt={stt_engine_name}  tts={tts_engine_name}  "
        f"intent_enabled={intent_enabled}  enable_thinking={enable_thinking_kwarg}"
    )

    # ----- STT load + VRAM probe -----
    print("\nLoading STT engine...")
    from ultron.transcription import make_stt_engine
    t = time.monotonic()
    stt = make_stt_engine(cfg.stt)
    print(f"  STT loaded in {time.monotonic() - t:.1f}s ({type(stt).__name__})")
    vram_after_stt = vram_used_mb()
    results["vram_mb"]["after_stt"] = vram_after_stt
    print(
        f"    VRAM: {vram_after_stt} MB "
        f"(delta {vram_after_stt - vram_idle:+d})"
    )

    # Memory module is CPU-only (FastEmbed + BM25 + embedded Qdrant) and
    # runs off the hot path (async writer, parallel retrieve during LLM
    # warmup). It has its own dedicated tests (tests/test_memory_qdrant.py
    # et al.). Baseline measures the STT -> LLM -> TTS critical path.
    print("  Memory module not loaded in baseline (off hot path; tested separately).")
    results["vram_mb"]["after_embedder"] = vram_after_stt

    # ----- LLM load + VRAM probe -----
    print("\nLoading LLM...")
    from ultron.llm import LLMEngine
    t = time.monotonic()
    llm = LLMEngine(memory=None)
    print(f"  LLM loaded in {time.monotonic() - t:.1f}s")
    vram_after_llm = vram_used_mb()
    results["vram_mb"]["after_llm"] = vram_after_llm
    print(
        f"    VRAM: {vram_after_llm} MB "
        f"(delta {vram_after_llm - vram_after_stt:+d})"
    )

    # ----- Intent recognizer warmup (when enabled in config) -----
    intent = None
    if intent_enabled and not args.skip_intent:
        print("\nLoading intent recognizer (Gemma-300M, q4)...")
        from ultron.intent.recognizer import UltronIntentRecognizer
        intent_cfg = cfg.intent
        t = time.monotonic()
        intent = UltronIntentRecognizer(
            model_name=getattr(intent_cfg, "model_name", "embeddinggemma-300m"),
            variant=getattr(intent_cfg, "variant", "q4"),
            threshold=getattr(intent_cfg, "threshold", 0.65),
        )
        loaded_ok = intent.ensure_loaded()
        print(
            f"  intent loaded in {time.monotonic() - t:.1f}s "
            f"({'ok' if loaded_ok else 'failed: ' + (intent._load_error or '')})"
        )
        vram_after_intent = vram_used_mb()
        results["vram_mb"]["after_intent"] = vram_after_intent
        print(
            f"    VRAM: {vram_after_intent} MB "
            f"(delta {vram_after_intent - vram_after_llm:+d})"
        )

    # ----- TTS load + warmup + VRAM probe -----
    print("\nLoading TTS engine...")
    from ultron.tts import make_tts_engine
    t = time.monotonic()
    _rvc, tts = make_tts_engine(cfg.tts)
    print(f"  TTS loaded in {time.monotonic() - t:.1f}s ({type(tts).__name__})")
    if hasattr(tts, "warmup"):
        t = time.monotonic()
        tts.warmup("Online.")
        print(f"  TTS warmup completed in {time.monotonic() - t:.1f}s")
    vram_loaded = vram_used_mb()
    results["vram_mb"]["full_stack_loaded"] = vram_loaded
    print(
        f"\n[loaded] Full stack VRAM: {vram_loaded} MB "
        f"(delta from before-load: {vram_loaded - vram_idle:+d} MB)"
    )

    # ----- LLM warmup (cancel after first token) -----
    print("\nWarming LLM (cancel after first token)...")
    warm_stream = llm.generate_stream(
        "Say 'ready' and nothing else.",
        enable_thinking=enable_thinking_kwarg,
    )
    for _tok in warm_stream:
        llm.cancel()
        break
    for _ in warm_stream:
        pass

    # ----- STT baseline -----
    print("\nMeasuring STT on 2.5s synthetic sample (5 reps)...")
    audio = synthetic_audio()
    try:
        stt.transcribe(audio)  # warmup
    except Exception as e:
        print(f"  WARN: STT warmup raised {e!r}; will keep measuring.")
    stt_ms: List[float] = []
    for _ in range(5):
        t = time.monotonic()
        try:
            stt.transcribe(audio)
        except Exception:
            pass
        stt_ms.append((time.monotonic() - t) * 1000)
    stt_median = statistics.median(stt_ms)
    results["latency_ms"]["stt_2_5s_sample"] = {
        "engine": type(stt).__name__,
        "min": min(stt_ms),
        "median": stt_median,
        "max": max(stt_ms),
        "samples": stt_ms,
    }
    print(
        f"  median={stt_median:.0f}ms  "
        f"range={min(stt_ms):.0f}-{max(stt_ms):.0f}ms"
    )

    # ----- LLM + TTS per-query measurement -----
    queries = REPRESENTATIVE_QUERIES[: args.queries_count]
    print(f"\nMeasuring {len(queries)} representative queries...")
    per_query: List[Dict[str, Any]] = []
    peak_vram = vram_loaded

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query}")

        t0 = time.monotonic()
        stream = llm.generate_stream(query, enable_thinking=enable_thinking_kwarg)
        ttft_ms, sentence_done_ms, sentence = first_sentence_of(stream, t0)
        llm.cancel()
        for _ in stream:
            pass

        synth_text = sentence or query
        t_synth = time.monotonic()
        try:
            pcm, sr = _synth_clip(tts, synth_text)
            tts_synth_ms = (time.monotonic() - t_synth) * 1000
            pcm_samples = int(getattr(pcm, "size", 0))
        except Exception as e:
            tts_synth_ms = float("nan")
            pcm_samples = 0
            sr = 0
            print(f"  WARN: TTS synth raised {e!r}; recording NaN.")

        ttfa_estimate_ms = stt_median + ttft_ms + tts_synth_ms

        v = vram_used_mb()
        if v > peak_vram:
            peak_vram = v

        per_query.append({
            "query": query,
            "first_token_ms": ttft_ms,
            "first_sentence_complete_ms": sentence_done_ms,
            "first_sentence_text": sentence[:200],
            "tts_synth_first_sentence_ms": tts_synth_ms,
            "tts_pcm_samples": pcm_samples,
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
    synth = [
        q["tts_synth_first_sentence_ms"] for q in per_query
        if not (isinstance(q["tts_synth_first_sentence_ms"], float)
                and q["tts_synth_first_sentence_ms"] != q["tts_synth_first_sentence_ms"])  # NaN guard
    ]
    ttfa = [
        q["ttfa_estimate_ms"] for q in per_query
        if not (isinstance(q["ttfa_estimate_ms"], float)
                and q["ttfa_estimate_ms"] != q["ttfa_estimate_ms"])
    ]
    results["latency_ms"]["aggregate"] = {
        "ttft_ms": {"min": min(ttft), "median": statistics.median(ttft), "max": max(ttft)},
        "tts_synth_ms": (
            {"min": min(synth), "median": statistics.median(synth), "max": max(synth)}
            if synth else None
        ),
        "ttfa_estimate_ms": (
            {"min": min(ttfa), "median": statistics.median(ttfa), "max": max(ttfa)}
            if ttfa else None
        ),
    }

    # ----- Cleanup -----
    if hasattr(tts, "stop"):
        try:
            tts.stop()
        except Exception:
            pass
    if intent is not None:
        try:
            intent.close()
        except Exception:
            pass

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Saved -> {OUTPUT_PATH}")
    print("=" * 60)
    agg = results["latency_ms"]["aggregate"]
    print(
        f"\nVRAM: before_load={vram_idle} MB, full_loaded={vram_loaded} MB "
        f"(delta {vram_loaded - vram_idle:+d}), peak={peak_vram} MB"
    )
    stt_line = (
        f"stt={stt_median:.0f} ms  ttft={agg['ttft_ms']['median']:.0f} ms"
    )
    if agg.get("tts_synth_ms"):
        stt_line += f"  tts_synth={agg['tts_synth_ms']['median']:.0f} ms"
    if agg.get("ttfa_estimate_ms"):
        stt_line += f"  ttfa~={agg['ttfa_estimate_ms']['median']:.0f} ms"
    print("Latency medians: " + stt_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
