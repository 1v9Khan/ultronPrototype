"""V1-gap B5: pre-flight LLM benchmark.

Settles V1-spec Part 1.5's open question: should the web-search gate's
pre-flight reasoning pass run on the main Qwen LLM (current behaviour)
or on a smaller dedicated CPU model (the spec's alternative -- e.g.,
``qwen2.5-1.5b-instruct-q4_k_m.gguf``, ``gemma-2-2b-it-q4_k_m.gguf``)?

The script:

  1. Loads the configured candidate models. Main LLM uses the live
     Ultron config; the CPU candidate is loaded into a separate
     ``llama_cpp.Llama`` with ``n_gpu_layers=0`` so no VRAM lands on
     the GPU (CPU-only path the spec asked about).
  2. Issues the existing pre-flight prompt against 30 representative
     queries on each backend.
  3. Records median / p95 / p99 latency and the per-query gate verdict.
  4. Compares verdicts to a hand-labeled ground truth so we can see
     whether the smaller model trades latency for accuracy.
  5. Writes results to ``baselines.json`` under a new
     ``preflight_benchmark`` block AND prints a Markdown summary table.

This is operator-only tooling: it loads heavy models, takes minutes
to run, and isn't part of the test sweep. Run from the main checkout
(``C:\\STC\\ultronPrototype``); the ``--candidate-model`` argument
points at a local GGUF.

Usage:

    python scripts/benchmark_preflight.py
    python scripts/benchmark_preflight.py --candidate-model models/qwen2.5-1.5b-instruct-q4_k_m.gguf
    python scripts/benchmark_preflight.py --skip-main  # only benchmark candidate
    python scripts/benchmark_preflight.py --queries 5  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project bootstrap: lift to repo root so ``ultron`` resolves.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


# ---------------------------------------------------------------------------
# Representative queries with hand-labeled ground truth.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BenchQuery:
    text: str
    expected_search: bool
    category: str  # "time-sensitive" | "factual" | "personal" | "creative" | "ambiguous"


_QUERIES: List[_BenchQuery] = [
    # Time-sensitive (should SEARCH)
    _BenchQuery("What's the weather today in Tampa?", True, "time-sensitive"),
    _BenchQuery("Is my flight on time?", True, "time-sensitive"),
    _BenchQuery("Who won last night's NFL game?", True, "time-sensitive"),
    _BenchQuery("What's the current price of NVDA?", True, "time-sensitive"),
    _BenchQuery("What's the latest version of Python?", True, "time-sensitive"),
    _BenchQuery("Did the Fed cut rates this week?", True, "time-sensitive"),
    # Factual / stable (should NOT SEARCH)
    _BenchQuery("Who was Nikola Tesla?", False, "factual"),
    _BenchQuery("How does a hash table work?", False, "factual"),
    _BenchQuery("Explain the speed of light.", False, "factual"),
    _BenchQuery("What is the boiling point of water?", False, "factual"),
    _BenchQuery("How tall is Mount Everest?", False, "factual"),
    _BenchQuery("What's the difference between TCP and UDP?", False, "factual"),
    # Personal / memory (should NOT SEARCH)
    _BenchQuery("What did I say earlier about the project?", False, "personal"),
    _BenchQuery("Do you remember my favorite coffee?", False, "personal"),
    _BenchQuery("Who am I?", False, "personal"),
    _BenchQuery("What's on my todo list?", False, "personal"),
    _BenchQuery("What was that thing I asked about last week?", False, "personal"),
    # Creative (should NOT SEARCH)
    _BenchQuery("Write me a poem about the rain.", False, "creative"),
    _BenchQuery("Compose a haiku about coding.", False, "creative"),
    _BenchQuery("Brainstorm names for a startup.", False, "creative"),
    _BenchQuery("Draft an email to my landlord.", False, "creative"),
    _BenchQuery("Pretend you're a detective.", False, "creative"),
    # Ambiguous (model needs to decide)
    _BenchQuery("Has the package been delivered?", True, "ambiguous"),
    _BenchQuery("Is it raining?", True, "ambiguous"),
    _BenchQuery("Who is the CEO of Anthropic?", True, "ambiguous"),
    _BenchQuery("What new features did Python 3.13 add?", True, "ambiguous"),
    _BenchQuery("How do I install LangGraph?", True, "ambiguous"),
    _BenchQuery("What movies just came out?", True, "ambiguous"),
    _BenchQuery("Is iOS 17 still the latest?", True, "ambiguous"),
    _BenchQuery("Did anything important happen in tech this morning?", True, "ambiguous"),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class _PerQueryResult:
    text: str
    category: str
    expected_search: bool
    actual_search: Optional[bool]
    knowledge_confidence: Optional[str]
    latency_ms: float
    correct: bool


@dataclass
class _BackendSummary:
    label: str
    samples: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    accuracy: float
    per_query: List[_PerQueryResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def _build_main_llm():
    """Construct the live Ultron LLMEngine."""
    from ultron.llm.inference import LLMEngine

    return LLMEngine(memory=None)


def _build_candidate_llm(model_path: Path, n_ctx: int = 4096):
    """Construct a lightweight CPU-only Llama wrapper for the candidate
    model. Mimics enough of LLMEngine's interface that
    ``classify_by_preflight`` can call ``_llm.create_chat_completion``
    via a small adaptor."""
    from llama_cpp import Llama

    if not model_path.exists():
        raise FileNotFoundError(
            f"candidate model GGUF not found: {model_path}",
        )
    llama = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=0,         # CPU-only; the spec's alternative path
        flash_attn=False,
        verbose=False,
    )

    class _Adapter:
        """Quack-typed: just enough surface for classify_by_preflight."""

        def __init__(self, llama_inst):
            self._llm = llama_inst

    return _Adapter(llama)


def _benchmark_backend(
    label: str,
    llm,
    queries: List[_BenchQuery],
) -> _BackendSummary:
    """Run the existing classify_by_preflight against every query."""
    from ultron.web_search.gating import classify_by_preflight

    per_query: List[_PerQueryResult] = []
    for q in queries:
        t0 = time.monotonic()
        try:
            verdict = classify_by_preflight(
                llm, q.text, memory_snippets=None,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            actual = verdict.decision.value == "SEARCH"
            confidence = verdict.knowledge_confidence
        except Exception as e:                                       # noqa: BLE001
            elapsed_ms = (time.monotonic() - t0) * 1000
            print(f"  [{label}] {q.text!r} -> ERROR: {e}")
            actual = None
            confidence = None
        per_query.append(_PerQueryResult(
            text=q.text,
            category=q.category,
            expected_search=q.expected_search,
            actual_search=actual,
            knowledge_confidence=confidence,
            latency_ms=elapsed_ms,
            correct=(actual == q.expected_search) if actual is not None else False,
        ))

    latencies = [r.latency_ms for r in per_query if r.actual_search is not None]
    accuracy = (
        sum(1 for r in per_query if r.correct) / max(1, len(per_query))
    )
    if not latencies:
        median = p95 = p99 = 0.0
    else:
        latencies.sort()
        median = statistics.median(latencies)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    return _BackendSummary(
        label=label,
        samples=len(per_query),
        median_ms=float(median),
        p95_ms=float(p95),
        p99_ms=float(p99),
        accuracy=float(accuracy),
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_markdown_summary(summaries: List[_BackendSummary]) -> None:
    print("\n## Pre-flight benchmark summary\n")
    print(
        "| Backend | N | Median | p95 | p99 | Accuracy |\n"
        "|---|---|---|---|---|---|"
    )
    for s in summaries:
        print(
            f"| {s.label} | {s.samples} | {s.median_ms:.0f} ms | "
            f"{s.p95_ms:.0f} ms | {s.p99_ms:.0f} ms | "
            f"{s.accuracy * 100:.1f}% |"
        )

    print("\n### Per-category accuracy\n")
    print("| Backend | time-sensitive | factual | personal | creative | ambiguous |")
    print("|---|---|---|---|---|---|")
    for s in summaries:
        per_cat: Dict[str, Tuple[int, int]] = {}
        for r in s.per_query:
            seen, ok = per_cat.get(r.category, (0, 0))
            per_cat[r.category] = (seen + 1, ok + (1 if r.correct else 0))
        cells = []
        for cat in ("time-sensitive", "factual", "personal", "creative", "ambiguous"):
            seen, ok = per_cat.get(cat, (0, 0))
            cells.append(f"{ok}/{seen}" if seen else "-/-")
        print(f"| {s.label} | " + " | ".join(cells) + " |")


def _write_baseline(
    summaries: List[_BackendSummary],
    *,
    out_path: Path,
) -> None:
    payload = {
        "preflight_benchmark": {
            "timestamp": time.time(),
            "backends": [
                {
                    "label": s.label, "samples": s.samples,
                    "median_ms": round(s.median_ms, 1),
                    "p95_ms": round(s.p95_ms, 1),
                    "p99_ms": round(s.p99_ms, 1),
                    "accuracy": round(s.accuracy, 4),
                    "per_query": [asdict(r) for r in s.per_query],
                }
                for s in summaries
            ],
        }
    }
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            existing.update(payload)
            payload = existing
        except json.JSONDecodeError:
            pass
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBaseline written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--candidate-model",
        type=Path,
        default=None,
        help="Path to a GGUF for the candidate (CPU-only) model. "
             "Skip the candidate benchmark when omitted.",
    )
    p.add_argument(
        "--skip-main", action="store_true",
        help="Skip the main-LLM benchmark (e.g., to compare two CPU candidates).",
    )
    p.add_argument(
        "--queries", type=int, default=0,
        help="Use only the first N queries (default: all). For quick smoke tests.",
    )
    p.add_argument(
        "--baseline", type=Path,
        default=_REPO / "baselines.json",
        help="Where to write/update the preflight_benchmark block.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    queries = _QUERIES if args.queries <= 0 else _QUERIES[: args.queries]

    summaries: List[_BackendSummary] = []

    if not args.skip_main:
        print("Loading main LLM...")
        try:
            main_llm = _build_main_llm()
        except Exception as e:                                       # noqa: BLE001
            print(f"ERROR: failed to construct main LLM: {e}")
            return 2
        print(f"Benchmarking main LLM on {len(queries)} queries...")
        summaries.append(_benchmark_backend("main", main_llm, queries))

    if args.candidate_model is not None:
        try:
            print(f"Loading candidate model {args.candidate_model}...")
            candidate = _build_candidate_llm(args.candidate_model)
        except Exception as e:                                       # noqa: BLE001
            print(f"ERROR: failed to construct candidate model: {e}")
            return 2
        label = f"candidate:{args.candidate_model.name}"
        print(f"Benchmarking {label} on {len(queries)} queries...")
        summaries.append(_benchmark_backend(label, candidate, queries))

    if not summaries:
        print("ERROR: nothing to benchmark (use --candidate-model or omit --skip-main).")
        return 2

    _print_markdown_summary(summaries)
    _write_baseline(summaries, out_path=args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
