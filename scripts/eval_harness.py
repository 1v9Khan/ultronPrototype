"""Ultron eval harness.

Runs labeled queries from a corpus file against Ultron's pure-Python
classifier surfaces (routing, addressing rules, web-search gating
rules), scores outcomes per dimension, writes a JSON report, and exits
non-zero when any per-dimension accuracy gate is below threshold.

The default mode is **classifier-only** -- it never loads the LLM,
Whisper, TTS, RVC, or Qdrant, so it is safe to run without the
voice-stack-concurrency ASK. Heavier modes (LLM preflight, full
retrieval) are scoped for later additions.

Corpus format: JSONL. Each row is one labeled query. Expected-* fields
are optional -- only the dimensions populated on a row contribute to
that dimension's score.

    {
        "id": "stable_string_id",
        "utterance": "what time is it",
        "expected_routing_kind": "conversational",       // optional
        "expected_addressing": "ADDRESSED",              // optional
        "expected_web_gate": "SEARCH",                   // optional
        "has_active_coding_task": false,                 // optional, default false
        "has_pending_clarification": false,              // optional, default false
        "seconds_since_response": 0.0,                   // optional, addressing input
        "tags": ["search", "time_sensitive"],            // optional, for filtering
        "notes": "Human-readable explanation."
    }

Usage::

    python scripts/eval_harness.py
    python scripts/eval_harness.py --corpus tests/eval/corpus.jsonl
    python scripts/eval_harness.py --dimensions routing,addressing
    python scripts/eval_harness.py --filter-tag search --verbose
    python scripts/eval_harness.py --output logs/eval_runs/custom.json

Exit codes: 0 = all gates met, 1 = one or more gates below threshold,
2 = invocation / IO failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The harness is intentionally lazy about imports so a corpus-only
# dry run can succeed even when the ultron package is partly broken.
# Concrete imports happen inside the dimension scorers below.

DEFAULT_CORPUS = PROJECT_ROOT / "tests" / "eval" / "corpus.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "eval_runs"

# Per-dimension accuracy floors. The harness exits non-zero if any
# scored dimension falls below its gate. Empty scored sets short-circuit
# to "passed=True" -- gates only apply when there's data to judge.
DEFAULT_GATES: dict[str, float] = {
    "routing_kind_accuracy": 0.95,
    "addressing_accuracy": 0.90,
    "web_gate_accuracy": 0.90,
}

KNOWN_DIMENSIONS = ("routing", "addressing", "web_gate")


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusRow:
    """One labeled utterance from the corpus file."""

    id: str
    utterance: str
    expected_routing_kind: Optional[str]
    expected_addressing: Optional[str]
    expected_web_gate: Optional[str]
    has_active_coding_task: bool
    has_pending_clarification: bool
    seconds_since_response: float
    tags: tuple[str, ...]
    notes: str


@dataclass
class DimensionScore:
    """Aggregate score for one classifier dimension across the corpus."""

    name: str
    total: int = 0
    correct: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 1.0
        return self.correct / self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "skipped": self.skipped,
            "failures": self.failures,
        }


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def parse_corpus_row(payload: Mapping[str, Any]) -> CorpusRow:
    """Validate + normalize one JSON object into a :class:`CorpusRow`.

    Missing optional fields default to None / False / 0.0 / ().
    Missing required fields (``id``, ``utterance``) raise ``ValueError``.
    """
    if "id" not in payload or not isinstance(payload["id"], str) or not payload["id"]:
        raise ValueError("corpus row missing string 'id'")
    if "utterance" not in payload or not isinstance(payload["utterance"], str):
        raise ValueError(f"corpus row {payload.get('id')!r} missing string 'utterance'")

    tags_raw = payload.get("tags") or ()
    if not isinstance(tags_raw, (list, tuple)):
        raise ValueError(f"corpus row {payload['id']!r} 'tags' must be a list")
    tags = tuple(str(t) for t in tags_raw)

    return CorpusRow(
        id=payload["id"],
        utterance=payload["utterance"],
        expected_routing_kind=_opt_str(payload.get("expected_routing_kind")),
        expected_addressing=_opt_str(payload.get("expected_addressing")),
        expected_web_gate=_opt_str(payload.get("expected_web_gate")),
        has_active_coding_task=bool(payload.get("has_active_coding_task", False)),
        has_pending_clarification=bool(payload.get("has_pending_clarification", False)),
        seconds_since_response=float(payload.get("seconds_since_response", 0.0)),
        tags=tags,
        notes=str(payload.get("notes", "")),
    )


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        return None
    return value


def load_corpus(path: Path) -> list[CorpusRow]:
    """Read JSONL ``path`` and return parsed rows.

    Empty lines and lines whose first non-whitespace char is ``#`` are
    skipped (so the corpus can carry comments if convenient).
    Malformed JSON or invalid row shapes raise ``ValueError`` with the
    line number for easy diagnosis.
    """
    rows: list[CorpusRow] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON ({exc.msg})"
                ) from exc
            try:
                row = parse_corpus_row(payload)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if row.id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate id {row.id!r}")
            seen_ids.add(row.id)
            rows.append(row)
    return rows


def filter_rows(
    rows: Sequence[CorpusRow], *, tag: Optional[str] = None
) -> list[CorpusRow]:
    """Return rows whose ``tags`` include ``tag`` (or all when ``tag`` is None)."""
    if tag is None:
        return list(rows)
    return [r for r in rows if tag in r.tags]


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------


def score_routing(rows: Iterable[CorpusRow]) -> DimensionScore:
    """Score the routing-classifier dimension across ``rows``.

    Rows without ``expected_routing_kind`` are skipped (counted in
    ``DimensionScore.skipped``).
    """
    from ultron.openclaw_routing.classifier import classify_routing

    score = DimensionScore(name="routing")
    for row in rows:
        if row.expected_routing_kind is None:
            score.skipped += 1
            continue
        score.total += 1
        try:
            intent = classify_routing(
                row.utterance,
                has_active_coding_task=row.has_active_coding_task,
                has_pending_clarification=row.has_pending_clarification,
            )
            actual = intent.kind.value
        except Exception as exc:  # noqa: BLE001 -- harness must never crash on a row
            score.failures.append(
                {
                    "id": row.id,
                    "utterance": row.utterance,
                    "expected": row.expected_routing_kind,
                    "actual": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "notes": row.notes,
                }
            )
            continue
        if actual == row.expected_routing_kind:
            score.correct += 1
        else:
            score.failures.append(
                {
                    "id": row.id,
                    "utterance": row.utterance,
                    "expected": row.expected_routing_kind,
                    "actual": actual,
                    "reason": getattr(intent, "reason", None),
                    "source": getattr(intent, "source", None),
                    "confidence": getattr(intent, "confidence", None),
                    "notes": row.notes,
                }
            )
    return score


def score_addressing(rows: Iterable[CorpusRow]) -> DimensionScore:
    """Score the addressing rule classifier across ``rows``.

    ``None`` from the rule classifier (no rule confidently fired) maps
    to the string ``"NONE"`` so the corpus can label "no rule should
    fire; caller should escalate to zero-shot" cases.
    """
    from ultron.addressing.rules import classify as addressing_classify

    score = DimensionScore(name="addressing")
    for row in rows:
        if row.expected_addressing is None:
            score.skipped += 1
            continue
        score.total += 1
        try:
            hit = addressing_classify(row.utterance, row.seconds_since_response)
            actual = hit.decision.value if hit is not None else "NONE"
            confidence = hit.confidence if hit is not None else None
            reason = hit.reason if hit is not None else None
        except Exception as exc:  # noqa: BLE001
            score.failures.append(
                {
                    "id": row.id,
                    "utterance": row.utterance,
                    "expected": row.expected_addressing,
                    "actual": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "notes": row.notes,
                }
            )
            continue
        if actual == row.expected_addressing:
            score.correct += 1
        else:
            score.failures.append(
                {
                    "id": row.id,
                    "utterance": row.utterance,
                    "expected": row.expected_addressing,
                    "actual": actual,
                    "confidence": confidence,
                    "reason": reason,
                    "notes": row.notes,
                }
            )
    return score


def score_web_gate(rows: Iterable[CorpusRow]) -> DimensionScore:
    """Score the web-search rule-classifier across ``rows``.

    Same ``None`` -> ``"NONE"`` convention as the addressing scorer.
    """
    from ultron.web_search.gating import classify_by_rules

    score = DimensionScore(name="web_gate")
    for row in rows:
        if row.expected_web_gate is None:
            score.skipped += 1
            continue
        score.total += 1
        try:
            verdict = classify_by_rules(row.utterance)
            actual = verdict.decision.value if verdict is not None else "NONE"
            reason = verdict.reason if verdict is not None else None
            confidence = verdict.confidence if verdict is not None else None
        except Exception as exc:  # noqa: BLE001
            score.failures.append(
                {
                    "id": row.id,
                    "utterance": row.utterance,
                    "expected": row.expected_web_gate,
                    "actual": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "notes": row.notes,
                }
            )
            continue
        if actual == row.expected_web_gate:
            score.correct += 1
        else:
            score.failures.append(
                {
                    "id": row.id,
                    "utterance": row.utterance,
                    "expected": row.expected_web_gate,
                    "actual": actual,
                    "confidence": confidence,
                    "reason": reason,
                    "notes": row.notes,
                }
            )
    return score


SCORERS = {
    "routing": (score_routing, "routing_kind_accuracy"),
    "addressing": (score_addressing, "addressing_accuracy"),
    "web_gate": (score_web_gate, "web_gate_accuracy"),
}


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    rows: Sequence[CorpusRow],
    dimensions: Sequence[str],
    *,
    gates: Mapping[str, float] = DEFAULT_GATES,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Run the requested ``dimensions`` and assemble a JSON-shaped report.

    Unknown dimensions are silently skipped (the CLI rejects them earlier
    so this only matters for programmatic callers).
    """
    results: dict[str, dict[str, Any]] = {}
    gate_outcomes: dict[str, dict[str, Any]] = {}
    overall_pass = True

    for dim in dimensions:
        scorer_entry = SCORERS.get(dim)
        if scorer_entry is None:
            continue
        scorer, gate_key = scorer_entry
        score = scorer(rows)
        results[dim] = score.as_dict()
        threshold = gates.get(gate_key)
        if threshold is None:
            continue
        passed = score.total == 0 or score.accuracy >= threshold
        gate_outcomes[gate_key] = {
            "threshold": threshold,
            "actual": round(score.accuracy, 4),
            "total_scored": score.total,
            "passed": passed,
        }
        if not passed:
            overall_pass = False

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(rows),
        "dimensions_run": list(dimensions),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "results": results,
        "gates": gate_outcomes,
        "overall_pass": overall_pass,
    }


def format_console_summary(report: Mapping[str, Any], *, verbose: bool = False) -> str:
    """Render a short human-readable summary of a report."""
    lines: list[str] = []
    lines.append(
        f"corpus_size={report['corpus_size']} "
        f"dimensions={','.join(report['dimensions_run'])} "
        f"elapsed={report['elapsed_seconds']}s"
    )
    for dim_name, payload in report["results"].items():
        acc = payload["accuracy"]
        lines.append(
            f"  {dim_name:11s} "
            f"correct={payload['correct']}/{payload['total']} "
            f"acc={acc:.3f} skipped={payload['skipped']} "
            f"failures={len(payload['failures'])}"
        )
        if verbose and payload["failures"]:
            for failure in payload["failures"]:
                lines.append(
                    f"    - [{failure['id']}] {failure['utterance']!r} "
                    f"expected={failure.get('expected')} "
                    f"actual={failure.get('actual')}"
                )
    if report["gates"]:
        lines.append("gates:")
        for gate_name, outcome in report["gates"].items():
            mark = "PASS" if outcome["passed"] else "FAIL"
            lines.append(
                f"  {mark} {gate_name}: "
                f"{outcome['actual']:.3f} vs threshold {outcome['threshold']:.3f}"
            )
    lines.append(f"overall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output_path: Path) -> Path:
    """Write ``report`` as JSON to ``output_path`` (creating parent dirs)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Ultron's classifier eval harness over a labeled corpus. "
            "Classifier-only by default (no voice stack load)."
        )
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"path to the JSONL corpus (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--dimensions",
        default=",".join(KNOWN_DIMENSIONS),
        help=(
            "comma-separated dimensions to evaluate. "
            f"valid values: {','.join(KNOWN_DIMENSIONS)}"
        ),
    )
    parser.add_argument(
        "--filter-tag",
        default=None,
        help="only run corpus rows whose 'tags' include this value",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSON report destination. defaults to "
            "logs/eval_runs/eval_<utc_timestamp>.json"
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="skip writing the JSON report (console only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="list per-failure detail in the console summary",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the console summary (still writes the JSON unless --no-write)",
    )
    return parser


def _parse_dimensions(raw: str) -> list[str]:
    candidates = [c.strip() for c in raw.split(",") if c.strip()]
    unknown = [c for c in candidates if c not in KNOWN_DIMENSIONS]
    if unknown:
        raise SystemExit(
            f"unknown dimensions: {unknown}; valid: {list(KNOWN_DIMENSIONS)}"
        )
    return candidates or list(KNOWN_DIMENSIONS)


def _default_output_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR / f"eval_{ts}.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry. Returns the process exit code (0 pass, 1 gate fail, 2 IO)."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dimensions = _parse_dimensions(args.dimensions)

    if not args.corpus.exists():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    try:
        rows = load_corpus(args.corpus)
    except (ValueError, OSError) as exc:
        print(f"failed to load corpus: {exc}", file=sys.stderr)
        return 2

    filtered = filter_rows(rows, tag=args.filter_tag)
    if not filtered:
        print("no rows after filter; nothing to score", file=sys.stderr)
        return 2

    started = time.perf_counter()
    report = build_report(filtered, dimensions, elapsed_seconds=0.0)
    elapsed = time.perf_counter() - started
    report["elapsed_seconds"] = round(elapsed, 3)

    if not args.no_write:
        output_path = args.output or _default_output_path()
        try:
            written = write_report(report, output_path)
        except OSError as exc:
            print(f"failed to write report: {exc}", file=sys.stderr)
            return 2
        report["output_path"] = str(written)

    if not args.quiet:
        print(format_console_summary(report, verbose=args.verbose))
        if not args.no_write:
            print(f"report written to {report['output_path']}")

    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
