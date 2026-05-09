"""Self-consistency sampling for high-stakes LLM calls.

Per the self-consistency paper (Wang et al., arXiv 2203.11171), sampling
N diverse reasoning paths at non-zero temperature and taking the most
consistent answer significantly outperforms greedy decoding on
chain-of-thought tasks (GSM8K +17.9 %, SVAMP +11.0 %, AQuA +12.2 %).

This module is the orchestration layer. It is **off by default** and
gated by ``llm.self_consistency.enabled`` so the voice path's hot loop
is byte-for-byte unchanged. When enabled, it is applied only at three
explicitly-marked sites (the projection-driven calls listed in the 4B
plan):

- coding correction-prompt generation
- HYBRID_TASK decomposition (JSON-output)
- pre-flight uncertainty (when initial confidence is borderline)

Each site decides locally whether to opt in via
:func:`should_apply_self_consistency`, which reads the config flag
plus an optional per-site enable.

The voice path is **never** routed through here. Generating 3 samples
per turn would 3× the TTFT cost.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ultron.utils.logging import get_logger

logger = get_logger("llm.self_consistency")


@dataclass
class ConsistencyResult:
    """Outcome of an N-sample self-consistency call.

    ``answer`` is the chosen response (majority-vote winner for
    structured output, mode for free-form text). ``votes`` is the
    distribution across samples — ``votes[answer]`` is the winning
    count. ``samples`` is the raw N responses, kept for audit.
    ``fallback_used`` indicates the orchestration fell back to the
    first sample (e.g. all samples were unparseable).
    """
    answer: str
    votes: Dict[str, int]
    samples: List[str]
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------


def should_apply_self_consistency(call_site: str, cfg: Any = None) -> bool:
    """Return True if self-consistency is enabled for ``call_site``.

    Two-stage gate:
      1. ``llm.self_consistency.enabled`` must be True.
      2. The call-site must not be in
         ``llm.self_consistency.disabled_sites`` (per-site opt-out).
    """
    if cfg is None:
        from ultron.config import get_config
        cfg = get_config()
    sc = cfg.llm.self_consistency
    if not sc.enabled:
        return False
    if call_site in sc.disabled_sites:
        return False
    return True


# ---------------------------------------------------------------------------
# Aggregators — one per output-shape family
# ---------------------------------------------------------------------------


def majority_vote_text(samples: List[str]) -> Tuple[str, Dict[str, int]]:
    """Aggregate free-form text samples by exact-string mode.

    The samples are normalised (whitespace-stripped) before counting.
    Ties resolve by first-occurrence in the input order — reproducible
    given the same sample list.
    """
    cleaned = [s.strip() for s in samples if s and s.strip()]
    if not cleaned:
        return "", {}
    counts = Counter(cleaned)
    # ``most_common`` is stable in 3.7+ — first-occurrence wins ties.
    winner, _ = counts.most_common(1)[0]
    return winner, dict(counts)


def majority_vote_json(samples: List[str]) -> Tuple[Optional[Dict], Dict[str, int]]:
    """Parse each sample as JSON and majority-vote the *normalised*
    serialised form.

    Returns ``(parsed_winner_dict, votes_by_serialised_key)``. Samples
    that fail to parse are ignored. ``parsed_winner_dict`` is ``None``
    if no sample parsed.
    """
    parsed: List[Tuple[str, Any]] = []
    for s in samples:
        try:
            obj = _extract_first_json(s)
            if obj is None:
                continue
            key = json.dumps(obj, sort_keys=True, separators=(",", ":"))
            parsed.append((key, obj))
        except Exception:
            continue
    if not parsed:
        return None, {}
    counts = Counter(k for k, _ in parsed)
    winner_key, _ = counts.most_common(1)[0]
    # Find the dict matching that key.
    for k, obj in parsed:
        if k == winner_key:
            return obj, dict(counts)
    return None, dict(counts)


def majority_vote_label(
    samples: List[str], allowed_labels: List[str],
) -> Tuple[Optional[str], Dict[str, int]]:
    """Find the first ``allowed_labels`` value mentioned in each sample
    and majority-vote across them.

    Used for short-output enums like ``CODING|AUTOMATION|HYBRID|UNCLEAR``
    or ``SEARCH|NO_SEARCH|UNCERTAIN``.
    """
    extracted: List[str] = []
    for s in samples:
        if not s:
            continue
        for label in allowed_labels:
            if re.search(rf"\b{re.escape(label)}\b", s, re.IGNORECASE):
                extracted.append(label.upper())
                break
    if not extracted:
        return None, {}
    counts = Counter(extracted)
    winner, _ = counts.most_common(1)[0]
    return winner, dict(counts)


# ---------------------------------------------------------------------------
# Driver — runs N samples, applies aggregator, returns result
# ---------------------------------------------------------------------------


AggregatorFn = Callable[[List[str]], Tuple[Any, Dict[str, int]]]


def run_self_consistency(
    sampler: Callable[[float], str],
    *,
    n: int = 3,
    temperature: float = 0.8,
    aggregator: Optional[AggregatorFn] = None,
) -> ConsistencyResult:
    """Run ``sampler(temperature)`` N times and aggregate.

    ``sampler`` is a callable that takes a temperature and returns a
    string response (the LLM call to perform). Decoupling the sampler
    keeps this module test-friendly and swappable across runtimes.

    ``aggregator`` defaults to text-mode (``majority_vote_text``).
    """
    n = max(1, int(n))
    samples: List[str] = []
    for _ in range(n):
        try:
            samples.append(sampler(temperature))
        except Exception as e:
            logger.warning("self-consistency sampler failed: %s", e)
            samples.append("")

    agg = aggregator or (lambda s: majority_vote_text(s))
    answer, votes = agg(samples)
    fallback = False
    if answer in (None, "") and samples:
        # All samples unaggregatable — fall back to the first non-empty.
        for s in samples:
            if s:
                answer = s
                fallback = True
                break
    if answer is None:
        answer = ""
    return ConsistencyResult(
        answer=str(answer) if not isinstance(answer, dict) else json.dumps(answer),
        votes=votes,
        samples=samples,
        fallback_used=fallback,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_first_json(text: str) -> Optional[Any]:
    """Find the first ``{...}`` or ``[...]`` block in ``text`` and
    parse it. Strips Qwen3 ``<think>`` blocks first."""
    if not text:
        return None
    text = _THINK_RE.sub("", text).strip()
    start_idx = -1
    end_idx = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start_idx = i
            break
    if start_idx < 0:
        return None
    # Find matching close — naive bracket-balancing.
    open_ch = text[start_idx]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx < 0:
        return None
    blob = text[start_idx:end_idx + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


__all__ = [
    "ConsistencyResult",
    "should_apply_self_consistency",
    "majority_vote_text",
    "majority_vote_json",
    "majority_vote_label",
    "run_self_consistency",
]
