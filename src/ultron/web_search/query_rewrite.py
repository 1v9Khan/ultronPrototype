"""Pre-search query reformulation (catalog 12 -- clawhub-felo-search T1).

Felo's API returns a ``query_analysis`` array -- the reformulated sub-queries
it derived server-side before searching -- which is evidence that decomposing
a complex question into several targeted queries improves recall. Ultron
previously forwarded the user's utterance (or the preflight's queries)
straight to the provider chain. This module adds an in-process reformulation
step over the FREE local-first ladder:

  * **Rule-based expansion** (default, zero-cost): structural rewrites that
    fire only on recognised shapes -- ``"X vs Y"`` -> two balanced queries;
    ``"how to X"`` -> ``"X tutorial"`` / ``"X guide"``; ``"best X"`` ->
    ``"X review"`` / ``"X comparison"``; a leading temporal qualifier
    (``"latest X"``) -> the bare subject. Pure regex; no LLM; no latency.
  * **LLM-based expansion** (opt-in via ``web_search.query_reformulation.use_llm``):
    one short in-process Qwen call (~150-250 ms, only on the SEARCH path
    which already pays a network round-trip) that decomposes the question
    into up to ``max_variants`` reformulated queries.

Both paths FAIL OPEN: any error returns no variants so the original query
still searches and the path never breaks. The reformulated variants are
merged into :meth:`ultron.web_search.search.WebSearchExecutor.run`'s existing
query list, deduped by its canonical form, and fanned out through the same
provider chain + URL-dedup + cache. A hard ceiling
(:data:`MAX_TOTAL_QUERIES`) bounds the total fan-out regardless of how many
variants are produced.

The reformulation pattern generalises beyond web search (memory RAG,
codebase exploration, UI discovery); the cross-system consumers live in the
catalog-12 deep-loop modules (batch E) -- this module is the web-search
surface + the reusable :func:`expand_query_rules` / :func:`expand_query_llm`
primitives.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from ultron.utils.logging import get_logger

logger = get_logger("web_search.query_rewrite")

#: Hard ceiling on the total number of queries fanned out per search turn
#: (original + any preflight queries + reformulated variants). Bounds the
#: provider-call count + latency regardless of how many variants the rules
#: or the LLM produce.
MAX_TOTAL_QUERIES = 5

#: Default cap on reformulated variants added for a single query.
DEFAULT_MAX_VARIANTS = 2


@dataclass(frozen=True)
class QueryReformulation:
    """Result of reformulating one query.

    Attributes:
        original: the query that was reformulated (stripped).
        variants: reformulated alternatives, NOT including the original.
        method: ``"rules"`` | ``"llm"`` | ``"none"`` -- which path produced
            the variants (``"none"`` means no variants were produced).
    """

    original: str
    variants: tuple[str, ...] = ()
    method: str = "none"

    @property
    def all_queries(self) -> List[str]:
        """Original first, then variants; deduped case-insensitively with
        order preserved. This is the list a caller fans out."""
        out: List[str] = []
        seen: set[str] = set()
        for q in (self.original, *self.variants):
            qs = (q or "").strip()
            key = qs.lower()
            if not qs or key in seen:
                continue
            seen.add(key)
            out.append(qs)
        return out


# ---------------------------------------------------------------------------
# Rule-based expansion (zero-cost default)
# ---------------------------------------------------------------------------

# "X vs Y" / "X versus Y". Non-greedy left so the FIRST " vs " splits.
_VS_RE = re.compile(r"^(.*?)\s+(?:vs\.?|versus)\s+(.+)$", re.IGNORECASE)
# "how to X" -> isolate X.
_HOWTO_RE = re.compile(r"^\s*how\s+to\s+(.+?)\s*\??$", re.IGNORECASE)
# "best / top / recommended X" -> isolate X.
_BEST_RE = re.compile(
    r"\b(?:best|top|recommended)\s+(.+?)\s*\??$", re.IGNORECASE
)
# Leading temporal qualifier -> bare subject for broader (non-time-boxed)
# recall alongside the time-sensitive original.
_LEADING_TEMPORAL_RE = re.compile(
    r"^\s*(?:the\s+)?(?:latest|newest|recent|current|most\s+recent)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)


def _split_comparison(query: str) -> List[str]:
    """Split ``"X vs Y [tail]"`` into two balanced queries.

    The right side often carries a shared tail (``"Go for backend
    services"``) while the left side is the bare first subject
    (``"Python"``). When that shape is detected the tail is grafted onto
    the left subject so both queries are balanced::

        "Python vs Go for backend services"
          -> ["Python for backend services", "Go for backend services"]
        "cats vs dogs"
          -> ["cats", "dogs"]

    Returns an empty list when no comparison shape is present.
    """
    m = _VS_RE.match(query.strip())
    if not m:
        return []
    left, right = m.group(1).strip(), m.group(2).strip()
    if not left or not right:
        return []
    right_tokens = right.split()
    if len(right_tokens) > 1 and len(left.split()) <= 2:
        tail = " ".join(right_tokens[1:])
        variants = [f"{left} {tail}".strip(), right]
    else:
        variants = [left, right]
    return [v for v in variants if v]


def expand_query_rules(
    query: str, *, max_variants: int = DEFAULT_MAX_VARIANTS
) -> List[str]:
    """Structural, zero-cost query expansion.

    Applies the recognised-shape rewrites and returns up to
    ``max_variants`` variants, deduped against the original and each
    other (case-insensitive, order-preserving). Returns an empty list
    when no rule matches or ``max_variants <= 0``.
    """
    q = (query or "").strip()
    if not q or max_variants <= 0:
        return []

    candidates: List[str] = []
    candidates.extend(_split_comparison(q))

    m = _HOWTO_RE.match(q)
    if m:
        subj = m.group(1).strip().rstrip("?.")
        if subj:
            candidates.append(f"{subj} tutorial")
            candidates.append(f"{subj} guide")

    if not _VS_RE.match(q):  # "best X vs Y" already handled by the split
        m = _BEST_RE.search(q)
        if m:
            subj = m.group(1).strip().rstrip("?.")
            if subj and "practice" not in subj.lower():
                candidates.append(f"{subj} review")
                candidates.append(f"{subj} comparison")

    m = _LEADING_TEMPORAL_RE.match(q)
    if m:
        subj = m.group(1).strip().rstrip("?.")
        if subj and subj.lower() != q.lower():
            candidates.append(subj)

    return _dedupe_against(q, candidates, max_variants)


# ---------------------------------------------------------------------------
# LLM-based expansion (opt-in)
# ---------------------------------------------------------------------------

_LLM_PROMPT = """Rewrite the user's search question into {n} alternative web-search queries that together cover it better than the original. Decompose multi-part questions into focused parts; make each query short, specific, and keyword-rich. Do NOT repeat the original query verbatim.

Return ONLY a JSON object, no commentary, no markdown fences:
{{"queries": ["query one", "query two"]}}

User question: {query}
"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_queries_json(text: str) -> List[str]:
    """Extract a list of query strings from the LLM response.

    Tolerant of ``<think>`` blocks, markdown fences, and prose preamble.
    Accepts either ``{"queries": [...]}`` / ``{"search_queries": [...]}``
    or a bare JSON array. Returns ``[]`` on any parse failure.
    """
    if not text:
        return []
    text = _THINK_RE.sub("", text).strip()

    candidates: List[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    i = text.find("{")
    if i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[i : j + 1])
                    break
    candidates.append(text)

    for c in candidates:
        try:
            v = json.loads(c)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(v, dict):
            qs = v.get("queries") or v.get("search_queries") or []
        elif isinstance(v, list):
            qs = v
        else:
            continue
        return [str(x).strip() for x in qs if isinstance(x, str) and str(x).strip()]
    return []


def expand_query_llm(
    query: str, llm, *, max_variants: int = DEFAULT_MAX_VARIANTS
) -> List[str]:
    """Reformulate ``query`` via one short in-process LLM call.

    Uses the same ``/no_think`` marker convention as the web-gate
    preflight (llama-cpp-python 0.3.22 rejects ``chat_template_kwargs``).
    FAIL-OPEN: returns ``[]`` on a missing engine, an LLM error, or an
    unparseable response, so the caller falls back to rules / the
    original query. Returns up to ``max_variants`` variants, deduped
    against the original.
    """
    q = (query or "").strip()
    if not q or max_variants <= 0 or llm is None:
        return []
    try:
        prompt = _LLM_PROMPT.format(n=max_variants, query=q)
        user_msg = prompt if "/no_think" in prompt else prompt.rstrip() + " /no_think"
        out = llm._llm.create_chat_completion(  # noqa: SLF001
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=128,
        )
        raw = (out["choices"][0]["message"]["content"] or "").strip()
        try:
            from ultron.llm.inference import strip_thinking_text

            raw = strip_thinking_text(raw).strip()
        except Exception:  # noqa: BLE001
            pass
        variants = _parse_queries_json(raw)
    except Exception as e:  # noqa: BLE001
        logger.debug("LLM query reformulation failed (%s); rule-based only", e)
        return []
    return _dedupe_against(q, variants, max_variants)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _dedupe_against(
    original: str, candidates: List[str], max_variants: int
) -> List[str]:
    """Drop blanks + the original + intra-list dups (case-insensitive),
    preserving first-seen order, capped at ``max_variants``."""
    out: List[str] = []
    seen = {original.strip().lower()}
    for v in candidates:
        vs = (v or "").strip()
        key = vs.lower()
        if not vs or key in seen:
            continue
        seen.add(key)
        out.append(vs)
        if len(out) >= max_variants:
            break
    return out


def reformulate_query(
    query: str,
    *,
    use_llm: bool = False,
    llm=None,
    max_variants: int = DEFAULT_MAX_VARIANTS,
    enabled: bool = True,
) -> QueryReformulation:
    """Reformulate ``query`` into targeted variants.

    When ``use_llm`` is True and an ``llm`` is supplied, the LLM path runs
    first; if it produces nothing (error / empty), we fall back to the
    rule-based path so reformulation degrades gracefully rather than
    vanishing. With ``use_llm=False`` only the rule-based path runs.
    Disabled or empty input yields ``method="none"`` with no variants.
    """
    q = (query or "").strip()
    if not enabled or not q:
        return QueryReformulation(original=q, variants=(), method="none")

    if use_llm and llm is not None:
        variants = expand_query_llm(q, llm, max_variants=max_variants)
        if variants:
            return QueryReformulation(original=q, variants=tuple(variants), method="llm")
        # Fall through to rules (fail-open): an LLM hiccup shouldn't drop
        # reformulation entirely when a cheap structural rewrite exists.

    variants = expand_query_rules(q, max_variants=max_variants)
    return QueryReformulation(
        original=q,
        variants=tuple(variants),
        method="rules" if variants else "none",
    )


def maybe_reformulate_queries(
    user_query: str,
    base_queries: Optional[List[str]],
    *,
    llm=None,
) -> List[str]:
    """Executor-facing helper: merge reformulated variants into the query list.

    Reads ``web_search.query_reformulation`` config. When enabled, the
    PRIMARY ``user_query`` is reformulated and its variants are appended to
    ``base_queries`` (which already contains the original / preflight
    queries), deduped case-insensitively and capped at
    :data:`MAX_TOTAL_QUERIES`. When disabled or on any failure, returns
    ``base_queries`` unchanged (or ``[user_query]`` when empty) so the
    search path is never broken. Logs reformulations to
    ``logs/search_reformulations.jsonl`` (best-effort).
    """
    base = [q.strip() for q in (base_queries or []) if q and q.strip()]
    fallback = base or ([user_query.strip()] if (user_query or "").strip() else [])

    try:
        from ultron.config import get_config

        cfg = get_config().web_search.query_reformulation
    except Exception:  # noqa: BLE001
        return fallback
    if not getattr(cfg, "enabled", False):
        return fallback

    try:
        reform = reformulate_query(
            user_query,
            use_llm=bool(getattr(cfg, "use_llm", False)),
            llm=llm,
            max_variants=int(getattr(cfg, "max_variants", DEFAULT_MAX_VARIANTS)),
            enabled=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("query reformulation skipped (%s)", e)
        return fallback

    if not reform.variants:
        return fallback

    merged: List[str] = list(base)
    seen = {q.lower() for q in merged}
    for v in reform.variants:
        if len(merged) >= MAX_TOTAL_QUERIES:
            break
        if v.lower() not in seen:
            merged.append(v)
            seen.add(v.lower())

    _log_reformulation(user_query, reform, merged)
    return merged


def _log_reformulation(
    user_query: str, reform: QueryReformulation, fanned_out: List[str]
) -> None:
    """Append one JSONL row to ``logs/search_reformulations.jsonl`` for
    offline tuning. Best-effort: never raises into the search path."""
    try:
        from ultron.config import LOGS_DIR

        path = LOGS_DIR / "search_reformulations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "query": user_query,
            "method": reform.method,
            "variants": list(reform.variants),
            "fanned_out": fanned_out,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "MAX_TOTAL_QUERIES",
    "DEFAULT_MAX_VARIANTS",
    "QueryReformulation",
    "expand_query_rules",
    "expand_query_llm",
    "reformulate_query",
    "maybe_reformulate_queries",
]
