"""End-to-end search execution: gate -> Brave -> Jina -> rank -> respond.

The orchestrator calls :meth:`WebSearchExecutor.run` once it has decided
the gate said SEARCH. The executor:

  1. Looks up the query in the Qdrant cache (``web_results`` collection).
     Hit means we skip Brave + Jina entirely.
  2. On miss, calls Brave for each query, dedupes by URL.
  3. Asks the LLM to rank the snippets for relevance to the original
     user question. Top N are kept.
  4. Fetches the top ``max_fetch`` snippets via Jina Reader IN PARALLEL
     for clean markdown extraction. A collective deadline caps the
     total wait; any fetch still in flight at deadline degrades to
     snippet-only. (2026-05-09 latency fix -- this loop was sequential
     before; one slow page could block the entire search path for
     10+ seconds while the TTS playback queue starved.)
  5. Caches the (url, snippet, full_text) bundles into ``web_results``.
  6. Hands back a structured :class:`SearchPayload` ready for the LLM
     prompt-augmentation step.

Failures degrade gracefully:
  - Brave failure -> empty result; caller falls back to base knowledge.
  - Jina failure -> we keep the snippet but skip full extraction.
  - Jina collective-deadline expiry -> abandoned fetches degrade to
    snippet-only; the executor returns immediately so the LLM can
    start streaming with whatever pages did come back in time.
  - Cache write failure -> log and continue; the user-visible flow
    still completes.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ultron.config import get_config
from ultron.utils.logging import get_logger
from ultron.web_search.brave import SearchResult, BraveSearchClient
from ultron.web_search.cache import WebResultsCache
from ultron.web_search.jina import JinaReaderClient

logger = get_logger("web_search.search")


@dataclass
class SearchSource:
    """A single retrieved source ready to feed into the LLM prompt."""

    url: str
    title: str
    snippet: str
    full_text: Optional[str]  # Jina extraction; None means snippet-only
    rank: int  # final rank after LLM re-ranking


@dataclass
class SearchPayload:
    """Result of running a search workflow."""

    query: str
    sources: List[SearchSource]
    cache_hit: bool
    elapsed_ms: float
    notes: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.sources)


# ---------------------------------------------------------------------------
# V1-gap B3: citation marker rendering
# ---------------------------------------------------------------------------


_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _render_inline_marker(index: int, *, fmt: str) -> str:
    """Render an inline citation marker in the configured format.

    ``"bracket"`` (default) -> ``"[1]"`` (terminal-friendly).
    ``"superscript"`` -> ``"¹"`` / ``"²"`` (matches the V1-spec
    Part 4.4 phrasing).

    The references list at the end of a reply always uses bracketed
    numbers regardless of the inline format, so the user can match
    inline markers to the source list even on terminal fonts that
    render Unicode superscripts oddly.
    """
    if fmt == "superscript":
        return str(index).translate(_SUPERSCRIPT_DIGITS)
    return f"[{index}]"


def _resolve_citation_format() -> str:
    try:
        cfg = get_config()
        return getattr(cfg.web_search.citation, "inline_marker_format", "bracket")
    except Exception:
        return "bracket"


# ---------------------------------------------------------------------------
# V1-gap B2: query deduplication
# ---------------------------------------------------------------------------


# Short stopwords to drop during canonicalisation. Conservative list:
# only dropping word-order fillers + possessive markers so "Tampa
# weather today" and "today's weather in Tampa" canonicalise the same.
_DEDUPE_STOPWORDS = frozenset({
    "the", "a", "an", "is", "in", "on", "at", "of", "for", "to",
    "and", "or", "but", "s", "current",
})


def _normalise_search_query(q: str) -> str:
    """Canonicalise ``q`` for dedup purposes.

    Lowercase, strip punctuation (so possessives like "today's" become
    "today" + "s"), drop short word-order stopwords, sort the
    remaining token set. Two queries with the same canonical form are
    treated as duplicates even when their phrasing differs.

    Examples:
      "Tampa weather today" -> "tampa today weather"
      "today's weather in Tampa" -> "tampa today weather"  (same)
    """
    if not q:
        return ""
    cleaned = re.sub(r"['']s\b", "", q.lower())     # strip possessive 's
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    tokens = sorted(
        t for t in cleaned.split()
        if t and t not in _DEDUPE_STOPWORDS
    )
    return " ".join(tokens)


def _dedupe_queries(queries: List[str]) -> List[str]:
    """Drop near-duplicates while preserving first-seen order."""
    seen: set = set()
    out: List[str] = []
    for q in queries:
        canonical = _normalise_search_query(q)
        if not canonical:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Snippet ranking: ask the LLM to pick the most relevant N from Brave's list.
# ---------------------------------------------------------------------------


_RANK_PROMPT = """You are ranking web-search snippets for relevance to a user question. Return a JSON object with a single key "ranked_indices" containing the 1-based indices of the most relevant snippets in priority order. Include ONLY the indices that would help answer the user's actual question; omit irrelevant or duplicate ones.

Return at most {top_n} indices. If none of the snippets seem useful, return an empty list.

Output ONLY the JSON object, no commentary, no markdown.

User question:
{query}

Snippets:
{snippets_block}
"""


# 2026-05-21 frontier: module-level cache for the cross-encoder
# instance. Shared across all WebSearchExecutor turns so we only pay
# the ~1-3 s model load once per process. The instance is constructed
# lazily on first call so the import cost stays out of orchestrator
# startup.
_CROSS_ENCODER_CACHE = None


_CROSS_ENCODER_LOAD_FAILED = object()  # sentinel marker for cached failure


def _get_cross_encoder():
    """Lazy-construct a shared cross-encoder reranker for snippet
    ranking. Returns ``None`` on construction failure (caller falls
    back to provider order). Idempotent + thread-safe via Python's
    GIL semantics for simple attribute reads."""
    global _CROSS_ENCODER_CACHE
    if _CROSS_ENCODER_CACHE is _CROSS_ENCODER_LOAD_FAILED:
        return None
    if _CROSS_ENCODER_CACHE is not None:
        return _CROSS_ENCODER_CACHE
    try:
        from ultron.memory.reranker import CrossEncoderReranker
        _CROSS_ENCODER_CACHE = CrossEncoderReranker()
        return _CROSS_ENCODER_CACHE
    except Exception as e:                                             # noqa: BLE001
        logger.warning(
            "Cross-encoder reranker construction failed (%s); "
            "ranking will fall back to provider order.", e,
        )
        # Sentinel so we don't keep retrying construction every query.
        _CROSS_ENCODER_CACHE = _CROSS_ENCODER_LOAD_FAILED
        return None


def _rank_snippets_cross_encoder(
    query: str,
    results: List[SearchResult],
    top_n: int = 3,
) -> List[SearchResult]:
    """Rank snippets using bge-reranker-v2-m3 cross-encoder.

    ~20-50 ms for 10-20 candidates on CPU vs the LLM path's 500-1500 ms.
    The cross-encoder is purpose-built for query-document relevance
    ranking and matches or beats LLM ranking on standard benchmarks
    for this task.

    Concatenates title + snippet as the candidate text so the model
    has more signal than just the snippet alone. Fail-open: returns
    ``results[:top_n]`` (provider order) on any failure.
    """
    if not results:
        return []
    if len(results) <= top_n:
        return results[:top_n]

    reranker = _get_cross_encoder()
    if reranker is None:
        return results[:top_n]

    try:
        # Trigger lazy model load. ``rerank()`` does this internally
        # but we're bypassing it -- our candidates have ``.snippet``,
        # not ``.content``, so we drop down to direct model.predict().
        if not reranker._ensure_model():                               # noqa: SLF001
            return results[:top_n]
        # Compose the candidate text. Title is short + high-signal;
        # snippet is the actual content preview. URL is excluded --
        # the cross-encoder's training corpus didn't include URLs
        # as query-document context.
        pairs = [
            (query, f"{r.title}. {r.snippet}" if r.title else r.snippet)
            for r in results
        ]
        scores = reranker._model.predict(                              # noqa: SLF001
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Sort indices by descending score, take top_n.
        ranked_indices = sorted(
            range(len(results)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:top_n]
        return [results[i] for i in ranked_indices]
    except Exception as e:                                             # noqa: BLE001
        logger.warning(
            "Cross-encoder ranking failed (%s); falling back to "
            "provider order.", e,
        )
        return results[:top_n]


def _rank_snippets_llm(llm, query: str, results: List[SearchResult], top_n: int = 3) -> List[SearchResult]:
    """Use the LLM to pick the top ``top_n`` results for ``query``.

    Legacy ranking path -- kept for swap-back via
    ``web_search.ranker: "llm"`` in config. ~500-1500 ms per call vs
    the cross-encoder's ~20-50 ms; quality is comparable on average
    but cross-encoder is more consistent (no JSON-parse failures).

    Falls back to provider's native ranking on any failure.
    """
    if not results:
        return []
    if len(results) <= top_n:
        return results[:top_n]

    snippets_block = "\n\n".join(
        f"[{i+1}] {r.title}\n    {r.snippet}\n    ({r.url})"
        for i, r in enumerate(results)
    )
    prompt = _RANK_PROMPT.format(top_n=top_n, query=query, snippets_block=snippets_block)
    try:
        out = llm._llm.create_chat_completion(  # noqa: SLF001
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        raw = (out["choices"][0]["message"]["content"] or "").strip()
        parsed = _parse_rank_response(raw)
        indices = [i - 1 for i in parsed if isinstance(i, int)]
        ranked = [results[i] for i in indices if 0 <= i < len(results)]
        if not ranked:
            return results[:top_n]
        # Trust LLM ordering but cap at top_n.
        return ranked[:top_n]
    except Exception as e:
        logger.warning("snippet ranking failed (%s); using Brave order", e)
        return results[:top_n]


def _rank_snippets(llm, query: str, results: List[SearchResult], top_n: int = 3) -> List[SearchResult]:
    """Dispatch snippet ranking based on ``web_search.ranker`` config.

    - ``"cross_encoder"`` (default): bge-reranker-v2-m3, ~20-50 ms.
    - ``"llm"``: local Qwen with JSON prompt, ~500-1500 ms.
    - ``"none"``: take provider order + slice to top_n, ~0 ms.
    """
    if not results:
        return []
    if len(results) <= top_n:
        return results[:top_n]
    try:
        ranker = get_config().web_search.ranker
    except Exception:                                                  # noqa: BLE001
        ranker = "cross_encoder"
    if ranker == "none":
        return results[:top_n]
    if ranker == "llm":
        return _rank_snippets_llm(llm, query, results, top_n=top_n)
    # Default: cross_encoder
    return _rank_snippets_cross_encoder(query, results, top_n=top_n)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_rank_response(text: str) -> List[int]:
    if not text:
        return []
    text = _THINK_RE.sub("", text).strip()
    candidates = []
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
                    candidates.append(text[i: j + 1])
                    break
    candidates.append(text)

    for c in candidates:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                ids = v.get("ranked_indices") or []
            elif isinstance(v, list):
                ids = v
            else:
                continue
            return [int(x) for x in ids if isinstance(x, (int, float))]
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return []


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class WebSearchExecutor:
    """Coordinator for the search workflow.

    Args:
        brave: a configured :class:`BraveSearchClient`.
        jina: a configured :class:`JinaReaderClient`.
        llm: an :class:`LLMEngine` (used only for ranking).
        cache: a :class:`WebResultsCache` over the ``web_results`` Qdrant
            collection. ``None`` disables caching.
        max_fetch: how many of the ranked snippets get full-text fetches.
    """

    def __init__(
        self,
        brave: BraveSearchClient,
        jina: JinaReaderClient,
        llm,
        cache: Optional[WebResultsCache] = None,
        max_fetch: Optional[int] = None,
        collective_deadline_seconds: Optional[float] = None,
    ) -> None:
        self.brave = brave
        self.jina = jina
        self.llm = llm
        self.cache = cache
        cfg_jina = get_config().web_search.jina
        self.max_fetch = (
            max_fetch
            if max_fetch is not None
            else cfg_jina.max_fetch
        )
        # Collective Jina-fetch deadline (seconds). After this many
        # seconds since the fetch fan-out started, any fetch still in
        # flight is abandoned and its source degrades to snippet-only.
        # Independent of per-fetch timeout. ``None`` -> read from
        # config; ``0.0`` disables (no collective cap; per-fetch
        # timeout is the only ceiling).
        self.collective_deadline_seconds = (
            collective_deadline_seconds
            if collective_deadline_seconds is not None
            else getattr(cfg_jina, "collective_deadline_seconds", 6.0)
        )

    def run(
        self,
        user_query: str,
        search_queries: Optional[List[str]] = None,
        top_n: int = 3,
    ) -> SearchPayload:
        """Run the full search workflow and return a :class:`SearchPayload`.

        Args:
            user_query: the user's original utterance, used for ranking.
            search_queries: queries to issue against Brave. Falls back to
                ``[user_query]`` when not given.
            top_n: how many ranked snippets to keep.
        """
        t0 = time.monotonic()
        notes: List[str] = []
        raw_queries = [q.strip() for q in (search_queries or [user_query]) if q and q.strip()]
        # V1-gap B2: dedupe near-duplicate Brave queries before the
        # fan-out. The pre-flight pass occasionally emits 2-3 queries
        # that share the same canonical-token set ("Tampa weather
        # today" / "weather in Tampa today" / "today weather Tampa").
        # Keep first-seen order so the cache lookup uses the original
        # phrasing the user is most likely to repeat.
        queries = _dedupe_queries(raw_queries)
        if len(queries) != len(raw_queries):
            notes.append(
                f"query_dedup:{len(raw_queries)}->{len(queries)}",
            )
        if not queries:
            return SearchPayload(query=user_query, sources=[], cache_hit=False,
                                 elapsed_ms=0.0, notes=["empty queries"])

        # Cache lookup: try each query (the first hit wins). Skip Brave/Jina
        # entirely when we can.
        if self.cache is not None:
            for q in queries:
                cached = self.cache.lookup(q)
                if cached:
                    notes.append(f"cache hit on {q!r}")
                    sources = [
                        SearchSource(
                            url=r.url, title=r.title, snippet=r.snippet,
                            full_text=full_text, rank=i,
                        )
                        for i, (r, full_text) in enumerate(cached)
                    ]
                    return SearchPayload(
                        query=user_query,
                        sources=sources[:top_n] if sources else [],
                        cache_hit=True,
                        elapsed_ms=(time.monotonic() - t0) * 1000,
                        notes=notes,
                    )

        # Brave fanout. Dedupe by URL; preserve first-seen order.
        all_results: List[SearchResult] = []
        seen_urls: set[str] = set()
        for q in queries:
            try:
                rs = self.brave.search(q)
            except Exception as e:
                logger.warning("Brave call failed for %r: %s", q, e)
                notes.append(f"brave_error:{q!r}")
                continue
            for r in rs:
                if r.url in seen_urls:
                    continue
                seen_urls.add(r.url)
                all_results.append(r)
        if not all_results:
            notes.append("no Brave results")
            return SearchPayload(query=user_query, sources=[], cache_hit=False,
                                 elapsed_ms=(time.monotonic() - t0) * 1000, notes=notes)

        # Rank.
        ranked = _rank_snippets(self.llm, user_query, all_results, top_n=top_n)
        if not ranked:
            ranked = all_results[:top_n]
            notes.append("ranking returned empty; using Brave order")

        # Fetch full content for the top ``max_fetch`` ranked snippets
        # IN PARALLEL with a collective deadline.
        #
        # Pre-2026-05-09: this loop was sequential. A single pathological
        # page (e.g. a slow Quora result at ~10 s) blocked the entire
        # search path while the user heard silence after the ack phrase.
        # Worst observed: 3 sequential fetches summed to 13.5 s; the TTS
        # playback queue starved waiting for tokens.
        #
        # New shape: every targeted URL gets its own daemon fetch thread.
        # ``concurrent.futures.wait`` returns once everything finishes
        # OR the collective deadline elapses. Anything still in flight
        # at deadline is abandoned (the source falls back to
        # snippet-only). Threads continue running in the background and
        # exit naturally on the per-fetch timeout -- ``pool.shutdown(
        # wait=False)`` ensures the executor returns immediately.
        rows: List[Tuple[SearchResult, Optional[str]]] = []
        to_fetch: List[Tuple[int, SearchResult]] = [
            (i, r) for i, r in enumerate(ranked) if i < self.max_fetch
        ]
        fetched: Dict[int, Optional[str]] = {}
        if to_fetch:
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(to_fetch)),
                thread_name_prefix="jina-fetch",
            )
            future_to_index: Dict[concurrent.futures.Future, Tuple[int, str]] = {
                pool.submit(self.jina.fetch, r.url): (i, r.url)
                for i, r in to_fetch
            }
            try:
                deadline = (
                    self.collective_deadline_seconds
                    if self.collective_deadline_seconds > 0
                    else None
                )
                done, not_done = concurrent.futures.wait(
                    future_to_index.keys(),
                    timeout=deadline,
                    return_when=concurrent.futures.ALL_COMPLETED,
                )
                for fut in done:
                    idx, url = future_to_index[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        logger.warning("Jina fetch failed for %s: %s", url, e)
                        notes.append(f"jina_error:{url}")
                        result = None
                    fetched[idx] = result
                    if not result:
                        notes.append(f"snippet_only:{url}")
                for fut in not_done:
                    idx, url = future_to_index[fut]
                    # Don't record a result for not_done futures; they
                    # implicitly degrade to snippet-only via the
                    # ``fetched.get(idx)`` lookup below.
                    notes.append(f"jina_deadline:{url}")
                    notes.append(f"snippet_only:{url}")
                    # Best-effort cancel; in-flight requests can't be
                    # aborted, but cancel() prevents queued futures from
                    # starting.
                    fut.cancel()
            finally:
                # wait=False so the executor doesn't block on slow
                # fetches still finishing in the background. Threads
                # exit on the per-fetch HTTP timeout.
                pool.shutdown(wait=False)
        for i, r in enumerate(ranked):
            rows.append((r, fetched.get(i)))

        # Cache write -- best-effort; failure doesn't block the response.
        if self.cache is not None and rows:
            try:
                # Cache against the FIRST (best) query so subsequent
                # identical queries hit. V1-gap B2 dedupes near-duplicate
                # variants at the query level before this point, so the
                # FIRST query is reliably the canonical one.
                self.cache.store(queries[0], rows)
            except Exception as e:
                logger.warning("cache store failed: %s", e)
                notes.append("cache_write_failed")

        sources = [
            SearchSource(
                url=r.url, title=r.title, snippet=r.snippet,
                full_text=full_text, rank=i,
            )
            for i, (r, full_text) in enumerate(rows)
        ]
        return SearchPayload(
            query=user_query,
            sources=sources,
            cache_hit=False,
            elapsed_ms=(time.monotonic() - t0) * 1000,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# LLM-prompt formatting
# ---------------------------------------------------------------------------


def format_sources_for_prompt(sources: List[SearchSource], max_chars_per_source: int = 1500) -> str:
    """Render sources into a prompt-ready block.

    Each source gets a numbered header (V1-gap B3: inline marker matches
    the configured format -- "bracket" or "superscript"), the URL on
    its own line for citation, the snippet, and (if available) a
    truncated extract from Jina. Truncation keeps the total prompt
    size sane on big articles.

    4B plan Item 4: each source body is optionally compressed when
    ``llm.compression.enabled`` AND ``llm.compression.compress_web``
    are both True. URL + title + numbering are NEVER compressed so
    citations stay accurate. Pass-through when disabled (default).
    """
    if not sources:
        return "(no sources)"
    fmt = _resolve_citation_format()
    blocks: List[str] = []
    for i, s in enumerate(sources, 1):
        marker = _render_inline_marker(i, fmt=fmt)
        body = (s.full_text or s.snippet or "").strip()
        if len(body) > max_chars_per_source:
            body = body[:max_chars_per_source] + "\n[truncated]"
        # Best-effort compression: never break the search path.
        try:
            from ultron.llm.compression import maybe_compress
            body = maybe_compress(body, surface="web")
        except Exception:
            pass
        blocks.append(
            f"{marker} {s.title}\n    URL: {s.url}\n    {body}"
        )
    return "\n\n".join(blocks)


def format_sources_for_transcript(sources: List[SearchSource]) -> str:
    """Render a one-line-per-source list for the visible transcript.

    The orchestrator prints this AFTER the spoken response so the user can
    verify what was consulted. Not sent to TTS.
    """
    if not sources:
        return ""
    lines = ["sources:"]
    for i, s in enumerate(sources, 1):
        title = s.title or s.url
        lines.append(f"  [{i}] {title} -- {s.url}")
    return "\n".join(lines)
