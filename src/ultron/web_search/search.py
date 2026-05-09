"""End-to-end search execution: gate -> Brave -> Jina -> rank -> respond.

The orchestrator calls :meth:`WebSearchExecutor.run` once it has decided
the gate said SEARCH. The executor:

  1. Looks up the query in the Qdrant cache (``web_results`` collection).
     Hit means we skip Brave + Jina entirely.
  2. On miss, calls Brave for each query, dedupes by URL.
  3. Asks the LLM to rank the snippets for relevance to the original
     user question. Top N are kept.
  4. Fetches the top 1-3 via Jina Reader for clean markdown extraction.
  5. Caches the (url, snippet, full_text) bundles into ``web_results``.
  6. Hands back a structured :class:`SearchPayload` ready for the LLM
     prompt-augmentation step.

Failures degrade gracefully:
  - Brave failure -> empty result; caller falls back to base knowledge.
  - Jina failure -> we keep the snippet but skip full extraction.
  - Cache write failure -> log and continue; the user-visible flow
    still completes.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ultron.config import get_config
from ultron.utils.logging import get_logger
from ultron.web_search.brave import BraveResult, BraveSearchClient
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


def _rank_snippets(llm, query: str, results: List[BraveResult], top_n: int = 3) -> List[BraveResult]:
    """Use the LLM to pick the top ``top_n`` results for ``query``.

    Falls back to Brave's native ranking on any failure.
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
    ) -> None:
        self.brave = brave
        self.jina = jina
        self.llm = llm
        self.cache = cache
        self.max_fetch = (
            max_fetch
            if max_fetch is not None
            else get_config().web_search.jina.max_fetch
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
        queries = [q.strip() for q in (search_queries or [user_query]) if q and q.strip()]
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
        all_results: List[BraveResult] = []
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

        # Fetch full content for the top ``max_fetch`` ranked snippets.
        rows: List[Tuple[BraveResult, Optional[str]]] = []
        for i, r in enumerate(ranked):
            full_text = None
            if i < self.max_fetch:
                try:
                    full_text = self.jina.fetch(r.url)
                except Exception as e:
                    logger.warning("Jina fetch failed for %s: %s", r.url, e)
                    notes.append(f"jina_error:{r.url}")
                if not full_text:
                    notes.append(f"snippet_only:{r.url}")
            rows.append((r, full_text))

        # Cache write -- best-effort; failure doesn't block the response.
        if self.cache is not None and rows:
            try:
                # Cache against the FIRST (best) query so subsequent identical
                # queries hit. We don't try to deduplicate per-search-query
                # variants; the simplest cache wins.
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

    Each source gets a numbered header, the URL on its own line for citation,
    the snippet, and (if available) a truncated extract from Jina. Truncation
    keeps the total prompt size sane on big articles.

    4B plan Item 4: each source body is optionally compressed when
    ``llm.compression.enabled`` AND ``llm.compression.compress_web``
    are both True. URL + title + numbering are NEVER compressed so
    citations stay accurate. Pass-through when disabled (default).
    """
    if not sources:
        return "(no sources)"
    blocks: List[str] = []
    for i, s in enumerate(sources, 1):
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
            f"[{i}] {s.title}\n    URL: {s.url}\n    {body}"
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
