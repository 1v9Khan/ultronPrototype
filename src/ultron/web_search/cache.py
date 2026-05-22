"""Per-query cache for web-search results, backed by the ``web_results``
Qdrant collection.

Cache key: the exact search query string. Values: the list of
:class:`SearchResult` rows plus any Jina-fetched ``full_text`` per URL.
Freshness is per-record so we can keep volatile categories (sports,
weather) shorter than stable ones (history, definitions).

The :func:`cleanup_web_cache` maintenance task already prunes stale
points -- we just have to write the right ``fetched_at`` and
``freshness_category`` values here.
"""

from __future__ import annotations

import time
import uuid
from typing import Iterable, List, Optional, Tuple

from ultron.config import get_config
from ultron.utils.logging import get_logger
from ultron.web_search.brave import SearchResult

logger = get_logger("web_search.cache")


# Heuristic: queries containing any of these tokens get the volatile TTL.
_VOLATILE_KEYWORDS = (
    "weather", "forecast", "temperature", "rain", "snow",
    "stock", "price", "ticker", "share",
    "score", "scores", "standings", "playoff", "playoffs",
    "election", "polls", "polling",
    "flight", "delay",
    "exchange rate", "currency", "fx",
    "live", "right now", "today", "tonight", "now",
    "breaking", "headline",
)


def freshness_category_for(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in _VOLATILE_KEYWORDS):
        return "volatile"
    return "stable"


def ttl_for(freshness_category: str) -> int:
    cache_cfg = get_config().web_search.cache
    if freshness_category == "volatile":
        return cache_cfg.ttl_volatile_seconds
    return cache_cfg.ttl_stable_seconds


# ---------------------------------------------------------------------------
# Lookup / store via Qdrant
# ---------------------------------------------------------------------------


class WebResultsCache:
    """Read/write helper for the ``web_results`` Qdrant collection.

    The cache is intentionally simple: per-query entries that expire by
    timestamp, no semantic dedupe across queries (a separate Phase-3
    maintenance task handles cross-query consolidation if needed).

    Args:
        client: an open ``QdrantClient`` (the orchestrator's existing one).
        embedder: a :class:`HybridEmbedder` so we embed snippet content for
            stored points (lets the maintenance script find related results
            cross-query later).
    """

    def __init__(self, client, embedder) -> None:
        self._client = client
        self._embedder = embedder

    # --- lookup -------------------------------------------------------------

    def lookup(self, query: str) -> Optional[List[Tuple[SearchResult, Optional[str]]]]:
        """Return cached ``(SearchResult, full_text_or_None)`` rows if a fresh
        entry for ``query`` exists; ``None`` otherwise.
        """
        query = (query or "").strip()
        if not query:
            return None

        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
        )

        try:
            points, _ = self._client.scroll(
                collection_name=get_config().qdrant.collections.web_results,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="query",
                        match=MatchValue(value=query),
                    )],
                ),
                limit=20,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.debug("cache lookup scroll failed: %s -- treating as miss", e)
            return None

        if not points:
            return None

        now = time.time()
        rows: List[Tuple[int, SearchResult, Optional[str]]] = []
        for pt in points:
            pl = pt.payload or {}
            fetched_at = float(pl.get("fetched_at", 0.0))
            ttl = ttl_for(str(pl.get("freshness_category", "stable")))
            if (now - fetched_at) > ttl:
                continue
            rank = int(pl.get("rank", 0))
            rows.append((rank, SearchResult(
                url=str(pl.get("url", "")),
                title=str(pl.get("title", "")),
                snippet=str(pl.get("snippet", "")),
                rank=rank,
            ), pl.get("full_text") or None))

        if not rows:
            return None
        rows.sort(key=lambda r: r[0])
        return [(r[1], r[2]) for r in rows]

    # --- store --------------------------------------------------------------

    def store(
        self,
        query: str,
        rows: Iterable[Tuple[SearchResult, Optional[str]]],
    ) -> int:
        """Upsert each (result, full_text_or_None) row into web_results.

        Idempotent on (query, url): re-storing replaces the previous record.
        Returns the count of points written.
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
            PointStruct,
            SparseVector,
        )

        query = (query or "").strip()
        if not query:
            return 0
        rows = list(rows)
        if not rows:
            return 0

        now = time.time()
        category = freshness_category_for(query)

        # Replace any existing points for this exact (query, url) pair.
        try:
            existing, _ = self._client.scroll(
                collection_name=get_config().qdrant.collections.web_results,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="query",
                        match=MatchValue(value=query),
                    )],
                ),
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            old_ids = [pt.id for pt in existing]
            if old_ids:
                from qdrant_client.models import PointIdsList
                self._client.delete(
                    collection_name=get_config().qdrant.collections.web_results,
                    points_selector=PointIdsList(points=old_ids),
                )
        except Exception as e:
            logger.debug("pre-store cleanup failed: %s", e)

        # Embed each result's title+snippet so cross-query retrieval works.
        text_for_embed = [
            f"{r.title}\n{r.snippet}" + (f"\n{full_text[:1000]}" if full_text else "")
            for r, full_text in rows
        ]
        try:
            dvecs = self._embedder.encode_dense(text_for_embed)
            svecs = self._embedder.encode_sparse(text_for_embed)
        except Exception as e:
            logger.warning("cache store: embedder failed (%s); skipping write", e)
            return 0

        points = []
        for (r, full_text), dv, sv in zip(rows, dvecs, svecs):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dv.tolist(),
                    "bm25": SparseVector(indices=sv.indices, values=sv.values),
                },
                payload={
                    "query": query,
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "full_text": full_text or "",
                    "rank": int(r.rank),
                    "fetched_at": now,
                    "freshness_category": category,
                },
            ))
        try:
            self._client.upsert(
                collection_name=get_config().qdrant.collections.web_results,
                points=points,
            )
        except Exception as e:
            logger.warning("cache store: upsert failed (%s)", e)
            return 0
        logger.info(
            "cached %d web results for %r (category=%s)",
            len(points), query[:60], category,
        )
        return len(points)
