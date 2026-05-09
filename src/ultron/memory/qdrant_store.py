"""Qdrant-backed conversation memory with hybrid search.

Architecture (per spec):
- Three collections: ``conversations`` (turn-level), ``facts`` (durable
  extracted statements), ``web_results`` (cached URL fetches, populated by
  Phase 4).
- Each conversation point carries a 384-dim dense bge-small vector + a BM25
  sparse vector. Hybrid retrieval issues a single Qdrant ``query_points``
  call with prefetch on both vectors and Reciprocal Rank Fusion.
- Hot-path write path: append to in-process recent-turns cache + push to a
  background queue. The writer thread embeds + upserts; failures log and
  drop. Worst-case <1 ms on the hot path.
- Hot-path read path: ``recent()`` reads the in-process cache; ``retrieve()``
  hits Qdrant + the embedder (~150 ms cold).

The public surface (``ConversationMemory.add / recent / retrieve / close``)
matches the legacy JSONL store so callers (LLMEngine, orchestrator) don't
need to change.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from ultron.config import get_config, resolve_path
from ultron.errors import QdrantUnavailableError
from ultron.memory.embedder import HybridEmbedder, _SparseVec
from ultron.resilience import get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("memory.qdrant_store")


# ---------------------------------------------------------------------------
# Public data class -- a "turn" returned to callers. We keep the legacy field
# names (id, ts, role, content) so existing call sites don't break, plus the
# Phase 3 fields populated by the maintenance script (summary, entities,
# topic_tags, cluster_id).
# ---------------------------------------------------------------------------


@dataclass
class MemoryTurn:
    id: int
    ts: float
    role: str  # "user" | "assistant"
    content: str
    session_id: str = ""
    summary: str = ""
    entities: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    cluster_id: Optional[int] = None


@dataclass
class FactRow:
    """A row from the ``facts`` collection.

    Populated by :meth:`ConversationMemory.search_facts`. The maintenance
    script (``scripts/maintenance.py:run_extract_facts``) writes the
    underlying Qdrant points; this dataclass is the read-side projection
    callers consume (Coordinator's clarification fast-path, in particular).
    """

    fact: str
    confidence: float
    last_confirmed: float
    category: str
    score: float                          # RRF score from the hybrid query
    extracted_at: float = 0.0
    extracted_from: List[int] = field(default_factory=list)
    retrieval_weight: float = 1.0


# ---------------------------------------------------------------------------
# ConversationMemory: same surface as the JSONL store, Qdrant under the hood.
# ---------------------------------------------------------------------------


class ConversationMemory:
    """Qdrant-backed conversation memory.

    Args:
        path: directory for the embedded Qdrant store. Created if missing.
        embedder: a :class:`HybridEmbedder`. Required for write + retrieve.
        recent_cache_size: how many recent turns to keep in process. Default
            big enough that ``recent(MEMORY_RECENT_TURNS)`` always serves
            from cache; older turns are still searchable via ``retrieve()``.
        session_id: tag for the current run. Lets ``retrieve()`` exclude
            current-session turns from RAG hits if desired.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        embedder: Optional[HybridEmbedder] = None,
        recent_cache_size: int = 100,
        session_id: Optional[str] = None,
    ) -> None:
        if embedder is None:
            raise ValueError(
                "ConversationMemory needs a HybridEmbedder. Pass embedder=HybridEmbedder()."
            )
        cfg = get_config()
        self.path = Path(path) if path is not None else resolve_path(cfg.qdrant.data_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._recent_cache_size = recent_cache_size
        self.session_id = session_id or _new_session_id()

        # Lazy-imported here so a missing qdrant-client install doesn't crash
        # at module import time -- the orchestrator's _load_memory_if_enabled
        # catches the resulting ValueError and disables memory gracefully.
        from qdrant_client import QdrantClient

        self._client = QdrantClient(path=str(self.path))
        self._lock = threading.RLock()

        self._ensure_collections()

        # Recent-turn cache + next-id tracking are warmed from Qdrant.
        self._recent: List[MemoryTurn] = []
        self._next_id: int = 0
        self._load_recent_cache_from_qdrant()

        # Async writer.
        self._write_queue: "queue.Queue[Optional[MemoryTurn]]" = queue.Queue(
            maxsize=cfg.memory.write_queue_maxsize
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="memory-writer"
        )
        self._writer_thread.start()

        logger.info(
            "ConversationMemory ready: %d recent turns cached, session=%s, path=%s",
            len(self._recent), self.session_id, self.path,
        )

    # --- collection bootstrap -----------------------------------------------

    def _ensure_collections(self) -> None:
        """Create the three collections on first run; no-op afterward."""
        from qdrant_client.models import (
            Distance,
            SparseVectorParams,
            VectorParams,
        )

        names = {c.name for c in self._client.get_collections().collections}

        common_dense = {"dense": VectorParams(size=self._embedder.dim, distance=Distance.COSINE)}
        common_sparse = {"bm25": SparseVectorParams()}

        if get_config().qdrant.collections.conversations not in names:
            self._client.create_collection(
                collection_name=get_config().qdrant.collections.conversations,
                vectors_config=common_dense,
                sparse_vectors_config=common_sparse,
            )
            logger.info("Created Qdrant collection %s", get_config().qdrant.collections.conversations)
        if get_config().qdrant.collections.facts not in names:
            self._client.create_collection(
                collection_name=get_config().qdrant.collections.facts,
                vectors_config=common_dense,
                sparse_vectors_config=common_sparse,
            )
            logger.info("Created Qdrant collection %s", get_config().qdrant.collections.facts)
        if get_config().qdrant.collections.web_results not in names:
            self._client.create_collection(
                collection_name=get_config().qdrant.collections.web_results,
                vectors_config=common_dense,
                sparse_vectors_config=common_sparse,
            )
            logger.info("Created Qdrant collection %s", get_config().qdrant.collections.web_results)

    def _load_recent_cache_from_qdrant(self) -> None:
        """Pull the most-recent N turns into the in-process cache + set next id.

        Avoids loading the entire history. Uses Qdrant's ``scroll`` API with
        ordering by id descending, then reverses to chronological order.
        """
        from qdrant_client.models import OrderBy, Direction

        try:
            # Newest-first scan, capped at recent_cache_size. The integer
            # turn-id we store in payload is the source of ordering.
            points, _ = self._client.scroll(
                collection_name=get_config().qdrant.collections.conversations,
                limit=self._recent_cache_size,
                with_payload=True,
                with_vectors=False,
                order_by=OrderBy(key="turn_id", direction=Direction.DESC),
            )
        except Exception as e:
            # An empty / fresh collection can sometimes raise on the
            # ordered-scroll path; fall back to a plain scroll.
            logger.debug("Ordered scroll failed (%s); falling back", e)
            try:
                points, _ = self._client.scroll(
                    collection_name=get_config().qdrant.collections.conversations,
                    limit=self._recent_cache_size,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                points = []

        turns = []
        max_id = -1
        for pt in points:
            payload = pt.payload or {}
            turn_id = int(payload.get("turn_id", 0))
            if turn_id > max_id:
                max_id = turn_id
            turns.append(_payload_to_turn(payload))
        # Sort chronologically (ascending id).
        turns.sort(key=lambda t: t.id)
        self._recent = turns[-self._recent_cache_size:]
        self._next_id = max_id + 1 if max_id >= 0 else 0

    # --- write path ---------------------------------------------------------

    def add(self, role: str, content: str) -> MemoryTurn:
        """Append a turn, return immediately. Persistence is async.

        The hot path:
          * stamps a turn id + timestamp,
          * appends to the in-process recent cache,
          * enqueues the turn for the writer thread.

        On queue overflow we log and drop the new turn rather than block --
        the spec gives us a hard "must not regress latency" budget.
        """
        with self._lock:
            turn = MemoryTurn(
                id=self._next_id,
                ts=time.time(),
                role=role,
                content=content,
                session_id=self.session_id,
            )
            self._next_id += 1
            self._recent.append(turn)
            if len(self._recent) > self._recent_cache_size:
                # Drop oldest entries -- they remain in Qdrant and are still
                # retrievable via the RAG path.
                self._recent = self._recent[-self._recent_cache_size:]
        try:
            self._write_queue.put_nowait(turn)
        except queue.Full:
            logger.warning(
                "Memory writer queue full (%d) -- dropping turn %d. "
                "Hot path stays responsive but this turn won't be RAG-indexed.",
                self._write_queue.maxsize, turn.id,
            )
        return turn

    def _writer_loop(self) -> None:
        """Background thread: drain the queue, embed, upsert into Qdrant."""
        while True:
            try:
                turn = self._write_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if turn is None:  # shutdown sentinel
                return
            try:
                self._upsert_turn(turn)
            except Exception as e:
                logger.warning("Async upsert failed for turn %d: %s", turn.id, e)
            finally:
                self._write_queue.task_done()

    def _upsert_turn(self, turn: MemoryTurn) -> None:
        from qdrant_client.models import PointStruct, SparseVector

        # Embed content as both dense + sparse. The role prefix is identical
        # to the legacy embedder (preserves retrieval behavior) but stripped
        # for BM25 since it'd act as a noisy stop-token.
        text_dense = f"{turn.role}: {turn.content}"
        dvec = self._embedder.encode_dense(text_dense)
        svec = self._embedder.encode_sparse(turn.content)[0]

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dvec.tolist(),
                "bm25": SparseVector(indices=svec.indices, values=svec.values),
            },
            payload={
                "turn_id": turn.id,
                "ts": turn.ts,
                "role": turn.role,
                "content": turn.content,
                "session_id": turn.session_id,
                "summary": turn.summary,
                "entities": turn.entities,
                "topic_tags": turn.topic_tags,
                "cluster_id": turn.cluster_id,
            },
        )
        self._client.upsert(
            collection_name=get_config().qdrant.collections.conversations,
            points=[point],
        )

    # --- read path ----------------------------------------------------------

    def recent(self, n: int) -> List[MemoryTurn]:
        """Return the last ``n`` turns chronologically, served from cache."""
        if n <= 0:
            return []
        with self._lock:
            return list(self._recent[-n:])

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        exclude_recent: Optional[int] = None,
    ) -> List[MemoryTurn]:
        """Top-``k`` turns by hybrid (dense + BM25, RRF-fused), excluding the
        last ``exclude_recent`` turn ids (the recent window the LLM already sees).

        Returns ``[]`` for empty query, empty store, or when everything is in
        the recent window.
        """
        if not query.strip():
            return []
        mem_cfg = get_config().memory
        if k is None:
            k = mem_cfg.rag_top_k
        if exclude_recent is None:
            exclude_recent = mem_cfg.rag_exclude_recent
        with self._lock:
            cutoff_id = max(0, self._next_id - exclude_recent)
        if cutoff_id <= 0:
            # Nothing older than the recent window yet.
            return []

        from qdrant_client.models import (
            FieldCondition,
            Filter,
            Fusion,
            FusionQuery,
            Prefetch,
            Range,
            SparseVector,
        )

        try:
            qdv = self._embedder.encode_query_dense(query)
            qsv: _SparseVec = self._embedder.encode_query_sparse(query)
        except Exception as e:
            logger.warning("Query embedding failed: %s", e)
            get_error_log().record(
                QdrantUnavailableError(
                    f"query embedding failed: {e}",
                    context={"query_len": len(query)},
                    recovery="returned empty retrieval; LLM responds from base knowledge",
                ),
                dependency="qdrant_embedder",
            )
            return []

        # Filter to turn_id < cutoff (the older-than-recent window).
        recency_filter = Filter(
            must=[FieldCondition(key="turn_id", range=Range(lt=cutoff_id))]
        )

        try:
            response = self._client.query_points(
                collection_name=get_config().qdrant.collections.conversations,
                prefetch=[
                    Prefetch(
                        query=qdv.tolist(),
                        using="dense",
                        filter=recency_filter,
                        limit=max(k * 4, 20),
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=qsv.indices, values=qsv.values
                        ),
                        using="bm25",
                        filter=recency_filter,
                        limit=max(k * 4, 20),
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=max(1, k),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning("Qdrant hybrid search failed: %s", e)
            get_error_log().record(
                QdrantUnavailableError(
                    f"hybrid search failed: {e}",
                    context={
                        "query_len": len(query),
                        "k": k,
                        "cutoff_id": cutoff_id,
                    },
                    recovery="returned empty retrieval; LLM responds from base knowledge",
                ),
                dependency="qdrant",
            )
            return []

        return [_payload_to_turn(pt.payload or {}) for pt in response.points]

    # --- V1-gap A2: multi-pass per-category retrieval ---------------------

    def retrieve_multi(
        self,
        primary_query: str,
        category_queries: List[str],
        *,
        k: Optional[int] = None,
        exclude_recent: Optional[int] = None,
    ) -> List[MemoryTurn]:
        """Multi-pass retrieval (V1-gap A2).

        Issues one hybrid Qdrant query per category sub-query (plus the
        primary if it isn't already in the list), unions the results,
        and ranks them via the composite scorer in
        :mod:`ultron.memory.ranking`.

        Args:
            primary_query: literal user utterance.
            category_queries: 2-4 category sub-queries from the gate's
                pre-flight pass. May be empty -- in that case we behave
                identically to :meth:`retrieve` so the orchestrator can
                call this unconditionally.
            k: max final hits.
            exclude_recent: exclude the latest N turns from results
                (anti-echo of the in-process recent cache).

        Returns:
            Up to ``k`` :class:`MemoryTurn` ranked by composite score.
            Empty on any failure (Qdrant down, embedder failure) -- same
            posture as :meth:`retrieve`.
        """
        if not (primary_query or "").strip():
            return []
        cfg = get_config()
        mem_cfg = cfg.memory
        if k is None:
            k = mem_cfg.rag_top_k
        if exclude_recent is None:
            exclude_recent = mem_cfg.rag_exclude_recent
        with self._lock:
            cutoff_id = max(0, self._next_id - exclude_recent)
        if cutoff_id <= 0:
            return []

        retrieval_cfg = mem_cfg.retrieval
        ranking_cfg = mem_cfg.ranking
        # Cap the category fan-out so a runaway pre-flight can't
        # multiply load.
        max_categories = max(0, retrieval_cfg.max_categories_per_query)
        categories = [
            q for q in (category_queries or [])
            if isinstance(q, str) and q.strip()
        ][:max_categories]
        # The primary query always gets a pass so we never miss
        # literal hits.
        all_queries = [primary_query] + [
            q for q in categories if q.strip() != primary_query.strip()
        ]
        if len(all_queries) == 1:
            # No category fan-out -- fall through to single-pass.
            return self.retrieve(primary_query, k=k, exclude_recent=exclude_recent)

        try:
            dense_batch = self._embedder.encode_query_dense_batch(all_queries)
            sparse_batch = self._embedder.encode_query_sparse_batch(all_queries)
        except Exception as e:
            logger.warning("retrieve_multi: query embedding failed: %s", e)
            get_error_log().record(
                QdrantUnavailableError(
                    f"multi-pass query embedding failed: {e}",
                    context={"queries": len(all_queries)},
                    recovery=(
                        "fell through to single-pass retrieve; "
                        "results may be narrower"
                    ),
                ),
                dependency="qdrant_embedder",
            )
            return self.retrieve(primary_query, k=k, exclude_recent=exclude_recent)

        primary_dense = dense_batch[0].tolist()

        from concurrent.futures import ThreadPoolExecutor
        from qdrant_client.models import (
            FieldCondition, Filter, Fusion, FusionQuery, Prefetch, Range,
            SparseVector,
        )

        recency_filter = Filter(
            must=[FieldCondition(key="turn_id", range=Range(lt=cutoff_id))],
        )

        per_query_limit = max(
            k * retrieval_cfg.candidates_per_category_multiplier, 20,
        )
        collection = cfg.qdrant.collections.conversations

        def _query(idx: int):
            try:
                response = self._client.query_points(
                    collection_name=collection,
                    prefetch=[
                        Prefetch(
                            query=dense_batch[idx].tolist(),
                            using="dense",
                            filter=recency_filter,
                            limit=per_query_limit,
                        ),
                        Prefetch(
                            query=SparseVector(
                                indices=sparse_batch[idx].indices,
                                values=sparse_batch[idx].values,
                            ),
                            using="bm25",
                            filter=recency_filter,
                            limit=per_query_limit,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=per_query_limit,
                    with_payload=True,
                    with_vectors=True,
                )
                return idx, list(response.points)
            except Exception as e:
                logger.warning(
                    "retrieve_multi sub-query %d failed: %s", idx, e,
                )
                return idx, []

        # Parallel fan-out. ThreadPoolExecutor caps concurrency at the
        # number of queries; the embedded Qdrant client is thread-safe
        # enough for this read-only fan-out.
        merged: dict[str, "_MergedCandidate"] = {}
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(all_queries))) as pool:
                results = list(pool.map(_query, range(len(all_queries))))
        except Exception as e:
            logger.warning("retrieve_multi fan-out failed: %s", e)
            return self.retrieve(primary_query, k=k, exclude_recent=exclude_recent)

        for idx, points in results:
            for pt in points:
                pid = str(pt.id)
                payload = pt.payload or {}
                rrf_score = float(getattr(pt, "score", 0.0) or 0.0)
                dense = _extract_dense_vector(pt.vector)
                if pid not in merged:
                    merged[pid] = _MergedCandidate(
                        candidate_id=pid,
                        payload=payload,
                        dense=dense,
                        primary_rrf=rrf_score if idx == 0 else 0.0,
                        category_rrf=0.0 if idx == 0 else rrf_score,
                    )
                    continue
                existing = merged[pid]
                if idx == 0:
                    existing.primary_rrf = max(existing.primary_rrf, rrf_score)
                else:
                    existing.category_rrf = max(existing.category_rrf, rrf_score)

        if not merged:
            return []

        from ultron.memory.ranking import (
            CandidateScore,
            RankingWeights,
            select_top_k,
        )

        weights = RankingWeights(
            rrf_weight=ranking_cfg.rrf_weight,
            recency_weight=ranking_cfg.recency_weight,
            recency_half_life_days=ranking_cfg.recency_half_life_days,
            surprise_weight=ranking_cfg.surprise_weight,
            redundancy_weight=ranking_cfg.redundancy_weight,
        )
        candidates: List[CandidateScore] = []
        for m in merged.values():
            # Use the larger of primary / category as the base RRF.
            base = max(m.primary_rrf, m.category_rrf)
            candidates.append(CandidateScore(
                candidate_id=m.candidate_id,
                payload=m.payload,
                rrf_score=base,
                dense=m.dense,
                primary_similarity=m.primary_rrf,
                category_similarity=m.category_rrf,
            ))
        picked = select_top_k(
            candidates, k=max(1, k), weights=weights,
            primary_dense=primary_dense,
        )
        return [_payload_to_turn(p.payload) for p in picked]

    def retrieve_for_query(
        self,
        primary_query: str,
        gate_verdict=None,
        *,
        k: Optional[int] = None,
        exclude_recent: Optional[int] = None,
    ) -> List[MemoryTurn]:
        """Single entry point that routes between single- and multi-pass.

        When ``memory.retrieval.multi_pass_enabled`` is ON AND the gate
        verdict carries category sub-queries, fan out via
        :meth:`retrieve_multi`. Otherwise call :meth:`retrieve`. Calling
        this with ``gate_verdict=None`` is equivalent to calling
        ``retrieve(...)`` directly -- callers that don't have a verdict
        (e.g., the existing RAG path before A2 lands fully) keep working.
        """
        cfg = get_config().memory
        if cfg.retrieval.multi_pass_enabled and gate_verdict is not None:
            categories = list(getattr(gate_verdict, "context_categories", []) or [])
            extra = list(getattr(gate_verdict, "memory_search_queries", []) or [])
            combined = [c for c in categories + extra if c]
            if combined:
                return self.retrieve_multi(
                    primary_query, combined, k=k, exclude_recent=exclude_recent,
                )
        return self.retrieve(
            primary_query, k=k, exclude_recent=exclude_recent,
        )

    # --- facts collection ---------------------------------------------------

    def search_facts(
        self,
        query: str,
        *,
        k: int = 5,
        min_confidence: float = 0.0,
        max_age_days: Optional[float] = None,
    ) -> List[FactRow]:
        """Hybrid (dense + BM25, RRF-fused) search of the ``facts`` collection.

        Args:
            query: free-text question to match against stored facts.
            k: max rows to return.
            min_confidence: drop facts with ``confidence`` below this.
            max_age_days: drop facts whose ``last_confirmed`` is older than
                this many days. ``None`` disables the age cap.

        Returns:
            Newest-first list of :class:`FactRow`. Empty on any failure
            (Qdrant down, embedder down, malformed payload). Failures are
            logged via :class:`ErrorLog`; no exception is propagated to the
            caller — the coordinator must keep working when memory is sick.
        """
        if not (query or "").strip() or k <= 0:
            return []

        from qdrant_client.models import (
            FieldCondition,
            Filter,
            Fusion,
            FusionQuery,
            Prefetch,
            Range,
            SparseVector,
        )

        try:
            qdv = self._embedder.encode_query_dense(query)
            qsv: _SparseVec = self._embedder.encode_query_sparse(query)
        except Exception as e:
            logger.warning("search_facts: query embedding failed: %s", e)
            get_error_log().record(
                QdrantUnavailableError(
                    f"facts query embedding failed: {e}",
                    context={"query_len": len(query)},
                    recovery=(
                        "returned empty facts list; coordinator falls "
                        "through to LLM/escalation"
                    ),
                ),
                dependency="qdrant_embedder",
            )
            return []

        must: List[Any] = []
        if min_confidence > 0:
            must.append(
                FieldCondition(
                    key="confidence",
                    range=Range(gte=float(min_confidence)),
                ),
            )
        if max_age_days is not None and max_age_days > 0:
            cutoff_ts = time.time() - (max_age_days * 86400.0)
            must.append(
                FieldCondition(
                    key="last_confirmed",
                    range=Range(gte=float(cutoff_ts)),
                ),
            )
        flt: Optional[Filter] = Filter(must=must) if must else None

        try:
            response = self._client.query_points(
                collection_name=get_config().qdrant.collections.facts,
                prefetch=[
                    Prefetch(
                        query=qdv.tolist(),
                        using="dense",
                        filter=flt,
                        limit=max(k * 4, 20),
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=qsv.indices, values=qsv.values
                        ),
                        using="bm25",
                        filter=flt,
                        limit=max(k * 4, 20),
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=max(1, k),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning("search_facts: Qdrant query failed: %s", e)
            get_error_log().record(
                QdrantUnavailableError(
                    f"facts hybrid search failed: {e}",
                    context={"query_len": len(query), "k": k},
                    recovery=(
                        "returned empty facts list; coordinator falls "
                        "through to LLM/escalation"
                    ),
                ),
                dependency="qdrant",
            )
            return []

        rows: List[FactRow] = []
        for pt in response.points:
            payload = pt.payload or {}
            try:
                rows.append(
                    FactRow(
                        fact=str(payload.get("fact", "")),
                        confidence=float(payload.get("confidence", 0.0)),
                        last_confirmed=float(
                            payload.get("last_confirmed", 0.0)
                        ),
                        category=str(payload.get("category", "")),
                        score=float(getattr(pt, "score", 0.0) or 0.0),
                        extracted_at=float(payload.get("extracted_at", 0.0)),
                        extracted_from=list(payload.get("extracted_from") or []),
                        retrieval_weight=float(
                            payload.get("retrieval_weight", 1.0)
                        ),
                    )
                )
            except (TypeError, ValueError) as e:
                logger.debug("search_facts: skipping malformed row: %s", e)
                continue
        return rows

    # --- introspection ------------------------------------------------------

    def __len__(self) -> int:
        try:
            return self._client.count(
                collection_name=get_config().qdrant.collections.conversations,
                exact=False,
            ).count
        except Exception:
            return len(self._recent)

    def close(self) -> None:
        """Drain the writer queue and close the Qdrant client."""
        try:
            # Wait for in-flight writes to complete (~ms each, capped).
            self._write_queue.join()
        except Exception:
            pass
        try:
            self._write_queue.put_nowait(None)
        except queue.Full:
            pass
        self._writer_thread.join(timeout=2.0)
        try:
            self._client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class _MergedCandidate:
    """Internal: per-point aggregator for the multi-pass fan-out.

    Tracks the best RRF score the candidate received against the
    primary query and the best score it received against any of the
    category sub-queries. The composite-score helper later prefers
    candidates with strong category match + weak primary match
    (the surprise score).
    """

    candidate_id: str
    payload: dict
    dense: Optional[List[float]]
    primary_rrf: float = 0.0
    category_rrf: float = 0.0


def _extract_dense_vector(vec) -> Optional[List[float]]:
    """Pull the dense vector out of a Qdrant point's ``vector`` field.

    Qdrant returns ``vector`` as a dict keyed by vector name when
    multiple vectors were configured (our ``dense`` + ``bm25`` setup);
    we fish out the dense one. Returns ``None`` when the point came
    back without vectors (``with_vectors=False``).
    """
    if vec is None:
        return None
    if isinstance(vec, dict):
        dense = vec.get("dense")
        if dense is None:
            return None
        return list(dense)
    # Plain list / array form (single-vector collection).
    try:
        return list(vec)
    except TypeError:
        return None


def _payload_to_turn(payload: dict) -> MemoryTurn:
    return MemoryTurn(
        id=int(payload.get("turn_id", 0)),
        ts=float(payload.get("ts", 0.0)),
        role=str(payload.get("role", "")),
        content=str(payload.get("content", "")),
        session_id=str(payload.get("session_id", "")),
        summary=str(payload.get("summary", "")),
        entities=list(payload.get("entities") or []),
        topic_tags=list(payload.get("topic_tags") or []),
        cluster_id=payload.get("cluster_id"),
    )
