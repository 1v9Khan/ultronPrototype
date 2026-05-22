"""Cross-encoder reranker for memory retrieval (frontier item 2).

Sits between :class:`~ultron.memory.qdrant_store.ConversationMemory`'s
dense + sparse hybrid retrieval and the final top-k selection. The
hybrid retriever pulls a wider candidate set (e.g., top-20); the
reranker scores each ``(query, candidate.content)`` pair directly
with a neural cross-encoder and produces a more accurate final
ranking.

Why a separate module:
- Cosine + RRF + recency composite (the current pre-rerank ranking
  signal in ``_retrieve_impl``) is a heuristic blend. A cross-encoder
  evaluates query-document semantic match jointly through a neural
  scorer, which is what industry-standard 2026 RAG pipelines do.
- The reranker is opt-in (``memory.reranking.enabled``, default OFF)
  because the default model (``BAAI/bge-reranker-v2-m3``) is a ~1.1
  GB download. Once you've run ``scripts/download_models.py`` and
  flipped the flag, the only behaviour change is the order of the
  retrieved top-k.

Cost model:
- Lazy-loaded; first ``rerank`` call pays ~1-3 s load. Subsequent
  calls are ~20-50 ms for ~20 candidates on CPU.
- Zero VRAM by default (``device="cpu"``). Move to GPU only if you've
  proven CPU is the latency bottleneck.

Fail-open contract:
- If model load fails -> log WARN, return pre-rerank order unchanged.
- If predict fails on a specific batch -> log WARN, return pre-rerank
  order unchanged.
- Voice never crashes on reranker issues; degrades to the cosine +
  RRF + recency baseline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, TYPE_CHECKING

from ultron.config import get_config
from ultron.utils.logging import get_logger


if TYPE_CHECKING:
    from ultron.memory.qdrant_store import MemoryTurn

logger = get_logger("memory.reranker")


@dataclass(frozen=True)
class RerankResult:
    """One reranked candidate.

    Carries the raw cross-encoder score alongside the original
    :class:`MemoryTurn` so callers can inspect why an item won/lost
    its position (e.g., debug "why didn't this match make top-k").
    """

    turn: "MemoryTurn"
    score: float
    pre_rerank_index: int


class CrossEncoderReranker:
    """Cross-encoder reranker wrapping :class:`sentence_transformers.CrossEncoder`.

    Args:
        model_name: HuggingFace model id. Defaults to the project's
            ``memory.reranking.model`` config value (typically
            ``BAAI/bge-reranker-v2-m3``).
        device: ``"cpu"`` (default), ``"cuda"``, or any torch device
            string. CPU is correct for typical candidate counts
            (~20); only switch to CUDA if you've measured CPU as
            the bottleneck AND the voice-path VRAM headroom permits.
        max_length: Max tokens per ``(query, candidate)`` pair. The
            cross-encoder truncates longer pairs. Default 512.
        eager: Load the model at construction. Default False so the
            first ``rerank`` call triggers the load; pass True from
            startup hooks where load latency is acceptable.

    The class is thread-safe via an internal lock on model load.
    Once loaded, ``predict`` is the cross-encoder's responsibility
    to handle concurrency (sentence-transformers' default behaviour
    is correct).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        eager: bool = False,
    ) -> None:
        cfg = get_config().memory.reranking
        self.model_name = model_name or cfg.model
        self.device = device or cfg.device
        self.max_length = max_length or int(cfg.max_length)
        self._model = None
        self._load_failed = False
        self._lock = threading.Lock()
        if eager:
            self._ensure_model()

    def _ensure_model(self) -> bool:
        """Load the cross-encoder lazily; cache + idempotent.

        Returns True if the model is available after this call,
        False if the load failed (in which case the caller falls
        back to the pre-rerank order).
        """
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            t0 = time.monotonic()
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                )
                logger.info(
                    "CrossEncoderReranker loaded: %s (device=%s) in %.2fs",
                    self.model_name, self.device, time.monotonic() - t0,
                )
                return True
            except Exception as e:                                     # noqa: BLE001
                self._load_failed = True
                logger.warning(
                    "CrossEncoderReranker load failed (%s); "
                    "retrieval will fall back to pre-rerank order.", e,
                )
                return False

    def rerank(
        self,
        query: str,
        candidates: Sequence["MemoryTurn"],
        top_k: int,
    ) -> List["MemoryTurn"]:
        """Rerank ``candidates`` for ``query`` and return the top
        ``top_k`` ordered by cross-encoder score (highest first).

        Fail-open: empty query OR empty candidates OR model load
        failure OR predict failure all return ``list(candidates)[:top_k]``
        (the pre-rerank order, truncated to top_k). Never raises.
        """
        if not candidates or top_k <= 0 or not (query and query.strip()):
            return list(candidates)[:max(0, top_k)]
        if not self._ensure_model():
            return list(candidates)[:top_k]
        try:
            pairs = [(query, str(c.content or "")) for c in candidates]
            scores = self._model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            ranked = sorted(
                zip(scores, range(len(candidates)), candidates),
                key=lambda row: float(row[0]),
                reverse=True,
            )
            return [c for _, _, c in ranked[:top_k]]
        except Exception as e:                                         # noqa: BLE001
            logger.warning(
                "CrossEncoderReranker.predict failed (%s); "
                "using pre-rerank order.", e,
            )
            return list(candidates)[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        candidates: Sequence["MemoryTurn"],
        top_k: int,
    ) -> List[RerankResult]:
        """Same as :meth:`rerank` but returns :class:`RerankResult`
        instances carrying the cross-encoder score + the original
        candidate index. Useful for debugging + observability.

        Fail-open returns the pre-rerank order with NaN scores so
        callers can distinguish "real score 0.0" from "no rerank
        applied".
        """
        from math import nan
        if not candidates or top_k <= 0 or not (query and query.strip()):
            return [
                RerankResult(turn=c, score=nan, pre_rerank_index=i)
                for i, c in enumerate(list(candidates)[:max(0, top_k)])
            ]
        if not self._ensure_model():
            return [
                RerankResult(turn=c, score=nan, pre_rerank_index=i)
                for i, c in enumerate(list(candidates)[:top_k])
            ]
        try:
            pairs = [(query, str(c.content or "")) for c in candidates]
            scores = self._model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            triples = sorted(
                zip(scores, range(len(candidates)), candidates),
                key=lambda row: float(row[0]),
                reverse=True,
            )
            return [
                RerankResult(turn=c, score=float(s), pre_rerank_index=i)
                for s, i, c in triples[:top_k]
            ]
        except Exception as e:                                         # noqa: BLE001
            logger.warning(
                "CrossEncoderReranker.predict failed (%s); "
                "using pre-rerank order with NaN scores.", e,
            )
            return [
                RerankResult(turn=c, score=nan, pre_rerank_index=i)
                for i, c in enumerate(list(candidates)[:top_k])
            ]


# ---------------------------------------------------------------------------
# Process-wide singleton (2026-05-22)
# ---------------------------------------------------------------------------
# Both ``ConversationMemory._apply_reranker`` and the web-search snippet
# ranker construct their own ``CrossEncoderReranker`` -- meaning the
# ~1.1 GB ``BAAI/bge-reranker-v2-m3`` model loads TWICE on a turn that
# does both a memory retrieve and a web search (each cold load takes
# ~2 s on CPU, so ~4 s of duplicate startup overhead). This factory
# provides a process-wide singleton both call sites should use.

_SHARED_RERANKER: Optional[CrossEncoderReranker] = None
_SHARED_LOCK = threading.Lock()


def get_shared_reranker() -> CrossEncoderReranker:
    """Return the process-wide :class:`CrossEncoderReranker` singleton.

    Thread-safe. Constructs lazily on first call; subsequent calls
    return the cached instance. The instance itself lazy-loads the
    underlying cross-encoder model on first :meth:`rerank` call.
    """
    global _SHARED_RERANKER
    if _SHARED_RERANKER is not None:
        return _SHARED_RERANKER
    with _SHARED_LOCK:
        if _SHARED_RERANKER is None:
            _SHARED_RERANKER = CrossEncoderReranker()
        return _SHARED_RERANKER


def reset_shared_reranker() -> None:
    """Test-only: drop the cached singleton so the next
    :func:`get_shared_reranker` call constructs fresh."""
    global _SHARED_RERANKER
    with _SHARED_LOCK:
        _SHARED_RERANKER = None


__all__ = [
    "CrossEncoderReranker",
    "RerankResult",
    "get_shared_reranker",
    "reset_shared_reranker",
]
