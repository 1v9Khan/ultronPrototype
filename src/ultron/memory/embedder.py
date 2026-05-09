"""Dense + sparse embedder for hybrid retrieval.

- Dense: BAAI/bge-small-en-v1.5 (384 dim, INT8 ONNX via FastEmbed). CPU-only;
  zero new VRAM. Per spec, FastEmbed handles the ONNX INT8 path so we don't
  need to manually export.
- Sparse: Qdrant/bm25 — FastEmbed's pretrained BM25 encoder. Produces
  ``SparseVector`` (indices + values) suitable for direct upsert into Qdrant.

Both models are lazy-loaded on first encode. First call pays a few-hundred-ms
download + load cost (cached after); subsequent calls are 3-30 ms per text.
"""

from __future__ import annotations

import threading
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ultron.config import get_config
from ultron.utils.logging import get_logger

logger = get_logger("memory.embedder")


class _SparseVec:
    """Thin wrapper to keep call sites independent of Qdrant's model classes.

    Carries the raw indices + values arrays from FastEmbed; the Qdrant store
    converts to ``qdrant_client.models.SparseVector`` at upsert time.
    """

    __slots__ = ("indices", "values")

    def __init__(self, indices: Sequence[int], values: Sequence[float]) -> None:
        self.indices = list(indices)
        self.values = list(values)

    def __len__(self) -> int:
        return len(self.indices)


class HybridEmbedder:
    """Encode text into a (dense, sparse) pair.

    Args:
        dense_model: HuggingFace ID for the dense model (FastEmbed).
        sparse_model: HuggingFace ID for the sparse BM25 encoder.
        eager: load both models at construction. Default False -- first
            ``encode_*`` call triggers the load. Set True at startup so the
            first hot-path call doesn't pay the load cost.
    """

    def __init__(
        self,
        dense_model: Optional[str] = None,
        sparse_model: Optional[str] = None,
        eager: bool = False,
    ) -> None:
        emb_cfg = get_config().embeddings
        self.dense_model_name = dense_model or emb_cfg.dense_model
        self.sparse_model_name = sparse_model or emb_cfg.sparse_model
        self._dense = None
        self._sparse = None
        self._lock = threading.Lock()
        if eager:
            self._ensure_dense()
            self._ensure_sparse()

    # --- internal: lazy loads ------------------------------------------------

    def _ensure_dense(self) -> None:
        if self._dense is not None:
            return
        with self._lock:
            if self._dense is not None:
                return
            from fastembed import TextEmbedding

            logger.info("Loading dense embedder %s on CPU...", self.dense_model_name)
            t0 = time.monotonic()
            # Cap ONNX threads so the embedder's idle pool doesn't compete
            # with Piper for CPU during conversation. Two threads is plenty
            # for bge-small at typical query rates.
            self._dense = TextEmbedding(self.dense_model_name, threads=2)
            logger.info(
                "Dense embedder ready in %.1fs", time.monotonic() - t0
            )

    def _ensure_sparse(self) -> None:
        if self._sparse is not None:
            return
        with self._lock:
            if self._sparse is not None:
                return
            from fastembed import SparseTextEmbedding

            logger.info("Loading sparse embedder %s on CPU...", self.sparse_model_name)
            t0 = time.monotonic()
            self._sparse = SparseTextEmbedding(self.sparse_model_name, threads=2)
            logger.info(
                "Sparse embedder ready in %.1fs", time.monotonic() - t0
            )

    # --- public API ----------------------------------------------------------

    @property
    def dim(self) -> int:
        return get_config().embeddings.dense_dim

    def encode_dense(self, texts: Iterable[str] | str) -> np.ndarray:
        """Return ``(N, 384)`` (or ``(384,)`` for a single string) float32 array."""
        self._ensure_dense()
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return np.zeros((0, self.dim), dtype=np.float32)
        # FastEmbed yields one np.ndarray per text.
        vecs = np.asarray(list(self._dense.embed(batch)), dtype=np.float32)
        return vecs[0] if single else vecs

    def encode_sparse(self, texts: Iterable[str] | str) -> List[_SparseVec]:
        """Return a list of :class:`_SparseVec` (one per input)."""
        self._ensure_sparse()
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return []
        out = [
            _SparseVec(s.indices.tolist(), s.values.tolist())
            for s in self._sparse.embed(batch)
        ]
        return out if not single else [out[0]]

    def encode_query_dense(self, query: str) -> np.ndarray:
        """Single-string query embedding via FastEmbed's query-side path.

        bge-small uses an asymmetric query encoding with a small instruction
        prefix; FastEmbed handles that for us.
        """
        self._ensure_dense()
        return np.asarray(
            list(self._dense.query_embed([query]))[0], dtype=np.float32
        )

    def encode_query_sparse(self, query: str) -> _SparseVec:
        """Single-string query sparse vector (BM25)."""
        self._ensure_sparse()
        s = list(self._sparse.query_embed([query]))[0]
        return _SparseVec(s.indices.tolist(), s.values.tolist())

    def encode_query_dense_batch(self, queries: Sequence[str]) -> np.ndarray:
        """V1-gap A2: batch the dense query encoder for multi-pass retrieval.

        Returns ``(N, 384)`` float32. Empty input -> zero rows. Wraps
        FastEmbed's ``query_embed`` in one call so the multi-pass path
        pays a single ONNX-runtime warmup instead of N.
        """
        self._ensure_dense()
        seq = list(queries)
        if not seq:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            list(self._dense.query_embed(seq)), dtype=np.float32,
        )

    def encode_query_sparse_batch(self, queries: Sequence[str]) -> List[_SparseVec]:
        """V1-gap A2: batch the sparse query encoder for multi-pass retrieval."""
        self._ensure_sparse()
        seq = list(queries)
        if not seq:
            return []
        return [
            _SparseVec(s.indices.tolist(), s.values.tolist())
            for s in self._sparse.query_embed(seq)
        ]
