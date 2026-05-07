"""Sentence-Transformer embedder used for conversation RAG.

Wraps a small bi-encoder (default: ``all-MiniLM-L6-v2``, 384 dim, ~22 M params)
so the rest of the codebase doesn't import sentence_transformers directly. The
encoder runs on CUDA when available; falls back to CPU otherwise — the model
is light enough that CPU is fine for retrieval-time queries.
"""

from __future__ import annotations

import time
from typing import Iterable, List

import numpy as np

from config import settings
from ultron.utils.logging import get_logger

logger = get_logger("memory.embeddings")


class Embedder:
    """Encode text into a fixed-dim float32 vector.

    Args:
        model_name: HuggingFace ID of the sentence-transformer to load.
        device: ``"cuda"`` / ``"cpu"`` / ``None``. ``None`` auto-selects CUDA
            if available, else CPU.
    """

    def __init__(
        self,
        model_name: str = settings.MEMORY_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device

        logger.info("Loading embedder %s on %s…", model_name, device)
        t0 = time.monotonic()
        self._model = SentenceTransformer(model_name, device=device)
        self._model.eval()
        self.dim = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            "Embedder ready in %.2fs (dim=%d)", time.monotonic() - t0, self.dim
        )

    def encode(self, texts: Iterable[str] | str) -> np.ndarray:
        """Return float32 array of shape ``(N, dim)`` (or ``(dim,)`` for a single string).

        Vectors are L2-normalized so cosine similarity is just a dot product.
        """
        single = isinstance(texts, str)
        batch: List[str] = [texts] if single else list(texts)
        if not batch:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        return vecs[0] if single else vecs
