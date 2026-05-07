"""JSONL-backed conversation memory with cosine-similarity RAG.

Each turn is appended to a single JSONL file as
``{"id": int, "ts": float, "role": "user|assistant", "content": str}``.
Embeddings live only in memory — they're recomputed at startup from the JSONL.
For thousands of turns this is fine; if it gets slow we'll persist them.

Two retrieval modes:
- :meth:`recent` returns the last N turns chronologically (always available).
- :meth:`retrieve` returns the top-K most semantically similar turns to a
  query, *excluding* the most recent ``exclude_recent`` turns so it never
  surfaces what the LLM is already going to see in the recent window.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import settings
from ultron.utils.logging import get_logger

logger = get_logger("memory.store")


@dataclass
class MemoryTurn:
    id: int
    ts: float
    role: str  # "user" | "assistant"
    content: str


class ConversationMemory:
    """Thread-safe append-on-add conversation log with optional RAG.

    Args:
        path: JSONL file. Created if missing.
        embedder: Optional :class:`Embedder`. Without one, only :meth:`recent`
            works — :meth:`retrieve` returns an empty list and logs once.
    """

    def __init__(
        self,
        path: Path = settings.MEMORY_PATH,
        embedder=None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._lock = threading.RLock()
        self._turns: List[MemoryTurn] = []
        self._embeddings: Optional[np.ndarray] = None
        self._next_id = 0
        self._warned_no_embedder = False

        self._load_from_disk()

    # --- loading ------------------------------------------------------------

    def _load_from_disk(self) -> None:
        if not self.path.is_file():
            logger.info("No prior conversation memory at %s — starting fresh", self.path)
            return
        loaded: List[MemoryTurn] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_num, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                    loaded.append(
                        MemoryTurn(
                            id=int(d["id"]),
                            ts=float(d["ts"]),
                            role=str(d["role"]),
                            content=str(d["content"]),
                        )
                    )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning("Skipping malformed memory line %d: %s", line_num, e)
        self._turns = loaded
        self._next_id = (loaded[-1].id + 1) if loaded else 0
        logger.info("Loaded %d prior conversation turns", len(loaded))

        if self._embedder is not None and loaded:
            t0 = time.monotonic()
            texts = [self._embed_text(t) for t in loaded]
            self._embeddings = self._embedder.encode(texts)
            logger.info(
                "Embedded %d turns in %.2fs", len(loaded), time.monotonic() - t0
            )

    @staticmethod
    def _embed_text(turn: MemoryTurn) -> str:
        return f"{turn.role}: {turn.content}"

    # --- writing ------------------------------------------------------------

    def add(self, role: str, content: str) -> MemoryTurn:
        """Append a turn, persist it, and update the in-memory index."""
        turn = MemoryTurn(
            id=self._next_id,
            ts=time.time(),
            role=role,
            content=content,
        )
        with self._lock:
            self._next_id += 1
            self._turns.append(turn)
            self._append_to_disk(turn)
            if self._embedder is not None:
                vec = self._embedder.encode(self._embed_text(turn)).reshape(1, -1)
                if self._embeddings is None:
                    self._embeddings = vec
                else:
                    self._embeddings = np.vstack([self._embeddings, vec])
        return turn

    def _append_to_disk(self, turn: MemoryTurn) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("Failed to persist memory turn: %s", e)

    # --- reading ------------------------------------------------------------

    def recent(self, n: int) -> List[MemoryTurn]:
        """Return the last ``n`` turns in chronological order."""
        if n <= 0:
            return []
        with self._lock:
            return list(self._turns[-n:])

    def retrieve(
        self,
        query: str,
        k: int = settings.MEMORY_RAG_TOP_K,
        exclude_recent: int = settings.MEMORY_RAG_EXCLUDE_RECENT,
    ) -> List[MemoryTurn]:
        """Top-``k`` turns by cosine similarity, excluding the last ``exclude_recent``.

        Returns ``[]`` if no embedder is configured, the query is empty, or
        there's nothing older than the recent window.
        """
        if not query.strip():
            return []
        if self._embedder is None:
            if not self._warned_no_embedder:
                logger.info("retrieve() called without embedder — returning empty")
                self._warned_no_embedder = True
            return []
        with self._lock:
            n_turns = len(self._turns)
            cutoff = max(0, n_turns - exclude_recent)
            if cutoff <= 0 or self._embeddings is None:
                return []
            corpus = self._embeddings[:cutoff]
            qvec = self._embedder.encode(query)
            sims = corpus @ qvec  # vectors are L2-normalized → dot = cosine
            if sims.size == 0:
                return []
            top = np.argsort(-sims)[: max(1, k)]
            return [self._turns[i] for i in top]

    # --- introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def close(self) -> None:
        """No-op; persistence is append-on-add. Provided for symmetry."""
        return None
