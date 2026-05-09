"""Conversation memory: Qdrant-backed hybrid retrieval (Phase 3).

Each turn is written asynchronously to an embedded Qdrant store with both a
dense bge-small embedding and a BM25 sparse vector. RAG retrieval combines
the two via Reciprocal Rank Fusion. Recent turns are cached in process so
``recent(n)`` is instant; older history is retrievable via ``retrieve()``.

The legacy JSONL store at :mod:`ultron.memory.store` is kept around purely as
the source for the one-time migration script
(``scripts/migrate_memory_to_qdrant.py``); production code should not import
it directly.
"""

from ultron.memory.embedder import HybridEmbedder
from ultron.memory.qdrant_store import ConversationMemory, FactRow, MemoryTurn

__all__ = ["ConversationMemory", "FactRow", "MemoryTurn", "HybridEmbedder"]
