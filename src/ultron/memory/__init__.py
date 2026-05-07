"""Conversation memory: rolling deque + JSONL persistence + embedding RAG.

The orchestrator holds one :class:`ConversationMemory` for the lifetime of the
process. Every user turn and assistant turn is appended to disk and embedded
into an in-memory matrix. The LLM hydrates each new prompt with
``recent(N)`` turns plus ``retrieve(query, k)`` snippets from older history.
"""

from ultron.memory.store import ConversationMemory, MemoryTurn

__all__ = ["ConversationMemory", "MemoryTurn"]
