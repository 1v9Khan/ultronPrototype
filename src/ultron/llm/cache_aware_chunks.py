"""Chunked prompt format with cache-control markers for HTTP LLM paths.

Pattern lifted in spirit (not in source) from aider's
``coders/chat_chunks.py`` + ``format_messages`` (Apache 2.0; see
``THIRD_PARTY_NOTICES.md``).

A prompt that supports prefix caching looks fundamentally different
from a plain "list of role/content" array. Anthropic's prompt
caching, for example, accepts ``cache_control: {"type": "ephemeral"}``
on individual content blocks; everything from the start of the
message list through the LAST block carrying that marker is cached
for 5 minutes server-side and reused on subsequent calls that share
the same prefix. With ~1-2 k tokens of repo map + ~3-5 k tokens of
system prompt, that's the difference between paying $0.30 and $0.03
per turn.

The trick is identifying which chunks are STABLE across turns. From
the catalog:

  * **system** — the persona / instructions / safety preamble.
    Stable for entire session.
  * **examples** — few-shot demonstrations. Stable for entire session.
  * **repo_map** — PageRank-weighted symbol map (batch 2). Stable
    for several turns at a time (changes when chat files do).
  * **readonly_files** — files in the chat but not being edited.
    Stable for several turns.
  * **chat_files** — files actively being edited. Less stable.
  * **history** — past turns. Grows turn by turn but tail-stable.
  * **current** — the user's just-sent message. NEVER cacheable.

We mark the first four cacheable, the next two semi-stable (cached
when present), and the last never. Within a chunk, the
``cache_control`` marker only goes on the LAST block — Anthropic
caches up through that marker, so marking earlier ones is wasted
metadata.

Public surface:

  * :class:`CacheableChunk` — frozen, one role/content block.
  * :class:`ChunkedPrompt` — collection with named slots.
  * :func:`to_anthropic_messages` — flatten to the Anthropic
    ``messages`` shape with ``cache_control`` injection.
  * :func:`to_plain_messages` — flatten without cache markers, for
    clients that don't support them (OpenAI, local llama-cpp).
  * :func:`count_cacheable_chars` — character count of the
    cacheable prefix (for visibility).
  * :data:`DEFAULT_CHUNK_STABILITY` — Mapping[chunk_name -> bool]
    declaring which slots are cacheable by default.

This module has zero hard dependencies on an LLM SDK — it's a
prompt-shape utility. Callers (a future Anthropic / litellm /
DeepSeek HTTP client) pick which serializer they want.

When no HTTP LLM with prompt caching is wired into ultron yet
(2026-05-22 state: only the CLI bridge), the local-LLM path can
still use :func:`to_plain_messages` to flatten the chunks for
``llama-cpp-python`` — the KV cache there benefits from stable
prompt prefixes the same way Anthropic's does, just without the
explicit cache_control marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


# Default stability mapping. The catalog's ordering (system → examples →
# repo_map → readonly_files → chat_files → history → current) is the
# stability ordering: earlier chunks change less often. ``True`` means
# "include in the cacheable prefix"; ``False`` means "exclude even if
# present (assume the chunk is dynamic)".
DEFAULT_CHUNK_STABILITY: Mapping[str, bool] = {
    "system": True,
    "examples": True,
    "repo_map": True,
    "readonly_files": True,
    "chat_files": True,
    "history": False,
    "current": False,
}


# Default sequence order — the catalog's recommended message ordering.
DEFAULT_CHUNK_ORDER: Sequence[str] = (
    "system",
    "examples",
    "repo_map",
    "readonly_files",
    "chat_files",
    "history",
    "current",
)


@dataclass(frozen=True)
class CacheableChunk:
    """One labelled role/content block.

    Attributes:
        role: ``"system"``, ``"user"``, or ``"assistant"``.
        content: The text payload.
        label: Free-form slot name (``"system"``, ``"repo_map"``,
            ...). Used by the serializer to look up stability.
        cacheable: Override per-block. When ``None``, the serializer
            consults :data:`DEFAULT_CHUNK_STABILITY` keyed by
            ``label``. Pass ``True`` / ``False`` to force.
    """

    role: str
    content: str
    label: str
    cacheable: Optional[bool] = None


@dataclass
class ChunkedPrompt:
    """Named slots for the catalog's prompt structure.

    Each slot is a list of :class:`CacheableChunk`. Empty lists are
    fine — the serializer skips empty slots. The order of slots in
    the serialized output follows :data:`DEFAULT_CHUNK_ORDER` unless
    the caller overrides via :func:`to_anthropic_messages`'s
    ``slot_order`` kwarg.
    """

    system: List[CacheableChunk] = field(default_factory=list)
    examples: List[CacheableChunk] = field(default_factory=list)
    repo_map: List[CacheableChunk] = field(default_factory=list)
    readonly_files: List[CacheableChunk] = field(default_factory=list)
    chat_files: List[CacheableChunk] = field(default_factory=list)
    history: List[CacheableChunk] = field(default_factory=list)
    current: List[CacheableChunk] = field(default_factory=list)

    def add_system(self, content: str) -> None:
        """Convenience: append a system message to the ``system`` slot."""
        self.system.append(CacheableChunk(
            role="system", content=content, label="system",
        ))

    def add_repo_map(self, content: str) -> None:
        """Convenience: append a repo-map block as a user message
        (Anthropic doesn't accept multiple system messages, so we
        encode the map as a user message labelled ``"repo_map"``)."""
        self.repo_map.append(CacheableChunk(
            role="user", content=content, label="repo_map",
        ))

    def add_history_turn(self, role: str, content: str) -> None:
        """Convenience: append a historical turn (not cacheable)."""
        self.history.append(CacheableChunk(
            role=role, content=content, label="history",
        ))

    def add_current(self, content: str) -> None:
        """Convenience: append the user's just-sent message (never cacheable)."""
        self.current.append(CacheableChunk(
            role="user", content=content, label="current",
        ))

    def slot(self, name: str) -> List[CacheableChunk]:
        """Look up a slot by name. Raises KeyError when name unknown."""
        slot_map = {
            "system": self.system,
            "examples": self.examples,
            "repo_map": self.repo_map,
            "readonly_files": self.readonly_files,
            "chat_files": self.chat_files,
            "history": self.history,
            "current": self.current,
        }
        if name not in slot_map:
            raise KeyError(
                f"unknown chunk slot {name!r}; "
                f"valid slots: {sorted(slot_map)}"
            )
        return slot_map[name]


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------


def to_anthropic_messages(
    prompt: ChunkedPrompt,
    *,
    slot_order: Sequence[str] = DEFAULT_CHUNK_ORDER,
    stability: Mapping[str, bool] = DEFAULT_CHUNK_STABILITY,
    cache_control_type: str = "ephemeral",
) -> List[Dict[str, Any]]:
    """Serialise the chunked prompt into Anthropic's ``messages`` format.

    Adds ``cache_control: {"type": cache_control_type}`` to the LAST
    block of the longest CONTIGUOUS cacheable prefix. Anthropic caches
    everything up through that marker; further markers are ignored
    by the API and waste metadata, so we emit exactly one.

    System chunks become ``role="system"`` (Anthropic accepts a single
    system block as a top-level field, but the list-of-messages shape
    used here is also valid via the ``messages`` array when the SDK
    accepts it; callers that need the top-level system field can pull
    the system blocks out themselves).

    Args:
        prompt: The :class:`ChunkedPrompt` to serialise.
        slot_order: Override the slot iteration order. Default is
            the catalog's stability-ordered sequence.
        stability: Override which slots are cacheable. Defaults to
            :data:`DEFAULT_CHUNK_STABILITY`.
        cache_control_type: The Anthropic ``cache_control`` type
            (currently only ``"ephemeral"`` is supported by the API).

    Returns:
        A list of dicts shaped like
        ``[{"role": ..., "content": [{"type": "text", "text": ..., "cache_control": ...?}]}]``.
        The cache_control field is present only on the final block of
        the longest cacheable prefix.
    """
    out: List[Dict[str, Any]] = []
    last_cacheable_index = -1

    # First pass: emit every chunk + track the index of the last
    # cacheable block.
    for slot_name in slot_order:
        slot = prompt.slot(slot_name)
        slot_cacheable = stability.get(slot_name, False)
        for chunk in slot:
            is_cacheable = chunk.cacheable if chunk.cacheable is not None else slot_cacheable
            out.append({
                "role": chunk.role,
                "content": [{
                    "type": "text",
                    "text": chunk.content,
                }],
            })
            if is_cacheable:
                last_cacheable_index = len(out) - 1

    # Second pass: inject cache_control on the marker block only.
    if last_cacheable_index >= 0:
        block = out[last_cacheable_index]["content"][0]
        block["cache_control"] = {"type": cache_control_type}

    return out


def to_plain_messages(
    prompt: ChunkedPrompt,
    *,
    slot_order: Sequence[str] = DEFAULT_CHUNK_ORDER,
) -> List[Dict[str, str]]:
    """Serialise without cache markers, for non-cache-aware clients.

    Returns a flat ``[{"role", "content": str}]`` list — the shape
    that OpenAI / litellm / local llama-cpp expect. Useful when the
    same ``ChunkedPrompt`` needs to drive both Anthropic and a
    different backend.
    """
    out: List[Dict[str, str]] = []
    for slot_name in slot_order:
        for chunk in prompt.slot(slot_name):
            out.append({"role": chunk.role, "content": chunk.content})
    return out


def count_cacheable_chars(
    prompt: ChunkedPrompt,
    *,
    slot_order: Sequence[str] = DEFAULT_CHUNK_ORDER,
    stability: Mapping[str, bool] = DEFAULT_CHUNK_STABILITY,
) -> int:
    """Character count of the cacheable prefix.

    For visibility: callers can log "warming N chars" before firing a
    keepalive ping (see :mod:`ultron.llm.cache_warmer`). Tokens are
    a more accurate metric but require a tokenizer; characters are a
    cheap proxy at ~4 chars/token for English.
    """
    total = 0
    for slot_name in slot_order:
        slot = prompt.slot(slot_name)
        slot_cacheable = stability.get(slot_name, False)
        for chunk in slot:
            is_cacheable = chunk.cacheable if chunk.cacheable is not None else slot_cacheable
            if is_cacheable:
                total += len(chunk.content)
    return total


__all__ = [
    "CacheableChunk",
    "ChunkedPrompt",
    "DEFAULT_CHUNK_ORDER",
    "DEFAULT_CHUNK_STABILITY",
    "count_cacheable_chars",
    "to_anthropic_messages",
    "to_plain_messages",
]
