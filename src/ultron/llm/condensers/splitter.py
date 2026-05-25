"""Token-share message splitter with tool-call boundary preservation.

T10 (OpenClaw catalog port; see ``THIRD_PARTY_NOTICES.md``). Splits
a long transcript into N roughly-equal-token chunks BUT only at
boundaries that don't break ``assistant->tool_call`` -> ``tool_result``
pairs. Tracks pending tool-call IDs in a set; only emits a chunk
boundary when no tool calls are dangling.

The splitter is the missing piece in long-coding-session compaction:
without it, a token-share split can land in the middle of an
``assistant.tool_use{id: X}`` <-> ``tool_result{tool_use_id: X}`` pair,
producing a dangling tool_use_id error on the next LLM call.

Supplements the existing ultron condensers (``RecentCondenser`` /
``LLMSummarizingCondenser`` / etc.) — those decide WHICH messages to
keep; this splitter decides HOW to chop the kept set into
summarisation-friendly chunks.

Constants mirror OpenClaw:

* ``BASE_CHUNK_RATIO = 0.4`` (target 40 % of the context window per chunk)
* ``MIN_CHUNK_RATIO = 0.15`` (floor below which we refuse to scale further)
* ``SAFETY_MARGIN = 1.2`` (20 % buffer for token-estimation inaccuracy)
* ``IDENTIFIER_PRESERVATION_SUFFIX`` — system-prompt directive appended
  to summariser prompts so opaque ids (UUIDs, hashes, hostnames,
  file names) survive verbatim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)


#: Target share of the context window each chunk consumes.
BASE_CHUNK_RATIO: float = 0.4

#: Floor — never scale chunk ratio below this; ratios smaller than
#: this produce more LLM round-trips than the summary saves.
MIN_CHUNK_RATIO: float = 0.15

#: Multiplier applied to per-message token estimates to absorb the
#: gap between cheap estimation and the actual tokenizer count.
SAFETY_MARGIN: float = 1.2

#: A single message above this share of context is "oversized" and
#: cannot be summarised individually. Triggers the 3-tier fallback.
OVERSIZED_MESSAGE_RATIO: float = 0.5

#: System-prompt directive appended to compaction prompts so the
#: summariser preserves opaque identifiers exactly. This is the
#: missing piece in most LLM compaction strategies — without it,
#: summarisers cheerfully rewrite UUIDs / hostnames / file names
#: and break later lookups.
IDENTIFIER_PRESERVATION_SUFFIX: str = (
    "Preserve all opaque identifiers exactly as written "
    "(no shortening or reconstruction), including UUIDs, hashes, "
    "IDs, hostnames, IPs, ports, URLs, and file names."
)

#: Fallback summary when an oversized message defeats every tier.
UNSUMMARISABLE_TEMPLATE: str = (
    "Context contained {total} messages ({oversized} oversized). "
    "Summary unavailable due to size limits."
)


class IdentifierPreservationPolicy(str, Enum):
    """Knob controlling whether the preservation suffix is appended.

    * ``STRICT`` — always append the suffix (default).
    * ``OFF`` — never append; summariser is free to rewrite ids.
    * ``CUSTOM`` — append a caller-supplied text instead.
    """

    STRICT = "strict"
    OFF = "off"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Message:
    """One transcript message used by the splitter.

    Generic shape (role + content) plus the optional ``tool_calls`` /
    ``tool_use_id`` / ``stop_reason`` fields that let the splitter
    recognise dangling tool-call pairs. Ultron's condenser callers
    already shape their messages this way; this dataclass formalises
    the contract.

    Attributes:
        role: ``"user"`` / ``"assistant"`` / ``"tool"`` / ``"system"``.
        content: Free-form message body (single string OR list of
            multimodal segments; the splitter treats both equally
            for token-estimation purposes).
        tool_calls: Optional sequence of ``{"id": str, ...}`` records
            describing tool calls the assistant initiated. Each ``id``
            is added to the pending set; the matching ``tool_result``
            removes it.
        tool_use_id: Optional id present on tool-result messages,
            paired with the ``tool_calls[*].id`` of the matching
            assistant message.
        stop_reason: Optional assistant ``stop_reason``. When ``"aborted"``
            or ``"error"`` the splitter does NOT add tool_calls to the
            pending set (the result is never coming).
        metadata: Free-form per-message metadata, opaque to the splitter.
    """

    role: str
    content: Any
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    tool_use_id: Optional[str] = None
    stop_reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def estimate_message_tokens(message: Message) -> int:
    """Cheap-and-cheerful token estimator: ``len(content) // 4``.

    Callers with a real tokenizer should pass ``token_counter`` to the
    splitter functions; this default keeps the module dependency-free
    so unit tests run without loading a transformer.
    """
    body = message.content
    if isinstance(body, str):
        n_chars = len(body)
    elif isinstance(body, list):
        n_chars = sum(
            len(part.get("text", "")) if isinstance(part, Mapping) else len(str(part))
            for part in body
        )
    else:
        n_chars = len(str(body))
    return max(1, n_chars // 4)


TokenCounter = Callable[[Message], int]


def _sum_tokens(messages: Iterable[Message], counter: TokenCounter) -> int:
    return sum(counter(m) for m in messages)


def is_oversized_for_summary(
    message: Message,
    *,
    context_window: int,
    token_counter: TokenCounter = estimate_message_tokens,
    ratio: float = OVERSIZED_MESSAGE_RATIO,
) -> bool:
    """``True`` when one message alone exceeds ``ratio`` of the window.

    An oversized message cannot be summarised inline — it must either
    be elided wholesale (with a placeholder note) or surfaced to the
    user as "I can't compress this; please trim it manually."
    """
    if context_window <= 0:
        return True
    return token_counter(message) * SAFETY_MARGIN > context_window * ratio


def compute_adaptive_chunk_ratio(
    messages: Sequence[Message],
    *,
    context_window: int,
    token_counter: TokenCounter = estimate_message_tokens,
    base_ratio: float = BASE_CHUNK_RATIO,
    min_ratio: float = MIN_CHUNK_RATIO,
) -> float:
    """Scale ``base_ratio`` downward when the average message is large.

    When the average per-message size (after the safety margin) is
    > 10 % of the context window, reduce the chunk ratio so chunks
    stay small enough for the summariser to swallow without
    re-truncating. Clamps to ``min_ratio``.
    """
    if not messages or context_window <= 0:
        return base_ratio
    n = len(messages)
    avg_tokens = _sum_tokens(messages, token_counter) / n
    avg_share = (avg_tokens * SAFETY_MARGIN) / context_window
    if avg_share <= 0.10:
        return base_ratio
    # Reduce by min(2 * avg_share, base - min) so the ratio stays
    # within [min_ratio, base_ratio].
    reduction = min(avg_share * 2.0, base_ratio - min_ratio)
    return max(min_ratio, base_ratio - reduction)


def _extract_tool_call_ids(message: Message) -> tuple[str, ...]:
    """Return the ids of every tool call the assistant initiated."""
    if message.role != "assistant" or not message.tool_calls:
        return ()
    if message.stop_reason in ("aborted", "error"):
        return ()
    out: list[str] = []
    for call in message.tool_calls:
        identifier = call.get("id")
        if isinstance(identifier, str) and identifier:
            out.append(identifier)
    return tuple(out)


@dataclass(frozen=True)
class ChunkPlan:
    """Result of a splitting pass.

    Attributes:
        chunks: tuple of message tuples (each chunk preserves the
            sequencing of the inputs).
        oversized_indices: indices of messages that individually
            exceed the per-chunk budget (caller surfaces via the
            oversized fallback).
        adaptive_ratio: the chunk ratio used after adaptive scaling.
        total_tokens: total tokens across all chunks (estimate).
        target_tokens: per-chunk target after adaptive scaling.
    """

    chunks: tuple[tuple[Message, ...], ...]
    oversized_indices: tuple[int, ...] = ()
    adaptive_ratio: float = BASE_CHUNK_RATIO
    total_tokens: int = 0
    target_tokens: int = 0


def chunk_messages_by_max_tokens(
    messages: Sequence[Message],
    *,
    max_tokens: int,
    token_counter: TokenCounter = estimate_message_tokens,
) -> ChunkPlan:
    """Split ``messages`` into chunks each at or under ``max_tokens``.

    Tool-call boundary preservation: tracks pending tool-call ids;
    only emits a chunk boundary when the pending set is empty. If the
    current chunk hits the budget mid-tool-pair, the boundary slides
    to just before the assistant message that opened the still-pending
    call.

    Args:
        messages: input transcript (chronological).
        max_tokens: per-chunk token ceiling.
        token_counter: callable estimating per-message tokens.

    Returns:
        :class:`ChunkPlan` with the produced chunks.
    """
    if not messages:
        return ChunkPlan(chunks=())
    if max_tokens <= 0:
        return ChunkPlan(chunks=(tuple(messages),))
    chunks: list[list[Message]] = []
    current: list[Message] = []
    current_tokens = 0
    pending_ids: set[str] = set()
    pending_chunk_start: Optional[int] = None  # index of assistant that opened a pending call
    oversized: list[int] = []
    total_tokens = 0
    for idx, message in enumerate(messages):
        m_tokens = token_counter(message)
        total_tokens += m_tokens
        if m_tokens > max_tokens:
            oversized.append(idx)
        # Track tool-call lifecycle.
        opened_ids = _extract_tool_call_ids(message)
        if opened_ids:
            if not pending_ids:
                pending_chunk_start = len(current)  # boundary candidate
            for tid in opened_ids:
                pending_ids.add(tid)
        if message.role == "tool" and message.tool_use_id:
            pending_ids.discard(message.tool_use_id)
            if not pending_ids:
                pending_chunk_start = None
        current.append(message)
        current_tokens += m_tokens
        # Emit boundary when over budget AND no pending calls.
        if current_tokens >= max_tokens and not pending_ids:
            chunks.append(current)
            current = []
            current_tokens = 0
            pending_chunk_start = None
        elif current_tokens >= max_tokens and pending_chunk_start is not None and pending_chunk_start > 0:
            # We're over budget AND there's a clean boundary earlier
            # in the current chunk; split there.
            head = current[:pending_chunk_start]
            tail = current[pending_chunk_start:]
            if head:
                chunks.append(head)
            current = tail
            current_tokens = _sum_tokens(current, token_counter)
            pending_chunk_start = 0 if pending_ids else None
    if current:
        chunks.append(current)
    return ChunkPlan(
        chunks=tuple(tuple(c) for c in chunks),
        oversized_indices=tuple(oversized),
        adaptive_ratio=BASE_CHUNK_RATIO,
        total_tokens=total_tokens,
        target_tokens=max_tokens,
    )


def split_messages_by_token_share(
    messages: Sequence[Message],
    *,
    parts: int = 4,
    context_window: int = 8192,
    token_counter: TokenCounter = estimate_message_tokens,
    base_ratio: float = BASE_CHUNK_RATIO,
    min_ratio: float = MIN_CHUNK_RATIO,
) -> ChunkPlan:
    """Split into roughly-equal-token chunks honouring tool-call pairs.

    Computes an adaptive per-chunk target from ``base_ratio`` /
    ``min_ratio`` based on average message size, then delegates to
    :func:`chunk_messages_by_max_tokens`. Final chunk count is roughly
    ``min(parts, ceil(total / target))``.

    Args:
        messages: input transcript.
        parts: desired chunk count (target; actual may differ when
            tool-call pairs force unequal cuts).
        context_window: LLM context budget; used by the adaptive ratio.
        token_counter: estimator.
        base_ratio: starting per-chunk share of context.
        min_ratio: floor below which adaptive scaling stops.

    Returns:
        :class:`ChunkPlan`.
    """
    if not messages:
        return ChunkPlan(chunks=())
    if parts < 1:
        parts = 1
    ratio = compute_adaptive_chunk_ratio(
        messages,
        context_window=context_window,
        token_counter=token_counter,
        base_ratio=base_ratio,
        min_ratio=min_ratio,
    )
    target_tokens = max(1, int(context_window * ratio))
    plan = chunk_messages_by_max_tokens(
        messages,
        max_tokens=target_tokens,
        token_counter=token_counter,
    )
    return ChunkPlan(
        chunks=plan.chunks,
        oversized_indices=plan.oversized_indices,
        adaptive_ratio=ratio,
        total_tokens=plan.total_tokens,
        target_tokens=target_tokens,
    )


# ----------------------------------------------------------------------
# Identifier-preservation prompt suffix


def identifier_preservation_text(
    *,
    policy: IdentifierPreservationPolicy = IdentifierPreservationPolicy.STRICT,
    custom_text: str = "",
) -> str:
    """Return the suffix string per ``policy``.

    Returns empty string for ``OFF``. Returns ``custom_text`` for
    ``CUSTOM`` (caller's responsibility to provide non-empty text).
    """
    if policy == IdentifierPreservationPolicy.OFF:
        return ""
    if policy == IdentifierPreservationPolicy.CUSTOM:
        return custom_text or ""
    return IDENTIFIER_PRESERVATION_SUFFIX


def with_identifier_preservation(
    system_prompt: str,
    *,
    policy: IdentifierPreservationPolicy = IdentifierPreservationPolicy.STRICT,
    custom_text: str = "",
    separator: str = "\n\n",
) -> str:
    """Append the identifier-preservation suffix to ``system_prompt``.

    No-op for ``OFF`` policy and for empty prompts.
    """
    suffix = identifier_preservation_text(policy=policy, custom_text=custom_text)
    if not suffix:
        return system_prompt
    if not system_prompt:
        return suffix
    return f"{system_prompt}{separator}{suffix}"


# ----------------------------------------------------------------------
# 3-tier oversized fallback


SummariseFn = Callable[[Sequence[Message]], str]


def summarise_with_fallback(
    messages: Sequence[Message],
    *,
    context_window: int,
    summarise_fn: SummariseFn,
    token_counter: TokenCounter = estimate_message_tokens,
    oversized_template: str = "[Large {role} (~{tokens} tokens) omitted from summary]",
) -> str:
    """Produce a summary string, falling back through 3 tiers on failure.

    1. **Tier 1** — call ``summarise_fn(messages)``. Return on success.
    2. **Tier 2** — drop messages where ``is_oversized_for_summary``
       fires; replace each with an ``oversized_template`` note; retry
       ``summarise_fn`` against the trimmed list. Skip when the
       trimmed list equals the original (every message fits).
    3. **Tier 3** — return :data:`UNSUMMARISABLE_TEMPLATE` rendered
       with total + oversized counts.

    Args:
        messages: messages to summarise.
        context_window: LLM context budget; used by oversized check.
        summarise_fn: caller's LLM-summarisation callable. Returns
            the summary string OR raises / returns empty on failure.
        token_counter: per-message token estimator.
        oversized_template: placeholder format for elided messages.

    Returns:
        Summary string. Always non-empty.
    """
    if not messages:
        return ""
    total = len(messages)
    # Tier 1.
    try:
        first = summarise_fn(messages)
    except Exception:  # noqa: BLE001 -- fall through to tier 2
        LOGGER.warning("tier-1 summarisation raised; falling back", exc_info=True)
        first = ""
    if first.strip():
        return first
    # Tier 2.
    oversized_count = 0
    trimmed: list[Message] = []
    for message in messages:
        if is_oversized_for_summary(message, context_window=context_window, token_counter=token_counter):
            oversized_count += 1
            note_text = oversized_template.format(
                role=message.role,
                tokens=token_counter(message),
            )
            trimmed.append(Message(role="system", content=note_text))
        else:
            trimmed.append(message)
    if oversized_count:
        # Indices differ between trimmed and original only when at
        # least one message was elided; structure is otherwise 1:1.
        try:
            second = summarise_fn(trimmed)
        except Exception:  # noqa: BLE001 -- fall through to tier 3
            LOGGER.warning("tier-2 summarisation raised; falling back", exc_info=True)
            second = ""
        if second.strip():
            return second
    # Tier 3.
    return UNSUMMARISABLE_TEMPLATE.format(total=total, oversized=oversized_count)


__all__ = [
    "BASE_CHUNK_RATIO",
    "ChunkPlan",
    "IDENTIFIER_PRESERVATION_SUFFIX",
    "IdentifierPreservationPolicy",
    "MIN_CHUNK_RATIO",
    "Message",
    "OVERSIZED_MESSAGE_RATIO",
    "SAFETY_MARGIN",
    "SummariseFn",
    "TokenCounter",
    "UNSUMMARISABLE_TEMPLATE",
    "chunk_messages_by_max_tokens",
    "compute_adaptive_chunk_ratio",
    "estimate_message_tokens",
    "identifier_preservation_text",
    "is_oversized_for_summary",
    "split_messages_by_token_share",
    "summarise_with_fallback",
    "with_identifier_preservation",
]
