"""Tail-preserve history compression with race protection.

Pattern lifted in spirit (not in source) from aider's ``history.py``
``ChatSummary`` (Apache 2.0; see ``THIRD_PARTY_NOTICES.md``). Implements
the catalog T6 algorithm: when a conversation history exceeds a token
budget, summarise the oldest half via an LLM call and prepend the
summary to the most-recent tail so total tokens land back inside the
budget. Recurses (max depth 3) when one pass still doesn't fit.

Race protection (catalog T21): the compression is intended to run on a
background thread while the foreground continues conversing. If the
foreground appends more turns *during* the LLM call, blindly applying
the result would lose those new turns. The race-protected wrapper
:func:`compress_history_with_guard` captures a snapshot at entry, runs
the compression on the snapshot, and validates the snapshot still
matches the live state on return. Mismatch -> result is discarded
(silently); the foreground keeps its newer state.

Public surface:

  * :func:`find_split_point` — pure function that picks where to split
    the message list. Respects an "ensure assistant boundary" flag so
    the head ends on an assistant message (LLMs are less brittle when
    the prompt ends "naturally").
  * :func:`compress_history` — single-pass compression. Calls
    ``summarize_fn(head_messages_text) -> str`` and returns the
    rebuilt message list.
  * :func:`compress_history_recursive` — recursive variant that
    re-compresses when one pass doesn't fit. Bounded by ``max_depth``;
    on the last attempt, summarises EVERYTHING (including the tail) to
    guarantee a result that fits the budget.
  * :func:`compress_history_with_guard` — race-protected wrapper.
    Takes a :class:`SnapshotGuard` plus a key, captures a snapshot of
    the messages at entry, runs the compression, and only returns the
    result if the snapshot still matches at exit.
  * :class:`CompressionResult` — frozen dataclass with the compressed
    messages + telemetry (iterations used, original-vs-final token
    counts, race_detected flag, etc.).

Message shape: each message is a ``Mapping`` with at least the keys
``"role"`` (``"user"`` / ``"assistant"`` / ``"system"``) and
``"content"`` (str). Callers using a different shape can pre-convert
via :func:`messages_to_dicts` or pass a custom ``role_of`` / ``text_of``
pair to the algorithm.

Fail-open: any LLM exception, any malformed input, returns a
:class:`CompressionResult` with ``compressed=None`` and
``error="..."``. Callers should treat this as "leave history as-is".

The module is pure-Python stdlib; no LLM client coupling. The caller
injects ``summarize_fn`` (typically wrapping the in-process LLM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence

from ultron.utils.snapshot_guard import SnapshotGuard
from ultron.utils.snapshot_guard import matches as snapshot_matches
from ultron.utils.snapshot_guard import take as snapshot_take


logger = logging.getLogger("ultron.memory.history_compression")


# Default summary preamble — prepended to the compressed message so
# the downstream LLM has explicit context for what the summary is.
DEFAULT_SUMMARY_PREAMBLE = (
    "Here's a summary of the conversation so far:\n\n"
)


# Default max recursion depth for :func:`compress_history_recursive`.
# Above this, the algorithm bails out and summarises everything in one
# pass. 3 mirrors aider's value.
DEFAULT_MAX_DEPTH = 3


# Type aliases. ``Message`` is a Mapping with "role" + "content" keys.
Message = Mapping[str, Any]
TokenCounter = Callable[[str], int]
SummarizeFn = Callable[[str], str]


@dataclass(frozen=True)
class CompressionResult:
    """Outcome of a single compression attempt.

    Attributes:
        compressed: The compressed message list, or ``None`` when the
            attempt failed (LLM error, race detected, no head to
            summarise, etc.).
        original_tokens: Token count of the input messages.
        final_tokens: Token count of the returned compressed messages,
            or 0 when ``compressed`` is None.
        head_summarised: How many of the input messages were folded
            into the summary.
        tail_preserved: How many of the input messages were carried
            through verbatim.
        depth: Recursion depth this result settled at (0 for
            single-pass).
        race_detected: True iff the race-protected wrapper observed
            a snapshot mismatch and discarded the result.
        error: Short description when ``compressed`` is None.
    """

    compressed: Optional[List[Message]]
    original_tokens: int
    final_tokens: int
    head_summarised: int
    tail_preserved: int
    depth: int = 0
    race_detected: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Pure split-point selection
# ---------------------------------------------------------------------------


def find_split_point(
    messages: Sequence[Message],
    max_tokens: int,
    token_counter: TokenCounter,
    *,
    prefer_assistant_boundary: bool = True,
) -> int:
    """Return the index where the tail-preserve split should land.

    Walks backwards from the end, accumulating tokens, until adding
    the next message would push the tail past ``max_tokens / 2``.
    That index is the split point: messages[:split] form the head
    (to be summarised), messages[split:] form the tail (verbatim).

    When ``prefer_assistant_boundary`` is True (default), the split
    point is nudged forward as needed so messages[split-1] is an
    assistant message — small LLMs tolerate that boundary better than
    splitting mid-user-turn.

    Edge cases:
      * Empty input → returns 0.
      * Single message → returns 0 (no split; whole thing is tail).
      * All-tail-fits case → returns 0.
      * No head can be made above the boundary preference → returns
        the raw split point even when it's mid-user.
    """
    n = len(messages)
    if n <= 1 or max_tokens <= 0:
        return 0

    target_tail = max(1, max_tokens // 2)
    accum = 0
    split = n
    for i in range(n - 1, -1, -1):
        msg_text = _text_of(messages[i])
        cost = token_counter(msg_text)
        if accum + cost > target_tail and i < n - 1:
            split = i + 1
            break
        accum += cost
    else:
        # Loop completed without breaking — whole conversation fits in
        # the tail half. No head to summarise.
        return 0

    if not prefer_assistant_boundary:
        return split

    # Back off to land on an assistant boundary: we want
    # messages[split - 1] to be assistant. Walk *backward* (shrinking
    # the head, growing the tail slightly) until we find one — this
    # mirrors aider's "back off split_index" behaviour. If no
    # assistant boundary exists at any earlier position, return 0 to
    # signal "skip compression; nothing to summarise cleanly".
    cand = split
    while cand > 0:
        if _role_of(messages[cand - 1]) == "assistant":
            return cand
        cand -= 1
    return 0


# ---------------------------------------------------------------------------
# Single-pass compression
# ---------------------------------------------------------------------------


def compress_history(
    messages: Sequence[Message],
    summarize_fn: SummarizeFn,
    *,
    max_tokens: int,
    token_counter: TokenCounter,
    summary_preamble: str = DEFAULT_SUMMARY_PREAMBLE,
    prefer_assistant_boundary: bool = True,
) -> CompressionResult:
    """Run ONE pass of tail-preserve compression.

    Algorithm:
      1. Compute :func:`find_split_point`.
      2. If split == 0 (nothing to summarise), return the input
         unchanged with ``head_summarised=0``.
      3. Render messages[:split] into a single text blob via
         :func:`messages_to_text`.
      4. Call ``summarize_fn(head_text) -> summary_str``.
      5. Build the new message list:
         ``[{"role": "user", "content": preamble + summary_str},
            *messages[split:]]``
      6. Return :class:`CompressionResult` with the new list.

    Any exception in step 4 returns a result with ``compressed=None``
    and ``error=<str>``.
    """
    n = len(messages)
    original_text = "".join(_text_of(m) for m in messages)
    original_tokens = token_counter(original_text) if original_text else 0

    if n == 0:
        return CompressionResult(
            compressed=list(messages),
            original_tokens=0,
            final_tokens=0,
            head_summarised=0,
            tail_preserved=0,
        )

    split = find_split_point(
        messages,
        max_tokens,
        token_counter,
        prefer_assistant_boundary=prefer_assistant_boundary,
    )
    if split <= 0:
        return CompressionResult(
            compressed=list(messages),
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            head_summarised=0,
            tail_preserved=n,
        )

    head = list(messages[:split])
    tail = list(messages[split:])
    head_text = messages_to_text(head)

    try:
        summary = summarize_fn(head_text) or ""
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(
            "history_compression: summarize_fn raised (%s); skipping",
            exc,
        )
        return CompressionResult(
            compressed=None,
            original_tokens=original_tokens,
            final_tokens=0,
            head_summarised=len(head),
            tail_preserved=len(tail),
            error=str(exc),
        )

    if not summary.strip():
        # Empty summary is useless; treat as failure rather than
        # silently dropping the head.
        return CompressionResult(
            compressed=None,
            original_tokens=original_tokens,
            final_tokens=0,
            head_summarised=len(head),
            tail_preserved=len(tail),
            error="empty summary",
        )

    summary_msg = {
        "role": "user",
        "content": f"{summary_preamble}{summary.strip()}",
    }
    compressed_list: List[Message] = [summary_msg, *tail]
    final_text = "".join(_text_of(m) for m in compressed_list)
    final_tokens = token_counter(final_text)
    return CompressionResult(
        compressed=compressed_list,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        head_summarised=len(head),
        tail_preserved=len(tail),
    )


# ---------------------------------------------------------------------------
# Recursive compression
# ---------------------------------------------------------------------------


def compress_history_recursive(
    messages: Sequence[Message],
    summarize_fn: SummarizeFn,
    *,
    max_tokens: int,
    token_counter: TokenCounter,
    summary_preamble: str = DEFAULT_SUMMARY_PREAMBLE,
    prefer_assistant_boundary: bool = True,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> CompressionResult:
    """Recurse :func:`compress_history` until the result fits or
    ``max_depth`` is exhausted.

    On the final allowed iteration, falls back to summarising
    EVERYTHING (the tail too) so the returned list is guaranteed to
    be small — matches aider's ``ChatSummary.summarize_real`` behaviour
    on deep recursion.
    """
    current: List[Message] = list(messages)
    last_result: Optional[CompressionResult] = None

    for depth in range(max_depth):
        result = compress_history(
            current,
            summarize_fn,
            max_tokens=max_tokens,
            token_counter=token_counter,
            summary_preamble=summary_preamble,
            prefer_assistant_boundary=prefer_assistant_boundary,
        )
        if result.compressed is None:
            # Single-pass failure — propagate.
            if last_result is not None and last_result.compressed is not None:
                # We have a previous partial success; return that.
                return CompressionResult(
                    compressed=last_result.compressed,
                    original_tokens=last_result.original_tokens,
                    final_tokens=last_result.final_tokens,
                    head_summarised=last_result.head_summarised,
                    tail_preserved=last_result.tail_preserved,
                    depth=last_result.depth,
                    error=result.error,
                )
            return CompressionResult(
                compressed=result.compressed,
                original_tokens=result.original_tokens,
                final_tokens=result.final_tokens,
                head_summarised=result.head_summarised,
                tail_preserved=result.tail_preserved,
                depth=depth,
                error=result.error,
            )
        last_result = CompressionResult(
            compressed=result.compressed,
            original_tokens=result.original_tokens,
            final_tokens=result.final_tokens,
            head_summarised=result.head_summarised,
            tail_preserved=result.tail_preserved,
            depth=depth,
        )
        if result.final_tokens <= max_tokens:
            return last_result
        current = result.compressed  # type: ignore[assignment]

    # Hit the depth cap. Fall back to summarising everything.
    everything_text = messages_to_text(current)
    try:
        summary = summarize_fn(everything_text) or ""
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(
            "history_compression: deep-fallback summarize_fn raised (%s)",
            exc,
        )
        if last_result is not None:
            return last_result
        return CompressionResult(
            compressed=None,
            original_tokens=last_result.original_tokens if last_result else 0,
            final_tokens=0,
            head_summarised=0,
            tail_preserved=0,
            depth=max_depth,
            error=str(exc),
        )
    if not summary.strip():
        if last_result is not None:
            return last_result
        return CompressionResult(
            compressed=None,
            original_tokens=0,
            final_tokens=0,
            head_summarised=0,
            tail_preserved=0,
            depth=max_depth,
            error="empty summary",
        )
    summary_msg: Message = {
        "role": "user",
        "content": f"{summary_preamble}{summary.strip()}",
    }
    final_text = _text_of(summary_msg)
    return CompressionResult(
        compressed=[summary_msg],
        original_tokens=last_result.original_tokens if last_result else 0,
        final_tokens=token_counter(final_text),
        head_summarised=len(current),
        tail_preserved=0,
        depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Race-protected wrapper
# ---------------------------------------------------------------------------


def compress_history_with_guard(
    messages_live_ref: Sequence[Message],
    summarize_fn: SummarizeFn,
    *,
    max_tokens: int,
    token_counter: TokenCounter,
    guard: Optional[SnapshotGuard] = None,
    key: str = "history",
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> CompressionResult:
    """Race-protected wrapper around :func:`compress_history_recursive`.

    Args:
        messages_live_ref: Reference to the live messages list. The
            wrapper snapshots its value at entry, runs compression on
            the snapshot, then re-checks equality at exit.
        summarize_fn: LLM call. Receives a single text blob, returns
            the summary string.
        max_tokens: Token budget the compressed history should fit in.
        token_counter: Function mapping text -> token count.
        guard: Optional caller-owned :class:`SnapshotGuard`. When
            supplied, the wrapper uses it (keyed by ``key``) so the
            caller can extend the race protection across more steps
            than just this call. When None, a one-shot snapshot is
            taken internally.
        key: Identifier used when ``guard`` is provided. Lets one guard
            track several concurrent compression jobs.
        max_depth: Forwarded to :func:`compress_history_recursive`.

    Returns:
        A :class:`CompressionResult`. The ``race_detected`` flag is True
        when the live messages changed during the LLM call, in which
        case ``compressed`` is None (the caller should leave its state
        as-is).
    """
    snapshot_value = list(messages_live_ref)
    if guard is not None:
        guard.snapshot(key, snapshot_value)
    else:
        # Internal one-shot snapshot via the functional API.
        token = snapshot_take(snapshot_value)

    result = compress_history_recursive(
        snapshot_value,
        summarize_fn,
        max_tokens=max_tokens,
        token_counter=token_counter,
        max_depth=max_depth,
    )

    if guard is not None:
        unchanged = guard.unchanged(key, list(messages_live_ref))
        guard.drop(key)
    else:
        unchanged = snapshot_matches(token, list(messages_live_ref))  # type: ignore[arg-type]

    if not unchanged:
        return CompressionResult(
            compressed=None,
            original_tokens=result.original_tokens,
            final_tokens=0,
            head_summarised=result.head_summarised,
            tail_preserved=result.tail_preserved,
            depth=result.depth,
            race_detected=True,
            error="live messages changed during compression",
        )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def messages_to_text(messages: Sequence[Message]) -> str:
    """Render messages as a single text blob suitable for an LLM
    summarisation prompt.

    Format: one ``# ROLE\\n<content>\\n`` block per message, separated
    by blank lines. Matches aider's history-summary input format.
    """
    blocks: List[str] = []
    for m in messages:
        role = _role_of(m).upper() or "USER"
        content = _text_of(m)
        blocks.append(f"# {role}\n{content}")
    return "\n\n".join(blocks)


def messages_to_dicts(messages: Sequence[Any]) -> List[Message]:
    """Coerce arbitrary message objects into ``{"role", "content"}`` dicts.

    Useful when the caller is passing in dataclass instances or
    :class:`ultron.memory.background_summarizer.TurnSnapshot` objects;
    the algorithm operates on plain dicts internally.
    """
    out: List[Message] = []
    for m in messages:
        if isinstance(m, Mapping):
            out.append({"role": _role_of(m), "content": _text_of(m)})
            continue
        role = getattr(m, "role", None) or "user"
        content = getattr(m, "content", None) or ""
        out.append({"role": str(role), "content": str(content)})
    return out


def _role_of(m: Any) -> str:
    if isinstance(m, Mapping):
        return str(m.get("role", "user"))
    return str(getattr(m, "role", "user"))


def _text_of(m: Any) -> str:
    if isinstance(m, Mapping):
        return str(m.get("content", ""))
    return str(getattr(m, "content", ""))


__all__ = [
    "CompressionResult",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_SUMMARY_PREAMBLE",
    "compress_history",
    "compress_history_recursive",
    "compress_history_with_guard",
    "find_split_point",
    "messages_to_dicts",
    "messages_to_text",
]
