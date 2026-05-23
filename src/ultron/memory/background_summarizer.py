"""Background conversation summarizer + structured fact extraction
(2026-05-19, Tracks 1c + 1d + 1e).

A token-density layer over ``ConversationMemory``: every N turns, the
background pass calls the same in-process LLM (lock-serialized) with
a JSON-mode prompt that returns BOTH a narrative summary AND a
structured fact list. The summary becomes a Qdrant entry tagged
``type=session_summary`` with span metadata; each extracted fact /
decision / preference becomes its own ``type=fact|decision|preference``
entry. The ranking layer (Track 1h) can then retrieve at the right
granularity for the query.

Design contracts (from the 2026-05-19 design conversation):

* **Single shared LLM instance.** No separate model load. The
  ``llama-cpp-python`` ``Llama`` instance is locked at the inference
  layer anyway -- the summarizer's call simply queues behind any
  foreground call. VRAM cost: zero.
* **Idle-threshold gating.** ``maybe_summarize`` only fires when the
  caller (the orchestrator) says the system has been idle for at
  least ``idle_threshold_seconds``. Prevents summarizer/foreground
  contention during active conversation. Default 30 s.
* **Cancellation.** ``cancel()`` flips a flag so the background
  thread exits its current LLM call ASAP. The orchestrator wires this
  to SPEECH_START so foreground responsiveness wins on contention.
* **Fail-open.** Any failure in the LLM call, JSON parse, or Qdrant
  write is logged WARN and swallowed. The summarizer's value is
  cumulative -- a missed pass costs nothing more than running it next
  time. Never raises.
* **Default OFF.** ``memory.background_summary.enabled`` defaults to
  False. With the flag off, the orchestrator never calls
  ``maybe_summarize`` and no Qdrant entries get the new types.

The module ships the building blocks; orchestrator wiring is
intentionally separate so we can land the safe machinery first and
flip the flag once live behaviour is verified.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# ----------------------------------------------------------------------
# Prompt scaffolding
# ----------------------------------------------------------------------


_SUMMARY_SYSTEM_PROMPT = (
    "You are an internal worker for the Ultron memory system. You "
    "are NOT the voice-facing assistant. Your job is to read a "
    "stretch of recent conversation and emit a compact, structured "
    "summary plus a list of facts the user established. You output "
    "VALID JSON only -- no preamble, no closing prose, no markdown "
    "fences."
)


_SUMMARY_INSTRUCTION_TEMPLATE = """\
Summarise the conversation below. Output VALID JSON matching this exact schema:

{{
  "summary": "<one paragraph, 60-120 words. Lead with the most consequential item.>",
  "facts": [
    {{"type": "fact", "subject": "<entity>", "predicate": "<verb phrase>", "object": "<value>"}}
  ],
  "decisions": [
    {{"topic": "<what was decided>", "outcome": "<the decision>", "status": "pending|made|reversed"}}
  ],
  "preferences": [
    {{"topic": "<area>", "value": "<what the user prefers>"}}
  ]
}}

Rules:
- Include only items that are EXPLICITLY established or decided by the user. No inference.
- Skip pleasantries, transitions, and tangents. They are noise.
- The "summary" must be self-contained -- a reader who has never seen these turns must be able to reconstruct what was discussed.
- If a category has no entries, return an empty list for it.

--- Conversation ---
{conversation}
--- End conversation ---

Respond with the JSON object only.
"""


def render_summary_prompt(
    turns: Sequence["TurnSnapshot"],
) -> str:
    """Render the summarization + fact-extraction prompt for ``turns``.

    Pure function; tests can call it directly. The turns are formatted
    as ``[role]: content`` lines, one per turn. Long content is left
    intact (the LLM call handles truncation if needed).
    """
    lines: List[str] = []
    for t in turns:
        role = (t.role or "user").strip()
        content = (t.content or "").strip()
        lines.append(f"[{role}] {content}")
    return _SUMMARY_INSTRUCTION_TEMPLATE.format(
        conversation="\n".join(lines),
    )


# ----------------------------------------------------------------------
# Data carriers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TurnSnapshot:
    """Minimal turn shape the summarizer needs.

    Decoupled from :class:`ultron.memory.qdrant_store.MemoryTurn` so
    tests can construct snapshots without instantiating the full
    memory store. The orchestrator constructs these from
    ``ConversationMemory.recent`` output.
    """

    turn_id: int
    ts: float
    role: str
    content: str


@dataclass(frozen=True)
class FactEntry:
    """One extracted fact."""

    subject: str
    predicate: str
    object: str

    def to_text(self) -> str:
        """Human-readable form used both for embedding + storage."""
        return f"{self.subject} {self.predicate} {self.object}".strip()


@dataclass(frozen=True)
class DecisionEntry:
    """One extracted decision."""

    topic: str
    outcome: str
    status: str

    def to_text(self) -> str:
        return f"{self.topic}: {self.outcome} (status: {self.status})".strip()


@dataclass(frozen=True)
class PreferenceEntry:
    """One extracted user preference."""

    topic: str
    value: str

    def to_text(self) -> str:
        return f"User preference -- {self.topic}: {self.value}".strip()


@dataclass(frozen=True)
class SummaryResult:
    """Full output of one summarizer pass."""

    summary: str
    facts: List[FactEntry] = field(default_factory=list)
    decisions: List[DecisionEntry] = field(default_factory=list)
    preferences: List[PreferenceEntry] = field(default_factory=list)
    turn_id_start: int = 0
    turn_id_end: int = 0
    span_seconds: float = 0.0

    @property
    def is_empty(self) -> bool:
        """True when nothing useful was extracted -- safe to discard."""
        return (
            not self.summary.strip()
            and not self.facts
            and not self.decisions
            and not self.preferences
        )


# ----------------------------------------------------------------------
# JSON parsing -- defensive against model-emitted markdown / preamble
# ----------------------------------------------------------------------


# Detect a fenced ```json ... ``` block or a bare leading "{" up
# through its balanced closing "}". The model is instructed to emit
# JSON only, but smaller models sometimes wrap it in markdown.
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)


def _extract_json_object(text: str) -> Optional[str]:
    """Return the JSON object substring from ``text``, or None.

    Tries fenced blocks first, then a brace-balanced scan from the
    first ``{``. Defensive against trailing prose ("Here's the
    summary:") that small models sometimes prepend / append.
    """
    if not text:
        return None
    # 1. Fenced markdown block.
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    # 2. Brace-balanced scan.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def parse_summary_response(
    response_text: str,
    *,
    turn_id_start: int = 0,
    turn_id_end: int = 0,
    span_seconds: float = 0.0,
) -> Optional[SummaryResult]:
    """Parse the LLM's JSON response into a :class:`SummaryResult`.

    Returns None on unparseable or empty responses. Fail-open by
    construction; the caller decides whether a None is OK to ignore.

    Schema-tolerant: missing keys default to empty / empty-list.
    Extra fields are ignored. Per-item type errors skip the bad
    entry without dropping the whole parse.
    """
    if not response_text:
        return None
    payload_str = _extract_json_object(response_text)
    if payload_str is None:
        return None
    try:
        payload = json.loads(payload_str)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    summary = str(payload.get("summary", "") or "").strip()
    facts = _parse_fact_list(payload.get("facts"))
    decisions = _parse_decision_list(payload.get("decisions"))
    preferences = _parse_preference_list(payload.get("preferences"))

    return SummaryResult(
        summary=summary,
        facts=facts,
        decisions=decisions,
        preferences=preferences,
        turn_id_start=turn_id_start,
        turn_id_end=turn_id_end,
        span_seconds=span_seconds,
    )


def _safe_str(value: Any, *, max_chars: int = 240) -> str:
    """Coerce arbitrary JSON values into a short trimmed string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:max_chars]
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_safe_str(v, max_chars=80) for v in value)[:max_chars]
    if isinstance(value, dict):
        return _safe_str(value.get("text") or value.get("value") or "", max_chars=max_chars)
    return ""


def _parse_fact_list(raw: Any) -> List[FactEntry]:
    if not isinstance(raw, list):
        return []
    out: List[FactEntry] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        subj = _safe_str(entry.get("subject"))
        pred = _safe_str(entry.get("predicate"))
        obj = _safe_str(entry.get("object"))
        if not subj and not obj:
            continue
        out.append(FactEntry(subject=subj, predicate=pred, object=obj))
    return out


def _parse_decision_list(raw: Any) -> List[DecisionEntry]:
    if not isinstance(raw, list):
        return []
    out: List[DecisionEntry] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        topic = _safe_str(entry.get("topic"))
        outcome = _safe_str(entry.get("outcome"))
        status = _safe_str(entry.get("status"))
        if not topic and not outcome:
            continue
        # Normalise common status synonyms.
        norm_status = status.lower().strip()
        if norm_status in {"open", "considering", "tentative"}:
            norm_status = "pending"
        elif norm_status in {"committed", "decided", "finalized", "done"}:
            norm_status = "made"
        elif norm_status in {"unmade", "rolled back", "rolled-back"}:
            norm_status = "reversed"
        elif not norm_status:
            norm_status = "made"
        out.append(DecisionEntry(
            topic=topic, outcome=outcome, status=norm_status,
        ))
    return out


def _parse_preference_list(raw: Any) -> List[PreferenceEntry]:
    if not isinstance(raw, list):
        return []
    out: List[PreferenceEntry] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        topic = _safe_str(entry.get("topic"))
        value = _safe_str(entry.get("value"))
        if not topic and not value:
            continue
        out.append(PreferenceEntry(topic=topic, value=value))
    return out


# ----------------------------------------------------------------------
# BackgroundSummarizer -- the stateful orchestrator-facing class
# ----------------------------------------------------------------------


# Type alias for the LLM call -- a callable that takes (prompt) and
# returns the generated text. Decoupled from any specific LLM client
# so tests can inject stubs.
GenerateFn = Callable[[str], str]


# Type alias for the storage hook -- a callable invoked with the
# parsed SummaryResult so the orchestrator can write entries to
# Qdrant (or any persistence layer). Tests inject a no-op or a
# capture-and-assert spy.
StoreFn = Callable[["SummaryResult"], None]


class BackgroundSummarizer:
    """Periodic conversation summarizer + fact extractor.

    Designed to be called from the orchestrator's idle path
    (post-turn, mid-silence). The orchestrator decides WHEN to call
    :meth:`maybe_summarize`; the summarizer decides whether the
    conditions are right and runs the LLM call if so.

    Args:
        generate_fn: callable that takes a prompt string and returns
            the LLM response. Typically ``llm_engine.generate``.
        store_fn: callable invoked with the parsed
            :class:`SummaryResult` so the caller can persist it.
            If None, the result is discarded (useful for testing).
        recent_turns_fn: callable that returns the list of recent
            ``TurnSnapshot`` objects since the last summary. The
            orchestrator typically wraps ``ConversationMemory.recent``.
        cadence_turns: trigger every N new turns since the last
            summary. Default 10.
        min_turns: skip the call when fewer than this many turns
            have accumulated since the last summary. Default 3.
        idle_threshold_seconds: minimum gap since the last
            foreground activity before the summarizer is allowed to
            run. Caller passes the timestamp into
            :meth:`maybe_summarize`.
        now_provider: injectable clock for tests.
    """

    def __init__(
        self,
        *,
        generate_fn: GenerateFn,
        store_fn: Optional[StoreFn] = None,
        recent_turns_fn: Callable[[], Sequence[TurnSnapshot]],
        cadence_turns: int = 10,
        min_turns: int = 3,
        idle_threshold_seconds: float = 30.0,
        now_provider: Callable[[], float] = time.monotonic,
        compress_summarize_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._store_fn = store_fn
        self._recent_turns_fn = recent_turns_fn
        self._cadence_turns = max(1, int(cadence_turns))
        self._min_turns = max(1, int(min_turns))
        self._idle_threshold_seconds = float(idle_threshold_seconds)
        self._now = now_provider
        self._lock = threading.Lock()
        self._last_summarized_turn_id: int = -1
        self._cancel_event = threading.Event()
        self._in_flight = False
        # 2026-05-22 catalog batch 3: separate free-form summarisation
        # callable for the tail-preserve history compression path. Kept
        # distinct from ``generate_fn`` (which expects JSON-structured
        # output for fact extraction) so the two LLM call shapes don't
        # bleed into each other. Falls back to ``generate_fn`` when
        # only one was wired, which is safe because the compression
        # caller is fault-tolerant -- a mangled JSON-looking summary
        # is rejected by ``compress_history``'s empty-summary guard.
        self._compress_summarize_fn = compress_summarize_fn or generate_fn
        # SnapshotGuard reused across compression jobs so callers can
        # opt-in to cross-job race tracking.
        from ultron.utils.snapshot_guard import SnapshotGuard
        self._compression_guard = SnapshotGuard()

    # ------------------------------------------------------------------

    @property
    def last_summarized_turn_id(self) -> int:
        with self._lock:
            return self._last_summarized_turn_id

    def cancel(self) -> None:
        """Signal an in-flight summarizer call to abort.

        Idempotent. The cancel flag is read between LLM calls and at
        the end of parse / store -- the summarizer can be aborted at
        any sub-step, including mid-LLM-generation if the LLM client
        respects the cancel event (most do not natively; orchestrator
        wires its own llm.cancel() on top).
        """
        self._cancel_event.set()

    def reset_cancel(self) -> None:
        """Clear the cancel flag before the next call.

        :meth:`maybe_summarize` clears the flag on entry, so this is
        usually unnecessary. Exposed for tests that need to reset
        state between simulated cancellations.
        """
        self._cancel_event.clear()

    def maybe_summarize(
        self,
        *,
        last_activity_monotonic: float,
    ) -> Optional[SummaryResult]:
        """Run a summarizer pass if the gating conditions are met.

        Returns the :class:`SummaryResult` on success, None when the
        gate said "not now" or any internal step failed.

        Gating order:

        1. Cancel flag already set -> skip (caller pre-cancelled).
        2. Idle threshold not satisfied -> skip.
        3. Fewer than ``min_turns`` accumulated since last summary -> skip.
        4. New turns < cadence threshold -> skip.

        On a successful pass, ``_last_summarized_turn_id`` advances
        to the highest turn id seen, so the next pass only considers
        the freshly-added turns.

        Thread-safe: the in_flight guard prevents two concurrent
        calls overlapping; the second caller short-circuits to None.
        """
        with self._lock:
            if self._in_flight:
                return None
            if self._cancel_event.is_set():
                self._cancel_event.clear()
                # Don't run if the previous call was cancelled;
                # caller can retry on the next idle tick.
                return None

            now = self._now()
            if (now - last_activity_monotonic) < self._idle_threshold_seconds:
                return None

            turns_all = list(self._recent_turns_fn() or [])
            new_turns = [
                t for t in turns_all
                if t.turn_id > self._last_summarized_turn_id
            ]
            if len(new_turns) < self._min_turns:
                return None
            if len(new_turns) < self._cadence_turns:
                # Not enough turns since the last pass -- wait.
                return None

            self._in_flight = True

        try:
            return self._run(new_turns)
        finally:
            with self._lock:
                self._in_flight = False

    def force_run(
        self,
        turns: Optional[Sequence[TurnSnapshot]] = None,
    ) -> Optional[SummaryResult]:
        """Bypass all gates and run the summarizer now.

        Used by tests + by operators who want to flush a summary
        regardless of cadence / idle. Returns the same result type as
        :meth:`maybe_summarize`.
        """
        with self._lock:
            if self._in_flight:
                return None
            self._in_flight = True
        try:
            payload = list(turns) if turns is not None else list(
                self._recent_turns_fn() or []
            )
            if not payload:
                return None
            return self._run(payload)
        finally:
            with self._lock:
                self._in_flight = False

    # ------------------------------------------------------------------

    def _run(self, turns: Sequence[TurnSnapshot]) -> Optional[SummaryResult]:
        """Execute one summarizer pass on the given turn slice."""
        if not turns:
            return None
        if self._cancel_event.is_set():
            self._cancel_event.clear()
            return None

        prompt = render_summary_prompt(turns)

        try:
            raw = self._generate_fn(prompt)
        except Exception as e:                                # noqa: BLE001
            # Fail-open. The summarizer's value is cumulative; a
            # missed pass costs only what the next pass will pick up.
            from ultron.utils.logging import get_logger
            get_logger("memory.background_summarizer").warning(
                "Summarizer LLM call failed (%s); discarding pass.", e,
            )
            return None

        if self._cancel_event.is_set():
            self._cancel_event.clear()
            return None

        turn_id_start = turns[0].turn_id
        turn_id_end = turns[-1].turn_id
        span_seconds = max(0.0, turns[-1].ts - turns[0].ts)

        result = parse_summary_response(
            raw,
            turn_id_start=turn_id_start,
            turn_id_end=turn_id_end,
            span_seconds=span_seconds,
        )

        if result is None or result.is_empty:
            # Don't advance the watermark -- the caller can retry
            # with more turns next time.
            return None

        # Persist via the storage hook (fail-open).
        if self._store_fn is not None:
            try:
                self._store_fn(result)
            except Exception as e:                            # noqa: BLE001
                from ultron.utils.logging import get_logger
                get_logger("memory.background_summarizer").warning(
                    "Summarizer store hook failed (%s); result lost.", e,
                )
                # Even on store failure we advance the watermark --
                # the LLM call already burned tokens, and the next
                # pass overlapping these turns would duplicate work.

        with self._lock:
            self._last_summarized_turn_id = max(
                self._last_summarized_turn_id, turn_id_end,
            )
        return result

    # ------------------------------------------------------------------
    # 2026-05-22 catalog batch 3: tail-preserve history compression
    # ------------------------------------------------------------------

    def compress_history_for_llm(
        self,
        messages: Sequence[Any],
        *,
        max_tokens: int,
        token_counter: Optional[Callable[[str], int]] = None,
        max_depth: int = 3,
        snapshot_key: str = "history_compression",
    ) -> Any:
        """Run a tail-preserve compression pass on ``messages``.

        Returns a
        :class:`ultron.memory.history_compression.CompressionResult`.
        The compressed list (if any) is the new history the caller
        should feed into the next LLM prompt. When
        ``result.compressed is None``, leave the live history as-is.

        Race-protected via the internal :class:`SnapshotGuard`: if
        ``messages`` mutates structurally during the summarisation
        LLM call (foreground appended a turn, etc.), the result is
        discarded and ``result.race_detected`` is True.

        Args:
            messages: Live list of message dicts. Pass the same list
                the caller will mutate; snapshot is taken internally.
            max_tokens: Target token budget for the compressed
                history.
            token_counter: Callable mapping text -> token count. When
                omitted, uses
                :func:`ultron.utils.token_budget.char_count_tokens`
                (length / 4 heuristic).
            max_depth: Forwarded to ``compress_history_recursive``.
            snapshot_key: Used by the internal ``SnapshotGuard`` so
                concurrent compression jobs don't trample each other.

        Failure modes (all return a CompressionResult with
        ``compressed=None``):
          * LLM call raises -> ``error="<exception>"``.
          * LLM returns empty -> ``error="empty summary"``.
          * Race detected -> ``race_detected=True``.
        """
        from ultron.memory.history_compression import (
            compress_history_with_guard,
            messages_to_dicts,
        )

        if token_counter is None:
            from ultron.utils.token_budget import char_count_tokens
            token_counter = char_count_tokens

        normalised = messages_to_dicts(messages)
        return compress_history_with_guard(
            normalised,
            self._compress_summarize_fn,
            max_tokens=max_tokens,
            token_counter=token_counter,
            guard=self._compression_guard,
            key=snapshot_key,
            max_depth=max_depth,
        )


__all__ = [
    "BackgroundSummarizer",
    "DecisionEntry",
    "FactEntry",
    "PreferenceEntry",
    "SummaryResult",
    "TurnSnapshot",
    "parse_summary_response",
    "render_summary_prompt",
]
