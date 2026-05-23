"""Helpers for subsystems to emit canonical observations.

Each call site imports the helper that matches its event and passes
structural data; the helper handles schema assembly + writer dispatch
in one place. This keeps subsystem modules free of observation-schema
knowledge -- if the schema changes, only this file moves.

Every helper:

* Returns the ``event_id`` of the emitted observation (or ``None`` if
  the writer is disabled / failed) so callers can chain
  ``parent_event_id`` into downstream observations.
* Is fail-open. The writer never raises; helpers never raise either.
* Stamps ``outcome='unknown_yet'`` by default. The outcome resolver
  tagging pass (separate module) writes resolution rows referencing
  the original ``event_id`` once the verdict (success / corrected /
  failed) becomes apparent.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .schema import Observation
from .writer import emit_observation


def observe_routing_verdict(
    *,
    utterance: str,
    intent_kind: str,
    confidence: float,
    source: str,
    reason: str,
    latency_ms: float,
    parent_event_id: Optional[str] = None,
) -> Optional[str]:
    """Emit one observation for a routing classifier verdict.

    ``utterance`` content is NOT stored -- only its length -- to keep
    the observation file low-cardinality and to leave full transcripts
    in their existing per-subsystem logs.
    """
    obs = Observation.create(
        subsystem="routing",
        event_type="classify_routing",
        intent_kind=intent_kind,
        latency_ms=latency_ms,
        outcome="unknown_yet",
        parent_event_id=parent_event_id,
        extra={
            "utterance_len": len(utterance or ""),
            "confidence": float(confidence),
            "source": source,
            "reason": reason,
        },
    )
    return obs.event_id if emit_observation(obs) else None


def observe_addressing_verdict(
    *,
    utterance: str,
    decision: str,
    confidence: float,
    reason: str,
    seconds_since_response: float,
    source: str,
    latency_ms: float,
    parent_event_id: Optional[str] = None,
) -> Optional[str]:
    """Emit one observation for an addressing classifier verdict."""
    obs = Observation.create(
        subsystem="addressing",
        event_type="classify_addressing",
        latency_ms=latency_ms,
        outcome="unknown_yet",
        parent_event_id=parent_event_id,
        extra={
            "utterance_len": len(utterance or ""),
            "decision": decision,
            "confidence": float(confidence),
            "reason": reason,
            "seconds_since_response": float(seconds_since_response),
            "classifier_source": source,
        },
    )
    return obs.event_id if emit_observation(obs) else None


def observe_retrieval(
    *,
    query: str,
    lineage_ids: Iterable[str],
    k: int,
    latency_ms: float,
    collection: str = "conversations",
    parent_event_id: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Emit one observation for a memory retrieval call.

    ``lineage_ids`` is the list of memory ids returned (or considered).
    Downstream consumers compare against response content via the
    post-response overlap pass to mark which were actually used.
    """
    base_extra: dict[str, Any] = {
        "query_len": len(query or ""),
        "k": int(k),
        "collection": collection,
        "result_count": sum(1 for _ in lineage_ids) if isinstance(lineage_ids, list) else None,
    }
    # Re-iterate lineage_ids since we may have consumed it for the count.
    ids_tuple = tuple(lineage_ids) if not isinstance(lineage_ids, tuple) else lineage_ids
    base_extra["result_count"] = len(ids_tuple)
    if extra:
        base_extra.update(extra)
    obs = Observation.create(
        subsystem="memory",
        event_type="retrieve",
        latency_ms=latency_ms,
        outcome="unknown_yet",
        lineage_ids=ids_tuple,
        parent_event_id=parent_event_id,
        extra=base_extra,
    )
    return obs.event_id if emit_observation(obs) else None


def observe_llm_call(
    *,
    event_type: str,
    user_message_len: int,
    tokens_used: Optional[int],
    latency_ms: float,
    streamed: bool,
    enable_thinking: Optional[bool] = None,
    parent_event_id: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Emit one observation for an LLM call.

    ``event_type`` distinguishes ``"generate"`` (blocking) from
    ``"generate_stream"`` (token stream) and any preflight variants.
    """
    base_extra: dict[str, Any] = {
        "user_message_len": int(user_message_len),
        "streamed": bool(streamed),
    }
    if enable_thinking is not None:
        base_extra["enable_thinking"] = bool(enable_thinking)
    if extra:
        base_extra.update(extra)
    obs = Observation.create(
        subsystem="llm",
        event_type=event_type,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        outcome="unknown_yet",
        parent_event_id=parent_event_id,
        extra=base_extra,
    )
    return obs.event_id if emit_observation(obs) else None


def observe_llm_thinking_drift_sample(
    *,
    user_text: str,
    response_text: str,
    user_message_len: int,
    response_message_len: int,
    parent_event_id: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Emit a sampled observation pairing user text with the no-think response.

    Stored verbatim (truncated) so a future review pass can spot
    regressions when ``enable_thinking=False`` is the active default.
    Production sampling is gated by
    :class:`ultron.config.LLMConfig.enable_thinking_drift_sample_rate`;
    when the dice roll lands the orchestrator calls this helper. The
    helper itself does no sampling -- the caller decides whether to
    emit.

    Texts longer than 4000 chars are truncated with an explicit
    ``"... <truncated>"`` marker so the observation file doesn't bloat
    on long search-augmented responses.

    Returns the event_id on success, None when the writer is
    disabled / fails (fail-open).
    """
    def _truncate(text: str, cap: int = 4000) -> str:
        if text is None:
            return ""
        if len(text) <= cap:
            return text
        return text[:cap] + "... <truncated>"

    base_extra: dict[str, Any] = {
        "user_text": _truncate(user_text),
        "response_text": _truncate(response_text),
        "user_message_len": int(user_message_len),
        "response_message_len": int(response_message_len),
        "enable_thinking": False,
    }
    if extra:
        base_extra.update(extra)
    obs = Observation.create(
        subsystem="llm",
        event_type="thinking_drift_sample",
        outcome="unknown_yet",
        parent_event_id=parent_event_id,
        extra=base_extra,
    )
    return obs.event_id if emit_observation(obs) else None
