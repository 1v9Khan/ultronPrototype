"""Generic loop-detection primitive with soft and hard escalation tiers.

Adapted from cline's ``checkRepeatedToolCall`` pattern (Apache 2.0;
see ``THIRD_PARTY_NOTICES.md``). The detector computes a canonical
JSON-serialised signature for an event (tool call, intent classification,
RAG snippet retrieval, ack-phrase selection) and counts consecutive
identical signatures. At :data:`DEFAULT_SOFT_THRESHOLD` it surfaces a
"soft warning" suitable for injecting into the next prompt as a hint;
at :data:`DEFAULT_HARD_THRESHOLD` it surfaces a "hard escalation" the
caller can convert into a user-facing apology + halt.

The detector is generic and stateless across callers. A single instance
is per-stream (per-session, per-channel); callers maintain their own
instances and clear them on stream end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from ultron.llm.response_format import loop_hard_escalation, loop_soft_warning

#: First escalation tier; soft hint injected into the next prompt.
DEFAULT_SOFT_THRESHOLD: int = 3

#: Second escalation tier; halt the loop and surface to the user.
DEFAULT_HARD_THRESHOLD: int = 5

#: Default set of "noisy" parameter keys stripped before signing.
#: These are metadata fields that legitimately vary across otherwise-
#: identical calls (progress markers, timestamps, request ids).
DEFAULT_NOISE_KEYS: frozenset[str] = frozenset({
    "task_progress",
    "timestamp",
    "ts",
    "request_id",
    "turn_id",
    "session_id",
    "trace_id",
    "correlation_id",
})


def tool_call_signature(
    name: str,
    parameters: Optional[Mapping[str, Any]] = None,
    *,
    noise_keys: Iterable[str] = DEFAULT_NOISE_KEYS,
) -> str:
    """Compute a canonical signature for a tool call or event.

    Args:
        name: tool / event name (string identifier).
        parameters: optional mapping of parameter name -> value. Values
            must be JSON-serialisable; non-serialisable values are
            coerced via ``repr``.
        noise_keys: iterable of keys to strip before signing (metadata
            fields that legitimately vary across otherwise-identical
            invocations).

    Returns:
        Stable canonical signature suitable for equality comparison.
    """
    cleaned: dict[str, Any] = {}
    if parameters:
        noise_set = set(noise_keys)
        for key in sorted(parameters.keys()):
            if key in noise_set:
                continue
            cleaned[key] = _coerce_serialisable(parameters[key])
    try:
        params_text = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        # Defensive: shouldn't happen after _coerce_serialisable but
        # never crash on signature compute.
        params_text = repr(cleaned)
    return f"{name}|{params_text}"


def _coerce_serialisable(value: Any) -> Any:
    """Recursively coerce ``value`` into a JSON-serialisable form."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {k: _coerce_serialisable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_coerce_serialisable(v) for v in value]
    return repr(value)


@dataclass
class LoopVerdict:
    """Outcome of a single :meth:`LoopDetector.observe` call.

    Attributes:
        signature: canonical signature of the most-recent event.
        count: consecutive-identical count INCLUDING the latest event.
        soft_warning: hint string when the soft threshold has been hit.
        hard_escalation: halt message when the hard threshold has been
            crossed; the caller should stop the loop and surface to
            the user.
    """

    signature: str
    count: int
    soft_warning: Optional[str] = None
    hard_escalation: Optional[str] = None

    @property
    def should_halt(self) -> bool:
        """True when the verdict carries a hard-escalation message."""
        return self.hard_escalation is not None


class LoopDetector:
    """Stream-local loop detector.

    Args:
        soft_threshold: count at which :attr:`LoopVerdict.soft_warning`
            is populated. Default :data:`DEFAULT_SOFT_THRESHOLD`.
        hard_threshold: count at which :attr:`LoopVerdict.hard_escalation`
            is populated. Default :data:`DEFAULT_HARD_THRESHOLD`. Must
            be > ``soft_threshold``.
        noise_keys: iterable of parameter keys to strip when signing
            (passed through to :func:`tool_call_signature`).

    Raises:
        ValueError: when ``hard_threshold <= soft_threshold`` or either
            is less than 2.
    """

    def __init__(
        self,
        *,
        soft_threshold: int = DEFAULT_SOFT_THRESHOLD,
        hard_threshold: int = DEFAULT_HARD_THRESHOLD,
        noise_keys: Iterable[str] = DEFAULT_NOISE_KEYS,
    ) -> None:
        if soft_threshold < 2 or hard_threshold < 2:
            raise ValueError("thresholds must be >= 2")
        if hard_threshold <= soft_threshold:
            raise ValueError("hard_threshold must exceed soft_threshold")
        self._soft = soft_threshold
        self._hard = hard_threshold
        self._noise_keys = frozenset(noise_keys)
        self._last_signature: Optional[str] = None
        self._count: int = 0
        self._halted: bool = False

    @property
    def consecutive_count(self) -> int:
        """Consecutive-identical count of the most-recent signature."""
        return self._count

    @property
    def last_signature(self) -> Optional[str]:
        """Most-recent canonical signature, or ``None`` before any observe."""
        return self._last_signature

    @property
    def halted(self) -> bool:
        """True after a :attr:`LoopVerdict.hard_escalation` has fired."""
        return self._halted

    def observe(
        self,
        name: str,
        parameters: Optional[Mapping[str, Any]] = None,
        *,
        signature: Optional[str] = None,
    ) -> LoopVerdict:
        """Record an event and return the resulting verdict.

        Args:
            name: tool / event name.
            parameters: optional parameters (passed through to
                :func:`tool_call_signature` when ``signature`` is None).
            signature: optional pre-computed signature (skips the
                hashing step; useful when callers already have one).

        Returns:
            :class:`LoopVerdict` describing the current state.
        """
        if self._halted:
            # Once halted, every subsequent observation returns the same
            # hard verdict so callers cannot accidentally bypass.
            current = signature or tool_call_signature(
                name, parameters, noise_keys=self._noise_keys,
            )
            return LoopVerdict(
                signature=current,
                count=self._count,
                hard_escalation=loop_hard_escalation(current, self._count),
            )
        current = signature or tool_call_signature(
            name, parameters, noise_keys=self._noise_keys,
        )
        if current == self._last_signature:
            self._count += 1
        else:
            self._last_signature = current
            self._count = 1
        verdict = LoopVerdict(signature=current, count=self._count)
        if self._count >= self._hard:
            verdict.hard_escalation = loop_hard_escalation(current, self._count)
            self._halted = True
        elif self._count >= self._soft:
            verdict.soft_warning = loop_soft_warning(current, self._count)
        return verdict

    def reset(self) -> None:
        """Clear the detector's state (e.g. on stream end)."""
        self._last_signature = None
        self._count = 0
        self._halted = False


__all__ = [
    "DEFAULT_HARD_THRESHOLD",
    "DEFAULT_NOISE_KEYS",
    "DEFAULT_SOFT_THRESHOLD",
    "LoopDetector",
    "LoopVerdict",
    "tool_call_signature",
]
