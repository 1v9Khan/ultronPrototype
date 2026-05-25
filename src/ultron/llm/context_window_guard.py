"""Multi-tier context-window guard.

T4 (OpenClaw catalog port; see ``THIRD_PARTY_NOTICES.md``). Resolves
the effective LLM context window from multiple sources, ranks them
by precedence, and produces a guard result with:

* ``hard_min_tokens`` — below this, the session refuses to start.
* ``warn_below_tokens`` — log a WARN; agent still runs.
* ``source`` — which layer won (caller_override / models_config /
  default / agent_cap).
* Format helpers emit operator-friendly messages that distinguish
  self-hosted (raise your server's ``--n-ctx``), configured (raise
  ``n_ctx`` in ``config.yaml``), or agent-capped (raise the agent
  budget).

The guard fires at orchestrator startup. A failed evaluation can
either block startup (when the resolved budget is below the hard
floor) or just emit a WARN (when below the soft floor). Voice-path
default of ``qwen3.5-4b`` at ``n_ctx=8192`` clears both thresholds
comfortably; the guard is primarily a catch for operator
misconfiguration (someone setting ``n_ctx=2048`` thinking it speeds
startup, hitting a sub-floor block with a clear diagnostic instead
of a confusing in-session crash).

The same pattern generalises beyond the LLM context window to
embedding-window, STT-window, TTS character cap, Qdrant cardinality
checks, disk space, VRAM peaks — see the catalog for the creative
extensions. This module provides the LLM-context core; callers map
the same shape onto other budgets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

LOGGER = logging.getLogger(__name__)

#: Absolute floor: below this, no LLM-driven turn can be expected to
#: produce a useful result. Mirrors OpenClaw's 4000-token floor.
DEFAULT_HARD_MIN_TOKENS: int = 4000

#: Soft floor: warn but allow. Sessions running this close to the
#: floor will trigger aggressive compaction (or fail mid-turn) on
#: real-world prompts. Mirrors OpenClaw's 8000-token warn.
DEFAULT_WARN_BELOW_TOKENS: int = 8000

#: Dynamic-ratio multipliers: when the configured context window is
#: very large, scale the floors up so the absolute numbers stay
#: meaningful (e.g. a 200k-token window deserves a 20k-token warn,
#: not 8k). Mirrors OpenClaw's `max(N, ctx * ratio)` resolution.
DEFAULT_HARD_MIN_RATIO: float = 0.10
DEFAULT_WARN_RATIO: float = 0.20


class ContextWindowSource(str, Enum):
    """Which source layer produced the effective context window.

    Used by the format helpers to tailor the operator-facing
    diagnostic ("raise your server's --n-ctx" vs "raise contextWindow
    in config.yaml").
    """

    CALLER_OVERRIDE = "caller_override"
    MODELS_CONFIG = "models_config"
    DEFAULT = "default"
    AGENT_CAP = "agent_cap"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ContextWindowInfo:
    """Resolved effective context window + provenance.

    Attributes:
        tokens: The effective token budget after all resolution.
            ``None`` when no source provided a value.
        source: Which layer produced ``tokens``.
        reference_tokens: When ``source == AGENT_CAP``, the original
            (un-capped) value the model could have offered. Lets the
            formatter say "your model supports 32k but you've capped
            the agent at 8k".
    """

    tokens: Optional[int]
    source: ContextWindowSource
    reference_tokens: Optional[int] = None


@dataclass(frozen=True)
class ContextWindowThresholds:
    """Computed warn + hard-min thresholds for an effective budget."""

    hard_min_tokens: int
    warn_below_tokens: int


@dataclass(frozen=True)
class ContextWindowGuardResult:
    """Outcome of a single guard evaluation.

    Attributes:
        info: The resolved :class:`ContextWindowInfo`.
        thresholds: The computed :class:`ContextWindowThresholds`.
        should_block: ``True`` when the resolved budget is below the
            hard floor (or unresolvable). The orchestrator should
            abort startup with the formatted block message.
        should_warn: ``True`` when below the soft floor. Log only;
            do not abort.
        block_message: Operator-facing diagnostic for the block case.
        warn_message: Operator-facing diagnostic for the warn case.
    """

    info: ContextWindowInfo
    thresholds: ContextWindowThresholds
    should_block: bool
    should_warn: bool
    block_message: str = ""
    warn_message: str = ""


def _normalise_positive_int(value: object) -> Optional[int]:
    """Coerce ``value`` to a positive ``int`` or ``None``.

    Accepts ``int`` / ``float`` (rounded down). Rejects negatives,
    zero, NaN / inf, and non-numeric types. Used to clean every
    caller-supplied or config-loaded token budget so downstream
    arithmetic isn't surprised by ``None`` or string inputs.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is a subclass of int but never a meaningful budget.
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN check
            return None
        if f <= 0 or f == float("inf"):
            return None
        return int(f)
    return None


def resolve_context_window_info(
    *,
    caller_override_tokens: Optional[int] = None,
    models_config_tokens: Optional[int] = None,
    default_tokens: Optional[int] = None,
    agent_cap_tokens: Optional[int] = None,
) -> ContextWindowInfo:
    """Resolve effective context window from priority-ordered sources.

    Priority (first non-empty wins):

    1. ``caller_override_tokens`` — explicit override from the call
       site (e.g. ``--n-ctx-override`` CLI flag).
    2. ``models_config_tokens`` — the value in ``config.yaml`` (e.g.
       ``llm.n_ctx`` resolved against the preset table).
    3. ``default_tokens`` — orchestrator's fallback default.

    After the base value is picked, ``agent_cap_tokens`` is applied
    as a ceiling: if the agent cap is lower than the base, the result
    is capped + the source is rewritten to :attr:`ContextWindowSource.AGENT_CAP`
    with ``reference_tokens`` set to the original base.

    Args:
        caller_override_tokens: Explicit per-call override.
        models_config_tokens: Value from the model registry.
        default_tokens: Orchestrator default fallback.
        agent_cap_tokens: Per-agent ceiling.

    Returns:
        :class:`ContextWindowInfo` with the resolved value, source,
        and (when capped) the original reference.
    """
    co = _normalise_positive_int(caller_override_tokens)
    mc = _normalise_positive_int(models_config_tokens)
    df = _normalise_positive_int(default_tokens)
    cap = _normalise_positive_int(agent_cap_tokens)

    base: Optional[int]
    source: ContextWindowSource
    if co is not None:
        base, source = co, ContextWindowSource.CALLER_OVERRIDE
    elif mc is not None:
        base, source = mc, ContextWindowSource.MODELS_CONFIG
    elif df is not None:
        base, source = df, ContextWindowSource.DEFAULT
    else:
        base, source = None, ContextWindowSource.UNKNOWN

    if base is None:
        return ContextWindowInfo(tokens=None, source=source)

    if cap is not None and cap < base:
        return ContextWindowInfo(
            tokens=cap,
            source=ContextWindowSource.AGENT_CAP,
            reference_tokens=base,
        )
    return ContextWindowInfo(tokens=base, source=source)


def resolve_thresholds(
    tokens: Optional[int],
    *,
    hard_min_tokens: int = DEFAULT_HARD_MIN_TOKENS,
    warn_below_tokens: int = DEFAULT_WARN_BELOW_TOKENS,
    hard_min_ratio: float = DEFAULT_HARD_MIN_RATIO,
    warn_ratio: float = DEFAULT_WARN_RATIO,
) -> ContextWindowThresholds:
    """Compute dynamic hard-min + warn-below thresholds for ``tokens``.

    Returns ``max(absolute_floor, tokens * ratio)`` for each. With
    ``tokens=None`` (no source), returns the absolute floors so the
    guard still fires.
    """
    if tokens is None or tokens <= 0:
        return ContextWindowThresholds(
            hard_min_tokens=int(hard_min_tokens),
            warn_below_tokens=int(warn_below_tokens),
        )
    return ContextWindowThresholds(
        hard_min_tokens=max(int(hard_min_tokens), int(tokens * hard_min_ratio)),
        warn_below_tokens=max(int(warn_below_tokens), int(tokens * warn_ratio)),
    )


def format_block_message(info: ContextWindowInfo, thresholds: ContextWindowThresholds, *, hint: str = "") -> str:
    """Operator-facing block message tailored to ``info.source``.

    Args:
        info: The resolved context-window info.
        thresholds: The thresholds the budget failed.
        hint: Optional environment hint ("self-hosted" / "managed").
            Appended to the suggestion when present.
    """
    base = (
        f"LLM context window too small: resolved {info.tokens} tokens "
        f"(source={info.source.value}); minimum is {thresholds.hard_min_tokens}. "
    )
    if info.source == ContextWindowSource.AGENT_CAP:
        suggest = (
            "The agent cap is below the model's offered window "
            f"(model={info.reference_tokens}); raise the agent cap or "
            "remove it from config."
        )
    elif info.source == ContextWindowSource.MODELS_CONFIG:
        suggest = (
            "Raise the n_ctx value in config.yaml or pick a preset with a "
            "larger default."
        )
    elif info.source == ContextWindowSource.CALLER_OVERRIDE:
        suggest = (
            "The explicit override is below the floor; remove it or pass a "
            "larger value."
        )
    elif info.source == ContextWindowSource.UNKNOWN:
        suggest = (
            "No source provided a value; check the LLM preset is wired "
            "correctly."
        )
    else:
        suggest = "Adjust the relevant configuration source."
    if hint:
        suggest = f"{suggest} ({hint})"
    return base + suggest


def format_warn_message(info: ContextWindowInfo, thresholds: ContextWindowThresholds) -> str:
    """Operator-facing warn message when budget is below the soft floor."""
    return (
        f"LLM context window close to the floor: resolved {info.tokens} tokens "
        f"(source={info.source.value}); recommended minimum is "
        f"{thresholds.warn_below_tokens}. Compaction may be aggressive "
        "on real-world prompts."
    )


def evaluate_context_window_guard(
    *,
    caller_override_tokens: Optional[int] = None,
    models_config_tokens: Optional[int] = None,
    default_tokens: Optional[int] = None,
    agent_cap_tokens: Optional[int] = None,
    hard_min_tokens: int = DEFAULT_HARD_MIN_TOKENS,
    warn_below_tokens: int = DEFAULT_WARN_BELOW_TOKENS,
    hard_min_ratio: float = DEFAULT_HARD_MIN_RATIO,
    warn_ratio: float = DEFAULT_WARN_RATIO,
    environment_hint: str = "",
) -> ContextWindowGuardResult:
    """One-call: resolve sources + compute thresholds + decide block/warn.

    Returns a :class:`ContextWindowGuardResult`. Caller inspects
    ``should_block`` (abort startup) and ``should_warn`` (log).

    Args:
        caller_override_tokens: Caller-supplied override (highest precedence).
        models_config_tokens: Value from the model registry.
        default_tokens: Orchestrator default fallback.
        agent_cap_tokens: Per-agent ceiling.
        hard_min_tokens: Absolute floor (default 4000).
        warn_below_tokens: Soft floor (default 8000).
        hard_min_ratio: Dynamic-floor multiplier (default 0.10).
        warn_ratio: Soft-floor multiplier (default 0.20).
        environment_hint: Optional string appended to messages
            (e.g. ``"self-hosted llama-cpp-server"``).

    Returns:
        :class:`ContextWindowGuardResult` with all info, thresholds,
        and pre-formatted messages.
    """
    info = resolve_context_window_info(
        caller_override_tokens=caller_override_tokens,
        models_config_tokens=models_config_tokens,
        default_tokens=default_tokens,
        agent_cap_tokens=agent_cap_tokens,
    )
    thresholds = resolve_thresholds(
        info.tokens,
        hard_min_tokens=hard_min_tokens,
        warn_below_tokens=warn_below_tokens,
        hard_min_ratio=hard_min_ratio,
        warn_ratio=warn_ratio,
    )
    if info.tokens is None:
        should_block = True
        should_warn = False
    else:
        should_block = info.tokens < thresholds.hard_min_tokens
        should_warn = (not should_block) and info.tokens < thresholds.warn_below_tokens
    block_message = format_block_message(info, thresholds, hint=environment_hint) if should_block else ""
    warn_message = format_warn_message(info, thresholds) if should_warn else ""
    return ContextWindowGuardResult(
        info=info,
        thresholds=thresholds,
        should_block=should_block,
        should_warn=should_warn,
        block_message=block_message,
        warn_message=warn_message,
    )


class ContextWindowGuardError(RuntimeError):
    """Raised when ``run_guard_or_raise`` decides to block."""


def run_guard_or_raise(
    *,
    caller_override_tokens: Optional[int] = None,
    models_config_tokens: Optional[int] = None,
    default_tokens: Optional[int] = None,
    agent_cap_tokens: Optional[int] = None,
    hard_min_tokens: int = DEFAULT_HARD_MIN_TOKENS,
    warn_below_tokens: int = DEFAULT_WARN_BELOW_TOKENS,
    environment_hint: str = "",
    logger_override: Optional[logging.Logger] = None,
) -> ContextWindowGuardResult:
    """Convenience: evaluate; raise on block; log on warn.

    Use from orchestrator init. On block, raises
    :class:`ContextWindowGuardError` with the formatted message so the
    operator sees the diagnostic immediately and startup aborts. On
    warn, logs WARN via ``logger_override`` (or this module's logger).

    Returns the :class:`ContextWindowGuardResult` for the warn case so
    the caller can inspect it for downstream consumers.
    """
    result = evaluate_context_window_guard(
        caller_override_tokens=caller_override_tokens,
        models_config_tokens=models_config_tokens,
        default_tokens=default_tokens,
        agent_cap_tokens=agent_cap_tokens,
        hard_min_tokens=hard_min_tokens,
        warn_below_tokens=warn_below_tokens,
        environment_hint=environment_hint,
    )
    if result.should_block:
        raise ContextWindowGuardError(result.block_message)
    if result.should_warn:
        log = logger_override or LOGGER
        log.warning(result.warn_message)
    return result


__all__ = [
    "ContextWindowGuardError",
    "ContextWindowGuardResult",
    "ContextWindowInfo",
    "ContextWindowSource",
    "ContextWindowThresholds",
    "DEFAULT_HARD_MIN_RATIO",
    "DEFAULT_HARD_MIN_TOKENS",
    "DEFAULT_WARN_BELOW_TOKENS",
    "DEFAULT_WARN_RATIO",
    "evaluate_context_window_guard",
    "format_block_message",
    "format_warn_message",
    "resolve_context_window_info",
    "resolve_thresholds",
    "run_guard_or_raise",
]
