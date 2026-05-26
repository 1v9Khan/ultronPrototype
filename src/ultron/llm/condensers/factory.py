"""Factory + intent-adaptive selector for condensers."""

from __future__ import annotations

import logging
from typing import Any, Callable

from ultron.llm.condensers.amortized import AmortizedCondenser
from ultron.llm.condensers.base import Condenser, CondenserError
from ultron.llm.condensers.llm_summarizing import LLMSummarizingCondenser
from ultron.llm.condensers.noop import NoOpCondenser
from ultron.llm.condensers.observation_masking import ObservationMaskingCondenser
from ultron.llm.condensers.recent import RecentCondenser

logger = logging.getLogger(__name__)


DEFAULT_CONDENSER_KIND = "noop"
KNOWN_CONDENSER_KINDS: tuple[str, ...] = (
    "noop",
    "recent",
    "amortized",
    "observation_masking",
    "llm_summarizing",
)


def build_condenser(
    kind: str,
    *,
    keep_first: int | None = None,
    max_events: int | None = None,
    max_size: int | None = None,
    max_tokens: int | None = None,
    attention_window: int | None = None,
    summarize_fn: Callable[[str], str] | None = None,
    summary_preamble: str = "",
) -> Condenser:
    """Construct a :class:`Condenser` from a config-string kind + knobs.

    Unknown kinds raise :class:`CondenserError`. Missing knobs fall to
    each concrete's documented defaults.
    """

    normalised = (kind or DEFAULT_CONDENSER_KIND).strip().lower()
    if normalised in ("", "none", "off", "noop"):
        return NoOpCondenser()
    if normalised == "recent":
        kwargs: dict[str, Any] = {}
        if keep_first is not None:
            kwargs["keep_first"] = keep_first
        if max_events is not None:
            kwargs["max_events"] = max_events
        return RecentCondenser(**kwargs)
    if normalised == "amortized":
        kwargs = {}
        if keep_first is not None:
            kwargs["keep_first"] = keep_first
        if max_size is not None:
            kwargs["max_size"] = max_size
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return AmortizedCondenser(**kwargs)
    if normalised in ("observation_masking", "mask"):
        kwargs = {}
        if attention_window is not None:
            kwargs["attention_window"] = attention_window
        return ObservationMaskingCondenser(**kwargs)
    if normalised in ("llm_summarizing", "llm_summary", "summary"):
        kwargs = {}
        if max_size is not None:
            kwargs["max_size"] = max_size
        if keep_first is not None:
            kwargs["keep_first"] = keep_first
        if summarize_fn is not None:
            kwargs["summarize_fn"] = summarize_fn
        if summary_preamble:
            kwargs["summary_preamble"] = summary_preamble
        return LLMSummarizingCondenser(**kwargs)
    raise CondenserError(f"unknown condenser kind: {kind!r}")


# -- intent-adaptive selector --


_INTENT_KIND_MAP: dict[str, str] = {
    # Voice-path intents: lean strategies for the latency budget.
    # ``noop`` is a zero-cost passthrough -- no per-turn churn, no
    # synthetic summary turn, no prompt-cache invalidation. Use it for
    # short conversational turns and lightweight quick-probe surfaces.
    "greeting": "noop",
    "ack": "noop",
    "conversational": "noop",
    "factual": "recent",
    "memory_recall": "amortized",
    "gaming": "recent",
    "gaming_mode": "noop",
    "system_status": "noop",
    "progress_query": "noop",
    "cancel": "noop",
    "mid_session_adjustment": "noop",
    "clarification_response": "noop",
    "model_switch": "noop",
    "active_window_query": "noop",
    "window_close_confirmation": "noop",
    # Coding-path intents: aggressive summarization OK because the
    # LLM call is the bottleneck, not the latency budget.
    "coding": "llm_summarizing",
    "refactor": "llm_summarizing",
    "code_task": "llm_summarizing",
    "hybrid_task": "llm_summarizing",
    # Desktop automation / window operations: lean recent-window shape;
    # the user typically threads a multi-step automation through
    # several recent turns but rarely re-reads older context.
    "browser_automation": "recent",
    "media_generation": "recent",
    "messaging": "recent",
    "file_operation": "recent",
    "shell_operation": "recent",
    "desktop_automation": "recent",
    "window_automation": "recent",
    "app_launch": "recent",
    "screen_context_query": "recent",
    "window_move": "recent",
    "window_close": "recent",
    "open_last_source": "recent",
    "navigate_to_site": "recent",
    "semantic_click": "recent",
    # Default for unknown intents.
    "default": "recent",
}


def select_condenser_for_intent(
    intent: str | None,
    *,
    fallback: Condenser | None = None,
    summarize_fn: Callable[[str], str] | None = None,
) -> Condenser:
    """Return a condenser tailored to ``intent``.

    Maps a small set of intent labels to a recommended strategy
    matching the catalog's "adaptive switching by intent" extension.
    Unknown intents fall to ``fallback`` (or a fresh :class:`RecentCondenser`).
    """

    if intent is None:
        return fallback if fallback is not None else RecentCondenser()
    kind = _INTENT_KIND_MAP.get(intent.lower().strip(), _INTENT_KIND_MAP["default"])
    try:
        return build_condenser(kind, summarize_fn=summarize_fn)
    except CondenserError:
        logger.warning(
            "select_condenser_for_intent: unknown kind %r for intent %r; "
            "falling back to NoOp",
            kind,
            intent,
        )
        return NoOpCondenser()
