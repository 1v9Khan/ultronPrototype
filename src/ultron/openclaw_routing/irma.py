"""IRMA-style input reformulation for the intent disambiguator.

Per the IRMA paper (arXiv) the disambiguator's accuracy on ambiguous
utterances improves substantially when the LLM sees, alongside the
raw utterance, *context* the model would otherwise have to guess at:

- recently used intent kinds (so the model doesn't suggest a tool that
  just failed two turns ago)
- a one-line summary of the active session state (so it knows which
  pipeline is in flight)
- routing hints / user-specific rules (e.g. "this user uses 'open' to
  mean browser, not file")

This module is the wrapper. The disambiguator calls
``InputReformulator.reformulate(utterance, ...)`` when
``cfg.routing.irma.enabled`` is True; otherwise the raw utterance flows
through unchanged. Default OFF — see the docstring on
:class:`ultron.config.RoutingIRMAConfig`.

Pure CPU; no LLM call here. The reformulator just shapes the prompt.
The disambiguator's downstream LLM call is unchanged in shape (still
``CODING | AUTOMATION | HYBRID | UNCLEAR``); only the context that
prompt sees is enriched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

from ultron.utils.logging import get_logger

logger = get_logger("openclaw_routing.irma")


@dataclass
class RecentDecision:
    """One historical routing decision, used as context for IRMA.

    Mirrors the fields routing_decisions.jsonl emits — caller can
    instantiate from those rows or build directly.
    """
    kind: str  # RoutingIntentKind value, e.g. "browser_automation"
    handler: str
    outcome: str  # "dispatched" / "stub" / "passthrough" / "failed" / ...
    raw_text_excerpt: str = ""

    @classmethod
    def from_log_row(cls, row: dict) -> "RecentDecision":
        return cls(
            kind=row.get("intent_kind", row.get("kind", "")),
            handler=row.get("handler", ""),
            outcome=row.get("outcome", ""),
            raw_text_excerpt=(row.get("raw_text", "") or "")[:80],
        )


@dataclass
class ReformulationContext:
    """Everything the reformulator needs to enrich an utterance.

    All fields are optional — the reformulator emits only the sections
    that have content, so callers can pass partial context without
    polluting the prompt with empty lines.
    """
    recent: List[RecentDecision] = field(default_factory=list)
    active_session_summary: Optional[str] = None
    routing_hints: List[str] = field(default_factory=list)


class InputReformulator:
    """Pure-text utterance reformulator.

    Given a raw utterance plus optional context, return an enriched
    string that the disambiguator's LLM call sees in place of the raw
    utterance. The output format is stable + grep-friendly so the
    routing audit trail can record exactly what the model saw.

    Example output (with full context):

    .. code-block:: text

        User utterance: "open the spreadsheet"
        Recent decisions (last 3):
        - browser_automation handled=stub for "open hacker news"
        - file_operation handled=stub for "list files in downloads"
        - conversational handled=passthrough for "thanks"
        Active session: coding task running ('flask app')
        Routing hints:
        - "open" historically maps to BROWSER, not FILE
    """

    def __init__(self, *, max_recent: int = 5) -> None:
        self._max_recent = max(0, int(max_recent))

    def reformulate(
        self,
        utterance: str,
        context: Optional[ReformulationContext] = None,
    ) -> str:
        utterance = (utterance or "").strip()
        ctx = context or ReformulationContext()
        # Normalised quote so the prompt template's quotes don't escape.
        safe_utt = utterance.replace('"', "'")
        lines: List[str] = [f'User utterance: "{safe_utt}"']

        if ctx.recent and self._max_recent > 0:
            # Take the *most recent* N — Python's [-0:] returns the full
            # list, so we explicitly skip the slice when max_recent==0
            # and omit the whole section.
            recent = list(ctx.recent)[-self._max_recent:]
            lines.append(f"Recent decisions (last {len(recent)}):")
            for r in recent:
                excerpt = f' for "{r.raw_text_excerpt}"' if r.raw_text_excerpt else ""
                lines.append(f"- {r.kind} handled={r.outcome}{excerpt}")

        if ctx.active_session_summary:
            lines.append(f"Active session: {ctx.active_session_summary}")

        if ctx.routing_hints:
            lines.append("Routing hints:")
            for h in ctx.routing_hints:
                lines.append(f"- {h}")

        return "\n".join(lines)


def build_default_reformulator(cfg: Any = None) -> InputReformulator:
    """Construct an :class:`InputReformulator` from the live config.

    Centralised so callers don't replicate the ``cfg.routing.irma.*``
    field-reading logic. ``cfg`` is the top-level :class:`UltronConfig`
    (or a stand-in with the same shape); pass ``None`` to read from
    :func:`get_config`.
    """
    if cfg is None:
        from ultron.config import get_config
        cfg = get_config()
    return InputReformulator(max_recent=cfg.routing.irma.max_recent_decisions)


__all__ = [
    "InputReformulator",
    "ReformulationContext",
    "RecentDecision",
    "build_default_reformulator",
]
