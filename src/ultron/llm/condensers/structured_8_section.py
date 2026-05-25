"""Structured 8-section history condenser (cline-style ``summarizeTask``).

Adapted from cline's ``summarizeTask`` / ``/compact`` / ``/smol``
pattern (Apache 2.0; see ``THIRD_PARTY_NOTICES.md``). The prompt
shape forces the model into a structured 8-section summary:

1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections
4. Problem Solving
5. Pending Tasks
6. Task Evolution (Original / Modifications / Current / Context)
7. Current Work
8. Next Step / Required Files

The summary is the entire compacted history — everything before the
condenser's call is dropped. Ultron's variant adds:

* A voice-friendly 3-section compressed renderer (``compact_for_voice``)
  that picks Primary Intent + Pending Tasks + Next Step to emit a
  conversational "where were we" ack.
* A best-effort sectional parser that recovers the section bodies from
  the model output so future callers can mutate or filter them
  without re-summarising.
* Validation: the parser reports missing-required-section so the
  caller can retry with a stricter prompt rather than silently
  shipping an incomplete summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from ultron.llm.condensers.base import (
    CondenseResult,
    Condenser,
    Turn,
    char_count_tokens_for_turns,
    turn_text,
)


#: Eight required section headers (in canonical order).
SECTION_HEADERS: tuple[str, ...] = (
    "Primary Request and Intent",
    "Key Technical Concepts",
    "Files and Code Sections",
    "Problem Solving",
    "Pending Tasks",
    "Task Evolution",
    "Current Work",
    "Next Step",
)

#: Sub-section titles emitted under ``Task Evolution``.
TASK_EVOLUTION_SUBSECTIONS: tuple[str, ...] = (
    "Original Request",
    "Modifications",
    "Current Scope",
    "Context for Changes",
)


SUMMARISE_PROMPT_TEMPLATE = """\
Summarise the conversation above into a single structured handoff.
The summary becomes the entire compacted history; everything before
this point will be dropped. Be COMPLETE — anything you omit is lost.

Produce EXACTLY these 8 sections, each starting with the ## header:

## Primary Request and Intent
One-paragraph description of what the user originally asked for and
the underlying intent.

## Key Technical Concepts
Bulleted list of the concepts, libraries, frameworks, and patterns
in play this session.

## Files and Code Sections
Bulleted list of the files we touched (or referenced) with one-line
descriptions of WHY each matters. Include file paths.

## Problem Solving
Bulleted list of the problems we encountered + how we resolved them
(or what's still open).

## Pending Tasks
Bulleted list of items the user asked for that are NOT yet done.

## Task Evolution
Use these four sub-headings:
* Original Request — what the user first asked.
* Modifications — how the scope changed mid-session.
* Current Scope — what we're working on right now.
* Context for Changes — why the scope shifted.

## Current Work
One paragraph describing what is actively in progress at this exact
moment.

## Next Step
The single most-important next step the agent (or the user) should
take. If there is none, say so plainly.

End the summary immediately after the Next Step section — do not
issue any tool calls, do not add commentary outside the sections.
"""


VOICE_INTRO_SENTENCE: str = "Picking up where we left off."


@dataclass(frozen=True)
class ParsedSummary:
    """Sectioned representation of a 8-section summary string.

    Attributes:
        sections: ordered mapping of header → body for the sections
            that were successfully parsed.
        missing: tuple of canonical headers that were NOT found.
        raw: the original input string.
    """

    sections: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = field(default_factory=tuple)
    raw: str = ""

    @property
    def has_all_required(self) -> bool:
        return not self.missing


def parse_summary(text: str) -> ParsedSummary:
    """Recover the 8 sections from a model-generated summary string.

    The parser is intentionally permissive: it accepts ``##`` and
    ``###`` headers, tolerates surrounding whitespace, and pairs each
    canonical header with the body that follows it (until the next
    header or end-of-string).

    Args:
        text: model-generated summary.

    Returns:
        :class:`ParsedSummary` with per-section bodies + the list of
        missing canonical headers.
    """
    sections: dict[str, str] = {}
    if not text:
        return ParsedSummary(sections={}, missing=SECTION_HEADERS, raw=text or "")

    # Match each section header (case-insensitive, trim trailing colons).
    # The header is captured + everything until the next header / EOF.
    section_pattern = re.compile(
        r"^\s*#{1,4}\s*(?P<header>[^\n#]+?)\s*:?\s*$",
        re.MULTILINE,
    )
    matches = list(section_pattern.finditer(text))
    if not matches:
        return ParsedSummary(sections={}, missing=SECTION_HEADERS, raw=text)
    for idx, match in enumerate(matches):
        header = match.group("header").strip()
        canonical = _resolve_header(header)
        if canonical is None:
            continue
        body_start = match.end()
        body_end = (
            matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        )
        body = text[body_start:body_end].strip()
        # Later matches for the same canonical header overwrite earlier
        # ones (we want the final, fully-rendered section).
        if body or canonical not in sections:
            sections[canonical] = body
    missing = tuple(h for h in SECTION_HEADERS if h not in sections)
    return ParsedSummary(sections=sections, missing=missing, raw=text)


def _resolve_header(header: str) -> Optional[str]:
    """Map a free-form header to its canonical form, if any."""
    cleaned = header.strip().rstrip(":").strip()
    cleaned_lower = cleaned.lower()
    for canonical in SECTION_HEADERS:
        if cleaned_lower == canonical.lower():
            return canonical
    # Heuristic: tolerate ``# next steps`` vs canonical ``Next Step``.
    aliases = {
        "next steps": "Next Step",
        "files": "Files and Code Sections",
        "problems": "Problem Solving",
        "pending": "Pending Tasks",
        "intent": "Primary Request and Intent",
        "current": "Current Work",
    }
    if cleaned_lower in aliases:
        return aliases[cleaned_lower]
    return None


def compact_for_voice(parsed: ParsedSummary, *, max_chars: int = 280) -> str:
    """Render a 3-section voice-friendly continuity ack.

    Args:
        parsed: output of :func:`parse_summary`.
        max_chars: hard cap on the rendered string (TTS-friendly).

    Returns:
        Short conversational string suitable for the orchestrator to
        speak as a "where were we" ack. Returns the intro alone when
        no relevant sections were found.
    """
    if not parsed.sections:
        return VOICE_INTRO_SENTENCE
    pieces: list[str] = [VOICE_INTRO_SENTENCE]
    intent = parsed.sections.get("Primary Request and Intent")
    if intent:
        pieces.append(f"You were focused on {_first_sentence(intent)}")
    pending = parsed.sections.get("Pending Tasks")
    if pending:
        pieces.append(f"Pending: {_first_sentence(pending)}")
    next_step = parsed.sections.get("Next Step")
    if next_step:
        pieces.append(f"Next: {_first_sentence(next_step)}")
    text = " ".join(p.rstrip(".") + "." for p in pieces if p.strip())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "."
    return text


def _first_sentence(text: str) -> str:
    """Return the first sentence-ish slice of ``text`` (capped at 140 chars)."""
    stripped = " ".join(text.strip().split())
    if not stripped:
        return ""
    for terminator in (". ", "? ", "! ", ".\n", "?\n", "!\n"):
        idx = stripped.find(terminator)
        if 0 < idx < 140:
            return stripped[:idx]
    if len(stripped) > 140:
        return stripped[:140].rstrip() + "..."
    return stripped


@dataclass
class StructuredEightSectionCondenser(Condenser):
    """:class:`Condenser` implementation that produces the 8-section summary.

    Args:
        summarize_fn: callable mapping ``(prompt, turns_text)`` to a
            model-generated string. Typically wraps
            :meth:`LLMEngine.generate`. None disables the condenser;
            attempting to call :meth:`condense` then returns a result
            with ``error="no summarize_fn provided"`` and the original
            turns intact.
        prompt_template: optional override of the default prompt.
        keep_tail_turns: number of latest turns (assistant + user) to
            preserve verbatim AFTER the summary. Default 4. Pass 0 to
            replace the entire history with the summary.
        attribution_role: role to assign to the summary turn. Default
            ``"system"``; some providers prefer ``"assistant"``.
    """

    summarize_fn: Optional[Callable[[str, str], str]] = None
    prompt_template: str = SUMMARISE_PROMPT_TEMPLATE
    keep_tail_turns: int = 4
    attribution_role: str = "system"

    def condense(self, turns: Sequence[Turn]) -> CondenseResult:  # noqa: D401
        if self.summarize_fn is None:
            return CondenseResult(
                turns=list(turns),
                dropped_turn_count=0,
                summary_inserted=False,
                token_estimate_before=char_count_tokens_for_turns(turns),
                token_estimate_after=char_count_tokens_for_turns(turns),
                notes=("no summarize_fn provided; condense is a no-op",),
                error="no summarize_fn provided",
            )
        if not turns:
            return CondenseResult(
                turns=list(turns),
                dropped_turn_count=0,
                summary_inserted=False,
                token_estimate_before=0,
                token_estimate_after=0,
            )
        tail_count = max(0, min(int(self.keep_tail_turns), len(turns)))
        head_turns = turns[: len(turns) - tail_count] if tail_count else turns
        tail_turns = list(turns[len(turns) - tail_count:]) if tail_count else []
        before_tokens = char_count_tokens_for_turns(turns)
        # Render the head into a single string for the summariser.
        rendered = "\n\n".join(turn_text(t) for t in head_turns)
        try:
            summary_text = self.summarize_fn(self.prompt_template, rendered) or ""
        except Exception as exc:  # noqa: BLE001
            return CondenseResult(
                turns=list(turns),
                dropped_turn_count=0,
                summary_inserted=False,
                token_estimate_before=before_tokens,
                token_estimate_after=before_tokens,
                notes=(f"summarize_fn raised: {type(exc).__name__}: {exc}",),
                error=f"{type(exc).__name__}: {exc}",
            )
        summary_text = summary_text.strip()
        if not summary_text:
            return CondenseResult(
                turns=list(turns),
                dropped_turn_count=0,
                summary_inserted=False,
                token_estimate_before=before_tokens,
                token_estimate_after=before_tokens,
                notes=("summarize_fn returned empty",),
                error="empty summary",
            )
        parsed = parse_summary(summary_text)
        notes: list[str] = []
        if parsed.missing:
            notes.append(
                "summary missing sections: " + ", ".join(parsed.missing),
            )
        summary_turn: Turn = (self.attribution_role, summary_text)
        new_turns: list[Turn] = [summary_turn]
        new_turns.extend(tail_turns)
        after_tokens = char_count_tokens_for_turns(new_turns)
        return CondenseResult(
            turns=new_turns,
            dropped_turn_count=len(head_turns),
            summary_inserted=True,
            token_estimate_before=before_tokens,
            token_estimate_after=after_tokens,
            notes=tuple(notes),
            error=None,
        )


def build_structured_8_section_condenser(
    summarize_fn: Optional[Callable[[str, str], str]] = None,
    **kwargs: object,
) -> StructuredEightSectionCondenser:
    """Convenience constructor mirroring the other condenser factories."""
    return StructuredEightSectionCondenser(
        summarize_fn=summarize_fn,
        **kwargs,  # type: ignore[arg-type]
    )


__all__ = [
    "SECTION_HEADERS",
    "SUMMARISE_PROMPT_TEMPLATE",
    "TASK_EVOLUTION_SUBSECTIONS",
    "VOICE_INTRO_SENTENCE",
    "ParsedSummary",
    "StructuredEightSectionCondenser",
    "build_structured_8_section_condenser",
    "compact_for_voice",
    "parse_summary",
]
