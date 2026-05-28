"""Voice-intent matcher for "log a concern" / "report that response".

Catalog openclaw-clawhub T12 wiring (deferred-primitive pass). The
:class:`ultron.feedback.report_queue.ReportQueue` primitive shipped
unwired; this module is the orchestrator-side trigger that turns a
spoken meta-command ("ultron, flag that last answer", "log a concern
that the response was wrong") into a filed :class:`Report`.

It is a strict regex matcher, NOT an LLM classifier -- it short-
circuits in the orchestrator run loop the same way
:func:`ultron.local_clock_reply.maybe_local_clock_reply` does, so a
report command never burns an LLM round-trip. Strictness matters: a
normal request that merely contains the word "report" ("give me a
report on the weather") must NOT trip the gate, so every pattern
requires an explicit concern / flag verb AND a reference to the
assistant's own output (response / answer / reply / that).

The matched :class:`ReportConcernMatch` carries a best-effort
:class:`ReportTargetKind` (RESPONSE for the usual "that answer"
case; MEMORY when the user references remembered facts) so the
audit-reviewer routing the catalog describes can pick the right
downstream pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ultron.feedback.report_queue import ReportTargetKind


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


# A reference to the assistant's own recent output. Used as a shared
# fragment so every trigger pattern is anchored to "the thing ultron
# just said" rather than an arbitrary noun.
_OUTPUT_NOUN = r"(?:response|answer|reply|that|it)"

# Explicit concern / flag verbs. The command must carry one of these
# so that benign sentences with "report" as a content noun don't match.
_REPORT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "log / file / raise a concern [about ...]"
    re.compile(r"\b(?:log|file|raise|note)\s+a\s+concern\b", re.IGNORECASE),
    # "report that (last) response/answer" ; "report that" alone
    re.compile(
        r"\breport\s+(?:that|the\s+last)(?:\s+" + _OUTPUT_NOUN + r")?\b",
        re.IGNORECASE,
    ),
    # "flag that / the last response/answer"
    re.compile(
        r"\bflag\s+(?:that|the\s+last)\s+" + _OUTPUT_NOUN + r"\b",
        re.IGNORECASE,
    ),
    # "that response/answer was wrong / bad / incorrect / unhelpful / ..."
    re.compile(
        r"\bthat\s+(?:response|answer|reply)\s+was\s+"
        r"(?:wrong|bad|incorrect|unhelpful|inappropriate|misleading|not\s+right)\b",
        re.IGNORECASE,
    ),
    # "that was a bad / wrong answer"
    re.compile(
        r"\bthat\s+was\s+a\s+(?:bad|wrong|terrible|poor)\s+(?:answer|response|reply)\b",
        re.IGNORECASE,
    ),
)

# When the command mentions memory / remembering, the concern is about
# a stored fact, not the just-spoken response. The trailing ``\w*``
# matches inflected forms ("misremembered", "remembered", "recalled").
_MEMORY_HINT = re.compile(
    r"\b(?:misremember|remember|memory|recall|forgot|stored)\w*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReportConcernMatch:
    """A matched report-a-concern voice command.

    Attributes:
        target_kind: best-effort :class:`ReportTargetKind`. RESPONSE
            for the usual "that answer was wrong" case; MEMORY when
            the utterance references remembered facts.
        reason: the verbatim user utterance -- it IS the explanation
            of the concern, stored on the filed report.
        raw_text: the original utterance (echo of ``reason``; kept
            distinct so a future variant can split a trailing reason
            clause off the trigger).
    """

    target_kind: ReportTargetKind
    reason: str
    raw_text: str


def match_report_concern(text: str) -> Optional[ReportConcernMatch]:
    """Return a :class:`ReportConcernMatch` when ``text`` is a
    report-a-concern command, else ``None``.

    Strict: requires an explicit concern / flag verb anchored to the
    assistant's own output. Empty / whitespace input returns None.
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()
    if not any(p.search(stripped) for p in _REPORT_PATTERNS):
        return None
    kind = (
        ReportTargetKind.MEMORY
        if _MEMORY_HINT.search(stripped)
        else ReportTargetKind.RESPONSE
    )
    return ReportConcernMatch(
        target_kind=kind,
        reason=stripped,
        raw_text=stripped,
    )


__all__ = [
    "ReportConcernMatch",
    "match_report_concern",
]
