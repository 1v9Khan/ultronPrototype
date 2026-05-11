"""Per-call response-style addenda.

Pure-function helpers that prepend short style directives to the user's
text before it reaches the LLM. They live OUTSIDE the persona file
(SOUL.md is voice-quality-locked) so the orchestrator can nudge the
model on a per-utterance basis without changing the system prompt.

Today only one addendum lives here -- a brevity hint for short
questions that the 4B model otherwise tends to over-explain ("What are
the Orcs in 40k?" → four-paragraph essay). The hint is ONLY applied to
the non-search conversational path; the web-search-augmented prompt
already carries its own length directive.
"""

from __future__ import annotations

from ultron.utils.logging import get_logger

logger = get_logger("response_style")


# Tuned to be a quiet system-style directive that the LLM treats as
# instruction rather than text to repeat (verified during the audio +
# memory quality pass: addenda in this shape never leaked into spoken
# output). Short and emphatic so it competes with the model's
# default-toward-verbose habit on simple questions.
_BREVITY_HINT = (
    "[Style: respond in 1-3 short sentences. The user's question is "
    "brief; match that brevity. Do not list, do not lecture, do not "
    "offer follow-up options unless asked.]"
)

# Heuristics for "this is a brief question that wants a brief answer".
# Tuned against the live-session log where 5-8-word queries like
# "What are the Orcs in 40k?" produced 4-paragraph responses.
_BREVITY_MAX_WORDS = 12
_BREVITY_MAX_CHARS = 80

# Keywords that signal "the user explicitly wants depth" -- skip the
# brevity hint even when the question is short. "Explain" / "in
# detail" / "step by step" / "walk me through" are the typical asks.
_DEPTH_MARKERS = (
    "explain",
    "in detail",
    "in depth",
    "thoroughly",
    "step by step",
    "step-by-step",
    "walk me through",
    "walk through",
    "elaborate",
    "expand on",
    "give me details",
    "give me the details",
    "tell me everything",
    "describe in",
    "list out",
    "list all",
    "everything you know",
)


def is_brief_question(user_text: str) -> bool:
    """True iff the user's text reads as a brief question that should
    get a brief answer.

    Brief = short (≤ 12 words or ≤ 80 chars after strip) AND not
    explicitly asking for depth via any of the
    :data:`_DEPTH_MARKERS` keywords. Empty / whitespace-only text is
    not "brief" -- there's nothing to size against.
    """
    stripped = (user_text or "").strip()
    if not stripped:
        return False
    word_count = len(stripped.split())
    char_count = len(stripped)
    if word_count > _BREVITY_MAX_WORDS and char_count > _BREVITY_MAX_CHARS:
        return False
    lowered = stripped.lower()
    if any(m in lowered for m in _DEPTH_MARKERS):
        return False
    return True


def apply_brevity_hint(user_text: str) -> str:
    """Prepend a brevity directive to ``user_text`` when the question
    is brief (per :func:`is_brief_question`); otherwise return
    ``user_text`` unchanged.

    The hint is on its own line(s) above a blank line, mirroring the
    uncertainty addendum format. Empty input is returned as-is so
    callers can apply this unconditionally without checking first.
    """
    if not is_brief_question(user_text):
        return user_text
    return f"{_BREVITY_HINT}\n\n{user_text}"
