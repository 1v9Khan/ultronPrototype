"""IntentDisambiguator — resolve ambiguous coding-vs-automation utterances.

Called when a rule-based classification turns out to be a tie or an
unconfident match between coding and automation categories. The
disambiguator asks Qwen a small question and either picks a winner or
returns UNCLEAR with a clarification question for the orchestrator to
ask the user.

Token budget for the LLM call: ~50 tokens output. Cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ultron.config import get_config
from ultron.openclaw_routing.intents import RoutingIntentKind
from ultron.utils.logging import get_logger

logger = get_logger("openclaw_routing.disambiguator")


_DISAMBIG_PROMPT = """\
The user said: "{utterance}"

This could be:
- A coding task (writing or editing code)
- A PC automation task (controlling existing applications, files, browser)
- Both (hybrid task)

Which is it? Output ONE of: CODING | AUTOMATION | HYBRID | UNCLEAR

If UNCLEAR, also output a one-sentence clarifying question on the next line.
Format:
VERDICT
optional question
"""


# 4B plan Item 5 — IRMA-enriched prompt. Used when
# ``routing.irma.enabled`` is True. The reformulator already produces a
# multi-line block that includes the original utterance and any
# context; we just frame the question around it.
_DISAMBIG_PROMPT_IRMA = """\
{enriched}

Given the above context, the user's request could be:
- A coding task (writing or editing code)
- A PC automation task (controlling existing applications, files, browser)
- Both (hybrid task)

Which is it? Output ONE of: CODING | AUTOMATION | HYBRID | UNCLEAR

If UNCLEAR, also output a one-sentence clarifying question on the next line.
Format:
VERDICT
optional question
"""


@dataclass
class DisambiguationResult:
    """Output of a disambiguation attempt.

    ``kind`` is one of:
      - RoutingIntentKind.CODE_TASK
      - RoutingIntentKind.HYBRID_TASK
      - RoutingIntentKind.CONVERSATIONAL  (used as the "AUTOMATION but
        unspecified" placeholder; caller can run a second classification
        pass on the original utterance with hybrid signals enabled)
      - None when the result is UNCLEAR — caller asks the user.

    ``clarification_question`` is populated only when the verdict is
    UNCLEAR; the orchestrator speaks it.
    """
    kind: Optional[RoutingIntentKind]
    clarification_question: Optional[str] = None
    raw_verdict: str = ""
    raw_response: str = ""


class IntentDisambiguator:
    """Two-shot question to the local LLM. Falls back to UNCLEAR on any
    parse failure so the orchestrator's safety net (asking the user)
    always engages.

    Optional ``reformulator`` (4B plan Item 5): when supplied AND
    ``routing.irma.enabled`` is True, the disambiguator wraps the raw
    utterance with relevant context (recent intents, active session,
    routing hints) before the LLM call. Default behaviour (no
    reformulator OR irma.enabled = False) is unchanged from before.
    """

    def __init__(self, llm: Any, *, reformulator: Optional[Any] = None) -> None:
        self._llm = llm
        self._reformulator = reformulator

    async def disambiguate(
        self,
        utterance: str,
        *,
        irma_context: Optional[Any] = None,
    ) -> DisambiguationResult:
        cfg = get_config().routing
        if not cfg.llm_disambiguation_enabled:
            # Disambiguation disabled in config -> always escalate to user.
            return DisambiguationResult(
                kind=None,
                clarification_question=(
                    "Was that a coding task, or did you want me to "
                    "automate something on the PC?"
                ),
                raw_verdict="",
            )

        # 4B plan Item 5 — optionally enrich the prompt with IRMA context.
        irma_enabled = (
            cfg.irma.enabled
            and self._reformulator is not None
        )
        if irma_enabled:
            try:
                enriched = self._reformulator.reformulate(utterance, irma_context)
                prompt = _DISAMBIG_PROMPT_IRMA.format(enriched=enriched)
            except Exception as e:
                # Reformulation must not break the disambiguator — fall
                # back to the unenriched prompt.
                logger.warning("IRMA reformulation failed: %s", e)
                prompt = _DISAMBIG_PROMPT.format(
                    utterance=utterance.replace('"', "'"),
                )
        else:
            prompt = _DISAMBIG_PROMPT.format(utterance=utterance.replace('"', "'"))

        try:
            raw = self._llm.generate(prompt) if self._llm is not None else ""
        except Exception as e:
            logger.warning("IntentDisambiguator LLM call failed: %s", e)
            raw = ""

        verdict, question = _parse_verdict(raw)
        kind: Optional[RoutingIntentKind]
        if verdict == "CODING":
            kind = RoutingIntentKind.CODE_TASK
        elif verdict == "AUTOMATION":
            # Caller maps this to the appropriate single category by
            # re-running the rule classifier; if that returns NONE the
            # voice path falls back to CONVERSATIONAL.
            kind = RoutingIntentKind.CONVERSATIONAL
        elif verdict == "HYBRID":
            kind = RoutingIntentKind.HYBRID_TASK
        else:
            kind = None
            if not question:
                question = (
                    "Was that a coding task, or did you want me to "
                    "automate something on the PC?"
                )
        return DisambiguationResult(
            kind=kind,
            clarification_question=question,
            raw_verdict=verdict,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_VERDICT_RE = re.compile(
    r"\b(CODING|AUTOMATION|HYBRID|UNCLEAR)\b",
    re.IGNORECASE,
)


def _parse_verdict(text: str) -> tuple[str, Optional[str]]:
    """Pull the verdict + optional question out of an LLM response."""
    if not text:
        return ("UNCLEAR", None)
    text = _THINK_RE.sub("", text).strip()
    m = _VERDICT_RE.search(text)
    if not m:
        return ("UNCLEAR", None)
    verdict = m.group(1).upper()

    # If the model emitted UNCLEAR, look for a follow-up question on the
    # next line(s).
    question: Optional[str] = None
    if verdict == "UNCLEAR":
        # Take everything after the verdict word up to the first sentence
        # terminator (or the whole thing if no terminator).
        rest = text[m.end():].strip()
        if rest:
            # First non-empty line that looks like a sentence.
            for line in rest.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Strip leading list markers / quotes.
                line = re.sub(r"^[\-*>\"']\s*", "", line).strip()
                if line:
                    question = line
                    break
    return (verdict, question)


__all__ = ["IntentDisambiguator", "DisambiguationResult"]
