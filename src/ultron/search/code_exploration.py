"""Strict voice-intent matcher for explicit deep CODE exploration.

Companion to :func:`ultron.memory.deep_recall.match_deep_recall` (memory) and
:func:`ultron.web_search.deep_research.match_deep_research` (web): gates the
orchestrator's ``_maybe_handle_code_exploration`` short-circuit, which runs a
bounded :class:`ultron.agent_loop.deep_loops.DeepExplorationLoop` (iterative
ripgrep: decompose -> search -> gap-fill -> search over the project source) and
reports where in the code the answer lives.

DELIBERATELY STRICT -- a normal CODING TASK ("build a calculator", "fix the
login bug") must NOT trip this; those belong to the coding engineer. We require
BOTH:

* a *search / locate* marker ("search", "find", "where is", "locate",
  "explore", "grep", "which files"), AND
* a *code* referent ("the codebase", "the code", "source", "the repo", "a
  function/class/module", "implementation", "defined", ...),

and we refuse anything that reads as a coding-task verb (build/create/fix/...),
a web-research request, or a memory-recall request (each handled by its own
matcher, which the orchestrator checks first / alongside).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

__all__ = ["CodeExplorationMatch", "match_code_exploration"]


# Search / locate marker -- the user wants to FIND something, not build it.
_SEARCH_RE = re.compile(
    r"\b("
    r"search|find|locate|look\s+for|where\s+(?:is|are|does|do)|"
    r"explore|grep|trace|show\s+me\s+where|which\s+files?"
    r")\b",
    re.IGNORECASE,
)

# Code referent -- the target is the project source, not the web/memory.
_CODE_RE = re.compile(
    r"\b("
    r"code\s*base|the\s+code|source(?:\s+code)?|the\s+repo(?:sitory)?|"
    r"the\s+project|in\s+the\s+source|function|method|class|module|"
    r"implementation|implemented|defined|definition|"
    r"import(?:ed|s)?|called\s+in\s+the\s+code"
    r")\b",
    re.IGNORECASE,
)

# Coding-TASK verbs -> the coding engineer owns these, not exploration.
_TASK_RE = re.compile(
    r"\b("
    r"build|create|make\s+(?:a|me|an)|write\s+(?:a|me|some)|add\s+|implement|"
    r"fix|edit|change|refactor|delete|remove|rename|"
    r"run\s+(?:it|the)|launch"
    r")\b",
    re.IGNORECASE,
)

# Web / memory -> defer to their own matchers.
_WEB_RE = re.compile(
    r"\b(search\s+(?:the\s+)?(?:web|internet)|online|latest\s+news|google)\b",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"\b(remember|recall|your\s+memory|we\s+discussed|i\s+told\s+you)\b",
    re.IGNORECASE,
)

# Topic extraction: prefer the clause after an explicit pivot.
_TOPIC_RE = re.compile(
    r"\b(?:for|where\s+(?:is|are|does|do)|find|locate|"
    r"the\s+code\s+for|the\s+implementation\s+of)\s+(.+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CodeExplorationMatch:
    """A matched deep code-exploration command.

    Attributes:
        topic: the search subject (best-effort; falls back to the whole
            utterance, which the loop's decomposer handles fine).
        raw_text: the original utterance.
    """

    topic: str
    raw_text: str


def _extract_topic(text: str) -> str:
    """Best-effort pull of the search subject from the utterance."""
    m = _TOPIC_RE.search(text)
    candidate = m.group(1) if m else text
    candidate = candidate.strip().strip("?.!,").strip()
    # Strip a leading code-referent so the term handed to ripgrep is clean.
    candidate = re.sub(
        r"^(?:in\s+(?:the\s+)?(?:code(?:base)?|source|repo(?:sitory)?|project)"
        r"|the\s+code(?:base)?|defined|implemented)\b\s*",
        "", candidate, flags=re.IGNORECASE,
    ).strip()
    return candidate


def match_code_exploration(text: str) -> Optional[CodeExplorationMatch]:
    """Return a :class:`CodeExplorationMatch` iff ``text`` is an explicit
    code-search command, else ``None``.

    Strict: requires both a search marker AND a code referent; suppressed for
    coding-task verbs, web-research, and memory-recall utterances. Empty /
    whitespace input returns ``None``.
    """
    if not text or not text.strip():
        return None
    if _WEB_RE.search(text) or _MEMORY_RE.search(text):
        return None
    if _TASK_RE.search(text):
        return None
    if not _SEARCH_RE.search(text):
        return None
    if not _CODE_RE.search(text):
        return None
    topic = _extract_topic(text) or text.strip()
    return CodeExplorationMatch(topic=topic, raw_text=text)
