"""Voice-utterance intent detection for the coding pipeline.

Three classifications matter to the orchestrator:

1. ``CODE_TASK``  -- the user wants Ultron to write/edit code. Routes
   into the coding runner instead of the normal LLM response path.
2. ``PROGRESS_QUERY`` -- the user is asking about an in-flight task
   ("how's it going?"). Resolves from runner state without spawning
   anything new.
3. ``CANCEL`` -- "stop", "abort", "kill the task". Cancels the active
   task if one is running.
4. ``NONE`` -- not a coding utterance; orchestrator falls through to
   the regular response path.

Pure rule-based today (regex over the transcribed text). Designed to
fail to ``NONE`` on ambiguity so we don't accidentally hijack a
non-coding query. A future LLM-fallback can be added the same way the
addressing classifier did it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CodingIntentKind(str, Enum):
    NONE = "none"
    CODE_TASK = "code_task"
    PROGRESS_QUERY = "progress_query"
    CANCEL = "cancel"
    # Phase 2 additions:
    MID_SESSION_ADJUSTMENT = "mid_session_adjustment"
    CLARIFICATION_RESPONSE = "clarification_response"


@dataclass
class CodingIntent:
    kind: CodingIntentKind
    confidence: float = 0.0
    reason: str = ""
    # CODE_TASK extras:
    is_new_project: bool = False
    project_reference: Optional[str] = None  # text the user used to refer to a project
    explicit_name: Optional[str] = None  # name from "called X" / "named X"
    task_text: str = ""  # the actual task body, with project markers stripped
    candidates_for_resolver: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule patterns
# ---------------------------------------------------------------------------


# "Stop", "cancel" etc. only count when there's clearly a coding task to
# stop -- we keep this list tight so we don't misclassify "stop the
# music" or "cancel my reminder".
_CANCEL_PATTERNS = re.compile(
    r"^\s*(?:"
    r"stop\s+(?:the\s+)?(?:claude|coding|task|build|generation)|"
    r"cancel\s+(?:the\s+)?(?:claude|coding|task|build|generation|run)|"
    r"abort\s+(?:the\s+)?(?:task|run|build)|"
    r"kill\s+(?:the\s+)?(?:task|run|build|claude)|"
    r"forget\s+the\s+coding\s+task"
    r")\b",
    re.IGNORECASE,
)


# Mid-session adjustment phrasing -- only fires when there's an active task.
# Captures the kind of natural-language directive the user gives mid-build:
# "actually have him use postgres", "tell him to add logging", "instead of X
# do Y", "change that to Y", "make him do Y".
_ADJUSTMENT_PATTERNS = re.compile(
    r"\b(?:"
    r"actually,?\s+(?:have\s+(?:him|claude)|tell\s+(?:him|claude)|change|make\s+(?:him|claude)|switch|use|do)|"
    r"instead\s+of\s+\w+,?\s+(?:have|tell|use|do|make)|"
    r"have\s+(?:him|claude|it)\s+(?:use|switch|change|make|do|stop|start|add|remove|drop)|"
    r"tell\s+(?:him|claude|it)\s+to\s+\w+|"
    r"make\s+(?:him|claude|it)\s+(?:use|switch|change|stop|start|add|remove|do|drop|focus)|"
    r"can\s+you\s+(?:have|tell|make)\s+(?:him|claude|it)|"
    r"don't\s+(?:have|let)\s+(?:him|claude|it)|"
    r"change\s+(?:that|it)\s+to|"
    r"forget\s+(?:that|the)\s+(?:approach|plan|idea)|"
    r"on\s+(?:second|2nd)\s+thought|"
    r"hold\s+on,?\s+(?:have|tell|use|do|make|change|switch)"
    r")\b",
    re.IGNORECASE,
)


# 2026-05-11 follow-up fix: the original pattern required ``going`` (or
# ``doing`` / ``done``) immediately after a tight subject group --
# ``it / things / claude / the task / that``. A real session ("How is
# that project going?") fell through to the conversational LLM because
# ``project`` appeared between ``that`` and ``going`` and the regex
# never re-anchored. The user got a generic hallucinated "progressing
# as expected" reply instead of the runner's actual status narration.
#
# Broadened to accept ``<determiner> [coding-noun]`` everywhere the old
# pattern accepted ``the task``. Determiners: the / that / this / your
# / our / my. Coding nouns: task / project / build / app / code / work /
# thing / run / job. The noun is optional so the legacy ``that going`` /
# ``the doing`` phrasings still fire. Added "coming (along)" as an
# alternate to "going" for "How's the project coming along?".
#
# Safety: these patterns only fire when has_active_task=True, so even
# the ungrammatical edge cases (``is the done``) can't hijack ordinary
# conversation -- there has to be a coding task in flight.
_DETERMINER_NOUN = (
    r"(?:the|that|this|your|our|my)"
    r"(?:\s+(?:task|project|build|app|code|work|thing|run|job))?"
)

_PROGRESS_PATTERNS = re.compile(
    r"(?:"
    # "How's it going" / "How is that project going" / "How are things going"
    r"\bhow(?:'s|\s+is|\s+are)\s+"
    r"(?:it|things|claude|" + _DETERMINER_NOUN + r")"
    r"\s+(?:going|coming(?:\s+along)?)|"
    # "What's claude doing" / "What's the project doing" / "What's it up to"
    r"\bwhat(?:'s|\s+is|\s+are)\s+"
    r"(?:claude|it|" + _DETERMINER_NOUN + r")?"
    r"\s*(?:doing|working\s+on|up\s+to)|"
    # "Are you done" / "Is it done" / "Is the project done yet"
    r"\bare\s+you\s+done|"
    r"\bis\s+(?:it|claude|" + _DETERMINER_NOUN + r")\s+done|"
    # Generic status markers
    r"\b(?:any\s+)?progress\b|"
    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:status|update)|"
    r"\bgive\s+me\s+(?:a\s+)?(?:status|update)|"
    r"\bhow\s+far\s+along|"
    r"\bstill\s+(?:running|going|working)\b|"
    r"\bfinished\s+yet|"
    # Bare status / update tokens (e.g. "Status?" -- short voice prompts).
    r"^\s*(?:status|update)\s*\??\s*$"
    r")",
    re.IGNORECASE,
)


# Coding-task triggers. Each is a verb + target pattern that strongly
# implies the user wants code written or edited.
# Common determiners we tolerate between the verb and the target noun.
# The lazy ``(?:[\w\-]+\s+){0,3}`` gap absorbs adjectives ("a small python
# tool", "another quick project") so the trigger fires reliably on natural
# speech without exploding the regex into combinatorial bits.
_DETERMINER = r"(?:an?|some|the|another|a\s+few)?"
_ADJ_GAP = r"(?:[\w\-]+\s+){0,3}"

_TARGET_NOUNS_CREATE = (
    r"python|javascript|js|typescript|ts|rust|go|c\+\+|cpp|java|bash|shell|"
    r"flask|fastapi|express|react|svelte|vue|cli|web|node|api|app|"
    r"script|tool|server|service|program|module|package|library|utility|"
    r"project|repo|repository|prototype|demo"
)
_TARGET_NOUNS_FIX = (
    r"bug|issue|problem|error|crash|test|"
    + _TARGET_NOUNS_CREATE
)

_CODE_TRIGGERS = re.compile(
    r"(?:"
    # Creation: "make a [adj] python tool", "spin up a quick project"
    r"\b(?:create|make|build|write|generate|scaffold|set\s+up|spin\s+up|whip\s+up)\s+"
    + r"(?:me\s+)?" + _DETERMINER + r"\s*" + _ADJ_GAP +
    r"(?:" + _TARGET_NOUNS_CREATE + r")\b|"
    # Editing / extending: "add a subtract function to ...", "implement an endpoint to ..."
    r"\b(?:add|implement|introduce)\s+" + _DETERMINER + r"\s*" + _ADJ_GAP +
    r"(?:function|method|class|endpoint|route|test|feature|fix|patch|"
    r"option|flag|argument|module|package|script|file|tool)\s+to\b|"
    # Fixing / debugging / editing existing code: "fix the bug in my flask app",
    # "edit my flask app", "update the dashboard"
    r"\b(?:fix|debug|patch|repair|edit|modify|update|extend|tweak)\s+(?:the\s+|my\s+)?" + _ADJ_GAP +
    r"(?:" + _TARGET_NOUNS_FIX + r")\b|"
    r"\b(?:refactor|rewrite|restructure)\b|"
    # Direct claude-code phrasings
    r"\bhave\s+claude\s+code\b|"
    r"\buse\s+claude\s+code\b|"
    r"\bsend\s+(?:a|the)\s+task\s+to\s+claude\b"
    r")",
    re.IGNORECASE,
)


# Pattern preferred for finding an EXISTING project reference. We split
# the lookup into two regexes so we can try the strong signal (preposition
# + my/the + ref + project-noun) before falling back to the weaker
# (verb + my/the + ref) one. The strong pattern reliably picks up
# "in my flask app" even when the sentence starts with "Fix the bug".
_EXISTING_PROJECT_STRONG = re.compile(
    r"(?:in|to|on|for|of)\s+(?:my|the|our)\s+"
    r"(?P<ref>[a-z0-9][a-z0-9 _\-]{1,40}?)"
    r"\s+(?:project|app|repo|script|service|tool|server|library|package|api)\b",
    re.IGNORECASE,
)
_EXISTING_PROJECT_WEAK = re.compile(
    r"\b(?:fix|edit|update|modify|extend|refactor|rewrite|tweak)\s+"
    r"(?:my|the|our)\s+"
    r"(?P<ref>[a-z0-9][a-z0-9 _\-]{1,40}?)"
    # Must end at a sentence boundary or prepositional phrase boundary so
    # we don't capture "the bug" from "Fix the bug in my flask app".
    r"(?=\s+(?:project|app|repo|script|service|tool|server|library|package|api|so|"
    r"to|for|with|by)|[.,!?]|$)",
    re.IGNORECASE,
)


def _find_existing_project_reference(text: str) -> str | None:
    """Return the first project reference found via either pattern, or None."""
    m = _EXISTING_PROJECT_STRONG.search(text)
    if m:
        return m.group("ref").strip()
    m = _EXISTING_PROJECT_WEAK.search(text)
    if m:
        return m.group("ref").strip()
    return None


# Look for "called X" / "named X" -- explicit project-name signal.
_EXPLICIT_NAME_PATTERN = re.compile(
    r"\b(?:called|named)\s+['\"]?(?P<name>[A-Za-z0-9][A-Za-z0-9 _\-]{0,40}?)['\"]?(?:\s|$|[.,!?])",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify(
    utterance: str,
    has_active_task: bool = False,
    has_pending_clarification: bool = False,
) -> CodingIntent:
    """Classify ``utterance`` for the orchestrator's coding pipeline.

    Args:
        utterance: the user's transcribed text.
        has_active_task: true if a coding task is currently running. Lets
            cancel/progress patterns fire only when there's something to
            actually act on; spares false positives when the user says
            "any progress" in casual conversation.
        has_pending_clarification: true if Claude is currently blocked
            waiting on the user to answer a clarification. When set,
            short utterances that don't match any other coding pattern
            are treated as the response to that clarification (lets the
            user answer naturally rather than having to phrase their
            answer as a coding command).
    """
    text = (utterance or "").strip()
    if not text:
        return CodingIntent(kind=CodingIntentKind.NONE, reason="empty utterance")

    # Cancel / progress / adjustment only count if a task is running.
    if has_active_task:
        if _CANCEL_PATTERNS.search(text):
            return CodingIntent(
                kind=CodingIntentKind.CANCEL,
                confidence=0.95,
                reason="cancel pattern matched while task running",
                task_text=text,
            )
        if _PROGRESS_PATTERNS.search(text):
            return CodingIntent(
                kind=CodingIntentKind.PROGRESS_QUERY,
                confidence=0.9,
                reason="progress query while task running",
                task_text=text,
            )
        if _ADJUSTMENT_PATTERNS.search(text):
            return CodingIntent(
                kind=CodingIntentKind.MID_SESSION_ADJUSTMENT,
                confidence=0.85,
                reason="adjustment pattern matched while task running",
                task_text=text,
            )
    else:
        # Even with no task running, "are you done" / "how's it going" with
        # nothing to compare against falls through to NONE -- a casual
        # check-in gets a normal LLM response, not a coding answer.
        pass

    # Clarification-response path: when Claude is parked on a question and
    # the user speaks anything that isn't a known coding command, treat
    # it as the answer. This is checked AFTER the high-confidence command
    # patterns so "stop the task" still cancels rather than answering.
    if has_pending_clarification and not _CODE_TRIGGERS.search(text):
        return CodingIntent(
            kind=CodingIntentKind.CLARIFICATION_RESPONSE,
            confidence=0.7,
            reason="utterance during pending clarification",
            task_text=text,
        )

    # Coding task creation / editing.
    if _CODE_TRIGGERS.search(text):
        intent = CodingIntent(
            kind=CodingIntentKind.CODE_TASK,
            confidence=0.85,
            reason="coding-trigger pattern matched",
            task_text=text,
        )
        # Detect explicit name first -- it can co-exist with a NEW or
        # EXISTING project signal.
        name_match = _EXPLICIT_NAME_PATTERN.search(text)
        if name_match:
            intent.explicit_name = name_match.group("name").strip()
        # Existing project reference?
        ref = _find_existing_project_reference(text)
        if ref:
            intent.is_new_project = False
            intent.project_reference = ref
            intent.candidates_for_resolver.append(ref)
            if intent.explicit_name and intent.explicit_name.lower() not in ref.lower():
                intent.candidates_for_resolver.append(intent.explicit_name)
            return intent
        # Otherwise treat it as a NEW project; the orchestrator will
        # double-check via the resolver before scaffolding.
        intent.is_new_project = True
        if intent.explicit_name:
            intent.candidates_for_resolver.append(intent.explicit_name)
        return intent

    return CodingIntent(kind=CodingIntentKind.NONE, reason="no rule matched")


def derive_project_name(intent: CodingIntent) -> str:
    """Produce a usable project name from a CODE_TASK intent.

    Order of preference:
      1. ``explicit_name`` from "called X" / "named X".
      2. A short slug of the first informative noun phrase in
         ``task_text`` (heuristic).
      3. Fallback: timestamp-style ``ultron_project_<hex>``.
    """
    if intent.explicit_name:
        return intent.explicit_name
    text = (intent.task_text or "").lower()
    # Try to capture the noun phrase right after a creation verb.
    m = re.search(
        r"\b(?:create|make|build|write|generate|scaffold|spin\s+up|whip\s+up)\b\s+"
        r"(?:me\s+)?(?:an?|some|the)?\s*"
        r"(?:python|javascript|js|typescript|ts|rust|go|java|bash|shell|"
        r"flask|fastapi|express|react|cli|web)?\s*"
        r"(?P<phrase>[a-z][a-z0-9 _\-]{2,40})",
        text,
    )
    if m:
        phrase = m.group("phrase")
        # Trim filler words off the head/tail.
        phrase = re.sub(r"^(?:script|app|tool|project|program|module|that|to)\s+", "", phrase)
        phrase = phrase.strip().split(".")[0].split(",")[0]
        words = [w for w in phrase.split() if w not in {"that", "to", "for", "with"}]
        if words:
            slug = "_".join(words[:3])
            return slug
    import uuid
    return f"ultron_project_{uuid.uuid4().hex[:6]}"
