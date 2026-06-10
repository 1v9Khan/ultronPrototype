"""Voice "scrap it" -- user-initiated cancel + revert of a coding task.

Production-hardening re-adjudication of finding #4 (the forfeit
escape-hatch). The catalog-ported :mod:`ultron.coding.forfeit`
``ForfeitController`` governs MODEL self-forfeit (minimum-effort gate +
three-tier salvage); a USER saying "scrap it" is a different contract:
an explicit instruction to abandon the work AND put the files back the
way they were. Plain "cancel" stops the task but leaves half-written
files behind; "scrap it" stops the task and rolls the edits back.

The roll-back rides the machinery that already exists end to end: the
catalog-09 batch-F pre-edit hook in :mod:`ultron.coding.direct_bridge`
records a :class:`~ultron.coding.file_history.FileHistory` snapshot
BEFORE every file edit the coding subprocess makes, and
:meth:`FileHistory.undo_last` restores one level (deleting a file whose
snapshot says "did not exist before"). Repeating ``undo_last`` until a
path's stack is empty therefore lands on the ORIGINAL pre-task content.

ARCHITECTURAL NOTE (why this is safe where mid-task auto-revert was
not): the infra-wiring campaign documented coding edit auto-revert as
architecturally inactive because silently restoring the coding agent's
files MID-TASK breaks its black-box mental model. That objection
evaporates here -- "scrap it" CANCELS the task first, so there is no
running agent whose state could desynchronise. The revert only ever
runs against a dead task's recorded snapshots.

Strict matcher discipline (the established short-circuit pattern): only
explicit scrap/trash/throw-away/revert-everything phrasings match;
ordinary conversation -- including bare "cancel", which keeps its
existing no-revert semantics -- falls through untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ultron.utils.logging import get_logger

logger = get_logger("coding.scrap")

#: Bound on undo iterations per file (defence against a corrupt store;
#: generous -- DEFAULT_MAX_HISTORY_PER_FILE is far smaller).
MAX_UNDO_STEPS_PER_FILE: int = 64

_SCRAP_PATTERNS: tuple[re.Pattern, ...] = (
    # "scrap it" / "scrap that" / "scrap this" / "scrap the project..."
    re.compile(
        r"^(?:ultron[,!.\s]+)?(?:just\s+)?scrap\s+"
        r"(?:it|that|this|everything|"
        r"the\s+(?:project|task|app|program|code|whole\s+thing))[.!]?$",
        re.IGNORECASE,
    ),
    # "throw it away" / "throw that out"
    re.compile(
        r"^(?:ultron[,!.\s]+)?(?:just\s+)?throw\s+(?:it|that|this)\s+"
        r"(?:away|out)[.!]?$",
        re.IGNORECASE,
    ),
    # "trash it" / "trash the project"
    re.compile(
        r"^(?:ultron[,!.\s]+)?(?:just\s+)?trash\s+"
        r"(?:it|that|this|the\s+(?:project|task|app|program|code))[.!]?$",
        re.IGNORECASE,
    ),
    # "revert everything" / "undo all of that" / "undo all the changes
    # you just made"
    re.compile(
        r"^(?:ultron[,!.\s]+)?(?:just\s+)?(?:revert|undo)\s+"
        r"(?:everything|all\s+(?:of\s+)?(?:it|that|this|the\s+changes|"
        r"those\s+changes|your\s+changes))"
        r"(?:\s+you\s+(?:just\s+)?(?:did|made|changed|wrote))?[.!]?$",
        re.IGNORECASE,
    ),
    # "cancel and revert" / "cancel it and undo the changes"
    re.compile(
        r"^(?:ultron[,!.\s]+)?cancel(?:\s+(?:it|that|the\s+task))?\s+and\s+"
        r"(?:revert|undo)(?:\s+\S.*)?[.!]?$",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class ScrapMatch:
    """A matched scrap command (truthiness mirrors ``matched``)."""

    matched: bool
    phrase: str = ""

    def __bool__(self) -> bool:  # pragma: no cover -- trivial
        return self.matched


@dataclass(frozen=True)
class ScrapRevertResult:
    """Outcome of reverting one session's recorded edits.

    :param files_restored: files written back to their pre-task content.
    :param files_deleted: files removed because the task created them.
    :param errors: per-file failures (the file is left as-is).
    :param had_history: whether ANY snapshot existed for the session.
    """

    files_restored: int = 0
    files_deleted: int = 0
    errors: int = 0
    had_history: bool = False

    @property
    def files_reverted(self) -> int:
        """Total files returned to their pre-task state."""
        return self.files_restored + self.files_deleted


def match_scrap_command(text: str) -> ScrapMatch:
    """Match an explicit scrap/revert-everything voice command.

    Strict by design: only unambiguous abandon-and-revert phrasings
    match ("scrap it", "throw that away", "undo everything you just
    did", "cancel it and revert"). Bare "cancel" / "stop" deliberately
    do NOT match -- they keep their existing no-revert semantics.
    Returns a falsy :class:`ScrapMatch` for everything else.
    """
    stripped = " ".join((text or "").split())
    if not stripped:
        return ScrapMatch(matched=False)
    for pattern in _SCRAP_PATTERNS:
        if pattern.match(stripped):
            return ScrapMatch(matched=True, phrase=stripped)
    return ScrapMatch(matched=False)


def revert_session_edits(session_id: str, *, history=None) -> ScrapRevertResult:
    """Roll every file the session edited back to its pre-task state.

    Walks each tracked path's snapshot stack via repeated
    :meth:`FileHistory.undo_last` (LIFO -- the final application is the
    OLDEST snapshot, i.e. the original content; a "file did not exist"
    snapshot deletes the created file). Clears the session's history on
    completion so a later scrap can't double-revert. ``history`` is
    injectable for tests; the default resolves the per-session
    singleton. Never raises -- per-file failures are counted and the
    file is left as undo_last left it.
    """
    result_restored = 0
    result_deleted = 0
    result_errors = 0
    had_history = False
    try:
        if history is None:
            from ultron.coding.file_history import get_file_history

            history = get_file_history(session_id)
        paths = list(history.all_paths())
        had_history = bool(paths)
        for path in paths:
            final_entry = None
            for _ in range(MAX_UNDO_STEPS_PER_FILE):
                try:
                    undo = history.undo_last(path)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("scrap undo failed for %s: %s", path, exc)
                    result_errors += 1
                    final_entry = None
                    break
                if not undo.applied:
                    break
                if undo.error:
                    result_errors += 1
                    final_entry = None
                    break
                final_entry = undo.entry
            if final_entry is not None:
                if final_entry.content is None:
                    result_deleted += 1
                else:
                    result_restored += 1
        try:
            history.clear_all()
        except Exception as exc:  # noqa: BLE001
            logger.debug("scrap clear_all failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("scrap revert failed for session %s: %s", session_id, exc)
        result_errors += 1
    return ScrapRevertResult(
        files_restored=result_restored,
        files_deleted=result_deleted,
        errors=result_errors,
        had_history=had_history,
    )


def summarize_scrap(*, cancelled: bool, result: ScrapRevertResult) -> str:
    """A short, TTS-safe summary of a scrap operation."""
    lead = "Scrapped. I cancelled the task and " if cancelled else "Scrapped. I "
    n = result.files_reverted
    if n > 0:
        msg = lead + f"reverted {n} file{'s' if n != 1 else ''}."
        if result.errors:
            msg += (
                f" {result.errors} file{'s' if result.errors != 1 else ''}"
                " could not be reverted; check the logs."
            )
    elif result.had_history and result.errors:
        msg = lead + "attempted the revert, but it hit errors; check the logs."
    else:
        msg = lead + "found no recorded edits to revert."
    return msg


__all__ = [
    "MAX_UNDO_STEPS_PER_FILE",
    "ScrapMatch",
    "ScrapRevertResult",
    "match_scrap_command",
    "revert_session_edits",
    "summarize_scrap",
]
