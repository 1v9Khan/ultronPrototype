"""Coder-mode descriptor registry (T26).

Pattern lifted in spirit (not in source) from aider's
``coders/__init__.py`` (Apache 2.0; see ``THIRD_PARTY_NOTICES.md``).

Catalog T26 is "refactor concept, not a discrete component" — aider
ships a dozen ``XxxCoder`` classes each with its own prompt template
+ edit format. Ultron's existing architecture has ONE coding-task
runner; refactoring it into a coder-per-mode hierarchy would be
disruptive. Instead, this module ships a lightweight *descriptor
registry*: each mode is a frozen dataclass listing its prompt-template
name, edit-format hint, and a short description. The existing runner
can opt to look up a mode descriptor when it wants per-mode behavior
without us refactoring the class hierarchy.

The registry is additive: NOTHING in the existing routing path
consults it. Future supervisor / dispatcher work can grow into it
gradually.

Public surface:

  * :class:`CoderMode` — frozen mode descriptor.
  * :class:`EditFormat` — enum of edit-format hints.
  * :data:`CODER_MODES` — registry mapping mode name to descriptor.
  * :func:`get_coder_mode` — lookup with fail-open None.
  * :func:`list_coder_modes` — sorted list of registered names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class EditFormat(str, Enum):
    """Edit-format hints. Maps to aider's edit_format strings."""

    # The mode doesn't produce edits — it just answers questions.
    NONE = "none"
    # SEARCH/REPLACE blocks (aider's "diff" / "diff-fenced").
    SEARCH_REPLACE = "search_replace"
    # Whole-file replacement.
    WHOLE_FILE = "whole_file"
    # Unified diff (udiff).
    UDIFF = "udiff"
    # V4A patch format (see ultron.coding.patch_v4a).
    PATCH_V4A = "patch_v4a"
    # Editor format dispatched FROM an architect plan (the editor's
    # natural format is one of the above; architect mode itself
    # produces prose only).
    ARCHITECT_DISPATCH = "architect_dispatch"


@dataclass(frozen=True)
class CoderMode:
    """One coding-mode descriptor.

    Attributes:
        name: Stable identifier (``"edit"``, ``"ask"``, ...).
        description: One-line human-readable summary. Used by future
            ``/help``-by-voice surfaces.
        prompt_template: Name of the prompt template to load (lives
            under ``prompts/coding/``). Loaders treat this as a hint
            — missing templates fall back to a generic default.
        edit_format: The :class:`EditFormat` the mode produces.
        produces_edits: True iff the mode is expected to write to the
            filesystem. False for ask / context / help modes.
        is_supervised: True iff the mode requires a supervisor decision
            before dispatch (architect mode, full-supervisor dispatch).
    """

    name: str
    description: str
    prompt_template: str
    edit_format: EditFormat
    produces_edits: bool
    is_supervised: bool = False


# Catalog T26 mode catalogue, adapted for ultron's terminology.
# Each entry is a single source of truth so the supervisor /
# dispatcher / help command can consult it without re-deriving.
CODER_MODES: Dict[str, CoderMode] = {
    "edit": CoderMode(
        name="edit",
        description="Modify existing code in the project (default).",
        prompt_template="edit",
        edit_format=EditFormat.SEARCH_REPLACE,
        produces_edits=True,
    ),
    "ask": CoderMode(
        name="ask",
        description="Answer questions about code without making edits.",
        prompt_template="ask",
        edit_format=EditFormat.NONE,
        produces_edits=False,
    ),
    "architect": CoderMode(
        name="architect",
        description="Produce a prose plan; defer edits to the editor coder.",
        prompt_template="architect",
        edit_format=EditFormat.ARCHITECT_DISPATCH,
        produces_edits=False,
        is_supervised=True,
    ),
    "context": CoderMode(
        name="context",
        description="Identify which files in the repo are relevant.",
        prompt_template="context",
        edit_format=EditFormat.NONE,
        produces_edits=False,
    ),
    "whole_file": CoderMode(
        name="whole_file",
        description="Rewrite whole files (no diff format).",
        prompt_template="whole_file",
        edit_format=EditFormat.WHOLE_FILE,
        produces_edits=True,
    ),
    "udiff": CoderMode(
        name="udiff",
        description="Produce unified-diff format edits.",
        prompt_template="udiff",
        edit_format=EditFormat.UDIFF,
        produces_edits=True,
    ),
    "patch_v4a": CoderMode(
        name="patch_v4a",
        description="Produce V4A patch-format edits (alternative to SEARCH/REPLACE).",
        prompt_template="patch_v4a",
        edit_format=EditFormat.PATCH_V4A,
        produces_edits=True,
    ),
    "help": CoderMode(
        name="help",
        description="Answer questions about Ultron itself.",
        prompt_template="help",
        edit_format=EditFormat.NONE,
        produces_edits=False,
    ),
}


def get_coder_mode(name: str) -> Optional[CoderMode]:
    """Look up a mode by name. Returns None when unknown."""
    if not name:
        return None
    return CODER_MODES.get(name.lower())


def list_coder_modes() -> List[str]:
    """Sorted list of all registered mode names."""
    return sorted(CODER_MODES)


def edit_modes() -> List[CoderMode]:
    """Subset of modes that produce file edits."""
    return [m for m in CODER_MODES.values() if m.produces_edits]


def read_only_modes() -> List[CoderMode]:
    """Subset of modes that don't write files."""
    return [m for m in CODER_MODES.values() if not m.produces_edits]


__all__ = [
    "CODER_MODES",
    "CoderMode",
    "EditFormat",
    "edit_modes",
    "get_coder_mode",
    "list_coder_modes",
    "read_only_modes",
]
