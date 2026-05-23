"""V4A patch-format parser + applier with fuzz tiers.

Pattern lifted in spirit (not in source) from aider's
``coders/patch_coder.py`` (Apache 2.0; see ``THIRD_PARTY_NOTICES.md``).

V4A is the alternative-to-SEARCH/REPLACE format that OpenAI's Codex
CLI prefers. Format:

    *** Begin Patch
    *** Update File: path/to/file.py
    @@ optional_scope_marker
     context_line
    -removed_line
    +added_line
     context_line
    *** End Patch

Three top-level actions: ``Add``, ``Update``, ``Delete``. Each
``Update`` block is a unified-diff-style hunk with 3 context lines
before and after each change; the optional ``@@`` line gives a
"jump to this scope" hint when 3 context lines isn't enough.

Fuzz tiers for context matching (catalog T17):

  * **Fuzz 0** — exact match.
  * **Fuzz 1** — ``rstrip()`` match (trailing whitespace differs).
  * **Fuzz 100** — full ``strip()`` match (any whitespace OK).
  * **Fuzz 10000** — EOF marker present but not actually at EOF
    (huge penalty; last-resort fallback).

The parser is deliberately tolerant: a malformed hunk produces a
``PatchError`` rather than corrupting the file. Apply is all-or-
nothing per file: if any hunk in a file's chain fails to locate,
the file is left untouched.

The catalog rated this ★ (defer) because ultron doesn't currently
have a Qwen-as-editor path that produces V4A. This module ships it
ready for that path — when batch 6's architect+editor split graduates
to local-editor mode, V4A is a one-line config flip via
:mod:`ultron.coding.coder_modes`.

Public surface:

  * :class:`PatchAction` — enum of action kinds.
  * :class:`PatchHunk` — frozen hunk record.
  * :class:`PatchFileBlock` — frozen per-file change block.
  * :class:`ParsedPatch` — full parsed patch.
  * :class:`PatchError` — raised on malformed input.
  * :func:`parse_v4a_patch` — parse the text format.
  * :func:`apply_patch` — apply a parsed patch to a virtual filesystem
    (``Mapping[path -> content]``); returns the updated mapping.

The applier never touches disk on its own — callers pass in the
file contents (typically loaded from disk earlier) and write the
returned contents back when satisfied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


logger = logging.getLogger("ultron.coding.patch_v4a")


# Marker tokens for the V4A format. Constants so callers + tests
# can reference them by name.
BEGIN_PATCH = "*** Begin Patch"
END_PATCH = "*** End Patch"
ADD_PREFIX = "*** Add File: "
UPDATE_PREFIX = "*** Update File: "
DELETE_PREFIX = "*** Delete File: "
SCOPE_PREFIX = "@@"
EOF_MARKER = "*** End of File"


# Fuzz penalty values per catalog T17.
FUZZ_EXACT = 0
FUZZ_RSTRIP = 1
FUZZ_STRIP = 100
FUZZ_EOF_MISMATCH = 10_000


class PatchAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True)
class PatchHunk:
    """One hunk within an Update block.

    Attributes:
        scope: Optional ``@@ <marker>`` text. Empty when not present.
        before_context: Lines preceded by " " in the patch (3 lines
            of context before the first change in this hunk).
        removed_lines: Lines preceded by ``-``.
        added_lines: Lines preceded by ``+``.
        after_context: Lines preceded by " " AFTER the changes.
        ends_at_eof: True when this hunk's final line is ``*** End of File``.
    """

    scope: str
    before_context: Tuple[str, ...]
    removed_lines: Tuple[str, ...]
    added_lines: Tuple[str, ...]
    after_context: Tuple[str, ...]
    ends_at_eof: bool = False


@dataclass(frozen=True)
class PatchFileBlock:
    """One file's change set in a parsed patch."""

    action: PatchAction
    file_path: str
    # For Update: list of hunks; for Add: a single hunk whose
    # added_lines is the whole file; for Delete: empty.
    hunks: Tuple[PatchHunk, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParsedPatch:
    """A complete parsed *** Begin Patch ... *** End Patch unit."""

    blocks: Tuple[PatchFileBlock, ...]


class PatchError(Exception):
    """Raised when the input doesn't parse as V4A."""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_v4a_patch(text: str) -> ParsedPatch:
    """Parse a V4A patch into a :class:`ParsedPatch`.

    Raises:
        PatchError: when ``Begin Patch`` / ``End Patch`` markers are
            missing or malformed, when a file block has an unknown
            action prefix, or when an Update block contains no hunks.
    """
    if not text:
        raise PatchError("empty patch")
    lines = text.splitlines()

    # Find Begin/End boundaries.
    begin_idx = end_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == BEGIN_PATCH:
            begin_idx = i
        if line.strip() == END_PATCH:
            end_idx = i
            break
    if begin_idx < 0 or end_idx < 0 or end_idx <= begin_idx:
        raise PatchError(
            "missing or out-of-order *** Begin Patch / *** End Patch markers"
        )

    body = lines[begin_idx + 1: end_idx]
    blocks: List[PatchFileBlock] = []
    i = 0
    while i < len(body):
        line = body[i]
        if line.startswith(UPDATE_PREFIX):
            block, consumed = _parse_update_block(body, i)
            blocks.append(block)
            i += consumed
        elif line.startswith(ADD_PREFIX):
            block, consumed = _parse_add_block(body, i)
            blocks.append(block)
            i += consumed
        elif line.startswith(DELETE_PREFIX):
            path = line[len(DELETE_PREFIX):].strip()
            if not path:
                raise PatchError(f"empty path in delete block at line {i + 1}")
            blocks.append(PatchFileBlock(
                action=PatchAction.DELETE,
                file_path=path,
            ))
            i += 1
        elif not line.strip():
            i += 1  # tolerate blank lines between blocks
        else:
            raise PatchError(
                f"unexpected line in patch body (no action prefix): "
                f"{line!r}"
            )
    if not blocks:
        raise PatchError("patch body contained no file blocks")
    return ParsedPatch(blocks=tuple(blocks))


def _parse_update_block(
    body: Sequence[str], start: int,
) -> Tuple[PatchFileBlock, int]:
    """Parse one *** Update File: <path> block, returning (block, n_lines)."""
    header = body[start]
    file_path = header[len(UPDATE_PREFIX):].strip()
    if not file_path:
        raise PatchError(f"empty path in update block at line {start + 1}")

    # Collect hunks until the next ``*** ...`` action line or end of body.
    hunks: List[PatchHunk] = []
    current_scope = ""
    current_before: List[str] = []
    current_removed: List[str] = []
    current_added: List[str] = []
    current_after: List[str] = []
    in_changes = False
    ends_at_eof = False

    def flush_hunk():
        nonlocal current_before, current_removed, current_added
        nonlocal current_after, in_changes, ends_at_eof
        if not (current_removed or current_added):
            # No changes recorded -> nothing to flush.
            current_before = []
            current_after = []
            in_changes = False
            return
        hunks.append(PatchHunk(
            scope=current_scope,
            before_context=tuple(current_before),
            removed_lines=tuple(current_removed),
            added_lines=tuple(current_added),
            after_context=tuple(current_after),
            ends_at_eof=ends_at_eof,
        ))
        current_before = list(current_after)  # leftover context becomes next hunk's before
        current_after = []
        current_removed = []
        current_added = []
        in_changes = False
        ends_at_eof = False

    i = start + 1
    while i < len(body):
        line = body[i]
        if line.startswith(("*** Update File:", "*** Add File:", "*** Delete File:")):
            break
        if line.startswith(SCOPE_PREFIX):
            flush_hunk()
            current_scope = line[len(SCOPE_PREFIX):].strip()
            current_before = []
            current_after = []
            current_removed = []
            current_added = []
            in_changes = False
            i += 1
            continue
        if line == EOF_MARKER:
            ends_at_eof = True
            i += 1
            continue
        if not line:
            # Empty body line — treat as " " (blank context).
            if in_changes:
                current_after.append("")
            else:
                current_before.append("")
            i += 1
            continue
        prefix, content = line[0], line[1:]
        if prefix == " ":
            if in_changes:
                current_after.append(content)
            else:
                current_before.append(content)
        elif prefix == "-":
            in_changes = True
            current_removed.append(content)
            current_after = []  # any prior trailing context belongs to PREVIOUS hunk
        elif prefix == "+":
            in_changes = True
            current_added.append(content)
            current_after = []
        else:
            raise PatchError(
                f"unrecognized line prefix {prefix!r} in update block at "
                f"line {i + 1}: {line!r}"
            )
        i += 1
    flush_hunk()
    if not hunks:
        raise PatchError(
            f"update block for {file_path!r} contained no hunks"
        )
    return (
        PatchFileBlock(
            action=PatchAction.UPDATE,
            file_path=file_path,
            hunks=tuple(hunks),
        ),
        i - start,
    )


def _parse_add_block(
    body: Sequence[str], start: int,
) -> Tuple[PatchFileBlock, int]:
    """Parse one *** Add File: <path> block."""
    header = body[start]
    file_path = header[len(ADD_PREFIX):].strip()
    if not file_path:
        raise PatchError(f"empty path in add block at line {start + 1}")
    added: List[str] = []
    i = start + 1
    while i < len(body):
        line = body[i]
        if line.startswith(("*** Update File:", "*** Add File:", "*** Delete File:")):
            break
        if line.startswith("+"):
            added.append(line[1:])
        elif not line.strip():
            added.append("")
        else:
            # Tolerate non-prefixed lines as literal content (some
            # emitters skip the ``+`` for Add blocks).
            added.append(line)
        i += 1
    return (
        PatchFileBlock(
            action=PatchAction.ADD,
            file_path=file_path,
            hunks=(PatchHunk(
                scope="",
                before_context=(),
                removed_lines=(),
                added_lines=tuple(added),
                after_context=(),
            ),),
        ),
        i - start,
    )


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------


def apply_patch(
    patch: ParsedPatch,
    files: Mapping[str, str],
) -> Dict[str, Optional[str]]:
    """Apply ``patch`` to a virtual filesystem.

    Args:
        patch: A :class:`ParsedPatch`.
        files: ``{path -> contents}`` mapping. Missing paths are
            treated as "file doesn't exist" — only Add blocks may
            target them.

    Returns:
        ``{path -> new_contents}`` for every modified path. Deleted
        paths map to ``None``. Untouched files are NOT included.

    Raises:
        PatchError: when any hunk in any file fails to locate a
            position to apply. The error is all-or-nothing per
            invocation — partial applications are never returned.
    """
    out: Dict[str, Optional[str]] = {}
    for block in patch.blocks:
        if block.action == PatchAction.DELETE:
            if block.file_path not in files:
                raise PatchError(
                    f"cannot delete missing file: {block.file_path}"
                )
            out[block.file_path] = None
            continue
        if block.action == PatchAction.ADD:
            if block.file_path in files:
                raise PatchError(
                    f"add target already exists: {block.file_path}"
                )
            new_content = "\n".join(block.hunks[0].added_lines)
            if not new_content.endswith("\n"):
                new_content += "\n"
            out[block.file_path] = new_content
            continue
        # Update.
        if block.file_path not in files:
            raise PatchError(
                f"update target missing: {block.file_path}"
            )
        new_text = _apply_update(files[block.file_path], block.hunks)
        out[block.file_path] = new_text
    return out


def _apply_update(original: str, hunks: Sequence[PatchHunk]) -> str:
    """Apply every hunk in order. Returns the new text.

    Raises :class:`PatchError` on any hunk that can't be located.
    """
    text = original
    for h_idx, hunk in enumerate(hunks):
        result = _apply_single_hunk(text, hunk)
        if result is None:
            raise PatchError(
                f"hunk {h_idx + 1} failed to locate its context"
            )
        text = result
    return text


def _apply_single_hunk(text: str, hunk: PatchHunk) -> Optional[str]:
    """Try fuzz tiers 0 → 1 → 100 in order. Returns None on miss."""
    text_lines = text.splitlines()
    # The "what to find" block is before_context + removed_lines + after_context.
    needle = list(hunk.before_context) + list(hunk.removed_lines) + list(hunk.after_context)
    # The "what to replace it with" is before + added + after.
    replacement = list(hunk.before_context) + list(hunk.added_lines) + list(hunk.after_context)

    for fuzz in (FUZZ_EXACT, FUZZ_RSTRIP, FUZZ_STRIP):
        location = _find_match(text_lines, needle, fuzz)
        if location is None:
            continue
        start = location
        out_lines = (
            text_lines[:start]
            + replacement
            + text_lines[start + len(needle):]
        )
        return ("\n".join(out_lines) + "\n") if text.endswith("\n") else "\n".join(out_lines)

    # EOF marker present but the hunk wasn't found at EOF — try matching
    # at the end of the file as a last resort (huge fuzz penalty).
    if hunk.ends_at_eof:
        tail_len = len(needle)
        if tail_len <= len(text_lines):
            tail_start = len(text_lines) - tail_len
            location = _find_match(
                text_lines[tail_start:],
                needle,
                FUZZ_STRIP,
            )
            if location is not None:
                start = tail_start + location
                out_lines = (
                    text_lines[:start]
                    + replacement
                    + text_lines[start + len(needle):]
                )
                return ("\n".join(out_lines) + "\n") if text.endswith("\n") else "\n".join(out_lines)
    return None


def _find_match(
    haystack: Sequence[str],
    needle: Sequence[str],
    fuzz: int,
) -> Optional[int]:
    """Walk ``haystack`` looking for ``needle`` under the given fuzz level.

    Returns the starting index or None. Requires UNIQUENESS — multiple
    matches are ambiguous (the LLM had a specific location in mind).
    """
    if not needle or len(needle) > len(haystack):
        return None
    transform = _transformer_for_fuzz(fuzz)
    needle_t = [transform(s) for s in needle]
    matches: List[int] = []
    for start in range(len(haystack) - len(needle) + 1):
        window = [transform(haystack[start + i]) for i in range(len(needle))]
        if window == needle_t:
            matches.append(start)
    if len(matches) != 1:
        return None
    return matches[0]


def _transformer_for_fuzz(fuzz: int):
    if fuzz <= FUZZ_EXACT:
        return lambda s: s
    if fuzz <= FUZZ_RSTRIP:
        return lambda s: s.rstrip()
    return lambda s: s.strip()


__all__ = [
    "ADD_PREFIX",
    "BEGIN_PATCH",
    "DELETE_PREFIX",
    "END_PATCH",
    "EOF_MARKER",
    "FUZZ_EOF_MISMATCH",
    "FUZZ_EXACT",
    "FUZZ_RSTRIP",
    "FUZZ_STRIP",
    "ParsedPatch",
    "PatchAction",
    "PatchError",
    "PatchFileBlock",
    "PatchHunk",
    "SCOPE_PREFIX",
    "UPDATE_PREFIX",
    "apply_patch",
    "parse_v4a_patch",
]
