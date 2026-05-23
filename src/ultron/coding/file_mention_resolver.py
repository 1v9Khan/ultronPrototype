"""File-mention auto-add heuristic.

Pattern lifted in spirit (not in source) from aider's
``base_coder.get_file_mentions`` / ``check_for_file_mentions``
(Apache 2.0; see ``THIRD_PARTY_NOTICES.md``).

Given a free-form user utterance and a list of candidate files,
return the subset the user *implicitly* referenced — without needing
an explicit `@filename` or `/add` command. The algorithm:

  1. Tokenise the utterance by whitespace, strip surrounding
     punctuation.
  2. For each candidate file path:
     a. **Exact relative-path match** → unambiguous; add it.
     b. **Basename match with disambiguation**:
        - basename appears verbatim in the token set;
        - basename contains at least one special character
          (``.``, ``_``, ``-``, ``/``, ``\\``) — i.e. it doesn't
          look like a plain English word;
        - basename is UNIQUE among the candidates (otherwise we
          can't tell which one the user meant);
        - basename is not in the ``already_in_chat`` set (the LLM
          already has it; no point re-adding).
  3. Result: a list of :class:`FileMention` records, each with a
     kind (``"exact"`` / ``"basename"``) and confidence score.

The disambiguation-via-special-chars heuristic is the catalog's key
insight. Without it, mentions of "run" or "make" or "test" would
spuriously match files like ``run.py`` / ``Makefile`` / ``test.go``.

Filtering: a caller-supplied ``ignore`` set ("never auto-add these")
is respected. The orchestrator typically populates this from a
persistent "permanently rejected" file so user opt-outs survive
restarts.

Output is deterministic on input ordering (no random tie-breaks);
when multiple basenames hit, they're returned in the input candidate
order. Use the result as ``mentioned_fnames`` input to
:class:`~ultron.coding.repo_map.RepoMap` to bias its personalization
vector toward the right files.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, List, Optional, Sequence, Set


logger = logging.getLogger("ultron.coding.file_mention_resolver")


# Special characters that signal a token is "filename-shaped" rather
# than a plain English word. Aider uses ``./\_-``; we add `~` for
# user-home references that occasionally show up.
_FILENAME_SPECIAL_CHARS = frozenset(".\\/-_~")


# Punctuation to strip from token edges before matching. ASCII +
# common smart variants.
_EDGE_PUNCTUATION = ',.;:!?()[]{}"‘’“”`'


# Common English-word basenames that should NEVER auto-add. Even with
# a special char in the basename ("e.g. run.py"), if the BARE word is
# in this set we still require explicit user disambiguation.
DEFAULT_NEVER_AUTO_ADD_BASENAMES: frozenset[str] = frozenset({
    "run",
    "make",
    "test",
    "main",
    "build",
    "data",
    "log",
    "config",
    "common",
    "core",
    "util",
    "utils",
    "helper",
    "helpers",
    "lib",
    "src",
    "tmp",
    "temp",
    "tools",
})


@dataclass(frozen=True)
class FileMention:
    """One detected file reference in the utterance.

    Attributes:
        path: Relative POSIX-form path (matches the candidate list).
        kind: ``"exact"`` (full relative path) or ``"basename"``
            (matched on basename + disambiguation).
        confidence: Heuristic score in [0, 1]. ``1.0`` for exact path,
            ``0.7`` for unique basename, ``0.3`` for ambiguous (which
            we generally don't return).
    """

    path: str
    kind: str
    confidence: float


def resolve_mentions(
    utterance: str,
    candidates: Sequence[str],
    *,
    already_in_chat: Optional[Iterable[str]] = None,
    ignore: Optional[Iterable[str]] = None,
    never_auto_add_basenames: Optional[Iterable[str]] = None,
) -> List[FileMention]:
    """Return implicit file mentions from ``utterance`` against ``candidates``.

    Args:
        utterance: The user's free-form text (typically the voice
            transcript or a typed message).
        candidates: Relative POSIX-form paths of files the user could
            be referencing. Typically the project's source tree.
        already_in_chat: Paths already visible to the LLM; excluded
            from the result (no point re-adding).
        ignore: Paths the user has permanently opted out of auto-add.
            Excluded from the result.
        never_auto_add_basenames: Override the
            :data:`DEFAULT_NEVER_AUTO_ADD_BASENAMES` blocklist of
            plain-English-word basenames that need explicit
            disambiguation.

    Returns:
        Deterministically ordered list of :class:`FileMention`. Empty
        when no mentions are detected.
    """
    if not utterance or not candidates:
        return []

    chat_set: Set[str] = set(already_in_chat or [])
    ignore_set: Set[str] = set(ignore or [])
    never_set: Set[str] = (
        set(never_auto_add_basenames)
        if never_auto_add_basenames is not None
        else set(DEFAULT_NEVER_AUTO_ADD_BASENAMES)
    )

    tokens = _tokenise(utterance)
    if not tokens:
        return []
    token_set = set(tokens)

    # Pre-compute candidate basename → list mapping for fast lookup.
    basename_map: dict[str, List[str]] = {}
    for cand in candidates:
        normalised = _normalise_path(cand)
        basename = _basename(normalised)
        basename_map.setdefault(basename, []).append(normalised)

    out: List[FileMention] = []
    seen: Set[str] = set()
    for cand in candidates:
        normalised = _normalise_path(cand)
        if normalised in chat_set or normalised in ignore_set or normalised in seen:
            continue

        # Exact relative-path match (against tokens and against the raw text).
        if normalised in token_set or normalised in utterance:
            out.append(FileMention(path=normalised, kind="exact", confidence=1.0))
            seen.add(normalised)
            continue

        # Basename match with disambiguation.
        basename = _basename(normalised)
        if basename not in token_set:
            continue
        if not _has_special_char(basename):
            # Plain word — too risky to auto-add.
            continue
        # Strip extension to also reject "run.py" when the bare word "run"
        # is in the blocklist. Note: we DO still require special-char in
        # the original basename — so "run.py" with stem "run" is
        # rejected, but "run_engine.py" with stem "run_engine" is fine.
        stem = _stem(basename)
        if stem in never_set:
            continue
        if len(basename_map.get(basename, [])) > 1:
            # Ambiguous — multiple files share this basename. Skip.
            continue

        out.append(FileMention(path=normalised, kind="basename", confidence=0.7))
        seen.add(normalised)

    return out


def _tokenise(text: str) -> List[str]:
    """Whitespace-split + strip edge punctuation."""
    parts = text.split()
    out: List[str] = []
    for raw in parts:
        cleaned = raw.strip(_EDGE_PUNCTUATION)
        if cleaned:
            out.append(cleaned)
    return out


def _normalise_path(path: str) -> str:
    """Convert OS-specific path separators to POSIX form."""
    return path.replace("\\", "/")


def _basename(posix_path: str) -> str:
    return os.path.basename(posix_path) or posix_path


def _stem(basename: str) -> str:
    """Filename without extension(s). 'foo.tar.gz' -> 'foo'."""
    stem = basename
    while "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return stem


def _has_special_char(basename: str) -> bool:
    """True iff basename contains any of the filename-shape chars."""
    return any(ch in _FILENAME_SPECIAL_CHARS for ch in basename)


__all__ = [
    "DEFAULT_NEVER_AUTO_ADD_BASENAMES",
    "FileMention",
    "resolve_mentions",
]
