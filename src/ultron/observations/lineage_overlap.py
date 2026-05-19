"""Lineage-overlap helper -- detects which retrieved memories were
actually consumed by a generated response.

The retrieval observation emitted by :func:`observe_retrieval` carries
``lineage_ids`` -- the memory point ids returned to the caller. After
the LLM generates a response, we want to know which of those ids
actually contributed to the response so importance scoring (A2 /
A5 in the V1-plus design notes) can up-weight memories that proved
useful and down-weight memories that get retrieved but ignored.

This module ships only the **pure detection primitive**.  Live wiring
(LLM call site -> retrieve memory contents -> compute overlap -> emit
``lineage_usage`` row) is deferred to a follow-up because the LLM
call site doesn't currently have a clean handle on the retrieved
memory CONTENT, only their ids.  The follow-up either:

* threads a memory-content-lookup callback through the LLM call, or
* runs the detection as a maintenance pass that re-queries Qdrant
  per-id.

Either way, ``compute_lineage_overlap`` is the shared primitive both
paths will call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

from .schema import Observation
from .writer import ObservationWriter, emit_observation


_WORD_RE = re.compile(r"[a-z0-9']+")


def _word_set(text: str) -> set[str]:
    """Return the unique lowercase word tokens of ``text``."""
    if not text:
        return set()
    return set(_WORD_RE.findall(text.lower()))


def _longest_common_substring(a: str, b: str) -> str:
    """Return the longest substring shared by ``a`` and ``b``.

    O(len(a) * len(b)) DP; fine for the short snippets we feed it
    (response chunks + memory content are kilobytes max). Returns an
    empty string when there's no overlap.
    """
    if not a or not b:
        return ""
    m, n = len(a), len(b)
    # Single-row DP -- we only need the previous row.
    prev_row = [0] * (n + 1)
    best_len = 0
    best_end = 0
    for i in range(1, m + 1):
        cur_row = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur_row[j] = prev_row[j - 1] + 1
                if cur_row[j] > best_len:
                    best_len = cur_row[j]
                    best_end = i
        prev_row = cur_row
    return a[best_end - best_len:best_end]


@dataclass(frozen=True)
class LineageOverlap:
    """Per-lineage-id detection result."""

    lineage_id: str
    used: bool
    word_overlap: int
    longest_substring: str

    def as_dict(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "used": self.used,
            "word_overlap": self.word_overlap,
            "longest_substring_len": len(self.longest_substring),
            "longest_substring_preview": self.longest_substring[:60],
        }


def compute_lineage_overlap(
    response_text: str,
    memory_contents: Mapping[str, str],
    *,
    min_word_overlap: int = 3,
    min_substring_chars: int = 12,
) -> list[LineageOverlap]:
    """For each lineage_id in ``memory_contents``, compute its overlap
    with ``response_text``.

    A lineage is marked ``used`` when EITHER of the heuristics fires:

    * at least ``min_word_overlap`` distinct content words shared with
      the response (loose semantic match), OR
    * a shared substring of at least ``min_substring_chars`` characters
      (tight literal match -- catches quoted snippets, names,
      titles).

    Returns a list of :class:`LineageOverlap` results in the same
    insertion order as ``memory_contents``.
    """
    response_words = _word_set(response_text)
    out: list[LineageOverlap] = []
    for lineage_id, content in memory_contents.items():
        content_words = _word_set(content)
        shared_words = response_words & content_words
        word_overlap = len(shared_words)
        substring = _longest_common_substring(response_text or "", content or "")
        used = (
            word_overlap >= min_word_overlap
            or len(substring) >= min_substring_chars
        )
        out.append(
            LineageOverlap(
                lineage_id=lineage_id,
                used=used,
                word_overlap=word_overlap,
                longest_substring=substring,
            )
        )
    return out


@dataclass
class UsageEmitSummary:
    """Counts from :func:`emit_lineage_usage_rows`."""

    emitted: int = 0
    failed: int = 0
    used_count: int = 0


def emit_lineage_usage_rows(
    parent_event_id: str,
    overlaps: Iterable[LineageOverlap],
    *,
    writer: Optional[ObservationWriter] = None,
) -> UsageEmitSummary:
    """Emit one ``lineage_usage`` observation per lineage id.

    Each row carries the lineage_id in ``extra`` plus the boolean
    ``used`` flag and the overlap diagnostics. Parent retrieval event
    is referenced via ``parent_event_id`` so reader code can group.

    Returns the summary; never raises.
    """
    summary = UsageEmitSummary()
    for overlap in overlaps:
        if overlap.used:
            summary.used_count += 1
        obs = Observation.create(
            subsystem="observations",
            event_type="lineage_usage",
            outcome="success" if overlap.used else "unknown_yet",
            parent_event_id=parent_event_id,
            lineage_ids=(overlap.lineage_id,),
            extra=overlap.as_dict(),
        )
        try:
            if writer is not None:
                ok = writer.emit(obs)
            else:
                ok = emit_observation(obs)
        except Exception:
            ok = False
        if ok:
            summary.emitted += 1
        else:
            summary.failed += 1
    return summary
