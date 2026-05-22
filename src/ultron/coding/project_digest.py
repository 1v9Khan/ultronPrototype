"""Project digest generator -- opencode's compaction template
re-shaped for project lifecycle.

After every Claude coding session finishes, this module distills the
session's outcome into a structured markdown digest:

    ## Goal
    ## Constraints & Preferences
    ## Progress
      ### Done
      ### In Progress
      ### Blocked
    ## Key Decisions
    ## Next Steps
    ## Critical Context
    ## Relevant Files

The template mirrors ``packages/opencode/src/session/compaction.ts``'s
``SUMMARY_TEMPLATE``, adapted for project state instead of arbitrary
conversation context (we replaced opencode's tail-protection with
file lists, since project digests target durable state, not recent
turns).

Digests are produced by a single LLM call against the in-process
Qwen voice model. The call runs on a background thread (mirrors
:class:`ultron.memory.background_summarizer.BackgroundSummarizer`'s
posture) so the voice loop is never blocked. Fail-open: any LLM
error, parse failure, or empty result falls back to a deterministic
template-from-task-metadata so the project still has *something*
indexed.

Public surface:

  * :class:`ProjectDigest` -- the produced artifact (path-on-disk,
    markdown body, structured-field cache).
  * :class:`DigestRequest` -- the inputs (project, task summary,
    file changes, optional previous digest).
  * :func:`generate_digest` -- pure sync function returning a
    :class:`ProjectDigest`.
  * :func:`render_template` -- exposed for tests; deterministic
    fallback when no LLM.
  * :func:`parse_digest_sections` -- re-extracts the markdown
    headings into a dict (used by callers that want section
    access without re-LLM-call).

The actual digest write-to-disk + Qdrant upsert is handled by
:class:`ultron.coding.project_index.ProjectIndex`. This module is
pure compute -- inputs in, dataclass out.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ultron.coding.project_digest")


# ---------------------------------------------------------------------------
# Template -- opencode's SUMMARY_TEMPLATE, project-flavored
# ---------------------------------------------------------------------------


# Verbatim section headers (matching opencode/src/session/compaction.ts
# lines 42-77 with file slot adapted for projects). Order matters --
# parse_digest_sections walks in this order.
DIGEST_SECTIONS: List[str] = [
    "Goal",
    "Constraints & Preferences",
    "Progress",
    "Key Decisions",
    "Next Steps",
    "Critical Context",
    "Relevant Files",
]

# Sub-sections under Progress (opencode style).
PROGRESS_SUBSECTIONS: List[str] = ["Done", "In Progress", "Blocked"]


SUMMARY_TEMPLATE = """## Goal
- {goal}

## Constraints & Preferences
{constraints}

## Progress
### Done
{progress_done}

### In Progress
{progress_in_progress}

### Blocked
{progress_blocked}

## Key Decisions
{key_decisions}

## Next Steps
{next_steps}

## Critical Context
{critical_context}

## Relevant Files
{relevant_files}
"""

# Prompt template fed to the LLM. Mirrors opencode's compaction prompt
# style (anchored summary, terse bullets, no meta-commentary). Built
# at call time with the actual session data substituted in.
DIGEST_PROMPT_PROLOGUE = """You are summarizing a coding-session outcome for long-term project memory.

Rules:
  * Use the markdown template below VERBATIM. Keep EVERY heading and sub-heading, even if empty -- write "(none)" in empty sections.
  * Terse bullets only. No paragraphs. No meta-commentary about the summarization itself.
  * Preserve EXACT file paths, command names, and error strings as they appear in the session record.
  * Under "Relevant Files", one line per file with format "- path: why it matters".
  * Under "Key Decisions", capture choices that future sessions need to honor (e.g. "chose Flask over FastAPI for simplicity", "PostgreSQL preferred over SQLite for production").
  * Under "Critical Context", include unresolved errors, open questions, environmental quirks.
  * Under "Next Steps", concrete actions a future session could take, ordered by priority."""

DIGEST_PROMPT_PRIOR_PROLOGUE = """You are UPDATING an existing project digest with new information from a recently-completed coding session.

Rules:
  * Start from the prior digest. Merge new facts. Drop superseded items.
  * Use the markdown template VERBATIM. Keep EVERY heading. Write "(none)" in empty sections.
  * Preserve user constraints and preferences across sessions unless the user explicitly changed them.
  * Move items from "In Progress" to "Done" when the session completed them.
  * Add new items to "In Progress" when work is partial.
  * Remove items from "Next Steps" that were done; add new ones surfaced by the session."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DigestRequest:
    """Inputs needed to generate a project digest.

    Attributes:
        project_name: canonical project name from
            :class:`ultron.coding.projects.ProjectRegistry`.
        project_path: absolute on-disk path.
        task_summary: final assistant text from the Claude session.
        files_created: paths created in this session.
        files_modified: paths modified.
        files_deleted: paths deleted.
        prior_digest_markdown: previous digest body when one exists;
            triggers update-mode prompt instead of create-mode.
        user_goal_hint: the original user utterance that started the
            session (e.g. "build a PDF-to-DOCX converter with a Tkinter UI"),
            used to seed the Goal section when no prior digest.
        language: detected language ("python", "javascript", ...);
            seeded by :mod:`ultron.coding.project_introspect`.
        entry_points: list of detected entry-point file paths.
    """

    project_name: str
    project_path: Path
    task_summary: str
    files_created: List[Path] = field(default_factory=list)
    files_modified: List[Path] = field(default_factory=list)
    files_deleted: List[Path] = field(default_factory=list)
    prior_digest_markdown: str = ""
    user_goal_hint: str = ""
    language: str = ""
    entry_points: List[Path] = field(default_factory=list)


@dataclass
class ProjectDigest:
    """Output of :func:`generate_digest`.

    Attributes:
        project_name: from the originating request.
        project_path: from the originating request.
        markdown: full markdown body (template-shaped).
        sections: parsed-out section map (key = heading text).
        generated_at: monotonic-equivalent timestamp (wall clock).
        elapsed_ms: time the digest call took.
        fallback: True when the LLM call failed or returned empty and
            we built a deterministic template from request metadata
            instead. Callers can use this to surface a hint that the
            digest is best-effort.
        source: "llm" | "template" | "manual" -- where the body came
            from. ``"manual"`` is reserved for future external writes.
    """

    project_name: str
    project_path: Path
    markdown: str
    sections: Dict[str, str] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)
    elapsed_ms: float = 0.0
    fallback: bool = False
    source: str = "llm"


# Type alias for the LLM-call callable. Accepts a single prompt string,
# returns the full completion text. Pass an
# :class:`ultron.llm.inference.LLMEngine`-bound function or a stub
# in tests.
LLMCallable = Callable[[str], str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_digest(
    request: DigestRequest,
    llm_call: Optional[LLMCallable] = None,
    *,
    max_files_in_prompt: int = 40,
    max_summary_chars: int = 4000,
) -> ProjectDigest:
    """Produce a :class:`ProjectDigest` from a completed coding session.

    Args:
        request: inputs (project + task outcome + optional prior digest).
        llm_call: a callable that takes the digest-generation prompt
            and returns the model completion. When ``None`` (or the
            call raises / returns empty), :func:`render_template` is
            used as a deterministic fallback.
        max_files_in_prompt: cap on the number of file paths embedded
            in the prompt context; protects against huge file lists
            inflating context.
        max_summary_chars: cap on the assistant-text snippet sent to
            the LLM. Long Claude summaries get truncated.

    Returns:
        :class:`ProjectDigest` with ``fallback=True`` when the LLM
        path didn't yield usable output.
    """
    t0 = time.monotonic()

    if llm_call is None:
        return _fallback_digest(request, t0)

    prompt = _build_digest_prompt(
        request,
        max_files=max_files_in_prompt,
        max_summary_chars=max_summary_chars,
    )

    try:
        completion = llm_call(prompt)
    except Exception as e:                                          # noqa: BLE001
        logger.warning(
            "project_digest: LLM call failed (%s); falling back to template.",
            e,
        )
        return _fallback_digest(request, t0)

    if not completion or not completion.strip():
        logger.info(
            "project_digest: LLM returned empty; falling back to template.",
        )
        return _fallback_digest(request, t0)

    markdown = _normalize_digest_markdown(completion)
    sections = parse_digest_sections(markdown)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    return ProjectDigest(
        project_name=request.project_name,
        project_path=request.project_path,
        markdown=markdown,
        sections=sections,
        elapsed_ms=elapsed_ms,
        fallback=False,
        source="llm",
    )


def render_template(
    request: DigestRequest,
) -> str:
    """Build a deterministic digest markdown body from request metadata.

    Used as the fallback when no LLM is provided OR when the LLM call
    fails. Doesn't try to be clever -- captures the goal hint + a
    factual rundown of file changes + the assistant summary verbatim.
    Future LLM-driven re-summarization can supplant it.
    """
    goal = (
        request.user_goal_hint.strip()
        or f"work on project {request.project_name}"
    )

    progress_done = _format_file_list(request.files_created, prefix="created")
    if request.files_modified:
        modified_block = _format_file_list(request.files_modified, prefix="modified")
        progress_done = (
            f"{progress_done}\n{modified_block}"
            if progress_done.strip() != "- (none)"
            else modified_block
        )
    if not progress_done.strip() or progress_done.strip() == "- (none)":
        if request.task_summary:
            progress_done = f"- {request.task_summary[:200].strip()}"

    relevant_files = _format_file_relevance(
        sorted({*request.files_created, *request.files_modified}),
        entry_points=request.entry_points,
    )

    deletions = _format_file_list(request.files_deleted, prefix="deleted")

    critical_context_parts: List[str] = []
    if request.language:
        critical_context_parts.append(f"- Language: {request.language}")
    if deletions.strip() and deletions.strip() != "- (none)":
        critical_context_parts.append(deletions)
    critical_context = (
        "\n".join(critical_context_parts) if critical_context_parts else "- (none)"
    )

    return SUMMARY_TEMPLATE.format(
        goal=goal,
        constraints="- (none)",
        progress_done=progress_done,
        progress_in_progress="- (none)",
        progress_blocked="- (none)",
        key_decisions="- (none)",
        next_steps="- (none)",
        critical_context=critical_context,
        relevant_files=relevant_files,
    ).strip()


def parse_digest_sections(markdown: str) -> Dict[str, str]:
    """Walk a digest body and return ``{section_header: body_text}``.

    Returns one entry per top-level ``## Header`` -- ``###`` sub-headers
    stay inside their parent's body. Body text is everything between
    one header and the next, stripped of leading/trailing whitespace.

    Robust to LLM-induced trailing whitespace, blank-line variance,
    and slight section-name capitalization drift (matches case-
    insensitively against :data:`DIGEST_SECTIONS`).
    """
    if not markdown:
        return {}

    sections: Dict[str, str] = {}
    current_key: Optional[str] = None
    buffer: List[str] = []

    # Build a lookup of canonical section names by lowercased form so
    # we can match the model's output even when it lowercases or
    # changes spacing of a heading.
    canonical = {name.lower(): name for name in DIGEST_SECTIONS}

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            # Flush previous section.
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            header = stripped[3:].strip()
            current_key = canonical.get(header.lower(), header)
            buffer = []
            continue
        buffer.append(line)

    # Flush trailing section.
    if current_key is not None:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


def extract_files_from_digest(markdown: str) -> List[str]:
    """Pull out the relevant-files list from a digest markdown body.

    Returns a list of file path strings (just the path portion,
    not the explanatory note that follows the colon). Empty when
    the section is missing or marked ``(none)``.
    """
    sections = parse_digest_sections(markdown)
    relevant = sections.get("Relevant Files", "")
    if not relevant or relevant.lower().startswith("- (none)"):
        return []

    paths: List[str] = []
    for raw_line in relevant.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:].strip()
        if not body or body.lower() == "(none)":
            continue
        # Format is "path: why it matters" per the template; take just
        # the path portion. Fall back to the whole entry when there's
        # no colon so we don't drop ill-formed entries silently.
        path_part = body.split(":", 1)[0].strip()
        if path_part:
            paths.append(path_part)
    return paths


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_digest_prompt(
    request: DigestRequest,
    *,
    max_files: int,
    max_summary_chars: int,
) -> str:
    """Assemble the prompt text the LLM gets.

    Mirrors opencode's compaction-prompt assembly in
    ``packages/opencode/src/session/compaction.ts``: prior-summary
    anchor when present, terse rules, template at the bottom.
    """
    if request.prior_digest_markdown.strip():
        prologue = DIGEST_PROMPT_PRIOR_PROLOGUE
        prior_block = (
            "\n\n<previous-summary>\n"
            f"{request.prior_digest_markdown.strip()}\n"
            "</previous-summary>"
        )
    else:
        prologue = DIGEST_PROMPT_PROLOGUE
        prior_block = ""

    files_summary = _summarize_files_for_prompt(
        request.files_created,
        request.files_modified,
        request.files_deleted,
        cap=max_files,
    )

    truncated_summary = (
        (request.task_summary or "").strip()[:max_summary_chars]
    )

    entry_points_block = ""
    if request.entry_points:
        ep_lines = "\n".join(
            f"- {p}" for p in request.entry_points[:5]
        )
        entry_points_block = f"\n\nDetected entry points:\n{ep_lines}"

    goal_block = ""
    if request.user_goal_hint:
        goal_block = (
            f"\n\nUser's original request that started this session:\n"
            f"  {request.user_goal_hint.strip()[:500]}"
        )

    language_block = (
        f"\n\nDetected language: {request.language}"
        if request.language
        else ""
    )

    template_block = (
        "\n\nUse this exact template:\n\n" + SUMMARY_TEMPLATE
    )

    return (
        f"{prologue}{prior_block}\n\n"
        f"Project: {request.project_name}\n"
        f"Path: {request.project_path}"
        f"{language_block}"
        f"{goal_block}"
        f"{entry_points_block}\n\n"
        f"Session file changes:\n{files_summary}\n\n"
        f"Session final assistant text:\n"
        f"  {truncated_summary}"
        f"{template_block}"
    )


def _summarize_files_for_prompt(
    created: List[Path],
    modified: List[Path],
    deleted: List[Path],
    *,
    cap: int,
) -> str:
    """Build a compact file-change summary for the prompt body.

    Caps at ``cap`` total entries; surplus is collapsed to a
    "+N more" trailer so the prompt doesn't blow past context on
    sessions that touched dozens of files.
    """
    if not (created or modified or deleted):
        return "  (no file changes)"

    lines: List[str] = []
    consumed = 0

    def _emit(prefix: str, paths: List[Path]) -> None:
        nonlocal consumed
        for p in paths:
            if consumed >= cap:
                return
            lines.append(f"  {prefix}: {p}")
            consumed += 1

    _emit("created", created)
    _emit("modified", modified)
    _emit("deleted", deleted)

    total = len(created) + len(modified) + len(deleted)
    if consumed < total:
        lines.append(f"  ... +{total - consumed} more")

    return "\n".join(lines)


def _format_file_list(
    paths: List[Path], *, prefix: str,
) -> str:
    """Format a file list as markdown bullets. Returns "- (none)" when empty."""
    if not paths:
        return "- (none)"
    return "\n".join(f"- {prefix}: {p}" for p in paths[:30])


def _format_file_relevance(
    paths: List[Path], *, entry_points: List[Path],
) -> str:
    """Build the "Relevant Files" section from touched files + entry points.

    Entry points always go first (they're durably relevant), then
    touched files. Each line is ``- path: why it matters``.
    """
    if not (paths or entry_points):
        return "- (none)"

    entry_set = {str(p): True for p in entry_points}
    lines: List[str] = []
    seen: set = set()

    for p in entry_points[:10]:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {p}: entry point")

    for p in paths[:30]:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        note = "modified this session"
        lines.append(f"- {p}: {note}")

    return "\n".join(lines) if lines else "- (none)"


def _normalize_digest_markdown(raw: str) -> str:
    """Tidy LLM output: strip code fences if the model wrapped the digest,
    collapse trailing whitespace, ensure newline-at-end-of-file."""
    text = raw.strip()
    # Strip a leading ```markdown / ```md / ``` fence if the model wrapped.
    fence_match = re.match(r"^```(?:markdown|md)?\s*\n", text)
    if fence_match:
        text = text[fence_match.end():]
        # Strip the trailing fence.
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text + "\n"


def _fallback_digest(request: DigestRequest, t0: float) -> ProjectDigest:
    """Build a :class:`ProjectDigest` from the deterministic template.

    Used when no LLM was provided OR the LLM call failed. The
    resulting digest has ``fallback=True`` + ``source="template"``
    so downstream consumers know it's best-effort.
    """
    markdown = render_template(request)
    sections = parse_digest_sections(markdown)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    return ProjectDigest(
        project_name=request.project_name,
        project_path=request.project_path,
        markdown=markdown,
        sections=sections,
        elapsed_ms=elapsed_ms,
        fallback=True,
        source="template",
    )


__all__ = [
    "DIGEST_SECTIONS",
    "PROGRESS_SUBSECTIONS",
    "SUMMARY_TEMPLATE",
    "DigestRequest",
    "LLMCallable",
    "ProjectDigest",
    "extract_files_from_digest",
    "generate_digest",
    "parse_digest_sections",
    "render_template",
]
