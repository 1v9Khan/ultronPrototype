"""Extended ``@``-mention resolvers (URLs, problems, memory, clipboard, etc.).

Adapted from cline's ``parseMentions`` pattern (Apache 2.0; see
``THIRD_PARTY_NOTICES.md``). Ultron already had a file-only resolver
ported from aider in :mod:`ultron.coding.file_mention_resolver`; this
module sits next to it as the EXTENDED resolver covering the
non-file forms that the cline catalog T14 calls out:

* ``@path/to/file.py`` — local file (delegated to the existing
  file resolver).
* ``@workspace:relpath`` — multi-root workspace prefix.
* ``@http(s)://url`` / ``@ftp://url`` — auto-fetched via an injected
  reader chain.
* ``@problems`` or ``@errors`` — pulls the active lint / diagnostic
  payload from a provider.
* ``@memory:<topic>`` — runs a RAG query and embeds the top-k snippets.
* ``@last`` — the most-recently touched file (provider-supplied).
* ``@diff`` — current git working-tree diff (provider-supplied).
* ``@clipboard`` — system clipboard contents (provider-supplied).
* ``@screenshot`` — VLM-described screenshot (provider-supplied).

Each mention resolves to a `<mention kind="..." source="...">...</mention>`
XML block that the LLM can parse. The orchestrator wires the
appropriate providers (reader chain, lint state, RAG, clipboard, VLM)
into a single :class:`MentionResolutionContext` and calls
:func:`resolve_extended_mentions` on the user text BEFORE the prompt
is built.

The resolver is intentionally provider-driven (every external surface
is a callable injected at construction time) so the unit tests stay
hermetic and the orchestrator chooses what to wire.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)


#: Cap on the rendered body of any single mention (chars) — keeps a
#: pathological URL or RAG return from blowing the prompt budget.
DEFAULT_MAX_BODY_CHARS: int = 8000

#: Cap on the number of mentions resolved per call. Anything beyond
#: this is reported as a truncation note instead of expanding inline.
DEFAULT_MAX_MENTIONS_PER_CALL: int = 16

#: Token used in rendered mention blocks (mirrors the aider resolver).
MENTION_OPEN: str = "<mention"
MENTION_CLOSE: str = "</mention>"

#: Heuristic mention regex. Accepts ``@thing`` where ``thing`` is one of:
#: ``http(s)://...``, ``ftp://...``, ``workspace:relpath``,
#: ``memory:topic``, ``problems``/``errors``/``last``/``diff``/
#: ``clipboard``/``screenshot``, or a bare path-like token.
_MENTION_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![\w@])@(?P<body>"
    r"(?:https?|ftp)://\S+"
    r"|workspace:[A-Za-z0-9_./\\\-:]+"
    r"|memory:[A-Za-z0-9_./\-]+"
    r"|problems|errors|last|diff|clipboard|screenshot"
    r"|[A-Za-z]:[/\\][A-Za-z0-9_./\\\-]+"
    r"|[A-Za-z0-9_./\\\-]+\.[A-Za-z0-9]{1,10}"
    r"|[A-Za-z0-9_./\\\-]+(?:/[A-Za-z0-9_./\\\-]+)+"
    r")",
)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

UrlFetcher = Callable[[str], Optional[str]]
"""Callable mapping a URL to its fetched text body (or None on failure)."""

LintProvider = Callable[[], str]
"""Callable returning the current lint / diagnostic payload as text."""

MemoryProvider = Callable[[str, int], Sequence[tuple[str, str]]]
"""Callable mapping ``(query, top_k)`` to ``[(label, body), ...]``."""

LastFileProvider = Callable[[], Optional[str]]
"""Callable returning the most-recently touched file path."""

DiffProvider = Callable[[], str]
"""Callable returning the current git working-tree diff."""

ClipboardProvider = Callable[[], str]
"""Callable returning the system clipboard text."""

ScreenshotProvider = Callable[[], Optional[str]]
"""Callable returning a textual description of a fresh screenshot."""

WorkspacePathResolver = Callable[[str, str], Optional[Path]]
"""Callable mapping ``(workspace_label, rel_path)`` to a resolved Path."""


@dataclass
class MentionResolutionContext:
    """Container for the provider callables consulted during resolution.

    Each provider is optional. When a mention's provider is missing,
    the resolver logs a WARN and emits a ``<mention kind="missing" .../>``
    block so the LLM knows the reference was unresolvable.

    Attributes:
        url_fetcher: fetches URL content for ``@http(s)://...``.
        lint_provider: returns the current lint payload for ``@problems``.
        memory_provider: runs RAG for ``@memory:topic``.
        last_file_provider: resolves ``@last``.
        diff_provider: returns git diff for ``@diff``.
        clipboard_provider: returns clipboard text for ``@clipboard``.
        screenshot_provider: returns screenshot description for
            ``@screenshot``.
        workspace_resolver: resolves ``@workspace:rel`` to an absolute
            path (caller-supplied; defaults to a single-workspace
            joiner).
        file_reader: callable reading an absolute path to a string. Used
            for both bare ``@path`` and ``@workspace:path`` forms.
        memory_top_k: default top-k for memory queries.
    """

    url_fetcher: Optional[UrlFetcher] = None
    lint_provider: Optional[LintProvider] = None
    memory_provider: Optional[MemoryProvider] = None
    last_file_provider: Optional[LastFileProvider] = None
    diff_provider: Optional[DiffProvider] = None
    clipboard_provider: Optional[ClipboardProvider] = None
    screenshot_provider: Optional[ScreenshotProvider] = None
    workspace_resolver: Optional[WorkspacePathResolver] = None
    file_reader: Optional[Callable[[Path], str]] = None
    memory_top_k: int = 3
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS
    max_mentions_per_call: int = DEFAULT_MAX_MENTIONS_PER_CALL


@dataclass(frozen=True)
class ResolvedMention:
    """Outcome of one mention resolution.

    Attributes:
        original: the literal ``@...`` substring that triggered the
            resolution.
        kind: classification (``file`` / ``url`` / ``problems`` /
            ``memory`` / etc.).
        source: short identifier (path / URL / topic / etc.).
        body: rendered body that will be embedded in the prompt.
        error: optional error string when resolution failed.
    """

    original: str
    kind: str
    source: str
    body: str = ""
    error: Optional[str] = None


@dataclass(frozen=True)
class MentionResolutionResult:
    """Outcome of a whole-text mention resolution pass.

    Attributes:
        original_text: the unmodified input text.
        rewritten_text: the text with each mention expanded into its
            rendered block (or kept verbatim when resolution failed).
        mentions: per-mention resolution records (in source order).
        truncated_count: number of mentions deferred past the per-call
            cap.
    """

    original_text: str
    rewritten_text: str
    mentions: tuple[ResolvedMention, ...] = field(default_factory=tuple)
    truncated_count: int = 0


# ---------------------------------------------------------------------------
# Per-kind resolution helpers
# ---------------------------------------------------------------------------

def _truncate(body: str, max_chars: int) -> str:
    if body is None:
        return ""
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip() + "\n... (truncated)"


def _render(
    kind: str,
    source: str,
    body: str,
    *,
    extra: Optional[Mapping[str, str]] = None,
) -> str:
    attrs = [f'kind="{kind}"', f'source="{_escape_attr(source)}"']
    if extra:
        for key, value in extra.items():
            attrs.append(f'{key}="{_escape_attr(value)}"')
    return f"{MENTION_OPEN} {' '.join(attrs)}>\n{body}\n{MENTION_CLOSE}"


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _missing(original: str, kind: str, source: str, reason: str) -> ResolvedMention:
    body = (
        f"[Note] {kind} mention '{source}' could not be resolved: {reason}"
    )
    return ResolvedMention(
        original=original,
        kind="missing",
        source=source,
        body=_render("missing", source, body, extra={"requested_kind": kind}),
        error=reason,
    )


def _classify(body: str) -> tuple[str, str]:
    """Map a mention body to ``(kind, source)``."""
    lowered = body.lower()
    if lowered.startswith(("http://", "https://", "ftp://")):
        return "url", body
    if lowered.startswith("workspace:"):
        return "workspace", body[len("workspace:"):]
    if lowered.startswith("memory:"):
        return "memory", body[len("memory:"):]
    if lowered in ("problems", "errors"):
        return "problems", body
    if lowered == "last":
        return "last", body
    if lowered == "diff":
        return "diff", body
    if lowered == "clipboard":
        return "clipboard", body
    if lowered == "screenshot":
        return "screenshot", body
    return "file", body


def _resolve_one(
    original: str,
    body: str,
    ctx: MentionResolutionContext,
) -> ResolvedMention:
    kind, source = _classify(body)
    try:
        if kind == "url":
            if ctx.url_fetcher is None:
                return _missing(original, kind, source, "no url_fetcher provider")
            fetched = ctx.url_fetcher(source)
            if fetched is None:
                return _missing(original, kind, source, "fetch returned None")
            body_text = _truncate(fetched, ctx.max_body_chars)
            return ResolvedMention(
                original=original, kind=kind, source=source,
                body=_render(kind, source, body_text),
            )
        if kind == "workspace":
            label, _, rel_path = source.partition(":")
            if not rel_path:
                label, rel_path = "", source
            if ctx.workspace_resolver is None or ctx.file_reader is None:
                return _missing(
                    original, kind, source,
                    "workspace_resolver or file_reader missing",
                )
            resolved = ctx.workspace_resolver(label, rel_path)
            if resolved is None:
                return _missing(
                    original, kind, source,
                    "workspace_resolver returned None",
                )
            content = ctx.file_reader(resolved)
            return ResolvedMention(
                original=original, kind="file", source=str(resolved),
                body=_render(
                    "file",
                    str(resolved),
                    _truncate(content, ctx.max_body_chars),
                    extra={"workspace": label or "default"},
                ),
            )
        if kind == "file":
            if ctx.file_reader is None:
                return _missing(original, kind, source, "no file_reader provider")
            try:
                content = ctx.file_reader(Path(source))
            except FileNotFoundError:
                return _missing(original, kind, source, "file not found")
            return ResolvedMention(
                original=original, kind=kind, source=source,
                body=_render(
                    kind, source, _truncate(content, ctx.max_body_chars),
                ),
            )
        if kind == "memory":
            if ctx.memory_provider is None:
                return _missing(
                    original, kind, source, "no memory_provider provider",
                )
            top_k = max(1, int(ctx.memory_top_k))
            snippets = ctx.memory_provider(source, top_k) or []
            if not snippets:
                return _missing(original, kind, source, "no snippets returned")
            rendered_parts = [
                f"### {label}\n{_truncate(snippet, ctx.max_body_chars // top_k)}"
                for label, snippet in snippets
            ]
            return ResolvedMention(
                original=original, kind=kind, source=source,
                body=_render(
                    kind, source, "\n\n".join(rendered_parts),
                    extra={"top_k": str(top_k)},
                ),
            )
        if kind == "problems":
            if ctx.lint_provider is None:
                return _missing(original, kind, source, "no lint_provider provider")
            text = ctx.lint_provider() or ""
            return ResolvedMention(
                original=original, kind=kind, source=source,
                body=_render(kind, source, _truncate(text, ctx.max_body_chars)),
            )
        if kind == "last":
            if ctx.last_file_provider is None:
                return _missing(
                    original, kind, source, "no last_file_provider provider",
                )
            target = ctx.last_file_provider()
            if not target:
                return _missing(original, kind, source, "no recent file")
            if ctx.file_reader is None:
                return _missing(
                    original, kind, source, "no file_reader provider",
                )
            content = ctx.file_reader(Path(target))
            return ResolvedMention(
                original=original, kind="file", source=target,
                body=_render(
                    "file", target, _truncate(content, ctx.max_body_chars),
                    extra={"resolved_from": "@last"},
                ),
            )
        if kind == "diff":
            if ctx.diff_provider is None:
                return _missing(
                    original, kind, source, "no diff_provider provider",
                )
            text = ctx.diff_provider() or ""
            return ResolvedMention(
                original=original, kind=kind, source=source,
                body=_render(kind, source, _truncate(text, ctx.max_body_chars)),
            )
        if kind == "clipboard":
            if ctx.clipboard_provider is None:
                return _missing(
                    original, kind, source, "no clipboard_provider provider",
                )
            text = ctx.clipboard_provider() or ""
            return ResolvedMention(
                original=original, kind=kind, source=source,
                body=_render(kind, source, _truncate(text, ctx.max_body_chars)),
            )
        if kind == "screenshot":
            if ctx.screenshot_provider is None:
                return _missing(
                    original, kind, source, "no screenshot_provider provider",
                )
            description = ctx.screenshot_provider() or ""
            if not description:
                return _missing(
                    original, kind, source, "screenshot returned empty",
                )
            return ResolvedMention(
                original=original, kind=kind, source=source,
                body=_render(
                    kind, source, _truncate(description, ctx.max_body_chars),
                ),
            )
        return _missing(original, kind, source, f"unknown kind {kind!r}")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "mention resolution raised for %s: %s", original, exc, exc_info=True,
        )
        return _missing(original, kind, source, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def find_mentions(text: str) -> list[str]:
    """Return the raw ``@thing`` substrings found in ``text`` (in order)."""
    if not text:
        return []
    out: list[str] = []
    seen: set[int] = set()
    for match in _MENTION_PATTERN.finditer(text):
        start = match.start()
        if start in seen:
            continue
        seen.add(start)
        out.append(match.group(0))
    return out


def resolve_extended_mentions(
    text: str,
    ctx: MentionResolutionContext,
) -> MentionResolutionResult:
    """Resolve every mention in ``text`` and return the rewritten string.

    Args:
        text: arbitrary user transcript (or any LLM-bound text).
        ctx: provider container.

    Returns:
        :class:`MentionResolutionResult` with the rewritten text +
        per-mention resolution records.
    """
    if not text:
        return MentionResolutionResult(
            original_text="", rewritten_text="",
        )
    matches = list(_MENTION_PATTERN.finditer(text))
    if not matches:
        return MentionResolutionResult(
            original_text=text, rewritten_text=text,
        )
    cap = max(1, int(ctx.max_mentions_per_call))
    keep = matches[:cap]
    truncated = len(matches) - len(keep)

    mentions: list[ResolvedMention] = []
    # Build replacement segments + rebuild text in one pass.
    rebuilt: list[str] = []
    cursor = 0
    seen_originals: dict[str, ResolvedMention] = {}
    for match in keep:
        rebuilt.append(text[cursor:match.start()])
        original = match.group(0)
        body = match.group("body")
        # Dedup identical mentions within a single call: reuse the prior
        # resolution to avoid duplicate work (and duplicate prompt bloat).
        if original in seen_originals:
            resolution = seen_originals[original]
        else:
            resolution = _resolve_one(original, body, ctx)
            seen_originals[original] = resolution
        mentions.append(resolution)
        rebuilt.append(resolution.body or original)
        cursor = match.end()
    rebuilt.append(text[cursor:])
    if truncated > 0:
        rebuilt.append(
            f"\n[Note] {truncated} additional mention(s) deferred past the "
            f"per-call cap of {cap}.",
        )
    return MentionResolutionResult(
        original_text=text,
        rewritten_text="".join(rebuilt),
        mentions=tuple(mentions),
        truncated_count=truncated,
    )


__all__ = [
    "DEFAULT_MAX_BODY_CHARS",
    "DEFAULT_MAX_MENTIONS_PER_CALL",
    "ClipboardProvider",
    "DiffProvider",
    "LastFileProvider",
    "LintProvider",
    "MemoryProvider",
    "MentionResolutionContext",
    "MentionResolutionResult",
    "ResolvedMention",
    "ScreenshotProvider",
    "UrlFetcher",
    "WorkspacePathResolver",
    "find_mentions",
    "resolve_extended_mentions",
]
