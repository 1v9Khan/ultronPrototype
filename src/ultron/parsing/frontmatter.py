"""YAML frontmatter parser with fail-open per-file semantics.

This is a clean-room adaptation of the ``_parse_skill_frontmatter`` pattern
from ``openhands/app_server/user/skills_router.py``. The shape stays the
same -- look for a leading ``---``, find the closing ``---``, ``yaml.safe_load``
the slice, catch ``yaml.YAMLError`` and log a warning -- but ultron's
version returns BOTH the frontmatter dict and the post-frontmatter body
text in a single :class:`FrontmatterResult`, because every ultron use site
(skills, identity overrides, project safety rules) wants both pieces.

Fail-open contract: every error path returns a :class:`FrontmatterResult`
with ``frontmatter=None`` and ``error`` populated. The directory walker
swallows per-file failures so one bad file never breaks discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

logger = logging.getLogger(__name__)

_FRONTMATTER_DELIMITER = "---"
_DEFAULT_FILE_EXTENSIONS: tuple[str, ...] = (".md",)
_DEFAULT_SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
)


@dataclass(frozen=True)
class FrontmatterResult:
    """Result of parsing a markdown / text file's YAML frontmatter.

    Attributes:
        path: Source file the result was produced from (may be a synthetic
            path for the ``parse_frontmatter_text`` entry point).
        frontmatter: The parsed YAML mapping, or ``None`` when the file has
            no frontmatter or the YAML failed to parse.
        body: The post-frontmatter text (the entire text when no
            frontmatter is present).
        error: ``None`` when parsing succeeded, otherwise a short
            human-readable description of what went wrong.

    The frozen dataclass shape mirrors ultron's other result records
    (``LintReport``, ``EditDiagnosticResult``, etc.) so consumers can
    pattern-match on the same attribute names.
    """

    path: Path
    frontmatter: dict[str, Any] | None
    body: str
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_frontmatter(self) -> bool:
        """``True`` iff the file had a parseable frontmatter block."""

        return self.frontmatter is not None

    @property
    def ok(self) -> bool:
        """``True`` iff parsing produced no error (frontmatter may still be absent)."""

        return self.error is None

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience: ``result.get("name", "default")`` reads from frontmatter."""

        if self.frontmatter is None:
            return default
        return self.frontmatter.get(key, default)


def parse_frontmatter_text(
    text: str,
    *,
    source_path: Path | str | None = None,
) -> FrontmatterResult:
    """Parse YAML frontmatter from a raw text string.

    Args:
        text: The file contents.
        source_path: Optional logical path used for logging + the returned
            result's ``path`` field. ``None`` becomes ``Path("<text>")``.

    Returns:
        A :class:`FrontmatterResult`. When the text has no leading
        ``---``, returns the text as the body with ``frontmatter=None``
        and ``error=None``. When YAML parsing fails, returns
        ``frontmatter=None`` with ``error`` populated; the body is the
        original text so callers can still process it.
    """

    if source_path is None:
        resolved_path = Path("<text>")
    elif isinstance(source_path, str):
        resolved_path = Path(source_path)
    else:
        resolved_path = source_path

    if not text.startswith(_FRONTMATTER_DELIMITER):
        return FrontmatterResult(
            path=resolved_path,
            frontmatter=None,
            body=text,
        )

    # Skip past the opening delimiter; tolerate trailing whitespace / newline.
    after_open = len(_FRONTMATTER_DELIMITER)
    # Allow either '---\n' or '---\r\n' immediately after the opener.
    if after_open < len(text) and text[after_open] in ("\n", "\r"):
        after_open += 1
        if after_open < len(text) and text[after_open - 1] == "\r" and text[after_open] == "\n":
            after_open += 1

    # Look for the closing delimiter either on its own line (preceded by '\n')
    # OR immediately after the opener (the "empty frontmatter" edge case).
    delim_offset: int
    delim_includes_leading_newline: bool
    if text.startswith(_FRONTMATTER_DELIMITER, after_open):
        delim_offset = after_open
        delim_includes_leading_newline = False
    else:
        newline_offset = text.find(f"\n{_FRONTMATTER_DELIMITER}", after_open)
        if newline_offset == -1:
            message = "no closing '---' delimiter found"
            logger.warning("Frontmatter parse failed for %s: %s", resolved_path, message)
            return FrontmatterResult(
                path=resolved_path,
                frontmatter=None,
                body=text,
                error=message,
            )
        delim_offset = newline_offset
        delim_includes_leading_newline = True

    frontmatter_slice = text[after_open:delim_offset]
    # Body starts AFTER the closing delimiter (+ the preceding '\n', if we matched one).
    body_start = delim_offset + len(_FRONTMATTER_DELIMITER) + (1 if delim_includes_leading_newline else 0)
    if body_start < len(text) and text[body_start] in ("\n", "\r"):
        body_start += 1
        if body_start < len(text) and text[body_start - 1] == "\r" and text[body_start] == "\n":
            body_start += 1
    body = text[body_start:]

    try:
        parsed = yaml.safe_load(frontmatter_slice)
    except yaml.YAMLError as exc:
        message = f"YAML error: {exc}"
        logger.warning("Frontmatter parse failed for %s: %s", resolved_path, message)
        return FrontmatterResult(
            path=resolved_path,
            frontmatter=None,
            body=body,
            error=message,
        )

    if parsed is None:
        # Empty frontmatter block (e.g. "---\n---\nbody"). Treat as present-but-empty.
        return FrontmatterResult(
            path=resolved_path,
            frontmatter={},
            body=body,
        )

    if not isinstance(parsed, dict):
        message = f"frontmatter is not a mapping (got {type(parsed).__name__})"
        logger.warning("Frontmatter parse failed for %s: %s", resolved_path, message)
        return FrontmatterResult(
            path=resolved_path,
            frontmatter=None,
            body=body,
            error=message,
        )

    return FrontmatterResult(
        path=resolved_path,
        frontmatter=parsed,
        body=body,
    )


def parse_frontmatter(
    file_path: Path | str,
    *,
    encoding: str = "utf-8",
) -> FrontmatterResult:
    """Parse YAML frontmatter from a file on disk.

    Reads the file, then delegates to :func:`parse_frontmatter_text`.

    Args:
        file_path: Path to read.
        encoding: Text encoding (default ``utf-8``).

    Returns:
        A :class:`FrontmatterResult`. File-read errors (missing file,
        permission denied, decode error) produce a result with
        ``error`` populated; the body is empty.
    """

    resolved_path = Path(file_path)

    try:
        text = resolved_path.read_text(encoding=encoding)
    except FileNotFoundError:
        message = "file not found"
        logger.warning("Frontmatter parse failed for %s: %s", resolved_path, message)
        return FrontmatterResult(
            path=resolved_path,
            frontmatter=None,
            body="",
            error=message,
        )
    except (OSError, PermissionError) as exc:
        message = f"I/O error: {exc}"
        logger.warning("Frontmatter parse failed for %s: %s", resolved_path, message)
        return FrontmatterResult(
            path=resolved_path,
            frontmatter=None,
            body="",
            error=message,
        )
    except UnicodeDecodeError as exc:
        message = f"decode error: {exc}"
        logger.warning("Frontmatter parse failed for %s: %s", resolved_path, message)
        return FrontmatterResult(
            path=resolved_path,
            frontmatter=None,
            body="",
            error=message,
        )

    return parse_frontmatter_text(text, source_path=resolved_path)


def walk_directory_with_frontmatter(
    directory: Path | str,
    *,
    extensions: Iterable[str] = _DEFAULT_FILE_EXTENSIONS,
    recursive: bool = True,
    skip_directories: Iterable[str] = _DEFAULT_SKIP_DIRECTORIES,
    skip_filenames: Iterable[str] = ("README.md", "readme.md", "Readme.md"),
    encoding: str = "utf-8",
) -> Iterator[FrontmatterResult]:
    """Walk a directory and yield :class:`FrontmatterResult` per matching file.

    Args:
        directory: Root directory to walk.
        extensions: File extensions to consider (must include the leading
            dot). Default: ``(".md",)``.
        recursive: When ``True``, descend into subdirectories. When ``False``,
            scan only the top level.
        skip_directories: Directory names to skip during the walk. Applied
            during ``rglob`` filtering. Defaults skip caches + VCS dirs.
        skip_filenames: Filenames to skip outright (case-sensitive).
        encoding: Text encoding to use when reading files.

    Yields:
        :class:`FrontmatterResult` per discovered file. Per-file exceptions
        beyond the documented read / parse modes are caught and logged;
        the iterator continues with subsequent files.
    """

    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return

    skip_dir_set = set(skip_directories)
    skip_name_set = set(skip_filenames)
    ext_lower = tuple(ext.lower() for ext in extensions)

    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.iterdir()

    for candidate in iterator:
        try:
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in ext_lower:
                continue
            if candidate.name in skip_name_set:
                continue
            # Skip any candidate whose path contains a skipped directory name.
            if any(part in skip_dir_set for part in candidate.parts):
                continue
        except OSError as exc:
            logger.warning("Walk skipped %s: %s", candidate, exc)
            continue

        try:
            yield parse_frontmatter(candidate, encoding=encoding)
        except Exception as exc:  # pragma: no cover - belt-and-braces fail-open
            logger.warning("Unexpected error parsing %s: %s", candidate, exc)
            yield FrontmatterResult(
                path=candidate,
                frontmatter=None,
                body="",
                error=f"unexpected: {exc}",
            )
