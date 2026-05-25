"""`.ultronignore` policy: workspace + project + global path-block file.

Adapted from cline's ``ClineIgnoreController`` pattern (Apache 2.0;
see ``THIRD_PARTY_NOTICES.md``). The on-disk syntax mirrors
``.gitignore`` (via the ``pathspec`` library) with two extensions:

* ``!include path/to/other-file`` — concatenate another ignore file's
  contents into the current one (so a monorepo can share a base list).
* ``validate_command(cmd)`` — parse a shell command via ``shlex`` and
  return the first ignored path argument, if any. Covers the cline
  POSIX list (``cat``, ``head``, ``tail``, ``less``, ``more``, ``grep``,
  ``awk``, ``sed``) PLUS the PowerShell equivalents (``gc``, ``type``,
  ``Get-Content``, ``Select-String``, ``sls``).

The controller stacks three layers from lowest-precedence to highest:
global at ``~/.ultron/.ultronignore``, project at
``<project_root>/.ultron/.ultronignore``, and an optional workspace
override at ``<project_root>/.ultronignore``. Each layer can include
others via ``!include``; cycle detection guards against infinite
loops. Compiled :class:`pathspec.PathSpec` objects are cached and
re-evaluated on file mtime change (the orchestrator subscribes a
file-watcher when one is available; offline / first-call evaluation
re-reads on every check).
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

LOGGER = logging.getLogger(__name__)

#: Default ignore filename searched at the workspace / project root.
DEFAULT_IGNORE_FILENAME: str = ".ultronignore"

#: Default directory under the project root where a versioned
#: project-level ignore lives (``<project>/.ultron/.ultronignore``).
DEFAULT_PROJECT_DIR: str = ".ultron"

#: Default location for the global ignore (per-user, untracked).
DEFAULT_GLOBAL_IGNORE: Path = Path.home() / ".ultron" / DEFAULT_IGNORE_FILENAME

#: Glyph displayed in file listings for ignored entries (matches
#: response_format.LOCK_TEXT_SYMBOL).
LOCK_GLYPH: str = "\U0001F512"

#: Prefix introducing the ``!include`` directive.
INCLUDE_DIRECTIVE: str = "!include"

#: File-reading shell commands whose arguments should be validated.
#: Includes POSIX + Windows-PowerShell variants.
COMMANDS_THAT_READ_FILES: frozenset[str] = frozenset({
    # POSIX
    "cat", "head", "tail", "less", "more",
    "grep", "egrep", "fgrep",
    "awk", "sed",
    "od", "xxd",
    # Windows PowerShell aliases / cmdlets
    "gc", "type", "get-content",
    "select-string", "sls",
    "format-hex",
})

#: Hard cap on recursion depth for nested ``!include`` directives.
MAX_INCLUDE_DEPTH: int = 8


def _load_pathspec():
    """Import ``pathspec`` lazily; helpful warning when absent."""
    try:
        import pathspec  # type: ignore
        return pathspec
    except ImportError as exc:  # pragma: no cover - import-time only
        raise RuntimeError(
            "pathspec is required for .ultronignore evaluation; "
            "ensure the venv install completed successfully."
        ) from exc


@dataclass(frozen=True)
class IgnoreVerdict:
    """Outcome of a single :meth:`IgnoreController.check_path` call.

    Attributes:
        ignored: True when the path is denied by policy.
        matched_layer: the layer (``global`` / ``project`` / ``workspace``)
            whose pattern fired. ``""`` when ``ignored`` is False.
        matched_pattern: the literal pattern that matched (best-effort
            recovery from the pathspec library; an empty string when
            the library does not expose it).
    """

    ignored: bool
    matched_layer: str = ""
    matched_pattern: str = ""


@dataclass(frozen=True)
class CommandValidation:
    """Outcome of a single :meth:`IgnoreController.validate_command` call.

    Attributes:
        denied_path: the first argument the policy denied, or ``None``
            when the command is allowed.
        program: the resolved command name (after stripping prefixes).
        reason: short explanation suitable for an audit log.
    """

    denied_path: Optional[str] = None
    program: str = ""
    reason: str = ""


@dataclass
class _Layer:
    """Internal record for one compiled ignore layer."""

    name: str
    source_path: Path
    mtime_ns: int
    patterns: tuple[str, ...]
    spec: object  # PathSpec
    includes: tuple[Path, ...] = field(default_factory=tuple)


class IgnoreController:
    """Three-layer ``.ultronignore`` evaluator with command-arg validation.

    Args:
        workspace_root: project / workspace root directory. ``None`` means
            "no workspace layer" — only the global layer is consulted.
        global_path: location of the per-user global ignore. Defaults to
            ``~/.ultron/.ultronignore``.
        project_path: optional project-level path override (defaults to
            ``<workspace_root>/.ultron/.ultronignore``).
        workspace_path: optional workspace-level path override (defaults to
            ``<workspace_root>/.ultronignore``).
        commands_that_read_files: optional override of the file-reading
            command list (extending the defaults).

    Notes:
        The controller compiles each layer on first use and re-compiles
        when the file's mtime changes. ``check_path`` is thread-safe.
    """

    def __init__(
        self,
        workspace_root: Optional[Path | str] = None,
        *,
        global_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
        commands_that_read_files: Optional[Iterable[str]] = None,
    ) -> None:
        self._workspace_root = Path(workspace_root).resolve() if workspace_root else None
        self._global_path = (global_path or DEFAULT_GLOBAL_IGNORE).resolve()
        if self._workspace_root is not None:
            self._project_path = (
                project_path
                if project_path is not None
                else (self._workspace_root / DEFAULT_PROJECT_DIR / DEFAULT_IGNORE_FILENAME)
            ).resolve()
            self._workspace_path = (
                workspace_path
                if workspace_path is not None
                else (self._workspace_root / DEFAULT_IGNORE_FILENAME)
            ).resolve()
        else:
            self._project_path = (
                Path(project_path).resolve() if project_path is not None else None
            )
            self._workspace_path = (
                Path(workspace_path).resolve() if workspace_path is not None else None
            )
        self._command_set = frozenset(
            (c.lower() for c in commands_that_read_files)
            if commands_that_read_files
            else COMMANDS_THAT_READ_FILES
        )
        self._lock = threading.RLock()
        self._layers: dict[str, Optional[_Layer]] = {
            "global": None,
            "project": None,
            "workspace": None,
        }

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def check_path(self, path: str | os.PathLike[str]) -> IgnoreVerdict:
        """Return the policy verdict for ``path``.

        Args:
            path: path to evaluate. May be absolute or relative; if
                relative and a workspace is configured, the workspace
                root is used as the anchor for pattern matching.

        Returns:
            :class:`IgnoreVerdict` describing the outcome.
        """
        candidate = self._normalise_candidate(path)
        with self._lock:
            for layer_name in ("workspace", "project", "global"):
                layer = self._ensure_layer(layer_name)
                if layer is None:
                    continue
                if self._spec_matches(layer.spec, candidate):
                    return IgnoreVerdict(
                        ignored=True,
                        matched_layer=layer_name,
                        matched_pattern=self._first_matching_pattern(
                            layer, candidate,
                        ),
                    )
        return IgnoreVerdict(ignored=False)

    def is_path_allowed(self, path: str | os.PathLike[str]) -> bool:
        """Convenience wrapper returning ``True`` when the path is allowed."""
        return not self.check_path(path).ignored

    def filter_paths(
        self, paths: Iterable[str | os.PathLike[str]],
    ) -> list[str]:
        """Return only the allowed paths from ``paths`` (preserving order)."""
        out: list[str] = []
        for path in paths:
            verdict = self.check_path(path)
            if not verdict.ignored:
                out.append(str(path))
        return out

    def validate_command(self, command: str) -> CommandValidation:
        """Validate a shell command's path arguments against the ignore rules.

        Args:
            command: full command string to tokenise and check.

        Returns:
            :class:`CommandValidation` describing the outcome. When the
            command is allowed, ``denied_path`` is ``None``; otherwise
            ``denied_path`` is the first path argument the policy denied.

        Notes:
            Empty / whitespace-only commands return an allowed result.
            Commands whose program name is not in
            :data:`COMMANDS_THAT_READ_FILES` are passed through without
            inspection — the safety validator handles other denials.
        """
        text = (command or "").strip()
        if not text:
            return CommandValidation()
        try:
            tokens = shlex.split(text, posix=(sys.platform != "win32"))
        except ValueError:
            # Malformed command — defer to the safety validator.
            return CommandValidation()
        if not tokens:
            return CommandValidation()
        program = self._resolve_program_token(tokens[0])
        if program not in self._command_set:
            return CommandValidation(program=program)
        for token in tokens[1:]:
            # Skip flags / options.
            if token.startswith("-") or token.startswith("/"):
                continue
            # Skip operator-like tokens.
            if token in {">", ">>", "|", "&&", "||", ";"}:
                continue
            verdict = self.check_path(token)
            if verdict.ignored:
                return CommandValidation(
                    denied_path=token,
                    program=program,
                    reason=(
                        f"path '{token}' blocked by {verdict.matched_layer} "
                        f"layer pattern '{verdict.matched_pattern}'"
                    ),
                )
        return CommandValidation(program=program)

    def invalidate(self) -> None:
        """Drop every cached layer (forces re-read on next check)."""
        with self._lock:
            for key in self._layers:
                self._layers[key] = None

    def configured_files(self) -> dict[str, Optional[Path]]:
        """Return the resolved per-layer file paths (read-only snapshot)."""
        return {
            "global": self._global_path,
            "project": self._project_path,
            "workspace": self._workspace_path,
        }

    def known_commands(self) -> frozenset[str]:
        """Return the lower-cased set of file-reading command names."""
        return self._command_set

    # ------------------------------------------------------------------
    # Layer compilation
    # ------------------------------------------------------------------

    def _ensure_layer(self, layer_name: str) -> Optional[_Layer]:
        """Return a fresh, mtime-validated layer (or ``None`` when absent)."""
        path = self._path_for_layer(layer_name)
        if path is None or not path.is_file():
            self._layers[layer_name] = None
            return None
        existing = self._layers.get(layer_name)
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            self._layers[layer_name] = None
            return None
        if existing is not None and existing.mtime_ns == mtime:
            return existing
        try:
            patterns, includes = self._read_with_includes(
                path,
                visited=set(),
                depth=0,
            )
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "failed to compile ignore layer '%s' at %s; falling back to empty.",
                layer_name,
                path,
                exc_info=True,
            )
            self._layers[layer_name] = None
            return None
        pathspec = _load_pathspec()
        try:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except Exception:  # noqa: BLE001
            self._layers[layer_name] = None
            return None
        layer = _Layer(
            name=layer_name,
            source_path=path,
            mtime_ns=mtime,
            patterns=tuple(patterns),
            spec=spec,
            includes=tuple(includes),
        )
        self._layers[layer_name] = layer
        return layer

    def _path_for_layer(self, layer_name: str) -> Optional[Path]:
        if layer_name == "global":
            return self._global_path
        if layer_name == "project":
            return self._project_path
        if layer_name == "workspace":
            return self._workspace_path
        return None

    def _read_with_includes(
        self,
        path: Path,
        *,
        visited: set[Path],
        depth: int,
    ) -> tuple[list[str], list[Path]]:
        """Recursively expand ``!include`` directives into the pattern list."""
        if depth >= MAX_INCLUDE_DEPTH:
            LOGGER.warning(
                "ignore include depth exceeded at %s; truncating.", path,
            )
            return [], []
        resolved = path.resolve()
        if resolved in visited:
            return [], []
        visited.add(resolved)
        try:
            raw = resolved.read_text(encoding="utf-8")
        except OSError:
            return [], []
        patterns: list[str] = []
        includes: list[Path] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                patterns.append(line)
                continue
            if stripped.startswith(INCLUDE_DIRECTIVE):
                target = stripped[len(INCLUDE_DIRECTIVE):].strip()
                if not target:
                    continue
                target_path = (resolved.parent / target).resolve()
                includes.append(target_path)
                sub_patterns, sub_includes = self._read_with_includes(
                    target_path, visited=visited, depth=depth + 1,
                )
                patterns.extend(sub_patterns)
                includes.extend(sub_includes)
                continue
            patterns.append(line)
        return patterns, includes

    # ------------------------------------------------------------------
    # Match helpers
    # ------------------------------------------------------------------

    def _spec_matches(self, spec: object, candidate: str) -> bool:
        try:
            return bool(spec.match_file(candidate))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return False

    def _first_matching_pattern(self, layer: _Layer, candidate: str) -> str:
        """Best-effort recovery of the first pattern that matched ``candidate``."""
        pathspec = _load_pathspec()
        for pattern_text in layer.patterns:
            stripped = pattern_text.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(INCLUDE_DIRECTIVE):
                continue
            try:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern_text])
                if spec.match_file(candidate):
                    return stripped
            except Exception:  # noqa: BLE001
                continue
        return ""

    def _normalise_candidate(self, path: str | os.PathLike[str]) -> str:
        """Render ``path`` as a workspace-relative forward-slash string."""
        candidate = Path(path)
        if self._workspace_root is not None:
            try:
                rel = candidate.resolve().relative_to(self._workspace_root)
                return rel.as_posix()
            except Exception:  # noqa: BLE001
                pass
        # Fall back to absolute path with forward slashes.
        try:
            return Path(path).resolve(strict=False).as_posix()
        except Exception:  # noqa: BLE001
            return str(path).replace("\\", "/")

    @staticmethod
    def _resolve_program_token(token: str) -> str:
        """Lower-case the program token after stripping path / .exe wrappers."""
        text = token.strip().strip('"').strip("'")
        if not text:
            return ""
        leaf = os.path.basename(text)
        if leaf.lower().endswith(".exe"):
            leaf = leaf[: -len(".exe")]
        return leaf.lower()


# ---------------------------------------------------------------------------
# Convenience module-level singleton (per-workspace)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, IgnoreController] = {}
_REGISTRY_LOCK = threading.RLock()


def get_ignore_controller(
    workspace_root: Optional[Path | str] = None,
    **kwargs: object,
) -> IgnoreController:
    """Return (and lazily construct) the controller for ``workspace_root``.

    Args:
        workspace_root: anchor directory (None reuses the global-only
            controller).
        **kwargs: forwarded to :class:`IgnoreController` on first
            construction; ignored thereafter.
    """
    key = str(Path(workspace_root).resolve()) if workspace_root else "__global__"
    with _REGISTRY_LOCK:
        controller = _REGISTRY.get(key)
        if controller is None:
            controller = IgnoreController(workspace_root, **kwargs)
            _REGISTRY[key] = controller
        return controller


def reset_ignore_controller_registry() -> None:
    """Drop every cached controller (test-only)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


__all__ = [
    "COMMANDS_THAT_READ_FILES",
    "CommandValidation",
    "DEFAULT_GLOBAL_IGNORE",
    "DEFAULT_IGNORE_FILENAME",
    "DEFAULT_PROJECT_DIR",
    "INCLUDE_DIRECTIVE",
    "IgnoreController",
    "IgnoreVerdict",
    "LOCK_GLYPH",
    "MAX_INCLUDE_DEPTH",
    "get_ignore_controller",
    "reset_ignore_controller_registry",
]
