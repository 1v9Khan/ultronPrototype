"""Project introspection -- cheap snapshot of on-disk project state.

Builds a structured ``ProjectSnapshot`` from a project directory by
walking the file tree (depth-limited), detecting the dominant
language(s), finding entry-point files, and -- for Python files only --
threading per-file AST metadata in via the existing
:mod:`ultron.coding.ast_metadata`.

This is the **non-LLM** side of project understanding. Cost target:
100-300 ms for a typical sandbox project (a few hundred files), no
external service calls, no model loads. The result is consumed by:

  * :class:`ultron.coding.project_supervisor.ProjectSupervisor` --
    when deciding whether the user's reference matches an existing
    project, the snapshot's entry points + language hint are used as
    additional features alongside the digest cosine match.
  * :class:`ultron.coding.project_digest.DigestRequest` -- when
    generating a project digest, the snapshot's language + entry
    points seed the prompt.
  * Enriched dispatch (Phase E) -- the snapshot's file tree summary
    is passed to Claude in the initial prompt so Claude doesn't
    re-explore the same paths.

The snapshot is cached per-path with a short TTL so repeated calls
during a single user turn don't repeat the walk.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from ultron.coding.ast_metadata import AstMetadata, extract_metadata_from_path

logger = logging.getLogger("ultron.coding.project_introspect")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# File extensions we consider source code for language detection.
# Maps extension -> language name (lowercase). The dominant language
# is the one with the most files.
LANGUAGE_BY_EXT: Mapping[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".clj": "clojure",
    ".ex": "elixir",
    ".exs": "elixir",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".lua": "lua",
    ".r": "r",
    ".m": "matlab",
    ".dart": "dart",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".vue": "vue",
    ".sql": "sql",
    ".sln": "csharp",
    ".cls": "apex",
    ".trigger": "apex",
}

# Project marker files -- their presence signals the project's language
# and conventional entry points. Order matters for "main" detection.
MARKER_FILES: Mapping[str, str] = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "requirements.txt": "python",
    "manage.py": "python",
    "app.py": "python",
    "main.py": "python",
    "__main__.py": "python",
    "package.json": "javascript",
    "tsconfig.json": "typescript",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "Package.swift": "swift",
    "Gemfile": "ruby",
    "composer.json": "php",
    "mix.exs": "elixir",
    "Pipfile": "python",
    "poetry.lock": "python",
    "Makefile": "make",
    "CMakeLists.txt": "cpp",
    "Dockerfile": "docker",
    "sfdx-project.json": "apex",
}

# Candidate entry-point filenames (in priority order). When multiple
# match, all are reported but the first is the "primary" entry point.
ENTRY_POINT_FILENAMES: Sequence[str] = (
    "manage.py",
    "main.py",
    "app.py",
    "__main__.py",
    "server.py",
    "run.py",
    "wsgi.py",
    "asgi.py",
    "index.js",
    "index.ts",
    "main.js",
    "main.ts",
    "server.js",
    "server.ts",
    "main.go",
    "main.rs",
    "Main.java",
    "Program.cs",
    "index.html",
    "index.php",
)

# Directories we always skip when walking. Keeps the snapshot focused
# on user code instead of dependency caches / build artifacts.
SKIP_DIRECTORIES: frozenset = frozenset({
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "htmlcov",
    "coverage",
    ".coverage",
    "site-packages",
    "vendor",
    "bower_components",
    "tmp",
    "temp",
    ".DS_Store",
    ".turbo",
    ".parcel-cache",
})

# Hard caps to keep snapshots fast and bounded. A typical sandbox
# project fits within these comfortably; pathological cases (e.g.
# user dropped node_modules into the project root after a build
# bypassed our skip list) get truncated rather than wedging.
DEFAULT_MAX_DEPTH = 6
DEFAULT_MAX_FILES = 500
DEFAULT_MAX_DIRECTORIES = 200
DEFAULT_AST_FILE_CAP = 30  # Only parse up to N Python files per snapshot.
DEFAULT_CACHE_TTL_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileInfo:
    """Per-file slot in the snapshot's file list."""

    path: Path  # absolute
    relative_path: str  # rel to project root, posix-style for stability
    size_bytes: int
    extension: str  # lowercased, including leading dot
    is_entry_point: bool = False


@dataclass
class ProjectSnapshot:
    """Structured view of a project's on-disk state.

    Attributes:
        project_path: absolute path to the project root.
        project_name: name component of the project root (basename).
        files: list of :class:`FileInfo` entries, depth-first walked,
            capped at :data:`DEFAULT_MAX_FILES`.
        directories: list of relative directory paths (excluding
            those in :data:`SKIP_DIRECTORIES`), capped.
        languages: dominant-first list of detected languages.
        language_counts: per-language file counts (for tie-breaking).
        entry_points: detected entry-point file paths (absolute).
        markers: detected project-marker files (e.g. pyproject.toml).
        ast_metadata: per-file AST snapshot for parsed Python files;
            empty when project has no Python files.
        captured_at: monotonic timestamp the snapshot was built.
        elapsed_ms: time the walk + parse took.
        truncated: True when caps were hit (file/depth/dir).
    """

    project_path: Path
    project_name: str
    files: List[FileInfo] = field(default_factory=list)
    directories: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    language_counts: Dict[str, int] = field(default_factory=dict)
    entry_points: List[Path] = field(default_factory=list)
    markers: List[str] = field(default_factory=list)
    ast_metadata: Dict[str, AstMetadata] = field(default_factory=dict)
    captured_at: float = field(default_factory=time.time)
    elapsed_ms: float = 0.0
    truncated: bool = False

    @property
    def dominant_language(self) -> str:
        """The language with the most files, or '' when none detected."""
        return self.languages[0] if self.languages else ""

    @property
    def file_count(self) -> int:
        return len(self.files)

    def render_tree_summary(self, max_lines: int = 50) -> str:
        """Build a markdown-style file-tree summary suitable for
        embedding in an LLM prompt.

        Lists directories (capped at 20) then files (capped at the
        rest of ``max_lines``). Designed to be cheap to render and
        token-bounded.
        """
        lines: List[str] = []
        lines.append(f"{self.project_name}/  (project root)")
        for d in self.directories[:20]:
            lines.append(f"  {d}/")
        remaining = max(1, max_lines - len(lines))
        for f in self.files[:remaining]:
            marker = "  [entry]" if f.is_entry_point else ""
            lines.append(f"  {f.relative_path}{marker}")
        if len(self.files) > remaining:
            lines.append(f"  ... +{len(self.files) - remaining} more files")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _SnapshotCache:
    """Per-path TTL cache for :class:`ProjectSnapshot` instances."""

    def __init__(self, ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._cache: Dict[str, tuple[ProjectSnapshot, float]] = {}

    def get(self, key: str) -> Optional[ProjectSnapshot]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            snap, when = entry
            if time.monotonic() - when > self.ttl_seconds:
                del self._cache[key]
                return None
            return snap

    def put(self, key: str, snap: ProjectSnapshot) -> None:
        with self._lock:
            self._cache[key] = (snap, time.monotonic())

    def invalidate(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._cache.clear()
                return
            self._cache.pop(key, None)

    def invalidate_for_file(self, file_path: str) -> int:
        """Drop any cached snapshot whose project_path is an ancestor of file_path.

        Args:
            file_path: absolute or relative path to the changed file.

        Returns:
            Number of cache entries dropped. Zero when no cached
            project contains the file (or when the cache is empty).
        """
        if not file_path:
            return 0
        try:
            norm = str(Path(file_path).resolve())
        except OSError:
            # Resolution can fail on Windows when the file is gone
            # (deleted event); fall back to string normalization.
            norm = os.path.normpath(file_path)
        norm_lower = norm.lower() if os.name == "nt" else norm

        with self._lock:
            to_drop: List[str] = []
            for key in self._cache:
                key_cmp = key.lower() if os.name == "nt" else key
                # An exact match means a file rename / mutation to the
                # project root itself; a prefix + separator means the
                # changed file lives inside that project tree.
                if (
                    norm_lower == key_cmp
                    or norm_lower.startswith(key_cmp + os.sep)
                    or norm_lower.startswith(key_cmp + "/")
                ):
                    to_drop.append(key)
            for key in to_drop:
                del self._cache[key]
        return len(to_drop)


_DEFAULT_CACHE = _SnapshotCache()


def invalidate_snapshot_cache(project_path: Optional[Path] = None) -> None:
    """Drop a cached snapshot.

    Without arguments, clears the entire cache (test escape hatch).
    With a project path, drops just that entry. Callers that mutate
    a project on disk (the FILE_CHANGE listener) should invalidate
    after the mutation so the next snapshot reflects truth.
    """
    if project_path is None:
        _DEFAULT_CACHE.invalidate()
        return
    _DEFAULT_CACHE.invalidate(str(project_path.resolve()))


def invalidate_snapshot_cache_for_file(file_path: str) -> int:
    """Drop any cached snapshot containing ``file_path`` as a descendant.

    Convenience wrapper around :meth:`_SnapshotCache.invalidate_for_file`
    used by bus subscribers (which receive a file path rather than a
    project root in :class:`CodingFileChangedEvent`).

    Returns the number of cache entries dropped.
    """
    return _DEFAULT_CACHE.invalidate_for_file(file_path)


# Bus-subscriber wiring -- single-installation guard prevents
# duplicate subscriptions when the orchestrator restarts or tests
# run install repeatedly. ``reset_bus_invalidator_for_testing``
# wipes the guard so tests after ``reset_bus_for_testing()`` can
# re-install cleanly.
_BUS_UNSUBSCRIBE: Optional[Callable[[], None]] = None


def install_bus_invalidator() -> Callable[[], None]:
    """Subscribe to :class:`CodingFileChangedEvent` for cache invalidation.

    The first call subscribes to the typed-event bus and stashes the
    unsubscribe callable. Subsequent calls return that callable
    without re-subscribing, so calling this from multiple init paths
    (orchestrator construction, voice controller setup, test fixtures)
    is safe.

    Returns:
        The unsubscribe callable. Calling it (or
        :func:`reset_bus_invalidator_for_testing`) clears the
        registration so a subsequent ``install_bus_invalidator()``
        subscribes fresh.

    Fail-open: when ``ultron.bus`` is unavailable for any reason
    (import error, mock environment), returns a no-op unsubscribe
    and logs a debug-level note. The cache still works manually via
    :func:`invalidate_snapshot_cache_for_file`.
    """
    global _BUS_UNSUBSCRIBE
    if _BUS_UNSUBSCRIBE is not None:
        return _BUS_UNSUBSCRIBE

    try:
        from ultron.bus import CodingFileChangedEvent, subscribe
    except Exception as e:                                         # noqa: BLE001
        logger.debug(
            "project_introspect: bus invalidator skipped (%s)", e,
        )
        _BUS_UNSUBSCRIBE = lambda: None  # noqa: E731
        return _BUS_UNSUBSCRIBE

    def _on_file_changed(payload) -> None:                          # noqa: ANN001
        try:
            file_path = payload.properties.get("file_path", "")
            if file_path:
                dropped = invalidate_snapshot_cache_for_file(file_path)
                if dropped:
                    logger.debug(
                        "project_introspect: invalidated %d snapshot(s) "
                        "after file_changed %s",
                        dropped, file_path,
                    )
        except Exception as e:                                     # noqa: BLE001
            # Subscriber must never raise -- the bus already swallows
            # but we add a layer because cache invalidation is
            # non-essential.
            logger.debug(
                "project_introspect: invalidator failed: %s", e,
            )

    _BUS_UNSUBSCRIBE = subscribe(CodingFileChangedEvent, _on_file_changed)
    return _BUS_UNSUBSCRIBE


def reset_bus_invalidator_for_testing() -> None:
    """Clear the install guard so a fresh ``install_bus_invalidator()``
    subscribes against the current (possibly reset) bus singleton.

    Test-only escape hatch. The previous subscription is left dangling
    on the old bus instance (if any); the test fixture is responsible
    for resetting the bus too.
    """
    global _BUS_UNSUBSCRIBE
    if _BUS_UNSUBSCRIBE is not None:
        try:
            _BUS_UNSUBSCRIBE()
        except Exception:                                          # noqa: BLE001
            pass
        _BUS_UNSUBSCRIBE = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def snapshot(
    project_path: Path,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_files: int = DEFAULT_MAX_FILES,
    max_directories: int = DEFAULT_MAX_DIRECTORIES,
    ast_file_cap: int = DEFAULT_AST_FILE_CAP,
    use_cache: bool = True,
) -> ProjectSnapshot:
    """Build a :class:`ProjectSnapshot` for ``project_path``.

    Args:
        project_path: project root to walk. Must exist; if it doesn't,
            an empty snapshot is returned with ``project_name`` set
            from the basename so callers can still report sensibly.
        max_depth: maximum subdirectory depth to walk.
        max_files: hard cap on the file count -- protects against
            pathological projects (e.g. node_modules slipped in).
        max_directories: hard cap on directory entries.
        ast_file_cap: max Python files to parse for AST metadata.
            Parsing is per-file ~5-50 ms; capped to keep snapshot
            cost bounded.
        use_cache: when True, consults / populates the process-wide
            TTL cache. Pass False from callers that need a fresh
            snapshot regardless (e.g. directly after a FILE_CHANGE).

    Returns:
        A :class:`ProjectSnapshot`. Always non-None; on errors during
        the walk, returns whatever could be collected (fail-open).
    """
    t0 = time.monotonic()
    resolved = project_path.resolve()
    cache_key = str(resolved)

    if use_cache:
        cached = _DEFAULT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    snap = ProjectSnapshot(
        project_path=resolved,
        project_name=resolved.name,
    )

    if not resolved.exists() or not resolved.is_dir():
        snap.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return snap

    _walk_project(
        snap,
        max_depth=max_depth,
        max_files=max_files,
        max_directories=max_directories,
    )
    _detect_languages(snap)
    _detect_entry_points(snap)
    _parse_ast_for_python_files(snap, cap=ast_file_cap)

    snap.elapsed_ms = (time.monotonic() - t0) * 1000.0

    if use_cache:
        _DEFAULT_CACHE.put(cache_key, snap)
    return snap


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------


def _walk_project(
    snap: ProjectSnapshot,
    *,
    max_depth: int,
    max_files: int,
    max_directories: int,
) -> None:
    """Depth-first walk filling ``snap.files`` and ``snap.directories``."""
    root = snap.project_path
    root_str = str(root)

    def _depth(path_str: str) -> int:
        # +1 for the root delimiter; depth at root = 0.
        rel = path_str[len(root_str):].strip(os.sep)
        return rel.count(os.sep) + (1 if rel else 0)

    truncated_flag = False
    files_collected = 0
    dirs_collected = 0

    for dirpath, dirnames, filenames in os.walk(root_str):
        depth = _depth(dirpath)

        # Prune skipped + over-depth directories in place (os.walk
        # honors mutations to dirnames).
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRECTORIES
            and not d.startswith(".")
            or d in {".github", ".claude"}  # specific carve-outs
        ]
        if depth >= max_depth:
            dirnames[:] = []
            truncated_flag = True

        if depth > 0:
            rel_dir = os.path.relpath(dirpath, root_str).replace(os.sep, "/")
            if dirs_collected < max_directories:
                snap.directories.append(rel_dir)
                dirs_collected += 1
            else:
                truncated_flag = True

        for fname in filenames:
            if files_collected >= max_files:
                truncated_flag = True
                break
            full = Path(dirpath) / fname
            try:
                size = full.stat().st_size
            except OSError:
                size = 0
            rel = os.path.relpath(str(full), root_str).replace(os.sep, "/")
            ext = full.suffix.lower()
            snap.files.append(FileInfo(
                path=full,
                relative_path=rel,
                size_bytes=size,
                extension=ext,
            ))
            files_collected += 1

            # Track project markers when encountered.
            if fname in MARKER_FILES and fname not in snap.markers:
                snap.markers.append(fname)

    snap.truncated = truncated_flag


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _detect_languages(snap: ProjectSnapshot) -> None:
    """Compute dominant + secondary languages from file extensions + markers."""
    counts: Dict[str, int] = {}
    for f in snap.files:
        lang = LANGUAGE_BY_EXT.get(f.extension)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1

    # Markers are a strong signal -- a project with pyproject.toml is
    # Python even if some auxiliary HTML/CSS counts overwhelm it.
    for marker in snap.markers:
        marker_lang = MARKER_FILES.get(marker)
        if marker_lang:
            counts[marker_lang] = counts.get(marker_lang, 0) + 10

    snap.language_counts = counts
    if not counts:
        snap.languages = []
        return
    # Dominant-first by count, then alphabetical to keep deterministic.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    snap.languages = [lang for lang, _ in ranked]


# ---------------------------------------------------------------------------
# Entry-point detection
# ---------------------------------------------------------------------------


def _detect_entry_points(snap: ProjectSnapshot) -> None:
    """Pick entry-point files from the walked file list.

    Prefers files at the project root over deeper ones. Within a
    depth tier, follows the order in :data:`ENTRY_POINT_FILENAMES`.
    Marks selected entries on the corresponding :class:`FileInfo`.
    """
    root_str = str(snap.project_path)
    root_files: List[Path] = []
    deeper_files: List[Path] = []

    target_set = {name.lower() for name in ENTRY_POINT_FILENAMES}
    for f in snap.files:
        base = f.path.name.lower()
        if base not in target_set:
            continue
        parent = str(f.path.parent)
        if parent == root_str:
            root_files.append(f.path)
        else:
            deeper_files.append(f.path)

    # Apply priority order from ENTRY_POINT_FILENAMES.
    name_priority = {n.lower(): i for i, n in enumerate(ENTRY_POINT_FILENAMES)}
    root_files.sort(key=lambda p: name_priority.get(p.name.lower(), 999))
    deeper_files.sort(key=lambda p: name_priority.get(p.name.lower(), 999))

    snap.entry_points = root_files + deeper_files[:5]  # Limit deeper picks.

    # Backfill is_entry_point flags onto the corresponding FileInfo entries.
    if not snap.entry_points:
        return
    entry_set = {str(p): True for p in snap.entry_points}
    new_files: List[FileInfo] = []
    for f in snap.files:
        if str(f.path) in entry_set and not f.is_entry_point:
            new_files.append(FileInfo(
                path=f.path,
                relative_path=f.relative_path,
                size_bytes=f.size_bytes,
                extension=f.extension,
                is_entry_point=True,
            ))
        else:
            new_files.append(f)
    snap.files = new_files


# ---------------------------------------------------------------------------
# AST parsing (Python only)
# ---------------------------------------------------------------------------


def _parse_ast_for_python_files(
    snap: ProjectSnapshot, *, cap: int,
) -> None:
    """Run :func:`extract_metadata_from_path` on up to ``cap`` Python files.

    Prioritizes entry points first (they're the most useful to
    summarize), then files near the project root, then alphabetic by
    relative path. The result is keyed by ``relative_path`` for
    portability across environments.
    """
    if cap <= 0:
        return

    py_files = [f for f in snap.files if f.extension in {".py", ".pyi"}]
    if not py_files:
        return

    # Stable priority order.
    def _priority(fi: FileInfo) -> tuple[int, int, str]:
        depth = fi.relative_path.count("/")
        entry_bias = 0 if fi.is_entry_point else 1
        return (entry_bias, depth, fi.relative_path)

    py_files.sort(key=_priority)

    parsed: Dict[str, AstMetadata] = {}
    for fi in py_files[:cap]:
        try:
            md = extract_metadata_from_path(fi.path)
            parsed[fi.relative_path] = md
        except Exception:                                           # noqa: BLE001
            # extract_metadata_from_path already fails-soft; defense
            # in depth against future changes.
            continue
    snap.ast_metadata = parsed


__all__ = [
    "DEFAULT_AST_FILE_CAP",
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_DIRECTORIES",
    "DEFAULT_MAX_FILES",
    "ENTRY_POINT_FILENAMES",
    "FileInfo",
    "LANGUAGE_BY_EXT",
    "MARKER_FILES",
    "ProjectSnapshot",
    "SKIP_DIRECTORIES",
    "install_bus_invalidator",
    "invalidate_snapshot_cache",
    "invalidate_snapshot_cache_for_file",
    "reset_bus_invalidator_for_testing",
    "snapshot",
]
