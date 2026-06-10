"""Project registry + voice-reference resolver.

The registry is a small JSON file at ``data/projects.json``. Each entry
is a known project with one canonical name, optional aliases (the casual
names the user actually says), an absolute path on disk, and a few
descriptive fields. The registry is the single source of truth for
"where on disk does the user mean when they say 'fix the flask app'".

The resolver maps a free-text user reference to a registered project:

  1. Exact name match (case-insensitive).
  2. Exact alias match.
  3. Substring match on name / alias / description.
  4. Semantic similarity via the existing :class:`HybridEmbedder` (when
     supplied) -- only fires above a configurable threshold so a vague
     reference doesn't quietly route to the wrong project.
  5. None of the above -> ``NOT_FOUND``; caller decides whether to
     prompt the user or auto-create a new project.

Ambiguity (multiple high-confidence matches) is preserved in the
resolution output so the voice layer can ask "did you mean X or Y?".
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import settings
from ultron.errors import FilesystemError
from ultron.resilience import get_error_log
from ultron.utils.logging import get_logger

logger = get_logger("coding.projects")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Project:
    """One entry in the registry."""

    name: str
    path: str  # absolute, stored as str for JSON portability
    aliases: List[str] = field(default_factory=list)
    language: str = ""
    description: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def path_obj(self) -> Path:
        return Path(self.path)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        return cls(
            name=str(d.get("name", "")),
            path=str(d.get("path", "")),
            aliases=list(d.get("aliases") or []),
            language=str(d.get("language", "")),
            description=str(d.get("description", "")),
            created_at=float(d.get("created_at", 0.0) or 0.0),
            last_accessed=float(d.get("last_accessed", 0.0) or 0.0),
            tags=list(d.get("tags") or []),
        )


# ---------------------------------------------------------------------------
# Registry: persistent CRUD
# ---------------------------------------------------------------------------


class ProjectRegistry:
    """File-backed project registry. Writes are atomic (write to .tmp + rename)."""

    def __init__(self, path: Path = settings.CODING_PROJECT_REGISTRY_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._projects: List[Project] = []
        self._load()

    # --- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            self._projects = []
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Project registry unreadable (%s) -- starting empty", e)
            get_error_log().record(
                FilesystemError(
                    f"project registry unreadable: {e}",
                    context={
                        "path": str(self.path),
                        "underlying": type(e).__name__,
                    },
                    recovery="started with empty registry; user must re-register",
                ),
                dependency="filesystem",
                include_traceback=False,
            )
            self._projects = []
            return
        rows = data.get("projects") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            self._projects = []
            return
        self._projects = [Project.from_dict(r) for r in rows if isinstance(r, dict)]

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"projects": [p.to_dict() for p in self._projects]}
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp.replace(self.path)
        except OSError as e:
            get_error_log().record(
                FilesystemError(
                    f"project registry write failed: {e}",
                    context={"path": str(self.path)},
                    recovery="in-memory registry retained; save will retry next time",
                ),
                dependency="filesystem",
                include_traceback=False,
            )
            raise

    # --- CRUD ---------------------------------------------------------------

    def list(self) -> List[Project]:
        with self._lock:
            return [Project.from_dict(p.to_dict()) for p in self._projects]

    def get(self, name: str) -> Optional[Project]:
        target = name.strip().lower()
        with self._lock:
            for p in self._projects:
                if p.name.lower() == target:
                    return Project.from_dict(p.to_dict())
        return None

    def add(self, project: Project) -> Project:
        with self._lock:
            if any(p.name.lower() == project.name.lower() for p in self._projects):
                raise ValueError(f"Project '{project.name}' already exists")
            self._projects.append(project)
            self._save()
        logger.info("Registered project %r at %s", project.name, project.path)
        return project

    def update(self, project: Project) -> Project:
        with self._lock:
            for i, existing in enumerate(self._projects):
                if existing.name.lower() == project.name.lower():
                    self._projects[i] = project
                    self._save()
                    return project
        raise KeyError(f"No such project: {project.name}")

    def remove(self, name: str) -> bool:
        target = name.strip().lower()
        with self._lock:
            for i, p in enumerate(self._projects):
                if p.name.lower() == target:
                    del self._projects[i]
                    self._save()
                    logger.info("Removed project %r", name)
                    return True
        return False

    def touch(self, name: str) -> None:
        with self._lock:
            for p in self._projects:
                if p.name.lower() == name.strip().lower():
                    p.last_accessed = time.time()
                    self._save()
                    return


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class ResolutionKind(str, Enum):
    EXACT = "exact"
    ALIAS = "alias"
    SUBSTRING = "substring"
    SEMANTIC = "semantic"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass
class ProjectResolution:
    kind: ResolutionKind
    project: Optional[Project] = None
    candidates: List[Project] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


class ProjectResolver:
    """Resolve a user reference to a project.

    Args:
        registry: the :class:`ProjectRegistry` to search.
        embedder: optional :class:`HybridEmbedder`. When supplied,
            ambiguous lexical matches escalate to a semantic similarity
            check using the bge-small dense vector. Without one, semantic
            resolution is skipped (cleanly degrades to lexical-only).
        semantic_threshold: minimum cosine similarity for a semantic
            match to win. 0.55 by default -- conservative to avoid
            confidently routing to the wrong project on a vague
            reference.
    """

    def __init__(
        self,
        registry: ProjectRegistry,
        embedder=None,
        semantic_threshold: float = 0.55,
    ) -> None:
        self.registry = registry
        self.embedder = embedder
        self.semantic_threshold = semantic_threshold

    def resolve(self, reference: str) -> ProjectResolution:
        """Map a free-text user reference to a project."""
        if reference is None:
            return ProjectResolution(ResolutionKind.NOT_FOUND, reason="empty reference")
        ref = reference.strip()
        if not ref:
            return ProjectResolution(ResolutionKind.NOT_FOUND, reason="empty reference")

        projects = self.registry.list()
        if not projects:
            return ProjectResolution(
                ResolutionKind.NOT_FOUND,
                reason="registry is empty",
            )

        ref_lower = ref.lower()

        # 1. Exact name.
        for p in projects:
            if p.name.lower() == ref_lower:
                return ProjectResolution(
                    ResolutionKind.EXACT,
                    project=p, confidence=1.0,
                    reason=f"exact name match: {p.name}",
                )

        # 2. Exact alias.
        for p in projects:
            if any(a.lower() == ref_lower for a in p.aliases):
                return ProjectResolution(
                    ResolutionKind.ALIAS,
                    project=p, confidence=0.95,
                    reason=f"alias match: {p.name}",
                )

        # 3. Substring match on name / alias / description.
        substring_hits = [
            p for p in projects
            if (
                ref_lower in p.name.lower()
                or any(ref_lower in a.lower() for a in p.aliases)
                or (p.description and ref_lower in p.description.lower())
                or any(ref_lower in t.lower() for t in p.tags)
            )
        ]
        if len(substring_hits) == 1:
            return ProjectResolution(
                ResolutionKind.SUBSTRING,
                project=substring_hits[0], confidence=0.8,
                reason=f"substring match: {substring_hits[0].name}",
            )
        if len(substring_hits) > 1:
            # Try semantic to pick the best, fall back to ambiguous.
            best = self._best_semantic(ref, substring_hits)
            if best is not None:
                p, score = best
                if score >= self.semantic_threshold:
                    return ProjectResolution(
                        ResolutionKind.SEMANTIC,
                        project=p, confidence=score,
                        reason=f"semantic disambiguation among substring hits: {p.name}",
                        candidates=substring_hits,
                    )
            return ProjectResolution(
                ResolutionKind.AMBIGUOUS,
                candidates=substring_hits,
                reason=f"{len(substring_hits)} substring matches; need disambiguation",
            )

        # 4. Pure semantic match across all projects.
        best = self._best_semantic(ref, projects)
        if best is not None:
            p, score = best
            if score >= self.semantic_threshold:
                return ProjectResolution(
                    ResolutionKind.SEMANTIC,
                    project=p, confidence=score,
                    reason=f"semantic match: {p.name}",
                )

        return ProjectResolution(ResolutionKind.NOT_FOUND, reason="no match")

    # --- internals ----------------------------------------------------------

    def _best_semantic(
        self, reference: str, candidates: List[Project]
    ) -> Optional[tuple[Project, float]]:
        if self.embedder is None or not candidates:
            return None
        try:
            qvec = self.embedder.encode_query_dense(reference)
            corpus_text = [_describe_for_embedding(p) for p in candidates]
            cvecs = self.embedder.encode_dense(corpus_text)
        except Exception as e:
            logger.debug("Semantic resolution failed (%s)", e)
            return None
        if cvecs.shape[0] == 0:
            return None
        # bge-small embeddings are L2-normalized so dot product == cosine.
        scores = cvecs @ qvec
        idx = int(np.argmax(scores))
        return candidates[idx], float(scores[idx])


def _describe_for_embedding(p: Project) -> str:
    """Concatenate the fields a free-text reference might lexically resemble."""
    fields = [p.name]
    if p.aliases:
        fields.append(" ".join(p.aliases))
    if p.description:
        fields.append(p.description)
    if p.language:
        fields.append(p.language)
    if p.tags:
        fields.append(" ".join(p.tags))
    return ". ".join(fields)


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


def slugify_for_path(name: str) -> str:
    """Make a project name safe for use as a directory."""
    keep = []
    for c in name.strip().lower():
        if c.isalnum():
            keep.append(c)
        elif c in (" ", "_", "-"):
            keep.append("_")
    slug = "".join(keep).strip("_")
    return slug or f"project_{uuid.uuid4().hex[:6]}"


def ensure_sandbox_isolation(
    project_dir: Path,
    *,
    sandbox_root: Optional[Path] = None,
    run_fn=None,
) -> bool:
    """Make a sandbox project its own git root so the coding subprocess
    treats IT as the project boundary.

    Production-hardening finding (the phase-11 voice-coding e2e): the
    sandbox lives at ``data/sandbox/<project>`` INSIDE the ultron repo,
    so the spawned coding CLI walked UP from the task cwd, discovered
    the ultron repo root, and loaded the repo's (very large) local
    orientation context into every voice coding task -- a hidden
    multi-thousand-token tax per task, and occasionally an outright
    hijack (the model responded to the orientation file instead of the
    task). A ``.git`` directory in the project makes the project dir
    its own root, stopping the upward walk -- and enables the coding
    CLI's own checkpointing as a bonus.

    Only acts on directories UNDER ``sandbox_root`` (never a user's own
    project folder), is idempotent (an existing ``.git`` short-circuits),
    and is fail-open at every layer (missing git binary / timeout / any
    error -> False, the task proceeds as before). Returns True iff the
    directory ends up git-initialised.
    """
    try:
        root = Path(sandbox_root) if sandbox_root is not None else Path(
            settings.CODING_SANDBOX_PATH
        )
        project_dir = Path(project_dir)
        try:
            project_dir.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            return False  # outside the sandbox -- never touch it
        if (project_dir / ".git").exists():
            return True
        if not project_dir.is_dir():
            return False
        import subprocess as _subprocess

        kwargs: dict = {
            "cwd": str(project_dir),
            "capture_output": True,
            "timeout": 10.0,
        }
        if os.name == "nt":  # pragma: no cover -- platform flag
            kwargs["creationflags"] = _subprocess.CREATE_NO_WINDOW
        runner = run_fn or _subprocess.run
        proc = runner(["git", "init", "-q", "."], **kwargs)
        return getattr(proc, "returncode", 1) == 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("sandbox isolation skipped for %s: %s", project_dir, exc)
        return False


def new_sandbox_project(
    registry: ProjectRegistry,
    *,
    name: str,
    aliases: Optional[List[str]] = None,
    language: str = "",
    description: str = "",
    sandbox_root: Path = settings.CODING_SANDBOX_PATH,
    create_dir: bool = True,
) -> Project:
    """Register a new project and create its sandbox directory.

    The directory is created under ``sandbox_root / slug(name)``. If a
    folder with that slug already exists (e.g. user re-uses a name),
    a uniqueness suffix is appended so we never silently merge two
    projects' files.
    """
    slug = slugify_for_path(name)
    target = Path(sandbox_root) / slug
    suffix = 1
    while target.exists():
        target = Path(sandbox_root) / f"{slug}_{suffix}"
        suffix += 1
    if create_dir:
        target.mkdir(parents=True, exist_ok=False)
        # Stop the coding CLI's upward project-context walk at the
        # project boundary (see ensure_sandbox_isolation). Fail-open.
        ensure_sandbox_isolation(target, sandbox_root=Path(sandbox_root))
    project = Project(
        name=name,
        path=str(target.resolve()),
        aliases=list(aliases or []),
        language=language,
        description=description,
    )
    return registry.add(project)
