"""Project index -- Qdrant-backed digest store + semantic search.

Companion to :mod:`ultron.coding.project_digest`. Whenever a project
digest is generated, :class:`ProjectIndex` upserts it into a dedicated
Qdrant collection (separate from conversational memory so RAG over
chat history doesn't surface project source).

Public surface:

  * :class:`ProjectIndexEntry` -- one row in the index (project_id,
    digest text + sections, embedding vector handled by Qdrant).
  * :class:`ProjectMatch` -- search result with cosine score.
  * :class:`ProjectIndex` -- upsert / search / get / list / delete.

Threading model:
  * Construction creates the collection and warms a recent cache.
  * Read methods (search, get, list) acquire the lock briefly.
  * Writes are synchronous (no background queue -- digests fire
    rarely enough that the overhead is fine).
  * When constructed with a BORROWED client (the production wiring:
    the orchestrator passes ConversationMemory's embedded client,
    because local-mode Qdrant allows one open client per path),
    upserts (bridge COMPLETE-listener thread) and searches (dispatch
    path) share that client with ConversationMemory's writer thread
    and the web cache -- the same already-proven concurrent pattern
    WebResultsCache uses on the disjoint ``web_results`` collection.

Fail-open:
  * Qdrant unavailable / connection error -> all methods return
    safe defaults (empty list, None). The orchestrator's supervisor
    handler checks for None and falls back to lexical
    :class:`ultron.coding.projects.ProjectResolver`.
  * Embedder failure during upsert -> WARN logged, entry NOT
    written; caller can retry.

Bus integration:
  * Every successful upsert publishes
    :data:`ultron.bus.events.ProjectIndexedEvent`. Subscribers can
    use this for liveness tracking or to invalidate adjacent
    caches.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ultron.bus import ProjectIndexedEvent, publish as bus_publish
from ultron.coding.project_digest import ProjectDigest

logger = logging.getLogger("ultron.coding.project_index")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProjectIndexEntry:
    """One indexed project digest.

    Stored both in Qdrant and (recent N) in an in-memory cache.

    Attributes:
        project_id: stable unique id; derived from absolute project
            path on first upsert. Stays constant across digest
            updates for the same project.
        project_name: canonical name from ProjectRegistry.
        project_path: absolute path string (for portability across
            re-mounts; Qdrant payloads JSON-encoded).
        digest_markdown: full digest body.
        digest_sections: parsed-out section map for fast section
            access without re-parsing.
        digest_text_summary: short summary text used for search /
            embedding (first ~500 chars of Goal + Critical Context).
        language: dominant language detected by introspect.
        entry_points: list of relative paths to entry-point files.
        tags: user-or-system tags (e.g. "active", "abandoned",
            "personal").
        last_modified_unix: wall-clock seconds.
        created_at_unix: wall-clock seconds.
        last_session_id: last Claude session id that touched this
            project (for resume).
    """

    project_id: str
    project_name: str
    project_path: str
    digest_markdown: str
    digest_sections: Dict[str, str] = field(default_factory=dict)
    digest_text_summary: str = ""
    language: str = ""
    entry_points: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    last_modified_unix: float = field(default_factory=time.time)
    created_at_unix: float = field(default_factory=time.time)
    last_session_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        """Serialize to Qdrant point payload."""
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_path": self.project_path,
            "digest_markdown": self.digest_markdown,
            "digest_sections": self.digest_sections,
            "digest_text_summary": self.digest_text_summary,
            "language": self.language,
            "entry_points": self.entry_points,
            "tags": self.tags,
            "last_modified_unix": self.last_modified_unix,
            "created_at_unix": self.created_at_unix,
            "last_session_id": self.last_session_id or "",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProjectIndexEntry":
        return cls(
            project_id=str(payload.get("project_id", "")),
            project_name=str(payload.get("project_name", "")),
            project_path=str(payload.get("project_path", "")),
            digest_markdown=str(payload.get("digest_markdown", "")),
            digest_sections=dict(payload.get("digest_sections") or {}),
            digest_text_summary=str(payload.get("digest_text_summary", "")),
            language=str(payload.get("language", "")),
            entry_points=list(payload.get("entry_points") or []),
            tags=list(payload.get("tags") or []),
            last_modified_unix=float(payload.get("last_modified_unix", 0.0) or 0.0),
            created_at_unix=float(payload.get("created_at_unix", 0.0) or 0.0),
            last_session_id=(
                str(payload.get("last_session_id") or "") or None
            ),
        )


@dataclass
class ProjectMatch:
    """A search result.

    Attributes:
        entry: the matched :class:`ProjectIndexEntry`.
        score: cosine similarity (0..1; 1 = identical).
        reason: short human-readable note (e.g. "semantic match"
            or "exact name match").
    """

    entry: ProjectIndexEntry
    score: float
    reason: str = ""


# ---------------------------------------------------------------------------
# ProjectIndex
# ---------------------------------------------------------------------------


class ProjectIndex:
    """Qdrant-backed project digest store + semantic search.

    Constructed once at orchestrator startup. Reuses the same Qdrant
    embedded path as ConversationMemory but a separate collection
    (configured via ``qdrant.collections.projects``).
    """

    def __init__(
        self,
        embedder,
        qdrant_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        recent_cache_size: int = 50,
        client: Optional[Any] = None,
    ) -> None:
        """Construct + open the underlying Qdrant collection.

        Args:
            embedder: a :class:`ultron.memory.embedder.HybridEmbedder`.
                Required for both upsert + search. Same embedder used
                by ConversationMemory -- vectors share the bge-small
                space so cross-pollination is well-defined (though we
                don't actually cross-search, separation of collections
                makes that boundary explicit).
            qdrant_path: directory for the embedded Qdrant store. None
                pulls from config (same path as conversation memory).
                Ignored for client construction when ``client`` is
                passed (the path still feeds logging).
            collection_name: override the default collection name.
                None pulls from config (``qdrant.collections.projects``).
            recent_cache_size: max entries kept in the in-process
                cache. List/get hits the cache first.
            client: an already-open Qdrant client to BORROW instead of
                opening a new one. Local-mode Qdrant allows ONE open
                client per path -- a second ``QdrantClient(path=...)``
                against ``data/qdrant`` raises "already accessed by
                another instance" (the live bug that forced the
                supervisor to registry-only on every boot). The
                orchestrator passes ConversationMemory's client here,
                mirroring the WebResultsCache borrow pattern. The
                OWNER controls the close lifecycle; :meth:`close` is a
                no-op on borrowed clients.
        """
        from ultron.config import get_config, resolve_path

        cfg = get_config()

        if embedder is None:
            raise ValueError(
                "ProjectIndex needs a HybridEmbedder. Pass embedder=HybridEmbedder()."
            )

        self._embedder = embedder
        self._lock = threading.RLock()
        self._recent: List[ProjectIndexEntry] = []
        self._recent_cache_size = recent_cache_size

        self.path = (
            Path(qdrant_path)
            if qdrant_path is not None
            else resolve_path(cfg.qdrant.data_dir)
        )
        self.path.mkdir(parents=True, exist_ok=True)

        self.collection = (
            collection_name
            if collection_name is not None
            else cfg.qdrant.collections.projects
        )

        self._owns_client = client is None
        if client is not None:
            # Borrowed client (see the ``client`` docstring above).
            self._client = client
        else:
            # Lazy import so a missing qdrant-client doesn't crash on
            # import.
            from qdrant_client import QdrantClient
            self._client = QdrantClient(path=str(self.path))
        self._ensure_collection()
        self._warm_recent_cache()

        logger.info(
            "ProjectIndex ready: collection=%s path=%s recent_cached=%d",
            self.collection, self.path, len(self._recent),
        )

    # --- collection bootstrap -----------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the project collection on first run; no-op afterward."""
        from qdrant_client.models import Distance, VectorParams

        names = {c.name for c in self._client.get_collections().collections}
        if self.collection in names:
            return

        # Dense-only for now (BM25 sparse not needed for project search;
        # the digest text is short + structured + we want semantic
        # similarity to drive resolution).
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": VectorParams(
                size=self._embedder.dim,
                distance=Distance.COSINE,
            )},
        )
        logger.info("Created Qdrant collection %s", self.collection)

    def _warm_recent_cache(self) -> None:
        """Pull the most-recent entries into the in-process cache."""
        try:
            # scroll all + sort by last_modified_unix desc, take recent_cache_size
            points, _ = self._client.scroll(
                collection_name=self.collection,
                limit=max(self._recent_cache_size, 100),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "ProjectIndex: scroll failed during warmup (%s); "
                "cache starts empty.", e,
            )
            return

        entries = [
            ProjectIndexEntry.from_payload(p.payload or {})
            for p in points
        ]
        entries.sort(key=lambda e: e.last_modified_unix, reverse=True)
        self._recent = entries[:self._recent_cache_size]

    # --- public API ---------------------------------------------------------

    def upsert(
        self,
        digest: ProjectDigest,
        *,
        project_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: str = "",
        entry_points: Optional[List[Path]] = None,
        last_session_id: Optional[str] = None,
    ) -> Optional[ProjectIndexEntry]:
        """Upsert a project digest into the index.

        Args:
            digest: the :class:`ProjectDigest` from
                :func:`ultron.coding.project_digest.generate_digest`.
            project_id: stable id. When omitted, derived from
                ``digest.project_path`` so the same project always
                lands on the same row (re-runs overwrite).
            tags: optional list of tags to attach (e.g. ["active"]).
            language: dominant language string for filtering.
            entry_points: list of entry-point file paths (absolute or
                relative -- stored as strings).
            last_session_id: Claude session id from the most recent
                Claude run that produced this digest. Used for resume.

        Returns:
            The persisted :class:`ProjectIndexEntry`, or ``None`` when
            upsert failed (logged at WARNING). The bus event is only
            published on success.
        """
        if not digest or not digest.markdown:
            logger.debug("upsert: empty digest, skipping.")
            return None

        pid = project_id or _derive_project_id(digest.project_path)
        summary = _build_digest_summary_for_search(digest.sections)

        entry = ProjectIndexEntry(
            project_id=pid,
            project_name=digest.project_name,
            project_path=str(digest.project_path),
            digest_markdown=digest.markdown,
            digest_sections=dict(digest.sections),
            digest_text_summary=summary,
            language=language,
            entry_points=[str(p) for p in (entry_points or [])],
            tags=list(tags or []),
            last_modified_unix=time.time(),
            last_session_id=last_session_id,
        )

        # Preserve created_at if upserting an existing entry.
        existing = self.get(pid)
        if existing is not None:
            entry.created_at_unix = existing.created_at_unix
            if not entry.tags and existing.tags:
                entry.tags = list(existing.tags)

        # Embed the summary text for search.
        try:
            vector = self._embedder.encode_query_dense(summary)
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "ProjectIndex: embedding failed for %s (%s); skipping upsert.",
                pid, e,
            )
            return None

        try:
            from qdrant_client.models import PointStruct
            self._client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=_qdrant_id_from_project_id(pid),
                        vector={"dense": vector.tolist()},
                        payload=entry.to_payload(),
                    ),
                ],
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning(
                "ProjectIndex: upsert failed for %s (%s); entry not stored.",
                pid, e,
            )
            return None

        with self._lock:
            # Refresh cache: drop any previous entry for this id, prepend.
            self._recent = [e for e in self._recent if e.project_id != pid]
            self._recent.insert(0, entry)
            if len(self._recent) > self._recent_cache_size:
                self._recent = self._recent[:self._recent_cache_size]

        try:
            bus_publish(ProjectIndexedEvent, {
                "project_id": entry.project_id,
                "project_name": entry.project_name,
                "digest_chars": len(entry.digest_markdown),
            })
        except Exception as e:                                      # noqa: BLE001
            logger.debug("bus publish failed for project.indexed (%s)", e)

        return entry

    def get(self, project_id: str) -> Optional[ProjectIndexEntry]:
        """Look up an entry by project_id. Returns None when missing."""
        if not project_id:
            return None
        with self._lock:
            for e in self._recent:
                if e.project_id == project_id:
                    return e
        # Fall through to Qdrant lookup.
        try:
            points = self._client.retrieve(
                collection_name=self.collection,
                ids=[_qdrant_id_from_project_id(project_id)],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:                                      # noqa: BLE001
            logger.debug("ProjectIndex.get: retrieve failed (%s)", e)
            return None
        if not points:
            return None
        return ProjectIndexEntry.from_payload(points[0].payload or {})

    def get_by_path(self, project_path: Path) -> Optional[ProjectIndexEntry]:
        """Look up by absolute project path."""
        return self.get(_derive_project_id(project_path))

    def list_all(self, limit: int = 100) -> List[ProjectIndexEntry]:
        """Return all entries, most-recently-modified first."""
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("ProjectIndex.list_all: scroll failed (%s)", e)
            return []
        entries = [
            ProjectIndexEntry.from_payload(p.payload or {})
            for p in points
        ]
        entries.sort(key=lambda e: e.last_modified_unix, reverse=True)
        return entries

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[ProjectMatch]:
        """Semantic search over project digests.

        Args:
            query: free-text query (typically the user's voice
                utterance, e.g. "edit the flask app").
            top_k: max results.
            min_score: lower bound on cosine similarity. Results
                below this are dropped. Callers use 0.55 for "could
                be a match", 0.75 for "high confidence".

        Returns:
            A list of :class:`ProjectMatch` ordered by descending
            score. Empty when no matches.
        """
        if not query or not query.strip():
            return []

        try:
            vector = self._embedder.encode_query_dense(query)
        except Exception as e:                                      # noqa: BLE001
            logger.warning("ProjectIndex.search: embed failed (%s)", e)
            return []

        try:
            response = self._client.query_points(
                collection_name=self.collection,
                query=vector.tolist(),
                using="dense",
                limit=max(1, top_k),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("ProjectIndex.search: qdrant search failed (%s)", e)
            return []

        # qdrant-client's query_points returns a response object with .points
        # (a list of ScoredPoint instances).
        hits = getattr(response, "points", None) or []
        results: List[ProjectMatch] = []
        for hit in hits:
            score = float(getattr(hit, "score", 0.0) or 0.0)
            if score < min_score:
                continue
            entry = ProjectIndexEntry.from_payload(hit.payload or {})
            results.append(ProjectMatch(
                entry=entry,
                score=score,
                reason=_score_reason(score),
            ))
        return results

    def search_by_name(
        self,
        name_substring: str,
        *,
        top_k: int = 10,
    ) -> List[ProjectIndexEntry]:
        """Lexical fallback: match project_name / tags for substring.

        Used by the supervisor when semantic search returns nothing
        confident. Cheap; in-memory + over the recent cache + a
        single scroll.
        """
        if not name_substring or not name_substring.strip():
            return []
        needle = name_substring.lower().strip()

        seen: set = set()
        hits: List[ProjectIndexEntry] = []
        with self._lock:
            for e in self._recent:
                if (
                    needle in e.project_name.lower()
                    or any(needle in t.lower() for t in e.tags)
                ):
                    hits.append(e)
                    seen.add(e.project_id)
                    if len(hits) >= top_k:
                        return hits[:top_k]

        # Fallback to full scroll for projects beyond cache.
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection,
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:                                      # noqa: BLE001
            logger.debug(
                "ProjectIndex.search_by_name: scroll failed (%s)", e,
            )
            return hits[:top_k]

        for p in points:
            e = ProjectIndexEntry.from_payload(p.payload or {})
            if e.project_id in seen:
                continue
            if (
                needle in e.project_name.lower()
                or any(needle in t.lower() for t in e.tags)
            ):
                hits.append(e)
                if len(hits) >= top_k:
                    break

        return hits[:top_k]

    def delete(self, project_id: str) -> bool:
        """Remove a project from the index. Returns True on success."""
        if not project_id:
            return False
        try:
            from qdrant_client.models import PointIdsList
            self._client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(
                    points=[_qdrant_id_from_project_id(project_id)],
                ),
            )
        except Exception as e:                                      # noqa: BLE001
            logger.warning("ProjectIndex.delete failed: %s", e)
            return False
        with self._lock:
            self._recent = [
                e for e in self._recent if e.project_id != project_id
            ]
        return True

    def count(self) -> int:
        """Return total entries in the collection (cheap)."""
        try:
            info = self._client.get_collection(self.collection)
            return int(getattr(info, "points_count", 0) or 0)
        except Exception:                                           # noqa: BLE001
            return 0

    def close(self) -> None:
        """Release the underlying Qdrant client IF this index owns it.

        Borrowed clients (``client=`` passed at construction) are
        never closed here -- the owner (ConversationMemory in the
        production wiring) controls that lifecycle. Never raises.
        """
        if not getattr(self, "_owns_client", False):
            return
        try:
            self._client.close()
        except Exception as e:                                      # noqa: BLE001
            logger.debug("ProjectIndex.close failed: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_project_id(project_path: Path) -> str:
    """Derive a stable project_id from an absolute path.

    Uses a UUID5 with a fixed namespace so the same project path
    always maps to the same id, even across process restarts.
    """
    namespace = uuid.UUID("8c0a6bf8-1f31-4d3a-8b6c-9e0b3a7d2a4f")
    return str(uuid.uuid5(namespace, str(project_path)))


def _qdrant_id_from_project_id(project_id: str) -> str:
    """Qdrant accepts UUID strings or ints as point ids.

    Since project_id is already a UUID5 string, return it directly.
    Surfaced as its own function in case we change the id strategy
    later (e.g. hash-based ids).
    """
    return project_id


def _build_digest_summary_for_search(
    sections: Mapping[str, str],
    *,
    max_chars: int = 500,
) -> str:
    """Build a short summary string used as the embedding target.

    Concatenates Goal + Critical Context + (truncated) Relevant Files.
    Short embeddings retrieve better than embedding the entire digest
    (the LLM's terse Goal line is the most semantically precise
    representation of what the project IS).
    """
    parts: List[str] = []
    goal = sections.get("Goal", "").strip()
    if goal:
        parts.append(_strip_bullets(goal, max_len=180))
    critical = sections.get("Critical Context", "").strip()
    if critical and critical.lower() != "- (none)":
        parts.append(_strip_bullets(critical, max_len=200))
    files = sections.get("Relevant Files", "").strip()
    if files and files.lower() != "- (none)":
        parts.append(_strip_bullets(files, max_len=200))
    summary = " | ".join(parts)
    return summary[:max_chars]


def _strip_bullets(block: str, *, max_len: int) -> str:
    """Trim a bullet list to a single-line summary."""
    cleaned = []
    for line in block.splitlines():
        s = line.strip().lstrip("-").strip()
        if s:
            cleaned.append(s)
    text = " ".join(cleaned)
    return text[:max_len]


def _score_reason(score: float) -> str:
    """Human-readable score band label."""
    if score >= 0.85:
        return "very high confidence semantic match"
    if score >= 0.75:
        return "high confidence semantic match"
    if score >= 0.55:
        return "possible semantic match"
    return "weak semantic match"


__all__ = [
    "ProjectIndex",
    "ProjectIndexEntry",
    "ProjectMatch",
]
