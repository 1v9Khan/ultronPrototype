"""Periodic / on-demand maintenance for the Qdrant memory store.

Run manually -- nothing here belongs on the hot path. All tasks are
idempotent: re-running with no new conversations is a no-op (or near-no-op),
and as you accumulate more turns each task picks up where it left off via
the SQLite metadata at ``data/maintenance.sqlite``.

Recommended cadence: run after each working session, or daily/weekly. A
typical run on a small store (< 1000 turns) takes a few minutes -- almost
all of it the LLM warmup.

Usage:
    python scripts/maintenance.py                          # all tasks
    python scripts/maintenance.py --task extract_facts     # one task
    python scripts/maintenance.py --task extract_facts --task cluster_conversations
    python scripts/maintenance.py --list-tasks

Tasks:

  backfill_metadata
      Fill missing summary / entities / topic_tags on conversation turns
      via the main LLM. Skips already-populated turns.

  extract_facts
      Slide a 6-turn window across new turns since last run. For each
      window, ask the LLM for durable facts (with confidence + category).
      Dedup against existing facts in the ``facts`` collection (cosine
      sim threshold 0.85): if similar fact exists, bump last_confirmed
      and confidence; if novel, insert.

  cluster_conversations
      k-means over dense vectors of the conversations collection. Adaptive
      k between 2 and 40 based on store size. Each cluster gets a 2-5 word
      LLM-generated topic label written back as ``cluster_label`` plus
      ``cluster_id`` on every point.

  daily_summary
      Last 24 h of turns -> 3-4 sentence summary -> ``data/summaries.jsonl``.

  decay_stale_facts
      Halve retrieval_weight on facts with last_confirmed older than 90 days
      and confidence < 0.7. No deletes.

  cleanup_web_cache
      Delete web_results older than per-point freshness (24 h volatile,
      30 d stable).

  resolve_observation_outcomes
      Offline pass over ``data/observations.jsonl``: resolve actions whose
      outcome was left ``unknown_yet`` at emit time by scanning the
      resolution window for follow-up rows, and emit the resolved outcome.
      Needs neither the LLM nor Qdrant -- pure log processing. Idempotent.

Most tasks need the main LLM. They share the same on-disk model file as the
live Ultron, so VRAM contention is real -- prefer running maintenance when
the live system isn't active.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Reach the main checkout so paths + models resolve.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from config import settings  # noqa: E402

_META_DB = _REPO / "data" / "maintenance.sqlite"


# ---------------------------------------------------------------------------
# Per-task last-processed bookkeeping
# ---------------------------------------------------------------------------


def _ensure_meta_db() -> sqlite3.Connection:
    _META_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_META_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_state (
            task TEXT PRIMARY KEY,
            last_processed_id INTEGER DEFAULT -1,
            last_processed_ts REAL DEFAULT 0,
            updated_at REAL
        )
        """
    )
    conn.commit()
    return conn


def _get_last_id(conn: sqlite3.Connection, task: str) -> int:
    row = conn.execute(
        "SELECT last_processed_id FROM task_state WHERE task = ?", (task,)
    ).fetchone()
    return int(row[0]) if row else -1


def _set_last_id(conn: sqlite3.Connection, task: str, last_id: int) -> None:
    conn.execute(
        """
        INSERT INTO task_state (task, last_processed_id, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(task) DO UPDATE SET
            last_processed_id = excluded.last_processed_id,
            updated_at = excluded.updated_at
        """,
        (task, int(last_id), time.time()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# LLM response parsing helpers
# ---------------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove all <think>...</think> blocks. Qwen3 emits these before answers."""
    return _THINK_RE.sub("", text).strip()


def _extract_json_payload(text: str):
    """Best-effort extract a JSON value from a possibly-noisy LLM response.

    Tries, in order:
      1. Strip ``<think>`` blocks.
      2. Pull contents of the first ``json`` (or unlabeled) code fence.
      3. Find the first balanced bracket / brace span.
      4. ``json.loads`` the whole stripped text.

    Returns the parsed value or raises ``ValueError`` if nothing parses.
    """
    text = _strip_thinking(text)
    if not text:
        raise ValueError("empty after thinking-strip")

    # Code fence first.
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Greedy first balanced span.
    for opener, closer in (("[", "]"), ("{", "}")):
        i = text.find(opener)
        if i == -1:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == opener:
                depth += 1
            elif text[j] == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[i: j + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Last shot: whole text.
    return json.loads(text)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


def _open_qdrant():
    """Open the embedded Qdrant client and ensure all three collections exist.

    Migration only seeds the ``conversations`` collection; the other two are
    created lazily by ``ConversationMemory.__init__`` in production. The
    maintenance script bypasses that constructor, so we replicate the
    bootstrap here -- otherwise ``decay_stale_facts`` and
    ``cleanup_web_cache`` blow up on a fresh store.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams

    client = QdrantClient(path=str(settings.MEMORY_QDRANT_PATH))
    existing = {c.name for c in client.get_collections().collections}
    common_dense = {"dense": VectorParams(size=settings.MEMORY_DENSE_DIM, distance=Distance.COSINE)}
    common_sparse = {"bm25": SparseVectorParams()}
    for name in (
        settings.MEMORY_QDRANT_CONVERSATIONS,
        settings.MEMORY_QDRANT_FACTS,
        settings.MEMORY_QDRANT_WEB_RESULTS,
    ):
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=common_dense,
                sparse_vectors_config=common_sparse,
            )
            print(f"  (created collection {name})")
    return client


def _all_conversations(client, with_vectors: bool = False) -> Iterable[Tuple[str, dict, Optional[list]]]:
    """Yield ``(point_id, payload, dense_vector_or_None)`` for every conversation point."""
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=settings.MEMORY_QDRANT_CONVERSATIONS,
            limit=256,
            with_payload=True,
            with_vectors=with_vectors,
            offset=offset,
        )
        for pt in page:
            dense = None
            if with_vectors and pt.vector:
                if isinstance(pt.vector, dict):
                    dense = pt.vector.get("dense")
                else:
                    dense = pt.vector
            yield pt.id, (pt.payload or {}), dense
        if offset is None:
            return


# ---------------------------------------------------------------------------
# Task: backfill_metadata
# ---------------------------------------------------------------------------


def run_backfill_metadata(llm, client, batch_size: int = 10) -> int:
    """Fill missing summary / entities / topic_tags on conversation points."""
    pending: List[Tuple[str, dict]] = []
    for pid, payload, _ in _all_conversations(client):
        if not payload.get("summary") or not payload.get("topic_tags"):
            pending.append((pid, payload))
    if not pending:
        print("  backfill_metadata: nothing to do")
        return 0

    print(f"  backfill_metadata: {len(pending)} points pending")
    processed = 0
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        items = "\n".join(
            f"{i+1}. ({p['role']}) {p['content']}"
            for i, (_, p) in enumerate(batch)
        )
        prompt = (
            "For each numbered conversation turn below, return a JSON list of "
            "objects with keys:\n"
            "  index (int, 1-based)\n"
            "  summary (one short sentence)\n"
            "  entities (list of named entities, can be empty)\n"
            "  topic_tags (1-3 short topic tags)\n"
            "Return ONLY the JSON list, no commentary, no markdown.\n\n"
            f"{items}\n"
        )
        try:
            data = _extract_json_payload(llm.generate(prompt))
            if not isinstance(data, list):
                raise ValueError(f"expected list, got {type(data).__name__}")
        except Exception as e:
            print(f"    batch {batch_start}: parse failed ({e}) -- skipping")
            continue

        for record in data:
            try:
                idx = int(record["index"]) - 1
                if not (0 <= idx < len(batch)):
                    continue
                pid, _ = batch[idx]
                client.set_payload(
                    collection_name=settings.MEMORY_QDRANT_CONVERSATIONS,
                    payload={
                        "summary": str(record.get("summary") or ""),
                        "entities": list(record.get("entities") or []),
                        "topic_tags": list(record.get("topic_tags") or []),
                    },
                    points=[pid],
                )
                processed += 1
            except Exception as e:
                print(f"    record skipped: {e}")
        print(f"    backfilled {processed}/{len(pending)}")
    return processed


# ---------------------------------------------------------------------------
# Task: extract_facts
# ---------------------------------------------------------------------------


_FACT_PROMPT = """Extract durable facts from this conversation. A durable fact is something that will remain true beyond this session: preferences, decisions, project information, personal context, persistent constraints, recurring patterns.

If no durable facts emerge, return an empty list [].

Return ONLY a JSON list, no commentary, no markdown. Each item:
{{"fact": "<one canonical sentence>", "confidence": <0.0-1.0>, "category": "preference|project|person|decision|constraint"}}

Conversation:
{transcript}
"""


def run_extract_facts(
    llm,
    client,
    conn,
    embedder,
    window_size: int = 6,
    overlap: int = 3,
    similarity_threshold: float = 0.85,
) -> int:
    """Sliding-window fact extraction over new turns since last run."""
    from qdrant_client.models import PointStruct, SparseVector

    last_id = _get_last_id(conn, "extract_facts")

    # Pull all turns; keep only those past last_id, sort chronologically.
    candidates: List[Tuple[int, str, str]] = []
    for _, payload, _ in _all_conversations(client):
        tid = int(payload.get("turn_id", -1))
        if tid > last_id:
            candidates.append((tid, payload.get("role", ""), payload.get("content", "")))
    candidates.sort(key=lambda t: t[0])

    if not candidates:
        print(f"  extract_facts: no new turns since id={last_id}")
        return 0

    print(f"  extract_facts: {len(candidates)} new turns to process")

    facts_added = 0
    facts_confirmed = 0
    high_water = last_id

    step = max(1, window_size - overlap)
    for window_start in range(0, len(candidates), step):
        window = candidates[window_start: window_start + window_size]
        if len(window) < 2:
            # A single isolated turn rarely yields a durable fact; require
            # at least one user/assistant pair.
            break

        transcript = "\n".join(f"{role}: {content}" for _, role, content in window)
        try:
            data = _extract_json_payload(llm.generate(_FACT_PROMPT.format(transcript=transcript)))
            if not isinstance(data, list):
                raise ValueError(f"expected list, got {type(data).__name__}")
        except Exception as e:
            print(f"    window starting at turn {window[0][0]}: parse failed ({e}) -- skipping")
            high_water = max(high_water, window[-1][0])
            continue

        source_turn_ids = [tid for tid, _, _ in window]
        for record in data:
            if not isinstance(record, dict):
                continue
            fact_text = str(record.get("fact", "")).strip()
            if not fact_text:
                continue
            try:
                confidence = float(record.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            category = str(record.get("category", "") or "").strip().lower()

            try:
                added, confirmed = _upsert_fact(
                    client,
                    embedder,
                    fact_text=fact_text,
                    confidence=confidence,
                    category=category,
                    source_turn_ids=source_turn_ids,
                    similarity_threshold=similarity_threshold,
                )
                facts_added += added
                facts_confirmed += confirmed
            except Exception as e:
                print(f"    fact upsert failed for {fact_text[:60]!r}: {e}")

        high_water = max(high_water, window[-1][0])

    _set_last_id(conn, "extract_facts", high_water)
    total = facts_added + facts_confirmed
    print(
        f"  extract_facts: {facts_added} new, {facts_confirmed} re-confirmed "
        f"(high-water turn id = {high_water})"
    )
    return total


def _upsert_fact(
    client,
    embedder,
    *,
    fact_text: str,
    confidence: float,
    category: str,
    source_turn_ids: List[int],
    similarity_threshold: float,
) -> Tuple[int, int]:
    """Insert ``fact_text`` if novel; otherwise update last_confirmed.

    Returns ``(new_inserted, re_confirmed)`` -- exactly one of which is 1.
    """
    from qdrant_client.models import PointStruct, SparseVector

    fact_dense = embedder.encode_dense(fact_text)
    fact_sparse = embedder.encode_sparse(fact_text)[0]
    now = time.time()

    # Search the facts collection for a similar existing fact via dense.
    similar = client.query_points(
        collection_name=settings.MEMORY_QDRANT_FACTS,
        query=fact_dense.tolist(),
        using="dense",
        limit=1,
        score_threshold=similarity_threshold,
        with_payload=True,
    ).points

    if similar:
        existing = similar[0]
        existing_payload = existing.payload or {}
        old_conf = float(existing_payload.get("confidence", 0.0))
        merged_conf = max(old_conf, confidence)
        new_sources = sorted(set(
            int(s) for s in (existing_payload.get("extracted_from") or [])
        ) | set(source_turn_ids))
        client.set_payload(
            collection_name=settings.MEMORY_QDRANT_FACTS,
            payload={
                "last_confirmed": now,
                "confidence": merged_conf,
                "extracted_from": new_sources,
            },
            points=[existing.id],
        )
        return (0, 1)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": fact_dense.tolist(),
            "bm25": SparseVector(indices=fact_sparse.indices, values=fact_sparse.values),
        },
        payload={
            "fact": fact_text,
            "confidence": confidence,
            "category": category,
            "extracted_from": source_turn_ids,
            "extracted_at": now,
            "last_confirmed": now,
            "retrieval_weight": 1.0,
        },
    )
    client.upsert(
        collection_name=settings.MEMORY_QDRANT_FACTS,
        points=[point],
    )
    return (1, 0)


# ---------------------------------------------------------------------------
# Task: cluster_conversations
# ---------------------------------------------------------------------------


_CLUSTER_LABEL_PROMPT = """Below are excerpts from a cluster of conversation turns that the system grouped together. Return a 2-5 word topic label that captures what they have in common. Return ONLY the label, no quotes, no commentary, no markdown.

Excerpts:
{excerpts}

Label:"""


def run_cluster_conversations(
    llm,
    client,
    min_points_for_clustering: int = 8,
) -> int:
    """Re-cluster the conversation collection and label each cluster.

    Re-runs from scratch each time (clusters are global, so we don't try to
    incrementally update). For small stores this is fast.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    points: List[Tuple[str, dict, list]] = []
    for pid, payload, dense in _all_conversations(client, with_vectors=True):
        if dense is not None:
            points.append((pid, payload, dense))

    if len(points) < min_points_for_clustering:
        print(
            f"  cluster_conversations: only {len(points)} points "
            f"(need >={min_points_for_clustering}); skipping"
        )
        return 0

    n = len(points)
    # Spec suggests k=20-40, but adapt down for small stores so each cluster
    # has at least ~5 turns.
    k = max(2, min(40, n // 5))
    print(f"  cluster_conversations: clustering {n} points into k={k}")

    X = np.vstack([np.asarray(v, dtype=np.float32) for _, _, v in points])
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    cluster_labels: dict[int, str] = {}
    for cluster_id in range(k):
        members = [
            (i, points[i]) for i in range(n) if int(labels[i]) == cluster_id
        ]
        if not members:
            continue

        centroid = km.cluster_centers_[cluster_id]
        members.sort(
            key=lambda im: float(np.linalg.norm(np.asarray(im[1][2]) - centroid))
        )
        sample = members[: min(10, len(members))]
        excerpts = "\n".join(
            f"  ({p[1].get('role','?')}) {(p[1].get('content','') or '').strip()[:160]}"
            for _, p in sample
        )

        try:
            raw = llm.generate(_CLUSTER_LABEL_PROMPT.format(excerpts=excerpts))
            label = _strip_thinking(raw).strip()
            label = label.split("\n", 1)[0].strip().strip('"').strip("'").strip()
            # Cap length so a chatty label doesn't pollute the payload.
            if len(label) > 60:
                label = label[:60]
            cluster_labels[cluster_id] = label or f"cluster {cluster_id}"
        except Exception as e:
            print(f"    cluster {cluster_id}: label failed ({e})")
            cluster_labels[cluster_id] = f"cluster {cluster_id}"

    updates = 0
    for i, (pid, _, _) in enumerate(points):
        cid = int(labels[i])
        client.set_payload(
            collection_name=settings.MEMORY_QDRANT_CONVERSATIONS,
            payload={
                "cluster_id": cid,
                "cluster_label": cluster_labels.get(cid, f"cluster {cid}"),
                "clustered_at": time.time(),
            },
            points=[pid],
        )
        updates += 1

    print(f"  cluster_conversations: labeled {len(cluster_labels)} clusters, updated {updates} points")
    for cid in sorted(cluster_labels):
        n_in = int((labels == cid).sum())
        print(f"    cluster {cid}: {n_in:3d} points -- {cluster_labels[cid]!r}")
    return updates


# ---------------------------------------------------------------------------
# Task: daily_summary
# ---------------------------------------------------------------------------


def run_daily_summary(llm, client) -> int:
    """Pull last 24 h of turns -> 3-4 sentence summary -> data/summaries.jsonl."""
    cutoff = time.time() - 24 * 3600
    recent: List[dict] = []
    for _, payload, _ in _all_conversations(client):
        if float(payload.get("ts", 0.0)) >= cutoff:
            recent.append(payload)
    if not recent:
        print("  daily_summary: no turns in last 24 h")
        return 0
    recent.sort(key=lambda p: float(p.get("ts", 0.0)))
    transcript = "\n".join(
        f"{p.get('role','?')}: {(p.get('content','') or '').strip()}" for p in recent
    )
    prompt = (
        "Summarize the following conversation transcript in 3-4 sentences. "
        "Highlight what was discussed, decisions made, and unresolved questions. "
        "Plain prose only -- no commentary on the summary itself, no markdown.\n\n"
        f"{transcript}\n"
    )
    try:
        summary = _strip_thinking(llm.generate(prompt))
    except Exception as e:
        print(f"  daily_summary: LLM call failed ({e})")
        return 0
    out_path = _REPO / "data" / "summaries.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "covered_seconds": 24 * 3600,
        "turn_count": len(recent),
        "summary": summary,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  daily_summary: wrote {len(summary)} chars covering {len(recent)} turns to {out_path}")
    return 1


# ---------------------------------------------------------------------------
# Task: decay_stale_facts
# ---------------------------------------------------------------------------


def run_decay_stale_facts(client) -> int:
    """Halve retrieval weight on facts older than 90 days with confidence < 0.7.

    No deletes. We only act on facts that haven't already been decayed, so
    re-running is a no-op once the decay has been applied.
    """
    cutoff = time.time() - 90 * 24 * 3600
    decayed = 0
    try:
        offset = None
        while True:
            page, offset = client.scroll(
                collection_name=settings.MEMORY_QDRANT_FACTS,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for pt in page:
                pl = pt.payload or {}
                last = float(pl.get("last_confirmed", pl.get("ts", pl.get("extracted_at", 0.0))))
                conf = float(pl.get("confidence", 1.0))
                if last < cutoff and conf < 0.7 and not pl.get("decayed_at"):
                    client.set_payload(
                        collection_name=settings.MEMORY_QDRANT_FACTS,
                        payload={
                            "decayed_at": time.time(),
                            "retrieval_weight": 0.5,
                        },
                        points=[pt.id],
                    )
                    decayed += 1
            if offset is None:
                break
    except Exception as e:
        print(f"  decay_stale_facts: scan failed ({e})")
        return 0
    print(f"  decay_stale_facts: {decayed} facts decayed")
    return decayed


# ---------------------------------------------------------------------------
# Task: cleanup_web_cache
# ---------------------------------------------------------------------------


def run_cleanup_web_cache(client) -> int:
    """Delete web_results older than per-point freshness."""
    now = time.time()
    deleted_ids: List[str] = []
    try:
        offset = None
        while True:
            page, offset = client.scroll(
                collection_name=settings.MEMORY_QDRANT_WEB_RESULTS,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for pt in page:
                pl = pt.payload or {}
                fetched = float(pl.get("fetched_at", 0.0))
                fresh = pl.get("freshness_category", "stable")
                ttl = 24 * 3600 if fresh == "volatile" else 30 * 24 * 3600
                if (now - fetched) > ttl:
                    deleted_ids.append(pt.id)
            if offset is None:
                break
    except Exception as e:
        print(f"  cleanup_web_cache: scan failed ({e}) -- collection may be empty")
        return 0
    if deleted_ids:
        from qdrant_client.models import PointIdsList

        client.delete(
            collection_name=settings.MEMORY_QDRANT_WEB_RESULTS,
            points_selector=PointIdsList(points=deleted_ids),
        )
    print(f"  cleanup_web_cache: {len(deleted_ids)} stale web results deleted")
    return len(deleted_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_resolve_observation_outcomes() -> int:
    """Resolve pending observation outcomes in ``data/observations.jsonl``.

    Offline pass over the canonical observation log: for each action whose
    outcome was emitted as ``unknown_yet``, scan the resolution window for the
    correlated follow-up rows and emit a resolved outcome. Needs neither the
    LLM nor Qdrant -- pure log processing. Idempotent (already-resolved rows
    are skipped). Returns the count resolved this run."""
    from ultron.observations import resolve_outcomes

    summary = resolve_outcomes()
    print(f"  observations resolved: {summary.as_dict()}")
    return summary.resolved_now


_TASKS = [
    "backfill_metadata",
    "extract_facts",
    "cluster_conversations",
    "daily_summary",
    "decay_stale_facts",
    "cleanup_web_cache",
    "resolve_observation_outcomes",
]


def _load_llm():
    """Lazy-load the main LLM. Memory module is intentionally NOT wired so
    the maintenance run doesn't pollute the hot-path memory.

    Maintenance prompts often want larger structured outputs (JSON lists of
    summaries / facts) than the live-chat 512-token budget allows --
    Qwen3.5's reasoning block alone consumes 200-500 tokens before the
    answer starts. Bump the global cap for this process so the LLM has
    room to finish.
    """
    settings.LLM_MAX_TOKENS = 2048
    from ultron.llm import LLMEngine

    return LLMEngine(memory=None)


def _load_embedder():
    from ultron.memory import HybridEmbedder

    return HybridEmbedder(eager=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument(
        "--task",
        choices=_TASKS,
        action="append",
        help="run a single task (can be repeated)",
    )
    args = parser.parse_args()

    if args.list_tasks:
        for t in _TASKS:
            print(f"  {t}")
        return 0

    chosen = args.task or _TASKS
    print(f"Maintenance run: {len(chosen)} tasks -> {chosen}")

    client = _open_qdrant()
    conn = _ensure_meta_db()

    needs_llm = any(
        t in chosen
        for t in (
            "backfill_metadata",
            "extract_facts",
            "cluster_conversations",
            "daily_summary",
        )
    )
    needs_embedder = "extract_facts" in chosen

    llm = None
    embedder = None
    if needs_llm:
        print("Loading main LLM (slow but maintenance is infrequent)...")
        llm = _load_llm()
    if needs_embedder:
        print("Loading HybridEmbedder...")
        embedder = _load_embedder()

    started = time.monotonic()
    summary: dict[str, int] = {}

    for task in chosen:
        print(f"\n[{task}]")
        try:
            if task == "backfill_metadata":
                summary[task] = run_backfill_metadata(llm, client)
            elif task == "extract_facts":
                summary[task] = run_extract_facts(llm, client, conn, embedder)
            elif task == "cluster_conversations":
                summary[task] = run_cluster_conversations(llm, client)
            elif task == "daily_summary":
                summary[task] = run_daily_summary(llm, client)
            elif task == "decay_stale_facts":
                summary[task] = run_decay_stale_facts(client)
            elif task == "cleanup_web_cache":
                summary[task] = run_cleanup_web_cache(client)
            elif task == "resolve_observation_outcomes":
                summary[task] = run_resolve_observation_outcomes()
        except Exception as e:
            print(f"  TASK FAILED: {task}: {e}")
            summary[task] = -1

    print(f"\nDone in {time.monotonic() - started:.1f}s")
    for task, count in summary.items():
        print(f"  {task:25s} {count}")
    conn.close()
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
