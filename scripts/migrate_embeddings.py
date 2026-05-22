"""Re-embed every entry in the Qdrant memory store with a new dense
model + dimension. Used when swapping the embedder
(frontier-enhancement Item 3, 2026-05-21).

The current bge-small (384-dim) -> jina-v3 (1024-dim) swap is a
one-way migration: Qdrant collections are immutable in
``vectors_config.size``, so the only path forward is to back up the
old collection, create a fresh one at the new dim, and re-embed
every stored ``content`` field.

This script is **idempotent + safe**:

- Reads from ``data/qdrant/`` (the live store).
- Backs up the old collection's points to ``data/qdrant_backup_<ts>/``.
- Creates a new Qdrant store under ``data/qdrant_new/`` with the
  new dim from config.
- Re-embeds every ``content`` with the new dense model and reinserts
  with the original payload preserved (turn_id, ts, channel, session_id,
  topic_id, discourse_type, role, etc.).
- Atomically swaps ``data/qdrant/`` -> ``data/qdrant_old/`` and
  ``data/qdrant_new/`` -> ``data/qdrant/``.

Usage::

    # Stop any running Ultron instance first (the qdrant store is
    # held with an exclusive lock).
    python scripts/migrate_embeddings.py

    # Dry run -- shows counts but doesn't write:
    python scripts/migrate_embeddings.py --dry-run

    # Custom paths (advanced):
    python scripts/migrate_embeddings.py \\
        --source data/qdrant \\
        --target data/qdrant_new \\
        --backup data/qdrant_backup_pre_jina_v3

Returns exit code 0 on success, 1 on failure. On failure the
original ``data/qdrant/`` is preserved untouched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional


# Add project root + src/ to path so `from ultron.config import ...` works
# whether you run from project root or wherever.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _resolve(path_str: str) -> Path:
    """Resolve a path relative to project root."""
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _open_old_store(source_path: Path):
    """Open the legacy Qdrant store read-only and return (client,
    collection_name, dim)."""
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(source_path))
    cols = client.get_collections().collections
    if not cols:
        raise RuntimeError(f"No collections found in {source_path}")
    # We migrate the "conversations" collection. Other collections
    # (facts, web_results) could be added if/when they get used.
    target_name = None
    for c in cols:
        if c.name == "conversations":
            target_name = c.name
            break
    if target_name is None:
        raise RuntimeError(
            f"'conversations' collection not found in {source_path}. "
            f"Available: {[c.name for c in cols]}"
        )
    info = client.get_collection(target_name)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        dense_dim = vectors["dense"].size
    else:
        dense_dim = vectors.size
    return client, target_name, dense_dim


def _read_all_points(client, collection: str) -> List:
    """Scroll the entire collection. Memory size is bounded by the
    payload count; conversational corpora are small (~hundreds to
    thousands of turns)."""
    all_points = []
    offset = None
    page = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            with_payload=True,
            with_vectors=False,                # we re-embed; don't need old vectors
            offset=offset,
        )
        all_points.extend(points)
        page += 1
        if not offset:
            break
    return all_points


def _create_new_store(target_path: Path, dense_dim: int):
    """Create a new Qdrant store at ``target_path`` with a fresh
    ``conversations`` collection at the requested dim."""
    if target_path.exists():
        raise RuntimeError(
            f"Target {target_path} already exists. Delete it or pass a "
            f"different --target.",
        )
    target_path.mkdir(parents=True, exist_ok=True)
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, SparseVectorParams, VectorParams,
    )
    client = QdrantClient(path=str(target_path))
    client.create_collection(
        collection_name="conversations",
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={"bm25": SparseVectorParams()},
    )
    return client


def _embed_and_insert(
    new_client,
    old_points,
    *,
    batch_size: int = 32,
) -> int:
    """Re-embed ``content`` for every point and upsert into the new
    store. Returns the number of points migrated.

    When ``memory.contextual_retrieval.enabled`` is True, each turn
    also gets a context-summary phrase generated (or reused if the
    legacy payload already had one). The phrase is prepended to the
    DENSE embedding text and persisted at ``payload["context_summary"]``.
    Idempotent: re-running the migration with contextualization on
    reuses the existing summary instead of regenerating.
    """
    from ultron.config import get_config
    from ultron.memory.embedder import HybridEmbedder
    from qdrant_client.models import PointStruct, SparseVector

    ctx_cfg = get_config().memory.contextual_retrieval
    contextualize = bool(getattr(ctx_cfg, "enabled", False))
    ctx_gen = None
    if contextualize:
        try:
            from ultron.memory.contextualizer import ContextGenerator
            ctx_gen = ContextGenerator(eager=True)
            print("  contextual retrieval ENABLED; will generate per-turn "
                  "summaries during re-embed.")
        except Exception as e:
            print(f"  WARNING: context generator init failed ({e}); "
                  f"falling back to non-contextualized re-embed.")
            ctx_gen = None
            contextualize = False

    embedder = HybridEmbedder(eager=True)
    total = 0
    for batch_start in range(0, len(old_points), batch_size):
        batch = old_points[batch_start:batch_start + batch_size]
        contents = [str((p.payload or {}).get("content", "")) for p in batch]
        roles = [str((p.payload or {}).get("role", "user")) for p in batch]

        # If contextualizing, compute per-turn embed text + summary
        # (reusing existing summary if present in legacy payload).
        if contextualize and ctx_gen is not None:
            embed_texts = []
            summaries = []
            for content, role, p in zip(contents, roles, batch):
                old_payload = p.payload or {}
                summary = str(old_payload.get("context_summary") or "")
                if not summary:
                    summary = ctx_gen.generate_context(content, role=role)
                summaries.append(summary)
                if summary:
                    embed_texts.append(f"[{summary}] {role}: {content}")
                else:
                    embed_texts.append(f"{role}: {content}")
        else:
            embed_texts = [
                f"{role}: {content}"
                for role, content in zip(roles, contents)
            ]
            summaries = [""] * len(batch)

        # Use document-side embed; query_embed has the wrong instruction
        # prefix for stored corpus entries.
        dense_vecs = embedder.encode_dense(embed_texts)
        # Sparse BM25 stays on the plain content (no synthesized
        # context) to avoid over-weighting LLM-generated tokens.
        sparse_vecs = embedder.encode_sparse(contents)
        new_points = []
        for i, p in enumerate(batch):
            payload = dict(p.payload or {})
            if summaries[i]:
                payload["context_summary"] = summaries[i]
            dense = dense_vecs[i].tolist()
            sparse_vec = sparse_vecs[i]
            new_points.append(PointStruct(
                id=p.id,
                vector={
                    "dense": dense,
                    "bm25": SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                },
                payload=payload,
            ))
        new_client.upsert(collection_name="conversations", points=new_points)
        total += len(new_points)
        print(f"  migrated {total}/{len(old_points)} points")
    return total


def _atomic_swap(source: Path, target: Path, backup: Path) -> None:
    """Move source -> backup, target -> source. Manual rollback path
    documented in the script docstring + status messages."""
    if backup.exists():
        raise RuntimeError(
            f"Backup target {backup} already exists. Delete or rename it."
        )
    print(f"  moving {source} -> {backup}")
    shutil.move(str(source), str(backup))
    print(f"  moving {target} -> {source}")
    shutil.move(str(target), str(source))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default="data/qdrant",
                        help="Path to existing Qdrant store (default: data/qdrant).")
    parser.add_argument("--target", default="data/qdrant_new",
                        help="Path to write the new Qdrant store (default: data/qdrant_new).")
    parser.add_argument(
        "--backup",
        default=f"data/qdrant_backup_pre_jina_v3_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Where to move the old store after success.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts but don't write or swap.",
    )
    args = parser.parse_args(argv)

    src = _resolve(args.source)
    tgt = _resolve(args.target)
    bk = _resolve(args.backup)

    print("Embedder migration")
    print("-" * 60)
    print(f"  source:  {src}")
    print(f"  target:  {tgt}")
    print(f"  backup:  {bk}")
    print()

    if not src.is_dir():
        print(f"ERROR: source {src} does not exist or is not a directory.")
        return 1

    try:
        from ultron.config import get_config
        cfg = get_config()
        new_dim = cfg.embeddings.dense_dim
        new_model = cfg.embeddings.dense_model
    except Exception as e:
        print(f"ERROR: could not read config: {e}")
        return 1

    print(f"  new model: {new_model}  (dim={new_dim})")
    print()

    try:
        old_client, collection, old_dim = _open_old_store(src)
    except Exception as e:
        print(f"ERROR: could not open old store: {e}")
        return 1
    print(f"  old store dim: {old_dim}  (collection: {collection})")

    if old_dim == new_dim:
        print()
        print("  -> source dim matches target dim; nothing to migrate.")
        print("     If you really need to re-embed (e.g., changed model "
              "within the same dim), delete data/qdrant and let Ultron "
              "rebuild from JSONL.")
        # Make sure the old client releases the lock so subsequent runs
        # of Ultron don't error out.
        try:
            old_client.close()
        except Exception:
            pass
        return 0

    try:
        print("  reading all points from old store...")
        points = _read_all_points(old_client, collection)
        print(f"  read {len(points)} points")
    except Exception as e:
        print(f"ERROR: could not scroll old collection: {e}")
        return 1
    finally:
        # Release the qdrant lock so the new store can open.
        try:
            old_client.close()
        except Exception:
            pass

    if args.dry_run:
        print()
        print(f"  DRY RUN -- would migrate {len(points)} points "
              f"from {old_dim}-dim to {new_dim}-dim.")
        return 0

    try:
        print(f"  creating new store at {tgt} (dim={new_dim})...")
        new_client = _create_new_store(tgt, new_dim)
    except Exception as e:
        print(f"ERROR: could not create new store: {e}")
        return 1

    try:
        print("  re-embedding + inserting...")
        t0 = time.monotonic()
        n_migrated = _embed_and_insert(new_client, points)
        elapsed = time.monotonic() - t0
        print(f"  re-embedded {n_migrated} points in {elapsed:.1f}s")
    except Exception as e:
        print(f"ERROR: re-embed failed: {e}")
        print(f"  partial new store left at {tgt}; old store untouched.")
        try:
            new_client.close()
        except Exception:
            pass
        return 1
    finally:
        try:
            new_client.close()
        except Exception:
            pass

    try:
        print()
        print("  swapping stores...")
        _atomic_swap(src, tgt, bk)
        print()
        print(f"  SUCCESS. Old store backed up at {bk}.")
        print("  If retrieval looks wrong after this, you can roll back:")
        print(f"    rmdir /s /q {src}")
        print(f"    move {bk} {src}")
        print("  ...and set ``dense_model``/``dense_dim`` in config.yaml "
              "back to the old values.")
        return 0
    except Exception as e:
        print(f"ERROR: swap failed: {e}")
        print(f"  Old store still at {src}; new store at {tgt}; backup at {bk}.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
