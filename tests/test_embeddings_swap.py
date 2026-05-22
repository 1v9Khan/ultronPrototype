"""Embedder swap tests (frontier item 3, 2026-05-21).

bge-small-en-v1.5 (384-dim, MTEB ~62.5) -> jina-embeddings-v3
(1024-dim, MTEB ~65.5). Both run through FastEmbed's ONNX CPU path.
The dim change is a one-way migration -- existing 384-dim Qdrant
collections need to be re-embedded via
``scripts/migrate_embeddings.py``.

Tests cover:
- Config defaults reflect the new model + dim.
- Old model is still settable via config (swap-back path).
- ``ConversationMemory`` startup raises a clear, actionable error
  when the existing collection's dim doesn't match the configured
  embedder dim -- so operators see the migration prompt instead of
  a cryptic mid-turn vector-size error.

No real FastEmbed download happens here -- the embedder is mocked
via the ``HybridEmbedder`` constructor that doesn't trigger ONNX
load until first encode.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ultron.config import EmbeddingsConfig, UltronConfig


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_embeddings_default_is_bge_small_after_revert():
    """2026-05-21 frontier-enhancement Item 3: we originally flipped
    the default to jina-v3 (1024 dim, MTEB ~65.5), but a live bench
    measured 568 ms per encode call (vs bge-small's 3 ms) -- a
    **183x slowdown** that makes it the wrong default for the
    voice memory write path. The default stays on bge-small;
    jina-v3 remains opt-in via explicit config + the migration
    script."""
    cfg = EmbeddingsConfig()
    assert cfg.dense_model == "BAAI/bge-small-en-v1.5"
    assert cfg.dense_dim == 384
    # Sparse stays on BM25; that part of the hybrid layer is unchanged.
    assert cfg.sparse_model == "Qdrant/bm25"


def test_embeddings_can_opt_in_to_jina_v3():
    """Setting the jina-v3 values via config still works -- the
    swap is bidirectional, just not the default."""
    cfg = EmbeddingsConfig(
        dense_model="jinaai/jina-embeddings-v3",
        dense_dim=1024,
    )
    assert cfg.dense_model == "jinaai/jina-embeddings-v3"
    assert cfg.dense_dim == 1024


def test_full_config_round_trip_with_bge_default():
    cfg = UltronConfig()
    assert cfg.embeddings.dense_model == "BAAI/bge-small-en-v1.5"
    assert cfg.embeddings.dense_dim == 384


def test_full_config_round_trip_with_explicit_jina_override():
    """Explicit YAML override flips on jina-v3."""
    cfg = UltronConfig.model_validate({
        "embeddings": {
            "dense_model": "jinaai/jina-embeddings-v3",
            "dense_dim": 1024,
        }
    })
    assert cfg.embeddings.dense_model == "jinaai/jina-embeddings-v3"
    assert cfg.embeddings.dense_dim == 1024


# ---------------------------------------------------------------------------
# Dimension mismatch detection
# ---------------------------------------------------------------------------


def _stub_qdrant_collection(existing_dim: int):
    """Build a mock Qdrant client whose ``get_collection`` reports
    a ``conversations`` collection at ``existing_dim``."""
    vectors_cfg = MagicMock()
    vectors_cfg.size = existing_dim

    info = MagicMock()
    info.config.params.vectors = {"dense": vectors_cfg}

    client = MagicMock()
    client.get_collections.return_value.collections = [
        SimpleNamespace(name="conversations"),
    ]
    client.get_collection.return_value = info
    return client


def _stub_memory_for_init():
    """Build a ConversationMemory-like object with just the fields
    ``_ensure_collections`` needs."""
    from ultron.memory.qdrant_store import ConversationMemory
    cm = object.__new__(ConversationMemory)
    cm._client = None
    cm._embedder = MagicMock()
    cm._embedder.dim = 1024  # new model dim
    return cm


def test_dim_mismatch_raises_with_actionable_message():
    """When configured dim (1024) doesn't match existing collection
    dim (384), startup raises with the migration instructions."""
    from ultron.memory.qdrant_store import ConversationMemory

    cm = _stub_memory_for_init()
    cm._client = _stub_qdrant_collection(existing_dim=384)

    with pytest.raises(RuntimeError) as exc_info:
        ConversationMemory._ensure_collections(cm)

    msg = str(exc_info.value)
    # Required: dimensions surfaced
    assert "384" in msg
    assert "1024" in msg
    # Required: migration command surfaced
    assert "migrate_embeddings.py" in msg
    # Required: revert path mentioned
    assert "dense_model" in msg or "config.yaml" in msg


def test_dim_match_does_not_raise():
    """When configured dim matches the existing collection, no error."""
    from ultron.memory.qdrant_store import ConversationMemory

    cm = _stub_memory_for_init()
    cm._client = _stub_qdrant_collection(existing_dim=1024)
    # Should not raise.
    ConversationMemory._ensure_collections(cm)


def test_no_existing_collection_does_not_raise():
    """Fresh install (no `conversations` collection yet) skips the
    dim-check and proceeds to create the collection."""
    from ultron.memory.qdrant_store import ConversationMemory

    cm = _stub_memory_for_init()
    # No collections at all.
    cm._client = MagicMock()
    cm._client.get_collections.return_value.collections = []
    # Should not raise; create_collection should be called.
    ConversationMemory._ensure_collections(cm)
    assert cm._client.create_collection.call_count >= 1


def test_introspect_failure_is_fail_open(monkeypatch, caplog):
    """If get_collection itself errors (corrupt metadata, future API
    change, etc.), we log WARN and continue. We don't want to brick
    the orchestrator on diagnostic IO."""
    from ultron.memory.qdrant_store import ConversationMemory

    cm = _stub_memory_for_init()
    cm._client = MagicMock()
    cm._client.get_collections.return_value.collections = [
        SimpleNamespace(name="conversations"),
    ]
    cm._client.get_collection.side_effect = RuntimeError("simulated probe failure")
    # Must not raise.
    ConversationMemory._ensure_collections(cm)
