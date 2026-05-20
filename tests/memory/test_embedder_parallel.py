"""Tests for HybridEmbedder.encode_query_dense_sparse (Track 2).

The new helper bundles dense + sparse encoding into a single call,
with optional parallelisation. Tests use a stub HybridEmbedder that
avoids the FastEmbed model load so the suite stays fast and runs
on any machine.
"""

from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import pytest

from ultron.memory.embedder import HybridEmbedder, _SparseVec


def _build_stubbed_embedder(
    dense_delay_ms: float = 0.0,
    sparse_delay_ms: float = 0.0,
) -> HybridEmbedder:
    """Construct a HybridEmbedder whose dense/sparse encoders are
    stubbed -- avoids the FastEmbed model load.

    ``dense_delay_ms`` and ``sparse_delay_ms`` let tests verify the
    parallel-vs-serial timing claim by sleeping inside each encoder.
    """
    embedder = HybridEmbedder.__new__(HybridEmbedder)
    embedder._dense = object()  # truthy; bypasses _ensure_dense reload
    embedder._sparse = object()
    embedder._lock = threading.Lock()
    embedder.dense_model_name = "stub-dense"
    embedder.sparse_model_name = "stub-sparse"

    # Override encode methods with stubs.
    def fake_dense(query: str) -> np.ndarray:
        if dense_delay_ms > 0:
            time.sleep(dense_delay_ms / 1000.0)
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    def fake_sparse(query: str) -> _SparseVec:
        if sparse_delay_ms > 0:
            time.sleep(sparse_delay_ms / 1000.0)
        return _SparseVec([1, 2, 3], [0.4, 0.5, 0.6])

    embedder.encode_query_dense = fake_dense  # type: ignore[assignment]
    embedder.encode_query_sparse = fake_sparse  # type: ignore[assignment]
    return embedder


def test_encode_query_dense_sparse_serial_returns_both():
    """Default ``parallel=False`` returns the dense + sparse tuple."""
    e = _build_stubbed_embedder()
    dense, sparse = e.encode_query_dense_sparse("test query")
    assert isinstance(dense, np.ndarray)
    assert dense.dtype == np.float32
    assert isinstance(sparse, _SparseVec)
    assert sparse.indices == [1, 2, 3]


def test_encode_query_dense_sparse_parallel_returns_both():
    """``parallel=True`` path returns the same outputs."""
    e = _build_stubbed_embedder()
    dense, sparse = e.encode_query_dense_sparse("test query", parallel=True)
    assert isinstance(dense, np.ndarray)
    assert dense.dtype == np.float32
    assert isinstance(sparse, _SparseVec)
    assert sparse.indices == [1, 2, 3]


def test_encode_parallel_overlaps_dense_and_sparse():
    """Wall-clock for parallel mode should be meaningfully shorter
    than serial. Each stub sleeps 200 ms so the timing margin is
    well above thread-launch noise (~20-40 ms on Windows under
    contention) -- this keeps the assertion robust when the suite
    runs alongside other heavy work.

    Retries up to 3 times because the timing assertion is
    contention-sensitive; an OS scheduler hiccup in the middle of
    one timed call can blur the signal. Three independent samples
    + a saved-ms floor avoids spurious failures on a busy machine
    without weakening the parallelism contract.
    """
    e = _build_stubbed_embedder(dense_delay_ms=200.0, sparse_delay_ms=200.0)

    last_serial_ms = 0.0
    last_parallel_ms = 0.0
    for _attempt in range(3):
        t0 = time.monotonic()
        e.encode_query_dense_sparse("x", parallel=False)
        last_serial_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        e.encode_query_dense_sparse("x", parallel=True)
        last_parallel_ms = (time.monotonic() - t0) * 1000

        # Serial should be ~400 ms; parallel should be ~200 ms.
        # Verify both: serial is in the right ballpark (stub not
        # broken) AND parallel saved at least 100 ms of wall-clock.
        if last_serial_ms >= 380.0 and (last_serial_ms - last_parallel_ms) >= 100.0:
            return

    # All retries failed -- the parallelism path is not overlapping.
    pytest.fail(
        f"Parallel did not save enough wall-clock vs serial after 3 "
        f"attempts: last serial={last_serial_ms:.0f} ms, "
        f"last parallel={last_parallel_ms:.0f} ms. Either the "
        f"ThreadPoolExecutor path is broken or the host is under "
        f"extreme CPU contention."
    )


def test_encode_parallel_with_exception_propagates():
    """If one of the encoders raises, the exception surfaces from
    the parallel path -- callers shouldn't get silent failure +
    None vectors."""
    e = HybridEmbedder.__new__(HybridEmbedder)
    e._dense = object()
    e._sparse = object()
    e._lock = threading.Lock()
    e.dense_model_name = "stub-dense"
    e.sparse_model_name = "stub-sparse"

    def fake_dense(_):
        raise RuntimeError("dense encoder is on fire")

    def fake_sparse(_):
        return _SparseVec([1], [1.0])

    e.encode_query_dense = fake_dense  # type: ignore[assignment]
    e.encode_query_sparse = fake_sparse  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="dense encoder is on fire"):
        e.encode_query_dense_sparse("x", parallel=True)


def test_encode_serial_with_exception_propagates():
    """Same exception contract on the serial path."""
    e = HybridEmbedder.__new__(HybridEmbedder)
    e._dense = object()
    e._sparse = object()
    e._lock = threading.Lock()
    e.dense_model_name = "stub-dense"
    e.sparse_model_name = "stub-sparse"

    def fake_dense(_):
        raise ValueError("dense issue")

    def fake_sparse(_):
        return _SparseVec([1], [1.0])

    e.encode_query_dense = fake_dense  # type: ignore[assignment]
    e.encode_query_sparse = fake_sparse  # type: ignore[assignment]

    with pytest.raises(ValueError, match="dense issue"):
        e.encode_query_dense_sparse("x", parallel=False)


def test_encode_dense_sparse_default_is_serial():
    """The ``parallel`` keyword defaults to False for byte-identical
    behaviour with the legacy two-call sequence. Explicit opt-in is
    required to engage the ThreadPoolExecutor path."""
    import inspect
    sig = inspect.signature(HybridEmbedder.encode_query_dense_sparse)
    parallel_param = sig.parameters["parallel"]
    assert parallel_param.default is False


def test_encode_dense_sparse_empty_string():
    """Empty string still passes through both encoders -- caller
    decides whether to validate."""
    e = _build_stubbed_embedder()
    dense, sparse = e.encode_query_dense_sparse("")
    assert isinstance(dense, np.ndarray)
    assert isinstance(sparse, _SparseVec)
