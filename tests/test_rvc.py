"""Tests for :class:`kenning.tts.rvc.RvcConverter`.

The converter wraps ``infer-rvc-python`` -- testing it end-to-end would
need GPU + the trained weights. These tests cover the bits that don't
need the model load: constructor's pre-flight file checks, the
``close()`` lifecycle, and the ``convert()`` not-loaded guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kenning.tts.rvc import RvcConverter


def _stub_paths(tmp_path):
    """Construct the kwargs tuple that points every path at tmp_path.

    We pass paths explicitly rather than monkeypatching ``settings``
    because :class:`RvcConverter`'s constructor reads the defaults at
    function-definition time (Python's standard mutable-default gotcha
    — patching ``settings.RVC_MODEL_PATH`` post-import leaves the
    ``__init__`` defaults bound to the original value).
    """
    return dict(
        model_path=tmp_path / "model.pth",
        index_path=tmp_path / "model.index",
        hubert_path=tmp_path / "hubert.pt",
        rmvpe_path=tmp_path / "rmvpe.pt",
        device="cpu",
    )


def test_constructor_rejects_missing_model(tmp_path):
    kwargs = _stub_paths(tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        RvcConverter(**kwargs)
    assert "RVC model not found" in str(exc_info.value)


def test_constructor_rejects_missing_index(tmp_path):
    kwargs = _stub_paths(tmp_path)
    # Drop in the model but leave the index missing.
    (tmp_path / "model.pth").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError) as exc_info:
        RvcConverter(**kwargs)
    assert "RVC index not found" in str(exc_info.value)


def test_close_releases_converter():
    """``close()`` clears ``_converter`` so VRAM can be reclaimed at GC."""
    # Build a barely-valid converter that skips the heavy load path.
    inst = RvcConverter.__new__(RvcConverter)
    inst._converter = object()
    inst.close()
    assert inst._converter is None


def test_close_is_idempotent():
    inst = RvcConverter.__new__(RvcConverter)
    inst._converter = object()
    inst.close()
    inst.close()  # second call must not raise
    assert inst._converter is None


def test_convert_raises_when_not_loaded():
    inst = RvcConverter.__new__(RvcConverter)
    inst._converter = None
    with pytest.raises(RuntimeError) as exc_info:
        inst.convert(np.zeros(16000, dtype=np.int16), 22050)
    assert "not loaded" in str(exc_info.value)


def test_convert_passes_through_empty_audio():
    """An empty buffer short-circuits before the inference call."""
    inst = RvcConverter.__new__(RvcConverter)
    inst._converter = object()  # truthy; doesn't matter what
    empty = np.zeros(0, dtype=np.int16)
    out, sr = inst.convert(empty, 22050)
    assert out.size == 0
    assert sr == 22050


def test_context_manager_releases_on_exit():
    inst = RvcConverter.__new__(RvcConverter)
    inst._converter = object()
    with inst as ctx:
        assert ctx is inst
        assert inst._converter is not None
    assert inst._converter is None
