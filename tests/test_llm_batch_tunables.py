"""Schema + plumbing tests for the LLM ``n_batch`` / ``n_ubatch`` config knobs.

2026-05-15 latency: added ``n_batch`` and ``n_ubatch`` to ``LLMConfig``
so operators can tune llama.cpp's prefill batching. Defaults are
``None`` -- inherit llama.cpp's per-version defaults so unknown
hardware isn't regressed. Setting them passes through to
``Llama(n_batch=..., n_ubatch=...)`` in ``_build_llama``.

These tests verify the schema accepts the knobs, that the defaults
are None, that out-of-range values are rejected, and that the
construction layer threads them through to the underlying Llama
constructor.

No real GGUF load happens here -- ``Llama`` is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ultron.config import LLMConfig, UltronConfig


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_n_batch_default_is_none():
    cfg = LLMConfig(model_path="x.gguf", preset="custom")
    assert cfg.n_batch is None


def test_n_ubatch_default_is_none():
    cfg = LLMConfig(model_path="x.gguf", preset="custom")
    assert cfg.n_ubatch is None


def test_n_batch_accepts_explicit_value():
    cfg = LLMConfig(model_path="x.gguf", preset="custom", n_batch=2048)
    assert cfg.n_batch == 2048


def test_n_ubatch_accepts_explicit_value():
    cfg = LLMConfig(model_path="x.gguf", preset="custom", n_ubatch=256)
    assert cfg.n_ubatch == 256


def test_n_batch_rejects_zero():
    with pytest.raises(Exception):
        LLMConfig(model_path="x.gguf", preset="custom", n_batch=0)


def test_n_ubatch_rejects_negative():
    with pytest.raises(Exception):
        LLMConfig(model_path="x.gguf", preset="custom", n_ubatch=-1)


def test_n_batch_rejects_huge_value():
    with pytest.raises(Exception):
        LLMConfig(model_path="x.gguf", preset="custom", n_batch=100000)


def test_n_ubatch_can_exceed_n_batch_in_schema():
    """The schema doesn't enforce n_ubatch <= n_batch -- that's
    llama.cpp's contract. We accept any in-range pair and let
    llama.cpp reject at load time if invalid."""
    cfg = LLMConfig(
        model_path="x.gguf", preset="custom",
        n_batch=128, n_ubatch=1024,
    )
    assert cfg.n_batch == 128
    assert cfg.n_ubatch == 1024


# ---------------------------------------------------------------------------
# _build_llama wiring
# ---------------------------------------------------------------------------


def _stub_engine_for_build_llama() -> object:
    """Build a partial LLMEngine skeleton just for _build_llama call."""
    from ultron.llm.inference import LLMEngine
    eng = object.__new__(LLMEngine)
    eng._memory = None
    return eng


def _stub_cfg(*, n_batch=None, n_ubatch=None) -> object:
    """Build a minimal cfg-like object for _build_llama."""
    return SimpleNamespace(
        flash_attn=True,
        kv_cache_type=8,
        gpu_layers=-1,
        n_ctx=8192,
        model_path="dummy.gguf",
        n_batch=n_batch,
        n_ubatch=n_ubatch,
    )


def test_build_llama_omits_batch_kwargs_when_none(tmp_path, monkeypatch):
    """Default config (n_batch=None) MUST NOT pass n_batch to Llama --
    that would override llama.cpp's per-version default. This is the
    safety contract for unknown hardware."""
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"dummy")
    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()
    monkeypatch.setattr("llama_cpp.Llama", mock_llama_cls)

    eng = _stub_engine_for_build_llama()
    cfg = _stub_cfg()
    eng._build_llama(cfg, gguf, 4096, -1)

    kwargs = mock_llama_cls.call_args.kwargs
    assert "n_batch" not in kwargs
    assert "n_ubatch" not in kwargs


def test_build_llama_passes_n_batch_when_set(tmp_path, monkeypatch):
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"dummy")
    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()
    monkeypatch.setattr("llama_cpp.Llama", mock_llama_cls)

    eng = _stub_engine_for_build_llama()
    cfg = _stub_cfg(n_batch=2048)
    eng._build_llama(cfg, gguf, 4096, -1)

    kwargs = mock_llama_cls.call_args.kwargs
    assert kwargs["n_batch"] == 2048
    assert "n_ubatch" not in kwargs


def test_build_llama_passes_n_ubatch_when_set(tmp_path, monkeypatch):
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"dummy")
    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()
    monkeypatch.setattr("llama_cpp.Llama", mock_llama_cls)

    eng = _stub_engine_for_build_llama()
    cfg = _stub_cfg(n_ubatch=256)
    eng._build_llama(cfg, gguf, 4096, -1)

    kwargs = mock_llama_cls.call_args.kwargs
    assert kwargs["n_ubatch"] == 256
    assert "n_batch" not in kwargs


def test_build_llama_passes_both_when_set(tmp_path, monkeypatch):
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"dummy")
    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()
    monkeypatch.setattr("llama_cpp.Llama", mock_llama_cls)

    eng = _stub_engine_for_build_llama()
    cfg = _stub_cfg(n_batch=1024, n_ubatch=128)
    eng._build_llama(cfg, gguf, 4096, -1)

    kwargs = mock_llama_cls.call_args.kwargs
    assert kwargs["n_batch"] == 1024
    assert kwargs["n_ubatch"] == 128


# ---------------------------------------------------------------------------
# Top-level UltronConfig round-trip
# ---------------------------------------------------------------------------


def test_full_config_default_keeps_batch_knobs_none():
    cfg = UltronConfig()
    assert cfg.llm.n_batch is None
    assert cfg.llm.n_ubatch is None


def test_full_config_accepts_batch_knobs():
    cfg = UltronConfig.model_validate({
        "llm": {"preset": "custom", "model_path": "x.gguf",
                "n_batch": 2048, "n_ubatch": 256},
    })
    assert cfg.llm.n_batch == 2048
    assert cfg.llm.n_ubatch == 256
