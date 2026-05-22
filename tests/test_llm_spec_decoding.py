"""Schema + plumbing tests for in-process speculative decoding.

2026-05-21 (Phase 1 frontier-enhancement pass) -- the in-process LLM
path now wires ``LlamaPromptLookupDecoding`` (PLD) when
``draft_model_path`` is non-None, matching the HTTP server's behaviour
at ``llama_cpp/server/model.py:211-215``. Closes the round-8d-surfaced
gap where spec decoding was HTTP-server-only.

Tests cover:
- Config schema accepts the new ``speculative_*`` knobs with safe
  defaults and validated ranges.
- ``_build_llama`` constructs and passes a PLD instance via
  ``draft_model=`` when ``draft_model_path`` is non-None.
- ``_build_llama`` omits ``draft_model`` from kwargs entirely when
  ``draft_model_path`` is None (so we don't accidentally disable
  llama.cpp's own internal defaults).
- The PLD constructor receives the configured tuning values, not
  hard-coded magic numbers.
- Failure to import ``LlamaPromptLookupDecoding`` is fail-open: the
  ``Llama`` constructor is still called without ``draft_model``;
  voice keeps working.

No real GGUF load happens here -- ``Llama`` and the PLD import are
both mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ultron.config import LLMConfig, UltronConfig


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_default_spec_decoding_knobs():
    cfg = LLMConfig()
    # PLD defaults match the HTTP server's
    # ``settings.draft_model_num_pred_tokens`` (10) and the library's
    # ``max_ngram_size`` default (2). Conservative for repetitive
    # prompts; tune up for higher-confidence drafts.
    assert cfg.speculative_max_ngram_size == 2
    assert cfg.speculative_num_pred_tokens == 10


def test_spec_decoding_knobs_validate_ranges():
    # max_ngram_size: [1, 8]
    with pytest.raises(Exception):  # noqa: PT011 -- pydantic ValidationError
        LLMConfig(speculative_max_ngram_size=0)
    with pytest.raises(Exception):  # noqa: PT011
        LLMConfig(speculative_max_ngram_size=9)
    # num_pred_tokens: [1, 64]
    with pytest.raises(Exception):  # noqa: PT011
        LLMConfig(speculative_num_pred_tokens=0)
    with pytest.raises(Exception):  # noqa: PT011
        LLMConfig(speculative_num_pred_tokens=65)


def test_spec_decoding_knobs_accept_in_range_values():
    cfg = LLMConfig(
        speculative_max_ngram_size=4,
        speculative_num_pred_tokens=20,
    )
    assert cfg.speculative_max_ngram_size == 4
    assert cfg.speculative_num_pred_tokens == 20


# ---------------------------------------------------------------------------
# _build_llama wiring (Llama + LlamaPromptLookupDecoding mocked)
# ---------------------------------------------------------------------------


def _stub_engine_for_build_llama() -> object:
    """Build a partial LLMEngine skeleton just for ``_build_llama``."""
    from ultron.llm.inference import LLMEngine
    eng = object.__new__(LLMEngine)
    eng._memory = None
    return eng


def _stub_cfg(
    *,
    draft_model_path=None,
    speculative_max_ngram_size=2,
    speculative_num_pred_tokens=10,
) -> object:
    """Build a minimal cfg-like object for ``_build_llama``."""
    return SimpleNamespace(
        flash_attn=True,
        kv_cache_type=8,
        gpu_layers=-1,
        n_ctx=8192,
        model_path="dummy.gguf",
        n_batch=None,
        n_ubatch=None,
        prefix_cache_ram_bytes=0,
        draft_model_path=draft_model_path,
        speculative_max_ngram_size=speculative_max_ngram_size,
        speculative_num_pred_tokens=speculative_num_pred_tokens,
    )


def test_build_llama_omits_draft_model_when_path_none(tmp_path, monkeypatch):
    """When ``draft_model_path is None``, ``draft_model`` MUST NOT be
    passed to ``Llama``. Otherwise we'd be force-feeding ``None`` to
    a kwarg that llama-cpp-python defaults differently per version."""
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"dummy")
    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()
    monkeypatch.setattr("llama_cpp.Llama", mock_llama_cls)

    eng = _stub_engine_for_build_llama()
    cfg = _stub_cfg(draft_model_path=None)
    eng._build_llama(cfg, gguf, 4096, -1)

    kwargs = mock_llama_cls.call_args.kwargs
    assert "draft_model" not in kwargs


def test_build_llama_wires_pld_when_path_set(tmp_path, monkeypatch):
    """When ``draft_model_path`` is non-None, PLD must be constructed
    and passed to ``Llama`` via ``draft_model=``. The path itself is
    NOT used by PLD (it's N-gram-based against the prompt), but it
    acts as the toggle."""
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"dummy")
    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()
    monkeypatch.setattr("llama_cpp.Llama", mock_llama_cls)

    mock_pld_cls = MagicMock()
    mock_pld_instance = MagicMock()
    mock_pld_cls.return_value = mock_pld_instance
    monkeypatch.setattr(
        "llama_cpp.llama_speculative.LlamaPromptLookupDecoding",
        mock_pld_cls,
    )

    eng = _stub_engine_for_build_llama()
    cfg = _stub_cfg(draft_model_path="models/draft.gguf")
    eng._build_llama(cfg, gguf, 4096, -1)

    # PLD constructed with the configured tuning
    mock_pld_cls.assert_called_once_with(
        max_ngram_size=2,
        num_pred_tokens=10,
    )
    # PLD instance was passed to Llama
    kwargs = mock_llama_cls.call_args.kwargs
    assert kwargs["draft_model"] is mock_pld_instance


def test_build_llama_pld_uses_configured_tuning(tmp_path, monkeypatch):
    """Custom ``speculative_max_ngram_size`` + ``speculative_num_pred_tokens``
    flow through to the PLD constructor."""
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"dummy")
    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()
    monkeypatch.setattr("llama_cpp.Llama", mock_llama_cls)

    mock_pld_cls = MagicMock()
    mock_pld_cls.return_value = MagicMock()
    monkeypatch.setattr(
        "llama_cpp.llama_speculative.LlamaPromptLookupDecoding",
        mock_pld_cls,
    )

    eng = _stub_engine_for_build_llama()
    cfg = _stub_cfg(
        draft_model_path="models/draft.gguf",
        speculative_max_ngram_size=5,
        speculative_num_pred_tokens=20,
    )
    eng._build_llama(cfg, gguf, 4096, -1)

    mock_pld_cls.assert_called_once_with(
        max_ngram_size=5,
        num_pred_tokens=20,
    )


def test_build_llama_pld_import_failure_is_fail_open(tmp_path, monkeypatch):
    """If ``LlamaPromptLookupDecoding`` import fails for any reason
    (hypothetical pinned wheel without the module), the voice path
    must still boot. ``Llama`` is constructed without ``draft_model``;
    a WARN is logged but no exception escapes."""
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"dummy")
    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()
    monkeypatch.setattr("llama_cpp.Llama", mock_llama_cls)

    # Force the PLD import to blow up by removing the symbol.
    import llama_cpp.llama_speculative as speculative_module
    monkeypatch.delattr(
        speculative_module, "LlamaPromptLookupDecoding", raising=False
    )

    eng = _stub_engine_for_build_llama()
    cfg = _stub_cfg(draft_model_path="models/draft.gguf")
    # Must not raise.
    eng._build_llama(cfg, gguf, 4096, -1)

    kwargs = mock_llama_cls.call_args.kwargs
    assert "draft_model" not in kwargs


# ---------------------------------------------------------------------------
# Top-level UltronConfig round-trip
# ---------------------------------------------------------------------------


def test_full_config_default_keeps_spec_decoding_defaults():
    cfg = UltronConfig()
    assert cfg.llm.speculative_max_ngram_size == 2
    assert cfg.llm.speculative_num_pred_tokens == 10


def test_full_config_accepts_spec_decoding_overrides():
    cfg = UltronConfig.model_validate({
        "llm": {
            "speculative_max_ngram_size": 3,
            "speculative_num_pred_tokens": 15,
        }
    })
    assert cfg.llm.speculative_max_ngram_size == 3
    assert cfg.llm.speculative_num_pred_tokens == 15
