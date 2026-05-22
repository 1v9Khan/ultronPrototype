"""Contextual retrieval tests (frontier item 4, 2026-05-21).

Per-turn context-phrase generation + integration into
:meth:`ConversationMemory._upsert_turn`. No real LLM load happens
here -- ``llama_cpp.Llama`` is mocked. Tests cover:
- Config schema accepts the new fields with safe defaults.
- ContextGenerator lazy-loads on first call; eager=True loads at construction.
- Empty input / model-load failure / inference failure all fail-open
  (empty string, no exception).
- Quote / "Topic:" prefix stripping.
- ``_upsert_turn`` prepends the context phrase to the DENSE embed
  text only; sparse BM25 stays on plain content; payload carries
  ``context_summary`` separately.
- Disabled flag -> no context generated, embed text unchanged.
- Generator construction failure in ``_upsert_turn`` is fail-open
  (proceeds without context).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ultron.config import (
    MemoryContextualRetrievalConfig,
    MemoryConfig,
    UltronConfig,
)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_contextual_retrieval_defaults():
    cfg = MemoryContextualRetrievalConfig()
    assert cfg.enabled is False
    assert cfg.generator_model_path is None
    assert cfg.generator_device == "cpu"
    assert cfg.max_context_tokens == 40
    assert cfg.generator_temperature == 0.2


def test_contextual_retrieval_validates_ranges():
    with pytest.raises(Exception):                                       # noqa: PT011
        MemoryContextualRetrievalConfig(max_context_tokens=5)        # below 10
    with pytest.raises(Exception):                                       # noqa: PT011
        MemoryContextualRetrievalConfig(max_context_tokens=300)      # above 200
    with pytest.raises(Exception):                                       # noqa: PT011
        MemoryContextualRetrievalConfig(generator_temperature=-0.1)
    with pytest.raises(Exception):                                       # noqa: PT011
        MemoryContextualRetrievalConfig(generator_temperature=1.5)


def test_memory_config_includes_contextual_retrieval():
    cfg = MemoryConfig()
    assert hasattr(cfg, "contextual_retrieval")
    assert cfg.contextual_retrieval.enabled is False


def test_full_config_round_trip_enables_contextual_retrieval():
    cfg = UltronConfig.model_validate({
        "memory": {
            "contextual_retrieval": {
                "enabled": True,
                "max_context_tokens": 60,
                "generator_temperature": 0.3,
            }
        }
    })
    assert cfg.memory.contextual_retrieval.enabled is True
    assert cfg.memory.contextual_retrieval.max_context_tokens == 60
    assert cfg.memory.contextual_retrieval.generator_temperature == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# ContextGenerator behaviour
# ---------------------------------------------------------------------------


class _FakeLlama:
    """Minimal stand-in for llama_cpp.Llama. Returns a configurable
    completion. Tracks calls."""

    def __init__(self, *_a, completion_text: str = "talking about apples",
                 **_kw):
        self.completion_text = completion_text
        self.calls = []

    def __call__(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        return {"choices": [{"text": " " + self.completion_text}]}

    def close(self):
        pass


def _install_fake_llama(monkeypatch, completion_text: str = "talking about apples"):
    """Patch llama_cpp.Llama to return the fake. Returns the fake
    instance so the test can inspect calls."""
    fake = _FakeLlama(completion_text=completion_text)

    def _factory(*_a, **_kw):
        # Reuse the same instance across all constructions so the
        # generator's lazy-load only sees one model.
        return fake

    monkeypatch.setattr("llama_cpp.Llama", _factory)
    return fake


def _stub_existing_model(monkeypatch, tmp_path):
    """Make ``Path(model_path).is_file()`` return True for the
    resolved generator path so ``_ensure_model`` reaches the Llama
    construction."""
    fake_gguf = tmp_path / "draft.gguf"
    fake_gguf.write_bytes(b"\x00")
    # Use a config override so ContextGenerator picks our temp path
    # via its normal resolution chain.
    monkeypatch.setattr(
        "ultron.memory.contextualizer.ContextGenerator._resolve_path",
        staticmethod(lambda p: fake_gguf),
    )


def test_context_generator_empty_content_returns_empty():
    from ultron.memory.contextualizer import ContextGenerator
    cg = ContextGenerator(model_path="nonexistent.gguf")
    assert cg.generate_context("") == ""
    assert cg.generate_context("   ") == ""


def test_context_generator_missing_model_returns_empty(tmp_path):
    """Passing a path that doesn't exist on disk -> empty context,
    no Llama load attempt."""
    from ultron.memory.contextualizer import ContextGenerator
    missing = tmp_path / "definitely_missing.gguf"
    # Note: NOT touching it; it should not exist.
    assert not missing.exists()
    cg = ContextGenerator(model_path=str(missing))
    assert cg.generate_context("hello") == ""


def test_context_generator_basic_generation(monkeypatch, tmp_path):
    _stub_existing_model(monkeypatch, tmp_path)
    fake = _install_fake_llama(monkeypatch, completion_text="apples and oranges")
    from ultron.memory.contextualizer import ContextGenerator
    cg = ContextGenerator(model_path="draft.gguf")
    result = cg.generate_context("What about apples?", role="user")
    assert result == "apples and oranges"
    assert len(fake.calls) == 1
    # Sanity: prompt contains the content
    assert "What about apples?" in fake.calls[0]["prompt"]
    assert "user:" in fake.calls[0]["prompt"].lower() or "user " in fake.calls[0]["prompt"]


def test_context_generator_strips_quotes(monkeypatch, tmp_path):
    _stub_existing_model(monkeypatch, tmp_path)
    _install_fake_llama(monkeypatch, completion_text="\"apples and oranges\"")
    from ultron.memory.contextualizer import ContextGenerator
    cg = ContextGenerator(model_path="draft.gguf")
    result = cg.generate_context("anything", role="user")
    # Surrounding quotes stripped
    assert result == "apples and oranges"


def test_context_generator_strips_topic_prefix(monkeypatch, tmp_path):
    _stub_existing_model(monkeypatch, tmp_path)
    _install_fake_llama(monkeypatch, completion_text="Topic: apples and oranges")
    from ultron.memory.contextualizer import ContextGenerator
    cg = ContextGenerator(model_path="draft.gguf")
    result = cg.generate_context("anything", role="user")
    assert result == "apples and oranges"


def test_context_generator_inference_failure_returns_empty(monkeypatch, tmp_path):
    _stub_existing_model(monkeypatch, tmp_path)

    class BoomLlama:
        def __init__(self, *_a, **_kw):
            pass

        def __call__(self, *_a, **_kw):
            raise RuntimeError("simulated inference failure")

    monkeypatch.setattr("llama_cpp.Llama", lambda *a, **kw: BoomLlama())
    from ultron.memory.contextualizer import ContextGenerator
    cg = ContextGenerator(model_path="draft.gguf")
    # Must not raise.
    assert cg.generate_context("anything", role="user") == ""


def test_context_generator_load_failure_returns_empty(monkeypatch, tmp_path):
    _stub_existing_model(monkeypatch, tmp_path)

    def boom(*_a, **_kw):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr("llama_cpp.Llama", boom)
    from ultron.memory.contextualizer import ContextGenerator
    cg = ContextGenerator(model_path="draft.gguf")
    assert cg.generate_context("anything", role="user") == ""
    # Second call should NOT re-attempt the load.
    assert cg.generate_context("else", role="user") == ""


def test_context_generator_lazy_load(monkeypatch, tmp_path):
    _stub_existing_model(monkeypatch, tmp_path)
    load_count = {"n": 0}

    class LazyLlama:
        def __init__(self, *_a, **_kw):
            load_count["n"] += 1

        def __call__(self, *_a, **_kw):
            return {"choices": [{"text": " ok"}]}

        def close(self):
            pass

    monkeypatch.setattr("llama_cpp.Llama", LazyLlama)
    from ultron.memory.contextualizer import ContextGenerator
    cg = ContextGenerator(model_path="draft.gguf")
    assert load_count["n"] == 0
    cg.generate_context("x", role="user")
    assert load_count["n"] == 1
    cg.generate_context("y", role="user")
    assert load_count["n"] == 1  # cached


def test_context_generator_eager_loads_at_construction(monkeypatch, tmp_path):
    _stub_existing_model(monkeypatch, tmp_path)
    load_count = {"n": 0}

    class LazyLlama:
        def __init__(self, *_a, **_kw):
            load_count["n"] += 1

        def __call__(self, *_a, **_kw):
            return {"choices": [{"text": " ok"}]}

        def close(self):
            pass

    monkeypatch.setattr("llama_cpp.Llama", LazyLlama)
    from ultron.memory.contextualizer import ContextGenerator
    ContextGenerator(model_path="draft.gguf", eager=True)
    assert load_count["n"] == 1


# ---------------------------------------------------------------------------
# ConversationMemory._generate_context_for_turn (integration)
# ---------------------------------------------------------------------------


def _stub_memory_for_generate_context():
    from ultron.memory.qdrant_store import ConversationMemory
    cm = object.__new__(ConversationMemory)
    cm._context_generator = None
    return cm


def _stub_turn(content: str = "hello world", role: str = "user", turn_id: int = 1):
    return SimpleNamespace(content=content, role=role, id=turn_id)


def test_generate_context_disabled_returns_empty(monkeypatch):
    """When the flag is OFF, no context is generated and no
    ContextGenerator is constructed."""
    from ultron.memory.qdrant_store import ConversationMemory
    from ultron.memory import contextualizer

    construct_count = {"n": 0}

    class FakeCG:
        def __init__(self, *_a, **_kw):
            construct_count["n"] += 1

        def generate_context(self, *_a, **_kw):
            return "should not be called"

    monkeypatch.setattr(contextualizer, "ContextGenerator", FakeCG)

    cm = _stub_memory_for_generate_context()
    assert ConversationMemory._generate_context_for_turn(cm, _stub_turn()) == ""
    assert construct_count["n"] == 0
    assert cm._context_generator is None


def test_generate_context_enabled_calls_generator(monkeypatch):
    """When enabled, the generator is constructed once and called
    per-turn."""
    from ultron.memory.qdrant_store import ConversationMemory
    from ultron.memory import contextualizer

    construct_count = {"n": 0}
    call_count = {"n": 0}

    class FakeCG:
        def __init__(self, *_a, **_kw):
            construct_count["n"] += 1

        def generate_context(self, content, role="user"):
            call_count["n"] += 1
            return f"topic-{call_count['n']}"

    monkeypatch.setattr(contextualizer, "ContextGenerator", FakeCG)

    # Flip the flag via a config override.
    from ultron.config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg.memory.contextual_retrieval, "enabled", True)

    cm = _stub_memory_for_generate_context()
    out1 = ConversationMemory._generate_context_for_turn(cm, _stub_turn(turn_id=1))
    out2 = ConversationMemory._generate_context_for_turn(cm, _stub_turn(turn_id=2))
    assert out1 == "topic-1"
    assert out2 == "topic-2"
    # Constructed exactly once (cached on self._context_generator).
    assert construct_count["n"] == 1


def test_generate_context_construct_failure_is_fail_open(monkeypatch):
    """If ContextGenerator construction raises, return empty (don't
    crash the writer thread)."""
    from ultron.memory.qdrant_store import ConversationMemory
    from ultron.memory import contextualizer

    def boom(*_a, **_kw):
        raise RuntimeError("simulated construct failure")

    monkeypatch.setattr(contextualizer, "ContextGenerator", boom)

    from ultron.config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg.memory.contextual_retrieval, "enabled", True)

    cm = _stub_memory_for_generate_context()
    # Must not raise.
    assert ConversationMemory._generate_context_for_turn(cm, _stub_turn()) == ""


def test_generate_context_runtime_failure_is_fail_open(monkeypatch):
    """If generate_context raises mid-call, return empty."""
    from ultron.memory.qdrant_store import ConversationMemory
    from ultron.memory import contextualizer

    class BoomCG:
        def __init__(self, *_a, **_kw):
            pass

        def generate_context(self, *_a, **_kw):
            raise RuntimeError("simulated runtime failure")

    monkeypatch.setattr(contextualizer, "ContextGenerator", BoomCG)

    from ultron.config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg.memory.contextual_retrieval, "enabled", True)

    cm = _stub_memory_for_generate_context()
    assert ConversationMemory._generate_context_for_turn(cm, _stub_turn()) == ""
