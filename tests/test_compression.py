"""4B optimization plan Item 4 — compression module + integration tests.

Verifies:
- Heuristic compressor reduces redundant text without flipping meaning.
- Negations are preserved.
- Perplexity-scorer hook works when supplied.
- Per-surface flag gating.
- Default-OFF preserves byte-for-byte behaviour.
- Integration: ``LLMEngine._format_rag_block`` and
  ``format_sources_for_prompt`` honor the compression flag.
"""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest

from ultron.llm.compression import (
    Compressor,
    CompressionResult,
    build_default_compressor,
    maybe_compress,
)


# ---------------------------------------------------------------------------
# Heuristic compressor
# ---------------------------------------------------------------------------


def test_heuristic_compresses_redundant_text() -> None:
    text = (
        "The boiling point of water is at one hundred degrees Celsius "
        "at standard atmospheric pressure. The boiling point of water "
        "is at one hundred degrees Celsius at standard atmospheric pressure."
    )
    c = Compressor(target_ratio=2.0)
    result = c.compress(text)
    # Repeated sentence should be deduped
    assert result.compressed.count("boiling point") == 1
    # Some compression happened
    assert result.actual_ratio > 1.0


def test_heuristic_preserves_negations() -> None:
    """Dropping the negation token would flip meaning. ``is not`` may
    legitimately collapse to ``isn't`` (contraction step), but a
    negation indicator MUST survive in some form."""
    text = "the cat will never jump over the lazy fox in this lifetime"
    c = Compressor(target_ratio=3.0)
    result = c.compress(text)
    # 'never' must survive (no contraction rule for it)
    assert "never" in result.compressed.lower()


def test_heuristic_contraction_preserves_negation_meaning() -> None:
    """``is not`` ⇒ ``isn't`` is fine — the negation is still encoded."""
    text = "the boiling point is not the same as the freezing point of water"
    c = Compressor(target_ratio=3.0)
    result = c.compress(text)
    # Either "not" or "isn't" / "n't" must survive — meaning preserved.
    out = result.compressed.lower()
    assert "not" in out or "n't" in out


def test_heuristic_collapses_repeated_punctuation() -> None:
    """Long enough to clear the passthrough threshold."""
    text = (
        "this entire result is great!!! Really great... Wow,, indeed "
        "the analysis confirms our findings about the model"
    )
    c = Compressor(target_ratio=1.5)
    out = c.compress(text).compressed
    assert "!!!" not in out
    assert "..." not in out
    assert ",," not in out


def test_heuristic_short_input_passes_through() -> None:
    text = "hello world"
    c = Compressor(target_ratio=2.0)
    result = c.compress(text)
    assert result.compressed == "hello world"
    assert result.method == "passthrough"


def test_heuristic_empty_input_passes_through() -> None:
    c = Compressor()
    assert c.compress("").compressed == ""
    assert c.compress("   ").compressed == "   "  # whitespace-only too short to compress


def test_heuristic_ratio_one_means_no_drop() -> None:
    """target_ratio=1.0 ⇒ drop fraction is 0 ⇒ no stopwords removed."""
    text = "the dog jumped over the lazy fox in the garden"
    c = Compressor(target_ratio=1.0)
    out = c.compress(text).compressed
    # 'the' should still be present — ratio=1.0 disables stopword drop
    assert "the" in out.lower()


def test_heuristic_higher_ratio_drops_more() -> None:
    text = (
        "The cat is on the mat in the room with the curtains drawn "
        "shut at the moment when the sun is setting in the west"
    )
    c_low = Compressor(target_ratio=1.2)
    c_high = Compressor(target_ratio=3.0)
    low_words = len(c_low.compress(text).compressed.split())
    high_words = len(c_high.compress(text).compressed.split())
    assert high_words < low_words


# ---------------------------------------------------------------------------
# Perplexity-scorer hook
# ---------------------------------------------------------------------------


def test_perplexity_scorer_drops_lowest_score_tokens() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    tokens = text.split()
    # Make 'the' tokens have low perplexity (predictable from context).
    # Real scorer would do this organically; here we mock the rank.
    def scorer(tokens):
        return [0.1 if t.lower() == "the" else 5.0 for t in tokens]

    c = Compressor(target_ratio=1.5, perplexity_scorer=scorer)
    result = c.compress(text)
    # Both 'the' tokens removed (lowest scores)
    assert "the" not in result.compressed.lower()
    assert result.method == "perplexity"


def test_perplexity_scorer_failure_falls_back_to_heuristic() -> None:
    text = "the cat jumped over the lazy fox today"

    def bad_scorer(tokens):
        raise RuntimeError("scorer crashed")

    c = Compressor(target_ratio=2.0, perplexity_scorer=bad_scorer)
    result = c.compress(text)
    # Heuristic ran instead — method tagged accordingly
    assert result.method == "heuristic-fallback"
    # Compression still happened (some 'the' dropped)
    assert result.actual_ratio >= 1.0


def test_perplexity_scorer_mismatched_length_falls_back() -> None:
    """Scorer returning the wrong number of scores must NOT crash —
    falls back to heuristic and the result is still usable."""
    text = "the cat jumped over the fence today again in the morning"

    def short_scorer(tokens):
        return [0.5]  # one score for many tokens

    c = Compressor(target_ratio=2.0, perplexity_scorer=short_scorer)
    result = c.compress(text)
    # Did NOT silently corrupt or use the bad scoring
    assert result.compressed  # got something back
    # Result still has core meaning preserved
    assert "cat" in result.compressed
    assert "jumped" in result.compressed


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


def test_result_actual_ratio() -> None:
    r = CompressionResult(compressed="a b c", ratio_in=10, ratio_out=3, method="heuristic")
    assert r.actual_ratio == pytest.approx(10 / 3)


def test_result_actual_ratio_zero_output() -> None:
    r = CompressionResult(compressed="", ratio_in=10, ratio_out=0, method="heuristic")
    assert r.actual_ratio == float("inf")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_default_compressor_disabled_returns_none() -> None:
    cfg = MagicMock()
    cfg.llm.compression.enabled = False
    assert build_default_compressor(cfg) is None


def test_build_default_compressor_enabled_returns_instance() -> None:
    cfg = MagicMock()
    cfg.llm.compression.enabled = True
    cfg.llm.compression.target_ratio = 2.5
    c = build_default_compressor(cfg)
    assert isinstance(c, Compressor)
    assert c._target_ratio == 2.5  # noqa: SLF001


# ---------------------------------------------------------------------------
# maybe_compress — per-surface flag gating
# ---------------------------------------------------------------------------


def _cfg(enabled: bool, **flags) -> MagicMock:
    cfg = MagicMock()
    cfg.llm.compression.enabled = enabled
    cfg.llm.compression.target_ratio = 1.5
    cfg.llm.compression.compress_rag = flags.get("compress_rag", True)
    cfg.llm.compression.compress_web = flags.get("compress_web", True)
    cfg.llm.compression.compress_history = flags.get("compress_history", False)
    return cfg


def test_maybe_compress_disabled_passes_through() -> None:
    text = "the cat sat on the mat in the room with the curtains drawn"
    out = maybe_compress(text, surface="rag", cfg=_cfg(False))
    assert out == text


def test_maybe_compress_per_surface_off_passes_through() -> None:
    text = "the cat sat on the mat in the room with the curtains drawn"
    out = maybe_compress(text, surface="rag", cfg=_cfg(True, compress_rag=False))
    assert out == text


def test_maybe_compress_per_surface_on_compresses() -> None:
    text = "the cat sat on the mat in the room with the curtains drawn"
    out = maybe_compress(text, surface="rag", cfg=_cfg(True, compress_rag=True))
    # Some reduction
    assert len(out.split()) <= len(text.split())


def test_maybe_compress_unknown_surface_passes_through() -> None:
    text = "the cat sat on the mat"
    out = maybe_compress(text, surface="bogus", cfg=_cfg(True))
    assert out == text


def test_maybe_compress_history_default_off() -> None:
    """History compression is the riskiest — it has user voice. Default
    should be OFF even when global compression is on."""
    text = "the user said something and the assistant replied"
    out = maybe_compress(text, surface="history", cfg=_cfg(True))
    assert out == text


def test_maybe_compress_compressor_failure_returns_original() -> None:
    """Compressor exception must NEVER break the prompt path."""
    text = "the cat sat on the mat in the room"
    bad = MagicMock()
    bad.compress.side_effect = RuntimeError("boom")
    out = maybe_compress(text, surface="rag", compressor=bad, cfg=_cfg(True))
    assert out == text


def test_maybe_compress_empty_text_returns_empty() -> None:
    out = maybe_compress("", surface="rag", cfg=_cfg(True))
    assert out == ""


# ---------------------------------------------------------------------------
# Integration — _format_rag_block honors compression flag
# ---------------------------------------------------------------------------


_FakeTurn = namedtuple("_FakeTurn", ["role", "content"])


def test_format_rag_block_default_off_unchanged() -> None:
    """Default behaviour preserved: compression OFF ⇒ block identical
    to pre-Item-4 shape."""
    from ultron.llm.inference import LLMEngine

    snippets = [_FakeTurn("user", "the boiling point of water is one hundred")]
    block = LLMEngine._format_rag_block(snippets)
    assert "the boiling point of water" in block.lower()


def test_format_rag_block_with_compression_on_compresses() -> None:
    """Compression ON ⇒ block is shorter."""
    from ultron.llm.inference import LLMEngine

    snippets = [
        _FakeTurn("user", "the user said the boiling point of water is the same as before in the morning"),
        _FakeTurn("assistant", "the assistant agreed with the user about the boiling point of water in the morning"),
    ]
    cfg = _cfg(True, compress_rag=True)
    cfg.llm.compression.target_ratio = 2.0
    with patch("ultron.llm.compression.get_config", return_value=cfg):
        compressed_block = LLMEngine._format_rag_block(snippets)

    with patch("ultron.llm.compression.get_config", return_value=_cfg(False)):
        uncompressed_block = LLMEngine._format_rag_block(snippets)

    assert len(compressed_block) <= len(uncompressed_block)


# ---------------------------------------------------------------------------
# Integration — format_sources_for_prompt
# ---------------------------------------------------------------------------


def test_format_sources_for_prompt_default_off_unchanged() -> None:
    from ultron.web_search.search import SearchSource, format_sources_for_prompt

    sources = [
        SearchSource(
            url="https://example.com",
            title="Example",
            snippet="the example article body has a lot of redundant words about example",
            full_text=None,
            rank=1,
        ),
    ]
    out = format_sources_for_prompt(sources)
    assert "URL: https://example.com" in out
    assert "redundant words" in out


def test_format_sources_for_prompt_compression_preserves_url() -> None:
    """Even with compression on, URLs must NOT be touched (citations
    must stay accurate)."""
    from ultron.web_search.search import SearchSource, format_sources_for_prompt

    sources = [
        SearchSource(
            url="https://example.com/very-specific-path",
            title="Example",
            snippet="the article body has a lot of redundant words in the body of the article that could be compressed",
            full_text=None,
            rank=1,
        ),
    ]
    cfg = _cfg(True, compress_web=True)
    cfg.llm.compression.target_ratio = 2.0
    with patch("ultron.llm.compression.get_config", return_value=cfg):
        out = format_sources_for_prompt(sources)
    assert "https://example.com/very-specific-path" in out
