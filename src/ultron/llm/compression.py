"""Context compression for RAG / web / history blocks.

Per LLMLingua (Jiang et al., arXiv 2310.05736) and its lighter
follow-ups, token-level compression of high-redundancy text before
LLM injection can free 1.5–5× context without measurable answer
quality loss. The 4B benefits more than the 9B because it has less
attention to spare on filler.

This module ships the **heuristic** version: no extra model, drops
stopwords / redundant punctuation / contractions / repeated paragraph
signatures. The compression is best-effort; actual ratio depends on
input redundancy.

A clean hook is plumbed in for swapping the heuristic out for a real
perplexity-based compressor later — pass a ``perplexity_scorer``
callable into :class:`Compressor` and the dispatcher will use it
instead of the heuristic. The Stage C speculative-decoding 0.8B is
the natural scorer (no additional VRAM cost when speculative is on).

Default OFF (gated by ``llm.compression.enabled``). The voice path
hot loop is byte-for-byte unchanged unless the user opts in. Even
when enabled, compression runs only on the per-block surfaces it's
configured for (compress_rag / compress_web / compress_history) —
the user message and persona are NEVER touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set

from ultron.utils.logging import get_logger

logger = get_logger("llm.compression")


# Stopwords + low-information tokens. Conservative list — words that
# are truly redundant in the *content* of a retrieved snippet without
# changing what the LLM understands. Preserves negations ("not", "no",
# "never") because dropping a negation can flip meaning.
_STOPWORDS: Set[str] = {
    # Articles
    "a", "an", "the",
    # Common copulas (drop only when redundant — not in negations)
    "is", "are", "was", "were", "be", "been", "being", "am",
    # Prepositions that are usually inferable
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "into", "onto", "about", "over", "under",
    # Conjunctions
    "and", "or", "but", "so", "yet", "nor",
    # Pronouns + auxiliaries that rarely carry content in third-person snippets
    "it", "this", "that", "these", "those",
    "do", "does", "did", "has", "have", "had",
    # Filler adverbs
    "very", "really", "quite", "just", "actually", "basically", "literally",
    "essentially", "simply", "perhaps", "maybe",
}


# Contraction expansions / simplifications. Keep the meaning.
_CONTRACTIONS = [
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bcan not\b", "can't"),
    (r"\bcannot\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"),
]


@dataclass
class CompressionResult:
    """Outcome of one compress() call.

    ``compressed`` is the output. ``ratio_in`` and ``ratio_out`` are
    rough word counts; ``actual_ratio`` is ratio_in / ratio_out (so
    >1.0 means we compressed). ``method`` records which scorer ran
    ("heuristic" / "perplexity").
    """

    compressed: str
    ratio_in: int
    ratio_out: int
    method: str

    @property
    def actual_ratio(self) -> float:
        if self.ratio_out == 0:
            return float("inf")
        return self.ratio_in / self.ratio_out


PerplexityScorer = Callable[[List[str]], List[float]]


class Compressor:
    """Heuristic + plug-in perplexity compressor.

    Args:
        target_ratio: desired compression. 1.5 ⇒ drop ~33 %.
        perplexity_scorer: optional callable that takes a list of
            tokens and returns per-token perplexity scores. When
            supplied, the compressor uses scorer-driven dropping
            instead of the heuristic. The Stage C 0.8B model is the
            natural fit; this module doesn't load any model itself.
    """

    def __init__(
        self,
        *,
        target_ratio: float = 1.5,
        perplexity_scorer: Optional[PerplexityScorer] = None,
    ) -> None:
        self._target_ratio = max(1.0, float(target_ratio))
        self._scorer = perplexity_scorer

    def compress(self, text: str) -> CompressionResult:
        """Compress ``text``. Empty / very short input passes through
        unchanged so we don't shred small snippets."""
        if not text or len(text.split()) < 8:
            return CompressionResult(
                compressed=text,
                ratio_in=len(text.split()) if text else 0,
                ratio_out=len(text.split()) if text else 0,
                method="passthrough",
            )

        ratio_in = len(text.split())
        if self._scorer is not None:
            try:
                compressed = self._compress_with_scorer(text)
                method = "perplexity"
            except Exception as e:
                logger.warning(
                    "perplexity scorer failed (%s); falling back to heuristic",
                    e,
                )
                compressed = self._compress_heuristic(text)
                method = "heuristic-fallback"
        else:
            compressed = self._compress_heuristic(text)
            method = "heuristic"

        return CompressionResult(
            compressed=compressed,
            ratio_in=ratio_in,
            ratio_out=len(compressed.split()),
            method=method,
        )

    # --- heuristic path ----------------------------------------------------

    def _compress_heuristic(self, text: str) -> str:
        """Conservative drop-stopwords + redundancy-collapse."""
        # 1. Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # 2. Apply contraction simplifications (preserve "not" first)
        for pattern, replacement in _CONTRACTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # 3. Collapse repeated punctuation (..., !!! -> ., !)
        text = re.sub(r"([.!?,;:])\1+", r"\1", text)
        # 4. Drop stopwords IF target_ratio asks for it. We aim for a
        #    rough fraction so the user's choice of target_ratio
        #    actually affects output size.
        target_drop_fraction = max(0.0, 1.0 - 1.0 / self._target_ratio)
        text = self._drop_stopwords(text, target_drop_fraction)
        # 5. De-duplicate consecutive sentences (common in noisy web text)
        text = self._dedupe_sentences(text)
        # 6. Trim leftover double spaces from drop steps
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _drop_stopwords(self, text: str, target_drop_fraction: float) -> str:
        """Drop stopword tokens up to ``target_drop_fraction`` of total.

        Preserves any token immediately preceded or followed by a
        negation (``not`` / ``no`` / ``never``) so meaning isn't
        flipped.
        """
        tokens = text.split()
        if not tokens:
            return text
        max_drops = int(len(tokens) * target_drop_fraction)
        if max_drops <= 0:
            return text

        keep: List[str] = []
        dropped = 0
        for i, tok in enumerate(tokens):
            base = re.sub(r"[^a-zA-Z']", "", tok).lower()
            if (
                dropped < max_drops
                and base in _STOPWORDS
                and not self._is_negation_neighbour(tokens, i)
            ):
                dropped += 1
                continue
            keep.append(tok)
        return " ".join(keep)

    @staticmethod
    def _is_negation_neighbour(tokens: List[str], i: int) -> bool:
        negations = {"not", "no", "never", "n't"}
        for j in (i - 1, i + 1):
            if 0 <= j < len(tokens):
                base = re.sub(r"[^a-zA-Z']", "", tokens[j]).lower()
                if base in negations or base.endswith("n't"):
                    return True
        return False

    @staticmethod
    def _dedupe_sentences(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        seen: Set[str] = set()
        out: List[str] = []
        for s in sentences:
            sig = s.strip().lower()
            if not sig:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            out.append(s)
        return " ".join(out)

    # --- perplexity path ---------------------------------------------------

    def _compress_with_scorer(self, text: str) -> str:
        """Score-driven token drop.

        Tokens with the LOWEST perplexity (most predictable from
        context) are the safest to drop. We sort by score, drop the
        bottom ``target_drop_fraction``, and stitch the remainder
        back together.

        This is intentionally simple — real LLMLingua does smarter
        budget control + question-aware coarse passes. The hook
        exists so that swap is a future drop-in.
        """
        assert self._scorer is not None
        tokens = text.split()
        if not tokens:
            return text
        scores = self._scorer(tokens)
        if len(scores) != len(tokens):
            logger.warning(
                "scorer returned %d scores for %d tokens; falling back",
                len(scores), len(tokens),
            )
            return self._compress_heuristic(text)
        target_drop = int(len(tokens) * max(0.0, 1.0 - 1.0 / self._target_ratio))
        if target_drop <= 0:
            return text
        # Indices sorted by score ASCENDING — lowest perplexity first.
        sorted_idx = sorted(range(len(tokens)), key=lambda i: scores[i])
        drop_set = set(sorted_idx[:target_drop])
        kept = [t for i, t in enumerate(tokens) if i not in drop_set]
        return " ".join(kept)


# ---------------------------------------------------------------------------
# Convenience: build from config + per-surface compress helpers
# ---------------------------------------------------------------------------


# Imported at module level so test patches against
# ``ultron.llm.compression.get_config`` are stable. The runtime cost is
# zero — ``ultron.config`` is already loaded by the time anything in
# the LLM stack runs.
from ultron.config import get_config  # noqa: E402  — see module docstring


def build_default_compressor(cfg: Any = None) -> Optional[Compressor]:
    """Construct the compressor only when enabled. Returns ``None``
    otherwise so callers can short-circuit."""
    if cfg is None:
        cfg = get_config()
    cc = cfg.llm.compression
    if not cc.enabled:
        return None
    return Compressor(target_ratio=cc.target_ratio)


def maybe_compress(
    text: str,
    *,
    surface: str,
    compressor: Optional[Compressor] = None,
    cfg: Any = None,
) -> str:
    """Compress ``text`` if the per-surface flag for ``surface`` is on.

    ``surface`` is one of: ``"rag"`` / ``"web"`` / ``"history"``.
    Returns the original text untouched when compression is disabled
    globally, when the per-surface flag is off, or when compression
    fails. Never raises.
    """
    if not text:
        return text
    if cfg is None:
        cfg = get_config()
    cc = cfg.llm.compression
    if not cc.enabled:
        return text
    surface_flag = {
        "rag": cc.compress_rag,
        "web": cc.compress_web,
        "history": cc.compress_history,
    }.get(surface, False)
    if not surface_flag:
        return text
    if compressor is None:
        compressor = build_default_compressor(cfg)
    if compressor is None:
        return text
    try:
        result = compressor.compress(text)
        return result.compressed
    except Exception as e:
        logger.warning("compression failed on %s surface: %s", surface, e)
        return text


__all__ = [
    "Compressor",
    "CompressionResult",
    "PerplexityScorer",
    "build_default_compressor",
    "maybe_compress",
]
