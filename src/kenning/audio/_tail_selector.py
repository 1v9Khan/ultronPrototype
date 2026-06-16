"""Semantic flavor-tail selection -- the embeddinggemma sidecar promoted to a
fine-SELECTOR that picks the tail best fitting THIS exact callout from an
already-correct (agent, situation) candidate set, with MMR diversity to avoid
repetition across a round.

Contract (board 2026-06-16): strictly ADDITIVE + FAIL-OPEN. ``select_tail``
returns a chosen tail TEXT, or None for ANY reason (sidecar down / latched /
empty / low-confidence / exception) -- in which case the caller falls back to the
deterministic anti-repeat picker (``_pick_flavor``). The coarse keyed route +
tag filter already produced a CORRECT cell; this only re-ranks within it, so it
can never change the character or the situation. numpy is in-process-legal (a
faster-whisper transitive dep; the firewall blocks only torch/transformers); the
only network is the existing loopback sidecar client.
"""
from __future__ import annotations

import os
from collections import deque
from typing import Optional, Sequence

try:
    import numpy as _np
except Exception:                                                # noqa: BLE001
    _np = None

#: doc-matrix cache keyed by the candidate tuple (tails are deterministic per
#: model, so a matrix is valid for the whole session even across sidecar restarts).
_DOC_CACHE: dict = {}
#: rolling window of recently-chosen tail vectors for MMR semantic anti-repeat.
_RECENT: deque = deque(maxlen=16)
#: abstain floor per pool kind -- below this top cosine, trust the LRU picker.
_THRESHOLD = {"agent": 0.30, "multi": 0.26, "generic": 0.20}


def _backend():
    try:
        from kenning.audio.command_router import get_embedding_backend
        return get_embedding_backend()
    except Exception:                                            # noqa: BLE001
        return None


def _query_for(agent: Optional[str], situation: Optional[str],
               active_tags: "frozenset[str]") -> str:
    """A short structured context sentence describing the callout, so the query
    embedding captures the whole scenario (agent + situation + loc/dmg/ability)."""
    parts: list[str] = []
    if agent:
        parts.append(agent)
    parts.append((situation or "spotted").replace("_", " "))
    for t in sorted(active_tags):
        parts.append(t.split(":", 1)[-1].replace("_", " "))
    return " ".join(parts)[:120]


def reset_recent() -> None:
    _RECENT.clear()


def select_tail(cands: Sequence[str], recent_lines: Optional[Sequence[str]] = None,
                *, agent: Optional[str] = None, situation: Optional[str] = None,
                active_tags: "frozenset[str]" = frozenset(),
                pool_kind: str = "agent") -> Optional[str]:
    """Best-fit tail text for the callout, or None -> caller uses _pick_flavor.

    Relevance (cosine to a structured query) + MMR semantic diversity + a HARD
    mask of tails already spoken this round (recent_lines), so a custom-fit tail
    is chosen WITHOUT repeating across a round."""
    try:
        if _np is None or not cands or len(cands) < 2:
            return None
        # LATENCY-FIRST DEFAULT (user directive 2026-06-16): the runtime tail embed
        # is OFF unless explicitly enabled. The DETERMINISTIC hierarchy (ult-lift +
        # verb->ability + side + situation routing) already selects the correct
        # curated cell from context with ZERO embeds; the LRU picker rotates within
        # it. The semantic re-ranker is opt-in (KENNING_ENABLE_TAIL_SELECTOR) for
        # A/B testing or once tail embeddings are precomputed offline (no live call).
        if not os.environ.get("KENNING_ENABLE_TAIL_SELECTOR"):
            return None
        be = _backend()
        if be is None:
            return None
        avail = getattr(be, "available", None)
        if callable(avail) and not avail():
            return None
        key = tuple(cands)
        mat = _DOC_CACHE.get(key)
        if mat is None:
            mat = be.prepare(list(cands))            # kind=document, generous timeout
            if mat is None or getattr(mat, "size", 0) == 0 or mat.shape[0] != len(cands):
                return None
            _DOC_CACHE[key] = mat
        q = be._embed([_query_for(agent, situation, active_tags)], kind="query")
        if q is None or q.size == 0 or q.shape[1] != mat.shape[1]:
            return None
        sims = mat @ q[0]
        if float(_np.max(sims)) < _THRESHOLD.get(pool_kind, 0.22):
            return None                              # low confidence -> LRU fallback
        lam = 0.85 if len(cands) >= 10 else 0.65
        score = _np.array(sims, dtype=_np.float32)
        if _RECENT:                                  # soft semantic diversity (MMR)
            pen = _np.zeros(len(cands), dtype=_np.float32)
            for r in _RECENT:
                if getattr(r, "shape", (0,))[0] == mat.shape[1]:
                    pen = _np.maximum(pen, mat @ r)
            score = lam * sims - (1.0 - lam) * pen
        if recent_lines:                             # HARD anti-repeat this round
            rl = "  ".join(recent_lines)
            mask = _np.array([c in rl for c in cands], dtype=bool)
            if mask.any() and not mask.all():
                score = score.copy()
                score[mask] = -1e9
        idx = int(_np.argmax(score))
        _RECENT.append(mat[idx])
        return cands[idx]
    except Exception:                                            # noqa: BLE001
        return None
