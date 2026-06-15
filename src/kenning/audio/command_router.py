"""Semantic command router -- the additive fallback layer beneath the existing
exact matchers.

Cascade (in the orchestrator):
    normalize -> EXACT matchers (relay / spotify / identity ...)   [unchanged]
              -> if they ALL miss: this router makes a COARSE family decision
                 (team_callout / spotify / identity / desktop_refuse /
                 conversational) by similarity to curated exemplars
              -> if a DETERMINISTIC family wins confidently: re-dispatch to that
                 family's existing handler (which does its own fine matching +
                 slot extraction -- nothing is bypassed)
              -> otherwise ABSTAIN -> conversational LLM.

The OOS / abstention gate is the load-bearing "zero mistakes" mechanism: a
family is committed to ONLY when its score clears a per-family threshold AND
beats the runner-up by a margin AND it isn't the CONVERSATIONAL anchor. Every
gate is biased toward ABSTAIN (a miss only costs an LLM round-trip; a false
route silently mis-executes). Thresholds are encoder-specific and meant to be
re-tuned on REAL transcripts -- see ``calibrate``.

This module imports no heavy ML: the similarity backend (lexical, or hybrid
with the embedding sidecar) lives in :mod:`kenning.audio._router_backends`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from kenning.audio._command_exemplars import (
    ABSTAIN_FAMILIES,
    DETERMINISTIC_FAMILIES,
    FAMILIES,
)
from kenning.audio._router_backends import SimilarityBackend, get_backend
from kenning.utils.logging import get_logger

logger = get_logger("audio.command_router")

# Conservative, abstain-biased DEFAULTS. These are starting points for the
# hybrid (lexical+embedding) backend; the research is emphatic that the real
# values are encoder-specific and must be tuned on REAL transcripts. Bias: a
# family must be clearly the best (margin) AND clearly relevant (threshold).
_DEFAULT_THRESHOLD = 0.50          # min top-family score to commit
_DEFAULT_MARGIN = 0.06             # min (top - runner_up) to commit
# Optional per-family threshold overrides (identity phrasings are tight + score
# high; callouts are multi-modal + score lower, so a slightly lower floor).
_FAMILY_THRESHOLDS: Dict[str, float] = {
    "identity": 0.55,
    "spotify": 0.50,
    "team_callout": 0.48,
    "desktop_refuse": 0.50,
}


@dataclass
class RoutingDecision:
    """Result of one routing call."""
    family: Optional[str]                 # committed deterministic family, or None
    abstained: bool                       # True => hand to the conversational LLM
    confidence: float                     # top-family score
    margin: float                         # top - runner_up
    reason: str                           # human-readable why
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def routed(self) -> bool:
        return self.family is not None and not self.abstained


class CommandRouter:
    """Coarse family router with an OOS abstention gate."""

    def __init__(
        self,
        backend: SimilarityBackend,
        families: "Dict[str, List[str]]" = FAMILIES,
        *,
        default_threshold: float = _DEFAULT_THRESHOLD,
        margin_delta: float = _DEFAULT_MARGIN,
        family_thresholds: "Optional[Dict[str, float]]" = None,
    ) -> None:
        self.backend = backend
        self.families = list(families.keys())
        self.default_threshold = float(default_threshold)
        self.margin_delta = float(margin_delta)
        self.family_thresholds = dict(_FAMILY_THRESHOLDS)
        if family_thresholds:
            self.family_thresholds.update(family_thresholds)
        # Embed/prepare each family's exemplars ONCE (the cost; route() is cheap).
        self._prepared = {name: backend.prepare(ex) for name, ex in families.items()}
        logger.info(
            "command router ready | backend=%s | families=%s | thr=%.2f margin=%.2f",
            backend.name, self.families, self.default_threshold, self.margin_delta,
        )

    def _threshold(self, family: str) -> float:
        return self.family_thresholds.get(family, self.default_threshold)

    def family_scores(self, text: str) -> Dict[str, float]:
        """Max-aggregated similarity of ``text`` to each family's exemplars."""
        out: Dict[str, float] = {}
        for name in self.families:
            sims = self.backend.score(text, self._prepared[name])
            out[name] = max(sims) if sims else 0.0
        return out

    def route(self, text: str) -> RoutingDecision:
        """Classify ``text`` into a deterministic family or ABSTAIN to the LLM."""
        if not text or not text.strip():
            return RoutingDecision(None, True, 0.0, 0.0, "empty", {})
        scores = self.family_scores(text)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_f, top_s = ranked[0]
        runner_s = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_s - runner_s
        # --- abstention gate (all biased toward LLM fallthrough) ---
        if top_f in ABSTAIN_FAMILIES:
            return RoutingDecision(None, True, top_s, margin,
                                   f"closest to conversational ({top_s:.2f})", scores)
        thr = self._threshold(top_f)
        if top_s < thr:
            return RoutingDecision(None, True, top_s, margin,
                                   f"{top_f} below threshold ({top_s:.2f}<{thr:.2f})", scores)
        if margin < self.margin_delta:
            return RoutingDecision(None, True, top_s, margin,
                                   f"ambiguous: {top_f} margin {margin:.3f}<{self.margin_delta:.3f}", scores)
        if top_f not in DETERMINISTIC_FAMILIES:
            return RoutingDecision(None, True, top_s, margin,
                                   f"{top_f} is not a deterministic handler", scores)
        return RoutingDecision(top_f, False, top_s, margin, "routed", scores)


# ---------------------------------------------------------------------------
# Lazy singleton (built once on first use; backend chosen from config).
# ---------------------------------------------------------------------------
_router: Optional[CommandRouter] = None
_router_lock = threading.Lock()
_router_failed = False


def get_command_router() -> Optional[CommandRouter]:
    """Return the process-wide router, building it on first call. Returns None
    (and never retries) if construction fails, so a router problem can NEVER
    break the voice loop -- the orchestrator simply skips the semantic layer."""
    global _router, _router_failed
    if _router is not None or _router_failed:
        return _router
    with _router_lock:
        if _router is not None or _router_failed:
            return _router
        try:
            from kenning.config import get_config
            rcfg = getattr(get_config(), "semantic_router", None)
            prefer = getattr(rcfg, "backend", "hybrid") if rcfg else "hybrid"
            host = getattr(rcfg, "sidecar_host", "127.0.0.1") if rcfg else "127.0.0.1"
            port = int(getattr(rcfg, "sidecar_port", 8772)) if rcfg else 8772
            emb_w = float(getattr(rcfg, "embedding_weight", 0.6)) if rcfg else 0.6
            # Poll for the sidecar (it loads EmbeddingGemma async at boot) so a
            # COLD boot still gets the embedding backend rather than latching
            # lexical-only. The sidecar is spawned EARLY, so this usually returns
            # after only a couple of seconds at the boot-end warmup.
            wait = float(getattr(rcfg, "sidecar_startup_timeout_seconds", 30.0)) if rcfg else 30.0
            backend = get_backend(prefer, host=host, port=port,
                                  emb_weight=emb_w, wait_seconds=wait)
            _router = CommandRouter(backend)
        except Exception as e:                                    # noqa: BLE001
            logger.warning("semantic command router unavailable (%s); "
                           "exact matchers + LLM fallback unaffected", e)
            _router_failed = True
            _router = None
    return _router


def reset_command_router() -> None:
    """Drop the cached router so the next get_command_router() REBUILDS it.

    Used by the boot respawn-on-lexical retry (re-spawn the sidecar, then rebuild
    the router with embedding) and by test isolation. Takes the singleton lock so
    it can't race a concurrent build."""
    global _router, _router_failed
    with _router_lock:
        _router = None
        _router_failed = False
