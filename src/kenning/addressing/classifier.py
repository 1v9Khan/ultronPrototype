"""Top-level addressing classifier: rules first, zero-shot fallback.

Wired into the orchestrator's WARM-mode (FOLLOW_UP_LISTENING) state to
decide whether each VAD-bounded utterance is a continuation of the
conversation with Kenning or stray speech.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from kenning.addressing.rules import (
    AddressingDecision,
    RuleHit,
    classify as classify_by_rules,
    features as rule_features,
)
from kenning.addressing.zero_shot import ZeroShotAddresseeModel
from kenning.utils.logging import get_logger

logger = get_logger("addressing.classifier")

# ---------------------------------------------------------------------------
# 2026-06-18 FUSION scorer. Combine the graded lexical/structural features
# (rules.features) + the zero-shot model's REAL P(YES) + recency into a single
# calibrated confidence in log-odds space, then threshold cost-asymmetrically.
# Hand-set log-odds weights (signs/magnitudes from the Amazon ICASSP'20 / Apple
# DDSD directionality the research board surfaced); fit by logistic regression
# later from the addressing JSONL the classifier already writes. The flan term
# is consulted ONLY when the lexical logit is in the undecided band, so clear
# commands ("Ultron, show me the stop button") and clear chatter resolve with
# ZERO model latency.
# ---------------------------------------------------------------------------
_ADDR_W = {
    "leading_wake": 3.0,
    "matcher_hit": 2.2,
    "initial_imperative": 1.6,
    "factual_question": 1.0,
    "continuation": 0.8,
    "second_person_q": 0.0,
    "embedded_or_3p_name": -2.2,
    "subj_pronoun_opener": -1.4,
    "particle_opener": -1.2,
    "phone_opener": -2.5,
    "interjection": -2.0,
    "third_party_narrative": -2.2,
    "possessive_q": -2.0,
    "trails_off": -1.0,
}
_ADDR_W_FLAN = 1.0
_ADDR_FLAN_BAND = 3.0   # consult flan only when |lexical logit| < this


def _addr_b0(t: float) -> float:
    """Recency prior in log-odds: +0.4 right after we spoke, decaying to -0.6 by
    >=20 s into the follow-up window (turn-taking strongly predicts addressing)."""
    return 0.4 - min(max(t, 0.0) / 20.0, 1.0) * 1.0


def _addr_tau() -> float:
    """Cost-asymmetric ADDRESSED threshold. Default 0.20 (a false-reject of a real
    command costs ~4x a false-accept). Raise via KENNING_ADDRESSING_TAU during a
    live stream to be more conservative about chatter."""
    try:
        return min(0.95, max(0.01, float(os.getenv("KENNING_ADDRESSING_TAU", "0.20") or 0.20)))
    except Exception:                                                # noqa: BLE001
        return 0.20


@dataclass
class AddressingVerdict:
    """The output of a single classification call.

    ``decision`` is the actionable verdict the caller should follow.
    ``confidence`` is on [0, 1]; below the rule threshold the dispatcher
    routes to the zero-shot pass. ``source`` records which layer produced
    the verdict for the review log.
    """

    decision: AddressingDecision
    confidence: float
    source: str  # "rule" | "zero_shot" | "default_silent"
    reason: str
    latency_ms: float
    rule_hit: Optional[str] = None
    zero_shot_raw: Optional[str] = None


class AddressingClassifier:
    """Rule-based + zero-shot addressing classifier (CPU-only).

    Args:
        rule_confidence_threshold: minimum confidence for a rule verdict to
            short-circuit zero-shot. 0.8 per spec.
        default_silent_on_uncertain: if both layers return UNCERTAIN, default
            to NOT_ADDRESSED (the safer behavior). Set False to be permissive.
        log_path: where decision logs land. ``None`` disables logging.
        zero_shot_model_name: HF model id for the zero-shot fallback.
        load_zero_shot_eagerly: if True, load Flan-T5-small at construction.
            Otherwise it loads on first ambiguous utterance. Eager loading
            avoids a ~8 s stall on the first WARM-mode utterance.
        recent_turns_provider: callable returning a list of (role, content)
            tuples for the last few conversation turns. Used as context for
            the zero-shot pass. ``None`` means no context is supplied.
    """

    def __init__(
        self,
        rule_confidence_threshold: float = 0.8,
        default_silent_on_uncertain: bool = True,
        log_path: Optional[Path] = None,
        zero_shot_model_name: str = "google/flan-t5-small",
        load_zero_shot_eagerly: bool = False,
        recent_turns_provider: Optional[Callable[[int], List[Tuple[str, str]]]] = None,
        zero_shot_addressed_min_confidence: float = 0.0,
    ) -> None:
        self.rule_threshold = rule_confidence_threshold
        self.default_silent = default_silent_on_uncertain
        # 2026-05-11: low-confidence zero-shot YES verdicts saturate
        # around 0.75 on borderline third-person utterances. When this
        # threshold is non-zero, a zero-shot YES below the bar is
        # downgraded (per ``default_silent_on_uncertain``). Default 0.0
        # preserves legacy behaviour for callers that don't opt in.
        self.zero_shot_addressed_min_confidence = float(zero_shot_addressed_min_confidence)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()
        self._zero_shot = ZeroShotAddresseeModel(zero_shot_model_name)
        self._recent_turns_provider = recent_turns_provider

        if load_zero_shot_eagerly:
            self._zero_shot._ensure_loaded()  # noqa: SLF001 -- intentional warmup

    # --- public API ----------------------------------------------------------

    def classify(
        self,
        utterance: str,
        seconds_since_response: float = 0.0,
        matcher_hit: bool = False,
    ) -> AddressingVerdict:
        """Classify ``utterance`` via the fused log-odds scorer. Always returns a
        verdict; never raises (fails open to the legacy rules->zero-shot cascade).

        ``matcher_hit`` lets the caller signal that a deterministic command matcher
        (relay / stop-button / spotify / settings) already parses the utterance --
        a very strong ADDRESSED feature (an executable command is almost certainly
        for us)."""
        t0 = time.monotonic()
        try:
            return self._classify_fused(
                utterance, seconds_since_response, matcher_hit, t0)
        except Exception as e:                                       # noqa: BLE001
            logger.warning("fused addressing failed (%s) -- legacy cascade", e)
            return self._classify_cascade(utterance, seconds_since_response)

    def _classify_fused(
        self,
        utterance: str,
        seconds_since_response: float,
        matcher_hit: bool,
        t0: float,
    ) -> AddressingVerdict:
        feats = rule_features(utterance, seconds_since_response)
        feats["matcher_hit"] = 1.0 if matcher_hit else 0.0
        lex = _addr_b0(feats.get("recency_s", 0.0)) + sum(
            _ADDR_W[k] * feats.get(k, 0.0) for k in _ADDR_W)
        logit = lex
        raw = None
        used_flan = False
        # Consult the zero-shot model ONLY when the lexical evidence is
        # undecided -- clear commands / clear chatter skip it (zero model latency).
        if abs(lex) < _ADDR_FLAN_BAND:
            try:
                context = (
                    self._recent_turns_provider(4)
                    if self._recent_turns_provider is not None else None)
            except Exception:                                        # noqa: BLE001
                context = None
            raw, zs_conf, _ = self._zero_shot.classify(
                utterance, context=context,
                seconds_since_response=seconds_since_response)
            used_flan = True
            p_yes = (zs_conf if raw == "YES"
                     else (1.0 - zs_conf) if raw == "NO" else 0.5)
            p_yes = min(max(p_yes, 1e-4), 1.0 - 1e-4)
            logit += _ADDR_W_FLAN * math.log(p_yes / (1.0 - p_yes))
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        tau = _addr_tau()
        decision = (
            AddressingDecision.ADDRESSED if p >= tau
            else (AddressingDecision.NOT_ADDRESSED if self.default_silent
                  else AddressingDecision.UNCERTAIN))
        verdict = AddressingVerdict(
            decision=decision,
            confidence=p,
            source="fusion",
            reason=(f"fused p={p:.2f}>=tau {tau:.2f}" if p >= tau
                    else f"fused p={p:.2f}<tau {tau:.2f}")
                   + f" | lex_logit={lex:.2f}"
                   + (f" +flan({raw})" if used_flan else " (lexical-decisive)"),
            latency_ms=(time.monotonic() - t0) * 1000,
            zero_shot_raw=raw,
        )
        self._log(utterance, verdict)
        return verdict

    def _classify_cascade(
        self,
        utterance: str,
        seconds_since_response: float = 0.0,
    ) -> AddressingVerdict:
        """Legacy rules->zero-shot cascade; the fail-open fallback for classify()."""
        t0 = time.monotonic()

        rule_hit = classify_by_rules(utterance, seconds_since_response)
        if rule_hit is not None and rule_hit.confidence >= self.rule_threshold:
            verdict = AddressingVerdict(
                decision=rule_hit.decision,
                confidence=rule_hit.confidence,
                source="rule",
                reason=rule_hit.reason,
                latency_ms=(time.monotonic() - t0) * 1000,
                rule_hit=rule_hit.reason,
            )
            self._log(utterance, verdict)
            return verdict

        # Below the rule threshold: fall through to zero-shot.
        try:
            context = (
                self._recent_turns_provider(4)
                if self._recent_turns_provider is not None
                else None
            )
        except Exception as e:
            logger.warning("recent_turns_provider failed: %s", e)
            context = None

        try:
            raw_verdict, zs_conf, zs_ms = self._zero_shot.classify(
                utterance,
                context=context,
                seconds_since_response=seconds_since_response,
            )
        except Exception as e:
            logger.warning("Zero-shot classifier failed: %s -- defaulting to silent", e)
            from kenning.errors import AddressingClassifierError
            from kenning.resilience import get_error_log
            get_error_log().record(
                AddressingClassifierError(
                    f"zero-shot classify failed: {e}",
                    context={"utterance_len": len(utterance)},
                    recovery=(
                        "default-silent verdict" if self.default_silent
                        else "uncertain verdict"
                    ),
                ),
                dependency="addressing_zero_shot",
            )
            verdict = AddressingVerdict(
                decision=AddressingDecision.NOT_ADDRESSED if self.default_silent
                else AddressingDecision.UNCERTAIN,
                confidence=0.30,
                source="default_silent",
                reason=f"zero-shot error: {e}",
                latency_ms=(time.monotonic() - t0) * 1000,
                rule_hit=rule_hit.reason if rule_hit else None,
            )
            self._log(utterance, verdict)
            return verdict

        decision = _map_zero_shot_to_decision(raw_verdict, self.default_silent)
        # 2026-05-11 false-positive guard: borderline third-person
        # narration was triggering zero-shot YES at exactly 0.75. When
        # the min-confidence gate is configured, demote low-confidence
        # YES verdicts to UNCERTAIN/NOT_ADDRESSED per default_silent.
        gated_for_low_confidence = False
        if (
            self.zero_shot_addressed_min_confidence > 0.0
            and decision == AddressingDecision.ADDRESSED
            and zs_conf < self.zero_shot_addressed_min_confidence
        ):
            decision = (
                AddressingDecision.NOT_ADDRESSED
                if self.default_silent
                else AddressingDecision.UNCERTAIN
            )
            gated_for_low_confidence = True

        # If the rule layer had a soft hint (>0.5) and zero-shot agrees, take
        # the higher confidence; otherwise trust the zero-shot.
        if rule_hit is not None and rule_hit.decision == decision:
            confidence = max(zs_conf, rule_hit.confidence)
            reason = f"zero-shot {raw_verdict} (agrees with rule: {rule_hit.reason})"
        else:
            confidence = zs_conf
            reason = f"zero-shot {raw_verdict}"
        if gated_for_low_confidence:
            reason = (
                f"zero-shot {raw_verdict} below ADDRESSED threshold "
                f"({zs_conf:.2f} < {self.zero_shot_addressed_min_confidence:.2f})"
            )

        verdict = AddressingVerdict(
            decision=decision,
            confidence=confidence,
            source="zero_shot",
            reason=reason,
            latency_ms=(time.monotonic() - t0) * 1000,
            rule_hit=rule_hit.reason if rule_hit else None,
            zero_shot_raw=raw_verdict,
        )
        self._log(utterance, verdict)
        return verdict

    def should_respond(self, utterance: str, seconds_since_response: float = 0.0) -> bool:
        """Convenience wrapper matching the orchestrator's old call signature.

        Returns True iff the verdict is ADDRESSED.
        """
        verdict = self.classify(utterance, seconds_since_response)
        return verdict.decision == AddressingDecision.ADDRESSED

    # --- logging -------------------------------------------------------------

    def _log(self, utterance: str, verdict: AddressingVerdict) -> None:
        logger.info(
            "addressing: %s (%s, conf=%.2f, %.0f ms) -- %r",
            verdict.decision.value,
            verdict.source,
            verdict.confidence,
            verdict.latency_ms,
            utterance[:80],
        )
        # Cross-cutting observation emit (fail-open). Stamped before the
        # log-file write so a JSONL write failure doesn't suppress the
        # observation row.
        try:
            from kenning.observations import observe_addressing_verdict

            observe_addressing_verdict(
                utterance=utterance,
                decision=verdict.decision.value,
                confidence=float(verdict.confidence or 0.0),
                reason=verdict.reason or "",
                seconds_since_response=0.0,
                source=verdict.source or "",
                latency_ms=float(verdict.latency_ms or 0.0),
            )
        except Exception:
            pass
        if self.log_path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "utterance": utterance,
            **{k: (v.value if hasattr(v, "value") else v) for k, v in asdict(verdict).items()},
        }
        try:
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.debug("Addressing log write failed: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _map_zero_shot_to_decision(
    raw_verdict: str, default_silent: bool
) -> AddressingDecision:
    if raw_verdict == "YES":
        return AddressingDecision.ADDRESSED
    if raw_verdict == "NO":
        return AddressingDecision.NOT_ADDRESSED
    # UNCLEAR or anything unrecognized.
    return (
        AddressingDecision.NOT_ADDRESSED
        if default_silent
        else AddressingDecision.UNCERTAIN
    )
