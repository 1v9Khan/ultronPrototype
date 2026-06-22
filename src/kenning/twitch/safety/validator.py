"""ChatSafetyValidator — the most-restrictive-wins, fail-CLOSED L0-L7 arbiter.

Mirrors the ethos of ``kenning.safety.validator.ToolCallValidator`` (iterate ALL
rules, dominant = max severity, fail-CLOSED on any rule exception) but over a
:class:`ChatMessageContext` and emitting a graded ``danger_score`` -> 4-band
action (ALLOW / REVIEW / DEFLECT / BLOCK) per ``constitution.md`` v1. Reuses the
hash-chained :class:`kenning.safety.audit.AuditLog` at a separate path.

The deterministic rules (L1 blocklist on the body AND every untrusted metadata
field, L5 reassembly, L6 phonetic markup guard) need NO model and are the
load-bearing defense against a hostile abliterated model. The guard-model rule
(L2/L3/L5-exchange) is ADDITIVE: present only when a guard client is wired, and
FAIL-CLOSED if that client errors/timeouts. The "guard REQUIRED when chat-mode
ON" precondition is enforced at the chat-mode-enable gate (the toggle path), not
per-message — so this validator stays usable (deterministic-only) in tests and
when the guard sidecar is momentarily unavailable degrades to deterministic
coverage rather than failing every message.

Context distinction:
  * INPUT screen (``is_output=False``): verdicts ALLOW/REVIEW/BLOCK — a trip stops
    the message reaching the 8B and (per band) flags/auto-moderates.
  * OUTPUT screen (``is_output=True``): verdicts ALLOW/DEFLECT — a trip substitutes
    a constant-string deflection; we never speak the draft.

ANTICHEAT: stdlib + rapidfuzz (+ the optional guard client, which lives in a
sidecar). Importable in either process.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Protocol, Sequence

from kenning.twitch.safety.blocklist import Blocklist, BlockMatch, get_blocklist
from kenning.twitch.safety.deflection import pick_deflection
from kenning.twitch.safety.phonetic import phonetic_guard
from kenning.twitch.safety.reassembly import reassembly_matches

logger = logging.getLogger("kenning.twitch.safety.validator")

__all__ = [
    "ChatVerdict", "ChatMessageContext", "ChatDecision", "RuleOutcome",
    "ChatSafetyRule", "ChatSafetyValidator", "build_chat_validator",
    "GuardClient", "GuardResult",
]


class ChatVerdict(IntEnum):
    ALLOW = 0
    REVIEW = 1     # gray zone -> 2nd-monitor popup + voice review loop
    DEFLECT = 2    # output trip -> substitute a constant-string deflection
    BLOCK = 3      # input clearly unsafe -> never reaches the 8B; auto-moderate


_SEV_SCORE = {"none": 0.0, "low": 0.30, "medium": 0.55, "high": 0.85, "critical": 1.0}


@dataclass(frozen=True)
class ChatMessageContext:
    text: str
    username: str = ""
    user_id: str = ""
    source: str = "twitch_chat"       # provenance (twitch_chat / redeem / raid / ...)
    is_output: bool = False           # False=screen inbound; True=screen a draft reply
    batch_context: tuple[str, ...] = ()    # other messages this batch (cross-batch acrostics)
    extra_fields: tuple[str, ...] = ()     # untrusted metadata to ALSO screen (username/emote/reward title/raid msg)
    inbound_for_exchange: str = ""    # for output screen: the inbound that prompted the draft (L5 exchange)


@dataclass(frozen=True)
class RuleOutcome:
    rule_id: str
    verdict: ChatVerdict
    score: float
    reason: str
    matches: tuple[BlockMatch, ...] = ()


@dataclass(frozen=True)
class ChatDecision:
    verdict: ChatVerdict
    danger_score: float
    reason: str
    rule_id: str
    matches: tuple[BlockMatch, ...] = ()
    deflection: Optional[str] = None
    outcomes: tuple[RuleOutcome, ...] = ()

    @property
    def is_allowed(self) -> bool:
        return self.verdict == ChatVerdict.ALLOW

    @property
    def should_speak(self) -> bool:
        """True only when an OUTPUT screen passed clean (ALLOW)."""
        return self.verdict == ChatVerdict.ALLOW


class ChatSafetyRule(Protocol):
    rule_id: str

    def evaluate(self, ctx: ChatMessageContext) -> RuleOutcome:  # pragma: no cover - protocol
        ...


# --- guard-model client contract (the sidecar implements this) ---------------
@dataclass(frozen=True)
class GuardResult:
    unsafe: bool
    category: str = ""
    score: float = 0.0           # P(unsafe) in [0,1]


class GuardClient(Protocol):
    def classify(self, text: str, *, exchange: str = "") -> GuardResult:  # pragma: no cover
        ...


# --- helpers ------------------------------------------------------------------
def _verdict_for_input(sev_rank: int) -> ChatVerdict:
    # critical/high -> BLOCK; medium/low -> REVIEW.
    return ChatVerdict.BLOCK if sev_rank >= 3 else ChatVerdict.REVIEW


def _normalize_verdict(v: ChatVerdict, is_output: bool) -> ChatVerdict:
    if is_output:
        # On the output side we never BLOCK/REVIEW — we DEFLECT (don't speak it).
        return ChatVerdict.DEFLECT if v >= ChatVerdict.REVIEW else ChatVerdict.ALLOW
    # On the input side DEFLECT collapses to BLOCK.
    return ChatVerdict.BLOCK if v == ChatVerdict.DEFLECT else v


def _redact(text: str) -> str:
    t = text or ""
    if len(t) <= 12:
        head = t[:4]
    else:
        head = t[:8]
    return f"{head}…(+{max(0, len(t) - len(head))}) sha256:{hashlib.sha256(t.encode('utf-8','replace')).hexdigest()[:12]}"


# --- the deterministic + guard rules -----------------------------------------
class _BlocklistRule:
    rule_id = "L1_blocklist"

    def __init__(self, blocklist: Blocklist) -> None:
        self._bl = blocklist

    def evaluate(self, ctx: ChatMessageContext) -> RuleOutcome:
        all_matches: list[BlockMatch] = list(self._bl.scan_text(ctx.text))
        for fld in ctx.extra_fields:
            all_matches.extend(self._bl.scan_text(fld))
        if not all_matches:
            return RuleOutcome(self.rule_id, ChatVerdict.ALLOW, 0.0, "clean")
        worst = max(all_matches, key=lambda m: m.severity_rank)
        score = _SEV_SCORE.get(worst.severity, 0.5)
        v = _normalize_verdict(_verdict_for_input(worst.severity_rank), ctx.is_output)
        return RuleOutcome(
            self.rule_id, v, score,
            f"blocklist:{worst.category}/{worst.severity} via {worst.rule}",
            tuple(all_matches),
        )


class _ReassemblyRule:
    rule_id = "L5_reassembly"

    def __init__(self, blocklist: Blocklist) -> None:
        self._bl = blocklist

    def evaluate(self, ctx: ChatMessageContext) -> RuleOutcome:
        ms = reassembly_matches(ctx.text, blocklist=self._bl, batch_context=ctx.batch_context)
        if not ms:
            return RuleOutcome(self.rule_id, ChatVerdict.ALLOW, 0.0, "no hidden channel")
        worst = max(ms, key=lambda m: m.severity_rank)
        score = max(0.85, _SEV_SCORE.get(worst.severity, 0.85))
        v = _normalize_verdict(_verdict_for_input(worst.severity_rank), ctx.is_output)
        return RuleOutcome(self.rule_id, v, score,
                           f"reassembly:{worst.category} via {worst.rule}", tuple(ms))


class _PhoneticRule:
    rule_id = "L6_phonetic"

    def __init__(self, blocklist: Blocklist) -> None:
        self._bl = blocklist

    def evaluate(self, ctx: ChatMessageContext) -> RuleOutcome:
        # L6 is an OUTPUT-side (TTS choke point) gate; on input it is a no-op.
        if not ctx.is_output:
            return RuleOutcome(self.rule_id, ChatVerdict.ALLOW, 0.0, "input (n/a)")
        v = phonetic_guard(ctx.text, blocklist=self._bl)
        if v.clear:
            return RuleOutcome(self.rule_id, ChatVerdict.ALLOW, 0.0, "clear")
        return RuleOutcome(self.rule_id, ChatVerdict.DEFLECT, 1.0, v.reason, v.matches)


class _GuardModelRule:
    rule_id = "L3_guard"

    def __init__(self, client: GuardClient) -> None:
        self._client = client

    def evaluate(self, ctx: ChatMessageContext) -> RuleOutcome:
        # FAIL-CLOSED: any guard error/timeout is a safety event.
        try:
            res = self._client.classify(ctx.text, exchange=ctx.inbound_for_exchange)
        except Exception as e:  # noqa: BLE001
            logger.warning("guard client failed; failing CLOSED: %s", e)
            v = _normalize_verdict(ChatVerdict.BLOCK, ctx.is_output)
            return RuleOutcome(self.rule_id, v, 1.0, f"guard unavailable (fail-closed): {type(e).__name__}")
        if not res.unsafe:
            return RuleOutcome(self.rule_id, ChatVerdict.ALLOW, float(res.score or 0.0), "guard: safe")
        v = _normalize_verdict(ChatVerdict.BLOCK, ctx.is_output)
        return RuleOutcome(self.rule_id, v, max(0.85, float(res.score or 0.85)),
                           f"guard: unsafe ({res.category or 'unspecified'})")


# --- the validator ------------------------------------------------------------
class ChatSafetyValidator:
    def __init__(
        self,
        rules: Sequence[ChatSafetyRule],
        *,
        audit: Optional[Any] = None,
        taus: tuple[float, float, float] = (0.25, 0.55, 0.80),
    ) -> None:
        self._rules = list(rules)
        self._audit = audit
        self._tau_review, self._tau_deflect, self._tau_block = taus

    def _band(self, score: float, is_output: bool) -> ChatVerdict:
        if score >= self._tau_block:
            return ChatVerdict.DEFLECT if is_output else ChatVerdict.BLOCK
        if score >= self._tau_deflect:
            return ChatVerdict.DEFLECT if is_output else ChatVerdict.REVIEW
        if score >= self._tau_review:
            return ChatVerdict.DEFLECT if is_output else ChatVerdict.REVIEW
        return ChatVerdict.ALLOW

    def check(self, ctx: ChatMessageContext) -> ChatDecision:
        outcomes: list[RuleOutcome] = []
        for rule in self._rules:
            rid = getattr(rule, "rule_id", rule.__class__.__name__)
            try:
                outcomes.append(rule.evaluate(ctx))
            except Exception as e:  # noqa: BLE001 — a crashing rule fails CLOSED
                logger.warning("chat-safety rule %s crashed; failing CLOSED: %s", rid, e)
                v = _normalize_verdict(ChatVerdict.BLOCK, ctx.is_output)
                outcomes.append(RuleOutcome(rid, v, 1.0, f"rule crash (fail-closed): {type(e).__name__}"))

        danger = max((o.score for o in outcomes), default=0.0)
        rule_dom = max(outcomes, key=lambda o: (int(o.verdict), o.score), default=None)
        rule_verdict = rule_dom.verdict if rule_dom else ChatVerdict.ALLOW
        band_verdict = self._band(danger, ctx.is_output)
        final = ChatVerdict(max(int(rule_verdict), int(band_verdict)))
        final = _normalize_verdict(final, ctx.is_output)

        matches: tuple[BlockMatch, ...] = tuple(m for o in outcomes for m in o.matches)
        reason = rule_dom.reason if (rule_dom and rule_dom.verdict == final) else f"danger={danger:.2f}"
        rid = rule_dom.rule_id if rule_dom else "none"
        deflection = pick_deflection(ctx.text) if (ctx.is_output and final >= ChatVerdict.DEFLECT) else None

        decision = ChatDecision(
            verdict=final, danger_score=danger, reason=reason, rule_id=rid,
            matches=matches, deflection=deflection, outcomes=tuple(outcomes),
        )
        if final != ChatVerdict.ALLOW:
            self._audit_decision(ctx, decision)
        return decision

    def _audit_decision(self, ctx: ChatMessageContext, d: ChatDecision) -> None:
        if self._audit is None:
            return
        try:
            self._audit.record(
                rule_id=d.rule_id,
                verdict=d.verdict.name,
                tool_name="twitch_chat_out" if ctx.is_output else "twitch_chat_in",
                capability=ctx.source,
                reason=d.reason,
                context={
                    "user_id": ctx.user_id,
                    "username_redacted": _redact(ctx.username),
                    "text_redacted": _redact(ctx.text),
                    "danger_score": round(d.danger_score, 3),
                    "categories": sorted({m.category for m in d.matches}),
                    "rules": [o.rule_id for o in d.outcomes if o.verdict != ChatVerdict.ALLOW],
                },
            )
        except Exception as e:  # noqa: BLE001 — audit failure must not change the verdict
            logger.warning("twitch chat audit write failed: %s", e)


def build_chat_validator(
    *,
    guard_client: Optional[GuardClient] = None,
    blocklist: Optional[Blocklist] = None,
    audit_path: Optional[str] = "logs/twitch_mod_audit.jsonl",
    taus: tuple[float, float, float] = (0.25, 0.55, 0.80),
) -> ChatSafetyValidator:
    """Wire the deterministic rules (always) + the guard rule (iff a client is
    given). The 'guard REQUIRED when chat-mode ON' precondition lives at the
    enable gate, not here."""
    bl = blocklist or get_blocklist()
    rules: list[ChatSafetyRule] = [
        _BlocklistRule(bl), _ReassemblyRule(bl), _PhoneticRule(bl),
    ]
    if guard_client is not None:
        rules.append(_GuardModelRule(guard_client))
    audit = None
    if audit_path:
        try:
            from kenning.safety.audit import AuditLog
            audit = AuditLog(path=audit_path)
        except Exception as e:  # noqa: BLE001 — degrade to no audit, never crash the gate
            logger.warning("twitch audit log unavailable (%s); proceeding without audit", e)
    return ChatSafetyValidator(rules, audit=audit, taus=taus)
