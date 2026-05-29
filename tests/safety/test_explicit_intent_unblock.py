"""Tests for the conditional NEEDS_EXPLICIT_INTENT unblock in the validator.

A NEI verdict is upgraded to an audited allow (LOG_ONLY) ONLY when the user's
current utterance explicitly names the action (verb + object). The security
invariants pinned here:

* explicit intent in the utterance -> NEI becomes allowed;
* no explicit intent -> NEI stays blocked;
* a BLOCK_HARD from any rule is NEVER overridden by explicit intent;
* empty utterance -> NEI stays blocked;
* the flag (safety.explicit_intent_matching_enabled) gates the whole thing.
"""

from __future__ import annotations

from ultron.safety.policy import load_policy
from ultron.safety.validator import (
    RuleContext,
    RuleResult,
    ToolCallValidator,
    Verdict,
)


class _FixedRule:
    def __init__(self, rule_id: str, verdict: Verdict):
        self.rule_id = rule_id
        self._verdict = verdict

    def evaluate(self, ctx, *, policy, resolver):  # noqa: ARG002
        return RuleResult(rule_id=self.rule_id, verdict=self._verdict, reason=self.rule_id)


def _validator(rules, *, explicit_intent=True):
    return ToolCallValidator(
        policy=load_policy(), rules=rules, explicit_intent_matching=explicit_intent,
    )


def _ctx(user_text: str):
    return RuleContext(
        tool_name="file.delete",
        arguments={},
        capability="test",
        paths=("/proj/sandbox/bar.py",),
        user_text=user_text,
    )


def test_nei_upgraded_on_explicit_intent():
    v = _validator([_FixedRule("NEI", Verdict.NEEDS_EXPLICIT_INTENT)])
    res = v.check(_ctx("please delete the bar.py file"))
    assert res.is_allowed
    assert res.verdict == Verdict.LOG_ONLY


def test_nei_stays_blocked_without_explicit_intent():
    v = _validator([_FixedRule("NEI", Verdict.NEEDS_EXPLICIT_INTENT)])
    res = v.check(_ctx("what's the weather like today"))
    assert not res.is_allowed
    assert res.verdict == Verdict.NEEDS_EXPLICIT_INTENT


def test_hard_block_never_overridden_by_explicit_intent():
    v = _validator([
        _FixedRule("NEI", Verdict.NEEDS_EXPLICIT_INTENT),
        _FixedRule("HARD", Verdict.BLOCK_HARD),
    ])
    res = v.check(_ctx("delete the bar.py file"))
    assert not res.is_allowed
    assert res.verdict == Verdict.BLOCK_HARD


def test_flag_off_keeps_nei_blocked():
    v = _validator(
        [_FixedRule("NEI", Verdict.NEEDS_EXPLICIT_INTENT)], explicit_intent=False,
    )
    res = v.check(_ctx("delete the bar.py file"))
    assert res.verdict == Verdict.NEEDS_EXPLICIT_INTENT


def test_empty_user_text_keeps_nei_blocked():
    v = _validator([_FixedRule("NEI", Verdict.NEEDS_EXPLICIT_INTENT)])
    res = v.check(_ctx(""))
    assert res.verdict == Verdict.NEEDS_EXPLICIT_INTENT


def test_plain_allow_is_unaffected():
    v = _validator([_FixedRule("OK", Verdict.ALLOW)])
    res = v.check(_ctx("delete the bar.py file"))
    assert res.is_allowed
    assert res.verdict == Verdict.ALLOW
