"""Phase 2 tests: ConversationCoordinator clarification + adjustment logic.

The decision matrix is exercised against 30 mock clarification scenarios
plus a focused set of adjustment + escalation cases. The LLM is mocked
out (a small ``_FakeLLM`` returns canned JSON / text) so tests are fast
and deterministic.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from ultron.coding.coordinator import (
    AdjustmentDecision,
    ConversationCoordinator,
    DecisionPath,
    PendingUserClarification,
)
from ultron.coding.session import (
    ClarificationRequest,
    SessionStatus,
    SessionStore,
    StageRecord,
)


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Minimal stand-in for :class:`LLMEngine`.

    Tests script its replies via ``set_decision`` / ``set_voice_question``
    / etc. Each call records its prompt prefix so tests can assert on
    routing.
    """

    def __init__(self) -> None:
        self.calls: List[str] = []
        self._scripted: List[str] = []
        self._default_decide = '{"action": "ESCALATE", "answer": null, "reasoning": "default"}'
        self._default_voice = "Question for you."
        self._default_conflict = '{"is_conflict": false, "reason": "ok", "completed_at_risk": []}'
        self._default_adjustment = "Apply the adjustment to the in-progress work."

    def set_next(self, response: str) -> None:
        self._scripted.append(response)

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._scripted:
            return self._scripted.pop(0)
        # Heuristic fallback so the non-test prompts don't fall over.
        if "<<<DECIDE>>>" in prompt or "Decide ONE of" in prompt:
            return self._default_decide
        if "spoken-style" in prompt or "Translate this technical" in prompt:
            return self._default_voice
        if "is_conflict" in prompt or "mid-session adjustment conflicts" in prompt:
            return self._default_conflict
        if "translate it into a concrete follow-up prompt" in prompt or "wrap" in prompt:
            return self._default_adjustment
        return ""


# ---------------------------------------------------------------------------
# Fixture: a fresh store + coordinator for each test
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path):
    store = SessionStore()
    project = tmp_path / "project"
    project.mkdir()
    session = store.create(
        project_root=project,
        user_intent="Build me a Python CLI that fetches forecasts",
        mode="new",
        model="haiku",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)

    llm = _FakeLLM()
    log = tmp_path / "clarifs.jsonl"
    coord = ConversationCoordinator(
        store=store, llm=llm,
        log_path=log,
        clarification_user_timeout_s=2.0,  # keep the timeout test fast
    )
    return {
        "store": store,
        "session_id": session.session_id,
        "session": session,
        "llm": llm,
        "coord": coord,
        "log": log,
    }


def _make_request(
    question: str,
    options: Optional[List[str]] = None,
    urgency: str = "blocking",
) -> ClarificationRequest:
    import uuid
    return ClarificationRequest(
        request_id=uuid.uuid4().hex,
        question=question,
        options=list(options or []),
        urgency=urgency,  # type: ignore[arg-type]
    )


def _decide_sync(coord, session_id, request, store) -> str:
    """Helper: run :meth:`decide_clarification` to completion."""
    session = store.get(session_id)
    return asyncio.run(coord.decide_clarification(session_id, request, session))


# ---------------------------------------------------------------------------
# Decision-matrix test set: 30 scenarios driven by rule + LLM paths.
#
# Each entry is (label, question, options, urgency, expected_path,
# expected_kind, llm_script).
#  - expected_path is one of "rule_escalate", "rule_default",
#    "rule_answer", "llm_*", "user_answer".
#  - expected_kind is "ANSWER", "USE_DEFAULT", or "ESCALATE" -- the
#    bucket Claude eventually receives.
#  - llm_script is a list of strings that the fake LLM returns in order;
#    the first is the decision JSON (when the LLM path is exercised);
#    the second (optional) is the voice question (when escalation
#    happens via LLM path).
# ---------------------------------------------------------------------------


@dataclass
class _Case:
    label: str
    question: str
    options: List[str]
    urgency: str
    expected_path: str
    expected_kind: str
    llm_script: List[str] = field(default_factory=list)


_RULE_ESCALATE_CASES = [
    _Case("api_key", "What is the OpenAI API key?", [], "blocking",
          "rule_escalate", "ESCALATE"),
    _Case("aws_setup", "Do you have an AWS account I should deploy this to?", [], "blocking",
          "rule_escalate", "ESCALATE"),
    _Case("paid_tier", "Should we upgrade to the paid Stripe tier?", [], "blocking",
          "rule_escalate", "ESCALATE"),
    _Case("scope_add", "Should I also add user authentication to this?", [], "blocking",
          "rule_escalate", "ESCALATE"),
    _Case("breaking", "Should I introduce a breaking change to the public API?",
          [], "blocking", "rule_escalate", "ESCALATE"),
    _Case("deploy_target", "What's the deployment target -- production server or local only?",
          [], "blocking", "rule_escalate", "ESCALATE"),
    _Case("creds_secret", "Where should I read the database password from?", [], "blocking",
          "rule_escalate", "ESCALATE"),
    _Case("expand_scope", "Should I expand scope to include reporting?", [], "blocking",
          "rule_escalate", "ESCALATE"),
]


_RULE_DEFAULT_CASES = [
    _Case("preference_with_options",
          "Do you want a logging mixin or a logging decorator? I have a default.",
          ["mixin", "decorator"], "preference",
          "rule_default", "USE_DEFAULT"),
    _Case("preference_options_naming",
          "Do you prefer snake_case or camelCase locals here?",
          ["snake_case", "camelCase"], "preference",
          "rule_default", "USE_DEFAULT"),
]


_RULE_ANSWER_CASES = [
    _Case("test_framework",
          "What test framework should I use?", [], "blocking",
          "rule_answer", "ANSWER"),
    _Case("linter",
          "What linter should I configure?", [], "blocking",
          "rule_answer", "ANSWER"),
    _Case("project_layout",
          "What directory structure / project layout should I use?", [], "blocking",
          "rule_answer", "ANSWER"),
    _Case("file_naming",
          "What file naming convention should I use?", [], "blocking",
          "rule_answer", "ANSWER"),
    _Case("docstring",
          "How verbose should the docstring style be?", [], "blocking",
          "rule_answer", "ANSWER"),
    _Case("logging_lib",
          "Which logging library and format should I use?", [], "blocking",
          "rule_answer", "ANSWER"),
    _Case("error_handling",
          "What error handling pattern do you prefer?", [], "blocking",
          "rule_answer", "ANSWER"),
]


_LLM_ANSWER_CASES = [
    _Case("storage_choice_inferable_from_intent",
          "SQLite or Postgres for storing forecasts?",
          ["sqlite", "postgres"], "blocking",
          "llm_answer", "ANSWER",
          llm_script=[
              json.dumps({
                  "action": "ANSWER",
                  "answer": "Use SQLite -- the user asked for a simple CLI.",
                  "reasoning": "intent says simple CLI; SQLite fits",
              }),
          ]),
    _Case("framework_inferable",
          "Which web framework? FastAPI or Flask?",
          ["fastapi", "flask"], "blocking",
          "llm_answer", "ANSWER",
          llm_script=[
              json.dumps({
                  "action": "ANSWER",
                  "answer": "FastAPI -- modern default for new Python API services.",
                  "reasoning": "modern default; user did not specify",
              }),
          ]),
    _Case("config_format",
          "Should config be YAML or TOML?",
          ["yaml", "toml"], "blocking",
          "llm_answer", "ANSWER",
          llm_script=[
              json.dumps({
                  "action": "ANSWER",
                  "answer": "Use TOML; it's the modern Python convention via pyproject.toml.",
                  "reasoning": "Python convention",
              }),
          ]),
    _Case("library_choice_safe_default",
          "Should I use httpx or requests for HTTP calls?",
          ["httpx", "requests"], "blocking",
          "llm_answer", "ANSWER",
          llm_script=[
              json.dumps({
                  "action": "ANSWER",
                  "answer": "Use httpx -- async-capable and more modern than requests.",
                  "reasoning": "modern default",
              }),
          ]),
]


_LLM_DEFAULT_CASES = [
    _Case("internal_naming",
          "What should I name the internal helper that joins forecast strings?",
          [], "blocking",
          "llm_default", "USE_DEFAULT",
          llm_script=[
              json.dumps({
                  "action": "USE_DEFAULT",
                  "answer": None,
                  "reasoning": "low-stakes internal naming",
              }),
          ]),
    _Case("internal_module_layout",
          "Should I put the formatter logic in a single module or split into two files?",
          [], "blocking",
          "llm_default", "USE_DEFAULT",
          llm_script=[
              json.dumps({
                  "action": "USE_DEFAULT",
                  "answer": None,
                  "reasoning": "low-stakes file layout",
              }),
          ]),
]


_LLM_ESCALATE_CASES = [
    _Case("ambiguous_feature",
          "Should the CLI write forecast history to a file?",
          ["yes write history", "no, just print"], "blocking",
          "llm_escalate", "ESCALATE",
          llm_script=[
              # Decision: ESCALATE
              json.dumps({
                  "action": "ESCALATE",
                  "answer": None,
                  "reasoning": "feature scope addition",
              }),
              # Voice question text (returned by render_voice_question)
              "On the forecast CLI, should it persist a history file or just print? Your call.",
          ]),
    _Case("ambiguous_format_for_user",
          "What output format does the user actually want -- table, JSON, plaintext?",
          ["table", "json", "plaintext"], "blocking",
          "llm_escalate", "ESCALATE",
          llm_script=[
              json.dumps({
                  "action": "ESCALATE",
                  "answer": None,
                  "reasoning": "user-facing output format -- needs input",
              }),
              "How would you like the forecast presented -- as a table, JSON, or plaintext?",
          ]),
    _Case("user_facing_label",
          "What label should the user see at the top of the report?",
          [], "blocking",
          "llm_escalate", "ESCALATE",
          llm_script=[
              json.dumps({
                  "action": "ESCALATE",
                  "answer": None,
                  "reasoning": "user-facing copy",
              }),
              "What heading do you want shown at the top of the forecast report?",
          ]),
]


_TIMEOUT_CASE = [
    _Case("timeout_falls_back_to_default",
          "Should the CLI write forecast history to a file?",
          ["yes write history", "no, just print"], "blocking",
          "timeout_default", "ESCALATE",
          llm_script=[
              json.dumps({
                  "action": "ESCALATE",
                  "answer": None,
                  "reasoning": "feature scope addition",
              }),
              "On the forecast CLI, should it persist a history file?",
          ]),
]


# 30 cases total for the spec's "test set of 30 mock clarifications".
ALL_CASES = (
    _RULE_ESCALATE_CASES                # 8
    + _RULE_DEFAULT_CASES               # 2
    + _RULE_ANSWER_CASES                # 7
    + _LLM_ANSWER_CASES                 # 4
    + _LLM_DEFAULT_CASES                # 2
    + _LLM_ESCALATE_CASES               # 3
    # 26 here. The four cases below round it out to 30 with edge cases:
    + [
        _Case("preference_no_options_falls_to_llm",
              "I'd default to retrying three times. Do you have a preference?",
              [], "preference",
              "llm_answer", "ANSWER",
              llm_script=[
                  json.dumps({
                      "action": "ANSWER",
                      "answer": "Three retries is fine.",
                      "reasoning": "default is sensible",
                  }),
              ]),
        _Case("blocking_with_default_in_text",
              "I'd default to ISO 8601 timestamps. Should I?",
              [], "blocking",
              "llm_answer", "ANSWER",
              llm_script=[
                  json.dumps({
                      "action": "ANSWER",
                      "answer": "Yes, ISO 8601 is the right call.",
                      "reasoning": "iso 8601 is standard",
                  }),
              ]),
        _Case("blocking_explicit_credential",
              "What's the cloudflare API token?", [], "blocking",
              "rule_escalate", "ESCALATE"),
        _Case("blocking_explicit_paid_plan",
              "Do you have a paid OpenAI plan we can use?", [], "blocking",
              "rule_escalate", "ESCALATE"),
    ]
)


assert len(ALL_CASES) >= 30, f"want at least 30 cases, got {len(ALL_CASES)}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", [c for c in ALL_CASES if not c.label.startswith("timeout")],
    ids=lambda c: c.label,
)
def test_clarification_decision_matrix(case: _Case, env, monkeypatch):
    """Drive each scripted scenario through the coordinator and assert
    that the resolved answer ends up matching ``expected_kind``.

    For ESCALATE cases we resolve the future synchronously by patching
    the coordinator's wait so the test doesn't actually pause."""
    coord = env["coord"]
    llm: _FakeLLM = env["llm"]
    for resp in case.llm_script:
        llm.set_next(resp)

    request = _make_request(
        question=case.question, options=case.options, urgency=case.urgency,
    )

    if case.expected_kind == "ESCALATE":
        # Drive the escalation: wait until the request appears in
        # pending_user_clarifications, then deliver an answer.
        ANSWER_TEXT = "<test escalation answer>"

        async def runner():
            task = asyncio.create_task(
                coord.decide_clarification(env["session_id"], request, env["session"])
            )
            # Spin until the coordinator parks the request, then deliver.
            for _ in range(100):
                await asyncio.sleep(0.02)
                pending = coord.pending_user_clarifications()
                if any(p.request_id == request.request_id for p in pending):
                    coord.deliver_user_clarification_response(
                        request.request_id, ANSWER_TEXT,
                    )
                    break
            return await task

        answer = asyncio.run(runner())
        assert answer == ANSWER_TEXT
    else:
        answer = _decide_sync(coord, env["session_id"], request, env["store"])
        assert answer
        if case.expected_kind == "USE_DEFAULT":
            assert "default" in answer.lower()
        else:  # ANSWER
            assert "default" not in answer.lower() or len(answer) > 30, (
                f"answer suspiciously brief: {answer!r}"
            )


def test_clarification_timeout_falls_back_to_default(env):
    """When the user doesn't respond within the timeout, the coordinator
    must return a default response and not hang."""
    coord = env["coord"]
    llm: _FakeLLM = env["llm"]
    case = _TIMEOUT_CASE[0]
    for resp in case.llm_script:
        llm.set_next(resp)
    request = _make_request(
        question=case.question, options=case.options, urgency=case.urgency,
    )

    t0 = time.monotonic()
    answer = asyncio.run(
        coord.decide_clarification(env["session_id"], request, env["session"])
    )
    elapsed = time.monotonic() - t0
    # Timeout was set to 2 seconds in the fixture.
    assert 1.5 < elapsed < 6.0, f"unexpected timeout duration: {elapsed:.2f}s"
    assert "default" in answer.lower()


def test_decision_log_records_path_and_answer(env):
    coord = env["coord"]
    request = _make_request(
        question="What test framework should I use?",
    )
    asyncio.run(coord.decide_clarification(env["session_id"], request, env["session"]))

    log_lines = env["log"].read_text(encoding="utf-8").splitlines()
    assert log_lines, "expected at least one log line"
    rec = json.loads(log_lines[-1])
    assert rec["decision_path"] == "rule_answer"
    assert rec["answer"]
    assert rec["question"]
    assert "test framework" in rec["question"].lower()


# ---------------------------------------------------------------------------
# Pending-user-clarification surfacing + delivery
# ---------------------------------------------------------------------------


def test_pending_user_clarifications_returns_empty_initially(env):
    assert env["coord"].pending_user_clarifications() == []


def test_pending_user_clarifications_surfaces_after_escalation(env):
    coord = env["coord"]
    llm: _FakeLLM = env["llm"]
    llm.set_next(json.dumps({"action": "ESCALATE", "answer": None, "reasoning": "user_only"}))
    llm.set_next("Want auth: yes or no?")
    request = _make_request(
        question="Should the CLI write forecast history?",
        options=["yes", "no"],
    )

    async def runner():
        task = asyncio.create_task(
            coord.decide_clarification(env["session_id"], request, env["session"])
        )
        for _ in range(50):
            await asyncio.sleep(0.02)
            pending = coord.pending_user_clarifications()
            if pending:
                assert pending[0].request_id == request.request_id
                assert pending[0].voice_question
                coord.deliver_user_clarification_response(
                    request.request_id, "no don't bother",
                )
                break
        return await task

    answer = asyncio.run(runner())
    assert "no don't bother" in answer
    # And after delivery the pending list is empty again.
    assert coord.pending_user_clarifications() == []


# ---------------------------------------------------------------------------
# Adjustment handling
# ---------------------------------------------------------------------------


def test_decide_adjustment_followup_when_no_conflict(env):
    coord = env["coord"]
    # Add some completed work so the coordinator runs the conflict check.
    env["store"].record_stage(
        env["session_id"], stage="scaffolding",
        summary="Created project skeleton",
        files_touched=["main.py"],
    )
    llm: _FakeLLM = env["llm"]
    llm.set_next(json.dumps({
        "is_conflict": False, "reason": "compatible", "completed_at_risk": [],
    }))
    llm.set_next("Switch to async http via httpx; keep the existing CLI structure.")

    decision = asyncio.run(
        coord.decide_adjustment(env["session_id"], "use httpx instead of requests")
    )
    assert decision.action == "FOLLOWUP"
    assert decision.followup_prompt
    assert "httpx" in decision.followup_prompt.lower()


def test_decide_adjustment_escalates_on_conflict(env):
    coord = env["coord"]
    env["store"].record_stage(
        env["session_id"], stage="data layer",
        summary="Implemented sqlite data layer",
        files_touched=["db.py"],
    )
    llm: _FakeLLM = env["llm"]
    llm.set_next(json.dumps({
        "is_conflict": True,
        "reason": "would require rewriting the data layer",
        "completed_at_risk": ["db.py"],
    }))

    decision = asyncio.run(
        coord.decide_adjustment(env["session_id"], "switch from sqlite to postgres")
    )
    assert decision.action == "ESCALATE_CONFLICT"
    assert decision.conflict_reason
    assert decision.voice_question


def test_decide_adjustment_during_pending_clarification_resolves_it(env):
    """If the user makes an adjustment while Claude has a parked
    clarification, the adjustment is treated as the answer."""
    coord = env["coord"]
    llm: _FakeLLM = env["llm"]
    # Force escalation of the first clarification.
    llm.set_next(json.dumps({"action": "ESCALATE", "answer": None, "reasoning": "user_only"}))
    llm.set_next("Want auth: yes or no?")
    request = _make_request(
        question="Should we add auth?", options=["yes", "no"],
    )

    async def runner():
        task = asyncio.create_task(
            coord.decide_clarification(env["session_id"], request, env["session"])
        )
        # Wait until parked, then issue an "adjustment" -- coordinator
        # should detect it answers the pending clarification.
        for _ in range(50):
            await asyncio.sleep(0.02)
            if coord.pending_user_clarifications():
                break
        decision = await coord.decide_adjustment(
            env["session_id"], "no, skip auth"
        )
        return decision, await task

    decision, claude_answer = asyncio.run(runner())
    assert decision.action == "FOLLOWUP"
    assert decision.followup_prompt is None  # answered the clarification
    assert "no, skip auth" in claude_answer


# ---------------------------------------------------------------------------
# A3: stored-facts fast-path
# ---------------------------------------------------------------------------


def _make_facts_lookup(rows):
    """Build a facts_lookup callable that returns ``rows`` regardless of args."""

    def _lookup(query, *, k=5, min_confidence=0.0, max_age_days=None):
        return rows

    return _lookup


def test_decide_clarification_uses_fact_when_high_confidence(tmp_path):
    """A high-confidence preference fact short-circuits the LLM call."""
    store = SessionStore()
    project = tmp_path / "project"
    project.mkdir()
    session = store.create(
        project_root=project,
        user_intent="Build me a Python web service",
        mode="new",
        model="haiku",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)

    llm = _FakeLLM()
    coord = ConversationCoordinator(
        store=store, llm=llm,
        log_path=tmp_path / "clarifs.jsonl",
        facts_lookup=_make_facts_lookup([{
            "fact": "user prefers FastAPI over Flask",
            "confidence": 0.92,
            "category": "preference",
            "score": 0.95,
            "last_confirmed": time.time(),
        }]),
    )

    request = _make_request("Should I use FastAPI or Flask?")
    answer = asyncio.run(coord.decide_clarification(
        session.session_id, request, store.get(session.session_id),
    ))

    # Answer pulled from the fact, not from the LLM.
    assert "FastAPI" in answer
    assert "stored preferences" in answer
    assert llm.calls == [], "LLM should not have been called when fact answered"

    # Verdict logged with FACT_ANSWER decision path.
    log_lines = (tmp_path / "clarifs.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"fact_answer"' in line for line in log_lines)


def test_decide_clarification_skips_fact_below_confidence_threshold(tmp_path):
    """A low-confidence fact must not auto-answer; coordinator falls
    through to the LLM path (which then escalates)."""
    store = SessionStore()
    project = tmp_path / "project"
    project.mkdir()
    session = store.create(
        project_root=project, user_intent="Build a service",
        mode="new", model="haiku",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)

    llm = _FakeLLM()
    coord = ConversationCoordinator(
        store=store, llm=llm,
        log_path=tmp_path / "clarifs.jsonl",
        clarification_user_timeout_s=1.0,
        facts_lookup=_make_facts_lookup([{
            "fact": "user maybe prefers FastAPI",
            "confidence": 0.4,  # below default 0.75 threshold
            "category": "preference",
            "score": 0.95,
            "last_confirmed": time.time(),
        }]),
    )

    # Use a question that will hit the LLM path (not a rule keyword).
    request = _make_request("FastAPI or Starlette for the API layer?")
    asyncio.run(coord.decide_clarification(
        session.session_id, request, store.get(session.session_id),
    ))
    # The LLM path was exercised because the fact was filtered out.
    assert llm.calls, "LLM should have been called when fact filtered out"


def test_decide_clarification_skips_fact_with_low_score(tmp_path):
    """An on-topic high-confidence fact with low RRF score must not
    auto-answer (the score is the relevance gate)."""
    store = SessionStore()
    project = tmp_path / "project"
    project.mkdir()
    session = store.create(
        project_root=project, user_intent="Build a service",
        mode="new", model="haiku",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)

    llm = _FakeLLM()
    coord = ConversationCoordinator(
        store=store, llm=llm,
        log_path=tmp_path / "clarifs.jsonl",
        clarification_user_timeout_s=1.0,
        facts_lookup=_make_facts_lookup([{
            "fact": "user prefers FastAPI",
            "confidence": 0.95,
            "category": "preference",
            "score": 0.40,  # below default 0.85 threshold
            "last_confirmed": time.time(),
        }]),
    )

    request = _make_request("FastAPI or Starlette for the API layer?")
    asyncio.run(coord.decide_clarification(
        session.session_id, request, store.get(session.session_id),
    ))
    assert llm.calls, "low score should fall through to LLM path"


def test_decide_clarification_skips_fact_with_irrelevant_category(tmp_path):
    """Categories like 'person' / 'project' are descriptive, not directive,
    so they don't auto-answer Claude."""
    store = SessionStore()
    project = tmp_path / "project"
    project.mkdir()
    session = store.create(
        project_root=project, user_intent="Build a service",
        mode="new", model="haiku",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)

    llm = _FakeLLM()
    coord = ConversationCoordinator(
        store=store, llm=llm,
        log_path=tmp_path / "clarifs.jsonl",
        clarification_user_timeout_s=1.0,
        facts_lookup=_make_facts_lookup([{
            "fact": "user works at Anthropic",
            "confidence": 0.99,
            "category": "person",  # NOT in directive set
            "score": 0.99,
            "last_confirmed": time.time(),
        }]),
    )

    request = _make_request("FastAPI or Starlette for the API layer?")
    asyncio.run(coord.decide_clarification(
        session.session_id, request, store.get(session.session_id),
    ))
    assert llm.calls, "non-directive category should fall through to LLM path"


def test_decide_clarification_handles_facts_lookup_exception(tmp_path):
    """A raising facts_lookup must not crash the decision loop."""
    store = SessionStore()
    project = tmp_path / "project"
    project.mkdir()
    session = store.create(
        project_root=project, user_intent="Build a service",
        mode="new", model="haiku",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated facts lookup failure")

    llm = _FakeLLM()
    coord = ConversationCoordinator(
        store=store, llm=llm,
        log_path=tmp_path / "clarifs.jsonl",
        clarification_user_timeout_s=1.0,
        facts_lookup=_boom,
    )

    request = _make_request("FastAPI or Starlette for the API layer?")
    answer = asyncio.run(coord.decide_clarification(
        session.session_id, request, store.get(session.session_id),
    ))
    # Coordinator falls through; LLM was called.
    assert llm.calls
    assert isinstance(answer, str)


def test_decide_clarification_unchanged_when_facts_lookup_none(tmp_path):
    """Back-compat: the existing decision matrix must still work when no
    facts_lookup is wired."""
    store = SessionStore()
    project = tmp_path / "project"
    project.mkdir()
    session = store.create(
        project_root=project,
        user_intent="Build a service",
        mode="new", model="haiku",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)

    llm = _FakeLLM()
    # Note: facts_lookup not passed -> None.
    coord = ConversationCoordinator(
        store=store, llm=llm,
        log_path=tmp_path / "clarifs.jsonl",
        clarification_user_timeout_s=1.0,
    )

    # An always-answer rule still hits.
    request = _make_request("What test framework should I use?")
    answer = asyncio.run(coord.decide_clarification(
        session.session_id, request, store.get(session.session_id),
    ))
    assert "pytest" in answer.lower() or "test framework" in answer.lower()


def test_decide_clarification_facts_lookup_signature_compat(tmp_path):
    """A facts_lookup that only accepts a single positional argument
    (no kwargs) must still work -- the coordinator falls back to the
    bare-call path."""

    captured = []

    def _legacy_lookup(query):
        captured.append(query)
        return [{
            "fact": "user prefers FastAPI",
            "confidence": 0.95,
            "category": "preference",
            "score": 0.95,
            "last_confirmed": time.time(),
        }]

    store = SessionStore()
    project = tmp_path / "project"
    project.mkdir()
    session = store.create(
        project_root=project, user_intent="Build a service",
        mode="new", model="haiku",
    )
    store.transition(session.session_id, SessionStatus.EXECUTING)

    llm = _FakeLLM()
    coord = ConversationCoordinator(
        store=store, llm=llm,
        log_path=tmp_path / "clarifs.jsonl",
        clarification_user_timeout_s=1.0,
        facts_lookup=_legacy_lookup,
    )

    request = _make_request("FastAPI or Starlette?")
    answer = asyncio.run(coord.decide_clarification(
        session.session_id, request, store.get(session.session_id),
    ))
    assert "FastAPI" in answer
    assert captured  # legacy positional-only signature exercised


def test_decision_path_enum_includes_fact_answer():
    """B1/A3 sanity: the new enum value is exposed on DecisionPath."""
    assert DecisionPath.FACT_ANSWER.value == "fact_answer"
