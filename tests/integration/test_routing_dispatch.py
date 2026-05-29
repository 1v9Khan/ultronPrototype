"""Integration test category 5 — routing-stub dispatch through the controller.

Verifies the full dispatch path:

  utterance → classify_routing() → CapabilityVoiceController.handle_capability_intent()
            → OpenClawDispatcher (stub) → VoiceResponse with stub message
            → routing-decision log entry

Each test uses the shared ``cap_stack`` and ``dispatch_utterance`` helper
from conftest. No real model loads.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import dispatch_utterance


# ---------------------------------------------------------------------------
# 10 routing-stub utterances per the spec
# ---------------------------------------------------------------------------


def test_browser_navigate_dispatches_stub(cap_stack, routing_log, read_routing):
    response = dispatch_utterance(cap_stack, "open hacker news")
    assert response is not None
    assert response.handled is True
    assert "page" in response.text.lower() or "gateway" in response.text.lower()
    rec = read_routing()[-1]
    assert rec["intent"] == "browser_automation"
    assert rec["outcome"] == "stub"
    assert "OpenClaw" in rec.get("stub_reason", "")


def test_media_generation_dispatches_stub(cap_stack, routing_log, read_routing):
    response = dispatch_utterance(cap_stack, "make me an image of a cat")
    assert response is not None
    assert "generate" in response.text.lower() or "gateway" in response.text.lower()
    rec = read_routing()[-1]
    assert rec["intent"] == "media_generation"
    assert rec["outcome"] == "stub"


def test_messaging_dispatches_stub(cap_stack, routing_log, read_routing):
    response = dispatch_utterance(cap_stack, "send a message to my phone")
    assert response is not None
    assert "send" in response.text.lower()
    rec = read_routing()[-1]
    assert rec["intent"] == "messaging"
    assert rec["outcome"] == "stub"


def test_file_operation_dispatches_stub(cap_stack, routing_log, read_routing):
    response = dispatch_utterance(cap_stack, "read the file at C:/test.txt")
    assert response is not None
    assert "files" in response.text.lower()
    rec = read_routing()[-1]
    assert rec["intent"] == "file_operation"
    assert rec["outcome"] == "stub"


def test_shell_operation_dispatches_stub(cap_stack, routing_log, read_routing):
    response = dispatch_utterance(cap_stack, "run dir on the desktop")
    assert response is not None
    assert "shell" in response.text.lower() or "command" in response.text.lower()
    rec = read_routing()[-1]
    assert rec["intent"] == "shell_operation"
    assert rec["outcome"] == "stub"


def test_hybrid_task_decomposes_and_dispatches(cap_stack, routing_log, read_routing):
    """HYBRID_TASK now runs through the HybridTaskDecomposer instead of a
    stale "gateway isn't connected" stub. With no real LLM in the fixture the
    decomposer falls back to a coding-only plan, which dispatches through the
    coding pipeline; the routing log records handler=HybridTaskDecomposer and
    outcome=decomposed."""
    response = dispatch_utterance(
        cap_stack, "set up a development environment for this project",
    )
    assert response is not None
    assert response.handled is True
    rec = read_routing()[-1]
    assert rec["intent"] == "hybrid_task"
    assert rec["handler"] == "HybridTaskDecomposer"
    assert rec["outcome"] == "decomposed"


def test_hybrid_excel_workflow_dispatches_stub(cap_stack, routing_log, read_routing):
    response = dispatch_utterance(cap_stack, "automate my excel workflow")
    assert response is not None
    rec = read_routing()[-1]
    assert rec["intent"] == "hybrid_task"


def test_hybrid_browser_script_dispatches_stub(cap_stack, routing_log, read_routing):
    response = dispatch_utterance(
        cap_stack, "write a script that opens chrome and clicks the login button",
    )
    assert response is not None
    rec = read_routing()[-1]
    assert rec["intent"] == "hybrid_task"


# ---------------------------------------------------------------------------
# Phase 13 — SYSTEM_STATUS voice handler routes through the bridge's reporter
# ---------------------------------------------------------------------------


def test_system_status_alerts_handled(cap_stack, routing_log, read_routing):
    """`what alerts did you flag` returns a voice response (no OpenClaw call)."""
    response = dispatch_utterance(cap_stack, "what alerts did you flag")
    assert response is not None
    assert response.handled is True
    # Either a real alert summary or "no pending alerts" — both are acceptable.
    msg = response.text.lower()
    assert "alert" in msg or "pending" in msg
    rec = read_routing()[-1]
    assert rec["intent"] == "system_status"
    assert rec["handler"] == "voice.system_status"


def test_system_status_projects_handled(cap_stack, routing_log, read_routing):
    """`what is Ultron working on` returns a voice response."""
    response = dispatch_utterance(cap_stack, "what is Ultron working on")
    assert response is not None
    assert response.handled is True
    rec = read_routing()[-1]
    assert rec["intent"] == "system_status"
    # Empty workspace → "Nothing active." is the canonical reply.
    msg = response.text.lower()
    assert "nothing active" in msg or "active" in msg or "session" in msg


def test_system_status_combined_handled(cap_stack, routing_log, read_routing):
    response = dispatch_utterance(cap_stack, "status report")
    assert response is not None
    assert response.handled is True
    rec = read_routing()[-1]
    assert rec["intent"] == "system_status"
    assert rec["handler"] == "voice.system_status"


def test_system_status_in_ultron_voice(cap_stack, routing_log):
    """Spot-check: status responses don't use forbidden filler phrases."""
    for utt in (
        "what alerts did you flag",
        "what is Ultron working on",
        "status report",
    ):
        response = dispatch_utterance(cap_stack, utt)
        assert response is not None
        msg = response.text.lower()
        for banned in (
            "certainly", "of course", "happy to",
            "i'd be happy", "absolutely",
        ):
            assert banned not in msg, (
                f"banned phrase {banned!r} in status response: {response.text!r}"
            )


# ---------------------------------------------------------------------------
# Stub voice quality: every stub message stays in Ultron's voice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utt", [
    "open wikipedia",
    "make me a song about coding",
    "text me when the build is done",
    "list the files in C:/Projects",
    "run git status",
])
def test_stub_voice_messages_in_character(cap_stack, routing_log, utt):
    """The stub voice messages defined in OpenClawDispatcher must NOT
    contain banned filler ('certainly', 'of course', 'happy to', etc.)
    per Ultron's system prompt."""
    response = dispatch_utterance(cap_stack, utt)
    assert response is not None
    msg = response.text.lower()
    banned = [
        "certainly", "of course", "happy to",
        "i'd be happy", "i'd love to", "absolutely",
    ]
    for phrase in banned:
        assert phrase not in msg, (
            f"banned phrase {phrase!r} in stub response: {response.text!r}"
        )


# ---------------------------------------------------------------------------
# Pass-through behavior: CONVERSATIONAL utterances return None, NOT a stub
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utt", [
    "good morning",
    "what is the boiling point of water",
    "tell me a joke",
    "I'm tired",
    "thanks",
])
def test_conversational_falls_through(cap_stack, routing_log, read_routing, utt):
    """CONVERSATIONAL utterances return None — orchestrator handles
    them via the normal LLM/TTS path, NOT via the dispatcher."""
    response = dispatch_utterance(cap_stack, utt)
    assert response is None, (
        f"expected None for conversational {utt!r}, got: {response}"
    )
    rec = read_routing()[-1]
    assert rec["intent"] == "conversational"
    assert rec["outcome"] == "passthrough"


# ---------------------------------------------------------------------------
# Routing-decision log shape verification
# ---------------------------------------------------------------------------


def test_routing_log_records_full_metadata(cap_stack, routing_log, read_routing):
    dispatch_utterance(cap_stack, "open hacker news")
    rec = read_routing()[-1]
    expected_keys = {
        "timestamp", "utterance", "intent", "confidence", "source",
        "reason", "rule_based", "handler", "outcome",
        "needs_clarification", "clarification_question",
    }
    assert expected_keys.issubset(rec.keys())
    assert rec["rule_based"] is True
    assert rec["confidence"] > 0


def test_routing_log_appends_across_dispatches(cap_stack, routing_log, read_routing):
    """Multiple dispatches accumulate routing-decision rows in order."""
    utterances = [
        "open hacker news",
        "good morning",
        "make me an image of a sunset",
    ]
    for u in utterances:
        dispatch_utterance(cap_stack, u)
    records = read_routing()
    assert len(records) == 3
    assert records[0]["intent"] == "browser_automation"
    assert records[1]["intent"] == "conversational"
    assert records[2]["intent"] == "media_generation"
