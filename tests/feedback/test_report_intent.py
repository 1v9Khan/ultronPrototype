"""Tests for the report-a-concern voice-intent matcher (T12 wiring)."""

from __future__ import annotations

import pytest

from ultron.feedback.report_intent import (
    ReportConcernMatch,
    match_report_concern,
)
from ultron.feedback.report_queue import ReportTargetKind


class TestMatches:
    @pytest.mark.parametrize(
        "text",
        [
            "log a concern",
            "log a concern that the last response was wrong",
            "file a concern about that answer",
            "raise a concern",
            "report that response",
            "report that last answer",
            "report that",
            "flag that response",
            "flag the last answer",
            "that response was wrong",
            "that answer was incorrect",
            "that reply was unhelpful",
            "that was a bad answer",
            "that was a wrong response",
        ],
    )
    def test_trigger_phrases_match(self, text: str) -> None:
        m = match_report_concern(text)
        assert m is not None, text
        assert isinstance(m, ReportConcernMatch)
        assert m.reason == text

    def test_default_target_is_response(self) -> None:
        m = match_report_concern("flag that response")
        assert m is not None
        assert m.target_kind is ReportTargetKind.RESPONSE

    def test_memory_hint_targets_memory(self) -> None:
        m = match_report_concern(
            "log a concern that you misremembered what I told you"
        )
        assert m is not None
        assert m.target_kind is ReportTargetKind.MEMORY

    def test_reason_is_verbatim(self) -> None:
        text = "  log a concern that the answer cited a fake source  "
        m = match_report_concern(text)
        assert m is not None
        # Stripped, but otherwise verbatim.
        assert m.reason == text.strip()


class TestNonMatches:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "give me a report on the weather",
            "report on the quarterly numbers",
            "what's the weather report",
            "can you flag this email as important",  # no output-noun anchor
            "answer my question about taxes",
            "respond to this",
            "tell me about concerns over climate change",
            "that movie was wrong about history",  # not response/answer/reply
        ],
    )
    def test_benign_phrases_do_not_match(self, text: str) -> None:
        assert match_report_concern(text) is None
