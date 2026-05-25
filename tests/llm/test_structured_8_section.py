"""Tests for ultron.llm.condensers.structured_8_section."""

from __future__ import annotations

import pytest

from ultron.llm.condensers import structured_8_section as s8


SAMPLE_SUMMARY = """\
## Primary Request and Intent
The user wants to refactor the auth middleware to drop session tokens
in favour of JWTs.

## Key Technical Concepts
* JWT vs session tokens
* Middleware injection
* Token rotation

## Files and Code Sections
* src/auth/middleware.py — main auth logic
* tests/test_auth.py — coverage

## Problem Solving
We hit a circular import; fixed by moving the JWT helper to
src/auth/jwt_helper.py.

## Pending Tasks
* Wire the new helper into the existing tests
* Update the docs

## Task Evolution
* Original Request — drop session tokens.
* Modifications — also rotate keys daily.
* Current Scope — JWT helper + tests + docs.
* Context for Changes — legal flagged session-token storage.

## Current Work
Updating tests/test_auth.py to expect the new JWT payload shape.

## Next Step
Re-run the test sweep and confirm green.
"""


# ---------------------------------------------------------------------------
# parse_summary
# ---------------------------------------------------------------------------

class TestParseSummary:
    def test_parses_all_sections(self) -> None:
        out = s8.parse_summary(SAMPLE_SUMMARY)
        for header in s8.SECTION_HEADERS:
            assert header in out.sections
        assert out.has_all_required is True
        assert out.missing == ()

    def test_handles_empty_input(self) -> None:
        out = s8.parse_summary("")
        assert out.sections == {}
        assert out.missing == s8.SECTION_HEADERS

    def test_reports_missing_sections(self) -> None:
        partial = "## Primary Request and Intent\nFoo\n\n## Pending Tasks\nBar\n"
        out = s8.parse_summary(partial)
        assert "Primary Request and Intent" in out.sections
        assert "Pending Tasks" in out.sections
        assert "Current Work" in out.missing

    def test_accepts_h3_headers(self) -> None:
        text = "### Primary Request and Intent\nFoo\n### Next Step\nBar\n"
        out = s8.parse_summary(text)
        assert out.sections["Primary Request and Intent"] == "Foo"
        assert out.sections["Next Step"] == "Bar"

    def test_alias_headers_resolved(self) -> None:
        text = "## Next Steps\nDo this\n## Files\nfoo.py\n"
        out = s8.parse_summary(text)
        # Aliases are mapped to canonical headers.
        assert "Next Step" in out.sections
        assert "Files and Code Sections" in out.sections

    def test_later_header_with_body_overwrites_earlier(self) -> None:
        text = (
            "## Primary Request and Intent\n\n"
            "## Primary Request and Intent\nReal content here\n"
        )
        out = s8.parse_summary(text)
        assert out.sections["Primary Request and Intent"] == "Real content here"


# ---------------------------------------------------------------------------
# compact_for_voice
# ---------------------------------------------------------------------------

class TestCompactForVoice:
    def test_renders_three_sections(self) -> None:
        parsed = s8.parse_summary(SAMPLE_SUMMARY)
        voice = s8.compact_for_voice(parsed)
        assert "Picking up where we left off" in voice
        assert "Pending" in voice
        assert "Next" in voice

    def test_empty_parsed_returns_intro_only(self) -> None:
        parsed = s8.parse_summary("")
        voice = s8.compact_for_voice(parsed)
        assert voice == s8.VOICE_INTRO_SENTENCE

    def test_max_chars_cap(self) -> None:
        parsed = s8.parse_summary(SAMPLE_SUMMARY)
        voice = s8.compact_for_voice(parsed, max_chars=80)
        assert len(voice) <= 80


# ---------------------------------------------------------------------------
# StructuredEightSectionCondenser
# ---------------------------------------------------------------------------

class TestCondenser:
    def test_no_summarize_fn_returns_passthrough(self) -> None:
        condenser = s8.StructuredEightSectionCondenser()
        turns = [("user", "hi"), ("assistant", "hello")]
        result = condenser.condense(turns)
        assert result.turns == turns
        assert result.summary_inserted is False
        assert result.error is not None

    def test_empty_turns_no_op(self) -> None:
        condenser = s8.StructuredEightSectionCondenser(
            summarize_fn=lambda _p, _t: "summary",
        )
        result = condenser.condense([])
        assert result.turns == []
        assert result.dropped_turn_count == 0

    def test_summarizes_with_tail_preserved(self) -> None:
        def summariser(prompt: str, body: str) -> str:
            return SAMPLE_SUMMARY

        condenser = s8.StructuredEightSectionCondenser(
            summarize_fn=summariser,
            keep_tail_turns=1,
        )
        turns = [
            ("user", "first"),
            ("assistant", "ack"),
            ("user", "second"),
            ("assistant", "another"),
            ("user", "third"),
        ]
        result = condenser.condense(turns)
        assert result.summary_inserted is True
        # First turn should be the summary; last turn should be preserved.
        assert result.turns[0][0] == "system"
        assert "Primary Request and Intent" in result.turns[0][1]
        assert result.turns[-1] == ("user", "third")
        assert result.dropped_turn_count == 4

    def test_summariser_exception_returns_error(self) -> None:
        def boom(_p: str, _t: str) -> str:
            raise RuntimeError("upstream LLM down")

        condenser = s8.StructuredEightSectionCondenser(summarize_fn=boom)
        result = condenser.condense([("user", "hi")])
        assert result.summary_inserted is False
        assert result.error is not None
        assert "RuntimeError" in result.error

    def test_empty_summary_returns_error(self) -> None:
        condenser = s8.StructuredEightSectionCondenser(
            summarize_fn=lambda _p, _t: "   ",
        )
        result = condenser.condense([("user", "hi")])
        assert result.summary_inserted is False
        assert result.error == "empty summary"

    def test_missing_sections_recorded_in_notes(self) -> None:
        partial = "## Primary Request and Intent\nFoo\n"
        condenser = s8.StructuredEightSectionCondenser(
            summarize_fn=lambda _p, _t: partial,
            keep_tail_turns=0,
        )
        result = condenser.condense([("user", "hi"), ("assistant", "ack")])
        assert any("missing" in n.lower() for n in result.notes)

    def test_keep_tail_zero_drops_everything(self) -> None:
        condenser = s8.StructuredEightSectionCondenser(
            summarize_fn=lambda _p, _t: SAMPLE_SUMMARY,
            keep_tail_turns=0,
        )
        turns = [("user", "a"), ("assistant", "b"), ("user", "c")]
        result = condenser.condense(turns)
        assert len(result.turns) == 1
        assert result.dropped_turn_count == 3


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_condenser() -> None:
    condenser = s8.build_structured_8_section_condenser(
        summarize_fn=lambda _p, _t: SAMPLE_SUMMARY,
    )
    assert isinstance(condenser, s8.StructuredEightSectionCondenser)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_section_headers_canonical_count() -> None:
    assert len(s8.SECTION_HEADERS) == 8


def test_voice_intro_is_sentence() -> None:
    assert s8.VOICE_INTRO_SENTENCE.endswith(".")
