"""4B optimization plan Item 5 — IRMA reformulator + disambiguator integration tests.

The reformulator is a pure-text shaper (no LLM call) that wraps the
disambiguator's input with relevant context. Default-OFF flag gating
means none of these tests change live behaviour — they verify the
shape of the enriched prompt and the disambiguator's plumbing.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ultron.openclaw_routing.disambiguator import IntentDisambiguator
from ultron.openclaw_routing.intents import RoutingIntentKind
from ultron.openclaw_routing.irma import (
    InputReformulator,
    RecentDecision,
    ReformulationContext,
    build_default_reformulator,
)


# ---------------------------------------------------------------------------
# InputReformulator — pure-text shape tests
# ---------------------------------------------------------------------------


def test_reformulator_with_no_context_emits_only_utterance() -> None:
    r = InputReformulator()
    out = r.reformulate("open the spreadsheet")
    assert out == 'User utterance: "open the spreadsheet"'


def test_reformulator_strips_whitespace() -> None:
    r = InputReformulator()
    out = r.reformulate("  open the spreadsheet  \n")
    assert out == 'User utterance: "open the spreadsheet"'


def test_reformulator_escapes_quotes_in_utterance() -> None:
    r = InputReformulator()
    out = r.reformulate('she said "yes"')
    # Use single quotes inside so the surrounding doubles aren't broken.
    assert "she said 'yes'" in out
    assert '"yes"' not in out


def test_reformulator_emits_recent_decisions() -> None:
    r = InputReformulator(max_recent=3)
    ctx = ReformulationContext(
        recent=[
            RecentDecision(kind="browser_automation", handler="d", outcome="stub", raw_text_excerpt="open hacker news"),
            RecentDecision(kind="file_operation", handler="d", outcome="stub", raw_text_excerpt="list files"),
            RecentDecision(kind="conversational", handler="voice", outcome="passthrough"),
        ],
    )
    out = r.reformulate("open the spreadsheet", ctx)
    assert "Recent decisions (last 3):" in out
    assert "browser_automation handled=stub" in out
    assert 'for "open hacker news"' in out
    assert "conversational handled=passthrough" in out


def test_reformulator_truncates_recent_to_max() -> None:
    r = InputReformulator(max_recent=2)
    ctx = ReformulationContext(
        recent=[
            RecentDecision(kind=f"k{i}", handler="h", outcome="o", raw_text_excerpt=f"u{i}")
            for i in range(5)
        ],
    )
    out = r.reformulate("u", ctx)
    assert "Recent decisions (last 2):" in out
    # Most recent ones — k3, k4 — must appear; k0 / k1 must not.
    assert "k4" in out
    assert "k3" in out
    assert "k0" not in out
    assert "k1" not in out


def test_reformulator_emits_active_session() -> None:
    r = InputReformulator()
    ctx = ReformulationContext(
        active_session_summary="coding task running ('flask app')",
    )
    out = r.reformulate("look at it", ctx)
    assert "Active session: coding task running ('flask app')" in out


def test_reformulator_emits_routing_hints() -> None:
    r = InputReformulator()
    ctx = ReformulationContext(
        routing_hints=[
            "'open' historically maps to BROWSER, not FILE",
            "user prefers single-tab navigation",
        ],
    )
    out = r.reformulate("open it", ctx)
    assert "Routing hints:" in out
    assert "- 'open' historically maps to BROWSER, not FILE" in out
    assert "- user prefers single-tab navigation" in out


def test_reformulator_max_recent_zero_omits_section() -> None:
    r = InputReformulator(max_recent=0)
    ctx = ReformulationContext(
        recent=[RecentDecision(kind="k", handler="h", outcome="o")],
    )
    out = r.reformulate("u", ctx)
    assert "Recent decisions" not in out


def test_recent_decision_from_log_row() -> None:
    row = {
        "intent_kind": "browser_automation",
        "handler": "OpenClawDispatcher.handle_browser",
        "outcome": "stub",
        "raw_text": "open hacker news please",
    }
    rd = RecentDecision.from_log_row(row)
    assert rd.kind == "browser_automation"
    assert rd.handler == "OpenClawDispatcher.handle_browser"
    assert rd.outcome == "stub"
    assert rd.raw_text_excerpt == "open hacker news please"


# ---------------------------------------------------------------------------
# Default factory — reads from config
# ---------------------------------------------------------------------------


def test_build_default_reformulator_uses_cfg() -> None:
    cfg = MagicMock()
    cfg.routing.irma.max_recent_decisions = 7
    r = build_default_reformulator(cfg)
    assert r._max_recent == 7  # noqa: SLF001 — internal probe


# ---------------------------------------------------------------------------
# Disambiguator integration — flag-gated
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_mock():
    m = MagicMock()
    m.generate.return_value = "CODING"
    return m


@pytest.mark.asyncio
async def test_disambiguator_default_off_uses_raw_utterance(llm_mock) -> None:
    """No reformulator + irma.enabled=False ⇒ raw utterance flows through
    unchanged. Back-compat guarantee."""
    d = IntentDisambiguator(llm_mock)
    with patch("ultron.openclaw_routing.disambiguator.get_config") as gc:
        cfg = MagicMock()
        cfg.routing.llm_disambiguation_enabled = True
        cfg.routing.irma.enabled = False
        gc.return_value = cfg
        await d.disambiguate("open the spreadsheet")
    # The utterance went into the prompt verbatim (template's
    # double-quoted slot)
    sent_prompt = llm_mock.generate.call_args.args[0]
    assert 'The user said: "open the spreadsheet"' in sent_prompt
    assert "Recent decisions" not in sent_prompt


@pytest.mark.asyncio
async def test_disambiguator_with_reformulator_but_flag_off_uses_raw(llm_mock) -> None:
    """Reformulator wired but flag off ⇒ behaviour identical to no-reformulator."""
    r = InputReformulator()
    d = IntentDisambiguator(llm_mock, reformulator=r)
    with patch("ultron.openclaw_routing.disambiguator.get_config") as gc:
        cfg = MagicMock()
        cfg.routing.llm_disambiguation_enabled = True
        cfg.routing.irma.enabled = False
        gc.return_value = cfg
        await d.disambiguate("open the spreadsheet")
    sent = llm_mock.generate.call_args.args[0]
    assert 'The user said: "open the spreadsheet"' in sent
    assert "Recent decisions" not in sent


@pytest.mark.asyncio
async def test_disambiguator_with_irma_enabled_uses_enriched_prompt(llm_mock) -> None:
    r = InputReformulator()
    d = IntentDisambiguator(llm_mock, reformulator=r)
    ctx = ReformulationContext(
        recent=[RecentDecision(kind="browser_automation", handler="d", outcome="stub")],
        active_session_summary="no active task",
    )
    with patch("ultron.openclaw_routing.disambiguator.get_config") as gc:
        cfg = MagicMock()
        cfg.routing.llm_disambiguation_enabled = True
        cfg.routing.irma.enabled = True
        gc.return_value = cfg
        await d.disambiguate("open the spreadsheet", irma_context=ctx)
    sent = llm_mock.generate.call_args.args[0]
    assert 'User utterance: "open the spreadsheet"' in sent
    assert "Recent decisions (last 1):" in sent
    assert "browser_automation handled=stub" in sent
    assert "Active session: no active task" in sent
    # The IRMA framing is used, not the legacy template.
    assert "Given the above context" in sent


@pytest.mark.asyncio
async def test_disambiguator_irma_failure_falls_back_to_raw(llm_mock) -> None:
    """If reformulate() raises, the disambiguator must fall back to the
    legacy prompt — never crash on the disambiguator path."""
    r = MagicMock()
    r.reformulate.side_effect = RuntimeError("boom")
    d = IntentDisambiguator(llm_mock, reformulator=r)
    with patch("ultron.openclaw_routing.disambiguator.get_config") as gc:
        cfg = MagicMock()
        cfg.routing.llm_disambiguation_enabled = True
        cfg.routing.irma.enabled = True
        gc.return_value = cfg
        result = await d.disambiguate("open it")
    sent = llm_mock.generate.call_args.args[0]
    # Fell back to legacy template
    assert 'The user said: "open it"' in sent
    # Disambiguator still returned a non-error result
    assert result.kind == RoutingIntentKind.CODE_TASK  # llm_mock returns "CODING"


@pytest.mark.asyncio
async def test_disambiguator_irma_with_no_context_still_emits_utterance(llm_mock) -> None:
    """irma_context=None + reformulator wired + flag on ⇒ enriched
    prompt with just the utterance (no recent / session / hints
    section)."""
    r = InputReformulator()
    d = IntentDisambiguator(llm_mock, reformulator=r)
    with patch("ultron.openclaw_routing.disambiguator.get_config") as gc:
        cfg = MagicMock()
        cfg.routing.llm_disambiguation_enabled = True
        cfg.routing.irma.enabled = True
        gc.return_value = cfg
        await d.disambiguate("open it")
    sent = llm_mock.generate.call_args.args[0]
    assert 'User utterance: "open it"' in sent
    assert "Given the above context" in sent
    assert "Recent decisions" not in sent
