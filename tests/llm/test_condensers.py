"""Tests for the condenser package (OpenHands catalog T4)."""

from __future__ import annotations

import pytest

from ultron.llm.condensers import (
    AmortizedCondenser,
    Condenser,
    CondenseResult,
    CondenserError,
    DEFAULT_CONDENSER_KIND,
    DEFAULT_MASK_TEMPLATE,
    KNOWN_CONDENSER_KINDS,
    LLMSummarizingCondenser,
    NoOpCondenser,
    ObservationMaskingCondenser,
    RecentCondenser,
    Turn,
    build_condenser,
    char_count_tokens_for_turns,
    select_condenser_for_intent,
    turn_text,
)


# -- helpers --


def _history(n: int, prefix: str = "u") -> list[Turn]:
    """Build a list of n turns alternating user/assistant."""

    out: list[Turn] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append((role, f"{prefix}_{i}: " + ("x" * 40)))
    return out


# -- base helpers --


def test_turn_text_extracts_content():
    assert turn_text(("user", "hello")) == "hello"
    assert turn_text(("user", "")) == ""


def test_char_count_tokens_zero_for_empty():
    assert char_count_tokens_for_turns([]) == 0


def test_char_count_tokens_uses_4_chars_per_token():
    # 8 chars / 4 = 2 tokens.
    assert char_count_tokens_for_turns([("user", "abcdefgh")]) == 2


def test_condense_result_ok_property():
    r = CondenseResult(turns=())
    assert r.ok is True
    r2 = CondenseResult(turns=(), error="boom")
    assert r2.ok is False


def test_condenser_abstract():
    with pytest.raises(TypeError):
        Condenser()  # type: ignore[abstract]


# -- NoOpCondenser --


def test_noop_passes_history_through():
    history = _history(5)
    result = NoOpCondenser().condense(history)
    assert result.turns == tuple(history)
    assert result.dropped_turn_count == 0
    assert result.summary_inserted is False
    assert result.token_estimate_before == result.token_estimate_after


def test_noop_empty_history():
    result = NoOpCondenser().condense([])
    assert result.turns == ()
    assert result.dropped_turn_count == 0


def test_noop_kind_stable():
    assert NoOpCondenser().kind == "noop"


# -- RecentCondenser --


def test_recent_short_history_unchanged():
    history = _history(5)
    result = RecentCondenser(keep_first=1, max_events=10).condense(history)
    assert result.dropped_turn_count == 0
    assert result.turns == tuple(history)


def test_recent_drops_middle():
    history = _history(20)
    result = RecentCondenser(keep_first=2, max_events=6).condense(history)
    assert len(result.turns) == 6
    assert result.dropped_turn_count == 14
    # First two preserved.
    assert result.turns[:2] == tuple(history[:2])
    # Last four preserved.
    assert result.turns[-4:] == tuple(history[-4:])


def test_recent_keep_first_zero_means_tail_only():
    history = _history(10)
    result = RecentCondenser(keep_first=0, max_events=3).condense(history)
    assert result.turns == tuple(history[-3:])


def test_recent_invalid_keep_first_raises():
    with pytest.raises(CondenserError):
        RecentCondenser(keep_first=-1)


def test_recent_max_events_less_than_keep_first_raises():
    with pytest.raises(CondenserError):
        RecentCondenser(keep_first=5, max_events=2)


def test_recent_token_estimate_falls_after_compression():
    history = _history(20)
    result = RecentCondenser(keep_first=1, max_events=4).condense(history)
    assert result.token_estimate_after is not None
    assert result.token_estimate_before is not None
    assert result.token_estimate_after < result.token_estimate_before


# -- AmortizedCondenser --


def test_amortized_short_history_unchanged():
    history = _history(5)
    result = AmortizedCondenser(max_size=10).condense(history)
    assert result.dropped_turn_count == 0
    assert result.turns == tuple(history)


def test_amortized_keeps_head_and_drops_to_max_size():
    history = _history(50)
    result = AmortizedCondenser(keep_first=1, max_size=10, max_tokens=10000).condense(history)
    assert result.dropped_turn_count > 0
    # Head preserved.
    assert result.turns[0] == history[0]
    # Result respects max_size cap.
    assert len(result.turns) <= 10


def test_amortized_pin_roles_preserved():
    history = [
        ("system", "task description"),
        ("user", "u1"),
        ("assistant", "a1"),
        ("user", "u2"),
        ("system", "important rule"),
        ("user", "u3"),
        ("assistant", "a2"),
    ]
    result = AmortizedCondenser(
        keep_first=0, max_size=3, max_tokens=10000
    ).condense(history)
    roles = [t[0] for t in result.turns]
    # Both system rows survive even though max_size=3.
    assert roles.count("system") == 2


def test_amortized_chronological_order_preserved():
    history = _history(40)
    result = AmortizedCondenser(keep_first=1, max_size=5, max_tokens=10000).condense(history)
    indices = [history.index(t) for t in result.turns]
    assert indices == sorted(indices)


def test_amortized_invalid_keep_first_raises():
    with pytest.raises(CondenserError):
        AmortizedCondenser(keep_first=-1)


def test_amortized_invalid_max_size_raises():
    with pytest.raises(CondenserError):
        AmortizedCondenser(keep_first=5, max_size=2)


def test_amortized_invalid_max_tokens_raises():
    with pytest.raises(CondenserError):
        AmortizedCondenser(max_tokens=-1)


# -- ObservationMaskingCondenser --


def test_mask_short_history_unchanged():
    history = _history(3)
    result = ObservationMaskingCondenser(attention_window=10).condense(history)
    assert result.turns == tuple(history)


def test_mask_replaces_old_observations():
    history = [
        ("user", "ask"),
        ("tool", "observation body with content"),
        ("assistant", "respond"),
        ("user", "next"),
        ("tool", "more obs"),
        ("assistant", "next response"),
    ]
    result = ObservationMaskingCondenser(
        attention_window=2,
    ).condense(history)
    # window=2 means the last 2 turns are unchanged; older tool turns get masked.
    tail = result.turns[-2:]
    assert tail == tuple(history[-2:])
    # The first tool turn (index 1) is OUTSIDE the window -> masked.
    masked = result.turns[1]
    assert "[Earlier observation" in masked[1]


def test_mask_only_touches_configured_roles():
    history = [
        ("user", "u long"),
        ("assistant", "a long"),
        ("user", "u3"),
        ("assistant", "a3"),
    ]
    result = ObservationMaskingCondenser(
        attention_window=1,
        mask_roles=frozenset({"tool"}),
    ).condense(history)
    # No 'tool' role present -> nothing is masked.
    assert result.turns == tuple(history)


def test_mask_default_template_includes_char_count():
    history = [("tool", "x" * 200)] + _history(10)
    result = ObservationMaskingCondenser(
        attention_window=3,
    ).condense(history)
    # The leading tool message should be masked + template includes the chars count.
    assert "200" in result.turns[0][1]


def test_mask_invalid_attention_window_raises():
    with pytest.raises(CondenserError):
        ObservationMaskingCondenser(attention_window=-1)


def test_mask_default_template_constant():
    assert "{chars}" in DEFAULT_MASK_TEMPLATE


# -- LLMSummarizingCondenser --


def test_summary_short_history_unchanged():
    history = _history(5)
    result = LLMSummarizingCondenser(
        summarize_fn=lambda t: "summary",
        max_size=10,
    ).condense(history)
    assert result.dropped_turn_count == 0
    assert result.summary_inserted is False


def test_summary_fires_at_threshold():
    history = _history(20)
    captured: list[str] = []

    def _fn(text: str) -> str:
        captured.append(text)
        return "compressed body"

    condenser = LLMSummarizingCondenser(
        summarize_fn=_fn,
        max_size=10,
        keep_first=1,
        keep_last=2,
    )
    result = condenser.condense(history)
    assert result.summary_inserted is True
    # head + summary + tail = 1 + 1 + 2.
    assert len(result.turns) == 4
    # Middle turn is the synthesised summary.
    assert "compressed body" in result.turns[1][1]
    assert len(captured) == 1


def test_summary_no_summarise_fn_falls_back_to_head_tail():
    history = _history(20)
    result = LLMSummarizingCondenser(
        summarize_fn=None,
        max_size=5,
        keep_first=1,
        keep_last=2,
    ).condense(history)
    assert result.summary_inserted is False
    assert result.error == "summarize_fn missing"
    assert len(result.turns) == 3


def test_summary_handles_summariser_exception():
    history = _history(20)

    def _broken(_text: str) -> str:
        raise RuntimeError("network down")

    result = LLMSummarizingCondenser(
        summarize_fn=_broken,
        max_size=5,
        keep_first=1,
        keep_last=2,
    ).condense(history)
    assert result.summary_inserted is False
    assert "summarize_fn raised" in (result.error or "")


def test_summary_empty_returns_fall_back():
    history = _history(20)
    result = LLMSummarizingCondenser(
        summarize_fn=lambda _: "",
        max_size=5,
        keep_first=1,
        keep_last=2,
    ).condense(history)
    assert result.summary_inserted is False
    assert "empty" in (result.error or "")


def test_summary_invalid_keep_negative_raises():
    with pytest.raises(CondenserError):
        LLMSummarizingCondenser(summarize_fn=lambda t: "x", keep_first=-1)


def test_summary_preamble_defaults_when_blank():
    cond = LLMSummarizingCondenser(summarize_fn=lambda t: "x", summary_preamble="")
    assert cond.summary_preamble  # non-empty after __post_init__


# -- factory --


def test_build_condenser_known_kinds():
    for kind in KNOWN_CONDENSER_KINDS:
        cond = build_condenser(kind)
        assert isinstance(cond, Condenser)
        assert cond.kind == kind


def test_build_condenser_default_kind_is_noop():
    assert DEFAULT_CONDENSER_KIND == "noop"
    assert isinstance(build_condenser(DEFAULT_CONDENSER_KIND), NoOpCondenser)


def test_build_condenser_aliases():
    assert isinstance(build_condenser("off"), NoOpCondenser)
    assert isinstance(build_condenser("summary"), LLMSummarizingCondenser)
    assert isinstance(build_condenser("mask"), ObservationMaskingCondenser)


def test_build_condenser_unknown_raises():
    with pytest.raises(CondenserError):
        build_condenser("bogus_kind")


def test_build_condenser_passes_knobs():
    cond = build_condenser("recent", keep_first=3, max_events=15)
    assert isinstance(cond, RecentCondenser)
    assert cond.keep_first == 3
    assert cond.max_events == 15


# -- intent selector --


def test_select_for_intent_known_returns_strategy():
    cond = select_condenser_for_intent("factual")
    assert isinstance(cond, RecentCondenser)
    cond_g = select_condenser_for_intent("greeting")
    assert isinstance(cond_g, NoOpCondenser)


def test_select_for_intent_unknown_uses_default():
    cond = select_condenser_for_intent("super_obscure_intent_kind")
    assert isinstance(cond, RecentCondenser)


def test_select_for_intent_none_uses_fallback():
    fallback = AmortizedCondenser()
    cond = select_condenser_for_intent(None, fallback=fallback)
    assert cond is fallback


def test_select_for_intent_passes_summarize_fn_when_relevant():
    def _fn(text: str) -> str:
        return "ok"

    cond = select_condenser_for_intent("coding", summarize_fn=_fn)
    assert isinstance(cond, LLMSummarizingCondenser)
    assert cond.summarize_fn is _fn


def test_select_for_intent_case_insensitive():
    cond_a = select_condenser_for_intent("CODING")
    cond_b = select_condenser_for_intent("coding")
    assert type(cond_a) is type(cond_b)


# -- catalog 09 batch G: RoutingIntentKind coverage --


@pytest.mark.parametrize(
    "intent_value,expected_cls",
    [
        # Conversational / lightweight voice path -> NoOp (zero-cost
        # passthrough, no churn).
        ("conversational", NoOpCondenser),
        ("greeting", NoOpCondenser),
        ("ack", NoOpCondenser),
        ("progress_query", NoOpCondenser),
        ("cancel", NoOpCondenser),
        ("mid_session_adjustment", NoOpCondenser),
        ("clarification_response", NoOpCondenser),
        ("model_switch", NoOpCondenser),
        ("gaming_mode", NoOpCondenser),
        ("system_status", NoOpCondenser),
        ("active_window_query", NoOpCondenser),
        ("window_close_confirmation", NoOpCondenser),
        # Factual + memory recall -> Recent / Amortized.
        ("factual", RecentCondenser),
        ("memory_recall", AmortizedCondenser),
        ("gaming", RecentCondenser),
        # Desktop automation / window operations -> Recent.
        ("browser_automation", RecentCondenser),
        ("media_generation", RecentCondenser),
        ("messaging", RecentCondenser),
        ("file_operation", RecentCondenser),
        ("shell_operation", RecentCondenser),
        ("desktop_automation", RecentCondenser),
        ("window_automation", RecentCondenser),
        ("app_launch", RecentCondenser),
        ("screen_context_query", RecentCondenser),
        ("window_move", RecentCondenser),
        ("window_close", RecentCondenser),
        ("open_last_source", RecentCondenser),
        ("navigate_to_site", RecentCondenser),
        ("semantic_click", RecentCondenser),
        # Coding path -> LLMSummarizing.
        ("coding", LLMSummarizingCondenser),
        ("refactor", LLMSummarizingCondenser),
        ("code_task", LLMSummarizingCondenser),
        ("hybrid_task", LLMSummarizingCondenser),
    ],
)
def test_select_for_intent_routing_intent_kinds(
    intent_value, expected_cls,
):
    """Each :class:`RoutingIntentKind` value maps to a documented
    condenser. Pin every value so an accidental rename of an enum
    member shows up here as a failure."""
    cond = select_condenser_for_intent(intent_value)
    assert isinstance(cond, expected_cls), (
        f"intent={intent_value!r} mapped to {type(cond).__name__}, "
        f"expected {expected_cls.__name__}"
    )


def test_select_for_intent_covers_every_routing_intent_kind():
    """Every value of :class:`RoutingIntentKind` MUST resolve to a
    non-default condenser (i.e. is in ``_INTENT_KIND_MAP`` explicitly).
    A new enum member that drifts past this guard would silently fall
    to the ``default`` (``recent``) entry, masking missed wiring."""
    from ultron.llm.condensers.factory import _INTENT_KIND_MAP
    from ultron.openclaw_routing.intents import RoutingIntentKind

    for member in RoutingIntentKind:
        assert member.value in _INTENT_KIND_MAP, (
            f"RoutingIntentKind.{member.name} ({member.value!r}) missing "
            "from _INTENT_KIND_MAP -- add an explicit entry"
        )
