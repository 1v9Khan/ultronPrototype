"""Tests for ANSI / control-character sanitisation + grapheme-width.

T18 (OpenClaw catalog port). Tests pure functions with no IO / threads /
voice-stack loads. Cap per test < 50 ms.
"""

from __future__ import annotations

import pytest

from ultron.utils.ansi_safe import (
    grapheme_width,
    is_full_width,
    is_zero_width,
    iter_graphemes,
    sanitize_for_log,
    split_graphemes,
    strip_ansi,
    truncate_to_visible_width,
    visible_width,
)


# ----------------------------------------------------------------------
# strip_ansi


def test_strip_ansi_removes_csi_color_codes() -> None:
    text = "\x1b[31mred\x1b[0m"
    assert strip_ansi(text) == "red"


def test_strip_ansi_removes_csi_cursor_moves() -> None:
    text = "before\x1b[2K\x1b[1Gafter"
    assert strip_ansi(text) == "beforeafter"


def test_strip_ansi_removes_osc_with_bel_terminator() -> None:
    text = "title\x1b]0;window title\x07body"
    assert strip_ansi(text) == "titlebody"


def test_strip_ansi_removes_osc_with_st_terminator() -> None:
    text = "title\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\done"
    assert strip_ansi(text) == "titlelinkdone"


def test_strip_ansi_preserves_plain_text() -> None:
    text = "hello world\nfoo\tbar"
    assert strip_ansi(text) == text


def test_strip_ansi_empty_input() -> None:
    assert strip_ansi("") == ""


def test_strip_ansi_handles_two_byte_esc() -> None:
    text = "before\x1b@after"
    assert strip_ansi(text) == "beforeafter"


def test_strip_ansi_preserves_tab_lf_cr() -> None:
    text = "\tline1\nline2\r"
    assert strip_ansi(text) == "\tline1\nline2\r"


# ----------------------------------------------------------------------
# sanitize_for_log


def test_sanitize_for_log_strips_ansi_and_control_chars() -> None:
    text = "\x1b[31mERROR\x1b[0m\x00\x01injected\x7f"
    assert sanitize_for_log(text) == "ERRORinjected"


def test_sanitize_for_log_preserves_tab_lf_cr() -> None:
    text = "\tindented\nnewline\rreturn"
    assert sanitize_for_log(text) == "\tindented\nnewline\rreturn"


def test_sanitize_for_log_strips_c1_controls() -> None:
    # NEL (U+0085) and other C1 controls (0x80-0x9F).
    text = "before\x85after\x9b"
    assert sanitize_for_log(text) == "beforeafter"


def test_sanitize_for_log_strips_del() -> None:
    text = "before\x7fafter"
    assert sanitize_for_log(text) == "beforeafter"


def test_sanitize_for_log_log_forging_defense() -> None:
    # CWE-117: attacker injects a fake log line via ANSI cursor +
    # synthetic timestamp. Both must be stripped.
    attack = "user said hi\x1b[2K\x1b[1G2025-01-01 ERROR fake log line"
    cleaned = sanitize_for_log(attack)
    assert "\x1b" not in cleaned
    assert "user said hi2025-01-01 ERROR fake log line" == cleaned


def test_sanitize_for_log_empty_input() -> None:
    assert sanitize_for_log("") == ""


def test_sanitize_for_log_plain_unicode_passes_through() -> None:
    text = "Café 中文 \U0001f600"
    assert sanitize_for_log(text) == text


def test_sanitize_for_log_module_source_is_ascii_safe() -> None:
    """Sanity-check: the module body uses no literal control chars."""
    from pathlib import Path
    src = Path(__file__).parent.parent.parent / "src" / "ultron" / "utils" / "ansi_safe.py"
    body = src.read_text(encoding="utf-8")
    for char in body:
        cp = ord(char)
        if 0 <= cp <= 0x1F and char not in ("\t", "\n", "\r"):
            pytest.fail(f"control char U+{cp:04X} found in source")


# ----------------------------------------------------------------------
# is_zero_width / is_full_width


def test_is_zero_width_combining_diacritic() -> None:
    # U+0301 COMBINING ACUTE ACCENT
    assert is_zero_width(0x0301)


def test_is_zero_width_zwj() -> None:
    # U+200D ZERO WIDTH JOINER
    assert is_zero_width(0x200D)


def test_is_zero_width_variation_selector() -> None:
    # U+FE0F VARIATION SELECTOR-16
    assert is_zero_width(0xFE0F)


def test_is_zero_width_returns_false_for_printable_ascii() -> None:
    assert not is_zero_width(ord("A"))


def test_is_full_width_cjk_ideograph() -> None:
    # U+4E2D '中'
    assert is_full_width(0x4E2D)


def test_is_full_width_emoji_smile() -> None:
    # U+1F600 '😀'
    assert is_full_width(0x1F600)


def test_is_full_width_returns_false_for_ascii_letter() -> None:
    assert not is_full_width(ord("A"))


# ----------------------------------------------------------------------
# split_graphemes / grapheme_width / visible_width


def test_split_graphemes_ascii_round_trip() -> None:
    assert split_graphemes("abc") == ["a", "b", "c"]


def test_split_graphemes_empty_input() -> None:
    assert split_graphemes("") == []


def test_split_graphemes_combining_mark_folds_into_base() -> None:
    # 'e' + U+0301 (COMBINING ACUTE ACCENT) -> single cluster
    text = "é"
    clusters = split_graphemes(text)
    assert clusters == ["é"]


def test_split_graphemes_emoji_with_vs16() -> None:
    # U+2764 HEAVY BLACK HEART + U+FE0F VARIATION SELECTOR-16
    text = "❤️"
    clusters = split_graphemes(text)
    assert clusters == ["❤️"]


def test_grapheme_width_ascii_letter() -> None:
    assert grapheme_width("A") == 1


def test_grapheme_width_combining_diacritic_only() -> None:
    # Standalone combining mark — no printable base.
    assert grapheme_width("́") == 0


def test_grapheme_width_emoji_double() -> None:
    assert grapheme_width("\U0001f600") == 2


def test_grapheme_width_cjk_double() -> None:
    assert grapheme_width("中") == 2


def test_grapheme_width_empty_cluster() -> None:
    assert grapheme_width("") == 0


def test_visible_width_strips_ansi_first() -> None:
    text = "\x1b[31mhello\x1b[0m"
    assert visible_width(text) == 5


def test_visible_width_combining_mark_does_not_count_twice() -> None:
    # "café" with combining mark — width 4, not 5.
    text = "café"
    assert visible_width(text) == 4


def test_visible_width_emoji_counted_as_two() -> None:
    text = "hi \U0001f600"
    assert visible_width(text) == 5


def test_visible_width_empty_input() -> None:
    assert visible_width("") == 0


def test_iter_graphemes_yields_one_at_a_time() -> None:
    items = list(iter_graphemes("ab中\U0001f600"))
    assert items == ["a", "b", "中", "\U0001f600"]


def test_iter_graphemes_empty_input_yields_nothing() -> None:
    assert list(iter_graphemes("")) == []


# ----------------------------------------------------------------------
# truncate_to_visible_width


def test_truncate_to_visible_width_under_budget_returns_unchanged() -> None:
    assert truncate_to_visible_width("hello", 10) == "hello"


def test_truncate_to_visible_width_over_budget_adds_ellipsis() -> None:
    result = truncate_to_visible_width("hello world this is long", 10)
    assert visible_width(result) <= 10
    assert result.endswith("...")


def test_truncate_to_visible_width_respects_emoji_width() -> None:
    # 3 emoji at width 2 each = 6; budget 5 should drop the last.
    text = "\U0001f600\U0001f601\U0001f602"
    result = truncate_to_visible_width(text, 5, ellipsis="")
    assert visible_width(result) <= 5


def test_truncate_to_visible_width_zero_budget_empty() -> None:
    assert truncate_to_visible_width("hello", 0) == ""


def test_truncate_to_visible_width_negative_budget_empty() -> None:
    assert truncate_to_visible_width("hello", -1) == ""


def test_truncate_to_visible_width_strips_ansi_before_counting() -> None:
    text = "\x1b[31mhello world\x1b[0m"
    result = truncate_to_visible_width(text, 8)
    assert "\x1b" not in result


def test_truncate_to_visible_width_no_ellipsis_when_set_empty() -> None:
    result = truncate_to_visible_width("hello world", 5, ellipsis="")
    assert result == "hello"
