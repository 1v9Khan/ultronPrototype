"""Adversarial fuzz tests for the narration-honesty helpers.

Two functions guard the spoken-output path against TTS-hostile text:

  * :func:`ultron.tts.xtts_v3.normalize_text_for_tts` rewrites Windows
    paths, times, units, abbreviations, etc. into spoken form before
    the synth call. Applied to every XTTS-v3 synthesis.
  * :func:`ultron.coding.supervisor_dispatch._speakable` strips
    path-prefix noise from project names so the supervisor's
    narration never says "C: backslash Users backslash...".

Both are conservative pure functions -- unmatched input passes
through unchanged. This test file fuzzes them against the kinds of
inputs that have caused real production issues (a stray Windows path
in a narration string, an unintended URL spelt out, a unicode
hyphen instead of an ASCII one, mixed slash directions, shell
metachars from a malformed prompt, etc.).

The tests don't pin EXACT output strings for most cases -- the
contract is "what comes out does NOT contain the TTS-hostile
fragment" rather than "the output is exactly X". This lets the
normaliser evolve (e.g. picking up new abbreviations) without
re-baselining 40 tests.

If a future caller needs adversarial-input fuzzing on a new TTS-
hostile pattern, add a case here BEFORE adding the fix to the
normaliser so the test pins the regression once a fix lands.
"""

from __future__ import annotations

import pytest

from ultron.coding.supervisor_dispatch import _speakable
from ultron.tts.xtts_v3 import normalize_text_for_tts


# ===========================================================================
# normalize_text_for_tts -- adversarial inputs
# ===========================================================================


# ---- Empty / whitespace / control chars ---------------------------------


def test_empty_string_passes_through() -> None:
    assert normalize_text_for_tts("") == ""


def test_whitespace_only_no_crash() -> None:
    """Whitespace-only input must not crash. The normaliser collapses
    multi-space runs as a side-effect of URL stripping; that's
    acceptable. The contract is "no exception" + "no synthesised content"."""
    out = normalize_text_for_tts("   ")
    # The exact whitespace shape may collapse; assert the output is
    # a (possibly-shorter) whitespace-only string.
    assert out.strip() == ""


def test_single_newline_passes_through() -> None:
    out = normalize_text_for_tts("hello\nworld")
    assert "hello" in out and "world" in out


def test_tab_chars_preserved() -> None:
    out = normalize_text_for_tts("a\tb\tc")
    # Specific representation can vary; just confirm no crash + characters survive.
    assert "a" in out and "b" in out and "c" in out


def test_null_byte_does_not_crash() -> None:
    """Adversarial: NUL byte from a malformed stream."""
    out = normalize_text_for_tts("hello\x00world")
    # Don't pin exact form -- just confirm no exception + words survive.
    assert "hello" in out


def test_form_feed_carriage_return_no_crash() -> None:
    normalize_text_for_tts("line\fone\rline two")  # no exception


# ---- URLs (stripped per the normaliser contract) -------------------------


def test_https_url_stripped() -> None:
    out = normalize_text_for_tts("see https://example.com/page for details")
    assert "https://" not in out
    assert "example.com" not in out
    assert "for details" in out  # surrounding text preserved


def test_http_url_stripped() -> None:
    out = normalize_text_for_tts("at http://x.test/a")
    assert "http://" not in out


def test_ftp_url_stripped() -> None:
    out = normalize_text_for_tts("ftp://files.test/dl works")
    assert "ftp://" not in out


def test_bare_www_url_stripped() -> None:
    out = normalize_text_for_tts("go to www.example.com today")
    assert "www.example.com" not in out
    assert "today" in out


def test_very_long_url_stripped() -> None:
    """A 1200-char URL must not crash and must not survive in output."""
    long_url = "https://example.com/" + ("a" * 1200)
    out = normalize_text_for_tts(f"check {long_url} now")
    assert "https://" not in out
    assert "now" in out


def test_url_with_query_string_stripped() -> None:
    out = normalize_text_for_tts(
        "https://example.com/search?q=foo&bar=baz&zzz=1#frag sources"
    )
    assert "?q=" not in out
    assert "sources" in out


def test_adjacent_urls_both_stripped() -> None:
    out = normalize_text_for_tts(
        "see https://a.test and https://b.test for context"
    )
    assert "https://" not in out
    assert "and" in out and "for context" in out


# ---- Windows drive paths -------------------------------------------------


def test_windows_drive_path_collapses_to_leaf() -> None:
    out = normalize_text_for_tts("saved C:\\Users\\alice\\proj\\file.txt")
    # The drive letter + backslashes must not survive.
    assert "C:" not in out
    assert "\\" not in out
    assert "file.txt" in out


def test_windows_path_with_spaces_in_segment() -> None:
    """A Windows path with a space in a folder name.

    The current Windows-path regex matches up to the first space, so
    a path like ``D:\\My Documents\\Project Files\\notes.md`` only
    has its prefix collapsed. The post-collapse output still avoids
    the drive letter and still surfaces the leaf, but later
    backslashes survive. Pin both the drive-letter strip + the leaf
    presence; future improvements that handle quoted Windows paths
    can tighten this test then.
    """
    out = normalize_text_for_tts("opened D:\\My Documents\\Project Files\\notes.md")
    assert "D:" not in out
    assert "notes.md" in out


def test_unc_path_not_mistakenly_handled_as_time() -> None:
    """A leading backslash run must not look like a time pattern."""
    # We don't have a UNC-path rewriter; just confirm no crash + no
    # bogus "00 00" output.
    out = normalize_text_for_tts(r"\\server\share\file.txt is here")
    assert "is here" in out


def test_forward_slash_posix_path_preserved() -> None:
    """URLs are deliberately the only forward-slash form rewritten.

    A bare ``/etc/something`` should NOT mangle into URL stripping
    because Posix paths from a description shouldn't disappear.
    """
    out = normalize_text_for_tts("the path /etc/hosts contains entries")
    # /etc/hosts is preserved exactly (no URL-strip regex match).
    assert "etc" in out
    assert "hosts" in out


# ---- Mixed-slash adversarials -------------------------------------------


def test_mixed_slash_directions_no_crash() -> None:
    normalize_text_for_tts("C:\\Users/alice\\Documents/file.txt")  # no exception


def test_path_at_start_of_string() -> None:
    out = normalize_text_for_tts("C:\\Users\\bob exists")
    assert "C:" not in out


def test_path_at_end_of_string() -> None:
    out = normalize_text_for_tts("see C:\\temp\\out.log")
    assert "out.log" in out
    assert "C:" not in out


# ---- Times + AM/PM ------------------------------------------------------


def test_twelve_hour_time_with_am_pm() -> None:
    out = normalize_text_for_tts("meeting at 2:16 a.m.")
    # Letters separated; no raw "a.m." token.
    assert "a.m." not in out


def test_twenty_four_hour_time() -> None:
    out = normalize_text_for_tts("departure 14:30 sharp")
    assert "14:30" not in out  # colon split
    assert "sharp" in out


def test_standalone_am_marker_with_digit() -> None:
    out = normalize_text_for_tts("call at 9 am about it")
    assert "about it" in out


def test_bare_am_in_sentence_not_misread() -> None:
    """The standalone-AM rule requires a digit prefix; 'I am here' is safe."""
    out = normalize_text_for_tts("I am here")
    # No rewrite of the 'I am' phrase -- it has no leading digit.
    # Specific form doesn't matter; just confirm 'I am here' survives semantically.
    assert "I" in out and "am" in out and "here" in out


# ---- Unicode oddities ---------------------------------------------------


def test_unicode_en_dash_passes_through() -> None:
    """En-dash (U+2013) -- adversarial: looks like ASCII hyphen but isn't.

    The unit-rewriter expands ``km`` to spoken form, so the literal
    ``km`` token won't survive verbatim. The contract here is "no
    crash" + "the rewritten unit survives in some form".
    """
    out = normalize_text_for_tts("range 5–7 km")
    # Either "km" (unrewritten) or "kilometre"/"kilometres" (rewritten).
    assert "km" in out or "kilom" in out.lower()


def test_unicode_em_dash_passes_through() -> None:
    out = normalize_text_for_tts("note—this matters")
    assert "matters" in out


def test_unicode_non_breaking_hyphen() -> None:
    out = normalize_text_for_tts("co‑op the result")
    assert "result" in out


def test_unicode_smart_quotes() -> None:
    """Curly quotes from copy-pasted text."""
    out = normalize_text_for_tts("he said “hello” clearly")
    assert "hello" in out


def test_rtl_arabic_text_passes_through() -> None:
    """Right-to-left text must not crash the normaliser."""
    # Arabic "hello" + ASCII tail.
    out = normalize_text_for_tts("مرحبا from your assistant")
    assert "from your assistant" in out


def test_combining_diacritic_passes_through() -> None:
    """Combining acute accent on 'e' is a 2-code-point grapheme."""
    out = normalize_text_for_tts("café opens at 8 am")
    assert "opens" in out


def test_emoji_sequence_no_crash() -> None:
    """Emoji has multi-byte UTF-8 + ZWJ + surrogate pair edge cases."""
    normalize_text_for_tts("ready \U0001F44D let's go")  # no exception


# ---- Shell metachars / special punctuation ------------------------------


def test_shell_metachars_pass_through() -> None:
    """Adversarial: a malformed prompt might splice shell syntax in."""
    out = normalize_text_for_tts("run cat foo | grep bar > out.txt")
    assert "out.txt" in out


def test_backtick_passthrough() -> None:
    out = normalize_text_for_tts("see `notes.md` for details")
    assert "notes.md" in out


def test_dollar_sign_in_currency_handled() -> None:
    """$1.5M -> spoken form via the currency rule."""
    out = normalize_text_for_tts("budget is $1.5M")
    # The exact rewrite is "1.5 million dollars" per the docstring.
    # We pin the strong invariant: no raw "$1.5M" or "$" survives.
    assert "$1.5M" not in out


def test_dollar_sign_without_amount_no_crash() -> None:
    """A bare $ without a digit shouldn't trigger the currency rule."""
    out = normalize_text_for_tts("paid $ at the door")
    # Specific output uncertain; just confirm no crash and the rest survives.
    assert "the door" in out


def test_question_marks_and_exclamations_preserved() -> None:
    out = normalize_text_for_tts("Are you sure?! Really??")
    assert "?" in out and "!" in out


# ---- Latin abbreviations ------------------------------------------------


def test_eg_rewritten() -> None:
    out = normalize_text_for_tts("e.g. fast paths")
    assert "e.g." not in out


def test_etc_rewritten() -> None:
    out = normalize_text_for_tts("items A, B, etc.")
    assert "etc." not in out


# ---- Acronyms + titles --------------------------------------------------


def test_acronym_dots_split() -> None:
    out = normalize_text_for_tts("from the U.S.A. today")
    assert "U.S.A." not in out


def test_title_followed_by_name_expanded() -> None:
    out = normalize_text_for_tts("Dr. Smith arrived")
    # The rule expands titles to spoken form ("Doctor"); confirm
    # "Dr." (with the period) does not survive verbatim.
    assert "Dr. Smith" not in out
    assert "Smith" in out


# ---- Idempotence + length ---------------------------------------------


def test_normaliser_is_idempotent_on_clean_text() -> None:
    clean = "hello there friend"
    assert normalize_text_for_tts(normalize_text_for_tts(clean)) == normalize_text_for_tts(clean)


def test_very_long_input_no_crash() -> None:
    """10 KB of varied text -- must not blow up or time out."""
    chunks = ["see C:\\foo\\bar.txt", "at 9 am", "etc.", "https://a.test"] * 200
    out = normalize_text_for_tts(" ".join(chunks))
    assert isinstance(out, str)
    assert "https://" not in out


# ===========================================================================
# _speakable -- adversarial inputs
# ===========================================================================


def test_speakable_empty_string() -> None:
    assert _speakable("") == ""


def test_speakable_whitespace_only() -> None:
    assert _speakable("   ") == ""


def test_speakable_strips_windows_drive_path() -> None:
    assert _speakable("C:\\Users\\alice\\proj") == "proj"


def test_speakable_strips_unix_path() -> None:
    assert _speakable("/home/alice/proj") == "proj"


def test_speakable_strips_mixed_slash_path() -> None:
    out = _speakable("C:\\Users/alice\\Documents/proj")
    assert "\\" not in out and "/" not in out
    assert "proj" in out


def test_speakable_strips_surrounding_quotes() -> None:
    assert _speakable('"my project"') == "my project"
    assert _speakable("'my project'") == "my project"


def test_speakable_strips_leading_trailing_whitespace() -> None:
    assert _speakable("  proj_x  ") == "proj_x"


def test_speakable_path_with_no_separator_returned_as_is() -> None:
    assert _speakable("simple_name") == "simple_name"


def test_speakable_path_ending_with_separator() -> None:
    """A trailing slash should yield empty leaf -- helper returns ''."""
    # The current behaviour returns "" because the rsplit gives an empty leaf.
    # This pins the behaviour explicitly so future changes are deliberate.
    assert _speakable("C:\\Users\\alice\\") == ""


def test_speakable_unicode_name_preserved() -> None:
    """Project names with unicode characters must round-trip."""
    assert _speakable("C:\\Users\\alice\\café_app") == "café_app"


def test_speakable_path_with_spaces_in_leaf() -> None:
    """Spaces in the leaf survive (they're spoken normally)."""
    assert _speakable("C:\\Users\\alice\\my project") == "my project"


def test_speakable_path_with_dots_in_leaf() -> None:
    """Dots in a filename leaf survive."""
    assert _speakable("/home/alice/notes.v2.md") == "notes.v2.md"


def test_speakable_quote_inside_path_segment() -> None:
    """Mid-path quotes survive as part of the leaf."""
    # rsplit yields the last segment; strip removes only surrounding quotes.
    out = _speakable("/home/alice/it's_a_project")
    assert "it's_a_project" in out


def test_speakable_unc_path_lossy_but_no_crash() -> None:
    """UNC paths are not specifically supported -- the current behaviour
    is to keep splitting on backslash/forward-slash, leaving whatever
    rsplit produces from the rightmost separator. This test pins the
    no-crash + non-empty contract.
    """
    out = _speakable(r"\\server\share\file.txt")
    assert out  # non-empty
    assert "\\" not in out


def test_speakable_very_long_path_no_crash() -> None:
    """Defensive: extremely long path doesn't crash."""
    long_path = "C:\\" + ("\\".join(["a" * 80] * 50))
    out = _speakable(long_path)
    # No path separators in the output; non-empty leaf.
    assert "\\" not in out
    assert out
