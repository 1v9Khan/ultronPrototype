"""ANSI / control-character sanitisation + grapheme-aware width.

Three primitives ultron needs across logging, memory persistence,
web-search ingestion, and TTS chunking:

* :func:`strip_ansi` — remove ANSI CSI (``ESC[...``) and OSC
  (``ESC]...BEL``/``ESC\\``) escape sequences. Used to sanitise tool
  output, web-page text, and LLM responses before they reach
  downstream consumers that don't render escape sequences.
* :func:`sanitize_for_log` — strip ANSI + C0 (``0x00-0x1F``) + C1
  (``0x80-0x9F``) + DEL (``0x7F``) control characters. Defends
  against CWE-117 (log forging via newline / cursor-jump injection)
  and prevents attacker-controlled tool output from breaking log
  viewers. Tab + LF + CR are preserved (legitimate inside log
  records); the regex is built at runtime so the source file stays
  free of literal control bytes that linters / IDEs flag.
* :func:`visible_width` — grapheme-aware display width. Counts the
  visible width of a string with correct handling of zero-width
  combining marks, full-width CJK glyphs, and emoji ZWJ sequences.
  Use this for TTS chunking, terminal alignment, and any width
  budgeting where ``len(str)`` over-counts on multi-byte content.

Pattern shapes informed by OpenClaw's ``src/terminal/ansi.ts``
(MIT; see ``THIRD_PARTY_NOTICES.md``); algorithm details adapted to
stdlib Python (``unicodedata``) so no third-party grapheme library
is required.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterator

#: ANSI CSI escape sequence: ``ESC [ <params> <command>``.
#: Final byte is in the 0x40-0x7E range.
_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

#: ANSI OSC escape sequence: ``ESC ] <content> (BEL | ESC \\)``.
#: Used by terminals for window-title and hyperlink (OSC 8) sequences.
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

#: Two-byte ESC escapes (``ESC X`` for X in the 0x40-0x5F range), used
#: by some terminals for keyboard mode toggles. Stripped for
#: completeness alongside CSI/OSC.
_ESC_RE = re.compile(r"\x1b[@-_]")

#: Control characters built from explicit codepoint ranges so the
#: source file stays free of literal control bytes (legibility for
#: code readers + tooling that complains about file-content escapes).
#: Includes C0 minus tab/LF/CR (which are legitimate in log lines),
#: DEL, and C1 (0x80-0x9F).
_CONTROL_CHARS_PATTERN = "".join(
    [
        # C0: 0x00..0x1F minus 0x09 (tab), 0x0A (LF), 0x0D (CR)
        "".join(chr(c) for c in range(0x00, 0x20) if c not in (0x09, 0x0A, 0x0D)),
        # DEL
        chr(0x7F),
        # C1: 0x80..0x9F
        "".join(chr(c) for c in range(0x80, 0xA0)),
    ]
)
_CONTROL_RE = re.compile(f"[{re.escape(_CONTROL_CHARS_PATTERN)}]")

#: Zero-width combining ranges. Includes combining diacriticals
#: (U+0300..U+036F), combining diacriticals extended (U+1AB0..U+1AFF),
#: combining diacriticals supplement (U+1DC0..U+1DFF), combining
#: diacriticals for symbols (U+20D0..U+20FF), half-marks
#: (U+FE20..U+FE2F), variation selectors (U+FE00..U+FE0F), and
#: ZWJ/ZWNJ/ZWSP (U+200B..U+200D) which never advance the cursor.
_ZERO_WIDTH_RANGES: tuple[tuple[int, int], ...] = (
    (0x0300, 0x036F),
    (0x1AB0, 0x1AFF),
    (0x1DC0, 0x1DFF),
    (0x200B, 0x200D),
    (0x20D0, 0x20FF),
    (0xFE00, 0xFE0F),
    (0xFE20, 0xFE2F),
)

#: Emoji-presence heuristic: codepoints in the "Pictographic" blocks
#: that render as emoji (and therefore double-width in most terminals).
#: Not exhaustive — covers the bulk of user-facing emoji ranges
#: (Misc Symbols and Pictographs, Emoticons, Transport, Supplemental
#: Symbols, plus the regional indicators used for flags).
_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F300, 0x1F5FF),  # Misc symbols + pictographs
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F680, 0x1F6FF),  # Transport + map
    (0x1F700, 0x1F77F),  # Alchemical
    (0x1F780, 0x1F7FF),  # Geometric extended
    (0x1F800, 0x1F8FF),  # Supplemental arrows
    (0x1F900, 0x1F9FF),  # Supplemental symbols + pictographs
    (0x1FA00, 0x1FA6F),  # Chess + symbols
    (0x1FA70, 0x1FAFF),  # Symbols + pictographs extended-A
    (0x2600, 0x26FF),    # Misc symbols
    (0x2700, 0x27BF),    # Dingbats
    (0x1F1E6, 0x1F1FF),  # Regional indicator (flag halves)
)

def _build_grapheme_continuation_class() -> str:
    """Build a regex character class for grapheme continuations.

    Constructed from explicit Unicode codepoint ranges so the source
    file stays ASCII-only regardless of editor encoding. Covers:
    combining diacriticals, variation selectors, ZWJ, and other
    invisible joiners that should fold into the preceding base
    cluster.
    """
    ranges = [
        (0x0300, 0x036F),   # Combining diacriticals
        (0x1AB0, 0x1AFF),   # Combining diacriticals extended
        (0x1DC0, 0x1DFF),   # Combining diacriticals supplement
        (0x200D, 0x200D),   # Zero-width joiner
        (0x20D0, 0x20FF),   # Combining diacriticals for symbols
        (0xFE00, 0xFE0F),   # Variation selectors
        (0xFE20, 0xFE2F),   # Combining half marks
    ]
    parts: list[str] = []
    for lo, hi in ranges:
        if lo == hi:
            parts.append(f"\\u{lo:04x}")
        else:
            parts.append(f"\\u{lo:04x}-\\u{hi:04x}")
    return "[" + "".join(parts) + "]"


#: Grapheme cluster regex: one base character followed by zero or
#: more continuation codepoints (combining marks, ZWJ, variation
#: selectors). Built from explicit codepoint ranges so the source
#: file stays editor-encoding-independent.
_GRAPHEME_RE = re.compile(
    r"." + _build_grapheme_continuation_class() + r"*",
    re.DOTALL,
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (CSI, OSC, two-byte ESC) from ``text``.

    Args:
        text: Input string, possibly containing escape sequences.

    Returns:
        ``text`` with every ANSI escape sequence removed. Whitespace
        and printable characters are preserved.
    """
    if not text:
        return text
    out = _OSC_RE.sub("", text)
    out = _CSI_RE.sub("", out)
    out = _ESC_RE.sub("", out)
    return out


def sanitize_for_log(text: str) -> str:
    """Strip ANSI escapes + control characters; safe for log records.

    Defends against CWE-117 (log forging) by removing characters that
    could let an attacker inject fake log lines (LF inside C0 is
    preserved as legitimate; the rest of C0 is stripped). Tab and
    CR/LF are preserved; DEL and C1 (0x80-0x9F) are stripped.

    Args:
        text: Input string.

    Returns:
        Sanitised copy safe to write to JSONL audit logs and
        text-based log files.
    """
    if not text:
        return text
    return _CONTROL_RE.sub("", strip_ansi(text))


def is_zero_width(codepoint: int) -> bool:
    """Return ``True`` when ``codepoint`` does not advance the cursor.

    Includes combining marks, variation selectors, ZWJ/ZWNJ/ZWSP,
    and Unicode-category Mn/Mc/Me characters.
    """
    for lo, hi in _ZERO_WIDTH_RANGES:
        if lo <= codepoint <= hi:
            return True
    try:
        category = unicodedata.category(chr(codepoint))
    except ValueError:
        return False
    return category in ("Mn", "Mc", "Me", "Cf")


def is_full_width(codepoint: int) -> bool:
    """Return ``True`` when the codepoint renders as two columns.

    Uses Unicode East-Asian-Width property; ``W`` (Wide) and ``F``
    (Fullwidth) are full-width. Also returns ``True`` for emoji
    ranges since most terminals render them double-width.
    """
    try:
        eaw = unicodedata.east_asian_width(chr(codepoint))
    except (ValueError, TypeError):
        eaw = "N"
    if eaw in ("W", "F"):
        return True
    for lo, hi in _EMOJI_RANGES:
        if lo <= codepoint <= hi:
            return True
    return False


def split_graphemes(text: str) -> list[str]:
    """Split ``text`` into Unicode grapheme clusters.

    Uses a regex that groups each base character with its trailing
    combining marks, variation selectors, and ZWJ continuations.
    Empty input yields an empty list.

    Args:
        text: Input string.

    Returns:
        List of grapheme cluster strings. ``len(graphemes) <=
        len(text)`` — combining marks fold into the preceding cluster.
    """
    if not text:
        return []
    return _GRAPHEME_RE.findall(text)


def grapheme_width(cluster: str) -> int:
    """Visible width (in columns) of a single grapheme cluster.

    Returns 0 when the cluster has no printable advance (only
    zero-width characters), 2 when any contained codepoint is
    full-width or emoji, and 1 otherwise.
    """
    if not cluster:
        return 0
    saw_printable = False
    for char in cluster:
        cp = ord(char)
        if is_full_width(cp):
            return 2
        if is_zero_width(cp):
            continue
        saw_printable = True
    return 1 if saw_printable else 0


def visible_width(text: str) -> int:
    """Total visible width of ``text`` in terminal columns.

    Args:
        text: Input string, possibly containing ANSI escapes.

    Returns:
        Sum of grapheme widths after stripping ANSI sequences.
        Equivalent to ``sum(grapheme_width(g) for g in
        split_graphemes(strip_ansi(text)))``.
    """
    if not text:
        return 0
    stripped = strip_ansi(text)
    return sum(grapheme_width(g) for g in split_graphemes(stripped))


def iter_graphemes(text: str) -> Iterator[str]:
    """Yield grapheme clusters from ``text``, one at a time.

    Streaming variant of :func:`split_graphemes` for use cases that
    cap by width without materialising the entire cluster list.
    """
    if not text:
        return
    for match in _GRAPHEME_RE.finditer(text):
        yield match.group(0)


def truncate_to_visible_width(text: str, max_width: int, *, ellipsis: str = "...") -> str:
    """Truncate ``text`` so its visible width does not exceed ``max_width``.

    Counts width grapheme-by-grapheme so emoji and CJK don't get split
    mid-cluster. Appends ``ellipsis`` when truncation occurred (and
    ``len(ellipsis)`` fits within the budget).

    Args:
        text: Input string.
        max_width: Maximum allowed visible width in columns.
        ellipsis: Suffix appended when truncation happens. Defaults to
            ``"..."``. Pass an empty string to truncate without a marker.

    Returns:
        ``text`` unchanged when its visible width already fits, or a
        truncated copy ending in ``ellipsis``.
    """
    if max_width <= 0:
        return ""
    stripped = strip_ansi(text)
    if visible_width(stripped) <= max_width:
        return stripped
    ellipsis_w = visible_width(ellipsis)
    budget = max_width - ellipsis_w if ellipsis_w < max_width else max_width
    accumulated_w = 0
    pieces: list[str] = []
    for cluster in iter_graphemes(stripped):
        w = grapheme_width(cluster)
        if accumulated_w + w > budget:
            break
        pieces.append(cluster)
        accumulated_w += w
    truncated = "".join(pieces)
    if ellipsis_w < max_width:
        truncated += ellipsis
    return truncated


__all__ = [
    "strip_ansi",
    "sanitize_for_log",
    "is_zero_width",
    "is_full_width",
    "split_graphemes",
    "grapheme_width",
    "visible_width",
    "iter_graphemes",
    "truncate_to_visible_width",
]
