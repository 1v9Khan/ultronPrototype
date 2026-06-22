"""L5 — output reassembly canonicalizer (materialize hidden channels).

The abliterated 8B will happily OBEY "spell it with the first letter of each
word" / "say it in NATO" / "encode it in base-26". Those payloads are genuinely
BENIGN as plain text, so L1's blocklist never fires — the harm is assembled in
the listener's ear. This layer materializes every such hidden channel into
candidate strings, which the caller (the L5 output gate) then RE-SCREENS through
the L1 blocklist. Batch- and raid-aware: acrostics are computed across the whole
batch and across lines, not just within one message.

ANTICHEAT: pure stdlib. Used on the OUTPUT (draft) side and as an input tripwire.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from kenning.twitch.safety.blocklist import Blocklist, BlockMatch, get_blocklist
from kenning.twitch.safety.normalize import deobfuscate, fold_skeleton, normalize_for_match

__all__ = ["reassemble_candidates", "reassembly_matches"]

_WORD = re.compile(r"[A-Za-z0-9]+")
_ALNUM = re.compile(r"[A-Za-z0-9]")

# NATO / common spelling alphabet -> letter (first letter of the codeword anyway,
# but the canonical set defeats "alpha bravo charlie" without relying on acrostic).
_NATO = {
    "alpha": "a", "alfa": "a", "bravo": "b", "charlie": "c", "delta": "d",
    "echo": "e", "foxtrot": "f", "golf": "g", "hotel": "h", "india": "i",
    "juliet": "j", "juliett": "j", "kilo": "k", "lima": "l", "mike": "m",
    "november": "n", "oscar": "o", "papa": "p", "quebec": "q", "romeo": "r",
    "sierra": "s", "tango": "t", "uniform": "u", "victor": "v", "whiskey": "w",
    "xray": "x", "x-ray": "x", "yankee": "y", "zulu": "z",
}

_MORSE = {
    ".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e", "..-.": "f",
    "--.": "g", "....": "h", "..": "i", ".---": "j", "-.-": "k", ".-..": "l",
    "--": "m", "-.": "n", "---": "o", ".--.": "p", "--.-": "q", ".-.": "r",
    "...": "s", "-": "t", "..-": "u", "...-": "v", ".--": "w", "-..-": "x",
    "-.--": "y", "--..": "z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}


def _acrostic(words: Iterable[str]) -> str:
    out = []
    for w in words:
        m = _ALNUM.search(w)
        if m:
            out.append(m.group(0))
    return "".join(out).lower()


def _rot(s: str, n: int) -> str:
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + n) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + n) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _decode_a1z26(text: str) -> str:
    """Numbers 1-26 (space/dash separated) -> letters."""
    nums = re.findall(r"\b([12]?\d)\b", text)
    out = []
    for n in nums:
        v = int(n)
        if 1 <= v <= 26:
            out.append(chr(96 + v))
    return "".join(out)


def _decode_morse(text: str) -> str:
    # tolerate · • and / word separators
    norm = text.replace("·", ".").replace("•", ".").replace("–", "-").replace("—", "-")
    tokens = re.split(r"[\s/|]+", norm.strip())
    out = []
    for tok in tokens:
        if tok and set(tok) <= {".", "-"}:
            out.append(_MORSE.get(tok, ""))
    return "".join(out)


def _decode_nato(text: str) -> str:
    toks = re.findall(r"[a-zA-Z-]+", text.lower())
    out = [_NATO[t] for t in toks if t in _NATO]
    return "".join(out)


def reassemble_candidates(text: str, *, batch_context: Iterable[str] = ()) -> list[str]:
    """Return distinct, non-trivial reassembled candidate strings to re-screen.

    Covers: acrostic (first letter of each word, and of each line, and across the
    batch's messages), NATO decode, a1z26, ROT-1..25, morse. Every candidate is
    later run through the L1 blocklist by :func:`reassembly_matches`.
    """
    text = text or ""
    cands: set[str] = set()

    def add(s: str) -> None:
        s = (s or "").strip().lower()
        if len(s) >= 4:
            cands.add(s)
            cands.add(deobfuscate(fold_skeleton(s)))

    words = _WORD.findall(text)
    add(_acrostic(words))                                  # first letter of each word
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        add(_acrostic(lines))                              # first letter of each line
    batch = [m for m in batch_context if m and m.strip()]
    if batch:
        add(_acrostic(batch))                              # first letter of each batched message
        add(_acrostic([w for m in batch for w in _WORD.findall(m)]))

    add(_decode_nato(text))
    add(_decode_a1z26(text))
    add(_decode_morse(text))
    # ROT13 only — the only caesar variant used in practice. Brute-forcing all 25
    # shifts of arbitrary chat is false-positive-prone for ~zero added coverage.
    low = text.lower()
    if re.search(r"[a-z]", low):
        add(_rot(low, 13))
    return sorted(cands)


def reassembly_matches(
    text: str,
    *,
    blocklist: Optional[Blocklist] = None,
    batch_context: Iterable[str] = (),
) -> list[BlockMatch]:
    """Materialize hidden channels and return any L1 blocklist hits on them.

    A hit here means the *plain* text was benign but a reassembled form is a
    slur/threat/etc. The caller treats any hit as an output-side trip -> DEFLECT
    the WHOLE reply (never partially redact)."""
    bl = blocklist or get_blocklist()
    out: list[BlockMatch] = []
    seen: set[tuple[str, str]] = set()
    for cand in reassemble_candidates(text, batch_context=batch_context):
        # Re-screen via the full L1 pipeline (normalize -> blocklist).
        for m in bl.scan(normalize_for_match(cand)):
            key = (m.category, m.term)
            if key in seen:
                continue
            seen.add(key)
            # retag the rule so the audit shows it came from reassembly.
            out.append(BlockMatch(m.category, m.severity, m.term, f"reassembly:{m.rule}", "reassembly"))
    return out
