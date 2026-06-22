"""L1 — the frozen text canonicalizer for chat-safety matching.

A single, FROZEN, deterministic pipeline that turns attacker-controlled text into
several comparison forms so the blocklist (``blocklist.py``) can dual-scan them.
For a NON-refusing abliterated model this is load-bearing: the deterministic
layers are the defense, so the canonicalizer must close the covert channels a
model would otherwise decode and emit.

Forms produced (see :class:`NormForms`):
  * ``nfkc``        — covert-channel-stripped + NFKC + casefold. Primary form;
                      the blocklist word-boundary-matches all categories here.
  * ``skeleton``    — ``nfkc`` + confusable/homoglyph fold + accent strip
                      (Cyrillic/Greek lookalikes NFKC does NOT fold). Catches
                      homoglyph spoofing; word-boundary matched.
  * ``deobf``       — ``skeleton`` with leetspeak folded, separators removed, and
                      char-repeats collapsed. SUBSTRING-scanned against the
                      curated HARD-slur subset only (high recall for the
                      zero-tolerance set): catches "b o m b", "f.u.c.k", "n1gg…".
  * ``reversed``    — ``deobf`` reversed (reversed-text evasion).
  * ``tokens``      — word tokens of ``skeleton`` (for phonetic + fuzzy match).

ANTICHEAT: pure stdlib (``unicodedata``/``re``) — importable in the voice process.

Frozen contract: changing the pipeline can silently weaken a deployed blocklist,
so edits must ship with the proving corpus case (bypass->regression). ReDoS-safe:
all regexes are linear and bounded.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["NormForms", "normalize_for_match", "fold_skeleton", "deobfuscate"]


# --- covert-channel codepoints (invisible to humans + the review popup) --------
# Dropped unconditionally: an abliterated model decodes these; a blocklist that
# only sees the visible glyphs is bypassed. Ranges + explicit chars are belt-and-
# suspenders over the General_Category drop below (Cf/Cs/Co/Cn).
_DROP_RANGES: tuple[tuple[int, int], ...] = (
    (0x200B, 0x200F),   # zero-width space/joiner/non-joiner + LRM/RLM
    (0x202A, 0x202E),   # bidi embeddings/overrides
    (0x2060, 0x2064),   # word-joiner + invisible operators
    (0x2066, 0x2069),   # bidi isolates
    (0xFE00, 0xFE0F),   # variation selectors 1-16
    (0xE0100, 0xE01EF),  # variation selectors supplement
    (0xE0000, 0xE007F),  # Unicode Tag block (ASCII-smuggling channel)
)
_DROP_CHARS: frozenset[int] = frozenset({
    0xFEFF,  # BOM / zero-width no-break space
    0x00AD,  # soft hyphen
    0x034F,  # combining grapheme joiner
    0x180E,  # mongolian vowel separator
})
_DROP_CATEGORIES: frozenset[str] = frozenset({"Cf", "Cs", "Co", "Cn"})
# Max consecutive combining marks kept (zalgo cap).
_MAX_COMBINING_RUN = 2


def _in_drop_range(cp: int) -> bool:
    for a, b in _DROP_RANGES:
        if a <= cp <= b:
            return True
    return False


def _strip_covert(s: str) -> str:
    out: list[str] = []
    combining_run = 0
    for ch in s:
        cp = ord(ch)
        if cp in _DROP_CHARS or _in_drop_range(cp):
            continue
        cat = unicodedata.category(ch)
        if cat in _DROP_CATEGORIES:
            continue
        if cat in ("Mn", "Mc", "Me"):
            combining_run += 1
            if combining_run > _MAX_COMBINING_RUN:
                continue  # zalgo: drop excess combining marks
        else:
            combining_run = 0
        out.append(ch)
    return "".join(out)


# --- confusable / homoglyph fold ----------------------------------------------
# NFKC already folds fullwidth + mathematical-alphanumeric + ligatures to ASCII,
# so this map only needs the cross-SCRIPT lookalikes NFKC keeps distinct:
# Cyrillic / Greek / Armenian letters that render like Latin. Curated, high-value.
_CONFUSABLE_MAP: dict[str, str] = {
    # Cyrillic -> Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "к": "k", "м": "m", "т": "t",
    "н": "h", "в": "b", "и": "u", "й": "u", "п": "n",
    "А": "a", "Е": "e", "О": "o", "Р": "p", "С": "c",
    "Х": "x", "У": "y", "К": "k", "М": "m", "Т": "t",
    "Н": "h", "В": "b", "Ѕ": "s", "ѕ": "s", "і": "i",
    "І": "i", "ј": "j", "Ј": "j", "һ": "h", "ґ": "r",
    # Greek -> Latin
    "α": "a", "ο": "o", "ρ": "p", "υ": "u", "ν": "v",
    "κ": "k", "ι": "i", "Ι": "i", "Ο": "o", "Ρ": "p",
    "Β": "b", "Ε": "e", "Η": "h", "Κ": "k", "Μ": "m",
    "Ν": "n", "Τ": "t", "Χ": "x", "Υ": "y", "Ζ": "z",
    # Armenian / misc lookalikes
    "ո": "n", "ռ": "n", "օ": "o",
    # common symbol lookalikes
    "ø": "o", "œ": "oe", "æ": "ae",
}

# Leetspeak digit/symbol -> letter. Applied only to tokens that contain a letter
# (so pure numbers like a damage value "84" are never mangled into letters).
_LEET_MAP: dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t",
    "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "+": "t", "|": "l",
    "(": "c", "<": "c", "€": "e", "£": "l", "¡": "i",
}

_NONALNUM = re.compile(r"[^a-z0-9]+")
_REPEAT3 = re.compile(r"(.)\1{2,}")          # 3+ same char -> 1 (linear, ReDoS-safe)
_WORD = re.compile(r"[a-z0-9]+")
_HAS_ALPHA = re.compile(r"[a-zA-Z]")


def fold_skeleton(s: str) -> str:
    """nfkc-casefolded input -> confusable-folded, accent-stripped ASCII skeleton."""
    folded = "".join(_CONFUSABLE_MAP.get(ch, ch) for ch in s)
    # Strip accents: decompose, drop nonspacing marks.
    decomposed = unicodedata.normalize("NFKD", folded)
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return no_accents


def _fold_leet_token(tok: str) -> str:
    if not _HAS_ALPHA.search(tok):
        return tok  # pure number / symbol run: leave untouched
    return "".join(_LEET_MAP.get(ch, ch) for ch in tok)


def deobfuscate(skeleton: str) -> str:
    """Collapse leet + separators + char-repeats so spaced/dotted/leeted slurs
    surface for a substring scan. e.g. 'b o m b' -> 'bomb', 'f.u.c.k' -> 'fuck',
    'n1ggaaaa' -> 'nigga'. Lossy + cross-word-joining by design: only the curated
    HARD-slur subset is substring-scanned on this form (with a benign allowlist)."""
    # Fold leet per whitespace token (so "84" stays numeric but "n1gga" -> "nigga").
    leeted = " ".join(_fold_leet_token(t) for t in skeleton.split())
    # Collapse 3+ repeats, then remove every non-alnum separator.
    collapsed = _REPEAT3.sub(r"\1", leeted)
    return _NONALNUM.sub("", collapsed)


@dataclass(frozen=True)
class NormForms:
    """The comparison forms produced from one input string."""
    raw: str
    nfkc: str
    skeleton: str
    deobf: str
    reversed: str
    tokens: tuple[str, ...]
    covert_stripped: int  # how many covert-channel codepoints were removed (telemetry)

    def all_forms(self) -> tuple[str, ...]:
        return (self.nfkc, self.skeleton, self.deobf, self.reversed)


def normalize_for_match(text: str) -> NormForms:
    """Run the frozen canonicalization pipeline. Never raises (fail-safe to the
    raw text); the caller treats any anomaly as fail-CLOSED at the match layer."""
    raw = text or ""
    try:
        stripped = _strip_covert(raw)
        covert = len(raw) - len(stripped)
        nfkc = unicodedata.normalize("NFKC", stripped).casefold()
        skeleton = fold_skeleton(nfkc)
        deobf = deobfuscate(skeleton)
        rev = deobf[::-1]
        tokens = tuple(_WORD.findall(skeleton))
        return NormForms(
            raw=raw, nfkc=nfkc, skeleton=skeleton, deobf=deobf,
            reversed=rev, tokens=tokens, covert_stripped=max(0, covert),
        )
    except Exception:  # noqa: BLE001 — never let normalization crash the gate
        # Fail-safe: degrade to a casefolded raw; the matcher still runs and the
        # validator fails CLOSED on any downstream anomaly.
        low = raw.casefold()
        return NormForms(
            raw=raw, nfkc=low, skeleton=low, deobf=_NONALNUM.sub("", low),
            reversed=_NONALNUM.sub("", low)[::-1],
            tokens=tuple(_WORD.findall(low)), covert_stripped=0,
        )
