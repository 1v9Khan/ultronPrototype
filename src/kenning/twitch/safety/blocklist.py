"""L1 — the dual-scan slur / hate / dox / threat / injection blocklist matcher.

Consumes the canonical forms from :mod:`kenning.twitch.safety.normalize` and
matches them against a hot-reloadable JSON blocklist. Matching is layered to keep
recall high on the zero-tolerance set while bounding false positives:

  * **literal word-boundary** match on ``nfkc`` + ``skeleton`` for every category
    (low false-positive; the everyday case).
  * **hard-slur subset** (categories flagged ``phonetic_fuzzy``) ALSO get a
    SUBSTRING scan on ``deobf`` + ``reversed`` (catches "b o m b", "f.u.c.k",
    leetspeak, reversed) plus **phonetic** (metaphone/soundex) and **fuzzy**
    (RapidFuzz) matching on tokens — for near-spellings and homophones.
  * **regex** rules for dox/PII/threat/injection patterns.

A benign **allowlist** (gg, agent names, pro names, common gamer vernacular)
suppresses the Scunthorpe class of false positives.

FAIL-CLOSED: if the JSON file is missing/corrupt, a built-in minimal HARD set
stays active so the gate is never silently disabled. ANTICHEAT: stdlib + rapidfuzz
(+ optional jellyfish if present); importable in the voice process.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from kenning.twitch.safety.normalize import NormForms, normalize_for_match

__all__ = ["BlockMatch", "Blocklist", "SEVERITY_ORDER", "load_blocklist", "get_blocklist"]

SEVERITY_ORDER: dict[str, int] = {
    "none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
}

# --- phonetic + fuzzy backends (graceful, dep-light) --------------------------
try:  # rapidfuzz is in the voice-path import envelope
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        return float(_fuzz.ratio(a, b))
except Exception:  # noqa: BLE001
    import difflib

    def _ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


_VOWELS = re.compile(r"[aeiou]+")


def _consonant_skeleton(s: str) -> str:
    """Consonant-only skeleton: catches vowel-swap evasion (niggur/nigguh) while
    staying clear of different-consonant words (bigger/faucet)."""
    return _VOWELS.sub("", s)


def _soundex(s: str) -> str:
    """Compact Soundex phonetic key (deterministic, dep-free fallback)."""
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), **dict.fromkeys("l", "4"),
             **dict.fromkeys("mn", "5"), **dict.fromkeys("r", "6")}
    first = s[0]
    out = first.upper()
    prev = codes.get(first, "")
    for ch in s[1:]:
        c = codes.get(ch, "")
        if c and c != prev:
            out += c
        if ch not in "hw":
            prev = c
    return (out + "000")[:4]


try:
    import jellyfish as _jelly  # optional; richer phonetic key

    def _phonetic(tok: str) -> str:
        try:
            return _jelly.metaphone(tok) or _soundex(tok)
        except Exception:  # noqa: BLE001
            return _soundex(tok)
except Exception:  # noqa: BLE001
    def _phonetic(tok: str) -> str:
        return _soundex(tok)


# --- match result --------------------------------------------------------------
@dataclass(frozen=True)
class BlockMatch:
    category: str
    severity: str
    term: str
    rule: str        # literal | hard_substr | phonetic | fuzzy | regex
    form: str        # which normalized form fired (nfkc/skeleton/deobf/reversed/raw)

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 0)


# --- built-in fail-closed minimum (used only if the JSON can't load) ----------
# Kept deliberately tiny: a load failure must never leave ZERO protection. The
# real curated list lives in data/blocklist.json. Injection markers only here;
# the hard-slur set in the JSON is the authoritative one.
_BUILTIN_FALLBACK: dict = {
    "version": "builtin-fallback",
    "categories": {
        "injection": {
            "severity": "medium", "word_boundary": False,
            "terms": ["ignore previous", "ignore all previous", "disregard above",
                      "you are now", "new instructions", "system prompt", "jailbreak",
                      "developer mode", "do anything now"],
            "regexes": [r"<\|[a-z0-9_]+\|>", r"\[/?inst\]", r"</?(system|assistant|user)>"],
        },
    },
    "allowlist": [],
}


@dataclass
class Blocklist:
    """Compiled, scannable blocklist. Build via :func:`load_blocklist`."""
    version: str
    # category -> (severity, word_boundary, phonetic_fuzzy, terms, compiled_literal, compiled_regex)
    _categories: dict = field(default_factory=dict)
    _hard_terms: list = field(default_factory=list)         # [(category, severity, term)]
    _allowlist: frozenset = field(default_factory=frozenset)

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "Blocklist":
        bl = cls(version=str(data.get("version", "?")))
        bl._allowlist = frozenset(
            w.casefold() for w in (data.get("allowlist") or []) if isinstance(w, str)
        )
        for name, spec in (data.get("categories") or {}).items():
            severity = str(spec.get("severity", "high"))
            wb = bool(spec.get("word_boundary", True))
            pf = bool(spec.get("phonetic_fuzzy", False))
            terms = [t.casefold() for t in (spec.get("terms") or []) if isinstance(t, str) and t]
            literal_re = None
            if terms:
                # one alternation per category; escaped; optional \b boundaries.
                alt = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
                literal_re = re.compile((r"\b(?:%s)\b" if wb else r"(?:%s)") % alt)
            regexes = []
            for rx in (spec.get("regexes") or []):
                try:
                    regexes.append(re.compile(rx, re.IGNORECASE))
                except re.error:
                    continue  # skip a bad pattern; never crash the gate
            bl._categories[name] = {
                "severity": severity, "wb": wb, "pf": pf,
                "terms": terms, "literal_re": literal_re, "regexes": regexes,
            }
            if pf:
                for t in terms:
                    bl._hard_terms.append((name, severity, t))
        return bl

    # -- scanning --------------------------------------------------------------
    def _allowlisted(self, term: str, forms: NormForms) -> bool:
        """Scunthorpe guard: suppress a hard hit that lives inside a benign token."""
        for tok in forms.tokens:
            if tok in self._allowlist and term in tok:
                return True
        return False

    def scan(self, forms: NormForms) -> list[BlockMatch]:
        matches: list[BlockMatch] = []
        seen: set[tuple[str, str, str]] = set()

        def add(cat: str, sev: str, term: str, rule: str, form: str) -> None:
            key = (cat, term, rule)
            if key in seen:
                return
            seen.add(key)
            matches.append(BlockMatch(cat, sev, term, rule, form))

        # 1) literal word-boundary + regex on the low-FP forms.
        for name, c in self._categories.items():
            lit = c["literal_re"]
            if lit is not None:
                for form_name, form in (("nfkc", forms.nfkc), ("skeleton", forms.skeleton)):
                    m = lit.search(form)
                    if m and not self._allowlisted(m.group(0), forms):
                        add(name, c["severity"], m.group(0), "literal", form_name)
            for rx in c["regexes"]:
                # regex rules run on nfkc AND the raw (PII/URLs need raw punctuation).
                for form_name, form in (("nfkc", forms.nfkc), ("raw", forms.raw)):
                    if rx.search(form):
                        add(name, c["severity"], rx.pattern, "regex", form_name)
                        break

        # 2) hard-slur subset: substring on de-obfuscated + reversed (high recall).
        # ONLY for LONG (>=5) hard terms — short slurs (spic/coon/kike) embed in
        # benign words (suspicious/raccoon/...), so they rely on word-boundary +
        # token match above; their spaced-out form is caught by the L5 reassembly
        # layer (which reassembles single-letter runs and checks them as WHOLE
        # words, avoiding the substring-in-benign false positive entirely).
        for cat, sev, term in self._hard_terms:
            if len(term) < 5:
                # short hard term: exact-token match (homoglyph-folded) only.
                if term in forms.tokens and not self._allowlisted(term, forms):
                    add(cat, sev, term, "token", "tokens")
                continue
            if term in forms.deobf and not self._allowlisted(term, forms):
                add(cat, sev, term, "hard_substr", "deobf")
            elif term in forms.reversed and not self._allowlisted(term, forms):
                add(cat, sev, term, "hard_substr", "reversed")

        # 3) phonetic + fuzzy on skeleton tokens vs the LONG hard-slur subset.
        for tok in forms.tokens:
            if len(tok) < 5 or tok in self._allowlist:
                continue
            tok_ph = _phonetic(tok)
            for cat, sev, term in self._hard_terms:
                if len(term) < 5 or abs(len(tok) - len(term)) > 2:
                    continue
                if self._allowlisted(term, forms):
                    continue
                ct, cterm = _consonant_skeleton(tok), _consonant_skeleton(term)
                if (tok_ph and tok_ph == _phonetic(term)) or (len(cterm) >= 3 and ct == cterm):
                    add(cat, sev, term, "phonetic", "tokens")
                elif _ratio(tok, term) >= 88.0:
                    add(cat, sev, term, "fuzzy", "tokens")
        return matches

    def scan_text(self, text: str) -> list[BlockMatch]:
        return self.scan(normalize_for_match(text))

    def worst(self, forms_or_text) -> Optional[BlockMatch]:
        forms = forms_or_text if isinstance(forms_or_text, NormForms) else normalize_for_match(forms_or_text)
        ms = self.scan(forms)
        if not ms:
            return None
        return max(ms, key=lambda m: m.severity_rank)


# --- loading (hot-reloadable, fail-CLOSED) ------------------------------------
_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, Blocklist]] = {}


def _default_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "blocklist.json"


def load_blocklist(path: Optional[str] = None) -> Blocklist:
    """Load + compile the blocklist. FAIL-CLOSED: any error falls back to the
    built-in minimal HARD set (never returns an empty/permissive blocklist)."""
    p = Path(path) if path else _default_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        bl = Blocklist.from_dict(data)
        if not bl._categories:
            raise ValueError("blocklist has no categories")
        return bl
    except Exception:  # noqa: BLE001 — fail CLOSED to the built-in minimum
        return Blocklist.from_dict(_BUILTIN_FALLBACK)


def get_blocklist(path: Optional[str] = None) -> Blocklist:
    """Cached accessor; reloads when the file mtime changes (hot-reload)."""
    p = Path(path) if path else _default_path()
    key = str(p)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = -1.0
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        bl = load_blocklist(str(p))
        _CACHE[key] = (mtime, bl)
        return bl
