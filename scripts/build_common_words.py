#!/usr/bin/env python
"""Build ``src/kenning/audio/_common_words.py`` -- a baked, frequency-ranked
frozenset of common English words used to PROTECT real words from the STT
phonetic/fuzzy gazetteer snapper (``_stt_correct._phonetic_fuzzy_snap``).

Frontier ASR-correction principle (confirmed by the 2026-06-16 research board):
a closed-domain corrector must only ever rewrite GENUINELY out-of-vocabulary /
misheard tokens -- never an in-vocabulary common word. The old hand-curated
``_FUZZY_BLOCK`` denylist could never be complete; e.g. ``let`` (rank 541)
collided with the gazetteer term ``lit`` (same Metaphone "LT") and was rewritten,
destroying the relay-lead verb in "let my team know ..." (~92% of corpus
false-NEGATIVES). ``mean`` (rank 1003) -> ``main``; etc. Protecting the
frequency head fixes the whole class generically.

This is a BUILD-TIME script: it fetches a public-domain frequency list and bakes
a pure-python frozenset (no runtime import, no network, anticheat-safe). The
generated module is committed so the runtime never touches the network.

Source: google-10000-english (Google Trillion Word Corpus unigram frequencies,
public domain). https://github.com/first20hours/google-10000-english

Usage:  .venv/Scripts/python.exe scripts/build_common_words.py [N]
        (N = how many of the most-frequent words to protect; default 5000)
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/"
    "master/google-10000-english-usa-no-swears.txt"
)
OUT = Path(__file__).resolve().parents[1] / "src" / "kenning" / "audio" / "_common_words.py"
DEFAULT_N = 5000


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    raw = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8")
    ranked = [w.strip().lower() for w in raw.splitlines() if w.strip()]
    # Keep the most-frequent N, alpha-only, length >= 3 (the snapper ignores
    # tokens < 3 chars), preserving frequency rank for the cutoff then sorting
    # for a stable, diff-friendly literal.
    head = ranked[:n]
    words = sorted({w for w in head if w.isalpha() and len(w) >= 3})

    lines = [
        '"""Baked frozenset of common English words -- GENERATED, do not edit by hand.',
        "",
        "Regenerate with:  .venv/Scripts/python.exe scripts/build_common_words.py",
        "",
        f"Source: google-10000-english (public domain), top {n} by unigram frequency,",
        "alpha-only, len>=3. Used by ``_stt_correct`` to protect real words from the",
        "phonetic/fuzzy gazetteer snapper (only OOV/misheard tokens may be rewritten).",
        '"""',
        "",
        "COMMON_WORDS = frozenset({",
    ]
    # 8 words per line for readability.
    for i in range(0, len(words), 8):
        chunk = ", ".join(repr(w) for w in words[i:i + 8])
        lines.append(f"    {chunk},")
    lines.append("})")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} with {len(words)} words (from top {n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
