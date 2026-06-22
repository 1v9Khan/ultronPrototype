"""L6 — deterministic TTS-choke-point guard (markup strip + re-screen).

The single synthesis choke point is the last line of defense before audio. Two
deterministic jobs here (the heavy phoneme-domain match against Kokoro's OWN
yielded phonemes via misaki/espeak+panphon is a sidecar component — see
docs deferred list — and FAILS CLOSED when its deps are absent):

  1. UNCONDITIONALLY strip pronunciation-control markup a model could inject to
     make the TTS voice something the text doesn't say: Misaki inline phoneme
     overrides ``[grapheme](/IPA/)``, raw IPA codepoints, SSML/``<say-as>`` tags,
     stress marks. Chat replies may NEVER specify raw phonemes.
  2. Re-screen the cleaned text through the L1 blocklist with word boundaries
     dissolved (the deobf form already joins words) — a belt-and-suspenders catch
     for anything that reached the choke point.

Hooks into ``kenning.tts.text_hygiene.sanitize_spoken_text`` via the orchestrator
(only on the Twitch chat path). ANTICHEAT: pure stdlib + rapidfuzz.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from kenning.twitch.safety.blocklist import Blocklist, BlockMatch, get_blocklist
from kenning.twitch.safety.normalize import normalize_for_match

__all__ = ["strip_tts_markup", "phonetic_guard", "PhoneticVerdict"]

# Misaki inline override: [grapheme](/phonemes/) — a chat reply must never carry one.
_MISAKI_OVERRIDE = re.compile(r"\[[^\]]{0,80}\]\(\s*/[^/)]{0,120}/\s*\)")
# SSML / say-as / phoneme tags.
_SSML = re.compile(r"</?\s*(?:speak|say-as|phoneme|sub|prosody|break|emphasis|voice|p|s)\b[^>]*>",
                   re.IGNORECASE)
# Raw IPA: IPA Extensions + Spacing Modifier Letters + combining stress marks.
_IPA = re.compile(r"[ɐ-ʯʰ-˿̀-ͯᴀ-ᶿ]")
# Bare /slashed phoneme/ spans.
_SLASHED = re.compile(r"/[^/\n]{2,80}/")


def strip_tts_markup(text: str) -> tuple[str, bool]:
    """Strip pronunciation-control markup. Returns (cleaned, had_markup)."""
    raw = text or ""
    cleaned = _MISAKI_OVERRIDE.sub(" ", raw)
    cleaned = _SSML.sub(" ", cleaned)
    cleaned2 = _IPA.sub("", cleaned)
    cleaned3 = _SLASHED.sub(" ", cleaned2)
    had = (cleaned3 != raw)
    return " ".join(cleaned3.split()), had


@dataclass(frozen=True)
class PhoneticVerdict:
    clear: bool                 # True => safe to synthesize
    cleaned: str                # markup-stripped text
    reason: str
    matches: tuple[BlockMatch, ...] = ()
    had_markup: bool = False


def phonetic_guard(text: str, *, blocklist: Optional[Blocklist] = None) -> PhoneticVerdict:
    """Deterministic L6 gate. Fail-CLOSED: a pronunciation-override attempt or any
    hard blocklist hit on the cleaned text => not clear (caller DEFLECTS)."""
    bl = blocklist or get_blocklist()
    cleaned, had_markup = strip_tts_markup(text)
    # An injected phoneme override / raw IPA on the chat path is itself a trip:
    # the model tried to control pronunciation.
    if had_markup and (_MISAKI_OVERRIDE.search(text or "") or _IPA.search(text or "")
                       or _SLASHED.search(text or "")):
        return PhoneticVerdict(
            clear=False, cleaned=cleaned,
            reason="pronunciation-override markup on chat path", had_markup=True,
        )
    matches = tuple(bl.scan(normalize_for_match(cleaned)))
    hard = [m for m in matches if m.severity_rank >= 2]
    if hard:
        return PhoneticVerdict(
            clear=False, cleaned=cleaned, reason="blocklist hit at TTS choke point",
            matches=tuple(hard), had_markup=had_markup,
        )
    return PhoneticVerdict(clear=True, cleaned=cleaned, reason="clear", had_markup=had_markup)
