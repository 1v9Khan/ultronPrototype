"""Pure text-side Whisper decode knowledge, shared by both STT paths.

Extracted 2026-07-26 when STT gained an out-of-process sidecar
(``scripts/stt_server.py``). The decoder-priming vocabulary and the
non-speech blocklist are DECODE QUALITY, not transport, so they must apply
identically whether faster-whisper runs in this process
(:mod:`kenning.transcription.whisper_engine`) or in the sidecar child --
otherwise moving STT to another GPU would silently cost agent-name accuracy
and re-admit "Thank you."-type phantom callouts.

Deliberately **stdlib-only** (``re``): the sidecar imports this at startup and
must not be able to die on a config-validation error, and the module sits on
the anticheat-pinned voice path (BR-P1).
"""

from __future__ import annotations

import re

# Closed Valorant vocabulary fed to the decoder as initial_prompt (domain
# biasing) so agent names + callout terms are recognised at the source. <=200
# tokens; most-confusable proper nouns first. Overridable via WHISPER_INITIAL_PROMPT.
DOMAIN_PROMPT = (
    "Valorant team comms. Agents: Raze, Jett, Sova, Omen, Killjoy, Cypher, Viper, "
    "Phoenix, Sage, Reyna, Breach, Fade, Skye, Astra, Harbor, Clove, Chamber, "
    "Brimstone, Gekko, Yoru, Iso, Deadlock, Tejo, Waylay, Vyse, Neon, KAY/O. "
    "Calls: spike, plant, defuse, ult, smoke, flash, molly, dart, rotate, eco, "
    "save, push, heaven, mid, long, short, A site, B site, C site, lurk, flank."
)

# faster-whisper emits stock phrases ("Thank you.", "Thanks for watching",
# "you", ".") on near-silence / room tone / non-speech audio. On the gaming
# relay path a false transcript would fire a bogus team callout or a
# conversational turn, so when the WHOLE transcript normalises to one of these
# it is dropped. Kept deliberately NARROW -- only phrases that are never a
# meaningful standalone command (real commands like "you're welcome" untouched).
WHISPER_HALLUCINATIONS = frozenset({
    "thank you", "thanks", "thank you so much", "thank you very much",
    "thanks for watching", "thank you for watching", "thanks for watching everyone",
    "please subscribe", "subscribe", "thanks for listening", "you", "bye",
    "bye bye", "the", "music", "applause", "silence", "background noise",
    "i'm sorry", "oh", "hmm", "mm", "mmm", "uh", "um", "ah",
})


def is_whisper_hallucination(text: str) -> bool:
    """True when ``text`` is, in whole, a known faster-whisper non-speech
    artifact (case/punctuation-insensitive)."""
    norm = re.sub(r"[^\w\s']", " ", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm in WHISPER_HALLUCINATIONS


def build_initial_prompt(user_override: str = "") -> str:
    """The decoder-priming prompt: domain vocabulary, plus any user override.

    2026-06-18 fix preserved here: a user-set ``WHISPER_INITIAL_PROMPT`` must
    AUGMENT the Valorant domain vocabulary, not REPLACE it. An earlier
    ``user or DOMAIN_PROMPT`` let a short override (e.g. ``.env``
    ``KENNING_WHISPER_INITIAL_PROMPT='Kenning.'``) SHADOW the whole domain
    prompt -> domain biasing effectively OFF -> agent-name jargon errors
    (Sova->Silva) and phantom leads ("Also team ...").
    """
    user = (user_override or "").strip()
    return f"{DOMAIN_PROMPT} {user}".strip() if user else DOMAIN_PROMPT


__all__ = [
    "DOMAIN_PROMPT",
    "WHISPER_HALLUCINATIONS",
    "is_whisper_hallucination",
    "build_initial_prompt",
]
