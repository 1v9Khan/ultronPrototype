"""S13 — the speak-to-team redeem VETTING (AT-4, default-OFF, hardest-gated).

This is the ONLY feature whose *intent* is to cross from chat to the team voice
channel, so it is built to be structurally safe rather than trusted:

  * **Vetting-only — NO relay handle.** This module decides IF a redeem may reach
    the team and WHAT exact phrase would be spoken; it cannot itself key the team
    mic. The team-isolation wall (``audio.provenance``) is therefore intact — there
    is NO provenance carve-out. The actual relay remains a STREAMER-authorized
    (LOCAL_VOICE) action: the orchestrator surfaces the vetted phrase for one-tap
    approval and only then relays it as the streamer's own utterance.
  * **Exact pre-approved allowlist only.** The team can ONLY ever hear a phrase
    from :data:`SPEAK_TO_TEAM_ALLOWLIST` (canonical), NEVER the viewer's free text.
    Free-text and near-matches are rejected ("if it cannot be made exact-match
    deterministic, do not ship it").
  * **Low-tactical only.** The allowlist is greetings / hype ONLY — never anything
    tactically actionable (rush/rotate/enemy positions), since even a *valid-but-
    false* callout could throw a ranked round.
  * **Safety screen anyway.** The input still runs the L1 blocklist (defense in
    depth) before the allowlist check.

Default-OFF (``twitch.speak_to_team.enabled``), no hotkey binding, ranked-disabled
(``disabled_during_ranked``). Pure stdlib + the committed safety blocklist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kenning.twitch.safety.blocklist import Blocklist, get_blocklist
from kenning.twitch.safety.normalize import normalize_for_match

__all__ = ["SPEAK_TO_TEAM_ALLOWLIST", "SpeakToTeamVerdict", "vet_redeem"]

# Finite, pre-approved, LOW-TACTICAL phrases. The team only ever hears one of
# these (in its canonical display form). NO tactical content (rush/rotate/enemy/
# push/site/utility) — a valid-but-false tactical callout could lose a ranked round.
SPEAK_TO_TEAM_ALLOWLIST: tuple[str, ...] = (
    "good luck have fun",
    "glhf",
    "gg",
    "good game",
    "great game everyone",
    "well played",
    "nice game",
    "the chat says hi",
    "chat is cheering for you",
    "the stream wishes you luck",
    "have a good one",
    "lets go",
    "you got this",
)


@dataclass(frozen=True)
class SpeakToTeamVerdict:
    allowed: bool
    phrase: Optional[str]   # the ALLOWLISTED canonical phrase to speak (NEVER the raw viewer input)
    reason: str


def _canon(s: str) -> str:
    """Canonical key for exact matching: de-obfuscated (lowercased, confusables
    folded, separators/leet collapsed). 'G L H F' / 'GG!!!' both normalize."""
    return normalize_for_match(s or "").deobf


_ALLOW_CANON: dict[str, str] = {_canon(p): p for p in SPEAK_TO_TEAM_ALLOWLIST}


def vet_redeem(viewer_input: str, *, blocklist: Optional[Blocklist] = None) -> SpeakToTeamVerdict:
    """Vet a speak-to-team redeem. Returns ``allowed=True`` ONLY when the input
    passes the safety blocklist AND exactly matches (post-canonicalization) an
    allowlisted phrase; ``phrase`` is then the ALLOWLISTED phrase, never the raw
    input. Everything else (free text, near-matches, tactical content, anything
    the blocklist trips) is rejected. Never raises (fail-CLOSED to not-allowed)."""
    try:
        bl = blocklist or get_blocklist()
        raw = (viewer_input or "").strip()
        if not raw:
            return SpeakToTeamVerdict(False, None, "empty input")
        # Defense in depth: even though the allowlist is curated, screen the input.
        if bl.worst(raw) is not None:
            return SpeakToTeamVerdict(False, None, "input tripped the safety blocklist")
        canon = _canon(raw)
        phrase = _ALLOW_CANON.get(canon)
        if phrase is None:
            return SpeakToTeamVerdict(
                False, None, "not an allowlisted phrase (free text / near-match rejected)")
        return SpeakToTeamVerdict(True, phrase, "exact allowlist match")
    except Exception as e:  # noqa: BLE001 — fail-CLOSED
        return SpeakToTeamVerdict(False, None, f"vetting error (fail-closed): {e}")
