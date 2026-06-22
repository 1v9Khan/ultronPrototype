"""Ultron 1.0 — always-listening 3-way (4-class) intent gate (optional wakeword).

Per the research synthesis (Decision 2 / board doc C_gate): when always-listening is ON, every
finalized transcript is classified into ONE of:
  RELAY_TO_TEAM  -- a tactical callout/command meant for teammates  (-> relay path / team mic)
  PRIVATE_REPLY  -- the player talking to Ultron, wants a ME-ONLY answer (-> desktop channel)
  COMMAND_LOCAL  -- a local control command (flavor/verbosity/thinking/device toggle, Spotify, stop)
  IGNORE         -- talking to Discord / stream / out loud, or ambiguous -> discard (no output)

DESIGN -- cost-asymmetric, FAIL-CLOSED to IGNORE. A false RELAY is broadcast to the team and is far
worse than a missed one; anything ambiguous defaults to IGNORE. This is a COMPOSITION of existing,
proven components (no new ML in-process):
  * COMMAND_LOCAL: the toggle/Spotify/stop matchers (relay_speech).
  * RELAY_TO_TEAM: ``match_relay_command`` (strict grammar) + the relay-intent gate
    (``_relay_intent.relay_intent_ok``, semantic+lexical) + ``is_complete_tactical_callout``.
  * PRIVATE_REPLY: the addressing YES-rules (factual question / imperative / direct address) AND/OR a
    wake-word/name signal -- ONLY when the utterance is clearly addressed to Ultron.
  * IGNORE: the addressing NO-rules (phone opener, third-person mention, third-party narrative,
    interjection) + the default.
  * ASR-confidence PRE-REJECT (``no_speech_prob`` / ``avg_logprob`` from faster-whisper) -> IGNORE --
    a free, high-value signal (Apple DDSD: +6.9% rel. EER) that also catches Whisper hallucinations
    on non-speech (which are ~40-52% on short/silent audio).
The LLM is consulted ONLY in the undecided band (PRIVATE vs IGNORE), with ``enable_thinking=False`` and a
single-token, fail-CLOSED parse (grammar+thinking conflict, llama.cpp #20345).

DEFAULT OFF (opt-in via ``addressing.always_listening``). The wake word stays the competitive default;
each false relay is a team-visible blast (asymmetric cost). Thresholds here are HEURISTIC starting points
-- calibrate on the labeled MP3 battery + real-session ``logs/addressing.jsonl`` (C_gate: needs ~200
labeled turns; binary-cascade decomposition for the 3-way split). PREREQUISITE: VoiceMeeter mic isolation
(if Discord/teammate audio bleeds into the user's mic bus, NO gate can help -- teammate speech == user speech).

Anticheat-safe: stdlib + the existing loopback embedder sidecar; nothing on a desktop-interaction surface.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class Scenario(str, Enum):
    RELAY_TO_TEAM = "RELAY_TO_TEAM"
    PRIVATE_REPLY = "PRIVATE_REPLY"
    COMMAND_LOCAL = "COMMAND_LOCAL"
    IGNORE = "IGNORE"


@dataclass(frozen=True)
class ScenarioVerdict:
    scenario: Scenario
    confidence: float
    reason: str
    # True when the cheap layers were undecided (PRIVATE vs IGNORE) and the caller MAY escalate to the
    # LLM (resolve_with_llm). Until escalation, ``scenario`` holds the fail-closed default (IGNORE).
    needs_llm: bool = False


# --- Calibratable thresholds (env-overridable; defaults are heuristic per C_gate) --------------------
def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


# faster-whisper no_speech_prob above this -> almost certainly ambient/non-speech -> IGNORE.
_NO_SPEECH_REJECT = _envf("KENNING_GATE_NO_SPEECH_REJECT", 0.60)
# faster-whisper avg_logprob below this -> very low-confidence transcript -> IGNORE (guard: gunfire
# bleed lowers avg_logprob even on clear speech, so keep this permissive).
_AVG_LOGPROB_REJECT = _envf("KENNING_GATE_AVG_LOGPROB_REJECT", -1.6)
# addressing-rule confidence needed to commit a NO (-> IGNORE) or a YES (-> PRIVATE) without the LLM.
_RULE_TAU = _envf("KENNING_GATE_RULE_TAU", 0.80)

# Pre-LLM reaction filter (2026-06-21): a bare agreement / reaction opener with no
# question and no name for Ultron is friend-chatter -- drop it WITHOUT spending the
# LLM (the live leak was "Yeah, I can." / "It's okay." / "I pranked you..." reaching
# the LLM band and being mislabelled PRIVATE). Conservative -- a real "Ultron, ..."
# carries the name token below, so this never suppresses a genuinely-addressed line.
_REACTION_OPENERS: frozenset[str] = frozenset({
    "yeah", "yep", "yup", "nah", "naw", "sure", "okay", "ok", "kay", "nice",
    "lol", "lmao", "haha", "hahaha", "lmfao", "damn", "dang", "bruh", "oh", "ah",
    "huh", "alright", "aight", "right", "true", "fair", "bet", "its", "it's",
    "thats", "that's", "mhm", "mmhm", "yikes", "oof", "sheesh", "welp", "mm",
})
# A direct name/address token for Ultron -- its presence vetoes the reaction filter.
_NAME_TOKEN_RE = re.compile(
    r"\b(?:ultron|kenning|machine|robot|hey\s+ai|the\s+ai)\b", re.IGNORECASE)


def _wake_present(text: str) -> bool:
    """Leading wake word ('ultron'/'kenning', incl. common ASR variants) = a strong addressed signal."""
    import re
    return bool(re.match(r"^\s*(?:hey[\s,]+|okay[\s,]+|ok[\s,]+)?"
                         r"(?:ultron|kenning|altron|ultraun|ultro)\b", text, re.IGNORECASE))


def _is_command_local(text: str) -> bool:
    """Local control commands: flavor/verbosity/thinking/relay/device toggles + Spotify + stop."""
    try:
        from kenning.audio import relay_speech as rs
        if (rs.match_flavor_toggle(text) is not None
                or rs.match_thinking_toggle(text) is not None
                or rs.match_relay_toggle(text) is not None
                or rs.match_verbosity_command(text) is not None
                or rs.match_llm_device_switch(text) is not None):
            return True
    except Exception:  # noqa: BLE001 - fail-open to "not a local command"
        pass
    # The STOP-window + LOGS-window summon/dismiss are local UI commands too, so
    # they reach the dispatch handlers in always-listening mode (not the relay or
    # private-reply path). Pure-regex matchers -> no tkinter import here.
    try:
        from kenning.audio.stop_button import match_stop_button_command
        if match_stop_button_command(text) is not None:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from kenning.audio.log_viewer import match_logs_command
        if match_logs_command(text) is not None:
            return True
    except Exception:  # noqa: BLE001
        pass
    # "ultron stop" / "stop" all-channel cancel is a local command too.
    import re
    if re.match(r"^\s*(?:ultron[\s,]+)?stop\b\s*[.!?]*$", text, re.IGNORECASE):
        return True
    return False


def _relay_signal(text: str, names: Optional[Sequence[str]]) -> Optional[float]:
    """Return a confidence in [0,1] that this is a team relay, or None if no relay signal.

    Applies ONLY the L1 STT correction first (``correct_callout_stt`` -- fixes casing/agent-name
    mishears so the slot parser recognizes 'sova hit 84 on a main'). It deliberately does NOT run the
    full ``normalize_command`` (its relay-lead RECOVERY aggressively prepends 'tell my team' to bare
    callouts and would FALSE-POSITIVE banter like 'the rotations feel clean' into RELAY -- the exact
    failure mode C_gate warns against). Strict matcher / complete-tactical-callout = high confidence;
    the semantic relay-intent gate = moderate. Fail-open to None (no relay) on any error.
    """
    norm = text
    try:
        from kenning.audio._stt_correct import correct_callout_stt
        norm = correct_callout_stt(text) or text
    except Exception:  # noqa: BLE001
        norm = text
    try:
        from kenning.audio import relay_speech as rs
        if rs.match_relay_command(norm, names=names) is not None:
            return 0.95
        if rs.is_complete_tactical_callout(norm):
            return 0.90
        # An AGENT + a concrete tactical token (count/location/ability) is a callout even when a
        # loose token like "main" keeps the strict slot parser from firing ("Sova hit 84 on A main").
        # Mirrors build_relay_line's tactical-literal pre-route; requires BOTH so a question that
        # merely contains a number ("what round is it") is not mistaken for a relay.
        nums, ags, locs, abils = rs._fact_tokens(norm)
        if ags and (len(nums) + len(locs) + len(abils)) >= 1:
            return 0.88
    except Exception:  # noqa: BLE001
        pass
    # NB (2026-06-21): the semantic relay-intent gate (_relay_intent.relay_intent_ok)
    # is deliberately NOT used here as a positive RELAY signal. It is a VETO tool
    # (tuned for recall in the normalizer -- biased to "plausibly a relay"), so as a
    # positive classifier it FALSE-POSITIVES conversation ("nice shot dude", "hey mom
    # how are you", "that's not even that long") into RELAY whenever the sidecar is
    # reachable -- and a false RELAY is broadcast to the team (the worst case in this
    # cost-asymmetric gate). RELAY_TO_TEAM therefore requires a STRONG, PRECISE signal
    # (strict matcher / complete tactical callout / agent+fact-token, above); a bare
    # directive-only callout the slot grammar can't structure is left to the wake word.
    return None


def _addressing_hit(text: str, seconds_since_response: float):
    try:
        from kenning.addressing import rules as addr_rules
        return addr_rules.classify(text, seconds_since_response)
    except Exception:  # noqa: BLE001
        return None


def classify_scenario(
    text: str,
    *,
    wake_present: Optional[bool] = None,
    seconds_since_response: float = 999.0,
    no_speech_prob: float = 0.0,
    avg_logprob: float = 0.0,
    names: Optional[Sequence[str]] = None,
) -> ScenarioVerdict:
    """Classify a finalized transcript into a Scenario (cheap layers only; fail-CLOSED to IGNORE).

    ``wake_present`` overrides the leading-wake detection when the caller already knows (e.g. the
    audio-domain wake detector fired). ASR-confidence args come from faster-whisper.
    """
    raw = (text or "").strip()
    if not raw:
        return ScenarioVerdict(Scenario.IGNORE, 0.99, "empty")

    # 1) ASR-confidence pre-reject (free; catches ambient + Whisper non-speech hallucinations).
    if no_speech_prob >= _NO_SPEECH_REJECT:
        return ScenarioVerdict(Scenario.IGNORE, 0.90, f"asr no_speech_prob {no_speech_prob:.2f}")
    if avg_logprob and avg_logprob <= _AVG_LOGPROB_REJECT:
        return ScenarioVerdict(Scenario.IGNORE, 0.80, f"asr avg_logprob {avg_logprob:.2f}")

    wake = _wake_present(raw) if wake_present is None else bool(wake_present)

    # 2) COMMAND_LOCAL (toggles / Spotify / stop) -- handled locally, definitely addressed.
    if _is_command_local(raw):
        return ScenarioVerdict(Scenario.COMMAND_LOCAL, 0.95, "local control command")

    # 3) RELAY_TO_TEAM -- strict matcher / tactical callout / semantic relay-intent.
    relay_conf = _relay_signal(raw, names)
    if relay_conf is not None:
        return ScenarioVerdict(Scenario.RELAY_TO_TEAM, relay_conf, "relay signal")

    # 4) Addressing rules: a confident NO -> IGNORE (talking to a person / stream / self).
    hit = _addressing_hit(raw, seconds_since_response)
    try:
        from kenning.addressing.rules import AddressingDecision
    except Exception:  # noqa: BLE001
        AddressingDecision = None  # type: ignore
    if hit is not None and AddressingDecision is not None:
        if hit.decision == AddressingDecision.NOT_ADDRESSED and hit.confidence >= _RULE_TAU:
            return ScenarioVerdict(Scenario.IGNORE, hit.confidence, f"addressing NO: {hit.reason}")
        # 5) Confident YES (or wake) -> PRIVATE_REPLY (talking to Ultron, not a relay).
        if wake or (hit.decision == AddressingDecision.ADDRESSED and hit.confidence >= _RULE_TAU):
            return ScenarioVerdict(Scenario.PRIVATE_REPLY,
                                   max(hit.confidence, 0.85 if wake else hit.confidence),
                                   "addressed to Ultron (private)")
        # 6) Undecided band (UNCERTAIN, or sub-tau) -> fail-closed IGNORE, flag for LLM escalation.
        return ScenarioVerdict(Scenario.IGNORE, 0.50, f"undecided: {hit.reason}", needs_llm=True)

    # No addressing hit at all: wake word alone still routes to PRIVATE.
    if wake:
        return ScenarioVerdict(Scenario.PRIVATE_REPLY, 0.85, "leading wake word")
    # Pre-LLM reaction filter: a bare reaction/agreement opener with no question and
    # no name for Ultron is friend-chatter -- drop it cheaply rather than letting the
    # LLM band mislabel it PRIVATE (the live "Yeah, I can." / "It's okay." leak).
    _low = (raw or "").strip().lower()
    _first = re.split(r"[\s,.!?]+", _low, maxsplit=1)[0] if _low else ""
    if (_first in _REACTION_OPENERS and "?" not in _low
            and not _NAME_TOKEN_RE.search(_low)):
        return ScenarioVerdict(Scenario.IGNORE, 0.70, "reaction opener (no address)")
    # Else fail-closed IGNORE, flagged for the LLM escalation.
    return ScenarioVerdict(Scenario.IGNORE, 0.55, "no addressing signal (fail-closed)", needs_llm=True)


# Single-token classification prompt for the LLM escalation in the undecided band (PRIVATE vs IGNORE).
_LLM_GATE_SYSTEM = (
    "You are Ultron, an AI teammate in a live Valorant match. The player is on voice with friends and "
    "their stream, so MOST of what you hear is NOT for you. Decide if a transcribed line is the player "
    "speaking DIRECTLY TO YOU and wanting a reply. Answer PRIVATE only when the line clearly names you "
    "(Ultron / the machine / hey AI), asks YOU a direct question, or gives YOU a direct command. Answer "
    "IGNORE for everything else: agreements and reactions ('yeah I can', \"it's okay\", 'sure', 'nice'); "
    "talking to teammates, the stream, or themselves; jokes, banter, narration; anything ambiguous. When "
    "unsure, answer IGNORE. Output ONE word only: PRIVATE or IGNORE.\n"
    "Examples:\n"
    "Ultron, what's their economy? -> PRIVATE\n"
    "hey can you tell me the round number -> PRIVATE\n"
    "machine, mute yourself -> PRIVATE\n"
    "Yeah, I can. -> IGNORE\n"
    "It's okay, it's okay. -> IGNORE\n"
    "I pranked you into thinking we could play. -> IGNORE\n"
    "nice shot dude -> IGNORE"
)


def resolve_with_llm(verdict: ScenarioVerdict, text: str, llm) -> ScenarioVerdict:
    """Escalate an undecided verdict to the LLM for a single-token {PRIVATE, IGNORE} decision.

    FAIL-CLOSED: any non-PRIVATE token, parse failure, or error -> keep IGNORE. enable_thinking=False
    (grammar/thinking conflict). Returns the resolved verdict (or the original if no escalation needed).
    """
    if not verdict.needs_llm or llm is None or not hasattr(llm, "generate_stream"):
        return verdict
    try:
        out = "".join(llm.generate_stream(
            f'Line: "{text.strip()}"\nOne word -- PRIVATE or IGNORE:',
            system_prompt=_LLM_GATE_SYSTEM,
            sampling={"max_tokens": 4, "temperature": 0.0},
            enable_thinking=False, suppress_memory_context=True, record_history=False,
        )).strip().upper()
    except Exception:  # noqa: BLE001 - fail closed
        return verdict
    # Strip any stray <think> and take the first alpha token.
    import re
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    first = next((w for w in re.findall(r"[A-Z]+", out)), "")
    # Fail-closed: ONLY an exact PRIVATE escalates; any other token (including a
    # PRIVATE-prefixed hallucination) stays IGNORE, and IGNORE carries the HIGHER
    # confidence so the default genuinely favours dropping non-addressed chatter.
    if first == "PRIVATE":
        return ScenarioVerdict(Scenario.PRIVATE_REPLY, 0.65, "LLM band escalation -> PRIVATE")
    return ScenarioVerdict(Scenario.IGNORE, 0.75, "LLM band escalation -> IGNORE (fail-closed)")
