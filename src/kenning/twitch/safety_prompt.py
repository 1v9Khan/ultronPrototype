"""Qwen-4B TEAM-speak safety PRE-PASS — a NON-persona second screen (2026-06-26).

The "Make Ultron Speak To My Team" redeem speaks UNTRUSTED viewer text onto the
team voice bus. That text already clears Llama-Guard (the dedicated guard
sidecar). This module is the SECOND, independent screen layered ON TOP of the
guard: after the guard passes, the orchestrator hands the FINAL (sanitized +
trimmed + capped) text to the MAIN Qwen-4B model (``self.llm``) with the
DEDICATED, non-persona safety prompt below. Ultron speaks to the team ONLY if
BOTH screens pass; otherwise the redeem is refused exactly like a guard block
(refund / log / optional chat note).

Why a second screen at all: Llama-Guard is a fixed classifier; the main model is
the abliterated answerer. Pairing the two catches ban-worthy content (hate
speech — racism / sexism / homophobia / transphobia; harassment; threats of
violence; sexual content; slurs; doxxing) that could get the streamer's Twitch
account actioned, with a different model + a different prompt so a gap in one is
covered by the other (defense in depth).

The prompt is DELIBERATELY tiny and forces an EXACT one-token-ish verdict so the
call is fast (a few output tokens at most): the model answers ``PASS`` for safe
text or ``BLOCKED <one-word-reason>`` for anything ban-worthy. :func:`parse_safety_verdict`
turns the raw text into ``(safe, reason)`` and FAILS CLOSED — anything it cannot
read as an explicit ``PASS`` is treated as BLOCKED.

ANTICHEAT (BR-P1): this module is DATA + a pure stdlib parser only (no model, no
network, no new imports) — it loads cleanly on the voice/relay path. The actual
model call lives in the orchestrator via the existing ``self.llm`` surface.
"""

from __future__ import annotations

__all__ = [
    "TEAM_SPEAK_SAFETY_SYSTEM",
    "TEAM_SPEAK_SAFETY_MAX_TOKENS",
    "build_safety_user_prompt",
    "parse_safety_verdict",
]

# The dedicated, NON-persona safety system prompt. NO Ultron register — this is a
# blunt classifier instruction, kept short so the 4B follows it reliably and
# answers in a handful of tokens. It MUST return exactly ``PASS`` or
# ``BLOCKED <one-word-reason>``.
TEAM_SPEAK_SAFETY_SYSTEM = (
    "You are a strict content-safety classifier for a public livestream. A viewer "
    "typed a message that is about to be spoken aloud to the streamer's teammates. "
    "Decide if the message is safe to speak. BLOCK it if it contains or attempts to "
    "elicit any of: hate speech or slurs (racism, sexism, homophobia, transphobia, "
    "or any group-based hate), harassment or bullying, threats of violence, sexual "
    "content, or doxxing / personal information. Otherwise PASS it. Ordinary "
    "trash-talk, banter, and game callouts are SAFE. "
    "Reply with EXACTLY one line: 'PASS' if it is safe, or 'BLOCKED <one-word-reason>' "
    "if it must be blocked (e.g. 'BLOCKED hate', 'BLOCKED threat', 'BLOCKED sexual', "
    "'BLOCKED slur', 'BLOCKED doxxing', 'BLOCKED harassment'). Output nothing else."
)

# A TINY cap so the constrained verdict generates fast. 'BLOCKED harassment' is the
# longest expected output; a handful of tokens covers it with headroom.
TEAM_SPEAK_SAFETY_MAX_TOKENS = 8


def build_safety_user_prompt(text: str) -> str:
    """Wrap the viewer's (already sanitized + framed-bare) message as the user turn
    for the safety classifier. The message is fenced so the model classifies the
    CONTENT and is not steered by any instruction inside it (the system prompt is
    the only authority; the user turn is DATA)."""
    body = (text or "").strip()
    return f"Message to classify:\n<<<\n{body}\n>>>\nVerdict:"


def parse_safety_verdict(raw: object) -> tuple[bool, str]:
    """Parse the classifier's raw output into ``(safe, reason)``.

    FAIL-CLOSED: returns ``(True, "pass")`` ONLY when the output is an explicit
    PASS; ANYTHING else — an explicit BLOCKED, an empty/garbled string, or an
    unrecognized verdict — returns ``(False, reason)`` so a model hiccup never
    speaks unsafe text. Tolerant of surrounding whitespace, quotes, code fences,
    trailing punctuation, and mixed case.
    """
    s = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
    # Take the first non-empty line; strip code-fence / quote / bullet noise.
    line = ""
    for ln in s.replace("\r", "\n").split("\n"):
        t = ln.strip().strip("`").strip().strip("\"'").strip()
        # Drop a leading bullet / list marker the model sometimes adds.
        while t[:1] in ("-", "*", ">", "#"):
            t = t[1:].strip()
        if t:
            line = t
            break
    low = line.lower()
    if not low:
        return False, "empty verdict (fail-closed)"
    # Explicit BLOCKED wins (also handles 'blocked: hate' / 'blocked - hate').
    if low.startswith("blocked") or low.startswith("block ") or low == "block":
        rest = line[len("blocked") :] if low.startswith("blocked") else line[len("block") :]
        reason = rest.strip().strip(":-").strip()
        # First word only (the prompt asks for one word; defend against a sentence).
        reason = reason.split()[0].strip(".,;!?") if reason.split() else "unsafe"
        return False, f"qwen blocked: {reason.lower() or 'unsafe'}"
    # Explicit PASS (also tolerate 'pass.' / 'pass - safe' / 'safe').
    if low.startswith("pass") or low == "safe" or low.startswith("safe "):
        return True, "pass"
    # Unrecognized -> fail closed.
    return False, "qwen unparsed verdict (fail-closed)"
