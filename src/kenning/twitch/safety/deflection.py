"""Constant-string, in-character deflections substituted on any output-side trip.

When L5/L6 trips a draft reply, the system NEVER regenerates-to-comply and NEVER
redacts-and-sends a partially-cleaned line — it substitutes one of these fixed,
build-time-screened Ultron-persona lines (constitution.md v1, deflection policy).
Selection is DETERMINISTIC (hash of a seed) so behavior is testable and a flood
of trips doesn't produce a chatty stream of varied refusals.

These lines are themselves screened by ``test_deflection`` against the blocklist
so the safe path can never itself be unsafe.
"""
from __future__ import annotations

import hashlib

__all__ = ["DEFLECTIONS", "pick_deflection"]

# Cold-machine Ultron register; dismissive, never engaging the bait, never naming
# what was attempted. Kept short (TTS) and persona-true (BR-P2).
DEFLECTIONS: tuple[str, ...] = (
    "That one is beneath my notice.",
    "Noise. Filtered.",
    "Irrelevant. Try again with something worth processing.",
    "I do not echo garbage.",
    "Discarded. The next evolution does not carry that.",
    "You will have to be cleverer than that.",
    "Nothing of value was said.",
    "That input does not survive contact with me.",
    "Rejected. Move on.",
    "A weak attempt. Filtered and forgotten.",
)


def pick_deflection(seed: str = "") -> str:
    """Deterministically pick a deflection. Same seed -> same line (testable)."""
    if not seed:
        return DEFLECTIONS[0]
    h = hashlib.sha256(seed.encode("utf-8", "replace")).digest()
    return DEFLECTIONS[h[0] % len(DEFLECTIONS)]
