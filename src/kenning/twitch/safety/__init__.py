"""Deterministic, anticheat-clean chat-safety primitives (L1/L5/L6 text layers).

Pure stdlib + rapidfuzz — importable in EITHER the main voice process or a
sidecar. These are the layers that work WITHOUT any model: the abliterated 8B is
treated as hostile, so the structural deterministic checks are the load-bearing
defense. Model-backed layers (Prompt-Guard-2, the guard model, Whisper L7, PII)
live in sidecar clients that fail-CLOSED when their sidecar/deps are absent.

Modules:
  * ``normalize``  — the frozen canonicalizer (covert-channel strip, NFKC,
    confusable fold, leet/separator/repeat de-obfuscation, multiple match forms).
  * ``blocklist``  — the dual-scan slur/hate/dox/threat/injection matcher
    (word-boundary + phonetic + fuzzy on the hard-slur subset), fail-CLOSED.
  * ``reassembly`` — (L5) materialize acrostics / spell-outs / NATO / ciphers so
    hidden output channels are re-screened.
  * ``phonetic``   — (L6) deterministic phonetic-key matching (cross-word slurs).
  * ``deflection`` — constant-string in-character deflections (never model-made).
"""
from __future__ import annotations

__all__: list[str] = []
