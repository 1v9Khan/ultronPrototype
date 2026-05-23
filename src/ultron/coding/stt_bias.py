"""STT bias-prompt manager (catalog T12, 2026-05-22 batch 14).

Accumulates domain-specific terms (file paths, function names, recent
identifiers) so an STT engine that supports a bias / initial-prompt
hook can favour those terms when transcribing the user's next
utterance. The aider pattern lifted here:

  * After every Claude edit / read / file mention, push the touched
    file's basename (sans extension) + any captured identifiers
    into the manager.
  * Before transcribing the next user utterance, the orchestrator
    asks the manager for the current bias prompt (one short string,
    capped at ``max_chars``) and forwards it to the engine.
  * Engines that don't support an initial-prompt hook ignore the
    string silently.

The manager is engine-agnostic: it just owns the prompt text. Engine
integration sits behind a tiny ``apply_bias_prompt`` helper
(consumers may keep their own).

Design notes:

* Bounded queue keyed by insertion order so the freshest terms win.
* Case-insensitive duplicate detection so we don't pollute the prompt
  with ``Foo`` + ``foo``.
* ``max_terms`` and ``max_chars`` both apply -- whichever hits first
  truncates. A tight budget protects engines (Whisper's
  ``initial_prompt`` window is ~224 tokens / ~880 chars).
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from typing import Iterable, List, Optional


logger = logging.getLogger("ultron.coding.stt_bias")


# Conservative identifier extractor. Catches snake_case, camelCase,
# and PascalCase; ignores leading numbers and single letters.
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def extract_identifiers(text: str) -> List[str]:
    """Return distinct identifier-like tokens from ``text``.

    Pure function. Order-preserving deduplication so callers can
    rank "most recent first" by simply prepending the result.
    """
    if not text:
        return []
    seen: dict[str, None] = {}
    for tok in _IDENTIFIER_RE.findall(text):
        # Skip Python keywords + a handful of common boilerplate words
        # that would dilute the prompt without helping STT.
        if tok.lower() in _IGNORE_TOKENS:
            continue
        if tok not in seen:
            seen[tok] = None
    return list(seen.keys())


_IGNORE_TOKENS = frozenset({
    "and", "the", "for", "with", "from", "this", "that", "into",
    "true", "false", "none", "self", "def", "class", "return",
    "import", "lambda", "yield", "async", "await", "raise", "try",
    "except", "finally", "while", "break", "continue", "pass",
    "global", "nonlocal", "elif", "else",
})


class STTBiasManager:
    """Process-wide bias term store with bounded freshness.

    Args:
        max_terms: Soft cap on stored terms. The oldest term is evicted
            when the queue exceeds this.
        max_chars: Hard cap on the generated prompt string (sum of
            term lengths + separator overhead). Picked to fit within
            common engine prompt windows (Whisper: ~224 tokens).
        separator: String inserted between terms in the rendered prompt.

    Thread-safe: all mutations and reads acquire an internal lock.
    """

    def __init__(
        self,
        *,
        max_terms: int = 64,
        max_chars: int = 600,
        separator: str = ", ",
    ) -> None:
        if max_terms < 1:
            raise ValueError("max_terms must be >= 1")
        if max_chars < 0:
            raise ValueError("max_chars must be >= 0")
        self._max_terms = int(max_terms)
        self._max_chars = int(max_chars)
        self._sep = separator
        self._lock = threading.Lock()
        # OrderedDict keys() preserve insertion order; we keep
        # case-canonical keys (lowercase) so duplicates collapse.
        self._terms: "OrderedDict[str, str]" = OrderedDict()

    # --- mutation ---------------------------------------------------------

    def add(self, term: str) -> None:
        """Push one term into the queue. No-op for empty / short terms."""
        cleaned = (term or "").strip()
        if len(cleaned) < 3:
            return
        key = cleaned.lower()
        with self._lock:
            # If the term already exists, move it to MRU position so
            # the newest insertion always wins.
            if key in self._terms:
                self._terms.pop(key)
            self._terms[key] = cleaned
            self._enforce_cap_locked()

    def add_many(self, terms: Iterable[str]) -> None:
        for t in terms:
            self.add(t)

    def add_from_text(self, text: str) -> None:
        """Extract identifiers from ``text`` and push each in order."""
        for ident in extract_identifiers(text):
            self.add(ident)

    def clear(self) -> None:
        """Drop every stored term."""
        with self._lock:
            self._terms.clear()

    def remove(self, term: str) -> bool:
        """Drop ``term`` if present. Returns whether it was present."""
        key = (term or "").strip().lower()
        with self._lock:
            return self._terms.pop(key, None) is not None

    # --- read -------------------------------------------------------------

    def terms(self) -> List[str]:
        """Snapshot of stored terms in MRU-last insertion order."""
        with self._lock:
            return list(self._terms.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._terms)

    def __contains__(self, term: str) -> bool:
        if not term:
            return False
        with self._lock:
            return (term or "").strip().lower() in self._terms

    # --- prompt rendering -------------------------------------------------

    def render_prompt(self) -> str:
        """Build the bias-prompt string. Empty when nothing's queued.

        Most-recent terms appear FIRST in the rendered prompt so the
        engine sees fresh context at the front. The string is then
        truncated at ``max_chars`` from the end (oldest gets dropped
        first).
        """
        with self._lock:
            # Most-recent first.
            mru_first = list(reversed(self._terms.values()))
        if not mru_first:
            return ""
        if self._max_chars == 0:
            return ""
        out: List[str] = []
        total = 0
        for term in mru_first:
            extra = len(term) + (len(self._sep) if out else 0)
            if total + extra > self._max_chars:
                break
            out.append(term)
            total += extra
        return self._sep.join(out)

    # --- internals --------------------------------------------------------

    def _enforce_cap_locked(self) -> None:
        """Drop oldest items while the queue is over ``max_terms``."""
        while len(self._terms) > self._max_terms:
            self._terms.popitem(last=False)


def apply_bias_prompt(stt_engine, prompt: str) -> bool:
    """Attempt to set ``prompt`` on ``stt_engine`` as a bias / initial
    prompt. Returns True when the engine accepted it, False otherwise.

    Heuristic: the engine accepts the prompt iff it has an attribute
    among the well-known names (``initial_prompt``, ``bias_prompt``,
    ``decoding_prompt``). Pure attribute-set; engines that read it from
    their own state pick it up on the next transcribe call.
    """
    if not prompt:
        return False
    for attr in ("initial_prompt", "bias_prompt", "decoding_prompt"):
        if hasattr(stt_engine, attr):
            try:
                setattr(stt_engine, attr, prompt)
                return True
            except Exception as exc:                              # noqa: BLE001
                logger.debug(
                    "stt_bias: setting %s on %s raised: %s",
                    attr, type(stt_engine).__name__, exc,
                )
                return False
    return False


__all__ = [
    "STTBiasManager",
    "apply_bias_prompt",
    "extract_identifiers",
]
