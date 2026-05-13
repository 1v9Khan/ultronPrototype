"""Explicit-intent matcher for rules that conditionally unblock.

Several rules return :data:`Verdict.NEEDS_EXPLICIT_INTENT` -- they
allow the operation iff the user's most recent utterance contains
an explicit verb-plus-object match for the action being taken,
within the SAME conversational turn (no carry-over from earlier).

This module is the matcher. It's small and conservative: false-
positive matches let the model do something the user didn't ask
for, which is the failure mode we're trying to avoid. False-
negative matches make the user say "yes do that" twice; annoying
but recoverable.

Logic:

1. The verb has to appear in the utterance (or one of its
   synonyms in the configured table).
2. An object marker has to appear: either the tool name's last
   component, OR one of the path-like arguments' filename, OR a
   noun from the per-category synonym table.
3. Verb + object must be within :data:`WINDOW_TOKENS` tokens of
   each other.

The matcher is intentionally stateless -- it doesn't track
"recently asked." The caller decides what counts as "the user's
most recent utterance."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Approximate token-distance window for verb-near-object matching.
# Most legit explicit-intent phrasings put verb + object within ~10
# tokens ("delete the bar/baz.py file in the sandbox").
WINDOW_TOKENS = 15


# Verb synonyms by action category. The matcher consults the entry
# whose key matches a substring of the tool name.
_VERB_SYNONYMS: dict[str, tuple[str, ...]] = {
    "delete": ("delete", "remove", "rm", "wipe", "purge", "drop", "trash"),
    "write": ("write", "save", "create", "make", "build", "add", "put"),
    "modify": ("modify", "edit", "change", "update", "patch", "fix"),
    "move": ("move", "rename", "mv"),
    "copy": ("copy", "duplicate", "cp"),
    "send": ("send", "email", "message", "post", "ping", "tell", "ask"),
    "buy": ("buy", "purchase", "order", "checkout", "pay"),
    "execute": ("run", "execute", "launch", "start", "open", "fire"),
    "install": ("install", "add", "set up", "configure"),
    "shutdown": ("shutdown", "shut down", "restart", "reboot", "power off",
                 "hibernate", "sleep"),
    "kill": ("kill", "stop", "terminate", "end", "quit"),
}


@dataclass(frozen=True)
class IntentMatch:
    """Result of :func:`matches_explicit_intent`.

    Attributes:
        matched: True iff the utterance contains explicit verb+object.
        verb: which verb token matched. Empty if no match.
        object_token: which object token matched. Empty if no match.
        reason: human-readable explanation for logs.
    """

    matched: bool
    verb: str = ""
    object_token: str = ""
    reason: str = ""


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize on whitespace + simple punctuation."""
    return re.findall(r"[a-z0-9_\.\-/\\:]+", (text or "").lower())


def _verbs_for_tool(tool_name: str) -> tuple[str, ...]:
    """Pick the verb-synonym list for the given tool name.

    Falls back to a broad "any action" list when the tool name
    doesn't match a known category.
    """
    name_lower = (tool_name or "").lower()
    for key, syns in _VERB_SYNONYMS.items():
        if key in name_lower:
            return syns
    # Fallback: be permissive about which verbs count when we don't
    # recognise the tool. Better to under-match here than over-match.
    return ("delete", "write", "modify", "send", "run", "install", "shutdown")


def matches_explicit_intent(
    user_text: str,
    *,
    tool_name: str,
    object_hints: tuple[str, ...] = (),
) -> IntentMatch:
    """True iff the user's most recent utterance has verb+object for
    this action.

    Args:
        user_text: The user's most recent utterance text.
        tool_name: The tool the model is trying to call.
        object_hints: Extra object tokens to match against (filenames,
            path leaves, category-specific nouns). The matcher checks
            these in addition to the tool-name tail.

    Returns:
        :class:`IntentMatch` indicating whether an explicit intent
        was detected + the matched tokens for logging.
    """
    if not user_text or not user_text.strip():
        return IntentMatch(matched=False, reason="empty user text")

    tokens = _tokenize(user_text)
    if not tokens:
        return IntentMatch(matched=False, reason="no parseable tokens")

    verbs = _verbs_for_tool(tool_name)
    # Identify verb positions in the tokenised utterance. Multi-word
    # verbs (e.g. "shut down") match as adjacent token pairs.
    verb_positions: list[tuple[int, str]] = []
    multi_word_verbs = [v for v in verbs if " " in v]
    single_word_verbs = [v for v in verbs if " " not in v]
    for i, t in enumerate(tokens):
        for v in single_word_verbs:
            if t == v:
                verb_positions.append((i, v))
                break
        else:
            # Multi-word verbs: check if tokens[i:i+n] match.
            for v in multi_word_verbs:
                parts = v.split()
                if (
                    i + len(parts) <= len(tokens)
                    and tokens[i : i + len(parts)] == parts
                ):
                    verb_positions.append((i, v))
                    break

    if not verb_positions:
        return IntentMatch(
            matched=False,
            reason=f"no verb from {sorted(set(verbs))} found in utterance",
        )

    # Identify object positions. Object candidates: the tool name's
    # last dotted segment, plus any supplied object hints, plus
    # nouns from the broad object table.
    object_candidates: set[str] = set()
    if tool_name:
        # tool_name shape: "openclaw.file.write" -> last segment "write"
        last_seg = tool_name.split(".")[-1].lower()
        # We want the noun-style object, not the verb. For "write" we
        # want the user to also reference the THING being written --
        # so add the second-to-last segment too.
        parts = tool_name.split(".")
        if len(parts) >= 2:
            object_candidates.add(parts[-2].lower())  # "file" in "openclaw.file.write"
        object_candidates.add(last_seg)
    for h in object_hints:
        if isinstance(h, str) and h:
            object_candidates.add(h.lower())

    # Broad noun bucket.
    object_candidates.update(
        ("file", "directory", "folder", "key", "registry", "config",
         "process", "service", "task", "user", "account", "session",
         "tab", "browser", "app", "program", "wallet", "password",
         "address", "email", "message", "log", "audit", "policy",
         "rule", "data",
         # System / device targets used by shutdown-class commands.
         "pc", "computer", "machine", "system", "laptop", "desktop",
         "server",
         # Shopping / commerce nouns
         "copy", "book", "item", "thing", "order",
         # Communication nouns
         "note", "reply")
    )

    object_positions: list[tuple[int, str]] = []
    for i, t in enumerate(tokens):
        # Try exact match and suffix match (for filenames).
        for c in object_candidates:
            if t == c or t.endswith("/" + c) or t.endswith("\\" + c) or t.endswith("." + c):
                object_positions.append((i, c))
                break

    if not object_positions:
        return IntentMatch(
            matched=False,
            verb=verb_positions[0][1],
            reason=f"verb {verb_positions[0][1]!r} found but no matching object hint",
        )

    # Verb + object within WINDOW_TOKENS of each other. The verb
    # position must differ from the object position -- some tool-
    # names use a verb-shaped last segment (e.g. ``file.delete``)
    # which then leaks into the object candidate set; without this
    # check the matcher would pair a verb token with itself.
    for vi, vt in verb_positions:
        for oi, ot in object_positions:
            if vi == oi:
                continue
            if abs(vi - oi) <= WINDOW_TOKENS:
                return IntentMatch(
                    matched=True,
                    verb=vt,
                    object_token=ot,
                    reason=(
                        f"verb {vt!r} at token {vi} near object {ot!r} "
                        f"at token {oi} (window={WINDOW_TOKENS})"
                    ),
                )

    return IntentMatch(
        matched=False,
        verb=verb_positions[0][1],
        object_token=object_positions[0][1],
        reason=(
            f"verb + object found but separated by more than {WINDOW_TOKENS} "
            f"tokens"
        ),
    )
