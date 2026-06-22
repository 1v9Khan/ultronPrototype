"""S10/S12 — closed-grammar, server-authoritative chat-command parser.

A hostile / abliterated chatter must never be able to steer a model with free
text. EVERY economy/game action enters the sidecar through THIS parser, which
maps a :class:`~kenning.twitch.clients.eventsub.ChatEvent` onto a *closed enum*
of :class:`CommandKind` with TYPED, RANGE-CHECKED arguments — never free text to
a model (MASTER.md §5 / SLICE 9: "deterministic closed-grammar parse"). An
unrecognised ``!token`` collapses to :data:`CommandKind.UNKNOWN` (so the caller
can emit one canned "unknown command" reply, not route it anywhere); a line that
is not prefixed at all returns ``None`` (it is ordinary chat, not a command).

Trust model
-----------
* ``is_mod`` is resolved HERE from the EventSub ``badges`` provenance
  (``set_id in {"moderator", "broadcaster"}``) — never from the message body and
  never from anything the chatter can spell. The broadcaster is implicitly a mod.
* Authoring commands that persist stored text (``!addcom`` / ``!addquote``) are a
  slur-injection vector and are Moderator-gated at the dispatch layer; the closed
  grammar here exposes none of them, so a non-mod simply cannot author.
* Numeric args are integers in ``[1, MAX_AMOUNT]``. Negative, zero, non-numeric,
  decimal, and overflow ("huge") amounts are REJECTED (the command parses to its
  kind with ``args["error"]`` set + ``args["amount"]`` absent) so a downstream
  ledger call can never see a poisoned amount. ``all`` is a first-class sentinel
  for the bet games (``!gamble all`` / ``!heist all`` / ``!slots all``).

ANTICHEAT (BR-P1): stdlib only (``re`` / ``logging`` / ``dataclasses`` / ``enum``
/ ``typing``). Importable in the anticheat-pinned voice process.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from kenning.twitch.clients.eventsub import ChatEvent

logger = logging.getLogger("kenning.twitch.commands")

__all__ = [
    "CommandKind",
    "Command",
    "parse_command",
    "MAX_AMOUNT",
    "ALL_SENTINEL",
]

# Upper bound on any wager / transfer amount. A "huge" amount (overflow attempt,
# or simply more points than any economy will hold) is rejected rather than
# silently clamped — the caller decides, but never on a poisoned value. Twitch
# point economies are well under this; the ceiling exists to make integer
# overflow / DoS-by-giant-number structurally impossible.
MAX_AMOUNT = 1_000_000_000  # 1e9

# Sentinel stored in ``args["amount"]`` for an "all-in" bet (``!gamble all``).
ALL_SENTINEL = "all"

# A bare integer amount: digits only, optional surrounding whitespace already
# stripped by tokenisation. We deliberately do NOT accept '+', '-', decimals,
# thousands separators, or scientific notation — the grammar is closed.
_INT_RE = re.compile(r"^\d+$")

# A target user reference for !duel / !give. Twitch logins are 4–25 chars,
# [a-zA-Z0-9_]; we accept an optional leading '@' and lower-case the result
# (logins are case-insensitive). This is a *login*, never a display name.
_TARGET_RE = re.compile(r"^@?([a-zA-Z0-9_]{1,25})$")

# Badge set_ids that confer moderator authority. The broadcaster is implicitly a
# moderator for command-authz purposes (they can do anything a mod can).
_MOD_BADGE_SET_IDS = frozenset({"moderator", "broadcaster"})


class CommandKind(Enum):
    """The closed set of recognised chat commands. UNKNOWN = a ``!token`` that is
    not one of the others (the caller emits a canned reply, routes nothing)."""

    POINTS = "points"
    GAMBLE = "gamble"
    WHEEL = "wheel"
    SLOTS = "slots"
    HEIST = "heist"
    DUEL = "duel"
    TRIVIA = "trivia"
    GIVE = "give"
    LEADERBOARD = "leaderboard"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Command:
    """A parsed, typed command. ``args`` holds only validated values.

    Common ``args`` keys by kind:
      * GAMBLE / HEIST / SLOTS: ``amount`` -> ``int`` or :data:`ALL_SENTINEL`;
        on a bad amount, ``amount`` is ABSENT and ``error`` describes why.
      * DUEL: ``target`` -> login ``str``, ``amount`` -> ``int`` / sentinel.
      * GIVE: ``target`` -> login ``str``, ``amount`` -> ``int`` (no 'all').
      * POINTS / WHEEL / TRIVIA / LEADERBOARD / HELP / UNKNOWN: no required args.
    ``raw`` is the original message text (untrusted; for audit/log only).
    """

    kind: CommandKind
    args: dict = field(default_factory=dict)
    user_id: str = ""
    user_login: str = ""
    is_mod: bool = False
    raw: str = ""


# --------------------------------------------------------------------------- #
# Badge / authz resolution
# --------------------------------------------------------------------------- #
def _resolve_is_mod(badges: object) -> bool:
    """True iff a moderator/broadcaster badge is present in the EventSub badges.

    Fail-SAFE: a malformed badge structure yields ``False`` (deny elevated
    authority) — we never grant mod from anything we cannot positively verify.
    """
    if not isinstance(badges, (list, tuple)):
        return False
    for badge in badges:
        if not isinstance(badge, dict):
            continue
        set_id = badge.get("set_id")
        if isinstance(set_id, str) and set_id in _MOD_BADGE_SET_IDS:
            return True
    return False


# --------------------------------------------------------------------------- #
# Amount / target validation
# --------------------------------------------------------------------------- #
def _parse_amount(token: str, *, allow_all: bool) -> tuple[object | None, str | None]:
    """Validate a wager/transfer amount token.

    Returns ``(value, error)`` where exactly one is non-None:
      * ``(int, None)`` for a valid ``1..MAX_AMOUNT`` amount,
      * ``(ALL_SENTINEL, None)`` for ``all`` when ``allow_all`` is set,
      * ``(None, "<reason>")`` on any rejection (negative/zero/non-numeric/huge/
        decimal/'all'-not-allowed).
    """
    tok = (token or "").strip().lower()
    if not tok:
        return None, "missing amount"
    if tok == ALL_SENTINEL:
        if allow_all:
            return ALL_SENTINEL, None
        return None, "'all' not permitted for this command"
    # Reject anything that is not a bare non-negative integer literal. This
    # catches '-5', '+5', '5.0', '5e3', '0x10', '1_000', '12abc', etc.
    if not _INT_RE.match(tok):
        return None, f"non-integer amount {token!r}"
    try:
        value = int(tok)
    except ValueError:
        # Unreachable given the regex, but never trust a parse to not raise.
        return None, f"unparseable amount {token!r}"
    if value <= 0:
        return None, "amount must be positive"
    if value > MAX_AMOUNT:
        return None, f"amount exceeds maximum {MAX_AMOUNT}"
    return value, None


def _parse_target(token: str) -> str | None:
    """Validate a ``@user`` target -> normalised login, or ``None`` if invalid."""
    m = _TARGET_RE.match((token or "").strip())
    if not m:
        return None
    login = m.group(1).lower()
    # Logins are >=1 char here; Twitch's real floor is 4, but we accept short
    # test logins and let the roster resolution layer reject a non-existent one.
    return login or None


def _tokenize(text: str) -> list[str]:
    """Whitespace-split the message body into tokens (collapses runs of space)."""
    return text.split()


# --------------------------------------------------------------------------- #
# Per-kind argument grammars
# --------------------------------------------------------------------------- #
def _args_bet(rest: list[str], *, allow_all: bool) -> dict:
    """Grammar for a single-amount bet (``!gamble`` / ``!heist`` / ``!slots``)."""
    if not rest:
        return {"error": "missing amount"}
    value, error = _parse_amount(rest[0], allow_all=allow_all)
    if error is not None:
        return {"error": error}
    return {"amount": value}


def _args_duel(rest: list[str]) -> dict:
    """Grammar for ``!duel @user <amount|all>`` -> target + amount."""
    if len(rest) < 2:
        return {"error": "usage: !duel @user <amount>"}
    target = _parse_target(rest[0])
    if target is None:
        return {"error": f"invalid target {rest[0]!r}"}
    value, error = _parse_amount(rest[1], allow_all=True)
    if error is not None:
        return {"target": target, "error": error}
    return {"target": target, "amount": value}


def _args_give(rest: list[str]) -> dict:
    """Grammar for ``!give @user <amount>`` -> target + amount (no 'all')."""
    if len(rest) < 2:
        return {"error": "usage: !give @user <amount>"}
    target = _parse_target(rest[0])
    if target is None:
        return {"error": f"invalid target {rest[0]!r}"}
    value, error = _parse_amount(rest[1], allow_all=False)
    if error is not None:
        return {"target": target, "error": error}
    return {"target": target, "amount": value}


# command-word -> (CommandKind, arg-builder). The arg-builder takes the tokens
# AFTER the command word and returns a validated ``args`` dict. A None builder
# means the command takes no arguments.
_NO_ARGS = None

_COMMAND_TABLE = {
    "points": (CommandKind.POINTS, _NO_ARGS),
    "balance": (CommandKind.POINTS, _NO_ARGS),   # common alias
    "gamble": (CommandKind.GAMBLE, lambda r: _args_bet(r, allow_all=True)),
    "wheel": (CommandKind.WHEEL, _NO_ARGS),
    "slots": (CommandKind.SLOTS, lambda r: _args_bet(r, allow_all=True)),
    "heist": (CommandKind.HEIST, lambda r: _args_bet(r, allow_all=True)),
    "duel": (CommandKind.DUEL, _args_duel),
    "trivia": (CommandKind.TRIVIA, _NO_ARGS),
    "give": (CommandKind.GIVE, _args_give),
    "leaderboard": (CommandKind.LEADERBOARD, _NO_ARGS),
    "top": (CommandKind.LEADERBOARD, _NO_ARGS),   # common alias
    "help": (CommandKind.HELP, _NO_ARGS),
    "commands": (CommandKind.HELP, _NO_ARGS),     # common alias
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_command(event: ChatEvent, *, prefix: str = "!") -> Command | None:
    """Parse a :class:`ChatEvent` into a typed :class:`Command`.

    Returns:
      * ``None`` if the message does not start with ``prefix`` (ordinary chat),
        or the event/text is structurally unusable.
      * a :class:`Command` with ``kind == CommandKind.UNKNOWN`` for an
        unrecognised ``!token``.
      * a :class:`Command` with the matched kind + validated ``args`` otherwise.

    Fail-SAFE: any unexpected error yields ``None`` (treat as non-command) and is
    logged — the parser never raises into the sidecar's receive loop.
    """
    try:
        if event is None:
            return None
        text = getattr(event, "text", None)
        if not isinstance(text, str):
            return None

        # A command must START with the prefix (after leading whitespace). Twitch
        # collapses to a single space, but be defensive about leading spaces.
        stripped = text.lstrip()
        if not prefix or not stripped.startswith(prefix):
            return None

        body = stripped[len(prefix):]
        tokens = _tokenize(body)
        if not tokens:
            # Just the bare prefix ("!") with nothing after it -> not a command.
            return None

        word = tokens[0].lower()
        rest = tokens[1:]

        user_id = getattr(event, "chatter_user_id", "") or ""
        user_login = getattr(event, "chatter_login", "") or ""
        is_mod = _resolve_is_mod(getattr(event, "badges", None))

        entry = _COMMAND_TABLE.get(word)
        if entry is None:
            logger.info(
                "twitch unknown command %r from user_id=%s", word, user_id or "?"
            )
            return Command(
                kind=CommandKind.UNKNOWN,
                args={"command": word},
                user_id=user_id,
                user_login=user_login,
                is_mod=is_mod,
                raw=text,
            )

        kind, builder = entry
        args = {} if builder is None else builder(rest)

        if "error" in args:
            logger.info(
                "twitch command %s rejected arg from user_id=%s: %s",
                kind.value, user_id or "?", args["error"],
            )
        else:
            logger.debug(
                "twitch command %s parsed from user_id=%s args=%s mod=%s",
                kind.value, user_id or "?", args, is_mod,
            )

        return Command(
            kind=kind,
            args=args,
            user_id=user_id,
            user_login=user_login,
            is_mod=is_mod,
            raw=text,
        )
    except Exception as exc:  # noqa: BLE001 — never raise into the receive loop
        logger.warning("twitch command parse failed; treating as non-command: %s", exc)
        return None
