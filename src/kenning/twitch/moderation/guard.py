"""S11 — ModerationGuard: the server-authoritative gate in front of Helix writes.

The guard is the deterministic kill-chain between a SPOKEN name and a Helix
``user_id`` moderation action. The abliterated 8B is NEVER consulted here.

Responsibilities (docs/twitch_integration/02_board/{MASTER,S_report}.md, SLICE 11):

  * RESOLVE a spoken name to a roster ``user_id``:
      1. exact login match (case-insensitive) — the unambiguous fast path;
      2. else RapidFuzz similarity + a phonetic (Soundex) key over the LIVE
         roster (logins + display names).
    Resolution returns AMBIGUOUS (no auto-pick) when the best score is low OR the
    top-2 margin is small — a homoglyph/near-collision must reach the human, never
    auto-target. ``user_id`` is ``None`` on ambiguous/no-match.

  * AUTHORIZE an action against a resolved ``target_id``:
      - REFUSE if the target is the bot itself, a moderator, or the broadcaster
        (the ``protected_ids`` set) — a hard role/self guard;
      - REFUSE if a mass-action circuit breaker has tripped (<= N applied actions
        per 60s, measured on a MONOTONIC clock window) — "ban everyone" is broken
        structurally at the action layer.

  * AUDIT every attempted AND applied action to an append-only ledger. Reuses
    :class:`kenning.safety.audit.AuditLog` (SHA-256 hash chain, fsync-per-write,
    ``sanitize_for_log`` against CWE-117 from hostile chat) when importable; falls
    back to a minimal JSONL writer otherwise. Default path
    ``logs/twitch_actions.jsonl``.

ANTICHEAT (BR-P1): stdlib + rapidfuzz only. No 8B, no network, no desktop/input
libs. The guard performs NO Helix call itself — it returns an authorization
verdict the caller hands to :class:`~kenning.twitch.moderation.helix.HelixClient`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

try:  # rapidfuzz is in the voice-path import envelope
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        return float(_fuzz.ratio(a, b))
except Exception:  # noqa: BLE001 - graceful degrade if rapidfuzz somehow absent
    import difflib

    def _ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


logger = logging.getLogger("kenning.twitch.moderation.guard")

__all__ = [
    "AuditWriter",
    "AuthorizeResult",
    "JsonlAuditWriter",
    "ModerationGuard",
    "ResolveResult",
    "RosterEntry",
]


# --- resolution tuning (deterministic, env-overridable) -----------------------
# Best fuzzy score below this => too weak to act on (ambiguous / no-match).
_MIN_SCORE = 80.0
# If the top-2 candidate scores are within this margin => ambiguous (no auto-pick).
_MIN_MARGIN = 8.0


def _norm(s: str) -> str:
    """Lowercase + strip whitespace/punctuation for matching. Pure stdlib."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _soundex(s: str) -> str:
    """Compact Soundex phonetic key (deterministic, dep-free).

    Mirrors the blocklist's helper so the moderation path stays anticheat-clean
    without depending on the optional ``jellyfish`` Metaphone backend.
    """
    s = re.sub(r"[^a-z]", "", (s or "").lower())
    if not s:
        return ""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        **dict.fromkeys("l", "4"),
        **dict.fromkeys("mn", "5"),
        **dict.fromkeys("r", "6"),
    }
    first = s[0]
    out = first.upper()
    prev = codes.get(first, "")
    for ch in s[1:]:
        c = codes.get(ch, "")
        if c and c != prev:
            out += c
        if ch not in "hw":
            prev = c
    return (out + "000")[:4]


@dataclass(frozen=True)
class RosterEntry:
    """One live-roster chatter eligible to be moderated.

    Only users who actually sent a message recently belong in the roster (the
    board's rule: "Only moderate user_ids that actually sent a message recently").
    """

    user_id: str
    login: str
    display_name: str = ""

    def names(self) -> tuple[str, ...]:
        out = [self.login]
        if self.display_name and self.display_name.lower() != self.login.lower():
            out.append(self.display_name)
        return tuple(out)


@dataclass(frozen=True)
class _Candidate:
    user_id: str
    login: str
    display_name: str
    score: float


@dataclass(frozen=True)
class ResolveResult:
    """The outcome of resolving a spoken name to a roster user.

    Attributes:
        user_id: the resolved id, or ``None`` when ambiguous / no match.
        candidates: the top scored candidates (login/display/score), best first —
            shown on the confirmation card so the human picks on ambiguity.
        ambiguous: True when no single user could be auto-selected (low best score
            OR small top-2 margin). When True, ``user_id`` is always ``None``.
        reason: short machine reason ("exact_login" / "fuzzy_unique" /
            "ambiguous_margin" / "ambiguous_low_score" / "no_match" / "empty").
    """

    user_id: Optional[str]
    candidates: tuple[dict[str, Any], ...] = ()
    ambiguous: bool = False
    reason: str = ""


@dataclass(frozen=True)
class AuthorizeResult:
    """The outcome of authorizing an action against a resolved target.

    ``allowed`` False with a ``reason`` of "protected_target" or
    "mass_action_breaker" means the guard REFUSED; the caller must not call Helix.
    """

    allowed: bool
    reason: str
    target_id: str = ""
    action: str = ""


# --- audit sink (reuse the hash-chained AuditLog if importable) ---------------
class AuditWriter(Protocol):
    """The minimal audit sink the guard needs."""

    def record(self, **fields: Any) -> None:  # pragma: no cover - protocol
        ...


class JsonlAuditWriter:
    """A minimal append-only JSONL audit writer (fallback when
    :class:`kenning.safety.audit.AuditLog` is not importable).

    Append-only, fsync-per-write, thread-safe, parent-dir auto-created. Control
    chars are stripped from string values (CWE-117: chat is hostile input). A
    write failure is logged and never raised — the verdict is already made.
    """

    # Strip C0 control chars + DEL. Includes CR (\x0d) and LF (\x0a): both can
    # forge a new JSONL line (CWE-117) from hostile chat-derived strings. (TAB
    # \x09 is kept.)
    _CTRL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("audit dir create failed (%s); best-effort to %s", e, self._path)

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        try:
            if isinstance(value, str):
                return cls._CTRL.sub("", value)
            if isinstance(value, dict):
                return {str(k): cls._sanitize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [cls._sanitize(v) for v in value]
            return value
        except Exception:  # noqa: BLE001 - sanitize is best-effort per leaf
            return value

    def record(self, **fields: Any) -> None:
        entry = {"ts": datetime.utcnow().isoformat(), **{k: self._sanitize(v) for k, v in fields.items()}}
        line = json.dumps(entry, ensure_ascii=False, default=str, sort_keys=True)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as e:
                logger.warning("twitch action audit write failed (%s); entry lost: %s", e, line[:200])


def _build_default_audit(audit_path: str | Path) -> AuditWriter:
    """Prefer the hash-chained :class:`AuditLog`; fall back to JSONL.

    The reused ``AuditLog.record`` has a fixed keyword signature; we adapt the
    guard's richer fields onto it (extra fields ride in ``context``).
    """
    try:
        from kenning.safety.audit import AuditLog  # noqa: PLC0415 - optional import by design

        real = AuditLog(audit_path)

        class _AuditLogAdapter:
            def record(self, **fields: Any) -> None:
                action = str(fields.get("action", ""))
                target_id = str(fields.get("target_id", ""))
                context = {
                    k: v
                    for k, v in fields.items()
                    if k not in {"action", "verdict", "reason", "target_id"}
                }
                context["target_id"] = target_id
                try:
                    real.record(
                        rule_id="TWITCH_MOD",
                        verdict=str(fields.get("verdict", "")),
                        tool_name=action or "moderation",
                        capability="twitch_moderation",
                        reason=str(fields.get("reason", "")),
                        context=context,
                    )
                except Exception as e:  # noqa: BLE001 - audit must never break the verdict
                    logger.warning("AuditLog.record failed (%s); action proceeds", e)

        return _AuditLogAdapter()
    except Exception as e:  # noqa: BLE001 - any import/init issue => JSONL fallback
        logger.info("AuditLog unavailable (%s); using JSONL audit at %s", e, audit_path)
        return JsonlAuditWriter(audit_path)


# --- the guard ----------------------------------------------------------------
class ModerationGuard:
    """Server-authoritative moderation gate (resolve -> authorize -> audit).

    Args:
        roster_provider: a zero-arg callable returning the LIVE roster as an
            iterable of :class:`RosterEntry` (or mappings with user_id/login/
            display_name). Called fresh on each resolve so a just-departed chatter
            is not targetable. A failing/empty provider fails CLOSED (no match).
        protected_ids: user_ids that may NEVER be a target — the bot's own id, the
            broadcaster, and every moderator. Resolution can still surface them as
            candidates (so the human sees who was meant) but :meth:`authorize`
            refuses to act.
        audit_path: where the action ledger is written (default
            ``logs/twitch_actions.jsonl``).
        audit: an explicit :class:`AuditWriter` (tests inject a fake). When None a
            default is built (hash-chained AuditLog if importable, else JSONL).
        breaker_limit / breaker_window_s: mass-action circuit breaker — at most
            ``breaker_limit`` APPLIED actions per ``breaker_window_s`` seconds
            (monotonic). Default 3 / 60s.
        min_score / min_margin: resolution thresholds (env-overridable via
            ``KENNING_TWITCH_MOD_MIN_SCORE`` / ``_MIN_MARGIN``).
        monotonic: injectable clock for the breaker window (tests drive it).
    """

    def __init__(
        self,
        roster_provider: Callable[[], Iterable[Any]],
        protected_ids: Optional[Iterable[str]] = None,
        audit_path: str | Path = "logs/twitch_actions.jsonl",
        *,
        audit: Optional[AuditWriter] = None,
        breaker_limit: int = 3,
        breaker_window_s: float = 60.0,
        min_score: Optional[float] = None,
        min_margin: Optional[float] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(roster_provider):
            raise ValueError("roster_provider must be callable")
        self._roster_provider = roster_provider
        self._protected: set[str] = {str(x) for x in (protected_ids or ()) if str(x)}
        self._audit: AuditWriter = audit if audit is not None else _build_default_audit(audit_path)
        self._breaker_limit = max(1, int(breaker_limit))
        self._breaker_window = max(1.0, float(breaker_window_s))
        self._min_score = float(min_score) if min_score is not None else _env_float(
            "KENNING_TWITCH_MOD_MIN_SCORE", _MIN_SCORE
        )
        self._min_margin = float(min_margin) if min_margin is not None else _env_float(
            "KENNING_TWITCH_MOD_MIN_MARGIN", _MIN_MARGIN
        )
        self._monotonic = monotonic
        self._action_times: list[float] = []  # monotonic timestamps of APPLIED actions
        self._lock = threading.Lock()

    # ---- roster ----
    def _load_roster(self) -> list[RosterEntry]:
        """Fetch the live roster; fail CLOSED (empty) on any provider error."""
        try:
            raw = self._roster_provider()
        except Exception as e:  # noqa: BLE001 - provider failure => no targets
            logger.warning("roster_provider raised (%s); resolving against empty roster", e)
            return []
        out: list[RosterEntry] = []
        for item in raw or ():
            try:
                if isinstance(item, RosterEntry):
                    entry = item
                elif isinstance(item, Mapping):
                    uid = str(item.get("user_id") or item.get("id") or "")
                    login = str(item.get("login") or item.get("user_login") or "")
                    disp = str(item.get("display_name") or item.get("user_name") or "")
                    if not uid or not login:
                        continue
                    entry = RosterEntry(user_id=uid, login=login, display_name=disp)
                else:
                    continue
            except Exception:  # noqa: BLE001 - skip a malformed roster row
                continue
            if entry.user_id and entry.login:
                out.append(entry)
        return out

    # ---- resolve ----
    def resolve(self, spoken_name: str) -> ResolveResult:
        """Resolve a spoken name to a roster ``user_id``.

        Exact login wins outright. Otherwise score every roster name (login +
        display) by RapidFuzz ratio, boosting an exact phonetic (Soundex) key
        match, and apply the ambiguity rules. Never auto-picks on a weak best
        score or a small top-2 margin.
        """
        spoken = (spoken_name or "").strip()
        if not spoken:
            return ResolveResult(user_id=None, candidates=(), ambiguous=False, reason="empty")
        roster = self._load_roster()
        if not roster:
            return ResolveResult(user_id=None, candidates=(), ambiguous=False, reason="no_match")

        norm_spoken = _norm(spoken)
        spoken_sx = _soundex(spoken)

        # 1) exact login (case-insensitive, punctuation-insensitive) — unambiguous.
        exact = [e for e in roster if _norm(e.login) == norm_spoken]
        if len(exact) == 1:
            e = exact[0]
            return ResolveResult(
                user_id=e.user_id,
                candidates=(_cand_dict(e, 100.0),),
                ambiguous=False,
                reason="exact_login",
            )
        if len(exact) > 1:
            # Two chatters with the same normalized login (homoglyph collision) —
            # never auto-pick.
            cands = tuple(_cand_dict(e, 100.0) for e in exact[:5])
            return ResolveResult(user_id=None, candidates=cands, ambiguous=True, reason="ambiguous_margin")

        # 2) fuzzy + phonetic over every name form.
        best_per_user: dict[str, _Candidate] = {}
        for e in roster:
            best = 0.0
            for name in e.names():
                score = _ratio(norm_spoken, _norm(name))
                if spoken_sx and _soundex(name) == spoken_sx:
                    score = max(score, 90.0)  # phonetic agreement floor
                best = max(best, score)
            prev = best_per_user.get(e.user_id)
            if prev is None or best > prev.score:
                best_per_user[e.user_id] = _Candidate(e.user_id, e.login, e.display_name, best)

        ranked = sorted(best_per_user.values(), key=lambda c: c.score, reverse=True)
        cands = tuple(_cand_dict_c(c) for c in ranked[:5])
        top = ranked[0]
        if top.score < self._min_score:
            return ResolveResult(user_id=None, candidates=cands, ambiguous=True, reason="ambiguous_low_score")
        if len(ranked) >= 2 and (top.score - ranked[1].score) < self._min_margin:
            return ResolveResult(user_id=None, candidates=cands, ambiguous=True, reason="ambiguous_margin")
        return ResolveResult(user_id=top.user_id, candidates=cands, ambiguous=False, reason="fuzzy_unique")

    # ---- authorize ----
    def authorize(self, action: str, target_id: str) -> AuthorizeResult:
        """Authorize an action against a resolved ``target_id``.

        REFUSE when the target is protected (self / moderator / broadcaster) or the
        mass-action circuit breaker has tripped. On allow, the breaker window is
        advanced (this attempt counts toward the rate). Every call is audited.
        """
        action = (action or "moderation").strip()
        target_id = str(target_id or "").strip()
        if not target_id:
            self._audit_action(action, target_id, "REFUSED", "empty_target", applied=False)
            return AuthorizeResult(allowed=False, reason="empty_target", target_id=target_id, action=action)

        if target_id in self._protected:
            self._audit_action(action, target_id, "REFUSED", "protected_target", applied=False)
            return AuthorizeResult(
                allowed=False, reason="protected_target", target_id=target_id, action=action
            )

        with self._lock:
            now = self._monotonic()
            self._prune_locked(now)
            if len(self._action_times) >= self._breaker_limit:
                tripped = True
            else:
                tripped = False
                self._action_times.append(now)
        if tripped:
            self._audit_action(action, target_id, "REFUSED", "mass_action_breaker", applied=False)
            logger.warning(
                "mass-action breaker tripped: >=%d actions / %.0fs; refusing %s on %s",
                self._breaker_limit, self._breaker_window, action, target_id,
            )
            return AuthorizeResult(
                allowed=False, reason="mass_action_breaker", target_id=target_id, action=action
            )

        self._audit_action(action, target_id, "ALLOWED", "authorized", applied=False)
        return AuthorizeResult(allowed=True, reason="authorized", target_id=target_id, action=action)

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self._breaker_window
        self._action_times = [t for t in self._action_times if t > cutoff]

    # ---- audit ----
    def record_applied(
        self,
        action: str,
        target_id: str,
        *,
        idempotent: bool = False,
        status: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Audit that an action was APPLIED at Helix (the caller invokes this after
        a successful :class:`HelixResult`)."""
        fields: dict[str, Any] = {
            "verdict": "APPLIED",
            "reason": "idempotent" if idempotent else "applied",
            "applied": True,
            "idempotent": bool(idempotent),
        }
        if status is not None:
            fields["status"] = int(status)
        if extra:
            fields["extra"] = dict(extra)
        self._audit_action(action, target_id, **fields)

    def _audit_action(
        self,
        action: str,
        target_id: str,
        verdict: str,
        reason: str = "",
        *,
        applied: bool = False,
        **extra: Any,
    ) -> None:
        try:
            self._audit.record(
                action=action,
                target_id=str(target_id),
                verdict=verdict,
                reason=reason,
                applied=bool(applied),
                **extra,
            )
        except Exception as e:  # noqa: BLE001 - audit must never block the decision
            logger.warning("moderation audit record failed (%s); decision stands", e)


# --- module helpers -----------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("env %s=%r is not a float; using default %.2f", name, raw, default)
        return default


def _cand_dict(e: RosterEntry, score: float) -> dict[str, Any]:
    return {
        "user_id": e.user_id,
        "login": e.login,
        "display_name": e.display_name or e.login,
        "score": round(float(score), 2),
    }


def _cand_dict_c(c: _Candidate) -> dict[str, Any]:
    return {
        "user_id": c.user_id,
        "login": c.login,
        "display_name": c.display_name or c.login,
        "score": round(float(c.score), 2),
    }
