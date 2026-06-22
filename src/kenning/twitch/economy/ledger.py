"""Append-only, event-sourced points ledger (SQLite WAL, idempotent, fail-safe).

The single authoritative store for the Ultron economy. Design (MASTER.md SLICE 9):

  * **Append-only event log.** Every mutation is an immutable row in ``events``;
    nothing is ever updated or deleted. The current balance is a *projection*
    (``balance_after`` cached per row, plus an authoritative recompute via
    :meth:`Ledger.rebuild_balances`).
  * **Durability.** ``PRAGMA journal_mode=WAL`` + ``synchronous=FULL`` so a
    committed money mutation survives a crash / power loss (MASTER: "synchronous
    = FULL on money commits").
  * **Idempotency.** Each mutation carries an ``idempotency_key`` with a UNIQUE
    constraint. A replay of the same key (the same Twitch redemption / message
    re-delivered with no EventSub replay) returns the SAME balance WITHOUT
    applying the delta twice (exactly-once locally).
  * **No negative balances.** A debit that would drive a balance below zero
    raises :class:`InsufficientFunds` and writes nothing.
  * **Thread-safe.** A single re-entrant lock serializes all writes (the sidecar
    is single-process but multi-threaded: EventSub callback thread, reconcile
    worker, overlay HTTP thread).
  * **Transfers OFF by default** — there is no peer-to-peer transfer primitive
    here; credit/debit are the only mutations, both attributed to a single
    ``user_id``. A transfer feature, if ever built, must be a separate gated path.

ANTICHEAT (BR-P1): stdlib only (``sqlite3``/``hashlib``/``threading``/``time``/
``logging``). No network, no third-party deps.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("kenning.twitch.economy.ledger")

__all__ = ["Ledger", "LedgerEvent", "LedgerError", "InsufficientFunds"]


class LedgerError(Exception):
    """Base class for ledger faults (validation, integrity, storage)."""


class InsufficientFunds(LedgerError):
    """A debit would drive the balance below zero; nothing was written."""

    def __init__(self, user_id: str, balance: int, amount: int) -> None:
        super().__init__(
            f"insufficient funds for {user_id!r}: balance={balance} "
            f"requested debit={amount}"
        )
        self.user_id = user_id
        self.balance = balance
        self.amount = amount


@dataclass(frozen=True)
class LedgerEvent:
    """One immutable row of the append-only event log."""

    id: int
    ts: float
    user_id: str
    delta: int
    reason: str
    idempotency_key: str
    balance_after: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    user_id         TEXT    NOT NULL,
    delta           INTEGER NOT NULL,
    reason          TEXT    NOT NULL,
    idempotency_key TEXT    NOT NULL UNIQUE,
    balance_after   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, id);
"""


def _coerce_amount(amount: int) -> int:
    """Validate a positive integer amount for credit/debit (input validation)."""
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise LedgerError(f"amount must be a non-bool int, got {type(amount).__name__}")
    if amount <= 0:
        raise LedgerError(f"amount must be a positive integer, got {amount}")
    return amount


def _coerce_str(value: str, field: str, *, max_len: int = 256) -> str:
    """Validate a non-empty, length-bounded identifier/reason field."""
    if not isinstance(value, str):
        raise LedgerError(f"{field} must be a str, got {type(value).__name__}")
    v = value.strip()
    if not v:
        raise LedgerError(f"{field} must be a non-empty string")
    if len(v) > max_len:
        raise LedgerError(f"{field} exceeds {max_len} chars ({len(v)})")
    return v


class Ledger:
    """Thread-safe, crash-durable, append-only points ledger.

    :param db_path: filesystem path to the SQLite DB. ``":memory:"`` is allowed
        for tests but is single-connection (no WAL durability across reopen).
    :param busy_timeout_ms: SQLite ``busy_timeout`` so concurrent writers wait
        for the write lock rather than raising ``database is locked``.
    """

    def __init__(self, db_path: str, *, busy_timeout_ms: int = 5000) -> None:
        self._db_path = str(db_path)
        self._busy_timeout_ms = int(busy_timeout_ms)
        # Re-entrant: balance() is called from inside credit/debit under the lock.
        self._lock = threading.RLock()
        self._closed = False

        parent = os.path.dirname(os.path.abspath(self._db_path))
        if self._db_path != ":memory:" and parent and not os.path.isdir(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as e:
                raise LedgerError(f"cannot create ledger dir {parent!r}: {e}") from e

        try:
            # check_same_thread=False because the RLock — not the connection — is
            # the concurrency boundary; every access holds self._lock.
            self._conn = sqlite3.connect(
                self._db_path,
                timeout=self._busy_timeout_ms / 1000.0,
                check_same_thread=False,
                isolation_level=None,  # autocommit; we manage BEGIN/COMMIT explicitly
            )
            self._conn.row_factory = sqlite3.Row
            self._configure()
            self._migrate()
        except sqlite3.Error as e:
            raise LedgerError(f"cannot open ledger at {self._db_path!r}: {e}") from e
        logger.info(
            "ledger opened path=%s journal=WAL synchronous=FULL", self._db_path
        )

    # -- setup ----------------------------------------------------------------
    def _configure(self) -> None:
        cur = self._conn.cursor()
        # WAL: durable, allows a reader concurrent with the single writer.
        # ":memory:" rejects WAL — fall back gracefully (tests only).
        try:
            cur.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.Error as e:  # pragma: no cover - platform/path dependent
            logger.warning("WAL unavailable (%s); continuing with default journal", e)
        # FULL: fsync the WAL on every commit -> a committed money mutation
        # survives power loss. This is the deliberate durability/latency trade.
        cur.execute("PRAGMA synchronous=FULL;")
        cur.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms};")
        cur.execute("PRAGMA foreign_keys=ON;")

    def _migrate(self) -> None:
        self._conn.executescript(_SCHEMA)

    # -- internal helpers (assume self._lock held) ----------------------------
    def _assert_open(self) -> None:
        if self._closed:
            raise LedgerError("ledger is closed")

    def _current_balance(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT balance_after FROM events WHERE user_id=? "
            "ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return int(row["balance_after"]) if row is not None else 0

    def _find_by_key(self, idempotency_key: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM events WHERE idempotency_key=? LIMIT 1",
            (idempotency_key,),
        ).fetchone()

    def _apply(
        self, user_id: str, delta: int, reason: str, idempotency_key: str
    ) -> int:
        """Append one event atomically; return the new balance. Lock held."""
        self._assert_open()

        # Idempotency fast-path: a replayed key returns the stored balance,
        # NEVER re-applying the delta. We compare the stored delta so a key reuse
        # with a *different* delta is caught as a programming error.
        existing = self._find_by_key(idempotency_key)
        if existing is not None:
            if int(existing["delta"]) != int(delta) or existing["user_id"] != user_id:
                raise LedgerError(
                    f"idempotency_key {idempotency_key!r} reused with a different "
                    f"mutation (stored user={existing['user_id']!r} "
                    f"delta={existing['delta']} vs new user={user_id!r} delta={delta})"
                )
            logger.info(
                "ledger replay key=%s user=%s delta=%+d -> balance=%d (no-op)",
                idempotency_key, user_id, delta, int(existing["balance_after"]),
            )
            return int(existing["balance_after"])

        balance = self._current_balance(user_id)
        new_balance = balance + delta
        if new_balance < 0:
            raise InsufficientFunds(user_id, balance, -delta)

        ts = time.time()
        try:
            self._conn.execute("BEGIN IMMEDIATE;")
            # Re-check under the write lock in case a concurrent writer inserted
            # the same key (UNIQUE would also catch it, but this returns cleanly).
            existing2 = self._find_by_key(idempotency_key)
            if existing2 is not None:
                self._conn.execute("ROLLBACK;")
                return int(existing2["balance_after"])
            self._conn.execute(
                "INSERT INTO events "
                "(ts, user_id, delta, reason, idempotency_key, balance_after) "
                "VALUES (?,?,?,?,?,?)",
                (ts, user_id, delta, reason, idempotency_key, new_balance),
            )
            self._conn.execute("COMMIT;")
        except sqlite3.IntegrityError:
            # Lost the race on the UNIQUE key — another thread committed it first.
            self._safe_rollback()
            row = self._find_by_key(idempotency_key)
            if row is None:  # pragma: no cover - integrity error must leave the row
                raise LedgerError(
                    f"integrity error but key {idempotency_key!r} absent"
                ) from None
            return int(row["balance_after"])
        except sqlite3.Error as e:
            self._safe_rollback()
            raise LedgerError(f"ledger write failed: {e}") from e

        logger.info(
            "ledger %s key=%s user=%s delta=%+d -> balance=%d reason=%s",
            "credit" if delta >= 0 else "debit",
            idempotency_key, user_id, delta, new_balance, reason,
        )
        return new_balance

    def _safe_rollback(self) -> None:
        try:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK;")
        except sqlite3.Error as e:  # pragma: no cover - defensive
            logger.warning("ledger rollback failed: %s", e)

    # -- public API -----------------------------------------------------------
    def credit(
        self, user_id: str, amount: int, reason: str, idempotency_key: str
    ) -> int:
        """Add ``amount`` (>0) to ``user_id``. Returns the new balance.

        A replay of ``idempotency_key`` returns the same balance without
        double-applying.
        """
        uid = _coerce_str(user_id, "user_id", max_len=128)
        amt = _coerce_amount(amount)
        rsn = _coerce_str(reason, "reason", max_len=256)
        key = _coerce_str(idempotency_key, "idempotency_key", max_len=256)
        with self._lock:
            return self._apply(uid, amt, rsn, key)

    def debit(
        self, user_id: str, amount: int, reason: str, idempotency_key: str
    ) -> int:
        """Subtract ``amount`` (>0) from ``user_id``. Returns the new balance.

        Raises :class:`InsufficientFunds` (writing nothing) if the balance would
        go below zero. A replay of ``idempotency_key`` returns the same balance
        without double-applying.
        """
        uid = _coerce_str(user_id, "user_id", max_len=128)
        amt = _coerce_amount(amount)
        rsn = _coerce_str(reason, "reason", max_len=256)
        key = _coerce_str(idempotency_key, "idempotency_key", max_len=256)
        with self._lock:
            return self._apply(uid, -amt, rsn, key)

    def balance(self, user_id: str) -> int:
        """Current balance of ``user_id`` (0 if never seen). Reads the cached
        ``balance_after`` of the latest event — O(1) via the index."""
        uid = _coerce_str(user_id, "user_id", max_len=128)
        with self._lock:
            self._assert_open()
            return self._current_balance(uid)

    def history(self, user_id: str, *, limit: int = 100) -> list[LedgerEvent]:
        """The most-recent events for ``user_id`` (newest first), for audit/!verify."""
        uid = _coerce_str(user_id, "user_id", max_len=128)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise LedgerError(f"limit must be a positive int, got {limit!r}")
        with self._lock:
            self._assert_open()
            rows = self._conn.execute(
                "SELECT * FROM events WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (uid, int(limit)),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def rebuild_balances(self) -> dict[str, int]:
        """Recompute every user's balance from the event log in id order and,
        for each user, return the authoritative total. Also asserts that the
        cached ``balance_after`` of each user's latest event matches the recompute
        (detects log/cache divergence). Used by ``!verify`` and crash recovery.
        """
        with self._lock:
            self._assert_open()
            totals: dict[str, int] = {}
            cached_last: dict[str, int] = {}
            for row in self._conn.execute(
                "SELECT user_id, delta, balance_after FROM events ORDER BY id ASC"
            ):
                uid = row["user_id"]
                totals[uid] = totals.get(uid, 0) + int(row["delta"])
                if totals[uid] < 0:
                    # Append-only + the debit guard means this is impossible
                    # unless the file was tampered with.
                    raise LedgerError(
                        f"ledger integrity: recomputed balance for {uid!r} "
                        f"went negative ({totals[uid]}) — log tampered?"
                    )
                cached_last[uid] = int(row["balance_after"])
            for uid, total in totals.items():
                if cached_last.get(uid) != total:
                    raise LedgerError(
                        f"ledger integrity: cached balance_after for {uid!r} "
                        f"({cached_last.get(uid)}) != recomputed ({total})"
                    )
            logger.info("ledger rebuild_balances over %d user(s)", len(totals))
            return totals

    def total_events(self) -> int:
        """Number of rows in the append-only log (for tests/metrics)."""
        with self._lock:
            self._assert_open()
            row = self._conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
            return int(row["n"])

    def checkpoint(self) -> None:
        """Force a WAL checkpoint (flush the WAL into the main DB file). Safe to
        call before a backup; fail-open (logged, never raised)."""
        with self._lock:
            if self._closed:
                return
            try:
                self._conn.execute("PRAGMA wal_checkpoint(FULL);")
            except sqlite3.Error as e:  # pragma: no cover - defensive
                logger.warning("ledger checkpoint failed: %s", e)

    def close(self) -> None:
        """Checkpoint + close the connection. Idempotent."""
        with self._lock:
            if self._closed:
                return
            try:
                self._conn.execute("PRAGMA wal_checkpoint(FULL);")
            except sqlite3.Error as e:  # pragma: no cover - defensive
                logger.warning("ledger checkpoint-on-close failed: %s", e)
            try:
                self._conn.close()
            except sqlite3.Error as e:  # pragma: no cover - defensive
                logger.warning("ledger close failed: %s", e)
            finally:
                self._closed = True
                logger.info("ledger closed path=%s", self._db_path)

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _row_to_event(row: sqlite3.Row) -> LedgerEvent:
    return LedgerEvent(
        id=int(row["id"]),
        ts=float(row["ts"]),
        user_id=str(row["user_id"]),
        delta=int(row["delta"]),
        reason=str(row["reason"]),
        idempotency_key=str(row["idempotency_key"]),
        balance_after=int(row["balance_after"]),
    )
