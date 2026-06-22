"""Tests for the append-only, idempotent, crash-durable points ledger."""
from __future__ import annotations

import threading

import pytest

from kenning.twitch.economy.ledger import (
    InsufficientFunds,
    Ledger,
    LedgerError,
)


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "economy.db")


@pytest.fixture()
def ledger(db_path):
    led = Ledger(db_path)
    try:
        yield led
    finally:
        led.close()


# --- basic credit / debit -----------------------------------------------------
def test_credit_then_balance(ledger):
    assert ledger.balance("alice") == 0
    bal = ledger.credit("alice", 100, "stream watch", "k-credit-1")
    assert bal == 100
    assert ledger.balance("alice") == 100


def test_debit_reduces_balance(ledger):
    ledger.credit("bob", 50, "bonus", "k1")
    bal = ledger.debit("bob", 20, "spin cost", "k2")
    assert bal == 30
    assert ledger.balance("bob") == 30


def test_balance_per_user_isolated(ledger):
    ledger.credit("alice", 100, "x", "a1")
    ledger.credit("bob", 7, "x", "b1")
    assert ledger.balance("alice") == 100
    assert ledger.balance("bob") == 7
    assert ledger.balance("never_seen") == 0


# --- idempotency: replay does NOT double-apply --------------------------------
def test_credit_idempotent_replay_same_balance(ledger):
    b1 = ledger.credit("alice", 100, "redeem", "redemption-XYZ")
    # Same redemption re-delivered (no EventSub replay) -> must NOT add again.
    b2 = ledger.credit("alice", 100, "redeem", "redemption-XYZ")
    b3 = ledger.credit("alice", 100, "redeem", "redemption-XYZ")
    assert b1 == b2 == b3 == 100
    assert ledger.balance("alice") == 100
    assert ledger.total_events() == 1


def test_debit_idempotent_replay_same_balance(ledger):
    ledger.credit("alice", 100, "seed", "seed-1")
    d1 = ledger.debit("alice", 30, "spin", "spin-key-1")
    d2 = ledger.debit("alice", 30, "spin", "spin-key-1")
    assert d1 == d2 == 70
    assert ledger.balance("alice") == 70
    # 1 credit + 1 debit row; the replay added nothing.
    assert ledger.total_events() == 2


def test_idempotency_key_reuse_with_different_mutation_raises(ledger):
    ledger.credit("alice", 100, "seed", "shared-key")
    # Same key, different delta -> caught as a programming error.
    with pytest.raises(LedgerError):
        ledger.credit("alice", 5, "seed", "shared-key")
    with pytest.raises(LedgerError):
        ledger.debit("alice", 100, "seed", "shared-key")
    # The original event is untouched.
    assert ledger.balance("alice") == 100
    assert ledger.total_events() == 1


# --- insufficient funds: never below zero -------------------------------------
def test_debit_below_zero_raises_and_writes_nothing(ledger):
    ledger.credit("alice", 40, "seed", "s1")
    with pytest.raises(InsufficientFunds) as ei:
        ledger.debit("alice", 41, "overspend", "bad-debit")
    assert ei.value.balance == 40
    assert ei.value.amount == 41
    # Nothing written: balance intact, the failing debit left no row.
    assert ledger.balance("alice") == 40
    assert ledger.total_events() == 1


def test_debit_to_exactly_zero_is_allowed(ledger):
    ledger.credit("alice", 40, "seed", "s1")
    bal = ledger.debit("alice", 40, "spend all", "d1")
    assert bal == 0
    assert ledger.balance("alice") == 0


def test_debit_on_zero_balance_raises(ledger):
    with pytest.raises(InsufficientFunds):
        ledger.debit("ghost", 1, "nope", "g1")
    assert ledger.balance("ghost") == 0


# --- input validation ---------------------------------------------------------
@pytest.mark.parametrize("amount", [0, -5, 1.5, True, "10"])
def test_credit_rejects_bad_amount(ledger, amount):
    with pytest.raises(LedgerError):
        ledger.credit("alice", amount, "x", f"k-{amount!r}")


def test_rejects_empty_user_and_key(ledger):
    with pytest.raises(LedgerError):
        ledger.credit("", 10, "x", "k1")
    with pytest.raises(LedgerError):
        ledger.credit("alice", 10, "x", "")
    with pytest.raises(LedgerError):
        ledger.credit("alice", 10, "", "k1")


# --- concurrency: many threads crediting -> exact final balance ---------------
def test_concurrent_credits_exact_final_balance(db_path):
    led = Ledger(db_path)
    try:
        threads = []
        n_threads = 16
        per_thread = 50  # 16*50 = 800 distinct credits of 1 each

        def worker(tid: int):
            for i in range(per_thread):
                led.credit("alice", 1, "concurrent", f"t{tid}-i{i}")

        for tid in range(n_threads):
            threads.append(threading.Thread(target=worker, args=(tid,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_threads * per_thread
        assert led.balance("alice") == expected
        assert led.total_events() == expected
        # The projection matches a full recompute.
        assert led.rebuild_balances()["alice"] == expected
    finally:
        led.close()


def test_concurrent_replays_of_same_key_apply_once(db_path):
    """Many threads racing the SAME idempotency key -> applied exactly once."""
    led = Ledger(db_path)
    try:
        threads = []
        results: list[int] = []
        results_lock = threading.Lock()

        def worker():
            bal = led.credit("alice", 100, "race", "ONE-KEY")
            with results_lock:
                results.append(bal)

        for _ in range(24):
            threads.append(threading.Thread(target=worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(b == 100 for b in results)
        assert led.balance("alice") == 100
        assert led.total_events() == 1
    finally:
        led.close()


# --- crash recovery: close + reopen -> balances intact, rebuild matches --------
def test_crash_recovery_balances_survive_reopen(db_path):
    led = Ledger(db_path)
    led.credit("alice", 100, "seed", "a1")
    led.debit("alice", 30, "spin", "a2")
    led.credit("bob", 250, "raid", "b1")
    # Simulate a crash: close WITHOUT any extra cleanup, then reopen the file.
    led.close()

    reopened = Ledger(db_path)
    try:
        assert reopened.balance("alice") == 70
        assert reopened.balance("bob") == 250
        # The append-only log + the cached projection agree after recovery.
        totals = reopened.rebuild_balances()
        assert totals == {"alice": 70, "bob": 250}
        # New mutations continue from the recovered state.
        assert reopened.credit("alice", 5, "post-recover", "a3") == 75
    finally:
        reopened.close()


def test_rebuild_matches_running_balance_across_many_ops(db_path):
    led = Ledger(db_path)
    try:
        running = {"alice": 0, "bob": 0}
        for i in range(200):
            led.credit("alice", 3, "c", f"a-c-{i}")
            running["alice"] += 3
            if running["alice"] >= 10:
                led.debit("alice", 2, "d", f"a-d-{i}")
                running["alice"] -= 2
            led.credit("bob", 1, "c", f"b-c-{i}")
            running["bob"] += 1
        totals = led.rebuild_balances()
        assert totals == running
        assert led.balance("alice") == running["alice"]
        assert led.balance("bob") == running["bob"]
    finally:
        led.close()


# --- history + context manager ------------------------------------------------
def test_history_newest_first(ledger):
    ledger.credit("alice", 10, "first", "h1")
    ledger.credit("alice", 20, "second", "h2")
    ledger.debit("alice", 5, "third", "h3")
    hist = ledger.history("alice", limit=10)
    assert [e.reason for e in hist] == ["third", "second", "first"]
    assert hist[0].balance_after == 25
    assert hist[0].delta == -5


def test_context_manager_closes(db_path):
    with Ledger(db_path) as led:
        led.credit("alice", 5, "x", "k1")
        assert led.balance("alice") == 5
    # After the context exits the ledger is closed.
    with pytest.raises(LedgerError):
        led.balance("alice")


def test_operations_after_close_raise(db_path):
    led = Ledger(db_path)
    led.credit("alice", 5, "x", "k1")
    led.close()
    with pytest.raises(LedgerError):
        led.credit("alice", 5, "x", "k2")
    with pytest.raises(LedgerError):
        led.balance("alice")


def test_close_is_idempotent(db_path):
    led = Ledger(db_path)
    led.close()
    led.close()  # second close must not raise
