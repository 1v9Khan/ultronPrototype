"""Gap-c — tests for the chat-command economy dispatcher (chat_games.py).

Fully offline: a :memory: Ledger, an injected drain (canned ChatEvents), and a
controllable FakeRNG so win/loss is deterministic. Covers the drain + flat-buffer
parse, the bet flow (debit-first, payout credit, leg-distinct idempotency),
insufficient funds, min/max bet, 'all', the per-stream loss cap, the per-user
cooldown, EventSub-replay dedup idempotency, watch-time earning, the
delete-moderation message-id index, and the RTP house-edge math.
"""
from __future__ import annotations

import json
import re
import types

import pytest

from kenning.twitch.economy.chat_games import (
    ChatGameRouter,
    DEFAULT_SLOT_SYMBOLS,
    chat_event_from_buffer,
    make_chat_command_drain_fn,
)
from kenning.twitch.economy.ledger import Ledger
from kenning.twitch.economy.rng import ProvablyFairRNG


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _flat(text, *, uid="u1", login="alice", mid=None, mod=False):
    """The read sidecar's FLAT buffered chat dict."""
    d = {"type": "chat", "message_id": mid or text, "chatter_user_id": uid,
         "chatter_login": login, "chatter_name": login, "text": text}
    if mod:
        d["badges"] = [{"set_id": "moderator", "id": "1"}]
    return d


def _ev(text, *, uid="u1", login="alice", mid=None, mod=False):
    return chat_event_from_buffer(_flat(text, uid=uid, login=login, mid=mid, mod=mod))


class FakeRNG:
    """A ProvablyFairRNG stand-in with controllable outcomes. ``uniform`` drives
    !gamble (a win is uniform < 0.5); ``slots_win`` forces a slots win (all reels
    symbol 0) or a guaranteed loss (each reel a distinct symbol)."""

    def __init__(self, *, uniform=0.99, slots_win=False):
        self._real = ProvablyFairRNG(default_client_seed="test")
        self._uniform = uniform
        self._slots_win = slots_win

    @property
    def default_client_seed(self):
        return self._real.default_client_seed

    def new_round(self):
        return self._real.new_round()

    def commit_for(self, server_seed):
        return self._real.commit_for(server_seed)

    def uniform_unit(self, server_seed, client_seed, nonce):
        return self._uniform

    def outcome(self, server_seed, client_seed, nonce, n):
        if self._slots_win:
            return 0
        m = re.search(r"reel(\d+)", str(client_seed))
        return (int(m.group(1)) if m else 0) % n

    def weighted_choice(self, server_seed, client_seed, nonce, weights):
        # delegate to the real RNG (the !wheel free-spin uses a weighted draw)
        return self._real.weighted_choice(server_seed, client_seed, nonce, weights)


def _cfg(**over):
    # defer_points_gamble_to_streamelements defaults False HERE so the existing
    # !points/!gamble mechanics tests keep exercising Ultron's dispatch path; the
    # dedicated deferral tests pass the flag explicitly. trivia_auto_interval_minutes
    # defaults 0 (disabled) so legacy tests aren't perturbed by auto-trivia.
    base = dict(earn_per_minute=10, gamble_rtp=0.90, per_stream_loss_cap=5000,
                currency_name="cores", command_cooldown_seconds=0, min_bet=1, max_bet=10000,
                defer_points_gamble_to_streamelements=False,
                trivia_auto_interval_minutes=0)
    base.update(over)
    return types.SimpleNamespace(**base)


def _router(events, *, ledger=None, rng=None, cfg=None, replies=None, now=None,
            epoch=None, chat_cfg=None):
    ledger = ledger or Ledger(":memory:")
    replies = replies if replies is not None else []
    kw = {}
    if now is not None:
        kw["now_fn"] = now
    if epoch is not None:
        kw["epoch_fn"] = epoch
    if chat_cfg is not None:
        kw["chat_cfg"] = chat_cfg
    r = ChatGameRouter(lambda: list(events), ledger=ledger, cfg=cfg or _cfg(),
                       rng=rng or FakeRNG(), announce_fn=replies.append, **kw)
    return r, ledger, replies


# --------------------------------------------------------------------------- #
# drain + flat-buffer parse
# --------------------------------------------------------------------------- #
def test_chat_event_from_buffer_maps_flat_fields():
    ev = chat_event_from_buffer(_flat("!points", uid="42", login="Bob", mid="m9"))
    assert ev is not None
    assert ev.text == "!points" and ev.chatter_user_id == "42"
    assert ev.chatter_login == "Bob" and ev.message_id == "m9"
    # a non-chat dict -> None
    assert chat_event_from_buffer({"type": "redeem"}) is None


def test_drain_filters_to_chat_and_advances_cursor():
    payloads = [
        json.dumps({"cursor": 5, "events": [
            {"seq": 1, "ts": 1.0, "event": _flat("!points", mid="a")},
            {"seq": 2, "ts": 1.0, "event": {"type": "redeem", "redemption_id": "r"}},
        ]}).encode(),
        json.dumps({"cursor": 5, "events": []}).encode(),
    ]
    calls = []

    def http_get(url, timeout):
        calls.append(url)
        return payloads[min(len(calls) - 1, len(payloads) - 1)]

    drain = make_chat_command_drain_fn("http://x", http_get=http_get)
    out = drain()
    assert len(out) == 1 and out[0].text == "!points"   # redeem filtered out
    assert "since=0" in calls[0]
    drain()
    assert "since=5" in calls[1]                          # cursor advanced


def test_drain_fail_safe_on_bad_body():
    drain = make_chat_command_drain_fn("http://x", http_get=lambda u, t: b"not json")
    assert drain() == []


# --------------------------------------------------------------------------- #
# read commands
# --------------------------------------------------------------------------- #
def test_points_reports_balance():
    led = Ledger(":memory:")
    led.credit("u1", 250, "seed", "seed")
    r, led, replies = _router([_ev("!points")], ledger=led)
    r.tick()
    assert replies == ["@alice you have 250 cores."]


def test_help_and_unknown():
    r, _led, replies = _router([_ev("!help", mid="h"), _ev("!frobnicate", mid="u")])
    r.tick()
    assert any("Commands:" in x for x in replies)
    assert any("unknown command" in x for x in replies)


def test_ultron_posts_commands_panel_with_doc_link():
    # !ultron posts the SAME condensed commands-panel text viewers see on the
    # periodic auto-post, including the guide link from the chat cfg's
    # commands_panel_doc_url (threaded in via chat_cfg).
    chat_cfg = types.SimpleNamespace(commands_panel_doc_url="https://example.test/guide")
    r, _led, replies = _router([_ev("!ultron", mid="ul")], chat_cfg=chat_cfg)
    r.tick()
    assert len(replies) == 1
    assert "Ultron games" in replies[0]
    assert "!slots" in replies[0] and "!help" in replies[0]
    assert "https://example.test/guide" in replies[0]   # doc link appended


def test_ultron_without_chat_cfg_still_posts_panel():
    # No chat cfg threaded (None) -> the panel still posts, just without a link.
    r, _led, replies = _router([_ev("!ultron", mid="ul")])
    r.tick()
    assert len(replies) == 1
    assert "Ultron games" in replies[0]
    assert "Full guide" not in replies[0]


def test_ultron_cooldown_throttles_spam():
    # A second !ultron from the SAME user inside the cooldown window is dropped
    # (mirrors the bet-game per-user throttle); a different user is not throttled.
    chat_cfg = types.SimpleNamespace(commands_panel_doc_url="")
    cfg = _cfg(command_cooldown_seconds=5)
    clock = {"t": 0.0}
    r, _led, replies = _router(
        [_ev("!ultron", uid="u1", mid="a"),
         _ev("!ultron", uid="u1", mid="b"),
         _ev("!ultron", uid="u2", login="bob", mid="c")],
        cfg=cfg, chat_cfg=chat_cfg, now=lambda: clock["t"])
    r.tick()
    # u1's first + u2's post fire; u1's second is throttled within the window.
    assert len(replies) == 2


def test_leaderboard_lists_top_balances():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "s1")
    led.credit("u2", 500, "s", "s2")
    # The caller (default uid u1) chats as login "z", so the leaderboard renders
    # u1 by its known display login "z"; u2 (never seen this session) shows its raw
    # uid. Top balance first. The board posts as ONE chat message (Twitch collapses
    # a message to a single line), ranks inline separated by " · ".
    r, led, replies = _router([_ev("!leaderboard", login="z")], ledger=led)
    r.tick()
    assert len(replies) == 1                              # one inline message
    assert replies[0] == "Top cores: 1. u2 (500) · 2. z (100)"


def test_leaderboard_caps_at_top_five():
    led = Ledger(":memory:")
    for i in range(8):
        led.credit(f"u{i}", (i + 1) * 10, "s", f"s{i}")
    r, led, replies = _router([_ev("!leaderboard")], ledger=led)
    r.tick()
    # ONE message, at most the top 5 ranks inline (Twitch single-line), never more.
    assert len(replies) == 1
    msg = replies[0]
    assert msg.startswith("Top cores: 1. u7 (80)")   # highest balance ranked first
    assert "5. u3 (40)" in msg                        # 5th rank present
    assert "6. " not in msg                           # capped at five


# --------------------------------------------------------------------------- #
# bet flow — gamble + slots
# --------------------------------------------------------------------------- #
def test_gamble_win_credits_multiplier_and_keys_legs():
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    r, led, replies = _router([_ev("!gamble 100", mid="m2")], ledger=led, rng=FakeRNG(uniform=0.1))
    r.tick()
    # win pays floor(100 * 0.9 / 0.5) = 180 gross; net +80
    assert led.balance("u1") == 1000 - 100 + 180
    keys = {e.idempotency_key for e in led.history("u1", limit=10)}
    assert "gamble:m2:bet" in keys and "gamble:m2:win" in keys
    assert any("WON 180" in x for x in replies)


def test_gamble_loss_debits_only():
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    r, led, replies = _router([_ev("!gamble 100", mid="m2")], ledger=led, rng=FakeRNG(uniform=0.9))
    r.tick()
    assert led.balance("u1") == 900
    keys = {e.idempotency_key for e in led.history("u1", limit=10)}
    assert "gamble:m2:bet" in keys and "gamble:m2:win" not in keys
    assert any("lost 100" in x for x in replies)


def test_slots_win_and_loss():
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    r, led, _ = _router([_ev("!slots 10", mid="w")], ledger=led, rng=FakeRNG(slots_win=True))
    r.tick()
    s = len(DEFAULT_SLOT_SYMBOLS)
    mult = int(0.90 * s * s)
    assert led.balance("u1") == 1000 - 10 + 10 * mult
    # loss
    led2 = Ledger(":memory:")
    led2.credit("u1", 1000, "seed", "seed")
    r2, led2, _ = _router([_ev("!slots 10", mid="l")], ledger=led2, rng=FakeRNG(slots_win=False))
    r2.tick()
    assert led2.balance("u1") == 990


def test_insufficient_funds_refuses_no_debit():
    led = Ledger(":memory:")
    led.credit("u1", 30, "seed", "seed")
    r, led, replies = _router([_ev("!gamble 100", mid="m")], ledger=led)
    r.tick()
    assert led.balance("u1") == 30  # untouched
    assert any("you only have 30" in x for x in replies)


def test_min_and_max_bet_enforced():
    led = Ledger(":memory:")
    led.credit("u1", 100000, "seed", "seed")
    r, led, replies = _router(
        [_ev("!gamble 0", mid="a"), _ev("!gamble 50000", mid="b")],
        ledger=led, cfg=_cfg(min_bet=5, max_bet=10000))
    r.tick()
    # '!gamble 0' is rejected by the PARSER (amount must be positive) -> error reply
    assert any("positive" in x or "minimum" in x for x in replies)
    assert any("maximum bet is 10000" in x for x in replies)
    assert led.balance("u1") == 100000


def test_all_in_bets_whole_balance():
    led = Ledger(":memory:")
    led.credit("u1", 77, "seed", "seed")
    r, led, _ = _router([_ev("!gamble all", mid="m")], ledger=led, rng=FakeRNG(uniform=0.9))
    r.tick()
    assert led.balance("u1") == 0  # lost all 77


def test_per_stream_loss_cap_refuses_past_ceiling():
    led = Ledger(":memory:")
    led.credit("u1", 100000, "seed", "seed")
    # cap=150: first 100 loss ok (net loss 100), second 100 would push to 200 > 150 -> refused
    r, led, replies = _router(
        [_ev("!gamble 100", mid="a"), _ev("!gamble 100", mid="b")],
        ledger=led, cfg=_cfg(per_stream_loss_cap=150, command_cooldown_seconds=0),
        rng=FakeRNG(uniform=0.9))
    r.tick()
    assert led.balance("u1") == 100000 - 100   # only the first bet went through
    assert any("loss cap" in x for x in replies)


def test_cooldown_throttles_second_bet():
    led = Ledger(":memory:")
    led.credit("u1", 10000, "seed", "seed")
    clock = {"t": 0.0}
    r, led, replies = _router(
        [_ev("!gamble 100", mid="a"), _ev("!gamble 100", mid="b")],
        ledger=led, cfg=_cfg(command_cooldown_seconds=5), rng=FakeRNG(uniform=0.9),
        now=lambda: clock["t"])
    r.tick()  # both in the same tick at t=0 -> the second is within cooldown -> silently dropped
    # exactly one bet event
    bets = [e for e in led.history("u1", limit=20) if e.reason == "gamble bet"]
    assert len(bets) == 1


# --------------------------------------------------------------------------- #
# replay dedup + earn + message-id
# --------------------------------------------------------------------------- #
def test_replay_is_idempotent():
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    events = [_ev("!gamble 100", mid="m2")]
    r, led, _ = _router(events, ledger=led, rng=FakeRNG(uniform=0.1))
    r.tick()
    bal = led.balance("u1")
    r.tick()  # same message_id -> dedup -> no second apply
    assert led.balance("u1") == bal


def test_watch_time_earn_credits_active_viewers_once_per_minute():
    led = Ledger(":memory:")
    clock = {"mono": 0.0, "epoch": 0.0}
    r, led, _ = _router([_ev("!points", uid="u1", login="alice", mid="m")], ledger=led,
                        cfg=_cfg(earn_per_minute=10),
                        now=lambda: clock["mono"], epoch=lambda: clock["epoch"])
    r.tick()                       # arms the earn clock (minute 0), no payout yet
    assert led.balance("u1") == 0
    clock["epoch"] = 60.0          # advance one minute; u1 still active (mono unchanged)
    r.tick()
    assert led.balance("u1") == 10
    r.tick()                       # same minute -> idempotent, no double credit
    assert led.balance("u1") == 10


def test_last_message_id_index_for_delete():
    r, _led, _ = _router([_ev("hello", login="Bob", mid="m1"),
                          _ev("world", login="Bob", mid="m2")])
    r.tick()
    assert r.last_message_id("bob") == "m2"   # latest, case-insensitive
    assert r.last_message_id("nobody") is None


# --------------------------------------------------------------------------- #
# economy invariants
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# !give — viewer -> viewer transfer (gated transfers_enabled)
# --------------------------------------------------------------------------- #
def test_give_disabled_by_default():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "s1")
    evs = [_ev("hi", uid="ub", login="bob", mid="b0"),
           _ev("!give @bob 40", uid="u1", login="alice", mid="g1")]
    r, led, replies = _router(evs, ledger=led, cfg=_cfg(transfers_enabled=False, earn_per_minute=0))
    r.tick()
    assert any("transfers are disabled" in x for x in replies)
    assert led.balance("u1") == 100 and led.balance("ub") == 0


def test_give_transfers_when_enabled_and_is_replay_idempotent():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "s1")
    evs = [_ev("hi", uid="ub", login="bob", mid="b0"),
           _ev("!give @bob 40", uid="u1", login="alice", mid="g1")]
    r, led, replies = _router(evs, ledger=led, cfg=_cfg(transfers_enabled=True, earn_per_minute=0))
    r.tick()
    assert led.balance("u1") == 60 and led.balance("ub") == 40
    r.tick()   # EventSub replay of the same message_id -> deduped, no double-move
    assert led.balance("u1") == 60 and led.balance("ub") == 40


def test_give_rejects_unknown_recipient_and_self():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "s1")
    evs = [_ev("!give @ghost 10", uid="u1", login="alice", mid="g1"),
           _ev("!give @alice 10", uid="u1", login="alice", mid="g2")]
    r, led, replies = _router(evs, ledger=led, cfg=_cfg(transfers_enabled=True, earn_per_minute=0))
    r.tick()
    assert any("don't know" in x for x in replies)
    assert any("can't give to yourself" in x for x in replies)
    assert led.balance("u1") == 100


# --------------------------------------------------------------------------- #
# !wheel — free spin (per-stream cap, house-funded payout)
# --------------------------------------------------------------------------- #
def test_wheel_free_spin_credits_and_caps_per_stream():
    led = Ledger(":memory:")
    evs = [_ev("!wheel", uid="u1", login="alice", mid="w1"),
           _ev("!wheel", uid="u1", login="alice", mid="w2")]
    r, led, replies = _router(evs, ledger=led, cfg=_cfg(wheel_free_per_stream=1, earn_per_minute=0))
    r.tick()
    assert led.balance("u1") > 0                          # first spin paid out
    assert any("no free spins left" in x for x in replies)  # second capped


def test_wheel_disabled_when_cap_zero():
    led = Ledger(":memory:")
    r, led, replies = _router([_ev("!wheel", uid="u1", login="alice", mid="w1")],
                              ledger=led, cfg=_cfg(wheel_free_per_stream=0, earn_per_minute=0))
    r.tick()
    assert led.balance("u1") == 0 and any("disabled" in x for x in replies)


# --------------------------------------------------------------------------- #
# !heist — group pooled bet, join window, house bonus
# --------------------------------------------------------------------------- #
def test_heist_win_pays_house_bonus_split():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "a")
    led.credit("u2", 100, "s", "b")
    clock = [0.0]
    evs = [_ev("!heist 100", uid="u1", login="alice", mid="h1"),
           _ev("!heist 100", uid="u2", login="bob", mid="h2")]
    r, led, replies = _router(
        evs, ledger=led,
        cfg=_cfg(heist_window_seconds=30, heist_house_bonus_pct=0.5,
                 heist_min_players=1, earn_per_minute=0, command_cooldown_seconds=0),
        rng=FakeRNG(uniform=0.9),   # draw 0.9 >= win_threshold 0.6 -> WIN
        now=lambda: clock[0])
    r.tick()
    assert led.balance("u1") == 0 and led.balance("u2") == 0   # both staked
    evs.clear()
    clock[0] = 31.0
    r.tick()   # deadline -> resolve. pot 200, +50% bonus = 300, per_head 150
    assert led.balance("u1") == 150 and led.balance("u2") == 150
    assert any("HEIST WIN" in x for x in replies)


def test_heist_fail_loses_stakes():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "a")
    clock = [0.0]
    evs = [_ev("!heist 100", uid="u1", login="alice", mid="h1")]
    r, led, replies = _router(
        evs, ledger=led,
        cfg=_cfg(heist_window_seconds=30, heist_min_players=1, earn_per_minute=0),
        rng=FakeRNG(uniform=0.1),   # draw 0.1 < partial_threshold 0.3 -> FAIL
        now=lambda: clock[0])
    r.tick()
    evs.clear()
    clock[0] = 31.0
    r.tick()
    assert led.balance("u1") == 0 and any("HEIST FAILED" in x for x in replies)


def test_heist_refunds_when_below_min_players():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "a")
    clock = [0.0]
    evs = [_ev("!heist 100", uid="u1", login="alice", mid="h1")]
    r, led, replies = _router(
        evs, ledger=led,
        cfg=_cfg(heist_window_seconds=30, heist_min_players=3, earn_per_minute=0),
        now=lambda: clock[0])
    r.tick()
    evs.clear()
    clock[0] = 31.0
    r.tick()
    assert led.balance("u1") == 100 and any("refunded" in x for x in replies)


# --------------------------------------------------------------------------- #
# !duel + !accept — 1v1 escrow challenge
# --------------------------------------------------------------------------- #
def test_duel_challenge_accept_settles_to_winner():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "a")
    led.credit("u2", 100, "s", "b")
    evs = [_ev("hi", uid="u2", login="bob", mid="b0"),
           _ev("!duel @bob 50", uid="u1", login="alice", mid="d1"),
           _ev("!accept", uid="u2", login="bob", mid="d2")]
    r, led, replies = _router(
        evs, ledger=led, cfg=_cfg(earn_per_minute=0, duel_window_seconds=60),
        rng=FakeRNG(uniform=0.1))   # draw 0.1 < win_bias 0.5 -> challenger (alice) wins
    r.tick()
    assert led.balance("u1") == 150 and led.balance("u2") == 50
    assert any("DUEL" in x and "alice" in x for x in replies)


def test_duel_expires_and_refunds_challenger():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "a")
    clock = [0.0]
    evs = [_ev("hi", uid="u2", login="bob", mid="b0"),
           _ev("!duel @bob 50", uid="u1", login="alice", mid="d1")]
    r, led, replies = _router(
        evs, ledger=led, cfg=_cfg(earn_per_minute=0, duel_window_seconds=60),
        now=lambda: clock[0])
    r.tick()
    assert led.balance("u1") == 50            # escrowed
    evs.clear()
    clock[0] = 61.0
    r.tick()
    assert led.balance("u1") == 100 and any("expired" in x for x in replies)


def test_accept_with_no_pending_duel():
    led = Ledger(":memory:")
    r, led, replies = _router([_ev("!accept", uid="u1", login="alice", mid="a1")],
                              ledger=led, cfg=_cfg(earn_per_minute=0))
    r.tick()
    assert any("no duel to accept" in x for x in replies)


# --------------------------------------------------------------------------- #
# !raffle — mod-opened entry window, house prize
# --------------------------------------------------------------------------- #
def test_raffle_mod_opens_then_draws_winner():
    led = Ledger(":memory:")
    clock = [0.0]
    evs = [_ev("!raffle", uid="um", login="mod", mid="r0", mod=True),
           _ev("!raffle", uid="u1", login="alice", mid="r1"),
           _ev("!enter", uid="u2", login="bob", mid="r2")]
    r, led, replies = _router(
        evs, ledger=led,
        cfg=_cfg(earn_per_minute=0, raffle_window_seconds=30, raffle_prize=500),
        now=lambda: clock[0])
    r.tick()
    assert any("RAFFLE open" in x for x in replies)
    assert any("entered the raffle" in x for x in replies)
    evs.clear()
    clock[0] = 31.0
    r.tick()   # FakeRNG.outcome -> 0 -> first entrant (alice) wins
    assert led.balance("u1") == 500 and any("RAFFLE winner" in x for x in replies)


def test_raffle_non_mod_cannot_open():
    led = Ledger(":memory:")
    r, led, replies = _router([_ev("!raffle", uid="u1", login="alice", mid="r1")],
                              ledger=led, cfg=_cfg(earn_per_minute=0))
    r.tick()
    assert any("no raffle is running" in x for x in replies)


def test_empty_message_id_bet_is_replay_idempotent():
    # A (malformed/synthetic) chat event with NO message_id must NOT double-spend on
    # replay: the stable content-hash surrogate (_event_key) dedups it AND keys its
    # ledger legs, so re-delivery applies the stake exactly once.
    led = Ledger(":memory:")
    led.credit("u1", 1000, "s", "s1")
    flat = {"type": "chat", "message_id": "", "chatter_user_id": "u1",
            "chatter_login": "alice", "chatter_name": "alice", "text": "!gamble 100"}
    ev = chat_event_from_buffer(flat)
    assert ev is not None and ev.message_id == ""
    events = [ev]
    r, led, _replies = _router(events, ledger=led, cfg=_cfg(earn_per_minute=0),
                               rng=FakeRNG(uniform=0.99))   # draw >= 0.5 -> loss
    r.tick()
    assert led.balance("u1") == 900           # one 100 stake debited, lost
    r.tick()                                   # SAME id-less event re-delivered
    assert led.balance("u1") == 900           # deduped -> NOT 800, no double-spend


def test_gamble_rtp_is_net_negative_over_many_rounds():
    # Real RNG, many rounds at a fixed stake: the house edge means the player's
    # expected return is ~rtp (< 1). Statistical, with a generous band.
    led = Ledger(":memory:")
    led.credit("u1", 10_000_000, "seed", "seed")
    rng = ProvablyFairRNG(default_client_seed="ev")
    cfg = _cfg(command_cooldown_seconds=0, max_bet=0)
    replies = []
    start = led.balance("u1")
    rounds = 2000
    evs = [_ev("!gamble 100", mid=f"g{i}") for i in range(rounds)]
    r = ChatGameRouter(lambda: evs, ledger=led, cfg=cfg, rng=rng, announce_fn=replies.append)
    r.tick()
    wagered = rounds * 100
    returned = (led.balance("u1") - start) + wagered   # net change + stakes = gross returned
    rtp_observed = returned / wagered
    assert 0.75 < rtp_observed < 1.05, rtp_observed   # centered on ~0.90


def test_config_defaults_off():
    from kenning.config import TwitchEconomyConfig
    c = TwitchEconomyConfig()
    assert c.chat_commands_enabled is False
    assert c.command_cooldown_seconds == 5
    assert c.min_bet == 1 and c.max_bet == 10000


def _trivia_router(led, *, prize=100, window=30.0, clock=None):
    cfg = _cfg(trivia_prize=prize, trivia_window_seconds=window)
    replies, batch = [], []
    kw = {"now_fn": (lambda: clock["t"])} if clock is not None else {}
    r = ChatGameRouter(lambda: list(batch), ledger=led, cfg=cfg,
                       rng=ProvablyFairRNG(default_client_seed="t"),
                       announce_fn=replies.append, **kw)
    return r, batch, replies


def test_trivia_mod_starts_first_correct_wins_and_closes():
    led = Ledger(":memory:")
    r, batch, replies = _trivia_router(led, prize=100)
    batch[:] = [_ev("!trivia", login="modder", mod=True, mid="t1")]
    r.tick()
    assert r._trivia is not None and any("TRIVIA" in x for x in replies)
    answer = r._trivia["question"].answer   # whatever was drawn
    batch[:] = [_ev(answer, uid="u2", login="bob", mid="a1")]
    r.tick()
    assert led.balance("u2") == 100
    assert r._trivia is None and any("got it" in x for x in replies)
    # a replayed / second correct answer does NOT double-award (round already closed)
    batch[:] = [_ev(answer, uid="u2", login="bob", mid="a2")]
    r.tick()
    assert led.balance("u2") == 100


def test_trivia_non_mod_cannot_start():
    led = Ledger(":memory:")
    r, batch, replies = _trivia_router(led)
    batch[:] = [_ev("!trivia", login="rando", mod=False, mid="t")]
    r.tick()
    assert r._trivia is None and any("only mods" in x for x in replies)


def test_trivia_wrong_answer_keeps_round_open():
    led = Ledger(":memory:")
    r, batch, _ = _trivia_router(led)
    batch[:] = [_ev("!trivia", mod=True, login="m", mid="t")]
    r.tick()
    batch[:] = [_ev("definitely not the trivia answer xyz", uid="u2", mid="w")]
    r.tick()
    assert r._trivia is not None and led.balance("u2") == 0


def test_trivia_times_out():
    led = Ledger(":memory:")
    clock = {"t": 0.0}
    r, batch, replies = _trivia_router(led, window=10.0, clock=clock)
    batch[:] = [_ev("!trivia", mod=True, login="m", mid="t")]
    r.tick()
    clock["t"] = 20.0      # past the 10s window
    batch[:] = []
    r.tick()
    assert r._trivia is None and any("timed out" in x for x in replies)


def test_trivia_already_running():
    led = Ledger(":memory:")
    r, batch, replies = _trivia_router(led)
    batch[:] = [_ev("!trivia", mod=True, login="m", mid="t1")]
    r.tick()
    batch[:] = [_ev("!trivia", mod=True, login="m", mid="t2")]
    r.tick()
    assert any("already running" in x for x in replies)


def test_orchestrator_wires_chat_games_gated_and_closes_ledger():
    import inspect
    from kenning.pipeline.orchestrator import Orchestrator
    hook = inspect.getsource(Orchestrator._start_twitch_chat_mode)
    assert "ChatGameRouter" in hook and "make_chat_command_drain_fn" in hook
    assert "chat_commands_enabled" in hook          # gated default-OFF
    assert "_twitch_chat_game_router" in hook and "_twitch_ledger" in hook
    # the economy ledger is checkpointed/closed somewhere in the orchestrator shutdown
    full = inspect.getsource(Orchestrator)
    assert "_twitch_ledger" in full and "_ledger.close()" in full


def test_no_banned_imports():
    import kenning.twitch.economy.chat_games as m
    src = m.__file__
    import ast
    tree = ast.parse(open(src, encoding="utf-8").read())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    banned = {"pyautogui", "mss", "pynput", "keyboard", "mouse", "win32api", "torch"}
    assert not (roots & banned), roots & banned


# --------------------------------------------------------------------------- #
# Change 1 — defer !points / !gamble to StreamElements (no double reply)
# --------------------------------------------------------------------------- #
def test_points_deferred_to_streamelements_no_reply():
    led = Ledger(":memory:")
    led.credit("u1", 250, "seed", "seed")
    r, led, replies = _router([_ev("!points")], ledger=led,
                              cfg=_cfg(defer_points_gamble_to_streamelements=True))
    handled = r.tick()
    # Ultron stays SILENT — StreamElements' bot answers !points.
    assert replies == [] and handled == 0


def test_gamble_deferred_to_streamelements_no_reply_no_ledger_touch():
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    r, led, replies = _router([_ev("!gamble 100", mid="m2")], ledger=led,
                              cfg=_cfg(defer_points_gamble_to_streamelements=True),
                              rng=FakeRNG(uniform=0.1))
    handled = r.tick()
    assert replies == [] and handled == 0
    assert led.balance("u1") == 1000               # never debited/credited
    assert led.history("u1", limit=10)[:] == led.history("u1", limit=10)  # no game legs
    keys = {e.idempotency_key for e in led.history("u1", limit=10)}
    assert "gamble:m2:bet" not in keys and "gamble:m2:win" not in keys


def test_balance_alias_also_deferred():
    led = Ledger(":memory:")
    led.credit("u1", 50, "seed", "seed")
    r, led, replies = _router([_ev("!balance")], ledger=led,
                              cfg=_cfg(defer_points_gamble_to_streamelements=True))
    assert r.tick() == 0 and replies == []


def test_other_commands_still_handled_when_deferring():
    # With deferral ON, every NON-points/gamble command still works.
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    cfg = _cfg(defer_points_gamble_to_streamelements=True, earn_per_minute=0,
               wheel_free_per_stream=1)
    # !slots, !wheel, !leaderboard, !help — all should produce replies.
    evs = [_ev("!slots 10", uid="u1", login="alice", mid="s1", ),
           _ev("!wheel", uid="u1", login="alice", mid="w1"),
           _ev("!leaderboard", uid="u1", login="alice", mid="lb"),
           _ev("!help", uid="u1", login="alice", mid="hp")]
    r, led, replies = _router(evs, ledger=led, cfg=cfg, rng=FakeRNG(slots_win=False))
    r.tick()
    joined = " || ".join(replies)
    assert "slots" in joined.lower()
    assert "spun the wheel" in joined.lower()
    assert "Top" in joined            # leaderboard
    assert "Commands:" in joined      # help


def test_help_omits_points_gamble_when_deferring():
    r, _led, replies = _router([_ev("!help", mid="h")],
                               cfg=_cfg(defer_points_gamble_to_streamelements=True))
    r.tick()
    help_line = next(x for x in replies if "Commands:" in x)
    assert "!points" not in help_line and "!gamble" not in help_line
    assert "!slots" in help_line and "!trivia" in help_line


def test_deferral_off_still_handles_points():
    led = Ledger(":memory:")
    led.credit("u1", 250, "seed", "seed")
    r, led, replies = _router([_ev("!points")], ledger=led,
                              cfg=_cfg(defer_points_gamble_to_streamelements=False))
    r.tick()
    assert replies == ["@alice you have 250 cores."]


# --------------------------------------------------------------------------- #
# Change 2 — currency renamed to 'one taps'
# --------------------------------------------------------------------------- #
def test_replies_use_one_taps_currency():
    led = Ledger(":memory:")
    led.credit("u1", 100, "seed", "seed")
    cfg = _cfg(currency_name="one taps", defer_points_gamble_to_streamelements=False)
    r, led, replies = _router([_ev("!points")], ledger=led, cfg=cfg)
    r.tick()
    assert replies == ["@alice you have 100 one taps."]


def test_config_default_currency_is_one_taps():
    from kenning.config import TwitchEconomyConfig
    assert TwitchEconomyConfig().currency_name == "one taps"


def test_config_defaults_defer_and_auto_trivia():
    from kenning.config import TwitchEconomyConfig
    c = TwitchEconomyConfig()
    assert c.defer_points_gamble_to_streamelements is True
    assert c.trivia_auto_interval_minutes == 8


# --------------------------------------------------------------------------- #
# Change 3 — auto-trivia on the periodic clock
# --------------------------------------------------------------------------- #
def test_auto_trivia_fires_on_interval_via_injected_clock():
    led = Ledger(":memory:")
    clock = {"mono": 0.0, "epoch": 0.0}
    cfg = _cfg(earn_per_minute=0, trivia_auto_interval_minutes=5,
               trivia_window_seconds=30)
    r = ChatGameRouter(lambda: [], ledger=led, cfg=cfg,
                       rng=ProvablyFairRNG(default_client_seed="t"),
                       announce_fn=(replies := []).append,
                       now_fn=lambda: clock["mono"], epoch_fn=lambda: clock["epoch"])
    r.tick()                       # arms the auto-trivia clock at epoch 0
    assert r._trivia is None and replies == []
    clock["epoch"] = 4 * 60.0      # 4 minutes — still before the 5-minute interval
    r.tick()
    assert r._trivia is None and replies == []
    clock["epoch"] = 5 * 60.0      # 5 minutes elapsed -> auto-start
    r.tick()
    assert r._trivia is not None
    assert any("TRIVIA" in x for x in replies)


def test_auto_trivia_does_not_start_while_round_active():
    led = Ledger(":memory:")
    clock = {"mono": 0.0, "epoch": 0.0}
    cfg = _cfg(earn_per_minute=0, trivia_auto_interval_minutes=5,
               trivia_window_seconds=600)   # long window so the manual round stays open
    batch = []
    r = ChatGameRouter(lambda: list(batch), ledger=led, cfg=cfg,
                       rng=ProvablyFairRNG(default_client_seed="t"),
                       announce_fn=(replies := []).append,
                       now_fn=lambda: clock["mono"], epoch_fn=lambda: clock["epoch"])
    # A mod starts a round at t=0.
    batch[:] = [_ev("!trivia", login="modder", mod=True, mid="t1")]
    r.tick()
    active_q = r._trivia
    assert active_q is not None
    starts_before = sum("TRIVIA for" in x for x in replies)
    # Advance past the auto interval; the manual round is still open (mono unchanged).
    batch[:] = []
    clock["epoch"] = 6 * 60.0
    r.tick()
    # No NEW trivia announced — the active round was not displaced.
    assert r._trivia is active_q
    assert sum("TRIVIA for" in x for x in replies) == starts_before


def test_auto_trivia_disabled_when_interval_zero():
    led = Ledger(":memory:")
    clock = {"mono": 0.0, "epoch": 0.0}
    cfg = _cfg(earn_per_minute=0, trivia_auto_interval_minutes=0)
    r = ChatGameRouter(lambda: [], ledger=led, cfg=cfg,
                       rng=ProvablyFairRNG(default_client_seed="t"),
                       announce_fn=(replies := []).append,
                       now_fn=lambda: clock["mono"], epoch_fn=lambda: clock["epoch"])
    r.tick()
    clock["epoch"] = 60 * 60.0     # an hour later
    r.tick()
    assert r._trivia is None and replies == []


def test_auto_trivia_can_restart_after_a_round_times_out():
    led = Ledger(":memory:")
    clock = {"mono": 0.0, "epoch": 0.0}
    cfg = _cfg(earn_per_minute=0, trivia_auto_interval_minutes=5,
               trivia_window_seconds=30)
    r = ChatGameRouter(lambda: [], ledger=led, cfg=cfg,
                       rng=ProvablyFairRNG(default_client_seed="t"),
                       announce_fn=(replies := []).append,
                       now_fn=lambda: clock["mono"], epoch_fn=lambda: clock["epoch"])
    r.tick()                       # arm
    clock["epoch"] = 5 * 60.0
    r.tick()                       # first auto round
    assert r._trivia is not None
    first_starts = sum("TRIVIA for" in x for x in replies)
    # Let the window lapse (advance the monotonic clock past the 30s deadline) so
    # _expire_trivia clears it, then advance another interval -> a second round.
    clock["mono"] = 100.0
    clock["epoch"] = 10 * 60.0
    r.tick()
    assert r._trivia is not None
    assert sum("TRIVIA for" in x for x in replies) == first_starts + 1


# --------------------------------------------------------------------------- #
# Change 4 — the expanded, multi-topic trivia question bank
# --------------------------------------------------------------------------- #
def test_expanded_trivia_bank_is_large_and_structurally_valid():
    from kenning.twitch.economy.games import Trivia, TriviaQuestion, _TRIVIA_POOL
    from kenning.twitch.economy.trivia_questions import TRIVIA_QUESTIONS

    assert _TRIVIA_POOL is TRIVIA_QUESTIONS            # default pool = the expanded bank
    assert len(TRIVIA_QUESTIONS) > 150, len(TRIVIA_QUESTIONS)
    t = Trivia()                                       # constructs with the default pool
    for i, q in enumerate(TRIVIA_QUESTIONS):
        assert isinstance(q, TriviaQuestion), i
        assert isinstance(q.question, str) and q.question.strip(), i
        assert isinstance(q.answer, str) and q.answer.strip(), i
        assert isinstance(q.accept, tuple), i
        # The canonical answer must be accepted by the matcher (case-insensitive).
        assert t.check_answer(q, q.answer), (i, q.answer)
        assert t.check_answer(q, q.answer.upper()), (i, q.answer)
        # Every declared alias must also match.
        for alias in q.accept:
            assert t.check_answer(q, alias), (i, alias)
        # A clearly-wrong answer must not match.
        assert not t.check_answer(q, "__definitely_wrong__"), i


def test_no_duplicate_trivia_questions():
    from kenning.twitch.economy.trivia_questions import TRIVIA_QUESTIONS
    prompts = [q.question.strip().casefold() for q in TRIVIA_QUESTIONS]
    assert len(prompts) == len(set(prompts)), "duplicate trivia prompts present"


def test_trivia_alias_answer_wins_round():
    # End-to-end: a chatter answering with an ACCEPTED ALIAS (not the canonical
    # answer) still wins the round.
    from kenning.twitch.economy.games import Trivia, TriviaQuestion
    led = Ledger(":memory:")
    pool = (TriviaQuestion("How many?", "13", accept=("thirteen",)),)
    cfg = _cfg(trivia_prize=100, trivia_window_seconds=30,
               defer_points_gamble_to_streamelements=True)
    replies, batch = [], []
    r = ChatGameRouter(lambda: list(batch), ledger=led, cfg=cfg,
                       rng=ProvablyFairRNG(default_client_seed="t"),
                       announce_fn=replies.append)
    r._trivia_game = Trivia(rng=ProvablyFairRNG(default_client_seed="t"), pool=pool)
    batch[:] = [_ev("!trivia", login="modder", mod=True, mid="t1")]
    r.tick()
    assert r._trivia is not None
    batch[:] = [_ev("thirteen", uid="u2", login="bob", mid="a1")]   # the ALIAS
    r.tick()
    assert led.balance("u2") == 100 and r._trivia is None


# --------------------------------------------------------------------------- #
# overlay emit — each chat-game OUTCOME emits a chat_game event that PASSES the
# real overlay validator; an overlay-emit exception never breaks the game/tick.
# --------------------------------------------------------------------------- #
from kenning.twitch.overlay.server import validate_event as _validate_overlay_event  # noqa: E402


def _overlay_router(events, *, ledger=None, rng=None, cfg=None, now=None, epoch=None):
    """A ChatGameRouter with a capturing overlay sink. Returns
    (router, ledger, replies, overlay_events)."""
    ledger = ledger or Ledger(":memory:")
    overlay_events: list = []
    replies: list = []
    kw = {}
    if now is not None:
        kw["now_fn"] = now
    if epoch is not None:
        kw["epoch_fn"] = epoch
    r = ChatGameRouter(
        lambda: list(events), ledger=ledger, cfg=cfg or _cfg(),
        rng=rng or FakeRNG(), announce_fn=replies.append,
        overlay_emit=overlay_events.append, **kw,
    )
    return r, ledger, replies, overlay_events


def _assert_valid_overlay(ev: dict) -> dict:
    """Every emitted chat_game event must pass the PRODUCTION overlay validator
    (the same one OverlayServer.emit runs) and carry the chat discriminators."""
    assert ev["type"] == "chat_game"
    assert ev["source"] == "chat"
    vetted = _validate_overlay_event(ev)   # raises OverlayError if the shape is bad
    assert vetted["game"] == ev["game"]
    assert vetted["source"] == "chat"
    return vetted


def test_overlay_emits_slots_win_with_settled_reels():
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    r, led, _replies, ov = _overlay_router(
        [_ev("!slots 10", mid="w")], ledger=led, rng=FakeRNG(slots_win=True))
    r.tick()
    cards = [e for e in ov if e.get("game") == "slots"]
    assert len(cards) == 1
    card = cards[0]
    _assert_valid_overlay(card)
    assert card["won"] is True and card["outcome"] == "WIN"
    # the reels carry the ACTUAL pulled symbols (all the win symbol on a triple)
    assert len(card["detail"]["reels"]) == 3
    assert card["detail"]["win_symbol"] == card["detail"]["reels"][0]
    assert card["amount"] > 0


def test_overlay_emits_slots_loss():
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    r, led, _replies, ov = _overlay_router(
        [_ev("!slots 10", mid="l")], ledger=led, rng=FakeRNG(slots_win=False))
    r.tick()
    card = [e for e in ov if e.get("game") == "slots"][0]
    _assert_valid_overlay(card)
    assert card["won"] is False and card["outcome"] == "LOSS"
    assert card["amount"] == 10        # stake lost


def test_overlay_emits_wheel_segment():
    led = Ledger(":memory:")
    r, led, _replies, ov = _overlay_router(
        [_ev("!wheel", uid="u1", login="alice", mid="w1")],
        ledger=led, cfg=_cfg(wheel_free_per_stream=1, earn_per_minute=0))
    r.tick()
    card = [e for e in ov if e.get("game") == "wheel"][0]
    _assert_valid_overlay(card)
    assert card["detail"]["segment"]   # the landed segment label is surfaced
    assert card["viewer"] == "alice"


def test_overlay_emits_heist_win_and_fail():
    # WIN
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "a")
    led.credit("u2", 100, "s", "b")
    clock = [0.0]
    evs = [_ev("!heist 100", uid="u1", login="alice", mid="h1"),
           _ev("!heist 100", uid="u2", login="bob", mid="h2")]
    r, led, _replies, ov = _overlay_router(
        evs, ledger=led,
        cfg=_cfg(heist_window_seconds=30, heist_house_bonus_pct=0.5,
                 heist_min_players=1, earn_per_minute=0, command_cooldown_seconds=0),
        rng=FakeRNG(uniform=0.9), now=lambda: clock[0])
    r.tick()
    evs.clear(); clock[0] = 31.0
    r.tick()
    card = [e for e in ov if e.get("game") == "heist"][0]
    _assert_valid_overlay(card)
    assert card["won"] is True and card["outcome"] in ("WIN", "PARTIAL")
    assert card["detail"]["crew"] == 2 and card["detail"]["pot"] == 200
    # FAIL
    led2 = Ledger(":memory:")
    led2.credit("u1", 100, "s", "a")
    clock2 = [0.0]
    evs2 = [_ev("!heist 100", uid="u1", login="alice", mid="hf1")]
    r2, led2, _r2, ov2 = _overlay_router(
        evs2, ledger=led2,
        cfg=_cfg(heist_window_seconds=30, heist_min_players=1, earn_per_minute=0),
        rng=FakeRNG(uniform=0.1), now=lambda: clock2[0])
    r2.tick()
    evs2.clear(); clock2[0] = 31.0
    r2.tick()
    fail = [e for e in ov2 if e.get("game") == "heist"][0]
    _assert_valid_overlay(fail)
    assert fail["won"] is False and fail["outcome"] == "FAIL"


def test_overlay_emits_duel_winner():
    led = Ledger(":memory:")
    led.credit("u1", 100, "s", "a")
    led.credit("u2", 100, "s", "b")
    evs = [_ev("hi", uid="u2", login="bob", mid="b0"),
           _ev("!duel @bob 50", uid="u1", login="alice", mid="d1"),
           _ev("!accept", uid="u2", login="bob", mid="d2")]
    r, led, _replies, ov = _overlay_router(
        evs, ledger=led, cfg=_cfg(earn_per_minute=0, duel_window_seconds=60),
        rng=FakeRNG(uniform=0.1))   # challenger (alice) wins
    r.tick()
    card = [e for e in ov if e.get("game") == "duel"][0]
    _assert_valid_overlay(card)
    assert card["won"] is True
    assert card["detail"]["winner"] == "alice" and card["detail"]["loser"] == "bob"
    assert card["amount"] == 100       # pot = wager * 2


def test_overlay_emits_trivia_winner_with_answer():
    led = Ledger(":memory:")
    r, batch, replies = _trivia_router(led, prize=100)
    ov: list = []
    r._overlay = ov.append   # attach a capturing sink
    batch[:] = [_ev("!trivia", login="modder", mod=True, mid="t1")]
    r.tick()
    answer = r._trivia["question"].answer
    batch[:] = [_ev(answer, uid="u2", login="bob", mid="a1")]
    r.tick()
    card = [e for e in ov if e.get("game") == "trivia"][0]
    _assert_valid_overlay(card)
    assert card["won"] is True
    assert card["detail"]["answer"] == answer
    assert card["detail"]["winner"] == "bob"


def test_overlay_emits_raffle_winner():
    led = Ledger(":memory:")
    clock = [0.0]
    evs = [_ev("!raffle", uid="um", login="mod", mid="r0", mod=True),
           _ev("!raffle", uid="u1", login="alice", mid="r1"),
           _ev("!enter", uid="u2", login="bob", mid="r2")]
    r, led, _replies, ov = _overlay_router(
        evs, ledger=led,
        cfg=_cfg(earn_per_minute=0, raffle_window_seconds=30, raffle_prize=500),
        now=lambda: clock[0])
    r.tick()
    evs.clear(); clock[0] = 31.0
    r.tick()
    card = [e for e in ov if e.get("game") == "raffle"][0]
    _assert_valid_overlay(card)
    assert card["won"] is True and card["amount"] == 500
    assert card["detail"]["winner"] and card["detail"]["entrants"] >= 1


def test_overlay_emit_exception_never_breaks_the_game():
    # A throwing overlay sink must NOT prevent the bet from settling (the slots
    # game still debits/credits the ledger + replies) and tick() must not raise.
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")

    def _boom(_ev):
        raise RuntimeError("overlay down")

    replies: list = []
    r = ChatGameRouter(
        lambda: [_ev("!slots 10", mid="w")], ledger=led, cfg=_cfg(),
        rng=FakeRNG(slots_win=True), announce_fn=replies.append, overlay_emit=_boom)
    r.tick()   # must not raise despite the throwing overlay
    s = len(DEFAULT_SLOT_SYMBOLS)
    mult = int(0.90 * s * s)
    assert led.balance("u1") == 1000 - 10 + 10 * mult   # game still settled
    assert any("WON" in x for x in replies)


def test_overlay_none_sink_is_safe():
    # No overlay wired (the overlay-disabled boot) -> games still run, no error.
    led = Ledger(":memory:")
    led.credit("u1", 1000, "seed", "seed")
    r, led, replies = _router([_ev("!slots 10", mid="w")], ledger=led,
                              rng=FakeRNG(slots_win=True))
    # _router does not pass overlay_emit -> self._overlay is None
    r.tick()
    assert any("WON" in x for x in replies)
