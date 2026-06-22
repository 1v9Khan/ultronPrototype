"""S11 moderation tests — HelixClient (idempotency / rate / 429 backoff) +
ModerationGuard (resolve / authorize / breaker / audit).

Fully offline: the Helix transport is injected and the clock is driven, so no
real network, credentials, or models are touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kenning.twitch.moderation import (
    HelixClient,
    HelixError,
    ModerationGuard,
    RateGovernor,
    RosterEntry,
)
from kenning.twitch.moderation.helix import TransportResponse


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeClock:
    """A monotonic clock that only advances when test code (or sleep) moves it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt

    def sleep(self, dt: float) -> None:
        # Sleeping advances the same fake clock so RateGovernor refills.
        self.t += dt


class ScriptedTransport:
    """An injected Helix transport that returns queued responses (or a default)
    and records every call (method, url, headers, body)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._queue: list[TransportResponse] = []
        self.default = TransportResponse(status=200, body=json.dumps({"data": []}))

    def queue(self, *responses: TransportResponse) -> "ScriptedTransport":
        self._queue.extend(responses)
        return self

    def __call__(self, method, url, headers, body):  # matches Transport signature
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": json.loads(body.decode()) if body else None,
            }
        )
        if self._queue:
            return self._queue.pop(0)
        return self.default


def make_client(transport, clock: FakeClock, **kw) -> HelixClient:
    gov = RateGovernor(rate=kw.pop("rate", 1000.0), burst=kw.pop("burst", 1000),
                       monotonic=clock.monotonic, sleep=clock.sleep)
    return HelixClient(
        client_id="cid",
        get_token=lambda: "tok",
        transport=transport,
        rate_governor=gov,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        base_backoff_s=1.0,
        max_backoff_s=8.0,
        **kw,
    )


# --------------------------------------------------------------------------- #
# HelixClient — idempotency
# --------------------------------------------------------------------------- #
def test_ban_success_posts_no_duration():
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=200, body=json.dumps({"data": [{"user_id": "42"}]})))
    client = make_client(tr, clock)
    res = client.ban_user("bcast", "mod", "42", reason="rule break")
    assert res.ok and not res.idempotent and res.action == "ban"
    # POST to /moderation/bans with the user_id but NO duration.
    call = tr.calls[0]
    assert call["method"] == "POST"
    assert "/moderation/bans" in call["url"]
    assert call["body"]["data"]["user_id"] == "42"
    assert call["body"]["data"]["reason"] == "rule break"
    assert "duration" not in call["body"]["data"]


def test_ban_409_is_idempotent_success_no_raise():
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=409, body=json.dumps({"message": "conflict"})))
    client = make_client(tr, clock)
    res = client.ban_user("bcast", "mod", "42")
    assert res.ok is True
    assert res.idempotent is True
    assert res.status == 409


def test_ban_already_banned_body_is_idempotent_success():
    clock = FakeClock()
    body = json.dumps({"error": "Bad Request", "status": 400,
                       "message": "The user specified in the user_id field is already banned."})
    tr = ScriptedTransport().queue(TransportResponse(status=400, body=body))
    client = make_client(tr, clock)
    res = client.ban_user("bcast", "mod", "42")
    assert res.ok is True and res.idempotent is True
    assert res.status == 400


def test_ban_local_idempotency_short_circuits_second_call():
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=200, body=json.dumps({"data": [{"user_id": "42"}]})))
    client = make_client(tr, clock)
    first = client.ban_user("bcast", "mod", "42")
    assert first.ok and not first.idempotent
    # Second ban of the same target must NOT hit the network again.
    second = client.ban_user("bcast", "mod", "42")
    assert second.ok and second.idempotent and second.status == 0
    assert len(tr.calls) == 1  # exactly one network call total


def test_ban_auth_failure_raises_loud():
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=401, body=json.dumps({"message": "invalid token"})))
    client = make_client(tr, clock)
    with pytest.raises(HelixError) as ei:
        client.ban_user("bcast", "mod", "42")
    assert ei.value.status == 401


def test_no_token_raises_before_network():
    clock = FakeClock()
    tr = ScriptedTransport()
    client = HelixClient(client_id="cid", get_token=lambda: "", transport=tr,
                         rate_governor=RateGovernor(rate=1000, burst=1000,
                                                    monotonic=clock.monotonic, sleep=clock.sleep),
                         monotonic=clock.monotonic, sleep=clock.sleep)
    with pytest.raises(HelixError):
        client.ban_user("bcast", "mod", "42")
    assert tr.calls == []  # never reached the transport


# --------------------------------------------------------------------------- #
# HelixClient — timeout
# --------------------------------------------------------------------------- #
def test_timeout_includes_duration():
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=200, body=json.dumps({"data": []})))
    client = make_client(tr, clock)
    res = client.timeout_user("bcast", "mod", "42", duration_s=600, reason="cool off")
    assert res.ok and res.action == "timeout"
    assert tr.calls[0]["body"]["data"]["duration"] == 600


def test_timeout_rejects_out_of_range_duration():
    clock = FakeClock()
    client = make_client(ScriptedTransport(), clock)
    for bad in (0, -5, 1_209_601):
        with pytest.raises(ValueError):
            client.timeout_user("bcast", "mod", "42", duration_s=bad)


def test_ban_and_timeout_have_distinct_idempotency_keys():
    clock = FakeClock()
    tr = ScriptedTransport().queue(
        TransportResponse(status=200, body=json.dumps({"data": []})),
        TransportResponse(status=200, body=json.dumps({"data": []})),
    )
    client = make_client(tr, clock)
    client.ban_user("bcast", "mod", "42")
    client.timeout_user("bcast", "mod", "42", duration_s=60)
    # Different actions on the same target => two real calls (keys differ).
    assert len(tr.calls) == 2


# --------------------------------------------------------------------------- #
# HelixClient — delete_message keys on message_id
# --------------------------------------------------------------------------- #
def test_delete_message_keys_on_message_id():
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=204, body=""))
    client = make_client(tr, clock)
    res = client.delete_message("bcast", "mod", "msg-1")
    assert res.ok and res.action == "delete_message"
    call = tr.calls[0]
    assert call["method"] == "DELETE"
    assert "message_id=msg-1" in call["url"]
    # Re-deleting the SAME message id short-circuits (no second call).
    res2 = client.delete_message("bcast", "mod", "msg-1")
    assert res2.ok and res2.idempotent and res2.status == 0
    assert len(tr.calls) == 1
    # A DIFFERENT message id is a distinct key -> a new call.
    tr.queue(TransportResponse(status=204, body=""))
    res3 = client.delete_message("bcast", "mod", "msg-2")
    assert res3.ok and not res3.idempotent
    assert len(tr.calls) == 2


def test_delete_message_404_is_idempotent_success():
    clock = FakeClock()
    tr = ScriptedTransport().queue(
        TransportResponse(status=404, body=json.dumps({"message": "message does not exist"}))
    )
    client = make_client(tr, clock)
    res = client.delete_message("bcast", "mod", "gone")
    assert res.ok is True and res.idempotent is True and res.status == 404


# --------------------------------------------------------------------------- #
# HelixClient — update_chat_settings
# --------------------------------------------------------------------------- #
def test_update_chat_settings_filters_to_allowed_keys():
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=200, body=json.dumps({"data": [{"slow_mode": True}]})))
    client = make_client(tr, clock)
    res = client.update_chat_settings(
        "bcast", "mod",
        {"slow_mode": True, "slow_mode_wait_time": 10, "evil_key": "x"},
    )
    assert res.ok
    body = tr.calls[0]["body"]
    assert body["slow_mode"] is True and body["slow_mode_wait_time"] == 10
    assert "evil_key" not in body


def test_update_chat_settings_rejects_no_known_keys():
    clock = FakeClock()
    client = make_client(ScriptedTransport(), clock)
    with pytest.raises(ValueError):
        client.update_chat_settings("bcast", "mod", {"nonsense": 1})


# --------------------------------------------------------------------------- #
# HelixClient — rate governor + 429 backoff
# --------------------------------------------------------------------------- #
def test_rate_governor_throttles():
    clock = FakeClock()
    # 2 tokens/sec, burst 2: the 3rd acquire in a burst must wait ~0.5s.
    gov = RateGovernor(rate=2.0, burst=2, monotonic=clock.monotonic, sleep=clock.sleep)
    assert gov.try_acquire() is True
    assert gov.try_acquire() is True
    assert gov.try_acquire() is False  # bucket drained
    t0 = clock.t
    gov.acquire()  # blocks -> sleeps advance the fake clock until a token refills
    assert clock.t > t0  # time genuinely passed (throttled)
    assert (clock.t - t0) >= 0.4  # ~0.5s for one token at 2/s


def test_rate_governor_caps_burst():
    clock = FakeClock()
    gov = RateGovernor(rate=1.0, burst=3, monotonic=clock.monotonic, sleep=clock.sleep)
    # Let plenty of time pass; bucket must NOT exceed burst capacity.
    clock.advance(100.0)
    got = sum(1 for _ in range(10) if gov.try_acquire())
    assert got == 3  # capped at burst, not 10


def test_429_then_success_backs_off_and_retries():
    clock = FakeClock()
    tr = ScriptedTransport().queue(
        TransportResponse(status=429, body=json.dumps({"message": "Too Many Requests"})),
        TransportResponse(status=429, body=json.dumps({"message": "Too Many Requests"})),
        TransportResponse(status=200, body=json.dumps({"data": [{"user_id": "42"}]})),
    )
    client = make_client(tr, clock, max_retries=4)
    t0 = clock.t
    res = client.ban_user("bcast", "mod", "42")
    assert res.ok and not res.idempotent and res.status == 200
    assert len(tr.calls) == 3  # two 429s then success
    # Exponential backoff actually slept: base 1.0 + 2.0 = >= 3s of fake time.
    assert (clock.t - t0) >= 3.0


def test_429_exhausts_retries_returns_loud_failure():
    clock = FakeClock()
    tr = ScriptedTransport()
    tr.queue(*[TransportResponse(status=429, body=json.dumps({"message": "rate"})) for _ in range(6)])
    client = make_client(tr, clock, max_retries=2)
    with pytest.raises(HelixError) as ei:
        client.ban_user("bcast", "mod", "42")
    assert ei.value.status == 429
    # initial try + 2 retries = 3 calls (never blind-retries beyond the cap).
    assert len(tr.calls) == 3


def test_write_never_blind_retries_on_500():
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=500, body=json.dumps({"message": "boom"})))
    client = make_client(tr, clock)
    with pytest.raises(HelixError) as ei:
        client.ban_user("bcast", "mod", "42")
    assert ei.value.status == 500
    assert len(tr.calls) == 1  # exactly one attempt — a 5xx is NOT retried


# --------------------------------------------------------------------------- #
# ModerationGuard — resolve
# --------------------------------------------------------------------------- #
def _roster():
    return [
        RosterEntry(user_id="1", login="shroud", display_name="Shroud"),
        RosterEntry(user_id="2", login="tenz", display_name="TenZ"),
        RosterEntry(user_id="3", login="aspas", display_name="aspas"),
        RosterEntry(user_id="4", login="asuna", display_name="Asuna"),
    ]


def make_guard(tmp_path: Path, protected=(), **kw) -> ModerationGuard:
    clock = kw.pop("clock", None)
    extra = {}
    if clock is not None:
        extra["monotonic"] = clock.monotonic
    return ModerationGuard(
        roster_provider=kw.pop("roster", _roster),
        protected_ids=protected,
        audit_path=tmp_path / "twitch_actions.jsonl",
        **extra,
        **kw,
    )


def test_resolve_exact_login(tmp_path):
    g = make_guard(tmp_path)
    r = g.resolve("shroud")
    assert r.user_id == "1" and not r.ambiguous and r.reason == "exact_login"


def test_resolve_exact_login_is_case_and_punct_insensitive(tmp_path):
    g = make_guard(tmp_path)
    r = g.resolve("  TenZ! ")
    assert r.user_id == "2" and not r.ambiguous


def test_resolve_fuzzy_unique(tmp_path):
    g = make_guard(tmp_path)
    # A near-spelling of a single distinct roster name resolves uniquely.
    r = g.resolve("shroudd")
    assert r.user_id == "1" and not r.ambiguous and r.reason == "fuzzy_unique"


def test_resolve_ambiguous_close_pair_no_autopick(tmp_path):
    # Two similar logins => a small top-2 margin => ambiguous, no auto-pick.
    roster = [
        RosterEntry(user_id="10", login="player1", display_name="player1"),
        RosterEntry(user_id="11", login="player2", display_name="player2"),
    ]
    g = make_guard(tmp_path, roster=lambda: roster)
    r = g.resolve("player")
    assert r.user_id is None and r.ambiguous is True
    assert r.reason == "ambiguous_margin"
    assert len(r.candidates) >= 2  # the human sees both


def test_resolve_low_score_is_ambiguous(tmp_path):
    g = make_guard(tmp_path)
    r = g.resolve("zzzzqqqq")  # nothing close
    assert r.user_id is None and r.ambiguous is True
    assert r.reason in ("ambiguous_low_score", "no_match")


def test_resolve_empty_and_empty_roster(tmp_path):
    g = make_guard(tmp_path)
    assert g.resolve("").reason == "empty"
    g2 = make_guard(tmp_path, roster=lambda: [])
    assert g2.resolve("shroud").user_id is None


def test_resolve_fails_closed_on_provider_error(tmp_path):
    def boom():
        raise RuntimeError("roster sidecar down")

    g = make_guard(tmp_path, roster=boom)
    r = g.resolve("shroud")
    assert r.user_id is None  # fail-CLOSED, no target


# --------------------------------------------------------------------------- #
# ModerationGuard — authorize (protected guard + breaker)
# --------------------------------------------------------------------------- #
def test_authorize_refuses_self_mod_broadcaster(tmp_path):
    g = make_guard(tmp_path, protected={"self-id", "mod-id", "bcast-id"})
    for pid in ("self-id", "mod-id", "bcast-id"):
        res = g.authorize("ban", pid)
        assert res.allowed is False and res.reason == "protected_target"


def test_authorize_allows_normal_target(tmp_path):
    g = make_guard(tmp_path, protected={"bcast-id"})
    res = g.authorize("ban", "99")
    assert res.allowed is True and res.reason == "authorized"


def test_mass_action_breaker_trips_after_n(tmp_path):
    clock = FakeClock()
    g = make_guard(tmp_path, clock=clock, breaker_limit=3, breaker_window_s=60.0)
    # First 3 distinct targets allowed...
    assert g.authorize("ban", "a").allowed is True
    assert g.authorize("ban", "b").allowed is True
    assert g.authorize("ban", "c").allowed is True
    # ...the 4th within the window trips the breaker.
    tripped = g.authorize("ban", "d")
    assert tripped.allowed is False and tripped.reason == "mass_action_breaker"
    # After the window slides past, actions are allowed again.
    clock.advance(61.0)
    assert g.authorize("ban", "e").allowed is True


def test_authorize_empty_target_refused(tmp_path):
    g = make_guard(tmp_path)
    res = g.authorize("ban", "")
    assert res.allowed is False and res.reason == "empty_target"


# --------------------------------------------------------------------------- #
# ModerationGuard — audit
# --------------------------------------------------------------------------- #
def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_audit_row_written_for_authorize_and_applied(tmp_path):
    from kenning.twitch.moderation.guard import JsonlAuditWriter

    audit_path = tmp_path / "twitch_actions.jsonl"
    # Inject the flat JSONL writer so we assert the documented twitch_actions.jsonl
    # contract (top-level action/target_id/verdict).
    g = ModerationGuard(
        roster_provider=_roster, protected_ids=(),
        audit=JsonlAuditWriter(audit_path),
    )
    g.authorize("ban", "99")
    g.record_applied("ban", "99", idempotent=False, status=200)
    rows = _read_audit(audit_path)
    assert len(rows) >= 2
    verdicts = [r.get("verdict") for r in rows]
    assert "ALLOWED" in verdicts
    assert "APPLIED" in verdicts
    applied = next(r for r in rows if r.get("verdict") == "APPLIED")
    assert applied["target_id"] == "99" and applied["action"] == "ban"
    assert applied["status"] == 200 and applied["applied"] is True


def test_audit_records_refusal(tmp_path):
    from kenning.twitch.moderation.guard import JsonlAuditWriter

    audit_path = tmp_path / "twitch_actions.jsonl"
    g = ModerationGuard(
        roster_provider=_roster, protected_ids={"99"},
        audit=JsonlAuditWriter(audit_path),
    )
    g.authorize("ban", "99")
    rows = _read_audit(audit_path)
    assert any(r.get("verdict") == "REFUSED" and r.get("reason") == "protected_target" for r in rows)


def test_default_audit_reuses_hash_chained_auditlog(tmp_path):
    """When kenning.safety.audit is importable, the default writer is the reused
    hash-chained AuditLog (rows carry a prev_hash; custom fields ride in context)."""
    audit_path = tmp_path / "twitch_actions.jsonl"
    g = ModerationGuard(roster_provider=_roster, protected_ids=(), audit_path=audit_path)
    g.authorize("ban", "99")
    rows = _read_audit(audit_path)
    assert rows, "the default AuditLog adapter should have written a row"
    row = rows[0]
    # AuditLog's fixed schema: hash chain + our fields adapted into context.
    assert row["prev_hash"] == "0" * 64  # genesis
    assert row["capability"] == "twitch_moderation"
    assert row["tool_name"] == "ban"
    assert row["context"]["target_id"] == "99"


def test_audit_uses_injected_writer(tmp_path):
    captured: list[dict] = []

    class FakeAudit:
        def record(self, **fields):
            captured.append(fields)

    g = ModerationGuard(roster_provider=_roster, protected_ids=(), audit=FakeAudit())
    g.authorize("ban", "99")
    assert captured and captured[-1]["action"] == "ban"


def test_audit_writer_sanitizes_control_chars(tmp_path):
    from kenning.twitch.moderation.guard import JsonlAuditWriter

    audit_path = tmp_path / "a.jsonl"
    w = JsonlAuditWriter(audit_path)
    # A forged-log-line attempt with CR/CSI control bytes.
    w.record(action="ban", target_id="x", note="evil\r\n\x1b[2Kfake")
    rows = _read_audit(audit_path)
    assert "\x1b" not in rows[0]["note"] and "\r" not in rows[0]["note"]


def test_audit_failure_never_blocks_decision(tmp_path):
    class ExplodingAudit:
        def record(self, **fields):
            raise OSError("disk full")

    g = ModerationGuard(roster_provider=_roster, protected_ids=(), audit=ExplodingAudit())
    # The verdict must still come back despite the audit blowing up.
    res = g.authorize("ban", "99")
    assert res.allowed is True


# --------------------------------------------------------------------------- #
# Integration: guard verdict gates the Helix call (the kill chain)
# --------------------------------------------------------------------------- #
def test_guard_refusal_means_no_helix_call(tmp_path):
    clock = FakeClock()
    tr = ScriptedTransport()
    client = make_client(tr, clock)
    g = make_guard(tmp_path, protected={"99"})

    decision = g.authorize("ban", "99")
    assert decision.allowed is False
    # The caller respects the verdict: no Helix write happens.
    if decision.allowed:  # pragma: no cover - guard refused
        client.ban_user("bcast", "mod", "99")
    assert tr.calls == []


def test_full_chain_resolve_authorize_ban(tmp_path):
    clock = FakeClock()
    tr = ScriptedTransport().queue(TransportResponse(status=200, body=json.dumps({"data": [{"user_id": "1"}]})))
    client = make_client(tr, clock)
    g = make_guard(tmp_path, protected={"bcast-id", "mod-id"})

    r = g.resolve("shroud")
    assert r.user_id == "1"
    decision = g.authorize("ban", r.user_id)
    assert decision.allowed
    res = client.ban_user("bcast", "mod", r.user_id, reason="spamming")
    assert res.ok
    g.record_applied("ban", r.user_id, idempotent=res.idempotent, status=res.status)
    assert len(tr.calls) == 1
