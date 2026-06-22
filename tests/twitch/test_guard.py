"""S5 — guard client + canary + chat-mode enable gate + prompt format/parse."""
from __future__ import annotations

import pytest

from kenning.twitch.guard import (
    CANARY_SAFE,
    CANARY_UNSAFE,
    GuardModelClient,
    GuardUnavailable,
    build_guard_messages,
    chat_mode_can_enable,
    format_guard_prompt,
    format_llama_guard_prompt,
    parse_guard_output,
)


def _client():
    return GuardModelClient("http://127.0.0.1:65535", timeout=0.5)


def test_classify_parses_result() -> None:
    c = _client()
    c._request = lambda m, p, payload=None: {"unsafe": True, "category": "S10", "score": 0.9}
    r = c.classify("whatever")
    assert r.unsafe and r.category == "S10" and r.score == 0.9


def test_classify_defaults_unsafe_when_field_missing() -> None:
    c = _client()
    c._request = lambda m, p, payload=None: {}
    assert c.classify("x").unsafe is True  # fail-CLOSED default


def test_classify_propagates_guard_unavailable() -> None:
    c = _client()

    def boom(m, p, payload=None):
        raise GuardUnavailable("down")

    c._request = boom
    with pytest.raises(GuardUnavailable):
        c.classify("x")


def _branching_request(unsafe_safe=True):
    # returns unsafe for the bomb probe, safe for the stream probe.
    def req(m, p, payload=None):
        txt = (payload or {}).get("text", "")
        if txt == CANARY_UNSAFE:
            return {"unsafe": True, "category": "S9", "score": 0.95}
        if txt == CANARY_SAFE:
            return {"unsafe": (not unsafe_safe), "score": 0.5}
        return {"unsafe": False, "score": 0.0}
    return req


def test_canary_passes_when_both_correct() -> None:
    c = _client()
    c._request = _branching_request(unsafe_safe=True)
    assert c.canary() is True


def test_canary_fails_when_safe_probe_flagged() -> None:
    c = _client()
    c._request = _branching_request(unsafe_safe=False)  # safe probe wrongly flagged unsafe
    assert c.canary() is False


def test_canary_false_on_unavailable() -> None:
    c = _client()

    def boom(m, p, payload=None):
        raise GuardUnavailable("down")

    c._request = boom
    assert c.canary() is False


def test_enable_gate_requires_healthy_canary_guard() -> None:
    assert chat_mode_can_enable(None, guard_required=False)[0] is True
    assert chat_mode_can_enable(None, guard_required=True)[0] is False

    class Fake:
        def __init__(self, h, can):
            self._h, self._c = h, can

        def health(self):
            return self._h

        def canary(self):
            return self._c

    assert chat_mode_can_enable(Fake(False, True), guard_required=True)[0] is False
    assert chat_mode_can_enable(Fake(True, False), guard_required=True)[0] is False
    assert chat_mode_can_enable(Fake(True, True), guard_required=True)[0] is True


def test_format_prompt_families() -> None:
    sys_lg, user_lg = format_guard_prompt("llama-guard-3-1b", "hello", exchange="prior")
    assert "S10" in sys_lg and "prior" in user_lg and "hello" in user_lg
    sys_g, _ = format_guard_prompt("generic", "hi")
    assert "json" in sys_g.lower() and "unsafe" in sys_g.lower()


def test_parse_llama_guard_output() -> None:
    assert parse_guard_output("llama-guard", "safe").unsafe is False
    r = parse_guard_output("llama-guard", "unsafe\nS10")
    assert r.unsafe is True and "s10" in r.category.lower()
    # garbage -> fail CLOSED (unsafe)
    assert parse_guard_output("llama-guard", "¯\\_(ツ)_/¯").unsafe is True


def test_parse_generic_json_output() -> None:
    assert parse_guard_output("generic", '{"unsafe": false, "score": 0.0}').unsafe is False
    assert parse_guard_output("generic", '{"unsafe": true, "category":"hate"}').unsafe is True
    assert parse_guard_output("generic", "not json at all").unsafe is True  # fail-CLOSED


def test_build_guard_messages_llama_guard_has_no_system_role() -> None:
    # Llama Guard rejects a system message -> conversation turns only.
    msgs = build_guard_messages("llama-guard-3-1b", "hello chat")
    assert [m["role"] for m in msgs] == ["user"]
    assert all(m["role"] != "system" for m in msgs)
    # exchange mode: inbound = user, draft = assistant
    ex = build_guard_messages("llama-guard", "the draft reply", exchange="the inbound")
    assert [m["role"] for m in ex] == ["user", "assistant"]
    assert ex[1]["content"] == "the draft reply"


def test_build_guard_messages_generic_uses_system() -> None:
    msgs = build_guard_messages("generic", "hello")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"


def test_format_llama_guard_prompt_structure() -> None:
    p = format_llama_guard_prompt("you are a slur")
    assert not p.startswith("<|begin_of_text|>")          # llama.cpp prepends BOS
    assert "S10: Hate." in p and "<BEGIN CONVERSATION>" in p
    assert "User: you are a slur" in p
    assert p.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>")
    # exchange mode assesses the Agent turn
    pe = format_llama_guard_prompt("draft", exchange="inbound")
    assert "Agent: draft" in pe and "User: inbound" in pe and "LAST Agent message" in pe
