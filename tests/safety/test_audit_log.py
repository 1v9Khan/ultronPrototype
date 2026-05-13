"""Tests for the tamper-evident audit log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultron.safety.audit import (
    GENESIS_PREV_HASH,
    AuditLog,
    _hash_line,
)


def test_creates_parent_dir(tmp_path):
    p = tmp_path / "deep" / "nested" / "audit.jsonl"
    AuditLog(path=p)
    assert p.parent.is_dir()


def test_records_entry_with_genesis_prev_hash(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AuditLog(path=p)
    a.record(
        rule_id="K1", verdict="BLOCK_HARD",
        tool_name="t", capability="c", reason="r",
    )
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["prev_hash"] == GENESIS_PREV_HASH
    assert entry["rule_id"] == "K1"
    assert entry["verdict"] == "BLOCK_HARD"


def test_chain_links_correctly_across_writes(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AuditLog(path=p)
    a.record(rule_id="K1", verdict="BLOCK_HARD", tool_name="t", capability="c", reason="r1")
    a.record(rule_id="K2", verdict="BLOCK_HARD", tool_name="t", capability="c", reason="r2")
    a.record(rule_id="K3", verdict="BLOCK_HARD", tool_name="t", capability="c", reason="r3")

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    e0 = json.loads(lines[0])
    e1 = json.loads(lines[1])
    e2 = json.loads(lines[2])

    assert e0["prev_hash"] == GENESIS_PREV_HASH
    assert e1["prev_hash"] == _hash_line(lines[0])
    assert e2["prev_hash"] == _hash_line(lines[1])


def test_verify_chain_returns_ok_on_clean_log(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AuditLog(path=p)
    for i in range(5):
        a.record(
            rule_id=f"K{i+1}", verdict="BLOCK_HARD",
            tool_name="t", capability="c", reason=f"r{i}",
        )
    ok, msg = a.verify_chain()
    assert ok, msg
    assert msg == "ok"


def test_verify_chain_returns_empty_when_no_file(tmp_path):
    p = tmp_path / "absent.jsonl"
    a = AuditLog(path=p)
    ok, msg = a.verify_chain()
    assert ok
    assert msg == "empty"


def test_verify_chain_detects_retroactive_deletion(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AuditLog(path=p)
    for i in range(5):
        a.record(
            rule_id=f"K{i+1}", verdict="BLOCK_HARD",
            tool_name="t", capability="c", reason=f"r{i}",
        )
    # Tamper: remove the middle line.
    lines = p.read_text(encoding="utf-8").splitlines()
    tampered = "\n".join(lines[:2] + lines[3:]) + "\n"
    p.write_text(tampered, encoding="utf-8")

    ok, msg = a.verify_chain()
    assert not ok
    assert "prev_hash mismatch" in msg


def test_verify_chain_detects_modified_entry(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AuditLog(path=p)
    a.record(rule_id="K1", verdict="BLOCK_HARD", tool_name="t", capability="c", reason="r1")
    a.record(rule_id="K2", verdict="BLOCK_HARD", tool_name="t", capability="c", reason="r2")

    # Modify the first line's content but preserve the prev_hash field.
    lines = p.read_text(encoding="utf-8").splitlines()
    e = json.loads(lines[0])
    e["reason"] = "ALTERED"
    lines[0] = json.dumps(e, sort_keys=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, msg = a.verify_chain()
    assert not ok
    assert "prev_hash mismatch" in msg


def test_tail_hash_survives_writer_recreation(tmp_path):
    """Construct a new AuditLog pointing at an existing file -- it
    must pick up the tail hash so subsequent writes link correctly."""
    p = tmp_path / "audit.jsonl"
    a1 = AuditLog(path=p)
    a1.record(rule_id="K1", verdict="BLOCK_HARD", tool_name="t", capability="c", reason="r1")
    a1.record(rule_id="K2", verdict="BLOCK_HARD", tool_name="t", capability="c", reason="r2")

    # Simulate a process restart by creating a fresh writer.
    a2 = AuditLog(path=p)
    a2.record(rule_id="K3", verdict="BLOCK_HARD", tool_name="t", capability="c", reason="r3")

    ok, msg = a2.verify_chain()
    assert ok, msg


def test_invalid_json_in_log_is_detected(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    a = AuditLog(path=p)
    ok, msg = a.verify_chain()
    assert not ok
    assert "invalid JSON" in msg
