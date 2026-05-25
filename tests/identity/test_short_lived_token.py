"""Tests for the T7 trusted-publisher short-lived token primitive."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest

from ultron.identity.short_lived_token import (
    ALGORITHM_HS256,
    DEFAULT_TTL_SECONDS,
    MAX_TTL_SECONDS,
    TokenExpiredError,
    TokenMintError,
    TokenSignatureError,
    TokenVerifyError,
    TrustedCaller,
    TrustedCallerClaimMismatch,
    TrustedCallerNotFoundError,
    VerifiedClaims,
    list_trusted_callers,
    load_trusted_caller,
    mint_token,
    register_trusted_caller,
    rotate_secret,
    verify_audit_chain,
    verify_token,
)


# ---------------------------------------------------------------------------
# register / load trusted callers


def test_register_and_load_caller(tmp_path: Path) -> None:
    caller = TrustedCaller(
        caller_id="mcp:tools",
        expected_claims_match={"scope_kind": "mcp"},
        allowed_scopes=("read_file", "list_files"),
    )
    register_trusted_caller(caller, project_root=tmp_path)
    loaded = load_trusted_caller("mcp:tools", project_root=tmp_path)
    assert loaded is not None
    assert loaded.caller_id == "mcp:tools"
    assert loaded.expected_claims_match == {"scope_kind": "mcp"}
    assert set(loaded.allowed_scopes) == {"read_file", "list_files"}


def test_register_empty_caller_id_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        register_trusted_caller(
            TrustedCaller(caller_id=""), project_root=tmp_path,
        )


def test_load_unknown_caller_returns_none(tmp_path: Path) -> None:
    assert load_trusted_caller("no-such", project_root=tmp_path) is None


def test_register_supersedes_earlier_row(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(
            caller_id="mcp:tools",
            allowed_scopes=("a",),
            notes="first",
        ),
        project_root=tmp_path,
    )
    register_trusted_caller(
        TrustedCaller(
            caller_id="mcp:tools",
            allowed_scopes=("a", "b"),
            notes="second",
        ),
        project_root=tmp_path,
    )
    loaded = load_trusted_caller("mcp:tools", project_root=tmp_path)
    assert loaded is not None
    assert set(loaded.allowed_scopes) == {"a", "b"}
    assert loaded.notes == "second"


def test_list_trusted_callers_returns_counts(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="a"), project_root=tmp_path,
    )
    register_trusted_caller(
        TrustedCaller(caller_id="a", notes="rotation"), project_root=tmp_path,
    )
    register_trusted_caller(
        TrustedCaller(caller_id="b"), project_root=tmp_path,
    )
    entries, counts = list_trusted_callers(project_root=tmp_path)
    assert len(entries) == 2
    assert counts == {"a": 2, "b": 1}


# ---------------------------------------------------------------------------
# mint_token


def test_mint_basic_token(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="mcp:tools", allowed_scopes=("x",)),
        project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path,
        caller_id="mcp:tools",
        audience="ultron-mcp-server",
        scope=["x"],
    )
    assert token.count(".") == 2


def test_mint_rejects_empty_caller(tmp_path: Path) -> None:
    with pytest.raises(TokenMintError):
        mint_token(
            project_root=tmp_path, caller_id="",
            audience="aud",
        )


def test_mint_rejects_empty_audience(tmp_path: Path) -> None:
    with pytest.raises(TokenMintError):
        mint_token(
            project_root=tmp_path, caller_id="mcp",
            audience="",
        )


def test_mint_rejects_zero_ttl(tmp_path: Path) -> None:
    with pytest.raises(TokenMintError):
        mint_token(
            project_root=tmp_path,
            caller_id="x",
            audience="aud",
            ttl_seconds=0,
        )


def test_mint_rejects_excessive_ttl(tmp_path: Path) -> None:
    with pytest.raises(TokenMintError):
        mint_token(
            project_root=tmp_path,
            caller_id="x",
            audience="aud",
            ttl_seconds=MAX_TTL_SECONDS + 1,
        )


def test_mint_respects_caller_max_ttl(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x", max_ttl_seconds=60),
        project_root=tmp_path,
    )
    with pytest.raises(TokenMintError):
        mint_token(
            project_root=tmp_path,
            caller_id="x",
            audience="aud",
            ttl_seconds=120,
        )


def test_mint_rejects_disallowed_scope(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x", allowed_scopes=("only_this",)),
        project_root=tmp_path,
    )
    with pytest.raises(TokenMintError):
        mint_token(
            project_root=tmp_path,
            caller_id="x",
            audience="aud",
            scope=["not_allowed"],
        )


def test_mint_extra_claims_pass_through(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path,
        caller_id="x",
        audience="aud",
        extra_claims={"task_id": "task-42", "sandbox": "/tmp/sandbox"},
    )
    parts = token.split(".")
    payload = json.loads(_b64url_decode(parts[1]))
    assert payload["task_id"] == "task-42"
    assert payload["sandbox"] == "/tmp/sandbox"


def test_mint_extra_claims_cannot_override_reserved(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path,
        caller_id="x",
        audience="aud",
        extra_claims={"aud": "evil", "exp": 0, "iss": "attacker"},
    )
    payload = json.loads(_b64url_decode(token.split(".")[1]))
    assert payload["aud"] == "aud"  # not overridden
    assert payload["iss"] == "ultron-local"


# ---------------------------------------------------------------------------
# verify_token (happy path)


def test_verify_happy_path(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x", allowed_scopes=("a",)),
        project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path,
        caller_id="x",
        audience="aud",
        scope=["a"],
    )
    verified = verify_token(
        token, project_root=tmp_path, expected_audience="aud",
    )
    assert isinstance(verified, VerifiedClaims)
    assert verified.caller_id == "x"
    assert verified.audience == "aud"
    assert verified.scope == ("a",)


def test_verify_carries_jti(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    verified = verify_token(token, project_root=tmp_path, expected_audience="aud")
    assert len(verified.jti) == 32


# ---------------------------------------------------------------------------
# verify_token (failure paths)


def test_verify_rejects_malformed_token(tmp_path: Path) -> None:
    with pytest.raises(TokenVerifyError):
        verify_token("not.a.real.jwt", project_root=tmp_path, expected_audience="x")
    with pytest.raises(TokenVerifyError):
        verify_token("only.two", project_root=tmp_path, expected_audience="x")


def test_verify_rejects_unknown_alg(tmp_path: Path) -> None:
    header = _b64url_encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64url_encode(
        json.dumps({"aud": "x", "exp": int(time.time()) + 100, "caller_id": "x"}).encode()
    )
    fake = f"{header}.{payload}."
    with pytest.raises(TokenVerifyError):
        verify_token(fake, project_root=tmp_path, expected_audience="x")


def test_verify_rejects_audience_mismatch(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud-real",
    )
    with pytest.raises(TokenVerifyError):
        verify_token(token, project_root=tmp_path, expected_audience="aud-other")


def test_verify_rejects_expired_token(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    base_time = [1_000_000]
    token = mint_token(
        project_root=tmp_path,
        caller_id="x",
        audience="aud",
        ttl_seconds=10,
        now=lambda: base_time[0],
    )
    # Move 5 minutes past expiry.
    base_time[0] += 300
    with pytest.raises(TokenExpiredError):
        verify_token(
            token,
            project_root=tmp_path,
            expected_audience="aud",
            now=lambda: base_time[0],
            clock_skew_seconds=0,
        )


def test_verify_rejects_future_nbf(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    base_time = [1_000_000]
    token = mint_token(
        project_root=tmp_path,
        caller_id="x",
        audience="aud",
        now=lambda: base_time[0],
    )
    base_time[0] -= 100  # roll the clock backward
    with pytest.raises(TokenExpiredError):
        verify_token(
            token,
            project_root=tmp_path,
            expected_audience="aud",
            now=lambda: base_time[0],
            clock_skew_seconds=0,
        )


def test_verify_rejects_signature_after_secret_rotation(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    rotate_secret(project_root=tmp_path)
    with pytest.raises(TokenSignatureError):
        verify_token(token, project_root=tmp_path, expected_audience="aud")


def test_verify_rejects_unknown_caller(tmp_path: Path) -> None:
    # Register, mint, then delete the caller -> verify should refuse.
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    # Wipe the trusted-caller file.
    callers_path = tmp_path / "data" / "identity" / "trusted_callers.jsonl"
    callers_path.unlink()
    with pytest.raises(TrustedCallerNotFoundError):
        verify_token(token, project_root=tmp_path, expected_audience="aud")


def test_verify_enforces_expected_claims_match(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(
            caller_id="x",
            expected_claims_match={"task_id": "task-42"},
        ),
        project_root=tmp_path,
    )
    # Token with WRONG task_id is rejected.
    token_bad = mint_token(
        project_root=tmp_path,
        caller_id="x",
        audience="aud",
        extra_claims={"task_id": "task-99"},
    )
    with pytest.raises(TrustedCallerClaimMismatch):
        verify_token(token_bad, project_root=tmp_path, expected_audience="aud")
    # Token with CORRECT task_id passes.
    token_good = mint_token(
        project_root=tmp_path,
        caller_id="x",
        audience="aud",
        extra_claims={"task_id": "task-42"},
    )
    verified = verify_token(
        token_good, project_root=tmp_path, expected_audience="aud",
    )
    assert verified.claims.get("task_id") == "task-42"


def test_verify_enforces_allowed_scopes_post_registration(tmp_path: Path) -> None:
    """Tighten the allowlist AFTER a token is minted -> verify refuses."""
    register_trusted_caller(
        TrustedCaller(caller_id="x", allowed_scopes=("a", "b")),
        project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path,
        caller_id="x",
        audience="aud",
        scope=["a", "b"],
    )
    # Tighten the allowlist (rotation).
    register_trusted_caller(
        TrustedCaller(caller_id="x", allowed_scopes=("a",)),
        project_root=tmp_path,
    )
    with pytest.raises(TrustedCallerClaimMismatch):
        verify_token(token, project_root=tmp_path, expected_audience="aud")


# ---------------------------------------------------------------------------
# Audit log


def test_audit_log_records_mint_and_verify(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    verify_token(token, project_root=tmp_path, expected_audience="aud")
    log_path = tmp_path / "data" / "identity" / "short_lived_tokens.jsonl"
    assert log_path.is_file()
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    ops = [r["op"] for r in rows]
    # register_trusted_caller + mint + verify_ok all logged.
    assert "register_caller" in ops
    assert "mint" in ops
    assert "verify_ok" in ops


def test_audit_chain_verifies_clean(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    verify_token(token, project_root=tmp_path, expected_audience="aud")
    assert verify_audit_chain(project_root=tmp_path) is True


def test_audit_chain_detects_tamper(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    verify_token(token, project_root=tmp_path, expected_audience="aud")
    log_path = tmp_path / "data" / "identity" / "short_lived_tokens.jsonl"
    raw = log_path.read_text(encoding="utf-8")
    # "aud" is the audience value present in every minted row.
    tampered = raw.replace('"aud"', '"evil-aud"', 1)
    assert tampered != raw  # sanity-check the replacement landed
    log_path.write_text(tampered, encoding="utf-8")
    assert verify_audit_chain(project_root=tmp_path) is False


# ---------------------------------------------------------------------------
# Round-trip / encoding


def test_secret_persists_across_mints(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    t1 = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    t2 = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    # Both verify under the same secret.
    assert verify_token(t1, project_root=tmp_path, expected_audience="aud")
    assert verify_token(t2, project_root=tmp_path, expected_audience="aud")


def test_secret_rotation_invalidates_old_tokens(tmp_path: Path) -> None:
    register_trusted_caller(
        TrustedCaller(caller_id="x"), project_root=tmp_path,
    )
    token = mint_token(
        project_root=tmp_path, caller_id="x", audience="aud",
    )
    rotate_secret(project_root=tmp_path)
    with pytest.raises(TokenSignatureError):
        verify_token(token, project_root=tmp_path, expected_audience="aud")


# ---------------------------------------------------------------------------
# Helpers


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
