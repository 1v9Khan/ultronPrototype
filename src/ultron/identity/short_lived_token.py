"""Trusted-publisher short-lived token mint + verify (T7).

T7 (openclaw-clawhub catalog port; see ``THIRD_PARTY_NOTICES.md``).
The marketplace pattern is "identity claim must match a pre-registered
tuple before a short-lived token is minted, replacing long-lived
secrets with revocation-by-expiry". The single-user ultron
adaptation maps the pattern onto four call sites the catalogue
called out:

* **MCP server startup** — instead of a long-lived
  ``ULTRON_MCP_TOKEN`` in the server's env, the orchestrator
  mints a short-lived JWT scoped to that server's PID + start
  time + tool allowlist.
* **Coding bridge subprocess** — the coding agent subprocess
  gets a token bound to the current task id, expiring at task
  completion + 5 min.
* **Skill execution token** — a skill that needs network egress
  to a specific allowlisted domain gets a token claiming
  ``{skill_id, sandbox_root, allowed_egress: [host]}`` for one
  turn.
* **Voice gaming-mode handoff** — engage mints a token authorising
  the lighter Llama 3.2 3B preset to take over for the gaming
  session; disengage revokes by letting the token expire.

Implementation: HMAC-SHA256 JWT (HS256) with a local signing
secret stored at ``data/identity/short_lived_token_secret.bin``.
Generated on first use; rotation supported via :func:`rotate_secret`.
RSA-256 with TPM-backed keys is documented as the future hardening
path (the JWT contract is identical -- only the algorithm and key
storage change). HS256 is sufficient for ultron's single-user
runtime where the verifier and minter share trust boundary.

Audit-log integration: every mint + verify is logged to a
hash-chained JSONL at ``data/identity/short_lived_tokens.jsonl``
so a compromised signing key still leaves forensic evidence of
which claims were minted and when.

Pre-registered trust tuples (the "before a token is minted, the
caller's claims must match a pre-registered shape" contract) live
in ``data/identity/trusted_callers.jsonl``. Tuple registration is
itself a K-category gate (modifying the trust registry requires
explicit voice intent + audit-logged event), so an attacker who
steals the signing key cannot register a fresh caller without
leaving evidence.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants


#: JWT signing algorithm. HS256 (HMAC-SHA256) is the only algorithm
#: this implementation supports. RSA-256 (RS256) is documented as
#: the future hardening path; the JWT contract is algorithm-neutral.
ALGORITHM_HS256: str = "HS256"

#: Default token lifetime. Short by design: revocation is by
#: expiry. Long-running operations need to refresh.
DEFAULT_TTL_SECONDS: int = 300

#: Max permitted token lifetime. Mints beyond this raise
#: :class:`TokenMintError`. Protects against accidental long-
#: lived issuance.
MAX_TTL_SECONDS: int = 60 * 60 * 6  # 6 hours

#: Tolerated clock skew on verify (seconds). Matches the upstream
#: 60-second window.
DEFAULT_CLOCK_SKEW_SECONDS: int = 60

#: Default subdir under PROJECT_ROOT.
DEFAULT_IDENTITY_SUBDIR: str = "data/identity"

#: Signing-secret filename.
SECRET_FILENAME: str = "short_lived_token_secret.bin"

#: Trusted-caller registry filename.
TRUSTED_CALLERS_FILENAME: str = "trusted_callers.jsonl"

#: Mint+verify audit log filename.
AUDIT_LOG_FILENAME: str = "short_lived_tokens.jsonl"

#: Header / payload keys preserved verbatim from the JWT spec
#: (these are RFC 7519 contract names; no copyright).
JWT_KEY_AUDIENCE: str = "aud"
JWT_KEY_ISSUER: str = "iss"
JWT_KEY_SUBJECT: str = "sub"
JWT_KEY_EXPIRES: str = "exp"
JWT_KEY_NOT_BEFORE: str = "nbf"
JWT_KEY_ISSUED_AT: str = "iat"
JWT_KEY_JWT_ID: str = "jti"

#: Issuer string emitted in minted tokens (the audit-log "who
#: signed this" anchor).
DEFAULT_ISSUER: str = "ultron-local"


# ---------------------------------------------------------------------------
# Exceptions


class TokenError(RuntimeError):
    """Base class for token-related failures."""


class TokenMintError(TokenError):
    """Raised when a mint request fails validation."""


class TokenVerifyError(TokenError):
    """Raised when a verify request fails."""


class TokenExpiredError(TokenVerifyError):
    """Subclass for the explicit-expiry failure mode (the most common)."""


class TokenSignatureError(TokenVerifyError):
    """Subclass for signature-mismatch (key rotation / tamper)."""


class TrustedCallerNotFoundError(TokenVerifyError):
    """Caller id in a token has no pre-registered trust tuple."""


class TrustedCallerClaimMismatch(TokenVerifyError):
    """Caller's pre-registered tuple disagrees with at least one claim."""


# ---------------------------------------------------------------------------
# Trusted-caller registry


@dataclass(frozen=True)
class TrustedCaller:
    """One pre-registered trust tuple.

    Fields:
        caller_id: opaque short string identifying the caller
            (e.g. ``"mcp:tools"`` / ``"coding:bridge"`` /
            ``"voice:gaming-engage"``).
        expected_claims_match: mapping of claim-name to required
            value. A mint that doesn't carry every expected value
            is refused at the boundary.
        allowed_scopes: optional set of scope strings the caller
            may request. When non-empty, every minted token's
            ``scope`` claim is required to be a subset of this set.
        max_ttl_seconds: optional per-caller TTL ceiling
            (overrides :data:`MAX_TTL_SECONDS` downward).
        registered_at: ISO-8601 timestamp.
        notes: free-form annotation.
    """

    caller_id: str
    expected_claims_match: Mapping[str, object] = field(default_factory=dict)
    allowed_scopes: tuple[str, ...] = ()
    max_ttl_seconds: Optional[int] = None
    registered_at: str = ""
    notes: str = ""


@dataclass(frozen=True)
class VerifiedClaims:
    """Result of :func:`verify_token` on a successfully-verified token."""

    caller_id: str
    audience: str
    issuer: str
    subject: str
    issued_at: int
    expires_at: int
    jti: str
    scope: tuple[str, ...]
    claims: Mapping[str, object]


# ---------------------------------------------------------------------------
# Base64url helpers (RFC 7515; padding stripped per JWT convention)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


# ---------------------------------------------------------------------------
# Signing-secret management


def _identity_dir(project_root: Path) -> Path:
    return Path(project_root) / DEFAULT_IDENTITY_SUBDIR


def _secret_path(project_root: Path) -> Path:
    return _identity_dir(project_root) / SECRET_FILENAME


def _trusted_callers_path(project_root: Path) -> Path:
    return _identity_dir(project_root) / TRUSTED_CALLERS_FILENAME


def _audit_log_path(project_root: Path) -> Path:
    return _identity_dir(project_root) / AUDIT_LOG_FILENAME


def _read_or_create_secret(project_root: Path) -> bytes:
    """Return the signing secret bytes, generating on first call.

    The secret is 32 random bytes (256 bits, matching HS256's
    recommended key length). File permissions are set to owner-
    only on POSIX; Windows leaves them at the user's umask but the
    ``data/identity/`` directory should already be ACL'd to the
    operator.
    """
    path = _secret_path(project_root)
    if path.is_file():
        try:
            return path.read_bytes()
        except OSError as exc:
            LOGGER.warning("Cannot read token signing secret at %s: %s", path, exc)
    secret = secrets.token_bytes(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(secret)
        try:
            os.chmod(path, 0o600)
        except (OSError, PermissionError):
            # Best-effort on Windows; do not fail mint when ACL
            # adjustment is not supported by the filesystem.
            pass
    except OSError as exc:
        LOGGER.warning("Cannot persist token signing secret to %s: %s", path, exc)
    return secret


def rotate_secret(*, project_root: Path) -> None:
    """Delete the stored signing secret so a fresh one is generated.

    Subsequent mints use the new key; tokens minted under the old
    key fail :class:`TokenSignatureError` on verify. The audit log
    is NOT touched -- historical mint records remain inspectable.
    """
    path = _secret_path(project_root)
    if path.is_file():
        try:
            path.unlink()
        except OSError as exc:
            LOGGER.warning("Cannot rotate token signing secret at %s: %s", path, exc)


# ---------------------------------------------------------------------------
# JWT mint + verify


def mint_token(
    *,
    project_root: Path,
    caller_id: str,
    audience: str,
    scope: Iterable[str] = (),
    subject: str = "",
    extra_claims: Optional[Mapping[str, object]] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    issuer: str = DEFAULT_ISSUER,
    now: Optional[Callable[[], int]] = None,
    secret: Optional[bytes] = None,
) -> str:
    """Mint a short-lived HS256 JWT.

    Raises :class:`TokenMintError` when ``ttl_seconds <= 0``, when
    ``ttl_seconds`` exceeds :data:`MAX_TTL_SECONDS`, when
    ``caller_id`` / ``audience`` is empty, OR when a trusted-caller
    tuple is registered for ``caller_id`` and the requested scope
    isn't a subset of the allowlist.

    ``now`` returns Unix-epoch seconds; defaults to :func:`time.time`.
    ``secret`` is read from disk when None.
    """
    if not caller_id or not caller_id.strip():
        raise TokenMintError("caller_id is required")
    if not audience or not audience.strip():
        raise TokenMintError("audience is required")
    if ttl_seconds <= 0:
        raise TokenMintError("ttl_seconds must be positive")
    if ttl_seconds > MAX_TTL_SECONDS:
        raise TokenMintError(
            f"ttl_seconds {ttl_seconds} exceeds MAX_TTL_SECONDS {MAX_TTL_SECONDS}"
        )

    caller = load_trusted_caller(caller_id, project_root=project_root)
    scope_tuple = tuple(s for s in scope if s)
    if caller is not None and caller.allowed_scopes:
        wanted = set(scope_tuple)
        allowed = set(caller.allowed_scopes)
        if not wanted.issubset(allowed):
            disallowed = sorted(wanted - allowed)
            raise TokenMintError(
                f"caller {caller_id!r} requested disallowed scopes: {disallowed}"
            )
    if caller is not None and caller.max_ttl_seconds is not None:
        if ttl_seconds > caller.max_ttl_seconds:
            raise TokenMintError(
                f"caller {caller_id!r} max_ttl_seconds is "
                f"{caller.max_ttl_seconds}; requested {ttl_seconds}"
            )

    now_fn = now or (lambda: int(time.time()))
    issued_at = int(now_fn())
    expires_at = issued_at + ttl_seconds
    jti = uuid.uuid4().hex

    header = {"alg": ALGORITHM_HS256, "typ": "JWT"}
    payload: dict[str, object] = {
        JWT_KEY_ISSUER: issuer,
        JWT_KEY_AUDIENCE: audience,
        JWT_KEY_SUBJECT: subject or caller_id,
        JWT_KEY_ISSUED_AT: issued_at,
        JWT_KEY_NOT_BEFORE: issued_at,
        JWT_KEY_EXPIRES: expires_at,
        JWT_KEY_JWT_ID: jti,
        "caller_id": caller_id,
        "scope": list(scope_tuple),
    }
    if extra_claims:
        for key, value in extra_claims.items():
            # Reserved JWT keys cannot be overridden via extra_claims
            # (the caller would corrupt the verify path).
            if key in payload:
                continue
            payload[key] = value

    signing_secret = secret if secret is not None else _read_or_create_secret(project_root)
    header_b64 = _b64url_encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(signing_secret, signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)
    token = f"{header_b64}.{payload_b64}.{signature_b64}"

    _append_audit_event(
        project_root,
        op="mint",
        caller_id=caller_id,
        payload={
            "audience": audience,
            "subject": payload[JWT_KEY_SUBJECT],
            "scope": list(scope_tuple),
            "jti": jti,
            "expires_at": expires_at,
        },
    )
    return token


def verify_token(
    token: str,
    *,
    project_root: Path,
    expected_audience: str,
    now: Optional[Callable[[], int]] = None,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
    secret: Optional[bytes] = None,
) -> VerifiedClaims:
    """Verify ``token`` and return the parsed :class:`VerifiedClaims`.

    Raises:
        :class:`TokenSignatureError` on signature mismatch.
        :class:`TokenExpiredError` on past-expiry or future-nbf.
        :class:`TokenVerifyError` on every other validation failure
            (malformed JWT, missing claim, wrong audience).
        :class:`TrustedCallerNotFoundError` when ``caller_id`` in
            the token has no pre-registered tuple.
        :class:`TrustedCallerClaimMismatch` when one of the
            ``expected_claims_match`` values doesn't agree with the
            corresponding claim.
    """
    if not isinstance(token, str) or token.count(".") != 2:
        raise TokenVerifyError("malformed JWT: expected 3 segments")
    parts = token.split(".")
    header_b64, payload_b64, signature_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
        signature = _b64url_decode(signature_b64)
    except (ValueError, json.JSONDecodeError) as exc:
        raise TokenVerifyError(f"malformed JWT segments: {exc}") from exc
    if header.get("alg") != ALGORITHM_HS256:
        raise TokenVerifyError(
            f"unsupported alg {header.get('alg')!r}; only HS256 allowed"
        )

    signing_secret = secret if secret is not None else _read_or_create_secret(project_root)
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(signing_secret, signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_sig):
        raise TokenSignatureError("signature mismatch")

    now_fn = now or (lambda: int(time.time()))
    now_value = int(now_fn())
    expires_at = payload.get(JWT_KEY_EXPIRES)
    not_before = payload.get(JWT_KEY_NOT_BEFORE)
    if not isinstance(expires_at, (int, float)):
        raise TokenVerifyError("missing or non-numeric 'exp' claim")
    if isinstance(not_before, (int, float)) and now_value < int(not_before) - clock_skew_seconds:
        raise TokenExpiredError(
            f"token not yet valid (nbf={int(not_before)}, now={now_value})"
        )
    if now_value > int(expires_at) + clock_skew_seconds:
        raise TokenExpiredError(
            f"token expired (exp={int(expires_at)}, now={now_value})"
        )

    audience = payload.get(JWT_KEY_AUDIENCE)
    if audience != expected_audience:
        raise TokenVerifyError(
            f"audience mismatch: expected {expected_audience!r}, got {audience!r}"
        )

    caller_id = payload.get("caller_id") or payload.get(JWT_KEY_SUBJECT)
    if not isinstance(caller_id, str) or not caller_id:
        raise TokenVerifyError("missing caller_id claim")
    caller = load_trusted_caller(caller_id, project_root=project_root)
    if caller is None:
        raise TrustedCallerNotFoundError(
            f"caller {caller_id!r} not registered as trusted"
        )
    for key, expected_value in caller.expected_claims_match.items():
        actual = payload.get(key)
        if actual != expected_value:
            raise TrustedCallerClaimMismatch(
                f"caller {caller_id!r} claim {key!r} mismatch: "
                f"expected {expected_value!r}, got {actual!r}"
            )

    scope_raw = payload.get("scope")
    if isinstance(scope_raw, list):
        scope_tuple = tuple(str(s) for s in scope_raw)
    else:
        scope_tuple = ()
    if caller.allowed_scopes:
        wanted = set(scope_tuple)
        allowed = set(caller.allowed_scopes)
        if not wanted.issubset(allowed):
            raise TrustedCallerClaimMismatch(
                f"caller {caller_id!r} scope contains disallowed entries"
            )

    _append_audit_event(
        project_root,
        op="verify_ok",
        caller_id=caller_id,
        payload={
            "audience": audience,
            "jti": payload.get(JWT_KEY_JWT_ID, ""),
            "expires_at": int(expires_at),
        },
    )

    return VerifiedClaims(
        caller_id=caller_id,
        audience=str(audience),
        issuer=str(payload.get(JWT_KEY_ISSUER, "")),
        subject=str(payload.get(JWT_KEY_SUBJECT, "")),
        issued_at=int(payload.get(JWT_KEY_ISSUED_AT, 0)),
        expires_at=int(expires_at),
        jti=str(payload.get(JWT_KEY_JWT_ID, "")),
        scope=scope_tuple,
        claims=dict(payload),
    )


# ---------------------------------------------------------------------------
# Trusted-caller registry CRUD


def register_trusted_caller(
    caller: TrustedCaller,
    *,
    project_root: Path,
    now: Optional[Callable[[], datetime]] = None,
) -> TrustedCaller:
    """Register ``caller`` in the trust-tuple file.

    Appends to ``data/identity/trusted_callers.jsonl``. Subsequent
    :func:`load_trusted_caller` calls return the latest registered
    row for ``caller_id`` (later registrations supersede earlier).

    Rotating a caller's tuple is therefore a single new
    registration; the historical row stays in the file for audit.
    """
    if not caller.caller_id or not caller.caller_id.strip():
        raise ValueError("caller_id is required")
    now_fn = now or (lambda: datetime.now(timezone.utc))
    timestamp = now_fn()
    if isinstance(timestamp, datetime):
        registered_at_iso = (
            timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        ).isoformat()
    else:
        registered_at_iso = datetime.fromtimestamp(
            float(timestamp), tz=timezone.utc
        ).isoformat()
    record = TrustedCaller(
        caller_id=caller.caller_id,
        expected_claims_match=dict(caller.expected_claims_match),
        allowed_scopes=tuple(caller.allowed_scopes),
        max_ttl_seconds=caller.max_ttl_seconds,
        registered_at=registered_at_iso,
        notes=caller.notes,
    )
    path = _trusted_callers_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "caller_id": record.caller_id,
        "expected_claims_match": dict(record.expected_claims_match),
        "allowed_scopes": list(record.allowed_scopes),
        "max_ttl_seconds": record.max_ttl_seconds,
        "registered_at": record.registered_at,
        "notes": record.notes,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    _append_audit_event(
        project_root,
        op="register_caller",
        caller_id=record.caller_id,
        payload={"allowed_scopes": list(record.allowed_scopes)},
    )
    return record


def load_trusted_caller(
    caller_id: str,
    *,
    project_root: Path,
) -> Optional[TrustedCaller]:
    """Return the latest :class:`TrustedCaller` row for ``caller_id`` or None.

    Reads the entire JSONL file -- linear in registrations. For
    ultron's single-user runtime this is fine; high-volume servers
    would index. Malformed rows are skipped (logged at debug).
    """
    path = _trusted_callers_path(project_root)
    if not path.is_file():
        return None
    latest: Optional[TrustedCaller] = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    LOGGER.debug(
                        "Skipping malformed trusted-caller row in %s",
                        path,
                    )
                    continue
                if not isinstance(record, dict):
                    continue
                if str(record.get("caller_id") or "") != caller_id:
                    continue
                try:
                    latest = TrustedCaller(
                        caller_id=str(record["caller_id"]),
                        expected_claims_match=dict(
                            record.get("expected_claims_match") or {}
                        ),
                        allowed_scopes=tuple(
                            str(s) for s in (record.get("allowed_scopes") or [])
                        ),
                        max_ttl_seconds=(
                            int(record["max_ttl_seconds"])
                            if record.get("max_ttl_seconds") is not None
                            else None
                        ),
                        registered_at=str(record.get("registered_at", "")),
                        notes=str(record.get("notes", "")),
                    )
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError as exc:
        LOGGER.warning("Cannot read trusted-caller registry at %s: %s", path, exc)
        return None
    return latest


def list_trusted_callers(*, project_root: Path) -> tuple[TrustedCaller, dict]:
    """Return (latest entries per caller, raw counts) tuple."""
    path = _trusted_callers_path(project_root)
    if not path.is_file():
        return (tuple(), {})
    latest_per_id: dict[str, TrustedCaller] = {}
    counts: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                cid = str(record.get("caller_id") or "")
                if not cid:
                    continue
                counts[cid] = counts.get(cid, 0) + 1
                try:
                    latest_per_id[cid] = TrustedCaller(
                        caller_id=cid,
                        expected_claims_match=dict(
                            record.get("expected_claims_match") or {}
                        ),
                        allowed_scopes=tuple(
                            str(s) for s in (record.get("allowed_scopes") or [])
                        ),
                        max_ttl_seconds=(
                            int(record["max_ttl_seconds"])
                            if record.get("max_ttl_seconds") is not None
                            else None
                        ),
                        registered_at=str(record.get("registered_at", "")),
                        notes=str(record.get("notes", "")),
                    )
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        return (tuple(), {})
    return (tuple(latest_per_id.values()), counts)


# ---------------------------------------------------------------------------
# Audit log


def _append_audit_event(
    project_root: Path,
    *,
    op: str,
    caller_id: str,
    payload: Mapping[str, object],
) -> None:
    """Append one mint/verify/register row to the audit log.

    Each row records the op + caller_id + timestamp + opaque payload.
    The chain hash links the row to the previous one so tamper
    detection mirrors the safety-audit pattern.
    """
    path = _audit_log_path(project_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        prev_hash = ""
        if path.is_file():
            try:
                with path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    file_size = handle.tell()
                    chunk = min(2048, file_size)
                    handle.seek(file_size - chunk, os.SEEK_SET)
                    tail = handle.read().decode("utf-8", errors="replace")
                last_line = ""
                for line in tail.splitlines():
                    text = line.strip()
                    if text:
                        last_line = text
                if last_line:
                    try:
                        record = json.loads(last_line)
                        prev_hash = str(record.get("hash", ""))
                    except json.JSONDecodeError:
                        prev_hash = ""
            except OSError:
                prev_hash = ""
        now_iso = datetime.now(timezone.utc).isoformat()
        canonical = json.dumps(
            {
                "op": op,
                "caller_id": caller_id,
                "ts": now_iso,
                "payload": dict(payload),
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        row = {
            "op": op,
            "caller_id": caller_id,
            "ts": now_iso,
            "payload": dict(payload),
            "prev_hash": prev_hash,
            "hash": digest,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    except OSError as exc:
        LOGGER.warning("Cannot append token audit row to %s: %s", path, exc)


def verify_audit_chain(*, project_root: Path) -> bool:
    """Return True iff every row's ``prev_hash`` matches.

    Replays the file and recomputes the chain. Returns False on
    any parse failure / mismatch / missing chain field.
    """
    path = _audit_log_path(project_root)
    if not path.is_file():
        return True
    prev_hash = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    return False
                if not isinstance(record, dict):
                    return False
                declared_prev = str(record.get("prev_hash", ""))
                declared_hash = str(record.get("hash", ""))
                if declared_prev != prev_hash:
                    return False
                canonical = json.dumps(
                    {
                        "op": str(record.get("op", "")),
                        "caller_id": str(record.get("caller_id", "")),
                        "ts": str(record.get("ts", "")),
                        "payload": record.get("payload") or {},
                        "prev_hash": declared_prev,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if computed != declared_hash:
                    return False
                prev_hash = declared_hash
    except OSError:
        return False
    return True


__all__ = [
    "ALGORITHM_HS256",
    "DEFAULT_TTL_SECONDS",
    "MAX_TTL_SECONDS",
    "DEFAULT_CLOCK_SKEW_SECONDS",
    "DEFAULT_ISSUER",
    "TrustedCaller",
    "VerifiedClaims",
    "TokenError",
    "TokenMintError",
    "TokenVerifyError",
    "TokenExpiredError",
    "TokenSignatureError",
    "TrustedCallerNotFoundError",
    "TrustedCallerClaimMismatch",
    "mint_token",
    "verify_token",
    "register_trusted_caller",
    "load_trusted_caller",
    "list_trusted_callers",
    "rotate_secret",
    "verify_audit_chain",
]
