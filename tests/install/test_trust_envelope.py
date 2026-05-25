"""Tests for the T1 trust envelope + T9 version-exact contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from ultron.install.reason_codes import (
    MALICIOUS_CODES,
    REASON_CODES,
    ModerationVerdict,
)
from ultron.install.trust_envelope import (
    ArtifactKind,
    ModerationState,
    PackageFamily,
    PackageRef,
    ReleaseRef,
    ScanInputs,
    TrustEnvelope,
    VersionExactRequest,
    VersionExactViolation,
    build_trust_envelope,
    derive_blocked_from_download,
    derive_reasons,
    derive_scan_status,
    fetch_for_version,
    is_pending_state,
    make_local_path_envelope,
    reason_for_moderation,
    reason_for_pending,
    reason_for_reports,
    reason_for_scan,
    reason_for_static_malicious,
    reason_for_vt_stale,
    refuse_if_blocked,
    validate_version_exact_request,
)


# ---------------------------------------------------------------------------
# Reason string helpers


def test_reason_for_moderation_none_returns_none() -> None:
    assert reason_for_moderation(None) is None


def test_reason_for_moderation_approved() -> None:
    assert reason_for_moderation(ModerationState.APPROVED) == "manual:approved"


def test_reason_for_moderation_quarantined() -> None:
    assert reason_for_moderation(ModerationState.QUARANTINED) == "manual:quarantined"


def test_reason_for_moderation_revoked() -> None:
    assert reason_for_moderation(ModerationState.REVOKED) == "manual:revoked"


def test_reason_for_scan_not_run_returns_none() -> None:
    assert reason_for_scan(ModerationVerdict.NOT_RUN) is None


def test_reason_for_scan_clean() -> None:
    assert reason_for_scan(ModerationVerdict.CLEAN) == "scan:clean"


def test_reason_for_scan_malicious() -> None:
    assert reason_for_scan(ModerationVerdict.MALICIOUS) == "scan:malicious"


def test_reason_for_reports_zero_returns_none() -> None:
    assert reason_for_reports(0) is None
    assert reason_for_reports(-1) is None


def test_reason_for_reports_positive() -> None:
    assert reason_for_reports(3) == "reports:3"


def test_reason_for_static_malicious_constant() -> None:
    assert reason_for_static_malicious() == "static:malicious"


def test_reason_for_vt_stale_constant() -> None:
    assert reason_for_vt_stale() == "vt:stale"


def test_reason_for_pending_constant() -> None:
    assert reason_for_pending() == "scan:pending"


# ---------------------------------------------------------------------------
# derive_scan_status (the T1 algorithm)


def test_scan_status_manual_approved_wins() -> None:
    inputs = ScanInputs(
        manual_moderation=ModerationState.APPROVED,
        static_scan_verdict=ModerationVerdict.MALICIOUS,
    )
    assert derive_scan_status(inputs) is ModerationVerdict.CLEAN


def test_scan_status_manual_quarantined_forces_malicious() -> None:
    inputs = ScanInputs(
        manual_moderation=ModerationState.QUARANTINED,
        static_scan_verdict=ModerationVerdict.CLEAN,
    )
    assert derive_scan_status(inputs) is ModerationVerdict.MALICIOUS


def test_scan_status_manual_revoked_forces_malicious() -> None:
    inputs = ScanInputs(manual_moderation=ModerationState.REVOKED)
    assert derive_scan_status(inputs) is ModerationVerdict.MALICIOUS


def test_scan_status_llm_verdict_used_when_set() -> None:
    inputs = ScanInputs(
        llm_verdict=ModerationVerdict.SUSPICIOUS,
        static_scan_verdict=ModerationVerdict.CLEAN,
    )
    assert derive_scan_status(inputs) is ModerationVerdict.SUSPICIOUS


def test_scan_status_static_malicious_when_no_llm() -> None:
    inputs = ScanInputs(static_scan_verdict=ModerationVerdict.MALICIOUS)
    assert derive_scan_status(inputs) is ModerationVerdict.MALICIOUS


def test_scan_status_verification_voided_when_both_suspicious() -> None:
    """Static + verification both suspicious -> verification voided, no escalation."""
    inputs = ScanInputs(
        static_scan_verdict=ModerationVerdict.SUSPICIOUS,
        verification_verdict=ModerationVerdict.SUSPICIOUS,
    )
    # static_scan_verdict isn't MALICIOUS, llm_verdict is None, and
    # verification is voided -> falls through to extra_codes (none),
    # content_sha256 (none) -> NOT_RUN.
    assert derive_scan_status(inputs) is ModerationVerdict.NOT_RUN


def test_scan_status_verification_malicious_wins() -> None:
    inputs = ScanInputs(verification_verdict=ModerationVerdict.MALICIOUS)
    assert derive_scan_status(inputs) is ModerationVerdict.MALICIOUS


def test_scan_status_verification_clean_requires_trusted_plugin() -> None:
    inputs = ScanInputs(verification_verdict=ModerationVerdict.CLEAN)
    # Untrusted plugin -> verification accepted (step 9 falls through)
    assert derive_scan_status(inputs) is ModerationVerdict.CLEAN
    inputs_trusted = ScanInputs(
        verification_verdict=ModerationVerdict.CLEAN,
        trusted_openclaw_plugin=True,
    )
    assert derive_scan_status(inputs_trusted) is ModerationVerdict.CLEAN


def test_scan_status_hash_only_returns_pending() -> None:
    inputs = ScanInputs(content_sha256="abc123")
    assert derive_scan_status(inputs) is ModerationVerdict.PENDING


def test_scan_status_no_signals_returns_not_run() -> None:
    inputs = ScanInputs()
    assert derive_scan_status(inputs) is ModerationVerdict.NOT_RUN


def test_scan_status_extras_escalate() -> None:
    inputs = ScanInputs(extra_codes=(REASON_CODES["DANGEROUS_EXEC"],))
    assert derive_scan_status(inputs) is ModerationVerdict.SUSPICIOUS


def test_scan_status_extras_malicious_escalate() -> None:
    inputs = ScanInputs(extra_codes=(REASON_CODES["CRYPTO_MINING"],))
    assert derive_scan_status(inputs) is ModerationVerdict.MALICIOUS


# ---------------------------------------------------------------------------
# derive_blocked_from_download


def test_blocked_when_quarantined() -> None:
    assert derive_blocked_from_download(
        scan_status=ModerationVerdict.CLEAN,
        moderation_state=ModerationState.QUARANTINED,
    )


def test_blocked_when_revoked() -> None:
    assert derive_blocked_from_download(
        scan_status=ModerationVerdict.CLEAN,
        moderation_state=ModerationState.REVOKED,
    )


def test_blocked_when_malicious() -> None:
    assert derive_blocked_from_download(
        scan_status=ModerationVerdict.MALICIOUS,
        moderation_state=None,
    )


def test_not_blocked_when_clean() -> None:
    assert not derive_blocked_from_download(
        scan_status=ModerationVerdict.CLEAN,
        moderation_state=None,
    )


def test_not_blocked_when_pending() -> None:
    """PENDING is a soft block via refuse_if_blocked; the hard predicate returns False."""
    assert not derive_blocked_from_download(
        scan_status=ModerationVerdict.PENDING,
        moderation_state=None,
    )


def test_not_blocked_when_suspicious() -> None:
    """SUSPICIOUS is also soft; doesn't hard-block by itself."""
    assert not derive_blocked_from_download(
        scan_status=ModerationVerdict.SUSPICIOUS,
        moderation_state=None,
    )


def test_not_blocked_when_approved_overrides_malicious() -> None:
    # If manual approval is in, the rollup should already have set
    # scan_status to CLEAN; the predicate is purely for the derived
    # gate so we expect approved + clean -> not blocked.
    assert not derive_blocked_from_download(
        scan_status=ModerationVerdict.CLEAN,
        moderation_state=ModerationState.APPROVED,
    )


# ---------------------------------------------------------------------------
# derive_reasons


def test_reasons_empty_when_clean_no_moderation() -> None:
    reasons = derive_reasons(
        scan_status=ModerationVerdict.CLEAN,
        moderation_state=None,
        static_scan_verdict=None,
    )
    assert reasons == ()


def test_reasons_include_manual_approved_and_scan_clean() -> None:
    reasons = derive_reasons(
        scan_status=ModerationVerdict.CLEAN,
        moderation_state=ModerationState.APPROVED,
        static_scan_verdict=None,
    )
    assert "manual:approved" in reasons
    assert "scan:clean" in reasons


def test_reasons_include_scan_malicious_only_when_not_clean() -> None:
    reasons = derive_reasons(
        scan_status=ModerationVerdict.MALICIOUS,
        moderation_state=None,
        static_scan_verdict=None,
    )
    assert "scan:malicious" in reasons


def test_reasons_include_static_when_malicious() -> None:
    reasons = derive_reasons(
        scan_status=ModerationVerdict.MALICIOUS,
        moderation_state=None,
        static_scan_verdict=ModerationVerdict.MALICIOUS,
    )
    assert "static:malicious" in reasons


def test_reasons_include_reports_count() -> None:
    reasons = derive_reasons(
        scan_status=ModerationVerdict.SUSPICIOUS,
        moderation_state=None,
        static_scan_verdict=None,
        report_count=5,
    )
    assert "reports:5" in reasons


def test_reasons_include_vt_stale() -> None:
    reasons = derive_reasons(
        scan_status=ModerationVerdict.SUSPICIOUS,
        moderation_state=None,
        static_scan_verdict=None,
        vt_stale=True,
    )
    assert "vt:stale" in reasons


def test_reasons_include_pending() -> None:
    reasons = derive_reasons(
        scan_status=ModerationVerdict.PENDING,
        moderation_state=None,
        static_scan_verdict=None,
        pending=True,
    )
    assert "scan:pending" in reasons


def test_reasons_deduplicated_and_sorted() -> None:
    reasons = derive_reasons(
        scan_status=ModerationVerdict.MALICIOUS,
        moderation_state=ModerationState.QUARANTINED,
        static_scan_verdict=ModerationVerdict.MALICIOUS,
        extras=("manual:quarantined", "static:malicious"),  # duplicates
    )
    # Dedup should leave one of each.
    assert reasons.count("manual:quarantined") == 1
    assert reasons.count("static:malicious") == 1
    # Sorted ascending case-insensitively.
    assert list(reasons) == sorted(reasons, key=str.casefold)


# ---------------------------------------------------------------------------
# is_pending_state


def test_pending_state_completed_returns_false() -> None:
    inputs = ScanInputs(content_sha256="abc")
    assert not is_pending_state(inputs, scan_completed=True)


def test_pending_state_hash_only_returns_true() -> None:
    inputs = ScanInputs(content_sha256="abc")
    assert is_pending_state(inputs, scan_completed=False)


def test_pending_state_some_engines_missing_returns_true() -> None:
    inputs = ScanInputs(
        static_scan_verdict=ModerationVerdict.CLEAN,
        # llm_verdict, verification_verdict are None
    )
    assert is_pending_state(inputs, scan_completed=False)


# ---------------------------------------------------------------------------
# build_trust_envelope


def test_build_envelope_clean() -> None:
    pkg = PackageRef(name="@user/example", family=PackageFamily.SKILL)
    release = ReleaseRef(version="1.0.0", artifact_kind=ArtifactKind.LOCAL_PATH)
    envelope = build_trust_envelope(
        package=pkg,
        release=release,
        inputs=ScanInputs(),
        engine_version="u1.0.0",
    )
    assert envelope.trust.scan_status is ModerationVerdict.NOT_RUN
    assert not envelope.trust.blocked_from_download
    assert envelope.trust.engine_version == "u1.0.0"
    assert envelope.trust.evaluated_at is not None


def test_build_envelope_malicious_marks_blocked() -> None:
    pkg = PackageRef(name="malware-pkg")
    release = ReleaseRef(version="0.1.0")
    envelope = build_trust_envelope(
        package=pkg,
        release=release,
        inputs=ScanInputs(static_scan_verdict=ModerationVerdict.MALICIOUS),
    )
    assert envelope.trust.scan_status is ModerationVerdict.MALICIOUS
    assert envelope.trust.blocked_from_download
    assert "scan:malicious" in envelope.trust.reasons
    assert "static:malicious" in envelope.trust.reasons


def test_build_envelope_quarantined_marks_blocked() -> None:
    pkg = PackageRef(name="quarantined-pkg")
    release = ReleaseRef(version="1.0.0")
    envelope = build_trust_envelope(
        package=pkg,
        release=release,
        inputs=ScanInputs(manual_moderation=ModerationState.QUARANTINED),
    )
    assert envelope.trust.blocked_from_download
    assert envelope.trust.moderation_state is ModerationState.QUARANTINED


def test_build_envelope_pending_when_scan_incomplete() -> None:
    pkg = PackageRef(name="in-flight-pkg")
    release = ReleaseRef(version="1.0.0")
    envelope = build_trust_envelope(
        package=pkg,
        release=release,
        inputs=ScanInputs(content_sha256="hash-here"),
        scan_completed=False,
    )
    assert envelope.trust.pending
    assert envelope.trust.scan_status is ModerationVerdict.PENDING
    assert not envelope.trust.blocked_from_download


def test_build_envelope_with_extras() -> None:
    pkg = PackageRef(name="example")
    release = ReleaseRef(version="1.0.0")
    envelope = build_trust_envelope(
        package=pkg,
        release=release,
        inputs=ScanInputs(),
        extras=("provider:rate_limited",),
    )
    assert "provider:rate_limited" in envelope.trust.reasons


# ---------------------------------------------------------------------------
# refuse_if_blocked


def test_refuse_allow_when_clean() -> None:
    envelope = build_trust_envelope(
        package=PackageRef(name="clean-pkg"),
        release=ReleaseRef(version="1.0.0"),
        inputs=ScanInputs(),
    )
    blocked, reasons = refuse_if_blocked(envelope)
    assert not blocked
    assert isinstance(reasons, tuple)


def test_refuse_blocks_when_malicious() -> None:
    envelope = build_trust_envelope(
        package=PackageRef(name="malware-pkg"),
        release=ReleaseRef(version="1.0.0"),
        inputs=ScanInputs(static_scan_verdict=ModerationVerdict.MALICIOUS),
    )
    blocked, reasons = refuse_if_blocked(envelope)
    assert blocked
    assert "scan:malicious" in reasons


def test_refuse_blocks_when_stale_unless_allowed() -> None:
    envelope = build_trust_envelope(
        package=PackageRef(name="example"),
        release=ReleaseRef(version="1.0.0"),
        inputs=ScanInputs(),
        vt_stale=True,
    )
    blocked, _ = refuse_if_blocked(envelope, allow_stale=False)
    assert blocked
    ok, _ = refuse_if_blocked(envelope, allow_stale=True)
    assert not ok


def test_refuse_blocks_when_pending_unless_allowed() -> None:
    envelope = build_trust_envelope(
        package=PackageRef(name="example"),
        release=ReleaseRef(version="1.0.0"),
        inputs=ScanInputs(content_sha256="hash"),
        scan_completed=False,
    )
    blocked, _ = refuse_if_blocked(envelope, allow_pending=False)
    assert blocked
    ok, _ = refuse_if_blocked(envelope, allow_pending=True)
    assert not ok


def test_refuse_hard_block_not_overridable_by_allow_flags() -> None:
    envelope = build_trust_envelope(
        package=PackageRef(name="malware-pkg"),
        release=ReleaseRef(version="1.0.0"),
        inputs=ScanInputs(manual_moderation=ModerationState.REVOKED),
    )
    blocked, _ = refuse_if_blocked(envelope, allow_stale=True, allow_pending=True)
    assert blocked


# ---------------------------------------------------------------------------
# T9 version-exact contract


def test_validate_version_exact_request_concrete() -> None:
    validate_version_exact_request(
        VersionExactRequest(package_name="@user/example", resolved_version="1.0.0")
    )


def test_validate_version_exact_rejects_latest() -> None:
    with pytest.raises(VersionExactViolation):
        validate_version_exact_request(
            VersionExactRequest(package_name="@user/example", resolved_version="latest")
        )


def test_validate_version_exact_rejects_empty_version() -> None:
    with pytest.raises(VersionExactViolation):
        validate_version_exact_request(
            VersionExactRequest(package_name="@user/example", resolved_version="")
        )


def test_validate_version_exact_rejects_empty_package() -> None:
    with pytest.raises(VersionExactViolation):
        validate_version_exact_request(
            VersionExactRequest(package_name="", resolved_version="1.0.0")
        )


def test_validate_version_exact_rejects_wildcard() -> None:
    for token in ("*", "main", "master", "HEAD", "next", "edge"):
        with pytest.raises(VersionExactViolation):
            validate_version_exact_request(
                VersionExactRequest(
                    package_name="@user/example", resolved_version=token
                )
            )


def test_validate_version_exact_case_insensitive_token_check() -> None:
    with pytest.raises(VersionExactViolation):
        validate_version_exact_request(
            VersionExactRequest(
                package_name="@user/example", resolved_version="LATEST"
            )
        )


def test_validate_version_exact_accepts_sha_like() -> None:
    # 40-char hex shas are fine.
    validate_version_exact_request(
        VersionExactRequest(
            package_name="@user/example",
            resolved_version="abcdef1234567890" * 2 + "abcdef12",
        )
    )


def test_fetch_for_version_calls_fetcher_with_validated_args() -> None:
    captured: dict = {}

    def fetcher(name: str, version: str) -> TrustEnvelope:
        captured["name"] = name
        captured["version"] = version
        return build_trust_envelope(
            package=PackageRef(name=name),
            release=ReleaseRef(version=version),
            inputs=ScanInputs(),
        )

    request = VersionExactRequest(package_name="@user/example", resolved_version="2.0.0 ")
    envelope = fetch_for_version(request, fetcher=fetcher)
    assert captured == {"name": "@user/example", "version": "2.0.0"}
    assert envelope.package.name == "@user/example"
    assert envelope.release.version == "2.0.0"


def test_fetch_for_version_validates_before_call() -> None:
    """Fetcher is NOT called when the request fails validation."""
    called = False

    def fetcher(name: str, version: str) -> TrustEnvelope:
        nonlocal called
        called = True
        raise AssertionError("must not be called")

    request = VersionExactRequest(
        package_name="@user/example", resolved_version="latest"
    )
    with pytest.raises(VersionExactViolation):
        fetch_for_version(request, fetcher=fetcher)
    assert called is False


# ---------------------------------------------------------------------------
# make_local_path_envelope


def test_make_local_path_envelope_clean() -> None:
    envelope = make_local_path_envelope(package_name="local-skill")
    assert envelope.package.name == "local-skill"
    assert envelope.release.artifact_kind is ArtifactKind.LOCAL_PATH
    assert not envelope.trust.blocked_from_download


def test_make_local_path_envelope_with_scan_codes() -> None:
    envelope = make_local_path_envelope(
        package_name="local-skill",
        scan_codes=[REASON_CODES["DANGEROUS_EXEC"]],
    )
    assert envelope.trust.scan_status is ModerationVerdict.SUSPICIOUS


def test_make_local_path_envelope_with_fingerprint() -> None:
    envelope = make_local_path_envelope(
        package_name="local-skill", fingerprint="abc123"
    )
    assert envelope.release.artifact_sha256 == "abc123"


def test_make_local_path_envelope_quarantine_blocks() -> None:
    envelope = make_local_path_envelope(
        package_name="local-skill",
        moderation_state=ModerationState.QUARANTINED,
    )
    assert envelope.trust.blocked_from_download
