"""Per-version trust envelope with derived `blocked` signal (T1) +
version-exact install contract (T9).

T1 + T9 (openclaw-clawhub catalog port; see ``THIRD_PARTY_NOTICES.md``).
A single typed envelope that consolidates the verdict-input signals
the scanner + moderation + report-queue all produce, plus a derived
``blocked_from_download`` boolean that downstream install code MUST
consult before materialising bytes onto disk.

The envelope is generalised beyond install-time scanning into a
universal "should I use this subsystem right now?" decision for any
ultron subsystem whose health is multi-source: install (the original
catalog use case), web-search providers, MCP servers, memory backend,
VLM / STT / TTS engines, and gaming-mode lifecycle transitions.

Three architectural pieces:

1. **Shape (T1).** A frozen dataclass :class:`TrustEnvelope` carrying
   ``package``, ``release``, ``trust`` blocks. The ``trust`` block has
   six fields: ``scan_status`` (re-used :class:`ModerationVerdict`
   from :mod:`ultron.install.reason_codes`), ``moderation_state``
   (optional :class:`ModerationState` enum -- approved / quarantined
   / revoked), ``blocked_from_download`` (the derived install gate),
   ``reasons`` (tuple of stable prefix-namespaced strings), ``pending``
   (one or more trust inputs still being computed), ``stale`` (verdict
   computed from outdated inputs; high-confidence decisions should
   wait for refresh).

2. **Derivation.** :func:`derive_scan_status` implements the
   short-circuit hierarchy verbatim:

   1. Manual moderation ``"approved"`` -> CLEAN
   2. Manual moderation ``"quarantined"`` or ``"revoked"`` -> MALICIOUS
   3. LLM verdict (malicious / suspicious / clean) -> that verdict
   4. Static-scan ``"malicious"`` -> MALICIOUS (without verification override)
   5. Verification + static both suspicious -> verification voided
   6. Verification ``"malicious"`` -> MALICIOUS
   7. Verification ``"suspicious"`` -> SUSPICIOUS
   8. Verification ``"clean"`` + trusted plugin -> CLEAN
   9. Verification present and not ``"not_run"`` -> that verdict
   10. ``content_sha256`` hash exists -> PENDING (scan in-flight)
   11. Otherwise -> NOT_RUN

   :func:`derive_blocked_from_download` returns ``True`` iff
   (a) manual moderation is quarantined / revoked, OR
   (b) scan_status is MALICIOUS.

   :func:`derive_reasons` produces the deduplicated prefixed-code
   list (``manual:approved``, ``scan:malicious``, ``static:malicious``,
   ``reports:N``, ``vt:stale``, etc.).

3. **Version-exact discipline (T9).** :class:`VersionExactRequest` +
   :func:`fetch_for_version` enforce the documented contract: clients
   resolve the target version first, THEN fetch the trust envelope
   for that specific version (never reuse a stale "latest" envelope).
   :func:`refuse_if_blocked` is the single decision point install
   code routes through; it returns either a ``(False, ())`` allow or
   ``(True, (reason, ...))`` refuse pair so the caller can render
   audit-log rows + voice narration without re-doing the derivation.

The TrustEnvelope is invariant by construction: callers cannot mark
an envelope as not-blocked when the inputs say block. The single
override path is the explicit ``allow_stale`` / ``allow_pending``
arguments to :func:`refuse_if_blocked` which clamp those soft
flags; the hard block (manual quarantined / revoked / scan malicious)
cannot be overridden through this surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional, Sequence

from ultron.install.reason_codes import (
    MALICIOUS_CODES,
    ModerationVerdict,
    StatusInputs,
    compute_status,
    normalize_reason_codes,
    verdict_from_codes,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums + constants


class ModerationState(str, Enum):
    """Manual-moderator verdict on a release.

    Mirrors the upstream catalogue:

    * :attr:`APPROVED` — explicitly cleared by a moderator. Forces
      scan_status to CLEAN regardless of other inputs.
    * :attr:`QUARANTINED` — hidden + non-installable but not removed.
      Forces blocked_from_download. May be lifted later.
    * :attr:`REVOKED` — permanently removed. Forces
      blocked_from_download. Never lifted.

    ``None`` (absent) means "no manual decision has been made yet";
    scan_status defaults to the underlying engine inputs.
    """

    APPROVED = "approved"
    QUARANTINED = "quarantined"
    REVOKED = "revoked"


class ArtifactKind(str, Enum):
    """Discriminator for the source-bytes shape of a release.

    Extends the upstream npm-pack / legacy-zip dichotomy with the
    ultron-specific shapes the marketplace primitive already supports
    (LOCAL_PATH for development, TARBALL_URL for arbitrary HTTP fetch,
    GIT_REF for git clone-by-ref, INLINE_MARKDOWN for trigger-loaded
    skills shipped as bare .md files).
    """

    NPM_PACK = "npm-pack"
    LEGACY_ZIP = "legacy-zip"
    LOCAL_PATH = "local-path"
    TARBALL_URL = "tarball-url"
    GIT_REF = "git-ref"
    INLINE_MARKDOWN = "inline-markdown"


class PackageFamily(str, Enum):
    """Release family. Matches the upstream taxonomy."""

    CODE_PLUGIN = "code-plugin"
    BUNDLE_PLUGIN = "bundle-plugin"
    SKILL = "skill"
    VOICEPACK = "voicepack"  # ultron extension
    MODEL = "model"          # ultron extension (LLM / VLM weights)


# Reason-code prefixes (API contracts; preserved verbatim from
# upstream for cross-system audit cross-ref).
REASON_PREFIX_MANUAL: str = "manual"
REASON_PREFIX_SCAN: str = "scan"
REASON_PREFIX_STATIC: str = "static"
REASON_PREFIX_REPORTS: str = "reports"
REASON_PREFIX_VT: str = "vt"
REASON_PREFIX_PACKAGE: str = "package"
REASON_PREFIX_VERIFICATION: str = "verification"


def reason_for_moderation(state: Optional[ModerationState]) -> Optional[str]:
    """Return the ``manual:<state>`` reason string or None."""
    if state is None:
        return None
    return f"{REASON_PREFIX_MANUAL}:{state.value}"


def reason_for_scan(status: ModerationVerdict) -> Optional[str]:
    """Return the ``scan:<status>`` reason string or None for not-run."""
    if status is ModerationVerdict.NOT_RUN:
        return None
    return f"{REASON_PREFIX_SCAN}:{status.value}"


def reason_for_static_malicious() -> str:
    """The ``static:malicious`` reason string (constant)."""
    return f"{REASON_PREFIX_STATIC}:{ModerationVerdict.MALICIOUS.value}"


def reason_for_reports(count: int) -> Optional[str]:
    """Return ``reports:<count>`` when ``count > 0``, else None."""
    if count <= 0:
        return None
    return f"{REASON_PREFIX_REPORTS}:{count}"


def reason_for_vt_stale() -> str:
    """The ``vt:stale`` reason string (constant)."""
    return f"{REASON_PREFIX_VT}:stale"


def reason_for_pending() -> str:
    """The ``scan:pending`` reason string (constant)."""
    return f"{REASON_PREFIX_SCAN}:{ModerationVerdict.PENDING.value}"


# ---------------------------------------------------------------------------
# Envelope structs


@dataclass(frozen=True)
class PackageRef:
    """The ``package`` block of a :class:`TrustEnvelope`."""

    name: str
    display_name: str = ""
    family: PackageFamily = PackageFamily.SKILL


@dataclass(frozen=True)
class ReleaseRef:
    """The ``release`` block of a :class:`TrustEnvelope`."""

    release_id: str = ""
    version: str = ""
    artifact_kind: ArtifactKind = ArtifactKind.LOCAL_PATH
    artifact_sha256: Optional[str] = None
    npm_integrity: Optional[str] = None
    npm_shasum: Optional[str] = None
    npm_tarball_name: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class TrustSignal:
    """The ``trust`` block of a :class:`TrustEnvelope`.

    Fields:
        scan_status: rolled-up verdict over all scan inputs (re-uses
            :class:`ModerationVerdict` so callers get the same value
            set as the T3 reason-code module).
        moderation_state: manual moderator decision (None when no
            human has reviewed).
        blocked_from_download: derived gate -- install code refuses
            when True.
        reasons: deduplicated tuple of prefix-namespaced reason
            strings (``manual:quarantined``, ``scan:malicious``,
            ``reports:3``, ...).
        pending: at least one trust input is still being computed.
            High-confidence decisions should defer.
        stale: verdict computed from outdated inputs. Used to flag
            "we have a verdict but you should refresh before relying
            on it".
        engine_version: scan-engine version that produced the
            verdict (mirrors :data:`MODERATION_ENGINE_VERSION` from
            :mod:`ultron.install.reason_codes`). Optional for
            non-scanner sources.
        evaluated_at: timestamp the envelope was assembled.
    """

    scan_status: ModerationVerdict = ModerationVerdict.NOT_RUN
    moderation_state: Optional[ModerationState] = None
    blocked_from_download: bool = False
    reasons: tuple[str, ...] = ()
    pending: bool = False
    stale: bool = False
    engine_version: str = ""
    evaluated_at: Optional[datetime] = None


@dataclass(frozen=True)
class TrustEnvelope:
    """Full per-version envelope.

    Construction is via :func:`build_trust_envelope`; callers don't
    normally build the envelope directly so the invariants
    (blocked_from_download is consistent with scan_status +
    moderation_state, reasons are deduplicated) are preserved by
    construction.

    The envelope is **generalised** -- consumers in ``web_search``,
    ``mcp``, ``memory``, ``vlm/stt/tts engine lifecycle`` can build a
    TrustEnvelope describing their subsystem's health and route the
    "should I use this right now?" decision through
    :func:`refuse_if_blocked`.
    """

    package: PackageRef
    release: ReleaseRef
    trust: TrustSignal


# ---------------------------------------------------------------------------
# Derivation logic (the T1 algorithm)


@dataclass(frozen=True)
class ScanInputs:
    """Raw inputs to :func:`derive_scan_status`.

    All fields optional / nullable so callers can populate whichever
    engines actually ran. Missing inputs are treated as "this engine
    didn't report" rather than "this engine returned clean".
    """

    manual_moderation: Optional[ModerationState] = None
    llm_verdict: Optional[ModerationVerdict] = None
    static_scan_verdict: Optional[ModerationVerdict] = None
    verification_verdict: Optional[ModerationVerdict] = None
    trusted_openclaw_plugin: bool = False
    content_sha256: Optional[str] = None
    extra_codes: tuple[str, ...] = ()


def derive_scan_status(inputs: ScanInputs) -> ModerationVerdict:
    """Run the short-circuit hierarchy returning the effective verdict.

    Mirrors the upstream algorithm step-for-step (catalog T1
    derivation tree):

    1. Manual moderation APPROVED -> CLEAN.
    2. Manual moderation QUARANTINED / REVOKED -> MALICIOUS.
    3. LLM verdict if set -> that.
    4. Static-scan MALICIOUS -> MALICIOUS.
    5. Verification + static both SUSPICIOUS -> verification voided.
    6. Verification MALICIOUS -> MALICIOUS.
    7. Verification SUSPICIOUS -> SUSPICIOUS.
    8. Verification CLEAN + trusted plugin -> CLEAN.
    9. Verification not None and not NOT_RUN -> that.
    10. content_sha256 present -> PENDING (scan in flight).
    11. Otherwise -> NOT_RUN.

    extra_codes contribute via :func:`verdict_from_codes` as the
    final fallback (so callers can supply post-hoc findings).
    """
    if inputs.manual_moderation is ModerationState.APPROVED:
        return ModerationVerdict.CLEAN
    if inputs.manual_moderation in (
        ModerationState.QUARANTINED,
        ModerationState.REVOKED,
    ):
        return ModerationVerdict.MALICIOUS
    if inputs.llm_verdict is not None and inputs.llm_verdict is not ModerationVerdict.NOT_RUN:
        return inputs.llm_verdict
    if inputs.static_scan_verdict is ModerationVerdict.MALICIOUS:
        return ModerationVerdict.MALICIOUS

    # Verification + static both suspicious -> verification voided.
    verification = inputs.verification_verdict
    if (
        verification is ModerationVerdict.SUSPICIOUS
        and inputs.static_scan_verdict is ModerationVerdict.SUSPICIOUS
    ):
        verification = None

    if verification is ModerationVerdict.MALICIOUS:
        return ModerationVerdict.MALICIOUS
    if verification is ModerationVerdict.SUSPICIOUS:
        return ModerationVerdict.SUSPICIOUS
    if verification is ModerationVerdict.CLEAN and inputs.trusted_openclaw_plugin:
        return ModerationVerdict.CLEAN
    if verification is not None and verification is not ModerationVerdict.NOT_RUN:
        return verification

    # extra_codes fallback before resorting to the hash-only PENDING heuristic.
    if inputs.extra_codes:
        extras_verdict = verdict_from_codes(inputs.extra_codes)
        if extras_verdict is not ModerationVerdict.CLEAN:
            return extras_verdict

    if inputs.content_sha256:
        return ModerationVerdict.PENDING

    return ModerationVerdict.NOT_RUN


def derive_blocked_from_download(
    *,
    scan_status: ModerationVerdict,
    moderation_state: Optional[ModerationState],
) -> bool:
    """Return True iff the release MUST NOT be installed right now.

    Block conditions (OR):

    1. ``moderation_state`` is :attr:`ModerationState.QUARANTINED`.
    2. ``moderation_state`` is :attr:`ModerationState.REVOKED`.
    3. ``scan_status`` is :attr:`ModerationVerdict.MALICIOUS`.

    All other combinations return False. ``PENDING`` and ``SUSPICIOUS``
    don't block by themselves (callers may treat them as soft blocks
    via :func:`refuse_if_blocked` ``allow_pending`` / ``allow_stale``
    arguments).
    """
    if moderation_state in (ModerationState.QUARANTINED, ModerationState.REVOKED):
        return True
    if scan_status is ModerationVerdict.MALICIOUS:
        return True
    return False


def derive_reasons(
    *,
    scan_status: ModerationVerdict,
    moderation_state: Optional[ModerationState],
    static_scan_verdict: Optional[ModerationVerdict],
    report_count: int = 0,
    vt_stale: bool = False,
    pending: bool = False,
    extras: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the prefixed reason-code tuple for the trust envelope.

    Algorithm:

    1. ``manual:<state>`` when moderation_state is set.
    2. ``scan:<status>`` when scan_status is not CLEAN / NOT_RUN, or
       when scan is CLEAN AND a manual decision is also recorded
       (mirrors the upstream "scan:clean only when manually
       approved" carve-out).
    3. ``static:malicious`` when the static scan verdict is MALICIOUS
       (and not already implied by scan_status being MALICIOUS).
    4. ``reports:<N>`` when report_count > 0.
    5. ``vt:stale`` when ``vt_stale=True``.
    6. ``scan:pending`` when ``pending=True`` (in addition to whatever
       scan_status renders).
    7. ``extras`` are appended as-is so callers can carry domain-
       specific reasons (``provider:rate_limited``, ``mcp:disconnected``).

    Output is deduplicated + sorted (case-insensitive).
    """
    collected: list[str] = []

    manual_reason = reason_for_moderation(moderation_state)
    if manual_reason is not None:
        collected.append(manual_reason)

    if scan_status is ModerationVerdict.CLEAN and moderation_state is ModerationState.APPROVED:
        collected.append(reason_for_scan(scan_status))  # type: ignore[arg-type]
    elif scan_status not in (ModerationVerdict.CLEAN, ModerationVerdict.NOT_RUN):
        scan_reason = reason_for_scan(scan_status)
        if scan_reason is not None:
            collected.append(scan_reason)

    if (
        static_scan_verdict is ModerationVerdict.MALICIOUS
        and scan_status is not ModerationVerdict.MALICIOUS
    ):
        collected.append(reason_for_static_malicious())
    elif static_scan_verdict is ModerationVerdict.MALICIOUS:
        # Even when scan_status is also MALICIOUS, surface the
        # static signal so audit consumers know which engine fired.
        collected.append(reason_for_static_malicious())

    reports_reason = reason_for_reports(report_count)
    if reports_reason is not None:
        collected.append(reports_reason)

    if vt_stale:
        collected.append(reason_for_vt_stale())

    if pending:
        collected.append(reason_for_pending())

    for extra in extras:
        if extra and isinstance(extra, str):
            collected.append(extra.strip())

    # Dedup + sort. normalize_reason_codes preserves case-fold ordering.
    seen: dict[str, None] = {}
    for value in collected:
        if not value:
            continue
        seen[value] = None
    return tuple(sorted(seen.keys(), key=str.casefold))


def is_pending_state(inputs: ScanInputs, *, scan_completed: bool = True) -> bool:
    """Return True iff the inputs indicate scan-in-flight.

    Used by :func:`build_trust_envelope` to set the ``pending`` flag.
    """
    if scan_completed:
        return False
    return bool(inputs.content_sha256) or any(
        v is None
        for v in (
            inputs.static_scan_verdict,
            inputs.llm_verdict,
            inputs.verification_verdict,
        )
    )


def build_trust_envelope(
    *,
    package: PackageRef,
    release: ReleaseRef,
    inputs: ScanInputs,
    report_count: int = 0,
    vt_stale: bool = False,
    scan_completed: bool = True,
    engine_version: str = "",
    evaluated_at: Optional[datetime] = None,
    extras: Iterable[str] = (),
) -> TrustEnvelope:
    """Build a :class:`TrustEnvelope` with derived fields populated.

    The single public construction surface; downstream call sites
    don't build TrustSignal / TrustEnvelope directly so the
    invariants (blocked_from_download consistent with scan_status +
    moderation_state, reasons deduplicated) are preserved.
    """
    scan_status = derive_scan_status(inputs)
    pending = is_pending_state(inputs, scan_completed=scan_completed)
    if pending and scan_status is ModerationVerdict.NOT_RUN:
        scan_status = ModerationVerdict.PENDING
    blocked = derive_blocked_from_download(
        scan_status=scan_status,
        moderation_state=inputs.manual_moderation,
    )
    reasons = derive_reasons(
        scan_status=scan_status,
        moderation_state=inputs.manual_moderation,
        static_scan_verdict=inputs.static_scan_verdict,
        report_count=report_count,
        vt_stale=vt_stale,
        pending=pending,
        extras=extras,
    )
    trust = TrustSignal(
        scan_status=scan_status,
        moderation_state=inputs.manual_moderation,
        blocked_from_download=blocked,
        reasons=reasons,
        pending=pending,
        stale=vt_stale,
        engine_version=engine_version,
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )
    return TrustEnvelope(package=package, release=release, trust=trust)


# ---------------------------------------------------------------------------
# Install decision surface


def refuse_if_blocked(
    envelope: TrustEnvelope,
    *,
    allow_stale: bool = False,
    allow_pending: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Return ``(blocked, reasons)`` for an install decision against ``envelope``.

    Block conditions (hard; cannot be overridden by the optional args):

    * Manual moderation QUARANTINED or REVOKED.
    * Scan status MALICIOUS.

    Soft block conditions (cleared by the optional args):

    * ``stale=True`` AND ``allow_stale=False`` -> block.
    * ``pending=True`` AND ``allow_pending=False`` AND scan status is
      PENDING -> block.

    A False return means the install can proceed; the returned
    reasons tuple is always the envelope's reasons (caller can log
    them for audit even when not blocking).
    """
    if envelope.trust.blocked_from_download:
        return (True, envelope.trust.reasons)
    if envelope.trust.stale and not allow_stale:
        return (True, envelope.trust.reasons + (f"{REASON_PREFIX_VT}:stale-refuse",))
    if (
        envelope.trust.pending
        and not allow_pending
        and envelope.trust.scan_status is ModerationVerdict.PENDING
    ):
        return (True, envelope.trust.reasons + (f"{REASON_PREFIX_SCAN}:pending-refuse",))
    return (False, envelope.trust.reasons)


# ---------------------------------------------------------------------------
# T9 version-exact contract


class VersionExactViolation(RuntimeError):
    """Raised when the version-exact contract is violated.

    Carries the offending request shape so audit consumers can show
    what went wrong without re-parsing the message.
    """

    def __init__(self, request: "VersionExactRequest", reason: str) -> None:
        super().__init__(
            f"Version-exact violation for {request.package_name!r}: {reason}"
        )
        self.request = request
        self.reason = reason


@dataclass(frozen=True)
class VersionExactRequest:
    """One trust-envelope lookup request.

    Fields:
        package_name: canonical package name (e.g. ``@owner/skill``).
        resolved_version: the EXACT version string about to be
            installed. ``"latest"`` / ``"*"`` / a tag are NOT
            valid here -- callers must resolve first.
        purpose: human-readable purpose string for audit logs
            (``"install"`` / ``"update"`` / ``"reload"``).
        actor: who's requesting (voice user / coding agent / CLI / etc.).
    """

    package_name: str
    resolved_version: str
    purpose: str = "install"
    actor: str = ""


#: Version strings the version-exact contract explicitly rejects.
_DISALLOWED_VERSION_TOKENS: frozenset[str] = frozenset({
    "",
    "latest",
    "*",
    "next",
    "main",
    "master",
    "head",
    "dev",
    "edge",
    "canary",
    "any",
})


def validate_version_exact_request(request: VersionExactRequest) -> None:
    """Raise :class:`VersionExactViolation` if ``request`` looks pre-resolution.

    The version string must be a concrete identifier -- semver,
    commit SHA, immutable tag, etc. Tag-like or floating identifiers
    raise so callers see the contract violation immediately rather
    than silently caching a stale "latest" envelope.
    """
    if not request.package_name or not request.package_name.strip():
        raise VersionExactViolation(request, "package_name is empty")
    version = request.resolved_version.strip()
    if not version:
        raise VersionExactViolation(request, "resolved_version is empty")
    if version.casefold() in _DISALLOWED_VERSION_TOKENS:
        raise VersionExactViolation(
            request,
            f"resolved_version {version!r} is a floating tag; resolve to a concrete version first",
        )


def fetch_for_version(
    request: VersionExactRequest,
    *,
    fetcher: Callable[[str, str], TrustEnvelope],
) -> TrustEnvelope:
    """Validate ``request`` then call ``fetcher(package_name, version)``.

    ``fetcher`` is the IO-doing callable -- typically an HTTP client
    against ``/api/v1/packages/{name}/versions/{version}/security``,
    or a local-scanner shim for PATH / GIT sources. Network IO is
    INJECTED so callers can compose this with provider chains,
    circuit breakers, and the T14 rate-limit tracker.

    Raises :class:`VersionExactViolation` BEFORE calling the fetcher
    when the request fails the contract.
    """
    validate_version_exact_request(request)
    return fetcher(request.package_name, request.resolved_version.strip())


def make_local_path_envelope(
    *,
    package_name: str,
    version: str = "",
    family: PackageFamily = PackageFamily.SKILL,
    scan_codes: Sequence[str] = (),
    moderation_state: Optional[ModerationState] = None,
    report_count: int = 0,
    fingerprint: Optional[str] = None,
) -> TrustEnvelope:
    """Build a trust envelope for a local PATH source.

    The local-scan path doesn't go through the upstream registry;
    this helper composes :func:`build_trust_envelope` with the
    pieces a local-path consumer typically has on hand (the scan
    findings + an optional fingerprint for the release identity).
    Used by :mod:`ultron.skills.marketplace` when building install
    plans from PATH / GIT / GIT_SUBDIR sources.
    """
    inputs = ScanInputs(
        manual_moderation=moderation_state,
        static_scan_verdict=verdict_from_codes(scan_codes) if scan_codes else None,
        extra_codes=tuple(scan_codes),
    )
    pkg = PackageRef(name=package_name, family=family)
    release = ReleaseRef(
        version=version,
        artifact_kind=ArtifactKind.LOCAL_PATH,
        artifact_sha256=fingerprint,
    )
    return build_trust_envelope(
        package=pkg,
        release=release,
        inputs=inputs,
        report_count=report_count,
        scan_completed=True,
    )


__all__ = [
    "ModerationState",
    "ArtifactKind",
    "PackageFamily",
    "REASON_PREFIX_MANUAL",
    "REASON_PREFIX_SCAN",
    "REASON_PREFIX_STATIC",
    "REASON_PREFIX_REPORTS",
    "REASON_PREFIX_VT",
    "REASON_PREFIX_PACKAGE",
    "REASON_PREFIX_VERIFICATION",
    "reason_for_moderation",
    "reason_for_scan",
    "reason_for_static_malicious",
    "reason_for_reports",
    "reason_for_vt_stale",
    "reason_for_pending",
    "PackageRef",
    "ReleaseRef",
    "TrustSignal",
    "TrustEnvelope",
    "ScanInputs",
    "derive_scan_status",
    "derive_blocked_from_download",
    "derive_reasons",
    "is_pending_state",
    "build_trust_envelope",
    "refuse_if_blocked",
    "VersionExactRequest",
    "VersionExactViolation",
    "validate_version_exact_request",
    "fetch_for_version",
    "make_local_path_envelope",
]
