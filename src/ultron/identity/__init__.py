"""Stable-identity infrastructure: alias graph, slug validation, reservation.

T6 (openclaw-clawhub catalog port; see ``THIRD_PARTY_NOTICES.md``).
Generalises the marketplace's slug-rename / merge / transfer +
30-day reservation pattern into a primitive any ultron subsystem
exposing a user-facing namespace can reuse (skill registry, voice
intent labels, sandbox project names, gaming-mode profile names,
persona overlays, voicepack ids, memory backend selectors).

Public surface lives in :mod:`ultron.identity.alias_graph`:

* :class:`AliasGraphEntry` — one alias graph node (canonical slug
  with redirects + reservation window).
* :class:`AliasGraph` — the in-memory + persistent alias graph;
  resolve / rename / merge / transfer / soft_delete / hard_delete
  operations.
* :func:`validate_slug` / :func:`normalize_slug` — slug shape rules.
* :data:`RESERVED_SLUGS` — names that can never be claimed.
"""

from ultron.identity.alias_graph import (
    DEFAULT_RESERVATION_DAYS,
    RESERVED_SLUGS,
    SLUG_PATTERN,
    AliasGraph,
    AliasGraphEntry,
    AliasOperation,
    AliasResolveError,
    InvalidSlugError,
    SlugReservedError,
    normalize_slug,
    validate_slug,
)
from ultron.identity.short_lived_token import (
    ALGORITHM_HS256,
    DEFAULT_CLOCK_SKEW_SECONDS,
    DEFAULT_ISSUER,
    DEFAULT_TTL_SECONDS,
    MAX_TTL_SECONDS,
    TokenError,
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

__all__ = [
    "ALGORITHM_HS256",
    "AliasGraph",
    "AliasGraphEntry",
    "AliasOperation",
    "AliasResolveError",
    "DEFAULT_CLOCK_SKEW_SECONDS",
    "DEFAULT_ISSUER",
    "DEFAULT_RESERVATION_DAYS",
    "DEFAULT_TTL_SECONDS",
    "InvalidSlugError",
    "MAX_TTL_SECONDS",
    "RESERVED_SLUGS",
    "SLUG_PATTERN",
    "SlugReservedError",
    "TokenError",
    "TokenExpiredError",
    "TokenMintError",
    "TokenSignatureError",
    "TokenVerifyError",
    "TrustedCaller",
    "TrustedCallerClaimMismatch",
    "TrustedCallerNotFoundError",
    "VerifiedClaims",
    "list_trusted_callers",
    "load_trusted_caller",
    "mint_token",
    "normalize_slug",
    "register_trusted_caller",
    "rotate_secret",
    "validate_slug",
    "verify_audit_chain",
    "verify_token",
]
