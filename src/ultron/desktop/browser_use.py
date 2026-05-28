"""``browser-use`` CLI wrapper -- CDP-backed browser automation tier.

Catalog 10 batch 1 (GREEN read foundation): T1 indexed state
enumeration, T2 DOM-native CSS/HTML/text/attribute/value/bbox
extraction, T5 wait-for-element/text synchronisation, T6 tab lifecycle
management. Plus the navigation helpers (``open`` / ``back`` /
``scroll`` / ``close``) needed to drive the read primitives.

Why a new tier on top of the existing :func:`ultron.desktop.uia.extract_browser_content`:

* UIA walks the accessibility tree -- fast, zero-GPU, but limited to
  what Windows exposes. It cannot query CSS selectors, execute JS,
  read cookies, or wait for DOM mutations.
* The ``browser-use`` CLI talks Chrome DevTools Protocol via Playwright.
  Indexed elements + CSS selectors + JS eval + cookie management +
  multi-session isolation -- all the things the UIA tier cannot do.
* Integration pattern (wired in batch 9): UIA stays the first tier in
  :func:`ultron.desktop.screen_context.build_screen_context`;
  ``browser-use`` slots in as a second tier when the UIA tree returns
  empty/sparse results; the Moondream2 VLM remains the third tier.

The plugin source under ``F:\\reference_repos\\quarantine\\plugins\\clawhub-browser-use``
is documentation-only (``SKILL.md`` + two recipe markdowns; no Python
source). This module wraps the documented public API of the external
``browser-use`` open-source CLI -- it does NOT import or vendor any
upstream code. See ``THIRD_PARTY_NOTICES.md`` for attribution.

Fail-open contract (matches every other ``desktop/`` module):

* When the CLI binary is missing OR the subprocess fails OR the daemon
  reports an error, every public method returns its result dataclass
  with ``success=False`` and ``error`` populated. Callers can treat
  every method as if it might no-op.
* No exception ever escapes a public method on the happy or sad path.
  Construction does not load anything; the binary is discovered lazily
  on first call via :func:`shutil.which`.

Security tiering for this batch (all GREEN per catalog 10):

* Read-only state enumeration (T1) -- no credential surface.
* Read-only extraction (T2) -- HTML / text / attributes / bbox / value.
  ``get_value`` can expose unmasked form-field values; password-type
  inputs are skipped by the upstream CLI but this module's caller
  should not log the result without filtering.
* Synchronisation (T5) -- pure blocking wait. No side effect.
* Tab lifecycle (T6) -- ``tab close`` is destructive but the operation
  is bounded to the daemon's own browser instance; no Cap-3 gate
  because the user must explicitly invoke this via voice intent.

Later batches (3-7) add YELLOW techniques (JS eval, cookies, session
isolation, profile connect, CDP passthrough) that require Cap-3 +
two-phase approval + static analysis gating. Batch 8 adds the
``BrowserSequenceRunner`` creative extension. Batch 9 wires this tier
into :mod:`ultron.desktop.screen_context`.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ultron.safety.path_resolver import PathResolver, get_path_resolver
from ultron.safety.two_phase_approval import (
    ApprovalHandle,
    ApprovalRegistry,
    ApprovalRequest,
    get_approval_registry,
)
from ultron.safety.validator import (
    RuleContext,
    Verdict,
    get_validator,
)
from ultron.utils.logging import get_logger

logger = get_logger("desktop.browser_use")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Binary names the upstream registers as aliases. Tried in order when
# no explicit ``binary_path`` is configured. ``bu`` is the shortest;
# ``browseruse`` is the no-hyphen variant some PATH conventions prefer.
BROWSER_USE_BINARY_CANDIDATES: tuple[str, ...] = (
    "browser-use",
    "bu",
    "browseruse",
)

# CREATE_NO_WINDOW on Windows suppresses the console flash that
# otherwise pops every time the CLI subprocess spawns. Matches the
# convention in :mod:`ultron.desktop.windows` and every subprocess
# site in :mod:`ultron.transcription.parakeet_engine`.
_CREATE_NO_WINDOW: int = 0x08000000 if sys.platform == "win32" else 0

# Default per-call subprocess wall-clock timeout. The upstream daemon
# documents ~50 ms per call when warm; cold-start is bounded by
# daemon-startup latency (~200-500 ms). 30 s headroom accommodates
# slow page-loads on ``open`` + ``wait_*`` commands.
DEFAULT_TIMEOUT_S: float = 30.0

# Default wait timeout for ``wait_selector`` / ``wait_text``. Matches
# the upstream CLI default of 30 s expressed in ms so the value passes
# straight to the ``--timeout`` flag without conversion.
DEFAULT_WAIT_TIMEOUT_MS: int = 30_000

# Allowed ``--state`` values for ``wait selector``. The upstream CLI
# documents these four; anything else is rejected at our boundary so
# typos surface as a clear error rather than an unhelpful CLI usage.
WAIT_SELECTOR_STATES: frozenset[str] = frozenset(
    {"visible", "hidden", "attached", "detached"}
)

# Allowed scroll directions. The upstream documents up/down; left/right
# are not exposed by the CLI surface we wrap so we reject them at our
# boundary.
SCROLL_DIRECTIONS: frozenset[str] = frozenset({"up", "down"})

# Environment variables that we strip from every subprocess call so
# ambient global state cannot silently change which session a call
# targets. ``BROWSER_USE_SESSION`` is the upstream env-var default for
# the session name; the catalog 10 "deliberately skip" list flags it
# explicitly because relying on it makes session boundaries unauditable.
_ENV_VARS_TO_SCRUB: tuple[str, ...] = (
    "BROWSER_USE_SESSION",
)

# Sentinel returned by ``state --json`` parsers when the CLI emitted
# non-JSON. Treated as a soft failure -- the raw text is preserved on
# the result so callers can fall back to substring matching.
_JSON_PARSE_FAILED: str = "__json_parse_failed__"

# Validator ``capability`` tag for every write-side method. Separates
# browser-use calls from the existing UIA + native input surfaces so
# per-capability rules can target them independently.
_VALIDATOR_CAPABILITY: str = "desktop_browser_use"

# Validator ``tool_name`` prefix. Audit log + dashboards group by this
# prefix when summarising browser-use activity.
_TOOL_NAME_PREFIX: str = "desktop.browser_use"

# Mime-type ordering for ``screenshot --no-path`` base64 decoding.
# The upstream CLI emits PNG by default; JPEG is the only other shape
# we tolerate so we don't surface a misleading "decoded successfully"
# for an unexpected payload.
_SCREENSHOT_DATA_URI_PREFIXES: tuple[str, ...] = (
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/jpg;base64,",
)


# ---------------------------------------------------------------------------
# T3 -- JavaScript eval static analysis
# ---------------------------------------------------------------------------


# Categories used by :func:`analyze_js_script` to bucket risky
# patterns. Each category triggers two-phase approval -- a category
# is "risky" iff its presence indicates the script can do something
# the user's spoken request might not have authorised:
#
#   network_egress       -- script can talk to arbitrary URLs from
#                           the page origin (cookies + headers
#                           attached automatically).
#   storage_write        -- script can write to persistent / session
#                           storage or cookies; reads the user did
#                           not authorise can be exfiltrated.
#   navigation           -- script can move the user to a new URL,
#                           losing the current authenticated context
#                           or leading to a phishing page.
#   second_order_eval    -- script can dynamically construct + run
#                           more code, defeating any allowlist that
#                           scans the literal text.
_JS_RISKY_CATEGORIES: tuple[str, ...] = (
    "network_egress",
    "storage_write",
    "navigation",
    "second_order_eval",
)


# Pattern catalog the analyzer scans. Each entry is
# (regex, category, short description).
#
# Regex shape conventions:
#   * ``\b`` boundaries to avoid matching identifiers like
#     ``mySafeFetch`` (would partial-match ``fetch``).
#   * ``\s*`` between identifier and the trailing ``(`` / ``=`` so
#     formatting variants don't escape detection.
#   * ``[^=]`` after ``=`` in assignment patterns so ``==`` / ``===``
#     comparisons don't false-positive.
#
# Mirrors both the catalog 10 T3 baseline list AND the independent
# Sonnet 4.6 security review's additions (sendBeacon, WebSocket,
# new Function, eval, import, RTCPeerConnection, navigator.sendBeacon).
_JS_RISKY_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\bfetch\s*\(", "network_egress", "fetch() call"),
    (r"\bXMLHttpRequest\b", "network_egress", "XMLHttpRequest reference"),
    (
        r"\bnavigator\.sendBeacon\s*\(",
        "network_egress",
        "navigator.sendBeacon()",
    ),
    (r"\bWebSocket\s*\(", "network_egress", "WebSocket constructor"),
    (
        r"\bRTCPeerConnection\s*\(",
        "network_egress",
        "RTCPeerConnection constructor",
    ),
    (r"\blocalStorage\.setItem\s*\(", "storage_write", "localStorage.setItem()"),
    (
        r"\bsessionStorage\.setItem\s*\(",
        "storage_write",
        "sessionStorage.setItem()",
    ),
    (r"\bdocument\.cookie\s*=\s*[^=]", "storage_write", "document.cookie assignment"),
    (r"\bwindow\.location\s*=\s*[^=]", "navigation", "window.location assignment"),
    (
        r"\bwindow\.location\.(?:replace|assign|href)\s*\(",
        "navigation",
        "window.location.replace/assign/href call",
    ),
    (r"\bdocument\.location\s*=\s*[^=]", "navigation", "document.location assignment"),
    (r"\beval\s*\(", "second_order_eval", "eval() second-order call"),
    (r"\bnew\s+Function\s*\(", "second_order_eval", "new Function constructor"),
    (r"\bimport\s*\(", "second_order_eval", "dynamic import()"),
    (r"\bdocument\.write\s*\(", "second_order_eval", "document.write()"),
)


# Compiled once at import; iterated for every analyse call. Multiline
# DOTALL is fine -- newlines don't change the meaning of any of the
# patterns above.
_JS_RISKY_COMPILED: tuple[
    tuple[re.Pattern[str], str, str], ...
] = tuple(
    (re.compile(pat), cat, desc) for pat, cat, desc in _JS_RISKY_PATTERNS
)


@dataclass(frozen=True)
class JsScriptAnalysis:
    """Outcome of a static analysis pass over a JS script body.

    Attributes:
        script_preview: short, single-line preview of the script
            (newlines collapsed, capped at 200 chars). Safe to ship
            into the safety audit log + the two-phase approval prompt.
        requires_two_phase: True when at least one risky marker
            matched. The caller MUST route through
            :class:`ApprovalRegistry` before executing.
        risky_markers: ordered tuple of short marker labels
            (``"fetch() call"``, ``"document.cookie assignment"``,
            etc.) detected in the script. Duplicates within a single
            script are deduped on description; ordering matches the
            scan order so the first match in the catalog wins for
            display purposes.
        categories: distinct categories that any marker belongs to,
            in the order :data:`_JS_RISKY_CATEGORIES` defines.
        char_count: length of the (non-stripped) script.
    """

    script_preview: str
    requires_two_phase: bool
    risky_markers: tuple[str, ...]
    categories: tuple[str, ...]
    char_count: int


def analyze_js_script(script: str) -> JsScriptAnalysis:
    """Run the static analysis pass over a JS script body.

    Pure function -- no I/O, no subprocess, no validator. Safe to
    call from any thread / on any code path including the voice hot
    path. The analysis runs at every :meth:`BrowserUseTool.eval`
    entry as a defense-in-depth check; the caller-side voice flow
    can call it explicitly when deciding whether to even prompt the
    user for approval.

    Args:
        script: the JavaScript source the caller plans to evaluate.

    Returns:
        :class:`JsScriptAnalysis`. ``requires_two_phase`` is True iff
        any risky pattern matched. When False the caller may proceed
        through the normal Cap-3 safety validator without the
        approval round-trip.

    Implementation notes:

    * Returns a "safe" analysis (no markers, requires_two_phase=False)
      for empty / whitespace-only input. The caller's argument
      validation will reject empty scripts upstream, but the analyzer
      doesn't second-guess that boundary.
    * Detection is intentionally generous: a script that mentions
      ``fetch`` inside a comment WILL trip the gate. False positives
      are acceptable; false negatives are not. Authors who need a
      string-literal ``fetch`` for legitimate reasons can route
      through the two-phase approval -- the gate's whole purpose is
      to surface the call to the user.
    """
    if not script:
        return JsScriptAnalysis(
            script_preview="",
            requires_two_phase=False,
            risky_markers=(),
            categories=(),
            char_count=0,
        )
    preview = _preview(script, cap=200)
    seen_descriptions: list[str] = []
    seen_categories: list[str] = []
    for pattern, category, description in _JS_RISKY_COMPILED:
        if pattern.search(script):
            if description not in seen_descriptions:
                seen_descriptions.append(description)
            if category not in seen_categories:
                seen_categories.append(category)
    return JsScriptAnalysis(
        script_preview=preview,
        requires_two_phase=bool(seen_descriptions),
        risky_markers=tuple(seen_descriptions),
        categories=tuple(seen_categories),
        char_count=len(script),
    )


# Canonical approval-request kind for browser-use JS eval. Used by
# the channel router to pick the right TTS narration template.
BROWSER_JS_APPROVAL_KIND: str = "browser_use_js_exec"

# Reason-code label that lands in the audit log when a JS eval is
# blocked OR approved. Mirrors the catalog 06 T3 reason-code namespace.
BROWSER_JS_REASON_CODE: str = "ultron.suspicious.browser_js_exec_unrestricted"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrowserUseResult:
    """Generic outcome of a CLI call.

    Every public method returns either this base type or a subclass
    that adds typed fields parsed out of the CLI's JSON or stdout.
    ``success=False`` is the universal sad-path signal; ``error`` is a
    short human-readable description (logged but never spoken verbatim
    without sanitisation).

    Attributes:
        success: True iff the CLI returned exit code 0 AND any expected
            parsing step succeeded.
        action: short label for the action performed (``"state"``,
            ``"open"``, ``"wait_selector"`` etc.); useful for the
            audit log and per-action telemetry.
        stdout: raw subprocess stdout (truncated to ``stdout_cap``).
            Always present so callers can fall back to substring
            matching when JSON parsing failed.
        stderr: raw subprocess stderr (truncated). Useful for surfacing
            the upstream daemon's actual error message.
        error: short failure reason when ``success=False``. None on
            happy path. Populated even when ``success=True`` if a
            partial-failure surfaced (e.g. JSON parse failed but exit
            code was 0).
        elapsed_ms: wall-clock time of the subprocess call, including
            spawn overhead. Useful for the latency dashboard.
        exit_code: subprocess exit code. None when the subprocess
            could not be spawned at all.
    """

    success: bool
    action: str
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    exit_code: Optional[int] = None


@dataclass(frozen=True)
class BrowserElement:
    """One element in the indexed-state enumeration.

    The upstream emits a numbered list of clickable / interactive
    elements; this dataclass captures the per-element record. Fields
    are best-effort: when the upstream output cannot be parsed (custom
    JSON shape, non-JSON output), the element list collapses to one
    entry with ``index=-1`` and the raw text in ``label`` so the caller
    can fall back to text-based matching.
    """

    index: int
    label: str = ""
    type: str = ""  # element type (button / link / input / ...)
    enabled: bool = True


@dataclass(frozen=True)
class BrowserState(BrowserUseResult):
    """T1 -- indexed state enumeration of the current page."""

    url: str = ""
    title: str = ""
    elements: tuple[BrowserElement, ...] = ()


@dataclass(frozen=True)
class BrowserHtmlResult(BrowserUseResult):
    """T2 -- ``get html [--selector]`` outcome."""

    html: str = ""
    selector: Optional[str] = None


@dataclass(frozen=True)
class BrowserTextResult(BrowserUseResult):
    """T2 -- ``get text <index>`` outcome."""

    text: str = ""
    index: int = -1


@dataclass(frozen=True)
class BrowserAttributesResult(BrowserUseResult):
    """T2 -- ``get attributes <index>`` outcome.

    ``attributes`` is best-effort parsed from the CLI output. JSON
    output yields a mapping; plain-text output yields one record with
    ``__raw__`` -> stdout so the caller can attempt their own parse.
    """

    attributes: Mapping[str, str] = field(default_factory=dict)
    index: int = -1


@dataclass(frozen=True)
class BrowserValueResult(BrowserUseResult):
    """T2 -- ``get value <index>`` outcome."""

    value: str = ""
    index: int = -1


@dataclass(frozen=True)
class BrowserBbox:
    """Bounding box for a single element. All four fields are physical
    pixels matching pyautogui's coordinate space. ``center_x`` /
    ``center_y`` are derived; callers can hand them directly to
    :meth:`ultron.desktop.input_control.InputController.click` to
    bridge protocol-level extraction with the safety-validated click
    gate stack.
    """

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    @property
    def center(self) -> tuple[int, int]:
        return (self.center_x, self.center_y)


@dataclass(frozen=True)
class BrowserBboxResult(BrowserUseResult):
    """T2 -- ``get bbox <index>`` outcome."""

    bbox: Optional[BrowserBbox] = None
    index: int = -1


@dataclass(frozen=True)
class BrowserTitleResult(BrowserUseResult):
    """``get title`` outcome -- thin convenience type."""

    title: str = ""


@dataclass(frozen=True)
class BrowserWaitResult(BrowserUseResult):
    """T5 -- ``wait selector`` / ``wait text`` outcome.

    ``matched=True`` when the condition was satisfied within the
    timeout. The CLI exits non-zero on timeout, so ``matched`` and
    ``success`` track together except in pathological cases (binary
    missing, CLI subprocess crashed).
    """

    matched: bool = False
    target: str = ""  # selector or text being waited on
    state: str = ""  # visible / hidden / attached / detached / text


@dataclass(frozen=True)
class BrowserTabInfo:
    """One open tab in the daemon's browser instance."""

    index: int
    url: str = ""
    title: str = ""
    active: bool = False


@dataclass(frozen=True)
class BrowserTabsResult(BrowserUseResult):
    """T6 -- ``tab list`` outcome."""

    tabs: tuple[BrowserTabInfo, ...] = ()


@dataclass(frozen=True)
class BrowserActionResult(BrowserUseResult):
    """T7 -- generic outcome for write actions (click / type / input /
    select / upload / hover / keys / dblclick / rightclick).

    ``action`` (inherited from :class:`BrowserUseResult`) carries the
    label (``"click_at_index"``, ``"type_text"``, etc.); ``target``
    carries the action-specific subject (the element index as a
    string, the typed text, the dropdown option, the file path, the
    key combo) for audit + telemetry.

    ``safety_verdict`` is the validator's aggregated verdict label
    (``"ALLOW"`` / ``"LOG_ONLY"`` / ``"BLOCK_HARD"`` /
    ``"NEEDS_EXPLICIT_INTENT"``); blank when the call short-circuited
    before the validator ran (binary missing, argument validation
    failure).
    """

    target: str = ""
    safety_verdict: str = ""


@dataclass(frozen=True)
class BrowserEvalResult(BrowserUseResult):
    """T3 -- ``eval`` outcome (JavaScript evaluation in the page).

    The eval flow has three terminal states:

    * **Approval required** (``success=False``,
      ``requires_two_phase=True``, ``approval_request_id`` set, and
      the subprocess was NOT invoked). The caller routes the
      approval through the voice / channel router; on grant the
      caller re-calls :meth:`BrowserUseTool.eval` with
      ``assume_preapproved=True``.
    * **Safety denied** (``success=False``, ``safety_verdict`` set
      to ``BLOCK_HARD`` / ``NEEDS_EXPLICIT_INTENT``). Validator
      blocked the call after static analysis cleared (or after
      pre-approval was granted but a separate rule fired).
    * **Executed** (``success=True``). ``raw_result`` carries the
      CLI's stdout, ``value`` carries the parsed JSON result when
      the CLI emitted parseable JSON (the upstream returns JSON for
      ``Runtime.evaluate(returnByValue: true)``).

    Attributes:
        raw_result: verbatim CLI stdout (truncated to the standard
            output cap). Useful when the JS result is a multi-line
            string that doesn't JSON-decode.
        value: parsed JSON value when stdout decodes as JSON, else
            ``None``. JSON shapes are preserved (str / int / float /
            bool / None / list / dict).
        requires_two_phase: True iff static analysis found markers
            requiring the two-phase approval flow AND the caller did
            not pass ``assume_preapproved=True``.
        approval_request_id: the registry key the caller routes the
            approval request through. Empty string when no approval
            was required.
        risky_markers: tuple of detected risky-pattern descriptions
            (audit-log friendly).
        categories: distinct categories the markers fall into.
        script_preview: short preview of the script body for the
            audit log / approval prompt.
        safety_verdict: validator's verdict label when the safety
            check ran; blank when the call short-circuited at static
            analysis OR argument validation.
    """

    raw_result: str = ""
    value: Optional[Any] = None
    requires_two_phase: bool = False
    approval_request_id: str = ""
    risky_markers: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    script_preview: str = ""
    safety_verdict: str = ""


@dataclass(frozen=True)
class BrowserScreenshotResult(BrowserUseResult):
    """T9 -- ``screenshot`` outcome.

    Two output shapes:

    * ``path`` set: the CLI wrote a file to ``path`` and the result's
      ``image_bytes`` is None unless ``read_back=True`` was requested.
    * ``path`` unset: the CLI emitted a base64 payload on stdout; we
      attempt to decode and populate ``image_bytes``.

    ``full_page`` mirrors the constructor arg so callers can verify
    they got what they asked for. The bytes can be handed directly to
    :meth:`ultron.desktop.vlm.Moondream2VLM.describe` for VLM
    analysis, matching the analyze-and-discard contract that
    :class:`ultron.desktop.sequence.DesktopSequenceRunner` uses on
    the desktop side (batch 8 builds the browser analog).
    """

    image_bytes: Optional[bytes] = None
    path: Optional[str] = None
    full_page: bool = False
    safety_verdict: str = ""


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


class BrowserUseTool:
    """Subprocess wrapper around the upstream ``browser-use`` CLI.

    The constructor is cheap: it does NOT discover or validate the
    binary, NOT spawn anything, NOT touch the network. Binary
    discovery runs lazily on the first :meth:`_invoke` call. This
    matches the rest of :mod:`ultron.desktop` -- constructed at
    orchestrator startup, lazy-resolves expensive dependencies.

    Args:
        binary_path: explicit path to the ``browser-use`` executable.
            ``None`` triggers PATH-based discovery against
            :data:`BROWSER_USE_BINARY_CANDIDATES`. An invalid path is
            tolerated -- :meth:`is_available` reflects the actual
            state and every invocation fails-open with a clear error.
        session: named session for this tool's calls. ``None`` means
            "no session flag" (the upstream defaults to ``default``).
            Multi-session orchestration arrives in batch 5 via
            :class:`ultron.desktop.browser_sessions.BrowserSessionManager`.
        default_timeout_s: per-call subprocess wall-clock timeout when
            an explicit ``timeout_s`` argument is omitted.
        headed: when True, every ``open`` call appends ``--headed``.
            Useful for debugging; the production default is headless.
        env_overrides: extra environment variables to set on each
            subprocess. The scrub list (:data:`_ENV_VARS_TO_SCRUB`)
            ALWAYS takes precedence -- callers cannot override
            ambient state through this kwarg.
    """

    def __init__(
        self,
        *,
        binary_path: Optional[str] = None,
        session: Optional[str] = None,
        default_timeout_s: float = DEFAULT_TIMEOUT_S,
        headed: bool = False,
        env_overrides: Optional[Mapping[str, str]] = None,
    ) -> None:
        if default_timeout_s <= 0:
            raise ValueError(
                f"default_timeout_s must be positive, got {default_timeout_s!r}"
            )
        if session is not None and not _is_valid_session_name(session):
            raise ValueError(
                f"session name must match [a-zA-Z0-9_-]{{1,32}}, got {session!r}"
            )
        self._binary_path_override: Optional[str] = binary_path
        self._resolved_binary: Optional[str] = None
        self._resolution_attempted: bool = False
        self._session: Optional[str] = session
        self._default_timeout_s: float = float(default_timeout_s)
        self._headed: bool = bool(headed)
        self._env_overrides: dict[str, str] = dict(env_overrides or {})

    # -- discovery -----------------------------------------------------

    def resolve_binary(self) -> Optional[str]:
        """Resolve the CLI binary, caching the result.

        Returns the absolute path on success, ``None`` when no
        candidate is on PATH. The cache survives until the next
        explicit :meth:`reset_binary_cache` call so PATH changes
        between calls do not surface (matches the upstream's own
        binary-cache pattern).
        """
        if self._resolution_attempted:
            return self._resolved_binary
        self._resolution_attempted = True
        # Explicit override wins, but is still validated against the
        # filesystem so a broken override is a clear None rather than
        # a deferred FileNotFoundError on subprocess spawn.
        if self._binary_path_override:
            candidate = shutil.which(self._binary_path_override) or (
                self._binary_path_override
                if _looks_like_existing_executable(self._binary_path_override)
                else None
            )
            if candidate:
                self._resolved_binary = candidate
                return candidate
            logger.warning(
                "browser_use: explicit binary path %r is not executable",
                self._binary_path_override,
            )
            return None
        for name in BROWSER_USE_BINARY_CANDIDATES:
            found = shutil.which(name)
            if found:
                self._resolved_binary = found
                return found
        return None

    def reset_binary_cache(self) -> None:
        """Forget the cached binary path so the next call re-discovers."""
        self._resolved_binary = None
        self._resolution_attempted = False

    def is_available(self) -> bool:
        """True iff the CLI binary is discoverable + executable."""
        return self.resolve_binary() is not None

    # -- session control (batch 5 builds on this) ---------------------

    @property
    def session(self) -> Optional[str]:
        return self._session

    def with_session(self, session: Optional[str]) -> "BrowserUseTool":
        """Return a new tool instance bound to ``session``.

        Used by :class:`BrowserSessionManager` (batch 5) to hand each
        managed session its own tool. Constructing a new instance is
        cheap because binary discovery is lazy.
        """
        if session is not None and not _is_valid_session_name(session):
            raise ValueError(
                f"session name must match [a-zA-Z0-9_-]{{1,32}}, got {session!r}"
            )
        return BrowserUseTool(
            binary_path=self._binary_path_override,
            session=session,
            default_timeout_s=self._default_timeout_s,
            headed=self._headed,
            env_overrides=self._env_overrides,
        )

    # -- navigation helpers (needed for read primitives to be useful) --

    def open(
        self,
        url: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserUseResult:
        """Navigate to ``url``. Returns a generic result.

        ``--headed`` is appended when the tool was constructed with
        ``headed=True``; otherwise the upstream's default (headless
        Chromium) applies.
        """
        url = (url or "").strip()
        if not url:
            return BrowserUseResult(
                success=False, action="open", error="empty url"
            )
        args: list[str] = []
        if self._headed:
            args.append("--headed")
        args.extend(["open", url])
        return self._invoke(args, action="open", timeout_s=timeout_s)

    def back(self, *, timeout_s: Optional[float] = None) -> BrowserUseResult:
        """Navigate back one entry in the tab's history."""
        return self._invoke(["back"], action="back", timeout_s=timeout_s)

    def scroll(
        self,
        direction: str = "down",
        *,
        amount: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> BrowserUseResult:
        """Scroll the page. ``direction`` must be one of
        :data:`SCROLL_DIRECTIONS`. ``amount`` is the pixel delta when
        given; the CLI's default applies when ``None``."""
        direction = (direction or "").strip().lower()
        if direction not in SCROLL_DIRECTIONS:
            return BrowserUseResult(
                success=False,
                action="scroll",
                error=f"direction must be one of {sorted(SCROLL_DIRECTIONS)}, "
                f"got {direction!r}",
            )
        args: list[str] = ["scroll", direction]
        if amount is not None:
            if amount <= 0:
                return BrowserUseResult(
                    success=False,
                    action="scroll",
                    error=f"amount must be positive, got {amount!r}",
                )
            args.extend(["--amount", str(amount)])
        return self._invoke(args, action="scroll", timeout_s=timeout_s)

    def close(
        self,
        *,
        all_sessions: bool = False,
        timeout_s: Optional[float] = None,
    ) -> BrowserUseResult:
        """Close the active browser + stop the daemon. With
        ``all_sessions=True`` closes every named session's daemon."""
        args: list[str] = ["close"]
        if all_sessions:
            args.append("--all")
        return self._invoke(args, action="close", timeout_s=timeout_s)

    # -- T1 state enumeration ------------------------------------------

    def state(self, *, timeout_s: Optional[float] = None) -> BrowserState:
        """T1 -- enumerate URL + title + indexed clickable elements.

        The upstream emits JSON when ``--json`` is passed; we always
        request it for parseability. On JSON parse failure the result
        still returns ``success=True`` (the CLI succeeded) but with
        ``elements`` empty and ``error="json parse failed"`` so
        callers can fall back to ``stdout`` substring matching.
        """
        result = self._invoke(["state", "--json"], action="state", timeout_s=timeout_s)
        if not result.success:
            return BrowserState(
                success=False,
                action="state",
                stdout=result.stdout,
                stderr=result.stderr,
                error=result.error,
                elapsed_ms=result.elapsed_ms,
                exit_code=result.exit_code,
            )
        parsed = _try_parse_state_json(result.stdout)
        if parsed is None:
            return BrowserState(
                success=True,
                action="state",
                stdout=result.stdout,
                stderr=result.stderr,
                error="json parse failed",
                elapsed_ms=result.elapsed_ms,
                exit_code=result.exit_code,
            )
        return BrowserState(
            success=True,
            action="state",
            stdout=result.stdout,
            stderr=result.stderr,
            error=None,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            url=parsed["url"],
            title=parsed["title"],
            elements=parsed["elements"],
        )

    # -- T2 DOM-native extraction -------------------------------------

    def get_html(
        self,
        selector: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserHtmlResult:
        """T2 -- raw page HTML or selector-scoped subtree."""
        args: list[str] = ["get", "html"]
        if selector is not None:
            selector = selector.strip()
            if not selector:
                return BrowserHtmlResult(
                    success=False,
                    action="get_html",
                    error="empty selector",
                )
            args.extend(["--selector", selector])
        result = self._invoke(args, action="get_html", timeout_s=timeout_s)
        return BrowserHtmlResult(
            success=result.success,
            action="get_html",
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            html=result.stdout if result.success else "",
            selector=selector,
        )

    def get_text(
        self,
        index: int,
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserTextResult:
        """T2 -- element text by index."""
        if index < 0:
            return BrowserTextResult(
                success=False,
                action="get_text",
                error=f"index must be non-negative, got {index!r}",
                index=index,
            )
        result = self._invoke(
            ["get", "text", str(index)],
            action="get_text",
            timeout_s=timeout_s,
        )
        return BrowserTextResult(
            success=result.success,
            action="get_text",
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            text=result.stdout.strip() if result.success else "",
            index=index,
        )

    def get_value(
        self,
        index: int,
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserValueResult:
        """T2 -- input / textarea current value by index.

        Caller responsibility: the returned value may include
        autofilled secrets (passwords are excluded by the upstream
        CLI for password-type inputs but other secret-bearing fields
        are not). Do not log the value verbatim without filtering.
        """
        if index < 0:
            return BrowserValueResult(
                success=False,
                action="get_value",
                error=f"index must be non-negative, got {index!r}",
                index=index,
            )
        result = self._invoke(
            ["get", "value", str(index)],
            action="get_value",
            timeout_s=timeout_s,
        )
        return BrowserValueResult(
            success=result.success,
            action="get_value",
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            value=result.stdout.rstrip("\n") if result.success else "",
            index=index,
        )

    def get_attributes(
        self,
        index: int,
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserAttributesResult:
        """T2 -- element attributes by index.

        JSON output is preferred; on parse failure the raw stdout is
        forwarded under the ``__raw__`` key so callers can attempt
        their own structured parse.
        """
        if index < 0:
            return BrowserAttributesResult(
                success=False,
                action="get_attributes",
                error=f"index must be non-negative, got {index!r}",
                index=index,
            )
        result = self._invoke(
            ["get", "attributes", str(index), "--json"],
            action="get_attributes",
            timeout_s=timeout_s,
        )
        attrs: dict[str, str] = {}
        parse_error: Optional[str] = result.error
        if result.success:
            try:
                payload = json.loads(result.stdout) if result.stdout else {}
                if isinstance(payload, Mapping):
                    attrs = {str(k): str(v) for k, v in payload.items()}
                else:
                    attrs = {"__raw__": result.stdout}
                    parse_error = "non-mapping json"
            except (ValueError, json.JSONDecodeError):
                attrs = {"__raw__": result.stdout}
                parse_error = "json parse failed"
        return BrowserAttributesResult(
            success=result.success,
            action="get_attributes",
            stdout=result.stdout,
            stderr=result.stderr,
            error=parse_error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            attributes=attrs,
            index=index,
        )

    def get_bbox(
        self,
        index: int,
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserBboxResult:
        """T2 -- bounding box by index.

        Returns physical pixel coordinates in pyautogui's coordinate
        space. The bridge to safety-gated clicks is:
        ``InputController.click(*result.bbox.center, user_text=...)``.
        """
        if index < 0:
            return BrowserBboxResult(
                success=False,
                action="get_bbox",
                error=f"index must be non-negative, got {index!r}",
                index=index,
            )
        result = self._invoke(
            ["get", "bbox", str(index), "--json"],
            action="get_bbox",
            timeout_s=timeout_s,
        )
        bbox: Optional[BrowserBbox] = None
        parse_error: Optional[str] = result.error
        if result.success:
            bbox, parse_error = _try_parse_bbox(result.stdout)
        return BrowserBboxResult(
            success=result.success and bbox is not None,
            action="get_bbox",
            stdout=result.stdout,
            stderr=result.stderr,
            error=parse_error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            bbox=bbox,
            index=index,
        )

    def get_title(
        self, *, timeout_s: Optional[float] = None
    ) -> BrowserTitleResult:
        """``get title`` -- page title convenience method."""
        result = self._invoke(
            ["get", "title"], action="get_title", timeout_s=timeout_s
        )
        return BrowserTitleResult(
            success=result.success,
            action="get_title",
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            title=result.stdout.strip() if result.success else "",
        )

    # -- T5 synchronisation barriers -----------------------------------

    def wait_selector(
        self,
        selector: str,
        *,
        state: str = "visible",
        timeout_ms: int = DEFAULT_WAIT_TIMEOUT_MS,
        timeout_s: Optional[float] = None,
    ) -> BrowserWaitResult:
        """T5 -- block until a CSS selector matches the requested state.

        ``state`` must be one of :data:`WAIT_SELECTOR_STATES`.
        ``timeout_ms`` bounds the page-level wait; ``timeout_s``
        bounds the subprocess (the latter defaults to
        ``(timeout_ms / 1000) + 5`` so the subprocess always outlives
        the page-level wait by a small margin).
        """
        selector = (selector or "").strip()
        if not selector:
            return BrowserWaitResult(
                success=False,
                action="wait_selector",
                error="empty selector",
                target=selector,
                state=state,
            )
        if state not in WAIT_SELECTOR_STATES:
            return BrowserWaitResult(
                success=False,
                action="wait_selector",
                error=f"state must be one of {sorted(WAIT_SELECTOR_STATES)}, "
                f"got {state!r}",
                target=selector,
                state=state,
            )
        if timeout_ms <= 0:
            return BrowserWaitResult(
                success=False,
                action="wait_selector",
                error=f"timeout_ms must be positive, got {timeout_ms!r}",
                target=selector,
                state=state,
            )
        effective_subprocess_timeout = (
            timeout_s if timeout_s is not None else (timeout_ms / 1000.0 + 5.0)
        )
        args = [
            "wait",
            "selector",
            selector,
            "--state",
            state,
            "--timeout",
            str(int(timeout_ms)),
        ]
        result = self._invoke(
            args, action="wait_selector", timeout_s=effective_subprocess_timeout
        )
        return BrowserWaitResult(
            success=result.success,
            action="wait_selector",
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            matched=result.success,
            target=selector,
            state=state,
        )

    def wait_text(
        self,
        text: str,
        *,
        timeout_ms: int = DEFAULT_WAIT_TIMEOUT_MS,
        timeout_s: Optional[float] = None,
    ) -> BrowserWaitResult:
        """T5 -- block until literal text appears on the page."""
        text = text or ""
        if not text:
            return BrowserWaitResult(
                success=False,
                action="wait_text",
                error="empty text",
                target=text,
                state="text",
            )
        if timeout_ms <= 0:
            return BrowserWaitResult(
                success=False,
                action="wait_text",
                error=f"timeout_ms must be positive, got {timeout_ms!r}",
                target=text,
                state="text",
            )
        effective_subprocess_timeout = (
            timeout_s if timeout_s is not None else (timeout_ms / 1000.0 + 5.0)
        )
        args = [
            "wait",
            "text",
            text,
            "--timeout",
            str(int(timeout_ms)),
        ]
        result = self._invoke(
            args, action="wait_text", timeout_s=effective_subprocess_timeout
        )
        return BrowserWaitResult(
            success=result.success,
            action="wait_text",
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            matched=result.success,
            target=text,
            state="text",
        )

    # -- T6 tab lifecycle ----------------------------------------------

    def tab_list(self, *, timeout_s: Optional[float] = None) -> BrowserTabsResult:
        """T6 -- enumerate open tabs."""
        result = self._invoke(
            ["tab", "list", "--json"],
            action="tab_list",
            timeout_s=timeout_s,
        )
        tabs: tuple[BrowserTabInfo, ...] = ()
        parse_error: Optional[str] = result.error
        if result.success:
            tabs, parse_error = _try_parse_tabs(result.stdout)
        return BrowserTabsResult(
            success=result.success,
            action="tab_list",
            stdout=result.stdout,
            stderr=result.stderr,
            error=parse_error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            tabs=tabs,
        )

    def tab_new(
        self,
        url: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserUseResult:
        """T6 -- open a new tab. ``url`` is optional; blank tab when None."""
        args: list[str] = ["tab", "new"]
        if url is not None:
            url = url.strip()
            if not url:
                return BrowserUseResult(
                    success=False, action="tab_new", error="empty url"
                )
            args.append(url)
        return self._invoke(args, action="tab_new", timeout_s=timeout_s)

    def tab_switch(
        self,
        index: int,
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserUseResult:
        """T6 -- switch the agent's active tab to ``index``.

        Note: this only changes the agent's logical focus -- it does
        NOT change which tab the USER sees in their browser window.
        For user-visible tab switching, see batch 7's
        ``cdp_python(...)`` with ``Target.activateTarget``.
        """
        if index < 0:
            return BrowserUseResult(
                success=False,
                action="tab_switch",
                error=f"index must be non-negative, got {index!r}",
            )
        return self._invoke(
            ["tab", "switch", str(index)],
            action="tab_switch",
            timeout_s=timeout_s,
        )

    def tab_close(
        self,
        indices: Sequence[int],
        *,
        timeout_s: Optional[float] = None,
    ) -> BrowserUseResult:
        """T6 -- close one or more tabs by index."""
        if not indices:
            return BrowserUseResult(
                success=False,
                action="tab_close",
                error="at least one index required",
            )
        for idx in indices:
            if idx < 0:
                return BrowserUseResult(
                    success=False,
                    action="tab_close",
                    error=f"all indices must be non-negative, got {idx!r}",
                )
        args = ["tab", "close", *(str(i) for i in indices)]
        return self._invoke(args, action="tab_close", timeout_s=timeout_s)

    # -- T7 form interaction (write primitives, Cap-3 gated) ----------

    def click_at_index(
        self,
        index: int,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- click the element at ``index`` (from a prior
        :meth:`state` call). Routed through the safety validator with
        ``tool_name=desktop.browser_use.click_at_index``."""
        if index < 0:
            return _failed_action(
                "click_at_index", f"index must be non-negative, got {index!r}"
            )
        denial = self._safety_check(
            action="click_at_index",
            arguments={"index": index},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["click", str(index)],
            action="click_at_index",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=str(index),
            safety_verdict="ALLOW",
        )

    def click_at_coords(
        self,
        x: int,
        y: int,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- click at pixel ``(x, y)`` in the active tab's viewport.

        Coordinate clicks are less typical than indexed clicks but
        the CLI exposes them; used by :class:`BrowserSequenceRunner`
        (batch 8) when handing off from a VLM-derived target to a
        direct page click without re-running ``state``.
        """
        if x < 0 or y < 0:
            return _failed_action(
                "click_at_coords",
                f"coords must be non-negative, got ({x!r}, {y!r})",
            )
        denial = self._safety_check(
            action="click_at_coords",
            arguments={"x": x, "y": y},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["click", str(x), str(y)],
            action="click_at_coords",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=f"{x},{y}",
            safety_verdict="ALLOW",
        )

    def type_text(
        self,
        text: str,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- type ``text`` into whatever element currently has
        focus. Use :meth:`input` to combine click + type.
        """
        if not text:
            return _failed_action("type_text", "empty text")
        denial = self._safety_check(
            action="type_text",
            arguments={"text_preview": _preview(text)},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["type", text],
            action="type_text",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=_preview(text),
            safety_verdict="ALLOW",
        )

    def input(
        self,
        index: int,
        text: str,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- click element at ``index`` then type ``text`` into it.

        Compound atomic primitive: the CLI handles the click+type
        sequence inside the daemon so focus loss between the two is
        not observable from the caller side.
        """
        if index < 0:
            return _failed_action(
                "input", f"index must be non-negative, got {index!r}"
            )
        if not text:
            return _failed_action("input", "empty text")
        denial = self._safety_check(
            action="input",
            arguments={"index": index, "text_preview": _preview(text)},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["input", str(index), text],
            action="input",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=f"{index}:{_preview(text)}",
            safety_verdict="ALLOW",
        )

    def select(
        self,
        index: int,
        option: str,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- set dropdown ``index`` to ``option``."""
        if index < 0:
            return _failed_action(
                "select", f"index must be non-negative, got {index!r}"
            )
        if not option:
            return _failed_action("select", "empty option")
        denial = self._safety_check(
            action="select",
            arguments={"index": index, "option": option},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["select", str(index), option],
            action="select",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=f"{index}:{option}",
            safety_verdict="ALLOW",
        )

    def upload(
        self,
        index: int,
        path: str,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
        path_resolver: Optional[PathResolver] = None,
    ) -> BrowserActionResult:
        """T7 upload (YELLOW per security review) -- provide a local
        file to a file-input element at ``index``.

        Goes through stricter gating than the other T7 writes because
        the path argument reads a local file and sends its contents
        into the browser process. Steps:

        1. Reject blank / negative arguments at the wrapper boundary.
        2. Canonicalise the path via :meth:`PathResolver.safe_realpath`
           so attacker-controlled symlinks / junctions / bidi-override
           filenames can never escape into the subprocess.
        3. Confirm the resolved path exists + is a regular file.
        4. Pass the resolved path tuple to the safety validator with
           ``capability=desktop_browser_use`` so Cap-3 and the
           file-read rules in category D / category A / Cap-2 see it.
        5. On allow, invoke the CLI with the resolved (NOT the raw)
           path so the daemon also sees the canonical form.

        The resolver argument is dependency-injected for tests.
        """
        if index < 0:
            return _failed_action(
                "upload", f"index must be non-negative, got {index!r}"
            )
        raw_path = (path or "").strip()
        if not raw_path:
            return _failed_action("upload", "empty path")
        resolver = path_resolver if path_resolver is not None else get_path_resolver()
        resolved = resolver.safe_realpath(raw_path)
        if resolved is None:
            return _failed_action(
                "upload",
                f"path does not resolve to a real file: {raw_path!r}",
            )
        if not resolved.is_file():
            return _failed_action(
                "upload",
                f"resolved path is not a regular file: {resolved}",
            )
        denial = self._safety_check(
            action="upload",
            arguments={
                "index": index,
                "path": str(resolved),
                "raw_path": raw_path,
            },
            user_text=user_text,
            paths=(resolved,),
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["upload", str(index), str(resolved)],
            action="upload",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=f"{index}:{resolved}",
            safety_verdict="ALLOW",
        )

    def hover(
        self,
        index: int,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- hover the element at ``index`` to reveal hidden
        menus / CSS hover state. Cheap; no actual click."""
        if index < 0:
            return _failed_action(
                "hover", f"index must be non-negative, got {index!r}"
            )
        denial = self._safety_check(
            action="hover",
            arguments={"index": index},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["hover", str(index)],
            action="hover",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=str(index),
            safety_verdict="ALLOW",
        )

    def keys(
        self,
        combo: str,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- send a key combo (``"Enter"`` / ``"Control+a"`` etc.)
        to the focused element.

        The upstream CLI accepts the same key-name syntax as Playwright's
        ``page.keyboard.press`` (modifier names ``Control`` / ``Alt`` /
        ``Shift`` / ``Meta`` joined with ``+`` before a key name like
        ``Enter`` / ``a`` / ``ArrowDown``). We do NOT validate the
        combo string at our boundary -- the surface is large and the
        CLI is a better validator than us.
        """
        combo = (combo or "").strip()
        if not combo:
            return _failed_action("keys", "empty key combo")
        denial = self._safety_check(
            action="keys",
            arguments={"combo": combo},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["keys", combo],
            action="keys",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=combo,
            safety_verdict="ALLOW",
        )

    def dblclick(
        self,
        index: int,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- double-click the element at ``index``."""
        if index < 0:
            return _failed_action(
                "dblclick", f"index must be non-negative, got {index!r}"
            )
        denial = self._safety_check(
            action="dblclick",
            arguments={"index": index},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["dblclick", str(index)],
            action="dblclick",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=str(index),
            safety_verdict="ALLOW",
        )

    def rightclick(
        self,
        index: int,
        *,
        user_text: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserActionResult:
        """T7 -- right-click the element at ``index`` (context menus)."""
        if index < 0:
            return _failed_action(
                "rightclick", f"index must be non-negative, got {index!r}"
            )
        denial = self._safety_check(
            action="rightclick",
            arguments={"index": index},
            user_text=user_text,
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["rightclick", str(index)],
            action="rightclick",
            timeout_s=timeout_s,
        )
        return _action_from_invoke(
            result,
            target=str(index),
            safety_verdict="ALLOW",
        )

    # -- T9 screenshot -------------------------------------------------

    def screenshot(
        self,
        path: Optional[str] = None,
        *,
        full_page: bool = False,
        user_text: str = "",
        timeout_s: Optional[float] = None,
        path_resolver: Optional[PathResolver] = None,
    ) -> BrowserScreenshotResult:
        """T9 -- capture a screenshot of the current page.

        Two output modes:

        * ``path`` set -- the CLI writes the PNG to disk at ``path``.
          The path is canonicalised + sandbox-checked via
          :class:`PathResolver` BEFORE the subprocess runs. The result
          carries the resolved path; ``image_bytes`` is None (callers
          read the file themselves if they need the bytes).
        * ``path`` unset -- the CLI emits base64 on stdout. We decode
          and populate ``image_bytes``; ``path`` is None on the result.

        ``full_page=True`` appends ``--full`` so the entire scrollable
        page is captured. The base64 output mode is the analyze-and-
        discard path (caller feeds the bytes to the VLM and lets the
        result dataclass go out of scope); the path output mode is
        for the "save this for the user to look at" path.
        """
        denial = self._safety_check_screenshot(
            path=path,
            full_page=full_page,
            user_text=user_text,
            path_resolver=path_resolver,
        )
        if isinstance(denial, BrowserScreenshotResult):
            return denial
        resolved_path: Optional[Path] = denial  # type: ignore[assignment]
        args: list[str] = ["screenshot"]
        if resolved_path is not None:
            args.append(str(resolved_path))
        if full_page:
            args.append("--full")
        result = self._invoke(
            args,
            action="screenshot",
            timeout_s=timeout_s,
        )
        if not result.success:
            return BrowserScreenshotResult(
                success=False,
                action="screenshot",
                stdout=result.stdout,
                stderr=result.stderr,
                error=result.error,
                elapsed_ms=result.elapsed_ms,
                exit_code=result.exit_code,
                path=str(resolved_path) if resolved_path is not None else None,
                full_page=full_page,
                safety_verdict="ALLOW",
            )
        image_bytes: Optional[bytes] = None
        decode_error: Optional[str] = None
        if resolved_path is None:
            image_bytes, decode_error = _decode_screenshot_payload(result.stdout)
        return BrowserScreenshotResult(
            success=True,
            action="screenshot",
            stdout=result.stdout,
            stderr=result.stderr,
            error=decode_error,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            image_bytes=image_bytes,
            path=str(resolved_path) if resolved_path is not None else None,
            full_page=full_page,
            safety_verdict="ALLOW",
        )

    def _safety_check_screenshot(
        self,
        *,
        path: Optional[str],
        full_page: bool,
        user_text: str,
        path_resolver: Optional[PathResolver],
    ) -> Optional[Path] | BrowserScreenshotResult:
        """Run the safety + path checks for ``screenshot``.

        Returns:
            * the resolved :class:`Path` (or ``None`` for base64 mode)
              on success.
            * a fully-populated :class:`BrowserScreenshotResult` with
              ``success=False`` on denial.

        Splitting this out keeps the main ``screenshot`` method linear.
        """
        resolved_path: Optional[Path] = None
        paths_for_safety: tuple[Path, ...] = ()
        if path is not None:
            raw_path = path.strip()
            if not raw_path:
                return BrowserScreenshotResult(
                    success=False,
                    action="screenshot",
                    error="empty path",
                    full_page=full_page,
                )
            resolver = (
                path_resolver if path_resolver is not None else get_path_resolver()
            )
            # ``safe_realpath`` returns None for paths that don't
            # exist on disk yet -- which is the common case for
            # screenshot output. Fall back to ``resolve`` (which
            # accepts non-existing paths) and validate the parent
            # directory exists + is writable separately.
            resolved = resolver.safe_realpath(raw_path)
            if resolved is None:
                # Path doesn't exist yet -- resolve via the lighter
                # path canonicalisation and check the parent.
                try:
                    resolved = resolver.resolve(raw_path)
                except (ValueError, OSError) as exc:
                    return BrowserScreenshotResult(
                        success=False,
                        action="screenshot",
                        error=f"path canonicalisation failed: {exc}",
                        full_page=full_page,
                    )
                if not resolved.parent.is_dir():
                    return BrowserScreenshotResult(
                        success=False,
                        action="screenshot",
                        error=f"parent directory does not exist: {resolved.parent}",
                        full_page=full_page,
                    )
            resolved_path = resolved
            paths_for_safety = (resolved,)
        denial = self._safety_check(
            action="screenshot",
            arguments={
                "path": str(resolved_path) if resolved_path is not None else None,
                "full_page": full_page,
                "output_mode": "file" if resolved_path is not None else "base64",
            },
            user_text=user_text,
            paths=paths_for_safety,
            result_factory=BrowserScreenshotResult,
            extra_result_kwargs={
                "path": str(resolved_path) if resolved_path is not None else None,
                "full_page": full_page,
            },
        )
        if denial is not None:
            return denial
        return resolved_path

    # -- T3 JavaScript evaluation (YELLOW) -----------------------------

    def eval(
        self,
        script: str,
        *,
        user_text: str = "",
        assume_preapproved: bool = False,
        approval_registry: Optional[ApprovalRegistry] = None,
        approval_timeout_s: Optional[float] = None,
        approval_scope_key: str = "",
        timeout_s: Optional[float] = None,
    ) -> BrowserEvalResult:
        """T3 (YELLOW) -- evaluate ``script`` in the active tab's
        JavaScript context.

        Three-phase pipeline:

        1. **Argument validation** -- blank script rejected at the
           wrapper boundary; no static analysis, no validator, no
           approval round-trip.
        2. **Static analysis** -- :func:`analyze_js_script` scans the
           body for write / network / navigation / second-order-eval
           markers. When ANY marker matches AND ``assume_preapproved``
           is False, an :class:`ApprovalRequest` is registered with
           the approval registry; the returned result carries
           ``requires_two_phase=True`` and ``approval_request_id`` so
           the channel router can ask the user yes/no. The CLI is
           NOT invoked yet.
        3. **Cap-3 safety check + subprocess** -- when (no markers)
           OR (markers + ``assume_preapproved=True``), the wrapper
           builds a :class:`RuleContext` with the script preview +
           markers + categories in ``arguments`` and runs the
           validator. On allow, the script is handed to the CLI
           verbatim; stdout is parsed as JSON when possible.

        Args:
            script: JavaScript source to evaluate.
            user_text: the user utterance that originated the call.
                Threaded into the validator for explicit-intent
                matching AND echoed in the approval prompt for voice
                confirmation.
            assume_preapproved: set True when the caller has already
                walked the user through the approval flow AND wants
                to execute the (still risky) script. The static
                analysis still runs as a defense-in-depth pass and
                the result records the markers, but the call proceeds
                past the approval phase.
            approval_registry: injectable registry override. Tests
                use this to verify the registration call. Production
                defaults to :func:`get_approval_registry`.
            approval_timeout_s: per-request approval timeout passed
                to :class:`ApprovalRequest`. ``None`` uses the
                registry's default.
            approval_scope_key: caller-supplied session id / scope
                key for the approval entry. Used by the registry's
                per-scope listing API.
            timeout_s: subprocess wall-clock timeout override.

        Returns:
            :class:`BrowserEvalResult`. Inspect ``requires_two_phase``
            first: True means the caller must drive the approval
            flow; False means ``success`` reflects the actual
            execution outcome.
        """
        script = script or ""
        if not script.strip():
            return BrowserEvalResult(
                success=False,
                action="eval",
                error="empty script",
            )
        analysis = analyze_js_script(script)
        if analysis.requires_two_phase and not assume_preapproved:
            registry = (
                approval_registry
                if approval_registry is not None
                else get_approval_registry()
            )
            handle = self._register_eval_approval(
                registry=registry,
                analysis=analysis,
                user_text=user_text,
                approval_timeout_s=approval_timeout_s,
                scope_key=approval_scope_key,
            )
            preapproved_decision = (
                handle.pre_resolved.outcome.value
                if handle.pre_resolved is not None
                else ""
            )
            return BrowserEvalResult(
                success=False,
                action="eval",
                error=(
                    f"two-phase approval required (markers: "
                    f"{', '.join(analysis.risky_markers)})"
                ),
                requires_two_phase=True,
                approval_request_id=handle.approval_id,
                risky_markers=analysis.risky_markers,
                categories=analysis.categories,
                script_preview=analysis.script_preview,
                safety_verdict=preapproved_decision,
            )
        # Static analysis cleared OR caller is asserting preapproval.
        # Run the regular Cap-3 safety check before invoking the CLI.
        denial = self._safety_check(
            action="eval",
            arguments={
                "script_preview": analysis.script_preview,
                "risky_markers": list(analysis.risky_markers),
                "categories": list(analysis.categories),
                "char_count": analysis.char_count,
                "assume_preapproved": assume_preapproved,
            },
            user_text=user_text,
            result_factory=BrowserEvalResult,
            extra_result_kwargs={
                "risky_markers": analysis.risky_markers,
                "categories": analysis.categories,
                "script_preview": analysis.script_preview,
                "requires_two_phase": analysis.requires_two_phase,
            },
        )
        if denial is not None:
            return denial
        result = self._invoke(
            ["eval", script],
            action="eval",
            timeout_s=timeout_s,
        )
        if not result.success:
            return BrowserEvalResult(
                success=False,
                action="eval",
                stdout=result.stdout,
                stderr=result.stderr,
                error=result.error,
                elapsed_ms=result.elapsed_ms,
                exit_code=result.exit_code,
                requires_two_phase=analysis.requires_two_phase,
                risky_markers=analysis.risky_markers,
                categories=analysis.categories,
                script_preview=analysis.script_preview,
                safety_verdict="ALLOW",
            )
        value, raw_result = _try_parse_eval_payload(result.stdout)
        return BrowserEvalResult(
            success=True,
            action="eval",
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_ms=result.elapsed_ms,
            exit_code=result.exit_code,
            raw_result=raw_result,
            value=value,
            requires_two_phase=analysis.requires_two_phase,
            risky_markers=analysis.risky_markers,
            categories=analysis.categories,
            script_preview=analysis.script_preview,
            safety_verdict="ALLOW",
        )

    def _register_eval_approval(
        self,
        *,
        registry: ApprovalRegistry,
        analysis: JsScriptAnalysis,
        user_text: str,
        approval_timeout_s: Optional[float],
        scope_key: str,
    ) -> ApprovalHandle:
        """Register a two-phase approval request for a risky eval.

        The prompt is constructed to be safe for TTS: no script
        body verbatim (the audit log gets the preview separately),
        just the marker labels and a one-line description so the
        channel router speaks "this script will <markers>. Proceed?".
        """
        request = ApprovalRequest(
            kind=BROWSER_JS_APPROVAL_KIND,
            prompt=(
                "Browser script wants to "
                + _humanize_categories(analysis.categories)
                + ". Proceed?"
            ),
            actor="desktop_browser_use",
            scope_key=scope_key or self._session or "",
            metadata={
                "script_preview": analysis.script_preview,
                "risky_markers": list(analysis.risky_markers),
                "categories": list(analysis.categories),
                "char_count": analysis.char_count,
                "user_text": user_text,
                "reason_code": BROWSER_JS_REASON_CODE,
            },
            timeout_seconds=approval_timeout_s,
            delivery_channel="voice",
        )
        return registry.register(request)

    # -- safety helper --------------------------------------------------

    def _safety_check(
        self,
        *,
        action: str,
        arguments: dict[str, Any],
        user_text: str,
        paths: tuple[Path, ...] = (),
        result_factory: Any = None,
        extra_result_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Run the validator. Returns ``None`` when the call is
        allowed, otherwise returns a populated failure result.

        The default ``result_factory`` is :class:`BrowserActionResult`.
        :meth:`screenshot` overrides to get a
        :class:`BrowserScreenshotResult` shape back.
        """
        factory = result_factory if result_factory is not None else BrowserActionResult
        try:
            validator = get_validator()
        except Exception:  # pragma: no cover -- defensive
            return None  # Validator unavailable -> permissive fall-through.
        try:
            ctx = RuleContext(
                tool_name=f"{_TOOL_NAME_PREFIX}.{action}",
                arguments=dict(arguments),
                capability=_VALIDATOR_CAPABILITY,
                paths=paths,
                user_text=user_text or "",
            )
            verdict = validator.check(ctx)
        except Exception as exc:
            logger.warning(
                "browser_use safety check raised; treating as deny: %s", exc
            )
            kwargs = {
                "success": False,
                "action": action,
                "error": f"safety check raised: {type(exc).__name__}",
                "safety_verdict": "BLOCK_HARD",
            }
            if extra_result_kwargs:
                kwargs.update(extra_result_kwargs)
            return factory(**kwargs)
        if verdict.is_allowed:
            return None
        verdict_label = verdict.verdict.value if isinstance(verdict.verdict, Verdict) else str(verdict.verdict)
        message = verdict.user_message or verdict.reason or "safety validator blocked the call"
        kwargs = {
            "success": False,
            "action": action,
            "error": f"safety denied ({verdict_label}): {message}",
            "safety_verdict": verdict_label,
        }
        if extra_result_kwargs:
            kwargs.update(extra_result_kwargs)
        if factory is BrowserActionResult:
            # BrowserActionResult also tracks ``target`` so callers
            # know what was attempted even when blocked.
            kwargs.setdefault("target", _short_target_label(arguments))
        return factory(**kwargs)

    # -- core subprocess invocation ------------------------------------

    def _invoke(
        self,
        args: Sequence[str],
        *,
        action: str,
        timeout_s: Optional[float] = None,
    ) -> BrowserUseResult:
        """Run a subprocess call and package the result.

        All public methods funnel through here. The shape is:

        1. Lazy-resolve the binary; fail-open with a clear error when
           it isn't on PATH.
        2. Prepend ``--session NAME`` when this instance is session-
           bound. Session goes BEFORE the subcommand per upstream
           convention; this is enforced by the order assembled here,
           not by callers.
        3. Build the env dict by scrubbing
           :data:`_ENV_VARS_TO_SCRUB` from the parent's env and
           layering ``env_overrides`` on top.
        4. Run via :func:`subprocess.run` with
           ``creationflags=_CREATE_NO_WINDOW`` on Windows.
        5. Bound stdout / stderr to ``_OUTPUT_CAP_BYTES`` so a runaway
           CLI cannot OOM the orchestrator. Larger payloads are
           truncated head + tail with an elision marker mirroring
           :func:`ultron.coding.observation_format.truncate_observation`.
        """
        binary = self.resolve_binary()
        if binary is None:
            return BrowserUseResult(
                success=False,
                action=action,
                error="browser-use binary not found on PATH",
            )
        cmd: list[str] = [binary]
        if self._session is not None:
            cmd.extend(["--session", self._session])
        cmd.extend(str(a) for a in args)
        effective_timeout = float(
            timeout_s if timeout_s is not None else self._default_timeout_s
        )
        if effective_timeout <= 0:
            return BrowserUseResult(
                success=False,
                action=action,
                error=f"timeout_s must be positive, got {effective_timeout!r}",
            )
        env = _build_scrubbed_env(self._env_overrides)
        # Use the safer subprocess.run flag tuple. CREATE_NO_WINDOW
        # is a no-op on non-Windows platforms.
        start = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 -- trusted binary lookup
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=effective_timeout,
                creationflags=_CREATE_NO_WINDOW,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000.0
            return BrowserUseResult(
                success=False,
                action=action,
                error=f"subprocess timeout after {effective_timeout:.1f}s",
                elapsed_ms=elapsed,
            )
        except (FileNotFoundError, PermissionError) as exc:
            elapsed = (time.monotonic() - start) * 1000.0
            return BrowserUseResult(
                success=False,
                action=action,
                error=f"subprocess spawn failed: {type(exc).__name__}: {exc}",
                elapsed_ms=elapsed,
            )
        except OSError as exc:
            elapsed = (time.monotonic() - start) * 1000.0
            return BrowserUseResult(
                success=False,
                action=action,
                error=f"subprocess os error: {exc}",
                elapsed_ms=elapsed,
            )
        elapsed = (time.monotonic() - start) * 1000.0
        stdout = _truncate(completed.stdout or "")
        stderr = _truncate(completed.stderr or "")
        if completed.returncode != 0:
            return BrowserUseResult(
                success=False,
                action=action,
                stdout=stdout,
                stderr=stderr,
                error=_extract_cli_error(stderr, stdout)
                or f"non-zero exit code {completed.returncode}",
                elapsed_ms=elapsed,
                exit_code=completed.returncode,
            )
        return BrowserUseResult(
            success=True,
            action=action,
            stdout=stdout,
            stderr=stderr,
            elapsed_ms=elapsed,
            exit_code=completed.returncode,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_tool_singleton: Optional[BrowserUseTool] = None


def get_browser_use_tool() -> Optional[BrowserUseTool]:
    """Return the module-level singleton, or ``None`` if unset.

    Matches the pattern used by :mod:`ultron.desktop.vlm` (lazy
    construction is the orchestrator's job; readers degrade gracefully).
    """
    return _tool_singleton


def set_browser_use_tool(tool: Optional[BrowserUseTool]) -> None:
    """Set or clear the module-level singleton. Tests / orchestrator
    init use this to install a configured instance."""
    global _tool_singleton
    _tool_singleton = tool


def reset_browser_use_tool_for_testing() -> None:
    """Clear the singleton. Tests should call this in teardown."""
    set_browser_use_tool(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Max bytes of stdout / stderr we capture per call. Large enough for
# typical state JSON + HTML snippets; small enough that a runaway CLI
# cannot OOM the orchestrator.
_OUTPUT_CAP_BYTES: int = 256 * 1024  # 256 KB

# Session names: alphanumeric / underscore / hyphen, 1-32 chars. Same
# shape the catalog 10 T8 plan recommends for :class:`BrowserSessionManager`
# (batch 5). Validated at both construction and ``with_session`` time
# so an invalid name can never become a subprocess argument.
_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _is_valid_session_name(name: str) -> bool:
    return bool(_SESSION_NAME_RE.match(name))


def _looks_like_existing_executable(path_str: str) -> bool:
    """Cheap "is this path executable" probe used only as a fallback
    when :func:`shutil.which` returns nothing for an explicit override.
    Avoids importing :mod:`pathlib` for a one-shot OS check."""
    try:
        import os

        return os.path.isfile(path_str) and os.access(path_str, os.X_OK)
    except OSError:
        return False


def _build_scrubbed_env(
    overrides: Mapping[str, str],
) -> dict[str, str]:
    """Build the subprocess env: start from parent env, drop the
    scrub list, layer overrides on top. Overrides cannot reintroduce
    the scrubbed keys (defensive against caller misuse)."""
    import os

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _ENV_VARS_TO_SCRUB
    }
    for k, v in overrides.items():
        if k in _ENV_VARS_TO_SCRUB:
            continue
        env[str(k)] = str(v)
    return env


def _truncate(text: str) -> str:
    """Cap large stdout / stderr payloads. Head + tail preservation
    is the readable shape; the elision marker matches
    :func:`ultron.coding.observation_format.truncate_observation`'s
    convention so existing log viewers handle it."""
    if not text:
        return ""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= _OUTPUT_CAP_BYTES:
        return text
    head = raw[: _OUTPUT_CAP_BYTES // 2]
    tail = raw[-(_OUTPUT_CAP_BYTES // 2):]
    elided = len(raw) - len(head) - len(tail)
    marker = f"\n... [{elided} bytes elided] ...\n".encode("utf-8")
    return (head + marker + tail).decode("utf-8", errors="replace")


def _extract_cli_error(stderr: str, stdout: str) -> Optional[str]:
    """Pull a short error description out of CLI output for the
    result ``error`` field. Prefers stderr first non-blank line; falls
    back to stdout when stderr is empty (some daemons emit errors on
    stdout for piping convenience)."""
    for source in (stderr, stdout):
        for raw in source.splitlines():
            line = raw.strip()
            if line:
                # Cap the surfaced error message length so a verbose
                # CLI cannot dominate the audit log.
                return line[:512]
    return None


def _try_parse_state_json(stdout: str) -> Optional[dict[str, Any]]:
    """Parse the JSON document the upstream emits for ``state --json``.

    The exact shape is daemon-version-dependent; this parser tolerates
    common variations:

    * ``{"url": ..., "title": ..., "elements": [{"index": N, "label": ..., "type": ..., "enabled": ...}, ...]}``
    * ``{"url": ..., "title": ..., "interactive_elements": [...]}``
    * Element entries may use ``text`` / ``name`` / ``label`` interchangeably.

    Returns ``None`` on irrecoverable parse failure.
    """
    if not stdout:
        return None
    try:
        payload = json.loads(stdout)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    url = str(payload.get("url", "") or "")
    title = str(payload.get("title", "") or "")
    raw_elements = (
        payload.get("elements")
        or payload.get("interactive_elements")
        or payload.get("clickable_elements")
        or []
    )
    elements: list[BrowserElement] = []
    if isinstance(raw_elements, Sequence) and not isinstance(raw_elements, str):
        for i, entry in enumerate(raw_elements):
            if not isinstance(entry, Mapping):
                continue
            try:
                index = int(entry.get("index", i))
            except (TypeError, ValueError):
                index = i
            label = str(
                entry.get("label")
                or entry.get("text")
                or entry.get("name")
                or entry.get("title")
                or ""
            )
            element_type = str(
                entry.get("type")
                or entry.get("kind")
                or entry.get("role")
                or ""
            )
            enabled_raw = entry.get("enabled", True)
            enabled = bool(enabled_raw) if enabled_raw is not None else True
            elements.append(
                BrowserElement(
                    index=index,
                    label=label,
                    type=element_type,
                    enabled=enabled,
                )
            )
    return {
        "url": url,
        "title": title,
        "elements": tuple(elements),
    }


def _try_parse_bbox(stdout: str) -> tuple[Optional[BrowserBbox], Optional[str]]:
    """Parse ``get bbox --json`` output. Tolerates:

    * ``{"x": N, "y": N, "width": N, "height": N}``
    * ``{"left": N, "top": N, "width": N, "height": N}``
    * ``{"x": N, "y": N, "w": N, "h": N}``

    Returns ``(bbox, None)`` on success, ``(None, error_string)`` on
    parse failure.
    """
    if not stdout:
        return None, "empty bbox output"
    try:
        payload = json.loads(stdout)
    except (ValueError, json.JSONDecodeError):
        return None, "json parse failed"
    if not isinstance(payload, Mapping):
        return None, "non-mapping bbox output"
    try:
        x = int(payload.get("x", payload.get("left", 0)) or 0)
        y = int(payload.get("y", payload.get("top", 0)) or 0)
        width = int(payload.get("width", payload.get("w", 0)) or 0)
        height = int(payload.get("height", payload.get("h", 0)) or 0)
    except (TypeError, ValueError):
        return None, "bbox fields not integral"
    if width < 0 or height < 0:
        return None, "bbox dimensions negative"
    return BrowserBbox(x=x, y=y, width=width, height=height), None


def _failed_action(action: str, error: str) -> BrowserActionResult:
    """Build a pre-subprocess failure result for a write method.

    Used when argument validation rejects the call before the safety
    validator or the subprocess can run (negative index, empty text,
    etc.). ``safety_verdict`` is blank because the validator never
    ran.
    """
    return BrowserActionResult(
        success=False,
        action=action,
        error=error,
        safety_verdict="",
    )


def _action_from_invoke(
    invoke_result: BrowserUseResult,
    *,
    target: str,
    safety_verdict: str,
) -> BrowserActionResult:
    """Project a generic ``_invoke`` result into a
    :class:`BrowserActionResult` with the action-specific fields
    populated. Subprocess outcomes (success / failure / timeout /
    spawn error) all flow through here uniformly.
    """
    return BrowserActionResult(
        success=invoke_result.success,
        action=invoke_result.action,
        stdout=invoke_result.stdout,
        stderr=invoke_result.stderr,
        error=invoke_result.error,
        elapsed_ms=invoke_result.elapsed_ms,
        exit_code=invoke_result.exit_code,
        target=target,
        safety_verdict=safety_verdict,
    )


def _preview(text: str, *, cap: int = 80) -> str:
    """Compact preview of arbitrary text for audit + denial messages.

    Caps at ``cap`` chars, replaces newlines with spaces, appends an
    elision marker when truncated. Used as the ``target`` field for
    :meth:`BrowserUseTool.type_text` and :meth:`BrowserUseTool.input`
    so the audit log can record what was typed without echoing
    arbitrarily long payloads.
    """
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= cap:
        return flat
    return flat[: cap - 1] + "…"  # ellipsis


def _short_target_label(arguments: Mapping[str, Any]) -> str:
    """Build a short human-readable target label from a write-method's
    arguments dict. Used as the ``target`` field on safety-denial
    results so audit log readers can see what the call was attempting
    even when the validator blocked it."""
    if "index" in arguments and "text_preview" in arguments:
        return f"{arguments['index']}:{arguments['text_preview']}"
    if "index" in arguments and "option" in arguments:
        return f"{arguments['index']}:{arguments['option']}"
    if "index" in arguments and "path" in arguments:
        return f"{arguments['index']}:{arguments['path']}"
    if "x" in arguments and "y" in arguments:
        return f"{arguments['x']},{arguments['y']}"
    if "index" in arguments:
        return str(arguments["index"])
    if "combo" in arguments:
        return str(arguments["combo"])
    if "text_preview" in arguments:
        return str(arguments["text_preview"])
    if "path" in arguments and arguments["path"] is not None:
        return str(arguments["path"])
    return ""


def _humanize_categories(categories: tuple[str, ...]) -> str:
    """Render a category tuple into a short TTS-safe phrase for the
    approval prompt. Maps the internal labels onto user-readable
    verbs so the voice prompt doesn't expose internal jargon.

    Multiple categories join with " and "; an empty tuple returns
    the safe fallback "execute custom code".
    """
    if not categories:
        return "execute custom code"
    label_map = {
        "network_egress": "make network requests",
        "storage_write": "write to storage or cookies",
        "navigation": "navigate to another page",
        "second_order_eval": "execute dynamically constructed code",
    }
    phrases = [label_map.get(c, c) for c in categories]
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _try_parse_eval_payload(stdout: str) -> tuple[Optional[Any], str]:
    """Parse the CLI's eval-result stdout.

    The upstream returns whatever ``Runtime.evaluate(returnByValue=True)``
    emitted -- JSON-encoded when feasible, otherwise raw text.
    Returns ``(parsed_value_or_None, raw_text)``.
    """
    if not stdout:
        return None, ""
    raw = stdout.strip()
    if not raw:
        return None, ""
    try:
        return json.loads(raw), raw
    except (ValueError, json.JSONDecodeError):
        return None, raw


def _decode_screenshot_payload(
    stdout: str,
) -> tuple[Optional[bytes], Optional[str]]:
    """Decode the base64 payload the CLI emits when ``screenshot`` is
    called without an output path.

    Tolerates two shapes:

    * ``data:image/<png|jpeg|jpg>;base64,<payload>`` URI form
    * raw base64 text (no prefix)

    Returns ``(bytes, None)`` on success, ``(None, error_string)`` on
    parse failure. Bytes-too-small (< 16 bytes after decode) is
    treated as a parse failure -- a real PNG / JPEG has more than
    that just in its header.
    """
    if not stdout:
        return None, "empty screenshot payload"
    raw = stdout.strip()
    # Strip a known data URI prefix if present.
    for prefix in _SCREENSHOT_DATA_URI_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    # base64 can include whitespace + newlines from CLI line-wrapping;
    # the upstream `base64.b64decode` strips them when validate=False
    # but we pass validate=True so we strip them ourselves first.
    raw = "".join(raw.split())
    if not raw:
        return None, "no base64 body after prefix strip"
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return None, "base64 decode failed"
    if len(decoded) < 16:
        return None, f"decoded payload too small ({len(decoded)} bytes)"
    return decoded, None


def _try_parse_tabs(
    stdout: str,
) -> tuple[tuple[BrowserTabInfo, ...], Optional[str]]:
    """Parse ``tab list --json`` output. Returns a tuple of tabs +
    optional parse error. Tolerates:

    * ``[{"index": N, "url": ..., "title": ..., "active": bool}, ...]``
    * ``{"tabs": [...]}``
    * Missing ``active`` flag (defaults to False)
    """
    if not stdout:
        return (), "empty tabs output"
    try:
        payload = json.loads(stdout)
    except (ValueError, json.JSONDecodeError):
        return (), "json parse failed"
    if isinstance(payload, Mapping):
        candidates = payload.get("tabs")
        if candidates is None:
            return (), "no 'tabs' key in mapping"
    elif isinstance(payload, Sequence) and not isinstance(payload, str):
        candidates = payload
    else:
        return (), "unexpected tabs payload shape"
    tabs: list[BrowserTabInfo] = []
    for i, entry in enumerate(candidates):
        if not isinstance(entry, Mapping):
            continue
        try:
            index = int(entry.get("index", i))
        except (TypeError, ValueError):
            index = i
        url = str(entry.get("url", "") or "")
        title = str(entry.get("title", "") or "")
        active = bool(entry.get("active", False))
        tabs.append(
            BrowserTabInfo(index=index, url=url, title=title, active=active)
        )
    return tuple(tabs), None


__all__ = [
    "BROWSER_JS_APPROVAL_KIND",
    "BROWSER_JS_REASON_CODE",
    "BROWSER_USE_BINARY_CANDIDATES",
    "BrowserActionResult",
    "BrowserAttributesResult",
    "BrowserBbox",
    "BrowserBboxResult",
    "BrowserElement",
    "BrowserEvalResult",
    "BrowserHtmlResult",
    "BrowserScreenshotResult",
    "BrowserState",
    "BrowserTabInfo",
    "BrowserTabsResult",
    "BrowserTextResult",
    "BrowserTitleResult",
    "BrowserUseResult",
    "BrowserUseTool",
    "BrowserValueResult",
    "BrowserWaitResult",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_WAIT_TIMEOUT_MS",
    "JsScriptAnalysis",
    "SCROLL_DIRECTIONS",
    "WAIT_SELECTOR_STATES",
    "analyze_js_script",
    "get_browser_use_tool",
    "reset_browser_use_tool_for_testing",
    "set_browser_use_tool",
]
