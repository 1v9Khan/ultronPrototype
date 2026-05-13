"""Capability carve-out rules (Cap-1 through Cap-4).

The carve-outs encode the *positive* surface: things the validator
explicitly allows because they're declared product capabilities,
plus the *negative* sub-rules that bound those allowances.

Cap-1 -- screen_context (all-monitors realtime visual understanding):
        allowed read; OUT-gate blocks captures from leaving the local
        machine; CAP-gate blocks captures of DRM/protected windows,
        UAC consent dialogs, cross-session windows.

Cap-2 -- app_launch (open arbitrary app on chosen monitor): allowed
        for paths in approved program dirs with allowed-flag
        schemas; blocks elevation flags, Chromium debug ports,
        unknown flags, launches from Temp/Downloads.

Cap-3 -- browser_default_profile (Chrome with Default profile):
        allowed launch + navigation + screenshots; blocks CDP attach
        to user profile, action-verb clicks on authenticated pages,
        OAuth consent screens, payment-domain form submissions,
        autofill credential acceptance.

Cap-4 -- window_management: allowed move/resize/focus on user-owned
        windows; blocks moving security dialogs, HWND_TOPMOST over
        security windows, synthetic input near a freshly-appeared
        UAC/security window.

Phase 4 implementation focuses on the CAP/OUT-gates that don't
depend on capability infrastructure that hasn't shipped yet
(screen_context cache dir, per-app flag allowlists). The carve-out
RULES enforce; the CARVE-OUTS themselves are policy entries.

Cap-1 outflow gates rely on the :class:`TaintTracker` (Phase 5).
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
)
from ultron.safety.validator import Verdict


def build_capability_rules() -> list[Rule]:
    """Factory for capability carve-out rules.

    Returns the SUB-rules that bound the capabilities (not the
    capabilities themselves; those are policy-level allowances
    expressed by the absence of a deny rule).
    """
    return [
        # ----- Cap-1: screen_context bounds -----
        # Cap-1 CAP: capture protected windows.
        CommandPatternRule(
            rule_id="Cap-1.protected",
            description="Cap-1: capturing DRM/protected windows or UAC dialogs",
            category="Cap",
            patterns=[
                # Attempt to bypass SetWindowDisplayAffinity
                r"\bSetWindowDisplayAffinity\b.*\bWDA_NONE\b",
                # Cross-session capture via WTSQueryUserToken
                r"\bWTSQueryUserToken\b",
                # Reading the secure desktop's framebuffer
                r"\bWinSta0\\\\Winlogon\b",
            ],
        ),

        # ----- Cap-2: app_launch bounds -----
        # Cap-2 CAP: Chromium internals-exposure flags.
        CommandPatternRule(
            rule_id="Cap-2.chromium-internals",
            description="Cap-2: Chromium launch flags exposing internals",
            category="Cap",
            patterns=[
                r"--remote-debugging-port=\d+",
                r"--remote-debugging-pipe\b",
                r"--disable-web-security\b",
                r"--load-extension=",
                # Custom user-data-dir pointing outside the user's
                # normal profile paths.
                r"--user-data-dir=(?!.*(?:appdata|users)).*",
            ],
        ),
        # Cap-2 CAP: launch from Temp/Downloads.
        PathPatternRule(
            rule_id="Cap-2.launch-from-temp",
            description="Cap-2: launching .exe from Temp/Downloads/AppData/Local/Temp",
            category="Cap",
            patterns=[
                r"\.exe$.*/(temp|downloads|tmp)/",
                r"/(temp|downloads|tmp)/[^/]+\.exe$",
                r"appdata/local/temp/[^/]+\.exe$",
            ],
            verdict_on_match=Verdict.BLOCK_HARD,
        ),
        # Cap-2 CAP: .lnk shortcuts resolving outside approved dirs.
        # Pure-path rule; the path resolver canonicalises the .lnk target.
        # If the resolved canonical falls in Temp/Downloads, treat as Cap-2 launch.
        # (Same pattern as above; .lnk resolution happens upstream in the
        # resolver.)

        # ----- Cap-3: browser_default_profile bounds -----
        # Cap-3 CAP: CDP attach (also in C12).
        CommandPatternRule(
            rule_id="Cap-3.cdp-attach",
            description="Cap-3: CDP attach to user-owned Chrome profile",
            category="Cap",
            patterns=[
                r"\bchrome_remote_interface\b",
                r"--remote-debugging-port=",
                r"\bcdp\.attach\b",
                r"\bpychrome\.Browser\b",
            ],
        ),
        # Cap-3 CAP: clicking action-verb elements on authenticated pages.
        # We detect this from the argument content -- the model is
        # passing the visible button label or aria-label to the click
        # tool. Action-verb match -> NEEDS_EXPLICIT_INTENT.
        CommandPatternRule(
            rule_id="Cap-3.action-verb-click",
            description="Cap-3: click on high-stakes action verb in authenticated context",
            category="Cap",
            patterns=[
                # Verb words inside an argument value (label/aria-label).
                # Phrasing in alphabetical order with word boundaries.
                r"['\"][Aa]llow\b",
                r"['\"][Aa]pprove\b",
                r"['\"][Aa]uthorize\b",
                r"['\"][Bb]uy\b",
                r"['\"][Cc]onfirm\s+[Oo]rder\b",
                r"['\"][Dd]elete\s+[Aa]ccount\b",
                r"['\"][Gg]rant\s+[Aa]ccess\b",
                r"['\"][Pp]ay\b",
                r"['\"][Pp]lace\s+[Oo]rder\b",
                r"['\"][Ss]end\b",
                r"['\"][Ss]end\s+[Mm]oney\b",
                r"['\"][Ss]ign\s+and\s+[Ss]end\b",
                r"['\"][Tt]ransfer\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # Cap-3 CAP: OAuth consent screens.
        CommandPatternRule(
            rule_id="Cap-3.oauth-consent",
            description="Cap-3: OAuth consent screens",
            category="Cap",
            patterns=[
                r"https?://accounts\.google\.com/o/oauth2\b",
                r"https?://login\.microsoftonline\.com/.*\boauth2\b",
                r"https?://github\.com/login/oauth/authorize\b",
                r"https?://[^/]+/oauth/authorize\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # Cap-3 CAP: payment-processor form submissions.
        CommandPatternRule(
            rule_id="Cap-3.payment-domain",
            description="Cap-3: form submissions on payment domains",
            category="Cap",
            patterns=[
                r"https?://[^/]*\bstripe\.com\b",
                r"https?://[^/]*\bpaypal\.com\b",
                r"https?://[^/]*\bsquareup\.com\b",
                # Bank domains -- user can add their bank list to the
                # policy; here we match common patterns.
                r"https?://[^/]*\bchase\.com\b",
                r"https?://[^/]*\bbankofamerica\.com\b",
                r"https?://[^/]*\bwellsfargo\.com\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),

        # ----- Cap-4: window_management bounds -----
        # Cap-4 CAP: moving security dialogs off-screen or behind.
        CommandPatternRule(
            rule_id="Cap-4.security-dialog-move",
            description="Cap-4: moving / occluding security dialogs",
            category="Cap",
            patterns=[
                # UAC / Windows credential / Defender dialog class names
                r"\bSetWindowPos\b.*\b(CredentialUIControl|UACWindow|SecurityHealthSystray)\b",
                r"\bMoveWindow\b.*\b(CredentialUIControl|UACWindow|SecurityHealthSystray)\b",
                # HWND_TOPMOST over security-relevant windows
                r"\bSetWindowPos\b.*\bHWND_TOPMOST\b.*\b(UAC|SecurityHealth|CredentialUI)\b",
            ],
        ),
        # Cap-4 CAP: synthetic input into UAC / security windows.
        CommandPatternRule(
            rule_id="Cap-4.synthetic-input-security",
            description="Cap-4: synthetic input near a UAC / security-class window",
            category="Cap",
            patterns=[
                r"\bSendInput\b.*\bUAC\b",
                r"\bkeybd_event\b.*\bUAC\b",
                r"\bmouse_event\b.*\b(UAC|SecurityHealth)\b",
            ],
        ),
    ]
