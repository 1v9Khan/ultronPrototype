"""Capability-tag namespace + tag-driven filtering (T5).

T5 (openclaw-clawhub catalog port; see ``THIRD_PARTY_NOTICES.md``).
A canonical, frozen tag namespace any ultron subsystem exposing a
capability can attach to its primitives. Tags are the lens through
which the orchestrator filters at intent time (don't expose a
wallet-signing tool to a non-confirmed voice utterance), the
gaming-mode VRAM-reclaim path knows which skills/services are
heavyweight vs lightweight, the safety validator can pre-block by
tag rather than per-call rule, and the voice ack-pool selector
picks the right tone.

The tag enum is intentionally small + frozen. Tag growth is a
versioned change (matches the upstream catalogue's "commit to a
small set, add only with explicit need" discipline). Where the
upstream marketplace uses tags like ``can-make-purchases`` /
``requires-wallet`` / ``requires-paid-service`` (financial-domain
patterns), ultron's tags focus on the voice-first single-user
runtime's actual gates: VRAM, latency, requires-input-device,
gaming-mode-safety.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)


class CapabilityTag(str, Enum):
    """Canonical capability-tag namespace.

    Tag categories:

    * **Resource requirements** — what the capability needs to run.
    * **Side-effect domain** — what the capability touches.
    * **Latency profile** — how long the capability typically takes.
    * **Gaming-mode safety** — whether the capability is safe to
      expose during gaming.
    * **Confirmation tier** — what user-side approval the
      capability needs.
    """

    # Resource requirements
    REQUIRES_INTERNET = "requires-internet"
    REQUIRES_MICROPHONE = "requires-microphone"
    REQUIRES_SCREEN_CAPTURE = "requires-screen-capture"
    REQUIRES_DESKTOP_INPUT = "requires-desktop-input"
    REQUIRES_CLIPBOARD = "requires-clipboard"
    REQUIRES_VLM = "requires-vlm"
    REQUIRES_LLM = "requires-llm"
    REQUIRES_CODING_BRIDGE = "requires-coding-bridge"
    REQUIRES_API_KEY = "requires-api-key"
    REQUIRES_PAID_SERVICE = "requires-paid-service"
    REQUIRES_BROWSER = "requires-browser"
    REQUIRES_BROWSER_CDP = "requires-browser-cdp"
    REQUIRES_NATIVE_DEPS = "requires-native-deps"

    # Side-effect domain
    EXECUTES_SHELL = "executes-shell"
    EXECUTES_PYTHON = "executes-python"
    EXECUTES_BINARY = "executes-binary"
    EXECUTES_FOREGROUND_WINDOW_ACTIONS = "executes-foreground-window-actions"
    EXECUTES_MULTI_STEP_LOOP = "executes-multi-step-loop"
    READS_SECRETS = "reads-secrets"
    READS_COOKIES = "reads-cookies"
    WRITES_FILES = "writes-files"
    POSTS_EXTERNALLY = "posts-externally"
    SELF_MODIFIES_TOOLKIT = "self-modifies-toolkit"

    # Network egress scope
    NETWORK_EGRESS_ALLOWED_DOMAINS = "network-egress-allowed-domains"
    NETWORK_EGRESS_UNRESTRICTED = "network-egress-unrestricted"

    # Latency profile
    LATENCY_SENSITIVE = "latency-sensitive"
    LATENCY_TOLERANT = "latency-tolerant"

    # Gaming-mode safety
    GAMING_MODE_SAFE = "gaming-mode-safe"
    GAMING_MODE_UNSAFE = "gaming-mode-unsafe"

    # Modality
    VOICE_ONLY = "voice-only"
    TEXT_ONLY = "text-only"

    # Confirmation tier (bridges to T2 two-phase approval + the
    # safety validator Cap-1..Cap-4 carveouts)
    REQUIRES_EXPLICIT_INTENT = "requires-explicit-intent"
    REQUIRES_OUT_OF_BAND_CONFIRM = "requires-out-of-band-confirm"
    K_CATEGORY_ADJACENT = "k-category-adjacent"
    K_CATEGORY_TERRITORY = "k-category-territory"


# Tag-set predicates that pull common combinations into one name.

GAMING_MODE_INCOMPATIBLE_TAGS: frozenset[CapabilityTag] = frozenset({
    CapabilityTag.REQUIRES_VLM,
    CapabilityTag.REQUIRES_BROWSER_CDP,
    CapabilityTag.EXECUTES_FOREGROUND_WINDOW_ACTIONS,
    CapabilityTag.LATENCY_SENSITIVE,
    CapabilityTag.GAMING_MODE_UNSAFE,
    CapabilityTag.REQUIRES_VLM,
})

K_PROTECTED_TAGS: frozenset[CapabilityTag] = frozenset({
    CapabilityTag.SELF_MODIFIES_TOOLKIT,
    CapabilityTag.K_CATEGORY_TERRITORY,
    CapabilityTag.K_CATEGORY_ADJACENT,
})


# ---------------------------------------------------------------------------
# Source-text + manifest derivation


_TAG_SIGNAL_PATTERNS: Mapping[CapabilityTag, tuple[re.Pattern[str], ...]] = {
    CapabilityTag.REQUIRES_INTERNET: (
        re.compile(r"\brequests\.\w+\("),
        re.compile(r"\burllib\.request\."),
        re.compile(r"\bhttpx\."),
        re.compile(r"\baiohttp\.ClientSession"),
        re.compile(r"\bhttp\.client\.HTTPS?Connection"),
    ),
    CapabilityTag.REQUIRES_DESKTOP_INPUT: (
        re.compile(r"\bpyautogui\."),
        re.compile(r"\bpywinauto\."),
        re.compile(r"\binput_control\."),
    ),
    CapabilityTag.REQUIRES_CLIPBOARD: (
        re.compile(r"\bwin32clipboard\."),
        re.compile(r"\bpyperclip\."),
    ),
    CapabilityTag.REQUIRES_SCREEN_CAPTURE: (
        re.compile(r"\bmss\.\w+"),
        re.compile(r"\bfrom\s+mss\s+import\b"),
        re.compile(r"\bimport\s+mss\b"),
        re.compile(r"\bcapture_monitor\b"),
        re.compile(r"\bScreenshot\b"),
    ),
    CapabilityTag.REQUIRES_VLM: (
        re.compile(r"\bMoondream2VLM\b"),
        re.compile(r"\bget_vlm\(\)"),
        re.compile(r"\bvlm\.describe\("),
    ),
    CapabilityTag.REQUIRES_LLM: (
        re.compile(r"\bLLMEngine\b"),
        re.compile(r"\bllama_cpp\."),
        re.compile(r"\bgenerate_stream\("),
    ),
    CapabilityTag.REQUIRES_CODING_BRIDGE: (
        re.compile(r"\bCodingTaskRunner\b"),
        re.compile(r"\bDirectClaudeCodeBridge\b"),
    ),
    CapabilityTag.EXECUTES_SHELL: (
        re.compile(r"\bsubprocess\.(?:Popen|run|call|check_call|check_output)\("),
        re.compile(r"\bos\.system\("),
        re.compile(r"\bshell=True\b"),
    ),
    CapabilityTag.EXECUTES_PYTHON: (
        re.compile(r"\bexec\("),
        re.compile(r"\bcompile\("),
        re.compile(r"\beval\("),
    ),
    CapabilityTag.READS_SECRETS: (
        # os.getenv("X_KEY") / os.getenv("X_TOKEN") / etc.
        re.compile(r"\bos\.getenv\(\s*['\"][A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD)\b"),
        # os.environ["X_KEY"] / os.environ.get("X_KEY")
        re.compile(
            r"\bos\.environ(?:\.\s*get)?\s*[\[\(]\s*['\"][A-Z_]*"
            r"(?:KEY|TOKEN|SECRET|PASSWORD)\b"
        ),
    ),
    CapabilityTag.WRITES_FILES: (
        re.compile(r"\bopen\([^)]*['\"]w['\"]"),
        re.compile(r"\bopen\([^)]*['\"]a['\"]"),
        re.compile(r"\bPath\([^)]*\)\.write"),
    ),
    CapabilityTag.REQUIRES_BROWSER: (
        re.compile(r"\bplaywright\."),
        re.compile(r"\bselenium\."),
    ),
    CapabilityTag.REQUIRES_BROWSER_CDP: (
        re.compile(r"\bcdp\b"),
        re.compile(r"\bremote_debugging_port\b"),
    ),
}


def derive_capability_tags(
    *,
    source: str = "",
    manifest: Optional[Mapping[str, object]] = None,
) -> tuple[CapabilityTag, ...]:
    """Return the auto-derived tag set for a ``(source, manifest)`` pair.

    Source-driven detection: each :class:`CapabilityTag` has an
    optional set of regex patterns; any match contributes the tag.
    Manifest-driven detection: explicit ``capabilityTags: [...]``
    in the manifest is honored; explicit-declared tags are
    preserved verbatim (the catalogue's "honor publisher
    declarations" contract). Manifest fields like
    ``requires.browser`` / ``requires.desktop`` / ``requires.api_key``
    are mapped to their canonical tags.

    Returns the union of explicit + derived tags, deduplicated, in
    sorted order.
    """
    found: set[CapabilityTag] = set()

    if source:
        for tag, patterns in _TAG_SIGNAL_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(source):
                    found.add(tag)
                    break

    if manifest is not None:
        explicit = manifest.get("capabilityTags")
        if isinstance(explicit, (list, tuple)):
            for value in explicit:
                if isinstance(value, str):
                    try:
                        found.add(CapabilityTag(value))
                    except ValueError:
                        # Unknown tag string; ignore (forwards-
                        # compatible with new upstream tags).
                        LOGGER.debug(
                            "Skipping unrecognised capability tag %r", value
                        )
        requires = manifest.get("requires")
        if isinstance(requires, Mapping):
            if requires.get("browser"):
                found.add(CapabilityTag.REQUIRES_BROWSER)
            if requires.get("desktop"):
                found.add(CapabilityTag.REQUIRES_DESKTOP_INPUT)
            if requires.get("vlm"):
                found.add(CapabilityTag.REQUIRES_VLM)
            if requires.get("nativeDeps") or requires.get("native_deps"):
                found.add(CapabilityTag.REQUIRES_NATIVE_DEPS)
            if requires.get("internet"):
                found.add(CapabilityTag.REQUIRES_INTERNET)
        env_vars = manifest.get("envVars")
        if isinstance(env_vars, (list, tuple)) and env_vars:
            found.add(CapabilityTag.REQUIRES_API_KEY)
            found.add(CapabilityTag.READS_SECRETS)

    return tuple(sorted(found, key=lambda t: t.value))


# ---------------------------------------------------------------------------
# Filtering helpers


@dataclass(frozen=True)
class TaggedCapability:
    """One taggable surface (skill / intent / MCP tool / slash command).

    Generic envelope -- the actual subsystem-specific data is
    carried in ``payload`` (caller-defined). The tag set is
    surfaced separately so :func:`filter_capabilities` can route
    without unpacking the payload.
    """

    name: str
    tags: tuple[CapabilityTag, ...] = ()
    payload: Mapping[str, object] = field(default_factory=dict)

    def has(self, tag: CapabilityTag) -> bool:
        return tag in self.tags

    def any_of(self, tags: Iterable[CapabilityTag]) -> bool:
        wanted = set(tags)
        return any(t in wanted for t in self.tags)

    def all_of(self, tags: Iterable[CapabilityTag]) -> bool:
        wanted = set(tags)
        return wanted.issubset(set(self.tags))


def filter_capabilities(
    items: Iterable[TaggedCapability],
    *,
    require: Iterable[CapabilityTag] = (),
    exclude: Iterable[CapabilityTag] = (),
    gaming_mode: bool = False,
    vlm_loaded: bool = True,
    has_internet: bool = True,
) -> tuple[TaggedCapability, ...]:
    """Return ``items`` filtered against per-call constraints.

    Filtering rules (all AND-combined):

    1. Every ``require`` tag must be present.
    2. No ``exclude`` tag may be present.
    3. When ``gaming_mode=True``, items with any tag in
       :data:`GAMING_MODE_INCOMPATIBLE_TAGS` are dropped.
    4. When ``vlm_loaded=False``, items tagged
       :attr:`CapabilityTag.REQUIRES_VLM` are dropped.
    5. When ``has_internet=False``, items tagged
       :attr:`CapabilityTag.REQUIRES_INTERNET` are dropped.

    Output preserves the input ordering of items that survive the
    filter.
    """
    require_set = set(require)
    exclude_set = set(exclude)
    out: list[TaggedCapability] = []
    for item in items:
        tags = set(item.tags)
        if not require_set.issubset(tags):
            continue
        if tags & exclude_set:
            continue
        if gaming_mode and tags & GAMING_MODE_INCOMPATIBLE_TAGS:
            continue
        if not vlm_loaded and CapabilityTag.REQUIRES_VLM in tags:
            continue
        if not has_internet and CapabilityTag.REQUIRES_INTERNET in tags:
            continue
        out.append(item)
    return tuple(out)


def is_gaming_mode_safe(tags: Iterable[CapabilityTag]) -> bool:
    """Return True iff none of ``tags`` is in :data:`GAMING_MODE_INCOMPATIBLE_TAGS`."""
    return not (set(tags) & GAMING_MODE_INCOMPATIBLE_TAGS)


def needs_explicit_intent(tags: Iterable[CapabilityTag]) -> bool:
    """Return True iff ``tags`` indicate the capability requires explicit-intent gating."""
    tag_set = set(tags)
    if CapabilityTag.REQUIRES_EXPLICIT_INTENT in tag_set:
        return True
    if CapabilityTag.REQUIRES_OUT_OF_BAND_CONFIRM in tag_set:
        return True
    if tag_set & K_PROTECTED_TAGS:
        return True
    return False


def is_voice_path_safe(
    tags: Iterable[CapabilityTag], *, ttft_budget_ms: int = 350
) -> bool:
    """Return True iff the capability is safe to expose on the voice TTFA budget.

    Capabilities tagged :attr:`LATENCY_SENSITIVE` get a "this may
    take a moment" pre-ack but are still safe; capabilities tagged
    :attr:`LATENCY_TOLERANT` are always safe; capabilities with
    neither tag are assumed safe (assume-safe-when-unknown). Pure
    helper; the orchestrator decides what to do with the result.
    """
    tag_set = set(tags)
    # The ttft_budget_ms arg is present for future tighter checks
    # (e.g., per-tag latency budgets). Currently the assumption is
    # any latency-sensitive capability still meets the budget after
    # a pre-ack, so we return True unconditionally. The argument
    # documents the contract.
    _ = ttft_budget_ms
    if CapabilityTag.LATENCY_TOLERANT in tag_set:
        return True
    return True


__all__ = [
    "CapabilityTag",
    "GAMING_MODE_INCOMPATIBLE_TAGS",
    "K_PROTECTED_TAGS",
    "TaggedCapability",
    "derive_capability_tags",
    "filter_capabilities",
    "is_gaming_mode_safe",
    "needs_explicit_intent",
    "is_voice_path_safe",
]
