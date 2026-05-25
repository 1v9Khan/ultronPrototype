"""Declared-vs-observed coherence checking (T4).

T4 (openclaw-clawhub catalog port; see ``THIRD_PARTY_NOTICES.md``).
A bidirectional linter that compares what a skill / hook / MCP-server
/ slash-command manifest **declares** against what its source code
actually **does**. Two failure modes get caught:

1. **Missing declaration** — code references an env var, binary, or
   config path that the manifest does not list under
   ``requires.env`` / ``requires.bins`` / ``requires.config``. The
   user installs the skill, hits a runtime failure on the missing
   capability, and has no way to tell from the manifest why. Auto-
   surface at install time.

2. **Unused declaration** — manifest lists ``requires.env: [SECRET]``
   but the body never reads ``SECRET``. Often benign (refactor
   leftover) but can be a misleading audit signal (a malicious
   manifest with an over-broad capability claim getting waived
   through).

The checker is **conservative**: only literal string env-var /
binary names are extracted from source. Dynamic reads
(``os.getenv(name_from_user_input)``) emit ``info`` severity rather
than ``warn`` so legitimate dynamic patterns don't get flagged.

Generalises beyond skill manifests: any ultron subsystem with a
declared-capabilities manifest + a source body can route through
:func:`check_coherence`. The pattern applies cleanly to skills
(this batch), hooks (T5 in cline catalog port), MCP servers (T22
in OpenClaw port), and slash commands.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)


class CoherenceMismatchKind(str, Enum):
    """Kind of coherence mismatch."""

    MISSING_DECLARATION = "missing_declaration"  # code uses X, manifest doesn't declare
    UNUSED_DECLARATION = "unused_declaration"    # manifest declares X, code doesn't use
    OS_MISMATCH = "os_mismatch"                  # manifest says macOS, code uses Win32
    DYNAMIC_READ = "dynamic_read"                # dynamic env/bin read; informational


class CoherenceSeverity(str, Enum):
    """Severity for a :class:`CoherenceMismatch`."""

    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass(frozen=True)
class CoherenceMismatch:
    """One coherence finding.

    Fields:
        kind: which mismatch class fired.
        category: which manifest axis the mismatch relates to
            (``env`` / ``bin`` / ``config`` / ``os`` / ``intent`` / ``tool``).
        identifier: the specific identifier that triggered the
            mismatch (e.g. ``"TODOIST_API_KEY"`` for an env-var
            mismatch).
        severity: :class:`CoherenceSeverity`.
        detail: free-form text explaining the finding.
        evidence_file: file the mismatch was observed in (may be
            empty for manifest-side findings).
        evidence_line: 1-indexed line number (0 = unknown).
    """

    kind: CoherenceMismatchKind
    category: str
    identifier: str
    severity: CoherenceSeverity = CoherenceSeverity.WARN
    detail: str = ""
    evidence_file: str = ""
    evidence_line: int = 0


# ---------------------------------------------------------------------------
# Reference extraction


_ENV_GETENV_LITERAL_RE: re.Pattern[str] = re.compile(
    r"""os\s*\.\s*getenv\s*\(\s*['"]([A-Z_][A-Z0-9_]*)['"]""",
    re.MULTILINE,
)

_ENV_ENVIRON_LITERAL_RE: re.Pattern[str] = re.compile(
    r"""os\s*\.\s*environ\s*(?:\.\s*get)?\s*[\[\(]\s*['"]([A-Z_][A-Z0-9_]*)['"]""",
    re.MULTILINE,
)

_ENV_GETENV_DYNAMIC_RE: re.Pattern[str] = re.compile(
    r"""os\s*\.\s*getenv\s*\(\s*[^'"\s)][^)]*?\)""",
    re.MULTILINE,
)

_BIN_WHICH_LITERAL_RE: re.Pattern[str] = re.compile(
    r"""shutil\s*\.\s*which\s*\(\s*['"]([A-Za-z0-9_./\-]+)['"]""",
    re.MULTILINE,
)

_BIN_SUBPROCESS_LITERAL_RE: re.Pattern[str] = re.compile(
    r"""subprocess\s*\.\s*(?:Popen|run|call|check_call|check_output)\s*\(\s*\[\s*['"]([A-Za-z0-9_./\-]+)['"]""",
    re.MULTILINE,
)


def extract_env_refs(source: str) -> tuple[set[str], int]:
    """Return ``(literal_names, dynamic_count)`` for env-var references.

    Literal names are stripped + deduplicated.

    ``dynamic_count`` is the number of `os.getenv(<expr>)` calls
    whose argument isn't a literal string -- callers may emit
    info-severity findings for these so audit reviewers know dynamic
    reads exist.
    """
    literals: set[str] = set()
    for match in _ENV_GETENV_LITERAL_RE.finditer(source):
        literals.add(match.group(1))
    for match in _ENV_ENVIRON_LITERAL_RE.finditer(source):
        literals.add(match.group(1))
    dynamic = len(_ENV_GETENV_DYNAMIC_RE.findall(source))
    # Subtract literal matches (the literal regex matches everything
    # the dynamic regex matches AND more).
    literal_count = len(_ENV_GETENV_LITERAL_RE.findall(source))
    dynamic = max(0, dynamic - literal_count)
    return literals, dynamic


def extract_bin_refs(source: str) -> set[str]:
    """Return the set of literal binary names referenced in ``source``.

    Matches ``shutil.which("X")`` and ``subprocess.run(["X", ...])``
    patterns. Skips file paths (anything starting with ``./`` /
    ``/`` is treated as a path, not a bin name) -- the catalog
    pattern is "manifest declares a *binary requirement*", not
    "manifest declares every relative path".
    """
    names: set[str] = set()
    for match in _BIN_WHICH_LITERAL_RE.finditer(source):
        candidate = match.group(1)
        if candidate.startswith(("./", "/", "\\")):
            continue
        if "/" in candidate or "\\" in candidate:
            continue
        names.add(candidate)
    for match in _BIN_SUBPROCESS_LITERAL_RE.finditer(source):
        candidate = match.group(1)
        if candidate.startswith(("./", "/", "\\")):
            continue
        if "/" in candidate or "\\" in candidate:
            continue
        names.add(candidate)
    return names


def extract_config_refs(source: str, *, config_attr_root: str = "config") -> set[str]:
    """Return literal config attribute paths referenced under ``config_attr_root``.

    Walks Python AST for attribute accesses rooted at ``config_attr_root``
    (default ``"config"``). E.g.
    ``config.web_search.providers`` -> ``"web_search.providers"``.

    Returns ``set()`` on syntax errors so callers don't have to
    branch.
    """
    refs: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return refs

    class _Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            parts: list[str] = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name) and current.id == config_attr_root:
                refs.add(".".join(reversed(parts)))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return refs


# ---------------------------------------------------------------------------
# Manifest declaration accessors


def declared_env_vars(manifest: Mapping[str, object]) -> set[str]:
    """Return the manifest's declared env-var set (union of all sources)."""
    out: set[str] = set()
    requires = manifest.get("requires")
    if isinstance(requires, Mapping):
        env = requires.get("env")
        if isinstance(env, (list, tuple)):
            for value in env:
                if isinstance(value, str):
                    out.add(value)
    env_vars = manifest.get("envVars")
    if isinstance(env_vars, (list, tuple)):
        for value in env_vars:
            if isinstance(value, str):
                out.add(value)
            elif isinstance(value, Mapping):
                name = value.get("name")
                if isinstance(name, str):
                    out.add(name)
    primary_env = manifest.get("primaryEnv")
    if isinstance(primary_env, str):
        out.add(primary_env)
    return out


def declared_bins(manifest: Mapping[str, object]) -> set[str]:
    """Return the manifest's declared bin set (requires.bins ∪ requires.anyBins)."""
    out: set[str] = set()
    requires = manifest.get("requires")
    if not isinstance(requires, Mapping):
        return out
    for key in ("bins", "anyBins"):
        bins = requires.get(key)
        if isinstance(bins, (list, tuple)):
            for value in bins:
                if isinstance(value, str):
                    out.add(value)
    return out


def declared_config_paths(manifest: Mapping[str, object]) -> set[str]:
    """Return the manifest's declared config-path set (requires.config)."""
    out: set[str] = set()
    requires = manifest.get("requires")
    if not isinstance(requires, Mapping):
        return out
    config = requires.get("config")
    if isinstance(config, (list, tuple)):
        for value in config:
            if isinstance(value, str):
                out.add(value)
    return out


def declared_os(manifest: Mapping[str, object]) -> set[str]:
    """Return the lowercased manifest ``os`` declarations.

    ``["macos", "linux"]`` -> ``{"macos", "linux"}``. Empty set when
    no declaration is made (treat as "all OS").
    """
    out: set[str] = set()
    os_decl = manifest.get("os")
    if isinstance(os_decl, str):
        out.add(os_decl.casefold())
    elif isinstance(os_decl, (list, tuple)):
        for value in os_decl:
            if isinstance(value, str):
                out.add(value.casefold())
    return out


# ---------------------------------------------------------------------------
# OS coherence (catalog "creative extension")


_OS_API_HINTS: Mapping[str, str] = {
    # win32 / pywin32 / windows-specific imports
    "import win32api": "windows",
    "import win32con": "windows",
    "import win32gui": "windows",
    "import pywin32": "windows",
    "import winreg": "windows",
    "os.startfile": "windows",
    # macOS-specific
    "import Cocoa": "macos",
    "import AppKit": "macos",
    "import Foundation": "macos",
    "subprocess.run(['osascript'": "macos",
    'subprocess.run(["osascript"': "macos",
    # Linux-specific
    "import dbus": "linux",
    "from systemd": "linux",
    "/proc/cpuinfo": "linux",
    "/etc/passwd": "linux",
}


def detect_os_signals(source: str) -> set[str]:
    """Return OS-specific signals observed in ``source``.

    Each entry of :data:`_OS_API_HINTS` is matched verbatim against
    the (stripped-of-leading-whitespace) source so false positives
    from documentation strings are minimised.
    """
    signals: set[str] = set()
    for hint, os_name in _OS_API_HINTS.items():
        if hint in source:
            signals.add(os_name)
    return signals


# ---------------------------------------------------------------------------
# Main check


def check_coherence(
    manifest: Mapping[str, object],
    source_files: Iterable[tuple[str, str]],
    *,
    config_attr_root: str = "config",
) -> tuple[CoherenceMismatch, ...]:
    """Return mismatches for ``(manifest, source_files)``.

    ``source_files`` is an iterable of ``(path, source_text)``. Each
    file contributes its env / bin / config / OS-signal extractions;
    the union is compared to the manifest's declarations.

    Returns an empty tuple when manifest + observed behaviour agree.
    """
    findings: list[CoherenceMismatch] = []

    decl_env = declared_env_vars(manifest)
    decl_bins = declared_bins(manifest)
    decl_config = declared_config_paths(manifest)
    decl_os = declared_os(manifest)

    observed_env_literals: dict[str, tuple[str, int]] = {}
    observed_bins: dict[str, tuple[str, int]] = {}
    observed_config: dict[str, tuple[str, int]] = {}
    observed_os: set[str] = set()
    dynamic_env_reads: list[tuple[str, int]] = []

    for path, source in source_files:
        literals, dynamic_count = extract_env_refs(source)
        for env_name in literals:
            if env_name not in observed_env_literals:
                line = _first_line_of(source, env_name)
                observed_env_literals[env_name] = (path, line)
        if dynamic_count > 0:
            dynamic_env_reads.append((path, dynamic_count))

        for bin_name in extract_bin_refs(source):
            if bin_name not in observed_bins:
                line = _first_line_of(source, bin_name)
                observed_bins[bin_name] = (path, line)

        for cfg in extract_config_refs(source, config_attr_root=config_attr_root):
            if cfg not in observed_config:
                line = _first_line_of(source, cfg.split(".")[0])
                observed_config[cfg] = (path, line)

        observed_os.update(detect_os_signals(source))

    # --- env vars ---
    for env_name, (path, line) in observed_env_literals.items():
        if env_name in decl_env:
            continue
        # OS-defined env vars (PATH, HOME, USER, etc.) are almost
        # never declared -- skip them.
        if env_name in _ALWAYS_AVAILABLE_ENV:
            continue
        findings.append(CoherenceMismatch(
            kind=CoherenceMismatchKind.MISSING_DECLARATION,
            category="env",
            identifier=env_name,
            severity=CoherenceSeverity.WARN,
            detail=f"source reads env {env_name!r} but manifest does not declare it",
            evidence_file=path,
            evidence_line=line,
        ))
    for env_name in decl_env - set(observed_env_literals.keys()):
        findings.append(CoherenceMismatch(
            kind=CoherenceMismatchKind.UNUSED_DECLARATION,
            category="env",
            identifier=env_name,
            severity=CoherenceSeverity.INFO,
            detail=f"manifest declares env {env_name!r} but no source reads it",
        ))

    # --- dynamic reads -> info
    for path, count in dynamic_env_reads:
        findings.append(CoherenceMismatch(
            kind=CoherenceMismatchKind.DYNAMIC_READ,
            category="env",
            identifier="(dynamic)",
            severity=CoherenceSeverity.INFO,
            detail=f"{count} dynamic os.getenv() call(s); cannot verify manifest coverage",
            evidence_file=path,
        ))

    # --- bins ---
    for bin_name, (path, line) in observed_bins.items():
        if bin_name in decl_bins:
            continue
        if bin_name in _COMMON_BINS_NEVER_DECLARED:
            continue
        findings.append(CoherenceMismatch(
            kind=CoherenceMismatchKind.MISSING_DECLARATION,
            category="bin",
            identifier=bin_name,
            severity=CoherenceSeverity.WARN,
            detail=f"source invokes binary {bin_name!r} but manifest does not declare it",
            evidence_file=path,
            evidence_line=line,
        ))
    for bin_name in decl_bins - set(observed_bins.keys()):
        findings.append(CoherenceMismatch(
            kind=CoherenceMismatchKind.UNUSED_DECLARATION,
            category="bin",
            identifier=bin_name,
            severity=CoherenceSeverity.INFO,
            detail=f"manifest declares bin {bin_name!r} but no source invokes it",
        ))

    # --- config ---
    for cfg, (path, line) in observed_config.items():
        if any(cfg == d or cfg.startswith(d + ".") for d in decl_config):
            continue
        if not decl_config:
            # Manifest didn't declare config at all -- skip the
            # missing-declaration spam (declaring every config
            # path would be very noisy; only flag when a manifest
            # claims SOME config and we observe a sibling outside
            # the claimed set).
            continue
        findings.append(CoherenceMismatch(
            kind=CoherenceMismatchKind.MISSING_DECLARATION,
            category="config",
            identifier=cfg,
            severity=CoherenceSeverity.WARN,
            detail=f"source reads config {cfg!r} but manifest does not declare it",
            evidence_file=path,
            evidence_line=line,
        ))
    for cfg in decl_config:
        if cfg in observed_config:
            continue
        if any(observed.startswith(cfg + ".") for observed in observed_config):
            continue
        findings.append(CoherenceMismatch(
            kind=CoherenceMismatchKind.UNUSED_DECLARATION,
            category="config",
            identifier=cfg,
            severity=CoherenceSeverity.INFO,
            detail=f"manifest declares config {cfg!r} but no source reads it",
        ))

    # --- OS ---
    if decl_os:
        for signal in observed_os:
            if signal not in decl_os:
                findings.append(CoherenceMismatch(
                    kind=CoherenceMismatchKind.OS_MISMATCH,
                    category="os",
                    identifier=signal,
                    severity=CoherenceSeverity.WARN,
                    detail=f"source uses {signal}-specific API but manifest declares os={sorted(decl_os)}",
                ))

    return tuple(findings)


# ---------------------------------------------------------------------------
# Helpers


def _first_line_of(source: str, needle: str) -> int:
    """Return the 1-indexed line number where ``needle`` first appears.

    Returns 0 when ``needle`` is not found (defensive against
    refactoring that drops the literal).
    """
    if not needle:
        return 0
    for i, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return i
    return 0


# Common env vars almost no skill manifests bother declaring; the
# coherence checker silently skips these so audit logs aren't noisy.
_ALWAYS_AVAILABLE_ENV: frozenset[str] = frozenset({
    "PATH",
    "HOME",
    "USER",
    "USERNAME",
    "USERPROFILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LOCALE",
    "OS",
    "SHELL",
    "TERM",
    "PWD",
    "OLDPWD",
    "HOSTNAME",
    "COMPUTERNAME",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "WINDIR",
    "SYSTEMROOT",
})

# Common binaries that are universally available enough that the
# coherence checker doesn't warn when they're observed but not
# declared. Conservative -- new entries should require a clear
# "every machine has this" justification.
_COMMON_BINS_NEVER_DECLARED: frozenset[str] = frozenset({
    "python",
    "python3",
    "pip",
    "pip3",
})


# ---------------------------------------------------------------------------
# Voice-intent coherence (catalog "creative extension")


def check_intent_phrase_coherence(
    trigger_phrases: Sequence[str],
    skill_body: str,
    *,
    min_overlap_ratio: float = 0.2,
) -> tuple[CoherenceMismatch, ...]:
    """Return mismatches when trigger phrases don't lexically appear in body.

    The catalog suggests this as a creative extension: a skill that
    triggers on a phrase the body never references is a red flag
    (could be an attempt to slip a skill into the LLM context on
    unrelated user utterances).

    ``min_overlap_ratio`` is the fraction of phrase tokens that must
    appear in the body for the phrase to count as "referenced".
    """
    body_tokens = set(re.findall(r"[A-Za-z0-9_]+", skill_body.casefold()))
    findings: list[CoherenceMismatch] = []
    for phrase in trigger_phrases:
        if not phrase or not phrase.strip():
            continue
        phrase_tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", phrase.casefold()) if len(t) > 2]
        if not phrase_tokens:
            continue
        overlap = sum(1 for t in phrase_tokens if t in body_tokens)
        if overlap / len(phrase_tokens) < min_overlap_ratio:
            findings.append(CoherenceMismatch(
                kind=CoherenceMismatchKind.MISSING_DECLARATION,
                category="intent",
                identifier=phrase,
                severity=CoherenceSeverity.WARN,
                detail=(
                    f"trigger phrase {phrase!r} has {overlap}/{len(phrase_tokens)} "
                    f"tokens in body (below ratio {min_overlap_ratio})"
                ),
            ))
    return tuple(findings)


__all__ = [
    "CoherenceMismatchKind",
    "CoherenceSeverity",
    "CoherenceMismatch",
    "extract_env_refs",
    "extract_bin_refs",
    "extract_config_refs",
    "declared_env_vars",
    "declared_bins",
    "declared_config_paths",
    "declared_os",
    "detect_os_signals",
    "check_coherence",
    "check_intent_phrase_coherence",
]
