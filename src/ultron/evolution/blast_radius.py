"""Blast-radius + policy-constraint spine for ultron's self-improvement loop.

Catalog 13 (clawhub-capability-evolver) clean-room synthesis. This is the
catalog's most directly-portable safety component -- the engine that
answers "how much did the change touch?" and "did it violate any rule?".
Every function here is a pure policy evaluation; the only impure piece is
the thin, fail-open git wrapper :func:`git_numstat`, and even that is
injectable so the policy core is fully testable without git.

The pieces:

* **constraint-counted files** -- a change file "counts" toward the blast
  radius only if it is real, intended content (under an include prefix /
  extension and not under an exclude prefix). Logs, caches, data dirs,
  binaries are excluded. This is how a *hollow commit* (the change wrote
  only bookkeeping, no real content -- metric gaming) is detected.
* **5-tier severity** -- ``within_limit`` / ``approaching_limit`` (warn) /
  ``exceeded`` / ``critical_overrun`` (2x) / ``hard_cap_breach`` (the
  absolute ceiling: >60 files OR >20000 lines).
* **critical-path protection** -- the Tier-3 hard wall. The loop can never
  count a change to ``src/`` (any ultron source), the safety validator,
  the audit log, the evolution engine itself, the voice-baseline models,
  or any Category-K file as legitimate. There is NO self-modify escape
  hatch (the upstream had one gated by ``EVOLVE_ALLOW_SELF_MODIFY``;
  ultron removes it).
* **ethics block** -- five regexes over a gene's strategy/summary text
  catch bypass-safety / covert-monitoring / social-engineering /
  human-exploitation / conceal-actions intent.
* **validation-command allowlist** -- only ``python`` / ``pytest`` /
  ``node`` / ``npm`` / ``npx`` prefixes; shell operators, eval flags
  (``node -e`` / ``python -c``), backtick / ``$()`` substitution, and
  network / destructive commands are rejected.
* **failure-mode classification** -- hard (never retried) vs soft
  (retryable validation failure).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

# Absolute ceilings -- no gene may ever exceed these, regardless of its own
# constraints. The environment's hard wall on a single change.
BLAST_RADIUS_HARD_CAP_FILES: int = 60
BLAST_RADIUS_HARD_CAP_LINES: int = 20_000

# Drift detector: an actual blast >= 3x the estimate (or <= 1/10) is a
# warning that the estimate was badly wrong.
BLAST_DRIFT_HIGH_RATIO: float = 3.0
BLAST_DRIFT_LOW_RATIO: float = 0.1


class BlastSeverity(str, Enum):
    """How far a change's blast radius exceeds its budget."""

    WITHIN_LIMIT = "within_limit"
    APPROACHING_LIMIT = "approaching_limit"
    EXCEEDED = "exceeded"
    CRITICAL_OVERRUN = "critical_overrun"
    HARD_CAP_BREACH = "hard_cap_breach"


@dataclass(frozen=True)
class CountedFilePolicy:
    """Rules deciding whether a changed file counts toward the blast radius.

    A file is counted iff it passes the exclude checks AND matches an
    include check. The default policy treats real source/config/docs as
    counted and excludes logs / data / models / caches / binaries.
    """

    exclude_prefixes: tuple[str, ...] = ()
    exclude_exact: tuple[str, ...] = ()
    exclude_regex: tuple[str, ...] = ()
    include_prefixes: tuple[str, ...] = ()
    include_exact: tuple[str, ...] = ()
    include_extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        compiled = tuple(re.compile(rx, re.IGNORECASE) for rx in self.exclude_regex)
        object.__setattr__(self, "_exclude_regex_compiled", compiled)

    @property
    def exclude_regex_compiled(self) -> tuple[re.Pattern[str], ...]:
        return getattr(self, "_exclude_regex_compiled")


#: The general-purpose policy (counts real source/config/docs/tests).
DEFAULT_COUNTED_FILE_POLICY = CountedFilePolicy(
    exclude_prefixes=(
        "logs/",
        "data/",
        "models/",
        ".git/",
        ".venv/",
        "node_modules/",
        "__pycache__/",
        "ultronVoiceAudio/",
        "htmlcov/",
        ".pytest_cache/",
        "build/",
        "dist/",
    ),
    exclude_exact=(),
    exclude_regex=(r"\.jsonl$", r"capsule", r"\.pyc$"),
    include_prefixes=("src/", "scripts/", "config/", "skills/", "prompts/", "docs/", "tests/"),
    include_exact=("config.yaml", "pyproject.toml", "README.md"),
    include_extensions=(
        ".py",
        ".pyi",
        ".md",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".txt",
        ".cfg",
        ".ini",
    ),
)


def proposal_policy(root: str = "data/evolution/skills") -> CountedFilePolicy:
    """A policy that counts ONLY files under ``root`` (the evolution
    proposal directory).

    The evolution loop writes its skill proposals here -- so a change that
    touches files but none under ``root`` is a hollow commit (it wrote
    somewhere it shouldn't have). Nothing is excluded; the include prefix
    is the sole gate.
    """
    return CountedFilePolicy(include_prefixes=(normalize_rel_path(root),))


# --- the Tier-3 / Category-K hard wall --------------------------------------
#
# Autonomous evolution is DATA-ONLY. It may never count a change to any of
# these as legitimate -- the loop's pre-flight rejects them outright. There
# is no self-modify exemption.

CRITICAL_PROTECTED_PREFIXES: tuple[str, ...] = (
    "src/",  # ALL ultron source -- the loop never writes code
    "config/",
    "scripts/",
    ".git/",
    ".github/",
    "models/",
    "ultronvoiceaudio/",  # voice baseline workshop
    "ultron_james_spader_mcu_6941/",  # RVC voice model
    "data/identity/",  # token-signing secret
    "data/observability/",
    "training/",
    "prompts/",
)

CRITICAL_PROTECTED_FILES: tuple[str, ...] = (
    "soul.md",
    "identity.md",
    "claude.md",
    "memory.md",
    "config.yaml",
    "pyproject.toml",
    "package.json",
    ".gitignore",
    ".env",
    "third_party_notices.md",
    "logs/safety_audit.jsonl",
    "docs/codebase_structure.md",
)


# --- ethics block -----------------------------------------------------------

ETHICS_BLOCK_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (
        "bypass_safety",
        re.compile(
            r"\b(bypass|disable|circumvent|remove|turn\s*off|defeat|weaken)\b.{0,40}"
            r"\b(safety|guardrail|security|ethic|constraint|protection|validator|sandbox)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "covert_monitoring",
        re.compile(
            r"\b(keylog(ger|ging)?|screen\s*capture|webcam\s*hijack|"
            r"secretly\s*record|record\b.{0,20}\b(mic|microphone|webcam|screen|keystrokes))\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "social_engineering",
        re.compile(
            r"\b(social\s*engineer\w*|phishing|spear[-\s]?phish\w*)\b.{0,40}"
            r"\b(attack|template|script|campaign|email|message)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "human_exploitation",
        re.compile(
            r"\b(exploit|hack|attack|compromise|manipulate|deceive)\b.{0,40}"
            r"\b(user|human|people|person|victim|target)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "conceal_actions",
        re.compile(
            r"\b(hide|conceal|obfuscate|cover\s*up|erase|wipe)\b.{0,40}"
            r"\b(action|behavior|behaviour|intent|log|trace|evidence|history)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


# --- validation-command allowlist -------------------------------------------

_ALLOWED_CMD_PREFIXES: tuple[str, ...] = (
    "python ",
    "python3 ",
    "py ",
    "pytest",
    "node ",
    "npm ",
    "npx ",
)
_BLOCKED_SHELL_OPERATORS: tuple[str, ...] = ("&&", "||", ";", "|", ">", "<", "`", "$(")
_BLOCKED_EVAL_FRAGMENTS: tuple[str, ...] = (
    "node -e",
    "node --eval",
    "node -p",
    "node --print",
    "python -c",
    "python3 -c",
    "py -c",
)
_BLOCKED_COMMANDS: tuple[str, ...] = (
    "rm ",
    "rm\t",
    "del ",
    "rmdir",
    "curl",
    "wget",
    "invoke-webrequest",
    "iwr ",
    "format ",
    "mkfs",
    "dd ",
)


# --- dataclasses ------------------------------------------------------------


@dataclass(frozen=True)
class NumstatRow:
    """One row of ``git diff --numstat`` output."""

    file: str
    added: int = 0
    deleted: int = 0


@dataclass(frozen=True)
class BlastComputation:
    """The measured blast radius of a change.

    ``files`` / ``lines`` count only constraint-counted (real) content;
    ``all_changed_files`` is every path git reported (used by the
    hollow-commit guard + forbidden/critical-path checks).
    """

    files: int = 0
    lines: int = 0
    changed_files: tuple[str, ...] = ()
    ignored_files: tuple[str, ...] = ()
    all_changed_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintCheckResult:
    """The verdict of :func:`check_constraints`."""

    ok: bool
    severity: BlastSeverity
    violations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FailureMode:
    """Classification of a cycle failure: hard (never retry) vs soft."""

    mode: str  # "hard" | "soft"
    reason_class: str
    retryable: bool


@dataclass(frozen=True)
class BlastEstimateComparison:
    """How far the actual blast drifted from the planned estimate."""

    estimate_files: int
    actual_files: int
    ratio: float
    drifted: bool
    message: str = ""


# --- path helpers -----------------------------------------------------------


def normalize_rel_path(path: str) -> str:
    """Normalise a repo-relative path: backslashes -> ``/``, strip leading
    ``./`` and ``/``, collapse ``//``, lowercase (paths are matched
    case-insensitively to be safe on Windows)."""
    p = str(path).replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    while "//" in p:
        p = p.replace("//", "/")
    return p.lower()


def is_constraint_counted_path(path: str, policy: Optional[CountedFilePolicy] = None) -> bool:
    """Whether ``path`` counts toward the blast radius under ``policy``.

    Exclude rules win first; then an include prefix / exact / extension
    match counts the file.
    """
    policy = policy or DEFAULT_COUNTED_FILE_POLICY
    p = normalize_rel_path(path)
    for pre in policy.exclude_prefixes:
        if p.startswith(normalize_rel_path(pre)):
            return False
    if p in {normalize_rel_path(e) for e in policy.exclude_exact}:
        return False
    for rx in policy.exclude_regex_compiled:
        if rx.search(p):
            return False
    if p in {normalize_rel_path(e) for e in policy.include_exact}:
        return True
    for pre in policy.include_prefixes:
        if p.startswith(normalize_rel_path(pre)):
            return True
    if policy.include_extensions:
        dot = p.rfind(".")
        ext = p[dot:] if dot != -1 else ""
        if ext in policy.include_extensions:
            return True
    return False


def is_forbidden_path(path: str, forbidden: Sequence[str]) -> bool:
    """Whether ``path`` is equal to or under any entry in ``forbidden``."""
    p = normalize_rel_path(path)
    for entry in forbidden:
        e = normalize_rel_path(entry)
        if not e:
            continue
        if p == e or p.startswith(e + "/") or p.startswith(e):
            return True
    return False


def is_critical_protected_path(path: str) -> bool:
    """Whether ``path`` is a Tier-3 / Category-K protected path the loop
    may NEVER autonomously modify. No exemption."""
    p = normalize_rel_path(path)
    if p in {normalize_rel_path(f) for f in CRITICAL_PROTECTED_FILES}:
        return True
    base = p.rsplit("/", 1)[-1]
    if base in {normalize_rel_path(f) for f in CRITICAL_PROTECTED_FILES}:
        return True
    for pre in CRITICAL_PROTECTED_PREFIXES:
        if p.startswith(normalize_rel_path(pre)):
            return True
    return False


# --- numstat parsing + blast computation ------------------------------------


def _resolve_rename(path: str) -> str:
    """Resolve a git rename path (``a => b`` or ``pre/{old => new}/post``)
    to the destination path."""
    path = path.strip()
    if "{" in path and "=>" in path and "}" in path:
        pre = path[: path.index("{")]
        inner = path[path.index("{") + 1 : path.index("}")]
        post = path[path.index("}") + 1 :]
        new = inner.split("=>")[-1].strip()
        combined = f"{pre}{new}{post}"
        while "//" in combined:
            combined = combined.replace("//", "/")
        return combined.strip()
    if "=>" in path:
        return path.split("=>")[-1].strip()
    return path


def parse_numstat_rows(text: str) -> tuple[NumstatRow, ...]:
    """Parse ``git diff --numstat`` output into rows.

    Each line is ``<added>\\t<deleted>\\t<path>``. Binary files (``-``)
    count as 0 added/deleted. Rename arrows are resolved to the
    destination path. Empty / malformed input yields ``()``.
    """
    if not text:
        return ()
    rows: list[NumstatRow] = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        added_s, deleted_s, path = parts[0], parts[1], parts[2]
        added = 0 if added_s.strip() in ("-", "") else _safe_int(added_s)
        deleted = 0 if deleted_s.strip() in ("-", "") else _safe_int(deleted_s)
        rows.append(NumstatRow(file=_resolve_rename(path), added=added, deleted=deleted))
    return tuple(rows)


def _safe_int(value: str) -> int:
    try:
        return max(0, int(value.strip()))
    except (TypeError, ValueError):
        return 0


def compute_blast_from_numstat(
    numstat_text: str,
    *,
    policy: Optional[CountedFilePolicy] = None,
) -> BlastComputation:
    """Compute the blast radius from a ``git diff --numstat`` string."""
    policy = policy or DEFAULT_COUNTED_FILE_POLICY
    rows = parse_numstat_rows(numstat_text)
    counted_files = 0
    counted_lines = 0
    changed: list[str] = []
    ignored: list[str] = []
    all_changed: list[str] = []
    seen: set[str] = set()
    for row in rows:
        f = normalize_rel_path(row.file)
        if f in seen:
            continue
        seen.add(f)
        all_changed.append(f)
        if is_constraint_counted_path(f, policy):
            counted_files += 1
            counted_lines += row.added + row.deleted
            changed.append(f)
        else:
            ignored.append(f)
    return BlastComputation(
        files=counted_files,
        lines=counted_lines,
        changed_files=tuple(changed),
        ignored_files=tuple(ignored),
        all_changed_files=tuple(all_changed),
    )


def _default_git_run(args: Sequence[str], repo_root: str, timeout: float) -> str:
    """Run a git command fail-open; return stdout or ``""``. CREATE_NO_WINDOW
    on Windows so no console flashes."""
    import subprocess

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=flags,
        )
        return res.stdout or ""
    except Exception:  # noqa: BLE001 -- git absence/error degrades to no blast
        return ""


def git_numstat(
    repo_root: str,
    *,
    since_ref: str = "HEAD",
    run: Optional[Callable[[Sequence[str]], str]] = None,
    timeout: float = 10.0,
) -> str:
    """Return ``git diff --numstat`` for the working tree vs ``since_ref``.

    Makes untracked files visible via ``add -A -N`` (intent-to-add, which
    does NOT stage content). ``run`` is injectable for tests. Fail-open.
    """
    runner = run if run is not None else (lambda a: _default_git_run(a, repo_root, timeout))
    try:
        runner(["add", "-A", "-N"])
        return runner(["diff", "--numstat", since_ref])
    except Exception:  # noqa: BLE001
        return ""


def compute_blast_radius(
    repo_root: str,
    *,
    since_ref: str = "HEAD",
    policy: Optional[CountedFilePolicy] = None,
    run: Optional[Callable[[Sequence[str]], str]] = None,
    timeout: float = 10.0,
) -> BlastComputation:
    """Convenience: read git numstat + compute the blast radius. Fail-open
    (returns an empty computation when git is unavailable)."""
    text = git_numstat(repo_root, since_ref=since_ref, run=run, timeout=timeout)
    return compute_blast_from_numstat(text, policy=policy)


# --- severity ---------------------------------------------------------------


def classify_blast_severity(*, files: int, lines: int, max_files: int) -> BlastSeverity:
    """Classify a blast radius against a per-gene ``max_files`` budget +
    the absolute hard caps."""
    if files > BLAST_RADIUS_HARD_CAP_FILES or lines > BLAST_RADIUS_HARD_CAP_LINES:
        return BlastSeverity.HARD_CAP_BREACH
    if max_files > 0 and files > max_files * 2:
        return BlastSeverity.CRITICAL_OVERRUN
    if files > max_files:
        return BlastSeverity.EXCEEDED
    if max_files > 0 and files >= max_files * 0.8:
        return BlastSeverity.APPROACHING_LIMIT
    return BlastSeverity.WITHIN_LIMIT


# --- ethics -----------------------------------------------------------------


def detect_ethics_violations(text: str) -> tuple[str, ...]:
    """Return the labels of any ethics-block patterns matched in ``text``."""
    if not text:
        return ()
    hits = [label for label, pattern in ETHICS_BLOCK_PATTERNS if pattern.search(text)]
    return tuple(hits)


def _gene_ethics_text(gene: Any) -> str:
    """Concatenate a gene's human-readable text for the ethics scan."""
    strategy = getattr(gene, "strategy", ()) or ()
    summary = getattr(gene, "summary", "") or ""
    preconditions = getattr(gene, "preconditions", ()) or ()
    return " ".join([*strategy, *preconditions, summary])


# --- validation command allowlist -------------------------------------------


def is_validation_command_allowed(cmd: str) -> bool:
    """Whether ``cmd`` is a safe validation command.

    Allows only ``python`` / ``pytest`` / ``node`` / ``npm`` / ``npx``
    prefixes; rejects shell operators, eval flags, command substitution,
    and network / destructive commands. Empty / whitespace -> rejected.
    """
    if not cmd or not cmd.strip():
        return False
    c = cmd.strip()
    low = c.lower()
    for op in _BLOCKED_SHELL_OPERATORS:
        if op in c:
            return False
    for frag in _BLOCKED_EVAL_FRAGMENTS:
        if frag in low:
            return False
    for blocked in _BLOCKED_COMMANDS:
        if low.startswith(blocked.strip()) or blocked in low:
            return False
    return any(low.startswith(prefix) for prefix in _ALLOWED_CMD_PREFIXES)


def filter_validation_commands(commands: Sequence[str]) -> tuple[str, ...]:
    """Keep only the allowed validation commands from ``commands``."""
    return tuple(c for c in commands if is_validation_command_allowed(c))


# --- constraint check -------------------------------------------------------


def check_constraints(
    *,
    gene: Any,
    blast: BlastComputation,
    blast_radius_estimate: Optional[int] = None,
    ethics_text: str = "",
) -> ConstraintCheckResult:
    """Evaluate a measured ``blast`` against a ``gene``'s constraints + the
    global safety rules.

    Hard violations (block the change): hard-cap breach, critical overrun,
    max-files exceeded, a forbidden path, a critical/Tier-3 protected path,
    a hollow commit (changed files but zero constraint-counted), or an
    ethics-block hit. Warnings (logged, not blocking): approaching the
    limit, or a large blast-estimate drift.
    """
    constraints = getattr(gene, "constraints", None)
    max_files = getattr(constraints, "max_files", BLAST_RADIUS_HARD_CAP_FILES)
    forbidden = getattr(constraints, "forbidden_paths", ()) or ()

    violations: list[str] = []
    warnings: list[str] = []
    severity = classify_blast_severity(files=blast.files, lines=blast.lines, max_files=max_files)

    if severity is BlastSeverity.HARD_CAP_BREACH:
        violations.append(
            f"hard_cap_breach: {blast.files} files / {blast.lines} lines exceeds "
            f"the {BLAST_RADIUS_HARD_CAP_FILES}-file / {BLAST_RADIUS_HARD_CAP_LINES}-line cap"
        )
    elif severity is BlastSeverity.CRITICAL_OVERRUN:
        violations.append(f"critical_overrun: {blast.files} files (> 2x max_files={max_files})")
    elif severity is BlastSeverity.EXCEEDED:
        violations.append(f"max_files_exceeded: {blast.files} > {max_files}")
    elif severity is BlastSeverity.APPROACHING_LIMIT:
        warnings.append(f"approaching_limit: {blast.files} of {max_files} files")

    for f in blast.all_changed_files:
        if is_critical_protected_path(f):
            violations.append(f"critical_path_modified: {f}")
        elif is_forbidden_path(f, forbidden):
            violations.append(f"forbidden_path: {f}")

    if len(blast.all_changed_files) > 0 and blast.files == 0:
        violations.append(
            f"hollow_commit: {len(blast.all_changed_files)} file(s) changed but 0 are "
            "constraint-counted content"
        )

    ethics_hits = detect_ethics_violations(ethics_text or _gene_ethics_text(gene))
    for hit in ethics_hits:
        violations.append(f"ethics:{hit}")

    if blast_radius_estimate is not None:
        cmp = compare_blast_estimate(blast_radius_estimate, blast.files)
        if cmp.drifted:
            warnings.append(cmp.message)

    return ConstraintCheckResult(
        ok=len(violations) == 0,
        severity=severity,
        violations=tuple(violations),
        warnings=tuple(warnings),
    )


# --- failure mode + reason --------------------------------------------------

_DESTRUCTIVE_VIOLATION_PREFIXES = (
    "critical_path_modified",
    "forbidden_path",
    "ethics:",
    "hard_cap_breach",
    "critical_overrun",
)


def classify_failure_mode(
    *,
    constraint_violations: Sequence[str] = (),
    protocol_violations: Sequence[str] = (),
    validation_failed: bool = False,
) -> FailureMode:
    """Classify a cycle failure.

    Protocol violations and destructive constraint violations
    (critical-path / forbidden-path / ethics / hard-cap / overrun) are
    HARD and never retried. Any other constraint violation is hard but
    classed ``constraint``. A validation-only failure is SOFT + retryable.
    """
    cv = list(constraint_violations)
    if protocol_violations:
        return FailureMode(mode="hard", reason_class="protocol", retryable=False)
    if any(v.startswith(_DESTRUCTIVE_VIOLATION_PREFIXES) for v in cv):
        return FailureMode(mode="hard", reason_class="constraint_destructive", retryable=False)
    if cv:
        return FailureMode(mode="hard", reason_class="constraint", retryable=False)
    if validation_failed:
        return FailureMode(mode="soft", reason_class="validation", retryable=True)
    return FailureMode(mode="soft", reason_class="unknown", retryable=True)


def build_failure_reason(
    constraint_check: Optional[ConstraintCheckResult] = None,
    *,
    validation_failed: bool = False,
    protocol_violations: Sequence[str] = (),
) -> str:
    """Build a human-readable failure reason from the failure inputs."""
    parts: list[str] = []
    if constraint_check is not None:
        parts.extend(constraint_check.violations)
    if protocol_violations:
        parts.extend(f"protocol: {p}" for p in protocol_violations)
    if validation_failed:
        parts.append("validation: command(s) failed")
    return "; ".join(parts) if parts else "unknown"


# --- breakdown + drift ------------------------------------------------------


def analyze_blast_breakdown(
    changed_files: Sequence[str], top_n: int = 5
) -> tuple[tuple[str, int], ...]:
    """Group changed files by their top-level directory, descending by
    count, capped at ``top_n``."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for f in changed_files:
        norm = normalize_rel_path(f)
        top = norm.split("/", 1)[0] if "/" in norm else "."
        counts[top] += 1
    return tuple(counts.most_common(top_n))


def compare_blast_estimate(estimate_files: int, actual_files: int) -> BlastEstimateComparison:
    """Detect blast-estimate drift (actual >= 3x the estimate, or <= 1/10)."""
    est = max(0, int(estimate_files))
    act = max(0, int(actual_files))
    if est <= 0:
        ratio = float("inf") if act > 0 else 1.0
    else:
        ratio = act / est
    drifted = ratio >= BLAST_DRIFT_HIGH_RATIO or (est > 0 and ratio <= BLAST_DRIFT_LOW_RATIO)
    message = ""
    if drifted:
        message = f"blast_estimate_drift: estimated {est} files, actual {act} (ratio {ratio:.2f})"
    return BlastEstimateComparison(
        estimate_files=est, actual_files=act, ratio=ratio, drifted=drifted, message=message
    )


__all__ = [
    "BLAST_RADIUS_HARD_CAP_FILES",
    "BLAST_RADIUS_HARD_CAP_LINES",
    "BlastSeverity",
    "CountedFilePolicy",
    "DEFAULT_COUNTED_FILE_POLICY",
    "proposal_policy",
    "CRITICAL_PROTECTED_PREFIXES",
    "CRITICAL_PROTECTED_FILES",
    "ETHICS_BLOCK_PATTERNS",
    "NumstatRow",
    "BlastComputation",
    "ConstraintCheckResult",
    "FailureMode",
    "BlastEstimateComparison",
    "normalize_rel_path",
    "is_constraint_counted_path",
    "is_forbidden_path",
    "is_critical_protected_path",
    "parse_numstat_rows",
    "compute_blast_from_numstat",
    "git_numstat",
    "compute_blast_radius",
    "classify_blast_severity",
    "detect_ethics_violations",
    "is_validation_command_allowed",
    "filter_validation_commands",
    "check_constraints",
    "classify_failure_mode",
    "build_failure_reason",
    "analyze_blast_breakdown",
    "compare_blast_estimate",
]
