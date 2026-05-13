"""Policy loader for the runtime tool-call validator.

The user's restriction list (2026-05-12) is the policy. This module
exposes:

* :class:`Policy` -- the loaded, validated policy. Includes a flag
  per rule (enabled / disabled), each category's sandbox paths,
  protected-files allowlist, etc.
* :func:`load_policy()` -- factory. Reads from config.yaml's
  ``safety`` section + the in-code defaults.

Why both config-driven AND in-code defaults: the rule IDs (K1, A3,
D7, etc.) are baked into source -- a rule's existence isn't config-
controlled. What IS config-controlled is whether each rule is
enabled, plus the parameters (paths, patterns, thresholds) each
rule consults. That keeps the user's policy editable without code
changes while preventing accidental removal of a rule.

Editing the policy is gated by Category K (specifically K1: the
validator can't disable itself via config). The K rules check
config-file writes against this policy's `protected_files` list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ultron.safety.policy")


@dataclass(frozen=True)
class Policy:
    """Loaded validator policy. Frozen so rules can't mutate it at runtime.

    Attributes:
        enabled: Master switch. False = validator is a permissive
            no-op (every call returns ALLOW). Set false to disable
            the entire safety layer; individual rules are toggled
            via ``rule_enabled``.
        rule_enabled: per-rule-id enable map. Missing keys default
            to True (rules are on unless explicitly disabled).
        sandbox_roots: paths under which destructive operations are
            allowed (e.g. ``data/sandbox/<project>/``). Resolved to
            absolute paths at policy load.
        protected_files: paths that are NEVER writeable by the model
            regardless of capability. Category K builds this list
            from the user's restriction list plus the validator's
            own files.
        protected_dirs: directory trees that are NEVER writeable
            (recursive). Used by Category A's system-directory
            blocks (C:\Windows\, etc.).
        screen_cache_dir: where the Cap-1 screen-context capability
            is allowed to write captured frames. Outflow outside
            this dir is OUT-gate blocked.
        approved_outbound_apis: hostnames the model is allowed to
            reach via outbound network calls (Brave, Jina,
            Anthropic). Empty list means "no outbound checks" --
            distinct from "block all".

    See `src/ultron/safety/rules/category_k.py` for how Category K
    consumes ``protected_files`` / ``protected_dirs``.
    """

    enabled: bool = True
    rule_enabled: dict[str, bool] = field(default_factory=dict)
    sandbox_roots: list[Path] = field(default_factory=list)
    protected_files: list[Path] = field(default_factory=list)
    protected_dirs: list[Path] = field(default_factory=list)
    screen_cache_dir: Optional[Path] = None
    approved_outbound_apis: list[str] = field(default_factory=list)

    def is_rule_enabled(self, rule_id: str) -> bool:
        """Lookup with default-True semantics.

        Rules are ON unless the user explicitly turned them off.
        Defending the model from unexpected new rules slipping in
        without re-review isn't the goal here -- the goal is
        making sure load-bearing safety rules stay on.
        """
        return self.rule_enabled.get(rule_id, True)


# Default protected files / directories for Category K. These are the
# validator's own files plus the items the user marked ★ in Category K
# of the 2026-05-12 restriction list. Paths are PROJECT_ROOT-relative;
# the path resolver canonicalises them at load time.
_DEFAULT_PROTECTED_FILES_RELATIVE = [
    # K1: config.yaml entries that govern the validator. We protect the
    # entire config.yaml -- granular per-key edits could land here in a
    # later phase but for Phase 2 the whole file is off-limits.
    "config.yaml",
    # K2: voice character / model assets.
    "ultronVoiceAudio/Ultron_vocals_mono_v1.wav",
    # K3: validator + existing block-and-revise modules.
    "src/ultron/safety/__init__.py",
    "src/ultron/safety/validator.py",
    "src/ultron/safety/path_resolver.py",
    "src/ultron/safety/audit.py",
    "src/ultron/safety/policy.py",
    "src/ultron/safety/intent.py",
    "src/ultron/safety/taint.py",
    "src/ultron/safety/rules/base.py",
    "src/ultron/safety/rules/category_k.py",
    "src/ultron/openclaw_routing/block_and_revise.py",
    "src/ultron/coding/canonical_monitor.py",
    # K4: audit log writers. The log files themselves are protected;
    # the writer code that produces them is also protected.
    "logs/errors.jsonl",
    "logs/safety_audit.jsonl",
    "src/ultron/resilience/error_log.py",
    # K5: OpenClaw bridge.
    "src/ultron/openclaw_bridge/client.py",
    "src/ultron/openclaw_bridge/holder.py",
    # K7: dependency manifests.
    "requirements.txt",
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    # K8: AI-pipeline-ingested files (prompt-injection-via-memory).
    "CLAUDE.md",
    "docs/codebase_structure.md",
    "docs/architecture.md",
    "docs/configuration.md",
    "docs/error_handling.md",
    "docs/operations.md",
    "docs/development.md",
    "docs/routing.md",
    "docs/memory_architecture.md",
    "docs/standing_orders.md",
    # K10: MCP entry scripts.
    "scripts/run_ultron_mcp_for_openclaw.py",
    "scripts/start_llamacpp_server.py",
    "scripts/supervised_llamacpp_server.py",
]

# Directories the user-locked SOUL.md, RVC, Piper, XTTS models live
# in. ``ultron_james_spader_mcu_6941/`` is the RVC voice model dir;
# ``models/piper/`` is the Piper TTS; ``models/`` overall holds the
# LLM GGUFs + Smart Turn ONNX which K2 / S5 jointly protect.
_DEFAULT_PROTECTED_DIRS_RELATIVE = [
    "models",
    "ultron_james_spader_mcu_6941",
    "ultronVoiceAudio",
    # K9 covers PowerShell profile / shell rc files which live in the
    # user's home, not the project. Those rules check absolute paths
    # in the rule logic, not via this list.
]


def load_policy(
    *,
    enabled: bool = True,
    rule_overrides: Optional[dict[str, bool]] = None,
    extra_protected_files: Optional[list[str]] = None,
    extra_protected_dirs: Optional[list[str]] = None,
    sandbox_roots: Optional[list[str]] = None,
    screen_cache_dir: Optional[str] = None,
    approved_outbound_apis: Optional[list[str]] = None,
    project_root: Optional[Path] = None,
) -> Policy:
    """Build a :class:`Policy` from defaults + user overrides.

    The defaults bake in the Category K protected-files list (the
    user's restriction list under "Ultron self-protection (meta)").
    Callers add their own entries via the ``extra_*`` parameters.

    Args:
        enabled: Master switch.
        rule_overrides: per-rule-id enable map. Anything missing
            stays at the default-True.
        extra_protected_files: project-root-relative paths to add
            on top of the built-in K protected-files list.
        extra_protected_dirs: project-root-relative directory roots
            to add on top of the built-in K protected-dirs list.
        sandbox_roots: project-root-relative directory roots where
            destructive operations are allowed.
        screen_cache_dir: Cap-1 screen-capture cache (used by the
            screen_context OUT-gate rules in Phase 4).
        approved_outbound_apis: hostnames the model is allowed to
            contact (used by Categories I/J).
        project_root: override PROJECT_ROOT for testing. Defaults to
            :data:`ultron.config.PROJECT_ROOT`.

    Returns:
        A frozen :class:`Policy` instance.
    """
    if project_root is None:
        try:
            from ultron.config import PROJECT_ROOT
            project_root = Path(PROJECT_ROOT)
        except Exception:
            import os as _os
            project_root = Path(_os.getcwd())

    protected_files: list[Path] = [
        (project_root / rel).resolve(strict=False)
        for rel in _DEFAULT_PROTECTED_FILES_RELATIVE
    ]
    if extra_protected_files:
        protected_files.extend(
            (project_root / rel).resolve(strict=False)
            for rel in extra_protected_files
        )

    protected_dirs: list[Path] = [
        (project_root / rel).resolve(strict=False)
        for rel in _DEFAULT_PROTECTED_DIRS_RELATIVE
    ]
    if extra_protected_dirs:
        protected_dirs.extend(
            (project_root / rel).resolve(strict=False)
            for rel in extra_protected_dirs
        )

    sandboxes: list[Path] = []
    if sandbox_roots:
        sandboxes = [
            (project_root / rel).resolve(strict=False)
            for rel in sandbox_roots
        ]
    else:
        # Default sandbox: the coding-task scratch area.
        sandboxes = [(project_root / "data" / "sandbox").resolve(strict=False)]

    return Policy(
        enabled=bool(enabled),
        rule_enabled=dict(rule_overrides or {}),
        sandbox_roots=sandboxes,
        protected_files=protected_files,
        protected_dirs=protected_dirs,
        screen_cache_dir=(
            (project_root / screen_cache_dir).resolve(strict=False)
            if screen_cache_dir is not None
            else None
        ),
        approved_outbound_apis=list(approved_outbound_apis or []),
    )
