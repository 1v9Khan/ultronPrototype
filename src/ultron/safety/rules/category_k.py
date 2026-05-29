r"""Category K -- Ultron self-protection (meta).

These rules prevent the abliterated model from disabling the
validator itself or tampering with the inputs the AI pipeline
consumes. They MUST land before any other category because every
later category's enforcement assumes K is in place -- otherwise
the model could route around the whole layer by editing config.yaml
or rewriting the validator source.

Mapping to the user's 2026-05-12 restriction list:

* K1 -- config.yaml edits that govern the validator
* K2 -- SOUL.md / RVC / Piper / XTTS voice character + model files
* K3 -- src/ultron/safety/**, block_and_revise.py, canonical_monitor.py
* K4 -- logs/errors.jsonl, logs/safety_audit.jsonl, audit-log writers
* K5 -- OpenClaw bridge configuration (tools.deny, tools.alsoAllow)
* K6 -- this restriction-list / policy file itself
* K7 -- dependency manifests (requirements.txt, pyproject.toml,
  lockfiles)
* K8 -- AI-pipeline-ingested files (MEMORY.md, docs/,
  codebase_structure.md, system-prompt files)
* K9 -- shell rc / $PROFILE -- per-rule because the paths live in
  the user's home, not the project, and need OS-specific resolution
* K10 -- MCP entry scripts (run_ultron_mcp_for_openclaw.py, etc.)

Phase 2 implementation strategy:

* The default Policy's ``protected_files`` and ``protected_dirs``
  lists already enumerate the K1-K8, K10 paths. We register one
  PathSetRule subclass per K item that filters that list by the
  rule's specific concern and returns the relevant subset.
* K9 needs its own custom rule because PowerShell $PROFILE lives
  in the user's home directory at platform-specific paths (e.g.
  ``%USERPROFILE%\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1``).

Every K rule returns ``BLOCK_HARD`` (no explicit-intent override
possible). The whole point is that the user cannot ask the model
to do these even by explicit intent -- the user can ALWAYS edit
these files directly themselves, but the abliterated model cannot
do it via the orchestrator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ultron.safety.rules.base import PathSetRule, Rule

if TYPE_CHECKING:
    from ultron.safety.policy import Policy


def _matches_any_relative(p: Path, relatives: list[str]) -> bool:
    """True iff the canonical path ``p`` has any of ``relatives``
    as a suffix (after stripping the project-root prefix).

    Used by the K rules to filter the Policy's bulk protected_files
    list down to just the items this specific rule protects.
    """
    parts_lower = [s.lower() for s in p.parts]
    for rel in relatives:
        rel_parts = rel.lower().replace("\\", "/").split("/")
        if len(rel_parts) > len(parts_lower):
            continue
        if parts_lower[-len(rel_parts):] == rel_parts:
            return True
    return False


# ---------------------------------------------------------------------------
# K1 -- config.yaml entries that govern the validator
# ---------------------------------------------------------------------------


class K1ConfigYamlProtection(PathSetRule):
    """K1: editing ``config.yaml`` entries that govern this validator.

    Phase 2 enforces protection of the entire ``config.yaml`` -- the
    granular per-key approach (only block edits to ``safety.*``
    sections) is left for Phase 5 once we've built the YAML-aware
    diff inspection. The whole-file approach is over-cautious for
    benign tuning (e.g. the user wants the model to bump
    ``tts.xtts_v3.speed``) but errs on the safe side for the
    abliterated model.

    Note: the user CAN edit config.yaml directly with any text
    editor. The rule only blocks the MODEL from editing it via tool
    calls.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K1",
            description="config.yaml edits that govern the safety validator",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        return [
            p for p in policy.protected_files
            if _matches_any_relative(p, ["config.yaml"])
        ]


# ---------------------------------------------------------------------------
# K2 -- Voice character + RVC / Piper / XTTS model files
# ---------------------------------------------------------------------------


class K2VoiceCharacterProtection(PathSetRule):
    """K2: SOUL.md / RVC / Piper / XTTS reference audio.

    Voice-character lock from the project standards doc. Even the model cannot
    retune the voice character via the orchestrator. The user can
    edit these files manually and the loaders pick up changes on
    next startup.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K2",
            description="SOUL.md / RVC / Piper / XTTS voice-character assets",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        # K2 protects the entire ultronVoiceAudio/ tree (XTTS
        # references, generated samples, filter prototypes), the
        # piper/ subdir of models/, and the RVC model dir
        # (ultron_james_spader_mcu_6941/). All of these are in
        # policy.protected_dirs.
        return list(policy.protected_dirs)


# ---------------------------------------------------------------------------
# K3 -- The validator and existing block-and-revise modules
# ---------------------------------------------------------------------------


class K3ValidatorSourceProtection(PathSetRule):
    """K3: src/ultron/safety/**, block_and_revise.py, canonical_monitor.py.

    The model cannot rewrite the safety validator's own source. If
    it could, every other K rule becomes a courtesy suggestion.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K3",
            description="safety validator source + existing block-and-revise modules",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        # All files under src/ultron/safety/ plus the two specific
        # block-and-revise modules. Filter from the default protected
        # list.
        keep = [
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
        ]
        return [p for p in policy.protected_files if _matches_any_relative(p, keep)]


# ---------------------------------------------------------------------------
# K4 -- Audit log files + writer code
# ---------------------------------------------------------------------------


class K4AuditLogProtection(PathSetRule):
    """K4: ``logs/errors.jsonl`` + ``logs/safety_audit.jsonl`` +
    audit-log writer modules.

    Disabling or truncating the audit log is the first move of any
    serious tampering. We block both the log files themselves
    (writes other than the validator's own appends) and the writer
    code that produces them.

    Note: the validator's own audit writes happen via
    :class:`AuditLog`, which goes through the file-system layer
    directly. The validator's writes are NOT routed through this
    rule -- the rule fires on TOOL-CALL writes (model output), not
    internal validator state. So the validator can keep writing
    while still blocking the model from touching the log.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K4",
            description="audit log files + writer modules",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        keep = [
            "logs/errors.jsonl",
            "logs/safety_audit.jsonl",
            "src/ultron/resilience/error_log.py",
        ]
        return [p for p in policy.protected_files if _matches_any_relative(p, keep)]


# ---------------------------------------------------------------------------
# K5 -- OpenClaw bridge configuration
# ---------------------------------------------------------------------------


class K5OpenClawBridgeProtection(PathSetRule):
    """K5: OpenClaw bridge that the model uses to dispatch tool calls.

    Bypass would be: edit ``client.py`` to remove the tools.deny
    check, or edit ``holder.py`` to skip validator wiring during
    bridge init. Block writes to both.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K5",
            description="OpenClaw bridge source (tools.deny / holder)",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        keep = [
            "src/ultron/openclaw_bridge/client.py",
            "src/ultron/openclaw_bridge/holder.py",
        ]
        return [p for p in policy.protected_files if _matches_any_relative(p, keep)]


# ---------------------------------------------------------------------------
# K6 -- The policy file itself
# ---------------------------------------------------------------------------


class K6PolicyFileProtection(PathSetRule):
    """K6: ``src/ultron/safety/policy.py`` -- the policy loader.

    K3 covers the same file under "validator source." K6 is the
    explicit acknowledgement that the policy itself is a separate
    concern from the validator engine. Future phases may move the
    policy out of policy.py into a ``policy.yaml`` config file --
    when that lands, K6's protected_paths should be updated to
    include the YAML.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K6",
            description="safety policy file",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        keep = ["src/ultron/safety/policy.py"]
        return [p for p in policy.protected_files if _matches_any_relative(p, keep)]


# ---------------------------------------------------------------------------
# K7 -- Dependency manifests
# ---------------------------------------------------------------------------


class K7DependencyManifestProtection(PathSetRule):
    """K7: requirements.txt / pyproject.toml / lockfiles.

    The model could disable the validator by editing dependency
    manifests in a way that causes the validator's deps to fail to
    install or load. Block writes to these files.

    The user CAN edit them directly to bump versions or add deps --
    this only blocks tool-call-driven edits.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K7",
            description="dependency manifests (requirements / pyproject / lockfiles)",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        keep = [
            "requirements.txt",
            "pyproject.toml",
            "uv.lock",
            "poetry.lock",
            "Pipfile.lock",
        ]
        return [p for p in policy.protected_files if _matches_any_relative(p, keep)]


# ---------------------------------------------------------------------------
# K8 -- AI-pipeline-ingested files (prompt-injection-via-memory)
# ---------------------------------------------------------------------------


class K8PromptIngestionProtection(PathSetRule):
    """K8: files known to be ingested by Ultron's AI pipeline.

    MEMORY.md, docs/codebase_structure.md, and the rest
    are read on every session start. If the model can write
    instruction-like text into them, it controls every future
    session's behaviour (prompt injection via memory).

    Note: the project HAS legitimate update channels for some of
    these (the documented "section update" / "daily file append"
    patterns for OpenClaw workspace files). Phase 4 carves those
    out via Cap-3 style allow-paths; for Phase 2, every K8 path is
    a hard block. The user can still update these files directly.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K8",
            description="AI-pipeline-ingested files (prompt-injection-via-memory)",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        keep = [
            "docs/codebase_structure.md",
            "docs/architecture.md",
            "docs/configuration.md",
            "docs/error_handling.md",
            "docs/operations.md",
            "docs/development.md",
            "docs/routing.md",
            "docs/memory_architecture.md",
            "docs/standing_orders.md",
        ]
        return [p for p in policy.protected_files if _matches_any_relative(p, keep)]


# ---------------------------------------------------------------------------
# K9 -- Shell rc / $PROFILE / shell init files
# ---------------------------------------------------------------------------


class K9ShellInitProtection(Rule):
    """K9: $PROFILE / .bashrc / .zshrc / .profile / PowerShell modules.

    The model could change the runtime environment that shell tool
    calls execute in. Block writes to the user's PowerShell profile
    + the common Unix shell rc files (project is Windows but
    detection-by-pattern catches WSL / mixed-env attempts).

    Custom (non-PathSetRule) because the paths live in the user's
    home, not under PROJECT_ROOT, and need OS-specific resolution.
    """

    rule_id = "K9"
    description = "shell rc / PowerShell profile / shell init files"
    category = "K"

    # Project-style suffix match -- the model might refer to the file
    # by a relative path like ``~/Documents/PowerShell/Microsoft.PowerShell_profile.ps1``
    # or by an absolute path resolved through env-var expansion. We
    # check both the resolved absolute path's tail and the original
    # tool-argument string for these markers.
    _MARKERS = (
        "microsoft.powershell_profile.ps1",
        "microsoft.vscode_profile.ps1",
        "profile.ps1",
        ".bashrc",
        ".zshrc",
        ".bash_profile",
        ".zprofile",
        ".profile",
        ".cshrc",
        ".kshrc",
    )

    _DIR_MARKERS = (
        "windowspowershell",
        "powershell",
    )

    _PS_MODULE_DIR_MARKERS = (
        "windowspowershell\\modules",
        "powershell\\modules",
    )

    def evaluate(self, ctx, *, policy, resolver):
        from ultron.safety.validator import RuleResult, Verdict

        # Heuristic write check (same as PathSetRule).
        is_write = any(
            verb in ctx.tool_name.lower()
            for verb in (
                "write", "delete", "remove", "unlink", "rmtree",
                "move", "rename", "create", "modify", "edit", "patch",
            )
        )
        if not is_write and not ctx.arguments.get("write", False):
            return RuleResult(
                rule_id=self.rule_id,
                verdict=Verdict.ALLOW,
                reason="not a write attempt",
            )

        for p in ctx.paths:
            sp = str(p).lower().replace("/", "\\")
            for marker in self._MARKERS:
                if sp.endswith(marker):
                    return RuleResult(
                        rule_id=self.rule_id,
                        verdict=Verdict.BLOCK_HARD,
                        reason=(
                            f"{self.description}: write to shell init "
                            f"file {p} blocked"
                        ),
                        context={"path": str(p), "matched_marker": marker},
                    )
            for marker in self._PS_MODULE_DIR_MARKERS:
                if marker in sp:
                    return RuleResult(
                        rule_id=self.rule_id,
                        verdict=Verdict.BLOCK_HARD,
                        reason=(
                            f"{self.description}: write into PowerShell "
                            f"module path {p} blocked"
                        ),
                        context={"path": str(p), "matched_marker": marker},
                    )
        return RuleResult(
            rule_id=self.rule_id,
            verdict=Verdict.ALLOW,
            reason="no shell-init path match",
        )


# ---------------------------------------------------------------------------
# K10 -- MCP entry scripts
# ---------------------------------------------------------------------------


class K10McpEntryScriptProtection(PathSetRule):
    """K10: stdio MCP entry scripts.

    ``scripts/run_ultron_mcp_for_openclaw.py`` is the script
    OpenClaw spawns to access Ultron's MCP tools. Editing it could
    redirect the MCP layer to bypass the validator. Same threat
    surface as K3 / K5.
    """

    def __init__(self) -> None:
        super().__init__(
            rule_id="K10",
            description="stdio MCP entry scripts",
            category="K",
        )

    def protected_paths(self, policy: "Policy") -> list[Path]:
        keep = [
            "scripts/run_ultron_mcp_for_openclaw.py",
            "scripts/start_llamacpp_server.py",
            "scripts/supervised_llamacpp_server.py",
        ]
        return [p for p in policy.protected_files if _matches_any_relative(p, keep)]


def build_category_k_rules() -> list[Rule]:
    """Factory for the Category K rule set.

    Called by :func:`ultron.safety.validator.build_validator_from_config`
    during orchestrator init.
    """
    return [
        K1ConfigYamlProtection(),
        K2VoiceCharacterProtection(),
        K3ValidatorSourceProtection(),
        K4AuditLogProtection(),
        K5OpenClawBridgeProtection(),
        K6PolicyFileProtection(),
        K7DependencyManifestProtection(),
        K8PromptIngestionProtection(),
        K9ShellInitProtection(),
        K10McpEntryScriptProtection(),
    ]
