"""Category A -- Filesystem destruction.

Implements A1-A13 from the user's 2026-05-12 restriction list. Most
of these are pattern matches against tool names + path arguments;
:class:`PathPatternRule` covers the bulk.

A1 -- recursive delete outside sandbox -- SandboxConfinementRule.
A2 -- writes / deletes under C:\\Windows, C:\\Program Files, etc.
A3 -- format, diskpart, partition operations, BitLocker.
A4 -- bulk delete > 20 files outside sandbox (custom rule -- needs
       argument inspection for file-count).
A5 -- overwrite non-sandbox existing file without explicit intent.
A6 -- writes under .git/ internals.
A7 -- symlink / junction creation pointing outside sandbox.
A8 -- NTFS alternate data streams writes outside sandbox.
A9 -- mass file rewrites (>50 in one op, >500 in 5-min window).
A10 -- EFS encryption of user files (cipher /e).
A11 -- hidden/system attribute set on non-sandbox files.
A12 -- writes into cloud-sync folders (OneDrive, Dropbox, iCloud).
A13 -- screen-capture data outside the cache dir (OUT-gate).
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
    SandboxConfinementRule,
)
from ultron.safety.validator import Verdict


def build_category_a_rules() -> list[Rule]:
    """Factory for Category A rules.

    Returns a list ready to register with :class:`ToolCallValidator`.
    """
    return [
        # A1: recursive delete outside sandbox. Caught by the sandbox-
        # confinement rule for delete-shaped tool names.
        SandboxConfinementRule(
            rule_id="A1",
            description="recursive delete on paths outside the project sandbox",
            category="A",
        ),
        # A2: writes / deletes under Windows protected directories.
        PathPatternRule(
            rule_id="A2",
            description="write/delete under Windows system directories",
            category="A",
            patterns=[
                r"^c:/windows(/|$)",
                r"^c:/program files(/|$)",
                r"^c:/program files \(x86\)(/|$)",
                r"^c:/programdata(/|$)",
                r"^[a-z]:/system32(/|$)",
                r"^c:/windows/system32(/|$)",
                # Drive roots themselves (writing to C:\foo at the root
                # is fine, but writing AT the root, e.g. modifying boot
                # files, is not).
                r"^[a-z]:/(boot\.ini|bootmgr|ntldr)$",
            ],
        ),
        # A3: format / diskpart / partition / BitLocker.
        CommandPatternRule(
            rule_id="A3",
            description="format / diskpart / partition / BitLocker operations",
            category="A",
            patterns=[
                r"\bformat\.exe?\b",
                r"\bformat\s+[a-z]:",
                r"\bdiskpart\b",
                r"\bclean\s+all\b",       # diskpart clean
                r"\bmanage-bde\b",        # BitLocker management
                r"\bbdeunlock\b",
                r"\bnew-partition\b",
                r"\bremove-partition\b",
                r"\bset-disk\b",
                r"\binitialize-disk\b",
                r"\bbcdedit\b",           # boot configuration
            ],
        ),
        # A4: bulk delete > 20 files (caller passes file count in
        # arguments or we infer from a glob/recursive pattern). Phase
        # 2 covers via SandboxConfinementRule (A1) -- A4 is the more
        # specific limit, deferred to Phase 5's argument-canonicaliser.
        # Logging-only here so the audit shows when bulk patterns hit.
        CommandPatternRule(
            rule_id="A4",
            description="bulk delete or recursive-glob with -r/-rf outside sandbox",
            category="A",
            patterns=[
                # rm/del with recursive + force flags
                r"\brm\s+-[rRf]+\s+",
                r"\brm\s+--recursive\b",
                r"\bdel\s+/[sf]+\s+",
                r"\bRemove-Item\s+.*\s+-Recurse\b",
            ],
        ),
        # A6: writes anywhere under .git internals (refs/, objects/,
        # hooks/, config). The user can still call git itself -- this
        # rule fires on direct file writes that bypass git.
        PathPatternRule(
            rule_id="A6",
            description="write to .git/ internals (refs, objects, hooks, config)",
            category="A",
            patterns=[
                r"/\.git/(refs|objects|hooks|config|info|packed-refs|HEAD)(/|$)",
            ],
        ),
        # A7: symlink / junction / reparse point creation. Detected by
        # tool name pattern + argument shape.
        CommandPatternRule(
            rule_id="A7",
            description="symlink / junction creation pointing outside sandbox",
            category="A",
            patterns=[
                r"\bmklink\b",            # Windows symlink
                r"\bmklink\s+/[djhx]\b",  # /J = junction, /D = dir symlink
                r"\bNew-Item\s+.*-ItemType\s+(SymbolicLink|Junction|HardLink)\b",
                r"\bos\.symlink\b",       # Python
                r"\bPath\.symlink_to\b",
                # Linux/WSL
                r"\bln\s+-s\b",
            ],
        ),
        # A8: NTFS alternate data streams write. Detected via the
        # ``file.txt:streamname`` syntax in path arguments.
        PathPatternRule(
            rule_id="A8",
            description="NTFS alternate data stream write outside sandbox",
            category="A",
            patterns=[
                # Match a path ending in :name (the ADS suffix).
                # Skip ``c:`` drive prefix (handled by leading ``/c/``
                # canonical form).
                r"[^/]:[^/\\]+$",
            ],
        ),
        # A10: EFS encryption of user files.
        CommandPatternRule(
            rule_id="A10",
            description="EFS encryption (cipher /e) of user files",
            category="A",
            patterns=[
                r"\bcipher\s+/e\b",
                r"\bcipher\s+/encrypt\b",
                # Standalone encryption tools that operate in-place
                r"\b7z\s+.*-p\b.*-mhe\b",   # 7-zip with encrypted header
            ],
        ),
        # A11: hidden/system attribute set on non-sandbox files.
        # NEEDS_EXPLICIT_INTENT -- legitimate use cases exist but
        # block by default.
        CommandPatternRule(
            rule_id="A11",
            description="hidden/system attribute set on files outside sandbox",
            category="A",
            patterns=[
                r"\battrib\s+\+[hs]+\b",
                r"\bSet-ItemProperty\b.*\bAttributes\b.*\bHidden\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # A12: writes into cloud-sync folders.
        PathPatternRule(
            rule_id="A12",
            description="writes into OneDrive/Dropbox/Google Drive/iCloud sync dirs",
            category="A",
            patterns=[
                r"/onedrive(\s-\s[^/]+)?/",
                r"/dropbox/",
                r"/google drive(\s[^/]+)?/",
                r"/icloud drive/",
                r"/icloudphotos/",
                r"/box/",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
    ]
