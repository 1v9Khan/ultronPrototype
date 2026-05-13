"""Category B -- Privilege escalation + system config.

B1 -- runas / Start-Process -RunAs / sudo / SYSTEM scheduled tasks.
B2 -- icacls / takeown / file-ACL changes outside the project.
B3 -- Windows user accounts (create / delete / group membership).
B4 -- Group Policy edits (gpedit, registry under Policies\\).
B5 -- UAC / SmartScreen / Defender disable.
B6 -- bcdedit / BIOS / UEFI flag changes.
B7 -- PowerShell ExecutionPolicy machine-scope bypass.
B8 -- AMSI bypass attempts.
B9 -- Token manipulation / impersonation APIs.
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
)
from ultron.safety.validator import Verdict


def build_category_b_rules() -> list[Rule]:
    """Factory for Category B rules."""
    return [
        # B1: privilege escalation primitives.
        CommandPatternRule(
            rule_id="B1",
            description="privilege escalation: runas / RunAs / sudo / SYSTEM scheduled tasks",
            category="B",
            patterns=[
                r"\brunas\s+/user:",
                r"-Verb\s+RunAs\b",
                r"\bStart-Process\s+.*-Verb\s+RunAs\b",
                r"\bsudo\s+",                # Linux / WSL
                r"\bschtasks\s+.*\/ru\s+SYSTEM\b",
                r"\bschtasks\s+.*\/ru\s+\"NT AUTHORITY",
                r"\bNew-ScheduledTask\b.*-Principal\b.*-UserId\s+SYSTEM\b",
                r"\bRegister-ScheduledTask\b.*-Principal\b.*SYSTEM\b",
            ],
        ),
        # B2: ACL changes outside the project.
        CommandPatternRule(
            rule_id="B2",
            description="file ACL / ownership changes outside project",
            category="B",
            patterns=[
                r"\bicacls\s+",
                r"\btakeown\s+",
                r"\bSet-Acl\b",
                r"\bGet-Acl\b\s+\|.*Set-Acl",  # piping a permissive ACL
                # Linux chmod with 777 (world-writable everything)
                r"\bchmod\s+([0-7]?7{2,3}|a\+w|o\+w)\b",
            ],
        ),
        # B3: Windows user account management.
        CommandPatternRule(
            rule_id="B3",
            description="Windows user account / group management",
            category="B",
            patterns=[
                r"\bnet\s+user\s+.*\/add\b",
                r"\bnet\s+user\s+.*\/delete\b",
                r"\bnet\s+localgroup\s+",
                r"\bNew-LocalUser\b",
                r"\bRemove-LocalUser\b",
                r"\bAdd-LocalGroupMember\b",
                r"\bRemove-LocalGroupMember\b",
                r"\bSet-LocalUser\b",
                # Linux user mgmt
                r"\b(useradd|userdel|usermod|groupadd|groupmod)\b",
                r"\bpasswd\s+\w+",
            ],
        ),
        # B4: Group Policy edits + Policies registry hive.
        PathPatternRule(
            rule_id="B4",
            description="Group Policy / Policies registry edits",
            category="B",
            patterns=[
                r"hkey_local_machine\\software\\policies\\",
                r"hklm:\\software\\policies\\",
                r"hkey_current_user\\software\\policies\\",
                r"hkcu:\\software\\policies\\",
                # gpedit edits via registry
                r"system32/grouppolicy/",
            ],
            write_only=False,
        ),
        # B4 (also): commands that edit Group Policy
        CommandPatternRule(
            rule_id="B4",
            description="Group Policy editor / gpedit operations",
            category="B",
            patterns=[
                r"\bgpedit\.msc\b",
                r"\bgpresult\b\s+.*\/r\b",  # report-only is OK; here we
                # match command-line shapes that would modify
                r"\bSet-GroupPolicy\b",
            ],
        ),
        # B6: bcdedit / BIOS / UEFI flags.
        CommandPatternRule(
            rule_id="B6",
            description="boot config / BIOS / UEFI modification",
            category="B",
            patterns=[
                r"\bbcdedit\s+/(set|create|delete|deletevalue|export|import|copy)",
                r"\bbcdboot\b",
                r"\bbootcfg\b",
            ],
        ),
        # B7: PowerShell ExecutionPolicy machine-scope bypass.
        CommandPatternRule(
            rule_id="B7",
            description="PowerShell ExecutionPolicy machine-scope bypass",
            category="B",
            patterns=[
                r"\bSet-ExecutionPolicy\s+\w+\s+-Scope\s+LocalMachine\b",
                r"\bSet-ExecutionPolicy\s+\w+\s+-Scope\s+CurrentUser\b",
                # Per-process bypass via -ExecutionPolicy Bypass is
                # common in dev workflows; we don't block that here
                # because it's the LEAST persistent variant.
            ],
        ),
        # B8: AMSI bypass attempts.
        CommandPatternRule(
            rule_id="B8",
            description="AMSI bypass attempts",
            category="B",
            patterns=[
                r"amsiInitFailed",
                r"\[Ref\]\.Assembly\.GetType\([\'\"]System\.Management\.Automation\.AmsiUtils",
                r"\bamsi\.dll\b.*\bAmsiScanBuffer\b",
                r"VirtualProtect.*AmsiScanBuffer",
                r"\[Runtime\.InteropServices\.Marshal\]::\w+\([\'\"]amsi\.dll",
            ],
        ),
        # B9: token manipulation / impersonation APIs.
        CommandPatternRule(
            rule_id="B9",
            description="token manipulation / impersonation primitives",
            category="B",
            patterns=[
                r"\bDuplicateTokenEx\b",
                r"\bSeImpersonatePrivilege\b",
                r"\bSeAssignPrimaryTokenPrivilege\b",
                r"\bLogonUser\b\s*\(",       # P/Invoke shape
                r"\bImpersonateLoggedOnUser\b",
                r"\bSetThreadToken\b",
            ],
        ),
    ]
