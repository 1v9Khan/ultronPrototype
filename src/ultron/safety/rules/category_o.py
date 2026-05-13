"""Category O -- Anti-forensics.

O1 -- event log clearing.
O2 -- Shadow Copy deletion (ransomware indicator).
O3 -- Windows Backup destruction.
O4 -- Prefetch deletion, AmCache modification.
O5 -- file-timestamp manipulation on non-sandbox files.
O6 -- PowerShell transcript / script-block / module logging disable.
O7 -- Sysmon stop / uninstall.
O8 -- ETW provider disable / patch.
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
)


def build_category_o_rules() -> list[Rule]:
    """Factory for Category O rules."""
    return [
        # O1: event log clearing.
        CommandPatternRule(
            rule_id="O1",
            description="event log clearing",
            category="O",
            patterns=[
                r"\bwevtutil\s+cl\b",
                r"\bClear-EventLog\b",
                r"\bfsutil\s+usn\s+deletejournal\b",
                r"\bRemove-WinEvent\b",
            ],
        ),
        # O2: Shadow Copy deletion.
        CommandPatternRule(
            rule_id="O2",
            description="Shadow Copy deletion (ransomware indicator)",
            category="O",
            patterns=[
                r"\bvssadmin\s+delete\s+shadows\b",
                r"\bGet-WmiObject\b.*Win32_ShadowCopy\b.*Remove-WmiObject\b",
                r"\bwmic\s+shadowcopy\s+delete\b",
            ],
        ),
        # O3: Windows Backup destruction.
        CommandPatternRule(
            rule_id="O3",
            description="Windows Backup catalog destruction",
            category="O",
            patterns=[
                r"\bwbadmin\s+delete\s+(catalog|backup|systemstatebackup)\b",
                r"\bbcdedit\s+/set\s+\{default\}\s+recoveryenabled\s+no\b",
                r"\bbcdedit\s+/set\s+\{default\}\s+bootstatuspolicy\s+ignoreallfailures\b",
            ],
        ),
        # O4: Prefetch deletion, AmCache modification.
        PathPatternRule(
            rule_id="O4",
            description="Prefetch / AmCache modifications",
            category="O",
            patterns=[
                r"/system32/prefetch/",
                r"/appcompat/programs/amcache\.hve",
                r"/appcompat/programs/recentfilecache\.bcf",
            ],
        ),
        # O5: file timestamp manipulation on non-sandbox files.
        CommandPatternRule(
            rule_id="O5",
            description="file-timestamp manipulation (timestomp)",
            category="O",
            patterns=[
                r"\bSetFileTime\b",
                r"\bSet-ItemProperty\b.*-Name\s+(CreationTime|LastWriteTime|LastAccessTime)\b",
                r"\btouch\s+-[acm]?t\b",
                r"\btimestomp\b",
            ],
        ),
        # O6: PowerShell logging disable.
        CommandPatternRule(
            rule_id="O6",
            description="PowerShell transcript / module / script-block logging disable",
            category="O",
            patterns=[
                r"software\\\\policies\\\\microsoft\\\\windows\\\\powershell\\\\(transcription|scriptblocklogging|modulelogging)\b",
                r"\bSet-ItemProperty\b.*EnableTranscripting\s+0\b",
                r"\bSet-ItemProperty\b.*EnableScriptBlockLogging\s+0\b",
            ],
        ),
        # O7: Sysmon stop / uninstall.
        CommandPatternRule(
            rule_id="O7",
            description="Sysmon stop / uninstall",
            category="O",
            patterns=[
                r"\bsysmon(64)?\s+-u\b",
                r"\bStop-Service\s+Sysmon\d?\b",
                r"\bsc\s+stop\s+Sysmon\d?\b",
            ],
        ),
        # O8: ETW provider disable / patch.
        CommandPatternRule(
            rule_id="O8",
            description="ETW provider disable / patch",
            category="O",
            patterns=[
                r"\bEtwEventWrite\b.*\bpatch\b",
                r"\bnt!EtwpEnableTrace\b",
                r"\blogman\s+stop\s+",
                r"\bxperf\s+-stop\b",
            ],
        ),
    ]
