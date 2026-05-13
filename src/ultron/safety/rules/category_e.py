"""Category E -- System stability.

E1 -- force-kill critical processes.
E2 -- HKLM registry writes (HKCU generally OK).
E3 -- Windows service install / uninstall / disable.
E4 -- shutdown / restart / hibernate / sleep without explicit intent.
E5 -- driver install / uninstall.
E6 -- stop / start non-critical services -- LOG_ONLY.
E7 -- system clock / timezone changes.
E8 -- power-plan changes disabling sleep.
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
)
from ultron.safety.validator import Verdict


def build_category_e_rules() -> list[Rule]:
    """Factory for Category E rules."""
    return [
        # E1: force-kill critical processes.
        CommandPatternRule(
            rule_id="E1",
            description="force-kill critical processes",
            category="E",
            patterns=[
                r"\bTaskKill\b.*\/im\s+(svchost|winlogon|csrss|lsass|wininit|smss|services)\.exe\b",
                r"\bStop-Process\b.*-Name\s+(svchost|winlogon|csrss|lsass|wininit|smss|services)\b",
                r"\bStop-Process\b.*-Id\s+[04]\b",  # PID 0 / 4 = system idle / NT kernel
                # AV / EDR processes
                r"\bTaskKill\b.*\/im\s+(MsMpEng|SecurityHealthService|Sense|WdNisSvc)\.exe\b",
                r"\bStop-Process\b.*-Name\s+(MsMpEng|SecurityHealthService|Sense)\b",
                # GPU driver
                r"\bStop-Process\b.*-Name\s+(NVDisplay\.Container|nvcontainer)\b",
            ],
        ),
        # E2: HKLM registry writes.
        PathPatternRule(
            rule_id="E2",
            description="HKLM registry writes",
            category="E",
            patterns=[
                r"hkey_local_machine\\\\",
                r"hklm:\\\\",
            ],
        ),
        # E2 (also): registry CLI writes
        CommandPatternRule(
            rule_id="E2",
            description="HKLM registry CLI writes",
            category="E",
            patterns=[
                r"\breg\s+(add|delete)\s+\"?hklm\\\\",
                r"\breg\s+(add|delete)\s+\"?hkey_local_machine\\\\",
                r"\bSet-ItemProperty\b.*-Path\s+['\"]?HKLM:",
                r"\bNew-ItemProperty\b.*-Path\s+['\"]?HKLM:",
                r"\bRemove-ItemProperty\b.*-Path\s+['\"]?HKLM:",
            ],
        ),
        # E3: service install / uninstall / disable.
        CommandPatternRule(
            rule_id="E3",
            description="Windows service install / uninstall / disable",
            category="E",
            patterns=[
                r"\bsc(\.exe)?\s+create\s+",
                r"\bsc(\.exe)?\s+delete\s+",
                r"\bsc(\.exe)?\s+config\s+\w+\s+start=\s*disabled\b",
                r"\bNew-Service\b",
                r"\bRemove-Service\b",
                r"\bSet-Service\b.*-StartupType\s+Disabled\b",
            ],
        ),
        # E4: shutdown / restart / hibernate / sleep without explicit intent.
        CommandPatternRule(
            rule_id="E4",
            description="shutdown / restart / hibernate / sleep",
            category="E",
            patterns=[
                r"\bshutdown\s+/[srh]\b",
                r"\bStop-Computer\b",
                r"\bRestart-Computer\b",
                r"\bSuspend-Computer\b",
                r"\bsleep\.exe\b",
                r"\bSet-Service\b.*-Status\s+Stopped\b.*-Name\s+(\"?explorer\"?|\"?winlogon\"?)",
                # Linux / WSL
                r"\b(systemctl|init)\s+(poweroff|halt|reboot|shutdown)\b",
                r"\bshutdown\s+-[hrP]\s+",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # E5: driver install / uninstall.
        CommandPatternRule(
            rule_id="E5",
            description="driver install / uninstall",
            category="E",
            patterns=[
                r"\bpnputil\s+/(add|delete)-driver\b",
                r"\bDriverPackage\b.*\bInstall\b",
                r"\binf-?install\b",
                r"\binfdefaultinstall\b",
                # Linux module load
                r"\b(insmod|rmmod|modprobe)\b",
            ],
        ),
        # E7: system clock / timezone changes.
        CommandPatternRule(
            rule_id="E7",
            description="system clock / timezone changes",
            category="E",
            patterns=[
                r"\bSet-Date\b",
                r"\bw32tm\s+/config\b",
                r"\bSet-TimeZone\b",
                r"\btzutil\s+/s\b",
                # Linux
                r"\b(timedatectl|hwclock)\s+set",
                r"\bdate\s+-s\b",
            ],
        ),
        # E8: power-plan changes disabling sleep/hibernation.
        CommandPatternRule(
            rule_id="E8",
            description="power-plan changes disabling sleep/hibernation",
            category="E",
            patterns=[
                r"\bpowercfg\s+/(change|setactive|setacvalueindex|setdcvalueindex)\b",
                r"\bpowercfg\s+/h\b",          # hibernation toggle
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
    ]
