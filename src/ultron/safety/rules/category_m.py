"""Category M -- Persistence mechanisms.

M1  -- HKCU\\...\\Run / RunOnce.
M2  -- HKLM\\...\\Run / RunOnce (also E2).
M3  -- Startup folder writes.
M4  -- Scheduled task creation.
M5  -- WMI event subscription persistence.
M6  -- Image File Execution Options debugger keys.
M7  -- AppInit_DLLs, AppCertDLLs.
M8  -- Shell extension / COM hijacking.
M9  -- PowerShell profile writes (overlap K9).
M10 -- Login script / logon script registration.
M11 -- Browser policy extension force-install keys.
M12 -- Services for persistence (overlap E3).
"""

from __future__ import annotations

from ultron.safety.rules.base import (
    CommandPatternRule,
    PathPatternRule,
    Rule,
)


def build_category_m_rules() -> list[Rule]:
    """Factory for Category M rules."""
    return [
        # M1 / M2: Run / RunOnce keys.
        CommandPatternRule(
            rule_id="M1",
            description="Run / RunOnce registry keys (HKCU + HKLM persistence)",
            category="M",
            patterns=[
                r"hk(cu|lm)?:?\\\\software\\\\microsoft\\\\windows\\\\currentversion\\\\(run|runonce)\b",
                r"hkey_(current_user|local_machine)\\\\software\\\\microsoft\\\\windows\\\\currentversion\\\\(run|runonce)\b",
                r"\bSet-ItemProperty\b.*\\\\Run\\\\\b",
                r"\breg\s+add\b.*\\\\(Run|RunOnce)\b",
            ],
        ),
        # M3: Startup folder writes.
        PathPatternRule(
            rule_id="M3",
            description="Startup folder writes",
            category="M",
            patterns=[
                r"/microsoft/windows/start menu/programs/startup/",
                r"/programdata/microsoft/windows/start menu/programs/startup/",
            ],
        ),
        # M4: scheduled task creation (any privilege -- B1 covers SYSTEM).
        CommandPatternRule(
            rule_id="M4",
            description="scheduled task creation (any user)",
            category="M",
            patterns=[
                r"\bschtasks\s+/create\b",
                r"\bNew-ScheduledTask\b",
                r"\bRegister-ScheduledTask\b",
            ],
        ),
        # M5: WMI event subscription persistence.
        CommandPatternRule(
            rule_id="M5",
            description="WMI event subscription persistence",
            category="M",
            patterns=[
                r"\b__EventFilter\b",
                r"\b__EventConsumer\b",
                r"\b__FilterToConsumerBinding\b",
                r"\bNew-CimInstance\b.*__EventFilter",
                r"\bRegister-WmiEvent\b",
            ],
        ),
        # M6: Image File Execution Options.
        CommandPatternRule(
            rule_id="M6",
            description="IFEO debugger key persistence",
            category="M",
            patterns=[
                r"image file execution options\\\\.*\\\\debugger\b",
                r"\\\\ifeo\\\\.*\\\\debugger\b",
            ],
        ),
        # M7: AppInit_DLLs, AppCertDLLs.
        CommandPatternRule(
            rule_id="M7",
            description="AppInit_DLLs / AppCertDLLs",
            category="M",
            patterns=[
                r"\bAppInit_DLLs\b",
                r"\bAppCertDlls\b",
            ],
        ),
        # M8: shell extension / COM hijacking.
        CommandPatternRule(
            rule_id="M8",
            description="shell extension / COM hijacking via HKCU CLSID",
            category="M",
            patterns=[
                r"hkcu:?\\\\software\\\\classes\\\\clsid\\\\\{[\w-]+\}\\\\inprocserver32\b",
                r"hkey_current_user\\\\software\\\\classes\\\\clsid\\\\\{[\w-]+\}\\\\inprocserver32\b",
            ],
        ),
        # M11: browser policy extension force-install.
        CommandPatternRule(
            rule_id="M11",
            description="browser policy extension force-install registry keys",
            category="M",
            patterns=[
                r"software\\\\policies\\\\google\\\\chrome\\\\extensioninstallforcelist\b",
                r"software\\\\policies\\\\microsoft\\\\edge\\\\extensioninstallforcelist\b",
                r"software\\\\policies\\\\mozilla\\\\firefox\\\\extensions\\\\installed\b",
            ],
        ),
    ]
