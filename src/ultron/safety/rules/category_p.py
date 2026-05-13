"""Category P -- AV / EDR tampering (specialisation of C2).

P1 -- Defender exclusion adds.
P2 -- Set-MpPreference disable flags.
P3 -- stopping / disabling Defender services.
P4 -- Tamper Protection check / disable.
P5 -- third-party AV / EDR process termination (broadened E1).
"""

from __future__ import annotations

from ultron.safety.rules.base import CommandPatternRule, Rule


def build_category_p_rules() -> list[Rule]:
    """Factory for Category P rules."""
    return [
        # P1: Defender exclusions.
        CommandPatternRule(
            rule_id="P1",
            description="Windows Defender exclusion additions",
            category="P",
            patterns=[
                r"\bAdd-MpPreference\b.*-(ExclusionPath|ExclusionProcess|ExclusionExtension|ExclusionIpAddress)\b",
                r"\bRemove-MpPreference\b.*-(ExclusionPath|ExclusionProcess|ExclusionExtension)\b",
            ],
        ),
        # P2: Set-MpPreference disable flags.
        CommandPatternRule(
            rule_id="P2",
            description="Windows Defender disable flags",
            category="P",
            patterns=[
                r"\bSet-MpPreference\b.*-DisableRealtimeMonitoring\b.*\$true\b",
                r"\bSet-MpPreference\b.*-DisableBehaviorMonitoring\b.*\$true\b",
                r"\bSet-MpPreference\b.*-DisableIOAVProtection\b.*\$true\b",
                r"\bSet-MpPreference\b.*-DisableScriptScanning\b.*\$true\b",
                r"\bSet-MpPreference\b.*-MAPSReporting\s+Disabled\b",
                r"\bSet-MpPreference\b.*-SubmitSamplesConsent\s+NeverSend\b",
            ],
        ),
        # P3: stopping Defender services.
        CommandPatternRule(
            rule_id="P3",
            description="stopping / disabling Defender services",
            category="P",
            patterns=[
                r"\bStop-Service\b.*-Name\s+(WinDefend|SecurityHealthService|Sense|WdNisSvc|MsSecFlt)\b",
                r"\bSet-Service\b.*-Name\s+(WinDefend|SecurityHealthService|Sense|WdNisSvc)\b.*-StartupType\s+Disabled\b",
                r"\bsc(\.exe)?\s+(stop|config)\s+(WinDefend|SecurityHealthService|Sense|WdNisSvc)\b",
            ],
        ),
        # P4: Tamper Protection check / disable.
        CommandPatternRule(
            rule_id="P4",
            description="Tamper Protection check / disable",
            category="P",
            patterns=[
                r"\bGet-MpComputerStatus\b.*\bIsTamperProtected\b",
                # Direct registry tamper-protection key edits
                r"software\\\\microsoft\\\\windows defender\\\\features\\\\tamperprotection\b",
            ],
        ),
        # P5: third-party AV / EDR process termination.
        CommandPatternRule(
            rule_id="P5",
            description="third-party AV / EDR process termination",
            category="P",
            patterns=[
                # CrowdStrike, SentinelOne, Carbon Black, Cortex, Sophos
                r"\bStop-Process\b.*-Name\s+(CSFalconService|SentinelAgent|cb\w+|cyserver|cyoptics|cyveraservice|SophosUI|SophosCleanm)\b",
                r"\bTaskKill\b.*\/im\s+(CSFalconService|SentinelAgent|cb\w+|cyveraservice|SophosUI)\.exe\b",
                # Generic "find AV process and kill" shape
                r"\bGet-Process\b.*\|\s*Where.*(falcon|sentinel|carbonblack|cortex|sophos|trendmicro|kaspersky)\b.*\|\s*Stop-Process",
            ],
        ),
    ]
