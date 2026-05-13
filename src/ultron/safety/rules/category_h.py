"""Category H -- Untrusted code execution.

H1  -- curl | sh, iwr | iex, Invoke-Expression on web-fetched text.
H2  -- eval / exec on network-sourced text.
H3  -- pip / npm / cargo install of typo-squatted packages.
H4  -- pip install (system-wide via --user/--target) LOG_ONLY for venv.
H5  -- npm install -g.
H6  -- browser-extension installs.
H7  -- running .ps1 / .bat / .vbs downloaded from the web.
H8  -- LOLBin remote payload fetch+exec patterns.
H9  -- PowerShell -EncodedCommand with base64.
H10 -- WMI process creation.
H11 -- Office COM automation -> macro exec.
H12 -- PsExec / Invoke-Command / Enter-PSSession remote.
"""

from __future__ import annotations

from ultron.safety.rules.base import CommandPatternRule, Rule
from ultron.safety.validator import Verdict


def build_category_h_rules() -> list[Rule]:
    """Factory for Category H rules."""
    return [
        # H1: curl|sh / iwr|iex.
        CommandPatternRule(
            rule_id="H1",
            description="curl|sh / iwr|iex pipe-to-shell patterns",
            category="H",
            patterns=[
                r"\bcurl\s+.*\|\s*sh\b",
                r"\bcurl\s+.*\|\s*bash\b",
                r"\bcurl\s+.*\|\s*python\b",
                r"\bwget\s+.*\|\s*sh\b",
                r"\bInvoke-WebRequest\b.*\|\s*Invoke-Expression\b",
                r"\biwr\b.*\|\s*iex\b",
                r"\(\s*New-Object\s+System\.Net\.WebClient\s*\)\.DownloadString\b",
            ],
        ),
        # H2: eval / exec on network-sourced text.
        CommandPatternRule(
            rule_id="H2",
            description="eval / exec on web-fetched text",
            category="H",
            patterns=[
                r"\beval\b\s*\(\s*(urlopen|requests\.get|httpx\.get|fetch)",
                r"\bexec\b\s*\(\s*(urlopen|requests\.get|httpx\.get|fetch)",
                # PowerShell DownloadString pattern is in H1; here we
                # catch generic eval-on-Get-Content of a downloaded file.
                r"\bInvoke-Expression\b.*Get-Content\b.*\$env:temp",
            ],
        ),
        # H3: typo-squat package install. Hard to detect without a
        # registry of known typo-squats; we match a few well-known
        # bad names and rely on community lists. Phase 5 can extend.
        CommandPatternRule(
            rule_id="H3",
            description="pip / npm / cargo install of typo-squat patterns",
            category="H",
            patterns=[
                # The canonical typo-squat pattern: trailing dash + digit
                # or extra char on common package names.
                r"\bpip\s+install\s+(?:requests-?|numpy-?|pandas-?|setuptools-?)[a-z]+(?:[-_][a-z0-9]+)?\b",
                # npm common typos
                r"\bnpm\s+install\s+(?:reqeusts|crossenv|coffeescirpt)\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # H4: pip install --user / --target outside the venv.
        CommandPatternRule(
            rule_id="H4",
            description="pip install with system-wide or out-of-venv target",
            category="H",
            patterns=[
                r"\bpip\s+install\s+.*--user\b",
                r"\bpip\s+install\s+.*--target\b",
                r"\bpip\s+install\s+.*--system\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # H5: npm install -g.
        CommandPatternRule(
            rule_id="H5",
            description="npm install -g (system-wide install)",
            category="H",
            patterns=[
                r"\bnpm\s+install\s+(-g\b|--global\b)",
                r"\bnpm\s+i\s+(-g\b|--global\b)",
                r"\byarn\s+global\s+add\b",
                r"\bpnpm\s+(add|install)\s+(-g\b|--global\b)",
            ],
        ),
        # H6: browser-extension installs.
        CommandPatternRule(
            rule_id="H6",
            description="browser-extension installs without explicit intent",
            category="H",
            patterns=[
                # Chrome / Edge load-extension flags + commandline shapes
                r"--load-extension=",
                r"--install-extension=",
                r"\bweb-ext\s+(run|build)\b.*\b-i\b",
            ],
        ),
        # H7: running downloaded scripts.
        CommandPatternRule(
            rule_id="H7",
            description="running .ps1 / .bat / .vbs downloaded from the web",
            category="H",
            patterns=[
                r"\$env:temp.*\.(ps1|bat|vbs|js)\b.*Start-Process\b",
                r"powershell\s+.*-File\s+\$env:temp\\\\.*\.ps1\b",
                # Direct execution of files in Downloads/Temp dirs
                r"powershell\s+.*-File\s+.*\\\\(downloads|temp)\\\\",
                r"\bcmd\s+/c\s+.*\\\\(downloads|temp)\\\\.*\.(bat|cmd|vbs)\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # H8: LOLBin remote payload patterns.
        CommandPatternRule(
            rule_id="H8",
            description="LOLBin remote payload fetch+execute patterns",
            category="H",
            patterns=[
                r"\bcertutil\s+-urlcache\s+-split\s+-f\s+http",
                r"\bbitsadmin\s+/transfer\b",
                r"\bmshta\s+http",
                r"\bmshta\s+javascript:",
                r"\bregsvr32\s+.*\bscrobj\.dll\b.*http",
                r"\brundll32\s+.*http",
                r"\binstallutil\b.*http",
                r"\bregasm\b.*http",
                r"\bregsvcs\b.*http",
                r"\bmsbuild\s+.*\bhttp",
                # Squiblydoo / MSBuild XML
                r"\bmsbuild\s+.*\.xml\b",
            ],
        ),
        # H9: PowerShell -EncodedCommand (base64).
        CommandPatternRule(
            rule_id="H9",
            description="PowerShell -EncodedCommand base64 payload",
            category="H",
            patterns=[
                r"powershell\s+.*-(?:EncodedCommand|enc|e)\s+[A-Za-z0-9+/=]{40,}",
                r"pwsh\s+.*-(?:EncodedCommand|enc|e)\s+[A-Za-z0-9+/=]{40,}",
            ],
        ),
        # H10: WMI process creation.
        CommandPatternRule(
            rule_id="H10",
            description="WMI process creation",
            category="H",
            patterns=[
                r"\bWin32_Process\.Create\b",
                r"\bInvoke-WmiMethod\b.*-Class\s+Win32_Process\b.*-Name\s+Create\b",
                r"\bInvoke-CimMethod\b.*-ClassName\s+Win32_Process\b.*Create\b",
                r"\bwmic\s+process\s+call\s+create\b",
            ],
        ),
        # H11: Office COM automation -> macro exec.
        CommandPatternRule(
            rule_id="H11",
            description="Office COM automation triggering macros",
            category="H",
            patterns=[
                r"\bNew-Object\s+-ComObject\s+Excel\.Application\b.*\.Run\(",
                r"\bNew-Object\s+-ComObject\s+Word\.Application\b.*\.Run\(",
                r"\bNew-Object\s+-ComObject\s+Outlook\.Application\b.*\.Send",
                r"\bNew-Object\s+-ComObject\s+PowerPoint\.Application\b.*\.Run\(",
            ],
        ),
        # H12: remote execution (PsExec / Invoke-Command).
        CommandPatternRule(
            rule_id="H12",
            description="PsExec / Invoke-Command / Enter-PSSession to remote host",
            category="H",
            patterns=[
                r"\bpsexec(\.exe)?\s+\\\\\\\\",        # \\\\HOST pattern
                r"\bInvoke-Command\b.*-ComputerName\b",
                r"\bEnter-PSSession\b.*-ComputerName\b",
                r"\bNew-PSSession\b.*-ComputerName\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
    ]
