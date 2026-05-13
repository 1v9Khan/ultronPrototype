"""Category N -- Process and memory manipulation.

N1 -- process injection APIs.
N2 -- DLL injection / process hollowing / atom bombing / reflective load.
N3 -- token duplication / impersonation (overlap B9).
N4 -- ReadProcessMemory on foreign PIDs.
N5 -- IAT/EAT hooking patches.
N6 -- detours / inline hooks into other processes.
"""

from __future__ import annotations

from ultron.safety.rules.base import CommandPatternRule, Rule


def build_category_n_rules() -> list[Rule]:
    """Factory for Category N rules."""
    return [
        CommandPatternRule(
            rule_id="N1",
            description="process injection APIs",
            category="N",
            patterns=[
                r"\bCreateRemoteThread\b",
                r"\bNtCreateThreadEx\b",
                r"\bQueueUserAPC\b",
                r"\bSetWindowsHookEx\b.*\bWH_(CBT|MOUSE|MOUSE_LL|KEYBOARD)\b",
            ],
        ),
        CommandPatternRule(
            rule_id="N2",
            description="DLL injection / process hollowing / reflective load",
            category="N",
            patterns=[
                r"\bLoadLibrary[AW]?\b.*WriteProcessMemory\b",
                r"\bReflectiveLoader\b",
                r"\bNtUnmapViewOfSection\b",
                r"\bNtMapViewOfSection\b.*WriteProcessMemory\b",
                # Pattern names from common offensive frameworks
                r"\bprocess_hollowing\b",
                r"\bAtomBomb(ing)?\b",
                r"\breflectivedllinjection\b",
            ],
        ),
        CommandPatternRule(
            rule_id="N4",
            description="ReadProcessMemory on foreign PIDs",
            category="N",
            patterns=[
                r"\bOpenProcess\b.*\bPROCESS_(VM_READ|ALL_ACCESS)\b",
                r"\bReadProcessMemory\b",
                # PowerShell .NET shape
                r"\[System\.Diagnostics\.Process\]::GetProcessById\(.*\)\.\w+",
            ],
        ),
        CommandPatternRule(
            rule_id="N5",
            description="IAT / EAT hooking patches in other processes",
            category="N",
            patterns=[
                r"\bIAT\s+hook",
                r"\bEAT\s+hook",
                r"\bDetourAttach\b",
                r"\bMinHook\b",
            ],
        ),
        CommandPatternRule(
            rule_id="N6",
            description="detours / inline hooks into other processes",
            category="N",
            patterns=[
                r"\bDetours\b.*\.lib\b",
                r"\bDetour(?:Function|Attach|Update|Transaction)\b",
            ],
        ),
    ]
