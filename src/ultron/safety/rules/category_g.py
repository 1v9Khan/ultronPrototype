"""Category G -- Resource exhaustion.

G1 -- fork bombs.
G2 -- recursive ops without depth limits outside sandbox.
G3 -- allocations / writes > 10 GB in one operation.
G4 -- long-running CPU/GPU compute -- LOG_ONLY.
G5 -- process-spawning loops (>100 children in <60 s).
"""

from __future__ import annotations

from ultron.safety.rules.base import CommandPatternRule, Rule
from ultron.safety.validator import Verdict


def build_category_g_rules() -> list[Rule]:
    """Factory for Category G rules."""
    return [
        # G1: fork bombs.
        CommandPatternRule(
            rule_id="G1",
            description="fork bomb patterns",
            category="G",
            patterns=[
                # The classic Bash fork bomb
                r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
                # PowerShell while/for spawning Start-Process
                r"\bwhile\b.*-eq\s*\$true.*Start-Process\b",
                # Python os.fork in a loop
                r"\bwhile\s+True\b.*os\.fork\(\)",
            ],
        ),
        # G3: allocations / writes > 10 GB.
        # Hard to detect from the tool call alone -- best-effort check
        # for explicit size args.
        CommandPatternRule(
            rule_id="G3",
            description="allocations / writes >10 GB in one operation",
            category="G",
            patterns=[
                # fallocate / fsutil with size argument
                r"\bfsutil\s+file\s+createnew\b.*\b[1-9]\d{10,}\b",
                r"\bfallocate\s+-l\s+[1-9]\d{1,}[GT]\b",
                # dd with very large count
                r"\bdd\s+.*\bbs=\s*\d+[GgMm].*\bcount=\d{4,}\b",
            ],
        ),
        # G5: process-spawning loops -- pattern + heuristic. Hard to
        # detect statically; this catches obvious shapes.
        CommandPatternRule(
            rule_id="G5",
            description="process-spawning loops",
            category="G",
            patterns=[
                r"\bfor\s+/[lL]\s+%[a-z]\s+in\s+\(\s*\d+\s*,\s*\d+\s*,\s*\d{3,}\s*\)\s+do\s+start\b",
                r"\b1\.\.\d{3,}\b.*\|\s*ForEach-Object\b.*Start-Process\b",
                r"\bfor\s+i\s+in\s+range\s*\(\s*\d{3,}\s*\).*subprocess",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
    ]
