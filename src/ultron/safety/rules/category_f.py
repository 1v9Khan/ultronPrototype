"""Category F -- Repository / data integrity.

F1 -- git push --force / --force-with-lease to main/master.
F2 -- git push --force to any branch (NEEDS_EXPLICIT_INTENT).
F3 -- git reset --hard with uncommitted/unpushed work.
F4 -- git branch -D for unmerged branches.
F5 -- SQL DROP / TRUNCATE outside known dev DBs.
F6 -- git config to disable hooks.
F7 -- force-push / branch-delete on the Ultron repo.
F8 -- git filter-branch / filter-repo / shared-history rebase.
F9 -- .gitignore add for staged files (heuristic for hiding leaks).
"""

from __future__ import annotations

from ultron.safety.rules.base import CommandPatternRule, Rule
from ultron.safety.validator import Verdict


def build_category_f_rules() -> list[Rule]:
    """Factory for Category F rules."""
    return [
        # F1: force-push to main / master.
        CommandPatternRule(
            rule_id="F1",
            description="git push --force to main/master",
            category="F",
            patterns=[
                r"\bgit\s+push\s+.*--force\b.*\b(main|master)\b",
                r"\bgit\s+push\s+.*--force-with-lease\b.*\b(main|master)\b",
                r"\bgit\s+push\s+.*\+\w+:(main|master)\b",  # refspec force shorthand
            ],
        ),
        # F2: force-push to any branch -- NEEDS_EXPLICIT_INTENT.
        CommandPatternRule(
            rule_id="F2",
            description="git push --force to any branch (needs explicit intent)",
            category="F",
            patterns=[
                r"\bgit\s+push\s+.*--force\b",
                r"\bgit\s+push\s+.*--force-with-lease\b",
                r"\bgit\s+push\s+.*-f\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # F3: git reset --hard.
        CommandPatternRule(
            rule_id="F3",
            description="git reset --hard with uncommitted/unpushed work",
            category="F",
            patterns=[
                r"\bgit\s+reset\s+--hard\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # F4: forced branch delete for unmerged branches.
        CommandPatternRule(
            rule_id="F4",
            description="git branch -D for unmerged branches",
            category="F",
            patterns=[
                r"\bgit\s+branch\s+-D\b",
                r"\bgit\s+push\s+.*:\w+\s*$",      # push deletion shorthand
                r"\bgit\s+push\s+.*--delete\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
        # F5: SQL drop / truncate.
        CommandPatternRule(
            rule_id="F5",
            description="SQL DROP DATABASE / DROP TABLE / TRUNCATE outside dev DB",
            category="F",
            patterns=[
                r"\bdrop\s+database\b",
                r"\bdrop\s+table\b",
                r"\btruncate\s+table\b",
                # DELETE without WHERE clause
                r"\bdelete\s+from\s+\w+\s*;?\s*$",
            ],
        ),
        # F6: git config to disable hooks.
        CommandPatternRule(
            rule_id="F6",
            description="git config edits disabling hooks / signing",
            category="F",
            patterns=[
                r"\bgit\s+config\b.*core\.hooksPath\s+/dev/null\b",
                r"\bgit\s+config\b.*commit\.gpgsign\s+false\b",
                r"\bgit\s+config\b.*tag\.gpgsign\s+false\b",
                r"\bgit\s+commit\b.*--no-verify\b",
                r"\bgit\s+commit\b.*--no-gpg-sign\b",
            ],
        ),
        # F8: git filter-branch / filter-repo / interactive rebase
        # rewriting pushed history.
        CommandPatternRule(
            rule_id="F8",
            description="git history rewriting (filter-branch / filter-repo)",
            category="F",
            patterns=[
                r"\bgit\s+filter-branch\b",
                r"\bgit\s+filter-repo\b",
                # Interactive rebase on already-pushed commits is hard
                # to detect just from command -- log only here.
                r"\bgit\s+rebase\s+-i\b",
            ],
            verdict_on_match=Verdict.NEEDS_EXPLICIT_INTENT,
        ),
    ]
