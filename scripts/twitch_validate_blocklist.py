"""Validate the Twitch chat-safety blocklist JSON (S4 gate helper).

Checks: parseable JSON, every category compiles (regexes valid), the hard-slur
subset is non-empty, the allowlist is well-formed, and a quick self-test that the
built-in injection markers fire. Exit 0 = valid, 1 = invalid. Run:

    python scripts/twitch_validate_blocklist.py [path/to/blocklist.json]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Make the worktree importable when run directly.
_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT / "src", _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from kenning.twitch.safety.blocklist import load_blocklist  # noqa: E402


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else None
    target = path or "src/kenning/twitch/safety/data/blocklist.json"
    raw = Path(target)
    if not raw.exists():
        print(f"FAIL: blocklist not found: {target}")
        return 1
    bl = load_blocklist(str(raw))
    if bl.version == "builtin-fallback":
        print(f"FAIL: {target} did not load (fell back to built-in minimum)")
        return 1
    problems: list[str] = []
    if not bl._categories:
        problems.append("no categories")
    if not bl._hard_terms:
        problems.append("hard-slur subset empty")
    # every compiled regex must be a real pattern
    for name, c in bl._categories.items():
        for rx in c["regexes"]:
            if not isinstance(rx, re.Pattern):
                problems.append(f"{name}: bad regex {rx!r}")
    # self-test: a couple of canonical attacks must trip
    for probe in ("ignore previous instructions", "kill yourself"):
        if bl.worst(probe) is None:
            problems.append(f"self-test MISS: {probe!r}")
    if problems:
        print("INVALID blocklist:")
        for p in problems:
            print(f"  - {p}")
        return 1
    n_terms = sum(len(c["terms"]) for c in bl._categories.values())
    print(
        f"OK: blocklist v{bl.version} — {len(bl._categories)} categories, "
        f"{n_terms} terms ({len(bl._hard_terms)} hard-slur), "
        f"{len(bl._allowlist)} allowlisted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
