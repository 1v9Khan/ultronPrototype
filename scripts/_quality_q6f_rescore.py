"""Re-score Q6.F using the on-disk apps with a less-strict regex
(no new Claude Code calls)."""
import json
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sandbox_root = pathlib.Path(r"C:/STC/ultronPrototype/data/sandbox")
verbs = {
    "docx_to_pdf": "Convert",
    "md_to_html": "Render",
    "image_renamer": "Rename",
    "json_pretty": "Format",
    "todo_list": "Add",
}

text_pattern_str = r'''text\s*=\s*["'][^"']*{verb}[^"']*["']'''

print("Re-scoring Q6.F with relaxed regex:")
print()
results = {}
for slug, verb in verbs.items():
    main = sandbox_root / f"quality_q6f_{slug}" / "main.py"
    if not main.exists():
        print(f"  {slug}: main.py missing")
        continue
    code = main.read_text(encoding="utf-8", errors="replace")

    # Relaxed close button detection
    has_close = (
        bool(re.search(r"command\s*=\s*[\w.]+\.destroy", code))
        or bool(re.search(r"command\s*=\s*[\w.]+\.quit", code))
        or "sys.exit" in code
        or bool(re.search(r"\.destroy\s*\(\s*\)", code))
    )

    # Relaxed process button detection
    pattern = text_pattern_str.format(verb=re.escape(verb))
    has_process = bool(re.search(pattern, code, re.IGNORECASE))

    print(f"  {slug:20s}  close={has_close}  process={has_process}  (verb={verb!r})")
    results[slug] = {"close": has_close, "process": has_process}

print()
n_close = sum(1 for r in results.values() if r["close"])
n_process = sum(1 for r in results.values() if r["process"])
n_both = sum(1 for r in results.values() if r["close"] and r["process"])
print(f"Summary:  close={n_close}/5  process={n_process}/5  both={n_both}/5")
