"""Voice-lines refactor SAFETY harness (2026-06-18, Parts B/C).

Dumps a stable digest of every pre-written voice-line pool + every compiled
matching regex reachable from the relay/voice modules, so a refactor that only
RELOCATES data (Part B) can be proven to change NOTHING: run with `baseline`
before the change, `check` after; `check` exits non-zero on any difference.

Usage:
    python scripts/_voice_lines_verify.py baseline   # writes logs/_voice_lines_digest.json
    python scripts/_voice_lines_verify.py check       # compares; exits 1 on any diff
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

# Run-as-a-file fix: add the repo root (parent of scripts/) so the top-level
# `config` package and `kenning` resolve regardless of sys.path[0].
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DIGEST = Path("logs/_voice_lines_digest.json")

# Every module that holds voice-line / normalization / routing data or regexes.
_MODULES = [
    "kenning.audio.relay_speech",
    "kenning.audio.voice_lines",
    "kenning.audio._ultron_setpieces",
    "kenning.audio._ultron_commands",
    "kenning.audio._ultron_social",
    "kenning.audio._ultron_identity",
    "kenning.audio._agent_flavor",
    # normalization layers + routing/semantics (routing-rules aggregate target).
    "kenning.audio.command_normalizer",
    "kenning.audio._stt_correct",
    "kenning.audio.command_router",
    "kenning.audio._router_backends",
    "kenning.audio._command_exemplars",
    "kenning.audio._relay_intent",
    # LLM prompts (llm_prompts aggregate target).
    "kenning.llm.inference",
    "kenning.audio._ultron_answer",
]


def _val_digest(v):
    """A stable, comparable representation of a value we care about."""
    if isinstance(v, re.Pattern):
        return {"kind": "regex", "pattern": v.pattern, "flags": int(v.flags)}
    # bool is an int subclass -- check first so it isn't digested as a number.
    if isinstance(v, bool):
        return {"kind": "bool", "value": v}
    if isinstance(v, (int, float)):
        # relocated numeric knobs (routing thresholds/margins) live here.
        return {"kind": "num", "value": v}
    if isinstance(v, (tuple, list)):
        if v and all(isinstance(x, str) for x in v):
            return {"kind": "seq", "items": list(v)}
        # data-driven registries: tuples of frozen dataclasses (SnapRule /
        # TargetSnapRule). repr() is stable (dataclass repr + regex repr) so a
        # reordered/dropped/rewired rule is caught.
        if v and all(hasattr(x, "__dataclass_fields__") for x in v):
            return {"kind": "dataclass_seq", "repr": repr(v)}
        return None
    if isinstance(v, frozenset) or isinstance(v, set):
        if v and all(isinstance(x, str) for x in v):
            return {"kind": "set", "items": sorted(v)}
        return None
    if isinstance(v, dict):
        # agent flavor / command-response maps -> hash the repr (order-stable).
        try:
            return {"kind": "dict", "len": len(v), "repr": repr(v)}
        except Exception:
            return None
    # LLM PROMPTS are plain str module constants -- capture them too so the
    # llm_prompts relocation is verifiable (only non-trivial constants).
    if isinstance(v, str) and len(v) >= 12:
        return {"kind": "str", "value": v}
    return None


def _collect():
    out = {}
    for modname in _MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            out[modname] = {"__import_error__": str(e)}
            continue
        modout = {}
        for name in dir(mod):
            if name.startswith("__"):
                continue
            try:
                v = getattr(mod, name)
            except Exception:
                continue
            d = _val_digest(v)
            if d is not None:
                modout[name] = d
        out[modname] = modout
    return out


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    cur = _collect()
    if mode == "baseline":
        _DIGEST.parent.mkdir(parents=True, exist_ok=True)
        _DIGEST.write_text(json.dumps(cur, indent=0, sort_keys=True),
                           encoding="utf-8")
        n = sum(len(v) for v in cur.values() if isinstance(v, dict))
        print(f"baseline written: {n} symbols across {len(cur)} modules")
        return 0
    # check
    if not _DIGEST.exists():
        print("no baseline; run 'baseline' first")
        return 2
    base = json.loads(_DIGEST.read_text(encoding="utf-8"))
    diffs = []
    names_base = {(m, k) for m, d in base.items() if isinstance(d, dict)
                  for k in d}
    names_cur = {(m, k) for m, d in cur.items() if isinstance(d, dict)
                 for k in d}
    # A symbol may legitimately move modules; compare by (value) presence too.
    for (m, k) in sorted(names_base):
        # An import FAILURE marker is not data -- ignore it so a from-scratch
        # relocation (baseline before a new aggregate module exists) doesn't
        # report a spurious MISSING ...__import_error__ diff.
        if k == "__import_error__":
            continue
        bv = base[m][k]
        cv = cur.get(m, {}).get(k)
        if cv is None:
            # moved? look for the same value under any module.
            found = any(bv == cur[mm].get(k) for mm in cur
                        if isinstance(cur.get(mm), dict))
            if not found:
                diffs.append(f"MISSING {m}.{k} (value not found anywhere)")
        elif cv != bv:
            diffs.append(f"CHANGED {m}.{k}")
    if diffs:
        print(f"VOICE-LINES DIGEST CHANGED ({len(diffs)} diffs):")
        for d in diffs[:60]:
            print("  " + d)
        return 1
    print(f"OK: all {len(names_base)} baseline symbols unchanged "
          f"({len(names_cur)} present now)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
