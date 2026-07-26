#!/usr/bin/env python
"""Scenario-router SHADOW report (2026-07-26).

In shadow mode the router classifies every live turn and logs a
``router:scenario`` line, while the ~24-matcher chain still decides. This
script pairs each router verdict with what the chain ACTUALLY did on the same
turn (the ``turn:flow`` line) and reports where they disagree.

That disagreement list is the whole point: the labelled corpus says 97.2%, but
it is MY corpus. The streamer's real phrasings are the actual test, and this is
how we see which of the two was right before anything is dispatched on the
router.

Read it as:
  * AGREE      -- router label matches the route the chain took.
  * DISAGREE   -- inspect. Either the router is wrong (corpus gap) or the CHAIN
                  is wrong, which is the bug we set out to fix. The utterance
                  is printed so you can judge which.
  * NO-VERDICT -- router was unavailable/unparsed; the chain ran alone.

Run:
  .venv\\Scripts\\python.exe scripts\\relay_test\\shadow_report.py [--log logs/kenning.log]
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# turn=N | ... | router:scenario | label='x' | ms=12.3 | actionable=false | ...
_ROUTER_RE = re.compile(
    r"turn=(?P<turn>\d+).*?router:scenario.*?label=(?P<label>\S+).*?ms=(?P<ms>[\d.]+)")
# turn=N | ... | turn:flow | route='x' | ... | raw='...'
_FLOW_RE = re.compile(
    r"turn=(?P<turn>\d+).*?turn:flow.*?route='(?P<route>[^']*)'.*?raw='(?P<raw>[^']*)'")

#: How a chain ROUTE name maps onto a router SCENARIO label. Only the pairs
#: that genuinely mean the same thing -- anything unmapped is reported rather
#: than silently counted as agreement.
_EQUIV = {
    "snap": {"relay_team", "relay_named"},
    "relay": {"relay_team", "relay_named"},
    "relay_llm": {"relay_team", "relay_named"},
    "team_callout": {"relay_team", "relay_named"},
    "conversational_llm": {"answer_question", "social", "identity"},
    "private_reply": {"answer_question", "social", "identity"},
    "tell_chat": {"tell_chat"},
    "spotify": {"spotify"},
    "identity": {"identity"},
    "desktop_refuse": {"desktop_refuse"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(_ROOT / "logs" / "kenning.log"))
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    path = Path(args.log)
    if not path.is_file():
        print(f"log not found: {path}")
        return 2

    text = path.read_text(encoding="utf-8", errors="replace")
    routers: dict[str, tuple[str, float]] = {}
    flows: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        m = _ROUTER_RE.search(line)
        if m:
            routers[m.group("turn")] = (
                m.group("label").strip("'\""), float(m.group("ms")))
            continue
        m = _FLOW_RE.search(line)
        if m:
            flows[m.group("turn")] = (m.group("route"), m.group("raw"))

    if not routers:
        print("No router:scenario lines found.\n"
              "The router logs only when it is ENABLED:\n"
              '  $env:KENNING_SCENARIO_ROUTER="1"\n'
              "then restart Ultron and speak a few turns.")
        return 1

    agree, disagree, unmapped, noflow = 0, [], [], 0
    latencies = []
    label_counts: Counter = Counter()
    by_route = defaultdict(Counter)

    for turn, (label, ms) in sorted(routers.items(), key=lambda kv: int(kv[0])):
        latencies.append(ms)
        label_counts[label] += 1
        flow = flows.get(turn)
        if not flow:
            noflow += 1
            continue
        route, raw = flow
        by_route[route][label] += 1
        equiv = _EQUIV.get(route)
        if equiv is None:
            unmapped.append((turn, route, label, raw))
        elif label in equiv:
            agree += 1
        else:
            disagree.append((turn, route, label, raw))

    paired = agree + len(disagree)
    print("=" * 74)
    print(f"SHADOW REPORT   turns classified={len(routers)}  paired with a "
          f"chain route={paired}")
    if paired:
        print(f"AGREEMENT       {agree}/{paired} = {100.0 * agree / paired:.1f}%")
    print("=" * 74)

    if latencies:
        s = sorted(latencies)
        print(f"\nrouter latency: median {s[len(s) // 2]:.0f} ms | "
              f"p90 {s[int(len(s) * 0.9)]:.0f} ms | max {max(s):.0f} ms")

    print("\nrouter labels seen:")
    for lab, n in label_counts.most_common():
        print(f"  {n:4d}  {lab}")

    if disagree:
        print(f"\nDISAGREEMENTS ({len(disagree)}) -- judge which was right:")
        for turn, route, label, raw in disagree[: args.show]:
            print(f"  turn={turn:>4s}  chain={route:<20s} router={label:<20s}")
            print(f"              {raw[:100]!r}")

    if unmapped:
        print(f"\nUNMAPPED chain routes ({len(unmapped)}) -- no equivalence "
              f"defined, not counted either way:")
        for turn, route, label, raw in unmapped[: args.show]:
            print(f"  turn={turn:>4s}  chain={route:<20s} router={label:<20s} "
                  f"{raw[:60]!r}")

    if noflow:
        print(f"\n{noflow} classified turn(s) had no turn:flow line "
              f"(handled by a command handler that does not emit one).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
