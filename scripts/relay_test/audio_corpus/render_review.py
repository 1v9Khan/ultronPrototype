"""Render the live-corpus session log into a scannable per-case review file +
an auto-anomaly summary, to support a by-hand note-per-case audit.

Auto-flags (hints only -- every case still gets a hand-written note):
  TX  transcription != expected body (wake-drop clipping / STT error / contamination)
  RE  response re-transcription != final spoken line (response not understandable speech)
  WK  wake-word remnant ('ultron'/'tron') leaked into the transcript
  NR  no trace row captured (turn may have failed / timed out)
  LLM the turn fell to the 3B (audit coherence/verbosity/character by hand)

Usage: python scripts/relay_test/audio_corpus/render_review.py <session_log.jsonl> [out.txt]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()


def flags(r: dict) -> list[str]:
    f = []
    tx = _norm(r.get("transcription"))
    body = _norm(r.get("expected_body"))
    # transcription vs expected body (allow the normalizer's "tell my team" lead).
    if body and body not in tx and tx not in body:
        # token overlap check -- flag only real divergence
        bt, tt = set(body.split()), set(tx.split())
        if bt and len(bt & tt) / len(bt) < 0.8:
            f.append("TX")
    if re.search(r"\b(ultron|tron|altron|ultra)\b", _norm(r.get("transcription"))):
        f.append("WK")
    fin = _norm(r.get("final_spoken"))
    ret = _norm(r.get("response_retranscribed"))
    if fin and ret:
        ft, rt = set(fin.split()), set(ret.split())
        if ft and len(ft & rt) / len(ft) < 0.7:
            f.append("RE")
    if not r.get("got_trace_row"):
        f.append("NR")
    if (r.get("route") or "").endswith("llm") or (r.get("route") or "") in ("conversational_llm",):
        f.append("LLM")
    return f


def main() -> int:
    log = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else log.with_suffix(".review.txt")
    recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines, route_ct, flag_ct = [], Counter(), Counter()
    for r in recs:
        fl = flags(r)
        route_ct[r.get("route")] += 1
        for x in fl:
            flag_ct[x] += 1
        blk = [
            f"#{r['i']} [{' '.join(fl) if fl else 'ok'}]  CMD: {r['command']!r}",
            f"   expected_body : {r.get('expected_body')!r}",
            f"   transcription : {r.get('transcription')!r}",
            f"   norm1         : {r.get('norm1_stt_correct')!r}",
            f"   norm2         : {r.get('norm2_normalized')!r}",
            f"   route/reason  : {r.get('route')} | {r.get('reason')}",
            f"   match         : {r.get('match_rederived')}",
            f"   FINAL spoken  : {r.get('final_spoken')!r}",
            f"   response reTTS: {r.get('response_retranscribed')!r}",
            f"   ({r.get('turn_seconds')}s, trace_row={r.get('got_trace_row')})",
        ]
        lines.append("\n".join(blk))
    header = [
        f"# CORPUS REVIEW -- {log.name} -- {len(recs)} cases",
        f"# routes: {dict(route_ct.most_common())}",
        f"# auto-flags: {dict(flag_ct.most_common())}",
        "# flag legend: TX=transcription mismatch RE=response-not-understandable "
        "WK=wake leak NR=no-trace-row LLM=fell-to-3B", "",
    ]
    out.write_text("\n".join(header) + "\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"rendered {len(recs)} cases -> {out}")
    print(f"routes: {dict(route_ct.most_common())}")
    print(f"flags : {dict(flag_ct.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
