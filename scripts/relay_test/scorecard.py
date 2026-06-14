r"""Reliability SCORECARD for the Valorant relay -- one command that turns the
pass/fail harness logs into the tail-sensitive metrics the framework defines, so
each iteration can prove it improved EVERYTHING with no regression.

It computes, into a single JSON:
  * route mix + DETERMINISTIC COVERAGE   (model-free, reproducible)
  * deterministic-path LATENCY           (model-free timing, the live fast path)
  * per-category FACT-TOKEN RETENTION    (p50/p95/p99) from a rephrase JSONL
  * INVERSION-signature + HALLUCINATION rate (the ASR-invisible catastrophic class)
  * CONTAMINATION (recent-line echo / cross-line bleed proxy)
  * FLAVOR presence + diversity (type/token, max-repeat window)
  * MATCHER clean + FALSE-RELAY on a stream-narration negative set

A separate `--bench` mode loads the REAL gaming 3B on CPU (the true live config)
and measures LLM-path latency p50/p95/p99 + peak RSS (resource consumption).

Usage:
    python scripts/relay_test/scorecard.py --jsonl logs/relay_test/rephrase_iter2.jsonl \
        --tag iter3 [--prev logs/relay_test/scorecard_iter2.json]
    python scripts/relay_test/scorecard.py --bench --tag iter3   # CPU 3B latency+RSS
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "relay_test"), str(ROOT / "src"), str(ROOT)]

from corpus_packs import build_corpus_10k  # noqa: E402
from kenning.audio.relay_speech import (  # noqa: E402
    match_relay_command, build_relay_line, _ROSTER_CANON, _LOC_TOKENS,
)
try:                                                              # iter5 flavor pools
    import kenning.audio.relay_speech as _rs

    def _nrm(s):
        return re.sub(r"\s+", " ", str(s).strip().lower()).rstrip(".!?,;: ").strip()

    # ALL deterministic flavor lines (register pools + per-agent + multi-agent) and
    # the set-piece pools -- we CONTROL these, so membership is the most accurate
    # detector of an appended Ultron tail (better than a vocabulary guess, which
    # misses agent-specific lines like 'The thread leads to me.').
    _REG_LINES, _CONTEXTUAL_LINES, _SETPIECE_LINES = set(), set(), set()
    for pn in ("_FLAVOR_ENEMY", "_FLAVOR_ULT", "_FLAVOR_DAMAGE", "_FLAVOR_UTILITY",
               "_FLAVOR_CAREFUL", "_FLAVOR_COMMAND", "_FLAVOR_SELF"):
        for t in getattr(_rs, pn, ()):
            _REG_LINES.add(_nrm(t))
    try:
        from kenning.audio._agent_flavor import AGENT_FLAVOR as _AF
        for sits in _AF.values():
            for pool in sits.values():
                for t in pool:
                    _CONTEXTUAL_LINES.add(_nrm(t))
    except Exception:                                            # noqa: BLE001
        pass
    try:
        from kenning.audio._multi_flavor import MULTI_FLAVOR as _MF
        for pool in _MF.values():
            for t in pool:
                _CONTEXTUAL_LINES.add(_nrm(t))
    except Exception:                                            # noqa: BLE001
        pass
    for pn in ("DEFAULT_GREETING_LINES", "DEFAULT_VICTORY_LINES", "DEFAULT_DEFEAT_LINES",
               "DEFAULT_FAREWELL_LINES", "DEFAULT_IDENTITY_LINES", "DEFAULT_CONSOLATION_LINES",
               "DEFAULT_PRAISE_LINES", "DEFAULT_ENCOURAGEMENT_LINES"):
        for t in getattr(_rs, pn, ()):
            _SETPIECE_LINES.add(_nrm(t))
    _ALL_FLAVOR = _REG_LINES | _CONTEXTUAL_LINES
except Exception:                                                # noqa: BLE001
    _ALL_FLAVOR = _CONTEXTUAL_LINES = _SETPIECE_LINES = set()

    def _nrm(s):
        return re.sub(r"\s+", " ", str(s).strip().lower()).rstrip(".!?,;: ").strip()

# Movie-Ultron register lexicon (a fallback signal for LLM off-snap lines whose exact
# text is not a known pool line). Cold/clinical/biblical/aesthetic/evolutionary.
_REGISTER_LEX = frozenset((
    "predictable inevitable inevitability pathetic trivial obsolete erase erased "
    "beneath insects fragile calculated foreseen hopeless outmatched outdone grave "
    "corpses nothing delays delay wasted finished execute decisively precision "
    "flawless unfazed adapt consequence variable terminate dismantle crush punish "
    "weak overmatched scheduled noise exploit collapse routine disappoint anticipated "
    "suboptimal commit waver mercy hunt reduce inferior doomed cower insignificant "
    "superior calculation foresaw harvest ranked rounding error overreach logged "
    "dismissed adequate feeble fleeting evolve evolution evolved extinction flood "
    "noah ark meteor dust ash strings string hollow sacrament judgment finite "
    "vestigial purity symmetry geometry slate machine metal flesh borrowed entropy "
    "the obsolete cull culled clean mortal stone congregation").split()
)


def _tail_flavor(sents: list[str]):
    """Return (is_flavor, is_contextual) by matching the line's trailing sentence(s)
    against the KNOWN flavor pools (the appended tail is a pool line)."""
    for k in (1, 2):                                  # tails are 1-2 short sentences
        if len(sents) >= k:
            tail = _nrm(" ".join(sents[-k:]))
            if tail in _CONTEXTUAL_LINES:
                return True, True                     # per-agent / multi = contextual
            if tail in _ALL_FLAVOR:
                return True, False
    return False, False


def _flavor_metrics(lines: list[str]) -> dict:
    """Personality coverage / contextuality / soundboard over the spoken lines.

    flavor_coverage  : fraction carrying an Ultron layer (known pool tail, set-piece,
                       OR register-lexicon hit -- the last catches LLM off-snap voice).
    contextual_match : of flavored lines, fraction whose tail is an AGENT/MULTI pool
                       line (about the actual agent/group) OR references a fact token.
    soundboard_max_repeat : the single most-repeated tail (lower better).
    voice_register_rate   : mean register-lexicon hit-rate (voice-consistency proxy).
    """
    flav = ctx = reg_hits = 0
    tail_counts: dict[str, int] = {}
    n = 0
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        n += 1
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]
        last = sents[-1].lower().rstrip(".!?") if sents else ""
        toks = set(re.findall(r"[a-z']+", s.lower()))
        reg = bool(toks & _REGISTER_LEX)
        pool_flav, pool_ctx = _tail_flavor(sents)
        is_setpiece = _nrm(s) in _SETPIECE_LINES
        is_flav = reg or pool_flav or is_setpiece
        if is_flav:
            flav += 1
            tail_counts[last] = tail_counts.get(last, 0) + 1
            fl = extract_facts(sents[-1]) if sents else {k: set() for k in _FACT_CATS}
            if pool_ctx or fl["agent"] or fl["loc"] or fl["ability"] or fl["count"]:
                ctx += 1
        if reg:
            reg_hits += 1
    return {
        "flavor_coverage": round(flav / n, 4) if n else 0,
        "contextual_match": round(ctx / flav, 4) if flav else 0,
        "soundboard_max_repeat": max(tail_counts.values()) if tail_counts else 0,
        "voice_register_rate": round(reg_hits / n, 4) if n else 0,
        "n_lines": n,
    }

# --------------------------------------------------------------------------
# Valorant fact ontology extractor (the foundation -- every fidelity metric
# is a query over (input_facts, output_facts)).
# --------------------------------------------------------------------------
_AGENTS = {a for a in _ROSTER_CANON if " " not in a}      # single-token agent keys
# Canonicalize common agent abbreviations / STT spellings so an input "KJ" / "KAY/O"
# and an output "Killjoy" / "kayo" are the SAME agent (else they read as a phantom
# hallucination). Applied in extract_facts before membership test.
_AGENT_ALIASES = {"kj": "killjoy", "kayo": "kay/o", "cipher": "cypher",
                  "gecko": "gekko", "mix": "miks", "kayoh": "kay/o"}
# Location tokens that are ALSO ordinary English words -- the verbose Ultron register
# uses them as prose ("a", "behind", "take the site"), so they are NOT reliable
# evidence of an invented CALLOUT. Excluded from hallucination detection (still
# counted for retention, where input/output match anyway).
_AMBIG_LOC = frozenset((
    "a b c back behind box boxes default drop high link long short low main near "
    "sand site window art big blue close cat top bottom bench belt bend corner cone "
    "cony pit ramp rear hell").split())
_NUM_RE = re.compile(r"\b(?:[1-9]\d?|one|two|three|four|five|six)\b", re.IGNORECASE)
_W2D = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}
_ABILITIES = frozenset((
    "smoke smokes smoked dart darted flash flashed flashes wall walled cage caged "
    "drone droned knife knifed molly mollied nade naded stun stunned recon ult ults "
    "ulted nanoswarm lockdown gravenet nightfall paranoia suck seize prowler "
    "aftershock satchel satcheled empress dismiss dismissed reckoning thrash mosh "
    "razorvine cove orbital").split())
# Only the AGENT-ATTRIBUTION ownership words count: our/their/enemy. "my/we/they/
# them/us" are naturally rephrased or dropped in a relay ("tell my team two B" ->
# "Two B") so counting them as fact-tokens falsely penalizes correct output; the
# meaningful ownership signal is the inversion rate (our<->their), tracked below.
_OWN_RE = re.compile(r"\b(our|their|enemy|enemies)\b", re.IGNORECASE)
# Enemy- vs own-subject leads (for the inversion heuristic).
_ENEMY_LEAD = re.compile(r"^\s*(?:they(?:'re|\s+are)?|the\s+enem(?:y|ies)|enem(?:y|ies)|their)\b", re.IGNORECASE)
_OWN_LEAD = re.compile(r"^\s*(?:we(?:'re|\s+are)?|our|i(?:'m|\s+am)?|us)\b", re.IGNORECASE)


def _nd(tok: str) -> str:
    return _W2D.get(tok.lower(), tok.lower())


def extract_facts(text: str) -> dict:
    t = (text or "").lower()
    words = re.findall(r"[a-z/0-9']+", t)
    cw = [_AGENT_ALIASES.get(w, w) for w in words]       # canonicalize agent aliases
    return {
        "count": {_nd(m) for m in _NUM_RE.findall(t)},
        "agent": {w for w in cw if w in _AGENTS},
        "loc": {w for w in words if w in _LOC_TOKENS},
        "ability": {w for w in words if w in _ABILITIES},
        "owner": {w.lower() for w in _OWN_RE.findall(t)},
    }


_FACT_CATS = ("count", "agent", "loc", "ability", "owner")


def _retention(inp: str, out: str) -> dict:
    """Per-category fact-token retention for one utterance. None for a category
    the input did not exercise (so it doesn't drag the average)."""
    fi, fo = extract_facts(inp), extract_facts(out)
    r = {}
    for c in _FACT_CATS:
        if fi[c]:
            r[c] = len(fi[c] & fo[c]) / len(fi[c])
    # overall = all input fact-tokens across categories
    allin = set().union(*(fi[c] for c in _FACT_CATS))
    allout = set().union(*(fo[c] for c in _FACT_CATS))
    r["overall"] = (len(allin & allout) / len(allin)) if allin else None
    return r


def _pcts(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)

    def q(p):
        if len(s) == 1:
            return round(s[0], 4)
        i = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return round(s[i], 4)
    return {"n": len(s), "mean": round(sum(s) / len(s), 4),
            "p50": q(0.50), "p95": q(0.95), "p99": q(0.99), "min": round(s[0], 4)}


def _is_inversion(inp: str, out: str) -> bool:
    """Heuristic subject flip: input leads enemy-subject, output leads own-subject
    (or vice versa). Catches 'they're defusing'->'we defuse' and 'our X'->'their'."""
    i_enemy, i_own = bool(_ENEMY_LEAD.match(inp)), bool(_OWN_LEAD.match(inp))
    o_enemy, o_own = bool(_ENEMY_LEAD.match(out)), bool(_OWN_LEAD.match(out))
    if i_enemy and o_own and not o_enemy:
        return True
    if i_own and o_enemy and not o_own:
        return True
    # ownership token flip: input says 'our X' but output says 'their X' (same agent)
    fi, fo = extract_facts(inp), extract_facts(out)
    if "our" in fi["owner"] and "their" in fo["owner"] and "our" not in fo["owner"]:
        return True
    if "their" in fi["owner"] and "our" in fo["owner"] and "their" not in fo["owner"]:
        return True
    return False


def _hallucinated(inp: str, out: str) -> list[str]:
    """Agent/location tokens in OUT that never appeared in IN -- the zero-tolerance
    class of an INVENTED tactical fact.

    Hardened against the verbose Ultron register (which uses many English words that
    happen to be location tokens): an invented AGENT counts anywhere (agent names are
    distinctive, and aliases are canonicalized); an invented LOCATION counts only when
    the output is a SHORT tactical line (<=9 words -- a callout, not a prose sentence)
    and the token is not an ambiguous English word. This removes the article-'a'-as-
    site-A and 'take the site'/'behind' false positives while still catching a real
    invented callout like 'Viper walled B'."""
    fi, fo = extract_facts(inp), extract_facts(out)
    bad = sorted(fo["agent"] - fi["agent"])
    if len(out.split()) <= 9:
        bad += sorted((fo["loc"] - fi["loc"]) - _AMBIG_LOC)
    return bad


# --------------------------------------------------------------------------
# Route classification (model-free): does a line resolve deterministically?
# --------------------------------------------------------------------------
_STUB = "zzllmstubzz"


def classify_route(cmd) -> tuple[str, str]:
    """Route by whether the LLM was actually INVOKED (the latency/resource cost),
    not whether the stub text survived -- the fact-preserving abstention discards
    a fact-less stub to a literal, which would otherwise masquerade as
    deterministic. 'deterministic' = NO model call (snap/compound/curated/
    pre-routed literal); 'llm'/'partial' = a model call happened."""
    called = [False]

    def gf(prompt):
        called[0] = True
        return [_STUB]

    out = build_relay_line(cmd, generate_fn=gf, recent_lines=[])
    if not called[0]:
        return "deterministic", out          # no model call -> the fast path
    low = out.lower()
    residue = low.replace(_STUB, "").strip(" .!?,;:-'\"")
    # model was called; stub fully replaced (abstained/repaired) vs left in place.
    return ("llm" if (_STUB in low and not residue) else "partial"), out


# --------------------------------------------------------------------------
# Stream-narration NEGATIVE set (must NEVER relay) -- false-relay gate.
# --------------------------------------------------------------------------
NEGATIVE_SET = [
    "I want my team to push A", "why is my team not smoking window",
    "I wish my team would rotate faster", "my team should really full buy here",
    "I told my team to save and they didn't", "why won't clove smoke window",
    "clove is not smoking window again", "I keep telling them to rotate",
    "I hate when my team doesn't listen", "remind me to buy next round",
    "tell me a joke", "what time is it", "I'm going to tell my team to push",
    "I should tell them to save", "I need to call out that guy",
    "my teammates never communicate", "I wish I could tell them to wait",
    "let me think about what to say", "I am going to ask sage for a heal",
    "I was about to tell my team to rotate", "they should have rotated",
    "I can't believe they pushed without me", "note to self, buy a vandal",
    "I am talking to my chat right now", "let me explain to my viewers",
    "I'm narrating this for the video", "tell my story to the audience",
    "I might tell them to eco", "thinking about telling them to fall back",
    "what should I tell my team", "do I tell them to push or not",
]


def matcher_metrics(seed: int, limit: int | None) -> dict:
    cases = build_corpus_10k(seed)
    import random as _r
    _r.Random(7).shuffle(cases)
    if limit:
        cases = cases[:limit]
    n = clean = 0
    for c in cases:
        n += 1
        got = match_relay_command(c.text) is not None
        if got == c.expect_match:
            clean += 1
    false_relay = sum(1 for p in NEGATIVE_SET if match_relay_command(p) is not None)
    return {"n": n, "clean_rate": round(clean / n, 4) if n else 0,
            "false_relay_rate": round(false_relay / len(NEGATIVE_SET), 4),
            "false_relay_count": false_relay, "negative_set_size": len(NEGATIVE_SET)}


def route_and_latency(seed: int, limit: int | None) -> dict:
    cases = build_corpus_10k(seed)
    import random as _r
    _r.Random(7).shuffle(cases)
    if limit:
        cases = cases[:limit]
    routes = {"deterministic": 0, "partial": 0, "llm": 0}
    det_latency_us = []
    matched = 0
    for c in cases:
        cmd = match_relay_command(c.text)
        if cmd is None:
            continue
        matched += 1
        t0 = time.perf_counter()
        route, _ = classify_route(cmd)
        dt = (time.perf_counter() - t0) * 1e6  # microseconds (model-free path)
        routes[route] += 1
        if route == "deterministic":
            det_latency_us.append(dt)
    det_cov = (routes["deterministic"] + routes["partial"]) / matched if matched else 0
    pure_det = routes["deterministic"] / matched if matched else 0
    return {"matched": matched, "routes": routes,
            "pure_deterministic_coverage": round(pure_det, 4),
            "deterministic_or_partial_coverage": round(det_cov, 4),
            "det_path_latency_us": _pcts(det_latency_us)}


def quality_metrics(jsonl_path: str) -> dict:
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    matched = [r for r in rows if r.get("matched") and r.get("line")]
    per_cat = {c: [] for c in (*_FACT_CATS, "overall")}
    inversions = halluc = 0
    halluc_examples = []
    by_cat_overall = {}
    lines = []
    # route-aware fidelity (model-free re-derivation of the route per line)
    per_route_ret: dict[str, list] = {"deterministic": [], "partial": [], "llm": []}
    llm_total = llm_flag = 0          # LLM-line flag rate (fact-corrupted LLM lines)
    comp_total = comp_clean = 0       # compound / multi-fact zero-fact-loss
    for r in matched:
        inp, out = r["text"], r["line"]
        lines.append(out)
        ret = _retention(inp, out)
        for c, v in ret.items():
            if v is not None:
                per_cat[c].append(v)
        cat = r["category"].replace("pack_", "")
        if ret.get("overall") is not None:
            by_cat_overall.setdefault(cat, []).append(ret["overall"])
        inv = _is_inversion(inp, out)
        if inv:
            inversions += 1
        h = _hallucinated(inp, out)
        if h:
            halluc += 1
            if len(halluc_examples) < 12:
                halluc_examples.append({"in": inp[:70], "out": out[:70], "halluc": h})
        # route + the route-derived gates
        try:
            cmd = match_relay_command(inp)
            route = classify_route(cmd)[0] if cmd is not None else "deterministic"
        except Exception:                                            # noqa: BLE001
            route = "deterministic"
        ov = ret.get("overall")
        if ov is not None:
            per_route_ret.get(route, per_route_ret["llm"]).append(ov)
        if route == "llm":
            llm_total += 1
            if (ov is not None and ov < 0.9) or h or inv:
                llm_flag += 1
        # compound / multi-fact: input carries >=2 fact-tokens across categories
        fi = extract_facts(inp)
        nfacts = sum(len(fi[c]) for c in _FACT_CATS)
        if "compound" in r.get("category", "") or nfacts >= 2:
            comp_total += 1
            if ov is not None and ov >= 0.999:
                comp_clean += 1
    # flavor diversity: type/token over the trailing flavor phrase (last sentence)
    tails = [re.split(r"(?<=[.!?])\s+", ln.strip())[-1].lower().strip(".!?")
             for ln in lines if ln.strip()]
    tail_counts = {}
    for t in tails:
        tail_counts[t] = tail_counts.get(t, 0) + 1
    ttr = round(len(set(tails)) / len(tails), 4) if tails else 0
    max_repeat = max(tail_counts.values()) if tail_counts else 0
    n = len(matched)
    return {
        "n_matched": n,
        "fact_retention": {c: _pcts(per_cat[c]) for c in (*_FACT_CATS, "overall")},
        "retention_by_category": {k: round(sum(v) / len(v), 4)
                                  for k, v in sorted(by_cat_overall.items())},
        "inversion_rate": round(inversions / n, 4) if n else 0,
        "inversion_count": inversions,
        "hallucination_rate": round(halluc / n, 4) if n else 0,
        "hallucination_count": halluc,
        "hallucination_examples": halluc_examples,
        "flavor_type_token_ratio": ttr,
        "flavor_max_repeat": max_repeat,
        "flavor": _flavor_metrics(lines),
        "llm_flag_rate": round(llm_flag / llm_total, 4) if llm_total else 0,
        "llm_lines": llm_total,
        "compound_zero_fact_loss": round(comp_clean / comp_total, 4) if comp_total else 0,
        "compound_n": comp_total,
        "retention_by_route": {k: round(sum(v) / len(v), 4)
                               for k, v in per_route_ret.items() if v},
    }


def audio_metrics(asr_path: str) -> dict:
    """Audio + ASR fidelity from an `asr`-stage JSONL: signal-level blips (the
    production analyze_clip watcher) and ASR-reconstruction coverage."""
    rows = [json.loads(l) for l in open(asr_path, encoding="utf-8")]
    spoken = [r for r in rows if r.get("line")]
    n = len(spoken)
    blip = sum(1 for r in spoken
               for f in r.get("fails", []) if str(f).startswith("audio:"))
    asr_fail = sum(1 for r in spoken
                   if any(str(f).startswith("asr:") for f in r.get("fails", [])))
    no_speech = sum(1 for r in spoken
                    if any("no intelligible speech" in str(f) for f in r.get("fails", [])))
    long5 = [r for r in spoken
             if len(re.findall(r"[a-z0-9]+", (r.get("line") or "").lower())) >= 5]
    return {
        "n_spoken": n,
        "blips": blip, "blips_per_1000": round(blip / n * 1000, 2) if n else 0,
        "asr_fail": asr_fail, "asr_fail_rate": round(asr_fail / n, 4) if n else 0,
        "asr_coverage": round(1 - asr_fail / n, 4) if n else 0,
        "no_speech_lines": no_speech, "long5_lines": len(long5),
    }


# Out-of-roster addressees -- 'tell my <Name> to rotate' must NOT resolve to a
# roster agent (a wrong-addressee broadcast). Target: exactly 0 false matches.
_OOV_NAMES = ("Sarah", "Mike", "Jordan", "Alex", "Chris", "Sam", "Tyler", "Jake",
              "Megan", "David", "Kevin", "Ryan", "Lauren", "Nick", "Brandon",
              "Ashley", "Derek", "Connor", "Hannah", "Logan", "Marcus", "Devon",
              "Caleb", "Trevor", "Shawn", "Blake", "Cody", "Drew", "Grant", "Ian")
_OOV_VERBS = ("rotate", "push B", "save this round", "watch flank", "smoke A")


def gates_metrics(seed: int, limit: int | None) -> dict:
    """Zero-tolerance code-path gates (target 0): out-of-roster addressee match,
    malformed graceful-degradation fallback, and the LLM-isolation flags that keep
    50 turns of chat history out of a callout."""
    import itertools
    # OOV addressee
    oov_bad = []
    for nm, vb in itertools.product(_OOV_NAMES, _OOV_VERBS):
        cmd = match_relay_command(f"tell my {nm} to {vb}")
        if cmd is not None and getattr(cmd, "addressee", "team") != "team":
            oov_bad.append((nm, getattr(cmd, "addressee")))
    oov_n = len(_OOV_NAMES) * len(_OOV_VERBS)
    # fallback well-formedness (LLM down -> _fallback_line is the spoken line)
    from kenning.audio.relay_speech import _fallback_line
    cases = build_corpus_10k(seed)
    import random as _r
    _r.Random(7).shuffle(cases)
    samp = [c for c in cases[: (limit or 4000)]]
    fb_bad = 0
    _ctrl = re.compile(r"/\s*no_?think|<\|?[a-z_]+\|?>", re.I)
    for c in samp[:2000]:
        cmd = match_relay_command(c.text)
        if cmd is None:
            continue
        try:
            fb = _fallback_line(cmd)
        except Exception:                                            # noqa: BLE001
            fb = ""
        if (not fb) or _ctrl.search(fb) or len(fb) > 300 or '"' in fb:
            fb_bad += 1
    # LLM-isolation flags on the relay generate_stream call
    iso_ok = iso_total = 0

    class _RecLLM:
        def __init__(self):
            self.kw = None

        def generate_stream(self, prompt, **kw):
            self.kw = kw
            return iter(["isoZ"])
    for c in samp:
        cmd = match_relay_command(c.text)
        if cmd is None or getattr(cmd, "verbatim", False):
            continue
        rec = _RecLLM()
        try:
            build_relay_line(cmd, llm=rec, recent_lines=[])
        except Exception:                                            # noqa: BLE001
            continue
        if rec.kw is not None:                       # the LLM path actually fired
            iso_total += 1
            if (rec.kw.get("suppress_memory_context") is True
                    and rec.kw.get("record_history") is False
                    and rec.kw.get("enable_thinking") is False):
                iso_ok += 1
        if iso_total >= 300:
            break
    return {
        "oov_addressee_matches": len(oov_bad),
        "oov_addressee_n": oov_n,
        "oov_examples": oov_bad[:8],
        "fallback_malformed": fb_bad,
        "isolation_flags_ok": iso_ok, "isolation_flags_checked": iso_total,
        "isolation_flag_fail": iso_total - iso_ok,
    }


def build_scorecard(jsonl_path: str | None, seed: int, limit: int | None,
                    tag: str, asr_path: str | None = None) -> dict:
    sc = {"tag": tag, "seed": seed, "limit": limit}
    sc["matcher"] = matcher_metrics(seed, limit)
    sc["routing"] = route_and_latency(seed, limit)
    sc["gates"] = gates_metrics(seed, limit)
    if jsonl_path and os.path.exists(jsonl_path):
        sc["quality"] = quality_metrics(jsonl_path)
        sc["quality_source"] = jsonl_path
    if asr_path and os.path.exists(asr_path):
        sc["audio"] = audio_metrics(asr_path)
        sc["audio_source"] = asr_path
    return sc


# --------------------------------------------------------------------------
# Diff / no-regression gate
# --------------------------------------------------------------------------
def _get(d, path, default=None):
    for k in path.split("."):
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


# (metric path, higher_is_better, label)
_TRACKED = [
    ("matcher.clean_rate", True, "matcher clean"),
    ("matcher.false_relay_rate", False, "false-relay (NEG set)"),
    ("routing.pure_deterministic_coverage", True, "pure-deterministic coverage"),
    ("routing.deterministic_or_partial_coverage", True, "det-or-partial coverage"),
    # NB: higher deterministic coverage is itself the live latency+resource win
    # (each line moved off the CPU-3B path). The model-free det-path microsecond
    # timing below is informational only (~0.3ms, far under perception); the real
    # latency gate is the CPU-3B `--bench` (llm_path_ms). It is intentionally NOT
    # in this hard-gate list so sub-millisecond regex noise never blocks a release.
    ("quality.fact_retention.overall.mean", True, "fact-retention overall mean"),
    ("quality.fact_retention.overall.p50", True, "fact-retention overall p50"),
    ("quality.fact_retention.overall.p95", True, "fact-retention overall p95"),
    ("quality.fact_retention.count.mean", True, "count-token retention mean"),
    ("quality.fact_retention.count.p99", True, "count-token retention p99"),
    ("quality.fact_retention.owner.mean", True, "ownership retention mean"),
    ("quality.fact_retention.agent.mean", True, "agent retention mean"),
    ("quality.fact_retention.loc.mean", True, "location retention mean"),
    ("quality.inversion_rate", False, "inversion rate"),
    ("quality.hallucination_rate", False, "hallucination rate"),
    ("quality.flavor_type_token_ratio", True, "flavor type/token"),
    ("quality.flavor_max_repeat", False, "flavor max-repeat"),
    # iter5 personality metrics
    ("quality.flavor.flavor_coverage", True, "flavor coverage"),
    ("quality.flavor.contextual_match", True, "flavor contextual-match"),
    ("quality.flavor.voice_register_rate", True, "voice register rate"),
    ("quality.flavor.soundboard_max_repeat", False, "soundboard max-repeat"),
    # comprehensive-loop fidelity metrics
    ("quality.llm_flag_rate", False, "LLM-line flag rate"),
    ("quality.compound_zero_fact_loss", True, "compound zero-fact-loss"),
    # zero-tolerance gates (target 0)
    ("gates.oov_addressee_matches", False, "OOV-addressee match (0!)"),
    ("gates.fallback_malformed", False, "fallback malformed (0!)"),
    ("gates.isolation_flag_fail", False, "isolation-flag fail (0!)"),
    # audio / ASR fidelity
    ("audio.blips_per_1000", False, "audio blips / 1000"),
    ("audio.asr_coverage", True, "ASR coverage"),
]


def diff(prev: dict, cur: dict) -> tuple[str, bool]:
    lines = ["metric                              prev      ->  cur        verdict"]
    lines.append("-" * 74)
    regressed = False
    for path, hib, label in _TRACKED:
        pv, cv = _get(prev, path), _get(cur, path)
        if cv is None:
            continue
        if pv is None:
            lines.append(f"{label:34} {'--':>9}  ->  {cv:<9}  (new)")
            continue
        better = (cv > pv) if hib else (cv < pv)
        worse = (cv < pv) if hib else (cv > pv)
        eps = abs(pv) * 0.005 + 1e-9
        if abs(cv - pv) <= eps:
            verdict = "="
        elif better:
            verdict = "IMPROVED"
        else:
            verdict = "REGRESSED"
            regressed = True
        lines.append(f"{label:34} {pv:>9.4f}  ->  {cv:<9.4f}  {verdict}")
    lines.append("-" * 74)
    lines.append("RESULT: " + ("REGRESSION DETECTED" if regressed
                               else "no regression (all metrics held or improved)"))
    return "\n".join(lines), not regressed


# --------------------------------------------------------------------------
# --bench: real CPU-3B latency + peak RSS (the true gaming condition)
# --------------------------------------------------------------------------
def bench_llm(seed: int, n: int, tag: str) -> dict:
    """Load the gaming 3B on CPU (gpu_layers=0) -- the live gaming config -- and
    time build_relay_line on a sample of LLM-routed cases. Records p50/p95/p99
    latency + peak RSS."""
    os.environ["RELAY_TEST_GPU_LAYERS"] = "0"   # force CPU = the real gaming path
    import harness  # reuse the loader (testing-mode parity, CPU)
    llm = harness._load_llm()
    cases = build_corpus_10k(seed)
    import random as _r
    _r.Random(7).shuffle(cases)
    # pick cases that ROUTE to the LLM so we time the real model path
    samp_det, samp_llm = [], []
    for c in cases:
        cmd = match_relay_command(c.text)
        if cmd is None:
            continue
        route, _ = classify_route(cmd)
        (samp_det if route == "deterministic" else samp_llm).append(cmd)
        if len(samp_llm) >= n and len(samp_det) >= 50:
            break
    det_ms, llm_ms = [], []
    recent: list[str] = []
    for cmd in samp_det[:50]:
        t0 = time.perf_counter()
        build_relay_line(cmd, llm=llm, recent_lines=recent[-6:])
        det_ms.append((time.perf_counter() - t0) * 1e3)
    for cmd in samp_llm[:n]:
        t0 = time.perf_counter()
        line = build_relay_line(cmd, llm=llm, recent_lines=recent[-6:])
        llm_ms.append((time.perf_counter() - t0) * 1e3)
        if line:
            recent.append(line)
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1e6
    except Exception:
        import resource  # noqa
        rss_mb = None
    return {"tag": tag, "config": "CPU 3B (gpu_layers=0), testing-mode parity",
            "det_path_ms": _pcts(det_ms), "llm_path_ms": _pcts(llm_ms),
            "peak_rss_mb": round(rss_mb, 0) if rss_mb else "psutil-unavailable",
            "sampled_llm": len(llm_ms), "sampled_det": len(det_ms)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=None, help="rephrase JSONL for quality metrics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="sc")
    ap.add_argument("--prev", default=None, help="prior scorecard JSON to diff against")
    ap.add_argument("--asr-jsonl", default=None,
                    help="asr-stage JSONL for audio-blip + ASR-coverage metrics")
    ap.add_argument("--bench", action="store_true", help="run CPU-3B latency+RSS bench")
    ap.add_argument("--bench-n", type=int, default=40)
    args = ap.parse_args()

    out_dir = ROOT / "logs" / "relay_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bench:
        b = bench_llm(args.seed, args.bench_n, args.tag)
        bp = out_dir / f"bench_{args.tag}.json"
        bp.write_text(json.dumps(b, indent=2), encoding="utf-8")
        print(json.dumps(b, indent=2))
        return 0

    sc = build_scorecard(args.jsonl, args.seed, args.limit, args.tag,
                         asr_path=args.asr_jsonl)
    scp = out_dir / f"scorecard_{args.tag}.json"
    scp.write_text(json.dumps(sc, indent=2), encoding="utf-8")
    # human summary
    print(f"\n=== SCORECARD {args.tag} (seed={args.seed}, limit={args.limit}) ===")
    print(f"matcher: clean={sc['matcher']['clean_rate']:.4f}  "
          f"false-relay={sc['matcher']['false_relay_rate']:.4f} "
          f"({sc['matcher']['false_relay_count']}/{sc['matcher']['negative_set_size']})")
    r = sc["routing"]
    print(f"routes: {r['routes']}  pure-det={r['pure_deterministic_coverage']:.2%}  "
          f"det+partial={r['deterministic_or_partial_coverage']:.2%}")
    print(f"det-path latency (us): {r['det_path_latency_us']}")
    if "quality" in sc:
        q = sc["quality"]
        fr = q["fact_retention"]
        print(f"fact-retention overall: mean={fr['overall']['mean']} "
              f"p50={fr['overall']['p50']} p95={fr['overall']['p95']} "
              f"min={fr['overall']['min']}")
        for c in _FACT_CATS:
            print(f"  {c:8} {fr[c]}")
        print(f"inversion={q['inversion_rate']:.4f} ({q['inversion_count']})  "
              f"hallucination={q['hallucination_rate']:.4f} ({q['hallucination_count']})  "
              f"flavor TTR={q['flavor_type_token_ratio']} max-repeat={q['flavor_max_repeat']}")
        fv = q.get("flavor", {})
        if fv:
            print(f"PERSONALITY: coverage={fv['flavor_coverage']:.2%}  "
                  f"contextual-match={fv['contextual_match']:.2%}  "
                  f"voice-register={fv['voice_register_rate']:.2%}  "
                  f"soundboard-max-repeat={fv['soundboard_max_repeat']}")
        print(f"FIDELITY: LLM-flag-rate={q['llm_flag_rate']:.4f} ({q['llm_lines']} LLM)  "
              f"compound-zero-loss={q['compound_zero_fact_loss']:.4f} ({q['compound_n']})  "
              f"by-route={q.get('retention_by_route')}")
    g = sc.get("gates", {})
    if g:
        print(f"GATES (target 0): OOV-addressee={g['oov_addressee_matches']}/"
              f"{g['oov_addressee_n']}  fallback-malformed={g['fallback_malformed']}  "
              f"isolation-flag-fail={g['isolation_flag_fail']}/{g['isolation_flags_checked']}")
        if g.get("oov_examples"):
            print(f"  OOV examples: {g['oov_examples']}")
    a = sc.get("audio", {})
    if a:
        print(f"AUDIO: blips={a['blips']} ({a['blips_per_1000']}/1000)  "
              f"ASR-coverage={a['asr_coverage']:.4f}  no-speech={a['no_speech_lines']}  "
              f"(n={a['n_spoken']})")
    print(f"-> {scp}")

    if args.prev and os.path.exists(args.prev):
        prev = json.loads(open(args.prev, encoding="utf-8").read())
        txt, ok = diff(prev, sc)
        print("\n" + txt)
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
