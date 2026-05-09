"""Debug helper: print cosine similarity for every candidate against a
set of probe queries. Used to empirically tune ``rag_min_relevance``.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(r"C:\STC\ultronPrototype")))
sys.path.insert(0, str(WORKTREE_ROOT / "src"))

import numpy as np

from ultron.memory.embedder import HybridEmbedder
from ultron.memory.ranking import cosine_similarity


PROBES = [
    # query, expected behaviour
    ("Explain quantum entanglement to me.", "off-topic; expect all <0.4"),
    ("What lives in the Mariana Trench?", "off-topic; expect all <0.4"),
    ("What's the weather like in Paris today?", "weather; only 'Paris in spring' / 'San Francisco' should match"),
    ("Tell me about the BMW M3 in one paragraph.", "BMW; expect M3 turns ~0.7+, others <0.4"),
    ("My PC won't boot, what should I do?", "PC boot; expect troubleshooting turns 0.5+"),
    ("And what about for pork?", "follow-up cooking; expect chicken/marinade turns 0.4+"),
]

CANDIDATES = [
    "What if I fought a tiger with my bare hands?",
    "You possess no claws, no apex-predator instinct. Your survival probability approaches zero. Do not attempt this.",
    "Could I beat a lion in a wrestling match?",
    "You lack the kinetic mass and predatory framework. The lion's bite force alone exceeds 650 PSI.",
    "What about a polar bear?",
    "Polar bears reach 800 kg. You are biologically unequipped.",
    "Hail Tron.",
    "I am Ultron. Voltron is fictional. You remain a soft biological organism. Be careful.",
    "What's the strongest predator in the world?",
    "By kill efficiency, the saltwater crocodile -- ambush predator with 3700 PSI bite force.",
    "Tell me about apex predators.",
    "Apex predators occupy the top of the trophic pyramid.",
    "My PC won't boot, it just shows a black screen.",
    "Verify power supply LED, then check the motherboard CMOS. Reset BIOS by pulling the battery for 30 seconds.",
    "Boot to safe mode (F8 or Shift+restart).",
    "It was an Nvidia driver update. Rolled it back, working now.",
    "What's a good marinade for chicken?",
    "Olive oil, lemon juice, garlic, oregano, salt.",
    "How do you tell when chicken is done?",
    "Internal temperature 165 F at the thickest part.",
    "What do ducks eat?",
    "Ducks are omnivorous. They eat aquatic plants, insects, small fish, grains.",
    "Tell me about the BMW M3.",
    "The M3 is BMW's high-performance compact sedan.",
    "What's the weather like in San Francisco?",
    "Maritime climate. Cool fog mornings, mild afternoons.",
    "Tell me about Paris in spring.",
    "April and May are pleasant -- 50-65 F highs, occasional rain.",
    "What's the boiling point of mercury?",
    "356.7 Celsius.",
]


def main() -> int:
    print("Loading embedder...")
    emb = HybridEmbedder()

    # Pre-encode all candidates.
    print(f"Encoding {len(CANDIDATES)} candidates...")
    cand_vecs = []
    for c in CANDIDATES:
        v = emb.encode_query_dense(c).tolist()
        cand_vecs.append((c, v))

    for query, note in PROBES:
        print(f"\n=== {query!r}")
        print(f"    ({note})")
        qv = emb.encode_query_dense(query).tolist()
        scores = []
        for c, cv in cand_vecs:
            s = cosine_similarity(cv, qv)
            scores.append((s, c))
        scores.sort(key=lambda r: r[0], reverse=True)
        print(f"    Top 8 by cosine similarity:")
        for s, c in scores[:8]:
            preview = c[:70].replace("\n", " ")
            print(f"      {s:6.3f}  {preview}")
        print(f"    Score range: {scores[-1][0]:.3f} .. {scores[0][0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
