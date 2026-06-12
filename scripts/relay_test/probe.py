r"""Quick isolation probe: rephrase a handful of suspect commands with NO
recent-line history, to separate real prompt bugs from the harness's
back-to-back recent_lines contamination. Loads the LLM once.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from kenning.audio.relay_speech import match_relay_command, build_relay_line
from kenning.llm.inference import LLMEngine
from kenning.memory.embedder import HybridEmbedder
from kenning.memory.qdrant_store import ConversationMemory

SUSPECTS = [
    "tell my team I am flanking",        # first person must survive
    "tell my team I am rotating",
    "tell my team to save",              # directive, not "I'm saving for op"
    "tell my team they are bots",        # insult, NOT the AI-identity line
    "tell my team they are terrible",
    "tell my phoenix to calm down",      # must address Phoenix, not Jett
    "tell my reyna to calm down",
    "tell my phoenix to flash for me",   # keep the ability "flash"
    "ask my jett how their day was",     # ASK, not answer-as-jett
    "tell my team there is one mid",
    "tell my team I saw one box",
    "tell my team to play their life",   # glossary: stay alive
    "tell my team I am anchoring",       # glossary: hold off-site
]

emb = HybridEmbedder()
mem = ConversationMemory(embedder=emb)
llm = LLMEngine(memory=mem)
try:
    llm.warmup()
except Exception:
    pass

for t in SUSPECTS:
    cmd = match_relay_command(t)
    if cmd is None:
        print(f"NONE | {t!r}"); continue
    line = build_relay_line(cmd, llm=llm, rephrase=True, recent_lines=[])
    print(f"IN  {t!r}\n -> {line!r}\n")
