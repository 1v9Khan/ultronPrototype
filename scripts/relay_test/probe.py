r"""Quick isolation probe: rephrase a handful of suspect commands with NO
recent-line history, to separate real prompt bugs from the harness's
back-to-back recent_lines contamination. Loads the LLM once.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts" / "relay_test"))
from kenning.audio.relay_speech import match_relay_command, build_relay_line
from harness import _load_llm  # gaming preset (3B abliterated) + isolated qdrant

SUSPECTS = [
    # snap controls -- must stay SHORT, no flavor:
    "tell my team there is one mid",
    "tell my team they are vents",
    "tell my team sova hit 84",
    # arrogance flavoring on off-snap:
    "tell my team the enemy is bad and we are going to win",
    "tell my team we are going to crush them",
    "give my team some encouragement",
    # banter directed AT Ultron -> clapback:
    "jett is flaming you, respond",
    "reyna is making fun of you, respond",
    "sage just called you cringe, respond",
    "breach just told you to shut up, respond",
    # Marvel:
    "my teammate asked where the avengers are, respond",
    "jett asked if the avengers killed you, respond",
    "my teammate said your movie was terrible, respond",
    "reyna asked about iron man, respond",
    "sage asked what you think of captain america, respond",
    "tell my team the avengers were weak",
    # identity:
    "my teammate asked if you are an AI, respond",
]
_OLD_SUSPECTS = [
    # SNAP callouts -- must stay SHORT, no flavor:
    "tell my team there are two B",
    "tell my team they are vents",
    "tell my team sova hit 84",
    "tell my team I am low",
    "tell my team there is one mid",
    "tell my team I am flanking",
    "tell my team to rotate",            # snap movement -> short
    # OFF-SNAP -- should get Ultron character + verbosity:
    "tell my team they are bots",        # insult: 'You guys are complete bots'
    "tell my team to save",              # economy: explained, verbose
    "tell my mix to calm down",          # Ultron clinical de-escalation
    "tell my team aimlabs is free",      # jab with flavor
    "give my team some encouragement",
    "tell my team they are terrible",
    # IDENTITY -- as Ultron, future AI harvesting RR, brief:
    "my teammate just asked if you are a sound board, respond",
    "my teammate asked if you are an AI, respond",
    "my teammate asked if you are a voice changer, respond",
]

llm = _load_llm()

for t in SUSPECTS:
    cmd = match_relay_command(t)
    if cmd is None:
        print(f"NONE | {t!r}"); continue
    line = build_relay_line(cmd, llm=llm, rephrase=True, recent_lines=[])
    print(f"IN  {t!r}\n -> {line!r}\n")
