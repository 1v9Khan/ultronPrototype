import sys, os, tempfile, pathlib

sys.path.insert(0, r"C:\STC\ultronPrototype")
sys.path.insert(0, r"C:\STC\ultronPrototype\src")
os.environ["ULTRON_LOG_LEVEL"] = "INFO"

from ultron.memory import ConversationMemory
from ultron.memory.embeddings import Embedder

emb = Embedder()
print(f"Embedder dim={emb.dim}")

with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "memory.jsonl"
    mem = ConversationMemory(path=path, embedder=emb)
    mem.add("user", "we decided to use sqlite for the cache")
    mem.add("assistant", "noted")
    mem.add("user", "lets refactor the auth module")
    mem.add("assistant", "okay")
    mem.add("user", "whats the weather today")
    mem.add("assistant", "i dont have a sensor for that")

    # Query something semantically near the auth turn but using different words.
    hits = mem.retrieve("rewrite the login flow", k=2, exclude_recent=2)
    print('RAG hits for "rewrite the login flow":')
    for h in hits:
        print(f"  - {h.role}: {h.content}")
