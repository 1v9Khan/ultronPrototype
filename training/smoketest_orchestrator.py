"""Construct the full orchestrator without entering run().

Verifies all components load cleanly, then shuts down. Useful when changing
pipeline wiring to catch breakage before launching the full app.
"""

import sys
import os

sys.path.insert(0, r"C:\STC\ultronPrototype")
sys.path.insert(0, r"C:\STC\ultronPrototype\src")
os.environ["ULTRON_LOG_LEVEL"] = "INFO"

from ultron.utils.logging import configure_logging
configure_logging()

from ultron.pipeline import Orchestrator

print("Constructing orchestrator…")
orch = Orchestrator()
print(f"  state={orch._state.value}")
print(f"  memory={'on' if orch.memory is not None else 'off'}"
      f" turns={len(orch.memory) if orch.memory is not None else 0}")
print(f"  rvc={'on' if orch.rvc is not None else 'off'}")
print(f"  wake_word={orch.wake.active_word} (fallback={orch.wake.using_fallback})")
print("Tearing down…")
orch.shutdown()
print("OK")
