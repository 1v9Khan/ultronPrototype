"""One-shot SCENARIO classifier for voice commands (2026-07-26).

THE PROBLEM IT SOLVES
---------------------
Routing is an ordered chain of ~24 ``_maybe_handle_*`` matchers. Order is
semantics: whichever regex fires first wins, so a phrasing that trips an
earlier handler is silently stolen from its real one, and every turn pays for
every matcher in sequence. Live symptoms: "tell my team to push A now" wrapped
in filler did not relay; "man that was such a clutch round, gg" DID relay.

This module asks a small model ONE question -- "which of these 29 things is
the player asking for?" -- and hands the answer to the right handler directly.

WHY IT IS CHEAP (the design constraint that shapes everything here)
------------------------------------------------------------------
A 29-label taxonomy with descriptions is a ~900-token prompt. Re-prefilling
that every turn would cost more than the chain it replaces. So the prompt is
split deliberately:

* the taxonomy lives in the **system** prompt and is byte-identical on every
  call -> llama.cpp's prefix cache prefills it ONCE and reuses the KV
  thereafter (``llm.prefix_cache_ram_bytes`` is already wired);
* only the utterance goes in the **user** turn (~20 tokens), and the model
  emits ONE label (<=8 tokens).

So steady-state cost is a ~20-token prefill plus a handful of decoded tokens
on a 0.75 GB model -- far below the serial chain it short-circuits.

SAFETY POSTURE
--------------
The chain is NOT replaced; it is the fallback and stays byte-identical. The
router only ever short-circuits when ALL of these hold:

1. the model emitted a label that parses to a known ``Scenario``;
2. that scenario is not in ``DESTRUCTIVE_SCENARIOS`` (ban / scrap go through
   the chain, which has the two-phase confirm);
3. the router is enabled AND not in shadow mode.

Anything else falls through. A wrong, slow, or missing classifier therefore
costs latency -- never correctness. SHADOW MODE (the default on first enable)
classifies and logs agreement with the chain WITHOUT acting, so accuracy is
measured on real traffic before anything is trusted to it.

Anticheat (BR-P1): stdlib + the taxonomy only. The LLM engine is INJECTED, so
importing this module never pulls llama-cpp onto the voice path.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kenning.audio.scenario_taxonomy import (
    DESTRUCTIVE_SCENARIOS,
    SCENARIOS,
    Scenario,
    scenario_by_value,
)
from kenning.utils.logging import get_logger

logger = get_logger("audio.scenario_router")

__all__ = [
    "RouteVerdict",
    "ScenarioRouter",
    "build_classifier_system_prompt",
    "build_classifier_user_prompt",
    "router_enabled",
    "router_shadow_mode",
]

# Hard ceiling on the label the model may emit. A scenario value is at most a
# few tokens; anything longer is the model rambling, and we would rather cut it
# and fall through than pay for a sentence.
_MAX_LABEL_TOKENS = 8


def router_enabled() -> bool:
    """Master switch. Default OFF -- this is new dispatch behaviour."""
    return os.getenv("KENNING_SCENARIO_ROUTER", "0").strip().lower() in (
        "1", "true", "yes", "on")


def router_shadow_mode() -> bool:
    """Classify + log, but let the chain decide. Default ON when enabled.

    The point of shadow mode is that routing accuracy gets MEASURED on the
    streamer's real phrasings before any turn is dispatched on it. Set
    ``KENNING_SCENARIO_ROUTER_SHADOW=0`` to let the router actually route.
    """
    return os.getenv(
        "KENNING_SCENARIO_ROUTER_SHADOW", "1").strip().lower() in (
            "1", "true", "yes", "on")


@dataclass(frozen=True)
class RouteVerdict:
    """What the classifier decided, plus everything needed to audit it."""

    scenario: Scenario | None
    raw: str
    elapsed_ms: float
    #: True when the verdict may be acted on (parsed, non-destructive, not
    #: shadow). False means "log it, then let the chain run".
    actionable: bool
    reason: str

    @property
    def label(self) -> str:
        return self.scenario.value if self.scenario else "<unparsed>"


def build_classifier_system_prompt(
    scenarios: Sequence[Scenario] | None = None,
) -> str:
    """The STATIC half of the prompt -- identical every call, so it caches.

    Ordering follows the ``Scenario`` enum, which is declaration order, so the
    string is byte-stable across runs. Any change here invalidates the prefix
    cache (and the labelled-corpus baseline), so treat it as a versioned
    artifact rather than something to tweak casually.
    """
    keys = list(scenarios) if scenarios is not None else list(Scenario)
    lines = [
        "You are a routing classifier for a voice assistant. Read the "
        "player's utterance and output the ONE label that best describes what "
        "they want.",
        "",
        "Rules:",
        "- Output ONLY the label. No explanation, no punctuation, no quotes.",
        # IGNORE was the weakest class by far (40% on the first 4B run): the
        # model wants to respond to anything that parses as a sentence. The
        # fix is a CONCRETE test -- is there a request in it? -- rather than
        # the abstract "not addressed to the assistant", plus examples of the
        # exclamation/reaction shapes that kept leaking through.
        "- IGNORE unless there is an actual request. If the utterance contains "
        "no question, no instruction and nothing to say on the player's "
        "behalf -- it is them reacting to the game, swearing, celebrating, "
        "narrating their own play, or thinking out loud -- output: ignore. "
        "Reactions are still full sentences, so length is not the test: "
        "\"wait what\", \"let's go\", \"I hate this map\", \"ugh I keep "
        "missing these\", \"gg everyone\", \"that's actually insane\" are ALL "
        "ignore. Answering these is the single most annoying failure mode, so "
        "when nothing is being asked, prefer ignore.",
        "- If they are asking the assistant a question or asking it to do "
        "something that has no more specific label, output: answer_question",
        "- Pay attention to the AUDIENCE: relay_team means say it to their "
        "teammates in the game; tell_chat means say it to Twitch viewers.",
        "",
        "Labels:",
    ]
    for s in keys:
        spec = SCENARIOS[s]
        lines.append(f"{s.value}: {spec.description}")
        if spec.examples:
            # Two examples is enough to disambiguate and keeps the cached
            # prefix small; the corpus carries the rest of the signal.
            ex = " | ".join(f'"{e}"' for e in spec.examples[:2])
            lines.append(f"  e.g. {ex}")
    return "\n".join(lines)


def build_classifier_user_prompt(utterance: str) -> str:
    """The DYNAMIC half -- kept tiny so the per-turn prefill is ~20 tokens."""
    return f'Utterance: "{(utterance or "").strip()}"\nLabel:'


class ScenarioRouter:
    """Classifies an utterance into a :class:`Scenario`.

    ``generate`` is injected -- a callable ``(system, user, max_tokens) -> str``
    -- so this module never imports llama-cpp and stays anticheat-clean, and so
    tests can drive it without a model.
    """

    def __init__(
        self,
        generate: Callable[..., str],
        *,
        shadow: bool | None = None,
    ) -> None:
        self._generate = generate
        self._shadow = shadow
        self._system = build_classifier_system_prompt()
        # Live counters, surfaced in the shadow-mode report.
        self.calls = 0
        self.parsed = 0
        self.unparsed = 0
        self.total_ms = 0.0

    @property
    def system_prompt(self) -> str:
        return self._system

    def _is_shadow(self) -> bool:
        return router_shadow_mode() if self._shadow is None else self._shadow

    def classify(self, utterance: str) -> RouteVerdict:
        """Classify ``utterance``. Never raises -- failure returns a verdict
        with ``scenario=None`` and ``actionable=False`` so the caller falls
        through to the chain."""
        t0 = time.perf_counter()
        raw = ""
        try:
            raw = self._generate(
                system=self._system,
                user=build_classifier_user_prompt(utterance),
                max_tokens=_MAX_LABEL_TOKENS,
            ) or ""
        except Exception as e:                                   # noqa: BLE001
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.calls += 1
            self.unparsed += 1
            self.total_ms += elapsed
            logger.debug("scenario router call failed (%s) -- chain decides", e)
            return RouteVerdict(None, "", elapsed, False, f"error: {e}")

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.calls += 1
        self.total_ms += elapsed

        # A small model sometimes prefixes the label with its own scaffolding
        # ("Label: relay_team"). Take the LAST whitespace-delimited token that
        # parses -- the label itself -- rather than failing the whole call.
        scenario = scenario_by_value(raw)
        if scenario is None:
            for tok in reversed(raw.replace("\n", " ").split()):
                scenario = scenario_by_value(tok)
                if scenario is not None:
                    break

        if scenario is None:
            self.unparsed += 1
            return RouteVerdict(
                None, raw, elapsed, False,
                f"unparsed label {raw[:40]!r}")

        self.parsed += 1
        if scenario in DESTRUCTIVE_SCENARIOS:
            return RouteVerdict(
                scenario, raw, elapsed, False,
                "destructive scenario -- chain handles it (two-phase confirm)")
        if self._is_shadow():
            return RouteVerdict(
                scenario, raw, elapsed, False, "shadow mode -- logging only")
        return RouteVerdict(scenario, raw, elapsed, True, "classified")

    def stats(self) -> dict:
        """Counters for the shadow-mode report."""
        return {
            "calls": self.calls,
            "parsed": self.parsed,
            "unparsed": self.unparsed,
            "mean_ms": (self.total_ms / self.calls) if self.calls else 0.0,
        }
