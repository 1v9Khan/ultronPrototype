"""Core data model for ultron's clean-room self-improvement system.

Catalog 13 (clawhub-capability-evolver) -- clean-room synthesis. NONE of
the upstream plugin's code was read, imported, executed, or
deobfuscated (its evolution engine ships as obfuscated JavaScript and
its network stack phones home to a paid remote hub). This module
reconstructs ONLY the GREEN, pure-data "GEP" (Genome Evolution Protocol)
schema documented in the reference catalog + the read-only test/spec
scan reports, re-implemented from scratch for ultron's local-only,
data-only, zero-network architecture.

The four headline types:

* :class:`Gene` -- a reusable "how to respond to this class of signal"
  strategy template.
* :class:`Capsule` -- a recorded successful evolution outcome bound to a
  gene + the signals that triggered it.
* :class:`EvolutionEvent` -- an append-only audit record of one cycle,
  stamped with the agent's :class:`PersonalityState`.
* :class:`PersonalityState` -- five response-temperament traits in
  ``[0, 1]`` that the system tunes from outcome feedback.

Plus the supporting records (:class:`Mutation`, :class:`GeneConstraints`,
:class:`Outcome`, :class:`BlastRadius`, :class:`EnvFingerprint`,
:class:`LearningHistoryEntry`, :class:`AntiPatternEntry`) and
content-addressable hashing helpers (:func:`canonicalize`,
:func:`compute_asset_id`, :func:`verify_asset_id`).

**Deliberate ultron-specific safety departures from the upstream schema:**

* No ``a2a`` / ``eligible_to_broadcast`` field -- ultron never broadcasts
  assets to any network. Genes + capsules are local-only data.
* :class:`EnvFingerprint` carries ONLY non-identifying local fields
  (platform, python version, capture time). The upstream's stable
  hardware ``device_id`` (machine-id / IOPlatformUUID / MAC-address
  harvest) is deliberately omitted -- it existed solely to correlate
  installs at a remote hub, which ultron has no concept of.
* Every type is a frozen dataclass. "Adaptation" returns a NEW object
  (:meth:`Gene.with_learning`) rather than mutating shared state, so a
  gene can never be silently rewritten out from under a caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform as _platform
import sys
import time
import uuid
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

# Fresh ultron-owned schema version (the upstream's was 1.6.0; this is a
# clean-room reimplementation so it starts its own line at 1.0.0).
EVOLUTION_SCHEMA_VERSION: str = "1.0.0"

# Per-gene default blast-radius ceiling. The upstream's hand-written
# genes allowed 20-25 file changes; distilled (auto-synthesised) genes
# are capped tighter because they are machine-generated.
DEFAULT_GENE_MAX_FILES: int = 20
DISTILLED_GENE_MAX_FILES: int = 12

# Paths a gene may never touch (the per-gene floor; the global protected
# set lives in :mod:`ultron.evolution.blast_radius`).
DEFAULT_FORBIDDEN_PATHS: tuple[str, ...] = (".git", "node_modules")

# Id prefixes for machine-synthesised genes (success vs failure path).
DISTILLED_ID_PREFIX: str = "gene_distilled_"
REPAIR_DISTILLED_ID_PREFIX: str = "gene_repair_distilled_"


class EvolutionCategory(str, Enum):
    """The intent class of a gene / mutation.

    ``"destroy"`` and any other value are deliberately NOT members -- a
    mutation can only ever repair, optimise, or innovate.
    """

    REPAIR = "repair"
    OPTIMIZE = "optimize"
    INNOVATE = "innovate"


class OutcomeStatus(str, Enum):
    """Terminal status of an evolution cycle's outcome."""

    SUCCESS = "success"
    FAILED = "failed"


class RiskLevel(str, Enum):
    """Risk tier of a proposed mutation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def clamp01(value: Any) -> float:
    """Clamp ``value`` to the closed unit interval ``[0.0, 1.0]``.

    Returns ``0.0`` for any non-finite or non-numeric input (``None``,
    ``NaN``, ``±Infinity``, a non-number) -- mirroring the upstream
    ``clamp01`` contract so a corrupted trait score can never escape the
    valid range.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_event_id() -> str:
    """Generate a fresh evolution-event id (``evt_<16 hex>``)."""
    return "evt_" + uuid.uuid4().hex[:16]


def new_capsule_id() -> str:
    """Generate a fresh, collision-free capsule id (``capsule_<epoch-ms>_<6 hex>``).

    The random suffix guarantees uniqueness even when many capsules are
    minted within the same millisecond -- otherwise the content-dedup in
    the distiller would silently collapse them.
    """
    return f"capsule_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def new_mutation_id() -> str:
    """Generate a fresh mutation id (``mut_<12 hex>``)."""
    return "mut_" + uuid.uuid4().hex[:12]


def new_gene_id(prefix: str = "gene_") -> str:
    """Generate a fresh gene id with the given prefix (``<prefix><10 hex>``)."""
    return f"{prefix}{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Content-addressable hashing (clean-room of contentHash.js, GREEN)
# ---------------------------------------------------------------------------


def _normalise_for_hash(value: Any) -> Any:
    """Recursively normalise ``value`` into JSON-serialisable primitives
    with a deterministic shape.

    Dataclasses become field-keyed dicts; enums become their ``.value``;
    sequences become lists; non-finite floats become ``None`` (the
    upstream's "non-finite -> null" rule). The result is safe to feed to
    :func:`json.dumps` with ``sort_keys=True``.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _normalise_for_hash(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _normalise_for_hash(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_for_hash(v) for v in value]
    if isinstance(value, Enum):
        return _normalise_for_hash(value.value)
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def canonicalize(obj: Any) -> str:
    """Render ``obj`` to a deterministic canonical JSON string.

    Keys are sorted, separators are tight, non-finite floats serialise as
    ``null``. Two structurally-equal objects always produce byte-identical
    output, which is the basis for content-addressable ids + dedup.
    """
    return json.dumps(
        _normalise_for_hash(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_asset_id(obj: Any, *, exclude_fields: Sequence[str] = ("asset_id",)) -> str:
    """Compute the content-addressable id of ``obj`` as ``sha256:<hex>``.

    The fields named in ``exclude_fields`` (the ``asset_id`` field itself
    by default) are dropped before hashing so an object's id never depends
    on its own id.
    """
    norm = _normalise_for_hash(obj)
    if isinstance(norm, dict):
        for name in exclude_fields:
            norm.pop(name, None)
    payload = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def verify_asset_id(obj: Any) -> bool:
    """Return True iff ``obj``'s recorded ``asset_id`` matches a fresh
    computation (tamper / corruption detection)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        recorded = getattr(obj, "asset_id", "")
    elif isinstance(obj, Mapping):
        recorded = obj.get("asset_id", "")
    else:
        return False
    if not recorded:
        return False
    return recorded == compute_asset_id(obj)


# ---------------------------------------------------------------------------
# Small coercion helpers shared by the frozen dataclasses
# ---------------------------------------------------------------------------


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """Coerce ``value`` to a tuple of strings. ``None`` -> ``()``; a bare
    string -> a single-element tuple (NEVER split into characters)."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _dedupe_preserve(values: Sequence[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication of a string sequence."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return tuple(out)


# ---------------------------------------------------------------------------
# Supporting records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvFingerprint:
    """A NON-identifying local environment snapshot.

    Deliberately omits any stable hardware identifier (the upstream's
    ``device_id`` / hostname / MAC harvest) because ultron never transmits
    this anywhere -- it is purely local diagnostic metadata stamped on a
    capsule so a future cross-environment comparison can group outcomes by
    coarse platform.
    """

    platform: str = ""
    python_version: str = ""
    captured_at: str = ""

    @classmethod
    def capture(cls) -> "EnvFingerprint":
        """Capture the current process's coarse, non-identifying env."""
        return cls(
            platform=f"{_platform.system()}-{_platform.machine()}".strip("-"),
            python_version=".".join(str(x) for x in sys.version_info[:3]),
            captured_at=_now_iso(),
        )


@dataclass(frozen=True)
class BlastRadius:
    """The size of a change: file count + line churn."""

    files: int = 0
    lines: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", max(0, int(self.files)))
        object.__setattr__(self, "lines", max(0, int(self.lines)))


@dataclass(frozen=True)
class Outcome:
    """The result of an evolution cycle: a status + a quality score."""

    status: OutcomeStatus = OutcomeStatus.SUCCESS
    score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", OutcomeStatus(self.status))
        object.__setattr__(self, "score", clamp01(self.score))

    @property
    def succeeded(self) -> bool:
        """True iff the status is :attr:`OutcomeStatus.SUCCESS`."""
        return self.status is OutcomeStatus.SUCCESS


@dataclass(frozen=True)
class GeneConstraints:
    """Per-gene blast-radius + protected-path constraints."""

    max_files: int = DEFAULT_GENE_MAX_FILES
    forbidden_paths: tuple[str, ...] = DEFAULT_FORBIDDEN_PATHS

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_files", max(1, int(self.max_files)))
        object.__setattr__(self, "forbidden_paths", _as_str_tuple(self.forbidden_paths))

    @classmethod
    def coerce(cls, value: Any) -> "GeneConstraints":
        """Build a :class:`GeneConstraints` from an existing instance, a
        mapping (``{"max_files": .., "forbidden_paths": [..]}``), or
        ``None`` (defaults)."""
        if isinstance(value, GeneConstraints):
            return value
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            return cls(
                max_files=value.get("max_files", DEFAULT_GENE_MAX_FILES),
                forbidden_paths=value.get("forbidden_paths", DEFAULT_FORBIDDEN_PATHS),
            )
        return cls()


@dataclass(frozen=True)
class LearningHistoryEntry:
    """One recorded outcome appended to a gene's learning history."""

    outcome: OutcomeStatus
    mode: str = "none"
    signals: tuple[str, ...] = ()
    at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", OutcomeStatus(self.outcome))
        object.__setattr__(self, "signals", _as_str_tuple(self.signals))
        object.__setattr__(self, "at", self.at or _now_iso())


@dataclass(frozen=True)
class AntiPatternEntry:
    """A recorded failure mode that down-weights a gene during selection."""

    mode: str = "hard"
    learning_signals: tuple[str, ...] = ()
    at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "learning_signals", _as_str_tuple(self.learning_signals))
        object.__setattr__(self, "at", self.at or _now_iso())


@dataclass(frozen=True)
class PersonalityState:
    """Five adaptive response-temperament traits, each in ``[0, 1]``.

    These tune ultron's response *shaping* (verbosity, hedging, etc.) via
    :mod:`ultron.evolution.personality` -> ``response_style`` -- they
    NEVER touch the locked voice character (SOUL.md / RVC / Piper / the
    TTS voicepack).
    """

    rigor: float = 0.5
    creativity: float = 0.5
    verbosity: float = 0.5
    risk_tolerance: float = 0.5
    obedience: float = 0.5

    def __post_init__(self) -> None:
        for name in ("rigor", "creativity", "verbosity", "risk_tolerance", "obedience"):
            object.__setattr__(self, name, clamp01(getattr(self, name)))

    @classmethod
    def balanced(cls) -> "PersonalityState":
        """The neutral default (all traits at 0.5)."""
        return cls()

    def is_high_risk(self) -> bool:
        """Whether this temperament is "high risk" (rigor < 0.4 OR
        risk_tolerance >= 0.7) -- used to down-grade an ``innovate``
        mutation to ``optimize``."""
        return self.rigor < 0.4 or self.risk_tolerance >= 0.7

    def allows_high_risk_mutation(self) -> bool:
        """Whether a high-risk mutation is permitted (rigor >= 0.6 AND
        risk_tolerance <= 0.5) -- the conservative AND-gate."""
        return self.rigor >= 0.6 and self.risk_tolerance <= 0.5

    def with_trait(self, name: str, value: float) -> "PersonalityState":
        """Return a copy with one trait nudged to ``value`` (clamped)."""
        if name not in ("rigor", "creativity", "verbosity", "risk_tolerance", "obedience"):
            raise ValueError(f"unknown personality trait: {name!r}")
        return replace(self, **{name: clamp01(value)})


# ---------------------------------------------------------------------------
# The three headline asset types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gene:
    """A reusable strategy template: "how to respond to this class of
    signal".

    A gene matches a set of signals (:attr:`signals_match`), carries an
    ordered list of natural-language :attr:`strategy` steps, and bounds
    the change it may produce (:attr:`constraints`). It accumulates a
    :attr:`learning_history` of outcomes and :attr:`anti_patterns` of
    failures that adjust its selection score over time.
    """

    id: str
    category: EvolutionCategory
    signals_match: tuple[str, ...] = ()
    strategy: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    constraints: GeneConstraints = field(default_factory=GeneConstraints)
    validation: tuple[str, ...] = ()
    summary: str = ""
    execution_mode: str = ""
    learning_history: tuple[LearningHistoryEntry, ...] = ()
    anti_patterns: tuple[AntiPatternEntry, ...] = ()
    schema_version: str = EVOLUTION_SCHEMA_VERSION
    type: str = "Gene"

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", EvolutionCategory(self.category))
        object.__setattr__(self, "signals_match", _as_str_tuple(self.signals_match))
        object.__setattr__(self, "strategy", _as_str_tuple(self.strategy))
        object.__setattr__(self, "preconditions", _as_str_tuple(self.preconditions))
        object.__setattr__(self, "validation", _as_str_tuple(self.validation))
        object.__setattr__(self, "constraints", GeneConstraints.coerce(self.constraints))
        object.__setattr__(self, "learning_history", tuple(self.learning_history))
        object.__setattr__(self, "anti_patterns", tuple(self.anti_patterns))

    @property
    def is_inplace(self) -> bool:
        """Whether this gene is restricted to parameter-only changes."""
        return self.execution_mode == "inplace"

    @property
    def is_distilled(self) -> bool:
        """Whether this gene was machine-synthesised by the distiller."""
        return self.id.startswith(DISTILLED_ID_PREFIX) or self.id.startswith(
            REPAIR_DISTILLED_ID_PREFIX
        )

    def with_learning(
        self,
        *,
        outcome: OutcomeStatus,
        learning_signals: Sequence[str] = (),
        mode: str = "none",
        at: Optional[str] = None,
    ) -> "Gene":
        """Return a NEW gene adapted from a learning outcome.

        On SUCCESS: extends :attr:`signals_match` with the ``problem:*``
        and ``area:*`` tags from ``learning_signals`` (NOT ``action:*``
        tags) and appends a success entry to :attr:`learning_history`.

        On FAILURE: does NOT extend :attr:`signals_match`; appends an
        :class:`AntiPatternEntry` + a failure entry to
        :attr:`learning_history`.

        The original gene is never mutated.
        """
        outcome = OutcomeStatus(outcome)
        stamp = at or _now_iso()
        signals = _as_str_tuple(learning_signals)
        history = self.learning_history + (
            LearningHistoryEntry(outcome=outcome, mode=mode, signals=signals, at=stamp),
        )
        if outcome is OutcomeStatus.SUCCESS:
            new_tags = tuple(
                s for s in signals if s.startswith("problem:") or s.startswith("area:")
            )
            new_signals = _dedupe_preserve(self.signals_match + new_tags)
            return replace(self, signals_match=new_signals, learning_history=history)
        anti = self.anti_patterns + (
            AntiPatternEntry(mode=mode or "hard", learning_signals=signals, at=stamp),
        )
        return replace(self, learning_history=history, anti_patterns=anti)


@dataclass(frozen=True)
class Mutation:
    """A proposed change produced by a cycle's plan phase.

    The category is gated by personality (a high-risk temperament
    down-grades ``innovate`` to ``optimize``) and the risk level is capped
    when high risk is not permitted -- both applied by the builder in
    :mod:`ultron.evolution.signals` / the loop, not here.
    """

    id: str
    category: EvolutionCategory
    trigger_signals: tuple[str, ...] = ()
    target: str = ""
    expected_effect: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    rationale: str = ""
    type: str = "Mutation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", EvolutionCategory(self.category))
        object.__setattr__(self, "risk_level", RiskLevel(self.risk_level))
        object.__setattr__(self, "trigger_signals", _as_str_tuple(self.trigger_signals))


@dataclass(frozen=True)
class Capsule:
    """A recorded successful evolution outcome.

    Bound to the :attr:`gene` that produced it (or ``"ad_hoc"``) and the
    :attr:`trigger` signals that were present. The :attr:`asset_id` is the
    content hash of every field except itself (auto-computed when empty).
    """

    id: str
    trigger: tuple[str, ...] = ()
    gene: str = "ad_hoc"
    summary: str = ""
    confidence: float = 0.0
    blast_radius: BlastRadius = field(default_factory=BlastRadius)
    outcome: Outcome = field(default_factory=Outcome)
    success_streak: int = 0
    env_fingerprint: EnvFingerprint = field(default_factory=EnvFingerprint.capture)
    schema_version: str = EVOLUTION_SCHEMA_VERSION
    type: str = "Capsule"
    asset_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger", _as_str_tuple(self.trigger))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "success_streak", max(0, int(self.success_streak)))
        if not isinstance(self.blast_radius, BlastRadius):
            br = self.blast_radius or {}
            object.__setattr__(
                self,
                "blast_radius",
                BlastRadius(files=br.get("files", 0), lines=br.get("lines", 0))
                if isinstance(br, Mapping)
                else BlastRadius(),
            )
        if not isinstance(self.outcome, Outcome):
            oc = self.outcome or {}
            object.__setattr__(
                self,
                "outcome",
                Outcome(status=oc.get("status", OutcomeStatus.SUCCESS), score=oc.get("score", 0.0))
                if isinstance(oc, Mapping)
                else Outcome(),
            )
        if not self.asset_id:
            object.__setattr__(self, "asset_id", compute_asset_id(self))


@dataclass(frozen=True)
class EvolutionEvent:
    """An append-only audit record of one evolution cycle.

    Stamped with the agent's :class:`PersonalityState` so a later analysis
    can rank which temperament produced the best outcomes. The
    :attr:`asset_id` is the content hash of every field except itself
    (auto-computed when empty).
    """

    id: str
    intent: str = ""
    signals: tuple[str, ...] = ()
    genes_used: tuple[str, ...] = ()
    mutation_id: str = ""
    personality_state: PersonalityState = field(default_factory=PersonalityState.balanced)
    blast_radius: BlastRadius = field(default_factory=BlastRadius)
    outcome: Outcome = field(default_factory=Outcome)
    parent: Optional[str] = None
    capsule_id: Optional[str] = None
    created_at: str = ""
    schema_version: str = EVOLUTION_SCHEMA_VERSION
    type: str = "EvolutionEvent"
    asset_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "signals", _as_str_tuple(self.signals))
        object.__setattr__(self, "genes_used", _as_str_tuple(self.genes_used))
        if not self.created_at:
            object.__setattr__(self, "created_at", _now_iso())
        if not isinstance(self.personality_state, PersonalityState):
            ps = self.personality_state or {}
            object.__setattr__(
                self,
                "personality_state",
                PersonalityState(**ps) if isinstance(ps, Mapping) else PersonalityState.balanced(),
            )
        if not isinstance(self.blast_radius, BlastRadius):
            br = self.blast_radius or {}
            object.__setattr__(
                self,
                "blast_radius",
                BlastRadius(files=br.get("files", 0), lines=br.get("lines", 0))
                if isinstance(br, Mapping)
                else BlastRadius(),
            )
        if not isinstance(self.outcome, Outcome):
            oc = self.outcome or {}
            object.__setattr__(
                self,
                "outcome",
                Outcome(status=oc.get("status", OutcomeStatus.SUCCESS), score=oc.get("score", 0.0))
                if isinstance(oc, Mapping)
                else Outcome(),
            )
        if not self.asset_id:
            object.__setattr__(self, "asset_id", compute_asset_id(self))


__all__ = [
    "EVOLUTION_SCHEMA_VERSION",
    "DEFAULT_GENE_MAX_FILES",
    "DISTILLED_GENE_MAX_FILES",
    "DEFAULT_FORBIDDEN_PATHS",
    "DISTILLED_ID_PREFIX",
    "REPAIR_DISTILLED_ID_PREFIX",
    "EvolutionCategory",
    "OutcomeStatus",
    "RiskLevel",
    "clamp01",
    "canonicalize",
    "compute_asset_id",
    "verify_asset_id",
    "new_event_id",
    "new_capsule_id",
    "new_mutation_id",
    "new_gene_id",
    "EnvFingerprint",
    "BlastRadius",
    "Outcome",
    "GeneConstraints",
    "LearningHistoryEntry",
    "AntiPatternEntry",
    "PersonalityState",
    "Gene",
    "Mutation",
    "Capsule",
    "EvolutionEvent",
]
