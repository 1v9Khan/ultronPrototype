"""Runtime service that drives ultron's evolution loop end-to-end.

Catalog 13 clean-room. This bundles the engine modules into a single
orchestrator-facing surface so the orchestrator's own wiring stays tiny +
obviously fail-open:

* :class:`EvolutionStore` -- lock-guarded, append-only JSONL persistence
  under ``data/evolution/`` (success capsules, failure records, a
  hash-chained audit ledger, the gate state, the personality profile);
* :class:`EvolutionService` -- holds the autonomy controller + personality
  tuner + the :class:`~ultron.evolution.evolution_loop.EvolutionLoop`,
  records per-turn satisfaction + success capsules, runs cycles
  single-flight (on a daemon thread for the autonomous trigger), and
  exposes the temperament hint + the periodic digest.

Everything is fail-open: a construction or runtime failure degrades to a
disabled service / a no-op, never to a crashed voice path. Zero network.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ultron.evolution.autonomy import TieredAutonomyController
from ultron.evolution.evolution_loop import (
    ApplyStatus,
    CheckpointHook,
    EvolutionLoop,
    EvolutionLoopConfig,
    EvolutionState,
)
from ultron.evolution.guardrails import GuardrailBaseline, GuardrailSample
from ultron.evolution.models import (
    Capsule,
    Outcome,
    OutcomeStatus,
    PersonalityState,
    canonicalize,
    new_capsule_id,
)
from ultron.evolution.personality import (
    PersonalityFeedback,
    PersonalityTuner,
    apply_temperament,
)
from ultron.evolution.signals import COSMETIC_SIGNALS, has_opportunity_signal, signal_base
from ultron.utils.logging import get_logger

logger = get_logger("evolution.service")

_GENESIS_HASH = "0" * 64
DEFAULT_CAPSULE_LOAD_LIMIT = 400
DEFAULT_CYCLE_CHECK_INTERVAL_TURNS = 25
DEFAULT_TURN_CAPSULE_SCORE = 0.8
PERSONALITY_SAVE_EVERY_TURNS = 10


class EvolutionStore:
    """Lock-guarded append-only JSONL persistence under ``data/evolution/``."""

    def __init__(self, data_dir: Path | str) -> None:
        self._dir = Path(data_dir)
        self._lock = threading.RLock()
        self.capsules_path = self._dir / "capsules.jsonl"
        self.failed_path = self._dir / "failed_capsules.jsonl"
        self.events_path = self._dir / "events.jsonl"
        self.state_path = self._dir / "state.json"
        self.personality_path = self._dir / "personality.json"

    def _append(self, path: Path, line: str) -> None:
        with self._lock:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception as exc:  # noqa: BLE001
                logger.debug("evolution store append failed (%s): %s", path.name, exc)

    def _read_lines(self, path: Path) -> list[str]:
        with self._lock:
            try:
                if not path.exists():
                    return []
                with open(path, encoding="utf-8") as fh:
                    return fh.readlines()
            except Exception as exc:  # noqa: BLE001
                logger.debug("evolution store read failed (%s): %s", path.name, exc)
                return []

    # -- capsules -----------------------------------------------------------

    def append_capsule(self, capsule: Capsule) -> None:
        self._append(self.capsules_path, canonicalize(capsule))

    def load_recent_capsules(self, limit: int = DEFAULT_CAPSULE_LOAD_LIMIT) -> list[dict]:
        return self._parse_jsonl(self.capsules_path, limit)

    def count_capsules(self) -> int:
        return len([ln for ln in self._read_lines(self.capsules_path) if ln.strip()])

    # -- failures -----------------------------------------------------------

    def append_failure(self, failure: dict) -> None:
        self._append(self.failed_path, json.dumps(failure, ensure_ascii=False))

    def load_failures(self, limit: int = DEFAULT_CAPSULE_LOAD_LIMIT) -> list[dict]:
        return self._parse_jsonl(self.failed_path, limit)

    def _parse_jsonl(self, path: Path, limit: int) -> list[dict]:
        out: list[dict] = []
        for ln in self._read_lines(path)[-limit:]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:  # noqa: BLE001 -- skip a torn / malformed tail line
                continue
        return out

    # -- hash-chained audit ledger ------------------------------------------

    def append_event(self, event: Any) -> None:
        with self._lock:
            prev = self._last_event_hash()
            body = canonicalize(event)
            digest = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
            try:
                row = json.dumps(
                    {"hash": digest, "prev": prev, "event": json.loads(body)}, ensure_ascii=False
                )
            except Exception:  # noqa: BLE001
                return
            self._append(self.events_path, row)

    def _last_event_hash(self) -> str:
        for ln in reversed(self._read_lines(self.events_path)):
            ln = ln.strip()
            if not ln:
                continue
            try:
                return str(json.loads(ln).get("hash", _GENESIS_HASH))
            except Exception:  # noqa: BLE001
                continue
        return _GENESIS_HASH

    def verify_event_chain(self) -> tuple[bool, Optional[int]]:
        """Re-walk the audit ledger; return ``(ok, first_break_index)``."""
        prev = _GENESIS_HASH
        for idx, ln in enumerate(self._read_lines(self.events_path)):
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
                body = canonicalize(row.get("event", {}))
                expected = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
            except Exception:  # noqa: BLE001
                return (False, idx)
            if row.get("prev") != prev or row.get("hash") != expected:
                return (False, idx)
            prev = expected
        return (True, None)

    # -- state + personality ------------------------------------------------

    def load_state(self) -> EvolutionState:
        with self._lock:
            try:
                if self.state_path.exists():
                    data = json.loads(self.state_path.read_text(encoding="utf-8"))
                    return EvolutionState(
                        last_distillation_at=data.get("last_distillation_at"),
                        last_data_hash=str(data.get("last_data_hash", "")),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("evolution state load failed: %s", exc)
        return EvolutionState()

    def save_state(self, state: EvolutionState) -> None:
        with self._lock:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                self.state_path.write_text(
                    json.dumps(
                        {
                            "last_distillation_at": state.last_distillation_at,
                            "last_data_hash": state.last_data_hash,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("evolution state save failed: %s", exc)

    def load_personality(self) -> dict:
        with self._lock:
            try:
                if self.personality_path.exists():
                    return json.loads(self.personality_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.debug("evolution personality load failed: %s", exc)
        return {}

    def save_personality(self, data: dict) -> None:
        with self._lock:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                self.personality_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.debug("evolution personality save failed: %s", exc)


def _build_checkpoint(data_dir: Path, proposal_dir: Path) -> Optional[CheckpointHook]:
    """Build a shadow-repo checkpoint over the proposal directory, or
    ``None`` (the loop falls back to delete-revert)."""
    try:
        from ultron.checkpoints.registry import CheckpointRegistry

        registry = CheckpointRegistry(checkpoints_root=data_dir.parent / "checkpoints")
        manager = registry.get_or_create("evolution-skills", workspace_path=proposal_dir)

        def _take() -> str:
            commit = manager.on_event("evolution", force=True)
            return commit.commit_hash if commit is not None else ""

        def _restore(token: str) -> bool:
            if not token:
                return False
            plan = manager.plan_workspace_rewind(target_commit_hash=token)
            outcome = manager.restore(plan)
            return bool(outcome.workspace_reset_succeeded)

        return CheckpointHook(take=_take, restore=_restore)
    except Exception as exc:  # noqa: BLE001
        logger.debug("evolution checkpoint unavailable (delete-revert fallback): %s", exc)
        return None


def _maybe_get_approval() -> Any:
    try:
        from ultron.safety.two_phase_approval import get_approval_registry

        return get_approval_registry()
    except Exception:  # noqa: BLE001
        return None


class EvolutionService:
    """The orchestrator-facing evolution runtime."""

    def __init__(
        self,
        *,
        config: Any,
        store: EvolutionStore,
        autonomy: TieredAutonomyController,
        personality: PersonalityTuner,
        loop: EvolutionLoop,
        state: EvolutionState,
        proposal_dir: Path,
        registry_reloader: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._store = store
        self._autonomy = autonomy
        self._personality = personality
        self._loop = loop
        self._state = state
        self._proposal_dir = proposal_dir
        self._registry_reloader = registry_reloader
        self._clock = clock
        self._cycle_lock = threading.Lock()
        self._turns_since_check = 0
        self._closed = False

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        project_root: Path | str,
        registry_reloader: Optional[Callable[[], None]] = None,
        guardrail_sampler: Optional[Callable[[], GuardrailSample]] = None,
        approval: Any = None,
        clock: Callable[[], float] = time.time,
    ) -> Optional["EvolutionService"]:
        """Build the full service from config, or ``None`` when evolution is
        disabled / construction fails (fail-open)."""
        ev = getattr(config, "evolution", None)
        if ev is None or not getattr(ev, "enabled", False):
            return None
        try:
            data_dir = Path(project_root) / "data" / "evolution"
            proposal_dir = data_dir / "skills"
            proposal_dir.mkdir(parents=True, exist_ok=True)
            store = EvolutionStore(data_dir)
            state = store.load_state()
            autonomy = TieredAutonomyController(
                pause_on_demote=bool(getattr(ev, "pause_on_demote", False))
            )
            personality = PersonalityTuner.from_dict(store.load_personality())
            checkpoint = _build_checkpoint(data_dir, proposal_dir)
            sampler = guardrail_sampler or (lambda: GuardrailSample())
            approval_registry = approval if approval is not None else _maybe_get_approval()
            loop = EvolutionLoop(
                repo_root=Path(project_root),
                proposal_dir=proposal_dir,
                capsules_provider=store.load_recent_capsules,
                autonomy=autonomy,
                baseline=GuardrailBaseline(),
                guardrail_sampler=sampler,
                checkpoint=checkpoint,
                approval=approval_registry,
                audit_sink=store.append_event,
                capsule_sink=None,  # capsules come from real turns, not from keeping a proposal
                failure_sink=store.append_failure,
                personality_provider=lambda: personality.state,
                state=state,
                config=EvolutionLoopConfig(
                    surface="skills",
                    enabled=True,
                    max_steps=int(getattr(ev, "max_steps", 3)),
                ),
                clock=clock,
            )
            return cls(
                config=ev,
                store=store,
                autonomy=autonomy,
                personality=personality,
                loop=loop,
                state=state,
                proposal_dir=proposal_dir,
                registry_reloader=registry_reloader,
                clock=clock,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("evolution service construction failed: %s", exc)
            return None

    # -- per-turn -----------------------------------------------------------

    def record_turn(
        self,
        *,
        user_text: str = "",
        signals: Sequence[str] = (),
        corrected: bool = False,
        re_asked: bool = False,
        barged_in: bool = False,
        response_summary: str = "",
    ) -> None:
        """Record a turn's satisfaction signals (tune the temperament) and,
        on a successfully-handled opportunity, a success capsule that feeds
        future distillation. Fail-open."""
        if self._closed:
            return
        try:
            feedback = PersonalityFeedback(corrected=corrected, re_asked=re_asked, barged_in=barged_in)
            self._personality.record_feedback(feedback)
            self._personality.record_outcome(1.0 if feedback.satisfied else 0.0)
            if feedback.satisfied and has_opportunity_signal(signals):
                opportunity = [s for s in signals if has_opportunity_signal([s])]
                capsule = Capsule(
                    id=new_capsule_id(),
                    trigger=opportunity or list(signals),
                    gene="ad_hoc",
                    summary=(response_summary or user_text)[:200],
                    confidence=DEFAULT_TURN_CAPSULE_SCORE,
                    outcome=Outcome(status=OutcomeStatus.SUCCESS, score=DEFAULT_TURN_CAPSULE_SCORE),
                    success_streak=1,
                )
                self._store.append_capsule(capsule)
            self._turns_since_check += 1
            if self._turns_since_check % PERSONALITY_SAVE_EVERY_TURNS == 0:
                self._store.save_personality(self._personality.to_dict())
        except Exception as exc:  # noqa: BLE001
            logger.debug("evolution record_turn failed: %s", exc)

    def maybe_run_autonomous_cycle(self) -> None:
        """If enough turns have elapsed and no cycle is running, run one on a
        daemon thread (single-flight, off the hot path). Fail-open."""
        if self._closed:
            return
        try:
            interval = int(getattr(self._config, "cycle_check_interval_turns", DEFAULT_CYCLE_CHECK_INTERVAL_TURNS))
            if self._turns_since_check < interval:
                return
            self._turns_since_check = 0
            if not self._cycle_lock.acquire(blocking=False):
                return  # a cycle is already running

            def _bg() -> None:
                try:
                    self._do_cycle()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("evolution autonomous cycle failed: %s", exc)
                finally:
                    self._cycle_lock.release()

            try:
                threading.Thread(target=_bg, name="evolution-cycle", daemon=True).start()
            except Exception:  # noqa: BLE001
                self._cycle_lock.release()
        except Exception as exc:  # noqa: BLE001
            logger.debug("evolution maybe_run_autonomous_cycle failed: %s", exc)

    # -- cycle --------------------------------------------------------------

    def run_cycle(self) -> dict:
        """Run a single evolution cycle now (the voice-command entry point).
        Single-flight: returns ``{"status": "busy"}`` if a cycle is already
        running. Never raises."""
        if self._closed:
            return {"status": "disabled"}
        if not self._cycle_lock.acquire(blocking=False):
            return {"status": "busy"}
        try:
            return self._do_cycle()
        finally:
            self._cycle_lock.release()

    def _do_cycle(self) -> dict:
        try:
            result = self._loop.run_once()
        except Exception as exc:  # noqa: BLE001
            logger.warning("evolution loop run failed: %s", exc)
            return {"status": "error", "error": str(exc)}
        try:
            self._store.save_state(self._state)
        except Exception:  # noqa: BLE001
            pass
        if result is None:
            return {"status": "no_proposal"}
        if result.status is ApplyStatus.KEPT and self._registry_reloader is not None:
            try:
                self._registry_reloader()
            except Exception as exc:  # noqa: BLE001
                logger.debug("evolution registry reload failed: %s", exc)
        return {
            "status": result.status.value,
            "slug": result.proposal.slug,
            "reasons": list(result.reasons),
        }

    # -- temperament --------------------------------------------------------

    def temperament_hint(self) -> str:
        """The current response-shaping hint (``""`` when balanced)."""
        try:
            return self._personality.current_hint()
        except Exception:  # noqa: BLE001
            return ""

    def apply_temperament(self, user_text: str) -> str:
        """Prepend the current temperament hint to ``user_text`` (fail-open)."""
        if self._closed or not getattr(self._config, "apply_temperament", True):
            return user_text
        try:
            return apply_temperament(user_text, self._personality.state)
        except Exception:  # noqa: BLE001
            return user_text

    # -- reporting ----------------------------------------------------------

    def digest(self) -> str:
        """The multi-line periodic digest (autonomy + personality ranking)."""
        try:
            return f"{self._autonomy.digest()}\n{self._personality.report()}"
        except Exception:  # noqa: BLE001
            return "Evolution digest unavailable."

    def status_line(self) -> str:
        """A short, TTS-safe one-line status for a voice query."""
        try:
            surfaces = [
                self._autonomy.state(s)
                for s in self._autonomy.known_surfaces()
                if self._autonomy.state(s).applied > 0
            ]
            kept = sum(s.kept for s in surfaces)
            reverted = sum(s.reverted for s in surfaces)
            capsules = self._store.count_capsules()
            return (
                f"I've recorded {capsules} learning samples; "
                f"kept {kept} self-improvements and auto-reverted {reverted}."
            )
        except Exception:  # noqa: BLE001
            return "Evolution is active."

    @property
    def autonomy(self) -> TieredAutonomyController:
        return self._autonomy

    @property
    def personality(self) -> PersonalityTuner:
        return self._personality

    @property
    def store(self) -> EvolutionStore:
        return self._store

    def shutdown(self) -> None:
        """Persist state + personality and stop accepting work."""
        try:
            self._store.save_state(self._state)
            self._store.save_personality(self._personality.to_dict())
        except Exception as exc:  # noqa: BLE001
            logger.debug("evolution shutdown persist failed: %s", exc)
        self._closed = True


__all__ = [
    "EvolutionStore",
    "EvolutionService",
]
