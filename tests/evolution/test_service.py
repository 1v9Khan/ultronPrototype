"""Tests for ultron.evolution.service -- the runtime bundle + JSONL store.
Hermetic: all persistence goes to tmp_path; no network, no model loads."""

from __future__ import annotations

from types import SimpleNamespace

from ultron.evolution.evolution_loop import EvolutionState
from ultron.evolution.models import Capsule, EvolutionEvent, Outcome, OutcomeStatus
from ultron.evolution.service import EvolutionService, EvolutionStore


def _cfg(**kw):
    ev = SimpleNamespace(
        enabled=True,
        max_steps=3,
        cycle_check_interval_turns=1000,
        pause_on_demote=False,
        apply_temperament=True,
    )
    for k, v in kw.items():
        setattr(ev, k, v)
    return SimpleNamespace(evolution=ev)


def _data_dir(tmp_path):
    return tmp_path / "data" / "evolution"


# --- EvolutionStore ---------------------------------------------------------


def test_store_capsules(tmp_path):
    store = EvolutionStore(_data_dir(tmp_path))
    cap = Capsule(id="c1", gene="ad_hoc", trigger=["capability_gap"], outcome=Outcome(status=OutcomeStatus.SUCCESS, score=0.8))
    store.append_capsule(cap)
    loaded = store.load_recent_capsules()
    assert len(loaded) == 1
    assert loaded[0]["gene"] == "ad_hoc"
    assert loaded[0]["outcome"]["status"] == "success"
    assert store.count_capsules() == 1


def test_store_failures(tmp_path):
    store = EvolutionStore(_data_dir(tmp_path))
    store.append_failure({"gene": "g", "reason_class": "validation"})
    assert store.load_failures()[0]["reason_class"] == "validation"


def test_store_event_chain_verifies(tmp_path):
    store = EvolutionStore(_data_dir(tmp_path))
    store.append_event(EvolutionEvent(id="e1", intent="optimize"))
    store.append_event(EvolutionEvent(id="e2", intent="repair"))
    ok, idx = store.verify_event_chain()
    assert ok is True
    assert idx is None


def test_store_event_chain_detects_tamper(tmp_path):
    store = EvolutionStore(_data_dir(tmp_path))
    store.append_event(EvolutionEvent(id="e1", intent="optimize"))
    # corrupt the ledger
    lines = store.events_path.read_text(encoding="utf-8").splitlines()
    store.events_path.write_text(lines[0].replace("optimize", "tampered") + "\n", encoding="utf-8")
    ok, idx = store.verify_event_chain()
    assert ok is False


def test_store_state_roundtrip(tmp_path):
    store = EvolutionStore(_data_dir(tmp_path))
    assert store.load_state().last_data_hash == ""
    store.save_state(EvolutionState(last_distillation_at=123.0, last_data_hash="abc"))
    s = store.load_state()
    assert s.last_distillation_at == 123.0
    assert s.last_data_hash == "abc"


def test_store_personality_roundtrip(tmp_path):
    store = EvolutionStore(_data_dir(tmp_path))
    store.save_personality({"rigor": 0.7})
    assert store.load_personality()["rigor"] == 0.7


# --- EvolutionService.from_config -------------------------------------------


def test_from_config_disabled_returns_none(tmp_path):
    assert EvolutionService.from_config(_cfg(enabled=False), project_root=tmp_path) is None


def test_from_config_enabled(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    assert svc is not None
    assert (tmp_path / "data" / "evolution" / "skills").exists()


# --- record_turn ------------------------------------------------------------


def test_record_turn_appends_success_capsule(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    svc.record_turn(signals=["capability_gap"], response_summary="explained the gap")
    assert svc.store.count_capsules() == 1


def test_record_turn_no_capsule_when_unsatisfied(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    svc.record_turn(signals=["capability_gap"], corrected=True)
    assert svc.store.count_capsules() == 0


def test_record_turn_no_capsule_without_opportunity(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    svc.record_turn(signals=["stable_success_plateau"])
    assert svc.store.count_capsules() == 0


def test_record_turn_tunes_personality(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    svc.record_turn(corrected=True)
    assert svc.personality.state.rigor > 0.5


# --- run_cycle --------------------------------------------------------------


def test_run_cycle_no_proposal_when_empty(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    assert svc.run_cycle()["status"] == "no_proposal"


def test_run_cycle_keeps_and_writes_skill(tmp_path):
    reloaded = []
    svc = EvolutionService.from_config(
        _cfg(), project_root=tmp_path, registry_reloader=lambda: reloaded.append(1)
    )
    for _ in range(10):
        svc.record_turn(signals=["capability_gap"], response_summary="resolved a capability gap")
    out = svc.run_cycle()
    assert out["status"] == "kept"
    assert reloaded == [1]
    skills_dir = tmp_path / "data" / "evolution" / "skills"
    md_files = [p for p in skills_dir.iterdir() if p.suffix == ".md"]
    assert md_files


# --- autonomous trigger -----------------------------------------------------


def test_maybe_autonomous_cycle_gating_and_thread(tmp_path):
    svc = EvolutionService.from_config(_cfg(cycle_check_interval_turns=1), project_root=tmp_path)
    svc.record_turn(signals=["capability_gap"])  # turns_since_check -> 1
    svc.maybe_run_autonomous_cycle()  # >= interval -> spawns a daemon cycle
    # wait for the background cycle to finish by acquiring its lock
    acquired = svc._cycle_lock.acquire(timeout=5)
    assert acquired
    svc._cycle_lock.release()
    # below the interval -> no-op (counter untouched)
    svc._turns_since_check = 0
    svc.maybe_run_autonomous_cycle()
    assert svc._turns_since_check == 0


# --- temperament ------------------------------------------------------------


def test_temperament_hint_after_barge_ins(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    for _ in range(5):
        svc.record_turn(barged_in=True)
    assert "concise" in svc.temperament_hint()
    assert svc.apply_temperament("what is x").startswith("[Tone:")


def test_apply_temperament_disabled(tmp_path):
    svc = EvolutionService.from_config(_cfg(apply_temperament=False), project_root=tmp_path)
    for _ in range(5):
        svc.record_turn(barged_in=True)
    assert svc.apply_temperament("hello") == "hello"


# --- reporting + shutdown ---------------------------------------------------


def test_digest_and_status(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    svc.record_turn(signals=["capability_gap"])
    assert svc.digest()
    assert svc.status_line()


def test_shutdown_persists(tmp_path):
    svc = EvolutionService.from_config(_cfg(), project_root=tmp_path)
    svc.record_turn(corrected=True)
    svc.shutdown()
    assert (tmp_path / "data" / "evolution" / "personality.json").exists()
    # closed service is a no-op
    svc.record_turn(signals=["capability_gap"])
    assert svc.store.count_capsules() == 0
