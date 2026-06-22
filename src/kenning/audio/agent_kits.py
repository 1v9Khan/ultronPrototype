"""Ultron 1.0 — Valorant agent-kit reference for LLM context injection.

Hot-swappable, VERSION-STAMPED per-agent kit facts injected into the relay/answer prompt so the LLM
never hallucinates an agent's kit (it mis-stated Sova's kit in early probing). The compact format
(~30 tokens/agent) is sourced from ``docs/ultron_1_0/02_research/board/B_valorant_kits.md`` with the
adversarially-verified corrections from ``C_domain.md`` applied inline (Iso Undercut now also
suppresses; Clove Not Dead Yet = 8 pts; Veto Evolution = 7 pts).

To update for a new patch/agent: edit ``AGENT_KITS`` below and bump ``KITS_VERSION`` — NO code change
(the loader just reads this dict). The LLM's training cutoff (~late 2024) cannot know Waylay (Mar 2025),
Veto (Oct 2025), or Miks (Mar 2026), nor Iso's Feb-2025 suppression change, so those are grounded here.

Anticheat-safe: pure data + stdlib (no heavy imports, nothing on a desktop-interaction surface).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

KITS_VERSION = "v2026-06-20 (Patch 12.10)"

# Canonical agent name -> "Role | C=.. Q=.. E=..(/sig) X=..(ult)". Compact (relay-accuracy) form.
# Corrections vs the raw B_valorant_kits compact block are marked [C_domain].
AGENT_KITS: Dict[str, str] = {
    # --- Duelists ---
    "Jett": "Duelist | C=Cloudburst(smoke), Q=Updraft(jump), E=Tailwind(dash/sig), X=Blade Storm(knives ult,7pts)",
    "Phoenix": "Duelist | C=Blaze(firewall), Q=Hot Hands(fireball+selfheal), E=Curveball(flash/sig), X=Run It Back(respawn ult,8pts)",
    "Raze": "Duelist | C=Boom Bot, Q=Blast Pack(satchel), E=Paint Shells(cluster nade/sig), X=Showstopper(rocket ult,8pts)",
    "Reyna": "Duelist | C=Leer(nearsight eye), Q=Devour(soul heal), E=Dismiss(soul invuln/sig), X=Empress(frenzy ult,7pts)",
    "Neon": "Duelist | C=Fast Lane(twin walls), Q=Relay Bolt(concuss), E=High Gear(sprint+slide/sig), X=Overdrive(beam ult,8pts)",
    "Yoru": "Duelist | C=Fakeout(decoy), Q=Blindside(flash), E=Gatecrash(teleport/sig), X=Dimensional Drift(invis ult,7pts)",
    "Iso": "Duelist | C=Contingency(bullet wall), Q=Undercut(fragile+suppress 4s,1chg,300cr)[C_domain], E=Double Tap(shield/sig), X=Kill Contract(1v1 ult,7pts)",
    "Waylay": "Duelist | C=Saturate(hinder AoE), Q=Light Speed(double dash), E=Refract(beacon teleport,invuln/sig), X=Convergent Paths(hinder zone ult,8pts) [post-cutoff: ground here]",
    # --- Initiators ---
    "Sova": "Initiator | C=Owl Drone(dart reveal), Q=Shock Bolt(damage), E=Recon Bolt(sonar scan/sig), X=Hunter's Fury(3 wall-pen blasts ult,8pts)",
    "Breach": "Initiator | C=Aftershock(through-wall burst), Q=Flashpoint(through-wall flash), E=Fault Line(daze/sig), X=Rolling Thunder(quake ult,8pts)",
    "Skye": "Initiator | C=Regrowth(team heal), Q=Trailblazer(tiger concuss), E=Guiding Light(steerable hawk flash/sig), X=Seekers(tracking nearsight ult,7pts)",
    "KAY/O": "Initiator | C=FRAG/ment(explosive), Q=FLASH/drive(flash), E=ZERO/point(ability suppress/sig), X=NULL/cmd(area suppress+overload ult,8pts)",
    "Fade": "Initiator | C=Prowler(nearsight), Q=Seize(tether+deafen+decay), E=Haunt(reveal eye/sig), X=Nightfall(map mark+deafen+decay ult,8pts)",
    "Gekko": "Initiator | C=Mosh Pit(AoE explosion), Q=Dizzy(blind), E=Wingman(clear/plant/defuse,sig), X=Thrash(detain ult,7pts); creatures reclaimable",
    "Tejo": "Initiator | C=Stealth Drone(dart reveal), Q=Special Delivery(sticky concuss), E=Guided Salvo(2 guided missiles/sig), X=Armageddon(airstrike ult,8pts) [post-cutoff: ground here]",
    # --- Controllers ---
    "Brimstone": "Controller | C=Stim Beacon(rapidfire), Q=Incendiary(fire), E=Sky Smoke(3 smokes/sig), X=Orbital Strike(beam ult,7pts)",
    "Viper": "Controller | C=Snake Bite(acid+fragile), Q=Poison Cloud(gas orb), E=Toxic Screen(gas wall/sig), X=Viper's Pit(gas cloud ult,8pts)",
    "Omen": "Controller | C=Shrouded Step(short teleport), Q=Paranoia(nearsight), E=Dark Cover(smoke/sig), X=From the Shadows(global teleport ult,7pts)",
    "Astra": "Controller | C=Gravity Well(pull), Q=Nova Pulse(concuss), E=Nebula/Dissipate(smoke/sig), X=Cosmic Divide(audio-dampening wall ult,8pts)",
    "Harbor": "Controller | C=High Tide(slow water wall,purchasable), Q=Storm Surge(whirlpool nearsight), E=Cove(bulletproof water smoke/sig), X=Reckoning(cone surge ult,8pts) [post-rework 11.10]",
    "Clove": "Controller | C=Pick-Me-Up(overheal on kill/assist), Q=Meddle(decay), E=Ruse(smokes/sig,usable after death), X=Not Dead Yet(self-revive ult,8pts)[C_domain]",
    "Miks": "Controller | C=M-Pulse(concuss OR team-heal waves), Q=Harmonize(combat stim ally), E=Waveform(map smokes/sig), X=Bassquake(knockback+deafen+slow ult,8pts) [post-cutoff: ground here]",
    # --- Sentinels ---
    "Sage": "Sentinel | C=Barrier Orb(ice wall), Q=Slow Orb(slow field), E=Healing Orb(heal/sig), X=Resurrection(revive ult,8pts)",
    "Cypher": "Sentinel | C=Trapwire, Q=Cyber Cage(vision block), E=Spycam(camera+dart/sig), X=Neural Theft(reveal ult,7pts)",
    "Killjoy": "Sentinel | C=Nanoswarm(damage field), Q=Alarmbot(vulnerable), E=Turret(auto-turret/sig), X=Lockdown(area detain ult,8pts)",
    "Chamber": "Sentinel | C=Trademark(slow trap), Q=Headhunter(pistol), E=Rendezvous(teleport anchors/sig), X=Tour De Force(sniper ult,8pts)",
    "Deadlock": "Sentinel | C=GravNet(force-crouch), Q=Sonic Sensor(sound stun), E=Barrier Mesh(wall/sig), X=Annihilation(cocoon ult,8pts)",
    "Vyse": "Sentinel | C=Razorvine(slow+dmg), Q=Shear(spike wall trap), E=Arc Rose(wall flash/sig), X=Steel Garden(weapon-jam ult,7pts)",
    "Veto": "Sentinel | C=Crosscut(teleport anchor), Q=Chokehold(tether+decay trap), E=Interceptor(destroys enemy utility/sig), X=Evolution(debuff-immunity+regen+stim ult,7pts,until death)[C_domain] [post-cutoff: ground here]",
}

# Lowercase index for tolerant lookup (the relay layer supplies canonical names, but be forgiving).
_LC_INDEX = {k.lower(): k for k in AGENT_KITS}


def _canon(agent: str) -> Optional[str]:
    if not agent:
        return None
    a = agent.strip()
    if a in AGENT_KITS:
        return a
    return _LC_INDEX.get(a.lower())


def agent_kit_fact(agent: Optional[str]) -> Optional[str]:
    """Return the compact 'Agent: Role | ...' kit fact line for one agent, or None if unknown."""
    canon = _canon(agent or "")
    if canon is None:
        return None
    return f"{canon}: {AGENT_KITS[canon]}"


def kit_facts_for(agents: Sequence[str], *, limit: int = 4) -> List[str]:
    """Return compact kit-fact lines for the given agents (canonical names), de-duped, capped.

    The cap keeps the injected context lean (research: prompt length raises hallucination risk).
    """
    out: List[str] = []
    seen: set[str] = set()
    for a in agents or ():
        canon = _canon(a)
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        out.append(f"{canon}: {AGENT_KITS[canon]}")
        if len(out) >= limit:
            break
    return out
