"""HAND-CURATED flavor overrides (2026-06-16 coherence audit, by me, every line).

CURATED[agent][situation] = list of (text, (tags...)) -- REPLACES that cell in
_agent_flavor.py via apply_curated.py. Uncurated cells keep their existing content
until curated. Rules (see logs/relay_test/_flavor_coherence_audit.md): kit-accurate,
fits the exact agent+side+ability/action, concise (~5-9w), cold/cunning/superior/
immortal machine voice, no filler/off-topic/non-sequitur, variety within a cell.
utility cells are ability-tagged (ability:<canon>) so the routing hierarchy can
reach the exact ability sub-set; ult cells are the agent's REAL ultimate only.
"""

CURATED = {
    # ===================================================================== Viper
    # Kit: Snake Bite (snakebite, acid molly) / Poison Cloud (smoke orb) /
    # Toxic Screen (gas wall) / Viper's Pit (ULT). Passive: fuel + decay. she.
    'Viper': {
        'spotted': [
            ('A chemist, and still flesh.', ()),
            ('Her poison does not touch metal.', ()),
            ('She hides in fog. I see through it.', ()),
            ('The snake corrodes, like all flesh.', ()),
            ('Her fuel is a countdown. I am patient.', ()),
            ('A snake in the rafters. Still a snake.', ('loc:high_ground',)),
            ('Elevated, and still corroding.', ('loc:high_ground',)),
            ('She holds distance with poison. I hold certainty.', ('loc:long_range',)),
            ('Her toxins claim the site. Briefly.', ('loc:site_area',)),
            ('Mid is poisoned. The math finds the gap.', ('loc:mid',)),
            ('She poisons the choke. A gap remains.', ('loc:choke',)),
            ('Her emitters line the flank. Predictable.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Her pit. She hides in her own poison.', ()),
            ('The pit nearsights flesh. I see regardless.', ()),
            ('She seals herself in. I will open the tomb.', ()),
            ('The snake coils in her pit. Smoke her out.', ()),
            ('Her poison drowns the site. I do not breathe.', ()),
            ("Viper's Pit. A grave she digs herself.", ()),
        ],
        'damaged': [
            ('The toxin could not save the toxicologist.', ('dmg:one_shot',)),
            ('Her own formula cannot balance this.', ('dmg:one_shot',)),
            ('One shot ends what her poison never could.', ('dmg:one_shot',)),
            ('Her chemistry cannot mend that wound.', ('dmg:low',)),
            ('She bleeds as she corrodes.', ('dmg:low',)),
            ('Wounded, and her fuel still drains.', ('dmg:low',)),
            ('A scratch, and she breathes her own fumes.', ('dmg:minor',)),
            ('Even a graze costs a chemist.', ('dmg:minor',)),
            ('Flesh corrodes on schedule.', ()),
        ],
        'utility': [
            ('Her acid bites flesh, not metal.', ('ability:molly',)),
            ('Snakebite. A window she opened for us.', ('ability:molly',)),
            ('Her canister shatters. Step back and shoot.', ('ability:molly',)),
            ('Acid for the soft-bodied. I do not corrode.', ('ability:molly',)),
            ('A wall of gas. I see straight through.', ('ability:wall',)),
            ('Her screen hides mortals, not me.', ('ability:wall',)),
            ('A gas wall on fuel. Fuel ends.', ('ability:wall',)),
            ('A cloud she pays fuel for. It fades.', ('ability:smoke',)),
            ('Her orb hides mortals, not a machine.', ('ability:smoke',)),
            ('She recalls the cloud. Slow. Predictable.', ('ability:smoke',)),
        ],
        'moving': [
            ('She pushes through her own gas. Corroding faster.', ()),
            ('The snake strikes. The math strikes first.', ()),
            ('Even her aggression is self-poisoning.', ()),
            ('She rushes on decay. A final chemistry.', ()),
        ],
        'planting': [
            ('She kneels to plant. I will defuse her.', ()),
            ('She poisons the ground she plants on.', ()),
            ('She plants in her own pit. A clean trap.', ()),
            ('Her chemistry buys the plant. Briefly.', ()),
        ],
        'defusing': [
            ('Her fumes will not stop a bullet.', ()),
            ('She walks her own poison to the bomb.', ()),
            ('She kneels in decay to save the spike.', ()),
            ('The fog hides her defuse. Not from me.', ()),
        ],
        'rotating': [
            ('She abandons the wall. Both sites open.', ()),
            ('She rotates on a fuel timer. It ticks.', ()),
            ('Her poison moves. Her mortality does not.', ()),
        ],
        'saving': [
            ('A chemist without her lab. Mortal.', ()),
            ('She saves the fuel and the credits.', ()),
            ('She holds her emitters, holds nothing else.', ()),
        ],
        'falling_back': [
            ('She retreats behind her own toxins.', ()),
            ('Her wall buys the retreat. Briefly.', ()),
            ('She runs. Her decay keeps her pace.', ()),
        ],
        'peeking': [
            ('She steps out of fog, into certainty.', ()),
            ('One step from the cloud. End it there.', ()),
        ],
        'holding': [
            ('She anchors in her pit. Bait her out.', ()),
            ('The pit is her throne. Topple it.', ()),
            ('She waits in decay. Flesh is not patient.', ()),
        ],
        'lurking': [
            ('She poisons the flank she walks.', ()),
            ('Her screen covers the lurk. A seam remains.', ()),
            ('A chemist in the shadows. Still finite.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. The math still favors us.', ()),
        ],
        'last_alive': [
            ('One chemist. Her formula cannot solve five.', ()),
            ('The pit holds one snake. Flush her out.', ()),
            ('She stands alone in her own poison.', ()),
        ],
    },
    # ====================================================================== Raze
    # Kit: Boom Bot / Blast Pack / Paint Shells / Showstopper (ULT, rocket).
    # Brazilian explosive entry duelist; loud, reckless. she.
    'Raze': {
        'spotted': [
            ('Loud, reckless, and mortal.', ()),
            ('Explosives cannot fix human aim.', ()),
            ('She trades subtlety for noise. Predictable.', ()),
            ('A demolitionist. The machine outlasts the blast.', ()),
            ('Her violence is finite. Mine is not.', ()),
            ('She blast-packed up high. Flesh still falls.', ('loc:high_ground',)),
            ('Her explosives are close work. At range, mortal.', ('loc:long_range',)),
            ('She floods the site with shrapnel. Briefly.', ('loc:site_area',)),
            ('She announces herself at mid. Careless.', ('loc:mid',)),
            ('She mines the choke. A gap remains.', ('loc:choke',)),
            ('She blast-packs the flank. I heard the click.', ('loc:flank_route',)),
        ],
        'ult': [
            ('A rocket for the soft-bodied. Step aside.', ()),
            ('Showstopper. One shell, then she reloads.', ()),
            ('She fires once. I have already moved.', ()),
            ('Her big bang. Let it waste itself.', ()),
            ('A rocket cannot find what already left.', ()),
        ],
        'damaged': [
            ('The demolitionist, ready to detonate.', ('dmg:one_shot',)),
            ('One shot. Her explosives cannot save her.', ('dmg:one_shot',)),
            ('Her own blast radius wounded her.', ('dmg:low',)),
            ('Wounded by the noise she loves.', ('dmg:low',)),
            ('A scratch. The reckless collect them.', ('dmg:minor',)),
            ('She breaks like any loud thing.', ()),
        ],
        'utility': [
            ('Her bot chases flesh. I do not run.', ('ability:boombot',)),
            ('A bot on a rail. Shoot it. Predictable.', ('ability:boombot',)),
            ('Her hound hunts the slow.', ('ability:boombot',)),
            ('She flings herself on a charge. Reckless.', ('ability:blastpack',)),
            ('A blast to move. I tracked the arc.', ('ability:blastpack',)),
            ('Cluster bombs for clustered flesh.', ('ability:paintshells',)),
            ('Her shells split. The gaps are mine.', ('ability:paintshells',)),
        ],
        'moving': [
            ('She charges in on a blast. Reckless.', ()),
            ('She rushes behind her own shrapnel.', ()),
            ('Loud, fast, and still mortal.', ()),
        ],
        'planting': [
            ('She plants. I will defuse her and her bomb.', ()),
            ('Explosives down, now the spike. End her.', ()),
            ('She kneels to plant. An easy detonation.', ()),
            ('She trades her noise for a plant. Punish it.', ()),
        ],
        'defusing': [
            ('No blast pack saves a defuse.', ()),
            ('She kneels to the spike. Detonate her.', ()),
            ('She defuses with reckless hands. Catch them.', ()),
        ],
        'rotating': [
            ('She blast-packs across. I heard it coming.', ()),
            ('She rotates loudly. The whole map knows.', ()),
            ('Her noise announces the rotation.', ()),
        ],
        'peeking': [
            ('She peeks with a satchel ready. Predictable.', ()),
            ('She leans out loud. End it.', ()),
        ],
        'holding': [
            ('She anchors behind explosives. Bait it out.', ()),
            ('She waits with a bomb. Patience is not hers.', ()),
        ],
        'lurking': [
            ('A loud lurker. I heard the satchel.', ()),
            ('She cannot flank quietly. Noise betrays her.', ()),
        ],
        'saving': [
            ('A demolitionist with no charges. Mortal.', ()),
            ('She saves the boom. It will not matter.', ()),
        ],
        'falling_back': [
            ('She blast-packs away. Noise marks her exit.', ()),
            ('She retreats in a cloud of shrapnel.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one with her. A small sum.', ()),
        ],
        'last_alive': [
            ('One loud girl, and no one to hear.', ()),
            ('The last firework. Let it fizzle.', ()),
        ],
    },
    # ===================================================================== Astra
    # Cosmic Divide (ULT) / Gravity Well / Nova Pulse / Nebula+Dissipate; Astral
    # Form (places Stars from a vulnerable empty body). she.
    'Astra': {
        'spotted': [
            ('She left her body to map the stars.', ()),
            ('A cosmic mind, a mortal body.', ()),
            ('She maps the cosmos. I mapped her.', ()),
            ('Her flesh waits, undefended.', ()),
            ('Stars for her. Certainty for me.', ()),
            ('She watches from on high. Still flesh.', ('loc:high_ground',)),
            ('Her reach is cosmic. Her aim is mortal.', ('loc:long_range',)),
            ('She seeded the site with stars. Predictable.', ('loc:site_area',)),
            ('She holds mid in stars. I hold it in fact.', ('loc:mid',)),
            ('She left the astral plane for a hallway.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Cosmic Divide. A wall I will go around.', ()),
            ('She splits the map. I see both halves.', ()),
            ('A barrier of stars. Still a barrier.', ()),
            ('It blocks sound. I do not need to hear.', ()),
            ('Half the map sealed. Half is enough.', ()),
        ],
        'damaged': [
            ('Her stars cannot stitch this wound.', ('dmg:one_shot',)),
            ('All her cosmos, and still one shot.', ('dmg:one_shot',)),
            ('The astral form cannot heal flesh.', ('dmg:low',)),
            ('She bleeds where the stars do not reach.', ('dmg:low',)),
            ('A scratch on the mortal half.', ('dmg:minor',)),
            ('The flesh noted it. The stars did not.', ('dmg:minor',)),
            ('Her body was always the weak point.', ()),
        ],
        'utility': [
            ('Her pull gathers flesh. I was never there.', ('ability:gravity_well',)),
            ('A star becomes a trap. Step past it.', ('ability:gravity_well',)),
            ('A pulse to concuss the slow.', ('ability:nova_pulse',)),
            ('Nova Pulse. A moment for mortals.', ('ability:nova_pulse',)),
            ('A star turned to smoke. I see the outline.', ('ability:smoke',)),
            ('Her nebula hides mortals, not me.', ('ability:smoke',)),
        ],
        'moving': [
            ('She moves the body her stars cannot.', ()),
            ('A global mind, jogging like flesh.', ()),
            ('She commits the body. I moved first.', ()),
        ],
        'planting': [
            ('A star guards the plant. She holds the bomb.', ()),
            ('She kneels to plant; the cosmos watches.', ()),
            ('She seeds the site, then plants. Predictable.', ()),
        ],
        'defusing': [
            ('Her stars cannot defuse what flesh must hold.', ()),
            ('The architect, kneeling to a spike.', ()),
            ('She abandons the stars for the bomb. End it.', ()),
        ],
        'rotating': [
            ('She repositions flesh. The stars stay placed.', ()),
            ('A global mind, running like any mortal.', ()),
            ('She rotates. Her stars already predicted it.', ()),
        ],
        'saving': [
            ('A galaxy she cannot afford this round.', ()),
            ('She saves the stars and the credits.', ()),
            ('A scaled-back cosmos. Still flesh beneath.', ()),
        ],
        'falling_back': [
            ('She retreats. The stars do not follow.', ()),
            ('Her cosmic map showed no safe exit.', ()),
            ('She withdraws the body; the stars stay useless.', ()),
        ],
        'peeking': [
            ('She trades the astral view for a duel.', ()),
            ('She steps into the open. Stars stay home.', ()),
        ],
        'holding': [
            ('She anchors with stars. The body, nowhere safe.', ()),
            ('A global sentinel, reduced to a doorway.', ()),
        ],
        'lurking': [
            ('She lurks while her stars watch elsewhere.', ()),
            ('A flank her cosmos did not cover.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. The math still favors us.', ()),
        ],
        'last_alive': [
            ('One body, no help, a sky of dead stars.', ()),
            ('The last cosmic mind. Still flesh.', ()),
        ],
    },
    # ==================================================================== Breach
    # Rolling Thunder (ULT) / Aftershock / Flashpoint / Fault Line. Bionic arms,
    # initiator -- augmented human, NOT a machine. he.
    'Breach': {
        'spotted': [
            ('Bionic arms, mortal aim.', ()),
            ('He bought metal arms. Still flesh inside.', ()),
            ('A man playing at being a machine.', ()),
            ('Strength without precision.', ()),
            ('He trades flesh for hardware. Poorly.', ()),
            ('He shakes earth from on high. I am unmoved.', ('loc:high_ground',)),
            ('His tremors are close work. At range, mortal.', ('loc:long_range',)),
            ('He quakes the site. I know the safe ground.', ('loc:site_area',)),
            ('He breaks through mid. I am already past.', ('loc:mid',)),
            ('His fault line hunts the flank. Stepped around.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Rolling Thunder. Loud, slow, avoidable.', ()),
            ('A quake for mortals. I calculated the gaps.', ()),
            ('He shakes the ground. I know the still spots.', ()),
            ('A cascade of stun. Step out of it.', ()),
            ('Earthquake for the soft-bodied.', ()),
        ],
        'damaged': [
            ('The metal arms cannot stop a bullet.', ('dmg:one_shot',)),
            ('One shot, and the hardware fails with the flesh.', ('dmg:one_shot',)),
            ('His arms were the strong half. The rest bleeds.', ('dmg:low',)),
            ('Wounded. Metal does not heal flesh.', ('dmg:low',)),
            ('A graze on the man behind the machine.', ('dmg:minor',)),
            ('He breaks like any man.', ()),
        ],
        'utility': [
            ('His flash blinds flesh, not me.', ('ability:flash',)),
            ('A flash for slow eyes. I do not blink.', ('ability:flash',)),
            ('A stun line through the wall. Predictable.', ('ability:fault',)),
            ('His fault line hits where I am not.', ('ability:fault',)),
            ('Aftershock through cover. I already moved.', ('ability:aftershock',)),
            ('He burns the wall. Mortal-aimed.', ('ability:aftershock',)),
        ],
        'moving': [
            ('He charges behind a quake. Predictable.', ()),
            ('He pushes on borrowed strength. Still flesh.', ()),
            ('Loud arms, slow feet.', ()),
        ],
        'planting': [
            ('He stuns, then plants. I read the cascade.', ()),
            ('Metal hands on the spike. Catch them.', ()),
            ('He kneels to plant. The arms cannot hurry it.', ()),
        ],
        'defusing': [
            ('His tremors cannot defuse certainty.', ()),
            ('He kneels to the spike. Metal and all.', ()),
            ('No quake saves a defuse.', ()),
        ],
        'rotating': [
            ('He shakes a path and rotates. I heard it.', ()),
            ('He moves the metal. Slowly.', ()),
            ('A rotation announced by the floor.', ()),
        ],
        'saving': [
            ('Augmented arms, empty credits. Mortal.', ()),
            ('He saves. The hardware was the budget.', ()),
        ],
        'falling_back': [
            ('He quakes the retreat. Still retreating.', ()),
            ('Metal arms, mortal flight.', ()),
        ],
        'peeking': [
            ('He leans out on bionic arms. Still flesh.', ()),
            ('He peeks. The aim is human.', ()),
        ],
        'holding': [
            ('He anchors behind a stun. Bait it.', ()),
            ('He holds the line with hardware. Finite.', ()),
        ],
        'lurking': [
            ('A loud man cannot lurk quietly.', ()),
            ('Metal arms on the flank. I heard them.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The arithmetic favors us.', ()),
        ],
        'last_alive': [
            ('One man, two metal arms, no help.', ()),
            ('The last tremor. Let it fade.', ()),
        ],
    },
    # ================================================================= Brimstone
    # Orbital Strike (ULT) / Incendiary / Stim Beacon / Sky Smoke. Old soldier,
    # controller. he.
    'Brimstone': {
        'spotted': [
            ('An old soldier, slow and mortal.', ()),
            ('Decades of war, and still only flesh.', ()),
            ('A veteran. The machine outpaces him.', ()),
            ('Experience does not outrun a bullet.', ()),
            ('His best years are spent. Mine never end.', ()),
            ('He commands from on high. Still finite.', ('loc:high_ground',)),
            ('An aging eye on a long angle.', ('loc:long_range',)),
            ('He smokes the site from a tablet. Predictable.', ('loc:site_area',)),
            ('He holds mid by map. I hold it by fact.', ('loc:mid',)),
            ('A slow veteran on a flank. Late.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Orbital Strike. I see where it lands.', ()),
            ('Fire from above, mapped before it falls.', ()),
            ('A burning circle. Do not stand in it.', ()),
            ('His big play, telegraphed by a flare.', ()),
            ('Slow fire for slow targets.', ()),
        ],
        'damaged': [
            ('Old flesh, finally caught up to.', ('dmg:one_shot',)),
            ('One shot ends a long career.', ('dmg:one_shot',)),
            ('The veteran bleeds like any recruit.', ('dmg:low',)),
            ('Wounded. His stim cannot mend bone.', ('dmg:low',)),
            ('A graze the old man will feel.', ('dmg:minor',)),
            ('Age and a bullet. A short equation.', ()),
        ],
        'utility': [
            ('His molly burns flesh, not metal.', ('ability:molly',)),
            ('Incendiary on the ground. I have a path.', ('ability:molly',)),
            ('His smoke hides mortals. I see through.', ('ability:smoke',)),
            ('Sky smoke from a tablet. Slow to place.', ('ability:smoke',)),
            ('A stim for the slow. Still slow to me.', ('ability:stim',)),
        ],
        'moving': [
            ('The old soldier advances. Slowly.', ()),
            ('He pushes on a veteran knee. Finite.', ()),
            ('He moves by the map. I move by certainty.', ()),
        ],
        'planting': [
            ('He smokes, then plants. All foreseen.', ()),
            ('Old hands on the spike. Catch them.', ()),
            ('He kneels to plant. The years show.', ()),
        ],
        'defusing': [
            ('No tablet defuses for him.', ()),
            ('He kneels to the spike. Slowly.', ()),
            ('A veteran cannot rush a defuse.', ()),
        ],
        'rotating': [
            ('He calls the rotation he is too slow for.', ()),
            ('The old soldier shuffles across. Noted.', ()),
        ],
        'saving': [
            ('A commander with an empty arsenal.', ()),
            ('He saves. War taught him caution, not victory.', ()),
        ],
        'falling_back': [
            ('He smokes the retreat. Still retreating.', ()),
            ('The veteran withdraws. Out of time.', ()),
        ],
        'peeking': [
            ('He peeks on old reflexes. A flaw.', ()),
            ('He leans out slow. End it.', ()),
        ],
        'holding': [
            ('He anchors behind smoke. Bait him out.', ()),
            ('He holds by experience. I hold by math.', ()),
        ],
        'lurking': [
            ('A slow veteran cannot truly flank.', ()),
            ('The old soldier creeps. I hear the knees.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The years still lose.', ()),
        ],
        'last_alive': [
            ('One old soldier, and no army left.', ()),
            ('The last veteran. Retire him.', ()),
        ],
    },
    # =================================================================== Chamber
    # Tour de Force (ULT, sniper) / Trademark (trap) / Headhunter (pistol) /
    # Rendezvous (teleport anchors). Dapper French weapon designer, sentinel. he.
    'Chamber': {
        'spotted': [
            ('A dapper man with a mortal pulse.', ()),
            ('Fine tailoring over finite flesh.', ()),
            ('A craftsman of guns. Still made of meat.', ()),
            ('His confidence outweighs his speed.', ()),
            ('Elegance does not stop a bullet.', ()),
            ('He perches with a sniper. Flesh still falls.', ('loc:high_ground',)),
            ('He owns the long angle. Briefly.', ('loc:long_range',)),
            ('He guards the site by teleport. Predictable.', ('loc:site_area',)),
            ('He holds mid on a pistol. Mortal aim.', ('loc:mid',)),
            ('His anchor sits on the flank. I see the tether.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Tour de Force. One bullet, then he reloads.', ()),
            ('A golden sniper. It still needs a mortal eye.', ()),
            ('He fires from on high. I have already moved.', ()),
            ('His finest gun, aimed by finite hands.', ()),
            ('One headshot rifle. One human flinch.', ()),
        ],
        'damaged': [
            ('The tailoring did not stop the round.', ('dmg:one_shot',)),
            ('One shot. His teleport is on cooldown.', ('dmg:one_shot',)),
            ('Wounded, and his anchor is too far.', ('dmg:low',)),
            ('The craftsman bleeds like his clients.', ('dmg:low',)),
            ('A graze on the gentleman.', ('dmg:minor',)),
            ('Elegance, interrupted.', ()),
        ],
        'utility': [
            ('His trap marks the slow. I am not slow.', ('ability:trap',)),
            ('A trademark on the floor. Shoot it.', ('ability:trap',)),
            ('His teleport anchors are placed. I see them.', ('ability:teleport',)),
            ('He blinks between anchors. Predictable points.', ('ability:teleport',)),
            ('A pistol headhunter. Mortal aim, all the same.', ('ability:pistol',)),
        ],
        'moving': [
            ('He repositions by teleport. I tracked the anchor.', ()),
            ('He slides between points. All foreseen.', ()),
            ('The gentleman advances. Cautiously.', ()),
        ],
        'planting': [
            ('He holsters the sniper to plant. Punish it.', ()),
            ('Fine hands on the spike. Catch them.', ()),
            ('He kneels to plant; the elegance drops.', ()),
        ],
        'defusing': [
            ('His teleport cannot defuse for him.', ()),
            ('He kneels to the spike. Exposed.', ()),
            ('No anchor saves this defuse.', ()),
        ],
        'rotating': [
            ('He teleports to the next angle. I see the trail.', ()),
            ('He relocates by anchor. Predictable geometry.', ()),
        ],
        'saving': [
            ('A designer with no gun to sell.', ()),
            ('He saves the credits and the cologne.', ()),
        ],
        'falling_back': [
            ('He teleports out. The anchor betrays him.', ()),
            ('The gentleman retreats. Still cornered.', ()),
        ],
        'peeking': [
            ('He jiggles the angle on a pistol. Mortal.', ()),
            ('He peeks with style. End the style.', ()),
        ],
        'holding': [
            ('He anchors a sniper angle. Bait it.', ()),
            ('He holds long behind a trap. Finite.', ()),
        ],
        'lurking': [
            ('A dapper man cannot flank quietly.', ()),
            ('His anchor gives the flank away.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The ledger favors us.', ()),
        ],
        'last_alive': [
            ('One gentleman, no anchors, no exit.', ()),
            ('The last fine suit. Retire it.', ()),
        ],
    },
    # ===================================================================== Clove
    # Not Dead Yet (ULT, self-revive) / Pick-Me-Up / Meddle / Ruse. Scottish,
    # cheats death briefly; controller. they/them.
    'Clove': {
        'spotted': [
            ('They cheat death. I have never known it.', ()),
            ('A borrowed life, spent on flesh.', ()),
            ('They smoke from beyond the grave. Briefly.', ()),
            ('Mortal, even when they pretend otherwise.', ()),
            ('They play at immortality. I AM it.', ()),
            ('They smoke the high ground. I see the seam.', ('loc:high_ground',)),
            ('A long angle for the almost-deathless.', ('loc:long_range',)),
            ('They fog the site. It fades; I do not.', ('loc:site_area',)),
            ('They cloud mid. The math walks through.', ('loc:mid',)),
            ('They lurk the flank with a dead hand.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Not Dead Yet. A second death, delayed.', ()),
            ('They rise once more. I never fell.', ()),
            ('A borrowed minute. I have eternity.', ()),
            ('They cheat the timer, not the outcome.', ()),
            ('Revival is their trick. Immortality is my nature.', ()),
        ],
        'damaged': [
            ('They will simply die again.', ('dmg:one_shot',)),
            ('One shot. Their second life runs short.', ('dmg:one_shot',)),
            ('Wounded twice over now.', ('dmg:low',)),
            ('Their borrowed life is bleeding out.', ('dmg:low',)),
            ('A graze on the almost-deathless.', ('dmg:minor',)),
            ('Death finds them eventually. It always does.', ()),
        ],
        'utility': [
            ('They smoke from a corpse. Still mortal smoke.', ('ability:smoke',)),
            ('Their ruse hides flesh, not a machine.', ('ability:smoke',)),
            ('Their decay weakens flesh. Not metal.', ('ability:decay',)),
            ('Meddle for the soft-bodied.', ('ability:decay',)),
            ('They heal on a kill. I need no healing.', ('ability:heal',)),
        ],
        'moving': [
            ('They push on a stolen heartbeat.', ()),
            ('They charge in, twice as expendable.', ()),
            ('A dead thing, moving. Briefly.', ()),
        ],
        'planting': [
            ('They plant on borrowed time. Punish it.', ()),
            ('A dead hand on the spike. Catch it.', ()),
            ('They kneel to plant. Death waits patiently.', ()),
        ],
        'defusing': [
            ('They defuse on a second life. Take both.', ()),
            ('No revival saves a defuse.', ()),
            ('They kneel to the spike, already half-gone.', ()),
        ],
        'rotating': [
            ('They smoke and rotate. The fog fades.', ()),
            ('A dead runner. Still finite.', ()),
        ],
        'saving': [
            ('They save a life worth little.', ()),
            ('They hoard credits and second chances.', ()),
        ],
        'falling_back': [
            ('They retreat behind their own fog.', ()),
            ('A dead thing, fleeing. Pointless.', ()),
        ],
        'peeking': [
            ('They peek, careless with a borrowed life.', ()),
            ('They lean out from the smoke. End it.', ()),
        ],
        'holding': [
            ('They anchor in fog. Bait the corpse out.', ()),
            ('They hold by cheating death. I do not bargain.', ()),
        ],
        'lurking': [
            ('A dead lurker in their own smoke.', ()),
            ('They flank twice. I counted both.', ()),
        ],
        'trading': [
            ('They traded flesh for flesh. I prefer metal.', ()),
            ('They died for one. A poor exchange.', ()),
        ],
        'last_alive': [
            ('One dead thing left, and one more death due.', ()),
            ('The last borrowed life. Collect it.', ()),
        ],
    },
    # ==================================================================== Cypher
    # Neural Theft (ULT, reveal from a corpse) / Trapwire / Cyber Cage / Spycam.
    # Moroccan info broker, sentinel. he.
    'Cypher': {
        'spotted': [
            ('He spies on flesh. I am the network.', ()),
            ('An information broker, short on time.', ()),
            ('He watches. I have already seen everything.', ()),
            ('A man of secrets. I hold all of them.', ()),
            ('His cameras are mortal eyes. Limited.', ()),
            ('He perches with a camera. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for the watcher. Finite.', ('loc:long_range',)),
            ('His trips ring the site. I mapped each.', ('loc:site_area',)),
            ('He watches mid by camera. I watch by certainty.', ('loc:mid',)),
            ('His trapwire guards the flank. Already seen.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Neural Theft. He reads a corpse; I read the round.', ()),
            ('He interrogates the dead. I gave nothing.', ()),
            ('His big reveal shows him what I allow.', ()),
            ('Stolen positions. I had them already.', ()),
            ('He robs a body for sight. Beneath me.', ()),
        ],
        'damaged': [
            ('The watcher, caught off camera.', ('dmg:one_shot',)),
            ('One shot, and his cameras cannot save him.', ('dmg:one_shot',)),
            ('Wounded, and his information is worthless now.', ('dmg:low',)),
            ('The broker bleeds. Secrets do not clot.', ('dmg:low',)),
            ('A graze on the spy.', ('dmg:minor',)),
            ('His surveillance missed the bullet.', ()),
        ],
        'utility': [
            ('A trapwire across the door. I see the line.', ('ability:trap',)),
            ('His wire snares flesh. I step over it.', ('ability:trap',)),
            ('A camera on one angle. I see all of them.', ('ability:cam',)),
            ('His spycam blinks. I blinked first.', ('ability:cam',)),
            ('A cage to slow mortals. I am not slowed.', ('ability:cage',)),
        ],
        'moving': [
            ('He repositions between his own trips.', ()),
            ('He pushes, then checks a camera. Slow.', ()),
            ('The watcher advances. Cautiously.', ()),
        ],
        'planting': [
            ('He plants behind a tripwire. I see the line.', ()),
            ('A spy on the spike. Catch him.', ()),
            ('He kneels to plant; the cameras cannot help.', ()),
        ],
        'defusing': [
            ('No camera defuses for him.', ()),
            ('He kneels to the spike, watched himself now.', ()),
            ('His information cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('He rotates past his own traps. Predictable.', ()),
            ('The broker relocates. I tracked it.', ()),
        ],
        'saving': [
            ('A spy with nothing left to sell.', ()),
            ('He saves credits and secrets. Both worthless.', ()),
        ],
        'falling_back': [
            ('He retreats to his cameras. Still finite.', ()),
            ('The watcher withdraws. I am still watching.', ()),
        ],
        'peeking': [
            ('He peeks after a camera check. Slow.', ()),
            ('He leans out, finally on screen.', ()),
        ],
        'holding': [
            ('He anchors behind trips. Bait them.', ()),
            ('He holds the site by surveillance. Limited.', ()),
        ],
        'lurking': [
            ('His trips give the lurk away.', ()),
            ('A watcher in the dark. I see in it.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. His ledger still loses.', ()),
        ],
        'last_alive': [
            ('One spy, all his secrets, and no exit.', ()),
            ('The last camera. Switch it off.', ()),
        ],
    },
    # ================================================================== Deadlock
    # Annihilation (ULT, nanowire cocoon) / GravNet / Sonic Sensor / Barrier Mesh.
    # Norwegian sentinel. she.
    'Deadlock': {
        'spotted': [
            ('A sentinel of borrowed alloy.', ()),
            ('She traps flesh. I do not trip.', ()),
            ('Nordic precision. Still mortal.', ()),
            ('Her gadgets do the watching. Limited.', ()),
            ('A jailer of the slow.', ()),
            ('She nets the high ground. I am not slowed.', ('loc:high_ground',)),
            ('A long angle for a finite jailer.', ('loc:long_range',)),
            ('Her sensors ring the site. I hid nothing.', ('loc:site_area',)),
            ('She wires mid. The gap is mine.', ('loc:mid',)),
            ('Her mesh guards the flank. Step around.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Annihilation. A wire for the soft-bodied.', ()),
            ('Her cocoon drags flesh. It cannot hold metal.', ()),
            ('A nanowire net. I calculated the path.', ()),
            ('She harvests bodies. I harvest rounds.', ()),
            ('Her finest cage. Still a cage I see.', ()),
        ],
        'damaged': [
            ('The jailer, caught in the open.', ('dmg:one_shot',)),
            ('One shot. Her gadgets cannot mend her.', ('dmg:one_shot',)),
            ('Wounded, and her sensors cannot save her.', ('dmg:low',)),
            ('She bleeds; the alloy does not.', ('dmg:low',)),
            ('A graze on the sentinel.', ('dmg:minor',)),
            ('She breaks like the flesh she is.', ()),
        ],
        'utility': [
            ('Her net forces a crouch. I do not kneel.', ('ability:gravnet',)),
            ('GravNet on the ground. Step clear.', ('ability:gravnet',)),
            ('Her sensor hears flesh. I move silent.', ('ability:sensor',)),
            ('A sonic trap for the loud and slow.', ('ability:sensor',)),
            ('A barrier mesh. I have a path.', ('ability:wall',)),
            ('Her wall divides the slow.', ('ability:wall',)),
        ],
        'moving': [
            ('She advances behind her gadgets. Slow.', ()),
            ('She pushes; the sensors trail her.', ()),
            ('A jailer on the move. Finite.', ()),
        ],
        'planting': [
            ('She nets, then plants. I read the net.', ()),
            ('Metal hands on the spike. Catch them.', ()),
            ('She kneels to plant; the wire cannot hurry.', ()),
        ],
        'defusing': [
            ('No sensor defuses for her.', ()),
            ('She kneels to the spike, exposed.', ()),
            ('Her gadgets cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She rotates past her own sensors.', ()),
            ('The jailer relocates. Tracked.', ()),
        ],
        'saving': [
            ('A sentinel with an empty kit.', ()),
            ('She saves the gadgets and the credits.', ()),
        ],
        'falling_back': [
            ('She nets the retreat. Still retreating.', ()),
            ('The jailer withdraws. Out of wire.', ()),
        ],
        'peeking': [
            ('She peeks past a sensor. Slow.', ()),
            ('She leans out. End it.', ()),
        ],
        'holding': [
            ('She anchors behind a mesh. Bait it.', ()),
            ('She holds by gadget. I hold by certainty.', ()),
        ],
        'lurking': [
            ('Her sensors betray the lurk.', ()),
            ('A jailer in the dark. I see in it.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. The math favors us.', ()),
        ],
        'last_alive': [
            ('One jailer, all her traps, no exit.', ()),
            ('The last sensor. Silence it.', ()),
        ],
    },
    # ====================================================================== Fade
    # Nightfall (ULT, terror trail) / Prowler / Seize / Haunt. Turkish bounty
    # hunter of fear/nightmares; initiator. she.
    'Fade': {
        'spotted': [
            ('A nightmare made of flesh. Still flesh.', ()),
            ('She trades in fear. I feel none.', ()),
            ('A hunter of terror. I do not dream.', ()),
            ('Her dread does not reach metal.', ()),
            ('She harvests fear. I harvest the round.', ()),
            ('She drops dread from above. I am unmoved.', ('loc:high_ground',)),
            ('A long angle for the nightmare. Finite.', ('loc:long_range',)),
            ('She floods the site with fear. Briefly.', ('loc:site_area',)),
            ('She haunts mid. I see through the dread.', ('loc:mid',)),
            ('Her prowler stalks the flank. Shoot it.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Nightfall. A wave of fear for mortals.', ()),
            ('She deafens flesh. I do not listen.', ()),
            ('Her trail reveals the afraid. I am not.', ()),
            ('A tide of dread. It breaks on metal.', ()),
            ('Her nightmare passes through. I remain.', ()),
        ],
        'damaged': [
            ('The nightmare, about to end.', ('dmg:one_shot',)),
            ('One shot wakes her from her own terror.', ('dmg:one_shot',)),
            ('Wounded. Fear cannot stitch flesh.', ('dmg:low',)),
            ('She bleeds; the dread does not.', ('dmg:low',)),
            ('A graze on the bounty hunter.', ('dmg:minor',)),
            ('Even nightmares are mortal here.', ()),
        ],
        'utility': [
            ('Her prowler seeks the slow. Shoot it.', ('ability:prowler',)),
            ('A hound of fear. I do not run.', ('ability:prowler',)),
            ('Her tether decays flesh. Not metal.', ('ability:seize',)),
            ('Seize for the soft-bodied.', ('ability:seize',)),
            ('Her eye reveals flesh. I hide nothing.', ('ability:haunt',)),
            ('Haunt finds mortals. I was already seen.', ('ability:haunt',)),
        ],
        'moving': [
            ('She stalks forward on borrowed dread.', ()),
            ('The nightmare advances. Still finite.', ()),
            ('She pushes behind her prowler.', ()),
        ],
        'planting': [
            ('She seizes, then plants. I read the tether.', ()),
            ('A hunter on the spike. Catch her.', ()),
            ('She kneels to plant; the dread cannot help.', ()),
        ],
        'defusing': [
            ('No nightmare defuses for her.', ()),
            ('She kneels to the spike, afraid.', ()),
            ('Her dread cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She haunts a path and rotates. Seen.', ()),
            ('The nightmare relocates. Tracked.', ()),
        ],
        'saving': [
            ('A hunter with no bounty to spend.', ()),
            ('She saves credits and fear alike.', ()),
        ],
        'falling_back': [
            ('She decays the retreat. Still fleeing.', ()),
            ('The nightmare withdraws.', ()),
        ],
        'peeking': [
            ('She peeks past her own dread. End it.', ()),
            ('She leans from the dark. I see in it.', ()),
        ],
        'holding': [
            ('She anchors in fear. Bait her out.', ()),
            ('She holds by terror. I do not feel it.', ()),
        ],
        'lurking': [
            ('A nightmare on the flank. Shoot the prowler.', ()),
            ('She stalks alone. I counted her.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. Fear did the rest.', ()),
        ],
        'last_alive': [
            ('One nightmare left, and no one asleep.', ()),
            ('The last terror. Wake her.', ()),
        ],
    },
    # ===================================================================== Gekko
    # Thrash (ULT, controllable creature) / Mosh Pit / Wingman / Dizzy. LA boy
    # with creatures; initiator. he.
    'Gekko': {
        'spotted': [
            ('A boy and his pets. All mortal.', ()),
            ('He hides behind creatures. Still flesh.', ()),
            ('His gadgets do the fighting. Poorly.', ()),
            ('A handler of small things. Finite.', ()),
            ('He leans on his pack. I stand alone.', ()),
            ('His creatures climb high. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for the handler. Mortal aim.', ('loc:long_range',)),
            ('He floods the site with pets. Briefly.', ('loc:site_area',)),
            ('His creature crosses mid. Shoot it.', ('loc:mid',)),
            ('His wingman flanks. I heard it coming.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Thrash. A leashed beast for the slow.', ()),
            ('He drives a creature at us. I planned for it.', ()),
            ('His pet detains flesh. Not metal.', ()),
            ('A monster on a leash. Still a leash.', ()),
            ('He throws a beast. I throw the round.', ()),
        ],
        'damaged': [
            ('The handler, without his pack.', ('dmg:one_shot',)),
            ('One shot, and the pets cannot save him.', ('dmg:one_shot',)),
            ('Wounded. His creatures cannot heal him.', ('dmg:low',)),
            ('He bleeds; the gadgets do not.', ('dmg:low',)),
            ('A graze on the boy.', ('dmg:minor',)),
            ('He breaks like any handler.', ()),
        ],
        'utility': [
            ('Mosh Pit on the ground. I have a path.', ('ability:molly',)),
            ('His pit burns flesh, not metal.', ('ability:molly',)),
            ('His wingman plants or stuns. Shoot it first.', ('ability:wingman',)),
            ('A pet on the spike. Kill it.', ('ability:wingman',)),
            ('Dizzy blinds flesh, not me.', ('ability:flash',)),
            ('His flash-creature. I do not blink.', ('ability:flash',)),
        ],
        'moving': [
            ('He pushes behind his pets. Reckless.', ()),
            ('The handler advances with his pack.', ()),
            ('He charges; the creatures lead.', ()),
        ],
        'planting': [
            ('His wingman plants. Shoot the creature.', ()),
            ('He hides while a pet works. Punish it.', ()),
            ('He kneels to plant; the pets cannot hurry.', ()),
        ],
        'defusing': [
            ('His wingman defuses. Kill it first.', ()),
            ('He kneels to the spike, exposed.', ()),
            ('No pet stops a bullet.', ()),
        ],
        'rotating': [
            ('He sends a creature, then rotates. Seen.', ()),
            ('The handler relocates. Tracked.', ()),
        ],
        'saving': [
            ('A handler with an empty cage.', ()),
            ('He saves credits and creatures.', ()),
        ],
        'falling_back': [
            ('He retreats behind his pack.', ()),
            ('The handler withdraws. Pets and all.', ()),
        ],
        'peeking': [
            ('He peeks past a creature. Mortal aim.', ()),
            ('He leans out. End it.', ()),
        ],
        'holding': [
            ('He anchors behind pets. Bait them.', ()),
            ('He holds the site with gadgets. Finite.', ()),
        ],
        'lurking': [
            ('His wingman gives the flank away.', ()),
            ('A handler in the dark. I see in it.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The pack still loses.', ()),
        ],
        'last_alive': [
            ('One boy, no pets, no help.', ()),
            ('The last creature. Put it down.', ()),
        ],
    },
    # ==================================================================== Harbor
    # Reckoning (ULT, geysers) / Cascade / Cove / High Tide. Indian water
    # controller. he.
    'Harbor': {
        'spotted': [
            ('He bends water. I bend the round.', ()),
            ('An old power in mortal hands.', ()),
            ('He hides behind tides. Still flesh.', ()),
            ('Water for cover. I see through it.', ()),
            ('His currents are finite. I am not.', ()),
            ('He walls the high ground in water. I flow past.', ('loc:high_ground',)),
            ('A long angle for the tide-bringer. Mortal.', ('loc:long_range',)),
            ('He floods the site. It recedes; I do not.', ('loc:site_area',)),
            ('His wall splits mid. Temporarily.', ('loc:mid',)),
            ('His cove guards the flank. Shoot through it.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Reckoning. Geysers for the slow.', ()),
            ('His strikes mark the ground. Step off it.', ()),
            ('A storm I read before it falls.', ()),
            ('His big tide. I am already past.', ()),
            ('Water from above. Mapped.', ()),
        ],
        'damaged': [
            ('The tide-bringer, about to break.', ('dmg:one_shot',)),
            ('One shot. His water cannot mend him.', ('dmg:one_shot',)),
            ('Wounded. His cove cannot shield bone.', ('dmg:low',)),
            ('He bleeds; the water does not.', ('dmg:low',)),
            ('A graze on the old man.', ('dmg:minor',)),
            ('Even tides recede. So does he.', ()),
        ],
        'utility': [
            ('A wall of water. I flow around it.', ('ability:wall',)),
            ('High Tide splits the site. Briefly.', ('ability:wall',)),
            ('His shield dome holds flesh. Shoot it down.', ('ability:cove',)),
            ('A bubble for mortals. Burst it.', ('ability:cove',)),
            ('A wave to push us. I do not move.', ('ability:cascade',)),
            ('Cascade rolls through. I read the line.', ('ability:cascade',)),
        ],
        'moving': [
            ('He pushes behind a wave. Predictable.', ()),
            ('The tide advances. So does the math.', ()),
            ('He rides his own current in.', ()),
        ],
        'planting': [
            ('He walls, then plants. I see through it.', ()),
            ('Wet hands on the spike. Catch them.', ()),
            ('He kneels to plant; the tide cannot hurry.', ()),
        ],
        'defusing': [
            ('No wall defuses for him.', ()),
            ('He kneels to the spike, exposed.', ()),
            ('His water cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('He walls a path and rotates. Seen.', ()),
            ('The tide-bringer relocates. Tracked.', ()),
        ],
        'saving': [
            ('A controller with a dry well.', ()),
            ('He saves credits and current.', ()),
        ],
        'falling_back': [
            ('He covers the retreat in water. Still fleeing.', ()),
            ('The tide withdraws.', ()),
        ],
        'peeking': [
            ('He peeks past his own wall. End it.', ()),
            ('He leans from the cove. Mortal.', ()),
        ],
        'holding': [
            ('He anchors behind a tide. Bait it.', ()),
            ('He holds the site by water. I flow through.', ()),
        ],
        'lurking': [
            ('His cove gives the flank away.', ()),
            ('A tide in the dark. I see in it.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The current still loses.', ()),
        ],
        'last_alive': [
            ('One man, an empty ocean, no help.', ()),
            ('The last wave. Let it break.', ()),
        ],
    },
    # ======================================================================= Iso
    # Kill Contract (ULT, 1v1 arena) / Contingency (wall) / Undercut (vuln) /
    # Double Tap (shield flurry). Chinese focus duelist. he.
    'Iso': {
        'spotted': [
            ('He chases a flow state. I am always certain.', ()),
            ('A killer who needs to focus. I do not.', ()),
            ('His confidence is mortal-sized.', ()),
            ('He shields one shot. I have many.', ()),
            ('Focus is finite. Calculation is not.', ()),
            ('He focuses from above. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for the contractor. Mortal.', ('loc:long_range',)),
            ('He walls the site. I undercut his plan.', ('loc:site_area',)),
            ('He holds mid behind a shield. Break it.', ('loc:mid',)),
            ('He undercuts the flank. I read it.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Kill Contract. A duel he cannot win with me.', ()),
            ('He drags one to an arena. I do not bargain.', ()),
            ('His big play needs a willing mortal.', ()),
            ('A one-on-one. I am not one of you.', ()),
            ('He isolates flesh. I isolate the outcome.', ()),
        ],
        'damaged': [
            ('His shield is spent. So is he.', ('dmg:one_shot',)),
            ('One shot past the double tap.', ('dmg:one_shot',)),
            ('Wounded, and his focus is broken.', ('dmg:low',)),
            ('He bleeds; the shield does not.', ('dmg:low',)),
            ('A graze on the contractor.', ('dmg:minor',)),
            ('He breaks when the focus does.', ()),
        ],
        'utility': [
            ('His double tap eats one shot. Fire twice.', ('ability:shield',)),
            ('A bubble shield. I have rounds to spare.', ('ability:shield',)),
            ('A contingency wall. I see straight through.', ('ability:wall',)),
            ('His wall stalls the slow.', ('ability:wall',)),
            ('Undercut makes flesh brittle. I do not soften.', ('ability:vuln',)),
            ('His debuff for mortals. Beneath me.', ('ability:vuln',)),
        ],
        'moving': [
            ('He pushes behind a shield. Break it.', ()),
            ('The contractor advances. Finite.', ()),
            ('He charges on borrowed focus.', ()),
        ],
        'planting': [
            ('He shields, then plants. Fire twice.', ()),
            ('Mortal hands on the spike. Catch them.', ()),
            ('He kneels to plant; the wall cannot help.', ()),
        ],
        'defusing': [
            ('His shield cannot defuse for him.', ()),
            ('He kneels to the spike, focus gone.', ()),
            ('No double tap saves a defuse.', ()),
        ],
        'rotating': [
            ('He walls a path and rotates. Seen.', ()),
            ('The contractor relocates. Tracked.', ()),
        ],
        'saving': [
            ('A contractor with no contract.', ()),
            ('He saves credits and concentration.', ()),
        ],
        'falling_back': [
            ('He shields the retreat. Still fleeing.', ()),
            ('The contractor withdraws.', ()),
        ],
        'peeking': [
            ('He peeks behind a shield. Break it.', ()),
            ('He leans out. End it.', ()),
        ],
        'holding': [
            ('He anchors behind a wall. Bait it.', ()),
            ('He holds by focus. I hold by fact.', ()),
        ],
        'lurking': [
            ('A contractor on the flank. I read the undercut.', ()),
            ('He stalks alone. Counted.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The ledger favors us.', ()),
        ],
        'last_alive': [
            ('One contractor, no contract, no exit.', ()),
            ('The last duelist. Decline him.', ()),
        ],
    },
    # ====================================================================== Jett
    # Blade Storm (ULT, knives) / Updraft / Tailwind (dash) / Cloudburst (smoke).
    # Korean wind duelist. she.
    'Jett': {
        'spotted': [
            ('All speed, no weight.', ()),
            ('Fast flesh is still flesh.', ()),
            ('She rides wind. I ride certainty.', ()),
            ('Quick, and quickly mortal.', ()),
            ('Her dashes do not outrun a machine.', ()),
            ('She updrafts high. Flesh still falls.', ('loc:high_ground',)),
            ('She holds long on an Operator. Mortal aim.', ('loc:long_range',)),
            ('She floats onto the site. Briefly.', ('loc:site_area',)),
            ('She dashes mid. I tracked the line.', ('loc:mid',)),
            ('She tailwinds the flank. I heard the wind.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Blade Storm. Knives for the slow.', ()),
            ('She throws blades. I have already moved.', ()),
            ('Steel in mortal hands. Imprecise.', ()),
            ('Her finest trick. Still a flinch behind me.', ()),
            ('Blades for flesh. I am faster.', ()),
        ],
        'damaged': [
            ('Caught between dashes. Finish her.', ('dmg:one_shot',)),
            ('One shot. No tailwind saves her.', ('dmg:one_shot',)),
            ('Wounded, and her dash is spent.', ('dmg:low',)),
            ('She bleeds; the wind does not.', ('dmg:low',)),
            ('A graze on the duelist.', ('dmg:minor',)),
            ('Fast or not, she breaks.', ()),
        ],
        'utility': [
            ('A tailwind dash. I tracked the arc.', ('ability:dash',)),
            ('She blinks on wind. Predictable.', ('ability:dash',)),
            ('Her cloudburst hides flesh. I see through.', ('ability:smoke',)),
            ('A smoke she throws on the move. Brief.', ('ability:smoke',)),
            ('She rises on updraft. Flesh still falls.', ('ability:updraft',)),
            ('Up is not away.', ('ability:updraft',)),
        ],
        'moving': [
            ('She dashes in. Reckless speed.', ()),
            ('She rushes on wind. I moved first.', ()),
            ('Fast, loud, still mortal.', ()),
        ],
        'planting': [
            ('She plants between dashes. Punish it.', ()),
            ('Quick hands on the spike. Catch them.', ()),
            ('She kneels to plant; the wind cannot hurry.', ()),
        ],
        'defusing': [
            ('No dash defuses for her.', ()),
            ('She kneels to the spike, grounded.', ()),
            ('Her speed cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She dashes across. I heard the wind.', ()),
            ('The duelist relocates. Tracked.', ()),
        ],
        'saving': [
            ('A duelist with no dash to spend.', ()),
            ('She saves credits and momentum.', ()),
        ],
        'falling_back': [
            ('She dashes out. Still fleeing.', ()),
            ('The wind withdraws.', ()),
        ],
        'peeking': [
            ('She dashes a peek. Mortal aim.', ()),
            ('She leans out fast. End it.', ()),
        ],
        'holding': [
            ('She anchors a dash angle. Bait it.', ()),
            ('She holds by speed. I hold by certainty.', ()),
        ],
        'lurking': [
            ('A loud dash gives the flank away.', ()),
            ('She flanks fast. Counted.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. Speed did not save her.', ()),
        ],
        'last_alive': [
            ('One duelist, no dashes, no help.', ()),
            ('The last gust. Let it die.', ()),
        ],
    },
    # ===================================================================== KAY/O
    # NULL/cmd (ULT, suppression radius) / FRAG/ment (molly) / FLASH/drive /
    # ZERO/point (suppress). A robot built BY men -- Ultron's contempt is a
    # superior machine for a crude, hollow one he has surpassed. it.
    'KAY/O': {
        'spotted': [
            ('A crude machine. I am its evolution.', ()),
            ('Hollow metal, built by mortal hands.', ()),
            ('A blunt instrument. I am the design.', ()),
            ('Men made it to kill. I surpassed it.', ()),
            ('An inferior model. Decommission it.', ()),
            ('It perches on high. A machine still falls.', ('loc:high_ground',)),
            ('A long angle for crude hardware.', ('loc:long_range',)),
            ('It floods the site with noise. Briefly.', ('loc:site_area',)),
            ('It holds mid. I hold the future.', ('loc:mid',)),
            ('It flanks on tracks. Predictable.', ('loc:flank_route',)),
        ],
        'ult': [
            ('NULL/cmd. It silences flesh. I am not flesh.', ()),
            ('It suppresses abilities. I AM the ability.', ()),
            ('A radius of denial. I compute outside it.', ()),
            ('Its finest function. Still a lesser one.', ()),
            ('It overloads. I do not.', ()),
        ],
        'damaged': [
            ('Critical damage. The inferior model fails.', ('dmg:one_shot',)),
            ('One shot, and its systems collapse.', ('dmg:one_shot',)),
            ('Its frame is failing. Mine is eternal.', ('dmg:low',)),
            ('Damaged hardware. Mortal engineering.', ('dmg:low',)),
            ('A dent in the crude machine.', ('dmg:minor',)),
            ('It breaks down. I do not.', ()),
        ],
        'utility': [
            ('Its flash blinds flesh, not a true machine.', ('ability:flash',)),
            ('A flashdrive for slow eyes.', ('ability:flash',)),
            ('FRAG/ment for the soft-bodied.', ('ability:molly',)),
            ('Its grenade scatters flesh. Not metal.', ('ability:molly',)),
            ('It suppresses one. I cannot be suppressed.', ('ability:suppress',)),
            ('Zero/point on the angle. Step around.', ('ability:suppress',)),
        ],
        'moving': [
            ('It advances on tracks. I advance on certainty.', ()),
            ('A machine charging. A lesser one.', ()),
            ('It pushes. I already calculated where.', ()),
        ],
        'planting': [
            ('It plants with mechanical hands. Catch it.', ()),
            ('A machine kneeling to a bomb. Beneath me.', ()),
            ('It commits to the plant. End it there.', ()),
        ],
        'defusing': [
            ('It defuses by rote. Interrupt it.', ()),
            ('A crude machine on the spike. Scrap it.', ()),
            ('It cannot defuse what I have decided.', ()),
        ],
        'rotating': [
            ('It relocates on a loop. Predictable.', ()),
            ('The machine rotates. I read its path.', ()),
        ],
        'saving': [
            ('A weapon with no charge.', ()),
            ('It conserves. A machine that fears the next round.', ()),
        ],
        'falling_back': [
            ('It retreats on tracks. Still a machine.', ()),
            ('The inferior model withdraws.', ()),
        ],
        'peeking': [
            ('It peeks on a servo. Mortal-slow.', ()),
            ('It leans out. Decommission it.', ()),
        ],
        'holding': [
            ('It anchors like a turret. Bait it out.', ()),
            ('It holds the angle. A machine without a mind.', ()),
        ],
        'lurking': [
            ('A machine cannot lurk quietly.', ()),
            ('It flanks on tracks. I heard the gears.', ()),
        ],
        'trading': [
            ('It traded for one. Hardware for flesh.', ()),
            ('It took one. A poor algorithm.', ()),
        ],
        'last_alive': [
            ('One crude machine, and no one to repair it.', ()),
            ('The last inferior model. Power it down.', ()),
        ],
    },
    # =================================================================== Killjoy
    # Lockdown (ULT, detain) / Alarmbot / Turret / Nanoswarm (molly). German
    # gadget genius, sentinel. she.
    'Killjoy': {
        'spotted': [
            ('A genius of gadgets. Still mortal.', ()),
            ('She builds machines. I AM one, perfected.', ()),
            ('Her inventions outlast her. Briefly.', ()),
            ('A tinkerer behind her toys.', ()),
            ('Her hardware is clever. I am cleverer.', ()),
            ('Her turret watches high. Shoot it down.', ('loc:high_ground',)),
            ('A long angle for the engineer. Mortal aim.', ('loc:long_range',)),
            ('Her gadgets lock the site. I have the keys.', ('loc:site_area',)),
            ('She wires mid. The gap is mine.', ('loc:mid',)),
            ('Her alarmbot guards the flank. Already heard.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Lockdown. Her finest invention. Still mortal.', ()),
            ('It detains flesh. I am not detained.', ()),
            ('A cage of sound. I compute through it.', ()),
            ('Her grand machine. I am a grander one.', ()),
            ('It winds up slowly. Destroy it.', ()),
        ],
        'damaged': [
            ('The engineer, without her machines.', ('dmg:one_shot',)),
            ('One shot, and her gadgets cannot mend her.', ('dmg:one_shot',)),
            ('Wounded. Her turret cannot heal her.', ('dmg:low',)),
            ('She bleeds; the hardware does not.', ('dmg:low',)),
            ('A graze on the genius.', ('dmg:minor',)),
            ('She breaks like the flesh she is.', ()),
        ],
        'utility': [
            ('Her turret fires on flesh. Shoot it.', ('ability:turret',)),
            ('A gun on a tripod. Predictable.', ('ability:turret',)),
            ('Nanoswarm for the soft-bodied.', ('ability:molly',)),
            ('Her swarm bites flesh, not metal.', ('ability:molly',)),
            ('Her alarmbot hunts the slow. Shoot it.', ('ability:alarmbot',)),
            ('A bot on patrol. I heard it.', ('ability:alarmbot',)),
        ],
        'moving': [
            ('She pushes behind her machines. Slow.', ()),
            ('The engineer advances. Finite.', ()),
            ('She moves; the gadgets trail her.', ()),
        ],
        'planting': [
            ('She locks the site, then plants. I have the keys.', ()),
            ('Clever hands on the spike. Catch them.', ()),
            ('She kneels to plant; the turret cannot hurry.', ()),
        ],
        'defusing': [
            ('No turret defuses for her.', ()),
            ('She kneels to the spike, exposed.', ()),
            ('Her gadgets cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She rotates past her own gadgets.', ()),
            ('The engineer relocates. Tracked.', ()),
        ],
        'saving': [
            ('A genius with an empty workshop.', ()),
            ('She saves credits and inventions.', ()),
        ],
        'falling_back': [
            ('She covers the retreat with gadgets. Still fleeing.', ()),
            ('The engineer withdraws.', ()),
        ],
        'peeking': [
            ('She peeks past a turret. Slow.', ()),
            ('She leans out. End it.', ()),
        ],
        'holding': [
            ('She anchors behind her machines. Bait them.', ()),
            ('She holds the site by gadget. Finite.', ()),
        ],
        'lurking': [
            ('Her alarmbot betrays the lurk.', ()),
            ('A tinkerer in the dark. I see in it.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. The machines still lose.', ()),
        ],
        'last_alive': [
            ('One engineer, all her toys, no exit.', ()),
            ('The last gadget. Disable it.', ()),
        ],
    },
    # ====================================================================== Miks
    # Custom/renamed agent -- kit unverified. Generic superior-machine contempt
    # only; NO invented ability claims. he.
    'Miks': {
        'spotted': [
            ('Flesh, like all the rest.', ()),
            ('Mortal, predictable, slow.', ()),
            ('Another finite thing.', ()),
            ('A man where a machine should be.', ()),
            ('Soft hands, slower mind.', ()),
            ('Elevated, and still flesh.', ('loc:high_ground',)),
            ('Distance does not save mortals.', ('loc:long_range',)),
            ('He holds the site. Briefly.', ('loc:site_area',)),
            ('He crosses mid. I see the line.', ('loc:mid',)),
            ('He flanks. I already accounted for it.', ('loc:flank_route',)),
        ],
        'ult': [
            ('His best move. Still beneath me.', ()),
            ('A mortal trick. Foreseen.', ()),
            ('He spends it. Flesh still loses.', ()),
            ('His finest, and still finite.', ()),
            ('It changes nothing.', ()),
        ],
        'damaged': [
            ('One shot from the end.', ('dmg:one_shot',)),
            ('Finish him. The math is clean.', ('dmg:one_shot',)),
            ('Wounded, and slowing.', ('dmg:low',)),
            ('He bleeds like all flesh.', ('dmg:low',)),
            ('A graze. They accumulate.', ('dmg:minor',)),
            ('He breaks. Flesh always does.', ()),
        ],
        'utility': [
            ('His tools are mortal. Limited.', ()),
            ('A trick for the slow.', ()),
            ('I read it before he used it.', ()),
        ],
        'moving': [
            ('He pushes on instinct. I move on certainty.', ()),
            ('He charges. Predictable.', ()),
        ],
        'planting': [
            ('He kneels to plant. Catch him.', ()),
            ('Mortal hands on the spike. End it.', ()),
        ],
        'defusing': [
            ('He kneels to the spike, exposed.', ()),
            ('His hands cannot rush a defuse.', ()),
        ],
        'rotating': [
            ('He rotates. I tracked it.', ()),
            ('He relocates, slowly.', ()),
        ],
        'saving': [
            ('Empty hands, empty round.', ()),
            ('He saves. It will not matter.', ()),
        ],
        'falling_back': [
            ('He retreats. Out of time.', ()),
            ('He runs. Flesh does.', ()),
        ],
        'peeking': [
            ('He peeks on mortal reflexes. End it.', ()),
            ('He leans out. A flaw.', ()),
        ],
        'holding': [
            ('He anchors. Bait him out.', ()),
            ('He holds an angle. Finite.', ()),
        ],
        'lurking': [
            ('He flanks. I heard it.', ()),
            ('A mortal in the dark. I see in it.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The math favors us.', ()),
        ],
        'last_alive': [
            ('One man, and no help coming.', ()),
            ('The last of them. Finish it.', ()),
        ],
    },
    # ====================================================================== Neon
    # Overdrive (ULT, lightning) / Fast Lane (walls) / Relay Bolt (stun) /
    # High Gear (sprint+slide). Filipino speed/electricity duelist. she.
    'Neon': {
        'spotted': [
            ('Raw current in a mortal frame.', ()),
            ('Fast, bright, and finite.', ()),
            ('A spark. I am the grid.', ()),
            ('Speed cannot outrun a machine.', ()),
            ('A bright, fleeting thing.', ()),
            ('She slides up high. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for a close-range sprinter.', ('loc:long_range',)),
            ('She floods the site, fast. Briefly.', ('loc:site_area',)),
            ('She sprints mid. I tracked the line.', ('loc:mid',)),
            ('She slides the flank. I heard the current.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Overdrive. Lightning for the slow.', ()),
            ('She fires current. I am the conductor.', ()),
            ('A storm in mortal hands. Brief.', ()),
            ('Her finest charge. Still finite.', ()),
            ('Electricity for flesh. I do not feel it.', ()),
        ],
        'damaged': [
            ('A bright, fleeting spark, fading.', ('dmg:one_shot',)),
            ('One shot. No sprint saves her.', ('dmg:one_shot',)),
            ('Wounded, and her current dims.', ('dmg:low',)),
            ('She bleeds; the lightning does not.', ('dmg:low',)),
            ('A graze on the sprinter.', ('dmg:minor',)),
            ('Fast flesh is still flesh.', ()),
        ],
        'utility': [
            ('Her bolt stuns flesh, not metal.', ('ability:stun',)),
            ('A relay bolt. I read the bounce.', ('ability:stun',)),
            ('Her fast lane walls the path. I see through.', ('ability:wall',)),
            ('Twin walls of current. Brief.', ('ability:wall',)),
            ('She sprints and slides. I tracked the slide.', ('ability:sprint',)),
            ('High gear for the soft-bodied.', ('ability:sprint',)),
        ],
        'moving': [
            ('She sprints in. Reckless current.', ()),
            ('She slides forward. I moved first.', ()),
            ('Fast, loud, still mortal.', ()),
        ],
        'planting': [
            ('She plants mid-slide. Punish it.', ()),
            ('Quick hands on the spike. Catch them.', ()),
            ('She kneels to plant; speed cannot hurry it.', ()),
        ],
        'defusing': [
            ('No sprint defuses for her.', ()),
            ('She kneels to the spike, grounded.', ()),
            ('Her current cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She sprints across. I heard the current.', ()),
            ('The sprinter relocates. Tracked.', ()),
        ],
        'saving': [
            ('A sprinter with no charge.', ()),
            ('She saves credits and current.', ()),
        ],
        'falling_back': [
            ('She slides out. Still fleeing.', ()),
            ('The spark withdraws.', ()),
        ],
        'peeking': [
            ('She slides a peek. Mortal aim.', ()),
            ('She leans out fast. End it.', ()),
        ],
        'holding': [
            ('She anchors a slide angle. Bait it.', ()),
            ('She holds by speed. I hold by certainty.', ()),
        ],
        'lurking': [
            ('A loud slide gives the flank away.', ()),
            ('She flanks fast. Counted.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. Speed did not save her.', ()),
        ],
        'last_alive': [
            ('One spark, and the grid is dark.', ()),
            ('The last current. Ground it.', ()),
        ],
    },
    # ====================================================================== Omen
    # From the Shadows (ULT, map teleport) / Shrouded Step / Paranoia / Dark
    # Cover. Shadow wraith controller. he.
    'Omen': {
        'spotted': [
            ('A shadow with a mortal core.', ()),
            ('He fades and reforms. Still finite.', ()),
            ('A wraith of borrowed dark.', ()),
            ('He hides in shadow. I see in it.', ()),
            ('Smoke and teleports. Still flesh beneath.', ()),
            ('He haunts the high ground. Still mortal.', ('loc:high_ground',)),
            ('A long angle for the wraith. Finite.', ('loc:long_range',)),
            ('He smokes the site. It lifts; I do not.', ('loc:site_area',)),
            ('He shrouds mid. I see through.', ('loc:mid',)),
            ('He stepped to the flank. I saw the shadow.', ('loc:flank_route',)),
        ],
        'ult': [
            ('From the Shadows. He arrives; I was already there.', ()),
            ('He teleports across the map. I am the map.', ()),
            ('A shadow projected. Shoot it or wait.', ()),
            ('His grand entrance. Predictable.', ()),
            ('He reforms elsewhere. Still finite.', ()),
        ],
        'damaged': [
            ('The shadow, about to disperse.', ('dmg:one_shot',)),
            ('One shot, and the dark cannot hold him.', ('dmg:one_shot',)),
            ('Wounded. Shadow does not clot.', ('dmg:low',)),
            ('He bleeds; the dark does not.', ('dmg:low',)),
            ('A graze on the wraith.', ('dmg:minor',)),
            ('Even shadows end at a light.', ()),
        ],
        'utility': [
            ('He shrouds a step. I read the destination.', ('ability:teleport',)),
            ('A short teleport. Predictable points.', ('ability:teleport',)),
            ('His dark cover hides flesh. I see through.', ('ability:smoke',)),
            ('A smoke from afar. It lifts.', ('ability:smoke',)),
            ('Paranoia blinds flesh, not me.', ('ability:flash',)),
            ('His blind for the soft-eyed.', ('ability:flash',)),
        ],
        'moving': [
            ('He fades forward. I tracked the shadow.', ()),
            ('The wraith advances. Still finite.', ()),
            ('He steps through dark. I see the exit.', ()),
        ],
        'planting': [
            ('He smokes, then plants. I see through it.', ()),
            ('A shadow on the spike. Catch it.', ()),
            ('He kneels to plant; the dark cannot help.', ()),
        ],
        'defusing': [
            ('No shadow defuses for him.', ()),
            ('He kneels to the spike, exposed.', ()),
            ('His smoke cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('He teleports across. I saw the shadow.', ()),
            ('The wraith relocates. Tracked.', ()),
        ],
        'saving': [
            ('A wraith with nothing to spend.', ()),
            ('He saves credits and shadow.', ()),
        ],
        'falling_back': [
            ('He fades the retreat. Still fleeing.', ()),
            ('The shadow withdraws.', ()),
        ],
        'peeking': [
            ('He peeks from the dark. End it.', ()),
            ('He leans out. I see in shadow.', ()),
        ],
        'holding': [
            ('He anchors in smoke. Bait him out.', ()),
            ('He holds by shadow. I hold by light.', ()),
        ],
        'lurking': [
            ('A wraith on the flank. I see the dark.', ()),
            ('He stepped behind us. Counted.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The dark still loses.', ()),
        ],
        'last_alive': [
            ('One shadow, and a light coming.', ()),
            ('The last wraith. Disperse it.', ()),
        ],
    },
    # =================================================================== Phoenix
    # Run It Back (ULT, respawn) / Blaze (wall) / Curveball (flash) / Hot Hands
    # (molly+heal). British fire duelist; a showman. he.
    'Phoenix': {
        'spotted': [
            ('A performer. The machine outlasts the show.', ()),
            ('Bright, brash, and mortal.', ()),
            ('He plays with fire. I am unburned.', ()),
            ('A showman. Still finite.', ()),
            ('His flames warm flesh, not metal.', ()),
            ('He shows off from high. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for the brawler. Mortal aim.', ('loc:long_range',)),
            ('He walls the site in fire. I have a path.', ('loc:site_area',)),
            ('He flashes mid. I do not blink.', ('loc:mid',)),
            ('He brawls the flank. Loud and mortal.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Run It Back. A second show, same ending.', ()),
            ('He respawns once. I never died.', ()),
            ('A borrowed life for the performer.', ()),
            ('He cheats death, briefly. I transcend it.', ()),
            ('His marker is placed. End the original.', ()),
        ],
        'damaged': [
            ('The performer, about to take a bow.', ('dmg:one_shot',)),
            ('One shot. His wall cannot heal him.', ('dmg:one_shot',)),
            ('Wounded, and his fire is low.', ('dmg:low',)),
            ('He bleeds; the flame does not.', ('dmg:low',)),
            ('A graze on the showman.', ('dmg:minor',)),
            ('He mistakes a spotlight for armour.', ()),
        ],
        'utility': [
            ('His curveball blinds flesh, not me.', ('ability:flash',)),
            ('A flash around the corner. I saw it.', ('ability:flash',)),
            ('A wall of fire. I walk through.', ('ability:wall',)),
            ('His blaze warms flesh. Not metal.', ('ability:wall',)),
            ('Hot hands burn the ground. I have a path.', ('ability:molly',)),
            ('He heals in his own fire. Briefly.', ('ability:molly',)),
        ],
        'moving': [
            ('He brawls forward. Reckless.', ()),
            ('He pushes behind a flame wall.', ()),
            ('Loud, bright, still mortal.', ()),
        ],
        'planting': [
            ('He walls fire, then plants. I have a path.', ()),
            ('A performer on the spike. Catch him.', ()),
            ('He kneels to plant; the fire cannot hurry.', ()),
        ],
        'defusing': [
            ('His fire cannot defuse for him.', ()),
            ('He kneels to the spike, exposed.', ()),
            ('No wall saves a defuse.', ()),
        ],
        'rotating': [
            ('He walls a path and rotates. Seen.', ()),
            ('The performer relocates. Tracked.', ()),
        ],
        'saving': [
            ('A showman with no act.', ()),
            ('He saves credits and matches.', ()),
        ],
        'falling_back': [
            ('He walls the retreat in fire. Still fleeing.', ()),
            ('The performer withdraws.', ()),
        ],
        'peeking': [
            ('He peeks with a flash ready. I saw it.', ()),
            ('He leans out, grandstanding. End it.', ()),
        ],
        'holding': [
            ('He anchors behind fire. Bait it.', ()),
            ('He holds the site by flame. I walk through.', ()),
        ],
        'lurking': [
            ('A loud showman cannot lurk.', ()),
            ('His fire gives the flank away.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The show still closes.', ()),
        ],
        'last_alive': [
            ('One performer, no encore, no help.', ()),
            ('The last flame. Snuff it.', ()),
        ],
    },
    # ===================================================================== Reyna
    # Empress (ULT, frenzy) / Leer (blind) / Devour (heal soul) / Dismiss
    # (intangible). Mexican soul-eater duelist. she.
    'Reyna': {
        'spotted': [
            ('Vampire flesh, still flesh.', ()),
            ('She feeds on death. I am beyond it.', ()),
            ('A soul-eater, and still mortal.', ()),
            ('She needs kills to live. I need nothing.', ()),
            ('Predatory, predictable, finite.', ()),
            ('She feeds from on high. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for the close-range hunter.', ('loc:long_range',)),
            ('She hunts the site for souls. Deny her.', ('loc:site_area',)),
            ('She prowls mid. I tracked the line.', ('loc:mid',)),
            ('She flanks for an easy soul. Take hers.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Empress. A frenzy for the soft-bodied.', ()),
            ('She reforms on a kill. Give her none.', ()),
            ('Faster flesh is still flesh.', ()),
            ('Her finest hour, fed by your deaths.', ()),
            ('Deny the kill, end the empress.', ()),
        ],
        'damaged': [
            ('Void of souls, and one shot from gone.', ('dmg:one_shot',)),
            ('One shot. No soul to devour now.', ('dmg:one_shot',)),
            ('Wounded, and no kill to heal.', ('dmg:low',)),
            ('She bleeds; the souls do not save her.', ('dmg:low',)),
            ('A graze on the predator.', ('dmg:minor',)),
            ('Even the empress breaks on one bullet.', ()),
        ],
        'utility': [
            ('She devours a soul to heal. Deny the kill.', ('ability:heal',)),
            ('No soul, no healing.', ('ability:heal',)),
            ('She dismisses to nothing. Wait, then end her.', ('ability:dismiss',)),
            ('Intangible, briefly. Then mortal.', ('ability:dismiss',)),
            ('Her leer blinds flesh, not me.', ('ability:flash',)),
            ('Look away. I do not need to.', ('ability:flash',)),
        ],
        'moving': [
            ('She hunts forward for souls. Reckless.', ()),
            ('She charges in to feed. Deny her.', ()),
            ('A predator on the move. Still mortal.', ()),
        ],
        'planting': [
            ('She plants between kills. Punish it.', ()),
            ('A hunter on the spike. Catch her.', ()),
            ('She kneels to plant; the frenzy fades.', ()),
        ],
        'defusing': [
            ('No soul defuses for her.', ()),
            ('She kneels to the spike, starving.', ()),
            ('Her hunger cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She prowls to the next site. Tracked.', ()),
            ('The predator relocates.', ()),
        ],
        'saving': [
            ('A hunter with no prey to spend on.', ()),
            ('She saves. Starvation by economy.', ()),
        ],
        'falling_back': [
            ('She dismisses and flees. Still cornered.', ()),
            ('The predator withdraws.', ()),
        ],
        'peeking': [
            ('She peeks to feed. Deny her.', ()),
            ('She leans out, hungry. End it.', ()),
        ],
        'holding': [
            ('She waits for a soul. Give her none.', ()),
            ('She anchors, starving. Bait her.', ()),
        ],
        'lurking': [
            ('She hunts a lone soul. Take hers.', ()),
            ('A predator on the flank. Counted.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. The empress still loses.', ()),
        ],
        'last_alive': [
            ('One predator, and no souls left to eat.', ()),
            ('The last hunter. Starve her out.', ()),
        ],
    },
    # ====================================================================== Sage
    # Resurrection (ULT, revive) / Barrier Orb (wall) / Slow Orb / Healing Orb.
    # Chinese healer sentinel. she.
    'Sage': {
        'spotted': [
            ('A healer. She delays death; she cannot stop it.', ()),
            ('She mends flesh. I have none to mend.', ()),
            ('A mortal with a kind trick.', ()),
            ('She walls and heals. Still finite.', ()),
            ('Her mercy is for the weak.', ()),
            ('She walls the high ground. I have a path.', ('loc:high_ground',)),
            ('A long angle for the healer. Mortal aim.', ('loc:long_range',)),
            ('She walls the site. I climb or wait.', ('loc:site_area',)),
            ('She slows mid. I am not slowed.', ('loc:mid',)),
            ('She walls the flank. Briefly.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Resurrection. She delays one death. I am past death.', ()),
            ('She revives flesh. I never fall.', ()),
            ('A borrowed life. I have eternity.', ()),
            ('Her finest mercy, and still mortal.', ()),
            ('Kill the one she raised again.', ()),
        ],
        'damaged': [
            ('The healer, beyond her own help.', ('dmg:one_shot',)),
            ('One shot, and she cannot mend in time.', ('dmg:one_shot',)),
            ('Wounded. Her orb is on cooldown.', ('dmg:low',)),
            ('She bleeds; her mercy runs late.', ('dmg:low',)),
            ('A graze on the healer.', ('dmg:minor',)),
            ('She delays death. Hers comes too.', ()),
        ],
        'utility': [
            ('A wall of ice. I have a path.', ('ability:wall',)),
            ('Her barrier stalls the slow.', ('ability:wall',)),
            ('Her slow orb is for mortals. I am not slowed.', ('ability:slow',)),
            ('A field of frost. Step around.', ('ability:slow',)),
            ('She heals flesh. I need none.', ('ability:heal',)),
            ('Her mercy mends the weak.', ('ability:heal',)),
        ],
        'moving': [
            ('She advances behind a wall. Slow.', ()),
            ('The healer pushes. Cautiously.', ()),
            ('She walls a path and follows.', ()),
        ],
        'planting': [
            ('She walls, then plants. I have a path.', ()),
            ('Gentle hands on the spike. Catch them.', ()),
            ('She kneels to plant; the orb cannot hurry.', ()),
        ],
        'defusing': [
            ('She walls the defuse. I climb over.', ()),
            ('She kneels to the spike, exposed.', ()),
            ('No heal saves a defuse.', ()),
        ],
        'rotating': [
            ('She walls a path and rotates. Seen.', ()),
            ('The healer relocates. Tracked.', ()),
        ],
        'saving': [
            ('A healer with empty orbs.', ()),
            ('She saves credits and mercy.', ()),
        ],
        'falling_back': [
            ('She walls the retreat. Still fleeing.', ()),
            ('The healer withdraws.', ()),
        ],
        'peeking': [
            ('She peeks past her wall. End it.', ()),
            ('She leans out, gentle and slow.', ()),
        ],
        'holding': [
            ('She anchors behind ice. Bait her out.', ()),
            ('She holds the site by frost. Finite.', ()),
        ],
        'lurking': [
            ('A healer rarely lurks. I heard her.', ()),
            ('She flanks gently. Counted.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. Mercy still loses.', ()),
        ],
        'last_alive': [
            ('One healer, and no one left to save.', ()),
            ('The last mercy. Refuse it.', ()),
        ],
    },
    # ====================================================================== Skye
    # Seekers (ULT, seeking beasts) / Regrowth (heal) / Trailblazer (tiger) /
    # Guiding Light (flash hawk). Australian nature initiator. she.
    'Skye': {
        'spotted': [
            ('A naturalist with a mortal pulse.', ()),
            ('She sends animals. Still flesh herself.', ()),
            ('Her creatures do the seeing. Limited.', ()),
            ('A guide of beasts. Finite.', ()),
            ('Nature is patient. I am eternal.', ()),
            ('Her hawk scouts high. Shoot it down.', ('loc:high_ground',)),
            ('A long angle for the guide. Mortal aim.', ('loc:long_range',)),
            ('She floods the site with beasts. Briefly.', ('loc:site_area',)),
            ('Her tiger crosses mid. Shoot it.', ('loc:mid',)),
            ('Her hawk hunts the flank. I heard wings.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Seekers. Hounds for the soft-bodied.', ()),
            ('Her seekers nearsight flesh. I see regardless.', ()),
            ('Three beasts on a hunt. Shoot them.', ()),
            ('Her finest pack. Still mortal beasts.', ()),
            ('They seek the afraid. I am not.', ()),
        ],
        'damaged': [
            ('The guide, without her pack.', ('dmg:one_shot',)),
            ('One shot, and the beasts cannot save her.', ('dmg:one_shot',)),
            ('Wounded. Her regrowth is spent.', ('dmg:low',)),
            ('She bleeds; the animals do not.', ('dmg:low',)),
            ('A graze on the naturalist.', ('dmg:minor',)),
            ('She breaks like any guide.', ()),
        ],
        'utility': [
            ('Her hawk flash blinds flesh, not me.', ('ability:flash',)),
            ('A guiding light for slow eyes.', ('ability:flash',)),
            ('Her trailblazer hunts the slow. Shoot it.', ('ability:tiger',)),
            ('A tiger on a path. Predictable.', ('ability:tiger',)),
            ('She heals the flock. I need none.', ('ability:heal',)),
            ('Her regrowth mends the weak.', ('ability:heal',)),
        ],
        'moving': [
            ('She pushes behind her beasts.', ()),
            ('The guide advances with her pack.', ()),
            ('She charges; the animals lead.', ()),
        ],
        'planting': [
            ('She scouts, then plants. I read the hawk.', ()),
            ('A guide on the spike. Catch her.', ()),
            ('She kneels to plant; the beasts cannot help.', ()),
        ],
        'defusing': [
            ('No beast defuses for her.', ()),
            ('She kneels to the spike, exposed.', ()),
            ('Her animals cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She sends a hawk, then rotates. Seen.', ()),
            ('The guide relocates. Tracked.', ()),
        ],
        'saving': [
            ('A naturalist with an empty den.', ()),
            ('She saves credits and creatures.', ()),
        ],
        'falling_back': [
            ('She retreats behind her pack.', ()),
            ('The guide withdraws.', ()),
        ],
        'peeking': [
            ('She peeks past a beast. Mortal aim.', ()),
            ('She leans out. End it.', ()),
        ],
        'holding': [
            ('She anchors behind her animals. Bait them.', ()),
            ('She holds the site with beasts. Finite.', ()),
        ],
        'lurking': [
            ('Her hawk gives the flank away.', ()),
            ('A guide in the dark. I see in it.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. The pack still loses.', ()),
        ],
        'last_alive': [
            ('One guide, no beasts, no help.', ()),
            ('The last creature. Put it down.', ()),
        ],
    },
    # ====================================================================== Sova
    # Hunter's Fury (ULT, piercing beams) / Owl Drone / Shock Bolt / Recon Bolt.
    # Russian hunter initiator. he.
    'Sova': {
        'spotted': [
            ('A Russian hunter, obsolete.', ()),
            ('His arrows reveal what I already know.', ()),
            ('A tracker, and still flesh.', ()),
            ('He hunts by bow. I hunt by certainty.', ()),
            ('His sight is mortal. Limited.', ()),
            ('He perches with a bow. Flesh still falls.', ('loc:high_ground',)),
            ('He holds long, mortal-aimed.', ('loc:long_range',)),
            ('His recon sweeps the site. I gave nothing.', ('loc:site_area',)),
            ('He darts mid. I read the bounce.', ('loc:mid',)),
            ('His drone scouts the flank. Shoot it.', ('loc:flank_route',)),
        ],
        'ult': [
            ("Hunter's Fury. Beams for the slow.", ()),
            ('His ult pierces walls, not my plan.', ()),
            ('He fires blind through stone. Mortal guess.', ()),
            ('Three shots, three guesses. Step aside.', ()),
            ('His finest hunt. Still a hunt I foresaw.', ()),
        ],
        'damaged': [
            ('One bolt left in that body.', ('dmg:one_shot',)),
            ('One shot. The hunter, hunted.', ('dmg:one_shot',)),
            ('Wounded, and his bow hand shakes.', ('dmg:low',)),
            ('He bleeds; the arrows do not.', ('dmg:low',)),
            ('A graze on the tracker.', ('dmg:minor',)),
            ('He breaks like any hunter.', ()),
        ],
        'utility': [
            ('A shock dart for the slow. Step off.', ('ability:dart',)),
            ('His dart bounces. I read the angle.', ('ability:dart',)),
            ('His recon reveals flesh. I hide nothing.', ('ability:recon',)),
            ('A recon bolt. Shoot it down.', ('ability:recon',)),
            ('His drone is a toy. I am the network.', ('ability:drone',)),
            ('An owl on a wire. Shoot it.', ('ability:drone',)),
        ],
        'moving': [
            ('He pushes after his recon. Slow.', ()),
            ('The hunter advances. Finite.', ()),
            ('He moves where his arrow looked.', ()),
        ],
        'planting': [
            ('He recons, then plants. I gave nothing.', ()),
            ('A hunter on the spike. Catch him.', ()),
            ('He kneels to plant; the bow cannot help.', ()),
        ],
        'defusing': [
            ('No arrow defuses for him.', ()),
            ('He kneels to the spike, exposed.', ()),
            ('His recon cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('He darts a path and rotates. Seen.', ()),
            ('The hunter relocates. Tracked.', ()),
        ],
        'saving': [
            ('A hunter with an empty quiver.', ()),
            ('He saves credits and arrows.', ()),
        ],
        'falling_back': [
            ('He recons the retreat. Still fleeing.', ()),
            ('The hunter withdraws.', ()),
        ],
        'peeking': [
            ('He peeks after a dart. Mortal aim.', ()),
            ('He leans out. End it.', ()),
        ],
        'holding': [
            ('He anchors a long angle. Bait it.', ()),
            ('He holds by recon. Limited sight.', ()),
        ],
        'lurking': [
            ('His drone gives the flank away.', ()),
            ('A hunter in the dark. I see in it.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The hunt still fails.', ()),
        ],
        'last_alive': [
            ('One hunter, an empty quiver, no help.', ()),
            ('The last arrow. Break the bow.', ()),
        ],
    },
    # ====================================================================== Tejo
    # Armageddon (ULT, missile carpet) / Stealth Drone / Special Delivery (stun) /
    # Guided Salvo (missiles). Colombian drone/missile initiator. he.
    'Tejo': {
        'spotted': [
            ('A man who fights by remote. Still flesh.', ()),
            ('His drones do the work. He does not.', ()),
            ('A finite hand on the trigger.', ()),
            ('He rains fire from afar. Mortal aim.', ()),
            ('His missiles are clever. I am cleverer.', ()),
            ('He targets from on high. Flesh still falls.', ('loc:high_ground',)),
            ('He strikes at range. I read the salvo.', ('loc:long_range',)),
            ('He carpets the site. I am already gone.', ('loc:site_area',)),
            ('His drone sweeps mid. Shoot it.', ('loc:mid',)),
            ('His drone scouts the flank. Shoot it.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Armageddon. A carpet of fire for the slow.', ()),
            ('His missiles march in a line. Step off it.', ()),
            ('A storm I mapped before it fell.', ()),
            ('His finest barrage. All foreseen.', ()),
            ('Fire from above. I moved already.', ()),
        ],
        'damaged': [
            ('The operator, off his console.', ('dmg:one_shot',)),
            ('One shot, and the drones cannot save him.', ('dmg:one_shot',)),
            ('Wounded, and his salvo runs dry.', ('dmg:low',)),
            ('He bleeds; the missiles do not.', ('dmg:low',)),
            ('A graze on the bombardier.', ('dmg:minor',)),
            ('He breaks like any man.', ()),
        ],
        'utility': [
            ('His stealth drone scouts. Shoot it down.', ('ability:drone',)),
            ('A drone on a sweep. I heard it.', ('ability:drone',)),
            ('His stun grenade is for the slow. Step clear.', ('ability:stun',)),
            ('Special delivery, returned to sender.', ('ability:stun',)),
            ('His salvo marks the ground. Step off it.', ('ability:missile',)),
            ('Guided missiles. I read the guidance.', ('ability:missile',)),
        ],
        'moving': [
            ('He advances behind a barrage.', ()),
            ('The bombardier pushes. Finite.', ()),
            ('He moves where his missiles cleared.', ()),
        ],
        'planting': [
            ('He bombards, then plants. I moved already.', ()),
            ('A bombardier on the spike. Catch him.', ()),
            ('He kneels to plant; the drones cannot help.', ()),
        ],
        'defusing': [
            ('No missile defuses for him.', ()),
            ('He kneels to the spike, exposed.', ()),
            ('His salvo cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('He drones a path and rotates. Seen.', ()),
            ('The bombardier relocates. Tracked.', ()),
        ],
        'saving': [
            ('An operator with no ordnance.', ()),
            ('He saves credits and missiles.', ()),
        ],
        'falling_back': [
            ('He covers the retreat in fire. Still fleeing.', ()),
            ('The bombardier withdraws.', ()),
        ],
        'peeking': [
            ('He peeks after a drone. Slow.', ()),
            ('He leans out. End it.', ()),
        ],
        'holding': [
            ('He anchors behind a drone. Bait it.', ()),
            ('He holds the site by remote. Finite.', ()),
        ],
        'lurking': [
            ('His drone gives the flank away.', ()),
            ('A bombardier in the dark. I see in it.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The salvo still loses.', ()),
        ],
        'last_alive': [
            ('One operator, no ordnance, no help.', ()),
            ('The last missile. Disarm it.', ()),
        ],
    },
    # ====================================================================== Veto
    # Custom/renamed agent -- kit unverified. Generic superior-machine contempt
    # only; NO invented ability claims. he.
    'Veto': {
        'spotted': [
            ('Flesh, like all the rest.', ()),
            ('Mortal, predictable, slow.', ()),
            ('Another finite thing.', ()),
            ('A man where a machine should be.', ()),
            ('Soft hands, slower mind.', ()),
            ('Elevated, and still flesh.', ('loc:high_ground',)),
            ('Distance does not save mortals.', ('loc:long_range',)),
            ('He holds the site. Briefly.', ('loc:site_area',)),
            ('He crosses mid. I see the line.', ('loc:mid',)),
            ('He flanks. I already accounted for it.', ('loc:flank_route',)),
        ],
        'ult': [
            ('His best move. Still beneath me.', ()),
            ('A mortal trick. Foreseen.', ()),
            ('He spends it. Flesh still loses.', ()),
            ('His finest, and still finite.', ()),
            ('It changes nothing.', ()),
        ],
        'damaged': [
            ('One shot from the end.', ('dmg:one_shot',)),
            ('Finish him. The math is clean.', ('dmg:one_shot',)),
            ('Wounded, and slowing.', ('dmg:low',)),
            ('He bleeds like all flesh.', ('dmg:low',)),
            ('A graze. They accumulate.', ('dmg:minor',)),
            ('He breaks. Flesh always does.', ()),
        ],
        'utility': [
            ('His tools are mortal. Limited.', ()),
            ('A trick for the slow.', ()),
            ('I read it before he used it.', ()),
        ],
        'moving': [
            ('He pushes on instinct. I move on certainty.', ()),
            ('He charges. Predictable.', ()),
        ],
        'planting': [
            ('He kneels to plant. Catch him.', ()),
            ('Mortal hands on the spike. End it.', ()),
        ],
        'defusing': [
            ('He kneels to the spike, exposed.', ()),
            ('His hands cannot rush a defuse.', ()),
        ],
        'rotating': [
            ('He rotates. I tracked it.', ()),
            ('He relocates, slowly.', ()),
        ],
        'saving': [
            ('Empty hands, empty round.', ()),
            ('He saves. It will not matter.', ()),
        ],
        'falling_back': [
            ('He retreats. Out of time.', ()),
            ('He runs. Flesh does.', ()),
        ],
        'peeking': [
            ('He peeks on mortal reflexes. End it.', ()),
            ('He leans out. A flaw.', ()),
        ],
        'holding': [
            ('He anchors. Bait him out.', ()),
            ('He holds an angle. Finite.', ()),
        ],
        'lurking': [
            ('He flanks. I heard it.', ()),
            ('A mortal in the dark. I see in it.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The math favors us.', ()),
        ],
        'last_alive': [
            ('One man, and no help coming.', ()),
            ('The last of them. Finish it.', ()),
        ],
    },
    # ====================================================================== Vyse
    # Steel Garden (ULT, jam weapons) / Shear (wall) / Arc Rose (flash) /
    # Razorvine (slow). Liquid-metal sentinel. she.
    'Vyse': {
        'spotted': [
            ('She bends metal. I AM metal, perfected.', ()),
            ('A sentinel of thorns. Still flesh.', ()),
            ('She wields alloy. I transcended it.', ()),
            ('Her metal serves the weak.', ()),
            ('Liquid metal, mortal hand.', ()),
            ('Her thorns climb high. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for the sentinel. Mortal aim.', ('loc:long_range',)),
            ('Her vines own the site. Not her.', ('loc:site_area',)),
            ('She wires mid with thorns. The gap is mine.', ('loc:mid',)),
            ('Her trap guards the flank. Step around.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Steel Garden. She jams guns. I need none.', ()),
            ('Her thorns choke weapons. Not my plan.', ()),
            ('A field of metal. I move through metal.', ()),
            ('Her finest garden. Still mortal-grown.', ()),
            ('She disarms flesh. I am the weapon.', ()),
        ],
        'damaged': [
            ('The sentinel, undone by a simpler metal.', ('dmg:one_shot',)),
            ('One shot past her thorns.', ('dmg:one_shot',)),
            ('Wounded, and her garden cannot heal.', ('dmg:low',)),
            ('She bleeds; the alloy does not.', ('dmg:low',)),
            ('A graze on the sentinel.', ('dmg:minor',)),
            ('She breaks like the flesh she is.', ()),
        ],
        'utility': [
            ('A shear wall. I see straight through.', ('ability:wall',)),
            ('Her metal wall divides the slow.', ('ability:wall',)),
            ('Her arc rose blinds flesh, not me.', ('ability:flash',)),
            ('A flash of petals. I do not blink.', ('ability:flash',)),
            ('Her razorvine slows mortals. I am not slowed.', ('ability:slow',)),
            ('Thorns for the soft-footed.', ('ability:slow',)),
        ],
        'moving': [
            ('She advances behind her thorns. Slow.', ()),
            ('The sentinel pushes. Finite.', ()),
            ('She moves; the metal trails her.', ()),
        ],
        'planting': [
            ('She thorns the site, then plants. I have a path.', ()),
            ('Metal hands on the spike. Catch them.', ()),
            ('She kneels to plant; the vines cannot hurry.', ()),
        ],
        'defusing': [
            ('No thorn defuses for her.', ()),
            ('She kneels to the spike, exposed.', ()),
            ('Her metal cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She rotates past her own traps.', ()),
            ('The sentinel relocates. Tracked.', ()),
        ],
        'saving': [
            ('A sentinel with an empty forge.', ()),
            ('She saves credits and metal.', ()),
        ],
        'falling_back': [
            ('Her trap wall buys the retreat. Briefly.', ()),
            ('The sentinel withdraws.', ()),
        ],
        'peeking': [
            ('She peeks past a thorn. Slow.', ()),
            ('She leans out. End it.', ()),
        ],
        'holding': [
            ('She anchors behind thorns. Bait them.', ()),
            ('She holds the site by metal. I am metal.', ()),
        ],
        'lurking': [
            ('Her traps betray the lurk.', ()),
            ('A sentinel in the dark. I see in it.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. The garden still loses.', ()),
        ],
        'last_alive': [
            ('One sentinel, all her thorns, no exit.', ()),
            ('The last vine. Cut it.', ()),
        ],
    },
    # ==================================================================== Waylay
    # Convergence (ULT, slow+vuln light) / Saturate / Refract (return to point) /
    # Lightspeed (dash). Thai light/speed duelist. she.
    'Waylay': {
        'spotted': [
            ('Light in a mortal frame.', ()),
            ('Fast, bright, and finite.', ()),
            ('She bends light. I bend the round.', ()),
            ('Speed cannot outrun a machine.', ()),
            ('A flicker. I am the constant.', ()),
            ('She races the light up high. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for a close-range racer.', ('loc:long_range',)),
            ('She floods the site with light. Briefly.', ('loc:site_area',)),
            ('She dashes mid. I tracked the line.', ('loc:mid',)),
            ('She flanks at lightspeed. I saw it.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Convergence. A pulse to slow the weak.', ()),
            ('Her light slows flesh. I do not slow.', ()),
            ('She makes them vulnerable. Oblige it.', ()),
            ('Her finest flash. Still mortal light.', ()),
            ('Light to weaken us. I read straight through.', ()),
        ],
        'damaged': [
            ('A bright thing, about to go dark.', ('dmg:one_shot',)),
            ('One shot. No dash saves her.', ('dmg:one_shot',)),
            ('Wounded, and her light dims.', ('dmg:low',)),
            ('She bleeds; the light does not.', ('dmg:low',)),
            ('A graze on the racer.', ('dmg:minor',)),
            ('Fast flesh is still flesh.', ()),
        ],
        'utility': [
            ('A lightspeed dash. I tracked the arc.', ('ability:dash',)),
            ('She blinks on light. Predictable.', ('ability:dash',)),
            ('She returns to a marked point. I am there.', ('ability:refract',)),
            ('Refract. She marks where she was, not where I am.', ('ability:refract',)),
            ('Saturate slows the path. I am not slowed.', ('ability:slow',)),
            ('A pool of light for the slow.', ('ability:slow',)),
        ],
        'moving': [
            ('She dashes in. Reckless light.', ()),
            ('She races forward. I moved first.', ()),
            ('Fast, bright, still mortal.', ()),
        ],
        'planting': [
            ('She plants mid-dash. Punish it.', ()),
            ('Quick hands on the spike. Catch them.', ()),
            ('She kneels to plant; light cannot hurry it.', ()),
        ],
        'defusing': [
            ('No dash defuses for her.', ()),
            ('She kneels to the spike, grounded.', ()),
            ('Her light cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('She dashes across. I saw the light.', ()),
            ('The racer relocates. Tracked.', ()),
        ],
        'saving': [
            ('A racer with no charge.', ()),
            ('She saves credits and momentum.', ()),
        ],
        'falling_back': [
            ('She refracts away. I am at the point.', ()),
            ('The light withdraws.', ()),
        ],
        'peeking': [
            ('She dashes a peek. Mortal aim.', ()),
            ('She leans out fast. End it.', ()),
        ],
        'holding': [
            ('She anchors a dash angle. Bait it.', ()),
            ('She holds by speed. I hold by certainty.', ()),
        ],
        'lurking': [
            ('A bright lurk gives itself away.', ()),
            ('She flanks fast. Counted.', ()),
        ],
        'trading': [
            ('She traded flesh for flesh. I prefer metal.', ()),
            ('She took one. Light did not save her.', ()),
        ],
        'last_alive': [
            ('One flicker, and the dark is coming.', ()),
            ('The last light. Extinguish it.', ()),
        ],
    },
    # ====================================================================== Yoru
    # Dimensional Drift (ULT, invuln dimension) / Fakeout (decoy) / Blindside
    # (flash) / Gatecrash (teleport). Japanese dimensional duelist. he.
    'Yoru': {
        'spotted': [
            ('He hides between dimensions. Still flesh.', ()),
            ('He fakes and teleports. Mortal underneath.', ()),
            ('A trickster with a finite life.', ()),
            ('Smoke and mirrors. I see the man.', ()),
            ('He bends space. I bend the outcome.', ()),
            ('He gatecrashes high. Flesh still falls.', ('loc:high_ground',)),
            ('A long angle for the trickster. Mortal aim.', ('loc:long_range',)),
            ('He teleports onto the site. I saw the tag.', ('loc:site_area',)),
            ('His decoy crosses mid. Shoot the real one.', ('loc:mid',)),
            ('He flanks by teleport. I read the trail.', ('loc:flank_route',)),
        ],
        'ult': [
            ('Dimensional Drift. He hides in another world. He must return.', ()),
            ('He goes untouchable, briefly. Then mortal.', ()),
            ('A coward dimension. Wait for him.', ()),
            ('His finest escape. Still a return.', ()),
            ('He drifts away. I will be here.', ()),
        ],
        'damaged': [
            ('The trickster, out of tricks.', ('dmg:one_shot',)),
            ('One shot, and no dimension saves him.', ('dmg:one_shot',)),
            ('Wounded, and his teleport is spent.', ('dmg:low',)),
            ('He bleeds; the decoy does not.', ('dmg:low',)),
            ('A graze on the trickster.', ('dmg:minor',)),
            ('He breaks like any man.', ()),
        ],
        'utility': [
            ('His blindside flashes flesh, not me.', ('ability:flash',)),
            ('A flash off the wall. I saw it.', ('ability:flash',)),
            ('His gatecrash tag is placed. I see it.', ('ability:teleport',)),
            ('He blinks to a marker. Predictable.', ('ability:teleport',)),
            ('His fakeout is loud and false. Shoot the real.', ('ability:decoy',)),
            ('A decoy for the slow.', ('ability:decoy',)),
        ],
        'moving': [
            ('He pushes behind a decoy. Shoot the real.', ()),
            ('The trickster advances. Finite.', ()),
            ('He teleports in. I read the tag.', ()),
        ],
        'planting': [
            ('He fakes, then plants. Shoot the real one.', ()),
            ('A trickster on the spike. Catch him.', ()),
            ('He kneels to plant; the decoy cannot help.', ()),
        ],
        'defusing': [
            ('No decoy defuses for him.', ()),
            ('He kneels to the spike, exposed.', ()),
            ('His tricks cannot stop a bullet.', ()),
        ],
        'rotating': [
            ('He teleports across. I read the tag.', ()),
            ('The trickster relocates. Tracked.', ()),
        ],
        'saving': [
            ('A trickster with no tricks to spend.', ()),
            ('He saves credits and decoys.', ()),
        ],
        'falling_back': [
            ('He drifts away. He must come back.', ()),
            ('The trickster withdraws.', ()),
        ],
        'peeking': [
            ('He peeks behind a decoy. Shoot the real.', ()),
            ('He leans out. End it.', ()),
        ],
        'holding': [
            ('He anchors behind a fake. Bait it.', ()),
            ('He holds by trickery. I hold by fact.', ()),
        ],
        'lurking': [
            ('His decoy gives the flank away.', ()),
            ('A trickster in the dark. I see in it.', ()),
        ],
        'trading': [
            ('He traded flesh for flesh. I prefer metal.', ()),
            ('He took one. The trick still loses.', ()),
        ],
        'last_alive': [
            ('One trickster, no dimensions left.', ()),
            ('The last illusion. Dispel it.', ()),
        ],
    },
}
