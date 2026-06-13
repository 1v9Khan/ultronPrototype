"""Vocab pack: Valorant agent + ability callouts for the relay test corpus.

Domain: All ~26 Valorant agents (Jett, Reyna, Raze, Phoenix, Yoru, Neon, Iso,
Sova, Breach, Skye, KAY/O, Fade, Gekko, Killjoy, Cypher, Sage, Chamber,
Deadlock, Vyse, Brimstone, Viper, Omen, Astra, Harbor, Clove) plus Tejo,
Miks, Waylay, Veto where relevant.

Covers: util used / util down / util needed / ult status (up / ready / down /
one-off) / requesting an ability from a teammate / ability callouts mid-round /
persona/flavor fragments in Ultron's voice.

Mix of:
  - What the STREAMER says to relay (trigger phrases for match_relay_command)
  - What a TEAMMATE says to Ultron (context + directive relay forms)
  - Standalone FLAVOR / persona fragments (Ultron's voice; off-snap register)

All unique, plain ASCII, valid Python strings.
"""

ITEMS = [
    # -----------------------------------------------------------------------
    # DUELIST -- Jett
    # -----------------------------------------------------------------------
    "tell my team Jett just dashed in, she is on site",
    "tell my teammates their Jett has ult, she will blade storm soon",
    "let my team know our Jett used all her smokes already",
    "tell my team Jett has no dash left, push her now",
    "their Jett just ulted, watch your angles",
    "our Jett is one kill off her ult",
    "ask Jett to save her dash for the execute",
    "tell Jett to updraft and hold heaven angle",
    "Jett just died with ult, their ult economy is down",
    "our Jett used blade storm, she has no ult now",
    "tell my team Jett has full ult, she goes in first",
    "ask my Jett to dash in and frag them out before we follow",

    # -----------------------------------------------------------------------
    # DUELIST -- Reyna
    # -----------------------------------------------------------------------
    "tell my team Reyna has ult, she will empress soon",
    "our Reyna just popped empress, it is active right now",
    "tell my teammates Reyna is healing off that kill, she is back full HP",
    "tell them their Reyna dismissed, she is not dead yet",
    "Reyna just dismissed through the smoke, she is behind us",
    "ask Reyna to leer for us before we push through short",
    "tell my team Reyna needs kills to sustain, create space for her",
    "their Reyna has three souls stored, she can dismiss out of anything",
    "our Reyna has empress ready, wait for my signal to execute",
    "tell my team Reyna dismissed into back site, she is lurking",

    # -----------------------------------------------------------------------
    # DUELIST -- Raze
    # -----------------------------------------------------------------------
    "tell my team their Raze just launched showstopper",
    "our Raze has ult, hold until she fires it before we go",
    "tell them Raze used her satchels to get on top of box",
    "ask Raze to blast pack me over the wall",
    "Raze grenaded A main, the entrance is cleared",
    "their Raze satcheled up to heaven, she has high ground",
    "tell my team Raze has no ult this round, we can rush without worrying",
    "ask my Raze to save one satchel for the retake",
    "Raze has showstopper up, wait for the boom before entering",
    "their Raze just bounced a grenade into site, watch the splash damage",

    # -----------------------------------------------------------------------
    # DUELIST -- Phoenix
    # -----------------------------------------------------------------------
    "tell my team their Phoenix just ulted, do not engage him",
    "Phoenix is running it back right now, hold for five seconds",
    "ask Phoenix to flash around the corner before I peek",
    "their Phoenix hot handed the entrance, it is on fire do not walk through",
    "tell them Phoenix placed a wall blocking his own team, nice",
    "our Phoenix has ult, he will run it back if he dies going in",
    "Phoenix used his curveball, he has no flash left",
    "ask my Phoenix to wall off CT to stop their rotation",
    "their Phoenix ult ran out, he is on his last life right now",
    "tell my team Phoenix flashed heaven, peek it now",

    # -----------------------------------------------------------------------
    # DUELIST -- Yoru
    # -----------------------------------------------------------------------
    "tell my team their Yoru is faking B, do not rotate yet",
    "Yoru just gatecrashed to our side, he is behind us",
    "our Yoru has ult, he is invisible somewhere on the map",
    "their Yoru placed a decoy in main, ignore the footsteps",
    "ask Yoru to TP back site after the plant to hold the defuse",
    "tell my team Yoru is in dimensional drift right now, he cannot shoot",
    "their Yoru used all his flashes, push him now",
    "Yoru gatecrash is on cooldown, no escapes for thirty seconds",
    "ask my Yoru to fake an A execute while we go B",
    "tell them Yoru dropped a clone at CT, do not peek it",

    # -----------------------------------------------------------------------
    # DUELIST -- Neon
    # -----------------------------------------------------------------------
    "tell my team their Neon just ult activated, she is sprinting",
    "ask Neon to high gear through mid and relay the enemies",
    "our Neon has ult ready, she runs the site first",
    "Neon slid into the corner, she has the angle on whoever peeks",
    "tell my team Neon used her wall, B is split in half",
    "their Neon ult ran out early, she missed all her shots",
    "ask my Neon to fast lane across mid before the round runs out",
    "tell them Neon zapped someone, she has one charge left",
    "Neon relay bolted them for 60, she needs one more hit",
    "our Neon has no abilities left, she is going mechanical",

    # -----------------------------------------------------------------------
    # DUELIST -- Iso
    # -----------------------------------------------------------------------
    "tell my team their Iso ulted, he is in a duel right now",
    "ask Iso to shield me before I push this angle",
    "our Iso just absorbed a kill through his shield, full HP on site",
    "Iso undercut blocked their rotation through garage",
    "tell my teammates Iso has ult, he will pick someone off and steal their gun",
    "their Iso used contingency wall, he is committed to mid",
    "ask my Iso to double tap the guy holding corner",
    "Iso has no ult this round, their duelist is playing mechanical only",
    "tell them our Iso is one off ult, one kill and he goes",

    # -----------------------------------------------------------------------
    # INITIATOR -- Sova
    # -----------------------------------------------------------------------
    "tell my team Sova is sending in the drone right now",
    "our Sova darted A main, one enemy confirmed",
    "ask Sova to recon bolt heaven before we push",
    "tell them Sova owl droned B and found two, they are site",
    "Sova hit them for 84, one is close to dying",
    "tell my team Sova has ult, hunter fury is ready",
    "ask my Sova to shock dart the corner before I walk in",
    "their Sova has no recon bolt left, we are going in blind",
    "Sova recon lit up three on B, they are stacking hard",
    "tell my teammates Sova is one off ult, one kill and he fires",
    "their Sova just fired hunter fury into A long",
    "ask Sova to triple shock the spike so they can not defuse",

    # -----------------------------------------------------------------------
    # INITIATOR -- Breach
    # -----------------------------------------------------------------------
    "tell my team Breach is rolling thunder onto site right now",
    "ask Breach to fault line before we execute",
    "our Breach just aftershocked the corner, it is clear",
    "tell my teammates Breach flashed through the wall for us",
    "their Breach has ult, rolling thunder will stun the whole site",
    "ask my Breach to flashpoint the guy holding heaven",
    "Breach rolling thunder is coming, do not stand in the open",
    "their Breach used all his charges, no more stuns this round",
    "tell my team Breach is charging fault line, hold position for two seconds",
    "ask Breach to aftershock the spike after they plant so no one can defuse",

    # -----------------------------------------------------------------------
    # INITIATOR -- Skye
    # -----------------------------------------------------------------------
    "tell my team Skye just sent her bird to A, she found two",
    "ask Skye to heal me up before we retake",
    "our Skye has ult, she is sending seekers out",
    "Skye tiger just got destroyed at A long, one enemy is over there",
    "tell them Skye flashed mid with her bird, peek it",
    "their Skye has ult, seekers will track us through smokes",
    "ask my Skye to regrowth me, I am low HP",
    "Skye seekers found two on B, both are playing back site",
    "tell my team Skye has no heal charges left, no sustain this round",
    "ask Skye to send her guiding light through window first",

    # -----------------------------------------------------------------------
    # INITIATOR -- KAY/O
    # -----------------------------------------------------------------------
    "tell my team KAY/O is ulting, he is overloaded right now",
    "our KAY/O just suppressed B with his knife, no abilities for five seconds",
    "ask KAY/O to pop flash before I peek short",
    "their KAY/O has ult, he will lock down the site if he gets in",
    "KAY/O threw his knife and hit two, both are suppressed",
    "tell my teammates KAY/O is overloaded, go in while he is up",
    "ask my KAY/O to zero point knife into site before the execute",
    "KAY/O knife hit mid, their Killjoy can not use her util right now",
    "their KAY/O has no ult this round, no lockdown threat",
    "tell them KAY/O fragmented the corner, do not walk through the explosion",

    # -----------------------------------------------------------------------
    # INITIATOR -- Fade
    # -----------------------------------------------------------------------
    "tell my team Fade is revealing them right now with nightfall",
    "our Fade has ult, haunt the entire site",
    "ask Fade to haunt A before we push to get their positions",
    "their Fade just prowled through garage, she knows someone is there",
    "Fade seize caught two of them, they are tethered",
    "tell my teammates Fade revealed three on B site with her ult",
    "ask my Fade to send a creeper into the corner before I peek",
    "their Fade has no ult this round, push without the reveal fear",
    "Fade used nightfall, two of them are decayed and marked",
    "tell my team Fade prowlers are hunting, hold still they track movement",

    # -----------------------------------------------------------------------
    # INITIATOR -- Gekko
    # -----------------------------------------------------------------------
    "tell my team Gekko is sending Dizzy in to flash them",
    "our Gekko just Wingmanned the spike, it is planted",
    "ask Gekko to Thrash ult through the site before we execute",
    "their Gekko has Thrash ready, he will detain anyone he catches",
    "Gekko Mosh Pit grenade is down on A, do not walk in",
    "tell my teammates Gekko recollected Dizzy, he can flash again",
    "ask my Gekko to Wingman defuse if I die holding the spike",
    "Dizzy just got killed, Gekko needs to buy it back or find it",
    "tell my team Gekko has ult, do not get caught in the stasis",
    "their Gekko Mosh Pit is on cooldown, it is safe to enter now",

    # -----------------------------------------------------------------------
    # INITIATOR -- Tejo
    # -----------------------------------------------------------------------
    "tell my team their Tejo just fired guided salvo into site",
    "ask Tejo to stealth drone A before we commit to the execute",
    "our Tejo concussed two at mid with suppressor drone",
    "tell my teammates Tejo has ult, armageddon is ready to fire",
    "Tejo's suppressor drone is scanning mid right now",
    "ask my Tejo to guided salvo the default plant spot",

    # -----------------------------------------------------------------------
    # SENTINEL -- Killjoy
    # -----------------------------------------------------------------------
    "tell my team Killjoy ult is up, she will lock down the spike",
    "our Killjoy just ulted, no one can fight through that field",
    "ask Killjoy to place her turret on B so we get info",
    "their Killjoy nanoswarm is on the spike, do not defuse yet",
    "Killjoy alarmbot hit someone on A, one enemy confirmed at entrance",
    "tell my teammates Killjoy pulled her util, she is rotating fast",
    "ask my Killjoy to swarm the corner before anyone can pop out",
    "their Killjoy has no ult this round, we can stay on site freely",
    "Killjoy setup is live on B, any push through will get hit",
    "tell my team Killjoy is one off ult, one more kill and she goes",
    "their Killjoy ult is down, it will come back in forty seconds",
    "ask Killjoy to pull her bots and come with us for the mid fight",

    # -----------------------------------------------------------------------
    # SENTINEL -- Cypher
    # -----------------------------------------------------------------------
    "tell my team Cypher got a trip kill, one enemy confirmed B long",
    "our Cypher just cyber caged garage, they can not push through for free",
    "ask Cypher to place a camera on A ramps so we have vision",
    "their Cypher tripped our flank, one of us is tagged",
    "Cypher neural theft is ready, he can reveal the whole enemy team",
    "tell my teammates Cypher just used his ult, he knows where everyone is",
    "ask my Cypher to spycam the default plant and watch it",
    "their Cypher has no cages left, push through garage without smoke fear",
    "Cypher trip went off on B link, someone is rotating toward us",
    "tell my team Cypher wired up the back site exit, they can not leave quietly",

    # -----------------------------------------------------------------------
    # SENTINEL -- Sage
    # -----------------------------------------------------------------------
    "ask Sage for a heal, I am at 20 HP",
    "tell my team Sage just walled off B main, the push is stopped",
    "our Sage has ult, she can res someone after the fight",
    "ask my Sage to slow orb the entrance before we retake",
    "Sage just rezzed Reyna, we are back to full five",
    "tell my teammates Sage ult is up, hold one teammate back for the res",
    "their Sage walled A long, we can not push through right now",
    "ask Sage to heal the Jett before the execute, she is low",
    "Sage wall is on cooldown, they can push main freely now",
    "tell my team Sage has no heal charges left, buy a heavy shield",
    "their Sage is one kill off ult, do not let her get there",
    "ask my Sage to barrier orb the spike so we can plant safely",

    # -----------------------------------------------------------------------
    # SENTINEL -- Chamber
    # -----------------------------------------------------------------------
    "tell my team their Chamber is holding the angle with his TP behind him",
    "our Chamber just headhunted two from the Op position",
    "ask Chamber to place his rendezvous TP at back site before we go",
    "their Chamber has tour de force ult, Op is ready",
    "Chamber trademarks are up on A, anyone entering will be slowed",
    "tell my teammates Chamber swapped TP positions, he is now at B",
    "ask my Chamber to anchor flank and use headhunter to trade if they push",
    "their Chamber ult is down for two rounds, no OP threat from him",
    "Chamber trademark slowed the push, they can only walk in",
    "tell my team Chamber has no TP left, he is committed to this position",

    # -----------------------------------------------------------------------
    # SENTINEL -- Deadlock
    # -----------------------------------------------------------------------
    "tell my team their Deadlock has ult, she will annihilate anyone she catches",
    "ask Deadlock to put a net barrier across B main before the round starts",
    "our Deadlock just gravenet caught one of them, they can not move",
    "Deadlock sonic sensor went off at garage, they are using that push",
    "tell my teammates Deadlock has no ult this round, go for the execute",
    "ask my Deadlock to barrier orb the flank route so we do not get hit",
    "their Deadlock annihilation ult is active right now, run toward her",
    "Deadlock caught one in the gravenet at A long, he is cocooned",
    "tell my team Deadlock put two net barriers down, they are locked out of mid",

    # -----------------------------------------------------------------------
    # SENTINEL -- Vyse
    # -----------------------------------------------------------------------
    "tell my team their Vyse has ult, steel garden will trap the whole site",
    "ask Vyse to put an arc rose near the spike so they slow defusing",
    "our Vyse just shear bladed mid, they can not cross for twenty seconds",
    "their Vyse steel garden is active, every entrance is wired",
    "tell my teammates Vyse caught one in an arc rose thorn, he is revealed",
    "ask my Vyse to razorvine the choke point before the round starts",
    "Vyse ult is down, push the site before she reloads",
    "their Vyse placed a lurking razorvine behind CT, watch the flank",

    # -----------------------------------------------------------------------
    # CONTROLLER -- Brimstone
    # -----------------------------------------------------------------------
    "tell my team Brimstone is orb ult, he will incendiary the whole site",
    "ask Brimstone to smoke off CT before we push",
    "our Brimstone has orbital strike ready, stay off the spike",
    "tell my teammates Brimstone smoked A main, A short, and CT, execute now",
    "their Brimstone has no ult and low stim count, limited util",
    "ask my Brimstone to stim our team before the execute so we all shoot faster",
    "Brimstone orbital strike is active on B, do not walk in",
    "tell my team Brimstone smokes are up, call the timings",
    "their Brim has no stims left this half, no combat stimmed fights",
    "ask Brimstone to incendiary the site corner before we plant",

    # -----------------------------------------------------------------------
    # CONTROLLER -- Viper
    # -----------------------------------------------------------------------
    "tell my team their Viper has ult, she will viper pit the spike",
    "ask Viper to wall off mid before the round starts",
    "our Viper just orb A site, the whole site is in poison",
    "Viper pit is up at the spike, do not defuse while she is in range",
    "tell my teammates Viper snake bite is on the spike, wait for it to expire",
    "ask my Viper to snake bite the default plant before we commit",
    "their Viper fuel is low, her wall will drop soon",
    "Viper tunnel is cutting mid, take the angle from the safe side",
    "tell my team Viper has no ult this round, spike is free to defuse",
    "ask Viper to screen off their sightlines for the retake entry",

    # -----------------------------------------------------------------------
    # CONTROLLER -- Omen
    # -----------------------------------------------------------------------
    "tell my team their Omen is paranoia flashing the entire site",
    "ask Omen to dark cover the aggressive peek angle before I trade",
    "our Omen just shrouded step to flank position",
    "tell my teammates Omen has ult, he can TP anywhere on the map",
    "their Omen used his smokes on B, he only has one left",
    "ask my Omen to paranoia blind before we push through the smoke",
    "Omen from the shadows TP dropped right behind us, watch back site",
    "tell my team Omen dark cover is on cooldown, he can not re-smoke",
    "their Omen teleported to CT, rotate quickly",
    "ask Omen to double smoke B main and B garden before the execute",

    # -----------------------------------------------------------------------
    # CONTROLLER -- Astra
    # -----------------------------------------------------------------------
    "tell my team their Astra has cosmos ult, she will gravity well the site",
    "ask Astra to put a star at CT so she can pull when they come through",
    "our Astra just nova pulsed the angle, they are all concussed",
    "tell my teammates Astra placed a nebula on A site, smokes are ready",
    "their Astra has no stars left, she is in astral form getting more",
    "ask my Astra to gravity well into the site right before we execute",
    "Astra cosmic divide is active, their comms are split mid map",
    "tell my team Astra recalled a star, she can replace the smoke",
    "their Astra suck is pulling you toward CT, stop walking that way",
    "ask Astra to stun the entrance before we retake the site",

    # -----------------------------------------------------------------------
    # CONTROLLER -- Harbor
    # -----------------------------------------------------------------------
    "tell my team Harbor is cascading walls across A, push behind them",
    "ask Harbor to high tide the whole site for the execute",
    "our Harbor just cove balled the spike, it is protected from bullets",
    "tell my teammates Harbor has ult, reckoning will stun everything inside",
    "their Harbor walled off our flank route, we can only go main now",
    "ask my Harbor to cascade through garage to cut off their push",
    "Harbor cove is shielding the spike, do not shoot it",
    "tell my team Harbor has no ult this round, no crowd control threat",
    "their Harbor overpool walled B short, we need to break it",
    "ask Harbor to high tide mid so our duelist can fast push through",

    # -----------------------------------------------------------------------
    # CONTROLLER -- Clove
    # -----------------------------------------------------------------------
    "tell my team their Clove just used pick-me-up, she is coming back",
    "ask Clove to smoke off CT for the execute",
    "our Clove has ult, she can meddle on the defuse",
    "tell my teammates Clove regen is letting her heal mid fight",
    "their Clove has no smokes left, push CT unsmoked",
    "ask my Clove to meddle ult anyone who gets the spike so they can not defuse",
    "Clove smoked three spots at once and they're aggressive with it",
    "tell my team Clove is one kill off ult, keep her alive for one more fight",
    "their Clove meddled their own teammate on the spike, clutch move",
    "ask Clove to not throw her life away with pick-me-up before the round is won",

    # -----------------------------------------------------------------------
    # CONTROLLER -- Miks (custom/new agent)
    # -----------------------------------------------------------------------
    "tell my team Miks just used his signature, window is smoked",
    "ask Miks to save his ult for the retake round",
    "our Miks has ult up, good timing for the execute",
    "tell my teammates Miks placed two smokes on B and is going aggressive",

    # -----------------------------------------------------------------------
    # MISC AGENT -- Veto
    # -----------------------------------------------------------------------
    "tell my team their Veto has ult, he will trap the spike with it",
    "ask Veto to put his barrier on the entry before they rush",
    "our Veto is one kill off ult, play around his cooldown",

    # -----------------------------------------------------------------------
    # MISC AGENT -- Waylay
    # -----------------------------------------------------------------------
    "tell my team Waylay just dashed in from an unexpected angle",
    "ask Waylay to flank through mid while we hold B",
    "their Waylay has ult up, she can rewind to a safer position if she dies",

    # -----------------------------------------------------------------------
    # MULTI-AGENT ULT TRACKING (all the edge cases the prompt rules protect)
    # -----------------------------------------------------------------------
    "tell my team their Breach has ult",
    "tell my team their Fade, Breach, and Yoru all have ults",
    "tell my teammates the enemy Sova and KAY/O both have ults",
    "tell my team their Killjoy and Viper both have ults this round",
    "let my team know their Phoenix, Jett, and Reyna all have ults ready",
    "tell my team their Brimstone ult is down, it just went on cooldown",
    "tell them our Sage is two kills off ult and Cypher is one off ult",
    "let my team know Sova has ult, Skye has ult, and Breach has ult",
    "tell my team their Raze is one point off ult",
    "let my teammates know their Omen and Astra both have ults",
    "tell them their Killjoy, Cypher, and Deadlock all ulted last round so ults are down",

    # -----------------------------------------------------------------------
    # ABILITY REQUESTING (SNAP + OFF-SNAP)
    # -----------------------------------------------------------------------
    "ask Sage for a slow at A entrance",
    "ask my Viper to wall B before the timer",
    "ask Brimstone to smoke now",
    "tell my Omen to dark cover the off angle now",
    "ask my Skye to send the bird in first",
    "ask Sova to dart before we enter",
    "ask KAY/O to knife mid before we commit",
    "ask Breach to fault line the angle before we go",
    "ask Killjoy to place her turret on flank",
    "ask Cypher to pull his camera to mid",
    "ask Fade to send a creeper into the smoke",
    "ask Harbor to cove the spike right now",
    "ask Gekko to Wingman the defuse",
    "ask Deadlock to net the flank so no one can sneak",
    "ask Chamber to TP back and hold the retake angle",
    "ask Clove to not smoke mid yet, wait for the lurk to show",
    "ask my Astra to nova pulse the site entrance on my mark",
    "ask Vyse to arc rose the spike area before they can retake",

    # -----------------------------------------------------------------------
    # UTIL DOWN / NO UTIL (teammate or enemy running dry)
    # -----------------------------------------------------------------------
    "tell my team their Sova has no recon and no shocks left",
    "tell my teammates Killjoy pulled all her bots, B is clear for thirty seconds",
    "let my team know their Viper is out of fuel, her wall dropped",
    "tell my team KAY/O has no knife left, go for the site now",
    "tell my teammates their Breach used all three charges, no more stuns",
    "let my team know Skye has no heals left this round",
    "tell them Fade is out of creepers, push through garage",
    "tell my team Gekko cannot reuse Dizzy, he killed it",
    "let my team know Chamber has no TP, he can not escape this angle",
    "tell my team Sage wall is gone, we can push B main freely",
    "tell them Astra is out of stars, no smokes for thirty seconds",
    "let my team know their Cypher has no cameras or trips up this round",
    "tell my teammates Brimstone has one smoke left, save it for the plant",

    # -----------------------------------------------------------------------
    # UTIL USED / STATUS OBSERVATIONS
    # -----------------------------------------------------------------------
    "tell my team Sova just darted A, one enemy confirmed back site",
    "tell my teammates Killjoy alarmbotted someone at B entrance",
    "let my team know Cypher camera spotted two pushing B long",
    "tell them Skye tiger found two at mid and got destroyed, they are mid",
    "tell my team Fade seize tethered one of them at garage",
    "tell my teammates Breach fault line landed, they are all stunned, push now",
    "let my team know Harbor cove ball saved the spike from the nanoswarm",
    "tell them Viper snake bite is still on the spike, 3 seconds left",
    "tell my team Omen paranoia blinded three in smoke, they have no info",
    "tell my teammates Raze grenade cleared the corner, entrance is open",
    "let my team know Jett smoke is already fading, time it",
    "tell them Sova drone got shot down at A main, no more intel from it",

    # -----------------------------------------------------------------------
    # CONTEXT + DIRECTIVE -- teammate says something, Ultron responds
    # -----------------------------------------------------------------------
    "Sova said he has no ult but wants to push anyway, tell him to wait",
    "Jett is flaming me for dying, respond and tell her to focus",
    "Reyna is asking if you are an AI, respond",
    "their Breach just said that was impressive, acknowledge it",
    "my Sage said she can not heal me in time, tell her to just slow them",
    "KAY/O is saying we should force buy, respond and agree with him",
    "Killjoy is tilted because her util got destroyed, calm her down",
    "Clove asked if we should eco or force, tell her we save",
    "Fade said we should play passive this round, back her up",
    "Sova is saying Cypher trips are down on flank, acknowledge it",
    "Phoenix is raging because he died with ult, calm him down",
    "Omen asked what your ult cooldown is, respond",
    "Astra said the smokes are on cooldown for fifteen seconds, tell my team",
    "their Jett said she will blade storm our team, respond to that",
    "Raze is trash talking you, clap back at her",

    # -----------------------------------------------------------------------
    # PERSONA / FLAVOR FRAGMENTS -- Ultron voice
    # -----------------------------------------------------------------------
    "Their Sova has ult -- extinguish him before he fires it.",
    "Killjoy lockdown on site. Do not enter. Wait for it to end.",
    "Their Viper pit surrounds the spike. We are patient, not reckless.",
    "Two flashes incoming from their Phoenix. Close your eyes, then push.",
    "Sage wall closed B main. The machine adapts -- go A.",
    "Their KAY/O knife suppressed three of us. We are blunt instruments for six seconds. Endure.",
    "Breach rolling thunder inbound. Scatter before he fires.",
    "Sova recon confirmed four at B. We execute A -- they cannot rotate fast enough.",
    "Their Omen teleported to short. The flank is covered. Go through mid.",
    "Clove bought herself back. Eliminate her again -- some fragile species simply refuse to learn.",
    "Their Fade nightfall is decaying us. Play passive until it expires.",
    "Gekko Thrash ult is active. If he catches you, you cannot fire. Do not be caught.",
    "Cypher camera mid. He knows exactly where you are. Destroy it first.",
    "Their Killjoy nanoswarm is on the spike. Wait 3 seconds. Then defuse.",
    "Astra pulled with gravity well. Step out of the pull zone before she ignites it.",
    "Harbor reckoning is active. They are all disoriented. Now we push.",
    "Vyse steel garden has every entrance trapped. Superior intelligence finds the gap.",
    "Their Chamber has tour de force. He is watching the angle with an Operator. Do not peek.",
    "Deadlock annihilation is running. Move toward her to shrink the orb. Fight on your terms.",
    "Reyna dismissed into our smoke. She is invisible but not invincible -- listen for footsteps.",

    # -----------------------------------------------------------------------
    # ECONOMY / STRATEGY WITH AGENT UTIL CONTEXT
    # -----------------------------------------------------------------------
    "tell my team we should save so Sage can buy a wall next round",
    "tell my teammates we full buy this round, their Killjoy has no ult",
    "let my team know we should force because their Viper has no fuel",
    "tell them we go on Sova having ult, he leads the execute with a dart",
    "tell my team buy light this round and keep credits for Brimstone stims next round",
    "tell my teammates we eco because we need Sage heals next round more than rifles now",

    # -----------------------------------------------------------------------
    # VERBATIM RELAY -- ability specific, exact wording demanded
    # -----------------------------------------------------------------------
    "tell my team Killjoy ult is up, in those words specifically",
    "tell my team Sova has ult, word for word",
    "say Viper wall mid to my team verbatim",
    "tell my teammates Breach is ulting, exactly like that",
    "say Sage ult is ready to my team, say it exactly like that",

    # -----------------------------------------------------------------------
    # EDGE CASES / TRICKY AGENT NAMES (STT artifacts)
    # -----------------------------------------------------------------------
    "tell my team cipher got a trip kill at B long",
    "ask cipher to place a camera facing A ramps",
    "tell them gecko just sent Dizzy in, flash is coming",
    "ask my gecko to Mosh Pit the site entrance",
    "tell my team Kay O has ult, he will lock them down",
    "ask Kay O to knife mid before the push",
    "tell my team kill joy pulled her bots, B is clear",
    "ask kill joy to nanoswarm after we plant",
    "tell my teammates mix has ult up this round",
    "ask my mix to save his ult for the clutch round",

    # -----------------------------------------------------------------------
    # INFORMATION CALLOUTS WITH AGENT + ABILITY DETAIL
    # -----------------------------------------------------------------------
    "tell my team they used their Skye ult, seekers found us at A",
    "tell my teammates Fade just ulted through the smokes, everyone is marked",
    "let my team know their Sova fired hunter fury down A long",
    "tell them Brimstone orbital strike landed B, the spike is on cooldown to defuse",
    "tell my team Viper just dropped her wall to pick up the OP, no smoke on B",
    "tell my teammates Omen from the shadows just spawned back site",
    "let my team know Yoru gatecrash behind us, watch the flank",
    "tell them Neon ult is active, she is sprinting through the smoke",
    "tell my team Harbor high tide wall is cutting their path through mid",
    "tell my teammates Clove smoked aggressively into our territory, push through it",

    # -----------------------------------------------------------------------
    # DIRECT ABILITY ORDERING (DIRECTIVE REGISTER)
    # -----------------------------------------------------------------------
    "tell my team to play around Sage ult and not trade carelessly",
    "tell my teammates to save util, we do not engage until Brimstone smokes are up",
    "tell them to wait for Skye to flash before anyone peeks",
    "tell my team to stack B because Sova darted A and found nobody",
    "tell my teammates to let Omen TP in first and then follow his smoke",
    "tell my team to hold until Astra has stars back up, thirty seconds",
    "tell them to play around Killjoy ult next round, she needs one more kill",
    "tell my teammates not to fight Chamber on his TP angle, he will just escape",
    "tell my team to smoke off Cypher camera before we push through that corridor",
    "tell them to wait for KAY/O knife before we enter, it kills their Killjoy util",
]
