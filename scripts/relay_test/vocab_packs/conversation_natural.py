"""Natural teammate conversation vocab pack for the Ultron relay test harness.

Domain: natural teammate-to-teammate and teammate-to-Ultron conversation a
streamer would relay mid-match -- greetings, GG, reactions, encouragement,
frustration, jokes, trash talk, small talk, agreeing/disagreeing, tilt,
clutch reactions, "wanna queue again", asking how someone is, hype, banter.

Each string is a single utterance: what the streamer says TO RELAY (e.g.
"tell my team gg"), what a teammate says TO ULTRON (e.g. "Reyna just asked
are you an AI, respond"), or a persona/flavor fragment representing a
conversational beat Ultron must handle.

Usage by the harness (harness.py / corpus.py):
    from scripts.relay_test.vocab_packs.conversation_natural import ITEMS
"""

ITEMS = [
    # -----------------------------------------------------------------------
    # GREETINGS / INTRODUCTIONS
    # -----------------------------------------------------------------------
    "tell my team gg let's win this",
    "tell my team hey guys",
    "tell my team good luck everyone",
    "tell my team let's get it",
    "tell my team let's go boys",
    "tell my team glad to be here",
    "tell my team what's up guys",
    "tell my team happy to be queuing with you",
    "greet my team",
    "introduce yourself to my team",
    "say hi to the squad and introduce yourself",
    "tell my team who you are",
    "say hey to my teammates",
    "tell my team first game of the night let's warm up",
    "tell my team fresh lobby, let's start clean",
    "tell my team morning boys, let's do this",

    # -----------------------------------------------------------------------
    # GG / MATCH END FAREWELLS
    # -----------------------------------------------------------------------
    "say gg to my team",
    "say goodbye to my team, we won",
    "say bye to my team, we lost",
    "tell my team gg ez",
    "tell my team gg wp",
    "tell my team good game everyone",
    "tell my team gg, well played all",
    "tell my team gg, that was a tough one",
    "tell my team rough game sorry guys",
    "close it out, we won",
    "close it out, we lost",
    "say gg to my team, we destroyed them",
    "wrap it up, we choked it",
    "give my team a goodbye statement",
    "tell my team thanks for the games",
    "tell my team gg see you next queue",
    "tell my team that was a good session",
    "tell my team rematch?",
    "tell my team same lobby again let's run it back",

    # -----------------------------------------------------------------------
    # NICE SHOT / PRAISE & COMPLIMENTS (streamer relaying to team)
    # -----------------------------------------------------------------------
    "tell my team nice shot",
    "tell my team great play",
    "tell my team that was clean",
    "tell my team incredible clutch",
    "tell my team nice clip that one",
    "tell my team well played",
    "tell my team that was nutty",
    "tell my team cracked",
    "tell my team insane aim bro",
    "tell my team good round everyone",
    "tell my team strong half nice work",
    "tell my team good half",
    "tell my team that was really smart",
    "tell my team nice read on that",
    "tell my team good call",
    "tell my team you carried that round",
    "tell my team that clutch was unreal",
    "tell my team nice job holding that angle",
    "tell my team good game sense",
    "tell my team beautiful execute",
    "tell my team okay that was actually sick",
    "tell my team okay I'm impressed",
    "tell my team you guys are popping off",
    "tell my team that's why we keep you around",
    "tell my team MVP behavior right there",
    "tell my team that entry was perfect",
    "tell my team okay now I respect it",
    "tell my team that rifle is doing work",
    "tell my team peeked that beautifully",
    "tell my team that was actually genius",

    # -----------------------------------------------------------------------
    # UNLUCKY / CONSOLATION
    # -----------------------------------------------------------------------
    "tell my team unlucky",
    "tell my team close one",
    "tell my team tough luck",
    "tell my team so close",
    "tell my team nice try",
    "tell my team that happens don't worry",
    "tell my team you almost had it",
    "tell my team that was a good fight",
    "tell my team rough round let's reset",
    "tell my team the RNG was terrible there",
    "tell my team bad timing that's all",
    "tell my team happens to everyone",
    "tell my team you played that well it just didn't go your way",
    "tell my team next round is ours",
    "tell my team shake it off",
    "tell my team these things happen in ranked",
    "tell my team that spray transfer was insane it just didn't hit",
    "tell my team two more and we get them",
    "tell my team almost there, one more round",
    "tell my team no worries let's just reset and focus",

    # -----------------------------------------------------------------------
    # MY BAD / APOLOGIES
    # -----------------------------------------------------------------------
    "tell my team my bad",
    "tell my team that was on me sorry",
    "tell my team I threw that round, sorry",
    "tell my team I should've rotated earlier my bad",
    "tell my team I missed that shot I'm sorry",
    "tell my team I messed up the execute sorry",
    "tell my team I gave him my gun I know I know",
    "tell my team I was on the wrong angle sorry",
    "tell my team that was a bad push on my part",
    "tell my team I panicked, won't happen again",
    "tell my team I called the wrong site sorry",

    # -----------------------------------------------------------------------
    # ENCOURAGEMENT / HYPE / MORALE
    # -----------------------------------------------------------------------
    "give my team some encouragement",
    "encourage my team",
    "hype up my team",
    "give my team a pep talk",
    "give my team a morale boost",
    "tell my team we can still win this",
    "tell my team don't give up",
    "tell my team we've got this",
    "tell my team stay focused",
    "tell my team lock in",
    "tell my team heads up",
    "tell my team we can come back from this",
    "tell my team trust the process",
    "tell my team just play your game",
    "tell my team we're the better team",
    "tell my team nothing's decided yet",
    "tell my team one round at a time",
    "tell my team this is winnable",
    "tell my team we have more than enough to close this out",
    "tell my team they're not as good as they think",
    "tell my team pressure is on them now",
    "tell my team we adapt",
    "tell my team mental stack reset go",
    "tell my team clean slate next round",
    "tell my team run it back same energy",
    "tell my team no shot we lose from here",
    "tell my team this is our round",
    "tell my team believe in yourselves for five seconds",

    # -----------------------------------------------------------------------
    # FRUSTRATION / TILT MANAGEMENT
    # -----------------------------------------------------------------------
    "tell my team calm down",
    "Reyna is tilted, calm her down",
    "Jett is raging, respond and calm him down",
    "Phoenix is malding, de-escalate",
    "Sova is flaming everyone, respond",
    "my teammate is crying in comms, calm him down",
    "Raze is really heated right now, reassure her",
    "my Killjoy is griefing and tilted, respond",
    "tell my team take a deep breath",
    "tell my team it's just ranked",
    "tell my team it's not that serious guys",
    "tell my team we're literally all on tilt right now and that needs to stop",
    "tell my team the game isn't over yet, chill",
    "tell my team blaming each other is making it worse",
    "tell my team let's just focus and stop the chat",
    "tell my team one bad round doesn't lose the match",
    "tell my team stop flaming we need each other",
    "tell my team muted, focusing",
    "tell my team ego checking isn't helping anybody",
    "tell my team save the argument for after the match",

    # -----------------------------------------------------------------------
    # TRASH TALK / INSULTS (team-directed and enemy-directed)
    # -----------------------------------------------------------------------
    "roast my team",
    "roast them",
    "flame my team",
    "tell my team they're bots",
    "tell my team that Jett has no idea what she's doing",
    "tell my team Phoenix has been inting all game",
    "tell my team that was genuinely the worst smoke I've ever seen",
    "tell my team their Reyna thinks she's a pro player",
    "tell my team the enemy is completely predictable",
    "tell my team these guys are hardstuck and it shows",
    "tell my team their aim is literally aimlabs-tier bad",
    "tell my team that Sage hasn't healed once this game",
    "tell my team the enemy controller has smoked themselves twice",
    "tell my team they literally have no comms",
    "tell my team the enemy is peeking the same angle every round",
    "tell my team Raze has been trolling all game",
    "tell my team Omen has been useless this half",
    "tell my team their strats are as creative as a coinflip",
    "tell my team their lurker is the most predictable player I've seen",
    "tell my team that's embarrassing for them honestly",

    # -----------------------------------------------------------------------
    # BANTER / JOKES / SMALL TALK
    # -----------------------------------------------------------------------
    "tell my team that was actually funny",
    "tell my team I can't stop laughing at that",
    "tell my team we need a highlight reel",
    "tell my team this is going in the compilation",
    "tell my team that could've gone worse somehow",
    "tell my team we're genuinely cursed this patch",
    "tell my team imagine losing to us",
    "tell my team honestly we don't deserve the W but we'll take it",
    "tell my team okay that was kinda cringe from me",
    "tell my team I've been in a dialogue with an AI all game",
    "tell my team the AI is talking smack again",
    "tell my team Ultron carried us again",
    "tell my team we should queue every day with Ultron",
    "tell my team peak gaming session right here",
    "tell my team this server is genuinely lagging",
    "tell my team I just had the most stupid idea for the execute",
    "tell my team that's what we do in the gutter Elo boys",
    "tell my team we are literally the most unserious team alive",
    "tell my team bro I almost alt-F4'd right there",
    "tell my team I genuinely have no idea how we win these",
    "tell my team I don't think this is what Riot intended",
    "tell my team ten dollars says they stack A every round",
    "tell my team okay respectfully that was kind of impressive from them",
    "tell my team I called it, I literally called it",
    "tell my team the RNG gods hate us specifically",
    "tell my team okay at least it was entertaining",

    # -----------------------------------------------------------------------
    # AGREEING / DISAGREEING
    # -----------------------------------------------------------------------
    "tell my team yeah that's a good call",
    "tell my team agree, let's go A",
    "tell my team no way, B is faster",
    "tell my team hard disagree, they're stacking B",
    "tell my team actually that might work",
    "tell my team I was thinking the same thing",
    "tell my team that's exactly what I was going to say",
    "tell my team I hear you but the timing's wrong",
    "tell my team sounds like a plan let's try it",
    "tell my team yeah okay fine let's do it your way",
    "tell my team no, we've been going A all half, switch it up",
    "tell my team trusting the call, let's go",
    "tell my team last round we did that and it didn't work",
    "tell my team I mean, you're the IGL not me",
    "tell my team I'll run it if you're confident",
    "tell my team I don't love it but okay",
    "tell my team that actually makes sense given their setup",
    "tell my team totally valid I was just thinking differently",

    # -----------------------------------------------------------------------
    # ASKING HOW SOMEONE IS / SMALL TALK BETWEEN ROUNDS
    # -----------------------------------------------------------------------
    "ask my Jett how their day was",
    "ask my Reyna how she's doing",
    "ask my team how everyone's feeling",
    "ask my Sova if they're warmed up",
    "ask my team if anyone needs a water break",
    "ask my Clove if they slept at all",
    "tell my team it's been a long day but I'm here",
    "tell my team first game back after work, bear with me",
    "tell my team I'm running on two hours of sleep",
    "tell my team finally logged in after a whole week",
    "tell my team good session yesterday, let's do it again",
    "ask my Killjoy what happened last game",
    "tell my team thanks for waiting on me",
    "tell my team I had to restart, sorry for the delay",
    "ask my team if they want to switch roles next half",
    "ask my team if everyone's comfortable with the comp",
    "ask my team if anyone has comms issues",
    "tell my team my mic was muted all of last round, sorry",
    "tell my team I might have to step out for a sec",
    "tell my team back now, what did I miss",

    # -----------------------------------------------------------------------
    # QUEUE AGAIN / REMATCH
    # -----------------------------------------------------------------------
    "ask my team if they want to queue again",
    "tell my team wanna run another",
    "ask my team if they're down for one more",
    "tell my team I'm queueing again if anyone wants to join",
    "tell my team last game for the night I think",
    "tell my team two more before I tap out",
    "tell my team I need to hit this rank tonight, one more",
    "tell my team I can't stop on a loss, one more",
    "tell my team I'm going to stop after this one win",
    "ask my team if they're partied up for the next game",
    "tell my team add me and let's party up",
    "tell my team I'll party if you invite me",
    "tell my team same five, let's run it",
    "ask my team if they're still down to stack",
    "tell my team that loss hurt but I'm requeuing",

    # -----------------------------------------------------------------------
    # TEAMMATES ASKING ULTRON ABOUT ITSELF (context + directive)
    # -----------------------------------------------------------------------
    "Reyna just asked if you are an AI, respond",
    "Jett wants to know if you're a real person, answer",
    "my Sova asked if you're a bot, respond",
    "Phoenix just asked what you are, respond",
    "Brimstone asked if you're a soundboard, handle it",
    "my Omen asked if you're a voice changer, respond",
    "Clove asked if you're actually human, answer them",
    "the team asked who are you, respond",
    "Sage asked what even is Ultron, respond",
    "my Killjoy asked how you work, answer",
    "Reyna asked if you're a streamer, respond",
    "Jett thinks you're a clip set on a soundboard, handle it",
    "Skye asked if you're live on Twitch right now, respond",
    "Phoenix called you a robot, respond",
    "Brimstone said prove you're not pre-recorded, handle it",
    "my Raze asked if you have feelings, respond",
    "Cypher asked if you're sentient, answer",
    "Yoru asked if you know you're going to die someday, respond",
    "Neon asked if AIs dream, respond",
    "Chamber asked where you come from, answer",

    # -----------------------------------------------------------------------
    # TEAMMATES INSULTING / BANTERING AT ULTRON (context + directive)
    # -----------------------------------------------------------------------
    "Reyna is making fun of you, respond",
    "Jett told you to shut up, respond",
    "Phoenix called you cringe, respond",
    "Raze said you sound annoying, handle it",
    "Sova is mocking you, clap back",
    "Brimstone told you to stop talking, shut him down",
    "my Omen said you're useless, respond",
    "Killjoy said you're just a bot reading lines, respond",
    "Neon laughed at your callout, respond",
    "my Clove called you a loser, set her straight",
    "Sage thinks you sound like a chatbot, respond",
    "Skye called you cringe, say something back",
    "Fade is roasting you, clap back",
    "Chamber called you mid, respond",
    "Yoru said you're not even good at this, handle it",
    "my Reyna is dissing you, back me up",
    "Phoenix is trolling you, respond",
    "Raze said you're just a recording, respond",
    "Iso told you your calls are trash, respond",
    "my Breach said he can play better without you, respond",

    # -----------------------------------------------------------------------
    # TEAMMATES TRASH-TALKING THE STREAMER (context + directive)
    # -----------------------------------------------------------------------
    "Jett just called me trash, defend me",
    "Reyna said I'm griefing them, back me up",
    "Phoenix is blaming me for the round, back me up",
    "my Sova said I should uninstall, back me up",
    "Raze called me a bot, defend me",
    "my Killjoy said I'm inting, defend me",
    "Brimstone accused me of throwing, back me up",
    "Omen said I have no game sense, respond and defend me",
    "Sage thinks I'm the problem, back me up",
    "Cypher called me hard stuck, respond",

    # -----------------------------------------------------------------------
    # TEAMMATES ASKING STRATEGY / GAME SENSE (context + directive)
    # -----------------------------------------------------------------------
    "Jett is asking what to do this round, respond",
    "my Sova wants to know the plan, respond",
    "Reyna asked if we should go A or B, respond",
    "Phoenix asked if we should save or force, respond",
    "my Clove asked whether we should play aggro or passive, respond",
    "Raze is wondering if she should ult now, respond",
    "my Killjoy asked if she should set up on site, respond",
    "Brimstone wants to know where to smoke, respond",
    "Omen asked if he should lurk or go with us, respond",
    "my Sage wants to know when to heal, respond",

    # -----------------------------------------------------------------------
    # MARVEL / IDENTITY BANTER (context + directive)
    # -----------------------------------------------------------------------
    "Reyna asked if you're the Ultron from the Avengers, respond",
    "Jett said she thought the Avengers killed you, respond",
    "Phoenix mentioned Tony Stark made you, respond",
    "Sova asked if you know Iron Man, respond",
    "my Brimstone brought up the Avengers, respond",
    "Clove said your movie was terrible, respond",
    "Raze said Sokovia was your fault, respond",
    "Yoru asked if you hate Tony Stark, respond",
    "Neon asked what you think of Vision, respond",
    "my Iso asked if you would beat Thanos, respond",
    "Chamber asked if Thor could stop you, respond",
    "my Skye said you remind her of the Avengers, respond",
    "Fade asked if you're from the MCU, respond",
    "Killjoy asked about Quicksilver, respond",
    "Deadlock asked what Scarlet Witch means to you, respond",

    # -----------------------------------------------------------------------
    # GENERAL KNOWLEDGE QUESTIONS (teammates asking Ultron in-game)
    # -----------------------------------------------------------------------
    "Reyna asked why the sky is blue, respond",
    "Jett wants to know how far the moon is, respond",
    "Phoenix asked what happened to the dinosaurs, respond",
    "my Sova asked what the meaning of life is, respond",
    "Raze asked how fast light travels, respond",
    "Clove asked if time travel is possible, respond",
    "Killjoy asked who invented the internet, respond",
    "Chamber asked how black holes form, respond",
    "Omen asked if we are alone in the universe, respond",
    "Brimstone asked what dark matter is, respond",
    "Skye asked how many stars there are in the galaxy, respond",
    "Sage asked why humans need sleep, respond",
    "Fade asked what causes earthquakes, respond",
    "Neon asked if the multiverse is real, respond",
    "Iso asked what the biggest number is, respond",

    # -----------------------------------------------------------------------
    # ACKNOWLEDGEMENT / AGREE / DIRECT RESPONSE DIRECTIVES
    # -----------------------------------------------------------------------
    "Reyna said good call, acknowledge her",
    "Jett said she agrees with the plan, acknowledge",
    "Phoenix said nice play, acknowledge him",
    "my Sova said she appreciates the callouts, acknowledge",
    "my Clove said the smokes were perfect, acknowledge",
    "my team said the info was helpful, acknowledge",
    "Sage said Ultron you're actually useful, agree with her",
    "Killjoy said this strat is working, acknowledge",
    "Brimstone said follow the IGL, agree with him",
    "Omen said we should play more patient, agree with him",

    # -----------------------------------------------------------------------
    # PURE MORALE / SHORT PHRASES (console/praise that the streamer relays)
    # -----------------------------------------------------------------------
    "tell my team let's go",
    "tell my team nice",
    "tell my team clutch",
    "tell my team gg",
    "tell my team almost",
    "tell my team nice work",
    "tell my team good round",
    "tell my team good game",
    "tell my team incredible",
    "tell my team you're the GOAT",
    "tell my team we run this",
    "tell my team they can't stop us",
    "tell my team we are not losing today",
    "tell my team that's what I'm talking about",
    "tell my team that's the energy right there",
    "tell my team okay we actually look good",
    "tell my team we are so back",
    "tell my team this is giving comeback arc",
    "tell my team we are literally built different",
    "tell my team said so",

    # -----------------------------------------------------------------------
    # MISCELLANEOUS CONVERSATIONAL REACTIONS
    # -----------------------------------------------------------------------
    "tell my team I love this team honestly",
    "tell my team this is the most fun I've had ranked in a while",
    "tell my team we should make this a regular stack",
    "tell my team I've been listening to Ultron all game and it's actually helping",
    "tell my team I did not expect that outcome but I'll take it",
    "tell my team we literally outplayed ourselves more than they outplayed us",
    "tell my team I think we need a better default honestly",
    "tell my team the vod review is going to be painful but educational",
    "tell my team I felt that in my soul",
    "tell my team that was some masterclass level trolling right there",
    "tell my team I'm going to need a moment to recover from that one",
    "tell my team okay we are genuinely on different pages this round",
    "tell my team let's just reset and run a clean default",
    "tell my team I don't know what just happened but it worked",
    "tell my team we've been getting these read by the same angle all half",
    "tell my team they know our timings, we need to mix it up",
    "tell my team we keep playing into their strengths",
    "tell my team I love the ambition but the execution killed us",
    "tell my team I think we underestimated their lurker",
    "tell my team next time we buy full on round three no matter what",
    "tell my team their rotates are stupid fast we need to account for that",
    "tell my team we take too long to commit and they abuse it every time",
    "tell my team one day we will run that execute without anyone dying early",
    "tell my team I promise I had them from my angle",
    "tell my team the sound design on this map is actually criminal",
    "tell my team no comms game next round, purely vibes",
    "tell my team we literally need a therapist to fix this team dynamic",
    "tell my team that's the most games I've played in one sitting in months",
    "tell my team somehow this was still fun despite everything",
    "tell my team respectfully that was a five-star disaster",

    # -----------------------------------------------------------------------
    # FUN FACT / ROAST CORPUS TRIGGERS
    # -----------------------------------------------------------------------
    "tell my team a fun fact",
    "give my team a fun fact",
    "drop a fun fact in chat",
    "share my team an interesting fact",
    "hit them with a random fact",
    "tell them a cool fact",
    "roast my squad",
    "roast everyone",
    "flame them",
    "give my team a roast",
]
