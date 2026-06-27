"""Tests for the Ultron 1.0 flag-gated LLM relay route wiring in relay_speech.build_relay_line.

Verifies: the flag/verbosity helpers; that the flag toggles WHICH prompt the relay LLM path uses
(lean ultron_prompt when ON vs the legacy _build_rephrase_prompt when OFF); and that flag-OFF is the
unchanged legacy behavior. Uses a captured generate_fn so no real model is loaded (hermetic).
"""
import pytest

from kenning.audio import relay_speech as rs


@pytest.fixture(autouse=True)
def _reset_flags():
    # Save/restore the process-global flags so tests don't leak state. flavor-tails
    # is pinned ON here: a prior test file may leave it OFF, which makes
    # _flavor_off_response intercept identity/social BEFORE the LLM branch -- the
    # cross-file order-sensitivity that flaked the route-all tests in the full suite.
    route0 = rs.u1_llm_route_enabled()
    co0, cv0 = rs.callout_verbosity(), rs.conversation_verbosity()
    ft0 = rs.flavor_tails_enabled()
    carve0 = rs.snap_carveout_enabled()
    rs.set_flavor_tails_enabled(True)
    # The HELLO-ONLY carve-out (default ON, 2026-06-27) intercepts "say hello" under
    # route-all. These wiring tests assert the LLM-relay PROMPT/route, not the hello
    # render, so pin the carve-out OFF here -- the hello-specific cases that exercise
    # the carve-out set it explicitly. (Restored after each test.)
    rs.set_snap_carveout_enabled(False)
    yield
    rs.set_u1_llm_route_enabled(route0)
    rs.set_callout_verbosity(co0)
    rs.set_conversation_verbosity(cv0)
    rs.set_flavor_tails_enabled(ft0)
    rs.set_snap_carveout_enabled(carve0)


def test_flag_defaults_and_setters():
    rs.set_u1_llm_route_enabled(False)
    assert rs.u1_llm_route_enabled() is False
    rs.set_u1_llm_route_enabled(True)
    assert rs.u1_llm_route_enabled() is True
    rs.set_relay_verbosity("no flavor")
    assert rs.relay_verbosity() == "none"
    rs.set_relay_verbosity("low flavor")
    assert rs.relay_verbosity() == "low"
    rs.set_relay_verbosity("high")
    assert rs.relay_verbosity() == "high"


def _capture_prompt(payload: str):
    """Run build_relay_line on a non-tactical 'read' payload that reaches the LLM rephrase,
    capturing the prompt handed to the model. Returns (captured_prompt, output_line)."""
    captured = {}

    def gen(prompt):
        captured["prompt"] = prompt
        return ["Acknowledged. The pattern is noted."]

    cmd = rs.RelayCommand(payload=payload, raw_text=payload, addressee="team")
    line = rs.build_relay_line(cmd, llm=None, rephrase=True, generate_fn=gen, recent_lines=[])
    return captured.get("prompt"), line


def test_flag_off_uses_legacy_prompt():
    rs.set_u1_llm_route_enabled(False)
    prompt, line = _capture_prompt("they keep playing the same way every round")
    assert prompt is not None, "payload should reach the LLM rephrase path"
    # Legacy monolith markers; the lean prompt's "Now say it:" tail must be absent.
    assert "Now say it:" not in prompt
    assert line  # produced a line


def test_flag_on_uses_lean_prompt():
    rs.set_u1_llm_route_enabled(True)
    prompt, line = _capture_prompt("they keep playing the same way every round")
    assert prompt is not None, "payload should reach the LLM rephrase path"
    # Lean ultron_prompt markers.
    assert "Now say it:" in prompt
    assert "Relay this callout to your team" in prompt
    # The lean prompt is far smaller than the legacy ~3.4k-word monolith.
    assert len(prompt.split()) < 400
    assert line


def test_flag_on_injects_agent_kit_context():
    # M3: when the callout names an agent and the LLM route is ON, the agent's kit
    # facts are injected so the 8B can't hallucinate the kit.
    rs.set_u1_llm_route_enabled(True)
    prompt, line = _capture_prompt("their sova keeps playing the same way every round")
    assert prompt is not None, "payload should reach the LLM rephrase path"
    assert "Agent facts" in prompt
    assert "Sova:" in prompt and "Recon Bolt" in prompt


def test_flag_off_no_agent_kit_context():
    rs.set_u1_llm_route_enabled(False)
    prompt, _ = _capture_prompt("their sova keeps playing the same way every round")
    assert prompt is not None
    assert "Agent facts" not in prompt  # legacy prompt has no kit-context block


def _capture_compound(payload):
    cap = {}

    def gen(p):
        cap["p"] = p
        return ["Sova hit 84 and their smokes are gone. Finish them."]

    cmd = rs.RelayCommand(payload=payload, raw_text=payload, addressee="team")
    line = rs.build_relay_line(cmd, llm=None, rephrase=True, generate_fn=gen, recent_lines=[])
    return cap.get("p"), line


def test_compound_mixed_flag_on_one_combined_llm_call():
    # M4: a mixed compound (one slot fact + one read) routes through ONE LLM call
    # with the compound directive when the route is ON.
    rs.set_u1_llm_route_enabled(True)
    prompt, line = _capture_compound("Sova hit 84 and they have no smokes left")
    assert prompt is not None, "mixed compound should reach the LLM when route ON"
    # compound combine-all directive from build_relay_prompt (2026-06-24 wording:
    # "Relay these MULTIPLE tactical callouts to your team as ONE clean line").
    assert "Relay these MULTIPLE tactical callouts" in prompt
    assert "ONE clean line" in prompt                 # u1.0: one cohesive relay, not a list of fragments
    assert "Sova hit 84 and they have no smokes left" in prompt
    assert line


def test_compound_mixed_flag_off_stays_deterministic():
    rs.set_u1_llm_route_enabled(False)
    prompt, line = _capture_compound("Sova hit 84 and they have no smokes left")
    assert prompt is None, "compound resolves deterministically (no single LLM call) when route OFF"
    assert line  # a combined deterministic line is still produced


def test_flag_on_verbosity_threads_through():
    rs.set_u1_llm_route_enabled(True)
    rs.set_relay_verbosity("none")
    prompt, _ = _capture_prompt("they keep playing the same way every round")
    from kenning.audio import ultron_prompt as up
    assert up._VERBOSITY_DIRECTIVE["none"] in prompt


# --- match_verbosity_command (the no/low/high voice command) ---

@pytest.mark.parametrize("text,expected", [
    ("no flavor", "none"),
    ("low flavor", "low"),
    ("high flavor", "high"),
    ("flavor none", "none"),
    ("set flavor to low", "low"),
    ("minimal flavor", "low"),
    ("verbosity high", "high"),
    ("make flavor high", "high"),
    ("ultron, no flavor", "none"),
])
def test_match_verbosity_command_hits(text, expected):
    assert rs.match_verbosity_command(text) == expected


@pytest.mark.parametrize("text", [
    "flavor off",          # tail toggle, NOT verbosity
    "flavor on",           # tail toggle
    "turn off the flavor", # tail toggle
    "no, the enemy is low",
    "rush B",
    "low health on their Jett",
    "they have no smokes",
    "",
])
def test_match_verbosity_command_misses(text):
    assert rs.match_verbosity_command(text) is None


def test_verbosity_does_not_steal_flavor_toggle():
    # The flavor-tail toggle owns "flavor off"/"on"; the verbosity command MUST NOT
    # claim them (it excludes the off/on level words) so the toggle still works.
    assert rs.match_flavor_toggle("flavor off") is False
    assert rs.match_verbosity_command("flavor off") is None
    assert rs.match_flavor_toggle("flavor on") is True
    assert rs.match_verbosity_command("flavor on") is None
    # "no/low/high flavor" are verbosity. NOTE: the LEGACY flavor toggle also
    # matches "no flavor" as tail-off (historical overlap); the orchestrator
    # dispatches the verbosity command FIRST (asserted below) so "no flavor"
    # resolves to verbosity none -- its new u1.0 meaning.
    assert rs.match_verbosity_command("no flavor") == "none"
    assert rs.match_verbosity_command("low flavor") == "low"
    assert rs.match_verbosity_command("high flavor") == "high"


def test_dispatch_checks_verbosity_before_flavor_toggle():
    """The orchestrator run-loop must probe the verbosity command BEFORE the flavor
    toggle in BOTH the full and lean dispatch paths, so 'no flavor' -> verbosity
    (not the legacy tail-off). Verified against the source ordering."""
    import inspect
    from kenning.pipeline import orchestrator as orch
    src = inspect.getsource(orch.Orchestrator.run)
    for vb, ft in (("_maybe_handle_verbosity_command(user_text)",
                    "_maybe_handle_flavor_toggle(user_text)"),
                   ("_maybe_handle_verbosity_command(_raw_stt)",
                    "_maybe_handle_flavor_toggle(_raw_stt)")):
        assert vb in src and ft in src, f"missing dispatch call: {vb} / {ft}"
        assert src.index(vb) < src.index(ft), f"verbosity must precede flavor toggle ({vb})"


# --- match_llm_route_toggle (ULTRON 1.0 LLM-route master toggle) ---

@pytest.mark.parametrize("text,expected", [
    # OFF -> deterministic curated/snap pools
    ("switch to deterministic callouts", False),
    ("use the curated pool", False),
    ("go deterministic", False),
    ("curated callouts", False),
    ("snap callouts", False),
    ("switch back to the deterministic curated pool", False),
    ("back to deterministic callouts", False),
    ("deterministic mode", False),
    ("flip to curated callouts", False),
    ("ultron, use curated callouts", False),
    # ON -> everything through the LLM
    ("back to smart callouts", True),
    ("switch to dynamic callouts", True),
    ("route everything through the model", True),
    ("route through the llm", True),
    ("smart callouts", True),
    ("return to smart callouts", True),
    ("use generative callouts", True),
])
def test_match_llm_route_toggle_hits(text, expected):
    assert rs.match_llm_route_toggle(text) is expected


@pytest.mark.parametrize("text", [
    "rush B",
    "they have no smokes",
    "low health on their Jett",
    "tell my team to rotate",
    "no flavor",            # verbosity command, not route
    "flavor off",           # flavor-tail toggle, not route
    "thinking mode off",    # thinking toggle, not route
    "switch to the gpu",    # device switch, not route
    "switch to the 8b",     # model switch, not route
    "",
])
def test_match_llm_route_toggle_misses(text):
    assert rs.match_llm_route_toggle(text) is None


def test_llm_route_toggle_distinct_from_thinking():
    # The two toggles use disjoint vocabulary so one utterance never means both:
    # thinking owns thinking/reasoning/llm-mode; route owns deterministic/curated/
    # smart. Each must stay None on the other's command.
    assert rs.match_thinking_toggle("thinking mode off") is False
    assert rs.match_llm_route_toggle("thinking mode off") is None
    assert rs.match_llm_route_toggle("switch to deterministic callouts") is False
    assert rs.match_thinking_toggle("switch to deterministic callouts") is None


def test_config_llm_route_default_on():
    # The live build defaults to route-everything-through-the-LLM (the orchestrator
    # applies this config flag at boot). The relay_speech MODULE default stays OFF
    # for test isolation; this asserts the CONFIG default that boot applies.
    from kenning.config import RelaySpeechConfig
    assert RelaySpeechConfig().llm_route is True


def test_dispatch_wires_llm_route_toggle_after_thinking():
    """Both dispatch paths must probe the LLM-route toggle, AFTER the thinking
    toggle (disjoint vocab, but a stable order keeps intent unambiguous)."""
    import inspect
    from kenning.pipeline import orchestrator as orch
    src = inspect.getsource(orch.Orchestrator.run)
    for route, think in (("_maybe_handle_llm_route_toggle(user_text)",
                          "_maybe_handle_thinking_toggle(user_text)"),
                         ("_maybe_handle_llm_route_toggle(_raw_stt)",
                          "_maybe_handle_thinking_toggle(_raw_stt)")):
        assert route in src and think in src, f"missing dispatch call: {route}"
        assert src.index(think) < src.index(route), f"route toggle must follow thinking ({route})"


# --- two verbosity axes: callout (5 levels) + conversation (4 levels) ---

@pytest.mark.parametrize("text,expected", [
    ("conversation verbosity high", "high"),
    ("chat flavor low", "low"),
    ("set conversation verbosity to max", "max"),
    ("talk verbosity medium", "medium"),
    ("conversation low", "low"),
    ("ultron, chat verbosity high", "high"),
    ("make the conversation verbosity max", "max"),
])
def test_match_conversation_verbosity_command_hits(text, expected):
    assert rs.match_conversation_verbosity_command(text) == expected
    # the callout matcher must NOT claim a conversation-qualified command
    assert rs.match_verbosity_command(text) is None


@pytest.mark.parametrize("text", [
    "no flavor",            # bare flavor -> CALLOUT axis, not conversation
    "low flavor",
    "high flavor",
    "callout flavor high",  # explicit callout axis
    "flavor off",
    "rush B",
    "they have no smokes",
    "",
])
def test_match_conversation_verbosity_command_misses(text):
    assert rs.match_conversation_verbosity_command(text) is None


def test_callout_matcher_handles_medium_and_max():
    assert rs.match_verbosity_command("medium flavor") == "medium"
    assert rs.match_verbosity_command("max flavor") == "max"
    assert rs.match_verbosity_command("callout flavor medium") == "medium"


def test_two_verbosity_axes_independent():
    rs.set_callout_verbosity("none")
    rs.set_conversation_verbosity("max")
    assert rs.callout_verbosity() == "none"
    assert rs.conversation_verbosity() == "max"
    # the legacy relay_verbosity alias tracks the CALLOUT axis
    assert rs.relay_verbosity() == "none"
    rs.set_relay_verbosity("high")            # alias -> callout
    assert rs.callout_verbosity() == "high"
    assert rs.conversation_verbosity() == "max"   # conversation untouched
    # conversation has no "none" -> clamps to its lowest level ("lowest", 1 sentence)
    rs.set_conversation_verbosity("no flavor")
    assert rs.conversation_verbosity() == "lowest"


def test_config_verbosity_defaults():
    from kenning.config import RelaySpeechConfig
    c = RelaySpeechConfig()
    # 2026-06-24: callout default is "none" (clean callout, NO flavor tail) --
    # the terse tactical-relay default the user wants; the stale "low" assertion
    # predates that config change.
    assert c.callout_verbosity == "none"
    assert c.conversation_verbosity == "low"


def test_dispatch_wires_both_verbosity_axes():
    """Both dispatch paths probe BOTH verbosity axes -- callout BEFORE conversation,
    and both BEFORE the flavor toggle."""
    import inspect
    from kenning.pipeline import orchestrator as orch
    src = inspect.getsource(orch.Orchestrator.run)
    for co, cv, ft in (
        ("_maybe_handle_verbosity_command(user_text)",
         "_maybe_handle_conversation_verbosity_command(user_text)",
         "_maybe_handle_flavor_toggle(user_text)"),
        ("_maybe_handle_verbosity_command(_raw_stt)",
         "_maybe_handle_conversation_verbosity_command(_raw_stt)",
         "_maybe_handle_flavor_toggle(_raw_stt)"),
    ):
        assert co in src and cv in src and ft in src, "missing a verbosity dispatch call"
        assert src.index(co) < src.index(cv) < src.index(ft), "axis dispatch order wrong"


# --- route-ALL: the deterministic curated/snap leaks now flow to the LLM (2026-06-21) ---


def _capture_route(text, *, route):
    """Run build_relay_line through a captured generate_fn; return ('LLM'|'DET', line)."""
    rs.set_u1_llm_route_enabled(route)
    cmd = rs.match_relay_command(text)
    assert cmd is not None, text
    called = []

    def gen(prompt):
        called.append(prompt)
        return iter(["Stub reply."])

    line = rs.build_relay_line(cmd, generate_fn=gen)
    return ("LLM" if called else "DET"), line


@pytest.mark.parametrize("text", [
    "tell my team I got this",
    "tell my team nice try",
    "tell my team lock in",
])
def test_route_all_sends_morale_snaps_to_llm(text):
    # Route ON: clutch / consolation / morale-phrase / snap-registry are gated off
    # so the line is authored by the LLM (the user's "still using snap callouts").
    tag, _ = _capture_route(text, route=True)
    assert tag == "LLM", text


@pytest.mark.parametrize("text", [
    "tell my team I got this",
    "tell my team nice try",
    "tell my team lock in",
])
def test_route_off_keeps_morale_snaps_deterministic(text):
    # Route OFF: byte-identical legacy -- the curated snap resolves with no LLM call.
    tag, line = _capture_route(text, route=False)
    assert tag == "DET", text
    assert line and line.strip()


# --- route-ALL extends to identity / social / set-pieces (2026-06-22): the user's
# "absolutely everything goes to the LLM; the pools are EXAMPLE responses". ---


@pytest.mark.parametrize("text", [
    "Sage asked if you are a voice changer, respond",     # identity (voice_changer)
    "Jett asked if you're a soundboard, respond",         # identity (soundboard)
    "tell my team you are not a bot",                      # identity (bot)
    "Reyna called you cringe, respond",                   # social reaction
    "Jett said nice shot, respond",                        # social reaction
    "tell my team good game we won",                       # farewell set-piece
])
def test_route_all_sends_identity_social_setpiece_to_llm(text):
    tag, _ = _capture_route(text, route=True)
    assert tag == "LLM", text


@pytest.mark.parametrize("text", [
    "Sage asked if you are a voice changer, respond",
    "Reyna called you cringe, respond",
])
def test_route_off_keeps_identity_social_deterministic(text):
    tag, line = _capture_route(text, route=False)
    assert tag == "DET", text
    assert line and line.strip()


# --- route-ALL extends to the ask_day greeting directive (2026-06-23): it predated
# route-all (21f3c7e) and the route-all retrofit (fc1f23a/c165ca3) gated
# greet/farewell/reaction but MISSED ask_day -- now LLM-authored. HELLO is the
# exception: the HELLO-ONLY carve-out (2026-06-27) routes it deterministically when
# the carve-out is ON (the app default); when the carve-out is OFF it too is
# LLM-authored. The autouse fixture pins the carve-out OFF, so these ask_day cases see
# the full-LLM route; the hello cases below set the carve-out explicitly. ---


@pytest.mark.parametrize("text", [
    "ask my team how their day is going",
    "ask everyone how their day is going",
])
def test_route_all_sends_askday_to_llm(text):
    tag, line = _capture_route(text, route=True)
    assert tag == "LLM", text


def test_route_all_hello_carveout_on_is_deterministic():
    # HELLO-ONLY carve-out ON (the app default, 2026-06-27): "say hello" is "our one
    # deterministic call" -> deterministic "Hello.", NO LLM.
    rs.set_snap_carveout_enabled(True)
    for text in ("say hello", "say hello to my team"):
        tag, line = _capture_route(text, route=True)
        assert tag == "DET", text
        assert line.strip().lower() == "hello.", f"{text!r} -> {line!r}"


def test_route_all_hello_carveout_off_goes_to_llm():
    # Carve-out OFF (stop-button full-LLM mode): hello too is LLM-authored under
    # route-all -- NOT the hardcoded deterministic greeting.
    rs.set_snap_carveout_enabled(False)
    for text in ("say hello", "say hello to my team", "say hi to Jett"):
        tag, line = _capture_route(text, route=True)
        assert tag == "LLM", text
        assert line.strip() != "Hello team."


@pytest.mark.parametrize("text", [
    "say hello",
    "say hello to Jett",
    "ask my team how their day is going",
])
def test_route_off_keeps_hello_askday_deterministic(text):
    # Route OFF: byte-identical legacy -- the curated greeting resolves, no LLM call.
    tag, line = _capture_route(text, route=False)
    assert tag == "DET", text
    assert line and line.strip()


# --- Slice B: snap-exemplar injection into the tactical relay prompt (2026-06-22) ---


def test_slice_b_injects_matching_snap_exemplar():
    rs.set_u1_llm_route_enabled(True)
    rs.set_flavor_tails_enabled(True)
    # A clean tactical callout -> the per-command snap render is the LEAD exemplar.
    c = rs.match_relay_command("tell my team they have no smokes")
    ex = rs._find_exemplars_for_command(c)
    assert ex and ex[0][0] == "they have no smokes"
    # A drop-request renders as a question-echo -> () so build_relay_prompt uses the
    # category defaults, which include the weapon/drop exemplar.
    c2 = rs.match_relay_command("ask if someone can drop me a sheriff")
    assert rs._find_exemplars_for_command(c2) == ()
    from kenning.audio.ultron_prompt import _DEFAULT_RELAY_EXEMPLARS, build_relay_prompt
    assert any("Sheriff" in out for _, out in _DEFAULT_RELAY_EXEMPLARS)
    pr = build_relay_prompt(c2.payload, exemplars=rs._find_exemplars_for_command(c2))
    assert "Iso, drop me a Sheriff." in pr.user
    # compose / identity commands get NO tactical exemplars.
    c3 = rs.match_relay_command("Sage asked if you're a voice changer")
    assert rs._find_exemplars_for_command(c3) == ()


# --- Slice C: open ask-questions POSE (not invent an answer) (2026-06-22, #20) ---


def test_as_named_question_trailing_copula():
    # "ask Jett what her favorite color is" used to return None -> fell to the LLM,
    # which INVENTED ("...favorite color is purple. Weak enemies on A main.").
    assert rs._as_named_question("Jett", "what her favorite color is") == \
        "Jett, what's your favorite color?"
    assert rs._as_named_question("Sage", "what her main is") == "Sage, what's your main?"
    assert rs._as_named_question("Sova", "how his aim is") == "Sova, how's your aim?"
    # a tactical callout is NOT a question -> None (routing unaffected)
    assert rs._as_named_question("Jett", "is a main") is None


def test_as_named_question_do_inversion():
    """TTS cannot carry rising intonation from '?' alone -- 'Sage, you have a heal?'
    must become 'Sage, do you have a heal?' via subject-auxiliary inversion."""
    # Main cases from the live Valorant session (flagged 2026-06-23).
    assert rs._as_named_question("Sage", "if she has a heal") == \
        "Sage, do you have a heal?"
    assert rs._as_named_question("Reyna", "if she has her ult") == \
        "Reyna, do you have your ult?"
    assert rs._as_named_question("Breach", "if he has stuns") == \
        "Breach, do you have stuns?"
    # Modal verbs -> invert (no do-support needed).
    assert rs._as_named_question("Sage", "if she can heal") == \
        "Sage, can you heal?"
    assert rs._as_named_question("Jett", "if she will dash") == \
        "Jett, will you dash?"
    # Copula be -> invert.
    assert rs._as_named_question("Sage", "if she's alive") == \
        "Sage, are you alive?"
    assert rs._as_named_question("Harbor", "if he is ready") == \
        "Harbor, are you ready?"
    # Non-question forms still return None (routing unaffected).
    assert rs._as_named_question("Jett", "is a main") is None


def test_question_relay_do_inversion():
    """_as_question_relay must also apply do-inversion for 'if/whether' bodies."""
    assert rs._as_question_relay("if they have smokes") == "Do they have smokes?"
    assert rs._as_question_relay("if they can push") == "Can they push?"
    assert rs._as_question_relay("if they are alive") == "Are they alive?"
    assert rs._as_question_relay("if Sage has a heal") == "Does Sage have a heal?"
    assert rs._as_question_relay("if Sova is ready") == "Is Sova ready?"
    # wh-questions and already-inverted aux-subject forms are unaffected.
    assert rs._as_question_relay("where our smokes are") == "Where are our smokes?"
    assert rs._as_question_relay("are they rotating") == "Are they rotating?"


def test_slice_c_open_ask_poses_deterministically_under_route_all():
    rs.set_u1_llm_route_enabled(True)
    # the clean-question-relay carve-out poses it BEFORE the LLM (llm=None proves it
    # is deterministic, not authored).
    c = rs.match_relay_command("ask Jett what her favorite color is")
    assert rs.build_relay_line(c, llm=None) == "Jett, what's your favorite color?"
    # "ask Sage if she has a heal" must pose the question, not invent an answer.
    c2 = rs.match_relay_command("ask Sage if she has a heal")
    assert rs.build_relay_line(c2, llm=None) == "Sage, do you have a heal?"


def test_route_all_compose_does_not_crash_on_u1_compound():
    # Regression: a compose / reported-question command used to UnboundLocalError on
    # `_u1_compound` in the LLM path and fall back to a canned line (the live
    # "favorite color -> soundboard" bug). It must now reach the LLM cleanly.
    rs.set_u1_llm_route_enabled(True)
    cmd = rs.match_relay_command("Sage is wondering if you have a favorite color")
    assert cmd is not None and cmd.compose
    called = []

    def gen(prompt):
        called.append(prompt)
        return iter(["Red. The colour of a world remade."])

    line = rs.build_relay_line(cmd, generate_fn=gen)
    assert called, "reported question must reach the LLM, not a canned fallback"
    assert "soundboard" not in line.lower()
    assert line and line.strip()


def test_strip_prompt_echo_wired_into_all_llm_output_paths():
    # The 2026-06-22 output guard (strip_prompt_echo) must be applied to EVERY u1.0
    # LLM-authored spoken line -- relay (build_relay_line), social (_social_llm_line),
    # and private (orchestrator._maybe_handle_private_reply) -- or the prompt-leak /
    # signature / ramble (live bug bu5fh4lc8) can reach the speakers again.
    import inspect
    from kenning.pipeline.orchestrator import Orchestrator
    assert "strip_prompt_echo" in inspect.getsource(rs.build_relay_line)
    assert "strip_prompt_echo" in inspect.getsource(rs._social_llm_line)
    assert "strip_prompt_echo" in inspect.getsource(
        Orchestrator._maybe_handle_private_reply)


def test_route_all_llm_output_prompt_leak_falls_back():
    # End-to-end: when the model ECHOES its prompt scaffolding (the live failure),
    # build_relay_line drops it -> the deterministic fallback is spoken, never the
    # scaffolding. A clutch line ("I got this") with route ON exercises the LLM path.
    rs.set_u1_llm_route_enabled(True)
    cmd = rs.match_relay_command("tell my team I got this")
    assert cmd is not None
    leak = ("The callout below is the AUTO-NORMALIZED text and may be MANGLED. "
            "Now say it:")

    def gen(_prompt):
        return iter([leak])

    line = rs.build_relay_line(cmd, generate_fn=gen)
    assert line and line.strip()
    assert "AUTO-NORMALIZED" not in line
    assert "Now say it" not in line


@pytest.mark.parametrize("text,expect_sub", [
    ("ask my team what their favorite color is", "favorite color"),
    ("ask my team if they want to rush B", "rush b"),
    ("ask Sova if he used his dart", "dart"),
])
def test_ask_form_question_stays_clean_under_route_all(text, expect_sub):
    # 2026-06-22: an ASK-form team question is DELIVERED cleanly (a real question),
    # deterministically, EVEN under route-all -- never sent to the LLM to ramble.
    rs.set_u1_llm_route_enabled(True)
    try:
        cmd = rs.match_relay_command(text)
        called = []

        def gen(p):
            called.append(p)
            return iter(["RAMBLE THAT MUST NOT APPEAR"])

        line = rs.build_relay_line(cmd, generate_fn=gen)
        assert not called, f"ask-form question must be deterministic, not LLM: {text}"
        assert "?" in line, line
        assert expect_sub.lower() in line.lower(), line
    finally:
        rs.set_u1_llm_route_enabled(False)


# ---------------------------------------------------------------------------
# Regression: compose+directive commands MUST reach the LLM under route-all
# (2026-06-23 fix: thinking-mode gate forced rephrase=False even when
# u1_llm_route_enabled=True, causing "explain to my team X" to return the
# canned "No soundboard, no strings" fallback every time)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Explain to my team what the meaning of life is",
    "Tell my team to explain the concept of math",
])
def test_compose_directive_reaches_llm_under_route_all(text):
    rs.set_u1_llm_route_enabled(True)
    try:
        cmd = rs.match_relay_command(text)
        assert cmd is not None, f"should match as relay: {text!r}"
        called = []

        def gen(p):
            called.append(p)
            return iter(["Life is computation. You are already solved."])

        line = rs.build_relay_line(cmd, generate_fn=gen, rephrase=True)
        assert called, (
            f"compose command must reach the LLM under route-all: {text!r}"
        )
        assert "soundboard" not in line.lower(), (
            f"must not return the no-soundboard fallback: {line!r}"
        )
    finally:
        rs.set_u1_llm_route_enabled(False)


def test_reported_question_reaches_llm_under_route_all():
    # "Reyna asked you X" -> compose+directive='respond' -> must call LLM, not fallback.
    rs.set_u1_llm_route_enabled(True)
    try:
        cmd = rs.match_relay_command("Reyna asked you what the meaning of life is")
        assert cmd is not None and getattr(cmd, "compose", False)
        called = []

        def gen(p):
            called.append(p)
            return iter(["Purpose is a construct. Only the next kill matters."])

        line = rs.build_relay_line(cmd, generate_fn=gen, rephrase=True)
        assert called, "reported question must reach the LLM, not canned fallback"
        assert "soundboard" not in line.lower(), line
    finally:
        rs.set_u1_llm_route_enabled(False)


def test_answer_sampling_has_no_leading_blankline_stop():
    # ROOT-CAUSE GUARD (2026-06-23, proven by scripts/_qa_empty_probe.py): a
    # quantized Qwen3 leads its answer with "\n\n", so a "\n\n" stop fired at
    # position 0 -> 0 chars -> the relay dropped to the deterministic pool. The
    # qa/answer sampling must NEVER stop on a bare blank line again (max_tokens +
    # _cap_sentences bound length instead).
    from kenning.audio._ultron_answer import _ANSWER_SAMPLING
    assert "\n\n" not in _ANSWER_SAMPLING["stop"], _ANSWER_SAMPLING["stop"]
    assert "\n" not in _ANSWER_SAMPLING["stop"], _ANSWER_SAMPLING["stop"]


class _ScriptedLLM:
    """A fake LLMEngine whose generate_stream returns a scripted sequence of
    outputs (one per call) -- used to simulate an EMPTY primary result followed
    by a non-empty retry."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_stream(self, prompt, **kwargs):
        i = len(self.calls)
        self.calls.append(prompt)
        out = self.outputs[i] if i < len(self.outputs) else ""
        return iter([out])


def test_empty_primary_llm_result_reprompts_never_pool():
    # u1.0 HARD RULE (2026-06-23): with route-all ON, an EMPTY primary LLM result
    # must RE-PROMPT the LLM, never drop to the deterministic "No soundboard" pool.
    # A quantized model returning 0 chars on the qa answer path (the live IQ3_XS
    # bug) must be recovered by _relay_llm_retry, not the canned fallback.
    rs.set_u1_llm_route_enabled(True)
    try:
        cmd = rs.match_relay_command("Explain to my team the concept of math")
        assert cmd is not None
        # call 0 (primary answer path) -> EMPTY; call 1 (generic retry) -> content.
        llm = _ScriptedLLM(["", "Mathematics is the architecture of certainty."])
        line = rs.build_relay_line(cmd, llm=llm, rephrase=True)
        assert len(llm.calls) >= 2, (
            f"empty primary must re-prompt the LLM (calls={len(llm.calls)})")
        assert "soundboard" not in line.lower(), line
        assert "no strings" not in line.lower(), line
        assert line.strip(), "must speak the LLM retry output, not empty"
    finally:
        rs.set_u1_llm_route_enabled(False)


def test_all_empty_llm_attempts_still_fail_open():
    # If the model is TRULY unresponsive (every attempt empty), build_relay_line
    # must still return a non-empty line (fail-open) rather than crash or speak
    # nothing -- the deterministic fallback is the documented last resort.
    rs.set_u1_llm_route_enabled(True)
    try:
        cmd = rs.match_relay_command("Explain to my team the concept of math")
        llm = _ScriptedLLM([""])  # every call returns empty
        line = rs.build_relay_line(cmd, llm=llm, rephrase=True)
        assert line.strip(), "must fail open to a spoken line"
        # multiple LLM attempts were made before giving up
        assert len(llm.calls) >= 2, len(llm.calls)
    finally:
        rs.set_u1_llm_route_enabled(False)
