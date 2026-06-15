"""Spotify voice commands: comprehensive matchers + dispatch -> spoken line.

Same discipline as the relay / scrap matchers: a wide-but-strict regex set
that understands the many natural ways to phrase a playback command, yet only
fires on clear music control and never on ordinary chatter. The orchestrator
places the ``_maybe_handle_spotify`` short-circuit AFTER run/launch + app-launch
so "play the calculator" (a sandbox program) wins over "play <song>".

Actions: play / queue / pause / resume / next / previous / restart /
now_playing / volume_up / volume_down / volume_set / mute / unmute / shuffle /
repeat / like / unlike. Replies are in Ultron's cold machine register, varied.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("kenning.spotify.voice")

__all__ = ["SpotifyCommand", "match_spotify_command", "handle_spotify_command"]

# How much a single "turn it up/down" nudges the volume.
VOLUME_STEP = 15

_SP = r"(?:on\s+)?spotify"
# Transport objects: "...it / that / the music / the song / playing".
_OBJ = r"(?:it|that|the\s+(?:music|song|playback|track|tune|audio)|playing)"


@dataclass(frozen=True)
class SpotifyCommand:
    """A parsed playback command.

    action: play / queue / pause / resume / next / previous / restart /
        now_playing / volume_up / volume_down / volume_set / mute / unmute /
        shuffle / repeat / like / unlike.
    argument: the search query for play/queue.
    kind: track / artist / album / playlist (for play).
    value: numeric payload (volume percent; shuffle/repeat bool-as-int).
    """

    action: str
    argument: str = ""
    kind: str = "track"
    value: int = 0


_NOW = re.compile(
    r"^(?:please\s+)?(?:hey\s+)?(?:"
    r"what(?:'?s|\s+is)\s+(?:playing|this(?:\s+song)?|the\s+(?:song|track))"
    r"|what\s+song\s+is\s+(?:this|playing)|what(?:'?s|\s+is)\s+this\s+(?:(?:song|track)(?:\s+called)?|called)"
    r"|who(?:'?s|\s+is)\s+(?:this|playing|singing)(?:\s+this)?|who\s+sings\s+this"
    r"|name\s+(?:this|the)\s+(?:song|track)|what\s+am\s+i\s+listening\s+to"
    r"|tell\s+me\s+what(?:'?s|\s+is)\s+playing|(?:the\s+)?(?:current|this)\s+song"
    r"|song\s+name|what\s+track\s+is\s+this"
    r")\s*[.!?]*$",
    re.IGNORECASE,
)
_MUTE = re.compile(
    r"^(?:please\s+)?(?:"
    r"mute(?:\s+(?:it|that|the\s+(?:music|volume|sound|audio)|spotify))?"
    r"|silence\s+the\s+(?:music|audio|sound)|kill\s+the\s+(?:volume|sound|audio|music)"
    r")\s*[.!?]*$",
    re.IGNORECASE,
)
_UNMUTE = re.compile(
    r"^(?:please\s+)?(?:"
    r"unmute(?:\s+(?:it|that|the\s+(?:music|volume|sound)|spotify))?"
    r"|restore\s+the\s+(?:volume|sound|audio)"
    r"|(?:bring|turn)\s+the\s+(?:sound|volume|music|audio)\s+back(?:\s+on)?"
    r")\s*[.!?]*$",
    re.IGNORECASE,
)
_PAUSE = re.compile(
    rf"^(?:please\s+)?(?:pause|stop|halt|freeze|hold)(?:\s+(?:{_OBJ}|{_SP}))?\s*[.!?]*$",
    re.IGNORECASE,
)
_RESUME = re.compile(
    r"^(?:please\s+)?(?:resume|unpause|un-pause|continue|keep\s+(?:playing|going)"
    rf"|carry\s+on|(?:hit|press)\s+play|(?:put\s+it\s+)?back\s+on|play)(?:\s+(?:{_OBJ}|{_SP}))?"
    r"\s*[.!?]*$",
    re.IGNORECASE,
)
_RESTART = re.compile(
    r"^(?:please\s+)?(?:restart(?:\s+(?:the\s+)?(?:song|track|it))?"
    r"|start\s+(?:it\s+)?over|(?:play\s+)?(?:it\s+)?from\s+the\s+(?:beginning|start|top)"
    r"|replay\s+(?:this(?:\s+(?:song|track))?|the\s+(?:song|track)|it)"
    r"|(?:go\s+)?back\s+to\s+the\s+(?:beginning|start))\s*[.!?]*$",
    re.IGNORECASE,
)
_PREV = re.compile(
    r"^(?:please\s+)?(?:previous(?:\s+(?:song|track|one))?"
    r"|go\s+back(?:\s+(?:a\s+(?:song|track)|one))?|(?:the\s+)?last\s+(?:song|track|one)"
    r"|(?:play\s+)?the\s+(?:previous|last)\s+(?:song|track)|back\s+(?:a\s+(?:song|track)|one)"
    r"|rewind|the\s+(?:song|one)\s+before)\s*[.!?]*$",
    re.IGNORECASE,
)
_NEXT = re.compile(
    rf"^(?:please\s+)?(?:next|skip|(?:play\s+)?(?:the\s+)?next)(?:\s+(?:song|track|one|tune|it|ahead|forward|this(?:\s+(?:song|one))?|{_SP}))?\s*[.!?]*$"
    r"|^(?:please\s+)?(?:change|skip)\s+(?:the\s+)?(?:song|track|tune|it|this)\s*[.!?]*$"
    r"|^(?:please\s+)?(?:another|a\s+different|different)\s+(?:song|track|tune|one)\s*[.!?]*$"
    r"|^(?:please\s+)?(?:i\s+)?(?:don'?t|do\s+not)\s+like\s+this(?:\s+(?:song|one))?\s*[.!?]*$",
    re.IGNORECASE,
)
_VOL_SET = re.compile(
    r"^(?:please\s+)?(?:"
    r"set\s+(?:the\s+)?volume\s+(?:to|at)\s+"
    r"|(?:put|make|turn)\s+(?:the\s+)?volume\s+(?:to\s+|at\s+)?"
    r"|volume\s+(?:to\s+|at\s+)?"
    r")(?P<n>\d{1,3})\s*(?:percent|%)?\s*[.!?]*$",
    re.IGNORECASE,
)
_VOL_UP = re.compile(
    r"^(?:please\s+)?(?:turn\s+(?:it|the\s+(?:music|volume|sound))\s+up(?:\s+a\s+(?:bit|little))?"
    r"|volume\s+up|louder|crank\s+(?:it|the\s+(?:volume|music))(?:\s+up)?"
    r"|(?:bump|pump|jack)\s+(?:it|the\s+volume)\s+up|make\s+it\s+louder|a\s+(?:bit|little)\s+louder"
    r"|(?:raise|increase|boost)\s+(?:the\s+)?volume|more\s+volume)\s*[.!?]*$",
    re.IGNORECASE,
)
_VOL_DOWN = re.compile(
    r"^(?:please\s+)?(?:turn\s+(?:it|the\s+(?:music|volume|sound))\s+down(?:\s+a\s+(?:bit|little))?"
    r"|volume\s+down|quieter|softer|lower\s+(?:it|the\s+volume)"
    r"|(?:bring|drop|knock)\s+(?:it|the\s+volume)\s+down|make\s+it\s+(?:quieter|softer)"
    r"|a\s+(?:bit|little)\s+(?:quieter|softer)|(?:decrease|lower|drop)\s+(?:the\s+)?volume|less\s+volume)\s*[.!?]*$",
    re.IGNORECASE,
)
_SHUFFLE = re.compile(
    r"^(?:please\s+)?(?:(?:turn\s+)?shuffle\s+(?P<on1>on|off)|turn\s+(?P<on2>on|off)\s+shuffle"
    r"|(?P<on3>enable|disable|start|stop)\s+shuffl(?:e|ing)"
    r"|shuffle(?:\s+(?:my\s+)?(?:music|playback|playlist|songs|it|everything))?"
    r"|(?:put\s+(?:it|the\s+music)\s+on|turn\s+on)\s+shuffle|randomi[sz]e(?:\s+(?:it|the\s+(?:music|playlist)))?"
    r"|mix\s+it\s+up)\s*[.!?]*$",
    re.IGNORECASE,
)
_REPEAT = re.compile(
    r"^(?:please\s+)?(?:repeat\s+(?P<what>this(?:\s+(?:song|track))?|the\s+(?:song|track)|it|on|off)"
    r"|(?:turn\s+)?repeat\s+(?P<onoff>on|off)|turn\s+(?P<onoff2>on|off)\s+repeat"
    r"|(?:enable|disable)\s+repeat|loop\s+(?P<loop>this(?:\s+(?:song|track))?|the\s+(?:song|track)|it|on|off)"
    r"|(?:put|play)\s+(?:it|this)\s+on\s+repeat|stop\s+(?:repeating|looping)|repeat)\s*[.!?]*$",
    re.IGNORECASE,
)
_LIKE = re.compile(
    r"^(?:please\s+)?(?:(?:like|save|heart|favou?rite)\s+(?:this(?:\s+(?:song|track))?|it|the\s+(?:song|track))"
    r"|(?:add|save)\s+(?:this(?:\s+(?:song|track))?|it)\s+to\s+(?:my\s+)?(?:liked\s+songs|library|favou?rites)"
    r"|(?:i\s+)?(?:like|love)\s+this(?:\s+song)?|thumbs\s+up|add\s+to\s+(?:my\s+)?(?:liked\s+songs|library))\s*[.!?]*$",
    re.IGNORECASE,
)
_UNLIKE = re.compile(
    r"^(?:please\s+)?(?:(?:unlike|unsave|unfavou?rite)(?:\s+(?:this(?:\s+(?:song|track))?|it))?"
    r"|remove\s+(?:this(?:\s+(?:song|track))?|it)\s+from\s+(?:my\s+)?(?:liked\s+songs|library|favou?rites)"
    r"|take\s+(?:this|it)\s+off\s+(?:my\s+)?(?:liked\s+songs|library)|thumbs\s+down)\s*[.!?]*$",
    re.IGNORECASE,
)
_QUEUE = re.compile(
    r"^(?:please\s+)?(?:queue(?:\s+up)?|add|throw(?!\s+on\b)|line\s+up|stick)\s+(?P<q>.+?)"
    r"(?:\s+(?:to|in|into|in\s+to)\s+(?:the\s+)?(?:queue|up\s+next))?\s*[.!?]*$",
    re.IGNORECASE,
)
# "play <X> next" = queue X after the current track (distinct from "play X").
_PLAY_NEXT = re.compile(
    r"^(?:please\s+)?play\s+(?P<q>.+?)\s+next\s*[.!?]*$", re.IGNORECASE)
# Play: lead-in verb + optional framing word (kind) + query; trailing
# "on spotify" / "on repeat|shuffle" / "playlist" stripped.
_PLAY = re.compile(
    r"^(?:please\s+)?(?:play|put\s+on|throw\s+on|start\s+playing|blast|spin\s+up"
    r"|i\s+(?:want|wanna)\s+(?:to\s+)?hear|let'?s\s+hear|let'?s\s+listen\s+to|listen\s+to)\s+(?:me\s+)?"
    r"(?P<frame>some\s+|the\s+song\s+|the\s+album\s+|the\s+playlist\s+|my\s+|the\s+artist\s+|artist\s+)?"
    r"(?P<q>.+?)"
    rf"(?:\s+(?:on\s+)?(?:repeat|shuffle))?(?:\s+playlist)?(?:\s+{_SP})?\s*[.!?]*$",
    re.IGNORECASE,
)

_RESUME_WORDS = {"music", "the music", "something", "anything", "it", "that", ""}


def match_spotify_command(text: str) -> Optional[SpotifyCommand]:
    """Match a Spotify playback command (wide phrasing), else None."""
    if not text:
        return None
    t = text.strip()
    if _NOW.match(t):
        return SpotifyCommand("now_playing")
    if _MUTE.match(t):
        return SpotifyCommand("mute")
    if _UNMUTE.match(t):
        return SpotifyCommand("unmute")
    if _PAUSE.match(t):
        return SpotifyCommand("pause")
    if _RESUME.match(t):
        return SpotifyCommand("resume")
    if _RESTART.match(t):
        return SpotifyCommand("restart")
    if _PREV.match(t):
        return SpotifyCommand("previous")
    if _NEXT.match(t):
        return SpotifyCommand("next")
    m = _VOL_SET.match(t)
    if m:
        return SpotifyCommand("volume_set", value=int(m.group("n")))
    if _VOL_UP.match(t):
        return SpotifyCommand("volume_up")
    if _VOL_DOWN.match(t):
        return SpotifyCommand("volume_down")
    m = _SHUFFLE.match(t)
    if m:
        word = (m.group("on1") or m.group("on2") or m.group("on3") or "on").lower()
        on = word not in ("off", "disable", "stop")
        return SpotifyCommand("shuffle", value=1 if on else 0)
    m = _REPEAT.match(t)
    if m:
        what = (m.group("what") or m.group("onoff") or m.group("onoff2")
                or m.group("loop") or "").lower()
        low = t.lower()
        off = what == "off" or "stop repeating" in low or "stop looping" in low \
            or "disable repeat" in low
        return SpotifyCommand("repeat", value=0 if off else 1)
    if _LIKE.match(t):
        return SpotifyCommand("like")
    if _UNLIKE.match(t):
        return SpotifyCommand("unlike")
    m = _PLAY_NEXT.match(t)               # "play <X> next" -> queue
    if m:
        q = m.group("q").strip().strip('"')
        if q and q.lower() not in ("it", "this", "that"):
            return SpotifyCommand("queue", argument=q)
    m = _QUEUE.match(t)                   # "queue <X>" / "add <X> to the queue"
    if m:
        q = (m.group("q") or "").strip().strip('"')
        if q and q.lower() not in ("it", "this", "that"):
            return SpotifyCommand("queue", argument=q)
    m = _PLAY.match(t)
    if m:
        frame = (m.group("frame") or "").strip().lower()
        q = m.group("q").strip().strip('"')
        kind = "track"
        if frame in ("some", "the artist", "artist"):
            kind = "artist"
        elif frame == "the album":
            kind = "album"
        elif frame in ("the playlist", "my") or t.lower().rstrip(".!?").endswith("playlist"):
            kind = "playlist"
        if not q or q.lower() in _RESUME_WORDS:
            return SpotifyCommand("resume")          # "play" / "play the music"
        return SpotifyCommand("play", argument=q, kind=kind)
    return None


# Ultron-voiced confirmations -- cold, brief, varied (the machine acknowledging
# the order). Dynamic content (track name, volume) is built in the handler.
_REPLIES = {
    "pause": ("The music halts.", "Silenced.", "Paused.", "Music stopped."),
    "resume": ("The music resumes.", "Playback restored.", "Continuing.", "Back on."),
    "next": ("Skipping it.", "On to the next.", "Next track.", "That one is dismissed."),
    "previous": ("Back a track.", "The previous one.", "Returning to it."),
    "restart": ("Back to the start.", "Restarting the track.", "From the beginning."),
    "mute": ("Muted.", "Silence.", "Volume cut."),
    "shuffle_on": ("Shuffle on.", "Order abandoned.", "Randomized."),
    "shuffle_off": ("Shuffle off.", "Order restored."),
    "repeat_on": ("Looping this track.", "Repeat on.", "This one, on repeat."),
    "repeat_off": ("Repeat off.", "The loop is broken."),
}


def _r(key: str) -> str:
    return random.choice(_REPLIES[key])


def handle_spotify_command(command: SpotifyCommand, client) -> str:
    """Execute ``command`` against a :class:`SpotifyClient`; return a spoken
    line in Ultron's voice. Fail-soft: API errors become a short message."""
    from kenning.spotify.client import SpotifyAPIError
    from kenning.spotify.auth import SpotifyAuthError

    try:
        action = command.action
        if action == "play":
            return client.play_query(command.argument, command.kind)
        if action == "queue":
            return client.queue_query(command.argument)
        if action == "pause":
            client.pause()
            return _r("pause")
        if action == "resume":
            client.resume()
            return _r("resume")
        if action == "next":
            client.next_track()
            return _r("next")
        if action == "previous":
            client.previous_track()
            return _r("previous")
        if action == "restart":
            client.seek(0)
            return _r("restart")
        if action == "now_playing":
            return client.now_playing().spoken()
        if action == "mute":
            cur = client.current_volume()
            client._premute_vol = cur if cur else 50      # noqa: SLF001
            client.set_volume(0)
            return _r("mute")
        if action == "unmute":
            vol = int(getattr(client, "_premute_vol", 0) or 0) or 50
            client.set_volume(vol)
            return f"Volume restored to {vol} percent."
        if action in ("volume_up", "volume_down"):
            cur = client.current_volume()
            cur = 50 if cur is None else cur
            step = VOLUME_STEP if action == "volume_up" else -VOLUME_STEP
            new = max(0, min(100, cur + step))
            client.set_volume(new)
            return f"Volume at {new} percent."
        if action == "volume_set":
            client.set_volume(command.value)
            return f"Volume at {max(0, min(100, command.value))} percent."
        if action == "shuffle":
            client.set_shuffle(bool(command.value))
            return _r("shuffle_on") if command.value else _r("shuffle_off")
        if action == "repeat":
            client.set_repeat("track" if command.value else "off")
            return _r("repeat_on") if command.value else _r("repeat_off")
        if action == "like":
            return client.save_current_track()
        if action == "unlike":
            return client.unsave_current_track()
    except SpotifyAuthError:
        return ("Spotify isn't connected yet. Run the Spotify setup "
                "once to authorize me.")
    except SpotifyAPIError as e:
        logger.warning("spotify command failed: %s", e)
        if "active" in str(e).lower() or "404" in str(e):
            return "Open Spotify on a device first, then ask me again."
        return "Spotify didn't take that one."
    except Exception as e:  # noqa: BLE001
        logger.warning("spotify command error: %s", e)
        return "Something went wrong with Spotify."
    return "I didn't catch that Spotify command."
