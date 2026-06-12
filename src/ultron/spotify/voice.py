"""Spotify voice commands: strict matchers + dispatch -> spoken line.

Same discipline as the relay / scrap matchers: a tight regex set that
fires only on clear playback commands and never on ordinary chatter.
The orchestrator places the ``_maybe_handle_spotify`` short-circuit
AFTER run/launch + app-launch so "play the calculator" (a sandbox
program) and app launches win over "play <song>".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ultron.spotify.voice")

__all__ = ["SpotifyCommand", "match_spotify_command", "handle_spotify_command"]

# How much a single "turn it up/down" nudges the volume.
VOLUME_STEP = 15


@dataclass(frozen=True)
class SpotifyCommand:
    """A parsed playback command.

    action: play / pause / resume / next / previous / now_playing /
        volume_up / volume_down / volume_set / shuffle / repeat / queue.
    argument: the search query for play/queue.
    kind: track / artist / album / playlist (for play).
    value: numeric payload (volume percent; shuffle/repeat bool-as-int).
    """

    action: str
    argument: str = ""
    kind: str = "track"
    value: int = 0


_SPOTIFY = r"(?:on\s+)?spotify"

# Transport commands that are unambiguous without naming Spotify.
_PAUSE = re.compile(
    r"^(?:please\s+)?(?:pause|stop)\s+(?:the\s+)?(?:music|song|playback|"
    rf"spotify)\b.*$|^(?:please\s+)?pause\s+{_SPOTIFY}?\s*[.!?]?$",
    re.IGNORECASE,
)
_PAUSE_BARE = re.compile(r"^(?:please\s+)?(?:pause|pause it)\s*[.!?]?$",
                         re.IGNORECASE)
_RESUME = re.compile(
    r"^(?:please\s+)?(?:resume|unpause|keep playing|continue)\s+"
    rf"(?:the\s+)?(?:music|song|playback|{_SPOTIFY})\s*[.!?]?$"
    r"|^(?:please\s+)?(?:resume|unpause|keep playing)\s*[.!?]?$",
    re.IGNORECASE,
)
_NEXT = re.compile(
    r"^(?:please\s+)?(?:next|skip)(?:\s+(?:song|track|this(?:\s+song)?|"
    rf"it|ahead|{_SPOTIFY}))?\s*[.!?]?$",
    re.IGNORECASE,
)
_PREV = re.compile(
    r"^(?:please\s+)?(?:previous(?:\s+(?:song|track))?|go\s+back"
    r"|last\s+song|play\s+the\s+(?:previous|last)\s+(?:song|track))\s*[.!?]?$",
    re.IGNORECASE,
)
_NOW = re.compile(
    r"^(?:please\s+)?(?:what(?:'s|\s+is)\s+(?:playing|this\s+song|this)"
    r"|what\s+song\s+is\s+this|who\s+(?:sings|is\s+playing)\s+this"
    r"|name\s+(?:this|the)\s+song)\s*[.!?]?$",
    re.IGNORECASE,
)
_VOL_UP = re.compile(
    r"^(?:please\s+)?(?:turn\s+(?:it|the\s+(?:music|volume))\s+up"
    r"|volume\s+up|louder|crank\s+it(?:\s+up)?)\s*[.!?]?$",
    re.IGNORECASE,
)
_VOL_DOWN = re.compile(
    r"^(?:please\s+)?(?:turn\s+(?:it|the\s+(?:music|volume))\s+down"
    r"|volume\s+down|quieter|lower\s+(?:it|the\s+volume))\s*[.!?]?$",
    re.IGNORECASE,
)
_VOL_SET = re.compile(
    r"^(?:please\s+)?set\s+(?:the\s+)?volume\s+to\s+(?P<n>\d{1,3})\s*"
    r"(?:percent|%)?\s*[.!?]?$",
    re.IGNORECASE,
)
_SHUFFLE = re.compile(
    r"^(?:please\s+)?(?:(?:turn\s+)?shuffle\s+(?P<on1>on|off)"
    r"|turn\s+(?P<on2>on|off)\s+shuffle"
    r"|(?P<on3>enable|disable)\s+shuffle"
    r"|shuffle\s+(?:my\s+)?(?:music|playback))\s*[.!?]?$",
    re.IGNORECASE,
)
_REPEAT = re.compile(
    r"^(?:please\s+)?(?:repeat\s+(?P<what>this(?:\s+song)?|track|off)"
    r"|(?:turn\s+)?repeat\s+(?P<onoff>on|off))\s*[.!?]?$",
    re.IGNORECASE,
)
_QUEUE = re.compile(
    r"^(?:please\s+)?(?:queue(?:\s+up)?|add)\s+(?P<q>.+?)"
    r"(?:\s+to\s+(?:the\s+)?queue)?\s*[.!?]?$",
    re.IGNORECASE,
)
# Play: optional framing word selects the kind; "<X> by <Y>" folds the
# artist into the search query; a trailing "on spotify" is stripped.
_PLAY = re.compile(
    r"^(?:please\s+)?play\s+(?:me\s+)?"
    r"(?P<frame>some\s+|the\s+song\s+|the\s+album\s+|the\s+playlist\s+"
    r"|my\s+|the\s+artist\s+)?"
    r"(?P<q>.+?)"
    r"(?:\s+playlist)?"
    rf"(?:\s+{_SPOTIFY})?\s*[.!?]?$",
    re.IGNORECASE,
)


def match_spotify_command(text: str) -> Optional[SpotifyCommand]:
    """Match a strict Spotify playback command, else None."""
    if not text:
        return None
    t = text.strip()
    if _PAUSE.match(t) or _PAUSE_BARE.match(t):
        return SpotifyCommand("pause")
    if _RESUME.match(t):
        return SpotifyCommand("resume")
    if _NEXT.match(t):
        return SpotifyCommand("next")
    if _PREV.match(t):
        return SpotifyCommand("previous")
    if _NOW.match(t):
        return SpotifyCommand("now_playing")
    if _VOL_UP.match(t):
        return SpotifyCommand("volume_up")
    if _VOL_DOWN.match(t):
        return SpotifyCommand("volume_down")
    m = _VOL_SET.match(t)
    if m:
        return SpotifyCommand("volume_set", value=int(m.group("n")))
    m = _SHUFFLE.match(t)
    if m:
        word = (m.group("on1") or m.group("on2") or m.group("on3")
                or "on").lower()
        on = word in ("on", "enable")
        return SpotifyCommand("shuffle", value=1 if on else 0)
    m = _REPEAT.match(t)
    if m:
        what = (m.group("what") or m.group("onoff") or "").lower()
        off = what == "off"
        return SpotifyCommand("repeat", value=0 if off else 1)
    m = _QUEUE.match(t)
    if m and not _PLAY.match(t):  # "add X" handled here; "play X" below
        q = m.group("q").strip()
        if len(q.split()) >= 1 and q.lower() not in ("it", "this"):
            return SpotifyCommand("queue", argument=q)
    m = _PLAY.match(t)
    if m:
        frame = (m.group("frame") or "").strip().lower()
        q = m.group("q").strip().strip('"')
        kind = "track"
        if frame in ("some", "the artist"):
            kind = "artist"
        elif frame == "the album":
            kind = "album"
        elif frame in ("the playlist", "my") or t.lower().rstrip(
                ".!?").endswith("playlist"):
            kind = "playlist"
        if not q or q.lower() in ("music", "something", "it"):
            # "play music" / bare resume.
            return SpotifyCommand("resume")
        return SpotifyCommand("play", argument=q, kind=kind)
    return None


def handle_spotify_command(command: SpotifyCommand, client) -> str:
    """Execute ``command`` against a :class:`SpotifyClient`; return a
    spoken line. Fail-soft: API errors become a short spoken message."""
    from ultron.spotify.client import SpotifyAPIError
    from ultron.spotify.auth import SpotifyAuthError

    try:
        action = command.action
        if action == "play":
            return client.play_query(command.argument, command.kind)
        if action == "queue":
            return client.queue_query(command.argument)
        if action == "pause":
            client.pause()
            return "Paused."
        if action == "resume":
            client.resume()
            return "Resuming."
        if action == "next":
            client.next_track()
            return "Skipping ahead."
        if action == "previous":
            client.previous_track()
            return "Going back."
        if action == "now_playing":
            return client.now_playing().spoken()
        if action in ("volume_up", "volume_down"):
            cur = client.current_volume()
            cur = 50 if cur is None else cur
            step = VOLUME_STEP if action == "volume_up" else -VOLUME_STEP
            new = max(0, min(100, cur + step))
            client.set_volume(new)
            return f"Volume {new} percent."
        if action == "volume_set":
            client.set_volume(command.value)
            return f"Volume set to {max(0, min(100, command.value))} percent."
        if action == "shuffle":
            client.set_shuffle(bool(command.value))
            return "Shuffle on." if command.value else "Shuffle off."
        if action == "repeat":
            client.set_repeat("track" if command.value else "off")
            return "Repeating this song." if command.value else "Repeat off."
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
