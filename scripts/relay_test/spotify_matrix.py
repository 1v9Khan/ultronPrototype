"""Exhaustive Spotify voice-command matrix: every action x many phrasings,
value checks (volume/shuffle/repeat), and negative cases (chatter + relay must
NOT route to Spotify). Run from repo root with the main venv."""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from kenning.spotify.voice import match_spotify_command as M

# (text, expected_action, expected_value_or_None)
CASES = [
    # --- now playing ---
    ("what's playing", "now_playing", None),
    ("what song is this", "now_playing", None),
    ("what is this song called", "now_playing", None),
    ("who is this", "now_playing", None),
    ("who sings this", "now_playing", None),
    ("name this song", "now_playing", None),
    ("what am i listening to", "now_playing", None),
    ("song name", "now_playing", None),
    ("what track is this", "now_playing", None),
    ("current song", "now_playing", None),
    # --- play (track/artist/album/playlist) ---
    ("play californication", "play", None),
    ("play some daft punk", "play", None),
    ("put on bohemian rhapsody", "play", None),
    ("throw on some jazz", "play", None),
    ("start playing radiohead", "play", None),
    ("blast master of puppets", "play", None),
    ("i want to hear hotel california", "play", None),
    ("i wanna hear the weeknd", "play", None),
    ("let's hear some metallica", "play", None),
    ("listen to pink floyd", "play", None),
    ("play the album discovery", "play", None),
    ("play the playlist focus flow", "play", None),
    ("play my liked songs playlist", "play", None),
    ("play the song wonderwall", "play", None),
    ("spin up some lofi", "play", None),
    # --- play X next  -> queue ---
    ("play californication next", "queue", None),
    ("play smells like teen spirit next", "queue", None),
    # --- queue ---
    ("queue despacito", "queue", None),
    ("queue up enter sandman", "queue", None),
    ("add toxicity to the queue", "queue", None),
    ("throw paranoid in the queue", "queue", None),
    ("line up some music", "queue", None),
    ("add hysteria to up next", "queue", None),
    ("stick another one in the queue", "queue", None),
    # --- pause ---
    ("pause", "pause", None),
    ("pause the music", "pause", None),
    ("stop", "pause", None),
    ("stop the music", "pause", None),
    ("halt", "pause", None),
    ("freeze the music", "pause", None),
    ("hold the music", "pause", None),
    ("pause spotify", "pause", None),
    # --- resume ---
    ("resume", "resume", None),
    ("play", "resume", None),
    ("play music", "resume", None),
    ("play the music", "resume", None),
    ("unpause", "resume", None),
    ("continue", "resume", None),
    ("keep playing", "resume", None),
    ("carry on", "resume", None),
    ("hit play", "resume", None),
    ("press play", "resume", None),
    ("back on", "resume", None),
    ("put it back on", "resume", None),
    ("resume spotify", "resume", None),
    # --- next ---
    ("next", "next", None),
    ("skip", "next", None),
    ("next song", "next", None),
    ("next track", "next", None),
    ("skip this", "next", None),
    ("skip this song", "next", None),
    ("play the next song", "next", None),
    ("change the song", "next", None),
    ("another song", "next", None),
    ("a different song", "next", None),
    ("i don't like this song", "next", None),
    ("skip ahead", "next", None),
    ("rewind", "previous", None),  # rewind -> prev
    # --- previous ---
    ("previous", "previous", None),
    ("previous song", "previous", None),
    ("go back", "previous", None),
    ("go back a song", "previous", None),
    ("last song", "previous", None),
    ("the last track", "previous", None),
    ("play the previous song", "previous", None),
    ("back a track", "previous", None),
    ("the song before", "previous", None),
    # --- restart ---
    ("restart", "restart", None),
    ("restart the song", "restart", None),
    ("start it over", "restart", None),
    ("start over", "restart", None),
    ("play it from the beginning", "restart", None),
    ("from the top", "restart", None),
    ("replay this song", "restart", None),
    ("replay it", "restart", None),
    ("go back to the beginning", "restart", None),
    # --- volume set ---
    ("set the volume to 50", "volume_set", 50),
    ("set volume at 30", "volume_set", 30),
    ("put the volume at 40", "volume_set", 40),
    ("make the volume 40", "volume_set", 40),
    ("turn the volume to 80", "volume_set", 80),
    ("volume 70", "volume_set", 70),
    ("volume to 30", "volume_set", 30),
    ("volume at 55", "volume_set", 55),
    ("set the volume to 100 percent", "volume_set", 100),
    # --- volume up ---
    ("turn it up", "volume_up", None),
    ("turn the volume up", "volume_up", None),
    ("turn it up a bit", "volume_up", None),
    ("volume up", "volume_up", None),
    ("louder", "volume_up", None),
    ("crank it", "volume_up", None),
    ("crank it up", "volume_up", None),
    ("bump it up", "volume_up", None),
    ("make it louder", "volume_up", None),
    ("a little louder", "volume_up", None),
    ("raise the volume", "volume_up", None),
    ("more volume", "volume_up", None),
    # --- volume down ---
    ("turn it down", "volume_down", None),
    ("turn the volume down", "volume_down", None),
    ("volume down", "volume_down", None),
    ("quieter", "volume_down", None),
    ("softer", "volume_down", None),
    ("lower it", "volume_down", None),
    ("lower the volume", "volume_down", None),
    ("bring it down", "volume_down", None),
    ("make it quieter", "volume_down", None),
    ("a bit softer", "volume_down", None),
    ("less volume", "volume_down", None),
    # --- mute / unmute ---
    ("mute", "mute", None),
    ("mute it", "mute", None),
    ("mute the music", "mute", None),
    ("silence the music", "mute", None),
    ("kill the sound", "mute", None),
    ("unmute", "unmute", None),
    ("unmute it", "unmute", None),
    ("restore the volume", "unmute", None),
    ("turn the sound back on", "unmute", None),
    ("bring the volume back", "unmute", None),
    # --- shuffle ---
    ("shuffle", "shuffle", 1),
    ("shuffle on", "shuffle", 1),
    ("turn on shuffle", "shuffle", 1),
    ("turn shuffle on", "shuffle", 1),
    ("enable shuffle", "shuffle", 1),
    ("shuffle my music", "shuffle", 1),
    ("randomize it", "shuffle", 1),
    ("mix it up", "shuffle", 1),
    ("shuffle off", "shuffle", 0),
    ("turn off shuffle", "shuffle", 0),
    ("turn shuffle off", "shuffle", 0),
    ("disable shuffle", "shuffle", 0),
    ("stop shuffling", "shuffle", 0),
    # --- repeat ---
    ("repeat", "repeat", 1),
    ("repeat this", "repeat", 1),
    ("repeat this song", "repeat", 1),
    ("repeat on", "repeat", 1),
    ("turn on repeat", "repeat", 1),
    ("loop this", "repeat", 1),
    ("loop this song", "repeat", 1),
    ("put it on repeat", "repeat", 1),
    ("play this on repeat", "repeat", 1),
    ("repeat off", "repeat", 0),
    ("turn off repeat", "repeat", 0),
    ("disable repeat", "repeat", 0),
    ("stop repeating", "repeat", 0),
    ("stop looping", "repeat", 0),
    # --- like / unlike ---
    ("like this song", "like", None),
    ("like it", "like", None),
    ("save this song", "like", None),
    ("save this track", "like", None),
    ("heart this", "like", None),
    ("favorite this", "like", None),
    ("favourite this song", "like", None),
    ("i love this song", "like", None),
    ("thumbs up", "like", None),
    ("add this to my liked songs", "like", None),
    ("add it to my library", "like", None),
    ("add to my liked songs", "like", None),
    ("unlike this song", "unlike", None),
    ("unlike it", "unlike", None),
    ("unsave this", "unlike", None),
    ("remove this from my liked songs", "unlike", None),
    ("take it off my library", "unlike", None),
    ("thumbs down", "unlike", None),
    # --- NEGATIVE: must NOT route to spotify (None) ---
    ("tell my team to rotate", None, None),
    ("rotate", None, None),
    ("they have ult", None, None),
    ("nice shot", None, None),
    ("what's the score", None, None),
    ("how are you", None, None),
    ("play the calculator", "play", None),  # NOTE: orchestrator gates app-launch BEFORE spotify
    ("ultron repeat to my team go b", None, None),
    ("what time is it", None, None),
    ("open spotify", None, None),
]


def main():
    bad = 0
    by_action = {}
    for t, exp_a, exp_v in CASES:
        c = M(t)
        got_a = c.action if c else None
        got_v = c.value if c else None
        ok = got_a == exp_a and (exp_v is None or got_v == exp_v)
        by_action.setdefault(exp_a, [0, 0])
        by_action[exp_a][1] += 1
        if ok:
            by_action[exp_a][0] += 1
        else:
            bad += 1
            extra = ""
            if exp_v is not None:
                extra = f"  (val exp={exp_v} got={got_v})"
            print(f"  MISMATCH  {t!r:42}  exp={exp_a}  got={got_a}{extra}")
    print("\n--- per action (pass/total) ---")
    for a in sorted(by_action, key=lambda x: (x is None, x)):
        p, tot = by_action[a]
        flag = "" if p == tot else "  <-- FAIL"
        print(f"  {str(a):14} {p}/{tot}{flag}")
    print(f"\nTOTAL: {len(CASES) - bad}/{len(CASES)} correct, {bad} mismatches")


if __name__ == "__main__":
    main()
