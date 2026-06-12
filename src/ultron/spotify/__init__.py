"""Spotify control for Ultron.

Voice-driven playback control via the Spotify Web API:

    auth.py   -- credential loading (gitignored ~/.ultron/spotify.json),
                 OAuth authorization-code flow + refresh-token caching.
    client.py -- the Web API wrapper (play / pause / skip / search-and-
                 play / queue / volume / devices / now-playing).
    voice.py  -- strict command matchers + dispatch -> spoken response.

The orchestrator wires a ``_maybe_handle_spotify`` short-circuit (same
pattern as the relay / scrap matchers). Credentials never enter the
repo: they live in a gitignored file outside the tree, referenced by
``spotify.credentials_path``. Playback control requires Spotify Premium
and a one-time browser authorization via ``scripts/spotify_setup.py``.
"""
