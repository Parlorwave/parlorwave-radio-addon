#!/usr/bin/env python3

import json
import os
import random
import re
import sys
from collections import defaultdict

import library_repo
import providers
from musicdb import artist_signals, get_scores_for_artists, holiday_allowed

# HTTP mechanics + the authoritative endpoint/model defaults live in
# providers.py; MODEL stays a module attribute for existing patch points.
MODEL = providers.OLLAMA_MODEL
COUNT = 30

# Max artists injected into the /station LLM prompt. Bounds prompt size so it does
# not grow with the library; at/under this count every artist is included, so
# behavior is unchanged for small libraries.
STATION_MAX_ARTISTS = int(os.environ.get("STATION_MAX_ARTISTS", "75"))

# Timeout for the tiny session-mode interpretation call. Deliberately far
# below the station call's 180s: mode mapping is optional and must fail fast.
MODE_LLM_TIMEOUT_S = float(os.environ.get("PLAYBACK_MODE_LLM_TIMEOUT_S", "5"))


def interpret_mode(utterance: str, modes: list[str],
                   timeout: float | None = None) -> str | None:
    """Map an utterance onto ONE of the given mode keys via a bounded LLM call
    (session mode selection). Returns the raw mode string or None; NEVER raises --
    Ollama down/slow/invalid output all degrade to None so the caller's keyword
    fallback takes over (Principle 6). The prompt contains only the utterance and
    the closed vocabulary; the caller validates the answer against it again."""
    if not (utterance or "").strip():
        return None
    prompt = f"""Classify a music listening request into exactly one mode.

Request: {utterance.strip()}

Modes: {", ".join(modes)}

Return JSON only: {{"mode": "<one of the modes>"}}
If none clearly fits, use "default_adaptive"."""
    try:
        resp = providers.ollama_generate(prompt, MODEL,
                                         timeout=timeout or MODE_LLM_TIMEOUT_S,
                                         format="json")
        return str(json.loads(resp["response"]).get("mode") or "") or None
    except Exception:
        return None


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def bounded_artists(artists, request, n=STATION_MAX_ARTISTS, signals_fn=artist_signals):
    """Return the artist names to show the LLM, capped at n by cheap grounded
    signals (prompt/genre token match, affinity, recency). At or under n, every
    artist is returned in library order -- an identical prompt to before."""
    names = [a["name"] for a in artists if a.get("name")]
    if len(names) <= n:
        return names
    req = _tokens(request)
    sig = signals_fn()

    def score(name):
        s = sig.get(name, {})
        toks = _tokens(name) | _tokens(s.get("genres", ""))
        token_match = len(req & toks)
        affinity = 2 * s.get("thumbs_up", 0) - 2 * s.get("thumbs_down", 0) - s.get("skips", 0)
        recency = 1 if s.get("last_played") else 0
        return 3 * token_match + affinity + recency

    ranked = sorted(names, key=lambda nm: (-score(nm), nm.lower()))
    return ranked[:n]


def effective_score(t, scores):
    s = scores.get(t.get("path", ""), {})
    base = int(s.get("public_score", 60))
    personal = int(s.get("personal_score", 0))
    skips = int(s.get("skip_count", 0))
    up = int(s.get("thumbs_up_count", 0))
    down = int(s.get("thumbs_down_count", 0))
    return base + personal + (up * 8) - (skips * 10) - (down * 40)


def allowed(t, scores, request):
    s = scores.get(t.get("path", ""), {})
    text_request = request.lower()

    if s.get("holiday_tag") and not holiday_allowed(s.get("holiday_tag")):
        return False

    if int(s.get("is_skit", 0)) and "skit" not in text_request:
        return False

    if int(s.get("is_intro", 0)) and not any(x in text_request for x in ["intro", "interlude", "skit"]):
        return False

    if effective_score(t, scores) < 20:
        return False

    return True


def select_tracks(weights, tracks, scores, request, count):
    """Build a radio-style playlist from weighted artists and the track pool.

    Pure (aside from shuffling): given artist weights, the track list, the
    score map and the request text, return up to ``count`` tracks that pass
    ``allowed()``, favouring higher ``effective_score``, with no exact
    duplicates and no artist repeated within the last 3 slots.
    """
    by_artist = defaultdict(list)
    for track in tracks:
        if allowed(track, scores, request):
            by_artist[track["artist"]].append(track)

    pool = []
    for artist, weight in weights.items():
        choices = by_artist.get(artist, [])
        choices = sorted(choices, key=lambda item: effective_score(item, scores), reverse=True)

        top = choices[:max(10, min(len(choices), weight * 8))]
        random.shuffle(top)
        pool.extend(top[:max(2, weight * 3)])

    random.shuffle(pool)

    playlist = []
    recent_artists = []
    seen = set()

    for track in pool:
        key = (track["artist"], track["title"], track["album"])
        artist = track["artist"]

        if key in seen or artist in recent_artists:
            continue

        playlist.append(track)
        seen.add(key)
        recent_artists.append(artist)
        recent_artists = recent_artists[-3:]

        if len(playlist) >= count:
            break

    return playlist


def build_station(request, count=COUNT):
    request = request.strip() or "90s rock"

    lib = library_repo.get_library()
    tracks = lib["tracks"]
    # A station is built from the LOCAL catalog: the model may only weight
    # artists this house actually owns. With no usable catalog there is
    # nothing to weight, and the honest answer is a refusal in a sentence
    # rather than an empty prompt, a doomed model call and a generic
    # "station build failed". The gate is the same one the capability
    # surface publishes, so the interface and the service agree.
    import catalog_status
    catalog_status.require(lambda: len(tracks))
    # Bounded candidate set for the prompt (identical to the full list at small
    # library sizes; capped so the prompt never grows without bound).
    artists = bounded_artists(lib["artists"], request)

    prompt = f"""
You are the listener's personal radio DJ.

The user request is:
{request}

You may ONLY choose artists from this exact library list:

{chr(10).join("- " + a for a in artists)}

Return JSON only.

Format:
{{
  "station_name": "short station name",
  "artist_weights": {{
    "Artist Name": 10,
    "Artist Name": 7
  }}
}}

Rules:
- Use only artists from the provided list.
- Pick 5 to 12 artists.
- Higher weight means play more.
- For moods/vibes, infer the best artists from the real library.
- Do not include artists not in the list.
"""

    resp = providers.ollama_generate(prompt, MODEL, timeout=180,
                                     format="json")
    data = json.loads(resp["response"])

    weights = data.get("artist_weights", {})
    weights = {k: int(v) for k, v in weights.items() if k in artists and int(v) > 0}

    # Candidate artists first, then scores + tracks for only their tracks
    # (targeted, indexed). Non-candidate tracks never enter the playlist, so this
    # is output-identical to loading the whole catalog's scores.
    candidates = set(weights)
    candidate_tracks = [t for t in tracks if t.get("artist") in candidates]
    scores = get_scores_for_artists(list(weights))

    playlist = select_tracks(weights, candidate_tracks, scores, request, count)

    return {
        "station_name": data.get("station_name", request),
        "request": request,
        "artists_text": json.dumps(weights, ensure_ascii=False),
        "artists": weights,
        "tracks": playlist,
    }


def main():
    request = " ".join(sys.argv[1:]).strip() or "90s rock"
    result = build_station(request, COUNT)

    print()
    print("Station:", result["station_name"])
    print("Request:", result["request"])
    print("Artists:", result["artists"])
    print()
    for i, track in enumerate(result["tracks"], 1):
        print(f"{i:02d}. {track['artist']} - {track['title']} [{track['album']}]")


if __name__ == "__main__":
    main()
