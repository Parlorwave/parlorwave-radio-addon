"""Local-first user-request resolver.

ONE resolver every request surface (phone, Studio text box, HA conversation)
ultimately calls, so request behaviour cannot drift between them. It answers a
single question -- "what should we play for this request?" -- and returns a
STRUCTURED result the conversational layer turns into words. It never speaks,
never enqueues, never reorders, and never invents a match.

Policy (deliberately small):

    request -> normalize artist/title intent
            -> search the OWNED/LOCAL library FIRST
            -> exact / strong match?  YES -> use the local track
                                      NO  -> report the provider capability truth

Match contract (bounded -- not a semantic search engine):

    exact        one track whose normalized artist AND title both match
    strong       one track whose title matches and artist is a confident
                 token-superset/subset (or the request gave no artist)
    ambiguous    the intent matches several distinct tracks (remasters, or
                 the same title by different artists) -- never silently pick
    artist_only  an artist with no title -- not a single-track intent
    local_not_found  nothing local matches

For Pandora (the only non-local source today) the capability truth is fixed:
current MA dynamic radio exposes NO arbitrary exact-track play, so a missing
local match must NOT become "playing that from Pandora". The result carries
`exact_provider_play_unavailable` so the conversational layer can offer a
station instead -- it does not hard-code any sentence.
"""

from __future__ import annotations

import re
from typing import Any

# result statuses
LOCAL_HIT = "local_hit"
AMBIGUOUS = "ambiguous"
ARTIST_ONLY = "artist_only"
LOCAL_NOT_FOUND = "local_not_found"

# confidences
EXACT = "exact"
STRONG = "strong"

_AMBIGUITY_CAP = 6   # how many candidates we hand back before it is just "many"


def _norm(text: Any) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Deterministic; no
    fuzzy stemming. '' for empty/None."""
    s = re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold())
    return " ".join(s.split())


def _tokens(text: Any) -> set[str]:
    return set(_norm(text).split())


def _track_artist(t: dict) -> str:
    """Artist string from either the Music Brain catalog shape (`artist`) or an
    MA library-item shape (`artists`: [{name}])."""
    if t.get("artist"):
        return str(t["artist"])
    arts = t.get("artists") or []
    return ", ".join(a.get("name") for a in arts if a.get("name"))


def _track_title(t: dict) -> str:
    return str(t.get("title") or t.get("name") or "")


def _track_album(t: dict) -> Any:
    a = t.get("album")
    return a.get("name") if isinstance(a, dict) else a


def _track_version(t: dict) -> str:
    """MA's own version label ('Live', 'Remastered 2011', 'Demo'); '' for
    the plain recording and for the local catalog, which has none."""
    return str(t.get("version") or "").strip()


def _track_view(t: dict) -> dict:
    return {"path": t.get("path"), "artist": _track_artist(t),
            "title": _track_title(t), "album": _track_album(t),
            "version": _track_version(t),
            "uri": t.get("uri") or t.get("media_uri"),
            "duration": t.get("duration")}


def _ambiguity(rows: list[dict]) -> str:
    """WHY several distinct recordings matched, for the person asking:
    'artist' (the same title by different artists), else 'version' (live,
    remastered, demo...), else 'album' (the same recording on two albums)."""
    if len({_norm(_track_artist(t)) for t in rows}) > 1:
        return "artist"
    if len({_norm(_track_version(t)) for t in rows}) > 1:
        return "version"
    return "album"


def resolve_local(artist: str | None, title: str | None, *,
                  tracks: list[dict] | None = None) -> dict[str, Any]:
    """Resolve a request against the LOCAL library only. Pure over `tracks`
    (the caller injects them in tests; production passes the real catalog)."""
    if tracks is None:
        import library_repo
        tracks = library_repo.get_tracks()
    na, nt = _norm(artist), _norm(title)
    query = {"artist": artist, "title": title}

    # artist-only intent is not a single-track request
    if nt == "" and na != "":
        return {"status": ARTIST_ONLY, "confidence": None, "track": None,
                "candidates": [], "query": query, "provider": None}
    if nt == "" and na == "":
        return {"status": LOCAL_NOT_FOUND, "confidence": None, "track": None,
                "candidates": [], "query": query,
                "provider": _provider_fallback()}

    exact: list[dict] = []
    strong: list[dict] = []
    # A request may name the version in the title -- "Quarry Light (Live)",
    # "Quarry Light - Remastered 2011" -- and then only that version matches.
    named: list[dict] = []
    named_strong: list[dict] = []
    for t in tracks:
        ta, tt = _norm(_track_artist(t)), _norm(_track_title(t))
        tv = _norm(_track_version(t))
        hit_version = bool(tv) and nt == _norm(f"{_track_title(t)} {_track_version(t)}")
        if tt != nt and not hit_version:
            continue
        if na == "" or ta == na:
            (named if hit_version else exact).append(t)
        elif _tokens(_track_artist(t)) & _tokens(artist):
            (named_strong if hit_version else strong).append(t)
    if named or named_strong:
        exact, strong = named, named_strong

    # de-duplicate so genuine multi-recording ambiguity is distinguished
    # from one track indexed once. The VERSION is part of the identity: a
    # live take and the studio cut on one deluxe album are two recordings,
    # and collapsing them used to play whichever came first (alpha.4 testing).
    def _distinct(rows: list[dict]) -> list[dict]:
        seen, out = set(), []
        for t in rows:
            key = (_norm(_track_artist(t)), _norm(_track_title(t)),
                   _norm(_track_album(t)), _norm(_track_version(t)))
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    exact_d = _distinct(exact)
    if len(exact_d) == 1:
        return {"status": LOCAL_HIT, "confidence": EXACT,
                "track": _track_view(exact_d[0]), "candidates": [],
                "query": query, "provider": None}
    if len(exact_d) > 1:
        return {"status": AMBIGUOUS, "confidence": None, "track": None,
                "candidates": [_track_view(t) for t in exact_d[:_AMBIGUITY_CAP]],
                "ambiguity": _ambiguity(exact_d),
                "query": query, "provider": None}

    strong_d = _distinct(strong)
    if len(strong_d) == 1:
        return {"status": LOCAL_HIT, "confidence": STRONG,
                "track": _track_view(strong_d[0]), "candidates": [],
                "query": query, "provider": None}
    if len(strong_d) > 1:
        return {"status": AMBIGUOUS, "confidence": None, "track": None,
                "candidates": [_track_view(t) for t in strong_d[:_AMBIGUITY_CAP]],
                "ambiguity": _ambiguity(strong_d),
                "query": query, "provider": None}

    return {"status": LOCAL_NOT_FOUND, "confidence": None, "track": None,
            "candidates": [], "query": query, "provider": _provider_fallback()}


def _provider_fallback() -> dict[str, Any]:
    """The capability truth to hand the conversational layer when local has no
    match. Pandora dynamic radio cannot force an exact track today; it may be
    able to SELECT an existing subscribed station (see match_pandora_station).
    The layer chooses the wording -- no sentence is hard-coded here."""
    return {
        "local_not_found": True,
        "exact_provider_play_unavailable": True,   # Pandora: no direct track play
        "station_selectable_possible": True,        # an existing station MIGHT match
    }


# --- existing Pandora station selection (no creation) -------------------------

STATION_SELECTED = "station_selected"
STATION_AMBIGUOUS = "station_ambiguous"
STATION_NOT_FOUND = "station_not_found"


def match_pandora_station(request_text: str,
                          stations: list[dict]) -> dict[str, Any]:
    """Match a request like 'play my Cold Orchard Pandora station' against the
    user's ACTUAL subscribed stations (caller supplies them from the provider).
    Returns a structured result; NEVER creates or seeds a station -- current
    MA has no such capability, so an unmatched request is `station_not_found`,
    not a fabricated station."""
    want = _norm(request_text)
    # strip filler that surrounds a station name
    for filler in ("play", "my", "the", "pandora", "station", "radio", "put on"):
        want = _norm(want.replace(filler, " "))
    want_tokens = set(want.split())
    if not want_tokens:
        return {"status": STATION_NOT_FOUND, "station": None, "candidates": []}
    scored = []
    for s in stations:
        name = s.get("name") or s.get("display_name") or ""
        ntok = set(_norm(name).split())
        if not ntok:
            continue
        if want == _norm(name):
            scored.append((2, s))                     # exact name
        elif want_tokens <= ntok or ntok <= want_tokens:
            scored.append((1, s))                     # strong token containment
    scored.sort(key=lambda x: -x[0])
    top = [s for score, s in scored if score == (scored[0][0] if scored else 0)]
    if len(top) == 1:
        return {"status": STATION_SELECTED, "station": top[0], "candidates": []}
    if len(top) > 1:
        return {"status": STATION_AMBIGUOUS, "station": None,
                "candidates": top[:_AMBIGUITY_CAP]}
    return {"status": STATION_NOT_FOUND, "station": None, "candidates": []}
