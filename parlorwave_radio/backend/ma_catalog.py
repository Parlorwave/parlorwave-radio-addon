#!/usr/bin/env python3
"""Music Assistant's LIBRARY, read for station programming. READ ONLY.

The Station Programmer builds stations from what the household actually
owns, and on a typical install that is Music Assistant's library, not
Parlorwave's optional local catalog (catalog_status): a typical install has
thousands of MA library tracks and an empty local catalog. So the station
resolver reads here.

EVERY COMMAND IS ON AN ALLOWLIST. `read()` refuses anything not in
READ_COMMANDS before a request is built, and every command on it is a
`library.read`-scoped listing (verified against the MA 2.10 command
reference). There is no player, queue, transport, favourite,
library-edit or metadata-refresh verb here, and there never may be: the
Test Station contract is that a preview performs ZERO Music Assistant
mutations, and this module is how that is made structural rather than a
promise. Pinned by test_station_preview_safety.

Two MA getters are deliberately NOT used: `music/*/get` and
`music/genres/get` default to `allow_update_metadata`, which schedules a
metadata refresh inside Music Assistant -- a write, even if not a queue
write. Everything here is a library listing.

WHAT MA 2.10 GIVES A STATION BUILDER:

    tracks        music/tracks/library_items   paged; album.year on every
                                               track; artists; duration;
                                               provider mapping; a `genre`
                                               id filter
    artists       music/artists/library_items  album artists
    genres        music/genres/library_items   first-class, but COARSE:
                                               subgenres fold into parents
                                               ("progressive rock" is an
                                               alias of Rock); `search`
                                               matches aliases
    genre sizes   music/genres/media_counts
    artist genres music/genres/genres_for_media_item
    related       music/artists/similar_artists  in-library artists only,
                                               ONE-DIRECTIONAL (an artist
                                               can list nobody while
                                               others list it)

There is no dependable popularity, mood, style or BPM data in a typical
library (metadata.popularity/mood/style are null), so nothing here pretends to
offer them.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("musicbrain")

#: The only commands this module will ever send. Adding one is a reviewed
#: change: it must be a `library.read` listing with no side effect.
READ_COMMANDS = frozenset({
    "music/tracks/library_items",
    "music/artists/library_items",
    "music/genres/library_items",
    "music/genres/media_counts",
    "music/genres/genres_for_media_item",
    "music/artists/similar_artists",
})

PAGE = 500
#: Hard bound on a catalog read. A library larger than this is read up to
#: the bound and reported as truncated, never loaded without limit.
MAX_TRACKS = 60000
MAX_ARTISTS = 5000
#: How many artists' similar-artist lists are read to build the (symmetric)
#: relation graph. Beyond it the graph is reported partial.
MAX_GRAPH_ARTISTS = 400
#: A snapshot is reused for this long. A preview is a person pressing a
#: button; a library does not change between two presses.
SNAPSHOT_TTL_S = 600.0
#: The relation graph costs one MA read per artist (many seconds for a
#: few dozen artists) and changes only when the library's artists do, so it
#: outlives many snapshots.
GRAPH_TTL_S = 6 * 3600.0

SOURCE = "music_assistant_library"


class CatalogError(RuntimeError):
    """The library could not be read. The message is safe to show."""


def _default_api(command: str, args: dict[str, Any]) -> Any:
    # The one MA transport (auth, URL, timeout) is ma_queue's. Only its
    # low-level call is borrowed; no ma_queue helper is ever reached from here.
    import ma_queue
    return ma_queue._api(command, args)


def read(command: str, args: dict[str, Any] | None = None, *,
         api_fn: Callable[[str, dict[str, Any]], Any] | None = None) -> Any:
    """Send ONE allowlisted read. Anything else raises before a request
    exists."""
    if command not in READ_COMMANDS:
        raise CatalogError(f"refusing non-read Music Assistant command: "
                           f"{command}")
    try:
        return (api_fn or _default_api)(command, dict(args or {}))
    except CatalogError:
        raise
    except Exception as exc:  # noqa: BLE001 - one boundary, one meaning
        raise CatalogError(f"Music Assistant library read failed "
                           f"({command})") from exc


def _pages(command: str, args: dict[str, Any], bound: int, api_fn
           ) -> tuple[list[dict[str, Any]], bool]:
    """Every item of a paged listing, up to `bound`. Returns (items,
    truncated). A short page ends the walk; the page size is always sent
    explicitly because MA's own default silently stops at 500."""
    out: list[dict[str, Any]] = []
    offset = 0
    while len(out) < bound:
        page = read(command, dict(args, limit=PAGE, offset=offset),
                    api_fn=api_fn)
        if not isinstance(page, list):
            raise CatalogError(f"unexpected answer from {command}")
        out.extend(x for x in page if isinstance(x, dict))
        offset += len(page)
        if len(page) < PAGE:
            return out, False
    return out[:bound], True


def _track(item: dict[str, Any]) -> dict[str, Any] | None:
    """One MA library track in the catalog-track shape station_selector
    already speaks. `path` IS the MA library URI: selector identity,
    exclusions and scores all key on `path`, and for this source the URI
    is the stable identity of the recording."""
    uri = str(item.get("uri") or "")
    title = str(item.get("name") or "").strip()
    artists = [str(a.get("name") or "").strip()
               for a in (item.get("artists") or []) if isinstance(a, dict)]
    artists = [a for a in artists if a]
    if not (uri and title and artists):
        return None
    album = item.get("album") if isinstance(item.get("album"), dict) else {}
    year = album.get("year")
    try:
        year = int(year) if year else None
    except (TypeError, ValueError):
        year = None
    if year is not None and not 1900 <= year <= 2100:
        year = None
    providers = [m for m in (item.get("provider_mappings") or [])
                 if isinstance(m, dict)]
    playable = bool(item.get("is_playable", True)) and bool(
        item.get("available", True)) and (
        not providers or any(m.get("available", True) for m in providers))
    return {
        "path": uri,
        "uri": uri,
        "item_id": str(item.get("item_id") or ""),
        "title": title,
        "version": str(item.get("version") or ""),
        "artist": artists[0],
        "artists": artists,
        "album": str(album.get("name") or ""),
        "album_uri": str(album.get("uri") or ""),
        "year": year,
        "duration": item.get("duration"),
        "favorite": bool(item.get("favorite")),
        "provider": str((providers[0] if providers else {}).get(
            "provider_domain") or item.get("provider") or ""),
        "playable": playable,
    }


@dataclass
class Snapshot:
    """One read of the library, reused for SNAPSHOT_TTL_S."""
    tracks: list[dict[str, Any]]
    artists: list[dict[str, Any]]          # {id, name}
    genres: list[dict[str, Any]]           # {id, name, tracks, artists}
    fetched_at: float
    truncated: bool = False
    # Lazily filled, per snapshot -- only artist radio needs them.
    _graph: dict[str, dict[str, set[str]]] | None = field(default=None,
                                                          repr=False)
    _graph_partial: bool = False
    _genre_uris: dict[str, set[str]] = field(default_factory=dict,
                                             repr=False)
    _artist_genres: dict[str, list[str]] = field(default_factory=dict,
                                                 repr=False)

    def describe(self) -> dict[str, Any]:
        return {"source": SOURCE, "tracks": len(self.tracks),
                "artists": len(self.artists),
                "genres_with_tracks": sum(1 for g in self.genres
                                          if g.get("tracks")),
                "truncated": self.truncated,
                "fetched_at": self.fetched_at}


def load_snapshot(api_fn=None, *, now: float | None = None) -> Snapshot:
    """Read tracks, artists and genres. Raises CatalogError."""
    items, truncated = _pages("music/tracks/library_items", {}, MAX_TRACKS,
                              api_fn)
    tracks = [t for t in (_track(i) for i in items) if t]
    raw_artists, _ = _pages("music/artists/library_items", {}, MAX_ARTISTS,
                            api_fn)
    artists = [{"id": str(a.get("item_id") or ""),
                "name": str(a.get("name") or "").strip()}
               for a in raw_artists if a.get("item_id") and a.get("name")]
    raw_genres, _ = _pages("music/genres/library_items", {}, 5000, api_fn)
    genres = [{"id": str(g.get("item_id") or ""),
               "name": str(g.get("name") or "").strip()}
              for g in raw_genres if g.get("item_id") and g.get("name")]
    counts: dict[str, Any] = {}
    if genres:
        try:
            got = read("music/genres/media_counts",
                       {"genre_ids": [g["id"] for g in genres]},
                       api_fn=api_fn)
            counts = got if isinstance(got, dict) else {}
        except CatalogError:
            # Sizes are a nicety for reporting; resolution does not need
            # them, so their absence is not a failure.
            counts = {}
    for g in genres:
        c = counts.get(g["id"]) if isinstance(counts.get(g["id"]), dict) \
            else {}
        g["tracks"] = int(c.get("track") or 0)
        g["artists"] = int(c.get("artist") or 0)
    return Snapshot(tracks=tracks, artists=artists, genres=genres,
                    fetched_at=time.time() if now is None else now,
                    truncated=truncated)


def search_genres(snapshot: Snapshot, text: str, api_fn=None
                  ) -> list[dict[str, Any]]:
    """MA's own alias-aware genre search: which parent genres claim `text`
    among their aliases. Returns genre dicts from the snapshot."""
    got = read("music/genres/library_items", {"search": text, "limit": 50},
               api_fn=api_fn)
    ids = [str(g.get("item_id")) for g in (got if isinstance(got, list)
                                           else []) if isinstance(g, dict)]
    by_id = {g["id"]: g for g in snapshot.genres}
    return [by_id[i] for i in ids if i in by_id]


def genre_track_uris(snapshot: Snapshot, genre_id: str, api_fn=None
                     ) -> set[str]:
    """URIs of the library tracks MA maps to one genre (cached per
    snapshot)."""
    if genre_id not in snapshot._genre_uris:
        try:
            gid: Any = int(genre_id)
        except ValueError:
            gid = genre_id
        items, _ = _pages("music/tracks/library_items", {"genre": gid},
                          MAX_TRACKS, api_fn)
        snapshot._genre_uris[genre_id] = {str(i.get("uri")) for i in items
                                          if i.get("uri")}
    return snapshot._genre_uris[genre_id]


def artist_genres(snapshot: Snapshot, artist_id: str, api_fn=None
                  ) -> list[str]:
    """The genre names MA maps to one artist (cached per snapshot)."""
    if artist_id not in snapshot._artist_genres:
        try:
            got = read("music/genres/genres_for_media_item",
                       {"media_type": "artist", "media_id": artist_id},
                       api_fn=api_fn)
        except CatalogError:
            got = []
        snapshot._artist_genres[artist_id] = [
            str(g.get("name")) for g in (got if isinstance(got, list) else [])
            if isinstance(g, dict) and g.get("name")]
    return snapshot._artist_genres[artist_id]


def similarity_graph(snapshot: Snapshot, api_fn=None
                     ) -> tuple[dict[str, dict[str, set[str]]], bool]:
    """{artist_name: {related_name: {provenance...}}}, SYMMETRIC.

    MA's similar-artist lists are one-directional on this library, so an
    artist nobody linked FROM can still be linked TO. The edge "A lists B"
    is recorded on both A and B, with the direction kept in its
    provenance. Returns (graph, partial)."""
    if snapshot._graph is not None:
        return snapshot._graph, snapshot._graph_partial
    names = tuple(sorted(a["name"] for a in snapshot.artists))
    if api_fn is None:
        with _LOCK:
            hit = _GRAPH.get(names)
        if hit is not None and time.time() - hit[0] < GRAPH_TTL_S:
            snapshot._graph, snapshot._graph_partial = hit[1], hit[2]
            return hit[1], hit[2]
    graph: dict[str, dict[str, set[str]]] = {}
    known = {a["name"] for a in snapshot.artists}
    partial = len(snapshot.artists) > MAX_GRAPH_ARTISTS
    subjects = snapshot.artists[:MAX_GRAPH_ARTISTS]
    # Serial on purpose: Music Assistant answers these one at a time anyway
    # (six-wide reads are barely faster than serial ones),
    # so parallel reads would only load MA for no gain. The graph is cached
    # far longer than a snapshot instead (GRAPH_TTL_S).
    for artist in subjects:
        try:
            got = read("music/artists/similar_artists",
                       {"item_id": artist["id"],
                        "provider_instance_id_or_domain": "library",
                        "limit": 25}, api_fn=api_fn)
        except CatalogError:
            partial = True
            continue
        for other in got if isinstance(got, list) else []:
            name = str((other or {}).get("name") or "").strip()
            if not name or name == artist["name"] or name not in known:
                continue
            graph.setdefault(artist["name"], {}).setdefault(
                name, set()).add("ma_similar_artists")
            graph.setdefault(name, {}).setdefault(
                artist["name"], set()).add("ma_similar_artists_reverse")
    snapshot._graph, snapshot._graph_partial = graph, partial
    if api_fn is None:
        with _LOCK:
            # Keyed by the artist roster: a library that gained or lost an
            # artist gets a fresh graph at its next snapshot.
            _GRAPH.clear()
            _GRAPH[names] = (time.time(), graph, partial)
    return graph, partial


# --- the shared snapshot cache -----------------------------------------------

_LOCK = threading.Lock()
_CACHE: dict[str, Snapshot] = {}
_GRAPH: dict[tuple[str, ...], tuple[float, dict, bool]] = {}


def snapshot(api_fn=None, *, max_age: float = SNAPSHOT_TTL_S,
             refresh: bool = False) -> tuple[Snapshot, bool]:
    """A library snapshot no older than `max_age`. Returns (snapshot,
    from_cache). An injected api_fn (tests) never touches the cache."""
    if api_fn is not None:
        return load_snapshot(api_fn), False
    with _LOCK:
        snap = _CACHE.get("default")
        if (snap is not None and not refresh
                and time.time() - snap.fetched_at < max_age):
            return snap, True
        snap = load_snapshot()
        _CACHE["default"] = snap
        return snap, False


def reset_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _GRAPH.clear()
