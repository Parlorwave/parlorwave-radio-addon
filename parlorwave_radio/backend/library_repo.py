#!/usr/bin/env python3
"""Library catalog access seam.

The single module that knows where catalog data lives. It can read from either
music_index.json or the play_history.db `tracks` table, selected by the
LIBRARY_SOURCE env var ("json" | "db", default "json"). Callers
(llm_station_builder, the /missing_albums handler) never touch the source
directly, so the JSON->DB cutover is a config flip behind this seam.

Rollback is instant: set LIBRARY_SOURCE=json.
"""

import json
import os
from pathlib import Path
from typing import Any

from paths import BASE_DIR

INDEX = BASE_DIR / "music_index.json"

# The DB is served by default; library_compare checks it is equivalent to the
# JSON. Instant rollback: set LIBRARY_SOURCE=json.
DEFAULT_SOURCE = "db"

# Catalog columns read from the DB, in the order they are exposed to callers.
_DB_TRACK_COLS = ["path", "artist", "title", "album", "genre", "year", "duration",
                  "favorite", "rating"]


def _source(override: str | None = None) -> str:
    return (override or os.environ.get("LIBRARY_SOURCE", DEFAULT_SOURCE)).strip().lower()


# --- JSON source -----------------------------------------------------------

def _load_index() -> dict[str, Any]:
    if not INDEX.exists():
        return {"artists": [], "tracks": []}
    with open(INDEX, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"artists": [], "tracks": []}


def _json_tracks() -> list[dict[str, Any]]:
    return _load_index().get("tracks", []) or []


def _json_artists() -> list[dict[str, Any]]:
    return _load_index().get("artists", []) or []


# --- DB source -------------------------------------------------------------

def _db_tracks() -> list[dict[str, Any]]:
    """Catalog tracks from the DB in insertion (rowid) order, which equals the
    music_index.json scan order after a clean seed -- so downstream ordering
    (and thus seeded-RNG station output) is identical to the JSON path."""
    import musicdb
    c = musicdb.conn()
    try:
        rows = c.execute(
            f"SELECT {', '.join(_DB_TRACK_COLS)} FROM tracks ORDER BY rowid"
        ).fetchall()
    finally:
        c.close()
    return [dict(zip(_DB_TRACK_COLS, row)) for row in rows]


def _db_artists() -> list[dict[str, Any]]:
    """Artists aggregated from the DB, shaped exactly like index_library.py's
    output: sorted by lowercased name, albums sorted, with album/track counts."""
    import musicdb
    c = musicdb.conn()
    try:
        rows = c.execute("SELECT artist, album FROM tracks").fetchall()
    finally:
        c.close()
    agg: dict[str, dict[str, Any]] = {}
    for artist, album in rows:
        entry = agg.setdefault(artist, {"albums": set(), "track_count": 0})
        if album:
            entry["albums"].add(album)
        entry["track_count"] += 1
    out = []
    for name in sorted(agg, key=lambda n: (n or "").lower()):
        albums = sorted(agg[name]["albums"])
        out.append({
            "name": name,
            "albums": albums,
            "album_count": len(albums),
            "track_count": agg[name]["track_count"],
        })
    return out


# --- Public seam -----------------------------------------------------------

def get_tracks(source: str | None = None) -> list[dict[str, Any]]:
    """All catalog tracks (from LIBRARY_SOURCE, or an explicit override)."""
    return _db_tracks() if _source(source) == "db" else _json_tracks()


def get_artists(source: str | None = None) -> list[dict[str, Any]]:
    """All catalog artists (name + owned albums)."""
    return _db_artists() if _source(source) == "db" else _json_artists()


def get_library(source: str | None = None) -> dict[str, Any]:
    """The whole library ({'artists': [...], 'tracks': [...]}) from one source."""
    src = _source(source)
    return {"artists": get_artists(src), "tracks": get_tracks(src)}


def count_tracks(source: str | None = None) -> int:
    """How many catalog tracks exist, WITHOUT materialising them.

    The capability surface asks this question on every health read, so it
    must not cost a full catalog load: the DB source counts rows, the JSON
    source is a file that is already parsed whole by every other reader.
    """
    if _source(source) == "db":
        import musicdb
        return musicdb.track_count()
    return len(_json_tracks())
