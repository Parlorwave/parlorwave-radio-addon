"""Station programming features: the four signals a programmer may weigh,
kept DISTINCT.

  familiarity  PERSONAL, per TRACK. How often this household has actually
               heard the song through, from `play_events` (completes,
               partials, skips). 0..1, deterministic, no LLM.
  affinity     PERSONAL, per ARTIST. The same evidence aggregated per artist
               and normalised across the station's roster. 0..1.
  fit          STATION, per TRACK. The frozen roster weight of the track's
               artist, normalised to the roster's strongest artist. 0..1.
               The selector already follows this; it is reported here so
               "fits the station" is never confused with "is familiar".
  hit          PUBLIC, per TRACK. Whether the wider world knows the song.
               There is NO local source for this: `tracks.public_score` is a
               title heuristic (live/demo/remix penalties), `ci_affinity` is
               a legacy value that nothing maintains, and the catalog
               carries no popularity tag. `public_hit()` is the seam and it
               answers None; nothing downstream may treat None as 0.

Why this is not a scorer: blending everything into one number gives a
number that is wrong in ways nobody can explain. Here each
signal keeps its own name, the station feel says which of them matter, and
a refill can be read back as "familiar because heard 5 times, fits because
weight 8/10". Deterministic for the same database state, computed from a
couple of small aggregate queries per refill -- never a model call per
refill.

PURE apart from `play_stats()`, the one loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# A track at or above this familiarity is "recognisable": something the
# household has heard through about twice. Discovery uses it to pick its
# anchors (station_feel).
ANCHOR_MIN_FAMILIARITY = 0.5


@dataclass(frozen=True)
class PlayStats:
    completes: int = 0
    partials: int = 0
    skips: int = 0
    starts: int = 0


@dataclass(frozen=True)
class TrackFeatures:
    familiarity: float      # 0..1 personal, track
    affinity: float         # 0..1 personal, artist
    fit: float              # 0..1 station, roster weight
    hit: float | None       # public; None = no source (always, today)

    def as_dict(self) -> dict[str, Any]:
        return {"familiarity": round(self.familiarity, 3),
                "affinity": round(self.affinity, 3),
                "fit": round(self.fit, 3), "hit": self.hit}


# --- PURE scoring -------------------------------------------------------------

def familiarity(stats: PlayStats | None) -> float:
    """0..1. Heard through twice ~0.5, four times ~0.75, six ~0.875; a
    partial listen counts a quarter; skips dilute in proportion.

    Explicit table (completes, partials, skips -> score):
      0,0,0 -> 0.0    2,0,0 -> 0.5    4,0,0 -> 0.75    2,0,2 -> 0.25
    """
    if stats is None:
        return 0.0
    heard = stats.completes + 0.25 * stats.partials
    if heard <= 0:
        return 0.0
    depth = 1.0 - 0.5 ** (heard / 2.0)
    return round(depth * (heard / (heard + stats.skips)), 6)


def affinity_by_artist(stats_by_path: dict[str, PlayStats],
                       artist_of: dict[str, str]) -> dict[str, float]:
    """Per-artist 0..1 from the same evidence: net heard-through (completes
    minus skips, floored at zero) normalised to the roster's most-heard
    artist. Empty evidence -> everyone 0.0, never a made-up 0.5."""
    net: dict[str, float] = {}
    for path, stats in stats_by_path.items():
        artist = artist_of.get(path)
        if not artist:
            continue
        net[artist] = net.get(artist, 0.0) + max(
            0.0, stats.completes + 0.25 * stats.partials - stats.skips)
    top = max(net.values(), default=0.0)
    if top <= 0:
        return {a: 0.0 for a in net}
    return {a: round(v / top, 6) for a, v in net.items()}


def fit_by_artist(artist_weights: dict[str, Any]) -> dict[str, float]:
    weights = {str(a): float(w) for a, w in (artist_weights or {}).items()
               if a and float(w) > 0}
    top = max(weights.values(), default=0.0)
    if top <= 0:
        return {}
    return {a: round(w / top, 6) for a, w in weights.items()}


def public_hit(track: dict[str, Any]) -> float | None:
    """The public hit-ness seam. No local source exists (see module doc), so
    this is None for every track. A future provider (chart data, MusicBrainz
    popularity, a curated list) plugs in here and nowhere else."""
    return None


def features_for(tracks: Iterable[dict[str, Any]], *,
                 stats_by_path: dict[str, PlayStats],
                 artist_weights: dict[str, Any]) -> dict[str, TrackFeatures]:
    """Features keyed by catalog PATH (the selector's identity), PURE."""
    tracks = list(tracks)
    artist_of = {str(t.get("path") or ""): str(t.get("artist") or "")
                 for t in tracks}
    affinity = affinity_by_artist(stats_by_path, artist_of)
    fit = fit_by_artist(artist_weights)
    out: dict[str, TrackFeatures] = {}
    for track in tracks:
        path = str(track.get("path") or "")
        if not path:
            continue
        artist = str(track.get("artist") or "")
        out[path] = TrackFeatures(
            familiarity=familiarity(stats_by_path.get(path)),
            affinity=affinity.get(artist, 0.0),
            fit=fit.get(artist, 0.0),
            hit=public_hit(track))
    return out


# --- the one loader -----------------------------------------------------------

def play_stats(artists: Iterable[str], db_path=None) -> dict[str, PlayStats]:
    """Aggregate play_events per path for the given artists: ONE grouped
    query against the play-history database. Any failure returns {} -- a
    refill never waits on, or fails for, its features."""
    names = [str(a).lower() for a in dict.fromkeys(artists) if a]
    if not names:
        return {}
    try:
        import musicdb
        import sqlite3
        c = sqlite3.connect(db_path) if db_path else musicdb.conn()
        try:
            marks = ",".join("?" for _ in names)
            rows = c.execute(
                "SELECT path, event, COUNT(*) FROM play_events "
                f"WHERE lower(artist) IN ({marks}) GROUP BY path, event",
                names).fetchall()
        finally:
            c.close()
    except Exception:
        return {}
    counts: dict[str, dict[str, int]] = {}
    for path, event, n in rows:
        counts.setdefault(str(path), {})[str(event)] = int(n)
    return {path: PlayStats(completes=ev.get("complete", 0),
                            partials=ev.get("partial", 0),
                            skips=ev.get("skip", 0),
                            starts=ev.get("start", 0))
            for path, ev in counts.items()}
