#!/usr/bin/env python3
"""The Station Programmer's one resolver: StationIntent -> station.

    StationIntent
         |
    resolve_station()      what the station IS: a roster of library artists
         |                 with roles and weights, a candidate pool of real
         |                 library tracks, per-artist caps, and every note
         |                 and warning about how well the catalog supports it
    build_candidates()     one batch: the SAME pure selector live refill
         |                 uses (station_selector), the same feel policy,
         |                 the same same-artist adjacency pass
       /        \\
  preview        live (station_live, Mode 2)
  no mutation    append the batch; call build_candidates again when the
                 runway runs low, with what has already played excluded

There is deliberately no second algorithm. Test Station calls exactly
resolve_station + build_candidates; station_live's start and refill call
exactly the same two with live exclusions filled in. "I tested this
station and liked it" therefore means "this is what it will play", which
is the only reason a preview is worth having. One verdict is shared too:
live_viability() is the minimum-viable-station rule the preview reports
and a live start enforces.

GROUNDED ONLY. Every artist on a roster is there because Music Assistant's
library says so (the artist's own tracks, MA's similar-artist lists, MA's
genre mapping, album years) or because the shipped Music Knowledge pack
documents a relationship (band membership, a shared member). No LLM is
asked anything, anywhere in this module, and the builder works with Smart
DJ off and no model configured. Search-result order is never programming:
nothing here ever matches a song TITLE against the query.

READ ONLY. The only Music Assistant access is through ma_catalog, whose
allowlist contains library listings and nothing else. Nothing here can
start a station: the live engine (station_live) imports this module,
never the other way round.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import ma_catalog
import service_clock
import station_intent
import station_selector

logger = logging.getLogger("musicbrain")

DEFAULT_COUNT = 25
MIN_COUNT, MAX_COUNT = 5, 40

# Roster weights. They are relative AIRTIME shares (the selector normalises
# by each artist's pool), so on an artist station with a full set of nine
# relatives the seed gets roughly a fifth of the music, not all of it.
SEED_WEIGHT = 8.0
RELATED_WEIGHT = 3.0
STRONG_RELATED_WEIGHT = 4.0      # two independent sources agree
SECOND_DEGREE_WEIGHT = 1.5
GENRE_NEIGHBOR_WEIGHT = 1.0

#: Below this many directly related artists, the roster widens one step
#: (relatives of relatives), then to shared-genre neighbours.
MIN_RELATED = 4
MAX_SECOND_DEGREE = 6
MAX_GENRE_NEIGHBORS = 8

# Per-batch caps, as a share of the batch.
SEED_CAP_SHARE = 0.30
ARTIST_CAP_SHARE = 0.16

#: The seed's target share of an artist station's airtime. A fixed seed
#: weight is diluted by a large family (a seed with many documented
#: relatives gets far less than its intended share); the target is kept by
#: scaling the seed against the whole roster instead.
SEED_TARGET_SHARE = 0.25

#: An artist joins a GENRE station only when Music Assistant tags at least
#: this share of the artist's library tracks with the genre. Track-level
#: genre tags on a typical library are noisy (a share of a rapper's tracks
#: can carry "Rock", of a psychedelic band's "Punk");
#: a stray tag must not put an artist on the wrong station.
GENRE_ARTIST_MIN_SHARE = 0.30

#: Recordings that are not radio programming whatever their tags say:
#: interview clips, commentary, deluxe-edition demos and work tapes. The
#: shared track classifier penalises some of these; a station excludes them.
_NON_RADIO = re.compile(
    r"\b(interview|excerpt|commentary|vocal idea|riff tape|work tape|"
    r"rough mix|demo|rehearsal|voice memo)\b|\briff$")

ROLE_SEED = "seed"
ROLE_RELATED = "related"
ROLE_SECOND = "second_degree"
ROLE_NEIGHBOR = "genre_neighbor"
ROLE_CATEGORY = "category"
RELATED_ROLES = (ROLE_RELATED, ROLE_SECOND, ROLE_NEIGHBOR)

#: MINIMUM VIABLE STATION -- the rule a live start enforces and a preview
#: reports, so a preset can never look healthier on the Dial than it would
#: be on air. Measured against the runway a live station keeps (25 to open,
#: topped up to 20 when 8 remain, up to 12 per refill, the selector's
#: 40-track / 4-hour rotation): below 40 songs a refill runs out of
#: material the listener has not just heard and falls to the selector's
#: continuity floor, which is repetition. Three artists and a 15-song
#: opening are what the per-artist caps allow a station that is genuinely
#: more than one band. Nothing is invented to reach a threshold: a thin
#: station is refused and says why.
LIVE_MIN_POOL_TRACKS = 40
LIVE_MIN_ARTISTS = 3
LIVE_MIN_OPENING = 15

#: STATION BUILDER REVISION. A saved preset's Test Station result is an
#: approval of what THIS builder programmed for its term; a result from a
#: different builder is out of date, and the preset must be tested again
#: before the Dial may start it (station_live) -- its term, label and Dial
#: settings are kept, never its old verdict or its old draw.
#:
#: BUMP IT for any change that can materially alter what a term resolves to
#: or which songs are programmed: resolver semantics, genre/era reading,
#: related-artist logic, duplicate/recording selection, artist caps and
#: diversity policy, candidate weighting, and the viability rule where it
#: changes a verdict. DO NOT bump it for copy, panel layout, logging, or
#: anything outside this builder (DJ, news, announcements).
#:
#:   1  the original builder (records carry no revision)
#:   2  recording choice reads MA's `version`: an ordinary take beats a
#:      labelled alternate of the same song (_dedupe)
STATION_BUILDER_REVISION = 2

TEST_CURRENT = "current"
TEST_STALE = "stale"
TEST_NONE = "none"


def test_state(last_test: Any, query: Any) -> str:
    """Is a preset's stored Test Station result an approval of what the
    builder would program for its term NOW? PURE.

      none     never tested (no record)
      stale    tested -- but for another term, or by another builder
               revision (a record from before revisions is revision 1)
      current  tested for this term by this builder

    Only a CURRENT record's verdict and draw mean anything: a stale one is
    neither "too thin" nor "tested"."""
    lt = last_test if isinstance(last_test, dict) else {}
    if not lt:
        return TEST_NONE
    if norm(lt.get("query")) != norm(query):
        return TEST_STALE
    if lt.get("builder_revision", 1) != STATION_BUILDER_REVISION:
        return TEST_STALE
    return TEST_CURRENT


def station_identity(kind: Any, term: Any, variant: Any) -> str:
    """What makes two stations THE SAME station: its kind, its term in the
    builder's comparison form, and the approved draw. PURE. The on-air
    station and a saved preset are compared by this -- never by preset id,
    which survives a retune (otherwise the Dial calls a retuned preset
    "On air now" and will not switch to its new term). A variant
    that is not a number (a preset with no current approval) matches
    nothing."""
    draw = variant if isinstance(variant, int) and not isinstance(
        variant, bool) else "-"
    return f"{kind or ''}|{norm(term)}|{draw}"


def preset_identity(preset: dict[str, Any]) -> str:
    """The station a saved preset starts NOW: its current term, and the draw
    its current approval names (none when the approval is not current)."""
    term = preset.get("query") or preset.get("source")
    lt = preset.get("last_test") or {}
    variant = lt.get("variant") if test_state(lt, term) == TEST_CURRENT \
        else None
    return station_identity(preset.get("kind"), term, variant)


STRATEGY_ARTIST = "artist_radio"
STRATEGY_GENRE = "genre"
STRATEGY_ERA = "era"

# Result statuses.
OK = "ok"
NO_MATCH = "no_match"
AMBIGUOUS = "ambiguous"
EMPTY_POOL = "empty_pool"
INVALID = "invalid_intent"
NOT_AVAILABLE = "kind_not_available"
CATALOG_UNAVAILABLE = "catalog_unavailable"


# --- text ---------------------------------------------------------------------

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"',
                         "”": '"', "–": "-", "—": "-"})


def norm(text: Any) -> str:
    """Comparison form: accents, quotes and punctuation folded, 'and' for
    '&', a leading 'the' dropped. Used on BOTH sides of every match."""
    s = unicodedata.normalize("NFKD", str(text or "").translate(_QUOTES))
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = s.replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    return s


def _words(text: str) -> list[str]:
    return text.split()


def _contains_words(haystack: str, needle: str) -> bool:
    """Whole-word containment on normalised text."""
    return bool(needle) and f" {needle} " in f" {haystack} "


_DECADE_WORDS = {"sixties": 1960, "seventies": 1970, "eighties": 1980,
                 "nineties": 1990}
_ERA_RE = re.compile(r"(?<![a-z0-9])(?:(19|20)(\d)0|'?(\d)0)'?s(?![a-z0-9])")
_FILLER = ("music", "radio", "station", "songs", "hits", "tunes")


def parse_era(text: str) -> tuple[dict[str, Any] | None, str]:
    """(era, remainder). Recognises '90s', "'90s", '1990s', '90's',
    '2000s', 'nineties'. One era per query; a second is refused by
    leaving it in the remainder, where it will fail to match."""
    raw = str(text or "").translate(_QUOTES).casefold()
    m = _ERA_RE.search(raw)
    start = None
    if m:
        if m.group(1):
            start = int(m.group(1) + m.group(2) + "0")
        else:
            d = int(m.group(3))
            start = 2000 + d * 10 if d <= 2 else 1900 + d * 10
        rest = raw[:m.start()] + " " + raw[m.end():]
    else:
        rest = raw
        for word, year in _DECADE_WORDS.items():
            if re.search(rf"\b{word}\b", raw):
                start = year
                rest = re.sub(rf"\b{word}\b", " ", raw, count=1)
                break
    if start is None:
        return None, raw
    label = f"{start}s"
    return {"start": start, "end": start + 9, "label": label}, rest


def _descriptor(rest: str) -> str:
    words = _words(norm(rest))
    while words and words[-1] in _FILLER:
        words.pop()
    return " ".join(words)


_QUALIFIER = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]|\s+-\s+.*$")


def base_title(title: str) -> str:
    """A song's title without bracketed or dashed qualifiers ('Slow Ferry -
    Live', 'Lost Bells (2008 Remix)' -> 'slow ferry', 'lost bells'). Two library
    tracks with one base title by one artist are one song for a station."""
    stripped = _QUALIFIER.sub("", str(title or ""))
    return norm(stripped) or norm(title)


# --- resolution ----------------------------------------------------------------

@dataclass
class Resolution:
    intent: station_intent.StationIntent
    status: str
    strategy: str | None = None
    seed_artist: str | None = None
    era: dict[str, Any] | None = None
    genres: list[dict[str, Any]] = field(default_factory=list)
    roster: dict[str, dict[str, Any]] = field(default_factory=dict)
    pool: list[dict[str, Any]] = field(default_factory=list)
    caps: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)   # for ambiguity

    @property
    def weights(self) -> dict[str, float]:
        return {a: r["weight"] for a, r in self.roster.items()}

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})

    def describe(self) -> dict[str, Any]:
        return {
            "status": self.status, "strategy": self.strategy,
            "seed_artist": self.seed_artist, "era": self.era,
            "genres": self.genres,
            "roster": [{"artist": a, "role": r["role"],
                        "weight": r["weight"], "because": r["because"],
                        "pool_tracks": r.get("pool_tracks", 0)}
                       for a, r in sorted(
                           self.roster.items(),
                           key=lambda kv: (-kv[1]["weight"], kv[0]))],
            "notes": list(self.notes), "warnings": list(self.warnings),
            "stats": dict(self.stats), "candidates": list(self.candidates),
        }


def _knowledge_related(seed: str, other: str) -> dict[str, Any] | None:
    try:
        import music_knowledge
        return music_knowledge.artists_related(seed, other)
    except Exception:  # noqa: BLE001 - relationships are optional evidence
        return None


def _match_artist(desc: str, names: list[str]) -> tuple[str | None, str,
                                                         list[str]]:
    """(artist, how, candidates). Exact normalised name first; then a
    UNIQUE whole-word containment of the term in one artist's name (the
    band's name without its first word, say)."""
    by_norm: dict[str, list[str]] = {}
    for n in names:
        by_norm.setdefault(norm(n), []).append(n)
    if desc in by_norm:
        return sorted(by_norm[desc])[0], "exact", []
    if len(desc) < 3:
        return None, "", []
    partial = sorted({n for key, group in by_norm.items() for n in group
                      if _contains_words(key, desc)})
    if len(partial) == 1:
        return partial[0], "partial", []
    return None, "", partial if len(partial) > 1 else []


def _match_genre(desc: str, snap: ma_catalog.Snapshot, api_fn
                 ) -> tuple[list[dict[str, Any]], str, list[str]]:
    """(genres, how, candidates). how: exact | alias | broadened | ''."""
    usable = [g for g in snap.genres if g.get("tracks", 1) > 0]
    squash = lambda s: norm(s).replace(" ", "")  # noqa: E731
    exact = [g for g in usable if squash(g["name"]) == squash(desc)]
    if exact:
        return exact[:1], "exact", []
    try:
        hits = [g for g in ma_catalog.search_genres(snap, desc, api_fn)
                if g.get("tracks", 1) > 0]
    except ma_catalog.CatalogError:
        hits = []
    if len(hits) > 1:
        named = [g for g in hits if _contains_words(desc, norm(g["name"]))]
        hits = named if len(named) == 1 else hits
    if len(hits) == 1:
        return hits, "alias", []
    if len(hits) > 1:
        return [], "", sorted(g["name"] for g in hits)
    inside = [g for g in usable if _contains_words(desc, norm(g["name"]))]
    if len(inside) == 1:
        return inside, "broadened", []
    if len(inside) > 1:
        return [], "", sorted(g["name"] for g in inside)
    return [], "", []


def _in_era(track: dict[str, Any], era: dict[str, Any] | None) -> bool:
    if era is None:
        return True
    year = track.get("year")
    return isinstance(year, int) and era["start"] <= year <= era["end"]


#: Music Assistant's track `version` is the recording's disambiguation
#: ("live at ...", "2019 remix", "instrumental", "early version", "radio
#: edit"). In a typical library every track that carries one names an
#: alternate of the ordinary recording, and a song that exists both with
#: and without one could be kept as the alternate, because a deluxe edition
#: files its live and demo cuts under the studio album's own name and year
#: and the tie fell to URI order. A remaster is the same recording, so it
#: counts as ordinary.
_SAME_RECORDING = re.compile(r"^(?:\d{4} )?remaster(?:ed)?(?: \d{4})?"
                             r"(?: version)?$")

#: How much a labelled alternate loses to the ordinary recording of the
#: same song: more than any single classifier penalty, less than an intro
#: or a skit -- so an ordinary take always wins, and a song that exists
#: ONLY as an alternate is still kept (there is nothing to prefer to it).
VERSION_PENALTY = 30


def is_alternate_version(version: Any) -> bool:
    """Does MA's `version` name something other than the ordinary
    recording? Empty and remaster-only are ordinary."""
    v = norm(version)
    return bool(v) and not _SAME_RECORDING.match(v)


def _dedupe(tracks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """One recording per (artist, base title). The kept one is the best by
    the SAME track classifier the selector's live policy uses (studio over
    live, album over intro), with a recording MA labels as an alternate
    ranked below the ordinary recording of the same song -- then the
    earliest year, then the URI, so the choice is deterministic and
    explainable. Preview and live share it: both build from this pool."""
    from musicdb import classify_track
    best: dict[tuple[str, str], tuple[tuple, dict[str, Any]]] = {}
    for t in tracks:
        key = (norm(t["artist"]), base_title(t["title"]))
        score = classify_track({"artist": t["artist"], "title": t["title"],
                                "album": t.get("album") or ""})[0]
        if is_alternate_version(t.get("version")):
            score -= VERSION_PENALTY
        rank = (-score, t.get("year") or 9999, t["uri"])
        if key not in best or rank < best[key][0]:
            best[key] = (rank, t)
    kept = sorted((v[1] for v in best.values()), key=lambda t: t["uri"])
    return kept, len(tracks) - len(kept)


def _genre_members(pool: list[dict[str, Any]],
                   library: list[dict[str, Any]]
                   ) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only artists the genre actually describes (see
    GENRE_ARTIST_MIN_SHARE). Returns (pool, left_out_artists)."""
    total: dict[str, int] = {}
    for t in library:
        total[t["artist"]] = total.get(t["artist"], 0) + 1
    tagged: dict[str, int] = {}
    for t in pool:
        tagged[t["artist"]] = tagged.get(t["artist"], 0) + 1
    members = {a for a, n in tagged.items()
               if n / max(1, total.get(a, n)) >= GENRE_ARTIST_MIN_SHARE}
    strays = sorted(set(tagged) - members)
    return [t for t in pool if t["artist"] in members], strays


def _cap_for(count: int, share: float) -> int:
    return max(2, math.ceil(count * share))


def resolve_station(intent: station_intent.StationIntent,
                    snap: ma_catalog.Snapshot, *, count: int = DEFAULT_COUNT,
                    api_fn=None,
                    knowledge_fn: Callable[[str, str], Any] | None = None
                    ) -> Resolution:
    """What station does this intent describe, in THIS library?"""
    res = Resolution(intent=intent, status=NO_MATCH)
    if not station_intent.KINDS[intent.kind]["buildable"]:
        res.status = NOT_AVAILABLE
        res.notes.append(f"'{intent.kind}' stations are a planned type and "
                         "cannot be built yet.")
        return res

    era, rest = parse_era(intent.query)
    desc = _descriptor(rest)
    res.era = era
    playable = [t for t in snap.tracks if t.get("playable", True)]
    unplayable = len(snap.tracks) - len(playable)
    library_artists = sorted({t["artist"] for t in playable})
    if era:
        res.notes.append(f"Era {era['label']}: album year {era['start']}-"
                         f"{era['end']} (Music Assistant album year).")

    # 1. An artist the library owns, exactly.  2. A genre, exactly.
    # 3. MA's alias-aware genre search.  4. An artist, partially.
    # 5. A genre named inside the query.  Never a song title.
    artist, how = None, ""
    genres: list[dict[str, Any]] = []
    ghow = ""
    candidates: list[str] = []
    if desc:
        artist, how, artist_cands = _match_artist(desc, library_artists)
        if how != "exact":
            genres, ghow, genre_cands = _match_genre(desc, snap, api_fn)
            if ghow in ("exact", "alias"):
                artist, how = None, ""           # 2/3 beat a partial name
            elif artist is not None:
                genres, ghow = [], ""            # 4 beats broadening
            elif ghow != "broadened":
                candidates = artist_cands or genre_cands
            elif artist_cands:
                # "broadened" is the weakest match there is; a query that
                # also names several artists is ambiguous, not a genre.
                genres, ghow, candidates = [], "", artist_cands
    if artist:
        res.strategy = STRATEGY_ARTIST
        res.seed_artist = artist
        res.notes.append(f"'{desc}' is the library artist {artist}"
                         + (" (partial name match)." if how == "partial"
                            else "."))
        _artist_roster(res, snap, artist, api_fn,
                       knowledge_fn or _knowledge_related)
        pool = [t for t in playable if t["artist"] in res.roster]
    elif genres:
        res.strategy = STRATEGY_GENRE
        g = genres[0]
        res.genres = [{"id": g["id"], "name": g["name"], "match": ghow,
                       "library_tracks": g.get("tracks")}]
        if ghow == "exact":
            res.notes.append(f"'{desc}' is the Music Assistant genre "
                             f"{g['name']}.")
        elif ghow == "alias":
            res.notes.append(f"Music Assistant files '{desc}' under its "
                             f"genre {g['name']}.")
            if norm(g["name"]) != desc:
                res.warn("genre_folded",
                         f"Music Assistant keeps no separate '{desc}' genre; "
                         f"it folds it into {g['name']}, so this station "
                         f"plays {g['name']}, not {desc} specifically.")
        else:
            res.notes.append(f"No '{desc}' genre in Music Assistant; the "
                             f"station was broadened to {g['name']}.")
            res.warn("genre_broadened",
                     f"'{desc}' is not a Music Assistant genre. The station "
                     f"was broadened to {g['name']}, which is much wider.")
        try:
            uris = ma_catalog.genre_track_uris(snap, g["id"], api_fn)
        except ma_catalog.CatalogError:
            res.status = CATALOG_UNAVAILABLE
            res.notes.append("The genre's tracks could not be read.")
            return res
        pool = [t for t in playable if t["uri"] in uris]
        share = (len(uris) / len(snap.tracks)) if snap.tracks else 0
        if share >= 0.6:
            res.warn("genre_too_broad",
                     f"{g['name']} covers {round(share * 100)}% of this "
                     f"library, so on its own it barely narrows anything.")
    elif era and not desc:
        res.strategy = STRATEGY_ERA
        res.notes.append("An era with no artist or genre: every genre in "
                         "that decade.")
        pool = list(playable)
    else:
        res.status = AMBIGUOUS if candidates else NO_MATCH
        res.candidates = candidates[:8]
        if candidates:
            res.notes.append(f"'{desc}' matches more than one thing in the "
                             "library: " + ", ".join(candidates[:8]) + ".")
        else:
            res.notes.append(f"Nothing in this library is an artist or a "
                             f"Music Assistant genre called '{desc or intent.query}'.")
        return res

    if res.strategy == STRATEGY_GENRE:
        pool, strays = _genre_members(pool, playable)
        if strays:
            res.notes.append(
                f"Left out {len(strays)} artist(s) with only stray "
                f"{res.genres[0]['name']} tags: " + ", ".join(strays[:6])
                + ("…" if len(strays) > 6 else "") + ".")
    before_era = len(pool)
    pool = [t for t in pool if _in_era(t, era)]
    outside_era = before_era - len(pool)
    radio = [t for t in pool if not _NON_RADIO.search(
        norm(f"{t['title']} {t.get('version') or ''}"))]
    non_radio = len(pool) - len(radio)
    pool, dupes = _dedupe(radio)
    res.pool = pool
    counts: dict[str, int] = {}
    for t in pool:
        counts[t["artist"]] = counts.get(t["artist"], 0) + 1

    if res.strategy != STRATEGY_ARTIST:
        # Category stations: every artist with matching material is on the
        # roster; deeper representation earns a little more airtime (log
        # scale), never proportionally more.
        res.roster = {a: {"role": ROLE_CATEGORY,
                          "weight": round(1.0 + math.log2(n), 3),
                          "because": [res.strategy]}
                      for a, n in counts.items()}
        default_cap = _cap_for(count, ARTIST_CAP_SHARE)
        res.caps = {a: default_cap for a in res.roster}
    else:
        res.roster = {a: r for a, r in res.roster.items() if counts.get(a)}
        res.caps = {a: (_cap_for(count, SEED_CAP_SHARE)
                        if r["role"] == ROLE_SEED
                        else _cap_for(count, ARTIST_CAP_SHARE))
                    for a, r in res.roster.items()}
    for a, r in res.roster.items():
        r["pool_tracks"] = counts.get(a, 0)

    res.stats = {"pool_tracks": len(pool), "pool_artists": len(counts),
                 "duplicates_rejected": dupes,
                 "non_radio_skipped": non_radio,
                 "outside_era": outside_era,
                 "unplayable_skipped": unplayable,
                 "library_tracks": len(snap.tracks)}
    if not pool:
        res.status = EMPTY_POOL
        res.notes.append("The library has no playable tracks for this "
                         "station" + (f" in the {era['label']}." if era
                                      else "."))
        return res
    if len(counts) == 1:
        only = next(iter(counts))
        res.warn("single_artist_catalog",
                 f"Every matching track is by {only}. A station needs "
                 f"several artists, so this preview is short on purpose.")
    elif len(counts) < 5:
        res.warn("few_artists",
                 f"Only {len(counts)} artists in the library match, so the "
                 f"station will repeat them often.")
    res.status = OK
    return res


def _artist_roster(res: Resolution, snap: ma_catalog.Snapshot, seed: str,
                   api_fn, knowledge_fn) -> None:
    """Seed + grounded relatives. Every edge carries where it came from."""
    roster: dict[str, dict[str, Any]] = {
        seed: {"role": ROLE_SEED, "weight": SEED_WEIGHT,
               "because": ["seed"]}}
    try:
        graph, partial = ma_catalog.similarity_graph(snap, api_fn)
    except ma_catalog.CatalogError:
        graph, partial = {}, True
    if partial:
        res.warn("relations_partial",
                 "Music Assistant's similar-artist data could not be read "
                 "for every artist, so relatives may be missing.")
    library = sorted({t["artist"] for t in snap.tracks})
    direct: dict[str, set[str]] = {a: set(p) for a, p in
                                   (graph.get(seed) or {}).items()}
    for other in library:
        if other == seed:
            continue
        rel = knowledge_fn(seed, other)
        if rel:
            kind = rel.get("kind") or "direct"
            direct.setdefault(other, set()).add(f"knowledge_pack_{kind}")
    for other, prov in direct.items():
        strong = any(p.startswith("ma_") for p in prov) and any(
            p.startswith("knowledge_pack") for p in prov)
        roster[other] = {"role": ROLE_RELATED,
                         "weight": STRONG_RELATED_WEIGHT if strong
                         else RELATED_WEIGHT,
                         "because": sorted(prov)}
    res.notes.append(f"{len(direct)} related artists from Music Assistant "
                     "similar-artist data and the Music Knowledge pack.")

    if len(direct) < MIN_RELATED:
        links: dict[str, set[str]] = {}
        for rel in direct:
            for other in (graph.get(rel) or {}):
                if other not in roster:
                    links.setdefault(other, set()).add(rel)
        second = sorted(links, key=lambda a: (-len(links[a]), a))
        for other in second[:MAX_SECOND_DEGREE]:
            roster[other] = {"role": ROLE_SECOND,
                             "weight": SECOND_DEGREE_WEIGHT,
                             "because": [f"related_to:{r}"
                                         for r in sorted(links[other])]}
        if second:
            res.notes.append(f"Widened by {min(len(second), MAX_SECOND_DEGREE)} "
                             "artists related to those relatives.")

    if len(roster) - 1 < MIN_RELATED:
        ids = {a["name"]: a["id"] for a in snap.artists}
        seed_genres = set(ma_catalog.artist_genres(snap, ids[seed], api_fn)) \
            if seed in ids else set()
        if seed_genres:
            shared: dict[str, set[str]] = {}
            for other in library:
                if other in roster or other not in ids:
                    continue
                g = set(ma_catalog.artist_genres(snap, ids[other], api_fn))
                if g & seed_genres:
                    shared[other] = g & seed_genres
            # Most specific overlap first: sharing two genres says more than
            # sharing the ubiquitous one.
            ranked = sorted(shared, key=lambda a: (-len(shared[a]), a))
            for other in ranked[:MAX_GENRE_NEIGHBORS]:
                roster[other] = {"role": ROLE_NEIGHBOR,
                                 "weight": GENRE_NEIGHBOR_WEIGHT,
                                 "because": [f"shares_genre:{g}"
                                             for g in sorted(shared[other])]}
            if ranked:
                res.warn("thin_relations",
                         f"Music Assistant knows few relatives of {seed}; the "
                         f"station also draws on artists sharing its genre "
                         f"({', '.join(sorted(seed_genres))}), which is a "
                         f"looser fit.")
    others = sum(r["weight"] for a, r in roster.items() if a != seed)
    if others:
        roster[seed]["weight"] = round(max(
            SEED_WEIGHT,
            others * SEED_TARGET_SHARE / (1.0 - SEED_TARGET_SHARE)), 3)
    if len(roster) == 1:
        res.warn("no_relatives",
                 f"Nothing in the library is documented as related to "
                 f"{seed}, so this preview is short rather than all {seed}.")
    res.roster = roster


# --- one batch -----------------------------------------------------------------

def _scores(pool: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The selector's live hard rules (skits, intros, out-of-season holiday
    songs, heavily penalised material) need a score row per track. The
    local catalog has one in its DB; an MA-sourced pool gets the SAME
    classifier the DB rows were built with, applied to the MA metadata."""
    from musicdb import classify_track
    out = {}
    for t in pool:
        score, intro, skit, holiday = classify_track(
            {"artist": t["artist"], "title": t["title"],
             "album": t.get("album") or ""})
        out[t["path"]] = {"public_score": score, "is_intro": intro,
                          "is_skit": skit, "holiday_tag": holiday}
    return out


def build_candidates(res: Resolution, *, count: int, rng: random.Random,
                     now: datetime,
                     hard_excluded_ids=(), recent_played=(),
                     recently_generated=(), feel: str = "balanced",
                     same_artist: dict[str, Any] | None = None,
                     features: dict[str, Any] | None = None
                     ) -> dict[str, Any]:
    """ONE batch of a resolved station. PURE apart from the caller's rng.

    This is the function a live refill will call. The preview passes empty
    exclusions because nothing has played yet; a running station passes
    what is queued (hard_excluded_ids), what has played (recent_played) and
    what it generated before (recently_generated) -- the selector's own
    rotation contract, unchanged."""
    import artist_policy
    import station_feel
    selection = station_selector.select_station_refill(
        artist_weights=res.weights, request_text=res.intent.query,
        catalog_tracks=res.pool, scores=_scores(res.pool), rng=rng,
        count=count, now=now, hard_excluded_ids=hard_excluded_ids,
        recent_played=recent_played, recently_generated=recently_generated,
        policy=station_feel.policy_for(feel), features=features,
        artist_caps=res.caps, normalize_by_artist_pool=True)
    sa = same_artist or {}
    adjacency = artist_policy.enforce(
        selection.tracks, enabled=bool(sa.get("enabled", True)),
        frequency=sa.get("frequency", "medium"))
    meta = selection.as_dict()
    meta.pop("tracks", None)
    return {"tracks": adjacency["order"], "selection": meta,
            "adjacency": adjacency["telemetry"], "feel": feel}


# --- the minimum viable station ------------------------------------------------

def _songs(n: int) -> str:
    return f"{n} song{'' if n == 1 else 's'}"


def live_viability(res: Resolution, opening_count: int) -> dict[str, Any]:
    """Could this station run live, endlessly? {viable, reasons, rule}.

    PURE, and the ONE rule: the preview reports it and station_live refuses
    a start that fails it. `opening_count` is how many songs the opening
    batch actually held after the per-artist caps. Each reason carries a
    sentence a person is shown."""
    rule = {"min_pool_tracks": LIVE_MIN_POOL_TRACKS,
            "min_artists": LIVE_MIN_ARTISTS,
            "min_opening": LIVE_MIN_OPENING}
    if res.status != OK:
        return {"viable": False, "rule": rule, "reasons": [
            {"code": res.status,
             "message": "There is no station to run: "
                        + (" ".join(res.notes) or res.status)}]}
    pool = int(res.stats.get("pool_tracks") or 0)
    artists = int(res.stats.get("pool_artists") or 0)
    reasons = []
    if pool < LIVE_MIN_POOL_TRACKS:
        reasons.append({
            "code": "pool_too_small",
            "message": f"Only {_songs(pool)} in the library fit this station. "
                       f"A station that runs for hours needs at least "
                       f"{LIVE_MIN_POOL_TRACKS}, or it starts repeating "
                       f"itself."})
    if artists < LIVE_MIN_ARTISTS:
        reasons.append({
            "code": "too_few_artists",
            "message": f"Only {artists} "
                       f"{'artist fits' if artists == 1 else 'artists fit'} "
                       f"this station; a live station needs at least "
                       f"{LIVE_MIN_ARTISTS}."})
    if opening_count < LIVE_MIN_OPENING:
        reasons.append({
            "code": "opening_too_short",
            "message": f"Only {_songs(opening_count)} could be programmed "
                       f"within the per-artist limits; a live station opens "
                       f"with at least {LIVE_MIN_OPENING}."})
    return {"viable": not reasons, "reasons": reasons, "rule": rule}


# --- preview (dry run) -----------------------------------------------------------

def seed_for(intent: station_intent.StationIntent, variant: int) -> int:
    """The deterministic draw for (station, variant). Test Station and a
    live start share it, so a start from a tested preset opens with the
    run that was tested unless the live queue excludes some of it."""
    digest = hashlib.sha256(f"{intent.key()}#{variant}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _why(track: dict[str, Any], res: Resolution) -> str:
    r = res.roster.get(track["artist"]) or {}
    role = r.get("role")
    if role == ROLE_SEED:
        return "seed artist"
    if role == ROLE_RELATED:
        src = []
        for p in r.get("because", []):
            if p.startswith("ma_similar"):
                src.append("Music Assistant similar artists")
            elif p.startswith("knowledge_pack"):
                src.append("shared members (Music Knowledge)")
        return f"related to {res.seed_artist}" + (
            f" ({', '.join(sorted(set(src)))})" if src else "")
    if role == ROLE_SECOND:
        via = [p.split(":", 1)[1] for p in r.get("because", [])
               if p.startswith("related_to:")]
        return "related via " + ", ".join(via[:2])
    if role == ROLE_NEIGHBOR:
        return "shares a genre with " + str(res.seed_artist)
    # The year is already beside the album; the why names the RULE.
    bits = [g["name"] for g in res.genres]
    if res.era:
        bits.append(res.era["label"])
    return " · ".join(bits) or "in the era"


def _quality(res: Resolution, tracks: list[dict[str, Any]], count: int
             ) -> list[dict[str, str]]:
    out = []
    n = len(tracks)
    if n and n < min(20, count):
        out.append({"code": "short",
                    "message": f"Only {n} of {count} tracks could be "
                               "programmed within the diversity limits."})
    if n:
        top = max(sum(1 for t in tracks if t["artist"] == a)
                  for a in {t["artist"] for t in tracks})
        if n >= 5 and top / n > 0.4:
            out.append({"code": "one_artist_dominates",
                        "message": "One artist is more than 40% of this "
                                   "preview."})
    return out


def settings_view(settings_fn=None) -> tuple[str, dict[str, Any]]:
    """Feel and same-artist policy, READ from settings (never written).
    Shared with station_live so a live batch honours exactly what the
    preview honoured."""
    try:
        if settings_fn is not None:
            doc = settings_fn()
        else:
            import settings
            doc = settings.load()
        station = doc.get("station") or {}
        import station_feel
        return (station_feel.current_feel(doc),
                dict(station.get("same_artist") or {}))
    except Exception:  # noqa: BLE001 - defaults are a faithful preview too
        return "balanced", {}


def preview_station(intent_data: Any, *, count: int = DEFAULT_COUNT,
                    variant: int = 0, api_fn=None, settings_fn=None,
                    knowledge_fn=None, now: datetime | None = None,
                    snapshot_fn=None) -> dict[str, Any]:
    """TEST STATION: resolve and build the first batch, and change nothing.

    No queue read, no queue write, no transport, no player selection, no
    settings write: the only side-effect-free Music Assistant access is
    ma_catalog's allowlisted library listings. The answer is the batch a
    real start would open with, plus everything needed to judge it."""
    try:
        intent = (intent_data if isinstance(intent_data,
                                            station_intent.StationIntent)
                  else station_intent.intent_from_dict(intent_data))
    except station_intent.IntentError as exc:
        return {"ok": False, "dry_run": True, "status": INVALID,
                "message": str(exc)}
    try:
        count = max(MIN_COUNT, min(MAX_COUNT, int(count)))
        variant = max(0, min(9999, int(variant)))
    except (TypeError, ValueError):
        return {"ok": False, "dry_run": True, "status": INVALID,
                "message": "count and variant must be numbers"}
    base = {"dry_run": True, "intent": intent.to_dict(), "count": count,
            "variant": variant}
    if not station_intent.KINDS[intent.kind]["buildable"]:
        return dict(base, ok=False, status=NOT_AVAILABLE,
                    message="All News is a planned station type. It can be "
                            "saved to the Dial, but it cannot be built or "
                            "previewed yet.")
    try:
        if snapshot_fn is not None:
            snap, cached = snapshot_fn(), False
        else:
            snap, cached = ma_catalog.snapshot(api_fn)
    except ma_catalog.CatalogError as exc:
        return dict(base, ok=False, status=CATALOG_UNAVAILABLE,
                    message="The Music Assistant library could not be "
                            "read, so no station could be built.",
                    detail=str(exc))
    res = resolve_station(intent, snap, count=count, api_fn=api_fn,
                          knowledge_fn=knowledge_fn)
    catalog = dict(snap.describe(), cached=cached)
    if res.status != OK:
        return dict(base, ok=False, status=res.status,
                    resolution=res.describe(), catalog=catalog,
                    live=live_viability(res, 0),
                    message=" ".join(res.notes) or res.status)
    feel, same_artist = settings_view(settings_fn)
    batch = build_candidates(res, count=count,
                             rng=random.Random(seed_for(intent, variant)),
                             now=now or service_clock.legacy_now(), feel=feel,
                             same_artist=same_artist)
    tracks = batch["tracks"]
    roles = [(res.roster.get(t["artist"]) or {}).get("role") for t in tracks]
    per_artist: dict[str, int] = {}
    for t in tracks:
        per_artist[t["artist"]] = per_artist.get(t["artist"], 0) + 1
    summary = {
        "total_tracks": len(tracks),
        "unique_artists": len(per_artist),
        "seed_matches": roles.count(ROLE_SEED),
        "related_matches": sum(roles.count(r) for r in RELATED_ROLES),
        "category_matches": roles.count(ROLE_CATEGORY),
        "duplicates_rejected": res.stats.get("duplicates_rejected", 0),
        "pool_tracks": res.stats.get("pool_tracks", 0),
        "pool_artists": res.stats.get("pool_artists", 0),
        "most_from_one_artist": max(per_artist.values()) if per_artist else 0,
        "relaxation_pass": batch["selection"].get("relaxation_pass"),
        "exhausted": batch["selection"].get("exhausted"),
    }
    rows = [{"position": i + 1, "title": t["title"], "artist": t["artist"],
             "album": t.get("album") or "", "year": t.get("year"),
             "uri": t["uri"], "provider": t.get("provider") or "",
             "role": (res.roster.get(t["artist"]) or {}).get("role"),
             "why": _why(t, res)}
            for i, t in enumerate(tracks)]
    return dict(base, ok=True, status=OK, summary=summary, tracks=rows,
                resolution=res.describe(), catalog=catalog,
                quality=_quality(res, tracks, count),
                live=live_viability(res, len(tracks)),
                selection=batch["selection"], adjacency=batch["adjacency"],
                feel=feel)


def last_test_record(result: dict[str, Any], at: str) -> dict[str, Any] | None:
    """The small, storable record of a successful preview, for a preset's
    `last_test`. Never the tracks themselves. It carries the live verdict
    (`viable`, and `weak` = the first reason it failed) so a thin station
    keeps looking thin on the Dial, and the approved draw (`variant`)."""
    if not result.get("ok"):
        return None
    s = result.get("summary") or {}
    live = result.get("live") or {}
    reasons = live.get("reasons") or []
    try:
        variant = max(0, min(9999, int(result.get("variant") or 0)))
    except (TypeError, ValueError):
        variant = 0
    return {"query": (result.get("intent") or {}).get("query", ""),
            "at": at[:40], "status": result.get("status", "")[:40],
            "strategy": str((result.get("resolution") or {}).get(
                "strategy") or "")[:40],
            "tracks": int(s.get("total_tracks") or 0),
            "artists": int(s.get("unique_artists") or 0),
            "viable": bool(live.get("viable")),
            "weak": str((reasons[0] or {}).get("code") or "")[:40]
                    if reasons else "",
            "variant": variant,
            "builder_revision": STATION_BUILDER_REVISION}
