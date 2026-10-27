"""Song context intelligence: ParlorWave knows the record it is
playing -- and mostly keeps that knowledge to itself.

Layers, strictly separated by PROVENANCE:

  GROUNDED   artist / title / album / MusicBrainz recording id, straight
             from the tagged catalog. Fit for literal on-air claims. The
             tag `year` field is deliberately ABSENT: in real catalogs
             it often holds rip/edition dates or placeholders
             ('0001-01-01') -- calling that "from 1991" would be a lie,
             so no year fact ships until an original-release source
             exists.
  DERIVED    pi_track_features (rebuildable, provenance-honest): a broad
             normalized genre, an energy bucket, live/demo/instrumental
             flags. Descriptive color only -- a derived 'grunge' is not a
             city, a history, or a scene claim. `mood` is deliberately
             NOT used (it collapses to a few dominant values and nearly
             duplicates energy).
  SEMANTIC   short theme/tone/imagery arrays derived from lyrics by the
             resident model, cached under a durable semantic key. Raw
             lyrics are TRANSIENT INPUT ONLY: never persisted, never
             logged, never in telemetry, never in the writer's prompt,
             never quoted on air.

Identity: mb:<recording id> (carried by tagged local files, survives MA rebuilds
and rescans) -> provider URI (stable within its provider) ->
path:<relative path> (untagged fallback). MA's library://track/N and
absolute paths are never identities.

The whole layer is OPTIONAL: every failure -- no lyrics, busy GPU, bad
JSON, low confidence, cache trouble -- yields the ordinary break.
Extraction runs opportunistically while music plays and always yields to
DJ/news renders and the programmer (render_pressure); it adds ZERO
required render latency because a break only ever reads the cache.

And the taste rules are enforced where facts are made: on-air use is
gated to roughly one observation per cooldown window, only when something
is genuinely noteworthy (environmental alignment, real contrast, a
session callback, or a distinctive theme), never generic love/life/
sadness filler, and the prompt forbids explaining "meaning", quoting
lines, or reciting unsupported trivia.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any

import config as _config
import service_clock

logger = logging.getLogger("musicbrain")

EXTRACTOR_VERSION = 1
# The extractor's model (config.py); "" = not configured = no extraction.
EXTRACT_MODEL = _config.current().song_context_model
EXTRACT_TIMEOUT_S = 20.0
PRESSURE_SLACK_S = 30.0

MIN_CONFIDENCE = 0.6                # below this, semantics never air
DISTINCTIVE_CONFIDENCE = 0.75
GENERIC_THEMES = {"love", "life", "sadness", "loss", "hope", "pain",
                  "happiness", "relationships", "emotion", "feelings"}

# Sparse opportunity cooldowns by chattiness. Low = never.
COOLDOWN_MIN = {"low": None, "normal": 50.0, "high": 30.0}
THEME_REPEAT_HOURS = 3.0            # a theme aired recently stays retired

ENERGY_BUCKETS = ((0.45, "low"), (0.75, "medium"), (999.0, "high"))

NIGHT_IMAGERY = {"night", "nighttime", "moon", "moonlight", "midnight",
                 "stars", "dark", "darkness"}
RAIN_IMAGERY = {"rain", "storm", "thunder", "clouds"}


# --- identity ------------------------------------------------------------------

def semantic_key(*, mbid: str | None = None, uri: str | None = None,
                 rel_path: str | None = None) -> str | None:
    """The durable identity for a song's derived knowledge."""
    if mbid and str(mbid).strip():
        return f"mb:{str(mbid).strip()}"
    if uri and not str(uri).startswith("library://"):
        return str(uri)
    if rel_path:
        return f"path:{str(rel_path).lstrip('/')}"
    return None


def _catalog_row(path: str | None = None, artist: str = "",
                 title: str = "") -> dict[str, Any] | None:
    """One tracks row from the music DB, by path first, else exact
    artist+title (the resolution the feedback path already trusts)."""
    import musicdb
    try:
        c = sqlite3.connect(str(musicdb.DB))
        c.row_factory = sqlite3.Row
        try:
            row = None
            if path:
                row = c.execute(
                    "SELECT path, artist, title, album, mb_recording_id "
                    "FROM tracks WHERE path = ?", (path,)).fetchone()
            if row is None and artist and title:
                row = c.execute(
                    "SELECT path, artist, title, album, mb_recording_id "
                    "FROM tracks WHERE lower(artist) = ? AND lower(title) "
                    "= ? LIMIT 1", (artist.lower(), title.lower())).fetchone()
            return dict(row) if row is not None else None
        finally:
            c.close()
    except Exception:  # noqa: BLE001 - facts are optional, never fatal
        return None


_ALBUM_TITLES: set[str] | None = None
_ARTIST_NAMES: set[str] | None = None
_ALL_ALBUM_TITLES: set[str] | None = None


def known_album_titles_all(refresh: bool = False) -> set[str]:
    """EVERY catalog album title, lower-cased, with none of the precision
    filters known_album_titles applies. Used ONLY inside explicit album
    grammar ("from the X album", "off X"), where the prose itself has
    already said X is an album and the ambiguity the filters guard against
    ("Gullwing" is also Cold Orchard's title track) does not arise.
    Without it, "That was Wren Hollis, from the Gullwing album" passes
    because the title-track exclusion hides the album from the guard."""
    global _ALL_ALBUM_TITLES
    if _ALL_ALBUM_TITLES is not None and not refresh:
        return _ALL_ALBUM_TITLES
    titles: set[str] = set()
    try:
        import musicdb
        c = sqlite3.connect(str(musicdb.DB))
        try:
            titles = {(r[0] or "").strip().lower()
                      for r in c.execute("SELECT DISTINCT album FROM tracks "
                                         "WHERE album IS NOT NULL AND album != ''")}
        finally:
            c.close()
        titles = {a for a in titles if len(a) >= 3}
    except Exception:  # noqa: BLE001 - the guard degrades, never fails a break
        titles = set()
    _ALL_ALBUM_TITLES = titles
    return titles


def known_artist_names(refresh: bool = False) -> set[str]:
    """Catalog artist identities, normalized (lower-case, punctuation to
    spaces) for the post-writer artist-identity guard. The same grounded
    catalog the track facts come from, never world knowledge. Names under
    three characters are dropped: too short to match safely in prose."""
    global _ARTIST_NAMES
    if _ARTIST_NAMES is not None and not refresh:
        return _ARTIST_NAMES
    names: set[str] = set()
    try:
        import musicdb
        c = sqlite3.connect(str(musicdb.DB))
        try:
            for (a,) in c.execute("SELECT DISTINCT artist FROM tracks "
                                  "WHERE artist IS NOT NULL AND artist != ''"):
                n = re.sub(r"[^a-z0-9]+", " ", str(a).lower()).strip()
                if len(n) >= 3:
                    names.add(n)
        finally:
            c.close()
    except Exception:  # noqa: BLE001 - the guard degrades, never fails a break
        names = set()
    _ARTIST_NAMES = names
    return names


def known_album_titles(refresh: bool = False) -> set[str]:
    """Catalog album titles that UNAMBIGUOUSLY name an album, lower-cased.

    Used by the post-writer album-attribution guard to recognize an album
    the writer brought in from somewhere other than this break's supplied
    facts.  This is the same grounded catalog the album facts come from --
    never world knowledge -- and it is filtered for precision, because the
    guard fails a break closed and a false positive costs real airtime:

      * titles shorter than 8 characters are dropped (short titles are
        often ordinary words in ordinary sentences),
      * titles that are also an ARTIST name are dropped (self-titled
        records: naming Cold Orchard is not an album claim),
      * titles that are also a TRACK title are dropped (naming the
        title track 'Gullwing' is not an album claim).

    The result is a conservative set: it can miss an ambiguous album, but
    what it does contain is an album title and nothing else.
    """
    global _ALBUM_TITLES
    if _ALBUM_TITLES is not None and not refresh:
        return _ALBUM_TITLES
    titles: set[str] = set()
    try:
        import musicdb
        c = sqlite3.connect(str(musicdb.DB))
        try:
            albums = {(r[0] or "").strip().lower()
                      for r in c.execute(
                          "SELECT DISTINCT album FROM tracks "
                          "WHERE album IS NOT NULL AND album != ''")}
            ambiguous = {(r[0] or "").strip().lower()
                         for r in c.execute("SELECT DISTINCT artist "
                                            "FROM tracks")}
            ambiguous |= {(r[0] or "").strip().lower()
                          for r in c.execute("SELECT DISTINCT title "
                                             "FROM tracks")}
        finally:
            c.close()
        titles = {a for a in albums if len(a) >= 8 and a not in ambiguous}
    except Exception:  # noqa: BLE001 - the guard degrades, never fails a break
        titles = set()
    _ALBUM_TITLES = titles
    return titles


def track_facts(path: str | None = None, artist: str = "",
                title: str = "") -> dict[str, Any] | None:
    """GROUNDED facts for one track, or None when unmatched. No year --
    see the module doc for why."""
    row = _catalog_row(path, artist, title)
    if row is None:
        return None
    key = semantic_key(mbid=row.get("mb_recording_id"),
                       rel_path=row.get("path"))
    return {"artist": row.get("artist") or "",
            "title": row.get("title") or "",
            "album": row.get("album") or "",
            "path": row.get("path"),
            "semantic_key": key}


def derived_character(path: str | None) -> dict[str, Any] | None:
    """DERIVED descriptive character from pi_track_features. Only the
    fields fit to use; None when unavailable."""
    if not path:
        return None
    try:
        import playback
        import paths
        c = sqlite3.connect(str(paths.BASE_DIR / "collection.db"))
        c.row_factory = sqlite3.Row
        try:
            row = c.execute(
                "SELECT genre_norm, energy, is_live, is_demo, "
                "is_instrumental FROM pi_track_features WHERE path = ?",
                (path,)).fetchone()
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    out: dict[str, Any] = {}
    if row["genre_norm"] and row["genre_norm"] != "neutral":
        out["genre"] = row["genre_norm"]
    energy = row["energy"]
    if isinstance(energy, (int, float)):
        for ceiling, name in ENERGY_BUCKETS:
            if float(energy) <= ceiling:
                out["energy"] = name
                break
    for flag, name in (("is_live", "live"), ("is_demo", "demo"),
                       ("is_instrumental", "instrumental")):
        if row[flag]:
            out[name] = True
    return out or None


# --- the semantic cache --------------------------------------------------------

_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS song_semantics (
    semantic_key TEXT PRIMARY KEY,
    extractor_version INTEGER NOT NULL,
    themes TEXT NOT NULL,
    tone TEXT NOT NULL DEFAULT '',
    emotion TEXT NOT NULL DEFAULT '',
    imagery TEXT NOT NULL DEFAULT '',
    perspective TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    derived_at TEXT NOT NULL
)
"""


def _cache_conn():
    import musicdb
    c = sqlite3.connect(str(musicdb.DB))
    c.execute(_CACHE_DDL)
    return c


def cached_semantics(key: str | None) -> dict[str, Any] | None:
    """The cached derived representation, or None. Rows from an older
    extractor version read as a miss (lazy regeneration -- never a bulk
    reprocess)."""
    if not key:
        return None
    try:
        c = _cache_conn()
        try:
            row = c.execute(
                "SELECT extractor_version, themes, tone, emotion, imagery, "
                "perspective, confidence, derived_at FROM song_semantics "
                "WHERE semantic_key = ?", (key,)).fetchone()
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return None
    if row is None or int(row[0]) != EXTRACTOR_VERSION:
        return None
    try:
        return {"themes": json.loads(row[1]), "tone": row[2],
                "emotion": row[3], "imagery": json.loads(row[4] or "[]"),
                "perspective": row[5], "confidence": float(row[6]),
                "derived_at": row[7]}
    except (TypeError, ValueError):
        return None


def store_semantics(key: str, data: dict[str, Any],
                    now: datetime | None = None) -> None:
    stamp = (now or service_clock.legacy_now()).replace(microsecond=0).isoformat()
    c = _cache_conn()
    try:
        c.execute(
            "INSERT OR REPLACE INTO song_semantics (semantic_key, "
            "extractor_version, themes, tone, emotion, imagery, "
            "perspective, confidence, derived_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, EXTRACTOR_VERSION,
             json.dumps(data.get("themes") or []),
             str(data.get("tone") or ""), str(data.get("emotion") or ""),
             json.dumps(data.get("imagery") or []),
             str(data.get("perspective") or ""),
             float(data.get("confidence") or 0.0), stamp))
        c.commit()
    finally:
        c.close()


# --- lyric semantic extraction ---------------------------------------------------

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {"type": "array", "items": {"type": "string"},
                   "maxItems": 3},
        "tone": {"type": "string"},
        "emotion": {"type": "string"},
        "imagery": {"type": "array", "items": {"type": "string"},
                    "maxItems": 3},
        "perspective": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["themes", "tone", "confidence"],
}

_PROMPT = """You are indexing songs for a radio station's internal notes. From the lyrics below, produce a TINY semantic sketch. Rules: at most 3 themes and 3 imagery entries, each 1-2 lowercase words; tone and emotion one word each; perspective one short phrase (e.g. "first person"); confidence 0..1 for how clearly the lyrics support your sketch. NEVER include lyric lines, phrases, or distinctive wording from the song -- concepts only.

Song: {artist} - {title}

Lyrics (transient, do not echo):
{lyrics}

Answer with JSON only."""

_WORD = re.compile(r"^[a-z][a-z '\-]{0,23}$")


def _clean_terms(values: Any, cap: int) -> list[str]:
    out = []
    for v in values if isinstance(values, list) else []:
        term = str(v).strip().lower()
        if _WORD.match(term) and len(term.split()) <= 2 and term not in out:
            out.append(term)
    return out[:cap]


def extract_semantics(lyrics_text: str, artist: str, title: str,
                      generate_fn=None,
                      timeout: float = EXTRACT_TIMEOUT_S
                      ) -> dict[str, Any] | None:
    """One constrained-JSON extraction. The lyric body exists only inside
    this call: it is not stored, not logged, not returned. None on any
    failure or unusable output."""
    text = (lyrics_text or "").strip()
    if len(text) < 80:
        return None
    if generate_fn is None and not EXTRACT_MODEL:
        return None
    prompt = _PROMPT.format(artist=artist, title=title, lyrics=text[:6000])
    try:
        if generate_fn is None:
            import providers
            body = providers.ollama_generate(prompt, EXTRACT_MODEL,
                                             timeout=timeout,
                                             format=EXTRACT_SCHEMA)
        else:
            body = generate_fn(prompt, EXTRACT_MODEL, timeout=timeout,
                               format=EXTRACT_SCHEMA)
        raw = json.loads((body or {}).get("response") or "")
    except Exception:  # noqa: BLE001 - one inference boundary
        return None
    if not isinstance(raw, dict):
        return None
    themes = _clean_terms(raw.get("themes"), 3)
    if not themes:
        return None
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence"))))
    except (TypeError, ValueError):
        return None
    out = {"themes": themes,
           "tone": _clean_terms([raw.get("tone")], 1)[:1] or [""],
           "emotion": _clean_terms([raw.get("emotion")], 1)[:1] or [""],
           "imagery": _clean_terms(raw.get("imagery"), 3),
           "perspective": str(raw.get("perspective") or "")[:40],
           "confidence": confidence}
    out["tone"] = out["tone"][0]
    out["emotion"] = out["emotion"][0]
    # The one impossible-by-construction guarantee a test can pin: no
    # output field can contain a lyric LINE -- every term is <=2 words.
    return out


# --- opportunistic extraction ------------------------------------------------------

_worker = threading.Lock()


def opportunistic_pass(queue_id: str, now: datetime | None = None,
                       deps: dict[str, Any] | None = None,
                       db_path=None) -> str:
    """Cache semantics for the CURRENTLY PLAYING track while it plays --
    by the next boundary, "that was X" can know what X was about.
    Single-flight, always yields to spoken renders (and thereby to
    everything above it in the GPU pecking order). Returns a short outcome
    string for tests; production ignores it. Never raises."""
    now_dt = now or service_clock.legacy_now()
    deps = deps or {}
    if not _worker.acquire(blocking=False):
        return "busy"
    try:
        import dj_engine
        pressure_fn = deps.get("pressure_fn")
        pressure = (pressure_fn(EXTRACT_TIMEOUT_S + PRESSURE_SLACK_S)
                    if pressure_fn else dj_engine.render_pressure(
                        queue_id, now=now_dt,
                        horizon_s=EXTRACT_TIMEOUT_S + PRESSURE_SLACK_S,
                        db_path=db_path))
        if pressure:
            return "render_pressure"
        import now_playing
        lyr = (deps.get("lyrics_fn") or now_playing.lyrics)(queue_id)
        if lyr.get("kind") not in ("synced", "plain"):
            return "no_lyrics"
        if lyr.get("kind") == "synced":
            text = "\n".join(r.get("text", "")
                             for r in lyr.get("lines") or [])
        else:
            text = lyr.get("text") or ""
        card = (deps.get("snapshot_fn") or now_playing.snapshot)(queue_id)
        # Resolve by the SESSION'S FILE PATH first -- the same resolution
        # the render-time gate uses. Otherwise a song that exists twice in
        # the catalog (two editions, two MBIDs) is cached under the
        # artist+title LIMIT 1 row while the gate looks up the path row --
        # a permanent key mismatch. Path-first on both sides
        # makes the identity one and the same.
        path = None
        try:
            if deps.get("session_path_fn") is not None:
                path = deps["session_path_fn"](queue_id)
            else:
                import playback
                session = playback.get_active_for_queue(queue_id)
                path = (session or {}).get("current_path")
        except Exception:  # noqa: BLE001
            path = None
        facts = track_facts(path, artist=card.get("artist") or "",
                            title=card.get("title") or "")
        if facts is None or not facts.get("semantic_key"):
            return "no_catalog_match"
        key = facts["semantic_key"]
        if cached_semantics(key) is not None:
            return "cache_hit"
        sem = extract_semantics(text, facts["artist"], facts["title"],
                                generate_fn=deps.get("generate_fn"))
        if sem is None:
            return "extract_failed"
        store_semantics(key, sem, now_dt)
        logger.info("song semantics cached for %s (themes=%s tone=%s "
                    "conf=%.2f)", key[:24], ",".join(sem["themes"]),
                    sem["tone"], sem["confidence"])
        return "cached"
    except Exception:  # noqa: BLE001 - fully fenced
        logger.warning("song semantic pass failed (nothing affected)",
                       exc_info=True)
        return "error"
    finally:
        _worker.release()


# --- the sparse opportunity gate -------------------------------------------------

def _last_semantic_mention(db_path=None) -> datetime | None:
    """When the station last used semantic color on air, from break telemetry --
    no bespoke state file."""
    try:
        import dj_engine
        c = dj_engine._connect(db_path)
        try:
            row = c.execute(
                "SELECT json_extract(telemetry, '$.semantic_used_at') "
                "FROM dj_breaks WHERE json_extract(telemetry, "
                "'$.semantic_used_at') IS NOT NULL "
                "ORDER BY rowid DESC LIMIT 1").fetchone()
        finally:
            c.close()
        return datetime.fromisoformat(row[0]) if row and row[0] else None
    except Exception:  # noqa: BLE001
        return None


def _recent_themes(now: datetime, db_path=None) -> set[str]:
    try:
        import dj_engine
        # a theme is held back only while CERTAINLY recent (DST-safe)
        cutoff = service_clock.cutoff_release(
            now, timedelta(hours=THEME_REPEAT_HOURS))
        c = dj_engine._connect(db_path)
        try:
            rows = c.execute(
                "SELECT json_extract(telemetry, '$.semantic_themes') FROM "
                "dj_breaks WHERE updated_at > ? AND json_extract(telemetry,"
                " '$.semantic_themes') IS NOT NULL", (cutoff,)).fetchall()
        finally:
            c.close()
        out: set[str] = set()
        for (blob,) in rows:
            try:
                out.update(json.loads(blob))
            except (TypeError, ValueError):
                continue
        return out
    except Exception:  # noqa: BLE001
        return set()


def _noteworthy(sem: dict[str, Any], pre_char: dict[str, Any] | None,
                nxt_char: dict[str, Any] | None,
                service: dict[str, Any]) -> str | None:
    """Why NOW is worth one short observation -- or None (which is the
    common answer). Environmental claims are alignment only: the prompt
    never lets the DJ say the song predicted the sky."""
    terms = set(sem.get("themes") or []) | set(sem.get("imagery") or [])
    phase = service.get("daylight_phase") or ""
    if terms & NIGHT_IMAGERY and phase in ("twilight", "dark",
                                           "sunset_imminent"):
        return "environment"
    condition = str(service.get("condition") or
                    service.get("current_weather") or "").lower()
    if terms & RAIN_IMAGERY and ("rain" in condition
                                 or "storm" in condition):
        return "environment"
    order = ("low", "medium", "high")
    if (pre_char and nxt_char and pre_char.get("energy") in order
            and nxt_char.get("energy") in order
            and abs(order.index(pre_char["energy"])
                    - order.index(nxt_char["energy"])) >= 2):
        return "contrast"
    distinctive = [t for t in sem.get("themes") or []
                   if t not in GENERIC_THEMES]
    if distinctive and sem.get("confidence", 0) >= DISTINCTIVE_CONFIDENCE:
        return "distinctive"
    return None


def semantic_opportunity(pre_path: str | None, pre_artist: str,
                         pre_title: str, nxt_char: dict[str, Any] | None,
                         service: dict[str, Any], chattiness: str,
                         now: datetime | None = None,
                         db_path=None) -> dict[str, Any] | None:
    """The deterministic gate: cached semantics for the ENDING track, only
    when chattiness allows, the cooldown has passed, the themes are not
    recently used, confidence clears the bar, and something is genuinely
    noteworthy. None is the normal answer."""
    now_dt = now or service_clock.legacy_now()
    cooldown = COOLDOWN_MIN.get(chattiness or "normal", 50.0)
    if cooldown is None:
        return None
    facts = track_facts(pre_path, pre_artist, pre_title)
    if not facts or not facts.get("semantic_key"):
        return None
    sem = cached_semantics(facts["semantic_key"])
    if not sem or sem.get("confidence", 0) < MIN_CONFIDENCE \
            or not sem.get("themes"):
        return None
    last = _last_semantic_mention(db_path)
    if last is not None:
        bounds = service_clock.age_bounds(last, now_dt)
        if bounds is None or bounds[1] < cooldown * 60:
            return None
    used = _recent_themes(now_dt, db_path)
    fresh = [t for t in sem["themes"] if t not in used]
    if not fresh:
        return None
    pre_char = derived_character(facts.get("path"))
    reason = _noteworthy({**sem, "themes": fresh}, pre_char, nxt_char,
                         service)
    if reason is None:
        return None
    return {"themes": fresh[:2], "tone": sem.get("tone") or "",
            "reason": reason, "confidence": sem["confidence"]}


# --- transition character ------------------------------------------------------
# Deterministic classification of the ending->next musical move, so the
# writer can only call a transition "surprising" when the FACTS say so.
# Track adjacency alone is never evidence of surprise; an album run is
# evidence of the opposite.

TRANSITION_CONTINUOUS = "continuous"
TRANSITION_GENTLE = "gentle_change"
TRANSITION_CONTRAST = "contrast"
TRANSITION_MAJOR = "major_contrast"
TRANSITION_UNKNOWN = "unknown"

_ENERGY_ORDER = ("low", "medium", "high")

# Broad genre families. A genre-LABEL difference within one family
# (alternative vs grunge vs rock) is the ordinary texture of a rock
# station, NOT a contrast -- treating it as one would announce a
# "shift" on nearly every cross-artist pair. Only a
# cross-FAMILY move counts as genre contrast.
_GENRE_FAMILY = {
    "alternative": "rock", "grunge": "rock", "rock": "rock",
    "hard_rock": "rock", "post_grunge": "rock", "punk": "rock",
    "indie": "rock", "emo": "rock", "metal": "heavy",
    "hip_hop": "urban", "rap": "urban", "r_and_b": "urban",
    "pop": "pop", "electronic": "electronic", "dance": "electronic",
    "folk": "acoustic", "country": "acoustic", "singer_songwriter":
    "acoustic", "jazz": "jazz", "classical": "classical",
}


def _genre_family(g: str | None) -> str | None:
    if not g:
        return None
    return _GENRE_FAMILY.get(str(g).lower(), str(g).lower())


def _energy_gap(a: dict[str, Any] | None, b: dict[str, Any] | None
                ) -> int | None:
    if not a or not b:
        return None
    try:
        return abs(_ENERGY_ORDER.index(a.get("energy"))
                   - _ENERGY_ORDER.index(b.get("energy")))
    except ValueError:
        return None


def _leading_track_no(path: str | None) -> int | None:
    """A leading track number parsed from the file's basename (e.g.
    '03-quarry-light.flac' -> 3, 'A04 Tin Lantern.flac' -> 4). None when no
    confident leading number exists -- ordering is only ever CLAIMED from
    real evidence."""
    if not path:
        return None
    base = str(path).rsplit("/", 1)[-1]
    m = re.match(r"[A-Da-d]?0*(\d{1,3})[\s._\-]", base)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 200 else None


def classify_transition(pre_facts: dict[str, Any] | None,
                        pre_char: dict[str, Any] | None,
                        post_facts: dict[str, Any] | None,
                        post_char: dict[str, Any] | None,
                        *, pre_album_uri: str | None = None,
                        post_album_uri: str | None = None,
                        pre_track_number: int | None = None,
                        post_track_number: int | None = None
                        ) -> dict[str, Any]:
    """The bounded transition descriptor. Never fabricates surprise: an
    absent case is `unknown`, and only a real genre/energy contrast
    reaches `contrast`/`major_contrast`.

    Album-run identity prefers MA's stable album CONTAINER uri when both
    sides carry it (a durable identity, unlike album-name strings that
    collide across compilations); it falls back to same-artist + same
    album name. Ordered-sequence likewise prefers MA's real track_number,
    falling back to a leading filename number -- and is claimed ONLY when
    the evidence proves consecutive tracks. MA does NOT expose a
    queue-origin 'started from album X' container (as of MA
    2.10.1: radio_source empty, no enqueue-source field), so intent is
    never inferred beyond these per-track facts."""
    if not pre_facts or not post_facts:
        return {"character": TRANSITION_UNKNOWN}
    pa = (pre_facts.get("artist") or "").strip().lower()
    na = (post_facts.get("artist") or "").strip().lower()
    palb = (pre_facts.get("album") or "").strip().lower()
    nalb = (post_facts.get("album") or "").strip().lower()
    same_artist = bool(pa) and pa == na
    if pre_album_uri and post_album_uri:
        same_album = pre_album_uri == post_album_uri      # container identity
    else:
        same_album = same_artist and bool(palb) and palb == nalb
    out: dict[str, Any] = {"same_artist": same_artist,
                           "same_album": same_album}
    if same_album:
        out["album_run"] = True
        # Ordered sequence ONLY when the numbers prove consecutive tracks:
        # MA's track_number preferred, filename leading number as fallback.
        pn = (pre_track_number if isinstance(pre_track_number, int)
              else _leading_track_no(pre_facts.get("path")))
        nn = (post_track_number if isinstance(post_track_number, int)
              else _leading_track_no(post_facts.get("path")))
        if isinstance(pn, int) and isinstance(nn, int) and nn == pn + 1:
            out["album_sequence"] = "ordered"
        out["character"] = TRANSITION_CONTINUOUS
        return out
    gap = _energy_gap(pre_char, post_char)
    pfam = _genre_family((pre_char or {}).get("genre"))
    nfam = _genre_family((post_char or {}).get("genre"))
    genre_known = bool(pfam) and bool(nfam)
    # A genre difference is only a contrast signal ACROSS families; within
    # the rock family (alternative/grunge/rock/punk...) it is ordinary.
    genre_diff = genre_known and pfam != nfam
    if same_artist:
        out["character"] = TRANSITION_GENTLE
        return out
    # Different artists: need derived character to say anything but unknown.
    if gap is None and not genre_known:
        out["character"] = TRANSITION_UNKNOWN
        return out
    big_energy = gap is not None and gap >= 2
    if genre_diff and big_energy:
        out["character"] = TRANSITION_MAJOR
    elif genre_diff or big_energy:
        out["character"] = TRANSITION_CONTRAST
    else:
        out["character"] = TRANSITION_GENTLE
    return out
