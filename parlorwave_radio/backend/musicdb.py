import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime, date
import json

from paths import BASE_DIR
import service_clock

DB = BASE_DIR / "play_history.db"
INDEX = BASE_DIR / "music_index.json"

# Busy timeout so a concurrent writer (e.g. a /feedback during startup seeding)
# waits briefly instead of raising "database is locked".
BUSY_TIMEOUT_MS = int(os.environ.get("MUSICBRAIN_SQLITE_BUSY_TIMEOUT_MS", "5000"))


def conn():
    c = sqlite3.connect(DB)
    c.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    # WAL improves read/write concurrency for the single-writer service. Enabled
    # best-effort: some networked/FUSE filesystems cannot back the -shm mapping,
    # and there the DELETE journal (with busy_timeout) remains correct.
    try:
        c.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
    return c


def journal_mode() -> str:
    """Return the journal mode actually in effect (for diagnostics/verification)."""
    c = conn()
    try:
        return c.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        c.close()

def holiday_allowed(tag):
    today = date.today()
    m, d = today.month, today.day
    if not tag:
        return True
    tag = tag.lower()
    if tag == "christmas":
        return (m == 11 and d >= 25) or (m == 12)
    if tag == "halloween":
        return m == 10
    if tag == "patriotic":
        return (m == 6 and d >= 20) or (m == 7 and d <= 7)
    return True

# Classification tokens matched on WORD BOUNDARIES over normalized text (non-
# alphanumeric -> space), so "live" no longer fires on "Olive"/"Delivery",
# "demo" not on "Democracy", "intro" not on "Introvert", etc. This is the
# word-boundary approach already proven in release_filters.
def _token_re(tokens):
    return re.compile(r"\b(?:" + "|".join(tokens) + r")\b")


_INTRO_RE = _token_re(["intro", "interlude", "outro", "skit", "dialogue",
                       "spoken", "talking", "phone call"])
_SKIT_RE = _token_re(["skit"])
_DEMO_RE = _token_re(["demo", "rehearsal", "rough mix", "early mix"])
_LIVE_RE = _token_re(["live"])
_INSTRUMENTAL_RE = _token_re(["instrumental"])
_CHRISTMAS_RE = _token_re(["christmas", "xmas", "holiday", "santa", "jingle",
                           "noel", "silent night"])
_HALLOWEEN_RE = _token_re(["halloween", "monster mash", "spooky"])
_PATRIOTIC_RE = _token_re(["america", "usa", "star spangled", "fourth of july"])


def _normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def classify_track(t):
    text = _normalize(" ".join([
        t.get("artist", ""),
        t.get("album", ""),
        t.get("title", ""),
        t.get("genre", ""),
        t.get("path", ""),
    ]))

    score = 60
    is_intro = 0
    is_skit = 0
    holiday = ""

    if _INTRO_RE.search(text):
        score -= 45
        is_intro = 1

    if _SKIT_RE.search(text):
        is_skit = 1
        score -= 35

    if _DEMO_RE.search(text):
        score -= 15

    if _LIVE_RE.search(text):
        score -= 8

    if _INSTRUMENTAL_RE.search(text):
        score -= 8

    if _CHRISTMAS_RE.search(text):
        holiday = "christmas"
        score -= 20
    elif _HALLOWEEN_RE.search(text):
        holiday = "halloween"
        score -= 15
    elif _PATRIOTIC_RE.search(text):
        holiday = "patriotic"
        score -= 10

    return max(1, min(100, score)), is_intro, is_skit, holiday

# Catalog fields that live in music_index.json and are backfilled into tracks
# (the catalog columns). Added via ALTER guards so an existing play_history.db gains
# them without a rebuild. (name, decl)
_CATALOG_COLUMNS = [
    ("genre", "TEXT"),
    ("year", "TEXT"),
    ("duration", "REAL"),
    ("favorite", "INTEGER DEFAULT 0"),
    ("rating", "INTEGER"),
]

# MusicBrainz IDs captured from the file tags by the indexer. Nullable
# (files may be untagged); mb_release_group_id enables MBID-first ownership
# matching in the collector.
_MB_COLUMNS = [
    ("mb_recording_id", "TEXT"),
    ("mb_release_group_id", "TEXT"),
    ("mb_artist_id", "TEXT"),
]


def ensure_schema():
    """Create the tracks table if it does not exist. Cheap; safe to call often.

    Also adds the catalog columns via ALTER guards and the expression
    index on lower(artist), lower(title) that backs feedback()'s artist+title
    lookup.
    """
    c = conn()
    cur = c.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tracks (
        path TEXT PRIMARY KEY,
        artist TEXT,
        title TEXT,
        album TEXT,
        public_score INTEGER DEFAULT 60,
        personal_score INTEGER DEFAULT 0,
        skip_count INTEGER DEFAULT 0,
        thumbs_up_count INTEGER DEFAULT 0,
        thumbs_down_count INTEGER DEFAULT 0,
        is_intro INTEGER DEFAULT 0,
        is_skit INTEGER DEFAULT 0,
        holiday_tag TEXT DEFAULT '',
        last_played TEXT
    )
    """)
    existing = {row[1] for row in cur.execute("PRAGMA table_info(tracks)")}
    for name, decl in _CATALOG_COLUMNS + _MB_COLUMNS:  # fixed literals -> safe to format
        if name not in existing:
            cur.execute(f"ALTER TABLE tracks ADD COLUMN {name} {decl}")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracks_artist_title "
        "ON tracks(lower(artist), lower(title))"
    )
    # Retrieval slices: constraint/request matching by genre and year.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_genre ON tracks(lower(genre))")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_year ON tracks(year)")
    # Recommendation lifecycle (user intent) lives in the persistent store and
    # survives every collection.db rebuild. Keyed by the stable recommendation id.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS recommendation_lifecycle (
        rec_id TEXT PRIMARY KEY,
        lane TEXT,
        release_group_mbid TEXT,
        status TEXT,
        updated_at TEXT
    )
    """)
    # Permanent per-play behavioral history. Fixes the lossiness of the
    # single overwritten tracks.last_played: start = track began; complete =
    # sufficiently completed (>=80%); skip = skipped (explicit or inferred);
    # partial = partially played / unknown ratio. Familiarity and play counts
    # derive from `complete` events only, never `start`s.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS play_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL,
        artist TEXT DEFAULT '',
        event TEXT NOT NULL,
        at TEXT NOT NULL,
        session_id TEXT,
        source TEXT DEFAULT ''
    )
    """)
    # Additive payload column (ALTER-guarded like the catalog
    # columns): 'resurfaced' rows carry {reason, confidence} JSON so the
    # resurfacing cooldown state is permanent, restart-proof, and auditable.
    pe_cols = {row[1] for row in cur.execute("PRAGMA table_info(play_events)")}
    if "detail" not in pe_cols:
        cur.execute("ALTER TABLE play_events ADD COLUMN detail TEXT")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_play_events_path ON play_events(path, at)"
    )
    c.commit()
    c.close()


VALID_LIFECYCLE_STATES = {"new", "seen", "dismissed", "proposed", "approved"}


def set_recommendation_status(rec_id, status, lane=None, release_group_mbid=None):
    """Upsert the lifecycle status of a recommendation (persistent). Raises on an
    unknown status. rec_id is the stable lane+release_group_mbid identity."""
    if status not in VALID_LIFECYCLE_STATES:
        raise ValueError(f"unknown lifecycle status: {status}")
    ensure_schema()
    c = conn()
    try:
        c.execute(
            "INSERT INTO recommendation_lifecycle "
            "(rec_id, lane, release_group_mbid, status, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(rec_id) DO UPDATE SET status=excluded.status, "
            "updated_at=excluded.updated_at, "
            "lane=COALESCE(excluded.lane, recommendation_lifecycle.lane), "
            "release_group_mbid=COALESCE(excluded.release_group_mbid, "
            "recommendation_lifecycle.release_group_mbid)",
            (rec_id, lane, release_group_mbid, status,
             datetime.now().isoformat(timespec="seconds")),
        )
        c.commit()
    finally:
        c.close()


def get_recommendation_statuses(rec_ids=None):
    """Return {rec_id: status} for the given ids (or all lifecycle rows if None)."""
    ensure_schema()
    c = conn()
    try:
        if rec_ids is None:
            rows = c.execute("SELECT rec_id, status FROM recommendation_lifecycle").fetchall()
        else:
            ids = list(rec_ids)
            if not ids:
                return {}
            placeholders = ",".join("?" for _ in ids)
            rows = c.execute(
                f"SELECT rec_id, status FROM recommendation_lifecycle "
                f"WHERE rec_id IN ({placeholders})", ids).fetchall()
    finally:
        c.close()
    return {rec_id: status for rec_id, status in rows}

def _catalog_values(t):
    """Extract the catalog columns from a music_index.json track, normalized for
    storage: genre/year as text, duration as-is, favorite as 0/1, rating passed
    through (NULL when absent)."""
    return (
        t.get("genre", "") or "",
        str(t.get("year", "") or ""),
        t.get("duration"),
        1 if t.get("favorite") else 0,
        t.get("rating"),
    )


def _mb_values(t):
    """Extract the MusicBrainz id columns from a track dict (None when untagged)."""
    return (
        t.get("mb_recording_id") or None,
        t.get("mb_release_group_id") or None,
        t.get("mb_artist_id") or None,
    )


def upsert_tracks(tracks):
    """Insert new tracks (learning defaults + catalog fields) and refresh catalog
    fields on existing rows, preserving learning state (scores/counts/flags).

    Shared by seed_from_index() and index_library.py so the indexer writes the DB
    directly. Returns the number of rows inserted.
    """
    ensure_schema()
    c = conn()
    cur = c.cursor()
    before = cur.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    for t in tracks:
        score, is_intro, is_skit, holiday = classify_track(t)
        path = t.get("path", "")
        genre, year, duration, favorite, rating = _catalog_values(t)
        mb_recording, mb_release_group, mb_artist = _mb_values(t)
        cur.execute("""
        INSERT OR IGNORE INTO tracks
        (path, artist, title, album, public_score, is_intro, is_skit, holiday_tag,
         genre, year, duration, favorite, rating,
         mb_recording_id, mb_release_group_id, mb_artist_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            path,
            t.get("artist", ""),
            t.get("title", ""),
            t.get("album", ""),
            score,
            is_intro,
            is_skit,
            holiday,
            genre, year, duration, favorite, rating,
            mb_recording, mb_release_group, mb_artist,
        ))
        # Refresh catalog + MB ids on existing rows (not learning state).
        cur.execute(
            "UPDATE tracks SET genre=?, year=?, duration=?, favorite=?, rating=?, "
            "mb_recording_id=?, mb_release_group_id=?, mb_artist_id=? WHERE path=?",
            (genre, year, duration, favorite, rating,
             mb_recording, mb_release_group, mb_artist, path),
        )
    after = cur.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    c.commit()
    c.close()
    return after - before


def seed_from_index():
    """Upsert tracks from music_index.json (full JSON parse). NOT for the request
    path; run at startup only when the DB is empty, or after re-indexing.
    Returns the number of rows inserted."""
    if not INDEX.exists():
        return 0
    with open(INDEX, encoding="utf-8") as handle:
        idx = json.load(handle)
    return upsert_tracks(idx.get("tracks", []) or [])


def track_count():
    """Number of tracks currently in the catalog table."""
    ensure_schema()
    c = conn()
    try:
        return c.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    finally:
        c.close()


def rescore():
    """Recompute classification (public_score, is_intro, is_skit, holiday_tag) for
    every track from its stored catalog fields, leaving learning state (personal
    score, thumbs, skips, last_played) untouched. Run once after a classifier
    change: `python -m musicdb rescore`. Returns the number of rows rescored."""
    ensure_schema()
    c = conn()
    cur = c.cursor()
    rows = cur.execute("SELECT path, artist, album, title, genre FROM tracks").fetchall()
    for path, artist, album, title, genre in rows:
        score, is_intro, is_skit, holiday = classify_track({
            "artist": artist or "", "album": album or "", "title": title or "",
            "genre": genre or "", "path": path or "",
        })
        cur.execute(
            "UPDATE tracks SET public_score=?, is_intro=?, is_skit=?, holiday_tag=? "
            "WHERE path=?",
            (score, is_intro, is_skit, holiday, path),
        )
    c.commit()
    n = len(rows)
    c.close()
    return n


def init_db():
    """Startup entry point. Ensures schema and seeds from the index ONLY when the
    catalog is empty -- once the indexer (index_library.py) keeps the DB current,
    cold start no longer parses the full JSON."""
    ensure_schema()
    if track_count() == 0:
        seed_from_index()

def get_scores():
    ensure_schema()
    c = conn()
    cur = c.cursor()
    if cur.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0:
        # Empty DB (e.g. fresh CLI run): seed lazily, once, instead of per call.
        c.close()
        seed_from_index()
        c = conn()
        cur = c.cursor()
    rows = cur.execute("SELECT * FROM tracks").fetchall()
    cols = [d[0] for d in cur.description]
    c.close()
    return {dict(zip(cols, r))["path"]: dict(zip(cols, r)) for r in rows}


def artist_signals():
    """Per-artist aggregates for candidate ranking (affinity, recency, genres).

    One GROUP BY row per artist -- cheap even at 20k artists (unlike a full-table
    scan of every track). Used only when the artist count exceeds the station
    prompt bound.
    """
    ensure_schema()
    c = conn()
    try:
        rows = c.execute("""
            SELECT artist,
                   COALESCE(SUM(thumbs_up_count), 0),
                   COALESCE(SUM(thumbs_down_count), 0),
                   COALESCE(SUM(skip_count), 0),
                   MAX(last_played),
                   GROUP_CONCAT(DISTINCT genre)
            FROM tracks GROUP BY artist
        """).fetchall()
    finally:
        c.close()
    return {
        artist: {
            "thumbs_up": up or 0,
            "thumbs_down": down or 0,
            "skips": skip or 0,
            "last_played": last,
            "genres": genres or "",
        }
        for artist, up, down, skip, last, genres in rows
    }


def artist_affinity_inputs():
    """Per-artist raw behavior signals for affinity aggregation, read from the
    persistent store. Returns {name: {artist_mbid, thumbs_up, thumbs_down, skips,
    played_tracks, last_played}}.

    NOTE: there is no play COUNT in the schema -- `played` feedback only overwrites
    last_played. `played_tracks` (tracks with a non-null last_played) is the only
    available plays proxy; recency derives solely from last_played.
    """
    ensure_schema()
    c = conn()
    try:
        rows = c.execute("""
            SELECT artist,
                   MAX(mb_artist_id),
                   COALESCE(SUM(thumbs_up_count), 0),
                   COALESCE(SUM(thumbs_down_count), 0),
                   COALESCE(SUM(skip_count), 0),
                   SUM(CASE WHEN last_played IS NOT NULL THEN 1 ELSE 0 END),
                   MAX(last_played)
            FROM tracks GROUP BY artist
        """).fetchall()
    finally:
        c.close()
    return {
        artist: {
            "artist_mbid": mbid,
            "thumbs_up": up or 0,
            "thumbs_down": down or 0,
            "skips": skip or 0,
            "played_tracks": played or 0,
            "last_played": last,
        }
        for artist, mbid, up, down, skip, played, last in rows
    }


def get_scores_for_artists(names):
    """Scores keyed by path for only the given artists' tracks -- a targeted,
    indexed query (idx_tracks_artist_title on lower(artist)) that replaces the
    full-table SELECT * on the request path. No lazy seeding (serving only)."""
    ensure_schema()
    names = [n for n in dict.fromkeys(n for n in names if n)]  # unique, drop empties
    if not names:
        return {}
    c = conn()
    cur = c.cursor()
    placeholders = ",".join("?" for _ in names)
    rows = cur.execute(
        f"SELECT * FROM tracks WHERE lower(artist) IN ({placeholders})",
        [n.lower() for n in names],
    ).fetchall()
    cols = [d[0] for d in cur.description]
    c.close()
    return {dict(zip(cols, r))["path"]: dict(zip(cols, r)) for r in rows}

VALID_PLAY_EVENTS = {"start", "complete", "skip", "partial", "resurfaced"}


def write_play_event(path, event, artist="", at=None, session_id=None, source="",
                     detail=None):
    """Append one permanent play_events row. Raises on an unknown event kind.
    `detail` (optional) is a JSON-serializable payload -- used by 'resurfaced'
    rows to carry {reason, confidence}."""
    if event not in VALID_PLAY_EVENTS:
        raise ValueError(f"unknown play event: {event}")
    if not (path or "").strip():
        raise ValueError("path required")
    ensure_schema()
    c = conn()
    try:
        c.execute(
            "INSERT INTO play_events (path, artist, event, at, session_id, source, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (path, artist or "",
             event, at or service_clock.legacy_now().isoformat(timespec="seconds"),
             session_id, source or "",
             json.dumps(detail, ensure_ascii=False) if detail is not None else None),
        )
        c.commit()
    finally:
        c.close()


def play_event_rows(path=None):
    """play_events rows (optionally for one path), oldest first (test/report helper)."""
    ensure_schema()
    c = conn()
    try:
        if path is None:
            rows = c.execute(
                "SELECT path, artist, event, at, session_id, source, detail "
                "FROM play_events ORDER BY id").fetchall()
        else:
            rows = c.execute(
                "SELECT path, artist, event, at, session_id, source, detail "
                "FROM play_events WHERE path = ? ORDER BY id", (path,)).fetchall()
    finally:
        c.close()
    cols = ["path", "artist", "event", "at", "session_id", "source", "detail"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("detail"):
            try:
                d["detail"] = json.loads(d["detail"])
            except (TypeError, ValueError):
                pass
        out.append(d)
    return out


VALID_FEEDBACK_ACTIONS = {"thumbs_up", "thumbs_down", "skip", "played"}

# Feedback actions that also constitute permanent play history. Thumbs are taste
# signals, not play events.
_FEEDBACK_PLAY_EVENTS = {"skip": "skip", "played": "complete"}

# The WHERE clause is always one of these two fixed literals; values stay
# parameterized. Constants (not f-string assembly) keep it injection-proof.
_WHERE_PATH = "path = ?"
_WHERE_ARTIST_TITLE = "lower(artist) = lower(?) AND lower(title) = lower(?)"


def feedback(path=None, artist=None, title=None, action="skip", now=None):
    # Validate before touching the DB. Strip for validation ONLY; the WHERE clause
    # below keeps the raw values so exact-match query behavior is unchanged.
    if action not in VALID_FEEDBACK_ACTIONS:
        raise ValueError(f"unknown action: {action}")
    _path = (path or "").strip()
    _artist = (artist or "").strip()
    _title = (title or "").strip()
    if not _path and not (_artist and _title):
        raise ValueError("path or artist+title required")

    ensure_schema()
    c = conn()
    cur = c.cursor()

    if path:
        where = _WHERE_PATH
        args = [path]
    else:
        where = _WHERE_ARTIST_TITLE
        args = [artist or "", title or ""]

    if action == "thumbs_up":
        cur.execute(f"UPDATE tracks SET thumbs_up_count = thumbs_up_count + 1, personal_score = personal_score + 10 WHERE {where}", args)
    elif action == "thumbs_down":
        cur.execute(f"UPDATE tracks SET thumbs_down_count = thumbs_down_count + 1, personal_score = personal_score - 40 WHERE {where}", args)
    elif action == "skip":
        cur.execute(f"UPDATE tracks SET skip_count = skip_count + 1, personal_score = personal_score - 15 WHERE {where}", args)
    elif action == "played":
        cur.execute(f"UPDATE tracks SET last_played = ? WHERE {where}", [(now or service_clock.legacy_now()).isoformat(timespec="seconds")] + args)

    changed = max(cur.rowcount, 0)

    # Mirror skip/played into permanent play history. Best-effort and
    # additive: it must never change feedback()'s behavior or return value. Only
    # written when the target track is unambiguous (explicit path, or the
    # artist+title WHERE matches exactly one row).
    event = _FEEDBACK_PLAY_EVENTS.get(action)
    if event and changed:
        try:
            if path:
                targets = [(path, artist or "")]
            else:
                targets = cur.execute(
                    f"SELECT path, artist FROM tracks WHERE {where}", args
                ).fetchall()
            if len(targets) == 1:
                cur.execute(
                    "INSERT INTO play_events (path, artist, event, at, session_id, source) "
                    "VALUES (?, ?, ?, ?, NULL, 'feedback')",
                    (targets[0][0], targets[0][1] or "", event,
                     (now or service_clock.legacy_now()).isoformat(timespec="seconds")),
                )
        except sqlite3.Error:
            pass

    c.commit()
    c.close()
    return changed


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rescore":
        print(f"rescored {rescore()} tracks")
    else:
        print("usage: python -m musicdb rescore")
