"""Grounded music knowledge (Library Pack V1) for Smart DJ breaks.

The station ships one frozen SQLite pack -- resolved artists from the
listener's own library, thousands of atomic facts, MusicBrainz and Wikidata
provenance, CC0. This module is the whole production surface over it:
one deterministic match, a scope gate, a relevance gate
(fact_relevance), an editorial gate (fact_editorial: which facts are
worth airtime on their own, and which RELATIONSHIPS between them are),
and a handful of candidates -- nothing else.

WHAT THIS IS NOT. It is not a second writer and not a second music brain.
It answers one question -- "is there anything TRUE and worth saying about
the music at this boundary?" -- and hands at most MAX_CANDIDATES bounded
propositions to the existing Smart DJ fact slot. The planner decides
whether any of them earns airtime, the writer performs it, and the
existing listener-visible anchor and authority guards decide whether it
actually aired. Knowledge never forces speech.

THE MATCH IS NOT THE MODEL'S. Matching a playing track to a canonical
recording is a deterministic lookup with explicit version handling
(see the version qualifiers below). An LLM is never asked which recording this
is, because the answer decides which facts are legal to say.

SCOPE IS THE WHOLE POINT. Every fact carries the subject it is ABOUT --
an artist, an album/release group, or one recording -- and the match
carries which of those scopes this audio has earned. A Grammy belongs to
the artist and may never become "this song won a Grammy". A producer
belongs to the album. `Lost Bells (Live at the Harbour)` matches the
canonical parent song and therefore earns artist-level facts ONLY: no
studio album, no studio producer, no studio release date, no reissue
date, no claim about this recording. Unknown or ambiguous abstains. A
wrong fact attached to the wrong recording is worse than silence, so the
gates here only ever narrow.

Zero dependencies beyond the standard library, and no network: the pack
is a file in the image. Every public entry point is fenced -- a missing,
corrupt or unreadable pack costs the station its knowledge facts and
nothing else.

    research/ and the artifact freeze are NOT runtime inputs. The pack
    that ships is assets/music_knowledge_library.db, pinned by SHA-256 to
    the accepted candidate (EXPECTED_SHA256); the research retriever and
    matcher were the reference this was written from, never imported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

import fact_relevance

logger = logging.getLogger("musicbrain")

_ASSETS = Path(__file__).resolve().parent / "assets"
DB_PATH = _ASSETS / "music_knowledge_library.db"
META_PATH = _ASSETS / "music_knowledge_library.json"

#: The accepted frozen candidate, pinned in CODE. The metadata file next
#: to the pack repeats it for provenance, but this constant is what the
#: build and the health report verify against: a pack and a manifest that
#: were swapped together would otherwise agree with each other.
EXPECTED_SHA256 = ("5defe68d5e9aaad4ea565e213b3ffdbcce52f842"
                   "b236fb4c5c3f61e46fb1ebef")

#: How many candidate facts may reach the planner packet, ever. The pack's
#: own ceiling: more than this dilutes the planner's attention
#: and tempts it to blend two facts into a third that is not true.
MAX_CANDIDATES = 4

#: Minimum match confidence. Below it the retriever returns nothing --
#: the pack's stated abstention floor.
MIN_CONFIDENCE = 0.85

_PREFIX = "kb:"          # packet fact ids: kb:<12 hex of the fact_id>
_ID_HEX = 12

#: How many relevance suppressions travel in the bounded diagnostics. A
#: boundary can legitimately suppress a dozen roster rows; the log line
#: is a sample, not a ledger.
_SUPPRESSED_MAX = 8

# --- scopes -----------------------------------------------------------------------
#: The scope a predicate's fact lives in. A fact is offered only when its
#: scope is allowed by the match AND not forbidden by it. `artist` facts
#: survive every match that identified an artist at all; everything else
#: needs the recording itself to have been identified.
SCOPE_ARTIST = "artist"
SCOPE_ALBUM = "canonical_album"
SCOPE_PRODUCER = "album_producer"
SCOPE_RELEASE_DATE = "original_recording_release_date"
SCOPE_REISSUE = "reissue_release_date"
SCOPE_RECORDING = "this_recording"

#: predicate -> (subject_type it must carry, scope it needs). A predicate
#: absent from this table NEVER reaches the planner: the table is the
#: whitelist, so a future pack that adds a predicate stays silent until
#: somebody decides how it may be spoken. `genre`, `influenced_by` (P737)
#: and anything work/songwriter-shaped are deliberately absent -- the pack's
#: works table is empty and songwriting commentary is deferred.
_PREDICATE_SCOPE = {
    # release group / album
    "album_producer":                 ("release_group", SCOPE_PRODUCER),
    "album_record_label":             ("release_group", SCOPE_ALBUM),
    "original_recording_release_date": ("release_group", SCOPE_RELEASE_DATE),
    # this recording
    "recording_length":               ("recording", SCOPE_RECORDING),
    "reissue_release_date":           ("recording", SCOPE_REISSUE),
    # artist -- group
    "group_formed_year":              ("artist", SCOPE_ARTIST),
    "group_disbanded_year":           ("artist", SCOPE_ARTIST),
    "group_origin_location":          ("artist", SCOPE_ARTIST),
    "band_member":                    ("artist", SCOPE_ARTIST),
    # artist -- person
    "artist_origin_location":         ("artist", SCOPE_ARTIST),
    "date_of_birth":                  ("artist", SCOPE_ARTIST),
    "date_of_death":                  ("artist", SCOPE_ARTIST),
    "member_of_group":                ("artist", SCOPE_ARTIST),
    "cross_band_membership":          ("artist", SCOPE_ARTIST),
    # artist -- honors
    "grammy_award":                   ("artist", SCOPE_ARTIST),
    "hall_of_fame_induction":         ("artist", SCOPE_ARTIST),
    "prestigious_award":              ("artist", SCOPE_ARTIST),
}

#: PERSON-vs-GROUP semantics.
#: "Formed in 1991" is nonsense about a human being and "born in 1962" is
#: nonsense about a band. The frozen pack already separates them; this is
#: the runtime's own refusal, so a future pack that regresses cannot put
#: the wrong shape of sentence on air.
_GROUP_ONLY = {"group_formed_year", "group_disbanded_year",
               "group_origin_location", "band_member"}
_PERSON_ONLY = {"artist_origin_location", "date_of_birth", "date_of_death",
                "member_of_group", "cross_band_membership"}

#: Editorial relevance to music that is playing RIGHT NOW, from the pack's
#: own documented ranking. Multiplied by the pack's `interestingness`, so
#: the pack's own judgement leads and this only says how well a kind
#: of fact suits a break between two songs.
_RELEVANCE = {
    "original_recording_release_date": 0.95,
    "album_producer": 0.90,
    "album_record_label": 0.85,
    "hall_of_fame_induction": 0.80,
    "grammy_award": 0.80,
    "prestigious_award": 0.80,
    "group_origin_location": 0.75,
    "artist_origin_location": 0.75,
    "group_formed_year": 0.70,
    "member_of_group": 0.70,
    "group_disbanded_year": 0.65,
    "date_of_birth": 0.65,
    "band_member": 0.60,
    "cross_band_membership": 0.60,
    "date_of_death": 0.50,
    "recording_length": 0.50,
    "reissue_release_date": 0.45,
}

#: A running time is a fact, but an ORDINARY running time is trivia
#: sludge: "that one runs 3:14" is true of most songs ever recorded and
#: would become the station's favourite sentence purely because it is
#: available for every track in the pack. It is demoted below everything
#: else unless it is deterministically unusual (below) -- never removed,
#: because a 47-second song really is worth a remark.
_LENGTH_SHORT_MS = 60_000    # under a minute: genuinely unusual
_LENGTH_LONG_MS = 600_000    # over ten minutes: genuinely unusual
_ORDINARY_LENGTH_FACTOR = 0.04

#: Country-level origins (a bare country name) are true but bland: they
#: rank below a city or region, which is what makes an origin worth
#: saying. RANKING ONLY -- a name missing from this set costs one dull
#: but accurate fact, never a wrong one.
_COUNTRY_AREAS = {
    "united states", "united kingdom", "england", "scotland", "wales",
    "northern ireland", "ireland", "canada", "australia", "new zealand",
    "germany", "france", "italy", "spain", "sweden", "norway", "denmark",
    "finland", "netherlands", "belgium", "japan", "brazil", "mexico",
    "iceland", "jamaica", "south africa", "argentina", "portugal",
}
_BLAND_ORIGIN_FACTOR = 0.55


# --- version qualifiers -----------------------------------------------------------
#: Tokens that change WHICH RECORDING this is. Their presence is what
#: turns a clean studio match into a canonical-parent match, and a
#: canonical-parent match may not borrow the studio album's facts.
_QUALIFIER_PATTERNS = (
    ("remix", r"[\(\[-]\s*[^\(\)\[\]]*\b(?:remix|rmx|dub mix|club mix|"
              r"extended mix|extended dance mix|[a-z]+ mix)\b[^\(\)\[\]]*[\)\]]?"),
    ("live", r"[\(\[-]?\s*\b(?:live)\b(?:\s+(?:at|in|from|on|version|"
             r"recording)\b[^\(\)\[\]]*)?[\)\]]?"),
    ("acoustic", r"[\(\[-]?\s*\b(?:acoustic|unplugged)\b[^\(\)\[\]]*[\)\]]?"),
    ("demo", r"[\(\[-]?\s*\b(?:demo|rehearsal|work tape|outtake)\b"
             r"[^\(\)\[\]]*[\)\]]?"),
    ("edit", r"[\(\[-]?\s*\b(?:radio edit|single edit|clean edit|edit)\b"
             r"[^\(\)\[\]]*[\)\]]?"),
    ("instrumental", r"[\(\[-]?\s*\b(?:instrumental|karaoke|backing track)\b"
                     r"[^\(\)\[\]]*[\)\]]?"),
    ("rerecorded", r"[\(\[-]?\s*\b(?:re-?recorded|re-?recording)\b"
                   r"[^\(\)\[\]]*[\)\]]?"),
    ("session", r"[\(\[-]?\s*\b(?:[a-z0-9 ]*session)\b[^\(\)\[\]]*[\)\]]?"),
    ("alternate", r"[\(\[-]?\s*\b(?:take \d+|alternate take|alternate "
                  r"version|alternate mix|alt take)\b[^\(\)\[\]]*[\)\]]?"),
)
#: Harmless: a remaster is the same performance. It keeps the album and
#: the producer; what it loses is the right to say THIS audio came out in
#: the original year, which the release-date phrasing handles by saying
#: "originally".
_REMASTER_RE = re.compile(
    r"[\(\[-]?\s*\b(?:\d{4}\s+)?(?:remaster(?:ed)?|remastered version|"
    r"deluxe|anniversary edition|expanded edition)\b[^\(\)\[\]]*[\)\]]?",
    re.IGNORECASE)
_FEAT_RE = re.compile(r"[\(\[-]?\s*\b(?:feat\.?|featuring|ft\.?|with)\s+"
                      r"[^\(\)\[\]]+[\)\]]?", re.IGNORECASE)
_TRAILING_YEAR_RE = re.compile(r"\s*[\(\[]\s*(?:19|20)\d\d\s*[\)\]]")


def _qualifiers(title: str, artist: str = "") -> tuple[str, dict[str, bool]]:
    """(base title, qualifier flags). Identity-sensitive tags are stripped
    from the title so the underlying song still matches, and REPORTED so
    the caller knows this audio is not the canonical studio cut."""
    out = str(title or "")
    flags: dict[str, bool] = {}
    for name, pat in _QUALIFIER_PATTERNS:
        rx = re.compile(pat, re.IGNORECASE)
        if rx.search(out):
            flags[name] = True
            out = rx.sub(" ", out)
    if _FEAT_RE.search(out) or _FEAT_RE.search(str(artist or "")):
        flags["featured"] = True
        out = _FEAT_RE.sub(" ", out)
    if _REMASTER_RE.search(out):
        flags["remaster"] = True
        out = _REMASTER_RE.sub(" ", out)
    out = _TRAILING_YEAR_RE.sub(" ", out)
    out = re.sub(r"\s*[-–—]\s*$", "", out).strip(" -[](){}")
    return out.strip(), flags


def _identity_sensitive(flags: dict[str, bool]) -> bool:
    return any(flags.get(k) for k in
               ("live", "acoustic", "demo", "remix", "edit", "instrumental",
                "rerecorded", "session", "alternate", "featured"))


def _norm(text: str) -> str:
    """The pack's own normalization: NFKD, ascii-folded, punctuation to
    spaces. It must match how recording_matches was keyed."""
    if not text:
        return ""
    text = re.sub(r"[‘’`´‚‛]", "'", text)
    text = re.sub(r"[“”„‟«»]", '"', text)
    text = re.sub(r"[‐-―−﹘﹣－]", "-", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_artist(artist: str) -> str:
    return _FEAT_RE.sub(" ", str(artist or "")).strip(" -&,")


def _dice(a: str, b: str) -> float:
    t1, t2 = set(a.split()), set(b.split())
    if not t1 or not t2:
        return 0.0
    return (2.0 * len(t1 & t2)) / (len(t1) + len(t2))


# --- integrity / availability ------------------------------------------------------

_conn: sqlite3.Connection | None = None
_integrity: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def integrity(refresh: bool = False) -> dict[str, Any]:
    """Is the SHIPPED pack the accepted candidate? Computed once per
    process (it hashes ~9 MiB) and cached. Never raises."""
    global _integrity
    if _integrity is not None and not refresh:
        return _integrity
    out: dict[str, Any] = {"present": False, "sha256": None, "bytes": None,
                           "expected_sha256": EXPECTED_SHA256,
                           "matches": False, "error": None}
    try:
        if not DB_PATH.is_file():
            out["error"] = "missing"
        else:
            out["present"] = True
            out["bytes"] = DB_PATH.stat().st_size
            out["sha256"] = _sha256(DB_PATH)
            out["matches"] = out["sha256"] == EXPECTED_SHA256
            if not out["matches"]:
                out["error"] = "sha256_mismatch"
    except Exception as exc:  # noqa: BLE001 - integrity must never raise
        out["error"] = type(exc).__name__
    _integrity = out
    return out


def metadata() -> dict[str, Any]:
    """The pack's provenance record (version, cohort, sources, licence),
    or {} when it cannot be read. Small, bounded, never on a prompt."""
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def available() -> bool:
    """Whether knowledge facts can be retrieved at all. A pack whose hash
    does not match the accepted candidate is NOT used: an unknown pack is
    an unaudited pack, and unaudited facts do not go on air."""
    return bool(integrity().get("matches"))


def _connection() -> sqlite3.Connection | None:
    """A read-only handle on the pack, or None. Opened once; the pack is
    immutable and shipped inside the image, so there is nothing to lock
    against and nothing to write."""
    global _conn
    if _conn is not None:
        return _conn
    if not available():
        return None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _conn = conn
        return _conn
    except Exception:  # noqa: BLE001
        logger.warning("music knowledge pack could not be opened",
                       exc_info=True)
        return None


def reset_cache() -> None:
    """Drop the cached handle and integrity result (tests, and a pack
    swapped under a running service)."""
    global _conn, _integrity
    try:
        if _conn is not None:
            _conn.close()
    except Exception:  # noqa: BLE001
        pass
    _conn = None
    _integrity = None


# --- the match ---------------------------------------------------------------------

#: Every match status, with the scopes it earns. `canonical_parent` is
#: the one that matters most: the song was identified, the RECORDING was
#: not, so only the artist may be spoken about.
_MATCH_SCOPES = {
    "exact":            ([SCOPE_ARTIST, SCOPE_ALBUM, SCOPE_PRODUCER,
                          SCOPE_RELEASE_DATE, SCOPE_RECORDING], []),
    "safe_remaster":    ([SCOPE_ARTIST, SCOPE_ALBUM, SCOPE_PRODUCER,
                          SCOPE_RELEASE_DATE, SCOPE_REISSUE], [SCOPE_RECORDING]),
    "canonical_parent": ([SCOPE_ARTIST],
                         [SCOPE_ALBUM, SCOPE_PRODUCER, SCOPE_RELEASE_DATE,
                          SCOPE_REISSUE, SCOPE_RECORDING]),
    "artist_only":      ([SCOPE_ARTIST],
                         [SCOPE_ALBUM, SCOPE_PRODUCER, SCOPE_RELEASE_DATE,
                          SCOPE_REISSUE, SCOPE_RECORDING]),
    "none":             ([], [SCOPE_ARTIST, SCOPE_ALBUM, SCOPE_PRODUCER,
                              SCOPE_RELEASE_DATE, SCOPE_REISSUE,
                              SCOPE_RECORDING]),
}
_FUZZY_MIN = 0.75          # below this the title is a different song
_FUZZY_MASTER = 0.95       # at/above it, and unqualified, it IS the cut


def _abstain(status: str = "none") -> dict[str, Any]:
    allowed, forbidden = _MATCH_SCOPES["none"]
    return {"matched": False, "match_status": status, "confidence": 0.0,
            "artist_mbid": None, "release_group_mbid": None,
            "recording_mbid": None, "canonical_artist": None,
            "canonical_title": None, "canonical_album": None,
            "album_source": None, "linked_release_kind": None,
            "release_kind": None,
            "artist_type": None, "qualifiers": {},
            "allowed_scopes": list(allowed),
            "forbidden_scopes": list(forbidden)}


# --- which RELEASE a track's album facts may come from ---------------------------
# The pack links each recording to ONE release group, and for a real share
# of its recordings that group is not the album the song belongs to: a
# compilation, a live album, a single, a demo or an EP. "Lost Bells"
# (Northern Static, 1976) linked to a later compilation that happens to be
# titled "Cold Orchard" would let the career-span bridge say the
# self-titled album came out decades after the band formed. A song can
# just as well sit on a single, or on a compilation reissue of a studio
# album. Album-date facts may come
# only from a STUDIO album: the linked one, or the studio album the
# catalog says is PLAYING. A shared title is not evidence -- a studio
# album "Cold Orchard" need not contain "Lost Bells" -- and the
# pack holds no track listings, so with neither the match keeps its
# artist facts and says nothing about any album.
KIND_STUDIO = "studio"


def release_kind(primary_type: Any, secondary_types_json: Any) -> str:
    """'studio' for an Album with no secondary type, else what it is
    ('compilation', 'live', 'single', 'ep', 'demo', ...)."""
    try:
        secondary = json.loads(secondary_types_json or "[]") or []
    except Exception:  # noqa: BLE001
        secondary = ["unknown"]
    primary = str(primary_type or "").strip().lower()
    if primary == "album" and not secondary:
        return KIND_STUDIO
    if secondary:
        return str(secondary[0]).strip().lower() or "other"
    return primary or "other"


def _release_row(cur: sqlite3.Cursor, rg_mbid: str | None):
    if not rg_mbid:
        return None
    return cur.execute(
        "SELECT release_group_mbid, artist_mbid, title, primary_type, "
        "secondary_types_json, first_release_date FROM release_groups "
        "WHERE release_group_mbid = ?", (rg_mbid,)).fetchone()


def _studio_titled(cur: sqlite3.Cursor, artist_mbid: str, title: str):
    """The artist's earliest STUDIO album with exactly this title, or None."""
    want = _norm(title)
    if not (artist_mbid and want):
        return None
    best = None
    for row in cur.execute(
            "SELECT release_group_mbid, artist_mbid, title, primary_type, "
            "secondary_types_json, first_release_date FROM release_groups "
            "WHERE artist_mbid = ?", (artist_mbid,)):
        if _norm(row["title"]) != want:
            continue
        if release_kind(row["primary_type"],
                        row["secondary_types_json"]) != KIND_STUDIO:
            continue
        if best is None or (row["first_release_date"] or "9999") < (
                best["first_release_date"] or "9999"):
            best = row
    return best


def _resolve_album(cur: sqlite3.Cursor, artist_mbid: str,
                   rg_mbid: str | None, album: str | None
                   ) -> tuple[Any, str, str]:
    """(release row or None, kind of the LINKED release, how).

    how: 'pack' (the linked release is a studio album and stands),
    'playing_album' (the catalog's album, found as a studio album),
    'contained' (no studio album: no album facts at all)."""
    linked = _release_row(cur, rg_mbid)
    kind = (release_kind(linked["primary_type"],
                         linked["secondary_types_json"])
            if linked is not None else "none")
    if album:
        playing = _studio_titled(cur, artist_mbid, album)
        if playing is not None:
            if linked is not None and \
                    playing["release_group_mbid"] == rg_mbid:
                return playing, kind, "pack"
            if kind != KIND_STUDIO:
                return playing, kind, "playing_album"
    if linked is None:
        return None, kind, "contained"
    if kind == KIND_STUDIO:
        return linked, kind, "pack"
    return None, kind, "contained"


def match(artist: str, title: str, album: str | None = None
          ) -> dict[str, Any]:
    """Which canonical entities this playing track is, and which scopes of
    fact it has therefore earned. Deterministic; never raises; abstains
    whenever it is not sure."""
    try:
        return _match(artist, title, album)
    except Exception:  # noqa: BLE001 - a match fault costs knowledge only
        logger.warning("music knowledge match failed (abstaining)",
                       exc_info=True)
        return _abstain()


def _match(artist: str, title: str, album: str | None) -> dict[str, Any]:
    conn = _connection()
    if conn is None or not (artist or title):
        return _abstain()
    base_title, flags = _qualifiers(title, artist)
    base_artist = _clean_artist(artist)
    n_artist, n_track = _norm(base_artist), _norm(base_title)
    cur = conn.cursor()

    arow = cur.execute(
        "SELECT artist_mbid, name, type FROM artists WHERE LOWER(name) = ? "
        "OR artist_mbid IN (SELECT artist_mbid FROM recording_matches "
        "WHERE artist_norm = ?) LIMIT 1",
        (base_artist.lower(), n_artist)).fetchone()

    row = cur.execute(
        "SELECT recording_mbid, release_group_mbid, artist_mbid, "
        "canonical_title, canonical_artist, canonical_album FROM "
        "recording_matches WHERE lookup_norm = ? ORDER BY score DESC LIMIT 1",
        (f"{n_artist} {n_track}",)).fetchone()
    if row is None:
        row = cur.execute(
            "SELECT recording_mbid, release_group_mbid, artist_mbid, "
            "canonical_title, canonical_artist, canonical_album FROM "
            "recording_matches WHERE artist_norm = ? AND track_norm = ? "
            "ORDER BY score DESC LIMIT 1", (n_artist, n_track)).fetchone()

    status: str | None = None
    confidence = 0.0
    if row is not None:
        if _identity_sensitive(flags):
            # The SONG is identified; this PERFORMANCE is not the one the
            # pack holds album facts for. Artist scope only.
            status, confidence = "canonical_parent", 0.85
        elif flags.get("remaster"):
            status, confidence = "safe_remaster", 0.98
        else:
            status, confidence = "exact", 1.0
    elif arow is not None:
        # The artist is in the pack; look for the song inside that catalog.
        best, best_score = None, 0.0
        for cand in cur.execute(
                "SELECT recording_mbid, release_group_mbid, artist_mbid, "
                "canonical_title, canonical_artist, canonical_album, "
                "track_norm FROM recording_matches WHERE artist_mbid = ?",
                (arow["artist_mbid"],)):
            score = _dice(n_track, cand["track_norm"])
            if score > best_score:
                best, best_score = cand, score
        if best is not None and best_score >= _FUZZY_MIN:
            row = best
            if best_score >= _FUZZY_MASTER and not _identity_sensitive(flags):
                status, confidence = "exact", round(best_score * 0.95, 2)
            else:
                status, confidence = "canonical_parent", 0.85
        else:
            row = None
            status, confidence = "artist_only", 0.90

    if status is None or confidence < MIN_CONFIDENCE:
        return _abstain(status or "none")

    allowed, forbidden = _MATCH_SCOPES[status]
    artist_mbid = (row["artist_mbid"] if row is not None
                   else arow["artist_mbid"])
    release_group = row["release_group_mbid"] if row is not None else None
    canonical_album = row["canonical_album"] if row is not None else None
    album_source, linked_kind = None, None
    if row is not None and SCOPE_ALBUM in allowed:
        rel, linked_kind, album_source = _resolve_album(
            cur, artist_mbid, release_group, album)
        if album_source != "pack":
            # The linked recording belongs to the release it was linked to,
            # not to the album now speaking for it: no recording facts.
            allowed = [sc for sc in allowed if sc != SCOPE_RECORDING]
            forbidden = sorted(set(forbidden) | {SCOPE_RECORDING})
        if rel is None:
            allowed = [sc for sc in allowed if sc == SCOPE_ARTIST]
            forbidden = sorted(set(forbidden) | {
                SCOPE_ALBUM, SCOPE_PRODUCER, SCOPE_RELEASE_DATE,
                SCOPE_REISSUE, SCOPE_RECORDING})
            release_group, canonical_album = None, None
        else:
            release_group = rel["release_group_mbid"]
            canonical_album = rel["title"]
    artist_type = None
    if arow is not None and arow["artist_mbid"] == artist_mbid:
        artist_type = arow["type"]
    else:
        trow = cur.execute("SELECT type, name FROM artists WHERE "
                           "artist_mbid = ?", (artist_mbid,)).fetchone()
        artist_type = trow["type"] if trow else None
    return {
        "matched": True, "match_status": status, "confidence": confidence,
        "artist_mbid": artist_mbid,
        "release_group_mbid": (release_group if SCOPE_ALBUM in allowed
                               else None),
        "recording_mbid": (row["recording_mbid"] if row is not None
                           and SCOPE_RECORDING in allowed else None),
        "canonical_artist": (row["canonical_artist"] if row is not None
                             else (arow["name"] if arow else None)),
        "canonical_title": (row["canonical_title"] if row is not None
                            else None),
        "canonical_album": (canonical_album if SCOPE_ALBUM in allowed
                            else None),
        # where the album facts come from, and what the pack had linked
        # ('pack' / 'playing_album' / 'contained')
        "album_source": album_source,
        "linked_release_kind": linked_kind,
        "release_kind": (KIND_STUDIO if album_source in (
            "pack", "playing_album") else None),
        "artist_type": artist_type,
        "qualifiers": {k: True for k, v in flags.items() if v},
        "allowed_scopes": list(allowed), "forbidden_scopes": list(forbidden),
    }


# --- phrasing ----------------------------------------------------------------------
# The spoken proposition is built HERE, in Python, from the predicate and
# its subject -- never left to the model to assemble from a bare value.
# That is what keeps an artist's Grammy an artist's Grammy: the sentence
# the writer is handed already names the right subject, and the anchor it
# must realize is that subject's name.

def _year(value: str) -> str:
    m = re.match(r"\s*((?:19|20)\d\d)", str(value or ""))
    return m.group(1) if m else ""


def _meta(row: sqlite3.Row) -> dict[str, Any]:
    try:
        return json.loads(row["object_metadata_json"] or "{}") or {}
    except Exception:  # noqa: BLE001
        return {}


def _area(meta: dict[str, Any], value: str) -> str:
    """The place, without the trailing country code the raw value carries
    ('Springfield, OR, US' -> 'Springfield, OR')."""
    area = str(meta.get("area") or "").strip()
    if area:
        return area
    return re.sub(r",\s*[A-Z]{2}\s*$", "", str(value or "")).strip()


def _duration(length_ms: Any) -> str:
    """Milliseconds as words a host can actually say, or '' when the pack
    did not supply a usable number."""
    try:
        total = int(length_ms) // 1000
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    minutes, seconds = divmod(total, 60)
    parts = []
    if minutes:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    if seconds:
        parts.append(f"{seconds} second" + ("s" if seconds != 1 else ""))
    return " and ".join(parts)


def _possessive(name: str) -> str:
    name = str(name or "").strip()
    return name + ("'" if name.endswith("s") else "'s")


def _phrase(predicate: str, value: str, meta: dict[str, Any], *,
            artist: str, album: str, track: str,
            remaster: bool
            ) -> tuple[str, str, list[str], str] | None:
    """(spoken proposition, anchor, grounds, object) for one fact, or None
    when the fact cannot be said safely.

    `anchor` is the name the listener must actually HEAR for the fact to
    count as aired -- the subject the fact is about. `grounds` are the
    factual fragments this proposition authorizes, and they are what the
    claim scanner masks when the writer cites the fact: a paraphrase of a
    supplied truth must be allowed, while the same words with no fact
    behind them must not. `object` is the value as this sentence says it,
    which is what the realization policy requires to be audible.
    """
    year = _year(value)
    if predicate == "album_producer":
        if not (album and value):
            return None
        return (f"{_possessive(artist)} album {album} was produced by "
                f"{value}.", album, ["produced by", "producer", value],
                value)
    if predicate == "album_record_label":
        if not (album and value):
            return None
        return (f"{album} came out on {value}.", album,
                ["record label", "label", value], value)
    if predicate == "original_recording_release_date":
        if not (album and year):
            return None
        if remaster:
            # The audio in the room is the remaster; only the ORIGINAL
            # release may carry the original year, and it must say so.
            return (f"{album} was originally released in {year}.", album,
                    ["released", "originally released", "came out", year],
                    year)
        return (f"{album} came out in {year}.", album,
                ["released", "came out", year], year)
    if predicate == "reissue_release_date":
        if not (track and year):
            return None
        return (f"{track} was reissued in {year}.", track,
                ["reissued", "reissue", "remaster", "remastered", year],
                year)
    if predicate == "recording_length":
        # Said from the milliseconds, not from the pack's own prose: its
        # value string is built for a manifest, not a microphone, and
        # reads "5 minutes, 1 seconds" often enough to matter.
        spoken = _duration(meta.get("length_ms"))
        if not (track and spoken):
            return None
        return (f"{track} runs {spoken}.", track,
                ["runs", "long", spoken], spoken)
    if predicate == "group_formed_year":
        if not (artist and year):
            return None
        return (f"{artist} formed in {year}.", artist,
                ["formed", "founded", year], year)
    if predicate == "group_disbanded_year":
        if not (artist and year):
            return None
        return (f"{artist} broke up in {year}.", artist,
                ["broke up", "disbanded", "split up", year], year)
    if predicate in ("group_origin_location", "artist_origin_location"):
        place = _area(meta, value)
        if not (artist and place):
            return None
        verb = "came out of" if predicate == "group_origin_location" else "is from"
        return (f"{artist} {verb} {place}.", artist,
                ["came out of", "is from", "hails from", "native of", place],
                place)
    if predicate == "band_member":
        member = str(meta.get("member_name") or value or "").strip()
        if not (artist and member):
            return None
        return (f"{member} played in {artist}.", artist,
                ["played in", "member", "line-up", "lineup", member],
                member)
    if predicate in ("member_of_group", "cross_band_membership"):
        group = str(meta.get("group_name") or value or "").strip()
        group = re.sub(r"\s*\([^)]*\)\s*$", "", group).strip()
        if not (artist and group):
            return None
        return (f"{artist} played in {group}.", artist,
                ["played in", "member", "side project", group], group)
    if predicate in ("date_of_birth", "date_of_death"):
        # Phrased so the fact EXISTS for the editorial layer (a birthday
        # today is a bridge to the calendar); on its own it is reference
        # material and fact_editorial never offers it alone.
        if not (artist and year):
            return None
        verb = "was born in" if predicate == "date_of_birth" else "died in"
        return (f"{artist} {verb} {year}.", artist,
                ["born", "died", year], year)
    if predicate == "grammy_award":
        if not (artist and value):
            return None
        return (f"{artist} won a Grammy, the {value}.", artist,
                ["grammy", "grammys", "won a grammy", value], value)
    if predicate == "hall_of_fame_induction":
        if not (artist and value):
            return None
        return (f"{artist} was inducted into the {value}.", artist,
                ["hall of fame", "inducted", "induction", value], value)
    if predicate == "prestigious_award":
        if not (artist and value):
            return None
        return (f"{artist} won the {value}.", artist,
                ["award", "won", value], value)
    return None


# --- predicate realization policy --------------------------------------------------
# The relationship a predicate ASSERTS, expressed as words the spoken copy
# must actually contain. The defect this closes, by example:
#
#     source : album_producer(Ada Venn, "Gullwing")
#     spoken : "he was working with Ada Venn at the time"
#
# The album is audible and the producer is named, so every noun-level
# check passes -- and a PRODUCTION CREDIT has quietly become a personal
# working relationship between two people, which the pack never
# asserted and the pack has no fact shape for.
#
# So a cited fact must show its relationship, not merely its nouns. Each
# entry lists alternatives; ONE is enough, because there are many honest
# ways to say "produced" and only the claim matters. Conservative by
# construction: a phrasing nobody anticipated is refused rather than
# guessed at, and refusal costs one break.
#
# `object` is the value side -- the producer, the label, the year, the
# place -- so a fact cannot be cited for its subject alone.
#
# Adding predicates (writer / composer / lyricist) later means one
# row each here and nothing else: the validator reads this table through
# the packet and knows none of these words itself.
# Matched as SUBSTRINGS of the normalized line, so entries are stems
# where a stem is safe: "releas" covers release/released/releasing and
# "arriv" covers arrive/arrived. That matters -- a table that listed
# "arrived" would refuse "Gullwing did arrive a while back", which is a
# faithful way to say a release date.
# Stems that would swallow an unrelated word are spelled out instead:
# "formed" rather than "form", because "form" is inside "performance".
_REALIZE_REQUIRE = {
    "album_producer": ("produc", "behind the board", "at the boards"),
    "album_record_label": ("label", "imprint", "records", "put out",
                           "releas", "issued", "signed", "carried on",
                           "out on", "home to", "via"),
    "original_recording_release_date": (
        "releas", "came out", "come out", "arriv", "appear", "emerg",
        "drop", "issued", "out in", "back in", "dates from", "landed",
        "from"),
    "reissue_release_date": ("reissu", "remaster", "repress"),
    "recording_length": ("runs", "long", "length", "clocks in", "minute",
                         "second"),
    "group_formed_year": ("formed", "forming", "formation", "founded",
                          "started", "got together", "began",
                          "came together"),
    "group_disbanded_year": ("broke up", "disband", "split", "ended",
                             "called it", "wound up", "packed it in"),
    "group_origin_location": ("from", "out of", "hails", "native", "based",
                              "grew up", "came up", "raised"),
    "artist_origin_location": ("from", "out of", "hails", "native", "based",
                               "grew up", "came up", "raised"),
    "band_member": ("played", "plays", "member", "joined", "line-up",
                    "lineup", "was in", "were in", "band", "drummer",
                    "guitarist", "bassist", "singer"),
    "member_of_group": ("played", "plays", "member", "joined", "was in",
                        "were in", "side project", "band"),
    "cross_band_membership": ("played", "plays", "member", "joined",
                              "was in", "were in", "side project", "band"),
    "date_of_birth": ("born",),
    "date_of_death": ("died", "passed away", "death"),
    "grammy_award": ("grammy", "award", "won", "winner", "took home"),
    "hall_of_fame_induction": ("hall of fame", "inducted", "induction"),
    "prestigious_award": ("award", "won", "winner", "took home", "received"),
}

#: Predicates whose object IS a date: the year is the thing that must be
#: heard, in either the four-digit or the apostrophe form.
_DATE_PREDICATES = {
    "original_recording_release_date", "reissue_release_date",
    "group_formed_year", "group_disbanded_year", "date_of_birth",
    "date_of_death",
}

_OBJ_STOP = {"the", "a", "an", "of", "for", "and", "in", "on", "at", "to",
             "by", "with", "record", "records"}


def _object_tokens(predicate: str, value: str, meta: dict[str, Any],
                   spoken_object: str) -> list[str]:
    """What counts as having SAID the value. A date wants its year; every
    other predicate wants a distinctive word of the thing itself."""
    if predicate in _DATE_PREDICATES:
        year = _year(value) or _year(str(meta.get("date") or ""))
        return [year] if year else []
    words = [w for w in re.split(r"[^\w']+", (spoken_object or value).lower())
             if len(w) >= 3 and w not in _OBJ_STOP]
    return words or [str(spoken_object or value).lower().strip()]


def _realization(predicate: str, value: str, meta: dict[str, Any],
                 spoken_object: str) -> dict[str, Any]:
    """The policy the validator enforces for this one fact. Travels ON the
    packet entry, so the planner applies it without knowing what a
    predicate is."""
    return {
        "predicate": predicate,
        "require": list(_REALIZE_REQUIRE.get(predicate, ())),
        "object": _object_tokens(predicate, value, meta, spoken_object),
    }


def _rank_factor(predicate: str, meta: dict[str, Any], value: str) -> float:
    """Deterministic demotions on top of relevance x interestingness."""
    if predicate == "recording_length":
        ms = meta.get("length_ms")
        try:
            ms = int(ms)
        except (TypeError, ValueError):
            return _ORDINARY_LENGTH_FACTOR
        if ms < _LENGTH_SHORT_MS or ms > _LENGTH_LONG_MS:
            return 1.0
        return _ORDINARY_LENGTH_FACTOR
    if predicate in ("group_origin_location", "artist_origin_location"):
        place = _area(meta, value)
        if _norm(place) in _COUNTRY_AREAS:
            return _BLAND_ORIGIN_FACTOR
    return 1.0


# --- retrieval ---------------------------------------------------------------------

def facts_for(artist: str, title: str, album: str | None = None, *,
              role: str = "ending", exclude_ids: set[str] | None = None,
              limit: int = MAX_CANDIDATES,
              boundary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bounded, scope-safe, RELEVANT candidate facts for one track.

    Returns {'match': <match record>, 'facts': [...], 'suppressed': [...]}
    -- `facts` is ranked, relevance-filtered, deduplicated by predicate,
    capped at `limit`, and empty whenever the match abstained.

    `boundary` is the transition these facts would be spoken across
    (fact_relevance.boundary / for_tracks). Scope safety asks whether a
    fact is TRUE of this audio; relevance asks whether it has any
    editorial business in THIS break, and a fact that fails the second
    is dropped BEFORE the ranking cap, so a relevant fact behind an
    irrelevant one is promoted rather than crowded out. With no boundary
    supplied nothing is filtered -- the caller has told us nothing to
    judge against, and silently inventing a judgement would be worse.
    Never raises.
    """
    out: dict[str, Any] = {"match": _abstain(), "facts": [],
                           "suppressed": [], "pool": [], "eligible": []}
    try:
        m = match(artist, title, album)
        out["match"] = m
        if not m["matched"]:
            return out
        conn = _connection()
        if conn is None:
            return out
        allowed = set(m["allowed_scopes"])
        forbidden = set(m["forbidden_scopes"])
        subjects: list[str] = []
        for sid in (m["artist_mbid"], m["release_group_mbid"],
                    m["recording_mbid"]):
            if sid:
                subjects.append(sid)
        if not subjects:
            return out
        rows = conn.execute(
            "SELECT fact_id, subject_type, subject_id, predicate, "
            "object_value, object_metadata_json, source_name, confidence, "
            "interestingness FROM facts WHERE subject_id IN "
            f"({','.join('?' for _ in subjects)})", subjects).fetchall()

        exclude = set(exclude_ids or ())
        remaster = bool((m.get("qualifiers") or {}).get("remaster"))
        is_group = (m.get("artist_type") or "").lower() == "group"
        is_person = (m.get("artist_type") or "").lower() == "person"
        scored: list[dict[str, Any]] = []
        for row in rows:
            pred = row["predicate"]
            spec = _PREDICATE_SCOPE.get(pred)
            if spec is None:
                continue                     # not whitelisted: never spoken
            want_subject, scope = spec
            if row["subject_type"] != want_subject:
                continue                     # the pack disagrees with itself
            if scope not in allowed or scope in forbidden:
                continue                     # this audio has not earned it
            if pred in _GROUP_ONLY and not is_group:
                continue                     # bands form; people do not
            if pred in _PERSON_ONLY and not is_person:
                continue                     # people are born; bands are not
            if row["fact_id"] in exclude:
                continue                     # aired too recently
            meta = _meta(row)
            built = _phrase(pred, str(row["object_value"] or ""), meta,
                            artist=m["canonical_artist"] or "",
                            album=m["canonical_album"] or "",
                            track=m["canonical_title"] or "",
                            remaster=remaster)
            if built is None:
                continue
            text, anchor, grounds, spoken_object = built
            factor = _rank_factor(pred, meta, str(row["object_value"] or ""))
            score = (float(row["interestingness"] or 0.5)
                     * _RELEVANCE.get(pred, 0.5)
                     * float(row["confidence"] or 1.0)
                     * float(m["confidence"])
                     * factor)
            entry = {
                "id": _PREFIX + row["fact_id"][:_ID_HEX],
                "fact_id": row["fact_id"],
                "about": role,
                "subject_type": row["subject_type"],
                "predicate": pred,
                "scope": scope,
                "value": str(row["object_value"] or ""),
                "source": row["source_name"],
                "text": text,
                "anchor": anchor,
                "grounds": [g for g in grounds if g],
                "realize": _realization(pred, str(row["object_value"] or ""),
                                        meta, spoken_object),
                "score": round(score, 4),
            }
            # A deterministically UNUSUAL running time is worth a remark;
            # an ordinary one is trivia sludge. The relevance layer needs
            # to know which this is and has no access to the metadata, so
            # the retriever states it.
            if pred == "recording_length":
                entry["notable"] = factor >= 1.0
            # What the EDITORIAL layer (fact_editorial) needs and cannot
            # see from the sentence: whose fact this is, whether an
            # origin is a whole country, and -- for a roster row -- the
            # member identity two bands could share. Bounded fields, no
            # provenance payload.
            entry["artist"] = m["canonical_artist"] or ""
            if row["subject_type"] == "release_group":
                # fact_editorial derives date RELATIONSHIPS only from a
                # studio album (see _resolve_album)
                entry["release_kind"] = m.get("release_kind")
            if pred in ("group_origin_location", "artist_origin_location"):
                place = _area(meta, str(row["object_value"] or ""))
                entry["place"] = place
                entry["bland"] = _norm(place) in _COUNTRY_AREAS
            if pred == "band_member":
                entry["meta"] = {
                    "member_mbid": str(meta.get("member_mbid") or ""),
                    "member_name": str(meta.get("member_name")
                                       or row["object_value"] or "")}
            scored.append(entry)
        # THE RAW POOL: every scope-safe, phraseable fact about this side,
        # before relevance and before any cap. Never offered as such --
        # it is what the editorial layer derives RELATIONSHIPS from, so
        # raw knowledge stays available for reasoning while only
        # editorially useful knowledge reaches the writer.
        scored.sort(key=lambda f: (-f["score"], f["fact_id"]))
        out["pool"] = list(scored)
        # RELEVANCE, BEFORE THE CAP. A true fact with no editorial reason
        # to be in this transition never reaches the ranking at all, so
        # the handful the planner sees is the best RELEVANT material
        # rather than the best available material.
        if boundary is not None:
            picked = fact_relevance.select(scored, boundary)
            scored = picked["facts"]
            out["suppressed"] = picked["suppressed"]
        out["eligible"] = list(scored)
        # One fact per predicate: 516 band_member rows must never become
        # a lineup recital, and two release dates are one fact twice.
        scored.sort(key=lambda f: (-f["score"], f["fact_id"]))
        best: list[dict[str, Any]] = []
        seen: set[str] = set()
        for f in scored:
            if f["predicate"] in seen:
                continue
            seen.add(f["predicate"])
            best.append(f)
            if len(best) >= max(0, int(limit)):
                break
        out["facts"] = best
    except Exception:  # noqa: BLE001 - knowledge is optional enrichment
        logger.warning("music knowledge retrieval failed (no facts)",
                       exc_info=True)
        out["facts"] = []
    return out


def candidates(ending: tuple[str, str, str | None] | None,
               nxt: tuple[str, str, str | None] | None = None, *,
               exclude_ids: set[str] | None = None,
               limit: int = MAX_CANDIDATES,
               boundary: dict[str, Any] | None = None,
               now: Any = None) -> dict[str, Any]:
    """The knowledge material for ONE break: the editorial OFFER.

    Three layers, in order, and only the last reaches the writer:

      retrieval   scope-safe, phraseable facts for each side (the raw
                  pool), minus anything aired inside the cooldown;
      relevance   what has any business in THIS transition
                  (fact_relevance);
      editorial   which of those are worth airtime on their own, and
                  which RELATIONSHIPS the raw pools support between the
                  two sides or inside one act's story (fact_editorial).

    The offer is at most `limit` entries: a couple of bridges, a couple
    of dj-ready facts, very often fewer, sometimes none -- an empty
    offer is a valid, good state. `now` is the station's local clock
    (datetime or date) for anniversary and recent-release bridges; with
    none, those are not derived. Nothing else in the queue is consulted.

    Returns {'facts': [...], 'diagnostics': {...}}; both are bounded and
    safe to log. Never raises.
    """
    out: dict[str, Any] = {"facts": [], "diagnostics": {}}
    try:
        import fact_editorial
        limit = max(0, int(limit))
        exclude = set(exclude_ids or ())
        # The transition these facts would have to earn their place in.
        # Derived from the two tracks when the caller supplied none, so a
        # relevance decision is ALWAYS made -- the fallback is the same
        # boundary minus the run/request context the caller knows.
        if boundary is None:
            boundary = fact_relevance.for_tracks(ending, nxt)
        diag: dict[str, Any] = {}
        suppressed: list[dict[str, Any]] = []
        eligible: list[dict[str, Any]] = []
        pools: dict[str, list[dict[str, Any]]] = {"ending": [], "next": []}
        sides: dict[str, dict[str, Any]] = {"ending": {}, "next": {}}
        for role, track in (("ending", ending), ("next", nxt)):
            if not (track and (track[0] or track[1])):
                continue
            res = facts_for(track[0], track[1], track[2], role=role,
                            exclude_ids=exclude, limit=MAX_CANDIDATES,
                            boundary=boundary)
            diag[role] = _diag(res)
            suppressed.extend(res.get("suppressed") or [])
            # the RAW pool, but only the rows that are about THIS side's
            # music: relation without standing (fact_relevance.on_topic).
            # A roster row seeds a shared-member bridge; an album row for
            # a record the catalog says is not playing seeds nothing.
            pools[role] = [f for f in (res.get("pool") or [])
                           if fact_relevance.on_topic(f, boundary)]
            eligible.extend(res.get("eligible") or [])
            m = res.get("match") or {}
            if m.get("matched"):
                sides[role] = {"artist": m.get("canonical_artist") or "",
                               "album": m.get("canonical_album") or "",
                               "title": m.get("canonical_title") or ""}
        # RELATIONSHIPS FIRST, from the RAW pools: two rows nobody would
        # read out can be one line worth saying. Evidence is the pack's
        # own fact_ids, so a bridge is exactly as grounded as its parts.
        bridges = fact_editorial.derive_bridges(
            pools["ending"], pools["next"], sides["ending"], sides["next"],
            today=now)
        # The ending side leads (it is what the listener just heard), so
        # its eligible facts come first in rank order, then the next
        # track's; the editorial layer keeps the next track to one.
        eligible.sort(key=lambda f: (0 if f.get("about") == "ending" else 1,
                                     -float(f.get("score") or 0.0),
                                     str(f.get("fact_id"))))
        members = {str((f.get("meta") or {}).get("member_name") or "")
                   for side in pools.values() for f in side
                   if f.get("predicate") == "band_member"}
        offer = fact_editorial.compose(eligible, bridges,
                                       member_names=members)
        picked = list(offer["facts"])[:limit]
        out["facts"] = picked
        # grounding-only rows (fact_editorial.MAX_REFERENCE): the packet
        # hides them from the writer and the validator reads them
        out["reference"] = list(offer.get("reference") or [])
        diag["offered"] = [{"id": f["id"], "fact_id": f["fact_id"],
                            "subject_type": f["subject_type"],
                            "predicate": f["predicate"],
                            "about": f["about"],
                            "relation": f.get("relation"),
                            "evidence": list(f.get("evidence") or [])}
                           for f in picked]
        # WHAT THE STATION DECIDED NOT TO OFFER, and why. The positive
        # space (a fact that aired) has always been recoverable; the
        # negative space -- true material Parlorwave held back because it
        # had no business in this transition, or no point on its own --
        # is the whole point of the relevance and editorial layers and
        # evaporates unless it is written down.
        suppressed.extend(offer.get("suppressed") or [])
        if suppressed:
            diag["suppressed"] = _summarise(suppressed)
        diag["editorial"] = {
            "raw_candidates": {"ending": len(pools["ending"]),
                               "next": len(pools["next"])},
            "eligible": len(eligible),
            **offer["diagnostics"],
        }
        out["diagnostics"] = diag
    except Exception:  # noqa: BLE001
        logger.warning("music knowledge candidates failed (none offered)",
                       exc_info=True)
        return {"facts": [], "diagnostics": {}}
    return out


def _summarise(suppressed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The relevance suppressions as COUNTS, not a list of rows.

    A band with 40 roster entries suppresses 40 facts for one reason; the
    useful statement is "band_member about the ending track, 40 of them,
    no editorial reason", and the useless one is forty near-identical
    objects in a diagnostics blob that travels into every forensic
    record."""
    buckets: dict[tuple[Any, Any, Any, Any], int] = {}
    for row in suppressed:
        key = (row.get("predicate"), row.get("about"),
               row.get("suppression_reason"), row.get("class"))
        buckets[key] = buckets.get(key, 0) + 1
    out = []
    for (p, a, r, c), n in buckets.items():
        entry = {"predicate": p, "about": a, "suppression_reason": r,
                 "count": n}
        if c:
            entry["class"] = c        # the editorial class (fact_editorial)
        out.append(entry)
    out.sort(key=lambda d: (-d["count"], str(d["predicate"])))
    return out[:_SUPPRESSED_MAX]


def _diag(result: dict[str, Any]) -> dict[str, Any]:
    """The bounded, loggable summary of one lookup. Identities and counts;
    never a source record and never a provenance payload."""
    m = result.get("match") or {}
    return {"match_status": m.get("match_status"),
            "confidence": m.get("confidence"),
            "artist_type": m.get("artist_type"),
            "allowed_scopes": list(m.get("allowed_scopes") or []),
            "forbidden_scopes": list(m.get("forbidden_scopes") or []),
            "qualifiers": sorted((m.get("qualifiers") or {}).keys()),
            "candidate_facts": len(result.get("facts") or [])}


# --- documented artist relationships ------------------------------------------------
#
# WHY THIS EXISTS, given that nothing else here reads `artist_relationships`.
# The deterministic callback (dj_callback) used to name an earlier artist
# purely because they had played recently. Recency is not an editorial
# reason, so the callback now needs a documented link between the artist
# it would recall and the music actually playing -- and the only
# authoritative source of such a link that ships with the station is this
# table. MusicBrainz relationship rows, frozen in the pack, CC0.
#
# Deliberately shallow: a DIRECT relationship row in either direction, or
# ONE shared person (a member of both acts). Two hops is where a
# relationship graph stops describing anything a listener would recognise
# -- everyone is two hops from everyone -- and this is a radio station,
# not a knowledge base.

#: Relationship types that count as a link worth recalling on air. The
#: table IS the whitelist: a type absent from it stays silent rather than
#: becoming a connection nobody decided was one. ("is person" is
#: MusicBrainz's alias-of-a-person row, which is an identity rather than
#: a relationship, and is deliberately absent.)
_SPOKEN_RELATIONSHIPS = frozenset({
    "member of band", "collaboration", "founder", "supporting musician",
})


def _artist_row(conn: sqlite3.Connection, name: str):
    """The pack's artist row for a played artist NAME, or None. The same
    two lookups `_match` uses, and no fuzzy fallback: a callback that
    named the wrong band would be a false occurrence claim."""
    clean = _clean_artist(name)
    if not clean:
        return None
    return conn.execute(
        "SELECT artist_mbid, name FROM artists WHERE LOWER(name) = ? "
        "OR artist_mbid IN (SELECT artist_mbid FROM recording_matches "
        "WHERE artist_norm = ?) LIMIT 1",
        (clean.lower(), _norm(clean))).fetchone()


def artists_related(a: str, b: str) -> dict[str, Any] | None:
    """A DOCUMENTED link between two artist names, or None.

    Returns {'kind', 'detail'} where `kind` is 'direct' or
    'shared_member' -- bounded, loggable, and never a sentence: how a
    relationship is SAID is the caller's business. Both artists must
    resolve exactly in the pack; an unresolved name is not a
    relationship, it is an unknown. Never raises: no pack, no link.
    """
    try:
        if not (a and b) or _norm(_clean_artist(a)) == _norm(_clean_artist(b)):
            return None
        conn = _connection()
        if conn is None:
            return None
        ra, rb = _artist_row(conn, a), _artist_row(conn, b)
        if ra is None or rb is None:
            return None
        ma, mb = ra["artist_mbid"], rb["artist_mbid"]
        if ma == mb:
            return None
        row = conn.execute(
            "SELECT rel_type, source_artist_mbid AS s FROM "
            "artist_relationships WHERE "
            "(source_artist_mbid = ? AND target_artist_mbid = ?) OR "
            "(source_artist_mbid = ? AND target_artist_mbid = ?) LIMIT 1",
            (ma, mb, mb, ma)).fetchone()
        if row is not None and row["rel_type"] in _SPOKEN_RELATIONSHIPS:
            out: dict[str, Any] = {"kind": "direct",
                                   "detail": row["rel_type"]}
            if row["rel_type"] == "member of band":
                # MusicBrainz stores membership person -> group. The
                # SPEAKABLE half: who played in what, in the pack's own
                # canonical names, so a callback can say WHY.
                person, group = ((ra, rb) if row["s"] == ma else (rb, ra))
                out["person"] = person["name"]
                out["group"] = group["name"]
            return out
        # ONE shared person. Both directions are collected because the
        # table stores a membership once, from whichever side MusicBrainz
        # recorded it.
        peers = {}
        for mbid, other in ((ma, mb), (mb, ma)):
            for r in conn.execute(
                    "SELECT source_artist_mbid AS s, target_artist_mbid AS t "
                    "FROM artist_relationships WHERE rel_type = 'member of "
                    "band' AND (source_artist_mbid = ? OR "
                    "target_artist_mbid = ?)", (mbid, mbid)):
                for peer in (r["s"], r["t"]):
                    if peer and peer != mbid:
                        peers.setdefault(mbid, set()).add(peer)
        shared = (peers.get(ma) or set()) & (peers.get(mb) or set())
        if shared:
            # The shared people BY NAME, from the pack's own roster facts,
            # so the link can be SAID. A relationship row with no name
            # behind it is still a relationship -- `members` is simply
            # empty, and the caller decides whether an unspeakable link
            # is worth recalling (dj_callback: it is not).
            members = _member_names(conn, shared, (ma, mb))
            return {"kind": "shared_member",
                    "detail": members[0] if members else None,
                    "members": members}
        return None
    except Exception:  # noqa: BLE001 - a relationship lookup is optional
        logger.debug("artist relationship lookup failed", exc_info=True)
        return None


def _member_names(conn: sqlite3.Connection, member_mbids: set[str],
                  bands: tuple[str, str]) -> list[str]:
    """Names for the shared member MBIDs, from `band_member` facts of
    either band, sorted; MBIDs the pack never named are left out."""
    names: set[str] = set()
    for band in bands:
        for r in conn.execute(
                "SELECT object_value, object_metadata_json FROM facts "
                "WHERE predicate = 'band_member' AND subject_id = ?",
                (band,)):
            meta = _meta(r)
            if str(meta.get("member_mbid") or "") in member_mbids:
                name = str(meta.get("member_name") or r["object_value"]
                           or "").strip()
                if name:
                    names.add(name)
    return sorted(names)


def shared_members(a: str, b: str) -> list[str]:
    """The people the pack documents in BOTH acts, by name (possibly
    empty). Never raises."""
    try:
        rel = artists_related(a, b)
    except Exception:  # noqa: BLE001
        return []
    if not rel or rel.get("kind") != "shared_member":
        return []
    return list(rel.get("members") or [])


# --- what actually aired -----------------------------------------------------------

def subjects_audible(fact: dict[str, Any], low_text: str) -> bool:
    """Is every subject this fact is about audible in the normalized
    copy? A plain fact has one anchor; a bridge has one GROUP of
    alternative names per subject, and each group needs one heard --
    the relationship is meaningless if either side is missing."""
    groups = fact.get("subjects")
    if isinstance(groups, list) and groups:
        for group in groups:
            alts = [_norm(str(a)) for a in (group or []) if a]
            alts = [a for a in alts if a]
            if alts and not any(a in low_text for a in alts):
                return False
        return True
    anchor = _norm(str(fact.get("anchor") or ""))
    return bool(anchor) and anchor in low_text


#: '97 -> 1997 for VALIDATION ONLY, mirroring smart_dj's rule so a spoken
#: year counts the same way in telemetry as it does in the validator.
_APOS_YEAR = re.compile(r"'(\d{2})\b")


def _expand_years(text: str) -> str:
    def _fix(m):
        n = int(m.group(1))
        return f"{2000 + n}" if n <= 29 else f"{1900 + n}"
    return _APOS_YEAR.sub(_fix, text or "")


def _token_said(token: str, low: str) -> bool:
    """Is this token audible? A NUMBER must stand alone -- a twenty-year
    gap is not 'said' by the 20 inside 2016 -- while words keep the
    substring rule the realization policy was written for (stems)."""
    t = _norm(str(token))
    if not t:
        return False
    if t.isdigit():
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low))
    return t in low


def proposition_realized(fact: dict[str, Any], text: str) -> bool:
    """Did this copy audibly express the PROPOSITION, not merely name its
    subject?

    THE SAME THREE QUESTIONS THE WRITER VALIDATOR ASKS (smart_dj
    .fact_realized), applied to the pack's own fact shape:

      SUBJECT   the thing the fact is about must be audible -- the anchor,
                and for a bridge every one of its subject groups, because
                a relationship the listener can only half hear is not a
                relationship.
      VALUE     the fact's own object -- the producer, the year, the
                member, the award -- must be said.
      RELATION  the predicate's relationship must survive: one of the
                cues the pack attached to this predicate must be audible.
                Naming both nouns is not asserting the credit that
                joins them.

    ONE DELIBERATE DIVERGENCE, and it is the point of this function. The
    validator is permissive where no policy exists: a fact carrying no
    `realize` rule passes on its subject alone, so that copy is never
    rejected for want of a rule. Telemetry must not inherit that
    permission -- it would report exactly the false positive this
    replaces. So here a fact with no usable policy is NOT realized.
    Prefer a false negative: under-reporting what aired is a gap, while
    over-reporting it is a lie about what the station said.

    A floor line such as "Ada Venn just wrapped up Slow Ferry. We'll keep
    it moving with Cold Orchard next. It's 8:40 PM." would pass a
    subject-only test and be credited with a knowledge fact. Both artists
    are audible; no proposition is. That case returns False."""
    low = _norm(_expand_years(text or ""))
    if not low:
        return False
    policy = fact.get("realize") or fact.get("_realize") or {}
    anchor = _norm(str(fact.get("anchor") or ""))
    if anchor and anchor not in low:
        return False
    groups = policy.get("subjects") or fact.get("subjects") or []
    for group in groups:
        alts = [_norm(str(a)) for a in (group or []) if a]
        alts = [a for a in alts if a]
        if alts and not any(a in low for a in alts):
            return False
    if not anchor and not groups:
        return False              # nothing identifies what this is about
    obj = [o for o in (policy.get("object") or []) if o]
    need = [r for r in (policy.get("require") or []) if r]
    if not obj and not need:
        return False              # no policy -> telemetry does not guess
    if obj and not any(_token_said(o, low) for o in obj):
        return False              # the value was never said
    if need and not any(_norm(str(r)) in low for r in need):
        return False              # the relationship was never asserted
    return True


def realized_fact_ids(knowledge_facts: list[dict[str, Any]] | None,
                      text: str) -> list[str]:
    """OBSERVABILITY ONLY: pack `fact_id`s whose PROPOSITION this copy
    audibly expressed, whether or not a writer cited them.

    Why this exists separately from `aired_fact_ids`. The deterministic
    floor does not cite: fact_editorial / dj_callback compose the
    sentence directly, so a real personnel bridge can go out on air while
    `aired_fact_ids` correctly reports nothing -- it requires a citation
    that never happened. A floor break saying "Wren Hollis played in both
    Cold Orchard and Marlow Kites" would show `knowledge_aired == []`.

    The converse is equally refused: a citation whose proposition never
    reached the words is not realized either, because this asks the copy
    what it said, never the writer what it meant to say.

    THIS MUST NEVER SPEND NOVELTY, and nothing here does: the cooldown is
    committed from `aired_fact_ids` alone (dj_engine's mark check), whose
    contract -- cited AND audible -- is unchanged, and which this function
    neither calls nor is called by. Attributing floor usage for the record
    is a measurement; treating it as a spend would retire facts the
    planner never chose.

    Never raises; on any failure it claims nothing."""
    out: list[str] = []
    try:
        for f in knowledge_facts or []:
            if not isinstance(f, dict):
                continue
            fid = f.get("fact_id")
            if not fid or not proposition_realized(f, text):
                continue
            for spent in (f.get("evidence") or [fid]):
                if spent and str(spent) not in out:
                    out.append(str(spent))
    except Exception:  # noqa: BLE001
        return out
    return out


def aired_fact_ids(knowledge_facts: list[dict[str, Any]] | None,
                   cited_ids: list[str] | None, text: str) -> list[str]:
    """The pack `fact_id`s the FINAL ACCEPTED COPY genuinely realized.

    Two things must both be true: the writer cited the fact, and the
    subject that fact is about is audible in the words that will play.
    This is the only thing that may spend a fact's novelty. A topic
    marked as used by every break that was OFFERED it would let a
    correctly qualified event silently spend its one mention on a break
    that never said it.

    Retrieval does not commit. Offering does not commit. The planner
    choosing FACT does not commit. Generating text that is then rejected
    does not commit. Never raises; on any failure it claims nothing,
    which at worst repeats a fact sooner than intended.
    """
    out: list[str] = []
    try:
        low = _norm(text or "")
        if not low:
            return out
        cited = set(cited_ids or ())
        for f in knowledge_facts or []:
            if not isinstance(f, dict):
                continue
            fid = f.get("fact_id")
            if not fid or f.get("id") not in cited:
                continue
            if not subjects_audible(f, low):
                continue
            # A BRIDGE spends the novelty of the facts it was built from:
            # the pack rows are what the cooldown is keyed on, and the
            # same two rows must not make the same line again next week.
            for spent in (f.get("evidence") or [fid]):
                if spent and str(spent) not in out:
                    out.append(str(spent))
    except Exception:  # noqa: BLE001
        logger.warning("aired fact attribution failed (claiming none)",
                       exc_info=True)
        return []
    return out
