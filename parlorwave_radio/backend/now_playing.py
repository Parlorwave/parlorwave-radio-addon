"""Booth data surfaces: now playing, lyrics, queue view,
program history. READ ONLY end to end -- nothing in this module can
reorder, skip, delete, mutate, or touch ownership; it exists so the Booth
can look like a radio, not so it can drive one.

Sources of truth:
  * Music Assistant queue snapshots/items (ordered list position +
    queue_item_id identity; the diagnostic index column is never
    consulted).
  * MA's own metadata on the current item -- including artwork proxy ids
    and lyrics. Artwork resolves to MA's /imageproxy/<proxy_id> URL, which
    is served anonymously by MA on the LAN (live-verified), so
    the browser fetches it directly and no credential of any kind exists
    to leak.
  * session_events for aired program elements (dj_break/news_bulletin are
    written at observed track START), joined to dj_breaks / news_bulletins
    for speech text and persona/voice metadata. Aired items only: listener
    history never shows planned/revoked/missed lifecycle noise.

Every item this module describes carries LIFECYCLE TRUTH. A
Parlorwave programme item -- a DJ break, a bulletin -- is `kind` dj/news
ONLY when Parlorwave's own durable state vouches for the token (a dj_breaks
row, a news_bulletins row, or a retired-token fact of the matching kind),
and it then says which of four things it is to a listener: `upcoming`,
`airing`, `spent` or `unknown`. The classification is queue hygiene's own,
through the read-only pair it publishes for exactly this
(`classification_context` + `classify_item`), so the drawer and the
cleanup engine cannot disagree about what a spent break is; this module
only translates the verdict into presentation words and never acts on it. A
route-shaped URL whose token nothing here knows is ordinary queue content
with Music Assistant's own title -- a name that says "DJ break" is not
ownership, and never was.

Lyrics are DISPLAY data: fetched on demand, returned to the UI,
and deliberately never logged, never persisted into telemetry or status,
and never sent to any LLM in this branch. Song semantics, when built on it,
derives BOUNDED semantic context from this same endpoint's source -- it
will not ship lyric bodies into prompts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Any

import media_ref

logger = logging.getLogger("musicbrain")

import config as _config

MA_PUBLIC_URL = _config.current().ma_public_url

ART_SIZE = 512
RECENT_COUNT = 5
UPCOMING_COUNT = 10
HISTORY_MAX = 25

_PROGRAM_LABELS = {"break": ("dj", "DJ break"),
                   "news": ("news", "News bulletin")}


def art_url(image: dict[str, Any] | None, size: int = ART_SIZE) -> str | None:
    """MA imageproxy URL for a MediaItemImage, or None. Built from the
    opaque proxy_id only -- provider names and library paths never reach
    the browser."""
    pid = (image or {}).get("proxy_id")
    if not isinstance(pid, str) or len(pid) != 64:
        return None
    return f"{MA_PUBLIC_URL}/imageproxy/{pid}?size={int(size)}"


def _item_kind(uri: str | None) -> tuple[str, str | None]:
    """('music'|'dj'|'news', program token|None) by ROUTE, never by title
    text. Route is where ownership STARTS (lyrics can refuse a programme
    item on it); whether the item is claimed as Parlorwave's, and what it
    is to a listener, is `_programme_classifier`'s answer."""
    kind, token = media_ref.programming_ref(uri)
    if not kind:
        return "music", None
    return _PROGRAM_LABELS.get(kind, ("dj", ""))[0], token


# --- programme lifecycle, as presentation --------------------------------------
#
# What a Parlorwave programme item IS to a listener right now. Derived on
# every read and never stored; the source is the same ownership + lifecycle
# classification queue hygiene runs (dj_breaks / news_bulletins rows, the
# retired-token ledger, the news schedule's in-flight lock), translated into
# four words a screen can act on. Nothing about what goes on the air is
# decided from any of this.
LIFECYCLE_UPCOMING = "upcoming"   # ours, live: will speak when reached
LIFECYCLE_AIRING = "airing"       # ours, on the air right now
LIFECYCLE_SPENT = "spent"         # ours, over: aired, missed, revoked, failed
                                  # or retired -- the URL answers a tombstone
LIFECYCLE_UNKNOWN = "unknown"     # ours, in a state this build cannot name

_DJ_UPCOMING = ("planned", "rendering", "ready", "queued")
_NEWS_UPCOMING = ("cached", "discovered", "downloading")


def _lifecycle(rec: dict[str, Any], news_airing) -> str | None:
    """queue_hygiene's classification record -> a presentation lifecycle,
    or None when the item is not claimed as Parlorwave's at all (music, a
    malformed or unknown token, a ledger fact of the wrong kind)."""
    import queue_hygiene
    if not rec.get("kind") or rec.get("verdict") == queue_hygiene.V_UNKNOWN:
        return None
    state = str(rec.get("lifecycle") or "")
    if rec["kind"] == queue_hygiene.KIND_DJ:
        # A terminal row is over whichever queue it belongs to: hygiene
        # declines to DELETE a break that sits on another queue's row, but
        # presentation has nothing to remove and states the row's truth.
        if state in queue_hygiene.DJ_TERMINAL_STATES or state == "retired":
            return LIFECYCLE_SPENT
        if state == "airing":
            return LIFECYCLE_AIRING
        if state in _DJ_UPCOMING:
            return LIFECYCLE_UPCOMING
        return LIFECYCLE_UNKNOWN
    if rec.get("verdict") == queue_hygiene.V_DELETE:
        return LIFECYCLE_SPENT
    if rec.get("reason") == "bulletin_locked":
        # The schedule has committed this bulletin to air: it is either
        # the one being heard or the one about to be.
        return (LIFECYCLE_AIRING if news_airing(rec.get("token") or "")
                else LIFECYCLE_UPCOMING)
    if state in _NEWS_UPCOMING:
        return LIFECYCLE_UPCOMING
    return LIFECYCLE_UNKNOWN


def _programme_classifier(queue_id: str, deps: dict[str, Any] | None = None,
                          db_path=None):
    """classify(item) -> (kind, lifecycle) for one view's worth of items.

    READ ONLY, and ONE CONTEXT PER VIEW. The classification is queue
    hygiene's public read-only pair (`classification_context` +
    `classify_item`), never a copy of it; the context is built once here and
    reused for every item, so a view costs one context build plus the reads
    that programme items in it actually need. Building the context reads
    nothing, and music asks it nothing: an ordinary queue of songs performs
    no lookups at all. The one question that is this module's own -- whether
    the schedule says a bulletin is the one ON AIR, which hygiene has no use
    for -- is asked at most once per view and remembered for it.

    `deps` takes the same `*_fn` seams hygiene's own callers inject, plus
    `news_airing_fn`. A lookup that fails claims nothing: the item is shown
    as ordinary queue content rather than as a break of ours in a state we
    could not read."""
    import queue_hygiene
    deps = deps or {}
    lookups = queue_hygiene.classification_context(deps, db_path)
    airing_token: list[str | None] = []

    def news_airing(token: str) -> bool:
        fn = deps.get("news_airing_fn")
        if fn is not None:
            return bool(fn(token))
        # The planner's state is ONE file describing ONE bulletin, so the
        # answer for every item in a view comes from a single read: which
        # token, if any, is on the air. Asking per item would re-read the
        # same file once per bulletin in the queue for one fact.
        if not airing_token:
            import news_schedule
            state = news_schedule.load_state()
            airing_token.append(
                state.get("bulletin_token")
                if state.get("state") == news_schedule.BULLETIN_AIRING else None)
        return bool(token) and airing_token[0] == token

    def classify(item: dict[str, Any]) -> tuple[str, str | None]:
        media = item.get("media_item") if isinstance(item.get("media_item"),
                                                     dict) else {}
        kind, _token = _item_kind(media.get("uri") or item.get("uri"))
        if kind == "music":
            return "music", None
        try:
            rec = queue_hygiene.classify_item(item, queue_id=queue_id,
                                              lookups=lookups)
            lifecycle = _lifecycle(rec, news_airing)
        except Exception:  # noqa: BLE001 - a read that failed claims nothing
            logger.debug("programme lifecycle unreadable; item shown as "
                         "ordinary queue content", exc_info=True)
            return "music", None
        if lifecycle is None:
            return "music", None
        # An announcement shares the break transport (its row and route
        # ARE a dj break's) but is operator programming to a listener, so
        # presentation names it. Purpose is read from the same row the
        # classification just used; a failed read shows a DJ break, which
        # is the transport truth.
        if rec["kind"] == queue_hygiene.KIND_DJ and rec.get("token"):
            try:
                row = lookups["dj_row"](rec["token"])
                purpose = ((row or {}).get("telemetry") or {}).get("purpose")
                if purpose == "announcement":
                    return "announcement", lifecycle
            except Exception:  # noqa: BLE001 - label nicety, never a claim
                pass
        return rec["kind"], lifecycle
    return classify


def _unclaimed(item: dict[str, Any]) -> tuple[str, str | None]:
    """The classifier that vouches for nothing. What `_describe` uses when
    it is handed none: every item is ordinary queue content."""
    return "music", None


def _describe(item: dict[str, Any], classify=None) -> dict[str, Any]:
    media = item.get("media_item") if isinstance(item.get("media_item"),
                                                 dict) else {}
    kind, lifecycle = (classify or _unclaimed)(item)
    out: dict[str, Any] = {"kind": kind,
                           "queue_item_id": item.get("queue_item_id")}
    if lifecycle is not None:
        out["lifecycle"] = lifecycle
    # Music Assistant's own duration for this item, when it reports one.
    # Absent is a real answer and stays absent: a queue summary that
    # invented a length would be arithmetic pretending to be a fact.
    duration = item.get("duration")
    if duration is None:
        duration = media.get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = None
    if duration is not None and duration > 0:
        out["duration_s"] = duration
    if kind == "music":
        artists = media.get("artists")
        artist = (artists[0].get("name") if isinstance(artists, list)
                  and artists and isinstance(artists[0], dict) else None)
        name = media.get("name") or item.get("name") or ""
        if not artist and " - " in name:
            artist, name = name.split(" - ", 1)
        out["title"] = media.get("name") or name
        out["artist"] = artist or ""
        album = media.get("album")
        if isinstance(album, dict) and album.get("name"):
            out["album"] = album["name"]
    else:
        # A program element displays as its semantic self -- never as a
        # filename, a token, or "0.4 seconds of silence".
        out["title"] = {"dj": "DJ break",
                        "announcement": "Announcement"}.get(kind,
                                                            "News bulletin")
        out["artist"] = "News" if kind == "news" else "ParlorWave Radio"
    return out


def snapshot(queue_id: str, state_fn=None, deps=None,
             db_path=None) -> dict[str, Any]:
    """The Now Playing card's data. elapsed is paired with elapsed_at
    (server epoch seconds) so a UI can advance a progress bar locally
    without polling anything at high frequency. The current item carries
    its programme lifecycle like any queue row: a screen must be able to
    tell a break that is speaking from a spent one the queue rewound onto."""
    import ma_queue
    import time
    state = (state_fn or ma_queue.raw_queue)(queue_id)
    ended = queue_ended(queue_id, state)
    if ended:
        # QUEUE ENDED. The finished queue's cursor still names
        # an old song with a leftover elapsed time; shown, it looks paused
        # partway through and invites a Play that replays old music. The
        # card carries no song at all -- MA's queue and the listener's
        # history are untouched, only this presentation is cleared.
        return {"queue_id": queue_id, "state": "ended",
                "ended_since": ended.get("since"),
                "ended_basis": ended.get("basis")}
    current = (state or {}).get("current_item") or {}
    out = _describe(current, _programme_classifier(queue_id, deps, db_path))
    media = current.get("media_item") if isinstance(current.get("media_item"),
                                                    dict) else {}
    out.update({
        "queue_id": queue_id,
        "state": (state or {}).get("state"),
        "duration": current.get("duration"),
        "uri": media.get("uri"),
        "art_url": art_url(current.get("image")),
        "elapsed": (state or {}).get("elapsed_time"),
        "elapsed_at": elapsed_sampled_at(state, time.time()),
    })
    return out


# MA's elapsed_time is stepped (refreshed about once a second), so at read
# time it is up to a second old, and elapsed_time_last_updated says when it
# was true, so a read is typically a fraction of a second stale. Stamping the
# value with the READ time made every card carry a random 0-1 s error, which
# a client interpolating from the card (the progress bar, the waveform
# Mouth's clock) saw as 0.5-1.0 s jumps whenever MA's tick re-phased, as it
# does at a track start. A last_updated in the future or older than this
# bound is not trusted (clock trouble, or a stalled player), and the read
# time is used as before.
ELAPSED_STAMP_MAX_AGE_S = 5.0


def queue_ended(queue_id: str, state: dict[str, Any] | None
                ) -> dict[str, Any] | None:
    """Has this queue genuinely finished? The observer's confirmed answer
    (queue_end, which also tells an ending from a pause, a stop with music
    ahead and a disconnection), or -- with no observer running -- Music
    Assistant's own `ended` flag on a queue that is not playing. Never an
    idle state or a stale elapsed time alone."""
    try:
        import queue_end
        got = queue_end.current(queue_id)
    except Exception:  # noqa: BLE001
        got = None
    if got is not None:
        return got if got.get("ended") else None
    st = state or {}
    if st.get("ended") is True and st.get("state") not in ("playing", "paused"):
        return {"ended": True, "since": None, "basis": "ma_ended"}
    return None


def elapsed_sampled_at(state: dict[str, Any] | None, now: float) -> float:
    """When MA's elapsed_time was true: its own last_updated when that is
    plausible, else `now`."""
    lu = (state or {}).get("elapsed_time_last_updated")
    try:
        lu = float(lu)
    except (TypeError, ValueError):
        return now
    if 0.0 <= now - lu <= ELAPSED_STAMP_MAX_AGE_S:
        return lu
    return now


# --- lyrics -------------------------------------------------------------------

_LRC_LINE = re.compile(r"\[(\d+):(\d{1,2}(?:\.\d{1,3})?)\]\s?(.*)")


def parse_lrc(text: str) -> list[dict[str, Any]]:
    """[[mm:ss.xx] line] -> [{t: seconds, text}], sorted; rows without a
    timestamp are skipped (never guessed)."""
    lines = []
    for raw in (text or "").splitlines():
        m = _LRC_LINE.match(raw.strip())
        if not m:
            continue
        t = int(m.group(1)) * 60 + float(m.group(2))
        lines.append({"t": round(t, 2), "text": m.group(3).strip()})
    lines.sort(key=lambda r: r["t"])
    return lines


def lyrics(queue_id: str, state_fn=None) -> dict[str, Any]:
    """The current track's lyrics, on demand. kind: synced|plain|none.
    Program items (DJ/news) answer none -- there are no lyrics to a
    bulletin. The payload carries the track identity so a UI can drop a
    stale response after a track change."""
    import ma_queue
    state = (state_fn or ma_queue.raw_queue)(queue_id)
    current = (state or {}).get("current_item") or {}
    media = current.get("media_item") if isinstance(current.get("media_item"),
                                                    dict) else {}
    uri = media.get("uri")
    base = {"queue_item_id": current.get("queue_item_id"), "uri": uri,
            "kind": "none"}
    kind, _ = _item_kind(uri)
    if kind != "music":
        return base
    md = media.get("metadata") if isinstance(media.get("metadata"),
                                             dict) else {}
    lrc = md.get("lrc_lyrics")
    if isinstance(lrc, str) and lrc.strip():
        lines = parse_lrc(lrc)
        if lines:
            return {**base, "kind": "synced", "lines": lines}
        # MA can store PLAIN text in lrc_lyrics
        # for some tracks -- no [mm:ss] rows at all. That is honest plain
        # lyrics, never "none", and never fake-synced.
        return {**base, "kind": "plain", "text": lrc}
    plain = md.get("lyrics")
    if isinstance(plain, str) and plain.strip():
        return {**base, "kind": "plain", "text": plain}
    return base


# --- the queue view ------------------------------------------------------------

def queue_view(queue_id: str, recent: int = RECENT_COUNT,
               upcoming: int = UPCOMING_COUNT, state_fn=None,
               items_fn=None, deps=None, db_path=None) -> dict[str, Any]:
    """Recently played + up next around the ORDERED live position.

    Anchoring is by queue_item_id identity inside an explicit-limit window
    around MA's positional hint (the ordered-read discipline the station
    subsystem proved; the diagnostic index column is metadata and is never
    consulted). An unlocatable current item answers honestly empty rather
    than guessing.

    Every row says what it is to a listener (`lifecycle`, see the module
    docstring). A rewind that puts an aired break back ahead of the play
    head is therefore visible as exactly that -- a spent break -- from the
    first read after it happens, whether or not queue hygiene has had its
    turn yet; when hygiene removes the item the row is simply gone, and
    nothing here has to be reset."""
    import ma_queue
    classify = _programme_classifier(queue_id, deps, db_path)
    recent = max(0, min(int(recent), 10))
    upcoming = max(0, min(int(upcoming), 20))
    state = (state_fn or ma_queue.queue_state)(queue_id)
    current_id = ((state or {}).get("current_item") or {}).get("queue_item_id")
    hint = (state or {}).get("current_index")
    out = {"queue_id": queue_id, "recent": [], "upcoming": [],
           "current": None, "complete": False,
           # MUSIC ASSISTANT'S OWN TOTALS, never counted here. `items` is
           # the queue's whole length and `current_index` its position in
           # it, both read from the same snapshot as the window below, so
           # "8 played, 34 to come" is MA's arithmetic and not a count of
           # the handful of rows this view happened to fetch. None when MA
           # did not report them: the screen then shows what it listed and
           # claims no total.
           "total": None, "played_count": None, "upcoming_count": None,
           "ended": False}
    if not current_id:
        return out
    # the history stays; the screen says the queue is over rather than
    # naming MA's stale cursor as "playing now"
    out["ended"] = bool(queue_ended(queue_id, state))
    total = (state or {}).get("items")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        out["total"] = total
        if isinstance(hint, int) and not isinstance(hint, bool) and 0 <= hint < total:
            out["played_count"] = hint
            out["upcoming_count"] = total - hint - 1
    fetch = items_fn or ma_queue.list_queue_items
    offset = max(0, int(hint or 0) - recent - 2)
    window = fetch(queue_id, limit=recent + upcoming + 8, offset=offset)
    pos = next((i for i, it in enumerate(window)
                if it.get("queue_item_id") == current_id), None)
    if pos is None:
        # The hint missed (queue moved). One wider retry from the top of
        # the plausible region; still bounded, still identity-anchored.
        offset = 0
        window = fetch(queue_id, limit=200, offset=0)
        pos = next((i for i, it in enumerate(window)
                    if it.get("queue_item_id") == current_id), None)
        if pos is None:
            return out
    out["current"] = _describe(window[pos], classify)
    out["recent"] = [_describe(it, classify)
                     for it in window[max(0, pos - recent):pos]]
    out["upcoming"] = [_describe(it, classify)
                       for it in window[pos + 1:pos + 1 + upcoming]]
    out["complete"] = True
    return out


# --- program history -----------------------------------------------------------

def program_ref(kind: str, token: str | None) -> str | None:
    """A stable, public id for one aired program element.

    The token is identity AND the credential the audio route accepts, so
    nothing a listener sees may carry it. This is a one-way digest of it:
    the same break always has the same ref -- the Air Check and the history
    row that describes the same break agree -- and the ref cannot
    be turned back into the token."""
    if not token:
        return None
    return hashlib.sha256(f"{kind}:{token}".encode()).hexdigest()[:16]


def program_history(limit: int = 10, db_path=None) -> list[dict[str, Any]]:
    """The last N AIRED ParlorWave program elements, newest first.

    The airing record is session_events (dj_break / news_bulletin rows are
    written at observed track start); speech text and persona/voice come
    from dj_breaks, bulletin titles from news_bulletins. Lifecycle noise
    (planned/rendering/revoked/missed, tokens, errors) never appears: this
    is listener history, not diagnostics."""
    import playback
    limit = max(1, min(int(limit), HISTORY_MAX))
    c = playback._connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        events = c.execute(
            "SELECT kind, at, payload FROM session_events WHERE kind IN "
            "('dj_break', 'news_bulletin') ORDER BY at DESC LIMIT ?",
            (limit,)).fetchall()
        out = []
        for ev in events:
            try:
                token = (json.loads(ev["payload"]) or {}).get("token")
            except (TypeError, ValueError):
                token = None
            entry: dict[str, Any] = {"at": ev["at"]}
            ref = program_ref(ev["kind"], token)
            if ref:
                entry["id"] = ref
            if ev["kind"] == "dj_break":
                brk = c.execute(
                    "SELECT spoken_text, telemetry FROM dj_breaks WHERE "
                    "token = ?", (token or "",)).fetchone()
                tel = {}
                if brk is not None:
                    try:
                        tel = json.loads(brk["telemetry"] or "{}") or {}
                    except (TypeError, ValueError):
                        tel = {}
                purpose = tel.get("purpose") or "scheduled"
                entry["type"] = (purpose if purpose in ("news_intro",
                                                        "announcement")
                                 else "dj_break")
                entry["text"] = (brk["spoken_text"] or "") if brk else ""
                if purpose == "announcement":
                    # Exact vs Smart vs deterministic origin, for Air Check
                    origin = (tel.get("announcement") or {}).get("origin")
                    if origin:
                        entry["origin"] = str(origin)
                    # Announcement voices: what the event asked for and
                    # whether another voice had to read it
                    av = tel.get("announcement_voice")
                    if isinstance(av, dict) and av.get("requested"):
                        entry["voice_requested"] = str(av["requested"])
                        if av.get("fallback"):
                            entry["voice_fallback"] = str(av.get("reason")
                                                          or "fallback")
                if tel.get("persona_profile"):
                    entry["personality"] = tel["persona_profile"]
                # the logical voice id -- a Home Assistant voice has
                # no Chatterbox asset, so the asset alone said nothing
                voice = tel.get("voice_id") or tel.get("tts_voice")
                if voice:
                    entry["voice"] = voice
                topics = tel.get("context_topics")
                if isinstance(topics, list) and topics:
                    entry["topics"] = [str(t) for t in topics][:4]
            else:
                entry["type"] = "news_bulletin"
                row = c.execute(
                    "SELECT title FROM news_bulletins WHERE token = ?",
                    (token or "",)).fetchone() if token else None
                entry["text"] = (row["title"] if row and row["title"]
                                 else "News bulletin")
            out.append(entry)
        return out
    finally:
        c.close()
