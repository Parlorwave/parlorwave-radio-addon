"""Pandora runtime wiring: the thin functions that connect the Pandora seams
to Music Brain's live paths. Bounded telemetry only; never a token or signed
URL. Every dependency is injectable so these are pure over their inputs in
tests and hit the real MA in production.

    adopt_source()          -- hosting posture for an active session
    handle_local_request()  -- one-off local request under any source
    request_station()       -- explicit existing-station selection
    arm_handoff()           -- the local->Pandora handoff intent
"""

from __future__ import annotations

from typing import Any, Callable

import source_capabilities as SC
import request_resolver as RR
import source_handoff


def source_of(queue_id: str, *, state_fn: Callable | None = None) -> dict:
    """Bounded source status for an active queue (telemetry/adoption)."""
    import ma_queue
    qs = (state_fn or ma_queue.queue_state)(queue_id)
    return SC.source_status(qs)


def adopt_source(queue_id: str, *, state_fn: Callable | None = None) -> dict:
    """Hosting posture for a session ParlorWave has detected on a DJ-enabled
    output. ParlorWave HOSTS any such session -- it may insert finite program
    elements -- but it does NOT seize queue ownership merely because Pandora was
    detected, never replaces the station, and runs its own station programming
    ONLY when it actually owns the music (LOCAL). Pandora keeps music
    authority; UNKNOWN is treated conservatively (host, but do not program)."""
    st = source_of(queue_id, state_fn=state_fn)
    mode = st["source_mode"]
    return {
        "host": True,
        "source_mode": mode,
        "music_authority": st["music_authority"],
        # ParlorWave station programming (selection/ordering/artist-policy/
        # refill) runs only when ParlorWave is the music authority -- never
        # for provider or unknown sources.
        "run_station_programming": mode == SC.LOCAL,
        "may_insert_program_elements": True,
        "may_reorder_music": st.get("reorderable") is True,
        "acquire_ownership": False,      # detection alone never seizes ownership
    }


def handle_local_request(queue_id: str, artist: str | None, title: str | None,
                         *, tracks: list[dict] | None = None,
                         state_fn: Callable | None = None,
                         insert_fn: Callable | None = None) -> dict:
    """Resolve a user song request local-first and, on a confident
    local hit, insert it as ONE user-intent item next in the current queue.
    A hit inside a Pandora session does NOT rebuild the source, run
    station programming, or touch surrounding Pandora order -- user intent is a
    single explicit intervention; Pandora stays the music authority afterwards.

    Returns {action, result/track, telemetry}. Ambiguous/artist_only/not_found
    do not insert -- the structured result goes back to the caller to speak,
    and a local miss carries the provider truth (no exact Pandora play)."""
    res = RR.resolve_local(artist, title, tracks=tracks)
    st = source_of(queue_id, state_fn=state_fn)
    mode = st["source_mode"]
    telem = {"request_resolution": res["status"], "intervention": None,
             "source_mode": mode}      # bounded: no urls/tokens

    if res["status"] != RR.LOCAL_HIT:
        return {"action": "none", "result": res, "telemetry": telem}

    uri = res["track"].get("uri")
    if not uri:
        # production catalog is path-only; the real path->library-uri resolution
        # reuses the existing search machinery. Without a uri we cannot insert.
        return {"action": "unresolved_uri", "result": res, "telemetry": telem}

    import ma_queue
    (insert_fn or ma_queue.insert_next_many)(queue_id, [uri])
    telem["request_resolution"] = "local"
    telem["intervention"] = ("one_off_local"
                             if mode == SC.PANDORA_DYNAMIC_RADIO else "local")
    return {"action": "inserted_local_next", "track": res["track"],
            "confidence": res["confidence"], "telemetry": telem}


def request_tracks() -> tuple[list[dict], str]:
    """(tracks, source) a single-song request resolves against: the local
    ParlorWave catalog when there is one, otherwise the Music Assistant
    library (alpha.4). Every shipped install has no local catalog -- the
    resolver used to find nothing there and answer `unresolved_uri` -- while
    MA's library carries a playable `library://track/N` URI on every track,
    which is all handle_local_request needs to insert ONE item next. Raises
    ma_catalog.CatalogError when the MA library cannot be read."""
    import library_repo
    local = library_repo.get_tracks()
    if local:
        return local, "local_catalog"
    import ma_catalog
    # snapshot() answers (snapshot, from_cache); the cached read is the one
    # station starts already use, so a request costs MA nothing extra
    # inside SNAPSHOT_TTL_S. A track MA marks unplayable is never offered.
    snap, _cached = ma_catalog.snapshot()
    return [t for t in snap.tracks if t.get("playable", True)], "ma_library"


# --- the panel's single-song request (alpha.4) --------------------------------

#: One request at a time per speaker: queue_id -> the request MA confirmed
#: directly after the song playing then. Process memory only -- a restart
#: forgets it, which costs at most one reordering, never a wrong insert.
_PENDING: dict[str, dict[str, Any]] = {}
PENDING_MAX_S = 3600.0
READBACK_TRIES = 4
READBACK_GAP_S = 0.5


class QueueMoved(Exception):
    """The queue changed between inspection and insertion: insert nothing."""


def _uri_of(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    media = item.get("media_item") if isinstance(item.get("media_item"), dict) else {}
    return str(media.get("uri") or item.get("uri") or "")


def _id_of(item: Any) -> str:
    return str((item or {}).get("queue_item_id") or "") if isinstance(item, dict) else ""


def _program(item: Any) -> bool:
    import media_ref
    return bool(media_ref.programming_ref(_uri_of(item))[0])


def _ahead(queue_id: str, st: dict, window_fn, n: int) -> list[dict] | None:
    """The n items after the current one, by identity, or None if MA's
    queue cannot be placed (then nothing is promised)."""
    import ma_queue
    cur = _id_of(st.get("current_item"))
    if not cur:
        return None
    try:
        win = (window_fn or ma_queue.read_window_items)(
            queue_id, cur, current_index=st.get("current_index"), back=0, ahead=n)
    except Exception:  # noqa: BLE001 - unplaceable is unknown, never position 0
        return None
    items = list(win.get("items") or [])
    for i, it in enumerate(items):
        if _id_of(it) == cur:
            return items[i + 1:i + 1 + n]
    return None


def _pending_still_waiting(queue_id: str, st: dict, window_fn, now: float) -> dict | None:
    rec = _PENDING.get(queue_id)
    if not rec:
        return None
    if now - rec["at"] > PENDING_MAX_S or _id_of(st.get("current_item")) == rec["queue_item_id"]:
        _PENDING.pop(queue_id, None)        # it is playing (or too old to matter)
        return None
    ahead = _ahead(queue_id, st, window_fn, 12)
    if ahead is not None and any(_id_of(it) == rec["queue_item_id"] for it in ahead):
        return rec
    if ahead is not None:
        _PENDING.pop(queue_id, None)        # played, or removed by hand
        return None
    return rec                              # cannot tell: keep the guard


def request_next(queue_id: str, artist: str | None, title: str | None, *,
                 state_fn: Callable | None = None, insert_fn: Callable | None = None,
                 window_fn: Callable | None = None, tracks_fn: Callable | None = None,
                 clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] | None = None) -> dict:
    """ONE song, played next, on `queue_id` -- or a refusal that says why.

    Against MA 2.10 there are three ways "Up next" can be untrue:
      * paused, non-Flow playback: MA's own play-next skips the item;
      * the next track already in MA's buffer: "next" lands behind it;
      * two requests while one waits: MA's "next" reverses them.
    So: only while PLAYING; only while MA's authoritative buffer has not
    moved past the current item (ma_queue.insertion_gate_open, the gate DJ
    breaks use); never next to program audio; one waiting request per
    speaker; the queue re-read immediately before the one insert (moved =
    nothing inserted); and the insert read back, so "Up next" is said only
    when MA shows the song directly after the current one with its buffer
    still in place. Nothing here removes, moves or replays anything."""
    import time
    import ma_catalog
    import ma_queue
    read = state_fn or ma_queue.queue_state
    now = (clock or time.time)()
    try:
        st = read(queue_id) or {}
    except Exception:  # noqa: BLE001 - unreadable is not playing
        st = {}
    if st.get("state") == "paused":
        return {"action": "none", "reason": "player_paused"}
    if st.get("state") != "playing":
        return {"action": "none", "reason": "nothing_playing"}
    if _program(st.get("current_item")) or _program(st.get("next_item")):
        return {"action": "none", "reason": "program_on_air"}
    if not ma_queue.insertion_gate_open(st):
        return {"action": "none", "reason": "next_already_buffered"}
    pend = _pending_still_waiting(queue_id, st, window_fn, now)
    if pend:
        return {"action": "none", "reason": "request_pending",
                "pending": {"title": pend.get("title"), "artist": pend.get("artist")}}
    try:
        tracks, source = (tracks_fn or request_tracks)()
    except ma_catalog.CatalogError:
        return {"action": "none", "reason": "library_unavailable"}

    seen = {"cur": _id_of(st.get("current_item")), "next": _id_of(st.get("next_item"))}

    def guarded_insert(qid: str, uris: list[str]) -> None:
        # THE LAST LOOK: the snapshot above may be seconds old (a large
        # library read sits between). Same current item, same next item,
        # buffer still on the current one, still music -- or nothing.
        again = read(qid) or {}
        if (again.get("state") != "playing"
                or _id_of(again.get("current_item")) != seen["cur"]
                or _id_of(again.get("next_item")) != seen["next"]
                or not ma_queue.insertion_gate_open(again)
                or _program(again.get("current_item")) or _program(again.get("next_item"))):
            raise QueueMoved()
        (insert_fn or ma_queue.insert_next_many)(qid, uris)

    try:
        out = handle_local_request(queue_id, artist, title, tracks=tracks,
                                   state_fn=lambda _q: st, insert_fn=guarded_insert)
    except QueueMoved:
        return {"action": "none", "reason": "queue_changed"}
    except ma_queue.MAQueueError:
        # one play_media call: refused means nothing was inserted
        return {"action": "none", "reason": "insert_failed"}
    out["source"] = source
    if out.get("action") != "inserted_local_next":
        return out
    # THE READBACK: said "next" only when MA shows it so.
    # MA publishes the changed queue a moment after play_media returns, so
    # the read is retried briefly; it never turns an absent item into a
    # promise.
    uri = (out.get("track") or {}).get("uri")
    confirmed, nxt = False, None
    for attempt in range(READBACK_TRIES):
        if attempt:
            (sleep or time.sleep)(READBACK_GAP_S)
        try:
            after = read(queue_id) or {}
        except Exception:  # noqa: BLE001
            after = {}
        nxt = (_ahead(queue_id, after, window_fn, 1) or [None])[0] if after else None
        if _id_of(after.get("current_item")) != seen["cur"]:
            break                       # the song changed: no longer provable
        if nxt is not None and _uri_of(nxt) == uri:
            confirmed = ma_queue.insertion_gate_open(after)
            break
    if not confirmed:
        out["action"] = "inserted_unconfirmed"
        return out
    trk = out.get("track") or {}
    _PENDING[queue_id] = {"queue_item_id": _id_of(nxt), "uri": uri, "at": now,
                          "title": trk.get("title"), "artist": trk.get("artist")}
    return out


def request_station(request_text: str, stations: list[dict]) -> dict:
    """Explicit existing-station selection. Delegates to the matcher;
    returns a STRUCTURED action for the caller. Never creates/seeds a station,
    and a missing song request must NOT be routed here as station intent."""
    return RR.match_pandora_station(request_text, stations)


def arm_handoff(queue_id: str, transition_queue_item_id: str,
                station_uri: str, *, now_iso: str | None = None) -> None:
    """Arm the one-shot local->Pandora handoff: when the intended ParlorWave
    transition item finishes, the executor starts the pre-selected station.
    `station_uri` must be an already-resolved existing station."""
    source_handoff.arm(queue_id, transition_queue_item_id, station_uri,
                       now_iso=now_iso)
