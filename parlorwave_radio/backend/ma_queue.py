#!/usr/bin/env python3
"""Music Assistant queue client for the queue-native DJ engine and the
station programmer.

Music Assistant OWNS playback transport. This client does exactly eight
things and nothing else:

  1. Read a PlayerQueue's structural state (current_index, index_in_buffer,
     queue item count) -- the evidence for the Flow-mode insertion gate.
  2. Insert one finite ParlorWave audio URL as the NEXT queue item.
  3. Read library tracks / current-item detail (station seeding + URI
     resolution; pure reads).
  4. Replace the FUTURE portion of a queue (MA QueueOption 'replace_next':
     everything AFTER the current item is replaced; the current item and its
     playback position are untouched -- verified against MA's
     enqueue option enum: play/replace/next/replace_next/add).
  5. Read the ORDERED queue items (player_queues/items) -- the future-runway
     evidence. player_queues/all reports `items` as an integer COUNT only, so
     this is the one command that can answer "what is actually queued after
     the current song".
  6. APPEND tracks at the queue TAIL (MA QueueOption 'add'): station
     replenishment. Proven on a live queue -- it lands after every existing
     item, in the order given, without touching the current item, its
     position, the Flow-Mode buffer or the near future.
  7. Read and set ONE queue's Autoplay flag ("Don't stop the music"):
     queue ownership. Music Assistant's per-queue
     `autoplay_enabled` is the switch behind its own similar-track /
     library auto-fill; while a ParlorWave station owns a queue that switch is
     turned OFF for that queue and that queue only, so exactly one
     autonomous programmer appends music. The GLOBAL Autoplay settings
     (`autoplay_mode`, `autoplay_playlist` in the player_queues core
     config) are never read for a decision and never written -- there is
     deliberately no helper for them. Contract verified read-only against
     MA 2.10: `player_queues/get {queue_id}` reports
     `autoplay_enabled`; `player_queues/autoplay {queue_id,
     autoplay_enabled}` sets it (scope QUEUES_CONTROL) and MA persists it
     in its own queue-state cache across restarts.
  8. DELETE one queue item by its exact `queue_item_id` string
     (`player_queues/delete_item`): queue hygiene. The ONLY
     removal verb in Music Brain, and it addresses an item by identity
     only -- never by list position. Two callers, both proving ownership
     first: queue_hygiene, which deletes nothing it cannot positively prove
     is Parlorwave's own stale program audio, and station_live's station
     switch, which removes only the OLD station's own not-yet-buffered music
     by the exact ids its ownership witness recorded when it appended them. Contract facts still to be VERIFIED LIVE on MA
     2.10 (see delete_queue_item): the response body is null, so the
     only proof of removal is a re-read; an index at or below
     index_in_buffer is refused silently; an unknown id errors.

The empirically proven safety rule (Flow Mode on hass_players outputs):
insertion is safe ONLY while

    index_in_buffer == current_index

i.e. the player has not yet buffered past the currently-audible item. Time
remaining is planning evidence, never the gate. When the buffer has advanced,
the boundary is MISSED: no insertion, no transport manipulation of any kind.

Transport contract verified against Music Assistant 2.9:
a single endpoint

    POST {MUSIC_ASSISTANT_URL}/api
    Authorization: Bearer <token>
    body: {"command": "<command>", "args": {...}}

with the commands

    player_queues/all          -> list of queues; each carries queue_id,
                                  state, items, flow_mode, current_index,
                                  index_in_buffer (top-level fields)
    player_queues/play_media   -> args {queue_id, media, option: "next"}
    player_queues/items        -> args {queue_id, limit?, offset?}; returns
                                  the queue's items as an ORDERED list of
                                  item objects (verified live against
                                  MA 2.9.11 -- see list_queue_items)

The token is read from MUSIC_ASSISTANT_TOKEN_FILE (a root-only file on the
host, outside any code or data mount) or
MUSIC_ASSISTANT_TOKEN, and is never logged.

This module never pauses, stops, skips, or seeks. There is deliberately no
method for any of those. It never moves, clears or reorders a queue either:
the one removal it can perform is a single item by exact id.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import config as _config

_cfg = _config.current()
MA_URL = _cfg.ma_url
MA_TIMEOUT_S = float(os.environ.get("MUSIC_ASSISTANT_TIMEOUT_S", "5"))
MA_TOKEN_FILE = _cfg.ma_token_file
MA_TOKEN = _cfg.ma_token_env


class MAQueueError(RuntimeError):
    """MA could not be read or the enqueue was not accepted. Callers must
    treat this as 'no break this boundary', never as a transport problem to
    fix -- music keeps playing without us."""


def enqueue_never_delivered(exc: BaseException) -> bool:
    """Did a failed enqueue provably NEVER reach Music Assistant?

    True only when the evidence is positive that the request never left
    this host: the call was refused before it was made (argument
    validation), or the connection was never established (refused,
    unreachable, name not resolved). Everything else -- a timeout, a reset
    mid-response, ANY HTTP status (MA read the request; what it did before
    answering is not ours to assume), an unreadable body -- may have
    happened after MA acted on the request, and is NOT evidence of
    non-delivery. Only the EXPLICIT cause chain (`raise ... from`, as _api
    does) is followed: an exception merely raised while handling another
    is not that other's outcome. Callers that must never double-air
    (announcements) retry only on True; nothing else changes behaviour on
    this answer."""
    import errno
    import socket
    not_connected = {getattr(errno, n) for n in
                     ("ECONNREFUSED", "EHOSTUNREACH", "ENETUNREACH")
                     if hasattr(errno, n)}
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, urllib.error.HTTPError):
            return False            # MA answered: it received the request
        if isinstance(cur, urllib.error.URLError):
            reason = cur.reason
            if isinstance(reason, (ConnectionRefusedError, socket.gaierror)):
                return True
            return (isinstance(reason, OSError)
                    and getattr(reason, "errno", None) in not_connected)
        if isinstance(cur, (ConnectionRefusedError, socket.gaierror)):
            return True
        if isinstance(cur, MAQueueError) and cur.__cause__ is None \
                and cur.__context__ is None:
            return True             # raised by our own validation, pre-call
        cur = cur.__cause__
    return False


class AnchorNotFound(MAQueueError):
    """The current item's queue_item_id was not in the queue's item list.

    A SUBCLASS so every existing `except MAQueueError` still catches it,
    but a distinct type because it means something different from a
    transport failure, and callers reason differently about the two:

        transport failure  we learned NOTHING. Fall back to whatever other
                           evidence exists; the queue is probably fine.
        anchor not found   we learned SOMETHING, and it is bad. MA's own
                           snapshot says the head is X and MA's own item
                           list does not contain X, so the two views of
                           this queue disagree. Nothing derived from that
                           snapshot's adjacency can be trusted, and a
                           caller that needs a position must fail closed
                           rather than fall back."""


def _token() -> str:
    """The MA API token, never logged. Env value wins; else the token file;
    else '' (the request is sent unauthenticated and MA's rejection surfaces
    as an ordinary MAQueueError = closed gate)."""
    if MA_TOKEN:
        return MA_TOKEN
    if MA_TOKEN_FILE:
        try:
            return Path(MA_TOKEN_FILE).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


def _api(command: str, args: dict[str, Any] | None = None) -> Any:
    """One MA server API call: POST {MA_URL}/api with the command in the
    body (the live-verified 2.9.11 contract). Small, synchronous, bounded."""
    body = json.dumps({"command": command, "args": args or {}}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{MA_URL}/api", data=body,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=MA_TIMEOUT_S) as resp:
            return json.load(resp)
    except Exception as exc:  # noqa: BLE001 - single boundary, single meaning
        raise MAQueueError(f"MA {command} failed: {exc}") from exc


def probe(timeout: float | None = None) -> dict[str, Any]:
    """READ-ONLY reachability and credential check for diagnostics: one
    players/all call whose HTTP status is kept apart from every other
    failure, so a rejected token can be reported as exactly that. Never
    raises; never mutates anything."""
    out: dict[str, Any] = {"ok": False, "url": MA_URL, "http_status": None,
                           "error": "", "players": 0,
                           "token_present": bool(_token())}
    if not MA_URL:
        out["error"] = "not configured"
        return out
    body = json.dumps({"command": "players/all", "args": {}}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{MA_URL}/api", data=body, headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout or MA_TIMEOUT_S) as resp:
            data = json.load(resp)
            out["http_status"] = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
        out["http_status"] = exc.code
        out["error"] = f"HTTP {exc.code}"
        return out
    except Exception as exc:  # noqa: BLE001 - one boundary, one meaning
        out["error"] = str(exc)[:200]
        return out
    out["ok"] = True
    out["players"] = len(data) if isinstance(data, list) else 0
    return out


# --- MA server version / foreign-media contract ------------------------------
# ONE place decides how a finite FOREIGN HTTP item is referenced when enqueued.
# MA 2.11+ silently DROPS a bare http URL passed to play_media and requires
# `builtin://track/<url>` (proven against MA 2.11); 2.10.x accepts
# the bare URL. Detected once from /info,
# cached, and pinned by tests. Detection failure falls back to the 2.10.x
# bare-URL form, so an unreachable probe can never alter production behaviour.

_MA_VERSION: tuple[int, int] | None = None
_MA_VERSION_PROBED = False


def _probe_server_version() -> tuple[int, int] | None:
    try:
        req = urllib.request.Request(f"{MA_URL}/info", method="GET")
        with urllib.request.urlopen(req, timeout=MA_TIMEOUT_S) as resp:
            info = json.load(resp)
        raw = str(info.get("server_version") or "")
        major, minor = raw.split(".")[:2]
        return (int(major), int(minor))
    except Exception:  # noqa: BLE001 - best-effort; None means "unknown"
        return None


def ma_server_version() -> tuple[int, int] | None:
    """(major, minor) of the connected MA server, probed once from /info and
    cached; None when undeterminable."""
    global _MA_VERSION, _MA_VERSION_PROBED
    if not _MA_VERSION_PROBED:
        _MA_VERSION = _probe_server_version()
        _MA_VERSION_PROBED = True
    return _MA_VERSION


def set_ma_version_for_test(version: tuple[int, int] | None) -> None:
    """Pin the detected MA version (tests only)."""
    global _MA_VERSION, _MA_VERSION_PROBED
    _MA_VERSION = version
    _MA_VERSION_PROBED = True


def reset_ma_version_cache() -> None:
    """Forget the probed version so the next read re-detects (tests only)."""
    global _MA_VERSION, _MA_VERSION_PROBED
    _MA_VERSION = None
    _MA_VERSION_PROBED = False


def foreign_media_requires_builtin_wrapper() -> bool:
    """True on MA 2.11+; False on 2.10.x and on an undetectable server."""
    v = ma_server_version()
    return v is not None and v >= (2, 11)


def foreign_media_ref(url: str) -> str:
    """The media reference to enqueue a finite FOREIGN HTTP item, correct for
    the connected MA version. Wraps http(s) URLs as `builtin://track/<url>` on
    2.11+ ONLY; every other URI (already builtin://, a local library:// track,
    ...) is returned unchanged, so this is safe to apply to any enqueue list."""
    if url.startswith(("http://", "https://")) \
            and foreign_media_requires_builtin_wrapper():
        return f"builtin://track/{url}"
    return url


#: ParlorWave program audio handed to MA as a SOUND EFFECT (the prototype,
#: playback.program_audio_as_sound_effect). A supported MA URI on 2.10:
#: helpers/uri parses `provider://media_type/item_id`, and the builtin
#: provider resolves media_type sound_effect through parse_item(
#: requested_media_type=SOUND_EFFECT). Why it matters, from MA's source:
#: queue_loader records only TRACK/ALBUM/PLAYLIST/ARTIST (or a dynamic
#: source) as Autoplay seeds; audio_buffer never starts analysis for a
#: sound effect, so Sonic Similarity never indexes it; the playlog skips it.
#: An MA without the type parses `sound_effect` as UNKNOWN and the builtin
#: provider falls back to its own tag test (ParlorWave audio is tagged, so:
#: a track, the old behaviour).
SOUND_EFFECT_PREFIX = "builtin://sound_effect/"


def program_media_ref(url: str, as_sound_effect: bool | None = None) -> str:
    """The media reference for one item ParlorWave inserts. Its OWN program
    audio (a break or bulletin URL, media_ref.programming_ref) becomes a
    sound effect when the prototype setting is on AND the bytes behind the
    URL were levelled for it (program_loudness.url_levelled). Everything
    else is exactly foreign_media_ref, never a sound effect: a local
    library track a request inserts, any foreign URL that is not ours, and
    our own clip when it was not levelled. MA does not normalise a sound
    effect, so an unlevelled clip goes in as a track, which MA normalises."""
    import media_ref
    if as_sound_effect is None:
        import settings
        as_sound_effect = settings.program_audio_as_sound_effect()
    if (as_sound_effect and url.startswith(("http://", "https://"))
            and media_ref.programming_ref(url)[0]):
        import program_loudness
        if program_loudness.url_levelled(url):
            return SOUND_EFFECT_PREFIX + url
        kind, token = media_ref.programming_ref(url)
        import logging
        logging.getLogger("musicbrain").warning(
            "program audio %s %s is not levelled for the sound-effect path; "
            "inserted as a track, which Music Assistant normalises",
            kind, token[:8])
    return foreign_media_ref(url)


def refuse_program_tail(queue_id: str, state_fn=None) -> None:
    """A sound effect must never be the queue's LAST item: MA's Autoplay
    does not refill after one (autoplay AUTOPLAY_EXCLUDED_MEDIA_TYPES) and
    MA never marks a queue ended on one (playback_tracker
    UNENDABLE_MEDIA_TYPES). An item inserted `next` is last exactly when the
    current item is the last one. Raises MAQueueError -- which every caller
    already reads as "no break / no news, music untouched" -- when that is
    so or cannot be told."""
    st = (state_fn or queue_state)(queue_id)
    items, cur = st.get("items"), st.get("current_index")
    if not (isinstance(items, int) and isinstance(cur, int)
            and not isinstance(items, bool) and not isinstance(cur, bool)):
        raise MAQueueError("program_tail_unknown")
    if cur >= items - 1:
        raise MAQueueError("program_would_be_queue_tail")


def _item_brief(item: Any) -> dict[str, Any] | None:
    """Read-only identity/duration summary of one queue item. queue_item_id
    is the adjacency identity (unique per queue entry, so the same song
    queued twice cannot be confused)."""
    if not isinstance(item, dict):
        return None
    media = item.get("media_item") if isinstance(item.get("media_item"),
                                                  dict) else {}
    album = media.get("album") if isinstance(media.get("album"), dict) else {}
    return {
        "name": item.get("name"),
        "uri": media.get("uri"),
        "queue_item_id": item.get("queue_item_id"),
        "duration": item.get("duration"),
        "index": item.get("index"),
        # Album-run evidence: MA's stable album container identity and
        # real track number, when present. Stronger than album-name strings
        # or filename parsing; absent for program items and some providers.
        "album_uri": album.get("uri"),
        "track_number": media.get("track_number"),
    }


def queue_state(queue_id: str) -> dict[str, Any]:
    """Structural + musical-time state for one queue (READ ONLY).

    current_index / index_in_buffer / items are the structural gate evidence
    (unchanged, still authoritative). The additional fields are planning
    evidence for airtime-aware scheduling: MA's own elapsed_time (correct
    even in Flow mode, unlike HA's media_position) plus the current/next
    item identities and durations. Planning evidence chooses WHEN TO RENDER;
    it is never the insertion gate.

    Raises MAQueueError when the queue cannot be read -- the caller treats
    that as a closed gate."""
    if not queue_id:
        raise MAQueueError("no queue_id")
    queues = _api("player_queues/all")
    if not isinstance(queues, list):
        raise MAQueueError("unexpected player_queues/all response")
    for q in queues:
        if isinstance(q, dict) and q.get("queue_id") == queue_id:
            return {
                "current_index": q.get("current_index"),
                "index_in_buffer": q.get("index_in_buffer"),
                "items": q.get("items"),
                "state": q.get("state"),
                "elapsed_time": q.get("elapsed_time"),
                "elapsed_time_last_updated": q.get("elapsed_time_last_updated"),
                "current_item": _item_brief(q.get("current_item")),
                "next_item": _item_brief(q.get("next_item")),
                # Dynamic-source flag (Pandora): corroborates the source
                # mode. None on installs that do not report it.
                "is_dynamic": q.get("is_dynamic"),
                # MA's per-queue "Don't stop the music" flag, surfaced from
                # the same snapshot so queue ownership can re-check it at
                # every refill evaluation without a second MA call. None
                # when the install does not report it (older MA), which the
                # ownership policy treats as "not measured", never as off.
                "autoplay_enabled": _autoplay_flag(q),
                # The listener's shuffle setting, READ ONLY: nothing in
                # Music Brain ever writes it. It changes what a MULTI-item
                # option:"next" does (MA 2.10 queue_loader: the first item
                # plays next and the rest of the batch is shuffled into the
                # tail), so the news block reads it before inserting. None
                # when the install does not report it.
                "shuffle_enabled": (q.get("shuffle_enabled")
                                    if isinstance(q.get("shuffle_enabled"), bool)
                                    else None),
                # MA's Flow Mode for this queue, READ ONLY: whether the
                # player is fed one continuous stream. A house speaker may
                # take a station by handover only in Flow Mode (station_live
                # .check_target). None when the install does not report it,
                # which never qualifies anything.
                "flow_mode": (q.get("flow_mode")
                              if isinstance(q.get("flow_mode"), bool) else None),
            }
    raise MAQueueError(f"queue not found: {queue_id}")


def raw_queue(queue_id: str) -> dict[str, Any]:
    """One queue's FULL object via `player_queues/get` (READ ONLY) --
    current_item with complete media_item metadata (artwork proxy ids,
    lyrics, album detail), which the trimmed queue_state deliberately
    omits. The Booth's surfaces read from here; nothing here can write."""
    if not queue_id:
        raise MAQueueError("no queue_id")
    queue = _api("player_queues/get", {"queue_id": queue_id})
    if not isinstance(queue, dict):
        raise MAQueueError(f"queue not found: {queue_id}")
    return queue


def _autoplay_flag(queue: dict[str, Any]) -> bool | None:
    """The per-queue Autoplay flag as MA reports it. 2.10.x serialises the
    field as `autoplay_enabled`; the same flag was named
    `dont_stop_the_music_enabled` on the wire in earlier releases and MA
    still accepts that alias, so both spellings are read."""
    for key in ("autoplay_enabled", "dont_stop_the_music_enabled"):
        value = queue.get(key)
        if isinstance(value, bool):
            return value
    return None


def queue_autoplay(queue_id: str) -> bool | None:
    """Read ONE queue's Autoplay flag (READ ONLY) via `player_queues/get`.

    Returns True/False, or None when MA does not report the flag. Raises
    MAQueueError when the queue cannot be read at all. This is the
    verification read behind queue ownership: a write is only ever believed
    after this says so."""
    if not queue_id:
        raise MAQueueError("no queue_id")
    queue = _api("player_queues/get", {"queue_id": queue_id})
    if not isinstance(queue, dict):
        raise MAQueueError(f"queue not found: {queue_id}")
    return _autoplay_flag(queue)


def set_queue_autoplay(queue_id: str, enabled: bool) -> None:
    """Set ONE queue's Autoplay flag: `player_queues/autoplay`.

    THE ONLY Music Assistant configuration write in Music Brain, and it is
    scoped to a single queue id on purpose. It never touches the global
    `autoplay_mode` / `autoplay_playlist`, never another queue, and never
    the transport: MA's handler flips the flag, signals an update, and (only
    when ENABLING with a short queue) schedules its own fill. Music Brain
    only ever calls this with `enabled=False` for a queue a ParlorWave
    station owns; the `enabled` parameter exists so a test can pin that. No
    retry: the caller reads the flag back and treats an unverified write as
    unverified."""
    if not queue_id:
        raise MAQueueError("no queue_id")
    _api("player_queues/autoplay",
         {"queue_id": queue_id, "autoplay_enabled": bool(enabled)})


def list_players() -> list[dict[str, Any]]:
    """The MA player inventory (READ ONLY): candidate ParlorWave DJ outputs.

    One live-verified command (players/all), no mutation of any kind. Each
    entry keeps only what output selection needs -- identity, display name,
    provider, availability, type -- because this feeds a UI selector, not a
    transport decision. player_id IS the queue_id for every MA player, which
    is what makes the selector's value directly usable as a station target.

    Raises MAQueueError when MA cannot be read; the caller reports an
    unavailable inventory rather than inventing one."""
    players = _api("players/all")
    if not isinstance(players, list):
        raise MAQueueError("unexpected players/all response")
    out = []
    for p in players:
        if not isinstance(p, dict) or not p.get("player_id"):
            continue
        out.append({
            "player_id": p.get("player_id"),
            "name": p.get("display_name") or p.get("name") or p.get("player_id"),
            "provider": p.get("provider"),
            "type": p.get("type"),
            "available": bool(p.get("available")),
            "state": p.get("state"),
        })
    return out


def player_topology() -> list[dict[str, Any]]:
    """The MA player inventory WITH its grouping facts (READ ONLY): what the
    Station Programmer needs to prove a target is isolated before it takes
    programming authority over it. Same one command as list_players
    (players/all); the extra fields are MA 2.10's own:

        type          "player" | "group" | "stereo_pair" | ...
        synced_to     the leader this player is synced to, or None
        active_group  the group player currently driving it, or None
        group_childs  a group's member player ids ([] for a plain player)

    A speaker that is a member of any group, synced to anything, or itself
    a group is never proven isolated -- the caller refuses it. Raises
    MAQueueError when MA cannot be read; an unreadable inventory is never
    an isolated one."""
    players = _api("players/all")
    if not isinstance(players, list):
        raise MAQueueError("unexpected players/all response")
    out = []
    for p in players:
        if not isinstance(p, dict) or not p.get("player_id"):
            continue
        childs = p.get("group_childs")
        if childs is None:
            childs = p.get("group_members")
        out.append({
            "player_id": str(p.get("player_id")),
            "name": p.get("display_name") or p.get("name") or p.get("player_id"),
            "provider": p.get("provider"),
            "type": p.get("type"),
            "available": bool(p.get("available")),
            "state": p.get("playback_state") or p.get("state"),
            # Reported, never decided on: an idle MA sync group says
            # powered=true while its member plays its own queue (MA 2.10).
            # between songs; None when the install does not report power.
            "powered": (p.get("powered") if isinstance(p.get("powered"), bool)
                        else None),
            "synced_to": p.get("synced_to") or None,
            "active_group": p.get("active_group") or None,
            "group_childs": [str(c) for c in (childs or []) if c],
        })
    return out


def insertion_gate_open(state: dict[str, Any]) -> bool:
    """The authoritative Flow-mode structural gate. Both indexes must be
    present and EQUAL; anything unknown is a closed gate, never a guess."""
    cur, buf = state.get("current_index"), state.get("index_in_buffer")
    if cur is None or buf is None:
        return False
    return cur == buf


def current_item_detail(queue_id: str) -> dict[str, Any] | None:
    """The currently-playing queue item with artist detail (READ ONLY):
    {title, artists, uri, queue_item_id} -- MA's canonical identity for
    'this song'. None when the queue has no current item."""
    if not queue_id:
        raise MAQueueError("no queue_id")
    queues = _api("player_queues/all")
    if not isinstance(queues, list):
        raise MAQueueError("unexpected player_queues/all response")
    for q in queues:
        if isinstance(q, dict) and q.get("queue_id") == queue_id:
            item = q.get("current_item")
            if not isinstance(item, dict):
                return None
            media = item.get("media_item") or {}
            artists = [a.get("name") for a in (media.get("artists") or [])
                       if isinstance(a, dict) and a.get("name")]
            return {"title": media.get("name") or item.get("name"),
                    "artists": artists,
                    "uri": media.get("uri"),
                    "queue_item_id": item.get("queue_item_id")}
    raise MAQueueError(f"queue not found: {queue_id}")


def search_library_tracks(search: str, limit: int = 8) -> list[dict[str, Any]]:
    """Library track search (READ ONLY): [{name, uri, artists}] via the
    live-verified music/tracks/library_items command."""
    items = _api("music/tracks/library_items",
                 {"search": search, "limit": limit})
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        out.append({
            "name": it.get("name"),
            "uri": it.get("uri"),
            "artists": [a.get("name") for a in (it.get("artists") or [])
                        if isinstance(a, dict) and a.get("name")],
        })
    return out


# MA's own DEFAULT page size for player_queues/items: on a queue of several
# hundred items, a call with NO `limit` returns
# exactly 500 items and silently omits the rest. An EXPLICIT limit is
# honoured beyond it (a larger limit returns that many, or all). So the 500 is
# a default, not a ceiling -- but a caller that omits `limit` and calls the
# result "the queue" is simply wrong.
MA_ITEMS_DEFAULT_LIMIT = 500

# Page size used when walking a queue. Any value works; pages are terminated
# by a SHORT page, which is only a reliable end-of-list signal when the page
# was explicitly requested (see read_future_items).
ITEMS_PAGE = 500

# How far back to scan when the current_index positional hint does not land on
# the current item. Bounds the worst case on a huge queue.
MAX_ANCHOR_SCAN = 5000


def list_queue_items(queue_id: str, *, limit: int | None = None,
                     offset: int | None = None) -> list[dict[str, Any]]:
    """One PAGE of the queue's items, in order, as raw item objects (READ ONLY).

    This is the ONLY command that exposes what is actually queued:
    player_queues/all reports `items` as an integer COUNT.

    ### The verified contract (three facts a small test queue would
    ### hide)

    **`limit` is NOT optional for any completeness-sensitive read.** Omitting
    it does not mean "everything": MA applies a default of
    MA_ITEMS_DEFAULT_LIMIT (500) and truncates in silence. A measurement
    reads as "no cap" when the queue happens to hold exactly 500
    items, so "all" and "capped" are indistinguishable. On a longer queue,
    `{}` returns 500 and the appended tail is invisible. Pass an explicit
    limit, or use read_future_items() which pages.

    **`offset` is a LIST POSITION, not a `sort_index`.** On a queue
    whose sort_index runs 0,1,2,4,5: offset=3 returns the item at
    sort_index 4, offset=4 returns sort_index 5, offset=5 returns
    []. Reading offset as "a starting sort_index" is right only
    while the two coincide.

    **`sort_index` is strictly increasing but NOT contiguous.** An
    option:"add" append can produce ...2, 4, 5 -- value 3 does not
    exist. sort_index is therefore DIAGNOSTIC METADATA ONLY. Never derive
    positions, counts, futures or tails from arithmetic on it.

    **`index` is not the position either** -- MA reports it as 0 on queue
    items.

    Ordering is the RETURNED LIST ORDER, and identity is `queue_item_id`.
    Those two are the only positional facts this module trusts.

    Items are returned WHOLE -- nothing is normalized away, because callers
    need queue_item_id, media_item.uri, duration and availability, and future
    callers will want more.

    UNKNOWN QUEUES ARE INDISTINGUISHABLE FROM EMPTY ONES: MA returns [] for a
    queue_id that does not exist exactly as for a queue with no items. Only
    queue_state() distinguishes them (it raises "queue not found"), and it is
    the read every boundary caller already performs.

    Cost: a 500-item page is tens of milliseconds and a few MB; a 40-item
    window is a few milliseconds and well under a MB, about what the
    player_queues/all read already costs.

    Raises MAQueueError when the queue cannot be read -- a closed gate,
    exactly like every other read here."""
    if not queue_id:
        raise MAQueueError("no queue_id")
    args: dict[str, Any] = {"queue_id": queue_id}
    if limit is not None:
        args["limit"] = limit
    if offset is not None:
        args["offset"] = offset
    items = _api("player_queues/items", args)
    if not isinstance(items, list):
        raise MAQueueError("unexpected player_queues/items response")
    return [it for it in items if isinstance(it, dict)]


def read_future_items(queue_id: str, current_queue_item_id: str,
                      *, current_index: int | None = None,
                      total_items: int | None = None,
                      page: int = ITEMS_PAGE) -> dict[str, Any]:
    """Every item queued AFTER the currently-playing one, in order (READ ONLY).

    Returns a RESULT, not a bare list, because "we could not see all of it"
    and "there is nothing left" are different answers and must never collapse
    into each other:

        {"future":           [item, ...]   ordered, exclusive of the current item
         "complete":         bool          the whole future region was read
         "current_position": int           list position the anchor was found at
         "scanned":          int           items examined
         "incomplete_reason": str          '' when complete}

    ANCHORING IS BY IDENTITY, NOT ARITHMETIC. The current item is located by
    matching `current_queue_item_id` -- from the player_queues/all snapshot
    the caller already holds -- against the returned list. Everything AFTER
    that list position is the future. This is unit-free: it cannot be broken
    by sort_index gaps (they exist), and it does not depend on
    whether MA's `current_index` is a position or a sort_index, which remains
    UNPROVEN because separating them would need playback to advance past a
    gap. No title/artist fallback: identity or nothing.

    `current_index` is used only as a POSITIONAL HINT, and only when it pays
    off: the first page is requested at that offset, and if the item found
    there carries the expected queue_item_id the anchor is confirmed without
    ever transferring history. That matters -- a mature station queue keeps
    every completed item (hundreds of them, after a long run), so scanning
    from zero would move megabytes at every boundary. A hint that does not
    land simply falls back to a bounded scan from position 0; a WRONG hint
    therefore costs time, never correctness.

    If the anchor is never found, this raises MAQueueError. That is the
    closed-gate contract: not knowing where playback is must never be
    reported as "no future items", because that reads as a drained runway and
    would trigger exactly the refill a confused caller should not perform.

    `total_items` (from the same snapshot) is used to cross-check
    completeness. Paging otherwise ends on a SHORT page, which is a reliable
    end-of-list signal only because every page here is requested with an
    explicit `limit`.

    Raises MAQueueError on an unreadable queue, a missing anchor id, or a
    non-positive page size."""
    if not current_queue_item_id:
        raise MAQueueError("no current_queue_item_id: cannot anchor the future")
    if page <= 0:
        raise MAQueueError(f"page must be positive: {page!r}")

    def _fetch(offset: int) -> list[dict[str, Any]]:
        return list_queue_items(queue_id, limit=page, offset=offset)

    def _index_of(items: list[dict[str, Any]]) -> int | None:
        for i, it in enumerate(items):
            if it.get("queue_item_id") == current_queue_item_id:
                return i
        return None

    start_offset = 0
    buffered: list[dict[str, Any]] = []
    anchor: int | None = None
    last_page_short = False

    # Fast path: trust current_index as a POSITIONAL hint, then verify.
    if isinstance(current_index, int) and not isinstance(current_index, bool) \
            and current_index >= 0:
        first = _fetch(current_index)
        hit = _index_of(first)
        if hit is not None:
            start_offset, buffered = current_index, first
            anchor = current_index + hit
            last_page_short = len(first) < page

    # Slow path: bounded scan from the top until the anchor appears.
    if anchor is None:
        start_offset, buffered, offset = 0, [], 0
        while offset < MAX_ANCHOR_SCAN:
            chunk = _fetch(offset)
            if not chunk:
                last_page_short = True
                break
            buffered.extend(chunk)
            offset += len(chunk)
            hit = _index_of(buffered)
            if len(chunk) < page:
                last_page_short = True
                anchor = hit
                break
            if hit is not None:
                anchor = hit
                break

    if anchor is None:
        raise MAQueueError(
            f"current item {current_queue_item_id} not found in queue "
            f"{queue_id}: refusing to report a runway we cannot anchor")

    future = buffered[anchor - start_offset + 1:]
    next_offset = start_offset + len(buffered)
    complete = last_page_short
    while not complete:
        chunk = _fetch(next_offset)
        if not chunk:
            complete = True
            break
        future.extend(chunk)
        next_offset += len(chunk)
        complete = len(chunk) < page
    scanned = next_offset

    reason = ""
    if complete and isinstance(total_items, int) and scanned < total_items:
        # A short page said "end of list" but the snapshot says otherwise --
        # the queue grew, or MA truncated. Either way this is not complete.
        complete, reason = False, (
            f"read {scanned} of {total_items} reported items")
    elif not complete:
        reason = "paging stopped before the end of the queue"
    return {"future": future, "complete": complete,
            "current_position": anchor, "scanned": scanned,
            "incomplete_reason": reason}


#: Default raw-item margins for read_window_items. These are RAW queue
#: items, not music items: a mature station queue interleaves DJ breaks and
#: bulletins with the songs (one program item in every two or three
#: is not unusual), so a window must be roughly
#: three times the number of MUSIC items its caller wants on each side.
#: 12 back + 24 ahead comfortably covers the DJ runway's 3 back / 4 ahead
#: even through a dense run of programming.
#:
#: Cost is the reason these are small. A 40-item window costs a few ms and
#: well under a MB -- about what the player_queues/all read already costs --
#: while a 500-item page costs tens of ms and a few MB. This read happens
#: once per break render, so it must stay in the cheap band.
WINDOW_BACK = 12
WINDOW_AHEAD = 24


def read_window_items(queue_id: str, current_queue_item_id: str, *,
                      current_index: int | None = None,
                      back: int = WINDOW_BACK,
                      ahead: int = WINDOW_AHEAD) -> dict[str, Any]:
    """One bounded window of queue items STRADDLING the current one, in
    order (READ ONLY).

        {"items":            [item, ...]  ordered, INCLUDING the current one
         "base_position":    int          list position of items[0]
         "current_position": int          list position of the current item
         "scanned":          int          items examined}

    WHY THIS EXISTS. Callers that want the music AROUND playback -- the DJ's
    runway -- need history as well as future, so read_future_items (which is
    exclusive of the current item) cannot serve them. The naive alternative,
    `list_queue_items(qid, limit=N)`, reads from position ZERO: on a mature
    station queue (hundreds of items, the head far down the list) that
    returns the first N items of the day and nothing near playback at all.

    ANCHORING IS BY IDENTITY, NOT ARITHMETIC -- the same contract
    read_future_items keeps, and for the same reasons. `current_index` is a
    positional HINT only: the window is requested at `current_index - back`
    and accepted only if `current_queue_item_id` is actually in it. A hint
    that does not land costs a bounded scan, never correctness, because
    sort_index gaps are real and whether MA's `current_index` is a
    position or a sort_index remains unproven.

    IF THE ANCHOR IS NEVER FOUND THIS RAISES, exactly like
    read_future_items. A runway we cannot place is not a runway starting at
    position 0: that would describe the beginning of the queue as though it
    were the music around the listener, which is worse than no runway at
    all. The caller's fence turns that into "no runway", which is honest.

    `back` and `ahead` are RAW item margins, not music counts; the caller
    filters. Both must be non-negative.

    Raises AnchorNotFound when the head is not in the queue's item list,
    and MAQueueError on an unreadable queue, a missing
    current_queue_item_id, or negative margins."""
    if not current_queue_item_id:
        raise MAQueueError("no current_queue_item_id: cannot anchor a window")
    if back < 0 or ahead < 0:
        raise MAQueueError(f"margins must be non-negative: {back!r}/{ahead!r}")

    def _index_of(items: list[dict[str, Any]]) -> int | None:
        for i, it in enumerate(items):
            if it.get("queue_item_id") == current_queue_item_id:
                return i
        return None

    scanned = 0

    # Fast path: trust current_index as a POSITIONAL hint, then verify.
    # Clamping at zero keeps the ahead margin whole -- a head near the top
    # of the queue simply has less history, never less runway.
    if isinstance(current_index, int) and not isinstance(current_index, bool) \
            and current_index >= 0:
        offset = max(0, current_index - back)
        limit = (current_index - offset) + 1 + ahead
        page = list_queue_items(queue_id, limit=limit, offset=offset)
        scanned = len(page)
        hit = _index_of(page)
        if hit is not None:
            return {"items": page, "base_position": offset,
                    "current_position": offset + hit, "scanned": scanned}

    # Slow path: bounded scan from the top until the anchor appears, then
    # cut the window out of what was read. Only reached when there is no
    # hint or the hint was wrong.
    buffered: list[dict[str, Any]] = []
    offset = 0
    anchor: int | None = None
    while offset < MAX_ANCHOR_SCAN:
        chunk = list_queue_items(queue_id, limit=ITEMS_PAGE, offset=offset)
        if not chunk:
            break
        buffered.extend(chunk)
        offset += len(chunk)
        scanned = offset
        hit = _index_of(buffered)
        if hit is not None:
            anchor = hit
            break
        if len(chunk) < ITEMS_PAGE:
            break

    if anchor is None:
        raise AnchorNotFound(
            f"current item {current_queue_item_id} not found in queue "
            f"{queue_id}: refusing to report a runway we cannot anchor")

    start = max(0, anchor - back)
    return {"items": buffered[start:anchor + 1 + ahead],
            "base_position": start, "current_position": anchor,
            "scanned": scanned}


def replace_future(queue_id: str, uris: list[str]) -> None:
    """Replace everything AFTER the current item with the given URIs (MA
    QueueOption 'replace_next'). The current item keeps playing at its
    current position -- this changes FUTURE programming only. Raises
    MAQueueError on any failure; the caller must treat that as 'queue
    unchanged'. Still no pause/stop/skip/seek anywhere in this module."""
    if not queue_id or not uris:
        raise MAQueueError("queue_id and uris required")
    _api("player_queues/play_media", {
        "queue_id": queue_id,
        "media": list(uris),
        "option": "replace_next",
    })


def insert_next(queue_id: str, url: str, title: str, artist: str) -> None:
    """Insert one finite audio URL as the NEXT item on the queue. Raises
    MAQueueError on any failure; the caller's contract is that a failed
    enqueue simply means no break -- music continues untouched."""
    if not queue_id or not url:
        raise MAQueueError("queue_id and url required")
    # Live-verified 2.9.11 argument shape: media is a list of URIs.
    # Display identity (title/artist) comes from the MP3's ID3 tags.
    insert_next_many(queue_id, [url])


def insert_next_many(queue_id: str, urls: list[str]) -> None:
    """Insert several finite audio URLs as the NEXT items, IN ORDER, in ONE
    queue mutation.

    This exists so a news event is atomic: the DJ intro and the bulletin
    it announces are enqueued together or not at all. Two separate writes
    could interleave with MA's own buffering and produce the one outcome
    that is worse than no news -- the DJ announcing a bulletin that never
    follows, or the two arriving reversed.

    Same command and same `media` list parameter that replace_future already
    uses with ordered multi-URI lists in production; only `option` differs
    ('next' rather than 'replace_next'), so nothing after the inserted items
    is disturbed and the existing music runway survives.

    NOTE: `option: next` with a MULTI-item list keeps its order ONLY while
    the queue's shuffle is off. With shuffle on, MA 2.10 plays the first
    item next and shuffles the rest of the batch, together with the queue's
    upcoming tail, into random positions (queue_loader `shuffle =
    queue.shuffle_enabled and len(queue_items) > 1`, which separates a
    news bulletin from its intro). A single-item list is never
    shuffled, so news_schedule inserts one item at a time on such a queue.

    Raises MAQueueError on any failure; a failed enqueue means no news and
    untouched music, exactly like a failed break enqueue."""
    if not queue_id or not urls:
        raise MAQueueError("queue_id and urls required")
    # Foreign HTTP media is referenced per the connected MA version: bare on
    # 2.10.x, builtin://track/<url> on 2.11+;
    # our own program audio as a sound effect when the prototype is on.
    media = [program_media_ref(u) for u in urls]
    if any(m.startswith(SOUND_EFFECT_PREFIX) for m in media):
        refuse_program_tail(queue_id)
    _api("player_queues/play_media", {
        "queue_id": queue_id,
        "media": media,
        "option": "next",
    })


def play_queue_index(queue_id: str, index: int) -> None:
    """Start playback at an ordered queue position: `player_queues/
    play_index`. THE ONE TRANSPORT COMMAND IN MUSIC BRAIN, and it exists
    for exactly one caller: queue_recovery's one-shot resume of an OWNED
    station queue that Music Assistant left idle at a rewound position
    after a flow-restart (full signature
    and justification in queue_recovery's module doc). Nothing else may
    call this -- pinned by test. Still no pause/stop/skip/seek anywhere
    in this module."""
    if not queue_id:
        raise MAQueueError("no queue_id")
    _api("player_queues/play_index", {"queue_id": queue_id,
                                      "index": int(index)})


def start_station(queue_id: str, station_uri: str) -> None:
    """Start a source/station on a queue: `player_queues/play_media` with
    option:"play". The Pandora handoff's SECOND (and last) transport command,
    for exactly one caller: the local->Pandora handoff executor's one-shot
    start of an already-selected station once the intended transition item
    has finished.
    Starting a station is a source-SELECT that rebuilds the upcoming queue (a
    start, like play_index -- never pause/stop/skip/seek). Nothing else may
    call this -- pinned by test."""
    if not (queue_id and station_uri):
        raise MAQueueError("queue_id and station_uri required")
    _api("player_queues/play_media", {"queue_id": queue_id,
                                      "media": station_uri, "option": "play"})


def append_tracks(queue_id: str, uris: list[str]) -> Any:
    """Append tracks at the END of the queue (MA QueueOption 'add').

    The station-replenishment primitive, and the only write here that adds
    without displacing: `replace_next` rewrites the whole future and `next`
    takes the immediate slot, but 'add' lands strictly after every existing
    item -- so a listener's manual additions, a queued DJ break and the
    atomic news pair are all untouched by construction.

    Verified against MA 2.9, on a playing Flow-Mode queue of
    several hundred items: both items landed after every pre-existing
    item, in the order sent, with fresh unique queue_item_ids; the current
    item, its elapsed position, current_index, index_in_buffer and next_item
    were all unchanged, and every pre-existing item kept its identity and
    relative order.

    THE RESPONSE TELLS YOU NOTHING. It is HTTP 200 with a body of literally
    `null` -- no inserted ids, no echo, no acknowledgement beyond acceptance.
    The value is returned anyway for diagnostics, but a caller must NEVER
    treat it as proof: only a subsequent ordered queue read can establish
    whether the append landed, and that read is the sole reconciliation
    evidence that will ever exist.

    NO IMPLICIT RETRY, here or anywhere above. An append whose outcome is
    unknown must not be repeated on a guess -- the duplicate it would create
    is worse than the gap it was trying to fill. Raises MAQueueError like
    every other write; the caller reconciles from the queue.

    Still no pause/stop/skip/seek anywhere in this module."""
    if not queue_id or not uris:
        raise MAQueueError("queue_id and uris required")
    return _api("player_queues/play_media", {
        "queue_id": queue_id,
        "media": list(uris),
        "option": "add",
    })


# --- queue hygiene: the one removal verb ---------------------------------------

#: The raw MA command. Pinned by test to appear in THIS file only: every other
#: module removes a queue item through delete_queue_item or not at all.
DELETE_ITEM_COMMAND = "player_queues/delete_item"


def delete_queue_item(queue_id: str, queue_item_id: str) -> Any:
    """Remove ONE item from a queue by its exact `queue_item_id` string.

    IDENTITY, NEVER POSITION. Music Assistant's handler accepts either an
    item id or a list index in the same argument; this wrapper accepts the
    id only. An int (or bool) is refused with TypeError before any request
    is built, because a position is a guess about where an item is, and a
    queue that shifts under the caller turns that guess into somebody
    else's song. The exact string id cannot name the wrong item.

    WHAT MA DOES WITH IT (from the MA source, to be verified live on 2.10
    before any automatic trigger is armed): resolves the id to its current
    index; refuses, silently, an index at or below `index_in_buffer` (the
    player already holds that audio); pops the item and republishes the
    queue. The response body is null -- an accepted request proves nothing.
    The caller must re-read the ordered queue and treat "still present" as
    a refusal, exactly as append_tracks callers reconcile from the queue.

    NO RETRY, here or above: a delete whose outcome is unknown is answered
    by reading the queue, never by sending it again. Raises MAQueueError on
    any transport failure or rejection, like every other write here.

    Still no pause/stop/skip/seek/move/clear anywhere in this module."""
    if isinstance(queue_item_id, bool) or not isinstance(queue_item_id, str):
        raise TypeError(
            "delete_queue_item takes the exact queue_item_id string, never "
            f"an index: {queue_item_id!r}")
    if not queue_id or not queue_item_id:
        raise MAQueueError("queue_id and queue_item_id required")
    return _api(DELETE_ITEM_COMMAND, {"queue_id": queue_id,
                                      "item_id_or_index": queue_item_id})
