#!/usr/bin/env python3
"""The Station Programmer on air: Parlorwave Mode 2.

    StationIntent (the Station tab's Start, the Request Line, the Dial --
         |         three callers of the ONE start_station below)
    resolve_station() -> build_candidates()      station_resolver: THE
         |                                        builder Test Station runs
    opening runway, 25 songs                      the same draw a preview
         |                                        of that variant showed
    ONE option:add append of exact library URIs   station_refill.commit_
         |                                        append's proof discipline
    refill forever: when the station's own runway falls to 8, the SAME two
    functions are called again with live exclusions, and one batch of up to
    12 is appended behind everything else in the queue

This module is the only one that holds Mode 2 programming authority. It adds
no recommendation logic: every song comes out of build_candidates, and what
is added here is LIVE STATE around it -- a target, a durable identity, the
exclusions a running station needs, and the transactions that put music in a
queue without ever guessing what happened.

WHAT A STATION IS HERE. An ordinary `stations` row (profile_source
PROFILE_PROGRAMMER) with its StationIntent in `program`. Its station_id is
its generation: a start, and every switch, mints a new one, and every
asynchronous writer re-checks it inside the station mutation lock before it
may touch the queue -- so a refill chosen for one station can never land
after the listener has switched to another.

WHAT IT MAY WRITE TO MUSIC ASSISTANT, and nothing else:
  * an option:add append (ma_queue.append_tracks) -- the opening runway and
    every refill, at the END of the queue, so manual additions, DJ breaks,
    news and announcements keep their places by construction;
  * a delete by exact queue_item_id (ma_queue.delete_queue_item) -- ONLY at
    a station switch, ONLY for the OLD station's own not-yet-buffered music,
    proven by the ownership witness recorded when it was appended;
  * the target queue's Autoplay flag set false (queue_ownership): Parlorwave,
    not MA Autoplay, owns refill.
There is no transport command anywhere in this module: no play, pause, stop,
skip, seek or index jump. Starting a station needs music already playing (or
paused) on the target; the station takes over after the current song.
Stopping a station stops PROGRAMMING, never playback.

SAFE TARGETING. The target is named explicitly by the request and proven
isolated before any write: it exists and is available, it is a plain player
(not a group), it is synced to nothing, driven by no group and a member of
no group, it is not the house's selected DJ output, and no other station is
running. Any doubt refuses. The observer is untouched: it follows the
listening session, not the station, so a station on an isolated speaker
never pulls the DJ away from the house output.

THE HOUSE HANDOVER. The one exception to isolation is the house speaker
itself -- the output the DJ station is following (often a whole-house
player). It is typically a member of several Cast groups and is where
Music Assistant's Autoplay is programming music right now, so a station
may take it over only by an explicit HANDOVER, never silently:

  * it must be a plain player in Flow Mode, synced to nothing, driven by no
    group, and no group it belongs to may be playing -- a group that is
    playing would be what the speaker plays (`powered` is NOT a signal: an
    idle Music Assistant sync group reports powered=true while its member
    plays its own queue -- as MA 2.10 behaves), and a non-Flow
    house speaker has not been qualified (unknown Flow Mode never
    qualifies; the old refusals stand);
  * the first start answers `handover_required` with what the person is
    about to take over (Autoplay on or off, the songs already queued and
    their length) and a short-lived token, and writes nothing;
  * the confirmed start names that token and one choice for the songs
    already queued: `keep` (they play first, the station follows) or
    `replace` (after the station's songs land, exactly the songs the person
    was shown are removed, by id, never a buffered one, a DJ-promised one,
    a break, a bulletin or anything that appeared after the question);
  * anything new ahead, or Autoplay flipped, since the question refuses as
    `handover_changed` with the new picture -- the person decides again;
  * Autoplay goes off for that queue, as for every owned queue, and if it
    was ON when the person confirmed, Stop turns it back on for that queue
    (queue_ownership.restore_after_handover), so the speaker returns to
    ordinary Music Assistant listening with exactly one refill authority at
    every moment: ParlorWave while the station runs, Autoplay after it.
Group membership is never changed, and no other speaker is touched.

THE CLOCK. A station refills from two existing heartbeats and nothing new:
the observed queue's boundary/tick hook (dj_engine._station_refill_pass,
which already evaluates the active station wherever it plays) and the
lifecycle reaper (every PLAYBACK_REAPER_INTERVAL_MIN). A station on a queue
the observer is not following is therefore checked at least every reaper
interval, and more often while the house plays. Tick-driven checks are
throttled to one a minute.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any

import ma_catalog
import ma_queue
import media_ref
import service_clock
import station_eligibility
import station_intent
import station_lock
import station_recovery
import station_refill
import station_resolver
import station_runway
import station_store

logger = logging.getLogger("musicbrain")

RECIPE_VERSION = 2

# --- the runway --------------------------------------------------------------
#
# Opening runway: the preview's own count, so "the station I tested" and "the
# station that opens" are the same 25-song draw on a clean queue. Refill is
# the shared eligibility policy, unchanged: at 8 station-owned songs left
# (~30 minutes at a typical ~3.5-minute song) top up to 20, at
# most 12 per batch, 90 s apart; an emergency top-up of up to 6 when only 3
# playable songs of ANY origin remain.
INITIAL_RUNWAY = station_resolver.DEFAULT_COUNT
REFILL_LOW_WATER = station_eligibility.NORMAL_LOW_WATER
REFILL_TARGET = station_eligibility.NORMAL_TARGET
REFILL_MAX_BATCH = station_eligibility.MAX_DESIRED_APPEND

#: A FRESH start appends behind what is already queued. More than this many
#: songs already ahead and the station would not be heard for a long time,
#: so the start is refused with the count -- nothing already queued is ever
#: removed to make room (a switch is different: see _retire_old_runway).
START_MAX_MUSIC_AHEAD = 5

#: Raw queue items read BEHIND the head as this queue's play history (MA's
#: own record of it, authoritative for an output the observer may not be
#: following). ~3x the selector's 40-track rotation window, because program
#: items interleave with songs.
HISTORY_BACK = 60

#: The selector's continuity floor protects the last 5 songs heard; below it
#: the selector would repeat anything but the very last song. A live station
#: never goes below the floor: a pick among the last 5 heard is dropped, and
#: an empty batch waits as `candidates_exhausted` rather than loop.
ANTI_LOOP_RECENT = 5

#: Tick-driven evaluations run at most this often (boundaries on the
#: station's own queue and reaper passes always evaluate).
MIN_EVAL_INTERVAL_S = 60.0

#: An unchanged runway check is re-recorded at most this often.
CHECK_RECORD_S = 600.0

#: Bound on the ids remembered as "already queued when the station started".
PREEXISTING_CAP = 200

#: A failed append is not judged NOT_LANDED on the first read: Music
#: Assistant may still apply a write whose response timed out. One short
#: settle, then the verdict -- for the opening, and (through commit_append's
#: settle_fn) for every refill.
WRITE_ERROR_SETTLE_S = 2.0

# --- stable reasons ---------------------------------------------------------------
R_TARGET_REQUIRED = "target_required"
R_INTENT_INVALID = "invalid_intent"
R_PRESET_NOT_FOUND = "preset_not_found"
R_PRESET_OFF_DIAL = "preset_off_dial"
R_PRESET_TOO_THIN = "preset_too_thin"
R_PRESET_NEEDS_TEST = "preset_needs_test"
R_KIND_NOT_AVAILABLE = "kind_not_available"
R_INVENTORY_UNREADABLE = "inventory_unreadable"
R_UNKNOWN_TARGET = "unknown_target"
R_TARGET_UNAVAILABLE = "target_unavailable"
R_TARGET_IS_GROUP = "target_is_group"
R_TARGET_GROUPED = "target_grouped"
R_TARGET_IN_GROUP = "target_in_group"
R_TARGET_IS_HOUSE_OUTPUT = "target_is_house_output"
R_LEGACY_STATION = "legacy_station_active"
R_ELSEWHERE = "station_active_elsewhere"
R_START_IN_FLIGHT = "start_in_flight"
R_QUEUE_UNREADABLE = "target_queue_unreadable"
R_NOT_PLAYING = "target_not_playing"
R_PROGRAMMING_ON_AIR = "programming_on_air"
R_CATALOG = "catalog_unavailable"
R_TOO_THIN = "station_too_thin"
R_MUSIC_AHEAD = "queue_has_music_ahead"
R_CHANGED = "station_changed_meanwhile"
R_REFILL_UNPROVEN = "refill_unproven"
R_BUSY = "station_mutation_busy"
R_NOT_LANDED = "runway_did_not_land"
R_UNPROVEN = "runway_unproven"
R_EXHAUSTED = "candidates_exhausted"
R_UNRESOLVABLE = "station_unresolvable"
R_SESSIONS_UNREADABLE = "session_state_unreadable"
R_ACTIVATION_LOST = "activation_lost"
R_NOT_ISOLATED = "target_not_isolated"
R_SUPERSEDED = "station_superseded"
R_HANDOVER_REQUIRED = "handover_required"
R_HANDOVER_CHANGED = "handover_changed"
R_GROUP_ACTIVE = "target_group_active"
R_HOUSE_NOT_FLOW = "target_not_flow_mode"

#: A handover question is good for this long; after it the person is shown
#: the speaker's queue again before anything is taken over.
HANDOVER_TTL_S = 300.0
#: Bound on handover questions remembered at once (one per press).
HANDOVER_CAP = 8
HANDOVER_MODES = ("keep", "replace")
#: Songs listed by name in a handover question (the rest are counted).
HANDOVER_LIST_CAP = 64
#: An Autoplay restore a Stop could not finish is never retried by itself:
#: it waits as a pending decision for the person, and is no longer offered
#: once it is this old (by then whatever is playing is the person's).
RESTORE_MAX_AGE_S = 6 * 3600.0
#: Group states in which a Cast group is driving its members.
GROUP_ACTIVE_STATES = ("playing", "paused", "buffering")

# Reasons that are PROBLEMS a person should see, as opposed to the ordinary
# weather of a running station (paused, healthy, cooldown, a break on air).
# Two families, tracked apart so neither can hide the other: a runway CHECK
# problem clears at the next check that does not see it; a REFILL problem
# clears only when a refill succeeds -- a healthy runway check says nothing
# about whether the station could refill.
CHECK_PROBLEMS = frozenset({
    station_eligibility.R_PLAYER_MISSING, station_eligibility.R_QUEUE_UNREADABLE,
    station_eligibility.R_INCOMPLETE_QUEUE,
    station_eligibility.R_ANCHOR_MISSING,
    station_eligibility.R_OWNERSHIP_UNPROVEN,
    station_eligibility.R_MA_AUTOPLAY_CONTESTED,
    station_eligibility.R_PROFILE_INVALID,
})
REFILL_PROBLEMS = frozenset({
    R_EXHAUSTED, R_UNRESOLVABLE, R_CATALOG, R_NOT_ISOLATED, "confirmation_lost",
    station_eligibility.R_MA_AUTOPLAY_CONTESTED,
    "append_did_not_land", "reconciliation_unproven",
    "reconciliation_unreadable", "queue_unreadable_or_incomplete",
})

#: The callers that start a SAVED station by its preset_id. The Request
#: Line has no preset: it sends an intent whose origin is request_line.
PRESET_ORIGINS = ("station_tab", "dial")

_MESSAGES = {
    R_PRESET_OFF_DIAL: "That station is not on the Dial. Turn on \"Show on "
                       "the Dial\" in Station Setup first.",
    R_PRESET_TOO_THIN: "Station Setup tested this station as too thin to go "
                       "live. Test it again there if the library has "
                       "changed.",
    R_PRESET_NEEDS_TEST: "This station needs a new Test Station run in "
                         "Station Setup before the Dial can start it.",
    R_TARGET_REQUIRED: "Choose the speaker the station should play on.",
    R_UNKNOWN_TARGET: "Music Assistant does not know that speaker.",
    R_TARGET_UNAVAILABLE: "That speaker is unavailable right now.",
    R_TARGET_IS_GROUP: "That output is a speaker group. A station can only "
                       "run on a single, ungrouped speaker in this release.",
    R_TARGET_GROUPED: "That speaker is currently grouped or synced with "
                      "another speaker.",
    R_TARGET_IN_GROUP: "That speaker belongs to a speaker group, so "
                       "programming it could change what the group plays.",
    R_TARGET_IS_HOUSE_OUTPUT: "That speaker is where the house's DJ station "
                              "is playing (or would play). A station cannot "
                              "take it over in this release.",
    R_SESSIONS_UNREADABLE: "ParlorWave could not tell where the house station "
                           "is playing, so no speaker could be checked.",
    R_ACTIVATION_LOST: "The songs were added but the start could not be "
                       "confirmed. Nothing further was changed.",
    R_INVENTORY_UNREADABLE: "Music Assistant's speakers could not be read, "
                            "so the speaker could not be checked.",
    R_LEGACY_STATION: "A Start Radio station is running. Stations cannot "
                      "run beside it.",
    R_ELSEWHERE: "A station is already on air on another speaker. Stop it "
                 "first.",
    R_START_IN_FLIGHT: "A station start is still being confirmed. Try "
                       "again in a moment.",
    R_QUEUE_UNREADABLE: "That speaker's queue could not be read, so nothing "
                        "was changed.",
    R_NOT_PLAYING: "Start something playing on that speaker first. The "
                   "station takes over after the current song.",
    R_PROGRAMMING_ON_AIR: "A DJ break or bulletin is on air there. Try again "
                          "after it.",
    R_CATALOG: "The Music Assistant library could not be read.",
    R_MUSIC_AHEAD: "Songs are already queued on that speaker. Clear them in "
                   "Music Assistant (or let them play), then start the "
                   "station.",
    R_CHANGED: "The station changed while this one was being built. "
               "Nothing was added.",
    R_REFILL_UNPROVEN: "The running station's last refill has not been "
                       "confirmed yet. Try again in a moment.",
    R_BUSY: "Another station change is in progress. Try again in a moment.",
    R_NOT_LANDED: "Music Assistant did not add the songs. Nothing changed.",
    R_UNPROVEN: "The songs were sent but could not be confirmed yet. "
                "ParlorWave will check the queue and finish the start "
                "without sending them again.",
    R_HANDOVER_REQUIRED: "Starting a station here hands this speaker's "
                         "programming to ParlorWave. Confirm the handover "
                         "first.",
    R_HANDOVER_CHANGED: "That speaker's queue changed while the handover "
                        "was being confirmed. Nothing was changed; look at "
                        "it again.",
    R_GROUP_ACTIVE: "A speaker group this speaker belongs to is "
                    "playing, so programming the speaker would change "
                    "what that group plays.",
    R_HOUSE_NOT_FLOW: "That speaker does not use Music Assistant's Flow "
                      "Mode. A station can take over the house speaker only "
                      "in Flow Mode in this release.",
}


class _Refusal(Exception):
    def __init__(self, reason: str, message: str = "", **extra: Any):
        super().__init__(reason)
        self.answer = {"ok": False, "reason": reason,
                       "message": message or _MESSAGES.get(reason, reason),
                       **extra}


# --- small helpers ------------------------------------------------------------------

def _iso(now: datetime | None = None) -> str:
    return (now or service_clock.legacy_now()).replace(microsecond=0).isoformat()


def _settings_doc(settings_fn=None) -> dict[str, Any]:
    try:
        if settings_fn is not None:
            return settings_fn() or {}
        import settings
        return settings.load()
    except Exception:  # noqa: BLE001 - a status read must not fail on this
        logger.warning("station programmer: settings unreadable",
                       exc_info=True)
        return {}


def _active_session_queues(db_path=None) -> list[str]:
    import playback
    c = playback._connect(db_path)
    try:
        return [r[0] for r in c.execute(
            "SELECT queue_id FROM sessions WHERE status = 'active' "
            "AND queue_id IS NOT NULL AND queue_id != ''")]
    finally:
        c.close()


def house_outputs(doc: dict[str, Any], db_path=None) -> set[str]:
    """Every output the house's DJ station is on, or would be on next,
    decided the way the observer decides (ma_observer.resolve_target): the
    selected DJ output, the queue of any active listening session, and --
    with nothing selected -- a single enabled output. FAILS CLOSED: session
    state that cannot be read raises a refusal rather than reading as
    "nowhere"."""
    outputs = ((doc.get("dj") or {}).get("outputs") or {})
    out = set()
    selected = str(outputs.get("default_request_output") or "").strip()
    if selected:
        out.add(selected)
    try:
        queues = _active_session_queues(db_path)
    except sqlite3.Error:
        raise _Refusal(R_SESSIONS_UNREADABLE) from None
    out.update(queues)
    import ma_observer
    target, _why = ma_observer.resolve_target(
        [{"queue_id": q} for q in queues], outputs, None)
    if target:
        out.add(target)
    return out


def authorized_targets(doc: dict[str, Any]) -> set[str]:
    """Speakers the OPERATOR has explicitly authorized as Station Programmer
    targets although they are house speakers (station.live.
    authorized_targets). Never inferred, empty by default, and an unreadable
    settings document authorizes nothing."""
    live = ((doc or {}).get("station") or {}).get("live") or {}
    raw = live.get("authorized_targets") if isinstance(live, dict) else None
    return {str(t).strip() for t in (raw if isinstance(raw, list) else ())
            if isinstance(t, str) and t.strip()}


def song_key(artist: Any, title: Any) -> tuple[str, str]:
    """One SONG, whatever the recording: the same (artist, base title) the
    resolver's duplicate cleanup keys on."""
    return (station_resolver.norm(artist), station_resolver.base_title(title))


def _item_song(item: dict[str, Any]) -> tuple[str, str] | None:
    media = item.get("media_item") if isinstance(item.get("media_item"),
                                                  dict) else {}
    title = media.get("name")
    artists = [a.get("name") for a in (media.get("artists") or [])
               if isinstance(a, dict) and a.get("name")]
    if not title or not artists:
        return None
    return song_key(artists[0], title)


def _uri(item: dict[str, Any]) -> str:
    return station_runway.item_uri(item)


def _intent_of(station: dict[str, Any]) -> station_intent.StationIntent:
    return station_intent.intent_from_dict(
        (station.get("program") or {}).get("intent") or {})


def is_active(db_path=None) -> bool:
    """Is a Station Programmer station on air? (A local read; no MA call.)"""
    try:
        return station_store.is_programmer(
            station_store.get_active_station(db_path=db_path))
    except sqlite3.Error:
        return False


# --- safe targeting -----------------------------------------------------------------

def check_target(target: str, players: list[dict[str, Any]], *,
                 house_outputs=(), authorized=(), flow=None) -> dict[str, Any]:
    """Is `target` PROVABLY isolated, or a house speaker that may be taken
    over by handover? PURE. {ok, reason, message, player, authorized,
    handover}.

    Every rule refuses on doubt. The house's selected DJ output is refused
    by name as well as by structure: a whole-house speaker is typically a
    member of several Music Assistant groups, and either rule alone keeps a
    station off it -- UNLESS the operator has explicitly authorized that
    speaker (`authorized`, from station.live.authorized_targets), or it
    qualifies for the HOUSE HANDOVER (module docstring): it is the house
    speaker, `flow` is True (its queue is in Flow Mode), and no group it
    belongs to is playing. Either way every other rule still
    applies to it: it must be a known, available, single speaker that is
    not grouped or synced right now. `handover` is True whenever the target
    is a house speaker: a start there must be confirmed as a handover.

    A speaker that is only a member of groups but is NOT the house speaker
    stays refused (`target_in_group`): this is the house's handover, not
    group support."""
    target = str(target or "").strip()
    if not target:
        return _verdict(R_TARGET_REQUIRED)
    player = next((p for p in players if p.get("player_id") == target), None)
    if player is None:
        return _verdict(R_UNKNOWN_TARGET)
    if not player.get("available"):
        return _verdict(R_TARGET_UNAVAILABLE, player)
    childs = [c for c in (player.get("group_childs") or []) if c != target]
    if (player.get("type") or "player") != "player" or childs:
        return _verdict(R_TARGET_IS_GROUP, player)
    if player.get("synced_to") or player.get("active_group"):
        return _verdict(R_TARGET_GROUPED, player)
    groups = sorted(str(p.get("name") or p.get("player_id")) for p in players
                    if p.get("player_id") != target
                    and target in (p.get("group_childs") or []))
    house = target in set(house_outputs or ())
    lifted = target in set(authorized or ())
    if house and not lifted:
        # THE HOUSE HANDOVER. Only a PROVEN Flow Mode queue qualifies; an
        # unknown answer falls through to the old refusals below.
        active = sorted(
            str(p.get("name") or p.get("player_id")) for p in players
            if p.get("player_id") != target
            and target in (p.get("group_childs") or [])
            and p.get("state") in GROUP_ACTIVE_STATES)
        if flow is True and active:
            return _verdict(R_GROUP_ACTIVE, player,
                            _MESSAGES[R_GROUP_ACTIVE]
                            + " (" + ", ".join(active[:4]) + ")")
        if flow is True:
            return {"ok": True, "reason": "", "message": "", "player": player,
                    "authorized": False, "handover": True,
                    "groups": groups}
        if flow is False:
            return _verdict(R_HOUSE_NOT_FLOW, player)
    if groups and not lifted:
        return _verdict(R_TARGET_IN_GROUP, player,
                        _MESSAGES[R_TARGET_IN_GROUP]
                        + " (" + ", ".join(groups[:4]) + ")")
    if house and not lifted:
        return _verdict(R_TARGET_IS_HOUSE_OUTPUT, player)
    return {"ok": True, "reason": "", "message": "", "player": player,
            # True when a lifted rule would otherwise have refused it
            "authorized": bool(lifted and (groups or house)),
            "handover": house, "groups": groups}


def _verdict(reason: str, player: dict[str, Any] | None = None,
             message: str = "") -> dict[str, Any]:
    return {"ok": False, "reason": reason,
            "message": message or _MESSAGES[reason], "player": player}


def queue_flow(queue_id: str) -> bool | None:
    """Is this queue in Music Assistant's Flow Mode? READ ONLY. None when it
    cannot be read or is not reported -- which never qualifies a handover."""
    try:
        flag = (ma_queue.queue_state(queue_id) or {}).get("flow_mode")
    except ma_queue.MAQueueError:
        return None
    return flag if isinstance(flag, bool) else None


# --- the house handover ---------------------------------------------------------------

_handover_lock = threading.Lock()
_handovers: dict[str, dict[str, Any]] = {}


def _handover_now() -> float:
    """The handover question's own timer: how long a person has had it on
    screen. Monotonic and real-time on purpose -- it is not station time."""
    return time.monotonic()


def _music_ahead(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [it for it in evidence.get("future") or ()
            if station_runway.classify_item(it)[0] == station_runway.MUSIC
            and it.get("queue_item_id")]


def _seconds(item: dict[str, Any]) -> float:
    for value in (item.get("duration"),
                  (item.get("media_item") or {}).get("duration")):
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and value > 0:
            return float(value)
    return 0.0


def _item_title(item: dict[str, Any]) -> tuple[str, str]:
    media = item.get("media_item") if isinstance(item.get("media_item"),
                                                  dict) else {}
    artists = [a.get("name") for a in (media.get("artists") or [])
               if isinstance(a, dict) and a.get("name")]
    title = str(media.get("name") or "")
    if not title:
        # "Artist - Title" is Music Assistant's own queue-row name
        name = str(item.get("name") or "")
        title = name.split(" - ", 1)[1] if " - " in name else name
        if not artists and " - " in name:
            artists = [name.split(" - ", 1)[0]]
    return title[:120], str(artists[0] if artists else "")[:120]


def _announced(target: str, db_path=None) -> set[str]:
    """Songs a DJ break Music Assistant already holds has promised ("up next,
    X"). READ ONLY. Unprovable promises make nothing removable that is not
    already protected again at delete time (_retire_items re-reads them)."""
    try:
        import dj_engine
        got = dj_engine.committed_handoffs(target, db_path=db_path)
    except Exception:  # noqa: BLE001 - a status read, protected again later
        return set()
    return set(got.get("ids") or ())


def handover_view(target: str, player_name: str, state: dict[str, Any],
                  evidence: dict[str, Any], *, db_path=None,
                  remember: bool = True) -> dict[str, Any]:
    """What the person is about to take over, and the token that confirms
    exactly this. Remembered (bounded, short-lived, in memory only: a
    restart simply asks again). READ ONLY against Music Assistant."""
    ahead = classify_ahead(evidence, None, db_path=db_path)
    music = _music_ahead(evidence)
    cur, buf = state.get("current_index"), state.get("index_in_buffer")
    ints = all(isinstance(v, int) and not isinstance(v, bool)
               for v in (cur, buf))
    # Items up to index_in_buffer are already with the player: they play,
    # whatever is chosen. Unknown positions make every song "buffered".
    held = (max(0, buf - cur) if ints else len(evidence.get("future") or ()))
    buffered = {it.get("queue_item_id")
                for it in (evidence.get("future") or ())[:held]}
    announced = _announced(target, db_path)
    flag = state.get("autoplay_enabled")
    autoplay = flag if isinstance(flag, bool) else None
    # Music Assistant does not record who queued a song (Autoplay and a
    # person both add through the same load), so every song is treated as
    # possibly someone's: listed by name, kept unless the person selects it.
    songs = []
    for it in music[:HANDOVER_LIST_CAP]:
        qiid = it["queue_item_id"]
        title, artist = _item_title(it)
        songs.append({"queue_item_id": qiid, "title": title, "artist": artist,
                      "duration_s": int(_seconds(it)),
                      "status": ("loaded" if qiid in buffered
                                 else "announced" if qiid in announced
                                 else "removable")})
    removable = frozenset(x["queue_item_id"] for x in songs
                          if x["status"] == "removable")
    view = {"target": {"queue_id": target, "name": player_name},
            "autoplay": autoplay,
            "restore_autoplay": autoplay is True,
            "music_ahead": len(music),
            "music_ahead_s": int(sum(_seconds(it) for it in music)),
            "replaceable": len(removable),
            "songs": songs,
            "songs_unlisted": max(0, len(music) - len(songs)),
            "program_ahead": (ahead["break"] + ahead["news"]
                              + ahead["announcement"]),
            "modes": list(HANDOVER_MODES),
            "expires_in_s": int(HANDOVER_TTL_S)}
    if remember:
        import secrets
        token = secrets.token_hex(12)
        with _handover_lock:
            now = _handover_now()
            for key in [k for k, v in _handovers.items()
                        if now - v["at"] > HANDOVER_TTL_S]:
                _handovers.pop(key, None)
            while len(_handovers) >= HANDOVER_CAP:
                _handovers.pop(min(_handovers, key=lambda k: _handovers[k]["at"]))
            _handovers[token] = {
                "target": target, "at": now, "autoplay": autoplay,
                "ids": frozenset(it["queue_item_id"] for it in music),
                "removable": removable}
        view["token"] = token
    return view


def _confirm_handover(raw: Any, target: str, player_name: str,
                      state: dict[str, Any], evidence: dict[str, Any],
                      db_path=None) -> dict[str, Any]:
    """The person's confirmed handover, proven to still describe this queue:
    {token, mode, autoplay, uris (the shown songs still ahead)}. Refuses --
    with a fresh picture -- when it was never asked, has expired, or the
    queue now holds a song the person was not shown or Autoplay flipped."""
    def ask(reason):
        raise _Refusal(reason, handover=handover_view(
            target, player_name, state, evidence, db_path=db_path))
    if raw is None:
        ask(R_HANDOVER_REQUIRED)
    if not isinstance(raw, dict) or raw.get("upcoming") not in HANDOVER_MODES \
            or not isinstance(raw.get("token"), str):
        raise _Refusal(R_INTENT_INVALID,
                       "A handover names its token and whether the songs "
                       "already queued are kept or replaced.")
    remove = raw.get("remove")
    if raw["upcoming"] == "replace" and (
            not isinstance(remove, list) or not remove
            or not all(isinstance(q, str) and q for q in remove)
            or len(remove) > HANDOVER_LIST_CAP):
        # Nothing queued is ever removed without the person naming it.
        raise _Refusal(R_INTENT_INVALID,
                       "Replacing queued songs names each song to remove.")
    with _handover_lock:
        rec = _handovers.get(raw["token"])
    if rec is None or rec["target"] != target \
            or _handover_now() - rec["at"] > HANDOVER_TTL_S:
        ask(R_HANDOVER_CHANGED)
    return _still_shown(rec, raw, target, player_name, state, evidence,
                        db_path)


def _still_shown(rec: dict[str, Any], raw: dict[str, Any], target: str,
                 player_name: str, state: dict[str, Any],
                 evidence: dict[str, Any], db_path=None) -> dict[str, Any]:
    flag = state.get("autoplay_enabled")
    autoplay = flag if isinstance(flag, bool) else None
    music = _music_ahead(evidence)
    # Songs only LEAVE the future as the queue plays; one that arrived since
    # the question is something the person never saw.
    selected = set(raw.get("remove") or ()) if raw["upcoming"] == "replace" \
        else set()
    if not {it["queue_item_id"] for it in music} <= rec["ids"] \
            or autoplay != rec["autoplay"] \
            or not selected <= rec.get("removable", frozenset()):
        # A song the person never saw, Autoplay flipped, or a selected song
        # that was never offered as removable (loaded, announced, unlisted).
        raise _Refusal(R_HANDOVER_CHANGED, handover=handover_view(
            target, player_name, state, evidence, db_path=db_path))
    return {"token": raw["token"], "mode": raw["upcoming"],
            "autoplay": rec["autoplay"], "remove": sorted(selected),
            "music_ahead": len(music),
            # exactly the selected songs still ahead -- the only deletions
            "uris": {it["queue_item_id"]: _uri(it) for it in music
                     if it["queue_item_id"] in selected}}


def _forget_handover(token: str) -> None:
    with _handover_lock:
        _handovers.pop(token, None)


def targets(*, settings_fn=None, players_fn=None, db_path=None
            ) -> dict[str, Any]:
    """Every Music Assistant player with its isolation verdict, eligible
    first -- what the Station tab offers as a station speaker. READ ONLY."""
    doc = _settings_doc(settings_fn)
    try:
        players = (players_fn or ma_queue.player_topology)()
    except ma_queue.MAQueueError:
        return {"ok": False, "reason": R_INVENTORY_UNREADABLE,
                "message": _MESSAGES[R_INVENTORY_UNREADABLE], "targets": []}
    try:
        house = house_outputs(doc, db_path)
    except _Refusal as refusal:
        return {**refusal.answer, "targets": []}
    enabled = set(((doc.get("dj") or {}).get("outputs") or {}).get(
        "enabled") or [])
    authorized = authorized_targets(doc)
    known = {p["player_id"] for p in players}
    # Flow Mode is read only for the house speaker(s): one queue read each.
    flows = {q: queue_flow(q) for q in house if q in known}
    rows = []
    for p in players:
        v = check_target(p["player_id"], players, house_outputs=house,
                         authorized=authorized, flow=flows.get(p["player_id"]))
        rows.append({"queue_id": p["player_id"], "name": p.get("name"),
                     "provider": p.get("provider"), "eligible": v["ok"],
                     "reason": v["reason"], "message": v["message"],
                     "dj_output": p["player_id"] in enabled,
                     "house_output": p["player_id"] in house,
                     "authorized": bool(v.get("authorized")),
                     # a start here must be confirmed as a handover
                     "handover": bool(v["ok"] and v.get("handover"))})
    rows.sort(key=lambda r: (not r["eligible"], not r["dj_output"],
                             str(r["name"] or "").casefold()))
    return {"ok": True, "targets": rows}


# --- the live exclusions around build_candidates ------------------------------------

def _history(queue_id: str, state: dict[str, Any]
             ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """(newest-first MUSIC items BEHIND the head, the full CURRENT item).

    Music Assistant's own record of what this queue played (or passed), and
    the playing item with its metadata -- the queue snapshot names the
    current song only by URI, which cannot say what SONG it is when the
    recording is not the station's own. Best effort: an unreadable window
    is no history, never a guess."""
    current = (state.get("current_item") or {}).get("queue_item_id")
    if not current:
        return [], None
    try:
        win = ma_queue.read_window_items(queue_id, current,
                                         current_index=state.get(
                                             "current_index"),
                                         back=HISTORY_BACK, ahead=0)
    except ma_queue.MAQueueError:
        return [], None
    items = win.get("items") or []
    head = win.get("current_position", 0) - win.get("base_position", 0)
    behind = items[:max(0, head)]
    now = items[head] if 0 <= head < len(items) else None
    return ([it for it in reversed(behind)
             if station_runway.classify_item(it)[0] == station_runway.MUSIC],
            now if (now or {}).get("queue_item_id") == current else None)


def exclusions(res: station_resolver.Resolution, evidence: dict[str, Any],
               history: list[dict[str, Any]], generated: list[dict[str, Any]],
               current: dict[str, Any] | None = None) -> dict[str, Any]:
    """The live state a batch is chosen around. PURE.

      hard       every recording of every song that is playing or queued
                 (identity = library URI; a different recording of a queued
                 song is excluded too, by the resolver's own song key)
      recent     this queue's history, newest first, in the selector's
                 recent_played shape -- other recordings of a recently heard
                 song count as heard
      generated  the station's own rotation ring (what it queued, played or
                 not -- a song the listener deleted must not come straight
                 back)
      loop_guard the last ANTI_LOOP_RECENT songs heard: never re-queued"""
    by_key: dict[tuple[str, str], list[str]] = {}
    by_uri: dict[str, dict[str, Any]] = {}
    for t in res.pool:
        by_key.setdefault(song_key(t["artist"], t["title"]), []).append(t["uri"])
        by_uri[t["uri"]] = t

    def expand(uri: str, key: tuple[str, str] | None) -> set[str]:
        out = {uri} if uri else set()
        if key is None and uri in by_uri:
            t = by_uri[uri]
            key = song_key(t["artist"], t["title"])
        if key is not None:
            out.update(by_key.get(key, ()))
        return out

    hard: set[str] = set()
    hard |= expand(evidence.get("current_uri") or "",
                   _item_song(current) if current else None)
    for item in evidence.get("future") or ():
        if station_runway.classify_item(item)[0] != station_runway.MUSIC:
            continue
        hard |= expand(_uri(item), _item_song(item))

    # ONE row per song heard, so the selector's "last N tracks" window
    # counts songs. The row names the pool's own recording of that song
    # (the resolver keeps exactly one per song), so a live cut heard
    # recently makes the studio version count as heard.
    recent: list[dict[str, Any]] = []
    loop_guard: set[str] = set()
    for n, item in enumerate(history):
        media = item.get("media_item") if isinstance(
            item.get("media_item"), dict) else {}
        album = media.get("album") if isinstance(media.get("album"),
                                                 dict) else {}
        artists = [a.get("name") for a in (media.get("artists") or [])
                   if isinstance(a, dict) and a.get("name")]
        uri = _uri(item)
        same = sorted(expand(uri, _item_song(item)) - {uri})
        ident = uri if uri in by_uri or not same else same[0]
        recent.append({"path": ident, "artist": artists[0] if artists else "",
                       "album": album.get("name") or "", "played_at": None})
        if n < ANTI_LOOP_RECENT:
            loop_guard.add(ident)
            loop_guard.update(same)
    ring = [{"path": str(e.get("path") or e.get("uri") or ""),
             "at": e.get("at")}
            for e in (generated or ()) if isinstance(e, dict)
            and (e.get("path") or e.get("uri"))]
    return {"hard": hard, "recent": recent, "generated": ring,
            "loop_guard": loop_guard}


def choose(res: station_resolver.Resolution, *, count: int,
           rng: random.Random, now: datetime, excl: dict[str, Any],
           settings_fn=None) -> dict[str, Any]:
    """ONE live batch: build_candidates exactly as the preview calls it, plus
    the live exclusions, minus anything the continuity guard refuses."""
    feel, same_artist = station_resolver.settings_view(settings_fn)
    batch = station_resolver.build_candidates(
        res, count=count, rng=rng, now=now,
        hard_excluded_ids=excl["hard"], recent_played=excl["recent"],
        recently_generated=excl["generated"], feel=feel,
        same_artist=same_artist)
    kept = [t for t in batch["tracks"] if t["uri"] not in excl["loop_guard"]
            and t["uri"] not in excl["hard"]]
    batch["loop_guarded"] = len(batch["tracks"]) - len(kept)
    batch["tracks"] = kept[:count]
    return batch


# --- the ownership taxonomy ---------------------------------------------------------

def _is_announcement(token: str, db_path=None) -> bool:
    import playback
    try:
        c = playback._connect(db_path)
        try:
            row = c.execute("SELECT telemetry FROM dj_breaks WHERE token = ?",
                            (token,)).fetchone()
        finally:
            c.close()
    except sqlite3.Error:
        return False
    if not row:
        return False
    try:
        tel = json.loads(row[0] or "{}")
    except ValueError:
        return False
    return isinstance(tel, dict) and tel.get("purpose") == "announcement"


def classify_ahead(evidence: dict[str, Any], station: dict[str, Any] | None,
                   *, db_path=None) -> dict[str, int]:
    """Who owns each item AHEAD of the head, by evidence -- never by title:

      station       queue_item_id is in this station's ownership witness
      manual        music this station did not queue and that was not there
                    when it started (a listener added it)
      preexisting   music already queued when the station started
      break / news / announcement   Parlorwave's finite programming
      unknown       anything that cannot be classified (never music)

    Only `station` is ever eligible for station reconciliation."""
    tail = {w.get("queue_item_id") for w in ((station or {}).get("tail")
                                              or ()) if w.get("queue_item_id")}
    pre = set(((station or {}).get("program") or {}).get("preexisting_ids")
              or ())
    out = {"station": 0, "manual": 0, "preexisting": 0, "break": 0,
           "news": 0, "announcement": 0, "unknown": 0}
    for item in evidence.get("future") or ():
        kind, detail = station_runway.classify_item(item)
        if kind == station_runway.PROGRAMMING:
            if detail == "break" and _is_announcement(
                    media_ref.programming_ref(_uri(item))[1], db_path):
                out["announcement"] += 1
            else:
                out[detail if detail in out else "unknown"] += 1
        elif kind == station_runway.MUSIC:
            qiid = item.get("queue_item_id")
            if qiid in tail:
                out["station"] += 1
            elif qiid in pre:
                out["preexisting"] += 1
            else:
                out["manual"] += 1
        else:
            out["unknown"] += 1
    return out


# --- start / switch -----------------------------------------------------------------

def _parse(request: Any, dial_fn=None
           ) -> tuple[station_intent.StationIntent, int, str]:
    if not isinstance(request, dict):
        raise _Refusal(R_INTENT_INVALID, "The request must be an object.")
    target = str(request.get("target") or "").strip()
    if not target:
        raise _Refusal(R_TARGET_REQUIRED)
    try:
        variant = max(0, min(9999, int(request.get("variant") or 0)))
    except (TypeError, ValueError):
        raise _Refusal(R_INTENT_INVALID, "variant must be a number") from None
    preset_id = request.get("preset_id")
    origin = request.get("origin")
    try:
        if preset_id:
            if request.get("intent"):
                raise _Refusal(R_INTENT_INVALID,
                               "Give a preset or an intent, not both.")
            # WHO pressed start on a saved station: the Station tab's form or
            # the Dial. Nothing else about the start depends on it.
            origin = origin or "station_tab"
            if origin not in PRESET_ORIGINS:
                raise _Refusal(R_INTENT_INVALID,
                               "A saved station is started from the "
                               "Station tab or the Dial.")
            if dial_fn is None:
                import settings
                dial_fn = settings.dial_config
            dial = dial_fn() or {}
            preset = next((p for p in dial.get("presets") or []
                           if p.get("id") == preset_id), None)
            if preset is None:
                raise _Refusal(R_PRESET_NOT_FOUND,
                               "That Dial station no longer exists.")
            if origin == "dial" and preset.get("enabled") is False:
                raise _Refusal(R_PRESET_OFF_DIAL)
            # The query is the SAVED preset's, read here and nowhere else:
            # retuning a preset retunes what its Dial position starts next.
            intent = station_intent.intent_for_preset(preset, origin=origin)
            # A kind nothing can build yet is refused AS that, before any
            # test rule: Test Station cannot qualify a news preset, so
            # "needs a test" would send the operator nowhere.
            _require_buildable(intent)
            lt = preset.get("last_test") or {}
            tested = station_resolver.test_state(lt, intent.query)
            current = tested == station_resolver.TEST_CURRENT
            if current and lt.get("viable") is False:
                # Station Setup's verdict on THIS term, by THIS builder,
                # stands for every caller, hardware included. A stale
                # verdict (another term, or an older builder revision) is
                # neither thin nor approved, and the live rule below is
                # re-run on every start regardless.
                raise _Refusal(R_PRESET_TOO_THIN)
            if origin == "dial" and not current:
                # The Dial starts only what Station Setup has approved as
                # the builder programs it NOW: never tested, retuned since
                # its test, or tested by an older builder -> test again.
                raise _Refusal(R_PRESET_NEEDS_TEST, test_state=tested)
            if "variant" not in request and isinstance(lt.get("variant"), int) \
                    and current:
                # The draw that was tested and approved on this preset.
                variant = lt["variant"]
        else:
            if origin is not None:
                raise _Refusal(R_INTENT_INVALID,
                               "An intent names its own origin.")
            intent = station_intent.intent_from_dict(request.get("intent"))
            _require_buildable(intent)
    except station_intent.IntentError as exc:
        raise _Refusal(R_INTENT_INVALID, str(exc)) from None
    return intent, variant, target


def _require_buildable(intent: station_intent.StationIntent) -> None:
    if not station_intent.KINDS[intent.kind]["buildable"]:
        raise _Refusal(R_KIND_NOT_AVAILABLE,
                       f"'{intent.kind}' stations cannot run yet.")


def start_station(request: Any, *, now: datetime | None = None, db_path=None,
                  settings_fn=None, dial_fn=None, players_fn=None,
                  knowledge_fn=None) -> dict[str, Any]:
    """start_station(intent, target) -- THE one way a station goes on air.

    request = {"target": <MA player id>, "intent": {...} | "preset_id": ...,
               "variant": n (optional),
               "origin": "station_tab" | "dial" (a preset only; an intent
                         carries its own origin, request_line for the
                         Request Line)}

    Every caller -- the Station tab's Start, the Request Line and the Dial
    -- arrives HERE, and the origin is recorded and never consulted again:
    the same intent and variant build the same opening whoever asked. A
    saved preset is read from the Dial settings by id, so its CURRENT term
    is what starts; a preset Station Setup tested as too thin for that term
    is refused for every caller before anything is built.

    On a target where this station programmer is already running, the same
    call is a SWITCH: the new intent becomes authoritative in one
    transaction and the old station's own queued music is retired.

    Refusals change nothing anywhere. Every read and all of the selection
    happens BEFORE the station mutation lock; inside it the world is
    re-read and re-checked, and the one append is proven by reading the
    queue back before the station is activated."""
    now_dt = now or service_clock.legacy_now()
    try:
        intent, variant, target = _parse(request, dial_fn)
        doc = _settings_doc(settings_fn)
        # 1. The target, proven isolated.
        verdict = _prove_target(target, doc, players_fn, db_path)
        player_name = str(verdict["player"].get("name") or target)
        # 2. What else is on air.
        old = _station_state(target, now_dt, db_path)
        # 3. The target's queue: playing or paused, music at the head.
        state, evidence = _read_target(target)
        # 3b. A house speaker is taken over only by a confirmed handover --
        # asked (and answered, writing nothing) before anything is built. A
        # switch on the station's own speaker is not a handover: ParlorWave
        # already holds its programming.
        handover = None
        if verdict.get("handover") and old is None:
            handover = _confirm_handover(
                request.get("handover"), target, player_name, state,
                evidence, db_path)
        # 4. The station, from the one builder.
        res = _resolve(intent, INITIAL_RUNWAY, knowledge_fn)
        history, playing = _history(target, state)
        excl = exclusions(res, evidence, history, [], current=playing)
        batch = choose(res, count=INITIAL_RUNWAY,
                       rng=random.Random(station_resolver.seed_for(intent,
                                                                   variant)),
                       now=now_dt, excl=excl,
                       settings_fn=lambda: doc)
        tracks = batch["tracks"]
        viability = station_resolver.live_viability(res, len(tracks))
        if not viability["viable"]:
            raise _Refusal(R_TOO_THIN,
                           " ".join(r["message"] for r in viability["reasons"]),
                           viability=viability,
                           summary={"tracks": len(tracks),
                                    "pool_tracks": res.stats.get("pool_tracks"),
                                    "pool_artists": res.stats.get(
                                        "pool_artists")})
        if old is None and handover is None:
            # A confirmed handover has already shown the person every song
            # ahead and let them choose; anything else keeps the old rule.
            _refuse_music_ahead(evidence, player_name, db_path)
        return _install(intent=intent, variant=variant, target=target,
                        player_name=player_name, res=res, batch=batch,
                        viability=viability, old=old, now_dt=now_dt,
                        db_path=db_path, doc=doc, players_fn=players_fn,
                        handover=handover)
    except _Refusal as refusal:
        answer = refusal.answer
        logger.info("station_live_start_refused reason=%s target=%s",
                    answer["reason"], str((request or {}).get("target")
                                          if isinstance(request, dict)
                                          else "")[:80])
        return answer


def _prove_target(target: str, doc: dict[str, Any], players_fn, db_path
                  ) -> dict[str, Any]:
    try:
        players = (players_fn or ma_queue.player_topology)()
    except ma_queue.MAQueueError:
        raise _Refusal(R_INVENTORY_UNREADABLE) from None
    house = house_outputs(doc, db_path)
    verdict = check_target(target, players, house_outputs=house,
                           authorized=authorized_targets(doc),
                           flow=queue_flow(target) if target in house else None)
    if not verdict["ok"]:
        raise _Refusal(verdict["reason"], verdict["message"])
    if verdict.get("authorized"):
        logger.warning("station_live target %s is a house speaker, allowed "
                       "only by station.live.authorized_targets", target)
    return verdict


def _refuse_music_ahead(evidence: dict[str, Any], player_name: str,
                        db_path) -> None:
    ahead = classify_ahead(evidence, None, db_path=db_path)
    music_ahead = ahead["manual"] + ahead["preexisting"] + ahead["station"]
    if music_ahead > START_MAX_MUSIC_AHEAD:
        raise _Refusal(R_MUSIC_AHEAD,
                       f"{music_ahead} songs are already queued on "
                       f"{player_name}. " + _MESSAGES[R_MUSIC_AHEAD],
                       music_ahead=music_ahead)


def _station_state(target: str, now_dt: datetime, db_path
                   ) -> dict[str, Any] | None:
    """The station this start would replace, or None. Refuses every state
    in which a start could change something other than `target`."""
    preparing = station_store.get_preparing_station(db_path=db_path)
    if preparing is not None:
        # A stranded start is settled by reading the queue -- never by
        # writing again -- before anything new may begin; serialized, so it
        # can never judge a write that is still in flight.
        station_recovery.recover_pass_serialized(now=now_dt, db_path=db_path)
        if station_store.get_preparing_station(db_path=db_path) is not None:
            raise _Refusal(R_START_IN_FLIGHT)
    active = station_store.get_active_station(db_path=db_path)
    if active is None:
        return None
    if not station_store.is_programmer(active):
        raise _Refusal(R_LEGACY_STATION)
    if (active.get("queue_id") or "") != target:
        raise _Refusal(R_ELSEWHERE)
    return active


def _read_target(target: str, history: int = 0
                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        state = ma_queue.queue_state(target)
    except ma_queue.MAQueueError:
        raise _Refusal(R_QUEUE_UNREADABLE) from None
    current = (state or {}).get("current_item") or {}
    if (state or {}).get("state") not in ("playing", "paused") \
            or not current.get("queue_item_id"):
        raise _Refusal(R_NOT_PLAYING)
    if media_ref.programming_ref(current.get("uri"))[0]:
        raise _Refusal(R_PROGRAMMING_ON_AIR)
    evidence = station_recovery.read_queue_evidence(target, history=history)
    if evidence is None:
        raise _Refusal(R_QUEUE_UNREADABLE)
    return state, evidence


def _resolve(intent: station_intent.StationIntent, count: int,
             knowledge_fn=None) -> station_resolver.Resolution:
    try:
        snap, _cached = ma_catalog.snapshot()
    except ma_catalog.CatalogError:
        raise _Refusal(R_CATALOG) from None
    res = station_resolver.resolve_station(intent, snap, count=count,
                                           knowledge_fn=knowledge_fn)
    if res.status != station_resolver.OK:
        raise _Refusal(res.status, " ".join(res.notes) or res.status,
                       candidates=list(res.candidates))
    return res


def _install(*, intent, variant, target, player_name, res, batch, viability,
             old, now_dt, db_path, doc, players_fn=None,
             handover=None) -> dict[str, Any]:
    """Under the station mutation lock: re-check, write the intent, append
    once, prove it, activate -- then retire the old station's runway, or
    (a confirmed `replace` handover) the songs the person chose to replace."""
    tracks = batch["tracks"]
    uris = [t["uri"] for t in tracks]
    stamp = _iso(now_dt)
    try:
        with station_lock.station_mutation():
            active = station_store.get_active_station(db_path=db_path)
            if (active or {}).get("station_id") != (old or {}).get(
                    "station_id"):
                raise _Refusal(R_CHANGED)
            if station_store.get_preparing_station(db_path=db_path):
                raise _Refusal(R_START_IN_FLIGHT)
            # The world may have moved while the station was being built:
            # prove the target again, under the lock, before any write.
            again = _prove_target(target, doc, players_fn, db_path)
            if old is None and bool(again.get("handover")) != (
                    handover is not None):
                # It became (or stopped being) the house speaker meanwhile:
                # what the person confirmed no longer describes this start.
                raise _Refusal(R_HANDOVER_CHANGED)
            if old is not None:
                old = station_store.get_station(old["station_id"],
                                                db_path=db_path) or old
                _settle_pending(old, now_dt, db_path)
                # Re-read: settling may have confirmed a refill, and the
                # retirement below must see every id this station owns.
                old = station_store.get_station(old["station_id"],
                                                db_path=db_path) or old
            proof = station_recovery.PROOF_HISTORY
            state, evidence = _read_target(target, history=proof)
            if handover is not None:
                # The last look before the write: still only songs the
                # person was shown, and Autoplay as it was described.
                with _handover_lock:
                    rec = _handovers.get(handover["token"])
                if rec is None:
                    raise _Refusal(R_HANDOVER_CHANGED)
                handover = _still_shown(
                    rec, {"token": handover["token"],
                          "upcoming": handover["mode"],
                          "remove": handover["remove"]},
                    target, player_name, state, evidence, db_path)
                _forget_handover(handover["token"])
            elif old is None:
                _refuse_music_ahead(evidence, player_name, db_path)
            ordered = station_recovery.ordered_queue(evidence)
            pre_tail = ordered[-1]
            music = station_recovery.future_music_uris(evidence["future"])
            old_tail = {w.get("queue_item_id")
                        for w in (old or {}).get("tail") or ()}
            preexisting = [str(i.get("queue_item_id")) for i in
                           evidence["future"] if i.get("queue_item_id")
                           and i.get("queue_item_id") not in old_tail]
            program = {
                "intent": intent.to_dict(), "variant": variant,
                "target": {"queue_id": target, "name": player_name},
                "strategy": res.strategy, "seed_artist": res.seed_artist,
                "genres": [g.get("name") for g in res.genres],
                "era": (res.era or {}).get("label"),
                "pool_tracks": res.stats.get("pool_tracks"),
                "pool_artists": res.stats.get("pool_artists"),
                "opening_count": len(uris),
                "viable": viability["viable"],
                "preexisting_ids": preexisting[:PREEXISTING_CAP],
                "switched_from": (old or {}).get("station_id") or "",
                "started_by": intent.origin,
            }
            carried = ((old or {}).get("program") or {}).get("handover")
            if handover is None and carried:
                # A switch on a speaker taken over by handover keeps that
                # handover: Stop still owes the queue back to Autoplay.
                program["handover"] = dict(carried,
                                           carried_from=old["station_id"])
            if handover is not None:
                program["handover"] = {
                    "mode": handover["mode"],
                    "autoplay_before": handover["autoplay"],
                    # Stop gives the queue back to Autoplay only when the
                    # person was told it was on and would come back.
                    "restore_autoplay": handover["autoplay"] is True,
                    "music_ahead": handover["music_ahead"],
                    "removal_selected": len(handover["remove"]),
                    "confirmed_at": stamp}
            station = station_store.create_preparing_station(
                request_text=intent.query or intent.source,
                artist_weights=res.weights,
                seed_artist=res.seed_artist or "",
                profile_source=station_store.PROFILE_PROGRAMMER,
                recipe_version=RECIPE_VERSION, queue_id=target,
                intended_uris=uris,
                pre_current_queue_item_id=evidence["current_queue_item_id"],
                pre_future_music_count=len(music),
                pre_future_music_hash=station_store.uri_list_hash(music),
                program=program, now=now_dt, db_path=db_path)
            sid = station["station_id"]
            claim = station_store.create_pending_batch(
                station_id=sid, queue_id=target, intended_uris=uris,
                pre_items_count=(state or {}).get("items") or len(ordered),
                pre_tail_queue_item_id=str(pre_tail.get("queue_item_id") or ""),
                pre_tail_position=len(ordered) - 1,
                pre_qiid_hash=station_store.uri_list_hash(
                    [str(i.get("queue_item_id") or "") for i in ordered]),
                now=now_dt, db_path=db_path)

            # THE ONE APPEND. No retry on any outcome.
            write_error = None
            try:
                ma_queue.append_tracks(target, uris)
            except Exception as exc:  # noqa: BLE001 - single write boundary
                write_error = str(exc)[:200]
                logger.warning("station programmer opening append failed or "
                               "was ambiguous: %s", write_error)
            verdict, why, after = _judge(claim, target)
            if write_error and verdict == station_recovery.NOT_LANDED:
                # A write that errored may still be applied late (a lost
                # response): look once more before calling it not landed.
                _write_error_wait()
                verdict, why, after = _judge(claim, target)
            if verdict == station_recovery.NOT_LANDED:
                station_store.resolve_batch(claim["batch_id"],
                                            station_store.NOT_LANDED,
                                            resolution=why, now=now_dt,
                                            db_path=db_path)
                station_store.fail_prepared_station(
                    sid, reason="runway_not_landed", now=now_dt,
                    db_path=db_path)
                raise _Refusal(R_NOT_LANDED, write_error=write_error)
            if verdict != station_recovery.LANDED:
                logger.warning("station_live_start_unproven station=%s "
                               "target=%s (%s)", sid[:8], target, why)
                return {"ok": False, "reason": R_UNPROVEN,
                        "message": _MESSAGES[R_UNPROVEN], "station_id": sid,
                        "reconciliation": why, "write_error": write_error}
            _activate(sid, claim, after, target, why, now_dt, db_path)
            live = station_store.get_station(sid, db_path=db_path) or {}
            if live.get("status") != station_store.ACTIVE:
                # Its claim was settled some other way meanwhile. Nothing
                # further is written: no retirement, no ownership.
                logger.error("station_live_start_activation_lost station=%s "
                             "status=%s", sid[:8], live.get("status"))
                raise _Refusal(R_ACTIVATION_LOST, station_id=sid)
            retired = (_retire_old_runway(old, target, db_path=db_path)
                       if old is not None else None)
            replaced = None
            if handover is not None and handover["mode"] == "replace":
                # Only now, with the station's songs landed and the station
                # active: exactly the songs the person was shown, by id.
                replaced = _retire_items(handover["uris"], target,
                                         db_path=db_path)
            handed = (None if handover is None else
                      {"mode": handover["mode"],
                       "autoplay_before": handover["autoplay"],
                       "restore_autoplay": handover["autoplay"] is True,
                       "music_ahead": handover["music_ahead"],
                       "removal_selected": len(handover["remove"]),
                       "replaced": replaced})
            ownership = _acquire_ownership(sid, target, now_dt, db_path)
            final = station_recovery.read_queue_evidence(target) or after
            if final is not after:
                final.pop("history", None)
            fresh = station_store.get_station(sid, db_path=db_path) or {}
            check = _check_record(fresh, final, "started", now_dt,
                                  db_path=db_path)
            station_store.update_live_status(
                sid, {"last_check": check, "check_problem": "",
                      "refill_problem": "",
                      "started": {"at": stamp, "opening": len(uris),
                                  "retired_old": retired,
                                  "handover": handed}},
                db_path=db_path)
    except station_lock.StationMutationBusy:
        raise _Refusal(R_BUSY) from None
    logger.info("station_live_started station=%s target=%s kind=%s "
                "strategy=%s opening=%d variant=%d switched_from=%s "
                "retired_old=%s kept_promised=%s ownership=%s handover=%s "
                "replaced=%s",
                sid[:8], target, intent.kind, res.strategy, len(uris),
                variant, ((old or {}).get("station_id") or "-")[:8],
                (retired or {}).get("removed", "-"),
                (retired or {}).get("promised", "-"),
                (ownership or {}).get("state"),
                (handed or {}).get("mode", "-"),
                ((handed or {}).get("replaced") or {}).get("removed", "-"))
    return {"ok": True, "reason": "", "station_id": sid,
            "switched_from": (old or {}).get("station_id") or None,
            "intent": intent.to_dict(), "variant": variant,
            "target": {"queue_id": target, "name": player_name},
            "station": {"strategy": res.strategy,
                        "seed_artist": res.seed_artist,
                        "genres": [g.get("name") for g in res.genres],
                        "era": (res.era or {}).get("label")},
            "opening": [{"artist": t["artist"], "title": t["title"],
                         "album": t.get("album") or "", "year": t.get("year")}
                        for t in tracks],
            "opening_count": len(uris), "viability": viability,
            "retired_old": retired, "ownership": ownership,
            "handover": handed,
            "runway": check, "loop_guarded": batch.get("loop_guarded", 0)}


def _judge(claim: dict[str, Any], target: str
           ) -> tuple[str, str, dict[str, Any] | None]:
    """(verdict, reason, evidence) for the opening append, read with history
    so a song boundary since the write cannot hide the anchor."""
    after = station_recovery.read_queue_evidence(
        target, history=station_recovery.PROOF_HISTORY)
    if after is None:
        return station_recovery.UNPROVEN, "readback_unreadable", None
    verdict, why = station_recovery.classify_append(claim, after)
    return verdict, why, after


def _write_error_wait() -> None:
    time.sleep(WRITE_ERROR_SETTLE_S)


def _settle_pending(old: dict[str, Any], now_dt: datetime, db_path) -> None:
    """A running station's unresolved refill is proven from the queue
    before a switch may replace it; unproven means wait."""
    batch = station_store.get_pending_batch(old["station_id"], db_path=db_path)
    if batch is None:
        return
    evidence = station_recovery.read_queue_evidence(
        batch.get("queue_id") or "", history=station_recovery.PROOF_HISTORY)
    if evidence is not None:
        verdict, _why = station_recovery.resolve_pending_batch(
            batch, evidence,
            confirm_fn=lambda b, e: station_refill.confirm_landed_batch(
                old, b, e, now=now_dt, db_path=db_path),
            now=now_dt, db_path=db_path)
        if verdict != station_recovery.UNPROVEN:
            return
    raise _Refusal(R_REFILL_UNPROVEN)


def _activate(sid: str, claim: dict[str, Any], evidence: dict[str, Any],
              target: str, why: str, now_dt: datetime, db_path) -> bool:
    appended = station_recovery.appended_items(claim, evidence)
    stamp = _iso(now_dt)
    ahead = {it.get("queue_item_id") for it in evidence.get("future") or ()}
    # The witness is RUNWAY: a song of the opening that a boundary has
    # already carried to the head (or past it) is in the ring, not here.
    witness = [{"queue_item_id": it.get("queue_item_id"), "uri": _uri(it),
                "last_region": "future"} for it in appended
               if it.get("queue_item_id") in ahead]
    ring = [{"uri": _uri(it), "path": _uri(it), "at": stamp}
            for it in appended]
    return station_store.activate_prepared_with_batch(
        sid, claim["batch_id"], queue_id=target, tail=witness,
        tail_is_partial=len(witness) > station_store.TAIL_CAP,
        generated=ring, resolution=why, now=now_dt, db_path=db_path)


def resolve_preparation(station: dict[str, Any], evidence: dict[str, Any],
                        *, now=None, db_path=None) -> str:
    """Startup/lifecycle recovery of a Station Programmer start stranded
    between its append and its activation. READ-ONLY against Music
    Assistant: the append is judged by its own claim, exactly like a refill,
    and a LANDED start is activated in the same one transaction the live
    request would have used. The old station's runway is NOT retired here
    -- that needs a write, and recovery never writes; it simply plays out."""
    now_dt = now or service_clock.legacy_now()
    sid = station["station_id"]
    evidence = station_recovery.read_queue_evidence(
        station.get("queue_id") or "",
        history=station_recovery.PROOF_HISTORY) or evidence
    claim = station_store.get_pending_batch(sid, db_path=db_path)
    if claim is None:
        # No claim was ever written, so no append was ever sent.
        station_store.fail_prepared_station(sid, reason="prepare_unclaimed",
                                            now=now_dt, db_path=db_path)
        return station_recovery.NOT_LANDED
    verdict, why = station_recovery.classify_append(claim, evidence)
    if verdict == station_recovery.LANDED:
        _activate(sid, claim, evidence, station.get("queue_id") or "", why,
                  now_dt, db_path)
        logger.info("station_live_start_recovered station=%s (%s)",
                    sid[:8], why)
    elif verdict == station_recovery.NOT_LANDED:
        station_store.resolve_batch(claim["batch_id"],
                                    station_store.NOT_LANDED, resolution=why,
                                    now=now_dt, db_path=db_path)
        station_store.fail_prepared_station(sid, reason="runway_not_landed",
                                            now=now_dt, db_path=db_path)
    return verdict


def _retire_old_runway(old: dict[str, Any], target: str, db_path=None
                       ) -> dict[str, Any]:
    """At a switch, remove the OLD station's own music that has not yet been
    buffered, by the exact ids its ownership witness recorded -- so the new
    station takes over after the current song instead of an hour later.

    Nothing else is ever a candidate: not the current item, not a manual
    addition (another queue_item_id, even for the same song), not a DJ
    break, bulletin or announcement, not anything unclassifiable, and not
    an item Music Assistant has already buffered (those simply play). Every
    delete is by id, never by position, and the result is proven by reading
    the queue back. Positions are compared with index_in_buffer only while
    this read proves the head's list position equals current_index; any
    doubt deletes nothing.

    A PROMISED SONG IS NEVER RETIRED. A DJ break Music Assistant already
    holds may have said "up next, X"; X is kept by its exact queue_item_id
    (dj_engine.committed_handoffs) and the new station follows it. The
    promises are read under dj_engine.HANDOFF_LOCK immediately before every
    delete, so a break queued while this runs is honoured too; `promised`
    counts the deletes a promise prevented. A committed break whose promise
    cannot be proven, or rows that cannot be read, stop the retirement
    before its next delete: an extra old song is always better than a
    broken promise. Keeping X gives the old station nothing back -- it is
    deactivated before this runs."""
    return _retire_items({w.get("queue_item_id"): w.get("uri")
                          for w in old.get("tail") or ()
                          if w.get("queue_item_id")}, target, db_path=db_path)


def _retire_items(witness: dict[str, str], target: str, db_path=None
                  ) -> dict[str, Any]:
    """Remove the not-yet-buffered MUSIC items named in `witness`
    ({queue_item_id: uri}), each by exact id and only while its URI still
    matches -- the one deletion discipline both a switch (the old station's
    runway) and a confirmed `replace` handover (the songs the person was
    shown) use. Every rule of _retire_old_runway applies."""
    import dj_engine
    out = {"candidates": 0, "removed": 0, "buffered": 0, "promised": 0,
           "failed": 0, "reason": ""}
    if not witness:
        return out

    def promised_now():
        """The committed ids, or None when the promises cannot be proven."""
        try:
            got = dj_engine.committed_handoffs(target, db_path=db_path)
        except Exception:  # noqa: BLE001 - any doubt deletes nothing
            return None
        return None if got["unproven"] else got["ids"]

    def head_of(st):
        return (((st or {}).get("current_item") or {}).get("queue_item_id"),
                (st or {}).get("current_index"),
                (st or {}).get("index_in_buffer"))

    try:
        state = ma_queue.queue_state(target)
        head = head_of(state)[0]
        read = ma_queue.read_future_items(
            target, head, current_index=state.get("current_index"),
            total_items=state.get("items"))
        settled = head_of(ma_queue.queue_state(target))
    except ma_queue.MAQueueError as exc:
        out["reason"] = f"unreadable:{str(exc)[:80]}"
        return out
    buf, cur = state.get("index_in_buffer"), state.get("current_index")
    ints = all(isinstance(v, int) and not isinstance(v, bool)
               for v in (buf, cur))
    # Positions are trusted only while ONE moment is proven: the head, its
    # index and the buffer were the same before and after the list read.
    if not read.get("complete") or not ints \
            or read.get("current_position") != cur \
            or settled != head_of(state):
        out["reason"] = "positions_unproven"
        return out
    doomed = []
    for offset, item in enumerate(read["future"]):
        qiid = item.get("queue_item_id")
        if qiid not in witness or _uri(item) != witness[qiid]:
            continue
        if station_runway.classify_item(item)[0] != station_runway.MUSIC:
            continue
        out["candidates"] += 1
        if read["current_position"] + 1 + offset <= buf:
            out["buffered"] += 1
            continue
        doomed.append(str(qiid))
    for qiid in doomed[:station_store.TAIL_CAP]:
        if not dj_engine.HANDOFF_LOCK.acquire(
                timeout=dj_engine.HANDOFF_WAIT_S):
            out["reason"] = "handoff_busy"
            break
        try:
            # Before EVERY delete, under the lock the DJ claims under: no
            # break Music Assistant holds has promised this song ...
            promised = promised_now()
            if promised is None:
                out["reason"] = "promise_unproven"
                break
            if qiid in promised:
                out["promised"] += 1
                continue
            # ... and the head and the buffer have not moved, so the
            # positions above still mean what they meant. A boundary
            # mid-way stops the retirement; what is left simply plays.
            if head_of(ma_queue.queue_state(target)) != head_of(state):
                out["reason"] = "head_moved"
                break
            ma_queue.delete_queue_item(target, qiid)
        except ma_queue.MAQueueError:
            out["failed"] += 1
            break
        finally:
            dj_engine.HANDOFF_LOCK.release()
    try:
        again = ma_queue.queue_state(target)
        head = ((again or {}).get("current_item") or {}).get("queue_item_id")
        left = ma_queue.read_future_items(
            target, head, current_index=again.get("current_index"),
            total_items=again.get("items"))
        present = {i.get("queue_item_id") for i in left["future"]}
        out["removed"] = sum(1 for q in doomed if q not in present)
    except ma_queue.MAQueueError:
        out["reason"] = "readback_unreadable"
    return out


def _acquire_ownership(sid: str, target: str, now_dt: datetime, db_path
                       ) -> dict[str, Any] | None:
    """MA Autoplay OFF for this queue, and only this queue: Parlorwave owns
    refill. Fenced -- a station on air is never undone by an unverified
    flag; the next refill re-checks it (queue_ownership's drift policy)."""
    try:
        import queue_ownership
        observed, err = None, ""
        try:
            flag = (ma_queue.queue_state(target) or {}).get("autoplay_enabled")
            observed = flag if isinstance(flag, bool) else None
        except ma_queue.MAQueueError as exc:
            err = str(exc)[:200]
        return queue_ownership.acquire(queue_id=target, station_id=sid,
                                       observed=observed, observed_error=err,
                                       now=now_dt, db_path=db_path,
                                       source="start_station")
    except Exception:  # noqa: BLE001 - never a station failure
        logger.warning("station programmer: queue ownership acquisition "
                       "failed (retried at the next refill)", exc_info=True)
        return None


# --- stop -------------------------------------------------------------------------

def stop_station(*, now: datetime | None = None, db_path=None,
                 reason: str = "stopped") -> dict[str, Any]:
    """Stop PROGRAMMING. Mode 2 goes inactive and no refill will ever be
    added for this station again. Deliberately nothing else:

      * no transport or queue call of any kind -- the player keeps playing,
        the queue is not cleared, the output is not changed;
      * MA Autoplay is left OFF (queue_ownership's restore policy is none),
        what is already queued plays out naturally -- EXCEPT after a
        confirmed house handover that found Autoplay ON: then that one
        queue's Autoplay is turned back on, inside the lock and after the
        station is inactive, so the speaker returns to ordinary Music
        Assistant listening with one refill authority at a time.

    Idempotent: stopping when nothing is on air answers ok."""
    now_dt = now or service_clock.legacy_now()
    try:
        with station_lock.station_mutation():
            stranded = station_store.get_preparing_station(db_path=db_path)
            if station_store.is_programmer(stranded):
                # A start that never proved itself: it will never be
                # activated now. Its songs, if they landed, simply play.
                _abandon_pending(stranded, "station_stopped", now_dt, db_path)
                station_store.deactivate_station(
                    stranded["station_id"], "stopped_before_confirmed",
                    now=now_dt, db_path=db_path)
            active = station_store.get_active_station(db_path=db_path)
            if not station_store.is_programmer(active):
                return {"ok": True, "was_active": False,
                        "legacy_station_active": active is not None,
                        "message": "No station is on air."}
            _abandon_pending(active, "station_stopped", now_dt, db_path)
            station_store.update_live_status(
                active["station_id"], {"stopped": {"at": _iso(now_dt),
                                                   "reason": reason}},
                db_path=db_path)
            # The restore is decided -- and written down as OWED -- before the
            # station stops. Only THIS moment may restore automatically: the
            # person has just pressed Stop, ParlorWave still holds the queue
            # uncontested, and the flag is read now. If it cannot finish here
            # (Music Assistant silent, the write unproven, the process ending
            # between the two steps) the debt becomes a PENDING DECISION the
            # person answers -- never a later automatic write, because an OFF
            # read later cannot tell ParlorWave's OFF from a person's choice.
            owed = _restore_owed(active, now_dt, db_path)
            station_store.deactivate_station(active["station_id"], reason,
                                             now=now_dt, db_path=db_path)
            restored = (_settle_restore(active["station_id"], now_dt, db_path,
                                        at_stop=True)
                        if owed and owed.get("state") == "owed" else owed)
    except station_lock.StationMutationBusy:
        return {"ok": False, "reason": R_BUSY, "message": _MESSAGES[R_BUSY]}
    target = (active.get("program") or {}).get("target") or {}
    logger.info("station_live_stopped station=%s target=%s reason=%s "
                "refills=%s autoplay_restore=%s", active["station_id"][:8],
                active.get("queue_id"), reason, active.get("refill_count"),
                (restored or {}).get("state", "-"))
    message = ("The station stopped programming. The speaker keeps playing "
               "what is already queued.")
    if restored is not None:
        where = target.get("name") or "that speaker"
        state = restored.get("state")
        if restored.get("restored"):
            message += f" Music Assistant's Autoplay is on again for {where}."
        elif state == "pending":
            message += (f" Autoplay could not be switched back on for {where}"
                        f"{' (Music Assistant did not answer)' if restored.get('reason') == 'ma_unavailable' else ''}"
                        ". The Station tab asks whether to switch it back on.")
        elif state == "not_ours":
            message += (f" Autoplay for {where} was left as it was set while "
                        "the station played.")
        else:
            message += (f" Autoplay could not be switched back on for {where}: "
                        "turn it on in Music Assistant if you want it.")
    return {"ok": True, "was_active": True,
            "station_id": active["station_id"], "target": target,
            "stopped_at": _iso(now_dt), "autoplay_restore": restored,
            "message": message}


#: Debt states. `owed` exists only between the decision and the attempt at
#: Stop; `pending` is the person's to answer; the rest are final.
RESTORE_OPEN = ("owed", "pending")
_PENDING_REASON = {"unreadable": "ma_unavailable", "failed": "write_failed",
                   "unverified": "unverified"}


def _queue_sources(queue_id: str) -> list[str] | None:
    """What the person last started on this queue (MA's `sources`), as URIs
    -- context for a pending decision ("music changed since"), never
    permission to write. None when it cannot be read."""
    try:
        raw = ma_queue.raw_queue(queue_id) or {}
    except ma_queue.MAQueueError:
        return None
    src = raw.get("sources")
    if not isinstance(src, list):
        return None
    return [str(x.get("uri") or "") for x in src if isinstance(x, dict)][:10]


def _read_autoplay(queue_id: str) -> bool | None:
    try:
        flag = ma_queue.queue_autoplay(queue_id)
    except ma_queue.MAQueueError:
        return None
    return flag if isinstance(flag, bool) else None


def _restore_owed(station: dict[str, Any], now_dt: datetime, db_path
                  ) -> dict[str, Any] | None:
    """Inside the lock, BEFORE deactivation: does this Stop owe the queue
    back to Autoplay? Only a confirmed handover that found Autoplay ON, and
    only while ParlorWave still holds that queue's ownership uncontested --
    a contest means the person switched Autoplay on themselves, a newer
    decision that is theirs, not ParlorWave's to repeat or undo. Recorded
    durably either way, with the original handover beside it so nothing of
    it is lost however the debt ends; None when nothing was ever promised."""
    handed = (station.get("program") or {}).get("handover") or {}
    queue_id = station.get("queue_id") or ""
    if handed.get("restore_autoplay") is not True or not queue_id:
        return None
    import queue_ownership
    rec = queue_ownership.get_record(queue_id, db_path=db_path) or {}
    mine = rec.get("station_id") == station["station_id"] and rec.get(
        "state") in (queue_ownership.OWNED, queue_ownership.UNVERIFIED)
    target = (station.get("program") or {}).get("target") or {}
    base = {"queue_id": queue_id, "speaker": target.get("name") or queue_id,
            "handover": dict(handed), "since": _iso(now_dt)}
    owed = ({**base, "state": "owed", "sources": _queue_sources(queue_id)}
            if mine else
            {**base, "state": "not_ours", "restored": False,
             "ownership": rec.get("state") or "none"})
    station_store.set_autoplay_restore(station["station_id"], owed,
                                       db_path=db_path)
    return owed


def _closed_by_events(debt: dict[str, Any], station_id: str,
                      now_dt: datetime, db_path) -> dict[str, Any] | None:
    """A debt the facts have settled without anyone's decision: a station
    on that queue now, another owner since, or too old to ask about."""
    import queue_ownership
    queue_id = debt.get("queue_id") or ""
    active = station_store.get_active_station(db_path=db_path)
    rec = queue_ownership.get_record(queue_id, db_path=db_path) or {}
    age = service_clock.age_bounds(debt.get("since"), now_dt)
    if active is not None and (active.get("queue_id") or "") == queue_id:
        return {"state": "superseded_by_station", "restored": False}
    if rec.get("station_id") not in (None, station_id):
        return {"state": "superseded_by_owner", "restored": False}
    if age is None or age[1] > RESTORE_MAX_AGE_S:
        return {"state": "expired", "restored": False}
    return None


def _settle_restore(station_id: str, now_dt: datetime, db_path, *,
                    at_stop: bool = False) -> dict[str, Any]:
    """Advance an open Autoplay debt. Inside the station lock.

    WRITES ONLY WITH at_stop=True -- the Stop that created the debt, a
    moment at which ownership and intent are established. Everywhere else
    (the reaper, after a restart) this only WATCHES: Autoplay seen ON
    closes the debt as `on_again` (someone switched it on: nothing left to
    do); a station or another owner on the queue, or age, closes it; an
    `owed` debt never attempted (the process ended mid-Stop) becomes
    `pending`; an OFF flag proves nothing about who chose OFF, so a pending
    debt stays pending for the person to answer (decide_restore).
    Persists only changes of state."""
    import queue_ownership
    st = station_store.get_station(station_id, db_path=db_path) or {}
    debt = (st.get("live_status") or {}).get("autoplay_restore") or {}
    if debt.get("state") not in RESTORE_OPEN:
        return debt
    queue_id = debt.get("queue_id") or ""
    result = _closed_by_events(debt, station_id, now_dt, db_path)
    if result is None and at_stop and debt.get("state") == "owed":
        result = queue_ownership.restore_after_handover(queue_id=queue_id)
        if result.get("state") in _PENDING_REASON:
            result = {**debt, "state": "pending",
                      "reason": _PENDING_REASON[result["state"]],
                      "error": result.get("error", "")}
    elif result is None:
        flag = _read_autoplay(queue_id)
        if flag is True:
            result = {"state": "on_again", "restored": False}
        else:
            now_src = _queue_sources(queue_id) if flag is False else None
            changed = (now_src is not None and debt.get("sources") is not None
                       and now_src != debt.get("sources"))
            result = {**debt, "state": "pending",
                      "reason": debt.get("reason") or "interrupted",
                      "manual_playback_since": bool(
                          debt.get("manual_playback_since") or changed)}
            if debt.get("sources") is None and now_src is not None:
                # MA could not be read at Stop: what is playing at the first
                # look after it answers becomes the baseline (context only).
                result["sources"] = now_src
    if result.get("state") not in RESTORE_OPEN:
        result = {**{k: debt.get(k) for k in ("queue_id", "speaker",
                                              "handover", "since")},
                  **result, "settled_at": _iso(now_dt)}
    def shape(d):
        return (d.get("state"), d.get("reason"), bool(d.get("manual_playback_since")),
                d.get("sources") is not None)
    if shape(result) != shape(debt):
        station_store.set_autoplay_restore(station_id, result, db_path=db_path)
        logger.info("station_live_autoplay_restore station=%s queue=%s "
                    "state=%s reason=%s", station_id[:8], queue_id,
                    result.get("state"), result.get("reason", "-"))
    return result


def settle_owed_restore(*, now: datetime | None = None, db_path=None
                        ) -> dict[str, Any] | None:
    """The reaper's look at an open Autoplay debt (it never writes Autoplay:
    _settle_restore). Only the most recent Station Programmer station can
    hold one (one station at a time; a newer station on that queue closes
    it). Cheap when nothing is open: one local read, no MA call."""
    try:
        last = station_store.last_programmer_station(db_path=db_path)
    except sqlite3.Error:
        return None
    debt = ((last or {}).get("live_status") or {}).get("autoplay_restore") or {}
    if debt.get("state") not in RESTORE_OPEN or (last or {}).get("status") == \
            station_store.ACTIVE:
        return None
    try:
        with station_lock.station_mutation():
            return _settle_restore(last["station_id"],
                                   now or service_clock.legacy_now(), db_path)
    except station_lock.StationMutationBusy:
        return None


RESTORE_DECISIONS = ("restore", "dismiss")


def decide_restore(decision: str, *, now: datetime | None = None,
                   db_path=None) -> dict[str, Any]:
    """The person's answer to a PENDING Autoplay restore: switch Autoplay
    back on for that queue (`restore`), or leave it as it is (`dismiss`).
    The only way a restore is written after the Stop that created the debt
    -- an explicit, current decision, so nothing ambiguous is overridden.
    Only the most recent Station Programmer station can hold the question
    (one station at a time), so it is answered for that one: a newer
    station since means nothing is pending, and the answer is refused.
    Writes at most the one flag (restore_after_handover)."""
    now_dt = now or service_clock.legacy_now()
    if decision not in RESTORE_DECISIONS:
        return {"ok": False, "reason": R_INTENT_INVALID,
                "message": "Choose restore or dismiss."}
    try:
        with station_lock.station_mutation():
            st = station_store.last_programmer_station(db_path=db_path) or {}
            debt = (st.get("live_status") or {}).get("autoplay_restore") or {}
            if debt.get("state") not in RESTORE_OPEN or \
                    st.get("status") == station_store.ACTIVE:
                return {"ok": False, "reason": "no_pending_restore",
                        "message": "Nothing is waiting for that decision.",
                        "autoplay_restore": debt or None}
            closed = _closed_by_events(debt, st["station_id"], now_dt, db_path)
            if closed is not None and closed["state"] != "expired":
                # A station (or another owner) has the queue now: the
                # question no longer applies, whatever the answer.
                closed = {**{k: debt.get(k) for k in ("queue_id", "speaker",
                                                      "handover", "since")},
                          **closed, "settled_at": _iso(now_dt)}
                station_store.set_autoplay_restore(st["station_id"], closed,
                                                   db_path=db_path)
                return {"ok": False, "reason": closed["state"],
                        "message": "A station is using that speaker now, so "
                                   "nothing was changed.",
                        "autoplay_restore": closed}
            if decision == "dismiss":
                result = {"state": "dismissed", "restored": False}
            else:
                import queue_ownership
                result = queue_ownership.restore_after_handover(
                    queue_id=debt.get("queue_id") or "")
                if result.get("state") == "restored":
                    result = {**result, "state": "restored_by_person"}
                elif result.get("state") in _PENDING_REASON:
                    return {"ok": False, "reason": "restore_failed",
                            "message": "Music Assistant did not switch Autoplay "
                                       "on. Try again in a moment.",
                            "autoplay_restore": debt}
            result = {**{k: debt.get(k) for k in ("queue_id", "speaker",
                                                  "handover", "since")},
                      **result, "settled_at": _iso(now_dt), "by": "person"}
            station_store.set_autoplay_restore(st["station_id"], result,
                                               db_path=db_path)
    except station_lock.StationMutationBusy:
        return {"ok": False, "reason": R_BUSY, "message": _MESSAGES[R_BUSY]}
    logger.info("station_live_autoplay_decision station=%s decision=%s "
                "state=%s", st["station_id"][:8], decision, result.get("state"))
    return {"ok": True, "autoplay_restore": result}


def _abandon_pending(station: dict[str, Any], why: str, now_dt, db_path
                     ) -> None:
    batch = station_store.get_pending_batch(station["station_id"],
                                            db_path=db_path)
    if batch is not None:
        station_store.resolve_batch(batch["batch_id"], station_store.ABANDONED,
                                    resolution=why, now=now_dt,
                                    db_path=db_path)


# --- the endless runway -----------------------------------------------------------

_maintain_lock = threading.Lock()
_last_eval: dict[str, float] = {}
_last_reason: dict[str, str] = {}


def on_refill_hook(queue_id: str, *, now: datetime | None = None,
                   db_path=None, trigger: str = "tick") -> bool:
    """The observed queue's boundary/tick hook asks: is refill mine?

    Returns False when no Station Programmer station is on air -- the
    legacy local-catalog refill then runs exactly as before. Returns True
    when one is: its maintenance runs here (throttled for ticks) and the
    legacy path must not."""
    try:
        station = station_store.get_active_station(db_path=db_path)
    except sqlite3.Error:
        return False
    if not station_store.is_programmer(station):
        return False
    own = queue_id == (station.get("queue_id") or "")
    maintain(trigger="boundary" if own and trigger == "boundary" else "tick",
             now=now, db_path=db_path)
    return True


def maintain(*, trigger: str = "reaper", now: datetime | None = None,
             db_path=None, force: bool = False) -> dict[str, Any]:
    """Keep the runway: measure, decide (station_eligibility's policy,
    unchanged), and refill when -- and only when -- the station's own music
    has run low. Single flight; every failure is a recorded reason, never
    an exception into a heartbeat."""
    if not _maintain_lock.acquire(blocking=False):
        return {"ok": True, "skipped": "busy"}
    try:
        station = station_store.get_active_station(db_path=db_path)
        if not station_store.is_programmer(station):
            # No station on air: the one thing left to keep is a promise --
            # an Autoplay restore a Stop could not finish.
            settled = settle_owed_restore(now=now, db_path=db_path)
            return {"ok": True, "active": False,
                    **({"autoplay_restore": settled} if settled else {})}
        sid = station["station_id"]
        for memo in (_last_eval, _last_reason):
            for old_sid in [k for k in memo if k != sid]:
                del memo[old_sid]
        clock = time.monotonic()
        if trigger == "tick" and not force and \
                clock - _last_eval.get(sid, -1e9) < MIN_EVAL_INTERVAL_S:
            return {"ok": True, "skipped": "throttled"}
        _last_eval[sid] = clock
        now_dt = now or service_clock.legacy_now()
        world, evidence = station_eligibility.gather_for_station(
            station, keep=True, now=now_dt, db_path=db_path)
        decision = station_eligibility.decide(world)
        _note_check(station, world, decision, evidence, trigger, now_dt,
                    db_path)
        if not decision.eligible:
            return {"ok": True, "eligible": False, "reason": decision.reason,
                    "station_id": sid}
        result = _refill(station, decision, evidence, trigger, now_dt,
                         db_path)
        _note_refill(station, decision, result, trigger, now_dt, db_path)
        return result
    except Exception:  # noqa: BLE001 - a heartbeat must never break
        logger.warning("station programmer maintenance failed (music "
                       "unaffected)", exc_info=True)
        return {"ok": False, "reason": "maintenance_failed"}
    finally:
        _maintain_lock.release()


def _refill(station: dict[str, Any], decision, evidence: dict[str, Any],
            trigger: str, now_dt: datetime, db_path) -> dict[str, Any]:
    sid, queue_id = station["station_id"], station.get("queue_id") or ""
    fail = {"ok": False, "appended_count": 0, "station_id": sid,
            "queue_id": queue_id, "mode": decision.mode, "trigger": trigger}
    if (station_store.get_active_station(db_path=db_path) or {}).get(
            "station_id") != sid:
        return dict(fail, reason=R_SUPERSEDED)
    # Still isolated? A speaker can be grouped after a station started on
    # it; a station never follows it into a group. (The house-output rule
    # is a START rule: the house moving onto this speaker is the owner's
    # doing, not the station's.) The operator's authorization is read
    # afresh: revoking it stops the next refill. A station that took over
    # the house speaker by HANDOVER is re-checked by the handover rule
    # itself: it refills only while that speaker is still in Flow Mode and
    # no group it belongs to has started playing.
    handed = bool((station.get("program") or {}).get("handover"))
    try:
        verdict = check_target(queue_id, ma_queue.player_topology(),
                               authorized=authorized_targets(_settings_doc()),
                               house_outputs={queue_id} if handed else (),
                               flow=queue_flow(queue_id) if handed else None)
    except ma_queue.MAQueueError:
        verdict = {"ok": False, "reason": R_INVENTORY_UNREADABLE}
    if not verdict["ok"]:
        return dict(fail, reason=R_NOT_ISOLATED, detail=verdict["reason"])
    ownership = station_refill._ownership(station, evidence, True, now_dt,
                                          db_path)
    if not ownership.get("refill_allowed", True):
        return dict(fail, reason=station_eligibility.R_MA_AUTOPLAY_CONTESTED,
                    ownership=ownership)
    desired = decision.desired_append_count
    try:
        intent = _intent_of(station)
        res = _resolve(intent, desired)
    except station_intent.IntentError:
        return dict(fail, reason=R_UNRESOLVABLE)
    except _Refusal as refusal:
        reason = refusal.answer["reason"]
        return dict(fail, reason=R_CATALOG if reason == R_CATALOG
                    else R_UNRESOLVABLE, detail=reason)
    history, playing = _history(queue_id, evidence.get("state") or {})
    excl = exclusions(res, evidence, history, station.get("generated") or [],
                      current=playing)
    seed = hashlib.sha256(f"{sid}#refill#{station.get('refill_count') or 0}#"
                          f"{len(station.get('generated') or [])}".encode()
                          ).digest()
    batch = choose(res, count=desired,
                   rng=random.Random(int.from_bytes(seed[:8], "big")),
                   now=now_dt, excl=excl)
    selection = batch.get("selection") or {}
    if not batch["tracks"]:
        return dict(fail, reason=R_EXHAUSTED,
                    pool_tracks=res.stats.get("pool_tracks"),
                    relaxation_pass=selection.get("relaxation_pass"),
                    loop_guarded=batch.get("loop_guarded", 0))
    resolved = [{"path": t["uri"], "uri": t["uri"], "artist": t["artist"],
                 "title": t["title"]} for t in batch["tracks"]]
    base = {"requested_count": desired, "selected_count": len(resolved),
            "resolved_count": len(resolved), "automatic": True,
            "mode": decision.mode, "trigger": trigger,
            "relaxation_pass": selection.get("relaxation_pass"),
            "loop_guarded": batch.get("loop_guarded", 0),
            "feel": batch.get("feel"), "ownership": ownership,
            "owned_before": decision.owned_future,
            "playable_before": decision.playable_future,
            "selected_tracks": [{"artist": t["artist"], "title": t["title"]}
                                for t in batch["tracks"]]}
    try:
        return station_refill.commit_append(station, queue_id, resolved,
                                            base=base, automatic=True,
                                            honour_switch=False,
                                            settle_fn=lambda: _write_error_wait(),
                                            now_dt=now_dt, db_path=db_path)
    except station_lock.StationMutationBusy:
        return dict(fail, reason=R_BUSY)


# --- diagnosis ---------------------------------------------------------------------

def _check_record(station: dict[str, Any], evidence: dict[str, Any] | None,
                  reason: str, now_dt: datetime, *, owned=None, playable=None,
                  trigger: str = "", db_path=None) -> dict[str, Any]:
    rec = {"at": _iso(now_dt), "reason": reason, "trigger": trigger}
    if evidence is not None:
        ahead = classify_ahead(evidence, station, db_path=db_path)
        rec["ahead"] = ahead
        rec["owned"] = ahead["station"] if owned is None else owned
        rec["playable"] = (station_runway.measure_runway(
            evidence.get("future") or [])["music_count"]
            if playable is None else playable)
        rec["before_station"] = _before_station(evidence, station)
    else:
        rec["owned"], rec["playable"] = owned, playable
    return rec


def _before_station(evidence: dict[str, Any], station: dict[str, Any]
                    ) -> dict[str, Any]:
    """How much music plays before the station's first own song: what a
    `keep` handover is waiting behind. {songs, seconds}; zero once the
    station's songs are next (or nothing of the station's is ahead)."""
    tail = {w.get("queue_item_id") for w in station.get("tail") or ()}
    songs = seconds = 0
    for item in evidence.get("future") or ():
        if item.get("queue_item_id") in tail:
            return {"songs": songs, "seconds": int(seconds)}
        if station_runway.classify_item(item)[0] == station_runway.MUSIC:
            songs += 1
            seconds += _seconds(item)
    return {"songs": 0, "seconds": 0}


def _note_check(station, world, decision, evidence, trigger, now_dt,
                db_path) -> None:
    """Record the runway check -- only when it CHANGED, or every
    CHECK_RECORD_S, so a quiet station costs no writes -- and log a reason
    only when it differs from the last one."""
    sid = station["station_id"]
    live = station.get("live_status") or {}
    prev = live.get("last_check") or {}
    rec = _check_record(station, evidence, decision.reason or "eligible",
                        now_dt, owned=world.owned_future,
                        playable=world.playable_future, trigger=trigger,
                        db_path=db_path)
    problem = decision.reason if decision.reason in CHECK_PROBLEMS else ""
    same = (prev.get("reason") == rec["reason"]
            and prev.get("owned") == rec["owned"]
            and prev.get("playable") == rec["playable"]
            and problem == (live.get("check_problem") or ""))
    bounds = service_clock.age_bounds(prev.get("at"), now_dt)
    age = bounds[1] if bounds is not None else None
    if not same or age is None or age >= CHECK_RECORD_S:
        station_store.update_live_status(
            sid, {"last_check": rec, "check_problem": problem},
            db_path=db_path)
    if _last_reason.get(sid) != rec["reason"]:
        _last_reason[sid] = rec["reason"]
        logger.info("station_live_check station=%s trigger=%s reason=%s "
                    "owned=%s playable=%s", sid[:8], trigger, rec["reason"],
                    rec["owned"], rec["playable"])


def _note_refill(station, decision, result, trigger, now_dt, db_path) -> None:
    sid = station["station_id"]
    rec = {"at": _iso(now_dt), "ok": bool(result.get("ok")),
           "reason": result.get("reason") or "",
           "appended": int(result.get("appended_count") or 0),
           "requested": decision.desired_append_count, "mode": decision.mode,
           "trigger": trigger,
           "relaxation_pass": result.get("relaxation_pass"),
           "batch_state": result.get("batch_state"),
           "owned_before": decision.owned_future,
           "playable_before": decision.playable_future}
    problem = "" if rec["ok"] else (rec["reason"] if rec["reason"]
                                    in REFILL_PROBLEMS else "")
    station_store.update_live_status(sid, {"last_refill": rec,
                                           "refill_problem": problem},
                                     db_path=db_path)
    logger.info("station_live_refill station=%s trigger=%s mode=%s owned=%s "
                "playable=%s requested=%s appended=%s pass=%s guarded=%s "
                "ok=%s %s", sid[:8], trigger, decision.mode,
                decision.owned_future, decision.playable_future,
                decision.desired_append_count, rec["appended"],
                rec["relaxation_pass"], result.get("loop_guarded"),
                rec["ok"], rec["reason"])


def _describe(st: dict[str, Any], db_path=None) -> dict[str, Any]:
    program = st.get("program") or {}
    live = st.get("live_status") or {}
    intent = program.get("intent") or {}
    pending = None
    if st.get("status") in (station_store.ACTIVE, station_store.PREPARING):
        b = station_store.get_pending_batch(st["station_id"], db_path=db_path)
        if b is not None:
            pending = {"claimed_at": b["claimed_at"],
                       "intended": len(b["intended_uris"])}
    return {"station_id": st["station_id"], "status": st["status"],
            "label": intent.get("label") or "",
            "query": intent.get("query") or intent.get("source") or "",
            "kind": intent.get("kind") or "",
            "preset_id": intent.get("preset_id"),
            "origin": intent.get("origin"),
            "variant": program.get("variant", 0),
            # WHICH station this is, in the form a saved preset is compared
            # in (/station/dial identities): a preset is this station only
            # while its term and approved draw still are, whatever its id.
            "identity": station_resolver.station_identity(
                intent.get("kind"), intent.get("query") or intent.get("source"),
                program.get("variant", 0)),
            "target": program.get("target") or {"queue_id": st.get("queue_id"),
                                                "name": st.get("queue_id")},
            "strategy": program.get("strategy"),
            "seed_artist": program.get("seed_artist"),
            "pool_tracks": program.get("pool_tracks"),
            "pool_artists": program.get("pool_artists"),
            "opening_count": program.get("opening_count"),
            "switched_from": program.get("switched_from") or None,
            # A house handover: how it was taken over, and whether Stop
            # gives the speaker back to Autoplay.
            "handover": program.get("handover") or None,
            # after Stop: whether the queue was given back to Autoplay
            # (restored / already_on / owed / not_ours / gave_up / ...)
            "autoplay_restore": live.get("autoplay_restore") or None,
            "created_at": st.get("created_at"),
            "started_at": st.get("activated_at"),
            "stopped_at": st.get("deactivated_at"),
            "stopped_reason": st.get("deactivated_reason") or "",
            "runway": live.get("last_check"),
            "refill": {"count": st.get("refill_count") or 0,
                       "last_at": st.get("last_replenished_at"),
                       "last": live.get("last_refill"),
                       "pending": pending,
                       "low_water": REFILL_LOW_WATER,
                       "target": REFILL_TARGET,
                       "max_batch": REFILL_MAX_BATCH},
            "owned_known": len(st.get("tail") or []),
            "witness_exact": not st.get("tail_is_partial"),
            "generated_count": len(st.get("generated") or []),
            "problem": (live.get("check_problem")
                        or live.get("refill_problem") or "")}


def dial_target(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """The speaker a Dial or Request Line start is aimed at. PURE.

    Neither has a speaker picker of its own, and neither may ever infer one
    from the house's DJ output. So the target comes from Station Programmer
    state and nothing else: the speaker the station on air is using (a
    start there is a SWITCH), else the speaker the last station ran on.
    With neither there is no target, and the caller must choose one on the
    Station tab. This only AIMS: start_station proves the speaker isolated
    again, and a speaker that has since joined a group is refused there."""
    st = snapshot.get("station") if snapshot.get("state") in (
        "on_air", "starting") else None
    source = "on_air" if st else "last_station"
    st = st or snapshot.get("last") or {}
    target = st.get("target") or {}
    queue_id = str(target.get("queue_id") or "")
    if not queue_id:
        return None
    return {"queue_id": queue_id, "name": target.get("name") or queue_id,
            "source": source}


def status(*, db_path=None) -> dict[str, Any]:
    """Mode 2, compactly, from durable state only (no MA call): is a station
    on air, which, where, how much runway at the last check, when it last
    refilled and how much it added, and why it last declined."""
    try:
        active = station_store.get_active_station(db_path=db_path)
        preparing = station_store.get_preparing_station(db_path=db_path)
        out: dict[str, Any] = {
            "ok": True, "active": False, "state": "off",
            "legacy_station_active": bool(active)
            and not station_store.is_programmer(active),
            "rules": {"opening": INITIAL_RUNWAY,
                      "low_water": REFILL_LOW_WATER,
                      "target": REFILL_TARGET, "max_batch": REFILL_MAX_BATCH,
                      "min_pool_tracks": station_resolver.LIVE_MIN_POOL_TRACKS,
                      "min_artists": station_resolver.LIVE_MIN_ARTISTS,
                      "min_opening": station_resolver.LIVE_MIN_OPENING}}
        if station_store.is_programmer(active):
            out.update(active=True, state="on_air",
                       station=_describe(active, db_path))
        elif station_store.is_programmer(preparing):
            out.update(state="starting", station=_describe(preparing, db_path))
        else:
            last = station_store.last_programmer_station(db_path=db_path)
            out["last"] = _describe(last, db_path) if last else None
        out["dial_target"] = dial_target(out)
        return out
    except sqlite3.Error:
        return {"ok": False, "active": False, "state": "unknown",
                "reason": "station_state_unreadable"}
