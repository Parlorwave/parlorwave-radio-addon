#!/usr/bin/env python3
"""Music Assistant observer: player-agnostic playback ingestion.

Without it, every playback event Music Brain receives must be forwarded by
Home Assistant automations bound to ONE speaker entity. This module reads the
same reality directly from Music Assistant — whose data is where HA's copy
comes from, and whose elapsed_time clock is the one the engine already
trusts everywhere else — and synthesizes the exact same event vocabulary:

    track_start / position / paused / resumed / idle / stop

Modes:

    off            (default)  nothing runs
    shadow         poll + synthesize + record for comparison ONLY
    authoritative  poll + synthesize + FEED the existing playback lifecycle
                   through its one canonical by-player entry point

ONE QUEUE, ONE AUTHORITY. Authoritative mode assumes HA ingestion for the
same queue has been operationally disabled first; there is deliberately no
dedup between the two feeds, because two reporters for one queue is the
bug, not a case to paper over. Shadow remains structurally incapable of
feeding: the worker only feeds through a feed function it was constructed
with, shadow construction passes none, and the only lifecycle references
in this module live inside the two _lifecycle_* seam functions. The module
still contains no MA write capability (the same "no method for it"
discipline as ma_queue's transport rule).
Observation failing must never cost the listener anything: on any error the
worker backs off and music simply continues, unobserved.

Queue hygiene: an authoritative worker asks the
hygiene engine to look at the queue -- through queue_hygiene_triggers,
fenced -- at a few named moments: its first stable playing reconcile of an
attach episode, a confirmed backward boundary (position_regressed),
playing again after a quiet stretch (quiet_recovery), a break or bulletin
leaving the air for music (program_boundary) and, as the
backstop for a missed event, PERIODIC_RECONCILE_S of settled playing with
no request at all (periodic).
A backward boundary is also recognised ACROSS a Music Assistant outage:
when the queue record comes back readable at a position earlier than the
last one this worker saw live, that is the same rewind (e.g. MA restarts
mid-song and restores its queue many items back while the speaker is off,
and the first boundary afterwards compares equal indexes).
The observer decides WHEN; the engine decides WHAT, if anything, is safe
to remove, under every gate it owns. Nothing here reads the queue's
future, pre-selects an item, or issues a playback command.

The synthesizer is a pure state machine over normalized snapshots, with all
time taken from the snapshot itself, so every rule is table-testable
without a network or a clock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Any, Callable

import config as _config
import ma_queue
import media_ref

logger = logging.getLogger("musicbrain")

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_AUTH = "authoritative"

# Target ownership: MA_OBSERVER_QUEUE holds either a fixed MA player id
# (a fixed target) or the sentinel "auto", which follows the actual
# ParlorWave station via the ownership resolver below.
TARGET_FIXED = "fixed"
TARGET_AUTO = "auto"
REASON_ACTIVE_SESSION = "active_session"
REASON_SELECTED_OUTPUT = "selected_output"
REASON_CONFIGURED_FALLBACK = "configured_fallback"
REASON_AMBIGUOUS = "ambiguous_sessions"
REASON_SELECTION_NOT_ENABLED = "selection_not_enabled"
REASON_NONE = "none"

# Cadence: boundaries and clock accuracy need single-digit-second latency
# (an error of a few seconds is what matters); every engine tolerance
# above that is minute-scale (TICK_LIVE_S 120, RENDER_LEAD_S 150, the news
# window -60/+240). 5s while watching is comfortably inside all of them.
POLL_S_DEFAULT = 5.0
POLL_S_MIN, POLL_S_MAX = 2.0, 60.0
ERROR_BACKOFF_MAX_S = 60.0

# Poll cadence and ENGINE position cadence are different dials. Boundaries,
# pause/resume and availability want the 5s poll (boundary lag of about
# one poll). But a position EVENT costs a playback hook, a DJ
# schedule pass and a news planner pass, and every consumer tolerance is
# minute-scale -- feeding one per poll would sextuple that machinery for
# nothing. Authoritative mode therefore feeds position evidence at the
# proven ~30s cadence of the HA path it replaces, while boundaries and
# state changes feed immediately.
POSITION_FEED_S = 30.0

# PROGRAM ITEMS ARE WATCHED CLOSELY. A short announcement, a few seconds
# long, can be current and gone between two 5 s polls; it then
# never gets its `air_start` and its row ends 'missed' although it aired --
# safe (nothing replays) but untrue on every card. While ParlorWave's own
# program item is current, or is next and the song before it has at most
# PROGRAM_POLL_LEAD_S left, the observer polls every PROGRAM_POLL_S. The
# evidence is unchanged and still MA's own report of the current item; only
# the reads are closer together, and only around program items. Position
# EVENTS keep their POSITION_FEED_S cadence, so no engine pass runs more often.
PROGRAM_POLL_S = 1.0
PROGRAM_POLL_LEAD_S = 15.0

# Debounces mirror the HA automations they will eventually replace: idle
# must hold for two minutes before it means "the listening stopped", and a
# vanished/unpowered player gets a short grace before it is treated as off.
# HA has no trigger at all for entity-unavailable -- a live gap this
# observer closes -- so 'gone' borrows the off semantics with a debounce.
IDLE_DEBOUNCE_S = 120.0
GONE_DEBOUNCE_S = 30.0
#: How long the player must have been QUIET (not playing: paused, idle,
#: off, gone, missing, or the queue unreadable) before resuming counts as a
#: recovery that asks hygiene to look. Shorter than the 30-minute ready/
#: queued TTL on purpose -- nothing ahead can go stale in less -- and long
#: enough that a cast group's momentary blip is not a pass.
RECOVERY_QUIET_S = 120.0
#: The periodic backstop. A routine
#: hygiene request after this long of PLAYING polls with no request and no
#: chain continuation, so a missed event can never leave spent programming
#: ahead indefinitely. One read of the queue's future when nothing is
#: spent; the engine keeps every gate and its own minimum gap.
PERIODIC_RECONCILE_S = 900.0
#: ...and after a PERSISTENT refusal (the setting is off, the latch is
#: tripped: answers that need a person) the backstop slows to this, so a
#: house that said no is not asked again every quarter hour.
PERIODIC_AFTER_PERSISTENT_REFUSAL_S = 3600.0

# The sanity bounds are copied from the HA track-change automation so the
# synthesized previous_position math is bit-compatible with today's.
PREV_GAP_SLACK_S = 300.0
PREV_POS_OVERRUN = 1.2


# --- snapshot -----------------------------------------------------------------

def _read_queues() -> Any:
    return ma_queue._api("player_queues/all")


def _read_players() -> Any:
    return ma_queue._api("players/all")


def _next_is_program(item: Any) -> bool | None:
    """True/False for MA's next item, None when MA names none."""
    if not isinstance(item, dict):
        return None
    media = item.get("media_item") if isinstance(item.get("media_item"),
                                                  dict) else {}
    return bool(media_ref.programming_ref(media.get("uri") or "")[0])


def program_near(snap: dict[str, Any], at: float | None = None) -> bool:
    """Is ParlorWave's own program item current, or next and about to
    start? Judged only from what MA reported in this snapshot."""
    item = snap.get("item") or {}
    if item.get("uri") and media_ref.programming_ref(item["uri"])[0]:
        return snap.get("state") in ("playing", "paused", "buffering")
    if not snap.get("next_is_program") or snap.get("state") != "playing":
        return False
    cur, buf = snap.get("current_index"), snap.get("index_in_buffer")
    if isinstance(cur, int) and isinstance(buf, int) and buf > cur:
        # the player already holds the program item: it starts within the
        # player's buffer. MA's elapsed for the outgoing song is not to be
        # trusted here -- it can read ~1 s just before the boundary (a
        # 5 s item can come and go between two 5 s polls)
        return True
    elapsed = _effective_elapsed(snap, snap.get("at") if at is None else at)
    try:
        duration = float(item.get("duration") or 0)
    except (TypeError, ValueError):
        return False
    if elapsed is None or duration <= 0:
        return False
    return duration - elapsed <= PROGRAM_POLL_LEAD_S


def take_snapshot(queue_id: str,
                  read_queues: Callable[[], Any] | None = None,
                  read_players: Callable[[], Any] | None = None,
                  now: float | None = None) -> dict[str, Any]:
    """One normalized observation of one queue (two bounded MA reads).

    Normalization decisions that matter:
      - `state` becomes "gone" when the PLAYER is unavailable or unpowered,
        whatever the queue claims: a queue can happily hold hundreds of
        items while its speaker is dead, so queue state alone must never
        mean "someone is listening".
      - the item keeps media_item's SEPARATED title/artists/album plus the
        uri, so downstream classification (programming_ref) sees exactly
        the identity HA forwards today.
      - `at` is stamped here and is the only clock the synthesizer uses.
    """
    at = time.time() if now is None else now
    queues = (read_queues or _read_queues)()
    players = (read_players or _read_players)()
    q = next((x for x in queues if isinstance(x, dict)
              and x.get("queue_id") == queue_id), None) \
        if isinstance(queues, list) else None
    p = next((x for x in players if isinstance(x, dict)
              and x.get("player_id") == queue_id), None) \
        if isinstance(players, list) else None

    snap: dict[str, Any] = {
        "queue_id": queue_id,
        "player_id": (p.get("player_id") or p.get("entity_id") or queue_id) if p else queue_id,
        "at": at, "state": "missing",
        "available": bool(p.get("available")) if p else False,
        "powered": bool(p.get("powered")) if p else False,
        "elapsed_time": None, "elapsed_time_last_updated": None,
        "current_index": None, "index_in_buffer": None, "items": None,
        "item": None,
    }
    if q is None:
        return snap
    snap.update({
        "state": q.get("state") or "idle",
        "elapsed_time": q.get("elapsed_time"),
        "elapsed_time_last_updated": q.get("elapsed_time_last_updated"),
        "current_index": q.get("current_index"),
        "index_in_buffer": q.get("index_in_buffer"),
        "items": q.get("items"),
        # Pandora: the queue's dynamic-source identity, so the source-aware
        # observer hook can classify the SESSION mode; and MA's end-of-queue
        # flag, the reliable "the last item finished" signal (a Cast device can
        # keep reporting 'playing' at duration, but MA marks the queue ended).
        "is_dynamic": q.get("is_dynamic"),
        "ended": q.get("ended"),
        # the listener's Repeat choice: a wrap under it is a new pass
        "repeat_mode": q.get("repeat_mode"),
        # is the item MA names as next ParlorWave's own program audio? Then
        # what was added after an ending is not music (queue_end)
        "next_is_program": _next_is_program(q.get("next_item")),
    })
    ci = q.get("current_item")
    if isinstance(ci, dict):
        mi = ci.get("media_item") if isinstance(ci.get("media_item"), dict) \
            else {}
        artists = mi.get("artists")
        album = mi.get("album")
        snap["item"] = {
            "queue_item_id": ci.get("queue_item_id"),
            "title": mi.get("name") or ci.get("name") or "",
            "artist": (artists[0].get("name")
                       if isinstance(artists, list) and artists
                       and isinstance(artists[0], dict) else ""),
            "album": (album.get("name")
                      if isinstance(album, dict) else "") or "",
            "uri": mi.get("uri"),
            "media_type": str(mi.get("media_type") or ""),
            "duration": ci.get("duration") or mi.get("duration"),
            # provider provenance for source classification
            "provider": (ci.get("streamdetails") or {}).get("provider"),
        }
    # A player that is unavailable/unpowered -- or absent from the player
    # inventory entirely while its queue record lingers -- is gone. A queue
    # row alone never means someone can hear anything.
    if p is None or not (snap["available"] and snap["powered"]):
        snap["state"] = "gone"
    return snap


def _effective_elapsed(snap: dict[str, Any], at: float) -> float | None:
    """MA elapsed + wall drift since MA's own checkpoint -- the same musical
    clock _remaining_s/estimated_air_at already trust. None when unusable."""
    elapsed = snap.get("elapsed_time")
    if elapsed is None:
        return None
    try:
        elapsed = float(elapsed)
    except (TypeError, ValueError):
        return None
    drift = 0.0
    updated = snap.get("elapsed_time_last_updated")
    if snap.get("state") == "playing" and updated is not None:
        try:
            drift = max(0.0, at - float(updated))
        except (TypeError, ValueError):
            drift = 0.0
    return max(0.0, elapsed + drift)


def _prev_effective(prev: dict[str, Any],
                    at: float) -> tuple[float | None, float | None]:
    """previous_position/previous_duration for a boundary, with the exact
    sanity bounds the HA automation applies (insane evidence -> None, never
    a guess)."""
    item = prev.get("item") or {}
    try:
        dur = float(item.get("duration") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    raw = prev.get("elapsed_time")
    if raw is None or dur <= 0:
        return None, (dur or None)
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return None, dur
    gap = 0.0
    updated = prev.get("elapsed_time_last_updated")
    if prev.get("state") == "playing" and updated is not None:
        try:
            gap = at - float(updated)
        except (TypeError, ValueError):
            return None, dur
    if gap < 0 or gap > dur + PREV_GAP_SLACK_S or raw > dur * PREV_POS_OVERRUN:
        return None, dur
    return min(raw + gap, dur), dur


def _index(snap: dict[str, Any] | None) -> int | None:
    """The queue's current_index when the queue record was readable and
    said so; None for an unreadable/absent record (or a non-int)."""
    idx = (snap or {}).get("current_index")
    if isinstance(idx, int) and not isinstance(idx, bool):
        return idx
    return None


def reconnect_rewind(live_index: int | None, snap: dict[str, Any]) -> int | None:
    """How far the queue went BACK across an unreadable stretch, or None.

    Called for the first readable queue record after one or more cycles in
    which the record could not be read at all (MA unreachable, or the queue
    absent from its inventory). It is a rewind only when all of:
      - this worker saw the player LIVE (playing) at `live_index`
        before the stretch -- a position it never saw live is not evidence;
      - the record is back at a strictly lower index;
      - the queue still holds the position it left (`items` beyond
        `live_index`): a queue that came back shorter is a different queue,
        not this one rewound, and nothing can be proven about it.
    Same or later position: None. A reconnect by itself is never a trigger."""
    cur = _index(snap)
    items = snap.get("items")
    if (live_index is None or cur is None or cur >= live_index
            or not isinstance(items, int) or isinstance(items, bool)
            or items <= live_index):
        return None
    return live_index - cur


def _nothing_took_it_on(summary: dict[str, Any] | None) -> bool:
    """Did a cleanup request leave its rewind with nobody to finish it --
    refused for want of authority (no session yet, and no playing proof on
    a record that is not playing), dropped because another request was
    running, or failed outright? A pass that ran, or a refusal the chain
    kept owed, took it on."""
    if summary is None:
        return True
    outcome = summary.get("outcome")
    return outcome == "skipped" or (
        outcome == "refused" and summary.get("reason") == "not_authoritative")


def track_start_payload(cur: dict[str, Any],
                        prev_pos: float | None = None,
                        prev_dur: float | None = None) -> dict[str, Any]:
    """The by-player track_start payload for the current snapshot. Used by
    the synthesizer (with observed previous evidence) and by restart
    reconciliation (with NONE -- an unobserved past is never fabricated)."""
    item = cur.get("item") or {}
    return {
        "queue_id": cur["queue_id"],
        "media_title": item.get("title") or "",
        "media_artist": item.get("artist") or "",
        "media_album_name": item.get("album") or "",
        "media_content_id": item.get("uri"),
        "media_content_type": item.get("media_type"),
        "duration": item.get("duration"),
        "position": _effective_elapsed(cur, cur["at"]),
        "previous_position": prev_pos,
        "previous_duration": prev_dur,
    }


# --- pure event synthesis -----------------------------------------------------

class EventSynthesizer:
    """previous snapshot -> current snapshot -> the HA event vocabulary.

    Pure: no I/O, no wall clock (time comes from snapshots), so every rule
    is deterministically testable. The first snapshot after construction is
    a BASELINE — no events are invented about a past the observer never
    saw; an already-playing track simply starts producing position ticks,
    and the next real boundary adopts exactly as the engine's existing
    external-boundary handling expects."""

    def __init__(self) -> None:
        self.prev: dict[str, Any] | None = None
        self._idle_since: float | None = None
        self._idle_sent = False
        self._gone_since: float | None = None
        self._gone_sent = False
        self._frozen: tuple[float | None, float | None] = (None, None)
        # Terminal-event live guard: until
        # this worker lifetime has actually SEEN the player playing or
        # paused, there is nothing whose stopping could truthfully be
        # reported. A stale populated queue is deliberately not "live" --
        # a dead speaker can still hold a long queue.
        self.seen_live = False
        # Grounded ownership, set by the worker from the CANONICAL session
        # resolver only (never inferred from queue contents): when the
        # target owns the active listening session, its total
        # disappearance is terminal evidence even if this synthesizer
        # lifetime never saw it live -- a closed Sendspin tab must be able
        # to end the session it owns.
        self.grounded = False
        self._absent_iid: Any = None

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _iid(snap: dict[str, Any] | None) -> Any:
        return ((snap or {}).get("item") or {}).get("queue_item_id")

    def _track_start(self, prev, cur):
        prev_pos, prev_dur = (None, None)
        if prev.get("item"):
            prev_pos, prev_dur = _prev_effective(prev, cur["at"])
        payload = track_start_payload(cur, prev_pos, prev_dur)
        # Rewind surface: a boundary whose ordered queue position went
        # BACKWARD. MA exposes no command provenance, so a listener
        # deliberately replaying an old item and an MA flow-restart resume
        # deep in history are indistinguishable
        # here -- which is exactly why this is EVIDENCE in the payload for
        # the worker to log, never a trigger for any transport command.
        prev_idx, cur_idx = prev.get("current_index"), cur.get("current_index")
        if (isinstance(prev_idx, int) and isinstance(cur_idx, int)
                and not isinstance(prev_idx, bool)
                and not isinstance(cur_idx, bool) and cur_idx < prev_idx):
            payload["position_regressed"] = prev_idx - cur_idx
            # WHERE it went, beside how far. The magnitude alone cannot be
            # reconciled against a queue afterwards; the two endpoints can,
            # and a rewind report that cannot name them is asking its
            # reader to open Music Assistant.
            payload["position_from"] = prev_idx
            payload["position_to"] = cur_idx
        return {"kind": "track_start", "payload": payload}

    def _clock_payload(self, cur):
        item = cur.get("item") or {}
        return {"queue_id": cur["queue_id"],
                "position": _effective_elapsed(cur, cur["at"]),
                "duration": item.get("duration")}

    # -- the state machine ----------------------------------------------
    def feed(self, cur: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        prev, self.prev = self.prev, cur
        if cur.get("state") in ("playing", "paused"):
            self.seen_live = True              # baseline counts: it was SEEN
        if prev is None:
            return events                      # baseline: observe, invent nothing

        state, pstate = cur.get("state"), prev.get("state")

        # Freeze the last trustworthy checkpoint the moment listening
        # stops being observable; idle/stop later report THIS position.
        if pstate in ("playing", "paused") and state not in ("playing",
                                                             "paused"):
            self._frozen = _prev_effective(prev, cur["at"]) \
                if pstate == "playing" else (prev.get("elapsed_time"),
                                             (prev.get("item") or {}).get("duration"))

        # --- absence: gone (unavailable/unpowered/uninventoried player)
        # or missing (queue vanished from MA entirely -- a closed Sendspin
        # tab). HA has NO trigger for either; borrow the 'off' -> stop
        # semantics with a debounce. Terminal eligibility is seen_live OR
        # grounded session ownership: a merely-selected target that was
        # never live and owns nothing stays absent forever in silence.
        if state in ("gone", "missing"):
            if self._gone_since is None:
                self._gone_since = cur["at"]
                self._absent_iid = self._iid(prev)
            if (not self._gone_sent and (self.seen_live or self.grounded)
                    and cur["at"] - self._gone_since >= GONE_DEBOUNCE_S):
                self._gone_sent = True
                pos, dur = self._frozen
                events.append({"kind": "stop", "payload": {
                    "queue_id": cur["queue_id"],
                    "position": pos, "duration": dur}})
            return events
        self._gone_since, self._gone_sent = None, False

        # --- idle: two-minute debounce, exactly like the HA automation.
        if state == "idle":
            if self._idle_since is None:
                self._idle_since = cur["at"]
            if (not self._idle_sent and self.seen_live
                    and cur["at"] - self._idle_since >= IDLE_DEBOUNCE_S):
                self._idle_sent = True
                pos, dur = self._frozen
                events.append({"kind": "idle", "payload": {
                    "queue_id": cur["queue_id"],
                    "position": pos, "duration": dur}})
            return events
        self._idle_since, self._idle_sent = None, False

        item_changed = (self._iid(cur) is not None
                        and self._iid(cur) != self._iid(prev))

        # --- back from total absence on the SAME item: the stream simply
        # continues; a boundary is never fabricated for a return.
        if (state == "playing" and pstate in ("gone", "missing")
                and self._iid(cur) is not None
                and self._iid(cur) == self._absent_iid):
            self._absent_iid = None
            events.append({"kind": "resumed",
                           "payload": self._clock_payload(cur)})
            return events

        # --- boundary: queue_item_id is the identity (the same song queued
        # twice is two boundaries; a re-announced title is none).
        if state == "playing" and item_changed:
            self._absent_iid = None
            events.append(self._track_start(prev, cur))
            return events

        # --- pause / resume on the SAME item.
        if pstate == "playing" and state == "paused":
            events.append({"kind": "paused",
                           "payload": self._clock_payload(cur)})
            return events
        if pstate in ("paused", "idle", "gone", "missing") \
                and state == "playing":
            if self._iid(cur) is not None and not item_changed \
                    and self._iid(prev) is not None:
                events.append({"kind": "resumed",
                               "payload": self._clock_payload(cur)})
                return events
            if self._iid(cur) is not None and self._iid(prev) is None:
                # First observable item after a quiet baseline period.
                events.append(self._track_start(prev, cur))
                return events

        # --- steady playing: a position tick every poll.
        if state == "playing" and self._iid(cur) is not None:
            events.append({"kind": "position",
                           "payload": self._clock_payload(cur)})
        return events


# --- single-station target ownership ------------------------------------------

def resolve_target(active_sessions: Any, outputs: Any,
                   current_target: str | None = None
                   ) -> tuple[str | None, str]:
    """Pure ownership resolver: which ONE queue does ParlorWave observe?

    Ownership order (single-station v1 is load-bearing):
      1. THE active listening session's queue -- an already-playing
         station is never teleported by a dropdown change, nor by an
         allow-list change (breaks on it are still gated per break by
         settings.dj_allowed_for_queue).
      2. dj.outputs.default_request_output -- the selected output becomes
         the observation/adoption target for the NEXT station start, but
         ONLY when it is DJ-enabled (in dj.outputs.enabled): the allow-list
         is the set of outputs Parlorwave Radio may adopt playback on, and
         observation never reaches outside it.
      3. A single configured enabled output -- the smallest safe fallback.
      4. Nothing. Never "some player that happens to be playing". A
         selection that is not DJ-enabled says so (REASON_SELECTION_NOT_
         ENABLED) so health can point at it.

    Multiple active sessions is a state the engine actively prevents; the
    resolver refuses to guess: it keeps the current grounded target when
    that target is one of them, otherwise targets nothing -- and either
    way says REASON_AMBIGUOUS so the ambiguity is visible."""
    sessions = [s for s in (active_sessions or [])
                if (s or {}).get("queue_id")]
    if len(sessions) == 1:
        return sessions[0]["queue_id"], REASON_ACTIVE_SESSION
    if len(sessions) > 1:
        if current_target in {s["queue_id"] for s in sessions}:
            return current_target, REASON_AMBIGUOUS
        return None, REASON_AMBIGUOUS
    outputs = outputs or {}
    selected = (outputs.get("default_request_output") or "").strip()
    enabled = [o for o in (outputs.get("enabled") or []) if o]
    if selected and selected in enabled:
        return selected, REASON_SELECTED_OUTPUT
    if len(enabled) == 1:
        return enabled[0], REASON_CONFIGURED_FALLBACK
    if selected:
        return None, REASON_SELECTION_NOT_ENABLED
    return None, REASON_NONE


# --- the observer worker ------------------------------------------------------

class ObserverWorker:
    """One long-lived polling loop for one queue.

    Shadow (no feed_fn): RECORDS ONLY. Candidate events go to counters and
    a bounded ring and nothing else happens -- the worker is structurally
    incapable of feeding because the feed function simply is not there.

    Authoritative (feed_fn injected by start_from_env): every candidate
    event is fed to the playback engine through the ONE injected seam --
    except `position`, which is downsampled to POSITION_FEED_S so the
    engine keeps the ~30s cadence it was built around, not the 5s poll.

    A feed failure never touches the poll loop: music continues, the error
    is counted, and the next cycle tries again.

    Target ownership: with sessions_fn injected (MA_OBSERVER_QUEUE=auto)
    the worker re-resolves its ONE target every cycle via resolve_target;
    otherwise queue_id is fixed for the worker's lifetime. Either way at
    most one queue is observed, and only that queue is ever fed.
    """

    RING = 50
    FED_RING = 20

    def __init__(self, queue_id: str | None, poll_s: float,
                 read_queues: Callable[[], Any] | None = None,
                 read_players: Callable[[], Any] | None = None,
                 feed_fn: Callable[[str, str, dict], Any] | None = None,
                 session_lookup: Callable[[str], Any] | None = None,
                 sessions_fn: Callable[[], Any] | None = None,
                 outputs_fn: Callable[[], Any] | None = None):
        self.queue_id = queue_id
        self.auto = sessions_fn is not None
        self._sessions_fn = sessions_fn
        self._outputs_fn = outputs_fn
        self.poll_s = poll_s
        self.program_near = False        # see program_near / next_wait
        self._read_queues = read_queues
        self._read_players = read_players
        self._feed = feed_fn
        self._session_lookup = session_lookup
        self.feeding = feed_fn is not None
        self.synth = EventSynthesizer()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at = time.time()
        self.polls = 0
        self.errors = 0
        self.last_error: str | None = None
        self.last_poll_at: float | None = None
        self.event_counts: Counter = Counter()
        self.recent: deque = deque(maxlen=self.RING)
        self.last_summary: dict[str, Any] = {}
        # feed-side accounting (all zero / empty in shadow)
        self.fed_counts: Counter = Counter()
        self.fed_errors = 0
        self.last_feed_error: str | None = None
        self.last_success_at: float | None = None
        self.recent_fed: deque = deque(maxlen=self.FED_RING)
        self._last_pos_fed_at: float | None = None
        self._reconciled = False
        # target-ownership diagnostics (static in fixed mode)
        self.target_reason = REASON_NONE if self.auto else TARGET_FIXED
        self.target_changed_at: float | None = None
        self.target_change_count = 0
        self.target_errors = 0
        self.last_target_error: str | None = None
        self.consecutive_errors = 0
        # Pandora: last source-mode status from the observer hook.
        self.last_source: dict | None = None
        # Queue hygiene: bounded accounting of the automatic requests
        # this worker has made (counts per trigger, the last summary, and
        # request failures). Requests are made only from poll_once, only
        # while feeding, only at the two qualifying moments.
        self.hygiene_counts: Counter = Counter()
        self.last_hygiene: dict[str, Any] | None = None
        self.hygiene_errors = 0
        self.last_hygiene_error: str | None = None
        # Rewind across an MA outage: the last index this worker saw LIVE
        # for the current target (re-baselined at each reconnect), and
        # whether the previous cycle could not read the queue record at all.
        self._live_index: int | None = None
        self._unreadable = False
        # A QUIET stretch: since when the player has not been playing
        # (paused, idle, off, gone, missing, or the queue unreadable),
        # for this target; whether the first playing record after it has
        # been compared with _live_index yet; and whether that stretch
        # already raised a regression (so the resume asks once, never twice).
        self._quiet_since: float | None = None
        self._quiet_checked = False
        self._quiet_regressed = False
        # A rewind first seen on a record that was NOT playing, which no
        # cleanup took on: the last live index it was measured from. The
        # stretch still owes it, and its first PLAYING record decides it.
        self._owed_from: int | None = None
        # The periodic backstop's clock (the last hygiene
        # request or continuation, else the first settled playing look) and
        # whether the last answer was one that needs a person.
        self._hygiene_anchor_at: float | None = None
        self._hygiene_persistent = False
        # Has the queue genuinely FINISHED, and is a restart a new playback
        # pass? (queue_end: a finished queue otherwise looks paused)
        import queue_end
        self._queue_end = queue_end.QueueEndTracker(queue_id or "")

    # one cycle, callable directly by tests
    def poll_once(self, now: float | None = None) -> list[dict[str, Any]]:
        if self.auto:
            self._resolve_target(now)
            # Only the canonical session resolver grounds terminal
            # eligibility -- never queue contents, never selection alone.
            self.synth.grounded = (
                self.target_reason == REASON_ACTIVE_SESSION)
        if self.queue_id is None:
            # nothing owns observation right now: idle cycle, invent nothing
            self.polls += 1
            self.last_poll_at = now if now is not None else time.time()
            self.last_summary = {"state": None, "available": None,
                                 "queue_item_id": None, "title": None,
                                 "elapsed": None}
            self.consecutive_errors = 0
            return []
        try:
            snap = take_snapshot(self.queue_id, self._read_queues,
                                 self._read_players, now=now)
        except Exception:
            self._unreadable = True          # MA unreachable this cycle
            if self._quiet_since is None:
                self._quiet_since = now if now is not None else time.time()
                self._quiet_checked = self._quiet_regressed = False
            raise
        # A readable record right after unreadable cycles is a reconnect;
        # decided before anything below can move either flag.
        reconnect = self._unreadable and _index(snap) is not None
        self._unreadable = _index(snap) is None
        self.program_near = program_near(snap)
        self.last_poll_at = snap["at"]
        self.polls += 1
        item = snap.get("item") or {}
        self.last_summary = {
            "state": snap.get("state"), "available": snap.get("available"),
            "queue_item_id": item.get("queue_item_id"),
            "title": item.get("title"),
            "elapsed": snap.get("elapsed_time"),
        }
        prev_snap = self.synth.prev          # before feed advances it
        events = self.synth.feed(snap)
        regressed = 0
        attached = False
        regressed_from: Any = None
        regressed_to: Any = None
        for ev in events:
            self.event_counts[ev["kind"]] += 1
            self.recent.append({"at": snap["at"], "kind": ev["kind"],
                                "title": item.get("title"),
                                "queue_item_id": item.get("queue_item_id")})
            payload = ev.get("payload") or {}
            steps = payload.get("position_regressed")
            if steps:
                regressed_from = payload.get("position_from")
                regressed_to = payload.get("position_to")
                # Log-only towards PLAYBACK: retired program
                # audio is already non-fatal, and a backward jump cannot be
                # told apart from a listener's own choice, so no automatic
                # correction is issued from here. It IS a hygiene
                # trigger (below): the items the station already spoke sit
                # in the future again, and only the engine decides which.
                logger.warning(
                    "ma_queue_position_regressed queue=%s steps=%d "
                    "from_index=%s to_index=%s "
                    "(listener history replay or MA flow-restart rewind; "
                    "no corrective command issued)",
                    snap.get("queue_id"), steps, regressed_from, regressed_to)
                regressed = max(regressed, int(steps))
        owed_from = None
        if reconnect:
            # The queue record is back. Compared against the last position
            # seen LIVE, not the previous snapshot: after an outage the
            # previous snapshot is the rewound one, and the next boundary
            # would compare equal indexes.
            # The same evidence as a backward boundary, so the same trigger;
            # when the synthesizer already reported this boundary the larger
            # figure stands and nothing is requested twice.
            steps = reconnect_rewind(self._live_index, snap)
            if steps and steps > regressed:
                regressed_from, regressed_to = self._live_index, _index(snap)
                logger.warning(
                    "ma_queue_position_regressed queue=%s steps=%d "
                    "from_index=%s to_index=%s (queue record back after an "
                    "outage earlier than the last live position; "
                    "no corrective command issued)",
                    snap.get("queue_id"), steps, regressed_from, regressed_to)
                regressed = steps
                if snap.get("state") != "playing":
                    owed_from = regressed_from
            # Whatever it said, this is now the position to reconcile the
            # NEXT outage against: the same restart seen twice is one rewind.
            self._live_index = _index(snap)
            if regressed:
                self._quiet_regressed = True
        owed_base = self._owed_from
        recovered, quiet_s = self._after_quiet(snap, regressed)
        if recovered == "regressed":
            base = self._live_index if owed_base is None else owed_base
            regressed_from, regressed_to = base, _index(snap)
            regressed = reconnect_rewind(base, snap) or regressed
            logger.warning(
                "ma_queue_position_regressed queue=%s steps=%d "
                "from_index=%s to_index=%s (resumed earlier than the last "
                "live position after %ds quiet; no corrective command "
                "issued)", snap.get("queue_id"), regressed, regressed_from,
                regressed_to, int(quiet_s or 0))
        elif recovered == "recovered":
            logger.info(
                "ma_queue_quiet_recovery queue=%s quiet_s=%d index=%s "
                "last_live_index=%s (playing again after a quiet stretch; "
                "asking hygiene to look)", snap.get("queue_id"),
                int(quiet_s or 0), _index(snap), self._live_index)
        self._observe_queue_end(snap)
        cur_idx = _index(snap)
        if cur_idx is not None:
            if snap.get("state") == "playing":
                self._live_index = cur_idx
            elif snap.get("state") == "paused":
                if self._live_index is None or cur_idx >= self._live_index:
                    self._live_index = cur_idx
        if self._feed is not None:
            if (not self._reconciled and snap.get("state") == "playing"
                    and item.get("queue_item_id") is not None):
                self._reconciled = True
                attached = True
                if not any(ev["kind"] == "track_start" for ev in events):
                    self._reconcile(snap)
            for ev in events:
                self._maybe_feed(ev, snap)
        # Rewind-and-stall recovery watches the OBSERVED sequence
        # (regression while owned+playing, then idle within the incident
        # window). Fully fenced; the recovery module verifies station
        # ownership itself (this observer never reaches into station
        # state) and does nothing on any doubt.
        try:
            import queue_recovery
            queue_recovery.observe(prev_snap, snap, events, snap["at"])
        except Exception:
            logger.warning("rewind recovery observe failed (nothing done)",
                           exc_info=True)
        # Pandora: source-mode classification (deduped) + firing an armed
        # local->Pandora handoff at the intended transition's natural end.
        # Fenced identically -- Pandora awareness must never cost observation.
        try:
            import pandora_observer
            self.last_source = pandora_observer.observe(prev_snap, snap, events)
        except Exception:
            logger.warning("pandora observe failed (nothing done)",
                           exc_info=True)
        # Queue hygiene: at most ONE request per cycle, only while this
        # worker is the playback authority, only at the two qualifying
        # moments -- the first stable playing reconcile of this attach
        # episode (the flag above, reset by a target change) and a
        # confirmed backward boundary. Runs last, after every feed of this
        # cycle, synchronously on this thread (it is bounded by the
        # engine's own budget and no worker is spawned for it). Fenced
        # identically to the other consumers: hygiene failing must never
        # cost observation.
        if self.feeding and (attached or regressed
                             or recovered == "recovered"):
            summary = self._request_hygiene(
                snap, attached, regressed, regressed_from, regressed_to,
                quiet_recovery=recovered == "recovered",
                recovered_verdict=recovered, quiet_s=quiet_s)
            if owed_from is not None and _nothing_took_it_on(summary):
                # Seen while nothing played, and nothing took it on: with
                # no session and no playing proof there is no authority yet
                # (a cleanup it could own would have been kept, owed, by the
                # chain). The stretch still owes it, and its first PLAYING
                # record decides it again -- never a guess made now, never
                # dropped. A second reconnect in the same stretch measures
                # from an already-rewound index: the earliest live position
                # owed stays the one to measure from.
                self._owed_from = (owed_from if self._owed_from is None
                                   else max(self._owed_from, owed_from))
                self._quiet_regressed = False
        elif self.feeding:
            # Offer this look to an armed reconciliation chain. A
            # rewind arrives as several regressions and one pass cannot
            # finish it, so the chain uses THIS poll as its clock -- head
            # identity is all it needs, and it still reads no queue. Costs
            # nothing on the overwhelming majority of polls (no chain ->
            # immediate None). Fenced like every other consumer.
            armed = self._continue_hygiene(snap)
            if not armed and self._boundary_due(prev_snap, snap):
                # A break or a bulletin (or the tombstone that
                # stood in for a retired one) just left the air and music is
                # playing again. The lifecycle's own moment to remove what
                # else is spent ahead -- before the head reaches it and MA
                # records a silent replay. Routine; every gate still applies.
                self._request_hygiene(snap, False, 0, program_boundary=True)
            elif not armed and self._periodic_due(snap):
                # The backstop for a missed event. Nothing has
                # asked for PERIODIC_RECONCILE_S of playing (longer after an
                # answer that needs a person). Routine; a clean queue costs
                # one read and no write.
                self._request_hygiene(snap, False, 0, periodic=True)
        self.consecutive_errors = 0
        return events

    def _observe_queue_end(self, snap: dict[str, Any]) -> None:
        """Feed the end-of-queue tracker and publish its answer. A restart
        it calls a NEW pass tells the DJ engine, so boundaries hosted on the
        previous pass are opportunities again. Fenced: never costs
        observation."""
        try:
            import queue_end
            ev = self._queue_end.observe(snap)
            queue_end.publish(snap.get("queue_id") or "",
                              self._queue_end.state())
            if ev and ev.get("kind") == queue_end.EV_ENDED:
                logger.info("ma_queue_ended queue=%s basis=%s items=%s "
                            "(the finite queue played to its end; nothing "
                            "is restarted or refilled)", snap.get("queue_id"),
                            ev.get("basis"), snap.get("items"))
            elif ev and ev.get("kind") == queue_end.EV_REOPENED:
                logger.info("ma_queue_reopened queue=%s ended_for_s=%s (too "
                            "soon to be a listener's restart: same pass)",
                            snap.get("queue_id"), ev.get("ended_for_s"))
            elif ev and ev.get("kind") == queue_end.EV_NEW_PASS:
                logger.info("ma_queue_new_pass queue=%s reason=%s evidence=%s "
                            "ended_for_s=%s", snap.get("queue_id"),
                            ev.get("reason"), ev.get("evidence"),
                            ev.get("ended_for_s"))
                import dj_engine
                dj_engine.begin_pass(snap.get("queue_id") or "",
                                     "%s/%s" % (ev.get("reason"),
                                                ev.get("evidence")))
        except Exception:
            logger.warning("queue end observe failed (nothing done)",
                           exc_info=True)

    def _after_quiet(self, snap: dict[str, Any], regressed: int
                     ) -> tuple[str | None, float | None]:
        """The QUIET-STRETCH rule (queue hygiene recovery).

        While Music Assistant stays reachable, a player that stops, drops,
        pauses or goes unavailable leaves the queue READABLE, so the reconnect
        outage rule never runs -- and the synthesizer compares the resume
        with the previous snapshot, which is already wherever MA moved the
        queue while nothing played. Two ways spent programming then sits
        ahead of the head with nobody asking hygiene to look:

          - MA moved current_index BACK while the player was quiet: the
            first PLAYING record after the stretch is compared with the
            last index this worker saw live -- the same evidence and the
            same test (reconnect_rewind) as the outage rule -> "regressed"
            (the mandatory position_regressed);
          - items waiting ahead went terminal while nothing played (a
            queued break's TTL, a bulletin's age) with no rewind at all:
            the first PLAYING record after a stretch of RECOVERY_QUIET_S
            or longer -> "recovered" (the routine quiet_recovery, which
            asks the routine-cleanup permission setting, like an attach).

        Once per stretch, never both, and nothing when the stretch already
        raised a regression (an outage reconnect) -- unless that rewind was
        seen on a record that was not playing and nothing took it on: then
        it is still OWED, and this first playing record measures it from the
        live index it was seen against. Short pauses (<120s)
        ask nothing; long pauses (>=120s) ask quiet_recovery on the first
        authoritative playing reading after resume (or position_regressed
        if rewound). Nothing mutates while paused or quiet. Returns
        (verdict or None, quiet seconds)."""
        at = snap["at"]
        state = snap.get("state")
        playing = state == "playing" and _index(snap) is not None
        if not playing:
            if self._quiet_since is None:
                self._quiet_since = at
                self._quiet_checked = self._quiet_regressed = False
            return None, None
        if self._quiet_since is None:
            return None, None
        quiet_s = at - self._quiet_since
        verdict = None
        if not self._quiet_checked:
            self._quiet_checked = True
            base = (self._live_index if self._owed_from is None
                    else self._owed_from)
            steps = (None if self._quiet_regressed
                     else reconnect_rewind(base, snap))
            if steps and steps > regressed:
                self._quiet_regressed = True
                verdict = "regressed"
        if (verdict is None and not regressed
                and not self._quiet_regressed
                and quiet_s >= RECOVERY_QUIET_S):
            verdict = "recovered"
        self._quiet_since = None
        self._quiet_checked = self._quiet_regressed = False
        self._owed_from = None             # this stretch is decided
        return verdict, quiet_s

    def _boundary_due(self, prev: dict[str, Any] | None,
                      snap: dict[str, Any]) -> bool:
        """Did a programme item just leave the air for MUSIC? The previous
        look's item was programme-shaped (a break, a bulletin, or the
        tombstone standing in for a retired one), this look's item is a
        different item that is not, and the queue is settled enough to be
        asked (playing, with a position and a buffer index). Decides
        nothing about the queue beyond that."""
        if not prev:
            return False
        p_item = prev.get("item") or {}
        c_item = snap.get("item") or {}
        p_kind, _ = media_ref.programming_ref(p_item.get("uri"))
        c_kind, _ = media_ref.programming_ref(c_item.get("uri"))
        if not p_kind or c_kind:
            return False
        if not c_item.get("queue_item_id") \
                or c_item.get("queue_item_id") == p_item.get("queue_item_id"):
            return False
        return self._settled(snap)

    def _periodic_due(self, snap: dict[str, Any]) -> bool:
        """Has the queue played for PERIODIC_RECONCILE_S (or, after a
        persistent refusal, PERIODIC_AFTER_PERSISTENT_REFUSAL_S) with no
        hygiene request and no chain continuation? The clock starts at the
        first settled playing look and is moved by every request."""
        if not self._settled(snap):
            return False
        at = snap["at"]
        if self._hygiene_anchor_at is None:
            self._hygiene_anchor_at = at
            return False
        wait = (PERIODIC_AFTER_PERSISTENT_REFUSAL_S if self._hygiene_persistent
                else PERIODIC_RECONCILE_S)
        return (at - self._hygiene_anchor_at) >= wait

    @staticmethod
    def _settled(snap: dict[str, Any]) -> bool:
        buf = snap.get("index_in_buffer")
        return (snap.get("state") == "playing" and _index(snap) is not None
                and isinstance(buf, int) and not isinstance(buf, bool))

    def _continue_hygiene(self, snap: dict[str, Any]) -> bool:
        """One chance for an armed chain to take its next bounded pass.
        Returns whether a chain is armed for this queue at all: the
        boundary and periodic moments stand aside for an armed chain (it
        is the active episode; they are backstops).

        Decides nothing about the queue: it hands over the head identity
        and the playing state, and the trigger module decides whether the
        queue has settled enough to be worth asking the engine again.
        Playing WITHOUT a position (Music Assistant reporting no
        current_index, or no index_in_buffer -- it publishes them one after
        the other while a queue restarts) is not a settled queue: the chain
        waits, kept, and nothing is asked on a guess."""
        buf = snap.get("index_in_buffer")
        try:
            import queue_hygiene_triggers
            summary = queue_hygiene_triggers.maybe_continue(
                self.queue_id, (snap.get("item") or {}).get("queue_item_id"),
                playing=(snap.get("state") == "playing"
                         and _index(snap) is not None
                         and isinstance(buf, int) and not isinstance(buf, bool)))
            armed = queue_hygiene_triggers.chain_status() is not None
        except Exception as exc:  # noqa: BLE001 - never costs observation
            self.hygiene_errors += 1
            self.last_hygiene_error = str(exc)[:200]
            logger.warning("queue hygiene continuation failed "
                           "(observation unaffected)", exc_info=True)
            return False
        if summary is None:
            return armed
        self._hygiene_anchor_at = snap["at"]
        self.hygiene_counts["chain_pass"] += 1
        self.last_hygiene = {
            "at": snap["at"], "trigger": summary.get("trigger"),
            "chain_generation": summary.get("chain_generation"),
            "chain_pass": summary.get("chain_pass"),
            "chain_kind": summary.get("chain_kind"),
            "policy": summary.get("policy"),
            "gap_waived": summary.get("gap_waived"),
            "ordinary_authority": summary.get("ordinary_authority"),
            "recovery_only_authority": summary.get("recovery_only_authority"),
            "recovery_authoritative": summary.get("recovery_authoritative"),
            "recovery_reason": summary.get("recovery_reason"),
            "retained_reasons": summary.get("retained_reasons"),
            "outcome": summary.get("outcome"),
            "reason": summary.get("reason"),
            "stale_found": summary.get("stale_found"),
            "stale_removed": summary.get("stale_removed"),
            "remaining": summary.get("remaining"),
            "more_remaining": summary.get("more_remaining"),
            "latch": summary.get("latch"),
            "retry": queue_hygiene_triggers.retry_class(
                summary.get("outcome"), summary.get("reason")),
        }
        return armed

    def _request_hygiene(self, snap: dict[str, Any], attached: bool,
                         regressed: int, regressed_from: Any = None,
                         regressed_to: Any = None, *,
                         quiet_recovery: bool = False,
                         recovered_verdict: str | None = None,
                         quiet_s: float | None = None,
                         program_boundary: bool = False,
                         periodic: bool = False
                         ) -> dict[str, Any] | None:
        """One coalesced automatic hygiene request for this cycle. The
        engine keeps every gate (latch, authority, lock, exclusions, gap,
        ownership, staleness, the serving belt); this method decides
        nothing about the queue and pre-selects nothing.

        The PERMISSION gate is the one the trigger module may waive, and
        only on a cycle carrying positive regression evidence -- the
        already-aired-items-are-ahead fault. That decision lives there,
        beside the reasoning for it; this method only reports what it
        observed, including WHERE the queue went. Returns the request's
        summary, or None when the request itself failed."""
        trigger = ("initial_attach" if attached
                   else "position_regressed" if regressed
                   else "quiet_recovery" if quiet_recovery
                   else "program_boundary" if program_boundary
                   else "periodic" if periodic
                   else "position_regressed")
        also = "position_regressed" if (attached and regressed) else None

        # Recovery evidence for a post-quiet event: the quiet-stretch rule's
        # verdicts, and a rewind raised ON the record that ends a quiet
        # stretch (an outage reconnect that is already playing) -- the same
        # event, so the same proof; the engine still decides with it. It is
        # handed to THIS request and held nowhere else: no other pass on the
        # queue can borrow it (see the module note above the worker state).
        rec_ev = None
        if self.feeding and (quiet_recovery
                             or recovered_verdict in ("recovered", "regressed")
                             or (regressed and quiet_s is not None)):
            rec_ev = {
                "trigger": trigger,
                "queue_id": self.queue_id,
                "player_id": snap.get("player_id") or self.queue_id,
                "state": snap.get("state"),
                "feeding": bool(self.feeding),
                "quiet_s": quiet_s,
                "quiet_recovery": bool(quiet_recovery),
                "after_quiet": bool(quiet_recovery or (quiet_s is not None and quiet_s >= 120.0)),
            }

        try:
            import queue_hygiene_triggers
            summary = queue_hygiene_triggers.request(
                trigger, self.queue_id, steps=regressed or None, also=also,
                from_index=regressed_from, to_index=regressed_to,
                recovery_evidence=rec_ev)
        except Exception as exc:  # noqa: BLE001 - hygiene never costs observation
            self.hygiene_errors += 1
            self.last_hygiene_error = str(exc)[:200]
            logger.warning("queue hygiene request failed (trigger=%s; "
                           "observation unaffected)", trigger, exc_info=True)
            return None
        self.hygiene_counts[trigger] += 1
        # Every request moves the periodic clock, and an answer
        # that needs a person (the setting off, the latch tripped) slows it.
        self._hygiene_anchor_at = snap["at"]
        self._hygiene_persistent = (
            queue_hygiene_triggers.retry_class(summary.get("outcome"),
                                               summary.get("reason"))
            == "persistent")
        self.last_hygiene = {
            "at": snap["at"], "trigger": trigger, "also": also,
            "retry": queue_hygiene_triggers.retry_class(
                summary.get("outcome"), summary.get("reason")),
            "steps": regressed or None,
            "from_index": regressed_from, "to_index": regressed_to,
            "policy": summary.get("policy"),
            "mandatory": summary.get("mandatory"),
            "permission_bypassed": summary.get("permission_bypassed"),
            "ordinary_authority": summary.get("ordinary_authority"),
            "recovery_only_authority": summary.get("recovery_only_authority"),
            "recovery_authoritative": summary.get("recovery_authoritative"),
            "recovery_reason": summary.get("recovery_reason"),
            "retained_reasons": summary.get("retained_reasons"),
            "outcome": summary.get("outcome"),
            "reason": summary.get("reason"),
            "stale_found": summary.get("stale_found"),
            "stale_removed": summary.get("stale_removed"),
            "preserved": summary.get("preserved"),
            "preserved_total": summary.get("preserved_total"),
            "deleted": summary.get("deleted"),
            "remaining": summary.get("remaining"),
            "more_remaining": summary.get("more_remaining"),
            "latch": summary.get("latch"),
        }
        return summary

    def _resolve_target(self, now: float | None = None) -> None:
        """Auto mode: re-resolve ownership every cycle. Resolver-input
        failure keeps the current grounded target -- one flaky read must
        never cause target thrash."""
        try:
            sessions = self._sessions_fn() if self._sessions_fn else []
            outputs = self._outputs_fn() if self._outputs_fn else {}
        except Exception as exc:  # noqa: BLE001 - resolver boundary
            self.target_errors += 1
            self.last_target_error = str(exc)[:200]
            return
        target, reason = resolve_target(sessions, outputs, self.queue_id)
        self.target_reason = reason
        if target != self.queue_id:
            self._set_target(target, now)

    def _set_target(self, target: str | None,
                    now: float | None = None) -> None:
        """A target change is a HARD observer context boundary: fresh
        synthesizer, fresh position cadence, fresh reconciliation. The old
        target simply stops being observed -- no terminal event is invented
        for it, and nothing from its stream may ground the new one."""
        self.queue_id = target
        self.synth = EventSynthesizer()
        self._last_pos_fed_at = None
        self._reconciled = False
        self._live_index, self._unreadable = None, False
        self._hygiene_anchor_at, self._hygiene_persistent = None, False
        self._quiet_since = None
        self._quiet_checked = self._quiet_regressed = False
        self._owed_from = None
        import queue_end
        self._queue_end = queue_end.QueueEndTracker(target or "")
        self.target_changed_at = now if now is not None else time.time()
        self.target_change_count += 1

    def _reconcile(self, snap: dict[str, Any]) -> None:
        """First-playing-snapshot reconciliation (restart truthfulness).

        The synthesizer baseline is deliberately silent, so a worker that
        starts against an already-playing player would otherwise never tell
        the engine what is on air. Ask the engine instead:

          - the active session is already on this exact item -> say nothing;
          - anything else (session diverged, or no session at all) -> ONE
            grounded track_start with NO previous_* evidence.

        An unobserved past is never fabricated: previous_position and
        previous_duration stay None because nobody watched that track end.
        """
        sess = None
        if self._session_lookup is not None:
            try:
                sess = self._session_lookup(self.queue_id)
            except Exception as exc:  # noqa: BLE001 - engine boundary
                self.fed_errors += 1
                self.last_feed_error = str(exc)[:200]
                return
        uri = (snap.get("item") or {}).get("uri") or ""
        if sess is not None and (sess.get("current_media_key") or "") == uri:
            return
        self._do_feed("track_start", track_start_payload(snap), snap,
                      reconcile=True)
        self._last_pos_fed_at = snap["at"]

    def _maybe_feed(self, ev: dict[str, Any], snap: dict[str, Any]) -> None:
        kind = ev["kind"]
        if kind == "position":
            at = snap["at"]
            if (self._last_pos_fed_at is not None
                    and at - self._last_pos_fed_at < POSITION_FEED_S):
                return
            self._last_pos_fed_at = at
        elif kind == "track_start":
            # a boundary restarts the position cadence; the payload already
            # carries the fresh position, so the next tick can wait ~30s
            self._last_pos_fed_at = snap["at"]
        self._do_feed(kind, ev["payload"], snap)

    def _do_feed(self, kind: str, payload: dict[str, Any],
                 snap: dict[str, Any], reconcile: bool = False) -> None:
        try:
            self._feed(self.queue_id, kind, payload)
        except Exception as exc:  # noqa: BLE001 - engine boundary
            self.fed_errors += 1
            self.last_feed_error = str(exc)[:200]
            return
        self.fed_counts[kind] += 1
        self.last_success_at = snap["at"]
        item = snap.get("item") or {}
        entry: dict[str, Any] = {"at": snap["at"], "kind": kind,
                                 "title": item.get("title"),
                                 "queue_item_id": item.get("queue_item_id")}
        if reconcile:
            entry["reconcile"] = True
        self.recent_fed.append(entry)

    def next_wait(self, backoff: float, ok: bool) -> float:
        """The pause before the next poll: PROGRAM_POLL_S around a program
        item (see program_near), the ordinary cadence (or error backoff)
        otherwise."""
        if ok and self.program_near:
            return min(backoff, PROGRAM_POLL_S)
        return backoff

    def _run(self) -> None:
        backoff = self.poll_s
        while not self.stop_event.is_set():
            ok = False
            try:
                self.poll_once()
                backoff = self.poll_s
                ok = True
            except Exception as exc:  # noqa: BLE001 - observe-only boundary
                self.errors += 1
                self.consecutive_errors += 1
                self.last_error = str(exc)[:200]
                backoff = min(backoff * 2, ERROR_BACKOFF_MAX_S)
            self.stop_event.wait(self.next_wait(backoff, ok))

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="ma-observer")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


# --- the ONE seam into the playback engine ------------------------------------
# These two functions are the ONLY place this module may name the playback
# lifecycle, and the static isolation test enforces exactly that: lifecycle
# references are permitted only inside functions whose names start with
# `_lifecycle_`. Shadow workers never receive them; authoritative workers
# receive them as constructor-injected callables and know nothing else.

def _lifecycle_feed(player_id: str, kind: str,
                    payload: dict[str, Any]) -> Any:
    import playback_lifecycle
    return playback_lifecycle.handle_event_by_player(player_id, kind, payload)


def _lifecycle_active_session(queue_id: str) -> Any:
    import playback_lifecycle
    return playback_lifecycle.get_active_for_queue(queue_id)


def _lifecycle_active_sessions() -> Any:
    import playback_lifecycle
    return playback_lifecycle.list_active_sessions()


def _dj_outputs() -> dict[str, Any]:
    """The dj.outputs settings block (selected output + enabled list),
    read-only. Observation ownership never writes settings: following a
    player and being ALLOWED to speak on it stay separate decisions."""
    import settings as _settings
    return dict((_settings.get_section("dj") or {}).get("outputs") or {})


# --- module lifecycle / env contract ------------------------------------------

_worker: ObserverWorker | None = None
_worker_lock = threading.Lock()

# There is deliberately NO module-level recovery evidence here. Evidence
# parked in a process-wide dictionary around a hygiene request, and read
# back by the engine for ANY pass on that queue, would let an unrelated
# dry-run diagnostic or manual pass in that window acquire recovery-only
# authority it had not earned. The evidence travels as an argument of the
# one request that earned it.


# The observer's settings come from the typed configuration boundary
# (config.py): MA_OBSERVER_MODE / MA_OBSERVER_QUEUE / MA_OBSERVER_POLL_S with
# the product defaults authoritative / auto. The function names are kept
# for the tests and the log lines that reference them.

def _env_mode() -> str:
    mode = _config.current().observer_mode
    if mode in (MODE_OFF, MODE_SHADOW, MODE_AUTH):
        return mode
    return MODE_OFF  # unknown value fails SAFE: nothing runs, nothing feeds


def _env_queue() -> tuple[str | None, str]:
    """(fixed_queue_or_None, target_mode). The configured queue is either
    the sentinel "auto" (dynamic ownership resolver, the default) or a
    fixed MA player id."""
    raw = (_config.current().observer_queue or "").strip()
    if not raw:
        return None, ""
    if raw.lower() == TARGET_AUTO:
        return None, TARGET_AUTO
    return raw, TARGET_FIXED


def _env_poll_s() -> float:
    try:
        val = float(_config.current().observer_poll_s or POLL_S_DEFAULT)
    except ValueError:
        val = POLL_S_DEFAULT
    return min(max(val, POLL_S_MIN), POLL_S_MAX)


def start_from_env() -> str:
    """App-startup hook. Returns the effective mode for the log line.

    authoritative: (the product default) candidate events FEED the playback
                   engine by player id. ONE QUEUE, ONE AUTHORITY -- no other
                   playback-event source may feed this player.
    shadow:        records candidate events only; something else is the
                   playback authority.
    off:           nothing runs.

    Both running modes need a queue target; the default (and an empty
    MA_OBSERVER_QUEUE) is "auto" -- the ownership resolver, which never
    guesses a player to watch."""
    global _worker
    mode = _env_mode()
    if mode == MODE_OFF:
        return MODE_OFF
    queue_id, target_mode = _env_queue()
    if not target_mode:
        logger.warning("MA_OBSERVER_MODE=%s but MA_OBSERVER_QUEUE is empty; "
                       "observer stays off", mode)
        return MODE_OFF
    with _worker_lock:
        if _worker is not None:
            return mode
        kwargs: dict[str, Any] = {}
        if mode == MODE_AUTH:
            kwargs.update(feed_fn=_lifecycle_feed,
                          session_lookup=_lifecycle_active_session)
        if target_mode == TARGET_AUTO:
            kwargs.update(sessions_fn=_lifecycle_active_sessions,
                          outputs_fn=_dj_outputs)
        _worker = ObserverWorker(queue_id, _env_poll_s(), **kwargs)
        _worker.start()
    # What this line says reaches every add-on log on a fresh install, so
    # it says only what is true THERE: which station is followed, and no
    # instruction about other event sources an installation may not have.
    target_desc = ("the active station (auto ownership)"
                   if target_mode == TARGET_AUTO else queue_id)
    if mode == MODE_AUTH:
        logger.info("MA observer authoritative for %s every %.0fs (feeding "
                    "the playback engine; positions ~%.0fs)",
                    target_desc, _worker.poll_s, POSITION_FEED_S)
    else:
        logger.info("MA shadow observer watching %s every %.0fs (records "
                    "only; Home Assistant remains the playback authority)",
                    target_desc, _worker.poll_s)
    return mode


def stop() -> None:
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker = None


def status() -> dict[str, Any]:
    """Bounded diagnostics for /observer/status. No filesystem paths, no
    tokens, no unbounded lists -- both rings are capped and summarised.
    `mode` + `feeding` answer the only question that matters during a
    cutover: is this process the playback authority right now?"""
    mode = _env_mode()
    cfg = _config.current()
    configured = {"configured_mode": cfg.observer_mode,
                  "configured_queue": cfg.observer_queue,
                  "queue_mode": _env_queue()[1] or None}
    w = _worker
    if w is None:
        return {"ok": True, "mode": mode, "active_mode": MODE_OFF,
                "running": False, "feeding": False, **configured}
    return {
        "ok": True,
        "mode": MODE_AUTH if w.feeding else MODE_SHADOW,
        "active_mode": MODE_AUTH if w.feeding else MODE_SHADOW,
        **configured,
        "running": True,
        "feeding": w.feeding,
        "queue_id": w.queue_id, "poll_s": w.poll_s,
        "target_mode": TARGET_AUTO if w.auto else TARGET_FIXED,
        "resolved_queue_id": w.queue_id,
        "target_reason": w.target_reason,
        "target_changed_at": w.target_changed_at,
        "target_change_count": w.target_change_count,
        "target_errors": w.target_errors,
        "last_target_error": w.last_target_error,
        "started_at": w.started_at, "polls": w.polls,
        "errors": w.errors, "last_error": w.last_error,
        "consecutive_errors": w.consecutive_errors,
        "last_poll_at": w.last_poll_at,
        "seen_live": w.synth.seen_live,
        "snapshot": w.last_summary,
        "events": dict(w.event_counts),
        "recent": list(w.recent),
        "fed": dict(w.fed_counts),
        "fed_errors": w.fed_errors,
        "last_feed_error": w.last_feed_error,
        "last_success_at": w.last_success_at,
        "recent_fed": list(w.recent_fed),
        "hygiene": {"requests": dict(w.hygiene_counts),
                    "last": w.last_hygiene,
                    "errors": w.hygiene_errors,
                    "last_error": w.last_hygiene_error,
                    # When the backstop next asks, and why it
                    # is waiting longer if it is
                    "periodic": {
                        "interval_s": PERIODIC_RECONCILE_S,
                        "slowed_by_persistent_refusal": w._hygiene_persistent,
                        "anchor_at": w._hygiene_anchor_at,
                        "due_at": (None if w._hygiene_anchor_at is None else
                                   w._hygiene_anchor_at
                                   + (PERIODIC_AFTER_PERSISTENT_REFUSAL_S
                                      if w._hygiene_persistent
                                      else PERIODIC_RECONCILE_S))}},
    }
