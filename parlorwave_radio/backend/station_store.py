#!/usr/bin/env python3
"""Durable persistence for station identity (schema v9).

PURE PERSISTENCE. This module reads and writes two tables and does nothing
else: no Music Assistant call, no queue mutation, no track selection, no
thread, no timer, no policy. It is the storage half of the station design, so
that the behavioural halves that follow can be written against a contract that
is already proven by tests.

WHY A STATION ROW EXISTS AT ALL. Without one, station identity evaporates the
moment it is created: `station_radio.start_from_current` writes 40 URIs to Music
Assistant and keeps nothing -- no seed, no recipe, no artist weights, no list
of what it queued. Every session it leaves behind is `origin=adopted,
queue_owned=0, request=''`. Replenishment is impossible without somewhere to remember what
the station IS.

THREE LIFECYCLE VALUES.

    preparing -> active -> deactivated

`preparing` is a WRITE-AHEAD INTENT record and the reason this module exists
in this shape. SQLite cannot enrol Music Assistant in a transaction, so the
only honest way to survive a crash around an external write is to commit the
intent first and prove the outcome afterwards by reading the queue back. The
OLD station stays `active` for the whole window, so a failed rebuild leaves
the listener exactly where they were -- the same discipline
`station_radio` already follows by retiring handoff sessions only AFTER the
queue mutation commits, and the same one `dj_engine.try_insert` follows by
claiming `ready->queued` BEFORE the enqueue.

There is deliberately no `suspended`: paused, missing player and a large
user-added runway are transient conditions of the world, derived at read time,
not durable facts about the station. And there is deliberately no
`generation`: every explicit Start Radio mints a new `station_id`, so identity
plus the partial unique indexes already fence what a counter would have.

CALLERS: the station path only -- station_radio (which writes the preparing
row and activates it), station_live (the Station Programmer, Mode 2, which
does the same for a station built from a StationIntent), station_recovery
(which proves stranded preparations), and app.py's read-only status surface.
Playback ingestion, the observer and the DJ engine deliberately know nothing
about station state.

TWO KINDS OF STATION, ONE TABLE. A legacy Start Radio station carries its
frozen LLM artist weights; a Station Programmer station (profile_source
PROFILE_PROGRAMMER) carries its StationIntent in `program` and is re-resolved
from it at every refill. Both are bound by the same invariants -- one active,
one preparing, one pending batch per station -- because they program the
same kind of thing: one queue, one autonomous programmer.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime
from typing import Any

import playback

# --- lifecycle vocabularies ---------------------------------------------------
PREPARING = "preparing"
ACTIVE = "active"
DEACTIVATED = "deactivated"
VALID_STATUS = (PREPARING, ACTIVE, DEACTIVATED)

PENDING = "pending"
CONFIRMED = "confirmed"
NOT_LANDED = "not_landed"
ABANDONED = "abandoned"
VALID_BATCH_STATE = (PENDING, CONFIRMED, NOT_LANDED, ABANDONED)

# Bounds. The tail is an OWNERSHIP witness, not play history: the largest
# station-owned future reachable is max(initial Start Radio 40, target runway
# 20 + refill batch 12) = 40, so 64 is headroom that keeps the witness EXACT
# in every reachable state while staying bounded. The rotation ring follows
# news_schedule's bounded-ring precedent (AIRED_TOKEN_CAP).
TAIL_CAP = 64
GENERATED_CAP = 300

#: profile_source of a Station Programmer (Mode 2) station. Refill for such a
#: station belongs to station_live alone; the legacy local-catalog refill
#: refuses it (station_refill) and the legacy eligibility reports it as
#: someone else's (station_eligibility).
PROFILE_PROGRAMMER = "station_programmer"

_STATION_COLS = [
    "station_id", "created_at", "status", "recipe_version", "request_text",
    "seed_artist", "seed_title", "seed_media_key", "profile_source",
    "artist_weights", "queue_id", "intended_uris",
    "pre_current_queue_item_id", "pre_future_music_count",
    "pre_future_music_hash", "prepared_at", "activated_at", "deactivated_at",
    "deactivated_reason", "last_seen_at", "last_replenished_at",
    "refill_count", "tail", "tail_is_partial", "generated",
    "program", "live_status",
]
_BATCH_COLS = [
    "batch_id", "station_id", "queue_id", "state", "intended_uris",
    "pre_items_count", "pre_tail_queue_item_id", "pre_tail_position",
    "pre_qiid_hash", "pre_uri_multiplicity", "claimed_at", "resolved_at",
    "resolution", "attempts",
]
_JSON_STATION = {"artist_weights": dict, "intended_uris": list,
                 "tail": list, "generated": list,
                 "program": dict, "live_status": dict}

#: live_status is a diagnosis, not a history: it is bounded here so no
#: caller can grow it into one.
LIVE_STATUS_MAX_BYTES = 4096
_JSON_BATCH = {"intended_uris": list, "pre_uri_multiplicity": dict}


def _connect(db_path=None) -> sqlite3.Connection:
    """One connection seam, so tests can inject faults exactly as the DJ
    fault-injection suite already does for dj_engine._connect."""
    return playback._connect(db_path)


def _iso(now: datetime | None = None) -> str:
    # The service clock, in the session database's own legacy stamp
    # encoding: the store's tables live in playback's database, and its
    # import contract is closed, so it takes the stamp from playback.
    return playback._iso(now)


def new_id() -> str:
    return secrets.token_hex(16)


def _loads(raw: Any, kind: type) -> Any:
    """JSON column -> Python, never raising. A corrupt cell degrades to the
    empty value of its kind rather than taking a read path down: these columns
    are witnesses and caches, and an unreadable witness means "prove it again",
    never "crash the caller"."""
    try:
        value = json.loads(raw or ("{}" if kind is dict else "[]"))
    except (TypeError, ValueError):
        return kind()
    return value if isinstance(value, kind) else kind()


def _row_to_dict(row, cols, json_cols) -> dict[str, Any]:
    d = dict(zip(cols, row))
    for key, kind in json_cols.items():
        d[key] = _loads(d.get(key), kind)
    return d


def is_programmer(station: dict[str, Any] | None) -> bool:
    """True for a Station Programmer (Mode 2) station row."""
    return bool(station) and station.get("profile_source") == PROFILE_PROGRAMMER


def uri_list_hash(uris: list[str]) -> str:
    """Stable digest of an ORDERED identity list -- used for the pre-write
    witness so a 500-item queue costs one column, not 500 rows. Order is part
    of the identity, so this is deliberately not a set hash."""
    h = hashlib.sha256()
    for u in uris:
        h.update(str(u).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --- reads --------------------------------------------------------------------

def _select_station(c, where: str, args: tuple) -> dict[str, Any] | None:
    row = c.execute(
        f"SELECT {', '.join(_STATION_COLS)} FROM stations WHERE {where} "
        "LIMIT 1", args).fetchone()
    return _row_to_dict(row, _STATION_COLS, _JSON_STATION) if row else None


def get_station(station_id: str, db_path=None) -> dict[str, Any] | None:
    playback.ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        return _select_station(c, "station_id = ?", (station_id,))
    finally:
        c.close()


def get_active_station(db_path=None) -> dict[str, Any] | None:
    """The one active station, or None. Singular by database invariant
    (ux_stations_one_active), not by convention."""
    playback.ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        return _select_station(c, "status = ?", (ACTIVE,))
    finally:
        c.close()


def get_preparing_station(db_path=None) -> dict[str, Any] | None:
    """The one in-flight Start Radio preparation, or None. A row here on
    startup means an external MA write MAY have happened and must be proven
    either way before anything else touches the queue."""
    playback.ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        return _select_station(c, "status = ?", (PREPARING,))
    finally:
        c.close()


def get_batch(batch_id: str, db_path=None) -> dict[str, Any] | None:
    playback.ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_BATCH_COLS)} FROM refill_batches "
            "WHERE batch_id = ?", (batch_id,)).fetchone()
    finally:
        c.close()
    return _row_to_dict(row, _BATCH_COLS, _JSON_BATCH) if row else None


def get_pending_batch(station_id: str, db_path=None) -> dict[str, Any] | None:
    """This station's single unresolved batch, or None. Singular by database
    invariant (ux_refill_one_pending)."""
    playback.ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_BATCH_COLS)} FROM refill_batches "
            "WHERE station_id = ? AND state = ?",
            (station_id, PENDING)).fetchone()
    finally:
        c.close()
    return _row_to_dict(row, _BATCH_COLS, _JSON_BATCH) if row else None


# --- station writes -----------------------------------------------------------

def create_preparing_station(*, request_text: str, artist_weights: dict,
                             seed_artist: str = "", seed_title: str = "",
                             seed_media_key: str = "",
                             profile_source: str = "station",
                             queue_id: str = "", intended_uris: list | None = None,
                             pre_current_queue_item_id: str = "",
                             pre_future_music_count: int = -1,
                             pre_future_music_hash: str = "",
                             recipe_version: int = 1,
                             program: dict | None = None,
                             station_id: str | None = None,
                             now: datetime | None = None,
                             db_path=None) -> dict[str, Any]:
    """Write the intent BEFORE the external queue mutation, and return the row.

    The old active station is deliberately not touched: it stays active and
    truthful until the MA write is PROVEN to have landed. Raises
    sqlite3.IntegrityError if a preparation is already in flight -- the
    partial unique index is the durable half of the single-station invariant,
    and the process-global mutation lock is the live half."""
    playback.ensure_sessions_schema(db_path)
    sid = station_id or new_id()
    stamp = _iso(now)
    c = _connect(db_path)
    try:
        c.execute(
            "INSERT INTO stations (station_id, created_at, status, "
            "recipe_version, request_text, seed_artist, seed_title, "
            "seed_media_key, profile_source, artist_weights, queue_id, "
            "intended_uris, pre_current_queue_item_id, "
            "pre_future_music_count, pre_future_music_hash, prepared_at, "
            "program) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, stamp, PREPARING, int(recipe_version), request_text,
             seed_artist, seed_title, seed_media_key, profile_source,
             json.dumps(artist_weights or {}), queue_id,
             json.dumps(list(intended_uris or [])), pre_current_queue_item_id,
             int(pre_future_music_count), pre_future_music_hash, stamp,
             json.dumps(program or {})))
        c.commit()
    finally:
        c.close()
    return get_station(sid, db_path)


def activate_prepared_station(station_id: str, *, queue_id: str,
                              tail: list | None = None,
                              tail_is_partial: bool = False,
                              now: datetime | None = None,
                              db_path=None) -> bool:
    """THE activation transaction, whole or not at all.

    In ONE SQLite transaction:
        preparing station          -> active (bound, with its queue witness)
        the previous active station -> deactivated(reason='superseded')
        that station's pending batches -> abandoned(reason='superseded')

    Returns True only for the caller whose conditional UPDATE matched, so a
    duplicate invocation or a restart replay is a no-op rather than a second
    activation -- the same exactly-once shape as dj_engine._transition.

    NO MUSIC ASSISTANT CALL MAY HAPPEN WHILE THIS IS OPEN. The caller performs
    the external write first, proves it landed, and only then calls this; the
    transaction is short and purely local by construction.

    Ordering inside the transaction matters: the old row leaves 'active'
    BEFORE the new one enters it, because ux_stations_one_active would
    otherwise reject the intermediate state."""
    stamp = _iso(now)
    c = _connect(db_path)
    try:
        try:
            c.execute("BEGIN IMMEDIATE")
            _supersede_active(c, station_id, stamp)
            cur = c.execute(
                "UPDATE stations SET status = ?, queue_id = ?, "
                "activated_at = ?, last_seen_at = ?, tail = ?, "
                "tail_is_partial = ? WHERE station_id = ? AND status = ?",
                (ACTIVE, queue_id, stamp, stamp,
                 json.dumps(list(tail or [])[:TAIL_CAP]),
                 1 if tail_is_partial else 0, station_id, PREPARING))
            if cur.rowcount != 1:
                c.rollback()
                return False
            c.commit()
            return True
        except Exception:
            c.rollback()
            raise
    finally:
        c.close()


def _supersede_active(c, station_id: str, stamp: str) -> None:
    """Inside an open transaction: every OTHER active station leaves
    'active' (reason 'superseded'), releases its ownership row and abandons
    its pending batches. Runs BEFORE the new row enters 'active', because
    ux_stations_one_active would otherwise reject the intermediate state."""
    prior = c.execute(
        "SELECT station_id FROM stations WHERE status = ? AND "
        "station_id != ?", (ACTIVE, station_id)).fetchall()
    for (old_id,) in prior:
        c.execute(
            "UPDATE stations SET status = ?, deactivated_at = ?, "
            "deactivated_reason = 'superseded' WHERE station_id = ?",
            (DEACTIVATED, stamp, old_id))
        _release_ownership(c, old_id, "superseded", stamp)
        c.execute(
            "UPDATE refill_batches SET state = ?, resolved_at = ?, "
            "resolution = 'superseded' WHERE station_id = ? "
            "AND state = ?", (ABANDONED, stamp, old_id, PENDING))


def activate_prepared_with_batch(station_id: str, batch_id: str, *,
                                 queue_id: str, tail: list,
                                 tail_is_partial: bool, generated: list,
                                 resolution: str = "landed",
                                 now: datetime | None = None,
                                 db_path=None) -> bool:
    """THE Station Programmer activation: a preparation whose opening runway
    was APPENDED (not a replaced future) is proven by its own pending batch,
    so activation and confirmation are one fact and land in ONE transaction:

        preparing station          -> active, with its ownership witness
                                      and its rotation ring
        its opening batch          -> confirmed
        the previous active station -> deactivated('superseded'), its
                                      ownership row released, its pending
                                      batches abandoned

    Whole or not at all. An active station whose opening batch is still
    pending would invite a second append of the same runway; a confirmed
    batch under a station that never activated would be music nobody owns.

    Conditional on the station still PREPARING and the batch still PENDING
    and belonging to it, so a replay (the live request and startup recovery
    racing, or a restart repeating a proof) is a no-op returning False.

    NO NETWORK CALL MAY HAPPEN INSIDE THIS: the caller appended, read the
    queue back and proved the run landed before calling."""
    stamp = _iso(now)
    c = _connect(db_path)
    try:
        try:
            c.execute("BEGIN IMMEDIATE")
            ok = c.execute(
                "SELECT 1 FROM refill_batches WHERE batch_id = ? AND "
                "station_id = ? AND state = ?",
                (batch_id, station_id, PENDING)).fetchone()
            live = c.execute(
                "SELECT 1 FROM stations WHERE station_id = ? AND status = ?",
                (station_id, PREPARING)).fetchone()
            if not ok or not live:
                c.rollback()
                return False
            _supersede_active(c, station_id, stamp)
            c.execute(
                "UPDATE stations SET status = ?, queue_id = ?, "
                "activated_at = ?, last_seen_at = ?, tail = ?, "
                "tail_is_partial = ?, generated = ? "
                "WHERE station_id = ? AND status = ?",
                (ACTIVE, queue_id, stamp, stamp,
                 json.dumps(list(tail or [])[:TAIL_CAP]),
                 1 if tail_is_partial else 0,
                 json.dumps(list(generated or [])[-GENERATED_CAP:]),
                 station_id, PREPARING))
            c.execute(
                "UPDATE refill_batches SET state = ?, resolved_at = ?, "
                "resolution = ? WHERE batch_id = ? AND state = ?",
                (CONFIRMED, stamp, resolution, batch_id, PENDING))
            c.commit()
            return True
        except Exception:
            c.rollback()
            raise
    finally:
        c.close()


def update_live_status(station_id: str, patch: dict[str, Any],
                       db_path=None) -> bool:
    """Merge `patch` into a station's live_status diagnosis (top-level keys
    replaced). Conditional on the row not being deactivated, so a late
    diagnosis can never rewrite a stopped station's last words. Bounded:
    a patch that would grow the document past LIVE_STATUS_MAX_BYTES is
    dropped rather than truncated into something unreadable."""
    c = _connect(db_path)
    try:
        row = c.execute(
            "SELECT live_status FROM stations WHERE station_id = ? AND "
            "status IN (?, ?)", (station_id, PREPARING, ACTIVE)).fetchone()
        if row is None:
            return False
        doc = _loads(row[0], dict)
        doc.update(patch or {})
        text = json.dumps(doc, sort_keys=True)
        if len(text) > LIVE_STATUS_MAX_BYTES:
            return False
        cur = c.execute(
            "UPDATE stations SET live_status = ? WHERE station_id = ? AND "
            "status IN (?, ?)", (text, station_id, PREPARING, ACTIVE))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


def set_autoplay_restore(station_id: str, record: dict[str, Any],
                         db_path=None) -> bool:
    """Write ONE key of a station's live_status -- `autoplay_restore`, the
    house-handover debt to give a queue back to MA Autoplay -- whatever the
    station's status. The single exception to "a stopped station's last
    words are final": the debt is born at Stop and settled after it, so it
    must outlive deactivation. Every other key is untouched; bounded like
    update_live_status."""
    c = _connect(db_path)
    try:
        row = c.execute("SELECT live_status FROM stations WHERE station_id = ?",
                        (station_id,)).fetchone()
        if row is None:
            return False
        doc = _loads(row[0], dict)
        doc["autoplay_restore"] = dict(record or {})
        text = json.dumps(doc, sort_keys=True)
        if len(text) > LIVE_STATUS_MAX_BYTES:
            return False
        cur = c.execute("UPDATE stations SET live_status = ? WHERE "
                        "station_id = ?", (text, station_id))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


def last_programmer_station(db_path=None) -> dict[str, Any] | None:
    """The most recently started Station Programmer station in ANY state --
    what a status surface shows as "last station" once none is active."""
    playback.ensure_sessions_schema(db_path)
    c = _connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_STATION_COLS)} FROM stations "
            "WHERE profile_source = ? ORDER BY created_at DESC, rowid DESC "
            "LIMIT 1", (PROFILE_PROGRAMMER,)).fetchone()
    finally:
        c.close()
    return _row_to_dict(row, _STATION_COLS, _JSON_STATION) if row else None


def fail_prepared_station(station_id: str, reason: str = "prepare_failed",
                          now: datetime | None = None, db_path=None) -> bool:
    """The failure side: the preparation dies, the OLD ACTIVE STATION IS NOT
    TOUCHED. Music kept playing, so the listener keeps the station they had --
    exactly what start_from_current already guarantees when replace_future
    fails."""
    return _set_terminal(station_id, (PREPARING,), reason, now, db_path)


def deactivate_station(station_id: str, reason: str,
                       now: datetime | None = None, db_path=None) -> bool:
    """Terminal, from either live state. Conditional, so a concurrent or
    replayed call cannot double-transition."""
    return _set_terminal(station_id, (PREPARING, ACTIVE), reason, now, db_path)


def _set_terminal(station_id: str, from_states: tuple, reason: str,
                  now: datetime | None, db_path) -> bool:
    stamp = _iso(now)
    marks = ",".join("?" for _ in from_states)
    c = _connect(db_path)
    try:
        cur = c.execute(
            f"UPDATE stations SET status = ?, deactivated_at = ?, "
            f"deactivated_reason = ? WHERE station_id = ? "
            f"AND status IN ({marks})",
            (DEACTIVATED, stamp, reason, station_id, *from_states))
        if cur.rowcount == 1:
            _release_ownership(c, station_id, reason, stamp)
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


def _release_ownership(c, station_id: str, reason: str, stamp: str) -> None:
    """A station leaving the live states releases its queue-ownership row in
    the SAME transaction. Persistence only: Music Assistant's Autoplay flag is
    deliberately NOT restored (queue_ownership's restore policy is "none"),
    so this touches one local row and nothing else."""
    c.execute(
        "UPDATE queue_ownership SET state = 'released', released_at = ?, "
        "released_reason = ?, updated_at = ? WHERE station_id = ? AND "
        "state != 'released'", (stamp, reason, stamp, station_id))


def update_station_witness(station_id: str, *, tail: list | None = None,
                           tail_is_partial: bool | None = None,
                           generated: list | None = None,
                           last_seen_at: datetime | None = None,
                           last_replenished_at: datetime | None = None,
                           bump_refill_count: bool = False,
                           db_path=None) -> bool:
    """Refresh the ownership witness and caches on an ACTIVE station.

    Conditional on status='active' so nothing can quietly resurrect a
    deactivated station by writing to it. Both JSON collections are capped
    here rather than by the caller, so the bound is a property of storage and
    cannot be forgotten at a call site."""
    sets, args = [], []
    if tail is not None:
        sets.append("tail = ?")
        args.append(json.dumps(list(tail)[:TAIL_CAP]))
    if tail_is_partial is not None:
        sets.append("tail_is_partial = ?")
        args.append(1 if tail_is_partial else 0)
    if generated is not None:
        sets.append("generated = ?")
        args.append(json.dumps(list(generated)[-GENERATED_CAP:]))
    if last_seen_at is not None:
        sets.append("last_seen_at = ?")
        args.append(_iso(last_seen_at))
    if last_replenished_at is not None:
        sets.append("last_replenished_at = ?")
        args.append(_iso(last_replenished_at))
    if bump_refill_count:
        sets.append("refill_count = refill_count + 1")
    if not sets:
        return False
    c = _connect(db_path)
    try:
        cur = c.execute(
            f"UPDATE stations SET {', '.join(sets)} "
            "WHERE station_id = ? AND status = ?",
            (*args, station_id, ACTIVE))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


# --- refill batch writes ------------------------------------------------------

def create_pending_batch(*, station_id: str, queue_id: str,
                         intended_uris: list, pre_items_count: int,
                         pre_tail_queue_item_id: str = "",
                         pre_tail_position: int = -1,
                         pre_qiid_hash: str = "",
                         pre_uri_multiplicity: dict | None = None,
                         batch_id: str | None = None,
                         now: datetime | None = None,
                         db_path=None) -> dict[str, Any]:
    """Claim a refill BEFORE the append, carrying the pre-write witness.

    player_queues/play_media returns HTTP 200 with body `null`:
    no inserted ids, no echo. So the ONLY way to answer "did my append land?"
    after a lost response is to compare the live queue against what was true
    just before the write -- which has to have been written down first.

    Raises sqlite3.IntegrityError if this station already has a pending batch
    (ux_refill_one_pending)."""
    playback.ensure_sessions_schema(db_path)
    bid = batch_id or new_id()
    c = _connect(db_path)
    try:
        c.execute(
            "INSERT INTO refill_batches (batch_id, station_id, queue_id, "
            "state, intended_uris, pre_items_count, pre_tail_queue_item_id, "
            "pre_tail_position, pre_qiid_hash, pre_uri_multiplicity, "
            "claimed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (bid, station_id, queue_id, PENDING,
             json.dumps(list(intended_uris)), int(pre_items_count),
             pre_tail_queue_item_id, int(pre_tail_position), pre_qiid_hash,
             json.dumps(pre_uri_multiplicity or {}), _iso(now)))
        c.commit()
    finally:
        c.close()
    return get_batch(bid, db_path)


def resolve_batch(batch_id: str, state: str, resolution: str = "",
                  now: datetime | None = None, db_path=None) -> bool:
    """pending -> confirmed | not_landed | abandoned. State-conditional, so a
    batch can only ever be resolved ONCE.

    NOTHING IN THIS MODULE RESOLVES A BATCH BY TIME. Reaching an age is not
    evidence about an append whose response was lost; only reading Music
    Assistant is. A timer may later TRIGGER that read -- it may never stand in
    for it."""
    if state not in (CONFIRMED, NOT_LANDED, ABANDONED):
        raise ValueError(f"not a terminal batch state: {state!r}")
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE refill_batches SET state = ?, resolved_at = ?, "
            "resolution = ? WHERE batch_id = ? AND state = ?",
            (state, _iso(now), resolution, batch_id, PENDING))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


def note_batch_attempt(batch_id: str, db_path=None) -> bool:
    """Count one reconciliation attempt. Observability for a bounded retry
    policy that lives with its caller; it decides nothing here."""
    c = _connect(db_path)
    try:
        cur = c.execute(
            "UPDATE refill_batches SET attempts = attempts + 1 "
            "WHERE batch_id = ? AND state = ?", (batch_id, PENDING))
        c.commit()
        return cur.rowcount == 1
    finally:
        c.close()


def status_snapshot(db_path=None) -> dict[str, Any]:
    """Compact READ-ONLY durable-station state for a status surface.

    Deliberately small: identity, lifecycle, binding, seed, and the SIZE and
    EXACTNESS of the ownership witness -- never the witness itself, never the
    frozen weights, never the intended URI list. Those are recovery evidence
    measured in hundreds of entries; putting them in a dashboard payload would
    make every poll carry a queue's worth of JSON to say one sentence.

    `witness_exact` is the honest half: when it is false, `witness_count` is a
    LOWER BOUND on station-owned runway, not a count."""
    out: dict[str, Any] = {"active": None, "preparing": None}
    try:
        for key, row in (("active", get_active_station(db_path)),
                         ("preparing", get_preparing_station(db_path))):
            if row is None:
                continue
            if is_programmer(row):
                # A Station Programmer station plays on ITS OWN speaker. This
                # snapshot is the house's station identity -- the HA Station
                # sensor and the Station tab's "Your station" read it -- and a
                # station on an isolated speaker is not the house's station.
                # Reported beside, never as. (station_live.status() is its
                # full surface.)
                label = str(((row.get("program") or {}).get("intent") or {})
                            .get("label") or "")
                out.setdefault("station_programmer", {})[key] = {
                    "station_id": row["station_id"], "label": label,
                    "queue_id": row["queue_id"]}
                continue
            out[key] = {
                "station_id": row["station_id"],
                "status": row["status"],
                "queue_id": row["queue_id"],
                "seed": (f"{row['seed_artist']} - {row['seed_title']}"
                         if row["seed_artist"] else ""),
                "profile_source": row["profile_source"],
                "created_at": row["created_at"],
                "activated_at": row["activated_at"],
                "witness_count": len(row["tail"]),
                "witness_exact": not row["tail_is_partial"],
                "generated_count": len(row["generated"]),
                "refill_count": row["refill_count"],
                "last_replenished_at": row["last_replenished_at"],
            }
        active = out.get("active")
        if active:
            # The unresolved claim, if any. Compact on purpose: counts and
            # state, never the intended URI list -- a status poll must not
            # carry a queue's worth of JSON to say one sentence.
            batch = get_pending_batch(active["station_id"], db_path)
            active["pending_refill"] = None if batch is None else {
                "batch_id": batch["batch_id"], "state": batch["state"],
                "intended_count": len(batch["intended_uris"]),
                "claimed_at": batch["claimed_at"],
                "attempts": batch["attempts"]}
    except Exception:
        # Status must never be the thing that breaks: an unreadable station
        # table reports nothing rather than failing the dashboard.
        return {"active": None, "preparing": None, "error": "unreadable"}
    return out


def confirm_batch_and_witness(batch_id: str, station_id: str, *,
                              tail: list, tail_is_partial: bool,
                              generated: list, resolution: str = "landed",
                              now: datetime | None = None,
                              db_path=None) -> bool:
    """Land a proven refill in ONE transaction: batch confirmed, ownership
    witness refreshed, rotation ring extended, counters stamped.

    Whole or not at all, because the halves are only true together -- a
    confirmed batch whose tracks are missing from the witness would look
    like unowned music, and a witness carrying tracks whose batch is still
    pending would invite a second append of the same songs.

    Conditional on the batch still being pending AND the station still being
    active, so a replay, a concurrent confirmation or a station superseded in
    the meantime is a no-op returning False rather than a second write.

    NO NETWORK CALL MAY HAPPEN INSIDE THIS. The caller performs the append,
    proves it landed by reading the queue back, and only then calls here; the
    transaction is short and purely local by construction."""
    stamp = _iso(now)
    c = _connect(db_path)
    try:
        try:
            c.execute("BEGIN IMMEDIATE")
            cur = c.execute(
                "UPDATE refill_batches SET state = ?, resolved_at = ?, "
                "resolution = ? WHERE batch_id = ? AND state = ?",
                (CONFIRMED, stamp, resolution, batch_id, PENDING))
            if cur.rowcount != 1:
                c.rollback()
                return False
            cur = c.execute(
                "UPDATE stations SET tail = ?, tail_is_partial = ?, "
                "generated = ?, last_replenished_at = ?, "
                "refill_count = refill_count + 1 "
                "WHERE station_id = ? AND status = ?",
                (json.dumps(list(tail)[:TAIL_CAP]),
                 1 if tail_is_partial else 0,
                 json.dumps(list(generated)[-GENERATED_CAP:]),
                 stamp, station_id, ACTIVE))
            if cur.rowcount != 1:
                c.rollback()
                return False
            c.commit()
            return True
        except Exception:
            c.rollback()
            raise
    finally:
        c.close()
