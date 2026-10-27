#!/usr/bin/env python3
"""Queue ownership: who is allowed to PROGRAM the music on a queue.

Music Assistant owns playback and queue transport. ParlorWave owns station
programming. Those are different jobs, and this module exists because they
collide: Music Assistant's own Autoplay ("Don't stop the music") can append
a batch of similar-track picks to a ParlorWave station's queue, seeded from
a DJ-break item (older queues still carry the legacy title "Jarvis DJ
Break"), while Endless Radio is also refilling it. Two autonomous
programmers, one queue.

THE RULE. When a ParlorWave station takes ownership of a queue, Music
Assistant's per-queue Autoplay flag is turned OFF for that queue -- and
only that queue. From then on exactly ONE autonomous programmer appends
music there: Endless Radio. That is "one autonomous programmer", not "one
entity allowed to touch the queue": the listener may still add, remove or
reorder anything by hand, and DJ breaks / news bulletins are program
ELEMENTS inserted into the station, not competing programmers.

WHAT OWNERSHIP IS. An `active` station row bound to a queue_id. That is the
only thing that ever creates one. A playback session adopted from a queue the
listener built themselves (DJ-only adoption: the DJ speaks between songs but
did not program them) creates NO ownership -- DJ-enabled is not the same as
"ParlorWave owns music refill", and MA Autoplay is left exactly as the listener
set it. Only Start Radio (station_radio.start_from_current) makes a station.

WHAT IS NEVER TOUCHED. MA's global Autoplay settings (`autoplay_mode`,
`autoplay_playlist`) and every other queue's flag. There is no code path here
that can write anything but `player_queues/autoplay {queue_id, false}` for
the owned queue.

RESTORE POLICY: NONE -- with ONE confirmed exception. When a station ends,
DJ is switched off, Endless Radio is switched off or Music Brain restarts,
the flag stays where it is. A restore would need a contract -- "what the
listener wanted" is not knowable from "what MA said when the station arrived",
which may itself have been the residue of an earlier station. Silence at the
end of a queue with Endless Radio off is the listener's explicit choice; the
flag they can flip in MA's own controls is documented, not fought.

The exception IS that contract: a Station Programmer HOUSE HANDOVER
(station_live). The person was shown that Autoplay was ON for the house
speaker and that Stop would give the queue back to it, and confirmed. Only
then, only at that station's Stop, only for that one queue, and only when
the flag reads OFF, `restore_after_handover` turns it back on and proves it.
Nothing else in the backend ever writes the flag on.

DRIFT. If MA (or a person, through MA's controls) turns the flag back on
while ParlorWave owns the queue, it turns it off again -- observed at the
next refill evaluation from the same `player_queues/all` snapshot the
eligibility engine already reads, so drift detection costs zero extra MA
calls. It is BOUNDED: after DRIFT_BUDGET corrections inside DRIFT_WINDOW_S
the queue is marked `contested` and ParlorWave stops correcting AND stops
autonomous refill for that station, because a config fight is worse than
either programmer winning. Contested clears the moment the flag is observed
off again, or when the listener explicitly asks ParlorWave to replenish.

The module is deliberately in two halves, like station_eligibility:

    PURE policy   plan(record, observed, now)  -> Plan   (table-testable)
    I/O           acquire / enforce / reassert (SQLite row + at most one
                                                MA write and one readback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import playback
import service_clock

logger = logging.getLogger("musicbrain")

# --- states -------------------------------------------------------------------
UNVERIFIED = "unverified"   # station active; MA flag not yet proven off
OWNED = "owned"             # MA flag proven off for this queue
CONTESTED = "contested"     # drift budget exhausted; ParlorWave yielded
RELEASED = "released"       # station ended; flag left as it was (no restore).
                            # Written by the station store inside the SAME
                            # transaction that deactivates the station.

# --- drift budget. Named, not magic --------------------------------------------
#
# Three corrections an hour is far more than any accident produces (MA never
# re-enables the flag on its own; a transfer copies it, a person flips it) and
# far fewer than a fight. Past it, the honest reading is "someone wants MA
# Autoplay on this queue", and ParlorWave yields rather than argues.
DRIFT_BUDGET = 3
DRIFT_WINDOW_S = 3600.0

# --- machine-readable outcomes -------------------------------------------------
ACT_NONE = ""                       # nothing to do
ACT_DISABLE = "disable"             # first acquisition: turn the flag off
ACT_CORRECT = "correct_drift"       # flag came back on: turn it off again
ACT_YIELD = "yield"                 # budget exhausted: mark contested
ACT_RECOVER = "recover"             # contested, but flag observed off again

# Log event names (stable; grep targets for the live proof).
EV_DISABLED = "ma_autoplay_disabled_for_owned_queue"
EV_ALREADY_OFF = "ma_autoplay_already_off_for_owned_queue"
EV_DRIFT = "ma_autoplay_drift_corrected"
EV_CONTESTED = "ma_autoplay_contested_yielding"
EV_RECOVERED = "ma_autoplay_contest_resolved"
EV_UNVERIFIED = "ma_autoplay_disable_unverified"


@dataclass(frozen=True)
class Plan:
    """What the policy wants done, given what MA was observed to say."""
    action: str
    next_state: str
    drift_corrections: int
    drift_window_started_at: str | None
    refill_allowed: bool


# --- PURE policy --------------------------------------------------------------

def _seconds_between(start: str | None, now: datetime) -> float | None:
    if not start:
        return None
    bounds = service_clock.age_bounds(start, now)
    return None if bounds is None else bounds[1]


def plan(record: dict[str, Any] | None, observed: bool | None,
         now: datetime, budget: int = DRIFT_BUDGET) -> Plan:
    """Decide from a snapshot. No I/O.

    `record` is the queue's ownership row (None = never acquired), `observed`
    is MA's flag as just read (None = could not be read). The plan never
    proposes ENABLING the flag: there is no action in this vocabulary that
    turns Autoplay on.

    `budget` is the drift corrections allowed per window. A station that
    took over a house speaker by HANDOVER passes 0: there, Autoplay turned
    back on is the person's newer decision, and ParlorWave yields at once
    rather than switching it off again."""
    state = (record or {}).get("state") or UNVERIFIED
    count = int((record or {}).get("drift_corrections") or 0)
    window = (record or {}).get("drift_window_started_at")
    stamp = now.replace(microsecond=0).isoformat()

    if observed is None:
        # Unreadable flag: nothing is written on a guess. An owned queue
        # stays owned (refill continues -- music continuity wins); an
        # unverified one stays unverified and is retried next time.
        keep = state if state in (OWNED, CONTESTED) else UNVERIFIED
        return Plan(ACT_NONE, keep, count, window,
                    refill_allowed=keep != CONTESTED)

    if observed is False:
        if state == CONTESTED:
            # Whoever wanted it on has turned it off (or MA did). The
            # contest is over; the budget starts fresh.
            return Plan(ACT_RECOVER, OWNED, 0, None, refill_allowed=True)
        return Plan(ACT_NONE, OWNED, count, window, refill_allowed=True)

    # observed is True: MA Autoplay is on for a queue ParlorWave programs.
    if state == CONTESTED:
        return Plan(ACT_NONE, CONTESTED, count, window, refill_allowed=False)
    if state == UNVERIFIED or record is None:
        # First acquisition (or a retry of one that never verified): this is
        # not drift, it is the take-over itself, and it costs no budget.
        return Plan(ACT_DISABLE, OWNED, count, window, refill_allowed=True)
    # state == OWNED and the flag came back: drift.
    age = _seconds_between(window, now)
    if age is None or age > DRIFT_WINDOW_S:
        count, window = 0, stamp
    if count >= budget:
        return Plan(ACT_YIELD, CONTESTED, count, window, refill_allowed=False)
    return Plan(ACT_CORRECT, OWNED, count + 1, window, refill_allowed=True)


# --- persistence --------------------------------------------------------------

_COLS = ("queue_id", "station_id", "state", "autoplay_before",
         "autoplay_verified_at", "disable_writes", "drift_corrections",
         "drift_window_started_at", "last_checked_at", "last_error",
         "acquired_at", "updated_at", "released_at", "released_reason")


def _iso(now: datetime | None = None) -> str:
    return (now or service_clock.legacy_now()).replace(microsecond=0).isoformat()


def get_record(queue_id: str, db_path=None) -> dict[str, Any] | None:
    if not queue_id:
        return None
    c = playback._connect(db_path)
    try:
        row = c.execute(
            f"SELECT {', '.join(_COLS)} FROM queue_ownership WHERE queue_id = ?",
            (queue_id,)).fetchone()
    finally:
        c.close()
    if row is None:
        return None
    rec = dict(zip(_COLS, row))
    if rec["autoplay_before"] is not None:
        rec["autoplay_before"] = bool(rec["autoplay_before"])
    return rec


def _upsert(queue_id: str, station_id: str, *, state: str, now: datetime,
            db_path=None, fresh: bool = False, **fields: Any) -> None:
    """Write the row. `fresh` re-starts the row for a new station (counters
    reset); otherwise only the named fields change."""
    stamp = _iso(now)
    c = playback._connect(db_path)
    try:
        existing = c.execute(
            "SELECT station_id FROM queue_ownership WHERE queue_id = ?",
            (queue_id,)).fetchone()
        if existing is None or fresh or existing[0] != station_id:
            c.execute(
                "INSERT OR REPLACE INTO queue_ownership (queue_id, station_id, "
                "state, autoplay_before, autoplay_verified_at, disable_writes, "
                "drift_corrections, drift_window_started_at, last_checked_at, "
                "last_error, acquired_at, updated_at, released_at, "
                "released_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "NULL, '')",
                (queue_id, station_id, state,
                 _int_or_none(fields.get("autoplay_before")),
                 fields.get("autoplay_verified_at"),
                 int(fields.get("disable_writes") or 0),
                 int(fields.get("drift_corrections") or 0),
                 fields.get("drift_window_started_at"),
                 fields.get("last_checked_at") or stamp,
                 str(fields.get("last_error") or ""), stamp, stamp))
        else:
            sets = ["state = ?", "updated_at = ?"]
            args: list[Any] = [state, stamp]
            for key, value in fields.items():
                if key == "autoplay_before":
                    value = _int_or_none(value)
                sets.append(f"{key} = ?")
                args.append(value)
            args.append(queue_id)
            c.execute(f"UPDATE queue_ownership SET {', '.join(sets)} "
                      "WHERE queue_id = ?", args)
        c.commit()
    finally:
        c.close()


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


# --- I/O: acquisition and enforcement ------------------------------------------

def _read(queue_id: str, read_fn) -> tuple[bool | None, str]:
    """One MA read. (flag, error) -- the flag is None when it could not be
    read, and the error text says why."""
    try:
        return read_fn(queue_id), ""
    except Exception as exc:  # noqa: BLE001 - single read boundary
        return None, str(exc)[:200]


def _disable(queue_id: str, write_fn, read_fn) -> tuple[bool, str]:
    """Turn the flag off and PROVE it by reading back. (verified, error).
    No retry: an unverified write is reported as unverified and revisited at
    the next evaluation; a second blind write would be a guess."""
    try:
        write_fn(queue_id, False)
    except Exception as exc:  # noqa: BLE001 - single write boundary
        return False, str(exc)[:200]
    observed, err = _read(queue_id, read_fn)
    if observed is False:
        return True, ""
    return False, err or f"readback reported autoplay_enabled={observed}"


def apply(*, queue_id: str, station_id: str, observed: bool | None,
          observed_error: str = "", now: datetime | None = None,
          db_path=None, read_fn=None, write_fn=None, fresh: bool = False,
          reset: bool = False, source: str = "refill",
          budget: int = DRIFT_BUDGET) -> dict[str, Any]:
    """Run the policy against ONE observation and act on it.

    Returns {state, action, verified, refill_allowed, drift_corrections,
    error}. `fresh=True` is the acquisition call from Start Radio: the row is
    (re)started for `station_id` and MA's pre-ownership reading recorded.
    `observed` may come from a `player_queues/all` snapshot the caller
    already holds (zero-cost drift check) -- only a decision to WRITE costs
    the verification read."""
    now_dt = now or service_clock.legacy_now()
    stamp = _iso(now_dt)
    record = None if fresh else get_record(queue_id, db_path=db_path)
    if record is not None and record.get("station_id") != station_id:
        record = None            # a different station's row: start over
    if record is not None and record.get("state") == RELEASED:
        record = None
    stored = record              # exactly what the row says right now
    if record is None:
        fresh = True
    elif reset:
        # A person asked ParlorWave to program this queue: any contest is
        # settled in ParlorWave's favour and the drift budget starts over. The
        # pre-ownership observation is kept -- it is history, not state.
        record = {**record, "state": UNVERIFIED, "drift_corrections": 0,
                  "drift_window_started_at": None}
    decision = plan(record, observed, now_dt, budget)

    fields: dict[str, Any] = {"last_checked_at": stamp,
                              "drift_corrections": decision.drift_corrections,
                              "drift_window_started_at":
                                  decision.drift_window_started_at}
    if fresh:
        fields["autoplay_before"] = observed
        fields["disable_writes"] = 0
    out = {"queue_id": queue_id, "station_id": station_id,
           "action": decision.action, "state": decision.next_state,
           "verified": False, "refill_allowed": decision.refill_allowed,
           "drift_corrections": decision.drift_corrections, "error": ""}
    short = station_id[:8]

    if decision.action in (ACT_DISABLE, ACT_CORRECT):
        import ma_queue
        verified, err = _disable(queue_id,
                                 write_fn or ma_queue.set_queue_autoplay,
                                 read_fn or ma_queue.queue_autoplay)
        writes = int((record or {}).get("disable_writes") or 0) + 1
        fields["disable_writes"] = writes
        out["verified"] = verified
        if verified:
            fields["autoplay_verified_at"] = stamp
            fields["last_error"] = ""
            out["state"] = OWNED
            if decision.action == ACT_DISABLE:
                logger.info("%s station=%s queue=%s (was on; %s)",
                            EV_DISABLED, short, queue_id, source)
            else:
                logger.warning("%s station=%s queue=%s correction %d/%d in "
                               "window (%s)", EV_DRIFT, short, queue_id,
                               decision.drift_corrections, DRIFT_BUDGET,
                               source)
        else:
            fields["last_error"] = err
            out["error"] = err
            # An acquisition that did not verify stays UNVERIFIED and is
            # retried at the next evaluation; a failed drift correction
            # leaves the queue OWNED (the budget was spent) so the retry
            # is itself budgeted. Neither blocks refill: music continuity
            # outranks the purity of the one-writer proof.
            out["state"] = (UNVERIFIED if decision.action == ACT_DISABLE
                            else OWNED)
            logger.warning("%s station=%s queue=%s: %s", EV_UNVERIFIED,
                           short, queue_id, err)
    elif decision.action == ACT_YIELD:
        fields["last_error"] = ""
        logger.warning("%s station=%s queue=%s: MA Autoplay re-enabled %d "
                       "times within %ds; Parlorwave stops correcting and "
                       "pauses autonomous refill until the flag is observed "
                       "off",
                       EV_CONTESTED, short, queue_id, DRIFT_BUDGET,
                       int(DRIFT_WINDOW_S))
    elif decision.action == ACT_RECOVER:
        fields["last_error"] = ""
        fields["autoplay_verified_at"] = stamp
        logger.info("%s station=%s queue=%s: flag observed off again",
                    EV_RECOVERED, short, queue_id)
    else:
        if observed is False:
            fields["autoplay_verified_at"] = stamp
            fields["last_error"] = ""
            out["verified"] = True
            if fresh:
                logger.info("%s station=%s queue=%s (already off; %s)",
                            EV_ALREADY_OFF, short, queue_id, source)
        elif observed is None and observed_error:
            fields["last_error"] = observed_error
            out["error"] = observed_error

    unchanged = (stored is not None and decision.action == ACT_NONE
                 and out["state"] == stored.get("state")
                 and decision.drift_corrections == int(
                     stored.get("drift_corrections") or 0))
    if (unchanged and not out["error"]) or (observed is None and not reset):
        # The quiet steady state (the flag is still off and nothing about
        # the record changes), or an observation that taught nothing (the
        # flag could not be read). Neither is written: a refill evaluation
        # must not cost a database transaction just to say "still fine",
        # and a row is never created from a guess.
        return out
    try:
        _upsert(queue_id, station_id, state=out["state"], now=now_dt,
                db_path=db_path, fresh=fresh, **fields)
    except Exception:
        # Persistence is the audit trail, never the safety mechanism: the
        # MA flag is already what it is. Report, do not raise.
        logger.warning("queue_ownership row not persisted (queue=%s)",
                       queue_id, exc_info=True)
    return out


def acquire(*, queue_id: str, station_id: str, observed: bool | None,
            observed_error: str = "", now: datetime | None = None,
            db_path=None, read_fn=None, write_fn=None,
            source: str = "start_radio") -> dict[str, Any]:
    """Start Radio: ParlorWave has just become the programmer of `queue_id`.

    `observed` is MA's flag from the queue snapshot the caller already read
    (None = unreadable, recorded as such). At most one write (turn the flag
    off) and one verification read follow. Called INSIDE the station
    mutation lock, after the queue rewrite landed and the station row is
    active, so ownership and the flag change are as close to atomic as an
    external system allows. Never raises: a failure here is an unverified
    ownership, not a failed station."""
    return apply(queue_id=queue_id, station_id=station_id, observed=observed,
                 observed_error=observed_error, now=now, db_path=db_path,
                 read_fn=read_fn, write_fn=write_fn, fresh=True,
                 source=source)


def enforce(*, station: dict[str, Any], observed: bool | None,
            now: datetime | None = None, db_path=None, read_fn=None,
            write_fn=None, source: str = "refill") -> dict[str, Any]:
    """Refill-time re-check for the ACTIVE station, from a flag the caller
    already observed. Acquires lazily when no row exists (a station started
    before ownership rows existed, or one whose Start Radio acquisition never
    wrote a row), corrects drift within budget, yields when it is exhausted."""
    queue_id = str(station.get("queue_id") or "")
    station_id = str(station.get("station_id") or "")
    if not queue_id or not station_id:
        return {"queue_id": queue_id, "station_id": station_id,
                "action": ACT_NONE, "state": UNVERIFIED, "verified": False,
                "refill_allowed": True, "drift_corrections": 0,
                "error": "station_unbound"}
    # A house HANDOVER station never corrects drift: the person turning
    # Autoplay back on there is a newer decision ParlorWave must not undo.
    handed = bool((station.get("program") or {}).get("handover"))
    return apply(queue_id=queue_id, station_id=station_id, observed=observed,
                 now=now, db_path=db_path, read_fn=read_fn,
                 write_fn=write_fn, source=source,
                 budget=0 if handed else DRIFT_BUDGET)


def reassert(*, station: dict[str, Any], observed: bool | None,
             now: datetime | None = None, db_path=None, read_fn=None,
             write_fn=None) -> dict[str, Any]:
    """An explicit, person-issued replenish: the listener has asked ParlorWave
    to program this queue, which settles any contest in its favour. The
    drift budget is reset and the flag re-acquired from the caller's
    observation; the original pre-ownership observation is kept."""
    queue_id = str(station.get("queue_id") or "")
    station_id = str(station.get("station_id") or "")
    if not queue_id or not station_id:
        return {"queue_id": queue_id, "station_id": station_id,
                "action": ACT_NONE, "state": UNVERIFIED, "verified": False,
                "refill_allowed": True, "drift_corrections": 0,
                "error": "station_unbound"}
    return apply(queue_id=queue_id, station_id=station_id, observed=observed,
                 now=now, db_path=db_path, read_fn=read_fn,
                 write_fn=write_fn, reset=True, source="manual_replenish")


# --- status --------------------------------------------------------------------

def status_for(station: dict[str, Any] | None, db_path=None
               ) -> dict[str, Any] | None:
    """Compact read-only ownership state for the status surface: the row for
    the active station's queue, or None."""
    if not station:
        return None
    try:
        rec = get_record(str(station.get("queue_id") or ""), db_path=db_path)
    except Exception:
        return None
    if rec is None or rec.get("station_id") != station.get("station_id"):
        return {"state": UNVERIFIED, "ma_autoplay_before": None,
                "verified_at": None, "drift_corrections": 0,
                "last_error": ""}
    return {"state": rec["state"], "ma_autoplay_before": rec["autoplay_before"],
            "verified_at": rec["autoplay_verified_at"],
            "drift_corrections": rec["drift_corrections"],
            "last_error": rec["last_error"]}


def restore_after_handover(*, queue_id: str, read_fn=None,
                           write_fn=None) -> dict[str, Any]:
    """Give ONE queue back to Music Assistant's Autoplay after a confirmed
    house handover's Stop (module docstring: the one restore). Called by
    station_live.stop_station INSIDE the station mutation lock, after the
    station is deactivated -- so ParlorWave has stopped refilling before
    Autoplay may start, and no other start can slip in between.

    At most one write (flag on) and one verification read. Written only when
    the flag is READ as off: on already, or unreadable, writes nothing.
    Never raises: a failure is reported, and the person is told to switch
    Autoplay on in Music Assistant."""
    import ma_queue
    read_fn = read_fn or ma_queue.queue_autoplay
    write_fn = write_fn or ma_queue.set_queue_autoplay
    observed, err = _read(queue_id, read_fn)
    if observed is True:
        return {"state": "already_on", "restored": True, "error": ""}
    if observed is None:
        logger.warning("ma_autoplay_restore_skipped queue=%s: flag unreadable "
                       "(%s)", queue_id, err)
        return {"state": "unreadable", "restored": False, "error": err}
    try:
        write_fn(queue_id, True)
    except Exception as exc:  # noqa: BLE001 - single write boundary
        logger.warning("ma_autoplay_restore_failed queue=%s: %s", queue_id,
                       str(exc)[:200])
        return {"state": "failed", "restored": False, "error": str(exc)[:200]}
    after, err = _read(queue_id, read_fn)
    if after is True:
        logger.info("ma_autoplay_restored_after_handover queue=%s", queue_id)
        return {"state": "restored", "restored": True, "error": ""}
    logger.warning("ma_autoplay_restore_unverified queue=%s: readback %s %s",
                   queue_id, after, err)
    return {"state": "unverified", "restored": False,
            "error": err or f"readback reported autoplay_enabled={after}"}
