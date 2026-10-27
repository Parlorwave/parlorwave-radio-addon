#!/usr/bin/env python3
"""Should an automatic refill be attempted right now? READ ONLY.

This module answers a question. It never appends, never claims, never selects
and never resolves a URI -- and nothing calls it automatically yet. It exists
so the policy that will one day drive automatic replenishment can be evaluated
against a running install, through a diagnostic endpoint, before any trigger is
wired to it.

THE MODULE IS DELIBERATELY IN TWO HALVES.

    gather_world(...)   touches settings, the station tables and Music
                        Assistant, and produces a plain snapshot
    decide(world)       is PURE -- a frozen snapshot in, a frozen decision out

Splitting them is what makes the thresholds testable as arithmetic instead of
as an integration test. Every boundary in `decide` is a table test; every
failure in `gather_world` is a truthful blocking reason.

TWO RUNWAYS, TWO QUESTIONS. They really do diverge: a station can own NOTHING
of the future while plenty of listener-chosen tracks are queued ahead.

    owned_future     items the durable station still owns, by witness identity
                     -> answers "does the STATION need more music?"
    playable_future  every ordinary playable track ahead, whoever queued it
                     -> answers "is the MUSIC about to run out?"

Confusing them breaks the feature in one of two ways. Gating replenishment on
playable_future means a station behind a long album never refills, its tail
drains to nothing, and the listener hits silence the moment their album ends --
exactly the failure replenishment exists to prevent. Gating urgency on
owned_future means panicking while seventeen songs are queued.

NOTHING HERE DEACTIVATES A STATION. Every blocking condition is transient by
construction: paused, unreadable, on-air programming, a pending batch, the
switch turned off. A station is retired as a product object, never because a
refill was inconvenient at one moment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import media_ref
import service_clock
import station_recovery
import station_runway
import station_store

logger = logging.getLogger("musicbrain")

# --- v1 tuning. Named, not magic ---------------------------------------------
#
# Normal low water at 8 is roughly 30 minutes of headroom at a typical song
# length of about four minutes -- enough for selection, up to a dozen serial MA
# resolutions, the append and one full retry before silence could threaten.
NORMAL_LOW_WATER = 8
# The runway a healthy station is topped up TO, ~75 minutes.
NORMAL_TARGET = 20
# Never write more than this in one transaction: a dozen serial resolutions
# fits comfortably inside one track, and smaller batches track evolving
# history instead of committing an hour of programming at once.
MAX_DESIRED_APPEND = 12

# Emergency is about SILENCE, not about station health, so it is measured on
# playable runway and sized to restore continuity quickly rather than to reach
# NORMAL_TARGET in one urgent transaction.
EMERGENCY_PLAYABLE_LOW_WATER = 3
EMERGENCY_DESIRED_APPEND = 6

# Storm guard between normal refills. 90s matches the existing
# dj_engine.FAILED_RETRY_COOLDOWN_S rather than inventing a second number.
NORMAL_COOLDOWN_S = 90.0

# Resolution misses are NORMAL -- a selection can easily resolve only some of
# its picks, because MA's exact artist+title matcher refuses to guess. So the selector is
# given a modest surplus to work with. Deliberately small: this compensates
# for ordinary misses, it does not chase a target, and the caller still
# appends at most desired_append_count.
RESOLUTION_SURPLUS = 2

MODE_NORMAL = "normal"
MODE_EMERGENCY = "emergency"

# Stable machine-readable blocking reasons.
R_OK = ""
R_SETTING_OFF = "setting_off"
R_NO_ACTIVE_STATION = "no_active_station"
R_PREPARING_STATION = "preparing_station"
R_PENDING_BATCH = "pending_batch"
R_PLAYER_MISSING = "player_missing"
R_QUEUE_UNREADABLE = "queue_unreadable"
R_INCOMPLETE_QUEUE = "incomplete_queue"
R_ANCHOR_MISSING = "current_anchor_missing"
R_PAUSED = "paused"
R_IDLE = "idle"
R_PROGRAMMING_ON_AIR = "programming_on_air"
R_PROFILE_INVALID = "station_profile_invalid"
R_OWNERSHIP_UNPROVEN = "ownership_unproven"
R_COOLDOWN = "cooldown"
R_RUNWAY_HEALTHY = "runway_healthy"
# Queue ownership: Music Assistant Autoplay was re-enabled on the owned
# queue past the drift budget and ParlorWave has yielded rather than fight. The
# durable `contested` record is read here (a local row, no MA call) and the
# decision blocks while the flag is still observed on; the moment the flag
# is seen off, the refill transaction's ownership step recovers the record.
R_MA_AUTOPLAY_CONTESTED = "ma_autoplay_contested"
# The active station is a Station Programmer (Mode 2) station, whose refill
# belongs to station_live. The legacy local-catalog path reports that and
# does nothing: one station, one refill engine.
R_PROGRAMMER_STATION = "station_programmer_owns_refill"


@dataclass(frozen=True)
class World:
    """A snapshot of everything the policy is allowed to reason about.

    Frozen and plain so `decide` can be exercised with a literal -- no MA, no
    database, no clock. `blocked` carries a precondition failure that policy
    cannot second-guess."""
    keep_radio_playing: bool = True
    station_id: str = ""
    queue_id: str = ""
    # None means NOT MEASURED -- the evaluation stopped before the runway was
    # ever counted. A measured zero and an unmeasured one are different facts,
    # and reporting both as 0 made a blocked evaluation look like an empty
    # queue: while a bulletin is on air the endpoint could report
    # playable_future 0 with several real tracks queued.
    owned_future: int | None = None
    playable_future: int | None = None
    witness_exact: bool = True
    seconds_since_refill: float | None = None   # None = never refilled
    # Music Assistant's per-queue Autoplay flag as reported in the SAME
    # player_queues/all snapshot the runway was measured from. None = not
    # measured (the queue was never read, or the install does not report
    # it). Reported here so queue ownership can re-check drift at every
    # evaluation without a second MA call; this module never acts on it.
    ma_autoplay: bool | None = None
    # The durable queue_ownership state for the station's queue ("" = no
    # record yet). Read locally; the policy only ever refuses on `contested`.
    ownership_state: str = ""
    blocked: str = R_OK


@dataclass(frozen=True)
class Decision:
    eligible: bool
    reason: str
    mode: str | None = None
    station_id: str = ""
    queue_id: str = ""
    owned_future: int | None = None      # None = not measured (see World)
    playable_future: int | None = None
    desired_append_count: int = 0
    selection_budget: int = 0
    cooldown_remaining: float = 0.0
    witness_exact: bool = True
    keep_radio_playing: bool = True
    ma_autoplay: bool | None = None
    ownership_state: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "reason": self.reason,
                "mode": self.mode, "station_id": self.station_id,
                "queue_id": self.queue_id, "owned_future": self.owned_future,
                "playable_future": self.playable_future,
                "desired_append_count": self.desired_append_count,
                "selection_budget": self.selection_budget,
                "cooldown_remaining": round(self.cooldown_remaining, 1),
                "witness_exact": self.witness_exact,
                "keep_radio_playing": self.keep_radio_playing,
                "ma_autoplay": self.ma_autoplay,
                "ownership_state": self.ownership_state}


# --- B. policy. PURE ----------------------------------------------------------

def selection_budget_for(desired: int) -> int:
    """How many candidates to ask the selector for, given how many we intend
    to append.

    Always >= desired, capped at MAX_DESIRED_APPEND. The surplus absorbs the
    ordinary resolution misses of MA's exact matcher; it never licenses
    appending more than desired, and it never justifies retrying resolution or
    loosening the exact artist+title match to reach a number."""
    if desired <= 0:
        return 0
    return min(MAX_DESIRED_APPEND, desired + RESOLUTION_SURPLUS)


def decide(world: World) -> Decision:
    """Pure policy. A snapshot in, a decision out -- no I/O of any kind."""
    base = dict(station_id=world.station_id, queue_id=world.queue_id,
                owned_future=world.owned_future,
                playable_future=world.playable_future,
                witness_exact=world.witness_exact,
                keep_radio_playing=world.keep_radio_playing,
                ma_autoplay=world.ma_autoplay,
                ownership_state=world.ownership_state)

    if world.blocked:
        return Decision(False, world.blocked, **base)
    if world.ownership_state == "contested" and world.ma_autoplay is not False:
        # ParlorWave yielded this queue's programming to Music Assistant after
        # the drift budget ran out. Only an observed OFF flag ends that (the
        # refill step then records the recovery); an unreadable flag keeps
        # the yield, because two autonomous programmers is the one thing
        # this whole policy exists to prevent.
        return Decision(False, R_MA_AUTOPLAY_CONTESTED, **base)
    if not world.keep_radio_playing:
        # The customer switch. It blocks AUTOMATIC behaviour only -- it is not
        # consulted by the explicit endpoint, and it changes no durable state.
        # Checked before the measurement guard because the switch is answered
        # without measuring anything at all.
        return Decision(False, R_SETTING_OFF, **base)
    if world.owned_future is None or world.playable_future is None:
        # Defensive: an unblocked world must carry measured runways. Policy
        # refuses to compare against a number nobody counted.
        return Decision(False, R_INCOMPLETE_QUEUE, **base)

    cooldown_left = 0.0
    if world.seconds_since_refill is not None:
        cooldown_left = max(0.0, NORMAL_COOLDOWN_S - world.seconds_since_refill)

    # EMERGENCY is evaluated FIRST and wins outright: silence is a worse
    # outcome than a slightly early refill, and it is measured on PLAYABLE
    # runway because that is what actually runs out.
    if world.playable_future <= EMERGENCY_PLAYABLE_LOW_WATER:
        desired = min(EMERGENCY_DESIRED_APPEND, MAX_DESIRED_APPEND,
                      max(0, NORMAL_TARGET - world.owned_future))
        if desired > 0:
            # Emergency bypasses the cooldown -- and ONLY the cooldown. Every
            # other precondition already blocked above.
            return Decision(True, R_OK, MODE_EMERGENCY,
                            desired_append_count=desired,
                            selection_budget=selection_budget_for(desired),
                            cooldown_remaining=cooldown_left, **base)

    # NORMAL is measured on OWNED runway, deliberately ignoring how much
    # listener content sits ahead. A station behind a long album still needs a
    # tail, and a station tail can wait behind unrelated music without
    # disturbing it.
    if world.owned_future > NORMAL_LOW_WATER:
        return Decision(False, R_RUNWAY_HEALTHY,
                        cooldown_remaining=cooldown_left, **base)
    if cooldown_left > 0:
        return Decision(False, R_COOLDOWN, cooldown_remaining=cooldown_left,
                        **base)
    desired = min(MAX_DESIRED_APPEND, max(0, NORMAL_TARGET - world.owned_future))
    if desired <= 0:
        return Decision(False, R_RUNWAY_HEALTHY,
                        cooldown_remaining=cooldown_left, **base)
    return Decision(True, R_OK, MODE_NORMAL, desired_append_count=desired,
                    selection_budget=selection_budget_for(desired),
                    cooldown_remaining=cooldown_left, **base)


# --- A. world-state gathering. Reads only -------------------------------------

def _seconds_since(stamp: str | None, now: datetime) -> float | None:
    if not stamp:
        return None
    bounds = service_clock.age_bounds(stamp, now)
    return None if bounds is None else max(0.0, bounds[1])


def count_runways(station: dict[str, Any], evidence: dict[str, Any]
                  ) -> tuple[int, int]:
    """(owned_future, playable_future) from a COMPLETE ordered read.

    Ownership is `queue_item_id` identity against the persisted witness --
    MA mints one per insertion, so a listener's own copy of a station song is
    never miscounted as ours. Programming and unavailable items are excluded
    from both counts by station_runway's classifier, so a queued DJ break
    or news bulletin can never make either runway look healthier than it is.

    No sort_index arithmetic: live sort_index values have gaps."""
    future = evidence.get("future") or []
    owned_ids = {w.get("queue_item_id")
                 for w in (station.get("tail") or ()) if w.get("queue_item_id")}
    measured = station_runway.measure_runway(future)
    owned = station_runway.count_owned_future(future, owned_ids)
    return owned, measured["music_count"]


def gather_world(*, now: datetime | None = None, db_path=None,
                 state_fn=None, future_fn=None,
                 settings_fn=None) -> World:
    """Read the world. Never writes anything, anywhere.

    Every failure becomes a truthful blocking reason rather than an exception:
    an unreadable queue is 'we cannot tell', which is a reason to do nothing,
    not a reason to change durable state."""
    now_dt = now or service_clock.legacy_now()
    try:
        if settings_fn is not None:
            keep = bool(settings_fn())
        else:
            import settings
            keep = bool(settings.load()["playback"]["keep_radio_playing"])
    except Exception:
        # Settings must never take a decision down; the safe default for an
        # unreadable switch is the shipped default.
        logger.warning("keep_radio_playing unreadable; assuming enabled",
                       exc_info=True)
        keep = True
    if not keep:
        # Checked FIRST, before any station or Music Assistant read. If the
        # customer has turned automatic replenishment off, the honest answer
        # is "the switch is off" -- not some incidental world-state detail
        # discovered on the way -- and there is no reason to poll MA to say
        # so. This is also what makes the reason stable regardless of what
        # the queue happens to look like.
        return World(keep_radio_playing=False)

    station = station_store.get_active_station(db_path=db_path)
    if station is None:
        return World(keep_radio_playing=keep, blocked=R_NO_ACTIVE_STATION)
    if station_store.is_programmer(station):
        return World(keep_radio_playing=keep,
                     station_id=station["station_id"],
                     queue_id=station.get("queue_id") or "",
                     blocked=R_PROGRAMMER_STATION)
    world, _evidence = gather_for_station(
        station, keep=keep, now=now_dt, db_path=db_path, state_fn=state_fn,
        future_fn=future_fn)
    return world


def gather_for_station(station: dict[str, Any], *, keep: bool = True,
                       now: datetime | None = None, db_path=None,
                       state_fn=None, future_fn=None
                       ) -> tuple[World, dict[str, Any] | None]:
    """The per-station half of gather_world, shared with station_live so a
    Station Programmer station is measured and gated by exactly the same
    reads and the same blocking reasons as any other: (World, evidence).

    `evidence` is the complete ordered queue read the runways were counted
    from (station_recovery.read_queue_evidence), or None when the world is
    blocked before one was taken. Never writes anything, anywhere."""
    now_dt = now or service_clock.legacy_now()
    sid = station["station_id"]
    queue_id = station.get("queue_id") or ""
    base = dict(keep_radio_playing=keep, station_id=sid, queue_id=queue_id,
                witness_exact=not station.get("tail_is_partial"),
                seconds_since_refill=_seconds_since(
                    station.get("last_replenished_at"), now_dt))

    if not station.get("artist_weights") or not queue_id:
        return (World(blocked=R_PROFILE_INVALID, **base), None)
    if station_store.get_preparing_station(db_path=db_path) is not None:
        return (World(blocked=R_PREPARING_STATION, **base), None)
    if station_store.get_pending_batch(sid, db_path=db_path) is not None:
        # A stronger block than any cooldown, and never resolved by time: an
        # unproven batch must be reconciled by reading MA, not waited out.
        return (World(blocked=R_PENDING_BATCH, **base), None)
    try:
        import queue_ownership
        record = queue_ownership.get_record(queue_id, db_path=db_path)
        if record and record.get("station_id") == sid:
            base["ownership_state"] = record.get("state") or ""
    except Exception:
        logger.warning("queue ownership record unreadable; ignoring",
                       exc_info=True)

    import ma_queue
    read_state = state_fn or ma_queue.queue_state
    try:
        state = read_state(queue_id)
    except Exception as exc:  # noqa: BLE001 - closed-gate boundary
        reason = (R_PLAYER_MISSING if "not found" in str(exc)
                  else R_QUEUE_UNREADABLE)
        return (World(blocked=reason, **base), None)
    flag = (state or {}).get("autoplay_enabled")
    base["ma_autoplay"] = flag if isinstance(flag, bool) else None
    playing = (state or {}).get("state")
    if playing == "paused":
        return (World(blocked=R_PAUSED, **base), None)
    if playing != "playing":
        return (World(blocked=R_IDLE, **base), None)
    current = (state or {}).get("current_item") or {}
    if not current.get("queue_item_id"):
        return (World(blocked=R_ANCHOR_MISSING, **base), None)
    if media_ref.programming_ref(current.get("uri"))[0]:
        return (World(blocked=R_PROGRAMMING_ON_AIR, **base), None)

    evidence = station_recovery.read_queue_evidence(
        queue_id, state_fn=read_state, future_fn=future_fn)
    if evidence is None:
        return (World(blocked=R_INCOMPLETE_QUEUE, **base), None)
    owned, playable = count_runways(station, evidence)
    if not station.get("tail") and owned == 0 and playable == 0:
        # Nothing owned, nothing playable and no witness at all: there is no
        # evidence this queue is still the station's. Refuse rather than guess.
        # Measured, and genuinely zero on both -- reported as 0, not null.
        return (World(owned_future=0, playable_future=0,
                      blocked=R_OWNERSHIP_UNPROVEN, **base), None)
    return (World(owned_future=owned, playable_future=playable, **base),
            evidence)


def evaluate(*, now: datetime | None = None, db_path=None, state_fn=None,
             future_fn=None, settings_fn=None) -> Decision:
    """gather + decide. The exact evaluation a future automatic caller will
    perform -- and the only thing the diagnostic endpoint runs."""
    return decide(gather_world(now=now, db_path=db_path, state_fn=state_fn,
                               future_fn=future_fn, settings_fn=settings_fn))
