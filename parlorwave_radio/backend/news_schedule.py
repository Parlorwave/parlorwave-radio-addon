#!/usr/bin/env python3
"""Automatic news planning -- DARK.

What this module decides, and nothing else:

  - whether an upcoming :30 news event should exist at all
  - when the cached bulletin needs refreshing for it
  - which natural music boundary would be eligible to carry it
  - that a target hour, and a bulletin, can each be used only once
  - what to report about all of the above

What it deliberately does NOT do, because no audio can be inserted yet:
no intro is written or synthesized, nothing is queued, MA is never
mutated, no ordinary DJ break is revoked or suppressed, cadence is
untouched, and there is no quiet guard. On-air behaviour lives elsewhere;
it inherits a planner that is already correct about
time, freshness, duplicates and restarts.

THE DARKNESS CONTRACT. `bulletins.enabled == false` means zero automatic
participation: no feed fetch, no download, no network of any kind, and no
mutation of persisted planning state. The only thing that runs while
disabled is read-only arithmetic for the status page. Manual Test Feed /
Fetch Latest / Preview are untouched and keep working exactly as before.

Timing is late-bound and evidence-based, never a timer: planning is
evaluated from the existing scheduler ticks, the station clock comes from
dj_context.station_now (the same HA-published basis the spoken clock
uses), and the boundary prediction comes from dj_engine.estimated_air_at.
There is no second thread, no second timer, and no second model of where
playback is.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import news_history
import news_sources
import paths
import service_clock

logger = logging.getLogger("musicbrain")

STATE_VERSION = 1

# --- the approved v1 air policy ----------------------------------------------
#
# The target is the top of the news half-hour. The window around it is
# deliberately asymmetric: a bulletin may start a minute EARLY, but the
# tolerance for running late is wider, because the alternative to a late
# start is skipping the hour entirely. A song is never interrupted, so the
# only question is ever "does the next natural boundary land in here".
TARGET_MINUTE = 30

# The station may aim at the top of the hour instead. These are the only
# two values, and the generalisation is deliberately this narrow: both are
# real broadcast conventions, both keep ONE target per clock hour -- which
# is what target_hour_key's identity, the duplicate guard and the restart
# arithmetic are all built on -- and both keep the proven window shape.
# Arbitrary minutes would not change this module's maths either, but they
# WOULD invite "every 10 minutes", which a slot (checked from 6 minutes
# before the target, airable until 10 minutes after it) cannot express
# without overlapping itself. Two conventions, one target an hour, same
# window: no new risk.
SUPPORTED_TARGET_MINUTES = (0, 30)
WINDOW_EARLY_S = 60.0       # xx:29:00 -- earliest acceptable air
WINDOW_LATE_S = 240.0       # xx:34:00 -- latest ON-TIME air
# THE LATE GRACE. Past the on-time
# window the slot is not over yet: a boundary up to SLOT_EXPIRY_S after the
# target may still carry the bulletin, as long as that bulletin is fresh at
# the boundary it would air on -- the same freshness rule every airing
# already passes. Five minutes is shorter than a lot of songs, so one long
# track used to span the whole window and lose the hour with the bulletin
# sitting fresh in the cache, well inside its freshness limit, and the
# first boundary only minutes past the window. A song is still never
# interrupted; the grace only stops
# "the first boundary came 30 s late" from meaning "no news this hour".
# Bounds: the grace must end before the other half-hour's feed check opens
# (xx:24 / xx:54) so the :00 and :30 slots never overlap, and inside it the
# DJ yields only to a bulletin News actually holds (bulletin_claims_boundary)
# so a dead feed still costs the DJ at most the on-time window. With minutes
# closer together than that (sources accept any minute) a slot in its grace
# yields to the next source's open feed check (_due_source).
SLOT_EXPIRY_S = 600.0       # xx:40:00 -- the slot's last chance (fresh only)
# How far before the target the feed may be checked. It OPENS at -6 min and
# stays open until the slot expires: a listener who starts the station at
# xx:29 must still be able to check immediately, and one who starts at
# xx:41 has simply missed this hour.
CHECK_LEAD_S = 360.0
# A cached bulletin older than this at its intended airtime is not news.
MAX_BULLETIN_AGE_MIN = 90.0

# Event states. One lifecycle, one state file: planning and on-air are the
# same event seen at different moments, so there is no second scheduler and
# no second table.
IDLE = "idle"               # no target worth watching
WATCHING = "watching"       # a target is upcoming; not yet time to check
REFRESHING = "refreshing"   # inside the check window, bulletin not ready
ARMED = "armed"             # fresh bulletin + an eligible predicted boundary
RENDERING = "rendering"     # the DJ's intro is being spoken into audio
READY = "ready"             # intro rendered; waiting for the insertion gate
QUEUED = "queued"           # intro+bulletin handed to MA as the next items
INTRO_AIRING = "intro_airing"        # the intro line is on air
BULLETIN_AIRING = "bulletin_airing"  # the publisher bulletin is on air
COMPLETED = "completed"     # the bulletin left the air
SKIPPED = "skipped"         # this target will not happen; reason recorded
DISABLED = "disabled"       # reported only; never persisted

# States in which the event owns audio MA has already been handed, or is
# playing right now. The cached bulletin file must not be replaced under
# any of them, and a restart must not re-queue the event.
IN_FLIGHT = (QUEUED, INTRO_AIRING, BULLETIN_AIRING)

# One pass at a time. Planning is driven from playback events, which can
# arrive on several threads; the render leg takes real seconds. The lock
# keeps a single evaluation in flight so two ticks cannot both render or
# both insert. (The dj_breaks state machine still provides the atomic
# claim that makes double-render impossible even across processes.)
_pass_lock = threading.Lock()


def _state_path() -> Path:
    return paths.BASE_DIR / "news_on_air_state.json"


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Write-then-rename so a crash mid-write can never leave a half state
    file behind -- the same pattern dj_context uses for its snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def empty_state() -> dict[str, Any]:
    """The dark planning state: nothing watched, nothing owed, nothing
    aired. This is also what a corrupt or missing file reads as."""
    return {
        "version": STATE_VERSION,
        "target_hour": None,          # station-local YYYY-MM-DDTHH
        "target_at": None,            # ISO, station-local, tz-aware
        "state": IDLE,
        "reason": "",
        "bulletin_token": None,
        "bulletin_source_id": None,
        # WHICH configured source this event belongs to, and what the
        # station calls it. bulletin_source_id above is the BULLETIN row's
        # source (derived from the feed URL); these two are the Newsroom
        # source document's own identity and display name, which is what
        # history and the screen report against.
        "source_id": None,
        "source_name": None,
        # Per-source bookkeeping: {source_id: {last_aired_token,
        # last_aired_at, last_result, last_result_at}}. The global
        # aired_tokens ring still prevents any token airing twice; this
        # answers "what did THIS source last do", which is what the
        # Newsroom shows and what per-source dedupe reports against.
        "sources": {},
        "bulletin_published_at": None,
        "intended_boundary_item_id": None,
        "estimated_air_at": None,     # service-clock ISO (naive UTC)
        "planned_at": None,
        "session_id": None,
        "queue_id": None,
        "feed_checked_for_hour": None,
        # --- on-air evidence -------------------------------------------
        "intro_break_id": None,       # dj_breaks token of the intro line
        "intro_state": None,          # mirrored for status; dj_breaks owns it
        "queued_at": None,
        "intro_air_start": None,
        "bulletin_air_start": None,
        "bulletin_completed_at": None,
        "last_result": None,
        "last_result_at": None,
        "last_aired_token": None,
        "last_aired_at": None,
        "last_completed_target_hour": None,
        "aired_tokens": [],           # bounded ring of already-used bulletins
        # Post-news silence. Released by OBSERVED playback, never a timer.
        "quiet_guard": None,
    }


AIRED_TOKEN_CAP = 24          # one day of hourly bulletins; bounded on purpose


def load_state() -> dict[str, Any]:
    """Persisted planning state, or the empty state.

    Fails SAFE in every direction: a missing file, unreadable file, corrupt
    JSON, wrong type or a version from the future all read as 'dark, plan
    nothing'. Planning state can never affect music, so losing it must cost
    at most one skipped bulletin."""
    try:
        raw = json.loads(_state_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError,
            UnicodeDecodeError):
        return empty_state()
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return empty_state()
    out = empty_state()
    for k, v in raw.items():
        if k in out:
            out[k] = v
    if not isinstance(out.get("aired_tokens"), list):
        out["aired_tokens"] = []
    if not isinstance(out.get("sources"), dict):
        out["sources"] = {}
    return out


# Last (target_hour, state, reason) written, so the planner's lifecycle is
# visible in the ordinary log as a handful of lines per hour -- one per
# transition, never one per tick. Without this, 'never evaluated', 'not
# due', 'lost the boundary' and 'stuck in flight' are indistinguishable
# after the fact.
_last_logged: tuple[Any, Any, Any] | None = None

# The last document actually written, as (path, canonical JSON). The
# planner is evaluated on every position tick -- every 30 s while music
# plays -- and for most of every hour the evaluation ends in exactly the
# state it started from ("watching, before_check_window"). Writing that
# again is a new inode, a rename and a directory update for nothing, and
# on an SD card it would be the single most frequent file write the
# service makes. Identical content is now
# recognised and skipped; every real transition still lands on disk at
# once, so a crash can lose nothing it could not lose before.
_last_written: tuple[str, str] | None = None


def _canonical(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, default=str)


def save_state(state: dict[str, Any]) -> None:
    """Best-effort persist. A write failure is logged and swallowed: the
    planner degrades to re-deciding next tick, which is always safe.

    Unchanged content is not rewritten (see _last_written). The file is
    checked to still exist so a document removed from under the service
    is restored on the next save rather than assumed present."""
    global _last_logged, _last_written
    key = (state.get("target_hour"), state.get("state"), state.get("reason"))
    if key != _last_logged:
        _last_logged = key
        logger.info("news planner %s: %s (%s)", *key)
    try:
        path = _state_path()
        canon = _canonical(state)
        if (_last_written is not None and _last_written == (str(path), canon)
                and path.exists()):
            return
        _atomic_write(path, state)
        _last_written = (str(path), canon)
    except Exception:
        logger.warning("news planning state write failed (planning only)",
                       exc_info=True)


# --- pure time helpers --------------------------------------------------------

def target_hour_key(target_at: datetime, source_id: str | None = None) -> str:
    """The identity of ONE bulletin event.

    Station-local YYYY-MM-DDTHH: two different days, or two different
    hours, are two different events; the same hour is always the same event
    however often it is evaluated.

    Since multi-source it also carries the SOURCE, because one clock hour
    can now hold two events -- a bulletin at :00 from one publisher and
    another at :30 from a second. Without the source in the key they would
    share one identity, and the duplicate guard that exists to stop an hour
    happening twice would instead stop the second source happening at all.

    source_id is optional so a caller that legitimately has no source (a
    skip recorded before one was chosen) still gets a usable key."""
    stamp = target_at.strftime("%Y-%m-%dT%H")
    return f"{stamp}#{source_id}" if source_id else stamp


def target_minute(bulletins: dict[str, Any] | None = None) -> int:
    """The configured minute of the hour bulletins aim for.

    Reads settings only when the caller did not already hold them, and
    falls back to the station default on anything unexpected: an unreadable
    or malformed settings document must cost the listener a preference, not
    the news."""
    if bulletins is None:
        try:
            import settings
            bulletins = settings.load().get("bulletins") or {}
        except Exception:
            bulletins = {}
    minute = ((bulletins or {}).get("schedule") or {}).get("target_minute")
    if isinstance(minute, bool) or not isinstance(minute, int):
        return TARGET_MINUTE
    return minute if minute in SUPPORTED_TARGET_MINUTES else TARGET_MINUTE


def target_for(local_now: datetime, minute: int | None = None) -> datetime:
    """The target of local_now's own hour (may be past or future)."""
    if minute is None:
        minute = TARGET_MINUTE
    return local_now.replace(minute=minute, second=0, microsecond=0)


def window_for(target_at: datetime) -> tuple[datetime, datetime]:
    """[earliest, latest] ON-TIME air for this target. The slot itself
    lives on until expiry_for(target_at); see SLOT_EXPIRY_S."""
    return (target_at - timedelta(seconds=WINDOW_EARLY_S),
            target_at + timedelta(seconds=WINDOW_LATE_S))


def expiry_for(target_at: datetime) -> datetime:
    """The last moment this slot's bulletin may air at all (late grace)."""
    return target_at + timedelta(seconds=SLOT_EXPIRY_S)


def active_target(local_now: datetime, minute: int | None = None) -> datetime:
    """The target this moment is working toward.

    The current hour's target stays the target until the slot EXPIRES ten
    minutes later; after that the hour is over and the next one begins.
    Restart-safe by construction: it is a pure function of the local clock
    and the configured minute, so a process that restarts inside the slot
    resumes the same target it was already watching, and one that restarts
    after it correctly moves on.

    The previous hour's target is asked first: a minute late in the hour
    expires in the NEXT clock hour (minute 55 expires at xx:05), and that
    slot must not be abandoned at the top of the hour. For :00 and :30 the
    previous hour's slot has always expired, so nothing changes there."""
    this_hour = target_for(local_now, minute)
    if this_hour.minute * 60 + SLOT_EXPIRY_S > 3600:
        # Only a slot that crosses the hour can still be live in the next
        # one, and only it pays for the question -- in absolute instants,
        # so a DST change cannot revive an aired or nonexistent slot.
        previous = this_hour - timedelta(hours=1)
        if _utc(local_now) <= _utc(expiry_for(previous)):
            return previous
    if local_now <= expiry_for(this_hour):
        return this_hour
    return this_hour + timedelta(hours=1)


def _utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt


def check_open_at(target_at: datetime) -> datetime:
    return target_at - timedelta(seconds=CHECK_LEAD_S)


def _due_source(bulletins: dict[str, Any] | None, local_now: datetime
                ) -> tuple[dict[str, Any] | None, datetime | None]:
    """WHOSE TURN, with the one rule the late grace adds: a slot past its
    on-time window yields to any other source whose own feed check has
    opened. A late bulletin never eats into another source's on-time slot.
    The :00/:30 pair never meets this (the :00 grace ends at xx:10, the :30
    check opens at xx:24); minutes closer together than that -- sources
    accept any minute -- keep exactly the turn order they had before the
    grace existed. Only a strictly LATER slot is yielded to: two sources on
    the same minute keep due_source's rule that the loser skips the hour,
    and never take over the winner's grace. Everything else is
    news_sources.due_source unchanged."""
    source, target_at = news_sources.due_source(bulletins, local_now,
                                                active_target)
    if (source is None or target_at is None
            or local_now <= window_for(target_at)[1]):
        return source, target_at
    best: tuple[datetime, int, dict[str, Any]] | None = None
    for order, src in enumerate(news_sources.enabled_sources(bulletins)):
        if src["id"] == source["id"]:
            continue
        other = active_target(local_now, src["target_minute"])
        if other > target_at and check_open_at(other) <= local_now and (
                best is None or (other, order) < (best[0], best[1])):
            best = (other, order, src)
    if best is None:
        return source, target_at
    return best[2], best[0]


def should_check_feed(local_now: datetime, target_at: datetime) -> bool:
    """Is the feed-check window open for this target?

    Opens at target-6min and stays open until the slot expires, so a
    listener who starts the station mid-window checks immediately rather
    than missing an hour they could still have made."""
    return check_open_at(target_at) <= local_now <= expiry_for(target_at)


def boundary_eligible(predicted_local: datetime,
                      target_at: datetime) -> tuple[bool, str]:
    """May a boundary predicted for this local time carry the bulletin?

    Too early is NOT a failure -- the music simply continues and the next
    boundary is evaluated -- so it reports 'too_early' rather than a skip.
    Past the on-time window but before the slot expires the boundary is
    still eligible ('boundary_late_grace'): every caller then requires the
    bulletin to be fresh AT that boundary, which is what makes a late
    airing news rather than stale audio. Past expiry this hour is gone."""
    early, late = window_for(target_at)
    if predicted_local < early:
        return False, "boundary_too_early"
    if predicted_local > expiry_for(target_at):
        return False, "boundary_too_late"
    if predicted_local > late:
        return True, "boundary_late_grace"
    return True, "boundary_eligible"


# --- two clocks, kept apart ---------------------------------------------------
#
# This module handles stamps from two different sources and they are NOT
# interchangeable. Parsing both with one function yields a subtraction
# that keeps every bulletin from airing on an installation west of UTC:
#
#   * OUR OWN stamps -- queued_at, last_result_at -- are written with
#     `_iso(now_dt)` from the service clock, naive. Under Supervisor the
#     add-on container inherits the HOME's timezone, so "the service clock"
#     is station-local wall time on a normal installation and UTC only on a
#     developer box that happens to run UTC. They are only ever compared
#     against another service-clock stamp, which is why they work.
#   * A PUBLISHER's stamp -- published_at -- is an absolute instant, and
#     arrives from the feed with a UTC offset.
#
# Mixing them subtracts two different clocks and yields the offset as a
# fake age: in a zone five hours behind UTC a bulletin published minutes
# ago looks five hours in the FUTURE, so `bulletin_published_in_future`
# refuses every one of them, forever. Anything that measures an ELAPSED
# REAL DURATION must therefore work in absolute instants -- timezone-aware
# -- and nothing may subtract a service-clock stamp from a publisher's.


def _instant(value: Any) -> datetime | None:
    """A PUBLISHER's ISO stamp -> timezone-aware UTC. None when unusable.

    A naive stamp here is assumed to be UTC. That is the right assumption
    for this input -- RSS/Atom dates carry an offset, and a feed that omits
    one is conventionally read as UTC -- and it is never applied to our own
    stamps, which go through _service_naive instead."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _service_naive(value: Any) -> datetime | None:
    """One of OUR OWN ISO stamps -> naive service clock, for comparison
    against another service-clock stamp and nothing else. An aware stamp
    (an older state file, or one written by a different basis) is converted
    into the service clock rather than silently compared across zones."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


def bulletin_fresh_enough(bulletin: dict[str, Any] | None,
                          air_at: datetime | None,
                          source_id: str,
                          max_age_min: float = MAX_BULLETIN_AGE_MIN
                          ) -> tuple[bool, str]:
    """Is this cached bulletin usable for a bulletin airing at air_at?

    Age is measured against the intended AIRTIME, not against now: a
    bulletin that is fine when planning starts can be too old by the time
    it would actually be heard. Note this answers 'not stale', never 'is
    the newest item' -- only a feed check can answer that.

    `air_at` MUST be timezone-aware. This is the one subtraction in the
    module that crosses from our clock to a publisher's, so it is the one
    place that has to work in absolute instants; see the two-clocks note
    above. A naive airtime is refused rather than guessed at, because
    guessing is precisely what lets that kind of defect go unseen."""
    if not bulletin:
        return False, "no_bulletin"
    if bulletin.get("state") != "cached":
        return False, "bulletin_not_cached"
    if source_id and bulletin.get("source_id") != source_id:
        return False, "bulletin_source_mismatch"
    published = _instant(bulletin.get("published_at"))
    if published is None:
        return False, "bulletin_published_at_invalid"
    if air_at is None:
        return False, "airtime_unknown"
    if air_at.tzinfo is None:
        # A caller handing over a wall-clock time cannot be served: there is
        # no offset to reconcile it with the publisher's instant.
        return False, "airtime_not_absolute"
    age_min = (air_at - published).total_seconds() / 60.0
    if age_min < 0:
        return False, "bulletin_published_in_future"
    if age_min > max_age_min:
        return False, "bulletin_stale"
    return True, "bulletin_fresh"


# --- duplicate prevention (the structures on-air code calls) -----------------------
#
# Two independent identities, because they fail differently. The TARGET HOUR
# stops one :30 event from happening twice however many times planning runs
# or the process restarts. The BULLETIN TOKEN stops one recording from being
# aired twice across different hours -- e.g. a publisher that has not posted
# a new item yet. Neither is inferred from news_bulletins.state: the cache
# lifecycle ('cached'/'expired') is about which FILE is current and says
# nothing about what has been ON AIR.

def may_plan_target_hour(state: dict[str, Any], hour_key: str) -> tuple[bool, str]:
    if state.get("last_completed_target_hour") == hour_key:
        return False, "target_hour_already_completed"
    if (state.get("target_hour") == hour_key
            and state.get("state") == SKIPPED):
        return False, "target_hour_already_skipped"
    return True, ""


def was_bulletin_aired(state: dict[str, Any], token: str | None) -> bool:
    return bool(token) and token in (state.get("aired_tokens") or [])


def backfill_spent_tokens(db_path=None, now: datetime | None = None) -> int:
    """Teach the spent ledger what this installation already knew.

    An installation upgrading into the durable ledger has cached rows and an
    `aired_tokens` ring, and nothing tying the two together -- so a bulletin
    that aired yesterday would be replayable until it next completed one,
    which for an idle station is never. This closes that window once.

    DETERMINISTIC AND BOUNDED, and from authoritative fields only: every
    entry of the ring (at most AIRED_TOKEN_CAP of them) plus
    `last_aired_token`, all of which mean exactly 'this token reached air'.
    Nothing is inferred from a name, a title or a timestamp.

    Idempotent, and cheap to keep calling: the ledger writes are INSERT OR
    IGNORE, so the second run and every run after it write nothing at all.

    NOT a substitute for recording spentness at completion. The ring is
    bounded and its entries age out; this only ensures the ledger starts
    knowing what the ring still remembers.
    """
    try:
        import program_tokens
        state = load_state()
        tokens = list(state.get("aired_tokens") or [])
        last = state.get("last_aired_token")
        if last and last not in tokens:
            tokens.append(last)

        # THE ONE TOKEN THAT MUST NEVER BE BACKFILLED: the one on the air.
        #
        # `aired_tokens` is written at the START of airing, and this runs on
        # the reaper's own five-minute timer -- the same order of magnitude
        # as a bulletin's length -- so a sweep lands mid-broadcast as a
        # matter of routine, not as a rare race. Without this exclusion it
        # would record the live token as spent and the next Flow retry would
        # be handed silence, mid-sentence.
        #
        # `bulletin_locked` is the existing authority for "this artifact is
        # committed to air right now" -- the same one that stops a refresh
        # deleting the file out from under Music Assistant. It reads the
        # planner's own IN_FLIGHT states. Nothing here infers anything from
        # a title, an age, a queue position or a timing heuristic.
        locked, active = bulletin_locked()
        if locked and active:
            tokens = [t for t in tokens if t != active]

        if not tokens:
            return 0
        return program_tokens.retire(tokens, program_tokens.KIND_NEWS,
                                     now, db_path)
    except Exception:  # noqa: BLE001 - best-effort, never breaks a sweep
        logger.warning("news: spent-token backfill failed", exc_info=True)
        return 0


#  What survives ANY reset of the working state. These four answer "what has
#  already gone to air", which no new hour, session or restart may forget --
#  losing them is how the same bulletin, or the same :30, happens twice.
HISTORY_KEYS = ("last_aired_token", "last_aired_at",
                "last_completed_target_hour", "aired_tokens", "sources")


def _carry_history(old: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    for k in HISTORY_KEYS:
        if old.get(k):
            fresh[k] = old[k]
    # The quiet guard belongs to the bulletin that JUST aired but protects
    # the music playing now, so it must outlive the event that opened it.
    # Without this, the first tick after the air window closes rolls the
    # target to the next hour, resets the working state, and silently drops
    # the guard -- letting an overdue break speak straight off the back of
    # the bulletin, which is the exact thing it exists to prevent.
    guard = old.get("quiet_guard")
    if guard and guard.get("active"):
        fresh["quiet_guard"] = guard
    return fresh


def mark_target_planned(state: dict[str, Any], hour_key: str,
                        target_at: datetime, now: datetime,
                        session_id: str | None = None) -> dict[str, Any]:
    """Claim this target hour. Idempotent: re-claiming the hour already
    being watched keeps its accumulated evidence rather than resetting it,
    which is what makes per-tick evaluation safe."""
    if state.get("target_hour") != hour_key:
        state = _carry_history(state, empty_state())
    state["target_hour"] = hour_key
    state["target_at"] = target_at.isoformat()
    # Stamped when the hour is CLAIMED, not on every re-evaluation: a
    # per-tick stamp made every otherwise-identical pass a changed document
    # and defeated save_state's unchanged-content check.
    if not state.get("planned_at"):
        state["planned_at"] = _iso(now)
    if session_id:
        state["session_id"] = session_id
    return state


def mark_target_skipped(state: dict[str, Any], hour_key: str, reason: str,
                        now: datetime) -> dict[str, Any]:
    state["target_hour"] = hour_key
    state["state"] = SKIPPED
    state["reason"] = reason
    state["last_result"] = reason
    state["last_result_at"] = _iso(now)
    state["intended_boundary_item_id"] = None
    state["estimated_air_at"] = None
    return state


def source_record(state: dict[str, Any], source_id: str | None
                  ) -> dict[str, Any]:
    """This source's own bookkeeping, or an empty one. Never raises."""
    if not source_id:
        return {}
    book = state.get("sources")
    rec = (book or {}).get(str(source_id)) if isinstance(book, dict) else None
    return dict(rec) if isinstance(rec, dict) else {}


def _put_source_record(state: dict[str, Any], source_id: str | None,
                       **fields: Any) -> None:
    if not source_id:
        return
    book = state.get("sources")
    if not isinstance(book, dict):
        book = {}
    rec = book.get(str(source_id))
    rec = dict(rec) if isinstance(rec, dict) else {}
    rec.update(fields)
    book[str(source_id)] = rec
    state["sources"] = book


def mark_bulletin_aired(state: dict[str, Any], token: str,
                        hour_key: str | None, now: datetime,
                        source_id: str | None = None) -> dict[str, Any]:
    """Record that a bulletin actually went to air.

    TWO ledgers, on purpose. `aired_tokens` is one bounded ring across every
    source: a token is unique to an episode, so nothing in it can ever air
    twice whichever source it came from. The per-source record answers a
    different question -- what did THIS newsroom last manage -- which is
    what the Newsroom screen shows and what makes 'do not replay an
    unchanged hourly item solely because another schedule occurs' visible
    rather than merely true.

    Only a bulletin that actually reached air lands here. A feed test, a
    fetch failure, a rejected over-long episode, an expired occurrence and
    a playback failure before completion all deliberately do not."""
    aired = [t for t in (state.get("aired_tokens") or []) if t != token]
    aired.append(token)
    state["aired_tokens"] = aired[-AIRED_TOKEN_CAP:]
    state["last_aired_token"] = token
    state["last_aired_at"] = _iso(now)
    if hour_key:
        state["last_completed_target_hour"] = hour_key
    _put_source_record(state, source_id or state.get("source_id"),
                       last_aired_token=token, last_aired_at=_iso(now),
                       last_result="aired", last_result_at=_iso(now))
    return state


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


# --- the planning pass --------------------------------------------------------

def _enabled(bulletins: dict[str, Any] | None) -> bool:
    return bool((bulletins or {}).get("enabled"))


def scheduler_pass(session: dict[str, Any] | None,
                   queue_id: str,
                   now: datetime | None = None,
                   deps: dict[str, Any] | None = None,
                   bulletins: dict[str, Any] | None = None,
                   db_path=None) -> dict[str, Any]:
    """One news planning evaluation, driven by an existing scheduler tick.

    Returns {"state": ..., "reason": ...} for logging and tests; the caller
    ignores it. NOTHING here touches the queue, the player, dj_breaks or
    cadence -- it decides and records, and that is all.

    Order matters, cheapest and most binding refusal first: disabled beats
    everything (and returns before any network or state write), then the
    active-listening gate, then duplicate protection, then time, then
    freshness, then the boundary.
    """
    deps = deps or {}
    now_dt = now or service_clock.legacy_now()
    if bulletins is None:
        try:
            import settings
            bulletins = settings.load().get("bulletins") or {}
        except Exception:
            bulletins = {}

    # 1. DARKNESS. Before any network call, any MA read, any state write.
    if not _enabled(bulletins):
        return {"state": DISABLED, "reason": "bulletins_disabled"}

    try:
        with _pass_lock:
            return _plan(session, queue_id, now_dt, deps, bulletins, db_path)
    except Exception:
        # Planning is never allowed to disturb playback or the DJ. A failure
        # simply means no news this pass.
        logger.warning("news planning pass failed (music unaffected)",
                       exc_info=True)
        return {"state": IDLE, "reason": "planner_error"}


def _plan(session, queue_id, now_dt, deps, bulletins, db_path):
    import dj_context
    import dj_engine
    import news_bulletins

    # 2. ACTIVE-LISTENING GATE. Automatic news exists for someone actually
    # listening; the wall clock reaching :24 is not a reason to do anything.
    if not session or not session.get("session_id") or not queue_id:
        return {"state": IDLE, "reason": "no_active_session"}
    if session.get("queue_id") and session["queue_id"] != queue_id:
        return {"state": IDLE, "reason": "queue_session_mismatch"}

    local_now = dj_context.station_now(now_dt)
    if local_now is None:
        # No station clock means no station-local :30 to aim at. Server time
        # is NOT a substitute.
        return {"state": IDLE, "reason": "station_clock_unknown"}

    state = load_state()
    # An event already handed to MA, or on air right now, is no longer a
    # planning question: it is observed to completion through the playback
    # lifecycle hooks. Re-planning here is exactly how a restart would
    # double-insert. The one exception is a handoff that provably died --
    # see _in_flight_lost -- which must release the planner, or every later
    # hour is silently lost behind a bulletin that will never play.
    if state.get("state") in IN_FLIGHT:
        lost = _in_flight_lost(state, local_now, now_dt, db_path)
        if lost is None:
            return {"state": state["state"], "reason": "event_in_flight"}
        logger.warning("news event for %s never aired (%s): releasing it so "
                       "later hours can be planned",
                       state.get("target_hour"), lost)
        intro = state.get("intro_break_id")
        if intro:
            dj_engine.revoke(intro, now_dt, db_path, reason="news_event_lost")
        # The bulletin was handed to MA and may still be sitting in its
        # queue. Releasing it makes it "not ours" -- so it must also stop
        # being servable as news, or a stranded copy plays at full length
        # whenever the head reaches it, possibly many minutes late.
        _retire_unaired_bulletin(state, now_dt, db_path, "news_event_lost")
        state = mark_target_skipped(state, state.get("target_hour") or "",
                                    "never_aired:" + lost, now_dt)
        state["intro_break_id"] = None
        state["intro_state"] = None
        save_state(state)
        _record_history(state, news_history.SKIPPED, now_dt,
                        reason="never_aired:" + lost, db_path=db_path)
    # WHOSE TURN. Each enabled source has its own minute of the hour, so
    # each has its own next target; the soonest is the one being worked
    # toward, ties broken by configured order. news_sources owns that rule;
    # only ONE event is ever in flight, because audio is serial.
    source, target_at = _due_source(bulletins, local_now)
    if source is None or target_at is None:
        return {"state": IDLE, "reason": "no_bulletin_source"}
    source_cfg = news_sources.as_feed_config(source, bulletins)
    hour_key = target_hour_key(target_at, source["id"])

    # A new session never inherits a previous session's CLAIM on an hour --
    # but it must still inherit what actually AIRED. Clearing everything
    # here would let a session change re-air an hour that already ran, which
    # is precisely the duplicate the target-hour identity exists to stop.
    if state.get("session_id") and state.get("session_id") != session["session_id"]:
        if state.get("target_hour") == hour_key:
            state = _carry_history(state, empty_state())

    # 3. DUPLICATE PROTECTION.
    may, why = may_plan_target_hour(state, hour_key)
    if not may:
        return {"state": SKIPPED, "reason": why}

    # The previous slot is about to be forgotten. If it ran out while the
    # planner was still trying -- waiting on a closed gate, an unknown
    # airtime, a boundary someone else owned, a feed that never produced a
    # fresh bulletin -- it gets a row saying so before its state is reset.
    _record_expired_slot(state, hour_key, now_dt, local_now, session,
                         bulletins, db_path)
    state = mark_target_planned(state, hour_key, target_at, now_dt,
                                session.get("session_id"))
    state["queue_id"] = queue_id

    # 4. TIME. Outside the check window there is nothing to do but watch.
    if local_now < check_open_at(target_at):
        state["state"] = WATCHING
        state["reason"] = "before_check_window"
        save_state(state)
        return {"state": WATCHING, "reason": "before_check_window"}

    # 5. PLAYBACK EVIDENCE. One read, shared by the boundary decision; a
    # player that is not playing has no boundary to predict.
    state_fn = deps.get("state_fn")
    if state_fn is None:
        import ma_queue
        state_fn = ma_queue.queue_state
    try:
        q = state_fn(queue_id)
    except Exception:
        state["state"] = REFRESHING
        state["reason"] = "queue_unreadable"
        save_state(state)
        return {"state": REFRESHING, "reason": "queue_unreadable"}

    air_at = dj_engine.estimated_air_at(q, None, now_dt)
    if air_at is None:
        # Paused, idle, buffered past the audible item, or state too stale
        # to trust. Never guess an airtime, and never fetch toward one. A
        # closed structural gate keeps its own long-standing reason code:
        # the boundary is gone, which is more specific than "unknown".
        import ma_queue
        why_none = ("gate_closed" if not ma_queue.insertion_gate_open(q)
                    else "airtime_unknown")
        state["state"] = WATCHING
        state["reason"] = why_none
        state["intended_boundary_item_id"] = None
        state["estimated_air_at"] = None
        save_state(state)
        return {"state": WATCHING, "reason": why_none}

    predicted_local = dj_context.station_project(local_now,
                                                  air_at - now_dt)
    eligible, why = boundary_eligible(predicted_local, target_at)
    if not eligible and why == "boundary_too_late":
        # Even the FIRST safe boundary lands past the slot's expiry (xx:40):
        # this hour is gone. Music continues untouched; nothing is
        # interrupted to make news fit. A boundary inside the late grace is
        # NOT this branch: it goes on to the freshness check like any other.
        state = mark_target_skipped(state, hour_key, why, now_dt)
        save_state(state)
        _record_history(state, news_history.SKIPPED, now_dt, reason=why,
                        source=source, target_at=target_at, db_path=db_path)
        return {"state": SKIPPED, "reason": why}

    # 6. FRESHNESS. Only inside the window, only for a real intended airtime.
    source_id = ""
    try:
        source_id, _url, _label = news_bulletins.resolve_publisher_source(
            source_cfg)
    except Exception:
        state["state"] = SKIPPED
        state["reason"] = "source_unconfigured"
        save_state(state)
        return {"state": SKIPPED, "reason": "source_unconfigured"}

    # THIS SOURCE's retained bulletin, asked for by name. While the audio
    # slot was global there was only ever one cached row, so "the newest
    # cached row" and "this source's bulletin" were the same object and the
    # question could be left vague; now that each source keeps its own, a
    # vague question would hand one source's planning pass another source's
    # bulletin, and refuse it as a source mismatch a line later.
    current = news_bulletins.current_bulletin(db_path, source_id=source_id)
    fresh, fresh_why = bulletin_fresh_enough(current, predicted_local,
                                             source_id)
    if fresh and was_bulletin_aired(state, (current or {}).get("token")):
        fresh, fresh_why = False, "bulletin_already_aired"
    if fresh and _bulletin_retired((current or {}).get("token"), db_path):
        fresh, fresh_why = False, "bulletin_retired"

    if not fresh or state.get("feed_checked_for_hour") != hour_key:
        # ONE bounded feed lookup per target hour, through the existing
        # proven path: it decides newest-GUID itself and returns
        # 'already_current' WITHOUT redownloading audio when the cache
        # already holds that item. No downloader logic is duplicated here.
        fetch = deps.get("fetch_fn") or news_bulletins.fetch_latest_bulletin
        try:
            res = fetch(bulletins=source_cfg, now=now_dt, db_path=db_path)
        except Exception as exc:  # noqa: BLE001 - network of any shape
            res = {"ok": False, "reason": "feed_failed", "error": str(exc)[:200]}
        state["feed_checked_for_hour"] = hour_key
        state["last_result"] = res.get("result") or res.get("reason")
        state["last_result_at"] = _iso(now_dt)
        if not res.get("ok"):
            # Keep watching: a later tick may succeed inside this window,
            # and the failure never reaches music. It is filed under the
            # hour's own id, so a later success in the same window REPLACES
            # this row rather than leaving the screen showing both.
            state["state"] = REFRESHING
            state["reason"] = res.get("reason") or "feed_failed"
            save_state(state)
            _record_history(state, news_history.FAILED, now_dt,
                            reason=state["reason"], source=source,
                            target_at=target_at, db_path=db_path)
            return {"state": REFRESHING, "reason": state["reason"]}
        current = news_bulletins.current_bulletin(db_path,
                                                  source_id=source_id)
        fresh, fresh_why = bulletin_fresh_enough(current, predicted_local,
                                             source_id)
        if fresh and was_bulletin_aired(state, (current or {}).get("token")):
            fresh, fresh_why = False, "bulletin_already_aired"
        if fresh and _bulletin_retired((current or {}).get("token"), db_path):
            fresh, fresh_why = False, "bulletin_retired"

    if not fresh:
        state["state"] = REFRESHING
        state["reason"] = fresh_why
        save_state(state)
        return {"state": REFRESHING, "reason": fresh_why}

    state["bulletin_token"] = current["token"]
    state["bulletin_source_id"] = current["source_id"]
    state["bulletin_published_at"] = current["published_at"]
    state["source_id"] = source["id"]
    state["source_name"] = news_sources.display_name(source)

    # 7. BOUNDARY. A too-early boundary is not a problem: keep watching and
    # let the music play on to the next one.
    if not eligible:
        state["state"] = WATCHING
        state["reason"] = why
        state["intended_boundary_item_id"] = None
        state["estimated_air_at"] = None
        save_state(state)
        return {"state": WATCHING, "reason": why}

    state["state"] = ARMED
    # boundary_eligible on time, boundary_late_grace past the on-time
    # window: the bulletin was just found fresh AT this boundary (step 6).
    state["reason"] = why
    state["intended_boundary_item_id"] = (q.get("current_item") or {}).get(
        "queue_item_id")
    state["estimated_air_at"] = _iso(air_at)
    save_state(state)
    # Armed and the boundary is ours to ask for: arbitrate, speak, insert.
    return _go_on_air(state, session, queue_id, q, local_now, now_dt, deps,
                      source_cfg, current, db_path)


#: States in which a slot was still being worked when its time ran out. A
#: slot that never reached its check window (before_check_window) was never
#: attempted and is not a loss worth a row.
_WORKING = (WATCHING, REFRESHING, ARMED, RENDERING, READY)
#: How soon after a slot's expiry the planner must be looking for the slot
#: to count as having run out WHILE it was being worked. A playing station
#: is evaluated at least every POSITION_FEED_S (30 s) and at every track
#: start; a first look minutes later means nobody was listening at expiry.
EXPIRED_RECORD_GRACE_S = 120.0


def _record_expired_slot(state: dict[str, Any], next_hour_key: str,
                         now_dt: datetime, local_now: datetime,
                         session: dict[str, Any] | None,
                         bulletins: dict[str, Any] | None,
                         db_path=None) -> bool:
    """File the slot the planner is about to move on from, if it ran out
    unaired while still being worked: 'expired:<what it was waiting on>'.

    Without this a slot lost to a closed gate, an unknown airtime, a
    boundary another programme owned or a feed that never produced a fresh
    bulletin simply vanished when the next slot was claimed -- no history
    row, no reason.
    Keyed by the slot's own identity, so news_history keeps its rule that a
    'failed' row (the feed fault) outranks the 'skipped' consequence.

    Only a slot that ran out while THIS session was listening is a loss:
    the same session, looking within EXPIRED_RECORD_GRACE_S of the expiry.
    A station stopped mid-slot (no session, so no passes), a session that
    starts hours later, or a source edited mid-slot leaves the old claim
    behind silently -- the active-listening gate applies to the record too.
    Best-effort: a history write never decides anything."""
    old = state.get("target_hour")
    if not old or old == next_hour_key:
        return False
    if not session or state.get("session_id") != session.get("session_id"):
        return False
    old_target = _parse_local(state.get("target_at"))
    if old_target is None:
        return False
    late_by = (_utc(local_now) - _utc(expiry_for(old_target))).total_seconds()
    if not 0 < late_by <= EXPIRED_RECORD_GRACE_S:
        return False
    if state.get("state") not in _WORKING:
        return False
    if state.get("last_completed_target_hour") == old:
        return False
    waiting = state.get("reason") or ""
    if not waiting or waiting == "before_check_window":
        return False
    reason = "expired:" + waiting
    source_id = state.get("source_id") or (old.split("#", 1)[1]
                                           if "#" in old else None)
    source = news_sources.by_id(bulletins, source_id) if source_id else None
    logger.info("news slot %s expired unaired (%s)", old, waiting)
    _record_history(state, news_history.SKIPPED, now_dt, reason=reason,
                    source=({"id": source_id,
                             "name": news_sources.display_name(source)}
                            if source else {"id": source_id}),
                    db_path=db_path)
    return True


def _record_history(state: dict[str, Any], result: str, now_dt: datetime,
                    *, reason: str | None = None, aired_at: Any = None,
                    source: dict[str, Any] | None = None,
                    target_at: Any = None, db_path=None) -> None:
    """File one outcome in the Newsroom's own record.

    Keyed by the EVENT identity (target_hour already carries source and
    hour), so the planner re-observing the same skipped hour on ten
    consecutive ticks writes one row, and an hour first filed as failed and
    later aired becomes the airing rather than sitting there as both.

    Best-effort by construction: news_history.record never raises, and
    nothing here is allowed to affect what goes to air.
    """
    try:
        token = state.get("bulletin_token")
        title = duration = None
        if token:
            try:
                import news_bulletins
                row = news_bulletins.get_bulletin(token, db_path) or {}
                # `get_bulletin` returns the TABLE ROW, whose column is
                # `title`; `item_title` is the name the presentation layer
                # gives it (news_bulletins builds the view with
                # "item_title": b["title"]). Reading the view's name off
                # the row yielded None every time, so every history entry
                # recorded an empty episode -- a history that can play a
                # bulletin back but cannot say which one it was.
                # Both spellings are accepted so a caller that hands in the
                # view is right too.
                title = row.get("title") or row.get("item_title")
                duration = row.get("duration_s")
            except Exception:  # noqa: BLE001 - presentation only
                pass
        news_history.record(
            result,
            entry_id=state.get("target_hour") or None,
            source_id=state.get("source_id") or (source or {}).get("id"),
            source_name=(state.get("source_name")
                         or (source or {}).get("name") or ""),
            item_title=title,
            token=token,
            scheduled_for=target_at or state.get("target_at"),
            aired_at=aired_at,
            duration_s=duration,
            reason=reason,
            now=now_dt)
    except Exception:  # noqa: BLE001 - the record must never reach the air
        logger.warning("news history not recorded (airing unaffected)",
                       exc_info=True)


#: How long past its estimated airtime a handed-over block is still
#: legitimately pending (a late boundary, an intro still airing).
LOST_HANDOFF_SLACK_S = 120.0


def _in_flight_lost(state: dict[str, Any], local_now: datetime,
                    now_dt: datetime, db_path=None) -> str | None:
    """Evidence that an event MA was handed will never air, or None.

    QUEUED and INTRO_AIRING are the states with no observed exit of their
    own: they advance only when MA actually plays the items. If the items
    never play -- the listener paused through the window, the queue was
    replaced by a request, MA restarted -- nothing ever moves the state and
    the planner refuses every later hour as 'event_in_flight'. (An event
    queued just before its window and paused inside it would block news
    for the rest of the day.) BULLETIN_AIRING is exempt: it completes
    on the next music start, which needs no queue cooperation.

    Evidence only, never a guess, and nothing until the event is OVERDUE:
    past the on-time window AND past the boundary it was queued for by
    LOST_HANDOFF_SLACK_S (an event queued in the late grace is pending
    until its own boundary has had a chance to air it; one with no
    recorded estimate, until the slot expires). Until then it is
    legitimately pending, and a late start is the existing policy. Then,
    in order:
      - the intro row, the DJ engine's own staleness authority: retired,
        missed, revoked or failed means the engine already declared this
        handoff dead (READY_TTL_MINUTES); completed without the bulletin
        ever starting means MA skipped the bulletin;
      - failing that (no intro, or a row the sweep has not reached yet),
        the same TTL applied to our own queued_at, so an intro-less event
        cannot outlive the policy an intro-bearing one is held to.
    Music is never touched: releasing is Music Brain state only. Items
    that somehow still play later are simply not ours any more -- the
    lifecycle hooks ignore tokens the state does not name."""
    import dj_engine
    if state.get("state") == BULLETIN_AIRING:
        return None
    target_at = _parse_local(state.get("target_at"))
    if target_at is None:
        return "target_lost"
    if local_now <= window_for(target_at)[1]:
        return None
    est = _service_naive(state.get("estimated_air_at"))
    if est is None:
        if local_now <= expiry_for(target_at):
            return None
    elif now_dt <= est + timedelta(seconds=LOST_HANDOFF_SLACK_S):
        return None
    intro = state.get("intro_break_id")
    if intro:
        brk = dj_engine.get_break(intro, db_path)
        if brk is None:
            return "intro_retired"
        if brk["state"] in ("missed", "revoked", "failed"):
            return "intro_" + brk["state"]
        if brk["state"] == "completed":
            return "bulletin_never_aired"
    queued_at = _service_naive(state.get("queued_at"))
    if queued_at is None:
        return "queued_at_unknown"
    age_min = (now_dt - queued_at).total_seconds() / 60.0
    if age_min > dj_engine.READY_TTL_MINUTES:
        return "handoff_ttl_expired"
    return None


# --- boundary arbitration -----------------------------------------------------

def _queued_elsewhere(token: str, q: dict[str, Any] | None,
                      enqueued_at: Any = None) -> bool:
    """Is this QUEUED programme item provably NOT at the boundary News is
    asking about? True only when Music Assistant positively names a
    different item as next (and the item is not the one playing): then the
    row sits somewhere further along the queue and owns some later
    boundary, not this one. No queue evidence, or no named next item, is
    'not proven' -- the row keeps the boundary exactly as before.

    A row handed to MA less than dj_engine.QUEUED_GRACE_S ago is never
    'elsewhere': try_insert claims 'queued' BEFORE its MA call, and this
    pass runs on its own worker beside the DJ's, so a queue read taken in
    between still names the old next item. The age is measured from the
    row's own `enqueue` stamp -- written by try_insert with the wall clock
    immediately before the insert -- never from updated_at, which carries
    the DJ pass's start time and can be minutes older than the claim. No
    stamp, or an unreadable one, is young. That is the same grace the
    engine's own missed-break reconciliation gives a young queued row."""
    if not isinstance(q, dict):
        return False
    import dj_engine
    bounds = service_clock.age_bounds(enqueued_at, service_clock.legacy_now())
    if bounds is None or bounds[0] < dj_engine.QUEUED_GRACE_S:
        return False
    nxt = q.get("next_item")
    if not isinstance(nxt, dict) or not nxt.get("uri"):
        return False
    import media_ref
    if media_ref.programming_ref(nxt.get("uri"))[1] == token:
        return False
    cur = q.get("current_item")
    if isinstance(cur, dict) and \
            media_ref.programming_ref(cur.get("uri"))[1] == token:
        return False
    return True


def arbitrate_boundary(queue_id: str, our_intro_token: str | None = None,
                       now: datetime | None = None,
                       db_path=None,
                       q_fn: Callable[[], dict[str, Any]] | None = None
                       ) -> tuple[bool, str, list[str]]:
    """May News take the upcoming boundary? -> (ok, reason, revoked_tokens).

    The policy, in priority order:

      - An explicit EVENT LINER (a station request the listener actually
        asked for) always wins its boundary. News waits for the next one.
      - An ordinary SCHEDULED break that is only planned/rendering/ready has
        nothing committed to MA yet, so News revokes it and it replans.
      - An ordinary break already QUEUED or AIRING is inside Music
        Assistant's buffer. It is not removed and not fought: News
        reevaluates a later boundary, and skips the hour if none remains.
      - A calendar ANNOUNCEMENT that is only planned/rendering/ready does
        NOT hold the boundary against News. It is deferred, not revoked:
        its own pass already waits whenever News owns the boundary
        (announcements._boundary_blocked -> ordinary_dj_block_reason), so
        it simply keeps its rendered audio and airs at a later safe
        boundary inside its late window -- no re-render, no lost attempt.
        Treating it as an event liner here made each side wait for the
        other until the bulletin's hour was lost. Once QUEUED or AIRING
        it is in Music Assistant's hands and owns the boundary like any
        speech already there.

    THE BOUNDARY, NOT THE QUEUE. "Queued" says a
    row was handed to MA, not WHERE it is. `q_fn` reads the queue NOW --
    after the rows, never a read taken before a feed download -- and only
    when a queued row needs it; when MA positively names a different next
    item, a queued row is somewhere further along and owns a later
    boundary, not this one. A break inserted earlier can be left many
    items ahead by an MA rewind, stay 'queued', and refuse News at every
    in-window boundary in a row. Without `q_fn`, an
    unreadable queue or no named next item, the old answer stands. An
    AIRING row always owns.

    All-or-nothing: the refusals are decided before anything is revoked,
    so a boundary News is refused never costs the DJ a rendered break.

    Nothing here touches transport. Revocation is Music Brain state only.
    """
    import dj_engine
    now_dt = now or service_clock.legacy_now()
    revoked: list[str] = []
    to_revoke: list[str] = []
    c = dj_engine._connect(db_path)
    try:
        rows = c.execute(
            "SELECT token, state, telemetry FROM dj_breaks "
            "WHERE queue_id = ? "
            f"AND state IN ({','.join('?' for _ in dj_engine.NON_TERMINAL_STATES)})",
            (queue_id, *dj_engine.NON_TERMINAL_STATES)).fetchall()
    finally:
        c.close()
    read: dict[str, Any] = {}

    def _q() -> dict[str, Any] | None:
        if "q" not in read:
            try:
                read["q"] = q_fn() if q_fn is not None else None
            except Exception:  # noqa: BLE001 - unreadable proves nothing
                read["q"] = None
        return read["q"]

    for token, brk_state, tel_raw in rows:
        if token == our_intro_token:
            continue                     # our own line
        try:
            tel = json.loads(tel_raw or "{}")
        except (TypeError, ValueError):
            tel = {}
        purpose = tel.get("purpose") or dj_engine.PURPOSE_SCHEDULED
        if purpose == dj_engine.PURPOSE_NEWS_INTRO:
            continue                     # a stale intro of ours; harmless
        if brk_state == "queued" and _queued_elsewhere(
                token, _q(), (tel or {}).get("enqueue")):
            continue                     # handed to MA, but not HERE
        if purpose == dj_engine.PURPOSE_ANNOUNCEMENT:
            if brk_state in ("queued", "airing"):
                return False, "announcement_owns_boundary", revoked
            continue                     # deferred: it waits for News
        if purpose != dj_engine.PURPOSE_SCHEDULED:
            # Explicit listener interaction. News never preempts it.
            return False, "event_liner_owns_boundary", revoked
        if brk_state in ("queued", "airing"):
            return False, "ordinary_break_owns_boundary", revoked
        to_revoke.append(token)
    for token in to_revoke:
        if dj_engine.revoke(token, now_dt, db_path,
                            reason="news_boundary_owned"):
            revoked.append(token)
    return True, "boundary_available", revoked


# --- the final gate -----------------------------------------------------------

def final_insertion_gate(state: dict[str, Any], session: dict[str, Any],
                         queue_id: str, q: dict[str, Any],
                         local_now: datetime, now_dt: datetime,
                         bulletins: dict[str, Any],
                         db_path=None) -> tuple[bool, str]:
    """Everything re-checked in the last moment before MA is mutated.

    Between arming and inserting, real seconds passed while the intro was
    synthesized. Any of these may have changed underneath us, and each one
    makes the insertion wrong rather than merely late."""
    import dj_engine
    import ma_queue
    import news_bulletins

    if not _enabled(bulletins):
        return False, "disabled_mid_event"
    if not session or session.get("session_id") != state.get("session_id"):
        return False, "session_changed"
    if queue_id != state.get("queue_id"):
        return False, "queue_changed"
    if (q.get("state") or "") != "playing":
        return False, "not_playing"
    item = (q.get("current_item") or {}).get("queue_item_id")
    if state.get("intended_boundary_item_id") and \
            item != state["intended_boundary_item_id"]:
        return False, "boundary_item_changed"
    air_at = dj_engine.estimated_air_at(q, None, now_dt)
    if air_at is None:
        return False, "airtime_unknown"
    target_at = _parse_local(state.get("target_at"))
    if target_at is None:
        return False, "target_lost"
    import dj_context
    predicted_local = dj_context.station_project(local_now,
                                                  air_at - now_dt)
    eligible, why = boundary_eligible(predicted_local, target_at)
    if not eligible:
        return False, why
    hour_key = state.get("target_hour") or ""
    if state.get("last_completed_target_hour") == hour_key:
        return False, "target_hour_already_completed"
    token = state.get("bulletin_token")
    if was_bulletin_aired(state, token):
        return False, "bulletin_already_aired"
    current = news_bulletins.current_bulletin(
        db_path, source_id=state.get("bulletin_source_id") or None)
    if not current or current.get("token") != token:
        return False, "bulletin_no_longer_current"
    # Fresh AT the boundary it would now air on. The planner measured this
    # a render ago; inside the late grace a few seconds can be the
    # difference, and "never aired late as stale news" is decided here,
    # at the last moment, on the latest prediction.
    fresh, fresh_why = bulletin_fresh_enough(
        current, predicted_local, state.get("bulletin_source_id") or "")
    if not fresh:
        return False, fresh_why
    if not ma_queue.insertion_gate_open(q):
        return False, "gate_closed"
    return True, "clear"


def _parse_local(iso: Any) -> datetime | None:
    if not isinstance(iso, str) or not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


# --- going on air -------------------------------------------------------------

def _go_on_air(state, session, queue_id, q, local_now, now_dt, deps,
               bulletins, current, db_path):
    """ARMED -> rendering -> ready -> queued, in one pass.

    The intro is rendered through the ordinary finite-audio path and then
    BOTH finite items are handed to MA in a single ordered mutation. The
    bulletin is the product and the intro is polish, so a failed intro
    degrades to airing the bulletin alone rather than losing the hour."""
    import dj_context
    import dj_engine
    import ma_queue
    import news_bulletins

    hour_key = state["target_hour"]

    # 1. Who owns this boundary?
    state_fn = deps.get("state_fn")
    if state_fn is None:
        state_fn = ma_queue.queue_state
    ok, why, revoked = arbitrate_boundary(
        queue_id, state.get("intro_break_id"), now_dt, db_path,
        q_fn=lambda: state_fn(queue_id))
    if not ok:
        # Not ours. Keep watching: a later boundary inside the window may
        # still work, and boundary_too_late will skip the hour when it does not.
        state["state"] = WATCHING
        state["reason"] = why
        state["intended_boundary_item_id"] = None
        save_state(state)
        return {"state": WATCHING, "reason": why}
    if revoked:
        # Named by the slot it was for: this line once said ":30" for every
        # slot, :00 ones included.
        target_at = _parse_local(state.get("target_at"))
        logger.info("news revoked %d ordinary break(s) for the %s slot (%s): "
                    "%s", len(revoked),
                    target_at.strftime("%H:%M") if target_at else "?",
                    hour_key, ", ".join(t[:8] for t in revoked))

    # 2. The intro line. Deterministic text, ordinary TTS path.
    intro_token = state.get("intro_break_id")
    intro_ready = False
    if intro_token:
        brk = dj_engine.get_break(intro_token, db_path)
        intro_ready = bool(brk and brk["state"] == "ready")
    if not intro_token:
        import news_intro
        air_at = dj_engine.estimated_air_at(q, None, now_dt)
        clock = None
        if air_at is not None:
            facts = dj_context.service_facts(now_dt, at=air_at)
            clock = facts.get("station_time")
        if clock and air_at is not None and (
                air_at - now_dt).total_seconds() > dj_engine.CLOCK_MAX_DRIFT_S:
            # The intro goes into Music Assistant NOW, for a boundary that
            # may be many minutes off (e.g. rendered and queued minutes
            # before a late-grace slot, behind a long track, with the time
            # projected to that track's end). Unlike an ordinary break it
            # is never revalidated before air (pre_air_revalidate), so a skip
            # of that long track would air the time early. A projection
            # further ahead than the drift the station tolerates for any
            # spoken clock cannot be vouched for: the clockless family.
            clock = None
        if clock:
            # The station-wide clock invariant (service presence): a
            # music break stated the time inside the gap, so this line uses
            # its clockless family. The intro still airs; only the clock is
            # left out, and a clockless intro records no clock_aired.
            # Measured to air_at -- when this line will be HEARD, the same
            # projection the clock above was written to -- never to the
            # render instant, which sits a boundary earlier (otherwise an
            # intro airing minutes after a music clock goes clockless
            # because at render that clock was seconds old). air_at is
            # non-None here: a clock exists only when it was projected.
            import service_presence
            if service_presence.clock_suppressed(air_at or now_dt,
                                                 queue_id, db_path):
                clock = None
        import settings as _settings
        text = news_intro.intro_text(
            clock, (current or {}).get("source_label"), hour_key,
            station_name=_settings.resolved_station_name())
        intro_token = dj_engine.create_news_intro(
            queue_id, {"news_token": state.get("bulletin_token"),
                       "target_hour": hour_key,
                       "target_at": state.get("target_at"),
                       "estimated_air_at": _iso(air_at) if air_at else None},
            session.get("session_id", ""), now_dt, db_path)
        state["intro_break_id"] = intro_token
        state["state"] = RENDERING
        state["reason"] = "rendering_intro"
        save_state(state)
        intro_ready = dj_engine.render_news_intro(
            intro_token, text, deps, now_dt, db_path)
        if not intro_ready:
            # Polish failed; the product still stands.
            logger.warning("news intro render failed; bulletin will air alone")
            state["intro_break_id"] = None
            state["last_result"] = "intro_render_failed"
            state["last_result_at"] = _iso(now_dt)

    # 3. Re-read the world: rendering took real seconds.
    try:
        state_fn = deps.get("state_fn")
        if state_fn is None:
            state_fn = ma_queue.queue_state
        q = state_fn(queue_id)
    except Exception:
        state["state"] = WATCHING
        state["reason"] = "queue_unreadable"
        save_state(state)
        return {"state": WATCHING, "reason": "queue_unreadable"}

    ok, why = final_insertion_gate(state, session, queue_id, q, local_now,
                                   now_dt, bulletins, db_path)
    if not ok:
        terminal = why in ("boundary_too_late", "target_hour_already_completed",
                           "bulletin_already_aired", "disabled_mid_event")
        if intro_token:
            dj_engine.revoke(intro_token, now_dt, db_path, reason=why)
            state["intro_break_id"] = None
        if terminal:
            state = mark_target_skipped(state, hour_key, why, now_dt)
        else:
            state["state"] = WATCHING
            state["reason"] = why
            state["intended_boundary_item_id"] = None
        save_state(state)
        if why in ("boundary_too_late", "disabled_mid_event"):
            # A slot given up at the last moment is a skip like any other:
            # it gets its row, not just a state the next slot overwrites.
            # (The other terminal reasons mean the slot or its bulletin
            # already aired: that row stands.)
            _record_history(state, news_history.SKIPPED, now_dt, reason=why,
                            db_path=db_path)
        return {"state": state["state"], "reason": why}

    # 4. Intro then bulletin, or bulletin alone, as the NEXT items.
    intro_url = (dj_engine.audio_url(intro_token)
                 if intro_token and intro_ready else None)
    bulletin_url = news_bulletins.audio_url(state["bulletin_token"])
    urls = ([intro_url] if intro_url else []) + [bulletin_url]

    # SHUFFLE SPLITS A MULTI-ITEM "NEXT". On a shuffle-enabled queue MA
    # 2.10.3 plays the FIRST item of a play-next batch next and shuffles the
    # rest of the batch into the tail (queue_loader: `shuffle =
    # queue.shuffle_enabled and len(queue_items) > 1` -> _load_pinned_first).
    # Unhandled, the intro airs on time and the bulletin lands at a random
    # later position. A ONE-item "next" is never shuffled (the
    # same condition), which is why ordinary breaks were unaffected. So under
    # shuffle the block goes in as two one-item inserts in REVERSE order --
    # bulletin, then intro -- each landing directly after the committed
    # item, which leaves intro, bulletin. The listener's shuffle setting is
    # read, never written. Without shuffle the single atomic mutation below
    # is unchanged.
    ordered = bool(intro_url) and q.get("shuffle_enabled") is True

    # Claim the intro BEFORE the enqueue so a concurrent pass cannot insert
    # it twice, exactly like try_insert does for an ordinary break.
    if intro_token and intro_ready:
        if not dj_engine._transition(intro_token, ("ready",), "queued",
                                     now_dt, db_path):
            state["state"] = WATCHING
            state["reason"] = "intro_claim_lost"
            save_state(state)
            return {"state": WATCHING, "reason": "intro_claim_lost"}

    insert_many = deps.get("insert_many_fn") or ma_queue.insert_next_many
    try:
        insert_many(queue_id, [bulletin_url] if ordered else urls)
    except Exception as exc:  # noqa: BLE001
        logger.warning("news enqueue failed (music untouched): %s", exc)
        if intro_token:
            dj_engine.revoke(intro_token, now_dt, db_path,
                             reason="news_enqueue_failed")
            state["intro_break_id"] = None
        # A failure MA may have acted on (a timeout, any answer at all) can
        # have left the bulletin in its queue with no event to own it. Only
        # a request that provably never left this host keeps the bulletin
        # servable for a retry inside the window.
        if not ma_queue.enqueue_never_delivered(exc):
            _retire_unaired_bulletin(state, now_dt, db_path,
                                     "news_enqueue_uncertain")
        state["state"] = WATCHING
        state["reason"] = "enqueue_failed"
        state["last_result"] = "enqueue_failed"
        state["last_result_at"] = _iso(now_dt)
        save_state(state)
        return {"state": WATCHING, "reason": "enqueue_failed"}

    if ordered:
        try:
            insert_many(queue_id, [intro_url])
        except Exception as exc:  # noqa: BLE001
            # The bulletin is MA's next item and the intro is not: the
            # product still stands and airs alone, exactly as after a failed
            # intro render. A copy that reached MA anyway answers silence.
            logger.warning("news intro enqueue failed; bulletin airs alone: "
                           "%s", exc)
            dj_engine.revoke(intro_token, now_dt, db_path,
                             reason="news_intro_enqueue_failed")
            intro_token = intro_url = None
            state["intro_break_id"] = None
        else:
            placed = _verify_news_block(queue_id, q, intro_token,
                                        state["bulletin_token"], deps)
            logger.info("news block inserted one item at a time (shuffle on): "
                        "%s", placed)
            if placed == "bulletin_next":
                # The committed item moved between the two inserts, so the
                # intro landed after its own bulletin. The bulletin is next:
                # it airs alone, and the misplaced intro answers silence.
                dj_engine.revoke(intro_token, now_dt, db_path,
                                 reason="news_intro_out_of_order")
                intro_token = intro_url = None
                state["intro_break_id"] = None
            elif placed == "separated":
                # Neither "intro, bulletin" nor "bulletin next": this block
                # cannot air as one. Give the hour up rather than air either
                # half out of place.
                dj_engine.revoke(intro_token, now_dt, db_path,
                                 reason="news_block_separated")
                _retire_unaired_bulletin(state, now_dt, db_path,
                                         "news_block_separated")
                state["intro_break_id"] = None
                state = mark_target_skipped(state, hour_key,
                                            "block_separated", now_dt)
                save_state(state)
                _record_history(state, news_history.SKIPPED, now_dt,
                                reason="block_separated", db_path=db_path)
                return {"state": SKIPPED, "reason": "block_separated"}

    state["state"] = QUEUED
    state["reason"] = "queued"
    state["queued_at"] = _iso(now_dt)
    state["intro_state"] = "queued" if (intro_token and intro_ready) else None
    state["last_result"] = "queued"
    state["last_result_at"] = _iso(now_dt)
    save_state(state)
    return {"state": QUEUED, "reason": "queued", "urls": len(urls),
            "intro": bool(intro_token and intro_ready)}


#: How far past the head the shuffle-path check looks for the block. The
#: two items land directly after the committed item, so a handful is plenty.
BLOCK_VERIFY_AHEAD = 8


def _verify_news_block(queue_id: str, q: dict[str, Any], intro_token: str,
                       bulletin_token: str, deps: dict[str, Any]) -> str:
    """Where the shuffle-path block actually landed, read back from MA.

    'adjacent'       the intro, then its bulletin immediately after it
    'bulletin_next'  the bulletin is the first item after the head and the
                     intro is not directly before it
    'separated'      anything else: the block cannot air as one
    'unverified'     MA could not be read; the lifecycle still owns the
                     event, and a release retires the bulletin

    READ ONLY, anchored on the head's identity (never on arithmetic)."""
    import ma_queue
    import media_ref
    head = (q.get("current_item") or {}).get("queue_item_id")
    if not head:
        return "unverified"
    window_fn = deps.get("window_fn") or ma_queue.read_window_items
    try:
        win = window_fn(queue_id, head, current_index=q.get("current_index"),
                        back=0, ahead=BLOCK_VERIFY_AHEAD)
        after = win["items"][win["current_position"] - win["base_position"] + 1:]
    except Exception:  # noqa: BLE001 - a read failure proves nothing
        return "unverified"

    def ref(item: dict[str, Any]) -> tuple[str, str]:
        media = item.get("media_item")
        uri = media.get("uri") if isinstance(media, dict) else ""
        return media_ref.programming_ref(str(uri or ""))

    refs = [ref(it) for it in after]
    intro_at = refs.index(("break", intro_token)) \
        if ("break", intro_token) in refs else None
    bulletin_at = refs.index(("news", bulletin_token)) \
        if ("news", bulletin_token) in refs else None
    if intro_at is not None and bulletin_at == intro_at + 1:
        return "adjacent"
    if bulletin_at == 0:
        return "bulletin_next"
    return "separated"


def _retire_unaired_bulletin(state: dict[str, Any], now_dt: datetime,
                             db_path, reason: str) -> bool:
    """Make this event's bulletin unservable as news: its URL answers the
    retired-silence tombstone from now on (news_bulletins.audio_available).

    For a bulletin that was handed to MA and will not air as this event --
    released, an uncertain enqueue, a block that could not be kept together.
    A queue item MA still holds for it then plays a short silence instead of
    stale news at full length; nothing in the queue is moved or deleted. A
    bulletin that already started airing is left to completion, which
    retires it the same way."""
    token = state.get("bulletin_token")
    if not token or was_bulletin_aired(state, token):
        return False
    import program_tokens
    if not program_tokens.retire([token], program_tokens.KIND_NEWS, now_dt,
                                 db_path):
        return False
    logger.info("news bulletin %s retired unaired (%s): a stranded queue "
                "item can no longer play it", token[:8], reason)
    return True


def _bulletin_retired(token: str | None, db_path=None) -> bool:
    """Never plan a bulletin the serving gate would answer with silence."""
    if not token:
        return False
    import program_tokens
    return program_tokens.is_retired(token, db_path)


# --- observed lifecycle (driven by real playback, never by a timer) -----------

def on_intro_start(intro_token: str, now: datetime | None = None,
                   db_path=None) -> bool:
    """The DJ's news intro line became the current MA item."""
    now_dt = now or service_clock.legacy_now()
    with _pass_lock:
        state = load_state()
        if state.get("intro_break_id") != intro_token:
            return False
        # ONE AIRING, HOWEVER OFTEN IT IS REPORTED. A repeated start of the
        # same intro keeps the first stamp; one that arrives after its
        # bulletin already started must not drag the event back to
        # intro_airing, where no music start would ever complete it and the
        # release path would retire a bulletin that did air.
        if state.get("state") == INTRO_AIRING:
            return True
        if state.get("state") in (BULLETIN_AIRING, COMPLETED):
            return False
        state["state"] = INTRO_AIRING
        state["reason"] = "intro_airing"
        state["intro_state"] = "airing"
        state["intro_air_start"] = _iso(now_dt)
        save_state(state)
    return True


def bulletin_audio_available(token: str, db_path=None):
    """The serving gate's answer for a bulletin token -- (status, path),
    exactly as news_bulletins.audio_available gives it -- for the playback
    handler's airing verdict.

    A pass-through on purpose: the session store may call the news
    SCHEDULER (it already does, for the bulletin hooks) but must not depend
    on the bulletin store itself, which holds the fetch and download paths
    (test_news_actions, DarknessTests). This is the one door, and it
    decides nothing."""
    import news_bulletins
    return news_bulletins.audio_available(token, db_path)


def on_bulletin_start(bulletin_token: str, now: datetime | None = None,
                      db_path=None) -> bool:
    """The publisher bulletin became the current MA item.

    This is the restart-safe point of no return: duplicate-prevention
    evidence is written NOW, not when the five minutes finish, so a restart
    halfway through a bulletin can never replay it. The linked intro is completed
    here too -- the next item after the intro is the bulletin, not music, so
    waiting for 'a real music track' would strand it airing for minutes."""
    import dj_engine
    now_dt = now or service_clock.legacy_now()
    with _pass_lock:
        state = load_state()
        if state.get("bulletin_token") != bulletin_token:
            # A bulletin we did not queue (manual preview through the queue,
            # or state lost). Never invent an event around it.
            return False
        # The same start reported twice is the same airing: the first air
        # start stands (it is the history row's aired_at). A start after
        # the event completed is not a new airing either -- completion
        # retired the token, and a replay is the serving gate's business.
        if state.get("state") == BULLETIN_AIRING:
            return True
        if state.get("state") == COMPLETED:
            return False
        intro_token = state.get("intro_break_id")
        if intro_token:
            dj_engine._transition(intro_token, ("queued", "airing"),
                                  "completed", now_dt, db_path)
            state["intro_state"] = "completed"
        state["state"] = BULLETIN_AIRING
        state["reason"] = "bulletin_airing"
        state["bulletin_air_start"] = _iso(now_dt)
        state = mark_bulletin_aired(state, bulletin_token,
                                    state.get("target_hour"), now_dt,
                                    state.get("source_id"))
        state["last_result"] = "bulletin_airing"
        state["last_result_at"] = _iso(now_dt)
        save_state(state)
    return True


def on_music_track_start(session: dict[str, Any] | None, queue_id: str,
                         now: datetime | None = None, db_path=None) -> str:
    """A real MUSIC track started. Two jobs, both evidence-driven:

    1. If a bulletin was airing, it has now left the air -- complete it and
       open the post-news quiet guard, baselined on the music history that
       existed BEFORE this track had a chance to finish.
    2. Otherwise, see whether the guard can be released: that requires a
       music row RESOLVED as played beyond the baseline, which this track
       cannot be yet."""
    now_dt = now or service_clock.legacy_now()
    with _pass_lock:
        state = load_state()
        if state.get("state") == BULLETIN_AIRING:
            state["state"] = COMPLETED
            state["reason"] = "bulletin_completed"
            state["bulletin_completed_at"] = _iso(now_dt)
            # DURABLY SPENT, AND NOT ONE MOMENT EARLIER.
            #
            # This is the point at which the bulletin has demonstrably LEFT
            # the air -- a music track has started. Everything before it is
            # too early, and the tempting earlier point is a trap:
            # `aired_tokens` is written by mark_bulletin_aired at the START
            # of airing, so keying spentness off that ring would tombstone a
            # bulletin in the middle of its own broadcast and turn any
            # Flow-mode retry or re-request into silence mid-sentence.
            #
            # The ring stays what it is -- the planner's duplicate guard,
            # and bounded at 24 entries. This is the serving authority, it
            # is unbounded by that ring, and it survives the row.
            token = state.get("bulletin_token")
            if token:
                try:
                    import program_tokens
                    program_tokens.retire([token], program_tokens.KIND_NEWS,
                                          now_dt, db_path)
                except Exception:  # noqa: BLE001 - never block completion
                    logger.warning("news: could not record spent token "
                                   "(transport safety only)", exc_info=True)
            state["last_result"] = "completed"
            state["last_result_at"] = _iso(now_dt)
            sid = (session or {}).get("session_id") or state.get("session_id")
            state["quiet_guard"] = {
                "active": True,
                "session_id": sid,
                "bulletin_token": state.get("bulletin_token"),
                "target_hour": state.get("target_hour"),
                "baseline_seq": _resolved_head_seq(sid, db_path),
                "started_at": _iso(now_dt),
            }
            save_state(state)
            # It provably aired: this transition is driven by MUSIC being
            # observed to start again, not by a timer. That is the only
            # moment worth writing "aired" -- anything earlier is a
            # bulletin that was merely handed to Music Assistant.
            _record_history(state, news_history.AIRED, now_dt,
                            aired_at=state.get("bulletin_air_start"),
                            db_path=db_path)
            return "bulletin_completed"
        if _release_guard_if_earned(state, db_path):
            save_state(state)
            return "quiet_guard_released"
    return "noop"


def _resolved_head_seq(session_id: str | None, db_path=None) -> int:
    """The highest session_tracks seq that has already RESOLVED.

    Deliberately not MAX(seq): record_queue pre-seeds the intended queue as
    'pending' rows, so the plain maximum is somewhere out in the future and
    nothing would ever exceed it. The guard needs the history HEAD -- the
    last track that actually finished -- so that the next track to resolve
    beyond it is genuinely a post-news listen."""
    if not session_id:
        return 0
    import playback
    c = playback._connect(db_path)
    try:
        row = c.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM session_tracks "
            "WHERE session_id = ? AND outcome != 'pending'",
            (session_id,)).fetchone()
        return int(row[0] or 0)
    except Exception:
        return 0
    finally:
        c.close()


def _guard_earned(guard: dict[str, Any], db_path=None) -> bool:
    """Has a COMPLETE post-news music track been observed?

    The evidence is a session_tracks row past the baseline whose outcome is
    'played'. Skipped and partial listens deliberately do not count: a
    listener skipping the first song after the news has not given the
    bulletin any room. News never creates session_tracks rows at all,
    so any row here is real music by construction."""
    sid = (guard or {}).get("session_id")
    if not sid:
        return True                      # nothing to protect
    import playback
    c = playback._connect(db_path)
    try:
        row = c.execute(
            "SELECT 1 FROM session_tracks WHERE session_id = ? AND seq > ? "
            "AND outcome = 'played' LIMIT 1",
            (sid, int(guard.get("baseline_seq") or 0))).fetchone()
        return row is not None
    except Exception:
        return False
    finally:
        c.close()


def _post_news_track_playing(guard: dict[str, Any], db_path=None) -> bool:
    """Is a post-news music track playing now, still unresolved?

    The guard's release evidence is a 'played' row, and that is written
    only when the NEXT track starts -- so a guard that blocked planning
    until release could never let the end of the first post-news song
    carry a break (several songs after the bulletin instead of
    one). While such a track is playing, any
    break planned now can only be placed after it, at its natural end,
    which is exactly "after one complete song". A skip resolves the row
    and clears current_seq, so the guard blocks again until the next
    track starts and the break moves to THAT track's end.

    Read from the session row itself, not a caller's snapshot. Fail closed:
    any doubt answers False and the guard keeps blocking as before."""
    sid = (guard or {}).get("session_id")
    started = service_clock.from_legacy(guard.get("started_at"))
    if not sid or started is None:
        return False
    import playback
    c = playback._connect(db_path)
    try:
        row = c.execute(
            "SELECT t.started_at FROM sessions s JOIN session_tracks t "
            "ON t.session_id = s.session_id AND t.seq = s.current_seq "
            "WHERE s.session_id = ? AND t.seq > ? AND t.outcome = 'pending'",
            (sid, int(guard.get("baseline_seq") or 0))).fetchone()
    except Exception:
        return False
    finally:
        c.close()
    track_started = service_clock.from_legacy(row[0]) if row else None
    return track_started is not None and track_started >= started


def _release_guard_if_earned(state: dict[str, Any], db_path=None) -> bool:
    guard = state.get("quiet_guard")
    if not guard or not guard.get("active"):
        return False
    if not _guard_earned(guard, db_path):
        return False
    state["quiet_guard"] = {**guard, "active": False,
                            "released_at": _iso(service_clock.legacy_now())}
    return True


def bulletin_claims_boundary(queue_id: str | None = None,
                             now: datetime | None = None,
                             bulletins: dict[str, Any] | None = None,
                             db_path=None,
                             boundary_fn: Callable[[], datetime | None] | None = None,
                             ) -> bool:
    """Is a scheduled bulletin legitimately due at the NEXT safe boundary?

    THE ARBITRATION RULE, stated once. A bulletin and an ordinary cadence
    break are two programme elements wanting the same thing: the end of the
    song that is playing. They used to race for it independently and the
    faster one won, which could mean this:

        hh:29-hh:34   the bulletin's air window
        hh:32:40      a safe boundary INSIDE the window
        hh:32:40      the DJ's cadence break (due since ~hh:31) takes it
        hh:32:51      the bulletin planner wakes; the next boundary is
                      ~hh:38, past the window -> hour skipped

    A song is never interrupted, so inside a five-minute window there may
    be exactly one boundary. Whoever asks first wins, and the bulletin --
    which can only air in that window -- loses to a break that could have
    aired at any point in the next hour. So the bulletin has priority and
    the cadence break defers.

    Deliberately NOT done by revoking afterwards: `arbitrate_boundary` can
    already take a boundary back from an ordinary break that is merely
    planned or rendering, but not from one already QUEUED -- by then the
    audio is in Music Assistant's hands and fighting it would mean touching
    the customer's queue. Prevention is the only thing that works at the
    moment it matters, so the deferral happens BEFORE the ordinary break is
    planned at all.

    Bounded on purpose, so a dead feed cannot cost the station its DJ:

      * only inside the air window -- five minutes in an hour, never the
        six-minute check lead as well -- or, in the late grace after it, at
        the one boundary a bulletin the planner already HOLDS would still
        be fresh on (never by the clock, never for a dead feed);
      * only while this hour is still winnable (`may_plan_target_hour`), so
        the moment the bulletin completes or is skipped the block lifts and
        ordinary cadence resumes at the very next boundary;
      * only when bulletins are enabled at all;
      * never when the event is already in flight -- that is
        `news_in_flight`'s job and it means something different.

    THE BOUNDARY, NOT THE CLOCK. "Inside the air window" used to
    be asked of the moment the DJ was evaluated, and that was the wrong
    instant. The DJ does not take a boundary when it arrives: it renders
    up to RENDER_LEAD_S before the end of the track and hands the audio to
    Music Assistant the moment the structural gate is open -- minutes
    before the boundary it is claiming. With a window hh:08-hh:13, a
    cadence break could be QUEUED for an hh:09:45 boundary before hh:08
    had arrived, so this rule saw "hh:07, not in the window" and let it
    through; from hh:08 on the queued item held the gate shut,
    the planner reported gate_closed every tick, never fetched the feed,
    and the hour was skipped -- while `arbitrate_boundary`, which can only
    revoke breaks that are not yet queued, was never reached.

    So the question is asked of the BOUNDARY: the caller may pass
    `boundary_fn`, which returns the station-local time the next safe
    boundary is predicted to land (`dj_engine.estimated_air_at` projected
    onto the station clock, exactly as the planner projects it), or None
    when that cannot be known. A boundary inside the window is the
    bulletin's, however early the DJ is looking at it. It is consulted
    only when the clock itself does not already answer, and only within
    `AIR_ESTIMATE_MAX_S` of the window opening -- the furthest ahead any
    airtime is ever estimated, so a boundary that could land in the
    window cannot be seen from further out and there is nothing to read.
    A caller that cannot supply the boundary keeps the clock rule alone.
    """
    if bulletins is None:
        try:
            import settings
            bulletins = settings.load().get("bulletins") or {}
        except Exception:
            return False
    if not _enabled(bulletins):
        return False
    try:
        import dj_context
        state = load_state()
        if state.get("state") in IN_FLIGHT:
            return False        # news_in_flight already owns this case
        ev_queue = state.get("queue_id")
        if queue_id and ev_queue and ev_queue != queue_id:
            return False        # another queue's event is not our business
        local_now = dj_context.station_now(now or service_clock.legacy_now())
        if local_now is None:
            return False        # no station clock, no :30 to defer to
        # THE SAME whose-turn and THE SAME event identity the planner uses.
        # This used to compute its own target from the legacy
        # schedule.target_minute and key it WITHOUT the source -- so once
        # the planner's state carried a source-qualified hour, the two keys
        # could never match, may_plan_target_hour always said "winnable",
        # and the DJ was deferred for the whole air window of an hour that
        # had already aired or had already been lost. One rule, one place.
        source, target_at = _due_source(bulletins, local_now)
        if source is None or target_at is None:
            return False        # nothing configured: nothing to defer to
        hour_key = target_hour_key(target_at, source["id"])
        may, _why = may_plan_target_hour(state, hour_key)
        if not may:
            return False        # this hour is already done or already lost
        early, late = window_for(target_at)
        expiry = expiry_for(target_at)
        if local_now > expiry:
            return False
        # A listener's explicit request ("DJ after this song") is never held
        # behind News: arbitrate_boundary already yields every boundary to
        # one, so deferring it here made each side wait for the other until
        # the slot was gone -- the announcement deadlock, for manual
        # breaks. News simply takes the next boundary.
        if _listener_request_pending(queue_id or ev_queue, db_path):
            return False
        in_window = early <= local_now <= late
        if boundary_fn is None:
            return in_window    # the clock rule alone
        import dj_engine
        horizon = timedelta(seconds=dj_engine.AIR_ESTIMATE_MAX_S)
        if local_now < early - horizon:
            return False        # no estimate can reach the window from here
        boundary = boundary_fn()
        if boundary is None:
            return in_window    # unknowable boundary: the clock rule stands
        if boundary <= late:
            # An on-time boundary is the bulletin's however early the DJ is
            # looking; one before the window is the DJ's.
            return in_window or boundary >= early
        if boundary <= expiry:
            # THE LATE GRACE is the bulletin's only when News HOLDS one that
            # will still be fresh there -- asked even inside the on-time
            # window, so a dead feed does not spend the DJ's render lead on
            # a boundary News cannot use. Never by the clock alone: a dead
            # feed must keep costing the DJ at most the on-time window.
            return _holds_fresh_bulletin(state, hour_key, boundary, db_path)
        return False
    except Exception:
        # Arbitration is a nicety; failing it must never silence the DJ.
        logger.debug("bulletin boundary claim check failed", exc_info=True)
        return False


def _listener_request_pending(queue_id: str | None, db_path=None) -> bool:
    """Is an explicit listener request (a manual 'DJ after this song', a
    station request liner) planned or in flight on this queue? Exactly the
    rows arbitrate_boundary answers with event_liner_owns_boundary."""
    if not queue_id:
        return False
    import dj_engine
    c = dj_engine._connect(db_path)
    try:
        rows = c.execute(
            "SELECT telemetry FROM dj_breaks WHERE queue_id = ? "
            f"AND state IN ({','.join('?' for _ in dj_engine.NON_TERMINAL_STATES)})",
            (queue_id, *dj_engine.NON_TERMINAL_STATES)).fetchall()
    finally:
        c.close()
    for (tel_raw,) in rows:
        try:
            purpose = (json.loads(tel_raw or "{}") or {}).get("purpose")
        except (TypeError, ValueError, AttributeError):
            continue            # unreadable: not proof of a request
        purpose = purpose or dj_engine.PURPOSE_SCHEDULED
        if purpose not in (dj_engine.PURPOSE_SCHEDULED,
                           dj_engine.PURPOSE_NEWS_INTRO,
                           dj_engine.PURPOSE_ANNOUNCEMENT):
            return True
    return False


def _holds_fresh_bulletin(state: dict[str, Any], hour_key: str,
                          boundary_local: datetime, db_path=None) -> bool:
    """Does the planner hold, for THIS slot, a bulletin that would still be
    fresh at `boundary_local`? The planner's own evidence only: the slot it
    claimed, the token it chose after the freshness check, and the same
    cached row and age rule every airing passes."""
    if state.get("target_hour") != hour_key:
        return False
    if state.get("state") not in (WATCHING, ARMED, RENDERING, READY):
        return False
    token = state.get("bulletin_token")
    if not token or was_bulletin_aired(state, token):
        return False
    import news_bulletins
    current = news_bulletins.current_bulletin(
        db_path, source_id=state.get("bulletin_source_id") or None)
    if not current or current.get("token") != token:
        return False
    fresh, _why = bulletin_fresh_enough(current, boundary_local,
                                        state.get("bulletin_source_id") or "")
    return fresh


def ordinary_dj_block_reason(session: dict[str, Any] | None,
                             queue_id: str | None = None,
                             db_path=None,
                             now: datetime | None = None,
                             boundary_fn: Callable[[], datetime | None] | None = None,
                             current_is_music_fn: Callable[[], bool] | None = None,
                             ) -> str:
    """Why ordinary SCHEDULED DJ speech may not run right now.

        "news_in_flight"  the news event is committed to the player queue
                          (queued / intro_airing / bulletin_airing). Without
                          this, a cadence-due break CREATED during the
                          bulletin takes the boundary at its final word and
                          speaks before the first post-news song. Once the
                          pair is in MA's hands, no new ordinary break may be
                          planned, rendered, readied or queued -- the event
                          owns the air until real music takes over.
        "quiet_guard"     the post-news one-complete-song protection.
        "bulletin_due"    a scheduled bulletin is inside its air window (or
                          holds a fresh bulletin for a boundary in its late
                          grace) and this hour is still winnable, so the
                          next safe boundary is the bulletin's. The cadence
                          break is
                          deferred rather than outrunning it to a boundary
                          that only the bulletin needs; see
                          bulletin_claims_boundary.
        ""                nothing blocks; ordinary scheduling proceeds.

    TWO DIFFERENT CONCEPTS, deliberately not conflated: the first means "the
    news event owns the air", the second means "the bulletin just ended and
    music has not yet earned back the microphone". The persisted quiet_guard
    is never redefined to cover in-flight states.

    `now` is the caller's own service clock, not this module's. The deferral
    below depends on where the station clock is, and a scheduler driven by
    an injected clock that then asked the wall clock what time it was would
    answer two different questions in one pass.

    `boundary_fn`, when the caller can supply one, answers where the NEXT
    safe boundary lands on the station clock (None when unknowable). The
    bulletin_due deferral is decided against that boundary rather than
    against the clock alone -- see bulletin_claims_boundary.

    `current_is_music_fn`, when the caller can supply one, answers from the
    caller's own live queue read whether Music Assistant's current item is
    real music; it is the second half of the evidence quiet_guard_blocks
    needs before it lets the first post-news song's end be planned for.

    The in-flight block is read from persisted state alone -- deliberately
    NOT from bulletins.enabled. Disabling news mid-event lets the queued
    pair air (the on-air contract), so the block must survive enabled=false for as
    long as the event does. The transition to quiet_guard is gapless by
    construction: on_music_track_start writes state=completed and the active
    guard in ONE save under the pass lock, so every reader sees in-flight or
    guard, never a hole between them.
    """
    try:
        state = load_state()
    except Exception:
        return ""
    if state.get("state") in IN_FLIGHT:
        ev_queue = state.get("queue_id")
        # Scoped to the event's own queue when both sides are known; an
        # unknown either way blocks conservatively -- silence is cheaper
        # than talking over the news.
        if not queue_id or not ev_queue or ev_queue == queue_id:
            return "news_in_flight"
    if quiet_guard_blocks(session, db_path=db_path,
                          current_is_music_fn=current_is_music_fn):
        return "quiet_guard"
    if bulletin_claims_boundary(queue_id, now=now, db_path=db_path,
                                boundary_fn=boundary_fn):
        return "bulletin_due"
    return ""


def quiet_guard_blocks(session: dict[str, Any] | None, db_path=None,
                       current_is_music_fn: Callable[[], bool] | None = None
                       ) -> bool:
    """Is ordinary SCHEDULED DJ speech currently held back?

    Checked by dj_engine.schedule_pass before it plans, renders or inserts
    anything, so no ordinary break is even prepared to sit behind the
    bulletin. Event liners are not consulted -- an explicit listener request
    is not the DJ chattering. Releases itself the moment the evidence
    arrives, so a stale guard cannot silence the station."""
    try:
        state = load_state()
    except Exception:
        return False
    guard = state.get("quiet_guard")
    if not guard or not guard.get("active"):
        return False
    sid = (session or {}).get("session_id")
    if sid and guard.get("session_id") and sid != guard["session_id"]:
        return False                     # a different session; not ours to hold
    if _guard_earned(guard, db_path):
        with _pass_lock:
            fresh = load_state()
            if _release_guard_if_earned(fresh, db_path):
                save_state(fresh)
        return False
    # The guard stays active; it lets the END of the post-news song now
    # playing be planned for (_post_news_track_playing) -- but only
    # when Music Assistant agrees that song is what is on air. A read that
    # still shows the bulletin would otherwise place the break straight
    # after it. No such evidence, or any doubt, keeps the guard blocking.
    if current_is_music_fn is None or not _post_news_track_playing(guard,
                                                                   db_path):
        return True
    try:
        return not current_is_music_fn()
    except Exception:  # noqa: BLE001 - unreadable is not music
        return True


# --- active bulletin protection ----------------------------------------------

def bulletin_locked(token: str | None = None) -> tuple[bool, str]:
    """Is the cached bulletin committed to air right now?

    The one-slot cache keeps exactly ONE publisher MP3, and activating a
    replacement expires the old row and deletes its file. That is fine
    until the old token is the one Music Assistant is about to play or is
    playing: its URL would start answering 410 mid-event. While the event
    is queued or airing, refreshes -- automatic or manual -- are refused
    rather than allowed to pull the artifact out from under MA."""
    try:
        state = load_state()
    except Exception:
        return False, ""
    if state.get("state") not in IN_FLIGHT:
        return False, ""
    active = state.get("bulletin_token")
    if token is not None and active is not None and token != active:
        return False, ""
    return True, active or ""

# --- status -------------------------------------------------------------------

def schedule_status(bulletins: dict[str, Any] | None = None,
                    now: datetime | None = None) -> dict[str, Any]:
    """The schedule block for /news/status. Read-only arithmetic only: it
    computes the theoretical next target even while disabled, and never
    fetches, downloads, writes state or exposes a filesystem path."""
    now_dt = now or service_clock.legacy_now()
    if bulletins is None:
        try:
            import settings
            bulletins = settings.load().get("bulletins") or {}
        except Exception:
            bulletins = {}
    enabled = _enabled(bulletins)
    minute = target_minute(bulletins)
    state = load_state()
    guard = state.get("quiet_guard") or {}
    out: dict[str, Any] = {
        "target_minute": minute,
        "source_id": None,
        "source_name": None,
        # An event already handed to MA keeps reporting its real state even
        # if News is switched off mid-event: it is going to air either way,
        # and saying "disabled" while the bulletin is playing would be a lie.
        "automation": (state.get("state") if state.get("state") in IN_FLIGHT
                       else (DISABLED if not enabled
                             else (state.get("state") or IDLE))),
        "state": state.get("state"),
        "reason": state.get("reason"),
        "target_hour": state.get("target_hour"),
        "target_at": None,
        "window_start": None,
        "window_end": None,
        # The slot's last chance: after the on-time window a fresh bulletin
        # may still take a boundary until this moment (SLOT_EXPIRY_S).
        "expires_at": None,
        "intended_boundary_item_id": state.get("intended_boundary_item_id"),
        "estimated_air_at": state.get("estimated_air_at"),
        "intro_break_id": state.get("intro_break_id"),
        "intro_state": state.get("intro_state"),
        "queued_at": state.get("queued_at"),
        "bulletin_token": state.get("bulletin_token"),
        "bulletin_published_at": state.get("bulletin_published_at"),
        "bulletin_air_start": state.get("bulletin_air_start"),
        "bulletin_completed_at": state.get("bulletin_completed_at"),
        "last_result": state.get("last_result"),
        "last_result_at": state.get("last_result_at"),
        "last_completed_target_hour": state.get("last_completed_target_hour"),
        "duplicate_block_reason": None,
        "quiet_guard": {
            "active": bool(guard.get("active")),
            "session_id": guard.get("session_id"),
            "bulletin_token": guard.get("bulletin_token"),
            "target_hour": guard.get("target_hour"),
            "baseline_seq": guard.get("baseline_seq"),
            "started_at": guard.get("started_at"),
        },
    }
    try:
        import dj_context
        local_now = dj_context.station_now(now_dt)
    except Exception:
        local_now = None
    if local_now is not None:
        # THE SAME WHOSE-TURN RULE THE PLANNER USES. This block
        # read the legacy `schedule.target_minute` -- the pre-multi-source
        # field, clamped to {0, 30} -- while `_plan` picks the due source
        # through news_sources.due_source and works toward THAT source's
        # minute. With two sources on different minutes past the hour, the
        # planner worked toward one of them while this reported xx:30, a window
        # matching no configured source, on the one sentence a person
        # reads to know when the news is next. The hero line and the
        # source list were right because they are derived from
        # `sources[]`, which made the disagreement harder to notice rather
        # than easier.
        #
        # The legacy field is still the answer when no source is
        # configured -- a document that has not been migrated yet, and the
        # factory state -- so nothing that predates sources changes.
        def _slot(at):
            src, when = _due_source(bulletins, at)
            if src is None or when is None:
                return None, active_target(at, minute)
            return src, when

        def _block(src, when):
            # Keyed WITH the source, exactly as the planner keys it. A
            # sourceless key could never match a source-qualified stored
            # one, so this block could only ever report "not blocked".
            return may_plan_target_hour(
                state, target_hour_key(when, src["id"] if src else None))

        source, target_at = _slot(local_now)
        may, why = _block(source, target_at)
        if not may and local_now > window_for(target_at)[1]:
            # Past its on-time window a slot that already aired (or was
            # skipped) is not the "next bulletin" any more, even though the
            # planner keeps its identity until expiry: report the slot after
            # it, as this did when the slot ended at the window's close.
            source, target_at = _slot(expiry_for(target_at)
                                      + timedelta(seconds=1))
            may, why = _block(source, target_at)
        if source is not None:
            minute = source["target_minute"]
            out["target_minute"] = minute
            out["source_id"] = source["id"]
            out["source_name"] = news_sources.display_name(source)
        early, late = window_for(target_at)
        out["target_at"] = target_at.isoformat()
        out["window_start"] = early.isoformat()
        out["window_end"] = late.isoformat()
        out["expires_at"] = expiry_for(target_at).isoformat()
        if not may:
            out["duplicate_block_reason"] = why
    return out
