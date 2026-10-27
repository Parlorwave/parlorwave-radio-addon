"""Calendar-driven station announcements.

A calendar event whose summary begins with ANNOUNCE (or ANNOUNCE EXACT)
is not DJ context -- it is intentional, operator-authored, finite station
programming. This module owns the OCCURRENCE side of that programming:
identity, durable lifecycle, timing, and the decision to commit one to
air. Everything downstream of that decision deliberately reuses the
trusted finite-program machinery: an announcement on the air IS a
dj_breaks row (purpose "announcement"), with the same token, audio
serving, atomic claims, boundary gate, tombstones, queue-hygiene
ownership and completion evidence as every break.

OCCURRENCE IDENTITY. Home Assistant's supported `calendar.get_events`
service returns only {start, end, summary, description, location} for
every provider -- Core filters the response through LIST_EVENT_FIELDS,
so a provider UID or recurrence id is NEVER available on this path
(verified against Core 2026.8.3). Identity is therefore the strongest
material that actually exists: the SOURCE CALENDAR plus the OCCURRENCE
START INSTANT. Editing the summary of an occurrence keeps its identity,
so an already-aired announcement whose wording is touched up afterwards
can never air a second time; the trade is that two distinct ANNOUNCE
events on the SAME calendar at the SAME instant collide (the first
observed wins, the collision is logged). Recurring events differ per
occurrence by start, so every instance is distinct.

TIMING. An occurrence becomes eligible at its scheduled start, waits for
a safe natural music boundary (the same structural Flow-mode gate every
break uses -- active music is never interrupted), and may air late up to
station.announcements.late_air_minutes. Past that it expires and stays
expired: stale copy never airs hours later.

DURABILITY. Occurrences live in BASE_DIR/announcements_state.json,
written atomically. A restart before air keeps the occurrence; a restart
while committed re-adopts the outcome from the dj_breaks row (or, when
the row has been reaped, from the retired-token ledger); a spent or
expired occurrence never comes back. In-memory state decides nothing.

The integration OBSERVES the calendars and posts what it saw; a feed
that is temporarily unreadable removes nothing here (an occurrence is
cancelled only when its source reported HEALTHY and the event is gone
while its start is still in the future -- that is an operator deletion,
not an outage)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import announce_copy
import paths
import service_clock

logger = logging.getLogger("musicbrain")

STATE_VERSION = 1

# --- occurrence lifecycle ----------------------------------------------------
#
# scheduled  observed, parsed, waiting for its start (or already inside
#            its late window and waiting for a boundary/commit)
# committed  a dj_breaks row exists for it; the row's own lifecycle
#            (planned/rendering/ready/queued/airing) is the live truth
# spent      proof of air (or of retirement): never airs again
# expired    the late window closed, or the outcome became unprovable:
#            never airs
# invalid    a directive that cannot air (empty payload, all-day/no-time,
#            payload too long); surfaced to the operator, never silent
SCHEDULED = "scheduled"
COMMITTED = "committed"
SPENT = "spent"
EXPIRED = "expired"
INVALID = "invalid"

TERMINAL_STATES = (SPENT, EXPIRED)

#: Late-air tolerance bounds (minutes); the default lives in settings.
LATE_AIR_MIN = 5
LATE_AIR_MAX = 120

#: One sync may carry at most this many directive events; one store may
#: hold at most OCC_CAP occurrences (newest starts win). Both are
#: safety valves, not expected operating points.
SYNC_EVENT_CAP = 64
OCC_CAP = 400
#: Terminal occurrences are kept for dedupe and Air Check context, then
#: pruned once their start is comfortably in the past.
RETAIN_TERMINAL_DAYS = 14

#: A commit whose render/insert keeps failing is retried across passes,
#: but never unboundedly inside one window.
MAX_ATTEMPTS = 3

INVALID_ALL_DAY = "all_day_unsupported"

# --- calendar selection --------------------------------------------------------
#
# A calendar is broadcast-eligible ONLY while it is selected. The selection
# is what the integration reports: every selected calendar appears in a
# sync's `sources` (healthy or failed); a calendar missing from `sources`
# has been deselected (an empty `sources` means none is selected). So:
#
#   scheduled / invalid    -> expired, SOURCE_DESELECTED (history kept)
#   committed, not on air  -> its row is revoked (a queued item is then
#                             removed by queue hygiene / served as silence);
#                             the occurrence expires SOURCE_DESELECTED, or
#                             UNCONFIRMED_AFTER_INSERT once MA had it
#   airing                 -> left alone: nothing on the air is interrupted
#
# The answer is asked again at every later step -- planning, the carrier
# bind, and immediately before the queue claim -- so a deselection that
# lands mid-render or mid-insert still keeps the words off the air.
# Re-selecting a calendar revives an occurrence withdrawn ONLY by
# deselection that never reached MA. SOURCE_DESELECTED is an EXPIRED
# reason, not a new state, so a store written here stays readable by an
# older build (rollback-safe).
SOURCE_DESELECTED = "source_deselected"
#: An occurrence born as the twin of one already aired, airing or pending
#: on another calendar id (same instant, same words): a renamed or
#: re-added calendar is a new identity, and must not air the event again.
DUPLICATE_OF = "duplicate_of:"

# --- hand-off evidence (the no-replay contract) --------------------------------
#
# An announcement is safety-biased: a duplicate airing is worse than a
# lost one. The observer polls Music Assistant every few seconds, so a
# one-line announcement can be inserted, play in full and be gone without
# its airing ever being seen -- and a row that ends 'missed' after that is
# NOT evidence that it never aired. So retry is gated on the opposite
# question: is there authoritative evidence it NEVER reached MA?
#
# An announcement row has exactly one door into Music Assistant:
# dj_engine.try_insert's atomic ready->queued claim, taken only by
# _insert_ready below (the DJ's schedule_pass never adopts an announcement
# row). try_insert calls our before_claim hook immediately before that
# claim, and the hook durably writes `handoff` into this store -- if the
# write fails, there is no claim. So:
#
#   no handoff for the row's token   the claim was never taken: the row
#                                    never reached MA -> may be retried
#   handoff 'pending'                the claim was (about to be) taken
#   handoff 'inserted'               MA accepted the insert call
#   handoff 'uncertain'              the insert call raised, and the
#                                    failure cannot prove non-delivery
#   handoff 'not_delivered'          the insert call raised and MA provably
#                                    never received it (connection refused
#                                    or unreachable, name not resolved --
#                                    never an HTTP status): the only
#                                    post-claim retry
#
# A row that reached MA and then ended without confirmed completion is
# EXPIRED with reason UNCONFIRMED_AFTER_INSERT, never rescheduled -- unless
# the player reported it on air (air_start), which is proof of air: SPENT.
HANDOFF_PENDING = "pending"
#: An occurrence whose copy rides an ordinary DJ
#: break instead of a row of its own. The occurrence's break_token is then
#: the DJ break's token, and every rule above holds unchanged -- the one
#: door into Music Assistant is the DJ's try_insert, which writes the same
#: pre-claim hand-off note through carrier_before_claim.
CARRIER_DJ_BREAK = "dj_break"
#: A carried occurrence's origin; how it was finally SAID (the DJ's own
#: words, or the floor reading it verbatim) is adopted from the row.
ORIGIN_DJ_BREAK = "dj_break"
HANDOFF_INSERTED = "inserted"
HANDOFF_UNCERTAIN = "uncertain"
HANDOFF_NOT_DELIVERED = "not_delivered"
UNCONFIRMED_AFTER_INSERT = "unconfirmed_after_insert"

_pass_lock = threading.Lock()
_state_lock = threading.Lock()


# --- durable state -----------------------------------------------------------

def _state_path() -> Path:
    return paths.BASE_DIR / "announcements_state.json"


def empty_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "synced_at": None,
            "sources": {}, "occurrences": {}}


class StoreUnavailable(RuntimeError):
    """The durable occurrence store exists but cannot be trusted.

    FAIL CLOSED. The store is the memory of what already aired: reading a
    corrupt one as empty would make every aired occurrence still on the
    calendar eligible again. So nothing is scheduled, nothing is written
    over the bad file (it stays exactly as found, for diagnosis), and the
    problem is surfaced (log + forensics once, /health/problems and the
    Station card continuously) until the operator repairs or restores it.
    `str(exc)` is the reason code."""


STORE_CORRUPT = "store_corrupt"              # not JSON
STORE_UNREADABLE = "store_unreadable"        # cannot be read at all
STORE_INVALID = "store_schema_invalid"       # JSON, but not a store we wrote
STORE_MISSING_WITH_HISTORY = "store_missing_with_history"

_VALID_STATES = (SCHEDULED, COMMITTED, SPENT, EXPIRED, INVALID)
_store_problem_seen: tuple | None = None


def load_state() -> dict[str, Any]:
    """The durable store, or StoreUnavailable. Only a GENUINELY ABSENT file
    is a first run (an empty store); a missing file while announcement rows
    still exist in dj_breaks means the store was lost, which is as unsafe
    as a corrupt one."""
    path = _state_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if _announcement_history_exists():
            raise _unavailable(STORE_MISSING_WITH_HISTORY, path)
        return empty_state()
    except (OSError, UnicodeDecodeError) as exc:
        raise _unavailable(STORE_UNREADABLE, path) from exc
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise _unavailable(STORE_CORRUPT, path) from exc
    if _schema_problem(raw):
        raise _unavailable(STORE_INVALID, path)
    raw.setdefault("sources", {})
    raw.setdefault("synced_at", None)
    return raw


def _schema_problem(raw: Any) -> str | None:
    """Is this a store this module wrote? Only the fields the lifecycle
    relies on are required; anything else is left alone."""
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return "version"
    occs = raw.get("occurrences")
    if not isinstance(occs, dict):
        return "occurrences"
    if not isinstance(raw.get("sources", {}), dict):
        return "sources"
    for oid, occ in occs.items():
        if not isinstance(occ, dict) or occ.get("id") != oid:
            return f"occurrence {oid!r}"
        if occ.get("state") not in _VALID_STATES:
            return f"occurrence {oid!r} state"
        for key in ("source", "start", "mode", "payload"):
            if not isinstance(occ.get(key), str):
                return f"occurrence {oid!r} {key}"
        if not isinstance(occ.get("handoff"), (dict, type(None))):
            return f"occurrence {oid!r} handoff"
        if not isinstance(occ.get("break_token"), (str, type(None))):
            return f"occurrence {oid!r} break_token"
        attempts = occ.get("attempts", 0)
        if not isinstance(attempts, int) or isinstance(attempts, bool):
            return f"occurrence {oid!r} attempts"
        for key in ("end", "origin", "spoken_text", "aired_at",
                    "expire_reason", "invalid_reason", "updated_at",
                    "voice_request"):
            if not isinstance(occ.get(key), (str, type(None))):
                return f"occurrence {oid!r} {key}"
        if not isinstance(occ.get("voice"), (dict, type(None))):
            return f"occurrence {oid!r} voice"
    return None


def _announcement_history_exists() -> bool:
    """Has this installation ever minted an announcement row (within the
    dj_breaks retention)? Unknown counts as yes: fail closed."""
    import sqlite3
    try:
        import dj_engine
        c = dj_engine._connect(None)
        try:
            row = c.execute(
                "SELECT 1 FROM dj_breaks WHERE telemetry LIKE ? "
                "OR telemetry LIKE ? LIMIT 1",
                ('%"purpose": "announcement"%',
                 f'%"carrier": "{CARRIER_DJ_BREAK}"%')).fetchone()
        finally:
            c.close()
        return row is not None
    except sqlite3.OperationalError as exc:
        return "no such table" not in str(exc)
    except Exception:  # noqa: BLE001
        return True


def _unavailable(reason: str, path: Path) -> StoreUnavailable:
    """Build the exception; say so loudly ONCE per distinct problem (the
    scheduler asks every few seconds)."""
    global _store_problem_seen
    try:
        st = path.stat()
        signature = (reason, st.st_mtime_ns, st.st_size)
    except OSError:
        signature = (reason, None, None)
    if signature != _store_problem_seen:
        _store_problem_seen = signature
        logger.error("announcements store %s (%s): announcements are held "
                     "until it is repaired; the file is left untouched",
                     reason, path)
        try:
            import dj_forensics
            dj_forensics.append({"event": "announcement_store_unavailable",
                                 "at": service_clock.legacy_now().isoformat(),
                                 "reason": reason, "path": str(path)})
        except Exception:  # noqa: BLE001
            logger.debug("store problem forensics failed", exc_info=True)
    return StoreUnavailable(reason)


def store_problem() -> str | None:
    """The store's trust problem right now, or None. For health and the
    Station card; never raises."""
    try:
        load_state()
        return None
    except StoreUnavailable as exc:
        return str(exc)
    except Exception:  # noqa: BLE001
        return STORE_UNREADABLE


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state, ensure_ascii=False, indent=1, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=".announcements-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- identity ----------------------------------------------------------------

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or _DATE_ONLY_RE.match(value.strip()):
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def occurrence_id(source: str, start: str) -> str:
    """sha1 of the source calendar and the start INSTANT.

    An aware start is keyed by its epoch second, so the same instant
    written with a different UTC offset stays the same occurrence. A
    naive or date-only start is keyed by its normalized string -- the
    only stability that exists for it."""
    dt = _parse_dt(start)
    if dt is not None and dt.tzinfo is not None:
        key = f"{source}|{int(dt.timestamp())}"
    else:
        key = f"{source}|{str(start).strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _start_local(occ: dict[str, Any], local_now: datetime) -> datetime | None:
    """The occurrence's start on the station clock; a naive start is read
    as station-local time. None for unreadable or date-only starts."""
    dt = _parse_dt(occ.get("start"))
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=local_now.tzinfo)
    return dt.astimezone(local_now.tzinfo)


def _payload_sha(mode: str, payload: str) -> str:
    return hashlib.sha1(f"{mode}|{payload}".encode("utf-8")).hexdigest()[:12]


# --- observation sync (the integration's report of the calendars) ------------

def _selected(state: dict[str, Any], source: str) -> bool:
    return source in (state.get("sources") or {})


def _instant_key(start: Any) -> str:
    dt = _parse_dt(start)
    if dt is not None and dt.tzinfo is not None:
        return str(int(dt.timestamp()))
    return str(start or "").strip()


def _reached_ma(occ: dict[str, Any]) -> bool:
    handoff = occ.get("handoff") or {}
    return bool(handoff) and handoff.get("state") != HANDOFF_NOT_DELIVERED


def _twin(occs: dict[str, Any], oid: str, start: Any,
          payload_sha: str | None) -> str | None:
    """Another occurrence of the same instant and words that is aired,
    on its way to air, or pending on a selected calendar -- whose twin
    must therefore never air. A twin withdrawn by deselection, or that
    expired unaired, does not count."""
    key = _instant_key(start)
    for other in occs.values():
        if other["id"] == oid or other.get("payload_sha") != payload_sha \
                or _instant_key(other.get("start")) != key:
            continue
        st, why = other["state"], other.get("expire_reason")
        if st in (SPENT, COMMITTED, SCHEDULED) or (
                st == EXPIRED and why == UNCONFIRMED_AFTER_INSERT):
            return other["id"]
    return None


def _withdraw(occ: dict[str, Any], stamp: str, db_path=None) -> bool:
    """Deselection, for one occurrence. True when anything changed."""
    if occ["state"] in (SCHEDULED, INVALID):
        occ.update(state=EXPIRED, expire_reason=SOURCE_DESELECTED,
                   updated_at=stamp)
        return True
    if occ["state"] != COMMITTED:
        return False
    import dj_engine
    token = occ.get("break_token") or ""
    row = dj_engine.get_break(token, db_path)
    if row is not None and row["state"] in ("planned", "rendering", "ready",
                                            "queued"):
        dj_engine.revoke(token, service_clock.legacy_now(), db_path,
                         reason="announcement_source_deselected")
        logger.info("announcement %s: calendar %s deselected; %s row %s "
                    "revoked before air", occ["id"], occ["source"],
                    row["state"], token[:8])
    return False        # the row's outcome is adopted by reconciliation



def observe(payload: dict[str, Any], now: datetime | None = None
            ) -> dict[str, Any]:
    """Reconcile one observation sync into the durable store.

    The integration reports the ANNOUNCE-prefixed events it found on the
    selected calendars, plus each source's health. Upserts follow the
    identity rules above; removal is conservative: only a HEALTHY
    source's missing, still-future, not-yet-committed occurrence is
    treated as deleted by the operator. Terminal occurrences are never
    resurrected, whatever the feed says."""
    if not isinstance(payload, dict):
        raise ValueError("observation payload must be an object")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) > SYNC_EVENT_CAP:
        raise ValueError(f"events must be a list of at most {SYNC_EVENT_CAP}")
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("sources must be an object of source -> status")

    import dj_context
    now_dt = now or service_clock.legacy_now()
    local_now = dj_context.station_now(now_dt)
    stamp = (local_now or now_dt).isoformat()

    accepted, invalid_n, removed, collisions = 0, 0, 0, 0
    with _state_lock:
        state = load_state()
        occs: dict[str, Any] = state["occurrences"]
        # The selection FIRST: a calendar renamed in one step (old id gone,
        # new id present) must have its old copy withdrawn before the new
        # one is judged, or the new one would be refused as its twin.
        state["sources"] = {str(k): str(v) for k, v in sources.items()}
        withdrawn = 0
        for occ in occs.values():
            if not _selected(state, occ["source"]):
                withdrawn += int(_withdraw(occ, stamp))
        seen: set[str] = set()
        for ev in events:
            if not isinstance(ev, dict):
                continue
            source = str(ev.get("source") or "")
            summary = ev.get("summary")
            start = ev.get("start")
            if not source or not isinstance(summary, str) \
                    or not isinstance(start, str) or not start.strip():
                continue
            if not announce_copy.is_directive(summary):
                continue                     # defense; integration filters
            parsed = announce_copy.parse_summary(summary)
            oid = occurrence_id(source, start)
            if oid in seen:
                collisions += 1
                logger.warning("announcement slot collision on %s @ %s; "
                               "first observed event wins", source, start)
                continue
            seen.add(oid)
            invalid_reason = parsed["invalid_reason"]
            if invalid_reason is None and _parse_dt(start) is None:
                invalid_reason = INVALID_ALL_DAY
            # Announcement voices: the event's voice REQUEST rides the
            # observation as its own field, read from the description's
            # `voice:` line by the integration. It is stored as a bounded
            # string and nothing more -- resolution and authorization
            # happen at commit, against the registry, never here.
            voice_request = announce_copy.normalize_voice_request(
                ev.get("voice"))
            occ = occs.get(oid)
            if occ is None:
                twin = _twin(occs, oid, start.strip(),
                             _payload_sha(parsed["mode"], parsed["payload"]))
                if twin and not invalid_reason:
                    logger.warning("announcement on %s @ %s duplicates %s "
                                   "(renamed or re-added calendar?); it "
                                   "will not air again", source, start,
                                   twin)
                occs[oid] = {
                    "id": oid, "source": source,
                    "start": start.strip(),
                    "end": (str(ev.get("end")).strip()
                            if isinstance(ev.get("end"), str) else None),
                    "mode": parsed["mode"], "payload": parsed["payload"],
                    "payload_sha": _payload_sha(parsed["mode"],
                                                parsed["payload"]),
                    "state": INVALID if invalid_reason else SCHEDULED,
                    "invalid_reason": invalid_reason,
                    "first_seen": stamp, "updated_at": stamp,
                    "break_token": None, "origin": None,
                    "spoken_text": None, "aired_at": None,
                    "attempts": 0, "expire_reason": None,
                    "edited_after_commit": False,
                    "voice_request": voice_request, "voice": None,
                }
                if twin and not invalid_reason:
                    occs[oid].update(state=EXPIRED,
                                     expire_reason=DUPLICATE_OF + twin)
                accepted += 1
                if invalid_reason:
                    invalid_n += 1
                continue
            if (occ["state"] == EXPIRED
                    and occ.get("expire_reason") == SOURCE_DESELECTED
                    and not _reached_ma(occ)
                    and not _twin(occs, oid, occ.get("start"),
                                  occ.get("payload_sha"))):
                # re-selected before its window closed and never handed to
                # MA: it is pending again (reconciliation still expires it
                # if the window has passed)
                occ.update(state=SCHEDULED, expire_reason=None,
                           break_token=None, handoff=None, carrier=None,
                           updated_at=stamp)
            if occ["state"] in TERMINAL_STATES:
                continue                     # aired/expired stays that way
            new_sha = _payload_sha(parsed["mode"], parsed["payload"])
            if occ["state"] == COMMITTED:
                if new_sha != occ.get("payload_sha") \
                        or voice_request != occ.get("voice_request"):
                    occ["edited_after_commit"] = True
                    occ["updated_at"] = stamp
                continue                     # committed copy stays committed
            if (new_sha != occ.get("payload_sha")
                    or occ.get("invalid_reason") != invalid_reason):
                occ.update(mode=parsed["mode"], payload=parsed["payload"],
                           payload_sha=new_sha,
                           state=INVALID if invalid_reason else SCHEDULED,
                           invalid_reason=invalid_reason, updated_at=stamp)
            if voice_request != occ.get("voice_request"):
                # a voice edit before commit wins, like a payload edit
                occ.update(voice_request=voice_request, updated_at=stamp)
            if invalid_reason:
                invalid_n += 1

        # operator deletions: healthy source, still-future start, not
        # committed, not terminal, and gone from this sync's view
        healthy = {s for s, st in sources.items()
                   if str(st).lower() == "healthy"}
        if local_now is not None and healthy:
            for oid in list(occs):
                occ = occs[oid]
                if oid in seen or occ["source"] not in healthy:
                    continue
                if occ["state"] not in (SCHEDULED, INVALID):
                    continue
                start_dt = _start_local(occ, local_now)
                if start_dt is not None and start_dt <= local_now:
                    continue                 # inside/past its window: keep
                del occs[oid]
                removed += 1

        state["synced_at"] = stamp
        _prune(state, local_now)
        _save_state(state)
    return {"ok": True, "accepted": accepted, "invalid": invalid_n,
            "removed": removed, "collisions": collisions,
            "withdrawn": withdrawn,
            "tracked": len(state["occurrences"])}


def _prune(state: dict[str, Any], local_now: datetime | None) -> None:
    occs = state["occurrences"]
    if local_now is not None:
        horizon = local_now - timedelta(days=RETAIN_TERMINAL_DAYS)
        for oid in list(occs):
            occ = occs[oid]
            if occ["state"] not in TERMINAL_STATES:
                continue
            start_dt = _start_local(occ, local_now)
            if start_dt is not None and start_dt < horizon:
                del occs[oid]
    if len(occs) > OCC_CAP:
        by_start = sorted(occs.values(),
                          key=lambda o: str(o.get("start") or ""))
        for occ in by_start[:len(occs) - OCC_CAP]:
            occs.pop(occ["id"], None)


# --- the scheduler pass --------------------------------------------------------

def _session_id(session: Any) -> str:
    """The production playback-session contract (playback._SESSION_COLS):
    the key is `session_id`. A real session has no `id`."""
    if not isinstance(session, dict):
        return ""
    return str(session.get("session_id") or "")


def scheduler_pass(session: dict[str, Any], queue_id: str,
                   now: datetime | None = None,
                   deps: dict[str, Any] | None = None,
                   db_path=None) -> str:
    """One planning pass, riding the same hooks as news: every playback
    tick and every music start. Never raises to its caller; one pass at
    a time. Returns a short reason word for tests and traces."""
    deps = deps or {}
    import settings as settings_mod
    cfg = settings_mod.announcements_config()
    if cfg.get("problem"):
        return "config_unavailable"      # fail closed: intent is uncertain
    if not cfg["enabled"]:
        return "disabled"
    if not queue_id or not _session_id(session):
        return "no_session"
    import dj_context
    now_dt = now or service_clock.legacy_now()
    local_now = dj_context.station_now(now_dt)
    if local_now is None:
        return "station_clock_unknown"
    if not _pass_lock.acquire(blocking=False):
        return "pass_in_progress"
    try:
        return _pass(session, queue_id, now_dt, local_now, cfg, deps, db_path)
    except StoreUnavailable:
        return "store_unavailable"       # fail closed: nothing is scheduled
    finally:
        _pass_lock.release()


def _window_end(start_dt: datetime, cfg: dict[str, Any]) -> datetime:
    return start_dt + timedelta(minutes=int(cfg["late_air_minutes"]))


def _pass(session: dict[str, Any], queue_id: str, now_dt: datetime,
          local_now: datetime, cfg: dict[str, Any],
          deps: dict[str, Any], db_path) -> str:
    import dj_engine
    with _state_lock:
        state = load_state()
        changed = _reconcile_outcomes(state, local_now, cfg, db_path)
        inflight = [o for o in state["occurrences"].values()
                    if o["state"] == COMMITTED]
        due = _due_occurrence(state, local_now, cfg)
        if changed:
            _save_state(state)

    if inflight:
        occ = inflight[0]
        if occ.get("carrier"):
            return "in_flight_carrier"   # the DJ's insert path owns it
        row = dj_engine.get_break(occ.get("break_token") or "", db_path)
        if row is not None and row.get("state") == "ready":
            return _insert_ready(occ, session, queue_id, now_dt, local_now,
                                 cfg, deps, db_path)
        return "in_flight"
    if due is None:
        return "idle"
    prior = _prior_airing(due, db_path)
    if prior == PRIOR_UNKNOWN:
        return "prior_airing_unknown"    # cannot tell: hold, ask again
    if prior is not None:
        return _adopt_prior_airing(due, prior, local_now)
    if cfg.get("combine_with_dj"):
        waiting = _carrier_wait(due, session, queue_id, now_dt, local_now,
                                cfg, db_path)
        if waiting:
            return waiting
    replaced: dict[str, Any] = {}
    blocked = _boundary_blocked(session, queue_id, now_dt, deps, db_path,
                                keep_rendered=bool(cfg.get("combine_with_dj")),
                                replaced=replaced)
    if blocked:
        return blocked
    return _commit(due, session, queue_id, now_dt, local_now, cfg, deps,
                   db_path, replaced=replaced or None)


def _reconcile_outcomes(state: dict[str, Any], local_now: datetime,
                        cfg: dict[str, Any], db_path) -> bool:
    # (see the calendar-selection notes at the top: a deselected source's
    # revoked row expires the occurrence; it is never rescheduled)
    """Adopt every committed occurrence's outcome from its row (or the
    retired-token ledger when the row is gone), and expire what the late
    window has passed by. The row is the live truth; this store only
    remembers what must survive the row's retention."""
    import dj_engine
    import program_tokens
    changed = False
    for occ in state["occurrences"].values():
        start_dt = _start_local(occ, local_now)
        if occ["state"] == COMMITTED:
            token = occ.get("break_token") or ""
            row = dj_engine.get_break(token, db_path)
            if row is None:
                spent = program_tokens.retired_kind(token, db_path) == \
                    program_tokens.KIND_DJ
                occ.update(state=SPENT if spent else EXPIRED,
                           expire_reason=None if spent else "row_lost",
                           updated_at=local_now.isoformat())
                changed = True
            elif row["state"] == "completed":
                occ.update(state=SPENT, aired_at=row.get("updated_at"),
                           spoken_text=(occ.get("spoken_text")
                                        if occ.get("carrier") else
                                        row.get("spoken_text")
                                        or occ.get("spoken_text")),
                           updated_at=local_now.isoformat())
                _adopt_voice_outcome(occ, row.get("telemetry") or {})
                _adopt_delivery(occ, row.get("telemetry") or {})
                occ["confirmation"] = dj_engine.airing_evidence(row)
                changed = True
            elif row["state"] in ("missed", "revoked", "failed"):
                tel = row.get("telemetry") or {}
                _adopt_voice_outcome(occ, tel)
                # what the row can PROVE, recorded with whatever follows
                occ["confirmation"] = dj_engine.airing_evidence(row)
                if tel.get("air_start"):
                    # The player reported it ON AIR before it ended: proof
                    # of air, whatever cut it short. Never again.
                    occ.update(state=SPENT, aired_at=tel.get("air_start"),
                               spoken_text=(occ.get("spoken_text")
                                            if occ.get("carrier") else
                                            row.get("spoken_text")
                                            or occ.get("spoken_text")))
                elif _may_have_reached_ma(occ, token, tel):
                    # Handed to Music Assistant, never confirmed. A short
                    # line can air between two observer polls, so 'missed'
                    # here is not evidence of silence: expire, never replay.
                    occ.update(state=EXPIRED,
                               expire_reason=UNCONFIRMED_AFTER_INSERT)
                else:
                    # Authoritatively never reached MA (failed render,
                    # revoked or timed out while ready, or an insert MA
                    # provably never received): retry inside the window.
                    occ["break_token"] = None
                    occ["handoff"] = None
                    # A carrier was never an attempt of its own: the DJ
                    # break it rode was revoked or failed, and that is the
                    # DJ's lifecycle. Bounded by the combine deadline.
                    occ["carrier"] = None
                    attempts = int(occ.get("attempts") or 0)
                    in_window = (start_dt is not None
                                 and local_now <= _window_end(start_dt, cfg))
                    if not _selected(state, occ["source"]):
                        occ.update(state=EXPIRED,
                                   expire_reason=SOURCE_DESELECTED)
                    elif in_window and attempts < MAX_ATTEMPTS:
                        occ["state"] = SCHEDULED
                    else:
                        occ.update(state=EXPIRED,
                                   expire_reason=("attempts_exhausted"
                                                  if attempts >= MAX_ATTEMPTS
                                                  else "late_window_passed"))
                occ["updated_at"] = local_now.isoformat()
                changed = True
        elif occ["state"] in (SCHEDULED, INVALID):
            if start_dt is None:
                if occ["state"] == SCHEDULED:
                    # unreadable start on a schedulable occurrence
                    occ.update(state=INVALID,
                               invalid_reason=INVALID_ALL_DAY,
                               updated_at=local_now.isoformat())
                    changed = True
                continue
            if local_now > _window_end(start_dt, cfg):
                occ.update(state=EXPIRED, expire_reason="late_window_passed",
                           updated_at=local_now.isoformat())
                changed = True
    return changed


def _adopt_delivery(occ: dict[str, Any], tel: dict[str, Any]) -> None:
    """A carrier's delivery mode, from its row: 'conversational' (the DJ's
    own words carried the whole message) or 'verbatim_floor'."""
    if occ.get("carrier"):
        mode = (tel.get("announcement_delivery") or {}).get("mode")
        if mode in ("conversational", "verbatim_floor"):
            occ["delivery"] = mode


def _adopt_voice_outcome(occ: dict[str, Any], tel: dict[str, Any]) -> None:
    """Copy the render's voice decision (who actually spoke, and why the
    requested voice did not) from the row into the occurrence, so the
    answer outlives the row's retention. Additive: the commit-time block
    (requested / authorized / rejection reason) stays."""
    outcome = tel.get("announcement_voice")
    if not isinstance(outcome, dict):
        return
    voice = dict(occ.get("voice") or {})
    for key in ("resolved_voice_id", "provider", "source", "fallback",
                "reason", "attempts"):
        if key in outcome:
            voice[key] = outcome[key]
    occ["voice"] = voice


PRIOR_UNKNOWN = "unknown"


def _prior_airing(occ: dict[str, Any], db_path) -> str | None:
    """A second line of defence behind the store: does dj_breaks already
    hold a row for THIS occurrence that aired, or was handed to Music
    Assistant? The store says scheduled, but a store restored from an older
    backup (or rebuilt by hand) can say that about something that already
    aired. 'spent' | 'unconfirmed' | None, or PRIOR_UNKNOWN when dj_breaks
    cannot be read: the caller HOLDS the occurrence (neither airs it nor
    records an outcome it cannot prove) and asks again next pass."""
    import dj_engine
    try:
        c = dj_engine._connect(db_path)
        try:
            rows = c.execute(
                "SELECT state, telemetry FROM dj_breaks WHERE telemetry LIKE ?",
                (f'%"occurrence_id": "{occ["id"]}"%',)).fetchall()
        finally:
            c.close()
    except Exception:  # noqa: BLE001 - cannot tell: never guess to air
        logger.warning("announcement prior-airing check failed; holding "
                       "occurrence %s", occ.get("id"), exc_info=True)
        return PRIOR_UNKNOWN
    verdict = None
    for state, tel_raw in rows:
        try:
            tel = json.loads(tel_raw or "{}")
        except ValueError:
            return PRIOR_UNKNOWN
        if not isinstance(tel, dict) or \
                (tel.get("announcement") or {}).get("occurrence_id") != occ["id"]:
            continue
        if state == "completed" or tel.get("air_start"):
            return "spent"
        if state in ("queued", "airing") or (
                tel.get("enqueue")
                and tel.get("enqueue_failure") != "not_delivered"):
            verdict = "unconfirmed"
    return verdict


def _adopt_prior_airing(occ: dict[str, Any], prior: str,
                        local_now: datetime) -> str:
    """Record what dj_breaks proves instead of airing again."""
    with _state_lock:
        state = load_state()
        live = state["occurrences"].get(occ["id"])
        if live is not None and live["state"] == SCHEDULED:
            if prior == "spent":
                live.update(state=SPENT, expire_reason=None)
            else:
                live.update(state=EXPIRED,
                            expire_reason=UNCONFIRMED_AFTER_INSERT)
            live["updated_at"] = local_now.isoformat()
            _save_state(state)
    logger.warning("announcement %s: dj_breaks shows it already %s; not "
                   "airing it again", occ.get("id"),
                   "aired" if prior == "spent" else "went to the player")
    return f"prior_airing:{prior}"


def _may_have_reached_ma(occ: dict[str, Any], token: str,
                         tel: dict[str, Any]) -> bool:
    """Is there ANY evidence this row was handed to Music Assistant? Only
    a 'not_delivered' insert failure counts as proof that it was not; the
    absence of every piece of evidence is the other proof (see the
    hand-off contract above)."""
    handoff = occ.get("handoff") or {}
    if token and handoff.get("token") == token:
        return handoff.get("state") != HANDOFF_NOT_DELIVERED
    if tel.get("enqueue"):          # defence in depth: any claim path
        return tel.get("enqueue_failure") != "not_delivered"
    return False


def _record_handoff(oid: str, token: str, handoff_state: str | None,
                    strict: bool = True) -> bool:
    """Durably note the hand-off of `token` for occurrence `oid`. False --
    and, for the pre-claim write, NO claim -- when the occurrence no longer
    owns this row or the store cannot be written."""
    try:
        with _state_lock:
            state = load_state()
            occ = state["occurrences"].get(oid)
            if occ is None or occ.get("state") != COMMITTED \
                    or occ.get("break_token") != token:
                return False
            if handoff_state is None:
                occ["handoff"] = None
            else:
                occ["handoff"] = {"token": token, "state": handoff_state,
                                  "at": service_clock.legacy_now().isoformat(
                                      timespec="seconds")}
            _save_state(state)
        return True
    except Exception:  # noqa: BLE001
        if strict:
            raise
        logger.warning("announcement hand-off note (%s) not saved; the "
                       "earlier pre-claim evidence stands", handoff_state,
                       exc_info=True)
        return False


def _due_occurrence(state: dict[str, Any], local_now: datetime,
                    cfg: dict[str, Any]) -> dict[str, Any] | None:
    due = [o for o in state["occurrences"].values()
           if o["state"] == SCHEDULED and _selected(state, o["source"])
           and (start := _start_local(o, local_now)) is not None
           and start <= local_now <= _window_end(start, cfg)]
    due.sort(key=lambda o: str(o.get("start") or ""))
    return due[0] if due else None


def _due_now(row: dict[str, Any], now_dt: datetime) -> bool:
    import dj_engine
    target = (row.get("telemetry") or {}).get("target_air_at")
    try:
        return datetime.fromisoformat(target) <= now_dt + timedelta(
            seconds=dj_engine.ANNOUNCEMENT_REPLACES_WITHIN_S)
    except (TypeError, ValueError):
        return False


def _boundary_blocked(session: dict[str, Any], queue_id: str,
                      now_dt: datetime, deps: dict[str, Any],
                      db_path, keep_rendered: bool = False,
                      replaced: dict[str, Any] | None = None) -> str | None:
    """May an announcement take the next boundary? News ownership wins
    outright (a bulletin's window is minutes wide; ours is tens of
    minutes). A QUEUED or AIRING break of any kind wins -- speech is not
    stacked onto speech. A merely planned/rendering/ready SCHEDULED break
    is revoked exactly as news revokes one (it replans automatically); a
    liner or manual break in flight is owner-initiated and waited out.

    The bulletin_due question is asked of the BOUNDARY this announcement
    would take, not of the clock alone, exactly as the DJ asks
    it: an announcement is queued the moment the gate opens, which can be
    before the bulletin's air window opens for a boundary that lands
    inside it.

    keep_rendered (station.announcements.combine_with_dj): a scheduled
    break that is already RENDERING or READY is not revoked -- it takes
    this boundary and the announcement the next one. Its render has
    already committed its calendar/weather/fact marks, and a revoke never
    returns them: an announcement that revoked it silently spent the DJ's
    owed calendar mention. Only a planned intent, which has committed
    nothing, is still revoked."""
    try:
        import news_schedule
        blocked = news_schedule.ordinary_dj_block_reason(
            session, queue_id, db_path=db_path, now=now_dt,
            boundary_fn=_next_boundary_fn(queue_id, now_dt, deps))
        if blocked:
            return f"news_owns_boundary:{blocked}"
    except Exception:  # noqa: BLE001 - fenced exactly like the DJ's check
        logger.warning("news block check failed (announcement waits)",
                       exc_info=True)
        return "news_check_failed"
    import dj_engine
    try:
        rows = dj_engine.active_breaks(queue_id, db_path)
    except Exception:  # noqa: BLE001
        logger.warning("break rows unreadable (announcement waits)",
                       exc_info=True)
        return "rows_unreadable"
    for row in rows:
        purpose = dj_engine.break_purpose(row.get("telemetry"))
        if purpose == dj_engine.PURPOSE_ANNOUNCEMENT:
            continue
        if row["state"] in ("queued", "airing"):
            return f"boundary_owned:{purpose}"
        if purpose == dj_engine.PURPOSE_SCHEDULED:
            if keep_rendered and row["state"] in ("rendering", "ready"):
                return "dj_break_rendered"
            if keep_rendered and _due_now(row, now_dt):
                # With keep_rendered, this announcement REPLACES the due DJ
                # break: the opportunity is consumed (one cadence later),
                # nothing it owed is marked said. Recorded for the row we mint.
                marker = dj_engine.yield_opportunity(
                    queue_id, row["token"], now_dt, db_path=db_path)
                if marker is not None:
                    if replaced is not None:
                        replaced.update(marker)
                    continue
            if keep_rendered and row["state"] == "planned":
                # With keep_rendered, an intent that is NOT due is left
                # alone. It renders nothing while the announcement is in
                # flight (the DJ's own pass is blocked), and revoking it
                # threw away a skip it had already taken -- the opportunity
                # an EXACT just replaced then came straight back one song
                # later.
                continue
            dj_engine.revoke(row["token"], service_clock.legacy_now(), db_path,
                             reason="announcement_boundary_owned")
        else:
            return f"liner_pending:{purpose}"
    if keep_rendered and replaced is not None and not replaced:
        # no intent planned yet (the DJ's boundary pass runs after this
        # one): an opportunity due within the window is consumed all the
        # same, or it would be planned already due, one song after us
        try:
            marker = dj_engine.yield_upcoming(queue_id, session, now_dt,
                                              db_path=db_path)
            if marker is not None:
                replaced.update(marker)
        except Exception:  # noqa: BLE001 - a cadence courtesy, never a block
            logger.warning("upcoming DJ opportunity not yielded", exc_info=True)
    return None


#: A standalone announcement does not claim the queue with less than this
#: much of the current track left; it waits for the next boundary (a new
#: track, with all of it ahead). An item inserted with only a few seconds
#: left can be skipped entirely -- the player has already been handed the
#: song after it and Music Assistant moves past the inserted item.
MIN_INSERT_LEAD_S = 15.0


def _too_late_in_track(queue_id: str, deps: dict[str, Any]) -> str | None:
    """'late_in_track' when the current track has less than
    MIN_INSERT_LEAD_S left; None otherwise, and whenever that cannot be
    read (the structural gate still decides)."""
    try:
        import dj_engine
        import ma_queue
        read = deps.get("state_fn") or ma_queue.queue_state
        remaining = dj_engine._remaining_s(read(queue_id))
    except Exception:  # noqa: BLE001
        return None
    if remaining is not None and remaining < MIN_INSERT_LEAD_S:
        return "late_in_track"
    return None


def _next_boundary_fn(queue_id: str, now_dt: datetime,
                      deps: dict[str, Any]):
    """Where the next safe boundary lands on the STATION clock (None when
    unknowable) -- the same projection dj_engine.schedule_pass hands the
    news check (estimated_air_at onto station_now). Lazy: the queue is
    read only if the news check actually needs the answer. Never raises."""
    def boundary():
        try:
            import dj_context
            import dj_engine
            import ma_queue
            read = deps.get("state_fn") or ma_queue.queue_state
            air_at = dj_engine.estimated_air_at(read(queue_id), None, now_dt)
            if air_at is None:
                return None
            local_now = dj_context.station_now(now_dt)
            if local_now is None:
                return None
            return dj_context.station_project(local_now, air_at - now_dt)
        except Exception:  # noqa: BLE001 - unknowable: the clock rule stands
            return None
    return boundary


def _build_copy(occ: dict[str, Any], cfg: dict[str, Any],
                deps: dict[str, Any]) -> tuple[str, str]:
    """(spoken copy, origin). EXACT is the operator's wording, verbatim.
    Plain ANNOUNCE tries one Smart rewrite when the setting is on and a
    writer is usable -- one draft, validated, never repaired -- and
    otherwise (or on any failure) renders deterministically. This can
    not return empty for a valid occurrence."""
    payload = str(occ.get("payload") or "")
    if occ.get("mode") == announce_copy.MODE_EXACT:
        return payload, "exact"
    if cfg.get("smart_rewriting") and _writer_usable(deps):
        draft, reject = _smart_draft(payload, deps)
        if draft is not None and reject is None:
            return draft, "smart"
        _record_rejected_draft(occ, draft, reject)
    return announce_copy.deterministic_copy(payload), "deterministic"


def _writer_usable(deps: dict[str, Any]) -> bool:
    if deps.get("announce_llm_fn"):
        return True
    try:
        import optional_services
        return bool(optional_services.effective_writer().get("usable"))
    except Exception:  # noqa: BLE001
        return False


def _smart_draft(payload: str, deps: dict[str, Any]
                 ) -> tuple[str | None, str | None]:
    """(accepted draft, None) or (raw/cleaned draft or None, reason)."""
    prompt = announce_copy.smart_prompt(payload)
    llm = deps.get("announce_llm_fn")
    if llm is None:
        def llm(p):  # noqa: ANN001
            from inference import llm_adapter
            return llm_adapter.generate_text(llm_adapter.ROLE_DJ_WRITER, p)
    try:
        raw = llm(prompt)
    except Exception as exc:  # noqa: BLE001 - the provider boundary
        logger.warning("announcement smart rewrite failed (%s: %s); "
                       "deterministic copy airs",
                       type(exc).__name__, str(exc)[:160])
        return None, "provider_error"
    raw_text = raw if isinstance(raw, str) else ""
    # The RAW draft is screened first: line breaks, markdown and assistant
    # talk are evidence, and normalising first would erase it.
    reject = announce_copy.screen_raw_draft(payload, raw_text)
    draft = None
    if reject is None:
        draft = announce_copy.clean_smart_draft(raw_text)
        reject = announce_copy.validate_smart(payload, draft)
    if reject:
        logger.info("announcement smart draft rejected (%s); deterministic "
                    "copy airs", reject)
        return raw_text, reject     # forensics keep what the model said
    return draft, None


def _record_rejected_draft(occ: dict[str, Any], draft: str | None,
                           reason: str | None) -> None:
    try:
        import dj_forensics
        dj_forensics.append({
            "event": "announcement_draft_rejected",
            "at": service_clock.legacy_now().isoformat(),
            "occurrence_id": occ.get("id"),
            "reason": reason or "provider_error",
            "draft": (draft or "")[:600] or None,
            "payload": str(occ.get("payload") or "")[:600],
        })
    except Exception:  # noqa: BLE001 - forensics never cost an airing
        logger.debug("announcement draft forensics failed", exc_info=True)


def _commit(occ: dict[str, Any], session: dict[str, Any], queue_id: str,
            now_dt: datetime, local_now: datetime, cfg: dict[str, Any],
            deps: dict[str, Any], db_path,
            replaced: dict[str, Any] | None = None) -> str:
    """The one write that turns an occurrence into programming: build the
    copy, mint the row, persist the commitment BEFORE rendering (a crash
    mid-render leaves a row the TTL fails and the next pass retries),
    then render and take the boundary if it is open right now."""
    import dj_engine
    copy, origin = _build_copy(occ, cfg, deps)
    voice = _resolve_voice(occ, cfg)
    token = dj_engine.create_announcement_break(
        queue_id, occ, _session_id(session), origin=origin,
        now=now_dt, db_path=db_path, voice=voice)
    if replaced:
        # provenance: which ordinary opportunity this announcement took
        dj_engine._note(token, {"replaced_dj_opportunity": {
            "revoked_token": replaced.get("revoked_token"),
            "suppressed_target": replaced.get("suppressed_target")}},
            db_path)
    with _state_lock:
        try:
            state = load_state()
        except StoreUnavailable:
            dj_engine.revoke(token, service_clock.legacy_now(), db_path,
                             reason="announcement_store_unavailable")
            raise
        live = state["occurrences"].get(occ["id"])
        if live is None or live["state"] != SCHEDULED:
            # observed away or raced; the row is abandoned to the TTL
            dj_engine.revoke(token, service_clock.legacy_now(), db_path,
                             reason="announcement_occurrence_gone")
            return "occurrence_gone"
        live.update(state=COMMITTED, break_token=token, origin=origin,
                    spoken_text=copy, voice=voice,
                    attempts=int(live.get("attempts") or 0) + 1,
                    committed_at=local_now.isoformat(),
                    updated_at=local_now.isoformat())
        _save_state(state)
        occ = live
    ok = dj_engine.render_announcement(token, copy, deps=deps,
                                       now=now_dt, db_path=db_path)
    if not ok:
        return "render_failed"      # row is 'failed'; next pass reconciles
    return _insert_ready(occ, session, queue_id, now_dt, local_now, cfg,
                         deps, db_path)


def _resolve_voice(occ: dict[str, Any], cfg: dict[str, Any]
                   ) -> dict[str, Any]:
    """The voice decision for this commit:

        requested               what the calendar asked for, as observed
        voice_id                the AUTHORIZED shelved voice it names, or None
        voice_identity          which voice that id holds right now (a
                                custom slot's sample fingerprint; the id
                                itself otherwise) -- the render checks the
                                slot still holds it before speaking
        reason                  why not (voice_library.REJECT_*), or None
        default_voice_id        station.announcements.default_voice_id now
        default_voice_identity  its identity now, on the same terms

    Authorization is the registry's answer alone (voice_library
    .resolve_announcement_voice): a request that names no shelved voice,
    or a voice the owner has not scoped for announcements, is refused
    here with its reason and the announcement airs in the next voice
    down. The words are decided elsewhere and are not touched by any of
    this. Never raises: a registry fault is a recorded refusal, not a
    lost announcement."""
    requested = occ.get("voice_request") or None
    default = cfg.get("default_voice_id") or None
    if cfg.get("combine_with_dj") and occ.get("mode") != announce_copy.MODE_EXACT:
        # Combined with the DJ, the announcement voice setting is EXACT's.
        # A plain ANNOUNCE is the DJ's line, in the on-air DJ voice,
        # standalone or not; an explicit per-event `voice:` request still applies.
        default = None
    out: dict[str, Any] = {"requested": requested, "voice_id": None,
                           "voice_identity": None, "reason": None,
                           "default_voice_id": default,
                           "default_voice_identity": None}
    try:
        import voice_library
        if default:
            out["default_voice_identity"] = voice_library.identity_of(default)
    except Exception:  # noqa: BLE001 - the registry boundary
        logger.debug("default voice identity unreadable", exc_info=True)
    if not requested:
        return out
    try:
        import voice_library
        res = voice_library.resolve_announcement_voice(requested)
        out["voice_id"] = res.get("voice_id")
        out["reason"] = res.get("reason")
        if out["voice_id"]:
            out["voice_identity"] = voice_library.identity_of(out["voice_id"])
    except Exception:  # noqa: BLE001 - the registry boundary
        logger.warning("announcement %s: voice request %r could not be "
                       "resolved; the station voice airs it",
                       occ.get("id"), requested, exc_info=True)
        out["reason"] = "voice_resolution_failed"
    if out["reason"]:
        logger.warning("announcement %s: requested voice %r refused (%s)",
                       occ.get("id"), requested, out["reason"])
    return out


def _insert_ready(occ: dict[str, Any], session: dict[str, Any],
                  queue_id: str, now_dt: datetime, local_now: datetime,
                  cfg: dict[str, Any], deps: dict[str, Any],
                  db_path) -> str:
    """Try to take the CURRENT boundary for a rendered announcement. A
    closed gate is normal -- the row stays ready and the next tick tries
    again -- but a window that closed while it waited revokes it for
    good: better silence than stale programming."""
    import dj_engine
    token = occ.get("break_token") or ""
    oid = occ.get("id") or ""
    if (occ.get("handoff") or {}).get("token") == token:
        # Only ever reached while the row is READY, and nothing but a
        # render moves a row INTO ready: a pending note for this token is
        # from a pass that died between writing it and taking the claim.
        # The row proves the claim never happened.
        _record_handoff(oid, token, None, strict=False)
    start_dt = _start_local(occ, local_now)
    if start_dt is None or local_now > _window_end(start_dt, cfg):
        dj_engine.revoke(token, service_clock.legacy_now(), db_path,
                         reason="announcement_window_expired")
        return "window_expired"    # reconciled to expired on the next pass
    try:
        deselected = not _selected(load_state(), occ.get("source") or "")
    except StoreUnavailable:
        return "store_unavailable"
    if deselected:
        dj_engine.revoke(token, service_clock.legacy_now(), db_path,
                         reason="announcement_source_deselected")
        return "source_deselected"
    # The SAME arbitration as the commit: this is the last look before
    # the claim, and at a music boundary the DJ plans its next intent in a
    # parallel pass -- so the intent is often only visible here (without
    # this, it was plain-revoked and the rendered-break protection did not
    # apply on this path).
    replaced: dict[str, Any] = {}
    blocked = _boundary_blocked(session, queue_id, now_dt, deps, db_path,
                                keep_rendered=bool(cfg.get("combine_with_dj")),
                                replaced=replaced)
    if blocked:
        return blocked
    late = _too_late_in_track(queue_id, deps)
    if late:
        return late
    if replaced:
        dj_engine._note(token, {"replaced_dj_opportunity": {
            "revoked_token": replaced.get("revoked_token"),
            "suppressed_target": replaced.get("suppressed_target")}},
            db_path)
    result = dj_engine.try_insert(
        token, service_clock.legacy_now(), state_fn=deps.get("state_fn"),
        insert_fn=deps.get("insert_fn"), db_path=db_path,
        items_fn=deps.get("items_fn"),
        # durable evidence BEFORE the claim; no evidence, no claim
        before_claim=lambda: _record_handoff(oid, token, HANDOFF_PENDING))
    if result == "queued":
        _record_handoff(oid, token, HANDOFF_INSERTED, strict=False)
        return "queued"
    if result == "failed":
        row = dj_engine.get_break(token, db_path) or {}
        never = ((row.get("telemetry") or {}).get("enqueue_failure")
                 == "not_delivered")
        _record_handoff(oid, token, HANDOFF_NOT_DELIVERED if never
                        else HANDOFF_UNCERTAIN, strict=False)
        return "failed"
    if result in ("gate_closed", "stale"):
        return "gate_closed"        # normal; the next tick retries
    if result == "not_ready":
        # The claim was lost: the row left READY between the gate and the
        # claim (a revoke or a TTL). Only this path claims an announcement
        # row, so unless the row shows it reached the player, the note
        # written for a claim that never happened must not outlive it.
        row = dj_engine.get_break(token, db_path) or {}
        if row.get("state") not in ("queued", "airing", "completed") \
                and not (row.get("telemetry") or {}).get("enqueue"):
            _record_handoff(oid, token, None, strict=False)
    return result                    # 'not_ready' | 'handoff_unrecorded'


# --- plain ANNOUNCE rides the next DJ break -------------------------------------
#
# Off unless station.announcements.combine_with_dj. A due plain-ANNOUNCE
# occurrence first waits (at most combine_wait_minutes past its start) for
# the ordinary DJ break that is about to render; dj_engine binds it there
# (bind_for_dj_break) immediately before render_break and appends its copy
# to the DJ's words. Nothing reaches the planner or the writer: the copy is
# built by the same _build_copy every announcement uses (the validated
# Smart rewrite when that is on, otherwise the deterministic renderer), so
# the DJ adds personality around the message, never inside it.
#
# Anything that prevents a carrier -- EXACT, an explicit voice request, DJ
# disabled or silenced, no ordinary opportunity before the deadline, the
# deadline itself -- leaves the standalone path to air it. The no-replay
# contract is unchanged: the carrier's insert writes the same pre-claim
# hand-off note, and the occurrence reconciles from the carrier row exactly
# as from its own.

def _combinable(occ: dict[str, Any]) -> bool:
    return (occ.get("state") == SCHEDULED
            and occ.get("mode") != announce_copy.MODE_EXACT
            and not occ.get("voice_request"))


def _combine_deadline(occ: dict[str, Any], local_now: datetime,
                      cfg: dict[str, Any]) -> datetime | None:
    start_dt = _start_local(occ, local_now)
    if start_dt is None:
        return None
    return start_dt + timedelta(minutes=int(cfg.get("combine_wait_minutes")
                                            or 0))


def _carrier_wait(occ: dict[str, Any], session: dict[str, Any],
                  queue_id: str, now_dt: datetime, local_now: datetime,
                  cfg: dict[str, Any], db_path) -> str | None:
    """'awaiting_dj_break' while a DJ break is expected to carry this
    occurrence before its combine deadline; None means air it standalone
    now. Fails toward standalone: an unreadable prediction never holds
    an announcement."""
    if not _combinable(occ):
        return None
    deadline = _combine_deadline(occ, local_now, cfg)
    if deadline is None or local_now >= deadline:
        return None
    try:
        import dj_engine
        if not (dj_engine.dj_enabled_for_queue(queue_id)
                and dj_engine.dj_speech_enabled()):
            return None
        nxt = dj_engine.next_ordinary_opportunity(queue_id, session, now_dt,
                                                  db_path)
    except Exception:  # noqa: BLE001
        logger.debug("carrier prediction failed; standalone", exc_info=True)
        return None
    if nxt is None or nxt > now_dt + (deadline - local_now):
        return None
    return "awaiting_dj_break"


def bind_for_dj_break(token: str, queue_id: str, now_dt: datetime,
                      deps: dict[str, Any] | None = None,
                      db_path=None) -> dict[str, Any] | None:
    """Bind the oldest due plain-ANNOUNCE occurrence to the ordinary DJ
    break `token`, which is about to render, and return what the render
    needs ({occurrence_id, copy, origin, ...}); None when there is nothing
    to carry. Idempotent per token: a break re-rendered after a crash gets
    the occurrence it already holds. Never raises -- a fault here costs the
    announcement its ride, never the DJ its break."""
    try:
        return _bind(token, queue_id, now_dt, deps or {}, db_path)
    except Exception:  # noqa: BLE001
        logger.warning("announcement carrier bind failed (DJ break "
                       "unaffected)", exc_info=True)
        return None


def _carrier_record(occ: dict[str, Any]) -> dict[str, Any]:
    return {"occurrence_id": occ["id"], "source": occ.get("source"),
            "start": occ.get("start"), "mode": occ.get("mode"),
            "origin": occ.get("origin"), "payload_sha": occ.get("payload_sha"),
            "copy": occ.get("spoken_text") or "",
            "carrier": CARRIER_DJ_BREAK}


def _bind(token: str, queue_id: str, now_dt: datetime,
          deps: dict[str, Any], db_path) -> dict[str, Any] | None:
    import settings as settings_mod
    cfg = settings_mod.announcements_config()
    if cfg.get("problem") or not cfg["enabled"] \
            or not cfg.get("combine_with_dj") or not token:
        return None
    import dj_context
    local_now = dj_context.station_now(now_dt)
    if local_now is None:
        return None
    with _state_lock:
        state = load_state()
        changed = _reconcile_outcomes(state, local_now, cfg, db_path)
        occs = state["occurrences"].values()
        held = next((o for o in occs if o["state"] == COMMITTED
                     and o.get("carrier")
                     and o.get("break_token") == token), None)
        busy = any(o["state"] == COMMITTED for o in occs)
        candidates = sorted(
            (o for o in occs if _combinable(o)
             and _selected(state, o["source"])
             and (start := _start_local(o, local_now)) is not None
             and start <= local_now
             and local_now < (_combine_deadline(o, local_now, cfg)
                              or local_now)),
            key=lambda o: str(o.get("start") or ""))
        if changed:
            _save_state(state)
    if held is not None:
        return _carrier_record(held)
    if busy or not candidates:
        return None             # one announcement in flight at a time
    occ = candidates[0]
    if _prior_airing(occ, db_path) is not None:
        return None             # the announcement pass adopts or holds it
    # The DJ writer is the one that puts it in its own words; what the
    # occurrence keeps is the message itself, which is also exactly what
    # the floor reads verbatim when the writer does not deliver it.
    copy = announce_copy.deterministic_copy(str(occ.get("payload") or ""))
    origin = ORIGIN_DJ_BREAK
    with _state_lock:
        state = load_state()
        live = state["occurrences"].get(occ["id"])
        if live is None or not _combinable(live) or any(
                o["state"] == COMMITTED
                for o in state["occurrences"].values()):
            return None         # raced: standalone commit or another carrier
        live.update(state=COMMITTED, break_token=token,
                    carrier=CARRIER_DJ_BREAK, origin=origin,
                    spoken_text=copy, voice=None, handoff=None,
                    committed_at=local_now.isoformat(),
                    updated_at=local_now.isoformat())
        _save_state(state)
    logger.info("announcement %s rides DJ break %s (%s copy)", occ["id"],
                token[:8], origin)
    return _carrier_record(live)


def carrier_before_claim(token: str, now: datetime | None = None):
    """The pre-claim evidence hook for a DJ break that carries an
    announcement (None when it carries none, or the store cannot say)."""
    oid = _carrier_oid(token)
    if oid is None:
        return None

    def hook() -> bool:
        # Past its late window the announcement is stale programming: no
        # evidence, no claim -- the DJ re-renders the break without it.
        if not _carrier_in_window(oid, now):
            return False
        return _record_handoff(oid, token, HANDOFF_PENDING)
    return hook


def _carrier_in_window(oid: str, now: datetime | None) -> bool:
    import dj_context
    import settings as settings_mod
    cfg = settings_mod.announcements_config()
    local_now = dj_context.station_now(now or service_clock.legacy_now())
    state = load_state()
    occ = state["occurrences"].get(oid)
    if cfg.get("problem") or local_now is None or occ is None:
        return False
    if not _selected(state, occ["source"]):
        return False            # deselected while it was being prepared
    start_dt = _start_local(occ, local_now)
    return start_dt is not None and local_now <= _window_end(start_dt, cfg)


def carrier_after_insert(token: str, result: str, db_path=None) -> None:
    """The post-claim half of the hand-off note, exactly as _insert_ready
    writes it for a standalone row. Never raises."""
    try:
        oid = _carrier_oid(token)
        if oid is None:
            return
        if result == "queued":
            _record_handoff(oid, token, HANDOFF_INSERTED, strict=False)
        elif result == "failed":
            import dj_engine
            row = dj_engine.get_break(token, db_path) or {}
            never = ((row.get("telemetry") or {}).get("enqueue_failure")
                     == "not_delivered")
            _record_handoff(oid, token, HANDOFF_NOT_DELIVERED if never
                            else HANDOFF_UNCERTAIN, strict=False)
    except Exception:  # noqa: BLE001
        logger.debug("carrier hand-off note failed", exc_info=True)


def _carrier_oid(token: str) -> str | None:
    try:
        state = load_state()
    except Exception:  # noqa: BLE001
        return None
    for occ in state["occurrences"].values():
        if occ["state"] == COMMITTED and occ.get("carrier") \
                and occ.get("break_token") == token:
            return occ["id"]
    return None


# --- reads --------------------------------------------------------------------

def ordinary_dj_block_reason(queue_id: str, db_path=None) -> str | None:
    """Does an announcement own the coming boundary? Read by the DJ's
    schedule_pass exactly where the news block is read, so an ordinary
    break is never rendered to stack behind (or race) a committed
    announcement. Fast fail-open: this is a courtesy ordering, and the
    boundary gate itself still serializes actual insertion."""
    try:
        state = load_state()
        committed = [o for o in state["occurrences"].values()
                     if o["state"] == COMMITTED and not o.get("carrier")]
        if not committed:
            return None
        import dj_engine
        for occ in committed:
            row = dj_engine.get_break(occ.get("break_token") or "", db_path)
            if row is not None and str(row.get("queue_id")) == queue_id \
                    and row["state"] in ("planned", "rendering", "ready",
                                         "queued", "airing"):
                return "announcement_in_flight"
        return None
    except Exception:  # noqa: BLE001
        logger.debug("announcement block check failed open", exc_info=True)
        return None


def status_snapshot(db_path=None) -> dict[str, Any]:
    """The Station tab's Announcements card, in one read. Committed
    occurrences borrow their live display state from the row; nothing
    here writes. A store or settings document that cannot be trusted is
    REPORTED (store_problem / config_problem), never papered over: the
    card then says nothing will air until it is repaired."""
    import settings as settings_mod
    cfg = settings_mod.announcements_config()
    store_problem_now = None
    try:
        state = load_state()
    except StoreUnavailable as exc:
        store_problem_now = str(exc)
        state = empty_state()
    import dj_context
    local_now = dj_context.station_now(service_clock.legacy_now())

    def display_state(occ: dict[str, Any]) -> str:
        if occ["state"] != COMMITTED:
            return occ["state"]
        try:
            import dj_engine
            row = dj_engine.get_break(occ.get("break_token") or "", db_path)
            return f"committed:{row['state']}" if row else "committed"
        except Exception:  # noqa: BLE001
            return "committed"

    def public(occ: dict[str, Any]) -> dict[str, Any]:
        out = {"id": occ["id"], "source": occ["source"],
               "start": occ["start"], "mode": occ["mode"],
               "payload": occ["payload"], "state": display_state(occ),
               "origin": occ.get("origin")}
        if occ.get("invalid_reason"):
            out["invalid_reason"] = occ["invalid_reason"]
        if occ.get("aired_at"):
            out["aired_at"] = occ["aired_at"]
        if occ.get("spoken_text") and (
                occ["state"] in (COMMITTED, SPENT)
                or occ.get("expire_reason") == UNCONFIRMED_AFTER_INSERT):
            out["spoken_text"] = occ["spoken_text"]
        if occ.get("expire_reason"):
            out["expire_reason"] = occ["expire_reason"]
        if occ.get("edited_after_commit"):
            out["edited_after_commit"] = True
        if occ.get("voice_request"):
            out["voice_request"] = occ["voice_request"]
        if occ.get("carrier"):
            out["carrier"] = occ["carrier"]
        if occ.get("delivery"):
            out["delivery"] = occ["delivery"]
        if occ.get("confirmation"):
            out["confirmation"] = occ["confirmation"]
        if occ.get("voice"):
            out["voice"] = occ["voice"]
        return out

    occs = list(state["occurrences"].values())
    active = sorted((o for o in occs
                     if o["state"] in (SCHEDULED, COMMITTED, INVALID)),
                    key=lambda o: str(o.get("start") or ""))
    finished = sorted((o for o in occs if o["state"] in TERMINAL_STATES),
                      key=lambda o: str(o.get("start") or ""),
                      reverse=True)
    return {
        "ok": True,
        "enabled": cfg["enabled"],
        "smart_rewriting": cfg["smart_rewriting"],
        "smart_available": _writer_usable({}),
        "late_air_minutes": cfg["late_air_minutes"],
        "default_voice_id": cfg.get("default_voice_id") or "",
        "voices": _announcement_voices(),
        "station_time": local_now.isoformat() if local_now else None,
        "synced_at": state.get("synced_at"),
        "sources": state.get("sources") or {},
        # This backend withdraws a deselected calendar's pending and
        # prepared announcements (SOURCE_DESELECTED), whatever the
        # combine_with_dj setting. The Settings panel shows the selection
        # as authoritative only when this is true; an older backend omits it.
        "selection_enforced": True,
        "upcoming": [public(o) for o in active[:8]],
        "recent": [public(o) for o in finished[:5]],
        "counts": {"tracked": len(occs),
                   "upcoming": len(active), "finished": len(finished)},
        "store_problem": store_problem_now,
        "config_problem": cfg.get("problem"),
    }


def _announcement_voices() -> list[dict[str, Any]]:
    try:
        import voice_library
        return voice_library.announcement_voices()
    except Exception:  # noqa: BLE001
        return []


def health_problems() -> list[dict[str, Any]]:
    """Findings for /health/problems: the two ways announcements are held
    closed that the operator must act on. Never raises."""
    import config as _config
    out: list[dict[str, Any]] = []
    try:
        problem = store_problem()
        if problem:
            out.append(_config.problem(
                "announcement_store_unreadable", _config.SEVERITY_ERROR,
                "announcements",
                f"the announcements record ({_state_path().name}) cannot "
                f"be trusted ({problem}); no announcement will air until it "
                "is repaired or restored, so nothing already aired can air "
                "twice",
                ("restore announcements_state.json from a backup; it is "
                 "missing although announcements have aired"
                 if problem == STORE_MISSING_WITH_HISTORY else
                 "restore announcements_state.json from a backup, or replace "
                 "it with a valid store; the unreadable file has been left "
                 "untouched for diagnosis")))
        import settings as settings_mod
        cfg_problem = settings_mod.announcements_config().get("problem")
        if cfg_problem:
            out.append(_config.problem(
                "announcement_settings_unreadable", _config.SEVERITY_ERROR,
                "announcements",
                f"the settings document cannot be read for announcements "
                f"({cfg_problem}); announcements are held off",
                "repair musicbrain_settings.json or save the Station "
                "settings again"))
    except Exception:  # noqa: BLE001
        logger.debug("announcement health check failed", exc_info=True)
    return out
