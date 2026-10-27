#!/usr/bin/env python3
"""Contextual enrichment for queue-native DJ breaks.

Home Assistant publishes a small sanitized snapshot of enrichment facts
(weather / upcoming calendar events / one news headline) to POST /dj/context,
INDEPENDENTLY of any break. At render time the deterministic topic editor in
this module decides which facts -- if any -- are eligible for the break being
written, and hands them to the prompt as pre-formatted grounded fact strings.

Hard boundaries:
  - Publishing a snapshot NEVER triggers planning, rendering, or any queue
    transport. It only stores facts.
  - Music Brain, not the LLM, decides eligibility. Staleness, repetition
    suppression, and topic budgets are evaluated here, deterministically.
  - A missing / stale / malformed snapshot simply yields a music-only break.
    Context failure can never fail music playback or the DJ transport.
  - Music remains the backbone: most breaks stay music-only; utility topics
    appear when noteworthy, never merely because data exists.

Storage: two small JSON files under the data root (no schema migration --
this is ephemeral operational context, not history):
  dj_context.json        the newest snapshot from HA (atomic replace)
  dj_context_state.json  last-mentioned bookkeeping for repetition suppression
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import paths
import service_clock

logger = logging.getLogger("musicbrain")

# --- eligibility policy (deterministic; internal, not customer settings) ------

WEATHER_MAX_AGE_MIN = 60.0        # snapshot weather older than this is stale
WEATHER_REPEAT_MIN = 45.0         # ordinary weather at most this often
WEATHER_NOTABLE_TEMP_DELTA = 8.0  # a change this large overrides suppression

# Calendar relevance (minutes before event start). A never-mentioned timed
# event is eligible for its FIRST mention continuously through the whole
# useful pre-event window -- there is deliberately no dead zone, and an
# event already inside the window when the station starts is not penalized for a
# theoretical earlier phase it never had. A SECOND near-term reminder may
# follow, only with real spacing after the first. Two mentions max, ever.
CAL_FIRST_WINDOW = (20.0, 150.0)     # first mention: the general heads-up
CAL_REMINDER_WINDOW = (20.0, 40.0)   # optional second: "X in half an hour"
CAL_REMINDER_SPACING_MIN = 30.0      # minimum gap after the first mention
CAL_MAX_EVENTS = 8
#: All-day events carry a DATE and no clock time, so none of the windows
#: above can apply to them -- and skipping them outright would mean a
#: calendar made mostly of all-day entries produces calendar
#: breaks that never come. They are announced
#: ONCE instead, on their own day or the day before, against the STATION's
#: date (station_now), never the container's. 1 = today and tomorrow.
CAL_ALLDAY_LEAD_DAYS = 1
#: An all-day mark must outlive its event's midnight: the key's timestamp is
#: date-only, so the ordinary >2h prune would forget it before the day it
#: describes is over and the same event could be announced twice.
CAL_ALLDAY_FORGET_MIN = 60.0 * 30    # a day and a half past its date
#: How long a music knowledge fact stays off the air after it was
#: genuinely spoken. The pack's guidance is 7-14 days; 7 is the
#: near end, because the cohort is a single listener's library, in which
#: the same artists come round often.
FACT_COOLDOWN_DAYS = 7.0
#: How long the SAME earlier-session callback relationship stays retired
#: after it aired. Without it, a night of callbacks can keep returning to
#: the same few relationships -- "Wren Hollis played in both Cold Orchard
#: and Marlow Kites", reused within one evening. A relationship is a
#: story; told twice in a night it is a tic.
CALLBACK_COOLDOWN_HOURS = 12.0
# A calendar snapshot older than this is not trusted for announcements: an
# event cancelled after the publisher stopped must never be read out.
CAL_MAX_AGE_MIN = 120.0
#: How long the published clock (clock.local_iso, a fixed UTC offset) may be
#: advanced by service-clock arithmetic. The publisher refreshes every ten
#: minutes; past this age its offset may be wrong (a DST or timezone
#: change), so the station clock is re-derived from the snapshot's IANA
#: timezone name when the runtime knows it, and is otherwise UNKNOWN --
#: never a stale local time, never the container's clock.
CLOCK_MAX_AGE_MIN = 90.0

NEWS_MAX_AGE_MIN = 90.0           # headline freshness cap

# Topic priority when the budget is smaller than what is eligible: an
# imminent reminder outranks everything; news outranks a distant heads-up
# only when the headline may stand alone -- keep it simple and fixed.
PRIORITY = ("calendar_reminder", "calendar_headsup", "calendar_allday",
            "news", "weather")


def _snapshot_path() -> Path:
    return paths.BASE_DIR / "dj_context.json"


def _state_path() -> Path:
    return paths.BASE_DIR / "dj_context_state.json"


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _parse_dt(value: Any) -> datetime | None:
    """ISO string -> datetime (aware or naive), else None. Never raises."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _age_minutes(stamp: Any, now: datetime) -> float | None:
    """Minutes since an ISO timestamp, tolerant of aware/naive mixes.

    A NAIVE value here is always the service's OWN clock reading -- either
    `datetime.now()` or one of our own stamps written from it -- so it is
    resolved in the runtime's local zone, NOT stamped as UTC.

    This used to read naive as UTC "matching the service clock", and that
    premise is false on the target platform: Supervisor gives an add-on
    container the station's timezone, so the service clock is local. Every
    timestamp the integration publishes is timezone-AWARE
    (`weather.observed_at`, `calendar.fetched_at`), so every mixed
    comparison came out wrong by the whole UTC offset -- ages too small
    west of Greenwich (a dead publisher's weather stayed "fresh" forever)
    and too large east of it, where a snapshot published seconds ago aged
    past WEATHER_MAX_AGE_MIN / NEWS_MAX_AGE_MIN / CAL_MAX_AGE_MIN on
    arrival and weather, headlines and calendar silently never reached a
    break at all. A calendar snapshot a few minutes old could report a
    calendar_age_minutes of several hundred negative minutes.

    `astimezone()` is right under either container clock: it resolves a
    naive value in whatever zone the runtime is actually in, and a
    UTC container still gets +00:00.

    DST-safe: both sides are compared as INSTANTS, so a spring-forward
    no longer adds an hour to a naive stamp's age. A stamp written in a
    repeated fall-back hour can be either of two ages; this returns the
    OLDER reading -- every caller here asks "is it fresh / may it repeat
    yet", and the older reading never calls a stale fact fresh and never
    holds a release past its true time.
    """
    if _parse_dt(stamp) is None:
        return None
    bounds = service_clock.age_bounds(stamp, now)
    return None if bounds is None else bounds[1] / 60.0


# --- snapshot validation / storage --------------------------------------------

class ContextError(ValueError):
    """Malformed snapshot payload (rejected; nothing stored)."""


class LabClockMismatch(ContextError):
    """Lab clock in force and the snapshot's clock is not the simulated
    station clock (rejected; nothing stored). Never raised in production:
    the check only runs under a nonzero lab offset."""


#: How far a snapshot's clock may sit from the simulated service clock
#: while the lab clock is in force (publish latency plus slack).
LAB_INGRESS_TOLERANCE_S = 120.0


def _lab_ingress_check(clean: dict[str, Any], now: datetime) -> None:
    if not service_clock.offset():
        return
    local = _parse_dt((clean.get("clock") or {}).get("local_iso"))
    if local is None or local.tzinfo is None:
        raise LabClockMismatch("lab clock in force: a snapshot must carry "
                               "the simulated station clock")
    skew = abs((local - service_clock.instant(now)).total_seconds())
    if skew > LAB_INGRESS_TOLERANCE_S:
        raise LabClockMismatch(
            f"lab clock in force: snapshot clock is {skew:.0f}s from the "
            "simulated service clock")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContextError(msg)


def validate_snapshot(payload: Any) -> dict[str, Any]:
    """Strictly validate an inbound snapshot. Returns the sanitized copy.

    Unknown top-level keys are rejected; each section is optional but must be
    well-formed when present. Strings are length-capped; events are capped.
    """
    _require(isinstance(payload, dict), "snapshot must be an object")
    known = {"weather", "calendar", "news", "clock"}
    unknown = set(payload) - known
    _require(not unknown, f"unknown snapshot fields: {', '.join(sorted(unknown))}")
    clean: dict[str, Any] = {}

    weather = payload.get("weather")
    if weather is not None:
        _require(isinstance(weather, dict), "weather must be an object")
        w: dict[str, Any] = {}
        for key in ("temperature", "forecast_low", "forecast_high"):
            val = weather.get(key)
            if val is not None:
                _require(isinstance(val, (int, float)) and not isinstance(val, bool)
                         and -100 <= float(val) <= 150,
                         f"weather.{key} must be a plausible number")
                w[key] = float(val)
        cond = weather.get("condition")
        if cond is not None:
            _require(isinstance(cond, str) and len(cond) <= 60,
                     "weather.condition must be a short string")
            w["condition"] = cond.strip()
        obs = weather.get("observed_at")
        if obs is not None:
            _require(_parse_dt(obs) is not None,
                     "weather.observed_at must be an ISO timestamp")
            w["observed_at"] = obs
        hourly = weather.get("hourly")
        if hourly is not None:
            _require(isinstance(hourly, list) and len(hourly) <= 40,
                     "weather.hourly must be a list of at most 40 entries")
            hrs = []
            for row in hourly:
                _require(isinstance(row, dict),
                         "weather.hourly entries must be objects")
                when = _parse_dt(row.get("time"))
                _require(when is not None and when.tzinfo is not None,
                         "weather.hourly time must be a timezone-aware ISO "
                         "timestamp")
                temp = row.get("temperature")
                _require(isinstance(temp, (int, float))
                         and not isinstance(temp, bool)
                         and -100 <= float(temp) <= 150,
                         "weather.hourly temperature must be a plausible "
                         "number")
                entry = {"time": row["time"], "temperature": float(temp)}
                cond = row.get("condition")
                if cond is not None:
                    _require(isinstance(cond, str) and len(cond) <= 60,
                             "weather.hourly condition must be a short string")
                    entry["condition"] = cond.strip()
                extra_h = set(row) - {"time", "temperature", "condition"}
                _require(not extra_h, "unknown weather.hourly fields: "
                         f"{', '.join(sorted(extra_h))}")
                hrs.append(entry)
            w["hourly"] = hrs
        extra = set(weather) - {"temperature", "forecast_low", "forecast_high",
                                "condition", "observed_at", "hourly"}
        _require(not extra, f"unknown weather fields: {', '.join(sorted(extra))}")
        clean["weather"] = w

    calendar = payload.get("calendar")
    if calendar is not None:
        _require(isinstance(calendar, dict), "calendar must be an object")
        events = calendar.get("events", [])
        _require(isinstance(events, list) and len(events) <= CAL_MAX_EVENTS,
                 f"calendar.events must be a list of at most {CAL_MAX_EVENTS}")
        out_events = []
        for ev in events:
            _require(isinstance(ev, dict), "calendar event must be an object")
            summary = ev.get("summary")
            _require(isinstance(summary, str) and 0 < len(summary.strip()) <= 200,
                     "calendar event summary must be a short non-empty string")
            start = ev.get("start")
            _require(_parse_dt(start) is not None,
                     "calendar event start must be an ISO timestamp")
            entry = {"summary": summary.strip(), "start": start}
            end = ev.get("end")
            if end is not None:
                _require(_parse_dt(end) is not None,
                         "calendar event end must be an ISO timestamp")
                entry["end"] = end
            extra = set(ev) - {"summary", "start", "end"}
            _require(not extra,
                     f"unknown calendar event fields: {', '.join(sorted(extra))}")
            out_events.append(entry)
        extra = set(calendar) - {"events", "fetched_at"}
        _require(not extra, f"unknown calendar fields: {', '.join(sorted(extra))}")
        c: dict[str, Any] = {"events": out_events}
        if calendar.get("fetched_at") is not None:
            _require(_parse_dt(calendar["fetched_at"]) is not None,
                     "calendar.fetched_at must be an ISO timestamp")
            c["fetched_at"] = calendar["fetched_at"]
        clean["calendar"] = c

    news = payload.get("news")
    if news is not None:
        _require(isinstance(news, dict), "news must be an object")
        headline = news.get("headline")
        _require(isinstance(headline, str) and 0 < len(headline.strip()) <= 400,
                 "news.headline must be a short non-empty string")
        source = news.get("source")
        _require(isinstance(source, str) and 0 < len(source.strip()) <= 60,
                 "news.source must be a short non-empty string")
        n = {"headline": headline.strip(), "source": source.strip()}
        if news.get("observed_at") is not None:
            _require(_parse_dt(news["observed_at"]) is not None,
                     "news.observed_at must be an ISO timestamp")
            n["observed_at"] = news["observed_at"]
        extra = set(news) - {"headline", "source", "observed_at"}
        _require(not extra, f"unknown news fields: {', '.join(sorted(extra))}")
        clean["news"] = n

    clock = payload.get("clock")
    if clock is not None:
        _require(isinstance(clock, dict), "clock must be an object")
        iso = clock.get("local_iso")
        _require(isinstance(iso, str) and bool(iso),
                 "clock.local_iso is required")
        parsed = _parse_dt(iso)
        # The container clock is UTC and is NOT the station timezone; a naive
        # stamp here would be ambiguous, so awareness is mandatory.
        _require(parsed is not None and parsed.tzinfo is not None,
                 "clock.local_iso must be a timezone-aware ISO timestamp")
        ck: dict[str, Any] = {"local_iso": iso}
        tz = clock.get("timezone")
        if tz is not None:
            _require(isinstance(tz, str) and 0 < len(tz) <= 64,
                     "clock.timezone must be a short string")
            ck["timezone"] = tz
        sun = clock.get("sun_up")
        if sun is not None:
            _require(isinstance(sun, bool), "clock.sun_up must be a boolean")
            ck["sun_up"] = sun
        for key in ("next_rising", "next_setting"):
            value = clock.get(key)
            if value is not None:
                parsed_sun = _parse_dt(value)
                _require(parsed_sun is not None
                         and parsed_sun.tzinfo is not None,
                         f"clock.{key} must be a timezone-aware ISO "
                         "timestamp")
                ck[key] = value
        extra = set(clock) - {"local_iso", "timezone", "sun_up",
                              "next_rising", "next_setting"}
        _require(not extra, f"unknown clock fields: {', '.join(sorted(extra))}")
        clean["clock"] = ck

    return clean


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
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


#: An unchanged snapshot is re-stamped on disk at most this often. The
#: integration republishes every CONTEXT_REFRESH_MINUTES (10) as a safety
#: net, and every publish carries fresh clock / observed_at / fetched_at
#: stamps even when nothing it describes has changed. Below this bound the
#: stored copy is kept: its stamps are at most this many minutes behind,
#: which is well inside every freshness rule that reads them
#: (WEATHER_MAX_AGE_MIN 60, CLOCK_MAX_AGE_MIN 90, CAL_MAX_AGE_MIN 120), so
#: a fact can never be judged stale because a write was skipped, and a
#: publisher that stops is still noticed -- at most this much later.
SNAPSHOT_REFRESH_MIN = 30.0

#: The stamps a publish regenerates whether or not the facts changed.
#: clock.local_iso is NOT one of them -- see _clock_moved.
_VOLATILE = (("captured_at",), ("weather", "observed_at"),
             ("calendar", "fetched_at"))

#: A republished station clock that differs from "the stored clock plus
#: the service time since it was stored" by at least this much has MOVED.
#: captured_at is stored in whole seconds, so a clock that merely advanced
#: always lands within a second of the prediction; a real jump is a whole
#: second or more. Exactly one second is therefore a move (the on-air
#: window edges are decided to the second).
CLOCK_TOLERANCE_S = 1.0


def _facts(snap: dict[str, Any]) -> dict[str, Any]:
    """The snapshot with its per-publish stamps removed: what it SAYS.
    The clock reading is removed too, because it is judged on its own
    terms by _clock_moved rather than by equality."""
    out = json.loads(json.dumps(snap))          # deep copy, plain types
    for path in _VOLATILE + (("clock", "local_iso"),):
        node: Any = out
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return out


def _clock_moved(prev: dict[str, Any], snap: dict[str, Any],
                 now_dt: datetime) -> bool:
    """Does the republished station clock say something the stored one
    plus elapsed service time does not?

    The station clock is a FACT, not a stamp: station_now derives every
    local time from stored local_iso + (service now - captured_at). A
    republish whose clock is exactly that prediction, in the same UTC
    offset, adds nothing. One that is not -- a DST change, a timezone
    change, a publisher whose clock jumped -- is new information and must
    be written at once. Anything unparseable counts as moved: when in
    doubt, write."""
    prev_clock = (prev.get("clock") or {})
    new_clock = (snap.get("clock") or {})
    prev_local = _parse_dt(prev_clock.get("local_iso"))
    new_local = _parse_dt(new_clock.get("local_iso"))
    if prev_local is None or new_local is None:
        return prev_local is not None or new_local is not None
    if prev_local.utcoffset() != new_local.utcoffset():
        return True
    elapsed = _age_minutes(prev.get("captured_at"), now_dt)
    if elapsed is None or elapsed < 0:
        return True
    expected = prev_local + timedelta(minutes=elapsed)
    return abs((new_local - expected).total_seconds()) >= CLOCK_TOLERANCE_S


def store_snapshot(clean: dict[str, Any],
                   now: datetime | None = None) -> dict[str, Any]:
    """Persist a validated snapshot (atomic replace). Server stamps
    captured_at; sections keep their own observed/fetched stamps.

    A snapshot that says nothing new -- the same facts, a clock that has
    merely advanced -- is not rewritten while the stored one is younger
    than SNAPSHOT_REFRESH_MIN; the stored document is returned instead, so
    the caller's captured_at is what is actually on disk.

    While a lab clock offset is in force, a snapshot whose clock is not
    the SIMULATED station clock is refused (LabClockMismatch) -- the live
    publisher must not drag the station back to real time mid-run."""
    now_dt = now or service_clock.legacy_now()
    _lab_ingress_check(clean, now_dt)
    snap = dict(clean)
    snap["captured_at"] = _iso(now_dt)
    prev = load_snapshot()
    if (prev is not None and _facts(prev) == _facts(snap)
            and not _clock_moved(prev, snap, now_dt)):
        age = _age_minutes(prev.get("captured_at"), now_dt)
        if age is not None and 0 <= age < SNAPSHOT_REFRESH_MIN:
            return prev
    _atomic_write(_snapshot_path(), snap)
    return snap


def load_snapshot() -> dict[str, Any] | None:
    """The newest snapshot, or None (missing/corrupt reads as absent --
    context must fail toward a music-only break, never toward an error)."""
    try:
        raw = json.loads(_snapshot_path().read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _load_state() -> dict[str, Any]:
    try:
        raw = json.loads(_state_path().read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


# --- deterministic fact formatting (the LLM phrases; it never computes) -------

def _fmt_clock(dt: datetime) -> str:
    """7:00 PM -- rendered manually so the format is cross-platform and the
    LLM never has to interpret a raw timestamp."""
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    if dt.minute:
        return f"{hour}:{dt.minute:02d} {ampm}"
    return f"{hour} {ampm}"


def _clock_forms(dt: datetime) -> list[str]:
    """The deterministic equivalents of the exact clock a writer may use
    instead of it, each bounded by the projected air minute: 'just after
    8 PM' only in the four minutes after the hour, 'coming up on 9 PM' only
    in the four minutes before it, and the same around the half hour. Never
    'about', 'almost' or a rounded guess -- an anchor is offered only when
    the arithmetic says the listener is genuinely that close to it."""
    forms: list[str] = []
    minute = dt.minute
    if 1 <= minute <= 4:
        forms.append("just after " + _fmt_clock(dt.replace(minute=0)))
    elif 56 <= minute <= 59:
        forms.append("coming up on " + _fmt_clock(
            (dt + timedelta(hours=1)).replace(minute=0)))
    elif 26 <= minute <= 29:
        forms.append("coming up on " + _fmt_clock(dt.replace(minute=30)))
    elif 31 <= minute <= 34:
        forms.append("just after " + _fmt_clock(dt.replace(minute=30)))
    return forms


def _fmt_lead(minutes: float) -> str:
    if minutes <= 75:
        return f"about {int(5 * round(minutes / 5))} minutes from now"
    halves = round(minutes / 30) / 2
    if halves == int(halves):
        halves = int(halves)
    return f"about {halves} hour{'s' if halves != 1 else ''} from now"


def _resolved_weather(snap: dict[str, Any],
                      local_now: datetime | None) -> dict[str, Any]:
    """The resolved weather facts for this snapshot at station-local `now`.
    Temporal truth is decided HERE (weather_context), never by the writer:
    the overnight low is the actual evening-to-morning minimum or absent,
    the high carries its upcoming/near/passed state, and daylight is a
    phase, not a boolean. {} when the station clock is unknown."""
    import weather_context
    return weather_context.resolve(snap.get("weather") or {},
                                   snap.get("clock") or {}, local_now)


def _fmt_weather(snap: dict[str, Any], local_now: datetime | None) -> str:
    import weather_context
    return weather_context.format_weather(_resolved_weather(snap, local_now))


#: The station's dayparts, by station-local hour: ONE definition, read by
#: the break's `daypart` colour and by the DJ shifts below.
MORNING_START_H = 5
AFTERNOON_START_H = 12
EVENING_START_H = 17
OVERNIGHT_START_H = 22


def _daypart(hour: int) -> str:
    if MORNING_START_H <= hour < AFTERNOON_START_H:
        return "morning"
    if AFTERNOON_START_H <= hour < EVENING_START_H:
        return "afternoon"
    if EVENING_START_H <= hour < OVERNIGHT_START_H:
        return "evening"
    return "overnight"


# --- DJ shifts -----------------------------------------------------------------
# Three air shifts cut from the SAME daypart boundaries, never a second
# set: Morning is the morning daypart, Afternoon the afternoon one, and
# Night is evening + overnight. Hours are STATION-LOCAL (station_now), so
# a DST change moves the shift with the wall clock, as a listener expects.

SHIFTS = ("morning", "afternoon", "night")
SHIFT_STARTS = {"morning": MORNING_START_H, "afternoon": AFTERNOON_START_H,
                "night": EVENING_START_H}


def shift_of(local: datetime) -> str:
    """The shift a station-local wall time falls in."""
    part = _daypart(local.hour)
    return part if part in ("morning", "afternoon") else "night"


def shift_instance(local: datetime) -> str:
    """WHICH occurrence of a shift a station-local time is in: the local
    date it STARTED on, and its name ("2026-01-15:night" runs from 17:00
    on the 15th to 05:00 on the 16th). Two times share an instance exactly
    when no shift boundary lies between them."""
    shift = shift_of(local)
    day = local.date()
    if shift == "night" and local.hour < MORNING_START_H:
        day = day - timedelta(days=1)
    return f"{day.isoformat()}:{shift}"


def shift_clock(now: datetime | None = None) -> datetime | None:
    """The clock shifts run on: station_now, and nothing else -- the Home
    Assistant station time, never the container's or a browser's. None
    when it is unknown, and then no shift is guessed."""
    return station_now(now)


def next_shift(shift: str) -> str:
    return SHIFTS[(SHIFTS.index(shift) + 1) % len(SHIFTS)]


# --- the topic editor ---------------------------------------------------------

def _event_key(ev: dict[str, Any]) -> str:
    return f"{ev.get('summary')}@{ev.get('start')}"


def select_topics(dj_settings: dict[str, Any],
                  now: datetime | None = None,
                  max_topics: int = 1,
                  allow: tuple[str, ...] = ("calendar", "news", "weather"),
                  snapshot: dict[str, Any] | None = None,
                  state: dict[str, Any] | None = None,
                  ) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Choose the optional contextual facts for ONE break.

    Returns (extras, marks): extras maps prompt roles (weather_context /
    upcoming_event / news_headline) to pre-formatted grounded fact strings;
    marks is the suppression-state update to commit IF AND ONLY IF the break
    actually renders (commit_marks). Empty extras = music-only break, which
    is the normal case. Never raises."""
    try:
        now_dt = now or service_clock.legacy_now()
        snap = snapshot if snapshot is not None else load_snapshot()
        if not snap or max_topics <= 0:
            return {}, None
        st = state if state is not None else _load_state()
        content = (dj_settings or {}).get("content") or {}
        news_cfg = (dj_settings or {}).get("news") or {}
        candidates: list[tuple[str, str, str, dict[str, Any]]] = []
        # (priority-key, role, fact, mark)

        # Calendar --------------------------------------------------------
        cal_snap = snap.get("calendar") or {}
        cal_age = _age_minutes(cal_snap.get("fetched_at")
                               or snap.get("captured_at"), now_dt)
        cal_fresh = cal_age is not None and cal_age <= CAL_MAX_AGE_MIN
        if content.get("calendar") and "calendar" in allow and cal_fresh:
            cal_state = st.get("calendar") or {}
            best: tuple[str, dict[str, Any], float] | None = None
            # All-day events are collected separately: they have no clock
            # time to measure a lead against, so they can never outrank a
            # timed event and are only used when nothing timed is eligible.
            best_allday: tuple[dict[str, Any], int] | None = None
            local_now = station_now(now_dt, snap)
            for ev in cal_snap.get("events") or []:
                raw_start = ev.get("start")
                start = _parse_dt(raw_start)
                if start is None:
                    continue                     # unparseable
                if ":" not in str(raw_start):
                    # Date-only: an all-day entry. Announced once, on its
                    # own day or the day before. Without a station clock
                    # there is no listener's date to compare against, and
                    # the container's is not a substitute.
                    if local_now is None:
                        continue
                    rec = cal_state.get(_event_key(ev)) or {}
                    if isinstance(rec, list):    # tolerate the older shape
                        rec = {p: None for p in rec}
                    if rec:
                        continue                 # one mention, ever
                    days_out = (start.date() - local_now.date()).days
                    if days_out < 0 or days_out > CAL_ALLDAY_LEAD_DAYS:
                        continue
                    if best_allday is None or days_out < best_allday[1]:
                        best_allday = (ev, days_out)
                    continue
                lead = _age_minutes(ev.get("start"), now_dt)
                if lead is None:
                    continue
                mins_to_start = -lead
                if mins_to_start <= 0:
                    continue  # already started/past
                rec = cal_state.get(_event_key(ev)) or {}
                if isinstance(rec, list):        # tolerate the older shape
                    rec = {p: None for p in rec}
                phase = None
                if (not rec
                        and CAL_FIRST_WINDOW[0] <= mins_to_start
                        <= CAL_FIRST_WINDOW[1]):
                    phase = "headsup"            # first-ever mention
                elif ("headsup" in rec and "reminder" not in rec
                        and CAL_REMINDER_WINDOW[0] <= mins_to_start
                        <= CAL_REMINDER_WINDOW[1]):
                    since_first = _age_minutes(rec.get("headsup"), now_dt)
                    if since_first is None or since_first >= CAL_REMINDER_SPACING_MIN:
                        phase = "reminder"       # second and final mention
                if phase is None:
                    continue
                if best is None or mins_to_start < best[2]:
                    best = (phase, ev, mins_to_start)
            if best is not None:
                phase, ev, mins_to_start = best
                start = _parse_dt(ev["start"])
                fact = (f"'{ev['summary']}' starts at {_fmt_clock(start)} "
                        f"({_fmt_lead(mins_to_start)})")
                candidates.append((f"calendar_{phase}", "upcoming_event", fact,
                                   {"calendar": {_event_key(ev): phase}}))
            elif best_allday is not None:
                ev, days_out = best_allday
                when = "today" if days_out == 0 else "tomorrow"
                # "(all day, no set time)" is the grounding: without it a
                # writer told an event "is today" will invent a clock time
                # for it, which is exactly the fabrication the facts rules
                # forbid everywhere else.
                fact = f"'{ev['summary']}' is {when} (all day, no set time)"
                candidates.append(("calendar_allday", "upcoming_event", fact,
                                   {"calendar": {_event_key(ev): "allday"}}))

        # News ------------------------------------------------------------
        if (content.get("news") and news_cfg.get("enabled")
                and "news" in allow):
            news = snap.get("news") or {}
            sources = news_cfg.get("sources") or []
            window = news_cfg.get("window_minutes") or []
            age = _age_minutes(news.get("observed_at")
                               or snap.get("captured_at"), now_dt)
            # The window is minutes past the hour on the STATION's clock --
            # not the container's, which differs in half-hour zones. Only a
            # snapshot with no station clock at all (a pre-clock publisher)
            # keeps the old reading of the service clock's own minute.
            minute = now_dt.minute
            if len(window) == 2:
                news_local = station_now(now_dt, snap)
                if news_local is not None:
                    minute = news_local.minute
            in_window = (len(window) != 2
                         or window[0] <= minute <= window[1])
            if (news.get("headline") and news.get("source") in sources
                    and age is not None and age <= NEWS_MAX_AGE_MIN
                    and in_window
                    and (st.get("news") or {}).get("last_headline")
                    != news["headline"]):
                fact = f"\"{news['headline']}\" (via {news['source'].upper()})"
                candidates.append(("news", "news_headline", fact,
                                   {"news": {"last_headline": news["headline"]}}))

        # Weather ---------------------------------------------------------
        if content.get("weather") and "weather" in allow:
            weather = snap.get("weather") or {}
            age = _age_minutes(weather.get("observed_at")
                               or snap.get("captured_at"), now_dt)
            fact = _fmt_weather(snap, station_now(now_dt, snap))
            if fact and age is not None and age <= WEATHER_MAX_AGE_MIN:
                w_state = st.get("weather") or {}
                since = _age_minutes(w_state.get("last_mention_at"), now_dt)
                changed = (
                    (weather.get("condition") and w_state.get("condition")
                     and weather["condition"] != w_state["condition"])
                    or (weather.get("temperature") is not None
                        and w_state.get("temperature") is not None
                        and abs(weather["temperature"] - w_state["temperature"])
                        >= WEATHER_NOTABLE_TEMP_DELTA))
                if since is None or since >= WEATHER_REPEAT_MIN or changed:
                    candidates.append(("weather", "weather_context", fact,
                                       {"weather": {
                                           "last_mention_at": _iso(now_dt),
                                           "condition": weather.get("condition"),
                                           "temperature": weather.get("temperature"),
                                       }}))

        candidates.sort(key=lambda c: PRIORITY.index(c[0])
                        if c[0] in PRIORITY else len(PRIORITY))
        extras: dict[str, str] = {}
        marks: dict[str, Any] = {}
        for _, role, fact, mark in candidates[:max_topics]:
            extras[role] = fact
            for section, update in mark.items():
                marks.setdefault(section, {}).update(update)
        return extras, (marks or None)
    except Exception:
        logger.warning("dj context selection failed (music-only break)",
                       exc_info=True)
        return {}, None


# --- the Smart DJ calendar packet (deterministic timing bands) ---------------
#
# The planner path's calendar, and the ONLY calendar the Smart DJ planner
# ever sees. It is deliberately NOT select_topics: that function formats a
# sentence for the deterministic writer's prompt and its two-mention
# policy (CAL_FIRST_WINDOW / CAL_REMINDER_WINDOW) is tuned for a writer
# that is handed a fact and told to state it. The planner is handed a
# bounded record and decides whether the event is worth any airtime at
# all, so its policy is stated in URGENCY BANDS instead.
#
# Every number here is computed in code against an injectable `now`. The
# model is never given a raw timestamp to subtract, and never asked how
# far away anything is: `minutes_until`, `timing_band`, `start_time` and
# `lead` all arrive pre-computed, and the prompt forbids arithmetic on
# them.

#: (band, lower-exclusive, upper-inclusive) minutes before start. The
#: bands are the product policy -- ~4h a single casual mention, ~2h
#: deliberate awareness, ~1h the priority topic -- widened into ranges so
#: that an ordinary break cadence actually lands inside one. An event more
#: than CAL_HORIZON_MIN away has no band and is not offered at all; an
#: event that has STARTED has no band either, which is what makes a
#: started event vanish rather than linger as "upcoming".
CAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("1h", 0.0, 90.0),
    ("2h", 90.0, 180.0),
    ("4h", 180.0, 300.0),
)
CAL_HORIZON_MIN = 300.0
#: The band an all-day entry gets instead: it has no clock time, so none
#: of the windows above can apply and it must never be delivered as an
#: imminent appointment. Lowest priority, one mention ever.
CAL_BAND_ALLDAY = "allday"
#: Which band wins when several events are eligible at once: urgency
#: first, all-day last.
CAL_BAND_PRIORITY = ("1h", "2h", "4h", CAL_BAND_ALLDAY)
#: How much of an event title may reach the packet.
CAL_TITLE_MAX = 64


def band_for(minutes_until: float) -> str | None:
    """The urgency band for an event `minutes_until` minutes away, or None
    when it has started, is past, or is beyond the horizon. Pure."""
    for name, lo, hi in CAL_BANDS:
        if lo < minutes_until <= hi:
            return name
    return None


def event_id(ev: dict[str, Any]) -> str:
    """A stable, NON-IDENTIFYING id for one calendar event.

    The suppression state file is keyed by `_event_key` (summary@start)
    and always has been, but a planner topic key travels further: into
    break telemetry, into the recent-break memory the next prompt is
    given, and into the dj_planner log line. Event summaries are the
    listener's private business and are never logged, so what travels is
    a hash of the same identity, not the summary."""
    import hashlib
    return hashlib.sha1(_event_key(ev).encode("utf-8")).hexdigest()[:10]


def calendar_packet(dj_settings: dict[str, Any] | None,
                    now: datetime | None = None,
                    snapshot: dict[str, Any] | None = None,
                    state: dict[str, Any] | None = None,
                    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The ONE calendar event worth offering the planner right now, and the
    suppression mark to commit IF AND ONLY IF it is actually spoken.

    Returns (event, marks). `event` is the bounded deterministic record

        {id, title, start_time, minutes_until, timing_band, all_day, lead}

    and nothing else: no location, no attendees, no description, no
    travel time, no preparation. The published snapshot carries only
    summary/start/end (the integration's publisher enforces that), so
    there is nothing else here to leak even by accident -- and the writer
    is told the same in the prompt.

    (None, None) whenever the calendar is off, the snapshot is missing or
    stale, or no event is inside a band it has not already used. Never
    raises: a calendar failure is a calendar-free break, never a lost one.
    """
    try:
        now_dt = now or service_clock.legacy_now()
        content = ((dj_settings or {}).get("content") or {})
        if not content.get("calendar"):
            return None, None
        snap = snapshot if snapshot is not None else load_snapshot()
        if not snap:
            return None, None
        cal_snap = snap.get("calendar") or {}
        age = _age_minutes(cal_snap.get("fetched_at")
                           or snap.get("captured_at"), now_dt)
        if age is None or age > CAL_MAX_AGE_MIN:
            return None, None       # a cancelled event must never be read out
        st = state if state is not None else _load_state()
        cal_state = st.get("calendar") or {}
        local_now = station_now(now_dt, snap)
        best: tuple[int, float, dict[str, Any]] | None = None
        for ev in (cal_snap.get("events") or [])[:CAL_MAX_EVENTS]:
            raw_start = ev.get("start")
            start = _parse_dt(raw_start)
            if start is None or not ev.get("summary"):
                continue
            rec = cal_state.get(_event_key(ev)) or {}
            if isinstance(rec, list):            # tolerate the older shape
                rec = {p: None for p in rec}
            if ":" not in str(raw_start):
                # Date-only: an all-day entry. Its own day or the day
                # before, once ever, against the STATION's date -- the
                # container's date is not the listener's.
                if local_now is None:
                    continue
                days_out = (start.date() - local_now.date()).days
                if days_out < 0 or days_out > CAL_ALLDAY_LEAD_DAYS:
                    continue
                if rec:
                    continue                     # one mention, ever
                cand = {"id": event_id(ev),
                        "title": _clip_title(ev["summary"]),
                        "start_time": None, "minutes_until": None,
                        "timing_band": CAL_BAND_ALLDAY, "all_day": True,
                        "lead": "today" if days_out == 0 else "tomorrow",
                        "_key": _event_key(ev)}
                rank = (CAL_BAND_PRIORITY.index(CAL_BAND_ALLDAY),
                        float(days_out))
            else:
                lead = _age_minutes(raw_start, now_dt)
                if lead is None:
                    continue
                minutes_until = -lead
                band = band_for(minutes_until)
                if band is None:
                    continue        # started, past, or beyond the horizon
                if band in rec:
                    continue        # this band has already been on air
                cand = {"id": event_id(ev),
                        "title": _clip_title(ev["summary"]),
                        "start_time": _fmt_clock(start),
                        "minutes_until": int(round(minutes_until)),
                        "timing_band": band, "all_day": False,
                        "lead": _fmt_lead(minutes_until),
                        "_key": _event_key(ev)}
                rank = (CAL_BAND_PRIORITY.index(band), minutes_until)
            if best is None or rank < best[:2]:
                best = (rank[0], rank[1], cand)
        if best is None:
            return None, None
        event = best[2]
        marks = {"calendar": {event.pop("_key"): event["timing_band"]}}
        return event, marks
    except Exception:
        logger.warning("dj calendar packet failed (calendar-free break)",
                       exc_info=True)
        return None, None


def _clip_title(summary: Any) -> str:
    s = str(summary or "").strip()
    if len(s) <= CAL_TITLE_MAX:
        return s
    return s[:CAL_TITLE_MAX - 1].rstrip() + "\u2026"


def station_now(now: datetime | None = None,
                snapshot: dict[str, Any] | None = None) -> datetime | None:
    """The current STATION-LOCAL, timezone-aware wall time, or None.

    THE one timezone basis for everything that reasons about the listener's
    clock. It comes only from the HA snapshot's timezone-aware
    clock.local_iso -- the container's own timezone is never the station's
    to assume, whatever Supervisor happened to set it to -- advanced by the
    service-clock minutes elapsed since the snapshot was captured, so a
    ten-minute-old publish still reads the current minute.

    A snapshot older than CLOCK_MAX_AGE_MIN (the publisher has stopped)
    is not advanced: local_iso carries a fixed UTC offset that a DST or
    timezone change would silently invalidate. The clock is then
    re-derived from clock.timezone (an IANA name) when the runtime has
    that zone, and is otherwise unknown.

    None when there is no snapshot, no clock, or a naive local_iso: callers
    must treat that as 'the station clock is unknown' and never substitute
    server time. Never raises.

    The service clock: elapsed time is measured between INSTANTS (the
    service clock's; a naive `now` is read honouring its fold), not between
    naive wall readings, and the advanced instant is expressed in the
    snapshot's IANA zone when the runtime knows it -- the offset in force
    at that instant, so a DST change no longer leaves the station an hour
    out until the next publish. The zone is trusted only when it AGREES
    with the published clock at the publish instant (_agreeing_zone);
    without such a zone the published fixed offset is kept, exactly as
    before."""
    try:
        now_inst = service_clock.instant(now)
        snap = snapshot if snapshot is not None else load_snapshot()
        if not snap:
            return None
        clock = snap.get("clock") or {}
        local = _parse_dt(clock.get("local_iso"))
        if local is None or local.tzinfo is None:
            return None
        zone = _zone(clock.get("timezone"))
        captured = _captured_instant(snap.get("captured_at"), local)
        elapsed = (None if captured is None
                   else (now_inst - captured).total_seconds() / 60.0)
        if elapsed is not None and elapsed > CLOCK_MAX_AGE_MIN:
            # the service clock, in the station zone (unchanged behaviour)
            return now_inst.astimezone(zone) if zone is not None else None
        if elapsed is not None and elapsed >= 0:
            local = local + timedelta(minutes=elapsed)
        agreeing = _agreeing_zone(zone, clock)
        return _civil(local, agreeing) if agreeing is not None else local
    except Exception:
        return None


def station_at(instant: datetime, snapshot: dict[str, Any] | None = None
               ) -> datetime | None:
    """The station-local wall time of ANY instant (a projection: when a
    break will be heard, where the next boundary lands), in the snapshot's
    IANA zone -- so a projection across a DST change carries the offset in
    force THEN. Falls back to the published fixed offset without a usable
    zone name; None without a published clock. Never raises."""
    try:
        inst = service_clock.instant(instant)
        snap = snapshot if snapshot is not None else load_snapshot()
        clock = (snap or {}).get("clock") or {}
        local = _parse_dt(clock.get("local_iso"))
        if local is None or local.tzinfo is None:
            return None
        zone = _agreeing_zone(_zone(clock.get("timezone")), clock)
        if zone is not None:
            return _civil(inst, zone)
        return inst.astimezone(local.tzinfo)
    except Exception:
        return None


def station_project(local: datetime, delta: timedelta,
                    snapshot: dict[str, Any] | None = None) -> datetime:
    """A station_now reading moved by `delta` of REAL time, as the station
    wall clock will read then: absolute arithmetic, then re-zoned. When the
    zone cannot be read the plain sum is returned, as before."""
    projected = station_at(local.astimezone(timezone.utc) + delta, snapshot)
    return projected if projected is not None else local + delta


def _zone(tz_name: Any):
    """The station's IANA zone, or None when the name is missing or unknown
    to the runtime (no tzdata, or a name it has never heard of)."""
    if not isinstance(tz_name, str) or not tz_name.strip():
        return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name.strip())
    except Exception:  # noqa: BLE001 - unknown zone / no tzdata
        return None


def _agreeing_zone(zone, clock: dict[str, Any]):
    """`zone` when its offset at the published instant IS the published
    offset (what Home Assistant always sends: local_iso is the zone's own
    reading), else None. A zone name that disagrees with the clock it came
    with is not evidence of anything -- the published offset stands, as it
    always did, and only a real change of offset AFTER the publish (DST)
    is taken from the zone."""
    if zone is None:
        return None
    local = _parse_dt(clock.get("local_iso"))
    if local is None or local.tzinfo is None:
        return None
    if local.astimezone(zone).utcoffset() != local.utcoffset():
        return None
    return zone


def station_zone(snapshot: dict[str, Any] | None = None):
    """The published station zone (a ZoneInfo), or None."""
    snap = snapshot if snapshot is not None else load_snapshot()
    return _zone(((snap or {}).get("clock") or {}).get("timezone"))


def _civil(inst: datetime, zone) -> datetime:
    """An instant on the station's wall clock with a FIXED-offset tzinfo
    (the zone's offset at that instant). Fixed, not the ZoneInfo itself:
    callers add and subtract timedeltas on station times, and Python does
    same-ZoneInfo arithmetic on the WALL clock -- a silent change of meaning
    across DST. A fixed offset keeps it absolute, as it always was."""
    loc = inst.astimezone(zone)
    return loc.replace(tzinfo=timezone(loc.utcoffset()))


def _captured_instant(stamp: Any, local: datetime) -> datetime | None:
    """captured_at as an instant. It is a legacy stamp (naive, process
    zone) written within seconds of local_iso, so in a repeated fall-back
    hour the reading nearest local_iso is the one that was meant."""
    dt = _parse_dt(stamp)
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    target = local.astimezone(timezone.utc)
    readings = {service_clock.from_legacy(dt.replace(fold=f)) for f in (0, 1)}
    readings.discard(None)
    if not readings:
        return None
    return min(readings, key=lambda r: abs((r - target).total_seconds()))


def service_facts(now: datetime | None = None,
                  snapshot: dict[str, Any] | None = None,
                  at: datetime | None = None) -> dict[str, Any]:
    """Standing service information for EVERY rendered break: the local
    station clock and fresh numeric weather. Unlike select_topics extras,
    these are never suppressed, never compete for topic slots, and carry
    no marks -- a radio station states the time and temperature all day.

    The clock is derived ONLY from the snapshot's timezone-aware
    clock.local_iso (the container's own timezone is never the station's
    to assume), advanced by the service-clock seconds elapsed since the
    snapshot was captured so a ten-minute-old publish still reads the
    current minute. (Advancing across a DST boundary can be off by an
    hour for one publish interval -- an accepted limit.) Empty dict when
    the snapshot lacks the data: older publishers keep working, and the
    break simply carries no service line. Never raises.

    `at` is the instant the words will actually be HEARD (an airtime
    estimate), and it is what station_time is projected to -- the listener
    hears "it's 8:37" at 8:37, not the minute the text was written. When
    `at` is None that instant is unknown, so the PRECISE clock is omitted
    entirely rather than stated stale: the caller's break simply carries no
    time, which the fact-conditional prompt rule already handles. The
    actual current station clock is still reported as station_time_now for
    diagnostics and provenance -- it is never shown to the writer.

    daypart is deliberately still derived from the best known instant even
    without `at`: it is coarse by nature, and a break that cannot state the
    minute can still say 'evening'. Daylight is a resolved PHASE,
    never the old sun_up boolean -- either the sun times support a real
    phase or daylight goes unmentioned."""
    try:
        now_dt = now or service_clock.legacy_now()
        snap = snapshot if snapshot is not None else load_snapshot()
        if not snap:
            return {}
        out: dict[str, Any] = {}
        clock = snap.get("clock") or {}
        local = station_now(now_dt, snap)
        spoken_at = None
        if local is not None:
            # The source clock is preserved untouched; the projection to
            # airtime is a separate, additive view of it.
            out["station_time_now"] = _fmt_clock(local)
            spoken_at = local
            if at is not None:
                lead = (at - now_dt).total_seconds()
                if lead > 0:
                    spoken_at = station_project(
                        local, timedelta(seconds=lead), snap)
                out["station_time"] = _fmt_clock(spoken_at)
                out["station_time_basis"] = "estimated_air"
                forms = _clock_forms(spoken_at)
                if forms:
                    out["station_time_forms"] = forms
            out["daypart"] = _daypart(spoken_at.hour)
        weather = snap.get("weather") or {}
        age = _age_minutes(weather.get("observed_at")
                           or snap.get("captured_at"), now_dt)
        # Resolution happens at the SPOKEN instant: minutes-to-sunset and
        # high/low tense describe the moment the words are heard. sun_up is
        # deliberately no longer surfaced -- a boolean invited "still
        # bright" minutes before sunset; either the sun
        # times resolve a real phase, or daylight goes unmentioned.
        resolved = _resolved_weather(
            snap, spoken_at if local is not None else None)
        import weather_context
        fact = weather_context.format_weather(resolved)
        if fact and age is not None and age <= WEATHER_MAX_AGE_MIN:
            out["current_weather"] = fact
            for key in ("high_state", "overnight_low_state"):
                if resolved.get(key):
                    out[key] = resolved[key]
        daylight_line = weather_context.format_daylight(resolved)
        if daylight_line:
            out["daylight_phase"] = resolved["daylight_phase"]
            out["daylight_line"] = daylight_line
        # A deterministic signature per service concept, for the
        # recent-repetition cooldown. Never shown to the writer -- the render
        # path reads it to gate repeats and to record what aired. A material
        # change (temp bucket, condition, high value/state, daylight phase,
        # or daypart) yields a different signature and re-enables the concept.
        out["service_sig"] = _service_signatures(out, resolved)
        return out
    except Exception:
        logger.warning("dj service facts failed (break continues without "
                       "service info)", exc_info=True)
        return {}


# --- service-repetition cooldown ---------------------------------------------

# Which context keys a suppressed concept removes. current_weather bundles
# temperature, condition and the daily high into one spoken line, so those
# gate together on one composite signature; daylight and the clock gate
# independently. station_time_now stays (diagnostic, never shown).
# Noise bands (degrees) for "did this fact MATERIALLY change?". Bounded in
# code, never LLM judgement: a 1-degree sensor/forecast wobble must not make a
# just-aired weather line eligible again.
_TEMP_BUCKET = 3
_HIGH_BUCKET = 2

_SERVICE_CONCEPT_KEYS = {
    "weather": ("current_weather", "high_state", "overnight_low_state"),
    "daylight": ("daylight_line", "daylight_phase"),
    "station_time": ("station_time", "station_time_basis",
                     "station_time_forms"),
}


def _bucket(value: Any, width: int) -> Any:
    """Quantize a number so tiny sensor/forecast noise is NOT a material
    change. Non-numbers pass through as '?'."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(round(float(value) / width) * width)
    return "?"


def _service_signatures(out: dict[str, Any],
                        resolved: dict[str, Any]) -> dict[str, str]:
    sig: dict[str, str] = {}
    if out.get("current_weather"):
        # The spoken weather line bundles temperature, condition (which carries
        # the precipitation/rain state) and the daily high, so they share one
        # signature; any one of them changing materially re-opens the topic.
        sig["weather"] = "|".join(str(x) for x in (
            _bucket(resolved.get("temperature_now"), _TEMP_BUCKET),
            resolved.get("condition") or "?",
            _bucket(resolved.get("forecast_high"), _HIGH_BUCKET),
            out.get("high_state") or "-",
            out.get("overnight_low_state") or "-"))
    if out.get("daylight_line"):
        sig["daylight"] = str(out.get("daylight_phase")
                              or resolved.get("daylight_phase") or "?")
    if out.get("station_time"):
        # The clock changes every minute, so material change is the DAYPART:
        # naming the time again inside the cooldown is the repetition to stop.
        sig["station_time"] = str(out.get("daypart") or "clock")
    return sig


# --- what the LISTENER actually heard -----------------------------------------
# Offering a service fact to the writer is not the same as airing it: the
# prompt says weather and time are OPTIONAL background, and the writer omits
# them roughly half the time. Recording an offered-but-unspoken fact as aired
# suppressed the topic for another cooldown window it never used up -- a ghost
# cooldown. These helpers read the FINAL accepted text and report which of the
# offered concepts were genuinely spoken.
#
# Detection is tied to the concrete values that were offered (this break's
# numbers, this break's clock, this break's daylight phase), never to generic
# mood words: a "warm, bright" guitar tone is not a weather report.

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
         7: "seventy", 8: "eighty", 9: "ninety"}


def _spellings(n: int) -> set[str]:
    """How a writer might render this number in prose: digits, and the
    hyphenated and spaced word forms ('82', 'eighty-two', 'eighty two')."""
    out = {str(n)}
    if 0 <= n < 20:
        out.add(_ONES[n])
    elif 20 <= n < 100:
        tens, ones = divmod(n, 10)
        word = _TENS[tens]
        out.add(word if not ones else f"{word}-{_ONES[ones]}")
        if ones:
            out.add(f"{word} {_ONES[ones]}")
    elif 100 <= n < 200:
        rest = n - 100
        if not rest:
            out.add("one hundred")
        else:
            for word in _spellings(rest):
                if not word.isdigit():
                    out.add(f"one hundred {word}")
                    out.add(f"a hundred {word}")
    return out


# Phrases that anchor a high/low claim to the offered forecast, mirroring
# weather_context.format_weather's own tense wording.
_HIGH_LOW_PHRASES = ("today's high", "todays high", "expected high",
                     "forecast high", "topped out", "the high of",
                     "high of", "overnight low", "the low of", "low of")

# Per-phase evidence for the daylight fact, drawn from DAYLIGHT_LINES. Bare
# 'dark' or 'bright' never qualifies -- a dark song is not a dark sky.
# "of daylight left" and "just after sunset" are here because a line such
# as "only forty minutes of daylight left" or "It's 8:40 PM, just after sunset"
# describes the phase and must be recorded. "of" keeps a
# story's "when the daylight left the valley" out.
_DAYLIGHT_EVIDENCE = {
    "early_daylight": ("sun came up", "sun's up", "sunrise", "just after "
                       "sunrise"),
    "full_daylight": ("full daylight", "broad daylight", "daylight"),
    "late_day": ("late-day", "late day", "daylight is starting to fade",
                 "light is fading", "daylight is fading", "fading light",
                 "of daylight left"),
    "sunset_imminent": ("sunset", "sun is about to set", "daylight is "
                        "fading"),
    "twilight": ("twilight", "sun has just set", "just set",
                 "just after sunset"),
    "dark": ("dark out", "dark outside", "it's dark", "it is dark",
             "after dark", "dark now"),
    "sunrise_imminent": ("sunrise", "still dark"),
}

_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", (text or "").lower())


#: What a weather mention stands next to. A number is a temperature, and a
#: condition word is the sky, only within WX_WINDOW_WORDS of one of these (or
#: of each other): "Room 4", "Two of their records", "Cold Orchard" and
#: "reached number nine" are not weather reports.
_WX_ANCHOR_RE = re.compile(
    r"^(?:degrees?|°|outside|outdoors|temperature|temp|skies|sky|weather|"
    r"forecast|currently|tonight|today|overnight)$")
_WX_ANCHOR_PHRASES = (("out", "there"), ("right", "now"), ("this", "morning"),
                      ("this", "afternoon"), ("this", "evening"))
WX_WINDOW_WORDS = 4
_WORD_RE = re.compile(r"[a-z0-9°]+(?:['’-][a-z0-9]+)*")


def _words(low: str) -> list[str]:
    return _WORD_RE.findall(low)


def _anchor_positions(words: list[str]) -> list[int]:
    out = [i for i, w in enumerate(words) if _WX_ANCHOR_RE.match(w)]
    for a, b in _WX_ANCHOR_PHRASES:
        out += [i for i in range(len(words) - 1) if words[i] == a and words[i + 1] == b]
    return out


def _number_positions(words: list[str], value: int) -> list[int]:
    """Where the text says this number, in digits or words ('fifty-four',
    'fifty four' spans two words and is found at its first)."""
    forms = _spellings(value)
    out = []
    for i, w in enumerate(words):
        if w in forms or (i + 1 < len(words) and f"{w} {words[i + 1]}" in forms):
            out.append(i)
    return out


def _run_matches(run: str, cond: str) -> bool:
    """A run of whole words names the condition: 'partly cloudy' for
    'partlycloudy', 'rainy' or 'raining' for 'rain', 'foggy' for 'fog'."""
    if run == cond:
        return True
    stem = lambda x: re.sub(r"(?:ing|y)$", "", x)
    a, b = stem(run), stem(cond)
    return bool(a) and (a == b or (len(a) > 2 and a[:-1] == b and a[-1] == a[-2]))


def _condition_positions(words: list[str], cond_letters: str) -> list[int]:
    out = []
    for i in range(len(words)):
        for k in (1, 2, 3):
            run = "".join(re.sub(r"[^a-z]", "", w) for w in words[i:i + k])
            if i + k <= len(words) and _run_matches(run, cond_letters):
                out.append(i)
                break
    return out


def _in_a_title(text: str, phrase: str) -> bool:
    """Is every occurrence of `phrase` capitalised in the middle of a
    sentence -- a name ("Wren Hollis with Daylight", "the Sunset Ferry")?"""
    hits = list(re.finditer(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])",
                            text or "", re.I))
    if not hits:
        return False
    for m in hits:
        before = (text or "")[:m.start()]
        first = m.group(0)[:1]
        sentence_start = not before.strip() or bool(re.search(r"[.!?:;]\s*$", before))
        if not first.isupper() or sentence_start:
            return False                   # one plain occurrence is enough
    return True


def _near(a: list[int], b: list[int]) -> bool:
    return any(abs(x - y) <= WX_WINDOW_WORDS for x in a for y in b)


def _weather_spoken(text: str, low: str, offered: dict[str, Any]) -> bool:
    """The OFFERED weather, said as weather: the current temperature next to
    a weather word or the condition, the forecast high or low beside its own
    phrase, or the condition (as whole words, not inside a title) next to a
    weather word or the temperature. Never a bare number or a substring:
    'Room 4', 'Cold Orchard', 'cloudberry', 'unclear' and 'reached
    number nine' must never pay owed weather."""
    fact = str(offered.get("current_weather") or "")
    if not fact:
        return False
    # a clock is not a temperature: "12:46" must not lend its digits to a
    # weather number
    words = _words(_CLOCK_RE.sub(" ", low))
    anchors = _anchor_positions(words)
    head = fact.split(";")[0]
    nums = [int(n) for n in re.findall(r"\d+", head)]
    temp = _number_positions(words, nums[0]) if nums else []
    cond = head.split(" and ")[-1].strip() if " and " in head else ""
    cond_letters = _letters(cond)
    conds = []
    if cond_letters and not _in_a_title(text, cond):
        conds = _condition_positions(words, cond_letters)
    if temp and (_near(temp, anchors) or _near(temp, conds)):
        return True
    if conds and (_near(conds, anchors) or _near(conds, temp)):
        return True
    # the high / low: its offered number right after "high"/"low" or one of
    # format_weather's own state phrases ("high of 75", "a high around eighty
    # two", "overnight low around 54"); a bare "topped out" is chart prose
    marks = {i for i, w in enumerate(words) if w in ("high", "highs", "low", "lows")}
    for phrase in _HIGH_LOW_PHRASES:
        pw = phrase.replace("'", "’" if "’" in low else "'").split()
        marks |= {i + len(pw) - 1 for i in range(len(words) - len(pw) + 1)
                  if words[i:i + len(pw)] == pw}
    for clause in fact.split(";")[1:]:
        for raw in re.findall(r"\d+", clause):
            if any(0 < p - m <= 3 for m in marks for p in _number_positions(words, int(raw))):
                return True
    return False


_FORM_RE = re.compile(r"^(just after|coming up on) (\d{1,2})(?::(\d{2}))? ([AP]M)$")
_STATION_TIME_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))? ([AP]M)$")
# Words that turn a stated clock into an estimate, or into a different
# instant altogether. "It's nearly 7:15 AM" and "It's roughly
# 8:40 PM" are time checks, yet neither is the supplied clock.
# A supplied FORM ("just after 8 PM") is matched whole, so its own anchor
# words are never read as a hedge on it.
_CLOCK_HEDGE_RE = re.compile(
    r"\b(?:close to|about|around|almost|nearly|roughly|approximately|near|"
    r"getting on for|going on|coming up on|coming up to|just after|"
    r"shortly after|a little after|just past|past|before|after|until|till|"
    r"by|since)\s*$")
_CLOCK_HEDGE_AFTER_RE = re.compile(r"^(?:\s*-?\s*ish\b|\s+or so\b)")


_HOUR_WORDS = ("twelve", "one", "two", "three", "four", "five", "six",
               "seven", "eight", "nine", "ten", "eleven", "twelve")
_TEEN_WORDS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
               "sixteen", "seventeen", "eighteen", "nineteen")
_TENS_WORDS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}


def _spelled_minute(minute: int) -> str:
    """Regex for a minute said in words: 'oh five', 'fifteen', 'fifty-eight'."""
    if minute < 10:
        return r"(?:oh|o)[\s-]+" + _HOUR_WORDS[minute]
    if minute < 20:
        return _TEEN_WORDS[minute - 10]
    tens = _TENS_WORDS[minute // 10]
    return tens if minute % 10 == 0 else (
        tens + r"[\s-]+" + _HOUR_WORDS[minute % 10])


def _clock_pattern(hour: str, minute: str | None, mer: str) -> str:
    """One supplied clock as spoken prose: the listed hour and minute (a
    top-of-hour clock may be said with or without ':00') and the listed
    meridiem, in any of the ordinary spellings (AM, am, a.m.). A bare
    'H:MM' is not the supplied time -- without its meridiem it is half of
    one.

    Or the same clock IN WORDS ("seven fifteen AM", "ten o'clock AM"):
    the speech engine reads digits out as words anyway, and since recent
    breaks do not show the writer digit clocks it writes
    the time in words now and then. It is the same time check, and the
    ledger still records the SUPPLIED form."""
    body = (r"(?<![\d:])" + hour
            + (":" + minute if minute and minute != "00" else r"(?::00)?"))
    try:
        h, m = int(hour), int(minute or 0)
        word = (r"\b" + _HOUR_WORDS[h % 12 if h != 12 else 12]
                + (r"(?:[\s-]+o'?\s?clock)?" if m == 0
                   else r"[\s-]+" + _spelled_minute(m)))
        body = r"(?:" + body + r"|" + word + r")"
    except (ValueError, IndexError):
        pass
    return body + r"\s*" + mer[0].lower() + r"\.?\s?m\.?(?![a-z])"


def clock_realizations(text: str, offered: dict[str, Any]
                       ) -> list[dict[str, Any]]:
    """Every place the text states the SUPPLIED clock -- the exact
    `station_time` or one of its `station_time_forms` -- in order, each
    flagged `hedged` when an approximation or a preposition bends it into
    something other than the time ('close to 7:15 AM', 'before 8 AM').

    Only authoritative supplied forms are recognised. A different clock, a
    daypart ('this morning') or a vague phrase ('around three') is not a
    realization of anything and does not appear here."""
    low = (text or "").lower()
    if not low:
        return []
    pats: list[tuple[str, str]] = []
    clock = str(offered.get("station_time") or "").strip()
    cm = _STATION_TIME_RE.match(clock)
    if cm:
        pats.append((clock, _clock_pattern(*cm.groups())))
    for form in offered.get("station_time_forms") or []:
        fm = _FORM_RE.match(str(form))
        if fm:
            words, hour, minute, mer = fm.groups()
            pats.append((str(form), re.escape(words) + r"\s+"
                         + _clock_pattern(hour, minute, mer)))
    found: list[dict[str, Any]] = []
    for form, pat in pats:
        for m in re.finditer(pat, low):
            if any(m.start() < f["end"] and f["start"] < m.end()
                   for f in found):
                continue
            hedged = bool(_CLOCK_HEDGE_RE.search(low[:m.start()])
                          or _CLOCK_HEDGE_AFTER_RE.match(low[m.end():]))
            found.append({"form": form, "start": m.start(), "end": m.end(),
                          "hedged": hedged})
    found.sort(key=lambda f: f["start"])
    return found


def spoken_clock(text: str, offered: dict[str, Any]) -> str | None:
    """The SUPPLIED clock the final text states -- the exact station_time,
    or one of its deterministic equivalent forms -- else None.

    ONE rule for both questions that ask it: whether an accepted draft paid
    a required clock (required_hosting) and whether the ledger may record a
    clock as delivered (`clock_aired`). The text must state the supplied
    clock EXACTLY ONCE, unhedged. A different or fabricated clock, an
    approximation ('close to 7:15 AM'), a daypart ('this morning'), a
    meridiem-less 'H:MM' or a repeated time check is not a verified
    delivery, and starts no suppression."""
    try:
        found = clock_realizations(text, offered)
    except Exception:  # noqa: BLE001 - an unreadable clock is no clock
        logger.warning("clock realization check failed (no clock recorded)",
                       exc_info=True)
        return None
    if len(found) != 1 or found[0]["hedged"]:
        return None
    return found[0]["form"]


def _station_time_spoken(text: str, low: str, offered: dict[str, Any]) -> bool:
    # A daypart is not the time. Recording "this morning" as station_time
    # aired would count it as a live reading for the
    # hosting payload and let a daypart stand in for a required clock.
    # Only a verified realization of the supplied clock is.
    return spoken_clock(text, offered) is not None


def _daylight_spoken(text: str, low: str, offered: dict[str, Any]) -> bool:
    """The offered phase's own evidence, as whole words and not inside a
    title: "Wren Hollis with Daylight", "the Sunset Ferry" and "Lost Bells
    Before Sunrise" must never pay the daylight fact."""
    phase = str(offered.get("daylight_phase") or "")
    for phrase in _DAYLIGHT_EVIDENCE.get(phase, ()):  # phase-specific only
        # a title capitalises the noun, not "of": "Hour of Daylight Left"
        anchor = phrase[3:] if phrase.startswith("of ") else phrase
        if re.search(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", low) \
                and not _in_a_title(text, anchor):
            return True
    return False


_SPOKEN_DETECTORS = {"weather": _weather_spoken,
                     "station_time": _station_time_spoken,
                     "daylight": _daylight_spoken}


def spoken_service_concepts(text: str,
                            offered: dict[str, Any]) -> set[str]:
    """Which of the OFFERED service concepts the final accepted text actually
    stated. Never raises: on any failure it reports nothing spoken, which
    starts no cooldown -- the safe direction, since the cost is at worst an
    earlier repeat rather than a fact silently suppressed for 15 minutes."""
    out: set[str] = set()
    try:
        low = (text or "").lower()
        if not low:
            return out
        for concept, detect in _SPOKEN_DETECTORS.items():
            keys = _SERVICE_CONCEPT_KEYS.get(concept, ())
            if not any(k in offered for k in keys):
                continue          # not offered this break: cannot have aired
            if detect(text, low, offered):
                out.add(concept)
    except Exception:  # noqa: BLE001
        logger.warning("spoken service detection failed (nothing recorded)",
                       exc_info=True)
        return set()
    return out


def recent_fact_ids(now: datetime | None = None) -> set[str]:
    """Music knowledge `fact_id`s that actually AIRED inside the cooldown.

    The durable half of fact novelty, in the ledger the station already
    keeps -- no second database. Expired entries are simply not returned
    (and are pruned on the next commit), so the file cannot grow without
    bound.

    An entry whose timestamp cannot be read is treated as EXPIRED, not as
    fresh: the alternative is a fact retired for good by one unparseable
    character, which is a silent, permanent loss and exactly the failure
    direction this module avoids everywhere else. Never raises -- on any
    failure nothing is suppressed, which at worst repeats a fact sooner
    than intended.
    """
    try:
        now_dt = now or service_clock.legacy_now()
        facts = (_load_state().get("facts") or {})
        if not isinstance(facts, dict):
            return set()
        horizon = FACT_COOLDOWN_DAYS * 24.0 * 60.0
        out = set()
        for fid, stamp in facts.items():
            age = _age_minutes(stamp, now_dt)
            if age is not None and age <= horizon:
                out.add(str(fid))
        return out
    except Exception:  # noqa: BLE001
        logger.warning("fact novelty read failed (nothing suppressed)",
                       exc_info=True)
        return set()


def recent_callback_keys(now: datetime | None = None) -> set[str]:
    """Callback relationship keys (dj_callback.link_key) that aired inside
    CALLBACK_COOLDOWN_HOURS. Same contract as recent_fact_ids: expired or
    unreadable entries do not count, and on any failure nothing is
    suppressed."""
    try:
        now_dt = now or service_clock.legacy_now()
        marks = (_load_state().get("callbacks") or {})
        if not isinstance(marks, dict):
            return set()
        horizon = CALLBACK_COOLDOWN_HOURS * 60.0
        return {str(k) for k, stamp in marks.items()
                if (lambda a: a is not None and a <= horizon)(
                    _age_minutes(stamp, now_dt))}
    except Exception:  # noqa: BLE001
        logger.warning("callback novelty read failed (nothing suppressed)",
                       exc_info=True)
        return set()


def commit_marks(marks: dict[str, Any] | None,
                 now: datetime | None = None) -> None:
    """Persist suppression bookkeeping AFTER a break actually rendered with
    the selected facts. Prunes calendar entries for events already started.
    Best-effort: a failed commit only weakens suppression, never breaks."""
    if not marks:
        return
    try:
        now_dt = now or service_clock.legacy_now()
        st = _load_state()
        if "weather" in marks:
            st["weather"] = marks["weather"]
        if "news" in marks:
            st["news"] = marks["news"]
        if marks.get("facts"):
            # Knowledge facts that were genuinely SPOKEN (the caller has
            # already checked the aired copy). Stamped now, and every
            # entry past the cooldown is dropped in the same pass so the
            # ledger stays the size of one cooldown window.
            facts = st.get("facts") or {}
            if not isinstance(facts, dict):
                facts = {}
            for fid in marks["facts"]:
                facts[str(fid)] = _iso(now_dt)
            horizon = FACT_COOLDOWN_DAYS * 24.0 * 60.0

            def _still_fresh(stamp: Any) -> bool:
                # NOT `(_age_minutes(...) or default)`: a fact stamped
                # this instant has age 0.0, which is falsy, and the `or`
                # form threw away the entry it had just written.
                age = _age_minutes(stamp, now_dt)
                return age is not None and age <= horizon

            st["facts"] = {f: t for f, t in facts.items()
                           if _still_fresh(t)}
        if marks.get("callbacks"):
            # earlier-session callbacks on a break that reached READY;
            # pruned to one cooldown window in the same pass
            cbs = st.get("callbacks") or {}
            if not isinstance(cbs, dict):
                cbs = {}
            for key in marks["callbacks"]:
                cbs[str(key)] = _iso(now_dt)
            horizon = CALLBACK_COOLDOWN_HOURS * 60.0
            st["callbacks"] = {
                k: t for k, t in cbs.items()
                if (lambda a: a is not None and a <= horizon)(
                    _age_minutes(t, now_dt))}
        if "calendar" in marks:
            cal = st.get("calendar") or {}
            for key, phase in marks["calendar"].items():
                rec = cal.get(key) or {}
                if isinstance(rec, list):        # tolerate the older shape
                    rec = {p: None for p in rec}
                rec.setdefault(phase, _iso(now_dt))
                cal[key] = rec
            # prune events whose start has passed
            pruned = {}
            for key, phases in cal.items():
                start = key.rsplit("@", 1)[-1]
                lead = _age_minutes(start, now_dt)
                # A date-only key is midnight: an all-day event announced
                # as "tomorrow" is still the same event all through the day
                # it names, so it is forgotten a day and a half later
                # rather than two hours after its midnight.
                horizon = (CAL_ALLDAY_FORGET_MIN if ":" not in start else 120)
                if lead is not None and lead > horizon:
                    continue
                pruned[key] = phases
            st["calendar"] = pruned
        _atomic_write(_state_path(), st)
    except Exception:
        logger.warning("dj context mark commit failed", exc_info=True)


# --- calendar diagnosis (Test Calendar) ---------------------------------
#: Why one event is or is not usable in a break right now. These are stable
#: machine keys; the panel owns the sentences a customer reads.
CAL_INELIGIBLE_REASONS = (
    "eligible_reminder", "eligible_headsup", "eligible_allday",
    "already_started", "already_announced", "outside_window",
    "beyond_allday_lead", "no_station_clock", "unreadable_start",
)


def weather_diagnosis(dj_settings: dict[str, Any] | None = None,
                      now: datetime | None = None,
                      snapshot: dict[str, Any] | None = None,
                      ) -> dict[str, Any]:
    """Exactly what Parlorwave holds of the weather, and what a break
    started this second would say about it.

    The read half of the same pair `calendar_diagnosis` is: `POST
    /dj/context` publishes weather and nothing answered "and what did you
    make of it". A station that never mentions the weather looks identical
    whether the entity was never chosen, the publisher stopped, the
    snapshot went stale, or the DJ is simply not allowed to mention it --
    four different things to do about it.

    It reports FACTS the station already holds; it is not a forecast
    service and contacts nothing. Resolution and wording come from the
    same `weather_context` code a real break uses, so the diagnosis cannot
    drift from the behaviour. Read-only, and never raises.
    """
    out: dict[str, Any] = {
        "received": False, "captured_at": None, "age_minutes": None,
        "station_time": None, "station_timezone": None,
        # dj.content.weather, read the way the engine reads it: absent
        # means on (schedule_pass asks for `.get("weather", True)`).
        "enabled_in_settings": bool(
            ((dj_settings or {}).get("content") or {}).get("weather", True)),
        "weather_present": False, "observed_at": None,
        "weather_age_minutes": None, "fresh": False,
        "condition": None, "temperature": None,
        "forecast_high": None, "forecast_low": None, "hourly_count": 0,
        "would_say": None, "daylight": None, "reason": "no_snapshot",
    }
    try:
        now_dt = now or service_clock.legacy_now()
        snap = snapshot if snapshot is not None else load_snapshot()
        if not snap:
            return out
        out["received"] = True
        out["captured_at"] = snap.get("captured_at")
        age = _age_minutes(snap.get("captured_at"), now_dt)
        out["age_minutes"] = round(age, 1) if age is not None else None
        local_now = station_now(now_dt, snap)
        out["station_time"] = _iso(local_now) if local_now else None
        out["station_timezone"] = (snap.get("clock") or {}).get("timezone")

        weather = snap.get("weather")
        if not isinstance(weather, dict) or not weather:
            out["reason"] = "no_weather_in_snapshot"
            return out
        out["weather_present"] = True
        out["condition"] = weather.get("condition")
        out["temperature"] = weather.get("temperature")
        out["forecast_high"] = weather.get("forecast_high")
        out["forecast_low"] = weather.get("forecast_low")
        hourly = weather.get("hourly")
        out["hourly_count"] = len(hourly) if isinstance(hourly, list) else 0
        out["observed_at"] = weather.get("observed_at")
        w_age = _age_minutes(weather.get("observed_at")
                             or snap.get("captured_at"), now_dt)
        out["weather_age_minutes"] = (round(w_age, 1)
                                      if w_age is not None else None)
        # The SAME freshness bound service_facts applies before it lets a
        # weather fact reach a break. Stale is its own answer, and it is
        # not the same answer as absent.
        out["fresh"] = w_age is not None and w_age <= WEATHER_MAX_AGE_MIN
        import weather_context
        resolved = _resolved_weather(snap, local_now)
        out["daylight"] = weather_context.format_daylight(resolved) or None
        said = weather_context.format_weather(resolved)
        if out["fresh"]:
            out["would_say"] = said or None
            out["reason"] = "ok"
        else:
            out["reason"] = "stale"
    except Exception:  # noqa: BLE001 - a diagnostic must never take the DJ down
        logger.warning("weather diagnosis failed", exc_info=True)
    return out


def calendar_diagnosis(dj_settings: dict[str, Any] | None = None,
                       now: datetime | None = None,
                       snapshot: dict[str, Any] | None = None,
                       state: dict[str, Any] | None = None,
                       ) -> dict[str, Any]:
    """Exactly what Parlorwave can see of the calendar, and why each event
    would or would not reach a break right now.

    The diagnostic behind Test Calendar. It answers the question the panel
    could not previously ask at all: POST /dj/context had no matching GET,
    so a calendar that never spoke was indistinguishable from a calendar
    that was never published. Read-only -- it selects nothing, commits no
    marks and changes no suppression state.

    Event summaries are returned because the person pressing the button is
    the person whose calendar it is. They are NEVER logged here, and the
    tests for this function use invented events only. Never raises.
    """
    out: dict[str, Any] = {
        "received": False, "captured_at": None, "age_minutes": None,
        "station_time": None, "station_timezone": None,
        "enabled_in_settings": bool(
            ((dj_settings or {}).get("content") or {}).get("calendar")),
        "calendar_present": False, "fetched_at": None,
        "calendar_age_minutes": None, "fresh": False,
        "event_count": 0, "events": [], "would_say": None,
    }
    try:
        now_dt = now or service_clock.legacy_now()
        snap = snapshot if snapshot is not None else load_snapshot()
        if not snap:
            return out
        out["received"] = True
        out["captured_at"] = snap.get("captured_at")
        age = _age_minutes(snap.get("captured_at"), now_dt)
        out["age_minutes"] = round(age, 1) if age is not None else None
        local_now = station_now(now_dt, snap)
        out["station_time"] = _iso(local_now) if local_now else None
        out["station_timezone"] = (snap.get("clock") or {}).get("timezone")

        cal_snap = snap.get("calendar")
        if not isinstance(cal_snap, dict):
            return out
        out["calendar_present"] = True
        out["fetched_at"] = cal_snap.get("fetched_at")
        cal_age = _age_minutes(cal_snap.get("fetched_at")
                               or snap.get("captured_at"), now_dt)
        out["calendar_age_minutes"] = (round(cal_age, 1)
                                       if cal_age is not None else None)
        out["fresh"] = cal_age is not None and cal_age <= CAL_MAX_AGE_MIN

        st = state if state is not None else _load_state()
        cal_state = st.get("calendar") or {}
        events = list(cal_snap.get("events") or [])
        out["event_count"] = len(events)
        rows: list[dict[str, Any]] = []
        for ev in events:
            raw_start = ev.get("start")
            start = _parse_dt(raw_start)
            all_day = start is not None and ":" not in str(raw_start)
            row = {"summary": ev.get("summary"), "start": raw_start,
                   "end": ev.get("end"), "all_day": all_day,
                   "reason": "unreadable_start", "minutes_away": None}
            rec = cal_state.get(_event_key(ev)) or {}
            if isinstance(rec, list):
                rec = {p: None for p in rec}
            if start is None:
                rows.append(row)
                continue
            if all_day:
                if local_now is None:
                    row["reason"] = "no_station_clock"
                elif rec:
                    row["reason"] = "already_announced"
                else:
                    days_out = (start.date() - local_now.date()).days
                    row["days_away"] = days_out
                    if days_out < 0:
                        row["reason"] = "already_started"
                    elif days_out > CAL_ALLDAY_LEAD_DAYS:
                        row["reason"] = "beyond_allday_lead"
                    else:
                        row["reason"] = "eligible_allday"
                rows.append(row)
                continue
            lead = _age_minutes(raw_start, now_dt)
            mins = None if lead is None else -lead
            row["minutes_away"] = round(mins, 1) if mins is not None else None
            if mins is None:
                row["reason"] = "unreadable_start"
            elif mins <= 0:
                row["reason"] = "already_started"
            elif "reminder" in rec:
                row["reason"] = "already_announced"
            elif not rec and CAL_FIRST_WINDOW[0] <= mins <= CAL_FIRST_WINDOW[1]:
                row["reason"] = "eligible_headsup"
            elif ("headsup" in rec
                    and CAL_REMINDER_WINDOW[0] <= mins <= CAL_REMINDER_WINDOW[1]):
                row["reason"] = "eligible_reminder"
            elif rec:
                row["reason"] = "already_announced"
            else:
                row["reason"] = "outside_window"
            rows.append(row)
        out["events"] = rows

        # What a break started this second would actually carry. Uses the
        # real selector so the diagnosis cannot drift from the behaviour.
        extras, _marks = select_topics(dj_settings or {}, now=now_dt,
                                       max_topics=1, allow=("calendar",),
                                       snapshot=snap, state=st)
        out["would_say"] = extras.get("upcoming_event")
    except Exception:  # noqa: BLE001 - a diagnostic must never take the DJ down
        logger.warning("calendar diagnosis failed", exc_info=True)
    return out


def calendar_evaluation(dj_settings: dict[str, Any] | None = None,
                        now: datetime | None = None,
                        snapshot: dict[str, Any] | None = None,
                        state: dict[str, Any] | None = None,
                        ) -> dict[str, Any]:
    """Structured evaluation of calendar availability and due state for forensics.

    Distinguishes:
    - disabled
    - unavailable (no snapshot, no calendar in snapshot, or stale snapshot)
    - checked_empty (0 events)
    - expired (all events already started)
    - available_not_due (outside window or already announced)
    - due_offered (event ready to offer to planner)
    """
    now_dt = now or service_clock.legacy_now()
    content = ((dj_settings or {}).get("content") or {})
    if not content.get("calendar"):
        return {
            "status": "disabled",
            "reason": "not_enabled",
            "calendar_due": False,
        }

    snap = snapshot if snapshot is not None else load_snapshot()
    if not snap:
        return {
            "status": "unavailable",
            "reason": "no_snapshot",
            "calendar_due": False,
        }

    cal_snap = snap.get("calendar")
    if not isinstance(cal_snap, dict):
        return {
            "status": "unavailable",
            "reason": "no_calendar_in_snapshot",
            "calendar_due": False,
        }

    age = _age_minutes(cal_snap.get("fetched_at") or snap.get("captured_at"), now_dt)
    if age is None or age > CAL_MAX_AGE_MIN:
        return {
            "status": "unavailable",
            "reason": "stale_snapshot",
            "calendar_due": False,
        }

    events = list(cal_snap.get("events") or [])
    if not events:
        return {
            "status": "checked_empty",
            "reason": "zero_events",
            "calendar_due": False,
        }

    # Check if all events already started
    all_started = True
    for ev in events:
        raw_start = ev.get("start")
        start = _parse_dt(raw_start)
        if start is None:
            continue
        if ":" not in str(raw_start):
            local_now = station_now(now_dt, snap)
            if local_now is not None:
                days_out = (start.date() - local_now.date()).days
                if days_out >= 0:
                    all_started = False
                    break
        else:
            lead = _age_minutes(raw_start, now_dt)
            if lead is not None and -lead > 0:
                all_started = False
                break

    if all_started:
        return {
            "status": "expired",
            "reason": "all_events_started",
            "calendar_due": False,
        }

    # Try calendar packet to see if an event is due
    cand, _ = calendar_packet(dj_settings, now=now_dt, snapshot=snap, state=state)
    if cand:
        return {
            "status": "due_offered",
            "reason": None,
            "calendar_due": True,
            "event_id": cand.get("id"),
            "timing_band": cand.get("timing_band"),
            "minutes_until": cand.get("minutes_until"),
        }

    return {
        "status": "available_not_due",
        "reason": "outside_window",
        "calendar_due": False,
    }


def weather_evaluation(dj_settings: dict[str, Any] | None = None,
                       now: datetime | None = None,
                       snapshot: dict[str, Any] | None = None,
                       context: dict[str, Any] | None = None,
                       budget: int | None = None,
                       ) -> dict[str, Any]:
    """Structured evaluation of weather availability and due state for forensics.

    Distinguishes:
    - disabled
    - unavailable (no snapshot, no weather in snapshot, or stale snapshot)
    - available_not_due (fresh snapshot, but not due under cooldown / repeat rules)
    - due_offered (fresh snapshot and due for presentation)
    """
    now_dt = now or service_clock.legacy_now()
    content = ((dj_settings or {}).get("content") or {})
    if not content.get("weather"):
        return {
            "status": "disabled",
            "reason": "not_enabled",
            "weather_due": False,
        }

    snap = snapshot if snapshot is not None else load_snapshot()
    if not snap:
        return {
            "status": "unavailable",
            "reason": "no_snapshot",
            "weather_due": False,
        }

    weather = snap.get("weather")
    if not isinstance(weather, dict) or not weather:
        return {
            "status": "unavailable",
            "reason": "no_weather_in_snapshot",
            "weather_due": False,
        }

    age = _age_minutes(weather.get("observed_at") or snap.get("captured_at"), now_dt)
    if age is None or age > WEATHER_MAX_AGE_MIN:
        return {
            "status": "unavailable",
            "reason": "stale_snapshot",
            "weather_due": False,
            "age_minutes": round(age, 1) if age is not None else None,
        }

    temp = weather.get("temperature")
    cond = weather.get("condition")
    forecast_high = weather.get("forecast_high")
    obs_age = round(age, 1) if age is not None else None

    weather_due = False
    ctx = context or {}
    if ctx.get("weather_context"):
        weather_due = True
    else:
        try:
            import service_presence
            if "weather" in service_presence.due(ctx, budget):
                weather_due = True
        except Exception:
            pass

    return {
        "status": "due_offered" if weather_due else "available_not_due",
        "reason": None if weather_due else "not_due",
        "weather_due": weather_due,
        "temperature": temp,
        "condition": cond,
        "forecast_high": forecast_high,
        "age_minutes": obs_age,
    }

