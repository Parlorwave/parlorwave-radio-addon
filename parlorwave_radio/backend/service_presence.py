"""When the clock, the weather and the light are DUE on air.

A real station reads the time constantly and the weather occasionally.
A single cooldown for all three service concepts -- one cadence-derived
window (15-45 min) keyed on a signature whose "material change" for the
clock is the daypart -- would withhold the clock for the whole window once
a time check aired, and then offer it only as optional background. Time
and weather do not share a cadence here:

  station time   due on every ordinary music break unless a truthful clock
                 ACTUALLY AIRED within the last 4:30 -- on any programme
                 element, a news intro's clock included; and the news
                 intro omits ITS clock under the same gap, so no two
                 clock announcements from any ParlorWave element land closer
                 than 4:30 apart
  weather        due 15 minutes after weather ACTUALLY AIRED, or at once on
                 a material change (the existing service signature:
                 temperature bucket, condition, high value/state, low
                 state); withheld in between, and OVERDUE at 30 minutes,
                 which is the ceiling the station may not quietly exceed
  daylight       optional seasoning: withheld while unchanged inside the
                 cadence window, flagged worth a mention when the phase
                 has changed since it was last mentioned, never required

The ledger is the existing dj_breaks telemetry -- `clock_aired` (the
supplied clock the final text stated) and `service_aired` (what the
listener heard, not what was offered) -- read newest-first. No new table, no
state file. Offering a fact never starts its interval; only airing it does,
and a failed or rejected break aired nothing.

Every gap is measured between AIRTIMES: from the air_start of the break
that spoke the fact to the instant THIS break's words will be heard. The
ledger stamps are air times, so the reference must be one too -- a break
renders up to RENDER_LEAD_S (~2.5 min) before it airs, and measuring to
the render instant understates every gap by that lead: a break that airs
4:45 after a clock is only about 2:15 past it when it renders, and would
be withheld for a gap it does not have.
The projected airtime is what the caller already computed to write the
clock itself; when there is no trustworthy projection the render instant
is the honest fallback (and the clock is usually absent anyway, since
service_facts omits it without an airtime).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger("musicbrain")

# 4:30, not 5:00: ordinary breaks a song or so apart often fall at 4:30-4:59,
# and the product wants the clock on the
# vast majority of music breaks. A repeat inside 4:30 is still a repeat.
TIME_MIN_GAP_S = 4 * 60 + 30
# WEATHER IS PROGRAMMING, NOT COLOUR. The target interval is
# 15 minutes and 30 is a ceiling: past it the station has quietly stopped
# doing one of its jobs, so an overdue weather line is required REGARDLESS
# of the word budget (see `due`). Between 15 and 30 the budget gate below
# still applies -- a cramped break may reasonably wait for a roomier one.
WEATHER_MIN_GAP_S = 15 * 60
WEATHER_MAX_GAP_S = 30 * 60
# Below this writer budget a due weather line stays optional: a coherent
# music handoff outranks utility material, and the clock (a few words) is
# the cheaper element to keep required. Overridden by `weather_overdue`.
SALIENT_MIN_WORDS = 45
_LEDGER_ROWS = 40
#: How far the (bucketed) temperature must move before weather is news again
#: inside WEATHER_MIN_GAP_S. The signature quantizes to 3 F buckets
#: (dj_context._TEMP_BUCKET), so ONE bucket is a reading that crossed a
#: bucket edge -- a one-degree change was enough. Two buckets is a real move.
WEATHER_MATERIAL_TEMP_STEP = 6


def material_weather_change(last_sig: str | None, sig: str | None) -> bool:
    """Has the weather changed enough to be worth saying again before the
    minimum gap is up?

    WEATHER CADENCE IS STABLE. Any difference in the signature
    used to re-open the topic and make it REQUIRED: a reading that crossed
    one bucket edge made the station say the weather twice in four minutes,
    again and again inside the 15-minute gap, and writer drafts that
    skipped the repeat were thrown away for it. Material now
    means the CONDITION changed (sunny -> rainy, which carries the
    precipitation state) or the temperature moved by
    WEATHER_MATERIAL_TEMP_STEP or more. A forecast high revision or the
    high/low narrative state advancing is not news; it waits for the gap.

    Signature: "temp|condition|high|high_state|low_state"
    (dj_context._service_signatures). An unreadable shape keeps the old
    rule (any difference), so nothing goes quiet on a format change."""
    if last_sig is None or sig is None or last_sig == sig:
        return False
    a, b = str(last_sig).split("|"), str(sig).split("|")
    if len(a) != 5 or len(b) != 5:
        return True
    if a[1] != b[1]:
        return True
    try:
        return abs(int(a[0]) - int(b[0])) >= WEATHER_MATERIAL_TEMP_STEP
    except ValueError:
        # a reading that came or went ('?') is not a change in the weather
        return False

_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}\s*[AP]M\b")
_NEWS_INTRO = "news_intro"


def _stamp(tel: dict[str, Any], updated: str | None) -> datetime | None:
    raw = tel.get("air_start") or updated
    try:
        ts = datetime.fromisoformat(raw) if raw else None
    except (TypeError, ValueError):
        return None
    if ts is not None and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def ledger(queue_id: str | None = None, db_path=None) -> dict[str, Any]:
    """Newest ACTUAL airing of each concept on this queue, from completed
    dj_breaks telemetry: {clock_at, weather_at, weather_sig, daylight_at,
    daylight_sig}. A break with a `coverage` verdict (cut short, or
    interrupted and not resumed where it was) is not evidence of anything. Any failure yields an empty ledger, which makes every
    concept due -- an extra time check is the safe direction."""
    out: dict[str, Any] = {"clock_at": None, "weather_at": None,
                           "weather_sig": None, "daylight_at": None,
                           "daylight_sig": None}
    try:
        import dj_engine
        c = dj_engine._connect(db_path)
        try:
            rows = c.execute(
                "SELECT telemetry, spoken_text, updated_at FROM dj_breaks "
                "WHERE state = 'completed'"
                + (" AND queue_id = ?" if queue_id else "")
                + " ORDER BY rowid DESC LIMIT ?",
                ((queue_id, _LEDGER_ROWS) if queue_id
                 else (_LEDGER_ROWS,))).fetchall()
        finally:
            c.close()
        for tel_json, spoken, updated in rows:
            try:
                tel = json.loads(tel_json or "{}")
            except Exception:  # noqa: BLE001
                continue
            ts = _stamp(tel, updated)
            if ts is None:
                continue
            if (tel.get("coverage") or {}).get("verdict"):
                # cut short or interrupted and resumed elsewhere: nothing
                # it said counts as heard (dj_engine._note_coverage)
                continue
            if out["clock_at"] is None and _clock_aired(tel, spoken):
                out["clock_at"] = ts
            aired = tel.get("service_aired") or {}
            if out["weather_at"] is None and aired.get("weather"):
                out["weather_at"], out["weather_sig"] = ts, aired["weather"]
            if out["daylight_at"] is None and aired.get("daylight"):
                out["daylight_at"] = ts
                out["daylight_sig"] = aired["daylight"]
    except Exception:  # noqa: BLE001
        logger.debug("service presence ledger unavailable", exc_info=True)
    return out


def _clock_aired(tel: dict[str, Any], spoken: str | None) -> bool:
    """A clock the listener really heard: the recorded supplied clock, or a
    news intro's template clock (deterministic text, aired verbatim)."""
    if tel.get("clock_aired"):
        return True
    return (tel.get("purpose") == _NEWS_INTRO
            and bool(_CLOCK_RE.search(spoken or "")))


def clock_suppressed(at: datetime, queue_id: str | None = None,
                     db_path=None) -> bool:
    """True when a truthful clock aired on this queue inside the gap -- the
    station-wide invariant for elements that carry their own clock (the
    news intro).

    `at` is the instant this element's words will be HEARD (the projected
    airtime), never the render instant: the ledger holds air times, so the
    gap is air-to-air. Never raises: an unreadable ledger suppresses
    nothing."""
    try:
        since = _elapsed(at, ledger(queue_id, db_path).get("clock_at"))
        return since is not None and since < TIME_MIN_GAP_S
    except Exception:  # noqa: BLE001
        return False


def _elapsed(now: datetime, then: datetime | None) -> float | None:
    if then is None:
        return None
    return (now - then).total_seconds()


def apply(service: dict[str, Any], state: dict[str, Any], at: datetime,
          daylight_window_s: int) -> dict[str, str]:
    """Decide presence for this break. Mutates `service` in place: a concept
    that is not due is REMOVED (never offered, so it cannot air and cannot
    be recorded); a due concept stays and is flagged (station_time_due,
    weather_due, daylight_salient) for the prompt. Returns concept ->
    signature for what remains offered, for the caller to reconcile against
    what was actually spoken.

    `at` is the instant these words will be HEARD -- the projected airtime
    the caller already used to write the clock -- so every gap is measured
    air-to-air against the ledger's air_start stamps. Callers with no
    trustworthy projection pass their render instant, which only shortens
    the measured gap and therefore only withholds."""
    import dj_context
    sig = service.pop("service_sig", None) or {}
    offered: dict[str, str] = {}
    keys = dj_context._SERVICE_CONCEPT_KEYS

    if "station_time" in service:
        since = _elapsed(at, state.get("clock_at"))
        if since is not None and since < TIME_MIN_GAP_S:
            for k in keys["station_time"]:
                service.pop(k, None)
        else:
            service["station_time_due"] = True
            offered["station_time"] = sig.get("station_time") or "clock"

    if "current_weather" in service:
        wsig = sig.get("weather") or "?"
        last_sig = state.get("weather_sig")
        changed = (last_sig is not None
                   and material_weather_change(last_sig, wsig))
        since = _elapsed(at, state.get("weather_at"))
        if since is None or changed or since >= WEATHER_MIN_GAP_S:
            service["weather_due"] = True
            if changed:
                service["weather_changed"] = True
            if since is not None and since >= WEATHER_MAX_GAP_S:
                # Past the ceiling: this one outranks the budget gate.
                service["weather_overdue"] = True
            offered["weather"] = wsig
        else:
            for k in keys["weather"]:
                service.pop(k, None)

    if "daylight_line" in service:
        dsig = sig.get("daylight") or "?"
        last_sig = state.get("daylight_sig")
        changed = last_sig is not None and dsig != last_sig
        since = _elapsed(at, state.get("daylight_at"))
        if changed:
            service["daylight_salient"] = True
            offered["daylight"] = dsig
        elif since is not None and since < daylight_window_s:
            for k in keys["daylight"]:
                service.pop(k, None)
        else:
            offered["daylight"] = dsig
    return offered


def due(context: dict[str, Any] | None, max_words: int | None = None
        ) -> list[str]:
    """The concepts this break is REQUIRED to state, after the word-budget
    policy: the clock whenever it is due; weather when the writer has room
    for it beside a coherent handoff, OR whenever weather is overdue past
    WEATHER_MAX_GAP_S, where room stops being the deciding question."""
    ctx = context or {}
    out: list[str] = []
    if ctx.get("station_time_due") and ctx.get("station_time"):
        out.append("station_time")
    if ctx.get("weather_due") and ctx.get("current_weather"):
        roomy = max_words is None or int(max_words) >= SALIENT_MIN_WORDS
        if roomy or ctx.get("weather_overdue") or ctx.get("required_hosting_only"):
            out.append("weather")
    return out


def omitted(text: str, context: dict[str, Any] | None,
            max_words: int | None = None) -> list[str]:
    """Which required concepts the text failed to state, judged by the same
    spoken-fact detectors the airing ledger uses -- a fabricated clock is
    an omission, not a time check."""
    import dj_context
    ctx = context or {}
    heard = dj_context.spoken_service_concepts(text, ctx)
    out = []
    for concept in due(ctx, max_words):
        if concept == "station_time":
            if not dj_context.spoken_clock(text, ctx):
                out.append(concept)
        elif concept not in heard:
            out.append(concept)
    return out


def repair_instruction(missing: list[str], context: dict[str, Any] | None
                       ) -> str:
    """One compact line appended to the SAME prompt for the single repair
    rerender. Adds no facts beyond the listed ones."""
    ctx = context or {}
    parts = []
    if "station_time" in missing:
        forms = ctx.get("station_time_forms") or []
        alt = (" (or " + " / ".join(f"'{f}'" for f in forms) + ")"
               if forms else "")
        parts.append(f"state the local time once, as listed: "
                     f"'{ctx.get('station_time')}'{alt}")
    if "weather" in missing:
        parts.append("give the listed current_weather once, briefly, with "
                     "its numbers and tense exactly as listed")
    if not parts:
        return ""
    return ("\n\nYour previous draft left out required station service. "
            "Keep the same handoff and, in the same short break, "
            + "; and ".join(parts) + ". Nothing else changes.")
