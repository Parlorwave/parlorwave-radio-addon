"""Deterministic weather/daylight temporal truth.

The context layer determines WHAT is true; the LLM only phrases resolved
facts. Three failures shape this module, each a temporal-truth
decision that must not be left to the writer:

  * an overnight low stated before midnight that is really higher than
    the actual pre-dawn low -- the provider's CALENDAR-DAY bucket low,
    not the evening-to-morning minimum.
  * brightness language minutes before sunset -- a binary sun_up
    boolean invites claims the sky no longer supports.
  * a future-tense expected high after the peak has already passed --
    without timing, the high is always phrased as future.

Everything here is PURE: aware datetimes in, resolved facts out. All
arithmetic happens in the STATION's timezone (every input timestamp must
carry an offset -- HA supplies them that way); naive timestamps are
ignored rather than guessed at, and the container/host clock never
participates. Missing context causes silence, not invention: no hourly
forecast means no overnight-low claim (the daily bucket low is exactly
the bug and is never re-labelled "overnight"), no sun times means no
daylight commentary, unknown high timing means a tense-neutral high.

Periods -- deliberately human, not meteorological:

  DAYTIME    sunrise..sunset (fallback 07:00..19:00 local): where the
             forecast high lives and ages.
  OVERNIGHT  sunset (fallback 18:00) .. next sunrise (fallback 07:30),
             crossing midnight as one unbroken period: "tonight's low"
             means the minimum across THIS span, and 00:10 is still the
             same night as 23:50.
  MORNING    sunrise ends the overnight period; only then does "tonight"
             roll forward to the coming night.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger("musicbrain")

# --- deterministic tolerances (named, not magic) ------------------------------

HIGH_NEAR_MIN = 45.0      # minutes around the peak hour that read as "now"
HIGH_NEAR_TEMP = 2.0      # deg: current within this of the high = "at it"
HIGH_PASSED_TEMP = 2.0    # deg: rest-of-day max this far under the high
                          # means the peak is behind us
LOW_NEAR_MIN = 45.0       # minutes around the overnight minimum = "now"
LOW_COVERAGE_SLACK_MIN = 90.0   # hourly data must reach within this of
                                # morning, or the minimum cannot be trusted
                                # (a short horizon revives the bucket-low bug)

LATE_DAY_MIN = 60.0       # daylight fading window before sunset
SUNSET_IMMINENT_MIN = 30.0
TWILIGHT_MIN = 40.0       # after sunset, before honest "dark"
SUNRISE_IMMINENT_MIN = 30.0
EARLY_DAYLIGHT_MIN = 60.0

MORNING_FALLBACK = (7, 30)    # overnight ends here without sunrise data
EVENING_FALLBACK = (18, 0)    # overnight begins here without sunset data
DAYTIME_FALLBACK = ((7, 0), (19, 0))

HIGH_UPCOMING = "upcoming"
HIGH_NEAR_NOW = "near_now"
HIGH_PASSED = "passed"

LOW_UPCOMING = "upcoming"
LOW_NEAR_NOW = "being_reached"

PHASES = ("early_daylight", "full_daylight", "late_day", "sunset_imminent",
          "twilight", "dark", "sunrise_imminent")


def _aware(value: Any) -> datetime | None:
    """Parse an ISO timestamp WITH offset; naive or malformed -> None."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _minutes(delta: timedelta) -> float:
    return delta.total_seconds() / 60.0


def _hourly(weather: dict[str, Any], now: datetime
            ) -> list[tuple[datetime, float]]:
    """[(aware time, temp)] sorted, recent past dropped, in station tz."""
    out = []
    for row in weather.get("hourly") or ():
        if not isinstance(row, dict):
            continue
        when = _aware(row.get("time"))
        temp = row.get("temperature")
        if when is None or not isinstance(temp, (int, float)) \
                or isinstance(temp, bool):
            continue
        if when < now - timedelta(hours=1):
            continue
        out.append((when.astimezone(now.tzinfo), float(temp)))
    out.sort()
    return out


# --- daylight phase ------------------------------------------------------------

def daylight(clock: dict[str, Any], now: datetime) -> dict[str, Any]:
    """{phase, minutes_to_sunset?, minutes_to_sunrise?} or {} when sun
    times are unavailable (silence, never a guess). `next_rising` /
    `next_setting` are HA sun.sun semantics: the NEXT future event, so the
    sun is up exactly when the next event is a setting; the matching
    sunrise/sunset of the current span is approximated as the next event
    minus ~24 h, which is plenty for phase boundaries measured in tens of
    minutes."""
    rising = _aware((clock or {}).get("next_rising"))
    setting = _aware((clock or {}).get("next_setting"))
    if rising is None or setting is None:
        return {}
    if rising < now and setting < now:
        # Both "next" events are in the past: the snapshot's sun facts are
        # stale (the publisher stopped). Silence, never yesterday's light.
        return {}
    out: dict[str, Any] = {}
    if setting < rising:                        # sun is up
        to_set = _minutes(setting - now)
        since_rise = _minutes(now - (rising - timedelta(hours=24)))
        if 0 <= since_rise <= EARLY_DAYLIGHT_MIN:
            out["phase"] = "early_daylight"
        elif to_set <= SUNSET_IMMINENT_MIN:
            out["phase"] = "sunset_imminent"
        elif to_set <= LATE_DAY_MIN:
            out["phase"] = "late_day"
        else:
            out["phase"] = "full_daylight"
        if to_set <= 90:
            out["minutes_to_sunset"] = max(0, int(round(to_set)))
    else:                                       # sun is down
        to_rise = _minutes(rising - now)
        since_set = _minutes(now - (setting - timedelta(hours=24)))
        if 0 <= since_set <= TWILIGHT_MIN:
            out["phase"] = "twilight"
        elif to_rise <= SUNRISE_IMMINENT_MIN:
            out["phase"] = "sunrise_imminent"
        else:
            out["phase"] = "dark"
        if to_rise <= 90:
            out["minutes_to_sunrise"] = max(0, int(round(to_rise)))
    return out


def _sun_today(clock: dict[str, Any], now: datetime,
               kind: str) -> datetime | None:
    """Today's sunrise ('rising') or sunset ('setting') from the NEXT-event
    attributes, whichever side of it we are on."""
    nxt = _aware((clock or {}).get(f"next_{kind}"))
    if nxt is None:
        return None
    nxt = nxt.astimezone(now.tzinfo)
    today = nxt - timedelta(hours=24) if nxt.date() != now.date() else nxt
    return today if today.date() == now.date() else None


# --- the daily high ------------------------------------------------------------

def resolve_high(weather: dict[str, Any], clock: dict[str, Any],
                 now: datetime) -> dict[str, Any]:
    """{forecast_high, high_state?, high_time?}. States only when
    defensible: hourly forecasts describe the FUTURE, so a peak that has
    already passed is detected by the rest of today's forecast never
    getting back near the daily high -- not by pretending to know history.
    Without hourly data the only honest state is 'the current temperature
    already touches the high' (near_now); otherwise no state at all."""
    high = weather.get("forecast_high")
    if not isinstance(high, (int, float)) or isinstance(high, bool):
        return {}
    out: dict[str, Any] = {"forecast_high": round(float(high))}
    temp = weather.get("temperature")
    at_high = (isinstance(temp, (int, float)) and not isinstance(temp, bool)
               and float(temp) >= float(high) - HIGH_NEAR_TEMP)
    sunset = _sun_today(clock, now, "setting")
    # Temporal evidence the peak is behind us. Sun-based when we have it
    # (past sunset), else a late-evening clock fallback. This OUTRANKS
    # numeric proximity to the high: late in the evening, a temperature
    # just under the day's high is the day cooling from its peak, not a
    # reading at today's high.
    after_peak_window = ((sunset is not None and now >= sunset)
                         or (sunset is None and now.hour >= 20))
    day_end = sunset or now.replace(
        hour=DAYTIME_FALLBACK[1][0], minute=DAYTIME_FALLBACK[1][1],
        second=0, microsecond=0)
    hourly = _hourly(weather, now)
    rest_of_day = [(t, v) for t, v in hourly
                   if t <= max(day_end, now) and t.date() == now.date()]
    # After sunset a "high" that is not meaningfully above the CURRENT
    # temperature is not a trustworthy DAYTIME peak -- at night the
    # provider's high field often just tracks the current/overnight reading
    # (late at night the high can equal the current one). Any "today's high"
    # claim is then misleading, so omit the high state entirely (in every
    # branch below) and let the overnight low carry the night.
    if after_peak_window and isinstance(temp, (int, float)) and \
            not isinstance(temp, bool) and float(high) - float(temp) \
            < HIGH_PASSED_TEMP:
        # Drop the number too: an untrustworthy nighttime "high" should
        # produce no high phrasing at all, not a tense-neutral one.
        out.pop("forecast_high", None)
        return out
    if rest_of_day:
        peak_time, peak_temp = max(rest_of_day, key=lambda tv: tv[1])
        # The rest of today never gets back near the high -> the peak has
        # passed. Numeric closeness of the CURRENT temp no longer vetoes
        # this: the forecast is the temporal evidence and it wins.
        if peak_temp <= float(high) - HIGH_PASSED_TEMP:
            out["high_state"] = HIGH_PASSED
            return out
        # near_now/upcoming are DAYTIME states: once past the day's peak
        # window (after sunset), a same-value hour near "now" is the day
        # holding warm into the evening, not the peak happening now.
        if not after_peak_window and (
                abs(_minutes(peak_time - now)) <= HIGH_NEAR_MIN or at_high):
            out["high_time"] = peak_time
            out["high_state"] = HIGH_NEAR_NOW
        elif peak_time > now and not after_peak_window:
            out["high_time"] = peak_time
            out["high_state"] = HIGH_UPCOMING
        else:
            out["high_state"] = HIGH_PASSED
        return out
    # No hourly data. Temporal evidence still outranks proximity.
    if after_peak_window:
        out["high_state"] = HIGH_PASSED
    elif at_high:
        out["high_state"] = HIGH_NEAR_NOW
    elif hourly:
        out["high_state"] = HIGH_PASSED
    return out


# --- the overnight low ---------------------------------------------------------

def _overnight_window(clock: dict[str, Any], now: datetime
                      ) -> tuple[datetime, datetime, bool]:
    """(start, end, inside_now) for the CURRENT overnight period.

    Inside-ness comes from the SUN STATE (next event is a rising = the sun
    is down = this is the night), never from "today's sunset" arithmetic --
    after midnight, today's sunset is tonight's FUTURE sunset, and naive
    date math inverted the window (the exact class of bug this module
    exists to kill). Before sunset the window is the coming night; from
    shortly before sunset until sunrise it is this night, crossing
    midnight as one unbroken span; sunrise rolls it forward."""
    rising = _aware((clock or {}).get("next_rising"))
    setting = _aware((clock or {}).get("next_setting"))
    if rising is not None:
        morning = rising.astimezone(now.tzinfo)
        if _minutes(morning - now) > 26 * 60:
            morning = morning - timedelta(hours=24)
    else:
        morning = (now if now.hour < MORNING_FALLBACK[0] else
                   now + timedelta(days=1)).replace(
            hour=MORNING_FALLBACK[0], minute=MORNING_FALLBACK[1],
            second=0, microsecond=0)
    if rising is not None and setting is not None:
        sun_down = rising.astimezone(now.tzinfo) < setting.astimezone(
            now.tzinfo)
    else:
        sun_down = now.hour >= 22 or now.hour < MORNING_FALLBACK[0]
    if sun_down:
        return now, morning, True             # this night, midnight or not
    if setting is not None:
        sunset = setting.astimezone(now.tzinfo)
    else:
        sunset = now.replace(hour=EVENING_FALLBACK[0],
                             minute=EVENING_FALLBACK[1], second=0,
                             microsecond=0)
        if sunset < now:
            sunset = now
    if now >= sunset - timedelta(minutes=30):
        return now, morning, True             # dusk: the night is beginning
    if morning <= sunset:
        morning += timedelta(hours=24)        # daytime: the COMING night
    return sunset, morning, False


def resolve_low(weather: dict[str, Any], clock: dict[str, Any],
                now: datetime) -> dict[str, Any]:
    """{overnight_low, overnight_low_state, overnight_low_time?} or {}.

    The minimum across the ACTUAL evening-to-morning span, from hourly
    data only -- and only when that data reaches close enough to morning
    that the minimum is real. The provider's daily bucket low is never
    re-labelled 'overnight': with no trustworthy hourly span there is
    simply no overnight-low fact (silence over invention)."""
    start, end, inside = _overnight_window(clock, now)
    hourly = _hourly(weather, now)
    span = [(t, v) for t, v in hourly if start <= t <= end]
    if not span:
        return {}
    if hourly[-1][0] < end - timedelta(minutes=LOW_COVERAGE_SLACK_MIN):
        # The DATA horizon (not the window slice) must reach morning, or
        # the minimum is a truncation artifact -- the bucket-low bug in
        # new clothes.
        return {}
    low_time, low = min(span, key=lambda tv: (tv[1], tv[0]))
    out = {"overnight_low": round(low), "overnight_low_time": low_time}
    if inside and abs(_minutes(low_time - now)) <= LOW_NEAR_MIN:
        out["overnight_low_state"] = LOW_NEAR_NOW
    else:
        out["overnight_low_state"] = LOW_UPCOMING
    return out


# --- the resolved fact model ---------------------------------------------------

def resolve(weather: dict[str, Any], clock: dict[str, Any],
            now: datetime | None) -> dict[str, Any]:
    """All resolved weather/daylight facts. Never raises; partial data
    yields partial facts. With no station clock (`now` None/naive), only
    the TIME-FREE facts survive: temperature, condition, and a
    tense-neutral forecast high -- no overnight low, no high state, no
    daylight, because every one of those is a temporal claim."""
    out: dict[str, Any] = {}
    try:
        weather = weather or {}
        temp = weather.get("temperature")
        if isinstance(temp, (int, float)) and not isinstance(temp, bool):
            out["temperature_now"] = round(float(temp))
        if weather.get("condition"):
            out["condition"] = str(weather["condition"])
        if now is None or now.tzinfo is None:
            high = weather.get("forecast_high")
            if isinstance(high, (int, float)) and not isinstance(high, bool):
                out["forecast_high"] = round(float(high))
            return out
        out.update(resolve_high(weather, clock or {}, now))
        out.update(resolve_low(weather, clock or {}, now))
        sun = daylight(clock or {}, now)
        if sun:
            out["daylight_phase"] = sun["phase"]
            for key in ("minutes_to_sunset", "minutes_to_sunrise"):
                if key in sun:
                    out[key] = sun[key]
    except Exception:  # noqa: BLE001 - context must never break a break
        logger.warning("weather resolution failed (facts omitted)",
                       exc_info=True)
    return out


def _clock12(when: datetime) -> str:
    hour = when.hour % 12 or 12
    return f"{hour}:{when.minute:02d} {'AM' if when.hour < 12 else 'PM'}"


# How a provider condition token is SAID. Home Assistant's weather
# vocabulary is machine-shaped ("clear-night", "partlycloudy",
# "lightning-rainy"); the canonical token stays untouched everywhere facts
# are compared (signatures, material-change checks, validation) and only
# the spoken fact string uses these. Same meteorology, plain words -- never
# a reinterpretation: "clear-night" is simply clear (the night is the
# daylight fact's business), "lightning-rainy" is rain and lightning.
# "exceptional" is HA's unclassified catch-all and carries no spoken
# meaning, so it is left unsaid rather than guessed at.
SPOKEN_CONDITIONS = {
    "clear-night": "clear",
    "sunny": "sunny",
    "partlycloudy": "partly cloudy",
    "cloudy": "cloudy",
    "fog": "foggy",
    "hail": "hail",
    "lightning": "lightning",
    "lightning-rainy": "rain and lightning",
    "pouring": "pouring rain",
    "rainy": "rainy",
    "snowy": "snowy",
    "snowy-rainy": "rain and snow",
    "windy": "windy",
    "windy-variant": "windy",
    "exceptional": "",
}


#: Daylight phases in which a clear sky is SAID "clear", never "sunny": the
#: sun is low (late day, sunset imminent) or down (twilight, dark, before
#: sunrise). Home Assistant keeps reporting "sunny" for a clear sky until its
#: own switch to "clear-night" some time after sunset, so without this an
#: evening break could say "sunny" after sunset and in twilight -- the
#: brightness claim the writer prompt already forbids late in the day
#: (break_packet). Same meteorology; only the word follows the light.
NOT_SUNNY_PHASES = ("late_day", "sunset_imminent", "twilight", "dark",
                    "sunrise_imminent")


def spoken_condition(token: Any, phase: str | None = None) -> str:
    """The spoken form of a condition token; an unknown token is spoken
    with its hyphens and underscores as spaces, never raw. With a verified
    daylight `phase`, a "sunny" reading after full daylight is said
    "clear"; without one, nothing is inferred."""
    raw = str(token or "").strip()
    if not raw:
        return ""
    key = raw.lower()
    if key == "sunny" and phase in NOT_SUNNY_PHASES:
        return "clear"
    if key in SPOKEN_CONDITIONS:
        return SPOKEN_CONDITIONS[key]
    return " ".join(raw.replace("-", " ").replace("_", " ").split())


def format_weather(facts: dict[str, Any]) -> str:
    """The grounded current_weather fact string. TENSE LIVES HERE, chosen
    from the resolved states -- the writer phrases; it never re-derives
    what 'tonight' or 'later' means. No brightness vocabulary ever:
    daylight is its own fact, and cloud conditions are not light levels.
    The condition is spoken (spoken_condition), never the provider token:
    otherwise a writer can read "clear-night" aloud."""
    bits = []
    if facts.get("temperature_now") is not None:
        bits.append(f"{facts['temperature_now']} degrees")
    cond = spoken_condition(facts.get("condition"),
                            facts.get("daylight_phase"))
    if cond:
        bits.append(cond)
    parts = []
    if bits:
        parts.append(" and ".join(bits))
    high, state = facts.get("forecast_high"), facts.get("high_state")
    if high is not None:
        if state == HIGH_PASSED:
            parts.append(f"already topped out around {high} today")
        elif state == HIGH_NEAR_NOW:
            parts.append(f"right around today's high of {high}")
        elif state == HIGH_UPCOMING:
            parts.append(f"expected high around {high} later today")
        else:
            parts.append(f"forecast high around {high}")
    low, low_state = facts.get("overnight_low"), \
        facts.get("overnight_low_state")
    if low is not None:
        when = facts.get("overnight_low_time")
        suffix = f" around {_clock12(when)}" if when is not None else ""
        if low_state == LOW_NEAR_NOW:
            parts.append(f"near the overnight low of {low} right now")
        else:
            parts.append(f"overnight low around {low}{suffix}")
    return "; ".join(parts)


DAYLIGHT_LINES = {
    "early_daylight": "the sun came up a short while ago",
    "full_daylight": "full daylight",
    "late_day": "late-day light, daylight is starting to fade",
    "sunset_imminent": "sunset is coming up shortly; daylight is fading",
    "twilight": "the sun has just set; twilight, not fully dark yet",
    "dark": "it is dark out",
    "sunrise_imminent": "still dark; sunrise is coming up shortly",
}


def format_daylight(facts: dict[str, Any]) -> str:
    """The daylight fact line, or ''. Built ONLY from sun geometry --
    deliberately no brightness claims (a clear sky eight minutes before
    sunset is not 'bright', and no illuminance sensor exists to say
    otherwise)."""
    phase = facts.get("daylight_phase")
    if phase not in DAYLIGHT_LINES:
        return ""
    line = DAYLIGHT_LINES[phase]
    if facts.get("minutes_to_sunset") is not None and phase in (
            "late_day", "sunset_imminent"):
        line += f" (sunset in about {facts['minutes_to_sunset']} minutes)"
    if facts.get("minutes_to_sunrise") is not None and phase == \
            "sunrise_imminent":
        line += f" (sunrise in about {facts['minutes_to_sunrise']} minutes)"
    return line
