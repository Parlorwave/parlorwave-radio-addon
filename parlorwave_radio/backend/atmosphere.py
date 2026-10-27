"""Atmospheric hosting: grounded weather, daylight and time-of-day colour.

The DJ should now and then talk naturally about the
weather, the light, the sunset and the time of day -- even when no formal
weather report is owed -- conversational, with harmless personality, and
never about anything Home Assistant has not reported. Owed weather, time and
calendar obligations stay exactly as reliable as before. Behind
`dj.atmosphere`, OFF by default.

    facts(snapshot, at)     what is TRUE at the moment the words are heard:
                            the station's resolver (weather_context.resolve /
                            spoken_condition, phase-aware) plus
                            derivations from the same snapshot -- weekday,
                            today's sun times, season and the day-length
                            trend (silent near solstices and equinoxes), and
                            rain arriving or easing within 3 h from HA's
                            hourly conditions. Stale weather (> 60 min) or
                            missing sun times mean silence.
    BANK                    code-owned lines. Each declares the facts that
                            LICENSE it, its topic, whether it states owed
                            weather VALUES (only inside the owed weather
                            sentence) or is condition-only BANTER (only when
                            no report is owed, never credited as one), the
                            ledger concept it records, and plain and playful
                            variants.
    choose(...)             the candidates for one break, best first. Weather
                            talk is an adaptive PREFERENCE for about one
                            mention an hour -- eagerness grows with the time
                            since the last one and is weighed by relevance (a
                            changed sky, rain), against a jittered threshold:
                            no quota, no minimum, no rigid cooldown. Topic
                            variety and the light's own phase changes decide
                            the rest.
    ledger_from(rows)       what aired, from break telemetry on EVERY queue
                            (dj_engine._atmosphere_rows): the atmosphere
                            records the render confirmed in the spoken text,
                            and the formal weather/daylight the ledger
                            recorded -- so a restart or another output
                            forgets nothing.
    screen(text, f, ...)    the grounding screen: any sky, light, season or
                            date word in code-owned atmosphere must be
                            licensed by the facts.
    planner_colour(...)     the INTERFACE for Smart DJ breaks (not wired):
                            licensed colour items a planner packet could
                            offer, from the same facts, policy and ledger.

Nothing here imports the story modules (test_story_rollout); split_hosting
places a chosen line, the engine records it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import dj_context as djc
import weather_context as wc

FAMILY = {"sunny": "clear", "clear-night": "clear", "partlycloudy": "partly_cloudy",
          "cloudy": "cloudy", "fog": "fog", "rainy": "rain", "pouring": "heavy_rain",
          "lightning-rainy": "storm", "lightning": "storm", "hail": "storm",
          "snowy": "snow", "snowy-rainy": "mix", "windy": "wind", "windy-variant": "wind"}
PRECIP = {"rain", "heavy_rain", "storm", "snow", "mix"}
#: the condition said before a noun ("on a clear fall evening")
ADJ = {"clear": "clear", "partly_cloudy": "partly cloudy", "cloudy": "cloudy",
       "fog": "foggy", "rain": "rainy", "snow": "snowy", "wind": "windy"}
MONTHS = ("January February March April May June July August September October "
          "November December").split()
WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()

# --- policy constants (minutes) ------------------------------------------------------
#: the target of about one atmospheric weather mention an hour: the point at
#: which weather talk is fully eager -- a preference, not a cooldown
WEATHER_TALK_TARGET_MIN = 60.0
#: the same sky said again (formally or as banter) inside this is not news;
#: after it, an unchanged sky may come back -- in other words -- at about the
#: hourly preference
SAME_SKY_MIN = 45.0
#: the evening arc -- an hour of daylight left / sun about to set / just set --
#: is ONE story per evening; night no sooner than an hour after it
ARC_MIN = 12 * 60.0
NIGHT_AFTER_ARC_MIN = 60.0
#: other topics come back no sooner than this
TOPIC_GAP_MIN = {"season.daylength": 18 * 60.0, "season.name": 6 * 60.0,
                 "week": 3 * 60.0, "wx.change": 60.0, "wx.trend": 90.0}
#: a variant (exact wording) is not repeated inside this
VARIANT_GAP_MIN = 3 * 60.0
#: how far back the ledger looks
LEDGER_HOURS = 14


# --- facts -----------------------------------------------------------------------------
def _dur(minutes: float | None) -> str | None:
    """A duration in words, never digits (no second clock, no stray number
    for the weather detector to credit)."""
    if minutes is None:
        return None
    m = float(minutes)
    for hi, words in ((7, None), (12, "about ten minutes"), (18, "about a quarter of an hour"),
                      (24, "about twenty minutes"), (37, "about half an hour"),
                      (52, "about forty-five minutes"), (75, "about an hour"),
                      (100, "about an hour and a half")):
        if m <= hi:
            return words
    return None


def facts(snapshot: dict[str, Any] | None, at: datetime | None) -> dict[str, Any]:
    """Atmosphere facts at `at` (station-local, timezone-aware: when the words
    are heard). {} without a station clock -- every claim here is temporal."""
    snapshot = snapshot or {}
    if at is None or at.tzinfo is None:
        return {}
    clock, weather = snapshot.get("clock") or {}, snapshot.get("weather") or {}
    part = djc._daypart(at.hour)
    f: dict[str, Any] = {"at": at, "weekday": WEEKDAYS[at.weekday()], "month": at.month,
                         "day": at.day, "daypart": "night" if part == "overnight" else part}
    resolved = wc.resolve(weather, clock, at)
    observed = wc._aware(weather.get("observed_at") or snapshot.get("captured_at"))
    fresh = observed is not None and timedelta(0) <= at - observed <= timedelta(
        minutes=djc.WEATHER_MAX_AGE_MIN)
    phase = resolved.get("daylight_phase")
    if phase:
        f["phase"] = phase
        for k in ("minutes_to_sunset", "minutes_to_sunrise"):
            if k in resolved:
                f[k] = resolved[k]
    if fresh and resolved.get("condition"):
        token = str(resolved["condition"]).lower()
        f["sky"] = FAMILY.get(token)
        f["cond"] = wc.spoken_condition(token, phase)
        if resolved.get("temperature_now") is not None:
            f["temp"] = resolved["temperature_now"]
        for k in ("overnight_low", "overnight_low_state"):
            if resolved.get(k) is not None:
                f[k] = resolved[k]
    rise, sett = wc._sun_today(clock, at, "rising"), wc._sun_today(clock, at, "setting")
    if rise and sett and sett > rise:
        f["sunset_today"] = sett
        day_h = (sett - rise).total_seconds() / 3600
        if rise <= at:
            f["minutes_since_sunrise"] = int((at - rise).total_seconds() // 60)
        mo = at.month
        north = (mo in (10, 11, 12, 1, 2) and day_h < 11.75) or (mo in (4, 5, 6, 7, 8) and day_h > 12.25)
        south = (mo in (10, 11, 12, 1, 2) and day_h > 12.25) or (mo in (4, 5, 6, 7, 8) and day_h < 11.75)
        if north or south:
            seasons = ["winter", "winter", "spring", "spring", "spring", "summer", "summer",
                       "summer", "autumn", "autumn", "autumn", "winter"]
            s = seasons[mo - 1]
            if south:
                s = {"winter": "summer", "summer": "winter", "spring": "autumn",
                     "autumn": "spring"}[s]
            f["season"] = s
            f["season_word"] = "fall" if s == "autumn" and str(
                getattr(at.tzinfo, "key", "")).startswith("America/") else s
            doy = at.timetuple().tm_yday
            if not (163 <= doy <= 182 or doy >= 345 or doy <= 1):
                shortening = 182 <= doy <= 345
                f["trend"] = ("shortening" if shortening else "lengthening") if north else \
                    ("lengthening" if shortening else "shortening")
    if f.get("sky"):
        ahead = []
        for row in weather.get("hourly") or ():
            t = wc._aware(row.get("time"))
            if t is not None and at < t <= at + timedelta(hours=3):
                ahead.append((t, FAMILY.get(str(row.get("condition") or "").lower())))
        ahead.sort()
        if f["sky"] not in PRECIP:
            wet = next(((t, fam) for t, fam in ahead if fam in PRECIP), None)
            if wet is not None:
                f["precip_later"] = wet[1]
                part = djc._daypart(wet[0].hour)
                f["precip_later_when"] = {"morning": "later this morning",
                                          "afternoon": "later this afternoon",
                                          "evening": "this evening"}.get(part, "tonight")
        elif len([1 for _, fam in ahead if fam not in PRECIP]) >= 2:
            f["drying_later"] = True
    return f


def facts_now(now_dt: datetime, air_at: datetime | None = None,
              snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """facts() for the station's own snapshot at the moment these words air
    (`air_at` on the engine's clock), projected the way service_facts
    projects the clock. {} when anything is missing."""
    try:
        snap = snapshot if snapshot is not None else djc.load_snapshot()
        if not snap:
            return {}
        local = djc.station_now(now_dt, snap)
        if local is None:
            return {}
        if air_at is not None:
            local = djc.station_project(local, air_at - now_dt, snap)
        return facts(snap, local)
    except Exception:  # noqa: BLE001 - colour never costs a break
        return {}


# --- the bank ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Line:
    id: str
    family: str
    kind: str                         # sentence | tail | weather | time
    when: Callable[[dict], Any]
    variants: tuple                   # callables f -> text; plain first, then playful
    topic: Callable[[dict], str] = lambda f: ""
    owed_weather: bool = False        # states weather VALUES: only in the owed sentence
    banter: bool = False              # condition-only: only when no report is owed
    records: str | None = None        # the service concept it states ("daylight")
    weather_talk: bool = False        # counts toward the ~hourly weather preference
    with_time: Callable | None = None  # (f, time) -> text, for a kind "weather" line


def _sunset(f):
    return _dur(f.get("minutes_to_sunset"))


def _noun(f):
    return {"morning": "morning", "afternoon": "afternoon", "evening": "evening"}.get(
        f["daypart"], "night")


def _part_of_month(f):
    return "Early" if f["day"] <= 10 else "Mid" if f["day"] <= 20 else "Late"


_NIGHT = ("twilight", "dark", "sunrise_imminent")


def _sky_now(f) -> dict[str, tuple]:
    """Condition-only banter per sky (no numbers, no trend, no view)."""
    night = f.get("phase") in _NIGHT
    return {
        "clear": (lambda f: "Clear skies out there tonight." if night
                  else "Clear skies out there right now.",
                  lambda f: "Not much in the way of clouds out there tonight." if night
                  else "Clear skies out there, for the record.",
                  lambda f: "It's a clear night out there." if night
                  else "Nothing but clear sky out there."),
        "partly_cloudy": (lambda f: "A few clouds out there right now.",
                          lambda f: "Partly cloudy out there, if you were wondering.",
                          lambda f: "Some clouds around out there right now."),
        "cloudy": (lambda f: "Cloudy out there right now.",
                   lambda f: "It's a cloudy one out there.",
                   lambda f: "Plenty of cloud cover out there right now."),
        "rain": (lambda f: "It's raining out there right now.",
                 lambda f: "Rain's coming down out there, so the radio's a good call."),
        "heavy_rain": (lambda f: "It's pouring out there right now.",
                       lambda f: "It is really coming down out there."),
        "storm": (lambda f: "There's thunder and lightning out there right now.",
                  lambda f: "Stormy out there, so stay put with us."),
        "snow": (lambda f: "It's snowing out there right now.",
                 lambda f: "Snow's coming down out there."),
        "mix": (lambda f: "Rain and snow out there right now.",
                lambda f: "A rain and snow mix out there right now."),
        "fog": (lambda f: "It's foggy out there right now.",
                lambda f: "Foggy out there, so take it easy if you're driving."),
        "wind": (lambda f: "It's windy out there right now.",
                 lambda f: "Windy one out there right now."),
    }


def _sky_variants(f):
    return _sky_now(f).get(f.get("sky"), ())


BANK: tuple[Line, ...] = (
    # --- the light: the resolver's authoritative phase; records daylight -----------
    Line("lt.late.left", "light.sunset", "sentence",
         lambda f: f.get("phase") == "late_day" and _sunset(f),
         (lambda f: f"There's {_sunset(f)} of daylight left.",
          lambda f: f"Only {_sunset(f)} of daylight left today."),
         topic=lambda f: "light:late_day", records="daylight"),
    Line("lt.late.tail", "light.sunset", "tail",
         lambda f: f.get("phase") == "late_day" and _sunset(f),
         (lambda f: f"with sunset {_sunset(f)} away",),
         topic=lambda f: "light:late_day", records="daylight"),
    Line("lt.soon", "light.sunset", "sentence",
         lambda f: f.get("phase") == "sunset_imminent",
         (lambda f: f"Sunset's {_sunset(f)} away." if _sunset(f) else "The sun's about to set.",
          lambda f: "The sun's on its way down."),
         topic=lambda f: "light:sunset_imminent", records="daylight"),
    Line("lt.dusk", "light.dusk", "sentence", lambda f: f.get("phase") == "twilight",
         (lambda f: "The sun's just set.", lambda f: "The sun's just gone down."),
         topic=lambda f: "light:twilight", records="daylight"),
    Line("lt.dusk.tail", "light.dusk", "tail", lambda f: f.get("phase") == "twilight",
         (lambda f: "just after sunset",),
         topic=lambda f: "light:twilight", records="daylight"),
    Line("lt.night.tail", "light.night", "tail", lambda f: f.get("phase") == "dark",
         (lambda f: "and it's dark out now", lambda f: "and it's officially dark out"),
         topic=lambda f: "light:dark", records="daylight"),
    Line("lt.morning", "light.morning", "sentence",
         lambda f: f.get("phase") == "early_daylight" and _dur(f.get("minutes_since_sunrise")),
         (lambda f: f"The sun came up {_dur(f.get('minutes_since_sunrise'))} ago.",),
         topic=lambda f: "light:early_daylight", records="daylight"),
    Line("lt.dawn", "light.dawn", "sentence", lambda f: f.get("phase") == "sunrise_imminent",
         (lambda f: (f"Still dark out, with sunrise {_dur(f.get('minutes_to_sunrise'))} away."
                     if _dur(f.get("minutes_to_sunrise")) else
                     "Still dark out, but sunrise isn't far off."),),
         topic=lambda f: "light:sunrise_imminent", records="daylight"),
    # --- season and day length: date + sun geometry ---------------------------------
    Line("ssn.short", "season.daylength", "sentence",
         lambda f: f.get("trend") == "shortening" and f.get("phase") in ("twilight", "dark")
         and f.get("sunset_today") is not None and f["sunset_today"].hour < 19
         and f["at"].hour < 21,
         (lambda f: f"{_part_of_month(f)} {MONTHS[f['month'] - 1]}, and the sun's setting "
                    "a little earlier every day.",
          lambda f: "The days keep getting shorter this time of year."),
         topic=lambda f: "season.daylength"),
    Line("ssn.long", "season.daylength", "sentence",
         lambda f: f.get("trend") == "lengthening" and f.get("phase") in (
             "late_day", "sunset_imminent") and f["at"].hour >= 19,
         (lambda f: "The days are getting longer, and the sun's still up.",),
         topic=lambda f: "season.daylength"),
    Line("ssn.name", "season.name", "weather",
         lambda f: f.get("season") and f.get("sky") in ADJ and f.get("temp") is not None,
         (lambda f: f"It's {f['temp']} degrees on a {ADJ[f['sky']]} {f['season_word']} {_noun(f)}.",),
         topic=lambda f: "season.name", owed_weather=True, weather_talk=True,
         with_time=lambda f, t: f"It's {t}, {f['temp']} degrees on a {ADJ[f['sky']]} "
                                f"{f['season_word']} {_noun(f)}."),
    # --- weather colour beside the OWED weather sentence -----------------------------
    Line("wx.rain_later", "wx.change", "tail",
         lambda f: f.get("precip_later") in ("rain", "heavy_rain", "storm"),
         (lambda f: "with rain in the forecast " + f["precip_later_when"],),
         topic=lambda f: "wx.change:rain", owed_weather=True, weather_talk=True),
    Line("wx.snow_later", "wx.change", "tail",
         lambda f: f.get("precip_later") in ("snow", "mix"),
         (lambda f: "with snow in the forecast " + f["precip_later_when"],),
         topic=lambda f: "wx.change:snow", owed_weather=True, weather_talk=True),
    Line("wx.drying", "wx.change", "tail", lambda f: f.get("drying_later"),
         (lambda f: "though it's supposed to dry out over the next couple of hours",),
         topic=lambda f: "wx.change:drying", owed_weather=True, weather_talk=True),
    Line("wx.low_tonight", "wx.trend", "tail",
         lambda f: f.get("overnight_low") is not None and f.get("phase") in ("twilight", "dark")
         and f.get("overnight_low_state") != wc.LOW_NEAR_NOW,
         (lambda f: f"headed for about {f['overnight_low']} overnight",),
         topic=lambda f: "wx.trend:low", owed_weather=True, weather_talk=True),
    # --- weather banter when NO report is owed: condition only -----------------------
    Line("wx.now", "wx.now", "sentence", lambda f: f.get("sky") in _sky_now(f),
         (lambda f: _sky_variants(f)[0](f), lambda f: _sky_variants(f)[1](f),
          lambda f: _sky_variants(f)[2](f)),
         topic=lambda f: "sky:" + str(f.get("sky")), banter=True, weather_talk=True),
    # --- the week: the station's own date --------------------------------------------
    Line("wk.on_a", "week", "time",
         lambda f: f["daypart"] in ("evening", "afternoon", "night")
         or f["weekday"] in ("Saturday", "Sunday"),
         (lambda f: f"on a {f['weekday']} {_noun(f)}",),
         topic=lambda f: "week"),
    Line("wk.friday", "week", "sentence",
         lambda f: f["weekday"] == "Friday" and f["daypart"] in ("evening", "night"),
         (lambda f: "Friday night, finally.", lambda f: "It's Friday night out there."),
         topic=lambda f: "week"),
)


# --- the ledger -------------------------------------------------------------------------
@dataclass
class Ledger:
    entries: list[dict[str, Any]]    # oldest first: at, queue_id, kind, ...

    def last(self, pred) -> dict[str, Any] | None:
        return next((e for e in reversed(self.entries) if pred(e)), None)


def ledger_from(rows: list[dict[str, Any]]) -> Ledger:
    """rows: [{at: datetime, queue_id, telemetry}] for breaks that aired or
    will (dj_engine._atmosphere_rows). Atmosphere records only when the
    render confirmed the line in the spoken text; formal weather/daylight
    from the service ledger, which records what the detector heard."""
    out = []
    for r in rows:
        tel, at, q = r.get("telemetry") or {}, r.get("at"), r.get("queue_id")
        if at is None:
            continue
        a = tel.get("atmosphere") or {}
        if a.get("aired"):
            out.append({"at": at, "queue_id": q, "kind": "atmos", "id": a.get("id"),
                        "family": a.get("family"), "topic": a.get("topic"),
                        "variant": a.get("variant"), "sky": a.get("sky"),
                        "phase": a.get("phase"), "weather_talk": bool(a.get("weather_talk"))})
        aired = tel.get("service_aired") or {}
        if aired.get("weather"):
            parts = str(aired["weather"]).split("|")
            out.append({"at": at, "queue_id": q, "kind": "report",
                        "sky": FAMILY.get(parts[1].lower()) if len(parts) > 1 else None})
        if aired.get("daylight"):
            out.append({"at": at, "queue_id": q, "kind": "daylight",
                        "phase": str(aired["daylight"])})
        mode = (tel.get("writer") or {}).get("mode") if isinstance(tel.get("writer"), dict) else None
        if "story_break" in tel or mode in DJ_MODES:
            # every DJ break, story or planner: the spacing between asides
            out.append({"at": at, "queue_id": q, "kind": "story_break",
                        "had_atmos": bool(a.get("aired"))})
    out.sort(key=lambda e: e["at"])
    return Ledger(out)


#: the writer modes of an ordinary DJ break (not a News intro, liner or announcement)
DJ_MODES = ("smart_planner", "deterministic_fallback", "deterministic_primary",
            "ollama_primary", "story_prepared")


# --- the policy -------------------------------------------------------------------------
def _u(salt: Any, key: str) -> float:
    """A deterministic draw in [0, 1) for this break and line: the jitter that
    keeps weather talk from landing on a timer."""
    h = hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:8]
    return int(h, 16) / 0x100000000


def _minutes(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 60.0


def _relevance(line: Line, f: dict[str, Any], ledger: Ledger, at: datetime) -> float:
    """How worth saying this weather talk is now: a sky nobody has mentioned
    lately or rain on its way is news; the same sky said a moment ago is not."""
    if line.family in ("wx.change",):
        return 1.0
    last_sky = ledger.last(lambda e: e.get("sky") is not None)
    if last_sky is not None and last_sky["sky"] == f.get("sky") and \
            _minutes(at, last_sky["at"]) < SAME_SKY_MIN:
        return 0.15
    if f.get("sky") in PRECIP:
        return 1.0
    if last_sky is not None and last_sky["sky"] != f.get("sky"):
        return 0.9                      # the sky has changed since it was last said
    return 0.5


def _weather_eager(ledger: Ledger, at: datetime) -> float:
    last = ledger.last(lambda e: e["kind"] == "atmos" and e.get("weather_talk"))
    if last is None:
        return 1.5
    return min(2.0, max(0.0, _minutes(at, last["at"])) / WEATHER_TALK_TARGET_MIN)


def _licensed(f: dict[str, Any], *, owed: set[str], offered: dict[str, Any]) -> list[Line]:
    weather_owed = "weather" in owed
    out = []
    for line in BANK:
        if line.owed_weather and not weather_owed:
            continue
        if line.banter and (weather_owed or offered.get("current_weather")):
            continue                  # a formal report is due or on offer: no banter
        if line.records == "daylight" and not offered.get("daylight_line"):
            continue                  # the daylight fact is withheld this break
        if line.kind == "tail" and not ({"weather", "station_time"} & owed):
            continue
        if line.kind == "time" and "station_time" not in owed:
            continue
        if line.kind == "weather" and not weather_owed:
            continue
        try:
            if line.when(f):
                out.append(line)
        except (KeyError, TypeError, IndexError):
            continue
    return out


def choose(f: dict[str, Any], *, owed: set[str], offered: dict[str, Any], ledger: Ledger,
           queue_id: str, salt: Any, at: datetime) -> list[dict[str, Any]]:
    """Rendered candidates for this break, best first ([] most of the time).
    `owed`: the obligation names ("weather", "station_time") this break owes;
    `offered`: its context (what the service ledger let through); `at`: the
    engine's own clock, the ledger's time base."""
    if not f:
        return []
    last_break = ledger.last(lambda e: e["kind"] == "story_break" and e["queue_id"] == queue_id)
    just_had = bool(last_break and last_break.get("had_atmos"))
    eager = _weather_eager(ledger, at)
    picked: list[tuple[float, dict[str, Any]]] = []
    for line in _licensed(f, owed=owed, offered=offered):
        topic = line.topic(f)
        score, news = 0.6, False
        if line.family.startswith("light."):
            said = ledger.last(lambda e: e["kind"] == "daylight" or (
                e["kind"] == "atmos" and str(e.get("family") or "").startswith("light.")))
            if said is not None and said.get("phase") == f.get("phase"):
                continue                  # light is news only when the phase has changed
            arc = ledger.last(lambda e: e["kind"] == "atmos" and e.get("family") in (
                "light.sunset", "light.dusk"))
            if arc is not None:
                gap = _minutes(at, arc["at"])
                if line.family in ("light.sunset", "light.dusk") and gap < ARC_MIN:
                    continue
                if line.family == "light.night" and gap < NIGHT_AFTER_ARC_MIN:
                    continue
            score, news = 1.0, True
        elif line.weather_talk:
            rel = _relevance(line, f, ledger, at)
            score, news = eager * rel, rel >= 0.9
            if score < 0.45 + 0.3 * _u(salt, line.id):
                continue                  # not yet: a preference, not a timer
        else:
            gap = TOPIC_GAP_MIN.get(line.family, 120.0)
            prev = ledger.last(lambda e: e["kind"] == "atmos" and e.get("topic") == topic)
            if prev is not None and _minutes(at, prev["at"]) < gap:
                continue
        if just_had and not news:
            continue                      # a plain break between asides, unless news
        # the exact wording (line, topic, variant) is not repeated within
        # VARIANT_GAP_MIN; when every wording was heard lately, the line
        # waits rather than repeat itself
        recent = {e.get("variant") for e in ledger.entries if e["kind"] == "atmos"
                  and _minutes(at, e["at"]) < VARIANT_GAP_MIN}
        options = []
        for i, variant in enumerate(line.variants):
            try:
                text = variant(f)
            except (KeyError, TypeError, IndexError):
                continue
            if f"{line.id}:{topic}#{i}" not in recent:
                options.append((i, text))
        if not options:
            continue
        vi, text = options[int(_u(salt, line.id + "#v") * len(options))]
        cand = {"id": line.id, "family": line.family, "kind": line.kind, "topic": topic,
                "variant": f"{line.id}:{topic}#{vi}", "text": text, "records": line.records,
                "weather_talk": line.weather_talk, "banter": line.banter,
                "owed_weather": line.owed_weather, "phase": f.get("phase"),
                "sky": f.get("sky") if line.weather_talk else None}
        if line.with_time is not None:
            cand["with_time"] = line.with_time
            cand["facts"] = f
        picked.append((score, cand))
    picked.sort(key=lambda x: -x[0])
    return [c for _, c in picked]


# --- the grounding screen ------------------------------------------------------------------
def screen(text: str, f: dict[str, Any], *, weather_owed: bool) -> list[str]:
    """Words in code-owned atmosphere that the facts do not license."""
    low = str(text or "").lower()
    ph, sky = f.get("phase"), f.get("sky")
    later = f.get("precip_later")
    wet = sky in ("rain", "heavy_rain", "storm") or later in ("rain", "heavy_rain", "storm") \
        or bool(f.get("drying_later"))
    rules = [
        (r"\bsunset\b|\bsun's (?:about to set|on its way down|just)\b|\bsun's just\b",
         ph in ("late_day", "sunset_imminent", "twilight") or f.get("trend") is not None),
        (r"\bsunrise\b|\bcame up\b", ph in ("early_daylight", "sunrise_imminent")),
        (r"\bdaylight\b", ph in ("early_daylight", "full_daylight", "late_day", "sunset_imminent")),
        (r"\bdark\b", ph in _NIGHT),
        (r"\b(?:rain\w*|pouring|coming down|dry out)\b",
         wet or sky in ("mix", "snow") and "coming down" in low),
        (r"\bsnow\w*\b", sky in ("snow", "mix") or later in ("snow", "mix")),
        (r"\bfog\w*\b", sky == "fog"),
        (r"\bcloud\w*\b", sky in ("partly_cloudy", "cloudy", "clear")),
        (r"\bclear\b", sky == "clear"),
        (r"\bwind\w*\b", sky == "wind"),
        (r"\b(?:thunder|lightning|storm\w*)\b", sky == "storm"),
        (r"\bdegrees\b|\bovernight\b", weather_owed),
        (r"\b(?:autumn|fall|winter|spring|summer)\b", f.get("season") is not None),
        (r"\b(?:shorter|longer|earlier every day)\b", f.get("trend") is not None),
        (r"\b(?:leaves|frost|stars|moon|look outside|you can see|sunshine|bright)\b", False),
    ]
    out = []
    for rx, ok in rules:
        m = re.search(rx, low)
        if m and not ok:
            out.append(m.group(0))
    for m in re.finditer(r"\b(" + "|".join(d.lower() for d in WEEKDAYS) + r")\b", low):
        if m.group(1) != f.get("weekday", "").lower():
            out.append(m.group(1))
    for m in re.finditer(r"\b(" + "|".join(x.lower() for x in MONTHS) + r")\b", low):
        if f.get("month") is None or m.group(1) != MONTHS[f["month"] - 1].lower():
            out.append(m.group(1))
    return out


# --- the planner's own words ----------------------------------------------------------------
#: sky and light words a writer may say only when the facts license them --
#: concrete conditions and sun events, never the sonic adjectives a song is
#: described with (dark, bright, warm, cool): those stay the writer guard's
_SPOKEN_RULES = (
    ("rain", r"\b(?:rain(?:y|ing|s)?|showers?|drizzl\w*|pouring)\b"),
    ("snow", r"\bsnow(?:y|ing|s)?\b"),
    ("fog", r"\bfog(?:gy)?\b"),
    ("cloud", r"\b(?:cloudy|overcast|clouds)\b"),
    ("clear", r"\bclear\s+(?:sky|skies|night|evening|morning|afternoon|day)\b"),
    ("sunny", r"\b(?:sunny|sunshine)\b"),
    ("storm", r"\b(?:thunder\w*|lightning|storm(?:y|s)?)\b"),
    ("wind", r"\b(?:windy|breez\w*)\b"),
    ("sunset", r"\b(?:sunset|sundown)\b"),
    ("sunrise", r"\b(?:sunrise|dawn)\b"),
    ("dusk", r"\b(?:dusk|twilight|nightfall)\b"),
    ("daylight", r"\bdaylight\b"),
    ("dark_out", r"\b(?:dark\s+(?:out|outside)|it's\s+dark|getting\s+dark)\b"),
    # "leaves" only as the noun: "that track leaves you wanting more" is a verb
    ("never", r"\b(?:(?:the|autumn|fall|falling|fallen|turning|changing)\s+leaves"
              r"|leaves\s+(?:are|were|have|turning|changing|falling)"
              r"|frost|the\s+moon|stars\s+out|look\s+outside)\b"),
)


def _spoken_ok(word: str, f: dict[str, Any]) -> bool:
    sky, ph, later = f.get("sky"), f.get("phase"), f.get("precip_later")
    return {
        "rain": sky in ("rain", "heavy_rain", "storm", "mix") or later in (
            "rain", "heavy_rain", "storm") or bool(f.get("drying_later")),
        "snow": sky in ("snow", "mix") or later in ("snow", "mix"),
        "fog": sky == "fog",
        "cloud": sky in ("cloudy", "partly_cloudy"),
        "clear": sky == "clear",
        "sunny": sky == "clear" and ph in ("early_daylight", "full_daylight"),
        "storm": sky == "storm",
        "wind": sky == "wind",
        "sunset": ph in ("late_day", "sunset_imminent", "twilight") or f.get("trend") is not None,
        "sunrise": ph in ("early_daylight", "sunrise_imminent"),
        "dusk": ph in ("sunset_imminent", "twilight"),
        "daylight": ph in ("early_daylight", "full_daylight", "late_day", "sunset_imminent"),
        "dark_out": ph in _NIGHT,
        "never": False,
    }[word]


def screen_spoken(text: str, f: dict[str, Any], masks: list[str] = ()) -> list[str]:
    """A writer's sky and light words that the facts do not license, after
    masking every supplied name (song, artist, album, station, event) -- so
    a song title with "Rain" in it is a title, and 'it's raining out there'
    on a clear night is not let through. Owed readings come from the same facts, so the
    words they license are licensed here too."""
    low = " " + _mask(str(text or ""), masks).lower() + " "
    out = []
    for word, rx in _SPOKEN_RULES:
        m = re.search(rx, low)
        if m and not _spoken_ok(word, f):
            out.append(m.group(0).strip())
    return out


def colour_subjects(items: list[dict[str, Any]]) -> list[str]:
    """The service subjects offered colour speaks of (writer_guard scope)."""
    out = set()
    for c in items or ():
        fam = str(c.get("family") or "")
        if fam.startswith("light.") or fam == "season.daylength":
            out.add("daylight")
        if c.get("weather_talk"):
            out.add("weather")
    return sorted(out)


def colour_used(text: str, items: list[dict[str, Any]], f: dict[str, Any],
                masks: list[str] = ()) -> dict[str, Any] | None:
    """Which offered colour item the accepted words actually carry -- by the
    words of its own topic, never by the item's exact phrasing (the writer
    paraphrases) -- as {**item, "said": the sentence that carries it}, or
    None. The sentence is what the ledger records: what was actually said."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", str(text or "")) if x.strip()]
    sky_words = {"clear": r"\bclear\b", "partly_cloudy": r"\bcloud", "cloudy": r"\bcloud|overcast",
                 "rain": r"\brain|showers|drizzl", "heavy_rain": r"\bpour|\brain|coming down",
                 "storm": r"thunder|lightning|storm", "snow": r"\bsnow", "mix": r"\bsnow|\brain",
                 "fog": r"\bfog", "wind": r"\bwind"}
    for c in items or ():
        fam = str(c.get("family") or "")
        if fam == "wx.now":
            rx = sky_words.get(f.get("sky") or "", r"(?!)")
        elif fam.startswith("light."):
            rx = r"sunset|sundown|daylight|sun's|the sun|dark out|it's dark|sunrise|dawn|dusk|twilight"
        elif fam == "season.daylength":
            rx = r"shorter|longer|earlier every|days are"
        elif fam == "week":
            rx = r"\b" + str(f.get("weekday") or "-").lower() + r"\b"
        else:
            continue
        for sentence in sentences:
            if re.search(rx, " " + _mask(sentence, masks).lower() + " "):
                return {**c, "said": sentence}
    return None


def _mask(text: str, masks) -> str:
    """Every supplied name replaced by a marker, as WHOLE words: a title
    "Rain" never hides "raining" or "Rain's", and "One" never eats "gone".
    A quote mark is not part of a word: the writer quotes titles ("here's
    'Lost Bells'"), and a title left unmasked inside its quotes would cost
    the break."""
    for m in sorted((x for x in masks or () if x and len(x.strip()) >= 2), key=len, reverse=True):
        text = re.sub(r"(?<!\w)(?<!\w['\u2019])" + re.escape(m) + r"(?!\w)(?!['\u2019]\w)",
                      " \u00a7 ", text, flags=re.I)
    return text


def record(cand: dict[str, Any] | None) -> dict[str, Any] | None:
    """The telemetry record of a chosen line (no callables)."""
    if not cand:
        return None
    return {k: cand.get(k) for k in ("id", "family", "kind", "topic", "variant", "text",
                                     "records", "weather_talk", "banter", "phase", "sky")}


# --- Smart DJ: the interface (NOT wired; separate qualification) ---------------------------
def planner_colour(f: dict[str, Any], *, owed: set[str], offered: dict[str, Any],
                   ledger: Ledger, queue_id: str, salt: Any, at: datetime,
                   limit: int = 2) -> list[dict[str, Any]]:
    """Colour a Smart DJ planner packet could offer as `moment.colour`: the
    same licensed lines, policy and ledger, limited to sentences a writer
    could weave in -- no weather values unless weather is owed (and then the
    planner's own owed reading carries them), no clock. Each item says what
    the writer may assert and what the ledger records if it airs:

        {"id", "topic", "family", "say", "records", "weather_talk"}

    The writer path stays OFF until it is qualified with a real writing model:
    the accepted text would pass `screen()` (an unlicensed sky, light,
    season or date word refuses the draft and the floor hosts), the existing
    detector/completion/guard are unchanged, and the ledger records the item
    the screen identifies in the accepted text. No extra blocking call: the
    items are computed with the packet, from facts already resolved."""
    out = []
    for c in choose(f, owed=owed, offered=offered, ledger=ledger, queue_id=queue_id,
                    salt=salt, at=at):
        if c["kind"] != "sentence" or c.get("owed_weather"):
            continue
        out.append({"id": c["id"], "topic": c["topic"], "family": c["family"],
                    "say": c["text"], "records": c["records"],
                    "weather_talk": c["weather_talk"]})
        if len(out) >= limit:
            break
    return out
