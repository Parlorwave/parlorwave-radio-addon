"""Split hosting: one spoken break, assembled in code from three owners.

    music segment   a prepared story or short fact from the DJ memory
                    (dj_memory), never asked to carry the time, the weather or
                    the calendar, and never written by a model at air time
    service         the owed station time / weather, phrased by code from the
                    authoritative values, in approved phrasings and places
    handoff         the authoritative next track, by code

The writer model never sees the clock or the weather on this path, so it can
neither misstate them nor stack them at the front of every break. What was owed
and what this break delivered are reported in the hosting ledger's own
vocabulary (`obligations_owed` / `obligations_met`), and a break that cannot
deliver an obligation it owes is NOT assembled here: anything the story path
cannot carry -- a calendar event, a riding announcement -- returns None and the
existing path keeps its delivery guarantee.

Speech-shaped for the voice engines: one service sentence (two verbless
fragments in a row sound choppy), "It's {temp} and {cond} out there" (a
form that opens on "Out there" tends to merge into the next sentence when spoken),
clock times with their AM/PM so the speech normaliser
spells them, and a spoken-SECONDS budget rather than a word count.

Flow: every service sentence has a verb ("It's 8:40 PM on
the station", never the bare "8:40 PM on the station."); before a story about
the NEXT act the service comes first, so the story runs straight into its
song; a time-only check may ride the handoff ("At 8:40 PM, here's ...").
Casual connectors ("Anyway,", "Alright,", "So,") stay part of the voice, but
one is used only when none was heard in the recent breaks -- variety, not a
ban.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import required_hosting as rh

LEAD_IN = {   # only when the telling does not name the act itself
    "ending": [("lead.end.that_was", "That was {artist}."),
               ("lead.end.there", "{artist} there.")],
    # no pronoun and no verb agreement: "Ada Venn are up next" and
    # "before we hear them" are wrong for every solo act
    "next": [("lead.next.coming", "A quick one about {artist}, coming up "
                                  "next."),
             ("lead.next.first", "Next up, {artist}. First, a quick story.")],
    "earlier": [("lead.earlier.heard", "Earlier we heard {artist}."),
                ("lead.earlier.back", "A little earlier we played {artist}.")],
}
SERVICE_TIME = [("svc.t.its", "It's {time}."),
                ("svc.t.its_on", "It's {time} on {station}."),
                ("svc.t.here_on", "It's {time} here on {station}.")]
#: a time-only check carried by the handoff instead of a sentence of its own
TIME_ON_HANDOFF = ("svc.t.at_handoff", "At {time}, ")
SERVICE_WX = [("svc.w.its_out_there", "It's {temp} and {cond} out there."),
              ("svc.w.its_degrees", "It's {temp} degrees and {cond} outside."),
              ("svc.w.right_now", "Right now it's {temp} and {cond}.")]
SERVICE_BOTH = [("svc.tw.its", "It's {time}, {temp} degrees and {cond}."),
                ("svc.tw.and_its", "{time}, and it's {temp} and {cond}."),
                ("svc.tw.at", "It's {temp} and {cond} at {time}."),
                ("svc.tw.on_station", "It's {time} on {station}, {temp} and "
                                      "{cond} outside.")]
#: casual connectors before a service sentence that follows the story; any
#: connector heard in the recent breaks benches them all (see _connector), so
#: they come every few breaks at most -- "" is weighted twice so an eligible
#: break still often has none (three of six weights on "", so the same
#: connector rarely repeats across a run of service breaks)
CONNECT = [("con.none", ""), ("con.none", ""),
           ("con.anyway", "Anyway, "), ("con.alright", "Alright, "),
           ("con.so", "So, ")]
HANDOFF = [("hand.heres", "Here's {artist} with {title}"),
           ("hand.next_up", "Next up, {artist}, with {title}"),
           ("hand.here_comes", "Here comes {title} from {artist}"),
           ("hand.now", "{artist} now, with {title}")]
#: after a story about the NEXT act, which already named them
HANDOFF_TITLE = [("hand.t.heres", "Here's {title}"),
                 ("hand.t.this_one", "This one's {title}"),
                 ("hand.t.coming_up", "Coming up, {title}")]
ENDING = [("end.that_was", "That was {title} from {artist}."),
          ("end.artist_with", "{artist} there, with {title}."),
          ("end.title_artist", "{title}, from {artist}.")]

#: words per minute by voice engine (typical rates for rendered breaks:
#: Chatterbox ~160, Kokoro ~130)
ENGINE_WPM = {"chatterbox": 160.0, "kokoro": 130.0}
DEFAULT_WPM = 150.0
#: the spoken-time ceiling for a whole story break, and for a short fact
STORY_BREAK_MAX_S = 24.0
NUGGET_BREAK_MAX_S = 18.0

#: what the story path can deliver; any other owed obligation hands the
#: break back to the existing path
CARRIES = {rh.ENDING, rh.NEXT, rh.STATION_TIME, rh.WEATHER}

_EDITION = re.compile(r"\s*\((?:[^()]*(?:remix|mix|live|take|version|demo|"
                      r"edit|remaster\w*|mono|stereo|session)[^()]*)\)\s*$",
                      re.I)


def split_name(name: str | None) -> tuple[str, str]:
    artist, _, title = str(name or "").partition(" - ")
    return artist.strip(), _EDITION.sub("", title).strip()


def weather_parts(prose: str | None) -> tuple[str, str] | None:
    """('59', 'clear') from the current-weather prose, or None."""
    m = re.match(r"\s*(-?\d{1,3}) degrees(?: and)? ([a-z][a-z ]*)",
                 str(prose or ""))
    if not m:
        return None
    return m.group(1), m.group(2).strip()


def spoken_time(station_time: str | None) -> str:
    """The station clock as said on air: '8:40 PM' stays '8:40 PM' -- the
    meridiem is what lets speech.normalize_for_tts spell it ('eight forty
    PM'); a bare '8:40' can reach Chatterbox as a cardinal number."""
    t = re.sub(r"\s+", " ", str(station_time or "")).strip()
    return re.sub(r"\s*([AaPp])\.?\s*[Mm]\.?$",
                  lambda m: " " + m.group(1).upper() + "M", t)


def spoken_seconds(text: str, wpm: float = DEFAULT_WPM) -> float:
    """Expected spoken length: words at the engine's rate, plus a short
    pause per sentence."""
    words = len(re.findall(r"[A-Za-z0-9']+", str(text or "")))
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", str(text or ""))))
    return round(words / (wpm / 60.0) + 0.3 * sentences, 1)


def engine_wpm(engine: str | None) -> float:
    return ENGINE_WPM.get(str(engine or "").lower(), DEFAULT_WPM)


def _pick(options: list[tuple[str, str]], recent: set[str], salt: int
          ) -> tuple[str, str]:
    fresh = [o for o in options if o[0] not in recent] or options
    return fresh[salt % len(fresh)]


def _connector(recent: set[str], salt: int) -> tuple[str, str]:
    """A casual connector, or none. Never two in a row: one heard in the
    recent breaks (recent_template_ids) means none this time."""
    if any(t.startswith("con.") and t != "con.none" for t in recent):
        return CONNECT[0]
    return CONNECT[salt % len(CONNECT)]


def _end(line: str) -> str:
    """A full stop, unless the title already ended the sentence."""
    line = line.rstrip()
    return line if line[-1:] in ".!?" else line + "."


def _names_act(text: str, artist: str) -> bool:
    low = str(text or "").lower().replace("’", "'")
    a = artist.lower()
    return bool(a) and (a in low or a.removeprefix("the ") in low)


def _opening(text: str, n: int = 2) -> str:
    return " ".join(re.findall(r"[a-z']+", str(text or "").lower()
                               .replace("’", "'"))[:n])


def assemble(context: dict[str, Any], material: dict[str, Any] | None, *,
             station_name: str, with_payoff: bool = True,
             recent_template_ids: Iterable[str] = (), salt: int = 0,
             recent_framing: Iterable[str] = (),
             wpm: float = DEFAULT_WPM,
             aside: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The break, or None when this path must not host it (no next track to
    hand to, or an obligation it cannot carry is owed).

    `aside`: one atmosphere candidate (atmosphere.choose), placed by its
    kind -- a tail on the service sentence, a weekday opener on the time
    check, the owed weather sentence itself, or a sentence of its own after
    the service. `out["atmosphere"]` records exactly what was placed, or is
    None when this break had no place for it."""
    owed = set(rh.obligations(context))
    if owed - CARRIES:
        return None                    # calendar / announcement: their own path
    nxt_artist, nxt_title = split_name(context.get("post_name"))
    if not (nxt_artist or nxt_title):
        return None
    recent = set(recent_template_ids or ())
    framed = {_opening(t) for t in recent_framing or () if t}
    parts: list[dict[str, Any]] = []
    used: list[str] = []
    met: list[str] = []

    if material:
        about = material.get("about") or "ending"
        artist = material.get("artist") or ""
        told = " ".join(p for p in (material.get("setup"), material.get(
            "telling")) if p)
        # an act from earlier in the hour ALWAYS gets its lead-in
        if about == "earlier" or not _names_act(told, artist):
            tid, tpl = _pick(LEAD_IN.get(about, LEAD_IN["ending"]), recent,
                             salt)
            parts.append({"role": "lead_in", "text": tpl.format(artist=artist),
                          "source": "code", "template_id": tid})
            used.append(tid)
        if material.get("setup") and _opening(material["setup"]) not in framed:
            parts.append({"role": "setup", "text": material["setup"],
                          "source": "prepared_framing",
                          "story_id": material.get("story_id")})
        parts.append({"role": "telling", "text": material["telling"],
                      "source": "prepared_story",
                      "story_id": material.get("story_id")})
        if with_payoff and material.get("payoff") and \
                _opening(material["payoff"]) not in framed:
            parts.append({"role": "payoff", "text": material["payoff"],
                          "source": "prepared_framing",
                          "story_id": material.get("story_id")})
        if about == "ending" and rh.ENDING in owed:
            met.append(rh.ENDING)

    time_due = rh.STATION_TIME in owed
    wx = weather_parts(context.get("current_weather")) if (
        rh.WEATHER in owed) else None
    t = spoken_time(context.get("station_time")) if time_due else ""
    # before a story about the NEXT act the service comes first, so the
    # story runs straight into its song
    service_first = bool(material) and material.get("about") == "next"
    story_named_next = bool(material) and material.get(
        "about") == "next" and bool(nxt_title)
    hid, htpl = _pick(HANDOFF_TITLE if story_named_next else HANDOFF,
                      recent, salt)
    handoff = _end(htpl.format(artist=nxt_artist or nxt_title,
                               title=nxt_title or nxt_artist))
    kind = (aside or {}).get("kind")
    placed: dict[str, Any] | None = None
    if t and not wx and handoff.startswith(("Here's", "Here comes")) and \
            kind not in ("tail", "time", "weather") and \
            TIME_ON_HANDOFF[0] not in recent and (salt // 4) % 3 == 1:
        # (salt // 4: independent of the handoff's own salt % len(HANDOFF))
        # "At 8:40 PM, here's Slow Ferry."
        handoff = TIME_ON_HANDOFF[1].format(time=t) + handoff[0].lower() + \
            handoff[1:]
        used.append(TIME_ON_HANDOFF[0])
        met.append(rh.STATION_TIME)
        t = ""
    if t or wx:
        if t and wx:
            tid, tpl = _pick(SERVICE_BOTH, recent, salt)
        elif t:
            tid, tpl = _pick(SERVICE_TIME, recent, salt)
        else:
            tid, tpl = _pick(SERVICE_WX, recent, salt)
        line = tpl.format(time=t, station=station_name,
                          temp=wx[0] if wx else "", cond=wx[1] if wx else "")
        if kind == "weather" and wx:
            line = (aside["with_time"](aside["facts"], t) if t
                    else aside["text"])
            tid, placed = aside["id"], {**aside, "text": line}
        elif kind == "time" and t:
            line = f"It's {t} {aside['text']}" + (
                f", {wx[0]} degrees and {wx[1]}" if wx else "") + "."
            tid, placed = "svc.t.week", {**aside, "text": aside["text"]}
        elif kind == "tail":
            line = line[:-1] + ", " + aside["text"] + line[-1]
            placed = dict(aside)
        if parts and not service_first:
            cid, glue = _connector(recent, salt + 1)
            if glue:
                keep_case = line[0].isdigit() or line.startswith(station_name)
                line = glue + (line if keep_case else
                               line[0].lower() + line[1:])
                used.append(cid)
        service = {"role": "service", "text": line, "source": "code",
                   "template_id": tid}
        if service_first:
            parts.insert(0, service)
        else:
            parts.append(service)
        used.append(tid)
        if t:
            met.append(rh.STATION_TIME)
        if wx:
            met.append(rh.WEATHER)
        if kind == "sentence":
            parts.insert(parts.index(service) + 1, {
                "role": "atmosphere", "text": aside["text"], "source": "code",
                "template_id": aside["id"]})
            placed = dict(aside)
    elif kind == "sentence" and material:
        parts.append({"role": "atmosphere", "text": aside["text"],
                      "source": "code", "template_id": aside["id"]})
        placed = dict(aside)
    if placed is not None and aside["id"] not in used:
        used.append(aside["id"])

    parts.append({"role": "handoff", "text": handoff, "source": "code",
                  "template_id": hid})
    used.append(hid)
    met.append(rh.NEXT)
    if not material and rh.ENDING in owed:
        end_artist, end_title = split_name(context.get("pre_name"))
        if end_artist:
            eid, etpl = _pick(ENDING, recent, salt)
            line = (_end(etpl.format(artist=end_artist, title=end_title))
                    if end_title else f"That was {end_artist}.")
            parts.insert(0, {"role": "ending", "text": line,
                             "source": "code", "template_id": eid})
            used.insert(0, eid)
            met.append(rh.ENDING)
    for p in parts:
        # markdown emphasis from the prepared text never reaches the voice
        p["text"] = re.sub(r"(?<!\w)[*_]+|[*_]+(?!\w)", "", p["text"])
        p["text"] = re.sub(r"\s+", " ", p["text"]).strip()
    text = " ".join(p["text"] for p in parts if p["text"]).strip()
    # owed obligations the story path did not deliver (ENDING when the story
    # was about another act) are reported, never claimed
    atmosphere = None
    if placed is not None:
        import atmosphere as _atm
        atmosphere = _atm.record(placed)
    return {"text": text, "parts": parts, "template_ids": used,
            "atmosphere": atmosphere,
            "obligations_met": sorted(set(met)),
            "obligations_owed": sorted(owed),
            "obligations_unmet": sorted(owed - set(met)),
            "story_id": (material or {}).get("story_id"),
            "words": len(text.split()),
            "spoken_s": spoken_seconds(text, wpm)}


def host_with_story(context: dict[str, Any], memory: Any, *,
                    station_name: str, break_token: str,
                    recent_artists: Iterable[tuple[str, float]] = (),
                    breaks_since_story: int = 99, last_break_act: str = "",
                    now: float | None = None, with_payoff: bool = True,
                    recent_template_ids: Iterable[str] = (), salt: int = 0,
                    recent_framing: Iterable[str] = (), wpm: float = DEFAULT_WPM,
                    max_words: int | None = None,
                    asides: Iterable[dict[str, Any]] = (),
                    atmosphere_facts: dict[str, Any] | None = None
                    ) -> dict[str, Any] | None:
    """The one call the DJ engine makes: choose material from the memory
    within the spoken-time budget, assemble the break, and RESERVE the story
    for this break token (a selection, not an airing). None when there is no
    eligible material that fits, or when this path must not host the break --
    the existing path then hosts it exactly as today."""
    end_artist, _ = split_name(context.get("pre_name"))
    nxt_artist, _ = split_name(context.get("post_name"))
    for material in memory.candidates(
            ending_artist=end_artist, next_artist=nxt_artist,
            recent_artists=recent_artists,
            breaks_since_story=breaks_since_story,
            last_break_act=last_break_act, now=now):
        ceiling = (STORY_BREAK_MAX_S if material["kind"] == "story"
                   else NUGGET_BREAK_MAX_S)
        chosen = None
        # an aside never costs the story: each candidate is tried, then none
        for aside in list(asides or ()) + [None]:
            out = assemble(context, material, station_name=station_name,
                           with_payoff=with_payoff,
                           recent_template_ids=recent_template_ids, salt=salt,
                           recent_framing=recent_framing, wpm=wpm, aside=aside)
            if out is None:
                return None                   # the path must not host it
            if aside is not None and not out.get("atmosphere"):
                continue                      # no place for it in this break
            if out["spoken_s"] > ceiling:
                continue
            if max_words and out["words"] > max_words:
                continue
            if aside is not None and not _aside_safe(out, context, aside,
                                                     atmosphere_facts):
                continue
            chosen = out
            break
        if chosen is None:
            continue
        if not memory.reserve(material, break_token, now=now):
            continue
        chosen["material"] = material
        chosen["selection"] = material.get("why")
        return chosen
    return None


def _aside_safe(out: dict[str, Any], context: dict[str, Any],
                aside: dict[str, Any], facts: dict[str, Any] | None) -> bool:
    """The aside is licensed by the facts (atmosphere.screen), leaves the
    supplied clock said exactly once, and is never heard as weather when no
    report is owed (dj_context's own detectors)."""
    try:
        import atmosphere
        import dj_context
        owed = set(out["obligations_owed"])
        placed = (out.get("atmosphere") or {}).get("text") or aside.get("text")
        if atmosphere.screen(placed, facts or {}, weather_owed=rh.WEATHER in owed):
            return False
        if rh.STATION_TIME in owed and dj_context.spoken_clock(
                out["text"], context) is None:
            return False
        if rh.WEATHER not in owed and context.get("current_weather") and \
                "weather" in dj_context.spoken_service_concepts(
                    placed, {"current_weather": context["current_weather"]}):
            return False
        return True
    except Exception:  # noqa: BLE001 - when unsure, no aside
        return False
