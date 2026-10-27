"""Speech-only text normalization: the last thing that touches DJ copy
before it reaches a TTS provider.

The transcript and the utterance are different artifacts. What Music Brain
stores, shows in /dj/status and hands to the dashboard is the writer's
text, unchanged; what a synthesizer needs is sometimes spelled
differently. Chatterbox reads "8:37 PM" as the cardinal number eight
thousand thirty-seven (observed repeatedly), so the TTS-bound COPY
says "eight thirty-seven PM" while the stored line keeps its colon.

Deliberately tiny and provider-independent: normalization is an editorial
property of the speech, not of any one synthesizer, so it lives above the
provider seam and applies to every writer/provider combination. Two rules
today: clocks, and the STATION'S OWN FREQUENCY. Chatterbox has no number
handling at all (its punc_norm only tidies punctuation), so a station name
like "101.5 QPRW" reaches the model as digits and can come back garbled --
even when every station mention in the written copy is exactly
right. The frequency is spelled the way radio says it,
but ONLY inside the configured station name (or as that same frequency
standing alone); no other number in a line is touched. The music-metadata pronunciation work (Live -> /laiv/,
context-sensitive read/lead/bass, learned user overrides) will add rules
here rather than replace this module -- but none of that machinery is built
yet, and this file must not grow a framework in anticipation of it.
"""

import re

# 12-hour clock in ordinary prose: 1-12, :00-:59, and an explicit AM/PM
# (with or without periods, either case, spaced or not). The meridiem is
# REQUIRED -- "version 8:37", "8:37 remaining" and every other
# colon-separated pair stay untouched -- and the leading word boundary
# keeps a 24-hour "18:37" from matching its own tail.
_CLOCK_RE = re.compile(
    r"\b(1[0-2]|0?[1-9]):([0-5][0-9])(\s*)([AaPp]\.?[Mm]\.?)(?![A-Za-z])")
# An hour with its meridiem and no minutes: "7 PM", "12 AM", "9 p.m.". Not
# after a digit, a point, a colon, a currency sign or a dash, so "88.5 PM",
# "$5 PM", the minutes of "8:37 PM" and the end of a range ("5-7 PM", left
# whole as before) can never be read as an hour.
_HOUR_CLOCK_RE = re.compile(
    r"(?<![\d.:,$\-\u2013])\b(1[0-2]|0?[1-9])(\s*)([AaPp]\.?[Mm]\.?)(?![A-Za-z])")

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
         7: "seventy", 8: "eighty", 9: "ninety"}


def _number_words(n: int) -> str:
    """Integers 0-99 in words (clock minutes, FM frequencies). A local
    three-liner rather than a spelling dependency."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    word = _TENS[tens]
    return f"{word}-{_ONES[ones]}" if ones else word


def _noon_or_midnight(hour: int, minute: int, meridiem: str) -> str | None:
    """12:00 PM is noon and 12:00 AM midnight: "twelve PM" is the one clock
    a listener has to stop and decode, and radio never says it."""
    if hour == 12 and minute == 0:
        return "noon" if meridiem[:1].lower() == "p" else "midnight"
    return None


def _keep_full_stop(match: re.Match, spoken: str) -> str:
    """A dotted "p.m." that ended its sentence took the full stop with it:
    when the meridiem is dropped (noon, midnight), the stop must stay."""
    meridiem = match.group(match.lastindex)
    rest = match.string[match.end():match.end() + 2]
    if meridiem.endswith(".") and (not rest or re.match(r"\s+[A-Z\"']|\s*$", rest)):
        return spoken + "."
    return spoken


def _speak_clock(match: re.Match) -> str:
    hour, minute, gap, meridiem = (int(match.group(1)), int(match.group(2)),
                                   match.group(3), match.group(4))
    named = _noon_or_midnight(hour, minute, meridiem)
    if named:
        return _keep_full_stop(match, named)
    spoken_hour = _number_words(hour)
    if minute == 0:
        # "eight zero zero AM" is not how anyone says it.
        return f"{spoken_hour}{gap or ' '}{meridiem}"
    if minute < 10:
        spoken_minute = f"oh {_number_words(minute)}"
    else:
        spoken_minute = _number_words(minute)
    return f"{spoken_hour} {spoken_minute}{gap or ' '}{meridiem}"


def _speak_hour_clock(match: re.Match) -> str:
    hour, gap, meridiem = int(match.group(1)), match.group(2), match.group(3)
    named = _noon_or_midnight(hour, 0, meridiem)
    if named:
        return _keep_full_stop(match, named)
    return f"{_number_words(hour)}{gap or ' '}{meridiem}"


def normalize_clock_expressions(text: str) -> str:
    """12-hour clock times as words: 8:37 PM -> eight thirty-seven PM,
    7:05 AM -> seven oh five AM, 8:00 AM -> eight AM, 7 PM -> seven PM,
    12:00 PM -> noon, 12 AM -> midnight, 05:38 PM -> five thirty-eight PM.
    The surrounding prose, including every other number, is returned
    verbatim."""
    said = _CLOCK_RE.sub(_speak_clock, text or "")
    return _HOUR_CLOCK_RE.sub(_speak_hour_clock, said)


# A BARE clock -- "It's 5:38." with no AM/PM. Chatterbox reads it as "five
# thousand thirty-eight" most of the time (the spelled form reliably
# right). Spelled only where it is unmistakably the time: after
# "it's", "at", "by" and their kin, right after the station's own name
# ("... FM, 5:38"), before "on <station>", or opening a sentence and ending
# its clause ("5:38. ...", "5:38, ...") -- so "version 8:37", "8:37
# AMPLIFIER" and every other colon pair stay exactly as written.
_BARE_CLOCK_RE = re.compile(
    r"(?<![\d:.])\b(1[0-2]|[1-9]):([0-5][0-9])\b(?!:\d)(?!\s*[AaPp]\.?[Mm]\.?(?![A-Za-z]))")
_CLOCK_LEAD_RE = re.compile(
    r"(?:\b(?:it'?s|it\s+is|at|by|until|till|after|before|around|about|nearly|"
    r"almost|just\s+after|coming\s+up\s+on)\s+|\b(?:FM|AM|Radio)\W{0,2}\s*)$",
    re.I)
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?][\"\u201d\u2019']?\s+)$")


def _speak_bare_clock(match: re.Match) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if minute == 0:
        return f"{_number_words(hour)} o'clock"
    spoken = (f"oh {_number_words(minute)}" if minute < 10
              else _number_words(minute))
    return f"{_number_words(hour)} {spoken}"


#: How far a bare clock may be from the station's own time and still be
#: given that time's AM/PM: "coming up on 7:15" at 7:12 AM, "just after
#: 8:40" at 8:43 PM. Beyond it the meridiem is unknown and none is said.
BARE_CLOCK_WINDOW_MIN = 60
_STATION_CLOCK_RE = re.compile(
    r"^\s*(1[0-2]|0?[1-9])(?::([0-5][0-9]))?\s*([AaPp])\.?\s*[Mm]\.?\s*$")


def station_minute(station_clock: str | None) -> int | None:
    """The station's own time ("5:38 PM", "7 PM": dj_context._fmt_clock) as a
    minute of the day, or None when it does not parse."""
    m = _STATION_CLOCK_RE.match(str(station_clock or ""))
    if not m:
        return None
    hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
    return hour * 60 + minute + (720 if m.group(3).lower() == "p" else 0)


def _meridiem_near(hour: int, minute: int, station: int) -> str | None:
    """'AM' or 'PM' for the reading of hour:minute within the window of the
    station's time, or None when neither reading is that close."""
    for candidate in ((hour % 12) * 60 + minute, (hour % 12) * 60 + minute + 720):
        gap = abs(candidate - station)
        if min(gap, 1440 - gap) <= BARE_CLOCK_WINDOW_MIN:
            return "AM" if candidate < 720 else "PM"
    return None


def normalize_bare_clocks(text: str, station_clock: str | None = None) -> str:
    """Bare 12-hour clocks in clock position, as words (see above). With the
    station's own time (`station_clock`, authoritative), a bare clock that
    is that time -- within BARE_CLOCK_WINDOW_MIN -- is said with its
    meridiem ("It's 5:38." at 5:38 PM -> "It's five thirty-eight PM."), and
    noon and midnight by name. Nothing is guessed: without the station time,
    or for any other bare clock, no meridiem is added."""
    text = text or ""
    station = station_minute(station_clock)

    def _one(m: re.Match) -> str:
        lead = text[max(0, m.start() - 40):m.start()]
        tail = text[m.end():m.end() + 6]
        if _CLOCK_LEAD_RE.search(lead) or re.match(r"\s+on\b", tail) or (
                _SENTENCE_START_RE.search(lead)
                and re.match(r"\s*(?:[,.;!?]|$)", tail)):
            hour, minute = int(m.group(1)), int(m.group(2))
            meridiem = _meridiem_near(hour, minute, station) if station is not None else None
            if meridiem:
                named = _noon_or_midnight(hour, minute, meridiem)
                if named:
                    return named
                if minute == 0:
                    return f"{_number_words(hour)} {meridiem}"
                return f"{_speak_bare_clock(m)} {meridiem}"
            return _speak_bare_clock(m)
        return m.group(0)
    return _BARE_CLOCK_RE.sub(_one, text)


# Title glyphs said as words (Chatterbox reads the "+" in "Quarry Light +
# Slow Ferry" as a garbled word). Only a glyph standing alone between
# spaces: "AC/DC" and "R&B" are left as written. The stored text keeps the
# glyph; only the speech copy changes.
_GLYPHS = ((re.compile(r"\s\+\s"), " and "), (re.compile(r"\s&\s"), " and "),
           (re.compile(r"\s/\s"), ", "))


def normalize_glyphs(text: str) -> str:
    for rx, said in _GLYPHS:
        text = rx.sub(said, text or "")
    return text


_NUMBER_WORD = "|".join(sorted({*_ONES, *_TENS.values(), "one hundred",
                                 "noon", "midnight"},
                               key=len, reverse=True))
_LOWER_SENTENCE_NUMBER = re.compile(
    r"([.!?][\"\u201d\u2019']?\s+)(" + _NUMBER_WORD + r")\b")


def capitalize_spelled_starts(text: str) -> str:
    """A sentence that now OPENS with a number spelled out after another one
    (". one oh one point five QPRW, ...") gets its capital back.
    Lowercase there makes Kokoro run the station ID into the story with no
    sentence pause (less than half the pause it gets capitalised). The
    very start of a line has nothing to run into and is left as it was."""
    return _LOWER_SENTENCE_NUMBER.sub(
        lambda m: m.group(1) + m.group(2)[0].upper() + m.group(2)[1:], text or "")


# An FM frequency as a station name writes it: "88.5", "101.5", "88,5", the
# point written as a space ("88 5"), or a bare "97" / "106". Two or three
# digits standing alone -- never the tail of "1999", never glued to letters
# ("Z100") -- and only accepted when the value is on the FM dial (below).
_FREQ_RE = re.compile(
    r"(?<![\w.,])(?P<int>\d{2,3})(?:(?P<sep>[.,]|[ \t]+)(?P<dec>\d{1,2}))?"
    r"(?!\w|[.,]\d)")
# After a bare frequency, these say it is a measurement, not the station.
_MEASURE_RE = re.compile(r"\s*(?:%|percent\b|per\s+cent\b|degrees?\b|°)", re.I)
_FM_LOW, _FM_HIGH = 87, 108


def _spoken_fm(whole: str, sep: str | None, dec: str | None) -> str | None:
    """An FM frequency the way radio says it, or None when the digits are
    not one: 88.5 -> eighty-eight point five, 101.5 -> one oh one point five,
    100.3 -> one hundred point three, a bare 106 -> one oh six. A point
    written as a space is said as a short pause: 88 5 -> eighty-eight, five."""
    n = int(whole)
    if not _FM_LOW <= n <= _FM_HIGH:
        return None
    if dec is not None and not 87.5 <= n + int(dec) / 10 ** len(dec) <= 108.0:
        return None
    if n < 100:
        words = _number_words(n)
    elif n == 100:
        words = "one hundred"
    else:
        words = f"one oh {_ONES[n - 100]}"
    if dec is None:
        return words
    digits = " ".join(_ONES[int(d)] for d in dec)
    return f"{words}, {digits}" if sep.strip() == "" else f"{words} point {digits}"


def _fm_matches(name: str) -> list[re.Match]:
    return [m for m in _FREQ_RE.finditer(name)
            if _spoken_fm(m.group("int"), m.group("sep"), m.group("dec"))]


def spoken_station_name(name: str) -> str:
    """The configured station name as it should be SAID: its FM frequency in
    words, every other character exactly as configured. A name with no
    frequency comes back unchanged."""
    return _FREQ_RE.sub(
        lambda m: _spoken_fm(m.group("int"), m.group("sep"), m.group("dec"))
        or m.group(0), name or "")


def normalize_station_name(text: str, station_name: str | None) -> str:
    """The station's own name, and its frequency standing alone, as words.

    The name is matched however a writer spelled its frequency ("88.5",
    "88,5", "88 5") and in any letter case, but only WHOLE -- the frequency
    next to the rest of the configured name -- and is always said the way
    the configured name says it. A bare frequency is rewritten only in its
    configured decimal form ("88.5", never a bare "97") and never before %,
    percent or degrees. Every other number in the line stays as written."""
    name = (station_name or "").strip()
    found = _fm_matches(name)
    if not text or not found:
        return text
    parts, pos = [], 0
    for m in found:
        parts.append(_literal_pattern(name[pos:m.start()]))
        sep = r"(?:\s*[.,]\s*|\s+)" if m.group("dec") is not None else ""
        parts.append(re.escape(m.group("int")) + sep + re.escape(m.group("dec") or ""))
        pos = m.end()
    parts.append(_literal_pattern(name[pos:]))
    whole = re.compile(r"(?<![\w.,])" + "".join(parts) + r"(?!\w)", re.I)
    said = spoken_station_name(name)
    text = whole.sub(lambda _m: said, text)
    for m in found:
        if m.group("dec") is None or m.group("sep").strip() == "":
            continue
        freq = re.compile(r"(?<![\w.,])" + re.escape(m.group("int")) + r"[.,]"
                          + re.escape(m.group("dec")) + r"(?!\w|[.,]\d)")
        words = _spoken_fm(m.group("int"), ".", m.group("dec"))
        text = freq.sub(lambda f: f.group(0) if _MEASURE_RE.match(f.string, f.end())
                        else words, text)
    return text


def _literal_pattern(literal: str) -> str:
    """A run of the configured name, matched with any whitespace between
    its words."""
    words = literal.split()
    if not words:
        return r"\s*" if literal else ""
    lead = r"\s+" if literal[:1].isspace() else ""
    tail = r"\s+" if literal[-1:].isspace() else ""
    return lead + r"\s+".join(re.escape(w) for w in words) + tail


def normalize_for_tts(text: str, station_name: str | None = None,
                      station_clock: str | None = None) -> str:
    """The speech copy of a finished line. Pure, total, and never a reason
    a break fails: an unexpected error returns the original text, so the
    worst case is the pronunciation we already ship today. `station_name`
    is the configured on-air name (settings.resolved_station_name); without
    it only clocks are spelled out. `station_clock` is the station's own
    time as the context supplied it ("5:38 PM"): a bare clock that is that
    time is said with its AM/PM (normalize_bare_clocks)."""
    try:
        said = normalize_glyphs(text)
        said = normalize_clock_expressions(normalize_station_name(said, station_name))
        said = normalize_bare_clocks(said, station_clock)
        return capitalize_spelled_starts(said)
    except Exception:      # noqa: BLE001 - speech polish is never load-bearing
        return text
