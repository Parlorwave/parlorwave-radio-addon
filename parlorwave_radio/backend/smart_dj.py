"""Smart DJ context planner: a bounded, grounded editorial decision at an
already-eligible break opportunity.

    deterministic context builder  (build_packet: the SMALL packet)
            |
    one bounded model call         (render_prompt -> call_model)
            |
    validated structured result    (parse_result + validate -> Decision)
            |
    speak  OR  say_nothing         (dj_engine decides transport; this module
                                    never touches the queue, the store, TTS
                                    or the lifecycle)

Airtime is earned. An eligible boundary does not oblige the DJ to talk:
SAY_NOTHING is a successful result, and it is also where every failure
lands -- a timeout, an unparseable answer, an unknown kind, a fact
reference the packet never supplied, a weather remark without the supplied
weather, a tease without a known next song, an authority-guard violation.
Nothing malformed can become aired text, because the only way out of this
module with words is `validate` saying yes.

The packet is the model's entire world. It is bounded by construction --
previous/ending/next only, at most five fact candidates, at most four
recent utterances, every human-readable field truncated -- and `fit`
enforces a hard serialized byte cap regardless of queue length, library
size, station runtime or break history. Nothing here reads the queue, the
database or the network: dj_engine hands over what the scheduler already
resolved for this very boundary.

Hosting contract: grounding decides whether a break may be
said; these decide whether it is worth saying and whether it finishes.
A break must carry a PAYLOAD the packet authorized -- a realized fact, a
service reading, the listener's event, a named handoff, or some concrete
detail -- because generic evaluation ("a certain intensity", "a sturdy
listen") may decorate a payload but may never be one. And a forward
reference must LAND: copy that points at what is coming has to name it
when the packet holds a next song. Neither rule is a template, neither
counts sentences, and both fail toward letting copy through -- the
expensive direction is silencing a good break, not airing a thin one.

The relationship between the two songs is the one thing the writer may
never invent, and the prompt must say so -- the delivery contract
otherwise invites exactly that claim, and then the omission, not the
guard, is what refuses most decisions.

Grounding contract: the model may be creative about tone, transitions,
wordplay and remarks on the SUPPLIED moment. Factual claims come only
from the packet -- the deterministic live context (clock, weather value,
daylight, the actual songs) or an explicit verified fact carrying an id
the result must cite. The model's own knowledge of these artists and songs
is never a source, and the prompt says so; `validate` and the existing
writer authority guard hold the line when the model does not listen.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import socket
from datetime import datetime
from typing import Any, Callable

import dj_persona

logger = logging.getLogger("musicbrain")

# --- decision kinds -------------------------------------------------------------
SAY_NOTHING = "say_nothing"
TRANSITION = "transition"
QUIP = "quip"
OBSERVATION = "observation"
FACT = "fact"
WEATHER = "weather"
DAYLIGHT = "daylight"
TEASE_NEXT = "tease_next"
CALENDAR = "calendar"
KINDS = (SAY_NOTHING, TRANSITION, QUIP, OBSERVATION, FACT, WEATHER,
         DAYLIGHT, TEASE_NEXT, CALENDAR)
SPEAKING_KINDS = tuple(k for k in KINDS if k != SAY_NOTHING)

# --- hard bounds (the packet stays small whatever grows around it) --------------
MAX_MUSIC_ITEMS = 3          # previous / ending / next -- never a queue
MAX_FACTS = 6                # verified fact candidates offered
#: Of those, how many may come from the grounded library-knowledge
#: channel the scheduler fills in (`knowledge_facts` on the context).
#: The rest of the slot stays with the catalog album facts, which are
#: where a break's album identity comes from -- a producer fact is
#: anchored on an album title the listener has to be able to hear.
#:
#: This module knows nothing about where that channel comes from and
#: imports nothing to read it: it is handed pre-matched, pre-scoped,
#: pre-ranked propositions and treats them exactly like any other
#: supplied fact.
MAX_KNOWLEDGE_FACTS = 4
#: Withheld pack facts carried for the VALIDATOR only (`_hidden`): the
#: writer never sees them; a correct mention it makes on its own is
#: grounded by them instead of dying as an ungrounded claim.
MAX_REFERENCE_FACTS = 4
MAX_RECENT = 4               # recent DJ utterances / topic keys
NAME_MAX = 64                # artist / title / album characters
ANNOUNCEMENT_MAX = 500       # announce_copy.MAX_PAYLOAD_CHARS
FACT_TEXT_MAX = 100
RECENT_TEXT_MAX = 100
LINE_MAX = 120               # weather / daylight lines
ANGLE_MAX = 110              # the editorial premise, one clause. Every
                             # built-in angle fits well inside it: a
                             # TRUNCATED instruction is a worse
                             # instruction, and the test pins that.
AVOID_MAX = 4                # structural habits named to the writer at once
#: How many of the recent breaks must share a structural habit before it is
#: named as something to break out of. Two is the point at which a shape
#: stops being a coincidence and starts being this station's voice.
STYLE_REPEAT_MIN = 2
PACKET_MAX_BYTES = 8192      # serialized packet, worst case, hard cap
RECENT_MAX_AGE_H = 6.0       # older utterances no longer count as "recent"

# The planner request's context contract. The whole prompt (rules, persona
# block, packet) is bounded by construction -- see the size tests -- but a
# runtime's DEFAULT context is not ours to assume: older Ollama builds and
# some Modelfiles run at 2048 tokens and silently truncate the FRONT of an
# oversized prompt, which is exactly where the grounding rules live. So the
# request states its needs explicitly: a 6144-token window (comfortably
# twice the worst-case prompt plus the answer) and a completion budget sized
# from the word budget, so a runaway answer is cut off -- and then fails
# parsing into say_nothing -- instead of running on.
#
# 6144, not the original 4096: the delivery/flair contract and the
# calendar block are what turned a terse announcer into a host, and they
# cost roughly 2.3 KB of prompt. At 4096 the worst-case prompt still fit,
# but the stated margin -- twice the prompt plus the answer -- no longer
# held, and that margin is the whole defence against a runtime that
# truncates the FRONT, where the grounding rules live.
PLANNER_NUM_CTX = 6144
PLANNER_NUM_PREDICT_BASE = 64        # JSON scaffolding and punctuation
PLANNER_NUM_PREDICT_PER_WORD = 2     # generous: ~1.3 tokens per word

# --- why a decision became SAY_NOTHING (bounded vocabulary, never prose) --------
REASON_MODEL = "model_said_nothing"        # the model chose silence: success
REASON_NO_MATERIAL = "no_material"         # no song facts at all: no call made
FAIL_TIMEOUT = "timeout"
FAIL_PROVIDER = "provider_error"
FAIL_INVALID = "invalid_result"
FAIL_UNKNOWN_KIND = "unknown_kind"
FAIL_NOT_CANDIDATE = "kind_not_eligible"
FAIL_EMPTY = "empty_text"
FAIL_FACT_REF = "fact_reference_mismatch"
FAIL_UNGROUNDED = "ungrounded_text"
FAIL_GUARD = "guard_rejected"
FAIL_SELF_REF = "self_reference"
FAIL_CLAIM = "ungrounded_claim"       # a fact-shaped assertion in ANY kind
FAIL_NO_ANCHOR = "no_hosting_anchor"  # banter with nothing to host
#: HOSTING VALUE. The anchor rule asks whether the listener was told what
#: the break is ABOUT; these two ask whether they were told anything worth
#: the interruption, and whether the break handed them back to the station.
FAIL_NO_PAYLOAD = "no_hosting_payload"      # evaluation WAS the whole break
FAIL_DANGLING = "dangling_forward_reference"  # a tease pointing at nothing
#: A break whose only substance was a catalogue date or label, dressed in
#: evaluation. Objective: decided by the fact's own PREDICATE, never by an
#: opinion about whether the fact was interesting.
FAIL_THIN_FACT = "thin_fact_payload"
#: A cited fact that reached air without actually being realized. Three
#: subtypes because "fact_reference_mismatch" told us a fact went wrong
#: and never which way, and the three have different fixes: the subject
#: was never named, the value was never said, or the RELATIONSHIP was
#: broadened into something the pack does not assert.
FAIL_FACT_SUBJECT = "fact_subject_not_said"
FAIL_FACT_VALUE = "fact_value_not_said"
FAIL_FACT_RELATION = "fact_relationship_drift"
#: The copy read our own bookkeeping out loud, e.g. a line ending
#: "...fact ID kb:0123abcd4567". A listener cannot be handed a database
#: key, and a line that does it is not copy with a blemish -- it is copy
#: written by something that thought the ids were part of the sentence,
#: so the whole line is refused rather than tidied up.
FAIL_METADATA = "internal_metadata_leak"
#: The copy states a clock the packet never supplied. The clock is OWED
#: only when due and then supplied exactly; when it is not due the packet
#: carries no time at all, and a model can fill the gap by copying the
#: clock from its own previous break -- minutes stale, passing every other
#: rule (the pre-air stale-clock check reads only SUPPLIED clocks).
#: A time check is a station fact like any
#: other: one the station did not give is refused.
FAIL_UNSUPPLIED_CLOCK = "unsupplied_clock"

#: Spoken markup: WARN ONLY.
#: Markdown emphasis in the raw answer is a PRESENTATION fault, not a
#: factual one, and the spoken cleaner already removes it.
#:
#: A hard refusal on the theory that italics are the cheapest detector of
#: invented ALBUMS does not hold up: markup usually wraps an album name
#: the packet actually SUPPLIED (*Northern Static*, *Gullwing*). It
#: catches nothing invented, while otherwise grounded copy is refused for
#: a stray asterisk.
#:
#: It would also mask better complaints: refusing here, ahead of the claim
#: scanner, relabels `ungrounded_claim` rejections as "spoken_markup"
#: and loses the fabricated claim from the record.
#:
#: So the detector stays, on the RAW answer -- `_clean_spoken` strips
#: [*_#`>[]] before `validate` sees the text, so this is the only place
#: the evidence exists -- and it reports. An INVENTED album is refused by
#: the factual guards that own that question, markup or no.
WARN_MARKUP = "i1_spoken_markup"
FAILURES = (FAIL_METADATA, FAIL_FACT_SUBJECT, FAIL_FACT_VALUE,
            FAIL_FACT_RELATION,
            FAIL_TIMEOUT, FAIL_PROVIDER, FAIL_INVALID, FAIL_UNKNOWN_KIND,
            FAIL_NOT_CANDIDATE, FAIL_EMPTY, FAIL_FACT_REF, FAIL_UNGROUNDED,
            FAIL_GUARD, FAIL_SELF_REF, FAIL_CLAIM, FAIL_NO_ANCHOR,
            FAIL_NO_PAYLOAD, FAIL_DANGLING, FAIL_THIN_FACT,
            FAIL_UNSUPPLIED_CLOCK)

# Daylight phases worth a remark on their own (near a sun event). Ordinary
# full daylight or settled dark is only offered when it just CHANGED.
SALIENT_DAYLIGHT = ("late_day", "sunset_imminent", "twilight",
                    "sunrise_imminent", "early_daylight")

#: The ONLY high/low relationship the writer may state, per resolved
#: high_state (weather_context.resolve_high). Each says what is true and
#: closes the inference the token used to leave open.
_HIGH_NOTE = {
    "near_now": ("the current temperature IS today's high -- it is "
                 "neither cooler nor warmer than the high, and no trend "
                 "is supplied"),
    "passed": ("today's high is over; the reading now is simply the "
               "current temperature, and no comparison between the two "
               "is supplied"),
    "upcoming": ("today's high is still ahead and has not happened yet; "
                 "no comparison with the current temperature is supplied"),
}

# The structured answer the model must give. Ollama >= 0.5 constrains the
# decoder to this schema (the same mechanism the station programmer and the
# song-context extractor already rely on); parse_result still validates the
# bytes as if nothing had been constrained.
RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(KINDS)},
        "text": {"type": ["string", "null"]},
        "fact_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "text", "fact_ids"],
}

# Words that mean the model is describing its own workings instead of
# talking to the room. Deliberately narrow; dj_persona's machinery rule
# stays the prompt-side instruction.
_SELF_REF_RE = re.compile(
    r"\b(according to (?:my|the) (?:context|packet|data|facts|notes|"
    r"instructions)|my context|the packet|the planner|as an ai|"
    r"language model|the prompt|the context|the data|fact_ids?|"
    r"say_nothing|tease_next)\b",
    re.IGNORECASE)

#: Talking ABOUT the station's own inputs rather than to the room: a line
#: that praises "the weather line" -- a listener has no idea what a weather
#: line is, because it is a field in a packet they cannot see.
#:
#: Deliberately a short, explicit list of the common shapes
#: plus their obvious neighbours. It is NOT a style censor: "the weather"
#: on its own is ordinary English and stays legal, and only a phrase that
#: names one of our INPUTS as an input is refused.
_META_RE = re.compile(
    r"\b(?:that |the |this )?(?:weather|daylight|calendar|context|packet|"
    r"service)\s+(?:line|lines|data|field|fields|entry|entries|feed|"
    r"listing|readout)\b"
    r"|\bthe information (?:you|i was) (?:gave|given|provided)\b"
    r"|\baccording to (?:the |my )?(?:packet|context|data|facts|notes|"
    r"listing|information)\b"
    r"|\b(?:supplied|provided|listed|given) (?:fact|facts|context|data)\b",
    re.IGNORECASE)

#: Two-digit years -- "'97", "’08" -- expanded ONLY for validation. The
#: claim scanner's year shape needs four digits, so "it came out in '97"
#: would sail past it while "1997" would have been caught.
#: 00-29 reads as 20xx, 30-99 as 19xx, which covers every year the pack
#: can hold. The SPOKEN COPY IS NEVER REWRITTEN: this builds a separate
#: string for the scanners to look at.
#: The apostrophe must OPEN the year: preceded by nothing, a space or an
#: opening bracket/quote, and followed directly by the digits. Allowing a
#: space after it would make the CLOSING quote of a title read as a year:
#: "...Slow Ferry.' 61 degrees" would become 1961 and a clean
#: weather-and-handoff draft would die as `ungrounded_claim`. "in '97" is
#: unaffected.
_APOS_YEAR_RE = re.compile(r"(?<![^\s(\[\"\u201c])[\u2018\u2019'](\d\d)\b")

#: Narrowest structural form: emphasis and code markup and
#: markdown links. `_` only where it is actually emphasising a word, so
#: ordinary prose is untouched. This is NOT a punctuation sanitizer --
#: dashes, quotes, ellipses and ampersands are ordinary spoken copy.
_MARKUP_RE = re.compile(r"\*|_[A-Za-z]|`|^#{1,6}\s|\[[^\]]*\]\([^)]*\)",
                        re.M)


def spoken_markup(text: str) -> str | None:
    """The markdown emphasis this spoken line carries, or None."""
    hits = sorted({m.group(0) for m in _MARKUP_RE.finditer(text or "")})
    return ",".join(hits) if hits else None

#: OUR OWN BOOKKEEPING, spoken. Every branch is a token shape that only
#: exists inside this service:
#:
#:   kb:9f2c...        a knowledge fact id as the packet writes it
#:   album:1a2b3c4d    the catalog fact id
#:   "fact ID"/"fact_id"   the label itself, read out
#:   a bare 12+ hex run    an id echoed without its prefix
#:   snake_case            album_producer, topic_key, context_offered --
#:                         the underscore IS the tell. "album producer"
#:                         and "the record label" are ordinary English
#:                         and stay legal; `album_producer` is a field
#:                         name and never belongs in a sentence.
#:
#: Narrow on purpose. It looks for machine tokens, not for topics, so no
#: word a host might reasonably say is on this list.
_METADATA_RE = re.compile(
    r"\bkb:\S+"
    r"|\balbum:[0-9a-f]{4,}\b"
    r"|\bfact[ _\-]?ids?\b"
    r"|\b[0-9a-f]{12,}\b"
    r"|\b[a-z]+_[a-z][a-z_]*\b",
    re.IGNORECASE)


def internal_metadata(text: str, packet: dict[str, Any]) -> str | None:
    """The internal token this copy would read out, or None.

    Supplied names are masked first, exactly as the claim scanner does,
    so a track genuinely titled with an underscore is the artist's
    problem and not ours -- only tokens the SERVICE introduced can trip
    this."""
    body = text or ""
    if not body:
        return None
    for mask in _supplied_strings(packet):
        body = re.sub(re.escape(mask), " \u00a7 ", body, flags=re.IGNORECASE)
    m = _METADATA_RE.search(body)
    return m.group(0)[:40] if m else None


def expand_apostrophe_years(text: str) -> str:
    """'97 -> 1997, '08 -> 2008, for VALIDATION ONLY."""
    def _fix(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        return f"{2000 + n}" if n <= 29 else f"{1900 + n}"
    return _APOS_YEAR_RE.sub(_fix, text or "")


# --- spoken clocks, as VALUES -----------------------------------------------
#: "7:15 AM", "7:15 a.m.", "9 PM", and a bare "7:15" with no meridiem. A
#: clock is compared by its VALUE, never its spelling, so "7:15 a.m." is the
#: supplied "7:15 AM" and "coming up on 9 PM" is the supplied form of it.
_CLOCK_TOKEN_RE = re.compile(
    r"\b(\d{1,2})(?::([0-5]\d))?\s*([ap])\.?\s?m\b\.?"
    r"|\b(\d{1,2}):([0-5]\d)\b", re.IGNORECASE)


_NUM_WORDS = {w: i for i, w in enumerate(
    ("zero one two three four five six seven eight nine ten eleven twelve "
     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split())}
_NUM_WORDS.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50})
#: A clock in words: "eight forty PM", "nine oh five a.m.", "ten
#: o'clock", "seven fifteen". An hour word alone is never a time ("six
#: songs", "one more"): it needs minutes, a meridiem or o'clock. The minute
#: words must end at a word boundary: "Two tenors sang..." is not 2:10 (the
#: minute "ten" would match the start of "tenors").
_SPELLED_CLOCK_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?:[\s-]+(?:(o'?\s?clock)|(?:oh|o)[\s-]+(one|two|three|four|five|"
    r"six|seven|eight|nine)|(ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen)|(twenty|thirty|forty|fifty)"
    r"(?:[\s-]+(one|two|three|four|five|six|seven|eight|nine))?)\b)?"
    r"\s*(?:([ap])\.?\s?m\b\.?)?", re.IGNORECASE)


def _spelled_clock_tokens(text: str) -> list[tuple[int, int, str | None]]:
    out: list[tuple[int, int, str | None]] = []
    for m in _SPELLED_CLOCK_RE.finditer(text or ""):
        oclock, mer = m.group(2), m.group(7)
        said_minutes = any(m.group(g) for g in (3, 4, 5))
        if not (oclock or mer or said_minutes):
            continue                  # "six songs" -- not a time at all
        minute = 0
        if m.group(3):
            minute = _NUM_WORDS[m.group(3).lower()]
        elif m.group(4):
            minute = _NUM_WORDS[m.group(4).lower()]
        elif m.group(5):
            minute = _NUM_WORDS[m.group(5).lower()] + (
                _NUM_WORDS[m.group(6).lower()] if m.group(6) else 0)
        out.append((_NUM_WORDS[m.group(1).lower()] % 12, minute,
                    mer.lower() if mer else None))
    return out


def clock_tokens(text: str) -> list[tuple[int, int, str | None]]:
    """Every clock-shaped reading in `text` as (hour 0-11, minute,
    meridiem 'a'/'p' or None) -- in digits or in words. Out-of-range
    readings are not clocks."""
    out: list[tuple[int, int, str | None]] = _spelled_clock_tokens(text)
    for m in _CLOCK_TOKEN_RE.finditer(text or ""):
        if m.group(1) is not None:
            hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), \
                m.group(3).lower()
            if not 1 <= hour <= 12:
                continue
        else:
            hour, minute, mer = int(m.group(4)), int(m.group(5)), None
            if not 0 <= hour <= 23:
                continue
            if hour > 12:
                hour, mer = hour - 12, "p"
        out.append((hour % 12, minute, mer))
    return out


def _supplied_clock_values(packet: dict[str, Any]
                           ) -> set[tuple[int, int, str | None]]:
    """Every clock value the packet itself states: the time and its
    equivalent forms, and any clock inside a supplied line (the forecast's
    "overnight low ... around 7:00 AM", a calendar start, a fact)."""
    moment = packet.get("moment") or {}
    texts = [moment.get("time"), moment.get("weather"),
             moment.get("daylight_line")]
    texts += list(moment.get("time_forms") or [])
    cal = packet.get("calendar") or {}
    texts += [cal.get(k) for k in ("title", "start_time", "lead")]
    texts += [f.get("text") for f in (packet.get("facts") or [])]
    texts.append((packet.get("announcement") or {}).get("message"))
    out: set[tuple[int, int, str | None]] = set()
    for t in texts:
        if t:
            out.update(clock_tokens(str(t)))
    return out


def unsupplied_clock(text: str, packet: dict[str, Any]) -> str | None:
    """The first clock the copy states that the packet did not supply, as
    'H:MM' for the record, else None. Song, artist and album names are
    masked first: a title like "2:45 AM" is a name, not a time check. A
    bare 'H:MM' matches a supplied clock with the same hour and minute."""
    low = text or ""
    for m in packet.get("music") or []:
        for k in ("artist", "title", "album"):
            if m.get(k):
                low = re.sub(re.escape(str(m[k])), " ", low,
                             flags=re.IGNORECASE)
    supplied = _supplied_clock_values(packet)
    bare = {(h, mi) for h, mi, _ in supplied}
    for hour, minute, mer in clock_tokens(low):
        if mer is None:
            if (hour, minute) in bare:
                continue
        elif (hour, minute, mer) in supplied:
            continue
        return f"{hour or 12}:{minute:02d}" + (f" {mer}m" if mer else "")
    return None


#: A temperature reading or a high/low figure, for recent-break scrubbing.
_RECENT_TEMP_RE = re.compile(
    r"\b\d{1,3}\s*(?:°|degrees?\b)|\bdegrees\b"
    r"|\b(?:high|low)s?\b[^.!?]{0,40}?\b\d{2,3}\b", re.IGNORECASE)
#: Sentence ends, including one closed by a quote: "...'Slow Ferry.' Next".
_SENTENCE_SPLIT_RE = re.compile(
    r"(?:(?<=[.!?…])|(?<=[.!?…]['\"’”]))\s+")


def scrub_recent_text(text: str) -> str:
    """A recent break's words WITHOUT their service readings, for the
    packet's `recent` memory.

    Recent breaks are shown so the writer does not repeat itself, and the
    writer imitates them instead: it can copy the previous break's clock
    ("It's 7:15 AM") into a break that owed no clock, so two breaks
    minutes apart carry the same clock-weather-handoff shape.
    A stale clock or temperature is never conversation
    worth remembering, so every SENTENCE carrying one is dropped and what
    the DJ actually talked about stays. Dropped rather than replaced with
    a placeholder: a placeholder is one more thing to copy. The repetition
    detector (`repeated_style`) still reads the ORIGINAL text."""
    keep = [s for s in _SENTENCE_SPLIT_RE.split((text or "").strip())
            if s and not clock_tokens(s) and not _RECENT_TEMP_RE.search(s)]
    return " ".join(keep).strip()


# Fact-SHAPED assertions about music that no packet can ground. The
# writer authority guard already refuses unauthorized albums, artists,
# genres, transition comparisons and station habits; this closes the rest
# of the factual surface for EVERY speaking kind, so a claim cannot ride
# out under `quip` or `transition` merely because it was not filed as
# `fact`. Deterministic vocabulary, word-bounded, scanned AFTER every
# supplied string (song, artist, album, fact text, weather, daylight,
# clock) has been masked out -- a title like "Twenty Years Late" is a
# name, not a claim. Not a classifier: creative, non-factual
# copy carries none of these shapes, and the cost of a false positive is
# one silent boundary. A place name with no predicate ("Port Avel's own")
# is the residual gap; the prompt forbids it and the guard's artist rule
# still bounds who may be named.
_CLAIM_SHAPES = (
    ("year", r"\b(?:19|20)\d\d\b"),
    ("decade", r"(?:\b|')(?:19|20)?[0-9]0s\b|\b(?:sixties|seventies|"
               r"eighties|nineties|noughties)\b"),
    ("band_history", r"\b(?:formed|founded|disbanded|broke up|split up|"
                     r"reunited|reunion)\b"),
    ("recording", r"\b(?:recorded|tracked|laid down)\b|\b(?:studio|"
                  r"studios|sessions)\b"),
    ("credit", r"\b(?:produced|engineered|mixed|mastered|written|penned|"
               r"composed|arranged)\s+by\b|\b(?:producer|producers|"
               r"songwriter|songwriters|wrote|penned|composed|co-wrote)\b"),
    ("release", r"\b(?:released|came out|debut|debuted|sophomore|"
                r"follow-up|b-side|bonus track|lead single|first single|"
                r"second single|the single|title track|opening track|"
                r"closing track|(?:first|second|third|fourth|new|latest|"
                r"last|final|solo|live|next|previous|earlier|later) album|"
                r"(?:their|his|her) album)\b"),
    ("chart_sales", r"\b(?:chart|charts|charted|billboard|top (?:ten|forty|"
                    r"10|40|100)|number one|number 1|no\. ?1|#1|went "
                    r"(?:gold|platinum)|platinum|gold record|million "
                    r"(?:copies|records|sales)|best-?sell\w*)"),
    ("award", r"\b(?:grammy|grammys|nominated|nomination|hall of fame)\b"),
    ("provenance", r"\b(?:cover of|covered by|a cover|originally by|original "
                   r"version|originally recorded|sampled|samples|remix of)\b"),
    ("lineup", r"\b(?:line-?up|frontman|frontwoman|vocalist|drummer|bassist|"
               r"guitarist|keyboardist|founding member|members of the band|"
               r"joined the band|left the band|side project|supergroup|"
               r"went on to)\b"),
    ("biography", r"\b(?:hometown|born in|born on|grew up|hails from|based in|"
                  r"native of|moved to|raised in|passed away|died)\b"),
    ("business", r"\b(?:signed to|signed with|record deal|record label|"
                 r"toured|on tour|world tour)\b"),
    ("meaning", r"\b(?:inspired by|(?:is|was|it's) about (?!to\b)|written "
                r"about|song about|tribute to|dedicated to)\b"),
    ("era", r"\b(?:years? ago|years old|anniversary|decades? ago|era)\b"),
)
_CLAIM_RE = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in _CLAIM_SHAPES]


def _supplied_strings(packet: dict[str, Any],
                      fact_ids: list[str] | None = None) -> list[str]:
    """Every string the packet GROUNDS, longest first, for the claim
    scanner to mask before it looks for invented assertions.

    A fact contributes its whole sentence and its anchor always. It
    contributes its `_grounds` -- the individual factual fragments it
    authorizes, like "produced by" or the year 2009 -- only when the
    writer actually CITED it in `fact_ids`. That asymmetry is the whole
    safety property: citing "Northern Static was produced by Ines Marlo" buys
    the right to paraphrase it, and the fact's own anchor check then
    forces the album to be audible. A break that never cites the fact
    still trips the scanner on "produced by", so a producer credit can
    never be smuggled into a quip with nothing to attach it to."""
    out: set[str] = set()
    cited = set(fact_ids or [])
    for m in packet.get("music") or []:
        for k in ("artist", "title", "album"):
            if m.get(k):
                out.add(str(m[k]))
    for f in packet.get("facts") or []:
        for k in ("text", "_anchor"):
            if f.get(k):
                out.add(str(f[k]))
        if f.get("id") in cited:
            for g in f.get("_grounds") or []:
                if g:
                    out.add(str(g))
    moment = packet.get("moment") or {}
    for k in ("weather", "daylight_line", "time", "daypart", "weekday"):
        if moment.get(k):
            out.add(str(moment[k]))
    for c in moment.get("colour") or []:
        out.add(str(c))
    # The calendar record is supplied material like any other: an event
    # called "90s Quiz night" or "Class of 2005 reunion" is a NAME, and masking it first
    # is what stops the claim scanner reading a year or decade in the
    # listener's own event title as a fabricated claim.
    cal = packet.get("calendar") or {}
    for k in ("title", "start_time", "lead"):
        if cal.get(k):
            out.add(str(cal[k]))
    ann = packet.get("announcement") or {}
    if ann.get("message"):
        out.add(str(ann["message"]))
    st = packet.get("station")
    if isinstance(st, dict) and st.get("name"):
        out.add(str(st["name"]))
    elif packet.get("station_name"):
        out.add(str(packet["station_name"]))
    return sorted((x for x in out if len(x.strip()) >= 2), key=len,
                  reverse=True)


def ungrounded_claim(text: str, packet: dict[str, Any],
                     fact_ids: list[str] | None = None) -> str | None:
    """'<shape>:<matched words>' when the text carries a fact-shaped music
    assertion the packet cannot ground, else None. Supplied strings are
    masked first, so only the model's own additions are scanned."""
    low = expand_apostrophe_years(text or "").lower()
    if not low:
        return None
    for mask in _supplied_strings(packet, fact_ids):
        low = re.sub(re.escape(mask.lower()), " \u00a7 ", low)
    # "from their album, Northern Static" names a SUPPLIED album (now masked)
    # and asserts nothing the packet did not: the music entry already
    # ties that album to that artist. Refusing it would refuse a supplied
    # album every time. Only the possessive directly before a
    # masked name is exempt; "their debut album" and "their album" with
    # no name behind it are claims as before.
    low = re.sub(r"\b(?:their|his|her)\s+album\b[\s,:]*(?:called|titled|"
                 r"named)?\s*(?=\u00a7)", " ", low)
    for label, rx in _CLAIM_RE:
        m = rx.search(low)
        if m:
            return f"{label}:{m.group(0).strip()[:24]}"
    return None


# --- elapsed spans of years ------------------------------------------------------
# The failure shape: "they formed in '04 and it took eighteen years to win
# them the lifetime achievement award." Eighteen is a real value -- the
# career-span bridge (formed 2004, an album in 2022) -- re-attached by the
# writer to the award row, which carries no year at all. With both facts
# cited, every check passes: the claim scanner masks a cited
# fact's grounds and the realization check looks at the whole line. A
# spoken span is therefore judged SENTENCE BY SENTENCE: its number must be
# a value of a cited relationship, the sentence must be about a release,
# and it may not hang the span on an award, a birth, a death or a "took N
# years to" achievement. Uncited, a span is never grounded.
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SPAN_RE = re.compile(
    r"\b(\d{1,3}|(?:" + "|".join(_TENS) + r")(?:[- ](?:" + "|".join(
        list(_UNITS)[:9]) + r"))?|" + "|".join(_UNITS) + r")[- ]years?\b"
    r"(?![- ]old)", re.IGNORECASE)
_SPAN_ELSEWHERE_RE = re.compile(
    r"\b(?:hall of fame|inducted|induction|grammy|grammys|award|awards|"
    r"nominat\w*|born|died|death|joined|signed|took|taken|to get)\b",
    re.IGNORECASE)
_SPAN_RELEASE_RE = re.compile(
    r"\b(?:album|record|records|came|came out|arrived|followed|released|"
    r"release|dropped|lp|apart|later)\b", re.IGNORECASE)
#: the bridge values that ARE spans (fact_editorial)
_SPAN_KEYS = ("years", "years_apart", "years_ago")


def _spoken_int(token: str) -> int | None:
    t = token.lower().replace("-", " ").split()
    if not t:
        return None
    if t[0].isdigit():
        return int(t[0])
    total = 0
    for w in t:
        if w in _TENS:
            total += _TENS[w]
        elif w in _UNITS:
            total += _UNITS[w]
        else:
            return None
    return total or None


def elapsed_span_problem(text: str, packet: dict[str, Any],
                         fact_ids: list[str] | None = None) -> str | None:
    """'elapsed_years:<n>' when a sentence states a span of years that no
    cited relationship supplies for THAT sentence, else None."""
    if not text:
        return None
    supplied = {f.get("id"): f for f in (packet.get("facts") or [])}
    spans: set[int] = set()
    for fid in fact_ids or []:
        values = (supplied.get(fid) or {}).get("values") or {}
        for k in _SPAN_KEYS:
            if isinstance(values.get(k), int):
                spans.add(int(values[k]))
    # Names are not claims: "Twenty Years Late" is a song, not a span. Mask the
    # music, the station and the calendar first (as the claim scanner does)
    # -- but NOT a cited fact's grounds, which carry the very numbers judged
    # here.
    masks: list[str] = []
    for m in packet.get("music") or []:
        masks += [str(m.get(k)) for k in ("artist", "title", "album")
                  if m.get(k)]
    st = packet.get("station")
    masks += [str(st.get("name"))] if isinstance(st, dict) and st.get(
        "name") else ([str(packet["station_name"])]
                      if packet.get("station_name") else [])
    cal = packet.get("calendar") or {}
    masks += [str(cal[k]) for k in ("title",) if cal.get(k)]
    text = str(text)
    for mask in sorted((x for x in masks if len(x.strip()) >= 2), key=len,
                       reverse=True):
        text = re.sub(re.escape(mask), " \u00a7 ", text, flags=re.I)
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for m in _SPAN_RE.finditer(sentence):
            n = _spoken_int(m.group(1))
            if n is None:
                continue
            if n not in spans or _SPAN_ELSEWHERE_RE.search(sentence) or \
                    not _SPAN_RELEASE_RE.search(sentence):
                return f"elapsed_years:{n}"
    return None


# --- small helpers ----------------------------------------------------------------

def _clip(value: Any, limit: int) -> str:
    s = str(value or "").strip()
    return s if len(s) <= limit else s[: max(0, limit - 1)].rstrip() + "…"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _h8(*parts: str) -> str:
    return hashlib.sha1("|".join(_norm(p) for p in parts).encode(
        "utf-8")).hexdigest()[:8]


def _split_name(name: str | None) -> tuple[str, str]:
    import break_packet
    return break_packet.split_name(name)


def _public(value: Any) -> Any:
    """The packet without its validator-only keys (a leading underscore)
    and without its validator-only ENTRIES (`_hidden`): the fact anchor
    is machine grounding, not model material, and a reference fact is
    grounding entirely."""
    if isinstance(value, dict):
        return {k: _public(v) for k, v in value.items()
                if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_public(v) for v in value
                if not (isinstance(v, dict) and v.get("_hidden"))]
    return value


def visible_facts(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """The facts the WRITER is shown."""
    return [f for f in (packet.get("facts") or [])
            if isinstance(f, dict) and not f.get("_hidden")]


def serialize(packet: dict[str, Any]) -> bytes:
    """The exact bytes the model sees for the packet (compact JSON)."""
    return json.dumps(_public(packet), ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def packet_bytes(packet: dict[str, Any]) -> int:
    return len(serialize(packet))


# --- the packet -------------------------------------------------------------------

def _music_item(role: str, artist: str, title: str, *, album: str = "",
                character: dict[str, Any] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"role": role}
    if artist:
        item["artist"] = _clip(artist, NAME_MAX)
    if title:
        item["title"] = _clip(title, NAME_MAX)
    if album:
        item["album"] = _clip(album, NAME_MAX)
    if character:
        char: dict[str, Any] = {}
        if character.get("genre"):
            char["genre"] = _clip(str(character["genre"]).replace("_", " "),
                                  32)
        if character.get("energy"):
            char["energy"] = _clip(character["energy"], 16)
        for flag in ("live", "demo", "instrumental"):
            if character.get(flag):
                char[flag] = True
        if char:
            item["character"] = char
    return item


def _previous_from(resolved: list[dict[str, Any]] | None,
                   ending: tuple[str, str]) -> tuple[str, str] | None:
    """The track that PLAYED before the ending one, from the session's
    resolved history (played outcome only), or None. Identity only."""
    end_key = (_norm(ending[0]), _norm(ending[1]))
    for row in reversed(list(resolved or [])[-6:]):
        if not isinstance(row, dict) or row.get("outcome") != "played":
            continue
        artist = str(row.get("artist") or "").strip()
        title = str(row.get("title") or "").strip()
        if not (artist or title):
            continue
        if (_norm(artist), _norm(title)) == end_key:
            continue
        return artist, title
    return None


def candidates(packet: dict[str, Any], recent_topics: set[str],
               context: dict[str, Any]) -> list[str]:
    """The kinds the model may choose from at THIS boundary: a deterministic
    editorial pre-filter, not a scoring engine. A kind is offered only when
    its grounding exists in the packet and the same topic was not just on
    air; SAY_NOTHING is always available."""
    out = [SAY_NOTHING]
    music = packet.get("music") or []
    roles = {m.get("role") for m in music}
    has_song = bool(roles & {"ending", "next"})
    if has_song:
        out += [TRANSITION, QUIP, OBSERVATION]
    if visible_facts(packet):
        out.append(FACT)
    moment = packet.get("moment") or {}
    if moment.get("weather"):
        key = topic_key(WEATHER, packet)
        if moment.get("weather_changed") or key not in recent_topics:
            out.append(WEATHER)
    if moment.get("daylight_line"):
        phase = moment.get("daylight")
        if ((context.get("daylight_salient") or phase in SALIENT_DAYLIGHT)
                and topic_key(DAYLIGHT, packet) not in recent_topics):
            out.append(DAYLIGHT)
    nxt = next((m for m in music if m.get("role") == "next"), None)
    if nxt and (nxt.get("title") or nxt.get("artist")):
        if topic_key(TEASE_NEXT, packet) not in recent_topics:
            out.append(TEASE_NEXT)
    # The calendar band was already qualified deterministically upstream
    # (dj_context.calendar_packet: band, novelty, freshness, privacy). The
    # only thing left to check here is the planner's own short memory, so
    # the same band cannot go out twice inside one recent window even if
    # the durable mark has not committed yet.
    if packet.get("calendar"):
        if topic_key(CALENDAR, packet) not in recent_topics:
            out.append(CALENDAR)
    return out


def topic_key(action: str, packet: dict[str, Any],
              fact_ids: list[str] | None = None) -> str | None:
    """A short stable key for novelty/cooldown. Weather keys on the
    supplied condition, daylight on its phase, a tease on the next song's
    identity, a fact on its id; the free kinds key on themselves."""
    moment = packet.get("moment") or {}
    if action == SAY_NOTHING:
        return None
    if action == WEATHER:
        cond = moment.get("weather_condition") or "now"
        return f"weather:{_norm(cond).replace(' ', '_') or 'now'}"
    if action == DAYLIGHT:
        return f"daylight:{moment.get('daylight') or 'now'}"
    if action == TEASE_NEXT:
        nxt = next((m for m in (packet.get("music") or [])
                    if m.get("role") == "next"), None) or {}
        return f"tease:{_h8(nxt.get('artist') or '', nxt.get('title') or '')}"
    if action == FACT:
        ids = list(fact_ids or [])
        return f"fact:{ids[0]}" if ids else "fact"
    if action == CALENDAR:
        cal = packet.get("calendar") or {}
        return (f"calendar:{cal.get('id') or 'event'}:"
                f"{cal.get('timing_band') or 'now'}")
    return action


def _condition_of(service_sig: str | None, weather_line: str) -> str:
    """The weather condition for the topic key: the deterministic service
    signature's condition component when the scheduler supplied it, else
    the condition phrase the formatted line carries."""
    if service_sig and "|" in service_sig:
        parts = service_sig.split("|")
        if len(parts) >= 2 and parts[1] not in ("?", ""):
            return parts[1]
    head = (weather_line or "").split(";")[0]
    return head.split(" and ")[-1].strip() if " and " in head else ""


# --- delivery: the anti-template signal -------------------------------------
#
# A station develops a house sentence, typically "That was X. <context>.
# Up next is Y." -- technically correct every time and dead on the third
# hearing. These are the SHAPES that sentence is made of, matched against
# the recent breaks' own text; a shape carried by STYLE_REPEAT_MIN of them
# is named back to the writer as something to break out of. This is a
# handful of regexes over at most four short strings, not a memory system,
# and it never touches what a break may CLAIM -- only how it is built.

_STYLE_SHAPES = (
    # the retrospective opening: the break starts by looking backwards
    ("looked_back_first",
     r"^\W*(?:and\s+)?(?:that (?:was|is)|that's|there (?:was|goes)|"
     r"you(?:'ve| have)? just heard|we just heard|you just heard)\b"),
    # the trailing signpost: the break ends by announcing what follows
    ("signposted_the_next_song",
     r"\b(?:up next|next up|coming up|here's|here is|stay (?:tuned|with)|"
     r"after this|we'll be back)\b[^.!?]*[.!?]?\s*$"),
    # the assistant's throat-clearing, wherever it appears
    ("assistant_scaffolding",
     r"^\W*(?:here'?s|here is|currently|right now|the (?:weather|time) is|"
     r"it (?:is|'s) currently)\b|\bthe weather is\b"),
    # a question asked of the listener
    ("asked_a_question", r"\?"),
    # spoke to the room directly
    ("direct_address", r"\b(?:you|your|you're|yourself|folks|listeners?)\b"),
    # the clock as the first words. Left alone, the writer opens most
    # breaks "It's <time>" -- a due time check is owed, its place in the
    # sentence is not
    ("opened_with_the_time",
     r"^\W*(?:(?:it'?s|it is|the time is)\s+)?(?:just\s+)?\d{1,2}:\d{2}\b"),
    # the dry-wit tic: a hedge tacked on instead of a thought ("a brisk
    # one, I suppose", "quite the evening, for sure")
    ("hedged_aside",
     r"\b(?:i suppose|i guess|for sure|quite the|understandable|brisk)\b"),
)
_STYLE_RE = [(k, re.compile(v, re.IGNORECASE)) for k, v in _STYLE_SHAPES]
#: Below this, a break is a one-liner rather than a built sentence.
_SHORT_QUIP_WORDS = 12


def style_flags(row: dict[str, Any]) -> set[str]:
    """The structural shapes ONE recent break used. Pure, bounded, and
    derived from the row the recent-memory reader already produced -- the
    spoken text plus the planner kind and topic key it was filed under, so
    'this break leaned on the weather' does not need the weather back."""
    out: set[str] = set()
    text = str(row.get("text") or "").strip()
    if text:
        for name, rx in _STYLE_RE:
            if rx.search(text):
                out.add(name)
        if len(text.split()) <= _SHORT_QUIP_WORDS:
            out.add("one_short_line")
    topic = str(row.get("topic") or "")
    kind = str(row.get("kind") or "")
    if topic.startswith("weather:") or kind == WEATHER:
        out.add("used_the_weather")
    if topic.startswith("calendar:") or kind == CALENDAR:
        out.add("used_the_calendar")
    if topic.startswith("tease:") or kind == TEASE_NEXT:
        out.add("signposted_the_next_song")
    # The house sentence is a COMBINATION, so it is judged last -- the
    # signpost half can come from the topic key as easily as from the
    # text, and judging it mid-way missed every break whose tease was
    # filed rather than written at the very end of the line.
    if "looked_back_first" in out and "signposted_the_next_song" in out:
        out.add("song_then_context_then_song")
    return out


#: Worst habit first. The list is truncated to AVOID_MAX, so this order is
#: what decides which tic gets named -- alphabetical would have dropped the
#: house sentence itself in favour of "asked a question".
_AVOID_SEVERITY = ("song_then_context_then_song", "opened_with_the_time",
                   "hedged_aside", "looked_back_first",
                   "signposted_the_next_song", "assistant_scaffolding",
                   "used_the_weather", "used_the_calendar", "one_short_line",
                   "asked_a_question")

#: Shapes that are DESCRIBED but never discouraged. Talking to the room is
#: a register, not a template -- it is one of the things this design
#: wants more of -- and `you` is common enough that it would be flagged
#: almost every break, quietly training the DJ back out of the warmth it
#: was just given. style_flags still reports it; only `avoid` ignores it.
_AVOID_NEVER = ("direct_address",)


def repeated_style(rows: list[dict[str, Any]] | None,
                   owed: Any = ()) -> list[str]:
    """The shapes at least STYLE_REPEAT_MIN of the recent breaks share and
    that are worth breaking out of -- worst first, bounded to AVOID_MAX.

    `owed` is what THIS break must carry ("weather", "calendar"). A habit
    is never named against an obligation: otherwise the prompt says
    "REQUIRED THIS BREAK: the current weather" and, a few lines later, "Do
    not do these again: reaching for the weather" (the recent breaks had
    paid the weather that was due), and drafts skip the reading and lose
    the slot to the deterministic floor."""
    owed = set(owed or ())
    counts: dict[str, int] = {}
    for row in list(rows or [])[:MAX_RECENT]:
        for flag in style_flags(row):
            counts[flag] = counts.get(flag, 0) + 1
    hits = [f for f, n in counts.items()
            if n >= STYLE_REPEAT_MIN and f not in _AVOID_NEVER
            and not (f == "used_the_weather" and "weather" in owed)
            and not (f == "used_the_calendar" and "calendar" in owed)]
    hits.sort(key=lambda f: (_AVOID_SEVERITY.index(f)
                             if f in _AVOID_SEVERITY else len(_AVOID_SEVERITY),
                             f))
    return hits[:AVOID_MAX]


# --- the editorial premise ---------------------------------------------------
#
# WHAT is worth talking about, in one clause, decided in code. The model is
# not asked to reason about the moment and then write; it is handed the
# angle and performs it. Deliberately a short fixed vocabulary: a premise
# is a nudge toward a shape of break, never prose to be edited into copy,
# and never a factual claim (every fact still comes from the packet's own
# fields and is still guarded exactly as before).

TONES = ("casual", "energized", "understated", "warm", "conversational",
         "dry")
_NO_OCCASION = ("nothing is going on -- only speak if a genuinely good "
                "line is already there", "dry")


def editorial(packet: dict[str, Any]) -> dict[str, str]:
    """The compact editorial premise for this break: {angle, tone}. Pure,
    deterministic, first match wins. Reads the PACKET, not the render
    context: everything the premise turns on has already been bounded and
    clipped on the way in, so the angle cannot outgrow the packet."""
    cal = packet.get("calendar") or {}
    moment = packet.get("moment") or {}
    transition = packet.get("transition")
    feel = packet.get("feel") or {}
    band = cal.get("timing_band")
    angle, tone = _NO_OCCASION
    occasion = True
    if band == "1h":
        angle, tone = ("the listener's next thing is close now -- let it "
                       "colour the break, never announce it like an alarm",
                       "casual")
    elif transition == "major_contrast":
        angle, tone = ("these two songs do not agree with each other -- "
                       "make the turn feel deliberate", "energized")
    elif transition == "contrast":
        angle, tone = ("the mood shifts across this break -- carry the "
                       "listener over it", "energized")
    elif transition in ("album_run", "album_run_ordered"):
        angle, tone = ("the record is still running -- stay out of its way",
                       "understated")
    elif band in ("2h", "4h", "allday"):
        angle, tone = ("there is something later on -- a passing nod at "
                       "most, worked into whatever you were saying anyway",
                       "casual")
    elif moment.get("weather_changed"):
        angle, tone = ("the weather has actually turned -- if it belongs "
                       "anywhere it belongs here", "conversational")
    elif feel.get("tone"):
        angle, tone = (f"lean into the {_clip(feel['tone'], 24)} of what "
                       "just played", "warm")
    elif moment.get("daypart"):
        angle, tone = (f"it is {_clip(moment['daypart'], 16)} -- let that "
                       "set the pace, not the subject", "conversational")
    else:
        occasion = False
    # The host has the last word on HOW it sounds (dj_persona.tone_for):
    # no occasion means the host's own register, never a blanket "dry".
    persona = packet.get("persona") or {}
    if persona.get("profile"):
        tone = dj_persona.tone_for(dj_persona.get_profile(persona["profile"]),
                                   tone, occasion)
    return {"angle": _clip(angle, ANGLE_MAX),
            "tone": tone if tone in TONES else "dry"}


# --- HOST FIRST, FLAIR SECOND: the hosting payload rule ----------------------
#
# The DJ's job is to host the music. Personality decorates a payload; it is
# not itself a reason to interrupt a song. Consider a break that passes
# every factual guard because it asserts nothing at all:
#
#     "Well, that was something different."
#
# Technically valid, grounded in nothing, and worth less than the silence
# it replaced. So a speaking decision must carry at least ONE authoritative
# anchor that the listener actually heard -- an identity, a fact, their
# calendar, or a real service reading.
#
# This is a FLOOR, not a template. It does not ask for both songs, it does
# not force weather or calendar, and it constrains nothing about phrasing:
# one anchor anywhere in the line is enough, and everything around it stays
# as free as the delivery/flair rules leave it. It also widens nothing -- an anchor
# only counts when the existing guards already consider it grounded.
#
# Kinds that already prove an anchor (fact / weather / daylight /
# tease_next / calendar) satisfy it for free; in practice it bites on
# exactly the three free kinds -- transition, quip, observation -- which is
# where the banter came from.

ANCHOR_SONG = "song_identity"
ANCHOR_FACT = "grounded_fact"
ANCHOR_CALENDAR = "calendar_event"
#: Service readings are anchors under the names the spoken-concept
#: detector already reports them by (weather / daylight / station_time),
#: so there is one vocabulary for "did the listener actually hear this".
ANCHORS = (ANCHOR_SONG, ANCHOR_FACT, ANCHOR_CALENDAR, WEATHER, DAYLIGHT,
           "station_time")

#: A supplied name shorter than this is not treated as an identity: a
#: two-character title cannot be told apart from ordinary words, and a
#: false anchor would re-admit the banter this rule exists to refuse.
#: Erring toward silence is the cheap direction -- airtime is earned.
ANCHOR_NAME_MIN = 3

#: The MUSIC-DERIVED kinds: their whole premise is "the music did
#: something". For these the listener must be told WHICH music, by a
#: supplied title or artist, in the words they hear.
#:
#: This is the listener-visible referent invariant, and it is stricter
#: than "the break hosted something". After the first hosting rule was
#: written the model still produced lines of this shape: a vague musing
#: about sounds, songs or stillness that names no song, act, time or
#: weather -- hosting in form, about nothing.
#:
#: Both were genuinely inspired by the real supplied music, and neither
#: ever tells the listener what they are about. A referent the packet
#: holds and the copy withholds is not a referent. So a weather reading
#: or a calendar event cannot rescue a `transition`: if the premise is
#: the diary, the answer is `calendar`, and it must say the event.
MUSIC_DERIVED = (TRANSITION, QUIP, OBSERVATION)

#: Kinds whose OWN validation already proved their referent before this
#: rule is reached: a fact whose phrase was stated, a tease that named the
#: next song, a weather or daylight line the listener heard, an event said
#: by name. Re-judging them here would be a second, stricter opinion about
#: a question already answered -- and it was: `tease_next` accepts a
#: one-character artist its own way, which this rule's word-boundary
#: minimum would then refuse. One referent, decided once, by whichever
#: check owns the kind.
_PRE_ANCHORED = {FACT: ANCHOR_FACT, TEASE_NEXT: ANCHOR_SONG,
                 WEATHER: WEATHER, DAYLIGHT: DAYLIGHT,
                 CALENDAR: ANCHOR_CALENDAR}


def _names_in(text_low: str, packet: dict[str, Any]) -> bool:
    """Did the text actually name the ending or the next song/artist?

    `previous` deliberately does not count: the prompt forbids describing
    it, so it can never be this break's payload."""
    for item in packet.get("music") or []:
        if item.get("role") not in ("ending", "next"):
            continue
        for key in ("artist", "title"):
            name = _norm(item.get(key) or "")
            if len(name) < ANCHOR_NAME_MIN:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])",
                         text_low):
                return True
    return False


def hosting_anchor(action: str, text: str, packet: dict[str, Any],
                   fact_ids: list[str] | None = None,
                   heard: set[str] | None = None) -> str | None:
    """The referent this spoken line actually REALIZES for `action`, or
    None when the listener was never told what the break was about.

    Judged against the text they will HEAR, never against what the model
    was offered or filed the answer under: a break that cites a fact id
    but never says the fact, or muses on a transition without naming
    either song, has realized nothing. A music-derived kind can only be
    realized by music identity -- a weather line does not explain what
    `quite a departure` was a departure FROM."""
    low = _norm(text or "")
    if not low:
        return None
    if action in MUSIC_DERIVED:
        return ANCHOR_SONG if _names_in(low, packet) else None
    if _PRE_ANCHORED.get(action):
        # proved by the kind's own check, immediately above the call
        return _PRE_ANCHORED[action]
    # An unknown speaking kind must still say something real.
    supplied = {f["id"]: f for f in (packet.get("facts") or [])}
    for fid in fact_ids or []:
        f = supplied.get(fid)
        if f is not None and fact_realized(f, text) is None:
            return ANCHOR_FACT
    if calendar_spoken(text, packet):
        return ANCHOR_CALENDAR
    for concept in (WEATHER, DAYLIGHT, "station_time"):
        if concept in (heard or set()):
            return concept
    return None


# --- HOSTING VALUE: evaluation is seasoning, never the payload --------------
#
# `hosting_anchor` asks whether the listener was told WHAT the break is
# about. It does not ask whether they were told anything worth the
# interruption. Lines like these pass every factual guard and the anchor
# rule, because each one names its music:
#
#     "That was Cold Orchard, and that one never gets old."
#     "Wren Hollis there, off Northern Static. Worth your time."
#     "Some songs just stay with you. That was Cold Orchard."
#
# The rule is NOT that these phrases are banned. "Never gets old" is
# good seasoning on a line that also has something in it, and a blanket
# ban would cost the station its voice. The rule is that generic
# evaluation may not be the PRINCIPAL PAYLOAD -- take the adjective out,
# and if nothing is left, the break was filler wearing a referent.
#
# So a music-derived break must carry ONE payload the packet actually
# authorized. Everything here is already proved grounded by a check above;
# this adds no new authority and widens nothing.
_PAYLOAD_FACT = "grounded_fact"
_PAYLOAD_SERVICE = "service_reading"
_PAYLOAD_CALENDAR = "calendar_event"
_PAYLOAD_HANDOFF = "next_identity"
_PAYLOAD_SPECIFIC = "specific_detail"

#: The SHAPES of generic evaluation -- a small, deliberately shallow set of
#: the constructions a writer model actually leans on, not a phrase
#: blacklist and not a style censor. It is consulted for exactly one
#: purpose: deciding whether a payload-less line had anything in it BESIDES
#: evaluation. A line that carries a payload is never looked at here.
_EVAL_SHAPES = re.compile(
    r"\ba\s+certain\s+\w+"
    r"|\bquite\s+(?:a|an|the|something)\b"
    r"|\ban?\s+(?:sturdy|solid|nice|good|fine|decent|lovely|pleasant|"
    r"wonderful|remarkable|spirited|interesting)\s+\w+"
    r"|\b(?:it'?s|that'?s|it\s+is|that\s+is)\s+(?:interesting|remarkable|"
    r"something|quite\s+\w+)\b"
    r"|\b(?:interesting|remarkable|funny|striking)\s+(?:how|that|to)\b"
    r"|\b(?:has|had|have|with)\s+an?\s+(?:certain|interesting|real)\s+\w+"
    r"|\b(?:wouldn'?t|doesn'?t|didn'?t|isn'?t|aren'?t|don'?t|hasn'?t)\s+"
    r"(?:you|it|he|she|they|we)\b",
    re.IGNORECASE)

#: Something concrete the listener could carry away. A digit is the
#: obvious one (a year, a temperature, a count). The other is a
#: capitalized word that does NOT open its sentence and is none of the
#: supplied song names -- a place, a person, a label, a day. Both are
#: crude on purpose: this decides "was there anything here at all", and
#: erring toward LETTING COPY THROUGH is the right direction for a rule
#: whose failure mode is silencing a good break.
_CAP_TOKEN_RE = re.compile(r"(?<=[a-z,;:]\s)([A-Z][A-Za-z']{2,})")


def _supplied_name_words(packet: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for item in packet.get("music") or []:
        for key in ("artist", "title", "album"):
            out.update(_norm(item.get(key) or "").split())
    for fact in packet.get("facts") or []:
        out.update(_norm(fact.get("text") or "").split())
    st = packet.get("station")
    if isinstance(st, dict) and st.get("name"):
        out.update(_norm(str(st["name"])).split())
    elif packet.get("station_name"):
        out.update(_norm(str(packet["station_name"])).split())
    return {w for w in out if w}


def specific_detail(text: str, packet: dict[str, Any]) -> bool:
    """Did the line say anything concrete beyond the song names and the
    opinion wrapped around them?"""
    if re.search(r"\d", text or ""):
        return True
    names = _supplied_name_words(packet)
    for m in _CAP_TOKEN_RE.finditer(text or ""):
        if _norm(m.group(1)) not in names:
            return True
    return False


def hosting_payload(action: str, text: str, packet: dict[str, Any],
                    fact_ids: list[str] | None = None,
                    heard: set[str] | None = None) -> str | None:
    """What this break is actually WORTH, or None when generic evaluation
    was the whole of it.

    Only the three free music-derived kinds are judged. The kinds that
    prove their own referent (fact / weather / daylight / tease_next /
    calendar) have a payload by construction -- a stated fact, a real
    reading, the listener's own event, a named next song -- and re-judging
    them here would be a second opinion on a settled question."""
    if action not in MUSIC_DERIVED:
        return _PRE_ANCHORED.get(action) or _PAYLOAD_SPECIFIC
    supplied = {f["id"]: f for f in (packet.get("facts") or [])}
    for fid in fact_ids or []:
        f = supplied.get(fid)
        if f is not None and fact_realized(f, text) is None:
            return _PAYLOAD_FACT
    for concept in (WEATHER, DAYLIGHT, "station_time"):
        if concept in (heard or set()):
            return _PAYLOAD_SERVICE
    if calendar_spoken(text, packet):
        return _PAYLOAD_CALENDAR
    # Handing the listener to the next song by name IS hosting value: it
    # is the close the whole design is about.
    if _named_next(text, packet):
        return _PAYLOAD_HANDOFF
    if specific_detail(text, packet):
        return _PAYLOAD_SPECIFIC
    # Nothing authorized, nothing concrete. If what is left is evaluation,
    # the evaluation WAS the break.
    return None if _EVAL_SHAPES.search(text or "") else _PAYLOAD_SPECIFIC


# --- THIN CHRONOLOGY IS NOT A BREAK ON ITS OWN -----------------------------
#
# Accepted under the hosting rules because `fact` is a pre-anchored kind and
# never reaches the payload test:
#
#     "Northern Static, the Cold Orchard album, came out back in 2009.
#      Quite a statement, that one."
#
# The album is named, the year is said, the citation is honest. It is
# still a catalogue row read aloud with an adjective on each end.
#
# This is NOT a judgement about whether a fact is interesting -- that
# rule was considered and deliberately refused, because no validator can
# hold an opinion. It is a judgement about a PREDICATE: a release date or
# a record label states when a thing came out or who put it out, and
# nothing else. Every other predicate the pack can carry -- who produced
# it, who was in the band, where they came from, what they won -- says
# something a listener could not have guessed, and stands alone fine.
#
# So: when EVERY fact a break cites is thin, the break must also carry one
# real station element. Generic evaluation is explicitly not one, which is
# the whole point -- "quite a record" is what the thin break reaches
# for when it has nothing else.
THIN_PREDICATES = ("original_recording_release_date",
                   "reissue_release_date",
                   "album_record_label")


def thin_fact_only(packet: dict[str, Any],
                   fact_ids: list[str] | None) -> bool:
    """Is every fact this break cites a bare catalogue date or label?

    False when nothing was cited (an uncited break is not this rule's
    business) and false as soon as ONE substantive fact is present. A
    fact carrying no predicate at all counts as substantive, so this can
    only ever bite on the library-knowledge predicates it names."""
    ids = [i for i in (fact_ids or []) if i]
    if not ids:
        return False
    supplied = {f["id"]: f for f in (packet.get("facts") or [])}
    seen = False
    for fid in ids:
        f = supplied.get(fid)
        if f is None:
            continue
        predicate = (f.get("_realize") or {}).get("predicate")
        if not predicate or predicate not in THIN_PREDICATES:
            return False
        seen = True
    return seen


# --- a tease may be editorial; it may not point into empty space -----------
#
# Lines of this shape were accepted:
#
#     "Cold Orchard, with real weight to it. Things head somewhere else
#      from here on out."
#     "Wren Hollis has a certain warmth, doesn't it? That sets the stage
#      nicely."
#
# Both point forward. Neither says where. The listener is told something
# is about to change and never told into what, which is worse than not
# mentioning it at all -- the station has opened a loop and walked away.
#
# Bounded deliberately: the shapes below are the forward-looking MOVES a
# host actually makes, and the rule fires only when the packet HAS a next
# identity to name. No next song, no obligation.
_FORWARD_SHAPES = re.compile(
    r"\bcoming\s+up\b|\bup\s+next\b|\bnext\s+up\b"
    r"|\bwhat(?:'?s|\s+is|\s+comes)\s+next\b|\bwhat'?s\s+(?:ahead|coming)\b"
    r"|\bfrom\s+here\b|\bhere\s+on\s+out\b|\bfrom\s+now\s+on\b"
    r"|\bsets?\s+(?:the|us)\s+(?:stage|up)\b"
    r"|\bchang(?:e|es|ing)\s+of\s+pace\b|\bshift(?:s|ing)?\s+gears\b"
    # "stick around" is gone: it refused "how long some bands stick
    # around", which is a remark about longevity and
    # not a tease at all. "stay tuned" has no such second reading.
    r"|\bstay\s+tuned\b|\bjust\s+ahead\b"
    r"|\bin\s+a\s+moment\b|\bafter\s+(?:this|the\s+break)\b"
    r"|\bwe'?ll\s+hear\b|\bis\s+waiting\b|\bare\s+waiting\b",
    re.IGNORECASE)

#: WHICH kinds a forward reference belongs to. The three free music kinds
#: and `fact`: their premise is the music, so "coming up" in one of them is
#: a reference to the next SONG and has to land. Deliberately NOT the
#: others:
#:
#:   tease_next  proves its own destination, its own way, one line above --
#:               and accepts a one-character artist this rule's
#:               word-boundary minimum would refuse. One referent, decided
#:               once, by whichever check owns the kind.
#:   weather / daylight / calendar
#:               "82 still to come", "the light goes in an hour", "before
#:               you leave" point at the FORECAST, the sky and the diary,
#:               not at a track. Reading them as a dangling song tease
#:               would silence good service copy to fix a music defect.
_FORWARD_KINDS = MUSIC_DERIVED + (FACT,)


def _next_names(packet: dict[str, Any]) -> list[str]:
    """Every name the next song has, however short."""
    nxt = next((m for m in (packet.get("music") or [])
                if m.get("role") == "next"), None) or {}
    return [n for n in (_norm(nxt.get("artist") or ""),
                        _norm(nxt.get("title") or "")) if n]


def _named_next(text: str, packet: dict[str, Any]) -> bool:
    """Did the listener hear the next song's identity, STRICTLY enough to
    credit it as this break's payload? Word-bounded and subject to
    ANCHOR_NAME_MIN, because a two-character title cannot be told apart
    from ordinary words and a false handoff would re-admit the banter the
    payload rule exists to refuse."""
    low = _norm(text or "")
    for name in _next_names(packet):
        if len(name) < ANCHOR_NAME_MIN:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", low):
            return True
    return False


def _mentions_next(text: str, packet: dict[str, Any]) -> bool:
    """Did the copy name the next song AT ALL? The same generous substring
    test `tease_next` has always used, one-character artists included.

    Deliberately looser than `_named_next`: this one only ever CLEARS the
    dangling rule, and the two tests pull in opposite directions. Refusing
    to credit a doubtful handoff costs a thin break its payload; refusing
    to notice one costs a perfectly good break its voice. The strict test
    reads "That was One from A. B takes it from here." as dangling -- the
    destination is right there."""
    low = _norm(text or "")
    return any(n in low for n in _next_names(packet))


def dangling_forward(action: str, text: str,
                     packet: dict[str, Any]) -> bool:
    """A forward-looking tease with no listener-visible destination, when
    the packet was holding one the whole time."""
    if action not in _FORWARD_KINDS:
        return False
    if not _FORWARD_SHAPES.search(text or ""):
        return False
    if not _next_names(packet):
        return False          # nothing to name: not this rule's business
    return not _mentions_next(text, packet)


def _said(token: str, low: str) -> bool:
    """Is this object token in the normalized copy? A NUMBER has to
    stand alone: otherwise a twenty-year gap is 'said' by the '20'
    inside 'Now it's 2016'. Words keep the substring rule the
    realization policy was written for (stems)."""
    t = _norm(str(token))
    if not t:
        return False
    if t.isdigit():
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])",
                              low))
    return t in low


def fact_realized(fact: dict[str, Any], text: str) -> str | None:
    """Did this spoken line actually REALIZE the fact it cites? None when
    it did, else the failure subtype.

    Runs for every cited fact under EVERY speaking kind, not only for
    `action == FACT`. Otherwise a line filed as an `observation` could
    cite a release-date fact, never name the album, and put a bare year
    on the air with nothing behind it. The novelty ledger refuses to
    credit such a line -- which is correct, and is exactly why the gap
    would be invisible from the outside. A citation is a claim whatever
    box it is filed under.

    Three questions, in the order that makes the failure legible:

      SUBJECT  the thing the fact is about must be audible. A producer
               credit with no album is not a producer credit.
      VALUE    the fact's own object -- the producer, the label, the
               year, the place -- must be said. Citing a fact for its
               subject alone states nothing.
      RELATION the predicate's relationship must survive. This is the
               producer defect: album_producer(Ada Venn, Northern Static)
               spoken as "she was in the studio with Ada Venn then" names
               both nouns and asserts a personal history nothing
               supplied can support. The policy travels on the fact (music pack), so
               this function enforces relationships it does not itself
               know -- and a new predicate needs no change here.

    Conservative: a fact carrying no policy is checked for its subject
    only, exactly as before, so nothing that used to pass starts failing
    for want of a rule."""
    low = _norm(expand_apostrophe_years(text or ""))
    if not low:
        return FAIL_FACT_SUBJECT
    anchor = _norm(fact.get("_anchor") or "")
    if anchor and anchor not in low:
        return FAIL_FACT_SUBJECT
    policy = fact.get("_realize") or {}
    # A RELATIONSHIP has more than one subject, and every one of them
    # has to be audible: "both came out in '95" with one album named is
    # not a relationship the listener can follow. Each group lists the
    # alternative names for one subject (album or artist); one per group
    # is enough. Plain facts carry no `subjects` and keep the anchor rule.
    for group in (policy.get("subjects") or []):
        alts = [_norm(str(a)) for a in (group or []) if a]
        alts = [a for a in alts if a]
        if alts and not any(a in low for a in alts):
            return FAIL_FACT_SUBJECT
    if not policy:
        return None
    obj = [o for o in (policy.get("object") or []) if o]
    if obj and not any(_said(o, low) for o in obj):
        return FAIL_FACT_VALUE
    need = [r for r in (policy.get("require") or []) if r]
    if need and not any(_norm(r) in low for r in need):
        return FAIL_FACT_RELATION
    return None


def _whole(token: str, low: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])",
                          low))


def infer_citations(text: str, packet: dict[str, Any],
                    cited: list[str] | None) -> list[str]:
    """Supplied facts the copy FULLY REALIZES without citing.

    Deliberately stricter than the check a cited fact gets: the value has
    to be unmistakable -- every object token as a whole word, or a
    numeric object (a year, a count) as a whole word -- and the fact must
    then pass `fact_realized` outright, subject and relationship
    included. So inference can only ever ADD a citation to a break that
    already says the fact properly; it can never turn an accepted break
    into a refused one. Facts with no value to test (the catalog album
    fact) are never inferred. Order is the packet's; never raises."""
    out: list[str] = []
    try:
        low = _norm(expand_apostrophe_years(text or ""))
        if not low:
            return out
        have = set(cited or [])
        for f in packet.get("facts") or []:
            fid = f.get("id")
            if not fid or fid in have:
                continue
            policy = f.get("_realize") or {}
            obj = [_norm(str(o)) for o in (policy.get("object") or []) if o]
            obj = [o for o in obj if o]
            if not obj:
                continue
            numeric = [o for o in obj if o.isdigit()]
            said = (all(_whole(o, low) for o in obj)
                    or any(_whole(o, low) for o in numeric))
            if not said:
                continue
            if fact_realized(f, text) is not None:
                continue
            out.append(str(fid))
    except Exception:  # noqa: BLE001 - inference is a courtesy, never a break
        return []
    return out


def calendar_spoken(text: str, packet: dict[str, Any]) -> bool:
    """Did this text actually put the supplied event on air? The same
    anchor test a cited fact gets, and the ONLY thing that may spend the
    event's timing band."""
    cal = packet.get("calendar") or {}
    anchor = _norm(cal.get("title") or "")
    return bool(anchor) and anchor in _norm(text or "")


def build_packet(context: dict[str, Any] | None,
                 resolved: list[dict[str, Any]] | None,
                 recent: list[dict[str, Any]] | None,
                 local_now: datetime | None = None,
                 weather_sig: str | None = None) -> dict[str, Any]:
    """The Smart DJ context packet for one eligible boundary. Pure: every
    input is something the scheduler already resolved (the render context,
    the session's resolved tracks, the bounded recent-break memory, the
    station-local clock). Missing information is OMITTED, never spelled
    'unknown' -- a gap in the packet is not an invitation to fill it."""
    ctx = dict(context or {})
    packet: dict[str, Any] = {"v": 1}

    import settings
    st_name = None
    if isinstance(ctx.get("station"), dict):
        st_name = ctx["station"].get("name")
    if not st_name and ctx.get("station_name"):
        st_name = ctx["station_name"]
    if not st_name:
        st_name = settings.resolved_station_name()

    import break_packet
    # station moment ---------------------------------------------------------
    moment: dict[str, Any] = {}
    if local_now is not None:
        moment["date"] = local_now.date().isoformat()
        moment["weekday"] = local_now.strftime("%A")
    if ctx.get("station_time"):
        moment["time"] = _clip(ctx["station_time"], 16)
        # Whether the clock is DUE is decided upstream by
        # service_presence.apply, which REMOVES station_time when it is
        # not -- so `time` present already implies due. It is stated in
        # `required` below, not as a `time_due` flag: a model given the
        # flag reads "the clock is due" back to the listener.
        if ctx.get("station_time_forms"):
            moment["time_forms"] = [_clip(f, 24) for f in
                                    list(ctx["station_time_forms"])[:2]]
    if ctx.get("daypart"):
        moment["daypart"] = _clip(ctx["daypart"], 16)
    if ctx.get("daylight_line"):
        moment["daylight"] = _clip(ctx.get("daylight_phase") or "", 24)
        moment["daylight_line"] = _clip(ctx["daylight_line"], LINE_MAX)
    # Atmospheric colour (dj.atmosphere): licensed, OPTIONAL remarks about
    # the sky, the light or the day -- never the owed reading, never credited
    # as one (atmosphere.planner_colour; the engine screens the answer)
    colour = [str(c.get("say") or "") for c in (ctx.get("atmosphere_colour") or [])
              if isinstance(c, dict) and c.get("say")]
    if colour:
        moment["colour"] = [_clip(c, LINE_MAX) for c in colour[:2]]
    if ctx.get("current_weather"):
        moment["weather"] = _clip(ctx["current_weather"], LINE_MAX)
        cond = _condition_of(weather_sig, str(ctx["current_weather"]))
        if cond:
            moment["weather_condition"] = _clip(cond, 24)
        if ctx.get("weather_changed"):
            moment["weather_changed"] = True
        # Whether the WEATHER is due is decided upstream exactly like the
        # clock (service_presence.due), and the packet must say so: told
        # nothing, the model skips a due reading and loses the slot
        # (required_hosting.draft_shortfall) -- a good line replaced by a
        # template for want of one sentence it was never told to write.
        try:
            import service_presence
            if "weather" in service_presence.due(
                    ctx, int(ctx.get("max_words") or 0) or None):
                moment["weather_due"] = True
        except Exception:  # noqa: BLE001 - a flag, never a break
            pass
        if ctx.get("high_state"):
            moment["high_state"] = _clip(ctx["high_state"], 16)
            # WHAT THAT TOKEN MEANS, spelled out. With a bare
            # `high_state` and a weather line whose current reading equals
            # the day's high, a model can call it "cooler now than the
            # high" -- a TREND nothing supplied.
            # `near_now` was the worse case still waiting to happen: the
            # current reading IS the high, and a token nobody glossed
            # invites exactly the comparison it forbids. So the packet
            # states the relationship in words instead of leaving the
            # writer to infer one from a label (a RENDERING clarification
            # of a supplied fact; no rule changed).
            note = _HIGH_NOTE.get(str(ctx["high_state"]))
            if note:
                moment["high_note"] = note
    required: list[str] = []
    if moment.get("weather_due"):
        required.append("weather")
    if moment.get("time") and ctx.get("station_time_due"):
        required.append("time")
    if required:
        # the obligation stated INSIDE the packet too, ahead of the moment
        # it refers to: `required` is the first thing the model reads there
        packet["required"] = required
    if moment:
        packet["moment"] = moment
    packet["station"] = {"name": st_name}

    # persona / talk budget ----------------------------------------------------
    profile = dj_persona.get_profile(ctx.get("persona_profile"))
    # An explicit context level is the internal test seam and still wins;
    # otherwise the production level. The profile has no say in airtime.
    chattiness = ctx.get("chattiness")
    if chattiness not in dj_persona.CHATTINESS_LEVELS:
        chattiness = dj_persona.PRODUCTION_CHATTINESS
    max_words = int(ctx.get("max_words") or break_packet.MAX_WORDS)
    packet["persona"] = {
        "profile": profile["persona_id"],
        "chattiness": chattiness,
        "max_words": max_words,
    }
    # The editorial TARGET, resolved by dj_persona from chattiness and
    # break_length. A ceiling alone never produced a paragraph: "at most
    # 90 words" is satisfied by eight. The target is what the writer aims
    # for WHEN THE MATERIAL IS THERE, and the prompt says so in the same
    # breath, because a target read as a minimum is how filler happens.
    target = ctx.get("target_words")
    if (isinstance(target, (list, tuple)) and len(target) == 2
            and all(isinstance(t, int) for t in target)):
        lo, hi = int(target[0]), int(target[1])
        if 0 < lo <= hi:
            packet["persona"]["target_words"] = [min(lo, max_words),
                                                 min(hi, max_words)]

    # music window: previous / ending / next -- never more -----------------------
    music: list[dict[str, Any]] = []
    end_artist, end_title = _split_name(ctx.get("pre_name"))
    nxt_artist, nxt_title = _split_name(ctx.get("post_name"))
    # THE TRACK BEFORE LAST IS NOT AUTOMATICALLY MATERIAL. It was offered
    # to the writer purely because the session history had it, which is
    # the same "true, therefore say it" reflex the fact relevance layer
    # exists to stop -- and it led the writer to open breaks by naming
    # acts with no connection to the transition happening. A
    # `previous` entry now travels only when it is part of the RUN the
    # station is actually programming: the same artist still going. That
    # is a real continuity a host would mention; "we also played someone
    # else a while back" is not.
    prev = _previous_from(resolved, (end_artist, end_title))
    if prev and _norm(prev[0]) and _norm(prev[0]) in (_norm(end_artist),
                                                      _norm(nxt_artist)):
        music.append(_music_item("previous", prev[0],
                                 break_packet.spoken_title(prev[1])))
    if end_artist or end_title:
        music.append(_music_item(
            "ending", end_artist, end_title,
            album=ctx.get("ending_track_album") or "",
            character=ctx.get("ending_track_character")))
    if nxt_artist or nxt_title:
        music.append(_music_item(
            "next", nxt_artist, nxt_title,
            album=ctx.get("next_track_album") or "",
            character=ctx.get("next_track_character")))
    packet["music"] = music[:MAX_MUSIC_ITEMS]

    tc = ctx.get("transition_character")
    if tc in ("contrast", "major_contrast"):
        packet["transition"] = tc
    elif ctx.get("album_run"):
        packet["transition"] = ("album_run_ordered"
                                if ctx.get("album_sequence") == "ordered"
                                else "album_run")
    if ctx.get("semantic_themes"):
        feel: dict[str, Any] = {"themes": [_clip(t, 24) for t in
                                           list(ctx["semantic_themes"])[:3]]}
        if ctx.get("semantic_tone"):
            feel["tone"] = _clip(ctx["semantic_tone"], 24)
        packet["feel"] = feel

    # the listener's calendar -----------------------------------------------
    # Already qualified, bounded and pre-computed by
    # dj_context.calendar_packet: a title, a clock time, a band and a lead
    # PHRASE. Nothing is derived here and nothing is added -- in
    # particular no location, no attendees, no purpose, no travel time and
    # no preparation, because none of those are ever published and none of
    # them may be invented.
    cal_in = ctx.get("calendar_event")
    if isinstance(cal_in, dict) and cal_in.get("title"):
        cal: dict[str, Any] = {
            "id": _clip(cal_in.get("id"), 16),
            "title": _clip(cal_in["title"], NAME_MAX),
            "timing_band": _clip(cal_in.get("timing_band"), 8),
            "all_day": bool(cal_in.get("all_day")),
            "lead": _clip(cal_in.get("lead"), 40),
        }
        if cal_in.get("start_time"):
            cal["start_time"] = _clip(cal_in["start_time"], 16)
        if cal_in.get("minutes_until") is not None:
            cal["minutes_until"] = int(cal_in["minutes_until"])
        packet["calendar"] = cal

    # Calendar announcement: an operator's message riding this break.
    # REQUIRED and SUPPLIED: the writer must work all of it in, and its
    # words are grounded material like the calendar's (validate refuses a
    # draft that drops any of it; the floor then reads it verbatim).
    ann_in = ctx.get("announcement_required")
    if isinstance(ann_in, dict) and str(ann_in.get("message") or "").strip():
        packet["announcement"] = {
            "message": _clip(ann_in["message"], ANNOUNCEMENT_MAX)}
        # stated with the other obligations, FIRST: a rule placed last
        # among the instructions is the one the model drops (the planner
        # writes a plain transition without it, the same failure as a
        # weather obligation stated last)
        packet["required"] = ["announcement"] + list(
            packet.get("required") or [])

    # verified facts -----------------------------------------------------------
    # THE CATALOG ALBUM IS ORIENTATION, NOT A CITEABLE FACT. When each
    # side's album also travelled as an `album:` entry in `facts`, the
    # writer filed it under `fact_ids` whether or not it said the album --
    # and most subject refusals were exactly that, on copy that named
    # nothing unsupplied.
    # The album stays on the `music` entry above, where the claim scanner
    # already masks it and the writer may name it freely; `facts` holds
    # only material with something to realize.
    recent_rows = list(recent or [])[:MAX_RECENT]
    recent_topics = {str(r.get("topic")) for r in recent_rows if r.get("topic")}
    facts: list[dict[str, Any]] = []
    # Grounded library knowledge: already matched, scope-gated, ranked
    # and capped by the scheduler before it ever reaches here. This only
    # takes what it was handed and applies the planner's own short
    # memory. The durable fact cooldown was applied at retrieval;
    # `recent_topics` is the much shorter "not twice in the same handful
    # of breaks" window.
    said: set[str] = set()
    for kf in list(ctx.get("knowledge_facts") or [])[:MAX_KNOWLEDGE_FACTS]:
        if not isinstance(kf, dict) or not (kf.get("id") and kf.get("text")):
            continue
        if f"fact:{kf['id']}" in recent_topics:
            continue
        if _norm(str(kf["text"])) in said:
            continue
        entry = {"id": str(kf["id"]), "about": kf.get("about") or "ending",
                 "text": _clip(kf["text"], FACT_TEXT_MAX),
                 "_anchor": str(kf.get("anchor") or ""),
                 "_grounds": [str(g) for g in (kf.get("grounds") or [])],
                 "_realize": dict(kf.get("realize") or {}),
                 "_fact_id": str(kf.get("fact_id") or "")}
        if kf.get("kind") == "bridge":
            # A DERIVED RELATIONSHIP (fact_editorial): the writer is told
            # the connection and its values outright, never left to
            # notice it from two rows. Its evidence -- the pack fact_ids
            # it was built from -- travels privately for the novelty
            # ledger and the forensics.
            entry["relationship"] = str(kf.get("bridge_type") or "")
            entry["values"] = _public_values(kf.get("values"))
            entry["_evidence"] = [str(e) for e in (kf.get("evidence") or [])]
        said.add(_norm(entry["text"]))
        facts.append(entry)
    packet["facts"] = facts[:MAX_FACTS]
    # REFERENCE ROWS, hidden. The editorial layer held these back from
    # the writer (a bare release year is not a break), but a writer that
    # says "Northern Static, from 2009" from its own memory and is RIGHT has
    # said something the pack can vouch for. They ride the packet under
    # `_hidden` -- stripped before serialization, so the prompt never
    # carries them -- and only the validator's citation inference reads
    # them, holding them to the same subject/value/relation realization
    # as any cited fact. Never offered, never marked as aired.
    shown = {f["id"] for f in packet["facts"]}
    for rf in list(ctx.get("knowledge_reference") or [])[:MAX_REFERENCE_FACTS]:
        if not isinstance(rf, dict) or not (rf.get("id") and rf.get("text")):
            continue
        if rf["id"] in shown:
            continue
        packet["facts"].append({
            "id": str(rf["id"]), "about": rf.get("about") or "ending",
            "text": _clip(rf["text"], FACT_TEXT_MAX),
            "_anchor": str(rf.get("anchor") or ""),
            "_grounds": [str(g) for g in (rf.get("grounds") or [])],
            "_realize": dict(rf.get("realize") or {}),
            "_fact_id": str(rf.get("fact_id") or ""),
            "_hidden": True})

    # recent DJ memory ------------------------------------------------------------
    recent_out = []
    for r in recent_rows:
        entry: dict[str, Any] = {}
        if r.get("kind"):
            entry["kind"] = _clip(r["kind"], 24)
        if r.get("topic"):
            entry["topic"] = _clip(r["topic"], 48)
        said = scrub_recent_text(r.get("text") or "")
        if said:
            entry["text"] = _clip(said, RECENT_TEXT_MAX)
        if entry:
            recent_out.append(entry)
    packet["recent"] = recent_out

    # delivery ---------------------------------------------------------------
    owed = set(packet.get("required") or ())
    if packet.get("calendar"):
        owed.add("calendar")
    avoid = repeated_style(recent_rows, owed)
    if avoid:
        packet["avoid"] = avoid
    packet["editorial"] = editorial(packet)

    packet["candidates"] = candidates(packet, recent_topics, ctx)
    return fit(packet)


def _public_values(values: Any) -> dict[str, Any]:
    """A bridge's structured values, bounded for the packet: short
    strings, ints, and short lists of those. Nothing else travels."""
    out: dict[str, Any] = {}
    if not isinstance(values, dict):
        return out
    for k, v in list(values.items())[:6]:
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out[str(k)[:24]] = v
        elif isinstance(v, str):
            out[str(k)[:24]] = _clip(v, NAME_MAX)
        elif isinstance(v, (list, tuple)):
            out[str(k)[:24]] = [_clip(x, NAME_MAX) for x in list(v)[:4]
                                if isinstance(x, (str, int))]
    return out


def fit(packet: dict[str, Any]) -> dict[str, Any]:
    """Enforce PACKET_MAX_BYTES. The caps above make this a no-op in every
    realistic case; it exists so the bound is a property of the code, not
    of the fixtures. Sheds the least valuable material first: recent
    texts, then recent rows, then facts beyond the first, then the facts
    entirely -- never the music window or the moment."""
    if packet_bytes(packet) <= PACKET_MAX_BYTES:
        return packet
    for row in packet.get("recent") or []:
        row.pop("text", None)
    if packet_bytes(packet) <= PACKET_MAX_BYTES:
        return packet
    packet["recent"] = []
    if packet_bytes(packet) <= PACKET_MAX_BYTES:
        return packet
    # Delivery guidance is the next least valuable thing: a duller break
    # beats a truncated one, and an over-budget packet is already a
    # pathological case.
    packet.pop("avoid", None)
    while packet_bytes(packet) > PACKET_MAX_BYTES and len(
            packet.get("facts") or []) > 0:
        packet["facts"] = packet["facts"][:-1]
    if FACT in (packet.get("candidates") or []) and not visible_facts(packet):
        packet["candidates"] = [c for c in packet["candidates"] if c != FACT]
    if packet_bytes(packet) > PACKET_MAX_BYTES:
        logger.warning("smart dj packet still %d bytes after shedding",
                       packet_bytes(packet))
    return packet


# --- the prompt ---------------------------------------------------------------------

_CHATTINESS_LINE = {
    "low": ("Talk budget LOW: speak only for a genuinely good reason. "
            "say_nothing is the expected answer most of the time. When "
            "you do speak, make it ONE thought -- a single payload, plus "
            "a handoff if it helps. Do not add a second fact or a second "
            "observation because one happened to be there: two unrelated "
            "true things make a worse break than one."),
    "normal": ("say_nothing is a normal answer. Speak when the moment "
               "offers ONE real thing worth saying; otherwise stay out of "
               "the way of the music. Develop that one thing rather than "
               "stacking a second unrelated one beside it."),
    "high": ("This station is set CHATTY: when a break is worth making, "
             "it is worth making properly -- take the room the length "
             "guidance gives you and actually host. But nothing is owed: "
             "say_nothing remains a good answer, a thin moment "
             "still deserves one line or none, and repeating yourself is "
             "worse than silence."),
}


_KIND_GLOSS = {
    SAY_NOTHING: "nothing beats a clean cut to the music (text null).",
    TRANSITION: "hand the listener from one song to the next BY NAME -- "
                "either song, by title or by artist, and both together "
                "whenever that is the better line. One is the floor, not "
                "the limit. Name them and hand off; the move between "
                "them is not yours to describe.",
    QUIP: "a dry, playful line about music you name -- title, artist, or "
          "both; no facts.",
    OBSERVATION: "a remark on the supplied music -- name the song or the "
                 "artist it is about (or both), so the listener knows "
                 "what you mean.",
    FACT: "state ONE supplied fact from `facts` in your own words and cite "
          "its id.",
    WEATHER: "ONE useful thing about the supplied weather, in its own "
             "numbers and tense exactly as written. You are not reading a "
             "forecast board: take the reading that actually matters and "
             "leave the rest of the line alone.",
    DAYLIGHT: "a natural remark using the supplied daylight line, never "
              "recited as a bare sentence.",
    TEASE_NEXT: "point at the next song by its listed title or artist.",
    CALENDAR: "work the listener's listed event into what you were saying. "
              "Say its title; use `start_time` or `lead` for WHEN, exactly "
              "as written, and never work the timing out yourself.",
}


#: How a repeated structural habit is said back to the writer. Same
#: vocabulary as `style_flags`, phrased as the thing to stop doing.
_AVOID_GLOSS = {
    "looked_back_first": "opening by looking back at the song that just "
                         "ended",
    "signposted_the_next_song": "ending by announcing what is coming",
    "song_then_context_then_song": "the shape 'that was X, <remark>, next "
                                   "is Y' -- do not build that sentence "
                                   "again",
    "assistant_scaffolding": "openers like 'Here's', 'Currently' or 'The "
                             "weather is'",
    "asked_a_question": "asking the listener a question",
    "direct_address": "addressing the listener directly",
    "one_short_line": "a single short line -- give this one some shape",
    "opened_with_the_time": "opening with the time (put it later)",
    "hedged_aside": "hedges like 'I suppose', 'for sure', 'brisk'",
    "used_the_weather": "reaching for the weather",
    "used_the_calendar": "reaching for the calendar",
}

#: The delivery contract. This is the half of the prompt that decides
#: whether the DJ sounds like a person or like a status line, and it is
#: deliberately written as permissions rather than as a template: the
#: failure it exists to fix is not a wrong break, it is a run of correct
#: breaks with the same skeleton.
_DELIVERY_RULES = (
    "DELIVERY. You are a radio host, not an announcer reading fields. "
    "Sound like someone who likes this music and is talking to one person "
    "in a room.\n"
    "Vary the shape of the break every time: change where you start, sentence "
    "length, and how you get out. Fragments are fine. Enthusiasm is fine "
    "when real, dry when not. Break length is set by the talk budget below, "
    "not habit.\n"
    "You do NOT have to mention the song that ended, the song that is "
    "coming, or the weather -- unless a REQUIRED line below says this "
    "break owes one of them. Mention what the angle below actually needs "
    "and leave the rest alone. When something IS worth mentioning, weave "
    "it into a sentence rather than giving it a sentence of its own.\n"
    "Do not lean on 'Here's', 'Up next', 'Currently', 'The weather is' or "
    "'That was' as a way into a break. They are ordinary English and may "
    "be used when they are genuinely the best words; they may not be the "
    "furniture every break is built on.\n"
    # AMENDED. This clause used to read "Opinions about
    # how music FEELS -- its swagger, its gloom, what a song does to a
    # room -- are yours to hold and to say with conviction", which
    # directly authorized the thing the character rule below
    # forbids: asserting energy/mood/tempo/texture with no supplied
    # character line. A prompt that says both at once is a prompt the
    # model gets to choose from. VOICE stays free -- dry, warm,
    # late-night, wry -- and so does reacting to the radio MOMENT; what
    # narrowed is claiming how a SONG sounds.
    "Your VOICE is yours: dry or warm, wry or sincere, and the radio moment "
    "is yours to react to. How a song sounds is not voice: it is a fact, "
    "governed by the character rule below. Taste is not a fact, and is "
    "not a licence to state one.\n"
    "HOST FIRST, FLAIR SECOND. The listener cannot see your packet. They "
    "hear only the words, so the words have to say what the break is "
    "ABOUT.\n"
    "If you are talking about the music -- transition, quip, observation "
    "-- NAME the music you mean, from the packet: a title, an artist, or "
    "both, and name both songs when the line wants them. ONE name "
    "anywhere is the floor, not a quota, and everything around it is "
    "yours. "
    "A thought the music gave you is not a break unless you say which "
    "music: lines like \"well, that was quite a departure\" leave the "
    "listener with no idea what you heard. Choose say_nothing instead.\n"
    "If the break is about the listener's event, their weather, the time "
    "or a supplied fact, then SAY that thing -- the event by name, the "
    "actual reading, the fact itself -- and still name a song. Do not "
    "gesture at it. A topic you "
    "thought about but never said did not happen.\n"
    "WORTH SAYING. A break needs a reason to exist, and 'a song ended' is "
    "not one: personality decorates something worth saying, never the whole "
    "reason to talk. The reason is a supplied fact, something specific "
    "about the music, the listener's event, a real weather or clock "
    "reading, or a genuine handoff to what is next.\n"
    "Evaluation is seasoning, not payload. Phrases like 'quite a pairing' "
    "or 'a certain intensity' may sit beside something real; they may not "
    "BE the thing you said. Take the adjective out of your line: if nothing "
    "is left, say_nothing.\n"
    "FINISHING. Land the break on something real -- the next song, the "
    "reading, the fact -- rather than trailing off, and never announce "
    "that you are finishing or handing over. If you point forward AT "
    "ALL -- 'coming up', 'next', 'from "
    "here' -- you must say WHERE: the next artist, the next title, or both. "
    "Never gesture forward without naming the destination: an ungrounded "
    "tease is worse than no tease.\n"
)


#: Fingerprint pinning the prompt's four reference lines below, so a
#: change to them is caught.
E2_PROMPT_SHA256 = ("66ec4c0d543e1bf00463a052dfb81b5b200afe74b9"
                    "ca8a77af15489e73557de7")

#: THE DELTA, VERBATIM: these four lines are reproduced byte for byte --
#: not paraphrased, not merged, not "improved". The source prompt itself
#: is NOT adopted: it is a standalone free-text prompt with its own
#: persona, its own word ceiling and no JSON contract, and swapping it
#: in would take the packet, the kinds and the schema with it. Only the
#: four lines ship.
_E2_RULES = (
    "Next track: mention an upcoming song or artist ONLY when a "
    "next_track line appears in the facts below. If there is no "
    "next_track line, you do not know what plays next -- never guess it, "
    "never infer it from anything else, and never imply it with forward "
    "phrasing.\n"
    "The absence of a next_track line is itself information: it means the "
    "next song is unknown to you. Do not turn any other name in the "
    "facts, or any artist you happen to know of, into an upcoming "
    "track.\n"
    "With no next_track listed, host from what you do have -- the song "
    "that just ended, the weather, the calendar, or a supplied music "
    "fact. A break that ends without promising anything is a complete, "
    "professional break.\n"
    "How the music sounds is a FACT, not a flourish: describe energy, "
    "mood, tempo, texture, genre or how a song feels ONLY when a "
    "character line or transition line below supplies it. With no such "
    "line, you have not heard the song -- say nothing about how it "
    "sounds, and nothing about the artist's career, era or history. "
    "Naming the artist and the moment is enough.\n")

#: The ONE adapter line. The delta is written against a flat fact list
#: and names `next_track`, a `character line` and a `transition line`;
#: this packet carries all three under different names. Without this,
#: "there is no next_track line" reads as TRUE at every boundary that has
#: a next song -- which would suppress legitimate handoffs, the obvious
#: regression to fear. It adds no rule and relaxes
#: none; it only says where each governed thing lives.
_E2_PACKET_GLOSS = (
    "Reading those three rules against this packet: the `next_track` "
    "line is the `music` entry whose role is \"next\" -- if there is no "
    "such entry, there is no next_track. A `character line` is the "
    "`character` block on a music entry. The `transition line` is THE "
    "MOVE paragraph above, and only where it reports a real contrast.\n")


def render_prompt(packet: dict[str, Any]) -> str:
    """One prompt, one call, one JSON answer. Rules first, packet last."""
    persona = packet.get("persona") or {}
    profile = dj_persona.get_profile(persona.get("profile"))
    chattiness = persona.get("chattiness") or dj_persona.DEFAULT_CHATTINESS
    max_words = int(persona.get("max_words") or 60)
    target = persona.get("target_words") or []
    if len(target) == 2:
        length_line = (
            f"Length: aim for roughly {target[0]}-{target[1]} words when "
            f"the material is genuinely there; never more than "
            f"{max_words}. That is a TARGET, not a quota and not a "
            "minimum -- if this moment only has one good sentence in it, "
            "say that sentence and stop, and if it has nothing, say "
            "nothing. Padding a thin break to fill the room is worse "
            "than a short one. ")
    else:
        length_line = f"Length: at most {max_words} words. "
    kinds = list(packet.get("candidates") or [SAY_NOTHING])
    allowed = ", ".join(kinds)
    glossary = " ".join(f"{k}: {_KIND_GLOSS[k]}" for k in kinds
                        if k in _KIND_GLOSS)
    ed = packet.get("editorial") or {}
    angle_line = ""
    if ed.get("angle"):
        tone = ed.get("tone") or dj_persona.tone_for(dj_persona.get_profile(
            (packet.get("persona") or {}).get("profile")), None, False)
        angle_line = (f"ANGLE for this break: {ed['angle']}. Tone: "
                      f"{tone}. The angle is the "
                      "premise, not a line to read out.\n")
    avoid = [_AVOID_GLOSS.get(a, a) for a in (packet.get("avoid") or [])]
    avoid_line = ""
    if avoid:
        avoid_line = ("Your last few breaks have all been built the same "
                      "way. Do not do these again this time: "
                      + "; ".join(avoid) + ". Find a different way in and "
                      "a different way out.\n")
    # One line, only when a fact is actually on offer. Facts in this
    # packet are pre-written propositions with a SUBJECT, and the failure
    # to guard against is not invention but re-attachment: an award that
    # belongs to a band becoming this song's award, a producer who made an
    # album becoming the producer of the track that just ended.
    fact_line = ""
    if visible_facts(packet):
        fact_line = (
            "SUPPLIED FACTS: each `facts` entry is a finished, verified "
            "sentence. Reword it, fold it into a bigger thought, react to "
            "it -- but it stays about the subject it names. An album's "
            "fact is not the song's; an artist's is not this recording's. "
            "Never move a name, year or credit onto another subject, never "
            "drop a qualifier, never add what the fact does not say. Say "
            "that subject out loud or the listener is lost. Put every fact "
            "you use in `fact_ids`; "
            + ("two at the very most, and only if the line is truly better "
               "for the second."
               if chattiness == "high" else
               "ONE at most -- a break is a remark, not a quiz round.")
            + " Using none is always fine. A fact is a starting point: "
            "either say what makes it worth knowing, or state it plainly and stop.\n"
            "A RELEASE YEAR OR A RECORD LABEL ON ITS OWN IS NOT A BREAK. "
            "When the only thing you have is when something came out or "
            "who put it out, give the listener something else too -- hand "
            "off to what is next by name, or give them the time or weather if "
            "listed. An adjective is not a thought; with nothing to add, "
            "say nothing.\n")
    # A supplied RELATIONSHIP (fact_editorial bridge). The one thing no
    # amount of fact ranking could give:
    # when two facts connect, SAY THE CONNECTION. The entry carries the
    # relationship's name and its values; the writer's job is to make
    # the observation, not to read both source facts out.
    if any(f.get("relationship") for f in (packet.get("facts") or [])):
        fact_line += (
            "RELATIONSHIPS: a `facts` entry carrying a `relationship` is "
            "a documented connection -- between the two songs, or across "
            "one act's own timeline -- with its `values` spelled out. If "
            "it gives you a real observation, make THAT the line: say the "
            "connection in your own words, naming both things it joins "
            "and stating the number or name it rests on (the year, the "
            "count of years, the person), rather than reading two dates "
            "or two credits out one after the other. List its id in "
            "`fact_ids` as you would any fact; "
            "an id is never spoken. Never infer a connection that is not "
            "supplied, and if it adds nothing, leave it out.\n")
    # THE ROOT CAUSE OF `unauthorized_transition_relationship`. The guard
    # refuses invented relationships between the two songs; a prompt that
    # never mentions the class -- while the delivery contract invites it
    # ("whether a transition lands") and `transition` is offered as a kind
    # at every boundary -- makes it a leading rejection cause, always the
    # same shape: a line announcing that the mood moves from one song's
    # tone to the next one's, or that the turn is "a welcome change". The
    # model is doing as it is told.
    #
    # So the rule is stated where the model can act on it, in the same
    # register as the FACTS rule -- and it is stated in the AUTHORIZED
    # direction too, because a real contrast is exactly when this station
    # DOES want the turn made.
    transition = packet.get("transition")
    if transition in ("contrast", "major_contrast"):
        relation_line = (
            "THE MOVE: the packet says these two songs genuinely contrast, "
            "so the turn between them IS yours to make -- say it in your "
            "own words. Nothing else about how they relate is supplied.\n")
    elif transition:
        relation_line = (
            "THE MOVE: the packet says the same record is still running. "
            "You may say so plainly, and otherwise stay out of its way. Do "
            "not characterise how the two tracks compare.\n")
    else:
        relation_line = (
            "THE MOVE BETWEEN THE SONGS IS NOT YOURS TO DESCRIBE. Name "
            "either song, name both, hand off, and say what one of them is "
            "like on its own -- all fine. What you may not do is RELATE "
            "THE TWO TO EACH OTHER: how they compare, how one sounds "
            "beside the other, what the passage between them is like, or "
            "that one prepares, answers, suits or works against the next. "
            "You were told what played, not how it fits together. Say the "
            "songs; leave the join between them alone. That includes the "
            "handoff verb: do not call it a shift, a change, a pivot, a "
            "transition or a move -- 'next, X' or 'here's X' is the whole "
            "move.\n")
    # The weather RENDERING clarification (not a new rule): the packet
    # already carries the tense inside `moment.weather`, and now carries
    # `moment.high_note` saying what today's high is doing. Stating where
    # those live is the same job `_E2_PACKET_GLOSS` does for the three
    # rules it glosses -- without it the model reads a bare `high_state` token and
    # supplies a trend of its own.
    weather_line = ""
    _moment = packet.get("moment") or {}
    if _moment.get("weather"):
        weather_line = (
            "WEATHER, EXACTLY AS GIVEN: `moment.weather` already carries "
            "the numbers AND their tense -- whether the high is expected, "
            "happening or over, and whose low that is. Say it in your own "
            "words without changing either.")
        if _moment.get("high_note"):
            weather_line += (
                " `moment.high_note` states the ONLY relationship between "
                "the current temperature and today's high that you may "
                "put on air: " + str(_moment["high_note"]) + ".")
        weather_line += (
            " Add no trend of your own -- not cooler, warmer, dropping, "
            "climbing, 'a change from', 'finally' -- and no comparison "
            "the packet does not make. A reading is not a story.\n")
    colour_line = ""
    if _moment.get("colour"):
        colour_line = (
            "COLOUR, OPTIONAL: `moment.colour` lists things that are TRUE "
            "right now about the sky, the light or the day. If one fits, "
            "work it in once, in your own words -- a passing remark, a "
            "reaction or a little joke is welcome. Use only what it says: "
            "no other weather, sky, temperature or trend, and nothing the "
            "listener can see. Leaving it out is fine.\n")
    # THE OBLIGATION, STATED FIRST. A weather-due line placed last among
    # the rules is easily ignored: the DELIVERY block has already said the
    # weather is optional, and the kind glossary files it under a kind of
    # its own. Said right after the glossary, in the register of an
    # obligation rather than a hint, the reading is far more likely to
    # appear.
    required_line = ""
    _required = packet.get("required") or []
    if "announcement" in _required and packet.get("announcement"):
        required_line += (
            "STATION ANNOUNCEMENT, REQUIRED THIS BREAK, whatever kind you "
            "choose: `announcement.message` is a message from the people "
            "who run this station. Say ALL of it, naturally, in your own "
            "voice, keeping its key words and every time, number, name and "
            "'not' exactly as written (times in digits). Say it BEFORE you "
            "introduce the next song, never tacked on at the end. Add "
            "nothing to it -- no reason, place, person, advice or urgency "
            "it does not state. A break without the whole message does "
            "not air.\n")
    # (This wording gets the message carried in the planner's own words
    # more reliably than an explicit ORDER line. Every refused draft is
    # read verbatim by the floor instead.)
    if "weather" in _required:
        required_line += (
            "REQUIRED THIS BREAK, whatever kind you choose: the current "
            "weather reading, once, in your own "
            "words, with the numbers and tense of `moment.weather` exactly "
            "as written. A break without it does not air; it is never a "
            "repeat.\n")
    # THE CLOCK, AS AN OBLIGATION. This used to be "THE CLOCK
    # IS DUE ... an OPTION, not an order", and the accepted-draft check
    # agreed. Smart breaks that owed the time could air without it: saying
    # nothing, a vague "this morning", a clock that was not the supplied
    # one, or the prompt's own heading ("... and the time is due") read
    # back on air. The line now says what to do and never names the
    # requirement in words the model could repeat.
    if "time" in _required:
        required_line += (
            "STATION TIME, REQUIRED THIS BREAK: say the local time once, "
            "naturally, exactly as `moment.time` is written"
            + (" or as one of `moment.time_forms`"
               if _moment.get("time_forms") else "")
            + ". Never round, approximate or hedge it -- no 'about', "
            "'around', 'close to', 'just about', 'almost' -- and never a "
            "time you worked out yourself. A part of the day ('this "
            "morning', 'tonight') is not the time. Say it once only. It "
            "sits naturally next to a song name. Do not describe it as "
            "anything: never say the time or the clock is due, owed or "
            "required.\n")
        if "opened_with_the_time" in (packet.get("avoid") or []):
            # the recent breaks opened with the clock: an avoid-list entry
            # alone did not move the writer model, a concrete
            # place for THIS break does
            required_line += ("This break, say the time in your LAST "
                              "sentence, not your first.\n")
    calendar_line = ""
    if packet.get("calendar"):
        calendar_line = (
            "CALENDAR: `calendar` is one entry from the listener's own "
            "diary. You may say its title and, for when it is, ONLY the "
            "supplied `start_time` or `lead` wording -- never your own "
            "arithmetic, never a countdown you worked out, never a "
            "different unit. You know nothing else about it: not where it "
            "is, not who is going, not why, not how long it takes to get "
            "there, not what they should do first. Do not invent any of "
            "that, and do not tell the listener what to do about it. An "
            "all_day entry has no time at all and is not an appointment -- "
            "mention it lightly or not at all.\n")
    announcement_line = ""
    if packet.get("announcement"):
        # the obligation itself is in required_line, stated first; this
        # only keeps the rest of the break from stacking on top of it
        announcement_line = ("This break carries a station announcement: "
                             "keep everything else short, no music fact.\n")
    station_line = ""
    st = packet.get("station")
    st_name = st.get("name") if isinstance(st, dict) else packet.get("station_name")
    if st_name:
        import json
        json_station = json.dumps(st_name, ensure_ascii=False)
        station_line = (
            f"STATION NAME (literal data): {json_station}. "
            f"Use only this name for station IDs; an occasional ID by a "
            f"song is fine, no slogans.\n"
        )
    return (
        "You are the DJ on the listener's personal radio station, deciding "
        "whether to say a few seconds of words between two songs. Airtime "
        "is earned: an eligible break does not mean you must talk, and "
        "choosing silence is a good result.\n"
        "Answer with ONE JSON object and nothing else: "
        '{"action": <kind>, "text": <spoken words or null>, '
        '"fact_ids": [<ids of supplied facts you stated>]}.\n'
        f"Allowed kinds for this break: {allowed}. Any other kind is invalid. "
        + glossary + "\n"
        + required_line
        + length_line
        + station_line
        + "Plain spoken prose, no "
        "lists or markdown. Your words play when the ending song finishes: "
        "past tense for it, never now playing. Name no song, artist or "
        "album beyond the packet. A `previous` entry is background only: "
        "never say it played, never describe or compare it.\n"
        + _DELIVERY_RULES
        + _E2_RULES + _E2_PACKET_GLOSS
        + relation_line
        + angle_line + avoid_line + fact_line + weather_line + colour_line
        + calendar_line
        + announcement_line
        + "FACTS: the packet is your entire world. Your own knowledge of "
        "these "
        "artists and songs is NOT a source: never state or imply origin, "
        "hometown, members, dates, years, eras, albums, charts, sales, "
        "producers, studios, recording or release history, lyrics, meaning "
        "or genre unless a supplied fact or listed field says exactly that. "
        "If it is not in the packet, it does not exist. A joke may be "
        "invented; a fact may not. Never state a time, temperature or "
        "forecast that is not listed, never round or reinterpret a listed "
        "one, never say what this station usually plays.\n"
        "Do not reuse anything in `recent`: not its observation, setup, "
        "joke, topic or phrasing. Repeating yourself is worse than "
        "silence. The one exception is a REQUIRED service reading, which "
        "is owed regardless of what recent breaks said.\n"
        "Never explain yourself, instructions or your workings; never "
        "narrate what you are doing.\n"
        + dj_persona.delivery_block(profile, chattiness)
        + _CHATTINESS_LINE.get(chattiness, _CHATTINESS_LINE["normal"]) + "\n"
        "PACKET:\n" + serialize(packet).decode("utf-8") + "\n"
        "JSON:"
    )


# --- the model call -----------------------------------------------------------------

def request_options(max_words: int) -> dict[str, int]:
    """The runtime options every planner request carries: the context
    window this prompt is entitled to and a completion budget sized from
    the word budget (PLANNER_NUM_CTX and friends above)."""
    words = max(1, int(max_words or 60))
    return {"num_ctx": PLANNER_NUM_CTX,
            "num_predict": PLANNER_NUM_PREDICT_BASE
            + PLANNER_NUM_PREDICT_PER_WORD * words}


def call_model(prompt: str, max_words: int = 60) -> str:
    """The production planner call: the configured DJ writer model and
    endpoint (the same resolution the free-text writer uses), one
    non-streaming generate constrained to RESULT_SCHEMA, with the context
    window and completion budget stated explicitly (request_options).
    Raises on transport failure; the caller classifies (timeout /
    provider_error) and lands on SAY_NOTHING."""
    import providers
    from inference import llm_adapter
    adapter = llm_adapter.default_adapter()
    role = llm_adapter.ROLE_DJ_WRITER
    model = adapter.model_override() or adapter.model_for(role)
    url, key = adapter.endpoint()
    body = providers.ollama_generate(prompt, model,
                                     timeout=adapter.timeout_for(role),
                                     format=RESULT_SCHEMA,
                                     url=url or None, api_key=key or None,
                                     options=request_options(max_words))
    return str((body or {}).get("response") or "")


# --- P2 personality-first writer (opt-in: DJ_SMART_WRITER=p2) ---------------
#
# An experimental conversational writer (no optional facts, no stories).
# It replaces ONLY the planner's prompt and
# its one model call; everything after the call is the production contract,
# unchanged: the free text is filed as a TRANSITION decision and judged by
# `validate` (claims, clocks, self-reference, metadata, anchor, payload,
# dangling, guard), then by render_break's guard / required-hosting
# shortfall / completion, and a refusal of ANY kind lands on the
# deterministic floor exactly as a production refusal does.
#
# Three P2-only refusals exist because free text has failure shapes JSON
# never had: a reply cut off at the token cap (it would air mid-sentence),
# spoken stage directions / emotes / bracketed asides (the cleaner does not
# remove parentheses, so TTS would read them), and markdown structure.
# None of them edits the copy: an unsafe script is refused, never patched.

WRITER_P2 = "p2"
#: The P2 word budget (90 x 1.25): the prompt's "At most"
#: line, the cleaner's word cap, required-hosting completion and the token
#: budget (request_options(112) -> num_predict 288).
P2_MAX_WORDS = 112
#: The P2 sampling. The production JSON planner sends neither key.
P2_SAMPLING = {"temperature": 0.7, "top_p": 0.9}
#: P2 runs only on a writer model the operator declared for it. The product names
#: no model (test_private_defaults), so the operator declares it with
#: DJ_SMART_WRITER_P2_MODEL and dj_engine._planner_for compares it with the
#: configured writer model; unset or different keeps the production planner.
FAIL_TRUNCATED = "truncated_generation"
FAIL_STAGE = "spoken_stage_direction"
FAILURES = FAILURES + (FAIL_TRUNCATED, FAIL_STAGE)


def p2_model_ok(model: str | None, qualified: str | None) -> bool:
    """True when the configured DJ writer model is a model declared for
    P2. `qualified` is the operator's DJ_SMART_WRITER_P2_MODEL (one or more
    comma-separated tags); a configured tag matches one exactly or as a
    '-'-suffixed variant of it (a quantization spelling). Nothing declared
    -> False: P2 never runs on an undeclared model."""
    m = str(model or "").strip().lower()
    allowed = [q.strip().lower() for q in str(qualified or "").split(",")
               if q.strip()]
    return bool(m) and any(m == q or m.startswith(q + "-") for q in allowed)


def p2_configured_model() -> str:
    """The DJ writer model call_model would use (same resolution)."""
    from inference import llm_adapter
    adapter = llm_adapter.default_adapter()
    return adapter.model_override() or adapter.model_for(
        llm_adapter.ROLE_DJ_WRITER)


def p2_part_of_day(hour: int) -> str:
    """The P2 prompt's day-part words."""
    h = int(hour)
    return ("the small hours" if h < 5 else "early morning" if h < 9
            else "morning" if h < 12 else "afternoon" if h < 17
            else "evening" if h < 21 else "late night")


def render_p2_prompt(context: dict[str, Any], packet: dict[str, Any],
                     local_now: datetime | None,
                     max_words: int = P2_MAX_WORDS) -> str:
    """The P2 personality-first prompt (no facts). Its inputs are
    render-context keys (station_time, current_weather, calendar_event,
    pre_name, post_name, persona_profile) and the production obligation
    ledger (required_hosting.obligations), so what is OWED is decided by
    production, not by this prompt. It also carries the station's own
    name, handles an all-day calendar event (no start time) and takes
    recent copy from the packet's clock/temperature-scrubbed memory."""
    import required_hosting
    ctx = context or {}
    owed = set(required_hosting.obligations(ctx, max_words))
    prof = dj_persona.get_profile(ctx.get("persona_profile"))
    station = str(ctx.get("station_name") or "").strip() or "the station"
    now = local_now or datetime.now()
    items: list[tuple[str, str, bool]] = []
    if ctx.get("station_time"):
        items.append(("the station time", str(ctx["station_time"]),
                      required_hosting.STATION_TIME in owed))
    if ctx.get("current_weather"):
        items.append(("the current weather", str(ctx["current_weather"]),
                      required_hosting.WEATHER in owed))
    ev = ctx.get("calendar_event")
    if isinstance(ev, dict) and ev.get("title"):
        say = (f"{ev['title']} at {ev['start_time']}" if ev.get("start_time")
               else str(ev["title"]))
        what = ("the listener's calendar event"
                + (f" ({ev['lead']})" if ev.get("lead") else ""))
        items.append((what, say, required_hosting.CALENDAR in owed))
    ann = ctx.get("announcement_required")
    ann_msg = (str(ann.get("message") or "").strip()
               if isinstance(ann, dict) else "")
    if ann_msg:
        items.append(("a message from the people who run the station",
                      ann_msg, True))
    has_next = bool(ctx.get("post_name"))
    if has_next:
        a, _, t = str(ctx["post_name"]).partition(" - ")
        items.append(("the next song (artist with title)", f"{a} with {t}",
                      True))
    if ctx.get("pre_name"):
        a, _, t = str(ctx["pre_name"]).partition(" - ")
        items.append(("the song that just ended (title from artist)",
                      f"{t} from {a}", False))
    lines = [f"You are the host of {station}, a small personal radio "
             "station, talking between two songs.",
             f"Your personality: {prof.get('description')} "
             f"{' '.join(prof.get('delivery') or ())}",
             f"It is {now.strftime('%A')}, {p2_part_of_day(now.hour)}.",
             "Sound like a real person who enjoys this job: natural spoken "
             "rhythm, warmth, a little character, an occasional understated "
             "dry remark -- never a forced joke, never a readout.",
             "What you know (the ONLY facts you may state):"]
    for what, say, is_owed in items:
        lines.append(f"- {what}: {say}"
                     + (" (you must say this)" if is_owed else ""))
    if not has_next:
        lines.append("- the next song is NOT known: do not name or promise "
                     "one.")
    lines += [f"At most {max_words} words; use the room only when there is "
              "something to deliver -- with little to say, keep it short. "
              "Opinions, feelings and clearly figurative remarks are "
              "welcome. Never invent anything about the artists, places, "
              "the weather or daylight, the listener, the news, or your own "
              "past.",
              "Never open with 'That was' or 'You just heard'. Name each "
              "song at most once.",
              "Say at most ONE music fact, and only if it fits the moment "
              "-- never stack several. A typical break is 25 to 55 words: "
              "the owed items plus one natural thought."]
    if ann_msg:
        lines.append("Say the station's message in one plain sentence in "
                     "your own voice, keeping its words and every time, "
                     "number, name and 'not' exactly as written, in digits. "
                     "Your personality goes AROUND it, never into it: no "
                     "comment, reason, joke, advice or explanation about the "
                     "message itself -- you know nothing more about it than "
                     "its words. Say it before the next song, and introduce "
                     "the next song last. With a message to deliver, skip the "
                     "music fact and keep the whole break under 60 words.")
    recent = [str(r.get("text")) for r in (packet.get("recent") or [])
              if isinstance(r, dict) and r.get("text")]
    if recent:
        lines.append("Recently on air -- do not reuse their openings or "
                     "phrasing: " + " | ".join(recent[-3:]))
    lines.append("Write only the words you will say.")
    return "\n".join(lines)


def call_model_p2(prompt: str, max_words: int = P2_MAX_WORDS, *,
                  timeout_s: float | None = None) -> dict[str, Any]:
    """The P2 call: the configured DJ writer model and endpoint (the same
    resolution as call_model), free text (no format), the P2
    sampling and token budget. Returns the DECODED BODY so the caller can see
    `done_reason` (a reply cut off at num_predict must never air) and the
    timing counters for forensics. Raises on transport failure; the caller
    classifies it exactly like call_model."""
    import providers
    from inference import llm_adapter
    adapter = llm_adapter.default_adapter()
    role = llm_adapter.ROLE_DJ_WRITER
    model = adapter.model_override() or adapter.model_for(role)
    url, key = adapter.endpoint()
    timeout = adapter.timeout_for(role)
    if timeout_s is not None:
        # a retry may only SHORTEN the configured writer timeout (to what is
        # left before the boundary), never lengthen it
        timeout = max(1.0, min(float(timeout), float(timeout_s)))
    body = providers.ollama_generate(
        prompt, model, timeout=timeout, format=None,
        url=url or None, api_key=key or None,
        options={**request_options(max_words), **P2_SAMPLING})
    out = dict(body or {})
    out.setdefault("model", model)
    return out


#: Marks a planner callable as the P2 writer (tests mark their stubs too).
call_model_p2.writer_style = WRITER_P2  # type: ignore[attr-defined]
#: It takes a per-call `timeout_s` (the retry's remaining budget).
call_model_p2.accepts_timeout = True  # type: ignore[attr-defined]

#: Bracketed non-speech. Parentheses / square brackets are refused unless
#: the bracketed words are part of a SUPPLIED song or artist name ("(We
#: Only) Drive at Night (Mostly)"); an asterisk-wrapped span is
#: refused only when it is an emote / sound cue, because asterisk EMPHASIS
#: is already removed by the cleaner without touching a word.
_P2_PAREN_RE = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_P2_EMOTE_RE = re.compile(
    r"\*[^*\n]{0,60}\b(sound|sounds|music|pause|pauses|needle|vinyl|static|"
    r"fade|fades|fading|sigh|sighs|laugh|laughs|chuckle|chuckles|beat|hum|"
    r"hums|click|clicks|clears|cough|coughs|whisper|whispers|silence|"
    r"applause|crackle|crackles|scratch|record|sips|smiles|grins)\b"
    r"[^*\n]{0,60}\*", re.IGNORECASE)
_P2_STRUCTURE_RE = re.compile(
    r"(?m)^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|(?:dj|host|announcer|narrator)"
    r"\s*:)", re.IGNORECASE)


def _p2_name_brackets(packet: dict[str, Any], context: dict[str, Any] | None
                      ) -> list[str]:
    """Bracketed segments that belong to supplied names, lower-cased."""
    names: list[str] = []
    for m in packet.get("music") or []:
        names += [str(m.get("title") or ""), str(m.get("artist") or "")]
    for k in ("pre_name", "post_name", "station_name"):
        names.append(str((context or {}).get(k) or ""))
    out: list[str] = []
    for n in names:
        out += [b.lower() for b in _P2_PAREN_RE.findall(n)]
    return out


def p2_unsafe_markup(text: str, packet: dict[str, Any],
                     context: dict[str, Any] | None = None) -> str | None:
    """The first spoken stage direction, emote or markdown structure in a
    P2 draft, or None. Judged on the RAW reply (the cleaner would hide the
    evidence). A hit refuses the draft; nothing is deleted from it."""
    raw = str(text or "")
    allowed = set(_p2_name_brackets(packet, context))
    for m in _P2_PAREN_RE.finditer(raw):
        if m.group(0).lower() not in allowed:
            return m.group(0)[:80]
    m = _P2_EMOTE_RE.search(raw)
    if m:
        return m.group(0)[:80]
    m = _P2_STRUCTURE_RE.search(raw)
    if m:
        return m.group(0).strip()[:80]
    return None


# --- P2 content screen: unsupported music claims + calendar framing ---------
#
# The P2 prompt carries NO facts, so P2 has no evidence for any externally
# checkable claim about the music beyond the names it was given (and the
# albums the packet supplies). `ungrounded_claim` (above, production) closes
# years, credits, releases, charts, awards, line-ups and the like. Claims
# of these shapes can still pass every validator (fictional examples):
#
#   album     "That one first appeared on Northern Static" (it is on another
#             album)
#   person    "Ada Venn's vocals" (someone else sings it), "the drummer is
#             self-taught", "the bassist originally wanted to call the song
#             something else"
#   history   "took nearly two years to finish", "almost cut from the record"
#   origin    "They began as a collective in a harbour town"
#   instrument "There's a glockenspiel on this one", "his harpsichord"
#   lyric     "the chorus keeps returning to that lighthouse"
#
# Like `ungrounded_claim` this is a bounded vocabulary of claim SHAPES,
# scanned after every supplied string is masked; it does not and cannot
# judge truth -- a true claim with no evidence is refused exactly like a
# false one, and an invented claim phrased in none of these shapes still
# airs (the residual risk this design accepts). Metaphor,
# opinion, persona feelings and general musings carry none of the shapes;
# the two places they could (instrument words, "did you know") have
# explicit exemptions for similes, generic preferences and context-only
# questions. Called ONLY on the P2 path: production is unchanged.
FAIL_CALENDAR = "calendar_misattribution"
#: Calendar announcement: a draft that does not carry the whole riding
#: message, or carries it only after the next song's introduction.
FAIL_ANNOUNCEMENT = "announcement_not_delivered"
FAIL_ANNOUNCEMENT_PLACEMENT = "announcement_after_outro"
FAIL_MUSIC_CLAIM = "unsupported_music_claim"
FAILURES = FAILURES + (FAIL_CALENDAR, FAIL_MUSIC_CLAIM, FAIL_ANNOUNCEMENT,
                       FAIL_ANNOUNCEMENT_PLACEMENT)

_P2_DAYWORDS = (r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                r"morning|afternoon|evening|night|tonight|today")
_P2_HEARSAY_RE = re.compile(
    r"\b(?:did you know|fun fact|trivia|legend has it|rumou?r has it"
    r"|(?:as )?the story goes|i(?:'ve| have)? (?:been )?read(?:ing)?"
    r"(?: that| somewhere)|i read (?:that|somewhere)|i'?m told|i was told"
    r"|i(?:'ve| have) heard (?:that|it)|reportedly|supposedly"
    r"|they say (?:that|it)|word is)\b", re.IGNORECASE)
#: "Did you know it's already Friday?" asks about the moment, not the music.
_P2_HEARSAY_MOMENT_RE = re.compile(
    r"\bdid you know (?:that )?(?:it'?s|it is|today is)\s+(?:(?:already|only"
    r"|still)\s+)?(?:" + _P2_DAYWORDS + r"|\d|\u00a7)", re.IGNORECASE)
_P2_ALBUM_RES = (
    re.compile(r"\b(?:appear(?:ed|s)?|originally|outtakes?|b-?sides?|off of"
               r"|was (?:first |originally )?(?:on|released on))\b[^.!?§]{0,30}?"
               r"\b(?:on|from|of|off)\s+(?:the\s+)?(?:album\s+|record\s+)?"
               r"(?=[A-Z])"),
    re.compile(r"\b(?:on|from|off)\s+(?:the|their|his|her)\s+(?:\w+\s+)?"
               r"(?:album|record|lp|ep|compilation|soundtrack)\s+(?=[A-Z])",
               re.IGNORECASE),
    re.compile(r"\b(?:[Ii]t'?s|[Tt]hat'?s|[Tt]his is|[Tt]h(?:is|at) one'?s"
               r"|[Tt]hat (?:song|track|one) was|[Ii]t was)\s+(?:also "
               r"|originally )?from\s+(?=[A-Z])"),
)
_P2_ORIGIN_RE = re.compile(
    r"\b(?:started|began|formed|got (?:their|his|her) start|came together"
    r"|grew up|hail(?:s|ed)?|came out of|comes? out of|based|roots|raised"
    r"|born|cut (?:their|his|her) teeth)\b[^.!?§]{0,40}?\b(?:in|from|of"
    r"|out of)\s+(?=[A-Z][a-z])")
_P2_HISTORY_RE = re.compile(
    r"\b(?:original(?:ly)? (?:\w+ ){0,3}?(?:intended|wanted|planned|meant"
    r"|called|titled|written|conceived|released|recorded|used|demo)"
    r"|initially (?:intended|meant|planned|released|called|written|recorded)"
    r"|(?:almost|nearly) (?:left|cut|scrapped|dropped|didn'?t make|wasn'?t)"
    r"|left off|(?:wasn'?t|was not|never|weren'?t) (?:originally |initially )?"
    r"(?:intended|meant|supposed|planned)"
    r"|took (?:them |him |her |the band )?(?:nearly |almost |about |over "
    r"|more than |just |only )?(?:\w+|\d+) (?:years?|months?|weeks?|days?"
    r"|hours?|takes|tries|attempts) to (?:record|make|write|finish|complete"
    r"|nail)|in (?:a single|one) (?:take|afternoon|session|day|night"
    r"|evening|weekend)|one take|first take|self-taught|taught (?:himself"
    r"|herself|themselves)|learn(?:ed|t) to play|wanted to call"
    r"|working title|wrote (?:it|that|this|the song) (?:for|about|in|after"
    r"|while|on)|(?:was|were) (?:inspired|written|recorded|cut|tracked"
    r"|made) (?:by|in|at|on|during|for|after|while)|members? (?:playing"
    r"|recording) (?:together|simultaneously|live)|all (?:three|four|five"
    r"|six) members|used on|recordings? (?:were|was)|been (?:recording"
    r"|making records|releasing))\b", re.IGNORECASE)
_P2_INSTRUMENT_RE = re.compile(
    r"\b(?:pianos?|pianist|guitars?|bass ?lines?|bass guitar|bass|drums?"
    r"|drumming|drum machine|percussion|strings|string section|violins?"
    r"|violas?|cellos?|fiddles?|brass|horns?|horn section|trumpets?"
    r"|trombones?|saxophones?|sax|clarinets?|flutes?|harmonicas?|banjos?"
    r"|mandolins?|ukuleles?|accordions?|organs?|hammond|mellotrons?|synths?"
    r"|synthesi[sz]ers?|moog|theremin|celest[ae]s?|glockenspiels?"
    r"|vibraphones?|marimbas?|xylophones?|harps?|harpsichords?|choirs?"
    r"|orchestra|orchestral|vocals|harmonies|falsetto|arpeggios?|riffs?"
    r"|instrumental(?!\s+(?:in|to)\b))\b", re.IGNORECASE)
#: "it rolls in like a brass band": a simile names no instrument on the record.
_P2_SIMILE_BEFORE_RE = re.compile(
    r"\b(?:like|as)\s+(?:a|an|the|some)\s+(?:[\w'-]+\s+){0,2}$", re.IGNORECASE)
#: "I'm partial to a great, warm bassline": a taste, not the recording --
#: unless the sentence points at the record ("on that track", "their").
_P2_GENERIC_BEFORE_RE = re.compile(
    r"\b(?:a good|a great|a solid|a nice|a decent|a proper|some big|big|some"
    r"|more|less|any|plenty of|a lot of|lots of|enough|too much),?\s+"
    r"(?:[\w,'-]+\s+){0,2}$", re.IGNORECASE)
_P2_DEICTIC_RE = re.compile(
    r"\b(?:th(?:at|is|ese|ose) (?:track|song|one|record|tune|number|piece)"
    r"|on (?:it|there)|in (?:it|there)|their|his|her|its)\b|§['’]?s\b",
    re.IGNORECASE)
_P2_LYRIC_RE = re.compile(
    r"\b(?:lyrics?|lyrically|th(?:e|at) chorus|th(?:e|at) verse|the refrain"
    r"|th(?:e|at) line (?:about|where)|sings? about|sang about|singing about"
    r"|motif|hear(?:ing)? about)\b", re.IGNORECASE)
_P2_KEY_RE = re.compile(
    r"\b(?:(?:minor|major) key|key of [a-g]\b|in [a-g](?:[- ](?:flat|sharp))?"
    r" (?:minor|major)|time signature|\d+/\d+ time|waltz time|\d+ bpm"
    r"|beats per minute)\b", re.IGNORECASE)
_P2_DURATION_RE = re.compile(
    r"\b(?:within|under|in just|in barely|just over|just under|clocks? in at"
    r"|runs?|lasts?)\s+(?:about |around |nearly |almost |just )?(?:\w+|\d+)"
    r"(?:[- ]and[- ]a[- ]half)?\s+(?:minutes|seconds)\b", re.IGNORECASE)
#: Capitalized words that open sentences or are ordinary words, never a name.
_P2_NOT_NAME = set("""a an and the that this those these it its it's i i'm i've
i'd i'll we you your our they their he she his her him them so but or then now
next up coming here there well right just speaking did do does what some one
if when while for from with not no yes quite rather good lovely perhaps maybe
still also even only time hope thanks don't let's let back over out today
tonight tomorrow happy oh ah okay ok great welcome meanwhile anyway also sure
indeed really honestly actually sometimes speaking am pm fm monday tuesday
wednesday thursday friday saturday sunday january february march april may
june july august september october november december""".split())


def _p2_prompted(context: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """(names, other) exactly as render_p2_prompt hands them to the model:
    the artist and title halves of pre_name/post_name as RENDERED (qualifier
    and all -- "Slow Ferry (live at the Cinder Rooms, Port Avel - spring 2011)"), and the clock,
    weather, station and calendar strings. What the prompt told the writer
    is authoritative context even where the packet stores a cleaned title."""
    ctx = context or {}
    names: list[str] = []
    for key in ("pre_name", "post_name"):
        a, _, t = str(ctx.get(key) or "").partition(" - ")
        names += [a, t]
    other = [str(ctx.get(k) or "")
             for k in ("station_time", "current_weather", "station_name")]
    ev = ctx.get("calendar_event")
    if isinstance(ev, dict):
        other += [str(ev.get(k) or "") for k in ("title", "start_time", "lead")]
    ann = ctx.get("announcement_required")
    if isinstance(ann, dict) and ann.get("message"):
        other.append(str(ann["message"]))
    keep = lambda xs: [x.strip() for x in xs if len(x.strip()) >= 2]  # noqa: E731
    return keep(names), keep(other)


def _p2_masks(packet: dict[str, Any],
              context: dict[str, Any] | None = None) -> list[str]:
    """Every supplied string, plus the spoken forms of each song, artist and
    album (qualifier stripped, a leading 'The' dropped): the writer says
    "Quarry Light" and "Tin Lantern", not always the stored string. With
    the render context, the names and items the P2 prompt itself showed."""
    import writer_guard
    out = set(_supplied_strings(packet))
    names, other = _p2_prompted(context)
    out |= set(names) | set(other)
    values = [str(m[k]) for m in packet.get("music") or []
              for k in ("artist", "title", "album") if m.get(k)] + names
    for value in values:
        forms: set[str] = set()
        writer_guard._mask_forms(forms, value)
        for f in list(forms):
            if f.lower().startswith("the "):
                forms.add(f[4:])
        out |= forms
    return sorted((x.replace("’", "'") for x in out if len(x.strip()) >= 2),
                  key=len, reverse=True)


def _p2_masked(text: str, masks: list[str]) -> str:
    keep = (text or "").replace("’", "'").replace("‘", "'")
    for mv in masks:
        keep = re.sub(r"(?<!\w)" + re.escape(mv) + r"(?!\w)", " § ", keep,
                      flags=re.IGNORECASE)
    return keep


def _p2_unsupplied_name(sentence: str, masks_low: list[str]) -> str | None:
    """Two or more capitalized words in a row that no supplied string holds:
    a person ("Ada Venn"), another work ("Quarry Light"), a place."""
    toks = [m.group(0) for m in re.finditer(r"[A-Za-z][\w'-]*", sentence)]
    run: list[str] = []
    for word in toks + [""]:
        cap = bool(re.match(r"[A-Z][a-z]", word))
        if cap and word.lower() not in _P2_NOT_NAME:
            run.append(word)
            continue
        if len(run) >= 2:
            name = " ".join(re.sub(r"'s?$", "", w) for w in run)
            if not any(name.lower() in m for m in masks_low):
                return name
        run = []
    return None


def p2_unsupported_claim(text: str, packet: dict[str, Any],
                         context: dict[str, Any] | None = None) -> str | None:
    """'<class>:<words>' for the first unsupported music claim SHAPE in a
    P2 draft (album, person/name, history, origin, instrument, lyric, key,
    duration, hearsay), else None. Supplied strings are masked first."""
    masks = _p2_masks(packet, context)
    keep = _p2_masked(text, masks)
    if not keep.strip():
        return None
    # A name the packet carries only INSIDE a title's qualifier is not a
    # supplied artist: "Lost Bells (Ask Teodor Vance)" spoken whole is masked
    # above, but "Ask Teodor Vance just played" names it as the act.
    # The words inside a supplied title's own qualifier ARE supplied ("Tin
    # Lantern", "Night Crossing"); one used as the ACT ("Ask Teodor Vance
    # just played...") is the specifics lock's identity check.
    masks_low = [m.lower() for m in masks + _p2_qualifier_names(packet, context)
                 if not re.search(r"[(\[]", m)]
    for sentence in _SENTENCE_SPLIT_RE.split(keep.strip()):
        m = _P2_HEARSAY_RE.search(sentence)
        if m and not _P2_HEARSAY_MOMENT_RE.search(sentence):
            return f"hearsay:{m.group(0)[:30]}"
        for rx in _P2_ALBUM_RES:
            m = rx.search(sentence)
            if m:
                return f"album:{m.group(0).strip()[:30]}"
        m = _P2_ORIGIN_RE.search(sentence)
        if m:
            return f"origin:{m.group(0).strip()[:30]}"
        m = _P2_HISTORY_RE.search(sentence)
        if m:
            return f"history:{m.group(0)[:30]}"
        name = _p2_unsupplied_name(sentence, masks_low)
        if name:
            return f"name:{name[:30]}"
        for m in _P2_INSTRUMENT_RE.finditer(sentence):
            before = sentence[:m.start()]
            if _P2_SIMILE_BEFORE_RE.search(before):
                continue
            if (_P2_GENERIC_BEFORE_RE.search(before)
                    and not _P2_DEICTIC_RE.search(sentence)):
                continue
            return f"instrument:{m.group(0)[:30]}"
        for label, rx in (("lyric", _P2_LYRIC_RE), ("key", _P2_KEY_RE),
                          ("duration", _P2_DURATION_RE)):
            m = rx.search(sentence)
            if m:
                return f"{label}:{m.group(0).strip()[:30]}"
    return None


# --- specifics lock (the P2 writer) ------------------------------------------
#
# P2's own words may not introduce a SPECIFIC the break's authoritative
# context did not supply: a year, a number that quantifies something, or a
# capitalised name, title or place. A false claim usually recombines names
# into a false relation (the wrong album for a song, the wrong singer for
# a vocal, a place or a span nobody supplied) -- a specific nobody gave it.
# Needing no evidence at all, refusing unsupplied specifics catches most
# such claims and relation swaps while refusing few legitimate lines;
# evidence MATCHING would catch far fewer.
#
# This is a conservative specifics screen, NOT a truth verifier: it cannot
# judge a relation between supplied names, nor a claim with no specific in
# it ("Cold Orchard's early recordings"). It is no phrase list -- it reads only
# the shapes of specifics (digits, quantified number words, capitalised
# runs) and the packet. Authoritative context = every supplied string and
# its spoken forms, the words inside a supplied title's own qualifier
# ("Tin Lantern" in `Gullwing ("Tin Lantern" sketch)`), and the numbers in the
# supplied clock, weather, calendar and titles, in digits or words.
FAIL_SPECIFICS = "unsupplied_specifics"
FAILURES = FAILURES + (FAIL_SPECIFICS,)

_P2_NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_P2_NUM_WORDS.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                      "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90})
_P2_NUM_SCALE = {"hundred": 100, "thousand": 1000, "million": 10 ** 6,
                 "billion": 10 ** 9}
#: A spelled number QUANTIFIES something only when a unit follows it ("four
#: years", "seventy-four degrees"); "two in a row" and "one of those" do not.
_P2_UNIT_RE = re.compile(
    r"^(?:[\s-]+(?:and a half|or so|odd|plus))?[\s-]+(?:years?|months?"
    r"|weeks?|days?|hours?|minutes?|seconds?|decades?|albums?|records?"
    r"|singles?|songs?|tracks?|members?|times|degrees?|copies|awards?"
    r"|grammys?|nominations?|shows?|countries|percent|million|billion"
    r"|thousand|hundred)\b", re.IGNORECASE)
_P2_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday",
                "saturday", "sunday"}
#: A qualifier-only name used AS THE ACT ("Ask Teodor Vance just played...",
#: "...from Ask Teodor Vance"): an identity error, not a title.
_P2_ACT_AFTER_RE = re.compile(
    r"^\W{0,3}(?:just\s+)?(?:played|finished|wrapped|performed|sang|gave us"
    r"|brought us|is up next|with)\b", re.IGNORECASE)
_P2_ACT_BEFORE_RE = re.compile(r"\b(?:from|by)\W{1,3}$", re.IGNORECASE)
#: "...", “...”, *...*, _..._ and '...' (a single quote only at word edges,
#: so "rock 'n' roll" and contractions are not quotes).
_P2_QUOTED_RE = re.compile(
    r"\*{1,2}([^*\n]{1,60}?)\*{1,2}|[\"“]([^\"“”\n]{1,60})[\"”]"
    r"|(?<![\w])_([^_\n]{1,60})_(?![\w])|(?<![\w'])'([^'\n]{1,60})'(?![\w])")


def _p2_norm(s: str) -> str:
    s = (s or "").replace("’", "'").lower()
    s = re.sub(r"'s\b", "", s)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _p2_titles(packet: dict[str, Any],
               context: dict[str, Any] | None = None) -> list[str]:
    """Every supplied song title: the packet's and the prompt's rendered
    pre/post titles."""
    titles = [str(m.get("title") or "") for m in packet.get("music") or []]
    for key in ("pre_name", "post_name"):
        titles.append(str((context or {}).get(key) or "").partition(" - ")[2])
    return [t.replace("’", "'") for t in titles if t.strip()]


def _p2_qualifier_names(packet: dict[str, Any],
                        context: dict[str, Any] | None = None) -> list[str]:
    """The words a supplied TITLE carries inside its own qualifier or quotes:
    "Tin Lantern" in `Gullwing ("Tin Lantern" sketch)`, "Night Crossing"
    in `Slow Ferry. (Night Crossing)`, "Ask Teodor Vance" in
    `Lost Bells (Ask Teodor Vance)`."""
    out: set[str] = set()
    for title in _p2_titles(packet, context):
        for inner in re.findall(r"[(\[]([^)\]]+)[)\]]", title):
            out.add(inner.strip())
            out |= {q.strip() for q in re.findall(r"[\"“”]([^\"“”]+)[\"“”]", inner)}
        out |= {q.strip() for q in re.findall(r"[\"“”]([^\"“”]+)[\"“”]", title)}
    return sorted((x.replace("’", "'") for x in out if len(x) >= 2),
                  key=len, reverse=True)


def _p2_spelled_numbers(text: str) -> list[tuple[int, int]]:
    """(value, end offset) of each spelled number ("seventy-four", "twenty
    one", "a hundred")."""
    out: list[tuple[int, int]] = []
    toks = list(re.finditer(r"[A-Za-z]+", text))
    i = 0
    while i < len(toks):
        w = toks[i].group(0).lower()
        if w not in _P2_NUM_WORDS and w not in _P2_NUM_SCALE:
            i += 1
            continue
        value, end, j = 0, toks[i].end(), i
        while j < len(toks):
            w = toks[j].group(0).lower()
            gap = text[toks[j - 1].end():toks[j].start()] if j > i else ""
            if j > i and gap.strip(" -") != "":
                break
            if w in _P2_NUM_WORDS:
                value += _P2_NUM_WORDS[w]
            elif w in _P2_NUM_SCALE:
                value = max(value, 1) * _P2_NUM_SCALE[w]
            else:
                break
            end = toks[j].end()
            j += 1
        out.append((value, end))
        i = max(j, i + 1)
    return out


def _p2_supplied_numbers(packet: dict[str, Any],
                         context: dict[str, Any] | None = None) -> set[float]:
    nums: set[float] = set()
    names, other = _p2_prompted(context)
    for s in _supplied_strings(packet) + names + other:
        for d in re.findall(r"\d+(?:\.\d+)?", s):
            nums.add(float(d))
            for part in d.split("."):
                nums.add(float(part))
        for v, _ in _p2_spelled_numbers(s):
            nums.add(float(v))
    cal = packet.get("calendar") or {}
    if isinstance(cal.get("minutes_until"), (int, float)):
        nums.add(float(cal["minutes_until"]))
    return nums


def p2_unsupplied_specific(text: str, packet: dict[str, Any],
                           context: dict[str, Any] | None = None) -> str | None:
    """'<kind>:<words>' for the first specific in a P2 draft that the packet
    or the P2 prompt's render context did not supply (year / number / name /
    identity), else None."""
    qualifiers = _p2_qualifier_names(packet, context)
    raw = (text or "").replace("’", "'")
    # a qualifier-only name used as the act; the whole title spoken
    # ("Quarry Light (Reprise) just finished") is the title, not the act
    names, _ = _p2_prompted(context)
    artists = {_p2_norm(m.get("artist")) for m in packet.get("music") or []}
    artists |= {_p2_norm(x) for x in names}
    titled = _p2_masked(raw, sorted(_p2_titles(packet, context), key=len,
                                    reverse=True))
    for q in qualifiers:
        if _p2_norm(q) in artists:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(q) + r"(?!\w)", titled,
                             re.IGNORECASE):
            if (_P2_ACT_AFTER_RE.search(titled[m.end():m.end() + 30])
                    or _P2_ACT_BEFORE_RE.search(
                        titled[max(0, m.start() - 8):m.start()])):
                return f"identity:{q[:30]}"
    masks = sorted(set(_p2_masks(packet, context)) | set(qualifiers),
                   key=len, reverse=True)
    supplied = [_p2_norm(x) for x in masks]
    nums = _p2_supplied_numbers(packet, context)
    keep = expand_apostrophe_years(_p2_masked(raw, masks))
    # A quoted or italicised span is offered AS a title: one the context did
    # not supply is refused even in ordinary words ("the *Brass Weather* album").
    for m in _P2_QUOTED_RE.finditer(keep):
        inner = next(g for g in m.groups() if g is not None).strip(" ,.;:!?-")
        if (re.match(r"[A-Z0-9]", inner) and "§" not in inner
                and inner.lower() not in _P2_NOT_NAME
                and not any(_p2_norm(inner) in s for s in supplied)):
            return f"title:{inner[:30]}"
    for sentence in _SENTENCE_SPLIT_RE.split(keep.strip()):
        for y in re.findall(r"\b(?:1[89]\d\d|20\d\d)\b", sentence):
            if float(y) not in nums:
                return f"year:{y}"
        for d in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", sentence):
            if float(d) not in nums:
                return f"number:{d}"
        for value, end in _p2_spelled_numbers(sentence):
            if _P2_UNIT_RE.match(sentence[end:]) and float(value) not in nums:
                return f"number:{sentence[max(0, end - 20):end + 12].strip()[:30]}"
        toks = [t.group(0) for t in re.finditer(r"[A-Za-z][\w'-]*|§", sentence)]
        run: list[str] = []
        start_index = None
        for idx, tok in enumerate(toks + [""]):
            word = re.sub(r"'(?:s|ve|re|ll|d)$", "", tok)
            low = word.lower()
            plural_day = low.endswith("s") and low[:-1] in _P2_WEEKDAYS
            if ((re.match(r"[A-Z](?:[a-z]|-[A-Z])", word)
                 or re.fullmatch(r"[A-Z]{2,}", word))
                    and low not in _P2_NOT_NAME and not plural_day):
                if not run:
                    start_index = idx
                run.append(word)
                continue
            if run and low in ("of", "the", "&", "for") and tok != "":
                run.append(word)
                continue
            while run and run[-1].lower() in ("of", "the", "&", "for"):
                run.pop()
            if run and not (len(run) == 1 and start_index == 0):
                name = " ".join(run)
                if not any(_p2_norm(name) in s for s in supplied):
                    return f"name:{name[:30]}"
            run, start_index = [], None
    return None


def p2_content_screen(text: str, packet: dict[str, Any],
                      context: dict[str, Any] | None = None
                      ) -> tuple[str, str] | None:
    """(failure, detail) when a P2 draft that passed the production contract
    still frames the listener's calendar as anyone else's, makes an
    unsupported music claim, or introduces a specific the context did not
    supply; else None. The draft is refused, never edited, and the
    deterministic floor (which carries every owed item, the calendar
    included) takes the slot -- after the one content retry."""
    import writer_guard
    event = packet.get("calendar") or (context or {}).get("calendar_event")
    hit = writer_guard.calendar_misattribution(text, event,
                                               _supplied_strings(packet))
    if hit:
        return FAIL_CALENDAR, hit
    hit = p2_unsupported_claim(text, packet, context)
    if hit:
        return FAIL_MUSIC_CLAIM, hit
    hit = p2_unsupplied_specific(text, packet, context)
    if hit:
        return FAIL_SPECIFICS, hit
    return None


def _p2_generation(body: dict[str, Any], wall_ms: int) -> dict[str, Any]:
    """Generation evidence for forensics (bounded, no text)."""
    ns = 1_000_000
    return {"model": body.get("model"), "done_reason": body.get("done_reason"),
            "eval_count": body.get("eval_count"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "total_ms": (int(body["total_duration"] / ns)
                         if isinstance(body.get("total_duration"), (int, float))
                         else None),
            "wall_ms": wall_ms}


def classify_failure(exc: BaseException) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return FAIL_TIMEOUT
    if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
        return FAIL_TIMEOUT
    return FAIL_PROVIDER


# --- result parsing and validation ----------------------------------------------------

def parse_result(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """(result, None) for a well-formed answer, (None, failure) otherwise.
    Tolerates leading/trailing chatter around exactly one JSON object;
    tolerates nothing else."""
    if not isinstance(raw, str) or not raw.strip():
        return None, FAIL_INVALID
    text = raw.strip()
    data: Any = None
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None, FAIL_INVALID
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return None, FAIL_INVALID
    if not isinstance(data, dict):
        return None, FAIL_INVALID
    action = data.get("action")
    if not isinstance(action, str):
        return None, FAIL_INVALID
    action = action.strip().lower()
    if action not in KINDS:
        return None, FAIL_UNKNOWN_KIND
    spoken = data.get("text")
    if spoken is not None and not isinstance(spoken, str):
        return None, FAIL_INVALID
    ids = data.get("fact_ids")
    if ids is None:
        ids = []
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return None, FAIL_INVALID
    return {"action": action, "text": spoken, "fact_ids": ids}, None


def _raw_draft(result: dict[str, Any]) -> str:
    """The model's own words for the record, whitespace-collapsed and
    bounded. Forensics only: never aired, never cleaned for speech."""
    return re.sub(r"\s+", " ", str((result or {}).get("text") or "")
                  ).strip()[:600]


def _bounded_ids(ids: list[str]) -> list[str]:
    """At most six ids of at most 64 characters: a record, not a dump."""
    return [str(i)[:64] for i in list(ids or [])[:6]]


def moment_refs(result: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    """The `fact_ids` entries that name one of the packet's own MOMENT
    readings rather than a fact: `time`, `weather`, `daylight_line`,
    `moment`, `moment.time` and the like.

    The schema asks for "ids of supplied facts you stated", and on a break
    that offers no facts the writer files the reading it DID state under the
    field's own name. That was the dominant cause of
    `fact_reference_mismatch` refusals -- ordinary handoffs that said the
    supplied clock, refused and replaced by the floor. Such a
    label is not a citation, and `_validate` drops it: the words are then
    judged as any uncited break's are (clock, weather and daylight by their
    own rules). Only a reading the packet really carries counts, so an
    invented label is still a mismatch."""
    moment = packet.get("moment") or {}
    supplied = {f.get("id") for f in packet.get("facts") or []}
    out: list[str] = []
    for raw in result.get("fact_ids") or []:
        ref = str(raw).strip()
        if not ref or ref in supplied:
            continue
        key = ref.lower()
        if key.startswith("moment."):
            key = key[len("moment."):]
        if (key == "moment" and moment) or key in moment:
            out.append(ref)
    return out


def _silence(reason: str | None, **extra: Any) -> dict[str, Any]:
    out = {"action": SAY_NOTHING, "text": None, "fact_ids": [],
           "topic_key": None, "reason": reason}
    out.update(extra)
    return out


def announcement_failure(text: str, packet: dict[str, Any]
                         ) -> tuple[str, str] | None:
    """(failure, detail) when an accepted draft does not deliver the riding
    announcement whole (announce_copy.delivery_problem), or delivers it
    only after it has introduced the next song -- the mechanical
    "...up next, X. And the library closes at nine." shape. None when it is fine."""
    import announce_copy
    msg = str((packet.get("announcement") or {}).get("message") or "")
    if not msg:
        return None
    problem = announce_copy.delivery_problem(msg, text)
    if problem:
        return FAIL_ANNOUNCEMENT, problem
    nxt = next((m for m in packet.get("music") or []
                if m.get("role") == "next"), None)
    span = announce_copy.message_span(msg, text)
    if nxt and span:
        low = (text or "").lower()
        lasts = [low.rfind(str(x).lower()) for x in (nxt.get("artist"),
                                                     nxt.get("title"))
                 if x and len(_norm(x)) >= ANCHOR_NAME_MIN]
        lasts = [i for i in lasts if i >= 0]
        # the next song is introduced LAST: a naming of it after the
        # message ends; named only before it, the message is a tag-on
        if lasts and max(lasts) < span[1]:
            return FAIL_ANNOUNCEMENT_PLACEMENT, "message_after_next_song"
    return None


def validate(result: dict[str, Any], packet: dict[str, Any], *,
             clean_fn: Callable[[str], str],
             guard_fn: Callable[[str], str | None] | None = None,
             spoken_fn: Callable[[str], set[str]] | None = None,
             warn_fn: Callable[[str], list[str]] | None = None
             ) -> dict[str, Any]:
    """The verdict, plus the warn-only observations on WHATEVER was said.

    The accept/reject decision is `_validate` below and is untouched by
    this wrapper. What the wrapper adds is telemetry on the REFUSED case:
    `_validate` returns at the first failing rule, so until now a draft
    that was rejected for, say, an ungrounded claim carried no record of
    the other things it also did -- spoken markup, an unsupported forward
    implication, an unsupported character claim. Forensic analysis
    needs all three of "what the model attempted", "why it was rejected"
    and "what else it exhibited"; the third was the missing one.

    OBSERVATION ONLY. Nothing computed here can change an outcome: the
    verdict is already decided when this runs, and a telemetry fault is
    swallowed.
    """
    out = _validate(result, packet, clean_fn=clean_fn, guard_fn=guard_fn,
                    spoken_fn=spoken_fn, warn_fn=warn_fn)
    if out.get("action") != SAY_NOTHING and packet.get("announcement"):
        failed = announcement_failure(out.get("text") or "", packet)
        if failed:
            out = _silence(failed[0], attempted=out.get("action"),
                           draft=out.get("text"), detail=failed[1])
    if out.get("action") == SAY_NOTHING:
        # The cleaned draft when the rule that fired captured one,
        # otherwise the raw answer -- a draft refused before cleaning
        # (empty text, an unknown kind) still has words worth measuring.
        draft = out.get("draft") or result.get("text") or ""
        if not out.get("draft") and _raw_draft(result):
            # every refusal keeps the words it refused (forensics only)
            out["draft"] = _raw_draft(result)
        warnings = list(out.get("warnings") or [])
        markup = out.get("markup")
        if markup is None:
            markup = spoken_markup(result.get("text") or "")
        if markup and not any(str(w).startswith(WARN_MARKUP)
                              for w in warnings):
            warnings.append(f"{WARN_MARKUP}:{markup}")
        if draft and warn_fn is not None:
            try:
                warnings.extend(warn_fn(draft) or [])
            except Exception:  # noqa: BLE001 - telemetry never costs a break
                pass
        out["warnings"] = warnings
        out["markup"] = markup
    # which readings the writer filed as facts (dropped; forensics counts)
    refs = moment_refs(result, packet)
    if refs:
        out["moment_refs"] = _bounded_ids(refs)
    return out


def _validate(result: dict[str, Any], packet: dict[str, Any], *,
              clean_fn: Callable[[str], str],
              guard_fn: Callable[[str], str | None] | None = None,
              spoken_fn: Callable[[str], set[str]] | None = None,
              warn_fn: Callable[[str], list[str]] | None = None
              ) -> dict[str, Any]:
    """Turn a parsed answer into a Decision, or into SAY_NOTHING with a
    failure category. `clean_fn` is the engine's spoken-text cleaner (word
    budget applied), `guard_fn` the writer authority guard bound to this
    break's flags (None -> no violation), `spoken_fn` the engine's
    supplied-service detector (which of weather/daylight the text really
    stated, judged against the offered values)."""
    action = result["action"]
    allowed = packet.get("candidates") or [SAY_NOTHING]
    if action == SAY_NOTHING:
        return _silence(REASON_MODEL)
    # A cited fact the packet never supplied is a fact-reference mismatch
    # whatever the kind, and it is judged first: it is the most precise
    # thing this answer got wrong.
    supplied = {f["id"]: f for f in (packet.get("facts") or [])}
    # a supplied reading named as if it were a fact is no citation at all
    readings = set(moment_refs(result, packet))
    ids = [i.strip() for i in result.get("fact_ids") or []
           if i.strip() and i.strip() not in readings]
    if any(i not in supplied for i in ids):
        # THE EVIDENCE, KEPT. A refusal here used to record nothing -- no
        # draft, no ids -- and on a packet that offered NO facts, what the
        # model thought it was citing could not be known.
        # The ids it invented (bounded) and its
        # words travel with the refusal; nothing about the verdict changes.
        return _silence(FAIL_FACT_REF, attempted=action,
                        draft=_raw_draft(result),
                        unknown_fact_ids=_bounded_ids(
                            [i for i in ids if i not in supplied]),
                        offered_fact_count=len(visible_facts(packet)))
    if action == FACT and not ids:
        return _silence(FAIL_FACT_REF, attempted=action,
                        draft=_raw_draft(result), unknown_fact_ids=[],
                        offered_fact_count=len(visible_facts(packet)))
    if action not in allowed:
        return _silence(FAIL_NOT_CANDIDATE, attempted=action)
    try:
        text = clean_fn(result.get("text") or "")
    except Exception:  # noqa: BLE001 - a cleaner fault is a silent break
        text = ""
    if not text:
        return _silence(FAIL_EMPTY, attempted=action)
    # A SUPPLIED FACT THE WRITER SAID BUT FORGOT TO CITE. A line like
    # "Wren Hollis is coming up with Quarry Light, a 2025 release" would
    # die as `ungrounded_claim year:2025` -- the recent-release fact
    # was in the packet, its subject and its
    # value were both audible, and the only thing missing was the id in
    # `fact_ids`. The citation is a bookkeeping act, not a truth
    # condition, so it is inferred here for a fact whose subject AND
    # value the copy already states, and the inferred fact is then held
    # to exactly the realization every cited fact gets. Nothing an
    # unsupplied claim could not already pass is loosened: a fact whose
    # subject is not said is not inferred and its words still trip the
    # claim scanner; an inferred fact that fails RELATION is refused as
    # before. Recorded as `inferred_fact_ids` so the record can tell the
    # two apart.
    inferred = infer_citations(text, packet, ids)
    if inferred:
        ids = ids + inferred
    # Judged BEFORE everything else it could be mistaken for: a line
    # carrying a fact id is not a self-reference and not an ungrounded
    # claim, and calling it either would send the next reader looking in
    # the wrong place. The cited ids ride along so the structured record
    # still knows which facts were in play -- `fact_ids` itself stays
    # empty, because nothing aired and nothing may be marked as used.
    leak = internal_metadata(text, packet)
    if leak:
        return _silence(FAIL_METADATA, attempted=action, draft=text,
                        leak=leak,
                        cited_fact_ids=[supplied[i].get("_fact_id") or i
                                        for i in ids])
    if _SELF_REF_RE.search(text) or _META_RE.search(text):
        return _silence(FAIL_SELF_REF, attempted=action, draft=text)
    claim = ungrounded_claim(text, packet, ids)
    if claim:
        return _silence(FAIL_CLAIM, attempted=action, claim=claim, draft=text)
    span = elapsed_span_problem(text, packet, ids)
    if span:
        return _silence(FAIL_CLAIM, attempted=action, claim=span, draft=text)
    clock = unsupplied_clock(text, packet)
    if clock:
        return _silence(FAIL_UNSUPPLIED_CLOCK, attempted=action, draft=text,
                        clock=clock)
    low = _norm(text)
    music = packet.get("music") or []
    # EVERY cited fact is judged, whatever kind this answer was filed
    # under. `fact` additionally has to cite something at all (above);
    # the realization itself is the same contract for all of them.
    for fid in ids:
        bad = fact_realized(supplied[fid], text)
        if bad:
            return _silence(bad, attempted=action, draft=text,
                            fact_id=supplied[fid].get("_fact_id") or fid,
                            predicate=(supplied[fid].get("_realize")
                                       or {}).get("predicate"))
    if action == CALENDAR and not calendar_spoken(text, packet):  # noqa: E501
        # An answer filed as `calendar` that never put the event on air is
        # not a calendar break -- and letting it through would spend the
        # event's timing band on a line that did not use it.
        return _silence(FAIL_UNGROUNDED, attempted=action, draft=text)
    if action == TEASE_NEXT:
        nxt = next((m for m in music if m.get("role") == "next"), None) or {}
        names = [_norm(nxt.get(k) or "") for k in ("title", "artist")]
        if not any(n and n in low for n in names):
            return _silence(FAIL_UNGROUNDED, attempted=action, draft=text)
    # Which supplied service readings the listener will actually hear.
    # Resolved for EVERY kind now, not just the two that are named after
    # one: the hosting rule below counts a real weather or clock reading
    # woven into a transition exactly as it counts one filed as `weather`.
    heard: set[str] = set()
    if spoken_fn is not None:
        try:
            heard = set(spoken_fn(text))
        except Exception:  # noqa: BLE001
            heard = set()
    if action in (WEATHER, DAYLIGHT) and action not in heard:
        return _silence(FAIL_UNGROUNDED, attempted=action, draft=text)
    # HOST FIRST, FLAIR SECOND. Personality may decorate an authorized
    # payload; it may not BE the payload.
    anchor = hosting_anchor(action, text, packet, ids, heard)
    if anchor is None:
        return _silence(FAIL_NO_ANCHOR, attempted=action, draft=text)
    # A tease that points into empty space. Judged before the payload
    # rule because it is the more precise complaint: the line may be
    # perfectly substantial and still leave the loop open.
    if dangling_forward(action, text, packet):
        return _silence(FAIL_DANGLING, attempted=action, draft=text)
    # HOST FIRST, FLAIR SECOND, and flair is not a reason to talk.
    payload = hosting_payload(action, text, packet, ids, heard)
    if payload is None:
        return _silence(FAIL_NO_PAYLOAD, attempted=action, draft=text)
    # A catalogue date or label is a fact, not a break. When that is ALL
    # the break cites, it has to carry one real station element besides --
    # a named handoff, a due service reading, or the listener's event.
    # Evaluation does not count, which is why this is decided on the
    # fact's predicate and the station signals, never on the words.
    if thin_fact_only(packet, ids):
        completed = (_named_next(text, packet)
                     or bool(heard & {WEATHER, DAYLIGHT, "station_time"}
                             if heard else False)
                     or calendar_spoken(text, packet))
        if not completed:
            return _silence(FAIL_THIN_FACT, attempted=action, draft=text,
                            cited_fact_ids=[supplied[i].get("_fact_id") or i
                                            for i in ids])
    if guard_fn is not None:
        try:
            category = guard_fn(text)
        except Exception:  # noqa: BLE001 - guard fault: fail toward silence
            category = "guard_error"
        if category:
            return _silence(FAIL_GUARD, attempted=action, draft=text,
                            guard_category=category)
    # FORWARD-IMPLICATION AND CHARACTER TELEMETRY: OBSERVED, NEVER
    # ENFORCED. These run only once the break has already been ACCEPTED,
    # and nothing below can change that outcome -- they exist so an
    # unsupported forward implication, or an unsupported character claim,
    # is countable in forensics without the guard becoming a style cop.
    # The forward check is lexicon-bound and the character class carries
    # an irreducible false-positive rate; neither is fit to refuse copy
    # on. Fenced: a telemetry fault never costs a break.
    warnings: list[str] = []
    # Spoken markup rides the same telemetry channel. Judged on the RAW
    # answer because the cleaner has already removed the evidence from
    # `text`, and recorded whether or not anything else went wrong.
    markup = spoken_markup(result.get("text") or "")
    if markup:
        warnings.append(f"{WARN_MARKUP}:{markup}")
    if warn_fn is not None:
        try:
            warnings.extend(warn_fn(text) or [])
        except Exception:  # noqa: BLE001
            pass
    return {"action": action, "text": text, "fact_ids": ids,
            "inferred_fact_ids": inferred,
            "warnings": warnings, "markup": markup,
            "topic_key": topic_key(action, packet, ids), "reason": None,
            "anchor": anchor, "payload": payload}


def decide(packet: dict[str, Any], model_fn: Callable[[str], str], *,
           clean_fn: Callable[[str], str],
           guard_fn: Callable[[str], str | None] | None = None,
           spoken_fn: Callable[[str], set[str]] | None = None,
           warn_fn: Callable[[str], list[str]] | None = None,
           p2_prompt: str | None = None,
           p2_context: dict[str, Any] | None = None,
           p2_retry_budget_s: float | None = None,
           p2_clock: Callable[[], float] | None = None
           ) -> dict[str, Any]:
    """The whole planner step for one boundary: nothing to talk about ->
    silence without a call; otherwise ONE model call, parsed and validated.
    Never raises; every failure is a SAY_NOTHING with its category.

    A planner marked `writer_style == "p2"` (call_model_p2, or a test stub)
    takes the P2 branch: `p2_prompt` is the rendered P2 prompt, the reply is
    free text, and it is judged by the same `validate` filed as TRANSITION.
    `p2_retry_budget_s` is the time this decision may still spend before the
    boundary (see _decide_p2's retry); None means no retry. Any other
    planner is the production path, unchanged."""
    roles = {m.get("role") for m in (packet.get("music") or [])}
    if not (roles & {"ending", "next"}):
        return _silence(REASON_NO_MATERIAL)
    if getattr(model_fn, "writer_style", None) == WRITER_P2:
        return _decide_p2(packet, model_fn, p2_prompt, p2_context,
                          clean_fn=clean_fn, guard_fn=guard_fn,
                          spoken_fn=spoken_fn, warn_fn=warn_fn,
                          retry_budget_s=p2_retry_budget_s, clock=p2_clock)
    prompt = render_prompt(packet)
    try:
        if model_fn is call_model:
            raw = model_fn(prompt, (packet.get("persona") or {}).get(
                "max_words") or 60)
        else:
            raw = model_fn(prompt)
    except Exception as exc:  # noqa: BLE001 - the provider boundary
        logger.warning("smart dj planner call failed (%s: %s)",
                       type(exc).__name__, str(exc)[:160])
        return _silence(classify_failure(exc))
    result, failure = parse_result(raw)
    if result is None:
        return _silence(failure)
    # `validate` records the warn-only observations on a refused draft as
    # well as an accepted one (its wrapper), so nothing is patched here.
    return validate(result, packet, clean_fn=clean_fn, guard_fn=guard_fn,
                    warn_fn=warn_fn, spoken_fn=spoken_fn)


#: ONE RETRY, CONTENT REFUSALS ONLY. A P2 draft
#: the contract refuses for what it SAYS gets at most one more fresh draw of
#: the identical prompt (no seed is sent, so Ollama samples anew); the second
#: draft is judged by exactly the same contract and screen, independently.
#: A content refusal is often a property of one draw, not of the prompt,
#: so one fresh re-sample recovers a share of the refused drafts.
#: Service and transport failures (timeout, provider
#: error), a missing prompt and everything after the decision (render, TTS,
#: queue, lifecycle) are never retried.
P2_RETRY_REASONS = frozenset({
    FAIL_CLAIM, FAIL_UNGROUNDED, FAIL_GUARD, FAIL_SELF_REF, FAIL_NO_ANCHOR,
    FAIL_NO_PAYLOAD, FAIL_DANGLING, FAIL_THIN_FACT, FAIL_FACT_SUBJECT,
    FAIL_FACT_VALUE, FAIL_FACT_RELATION, FAIL_FACT_REF, FAIL_METADATA,
    FAIL_UNSUPPLIED_CLOCK, FAIL_EMPTY, FAIL_TRUNCATED, FAIL_STAGE,
    FAIL_CALENDAR, FAIL_MUSIC_CLAIM, FAIL_SPECIFICS, FAIL_ANNOUNCEMENT,
    FAIL_ANNOUNCEMENT_PLACEMENT})
#: Calendar on the P2 path. When a break owes the listener's calendar entry, P2
#: does not write it at all: no generation, no retry. Free text hands the
#: listener's entry to the audience or the host in ever-new phrasings
#: (addressing it to "everyone out there", or claiming it as the host's
#: own plans) faster than any guard could list them, and a second draw is
#: a second chance at a new one.
#: The deterministic floor already says it in the
#: listener's own framing ("On your calendar, {title} at {start}") together
#: with every other owed item, so this decline hands it the slot
#: (required_hosting.ENRICHMENT_DECLINES). Same "owed" rule as
#: required_hosting.obligations: an offered event with a title.
FAIL_P2_CALENDAR = "p2_calendar_deterministic"
FAILURES = FAILURES + (FAIL_P2_CALENDAR,)


def p2_calendar_owed(packet: dict[str, Any],
                     context: dict[str, Any] | None) -> bool:
    event = (context or {}).get("calendar_event") or packet.get("calendar")
    return isinstance(event, dict) and bool(event.get("title"))


#: The retry runs only when the time left in the budget covers this estimate
#: of one more generation: 1.5x the first attempt's own wall time, never
#: below 6 s (just above a typical P2 generation time).
P2_RETRY_EST_FACTOR = 1.5
P2_RETRY_EST_MIN_S = 6.0


def _p2_attempt(packet: dict[str, Any], model_fn, prompt: str,
                context: dict[str, Any] | None, *, clean_fn, guard_fn,
                spoken_fn, warn_fn, timeout_s: float | None = None
                ) -> dict[str, Any]:
    """ONE P2 attempt: one free-text call, then the production contract and
    the P2 content screen. Every outcome carries `writer_style` and, when the
    model was reached, `generation` (done_reason, token counts, timing)."""
    import time as _time
    style = {"writer_style": WRITER_P2}
    t0 = _time.monotonic()
    try:
        if timeout_s is not None and getattr(model_fn, "accepts_timeout", False):
            body = model_fn(prompt, P2_MAX_WORDS, timeout_s=timeout_s)
        else:
            body = model_fn(prompt, P2_MAX_WORDS)
    except Exception as exc:  # noqa: BLE001 - the provider boundary
        logger.warning("smart dj p2 writer call failed (%s: %s)",
                       type(exc).__name__, str(exc)[:160])
        return _silence(classify_failure(exc), **style,
                        generation={"wall_ms": int(
                            (_time.monotonic() - t0) * 1000)})
    if not isinstance(body, dict):
        body = {"response": str(body or "")}
    gen = _p2_generation(body, int((_time.monotonic() - t0) * 1000))
    raw = str(body.get("response") or "").strip().strip('"')
    if body.get("done_reason") == "length":
        # cut off at num_predict: free text would air mid-sentence
        return _silence(FAIL_TRUNCATED, attempted=TRANSITION,
                        draft=_raw_draft({"text": raw}), generation=gen,
                        **style)
    if not raw:
        return _silence(FAIL_EMPTY, attempted=TRANSITION, generation=gen,
                        **style)
    stage = p2_unsafe_markup(raw, packet, context)
    if stage:
        return _silence(FAIL_STAGE, attempted=TRANSITION,
                        draft=_raw_draft({"text": raw}), stage=stage,
                        generation=gen, **style)
    out = validate({"action": TRANSITION, "text": raw, "fact_ids": []},
                   packet, clean_fn=clean_fn, guard_fn=guard_fn,
                   spoken_fn=spoken_fn, warn_fn=warn_fn)
    if out.get("action") != SAY_NOTHING:
        # The production contract passed; P2's own content screen last
        # (refusal attribution stays with the production check that fires).
        hit = p2_content_screen(raw, packet, context)
        if hit:
            return _silence(hit[0], attempted=TRANSITION,
                            draft=_raw_draft({"text": raw}), claim=hit[1],
                            generation=gen, **style)
    out.update(style)
    out["generation"] = gen
    return out


def _attempt_record(n: int, d: dict[str, Any]) -> dict[str, Any]:
    """What forensics keeps of one attempt: verdict, evidence, the draft
    when it was refused (bounded; never aired, never cleaned for speech)."""
    rec = {"attempt": n, "action": d.get("action"), "reason": d.get("reason"),
           "generation": dict(d.get("generation") or {})}
    for k in ("claim", "stage", "guard_category", "clock"):
        if d.get(k):
            rec[k] = d[k]
    if d.get("action") == SAY_NOTHING and d.get("draft"):
        rec["draft"] = d["draft"]
    return rec


def _decide_p2(packet: dict[str, Any], model_fn, prompt: str | None,
               context: dict[str, Any] | None, *,
               clean_fn: Callable[[str], str],
               guard_fn: Callable[[str], str | None] | None = None,
               spoken_fn: Callable[[str], set[str]] | None = None,
               warn_fn: Callable[[str], list[str]] | None = None,
               retry_budget_s: float | None = None,
               clock: Callable[[], float] | None = None
               ) -> dict[str, Any]:
    """The P2 branch of `decide`: one attempt, and -- only when it was refused
    for a CONTENT reason (P2_RETRY_REASONS) and the remaining budget covers
    another generation -- one more, judged independently. An accepted first
    attempt is final. The returned decision is the last attempt's, carrying
    `attempts` (both verdicts and drafts) and `retry` (eligibility, whether
    it ran, why not, the budget it had) for forensics.

    `retry_budget_s` is the time, from this call, that the decision may still
    spend: the caller derives it from the boundary deadline and the planner
    claim (dj_engine._p2_retry_budget). None (no known deadline) means no
    retry. The retry's own HTTP timeout is capped to what remains."""
    import time as _time
    now = clock or _time.monotonic
    style = {"writer_style": WRITER_P2}
    if p2_calendar_owed(packet, context):
        # No model call: the floor hosts the calendar (FAIL_P2_CALENDAR).
        return _silence(FAIL_P2_CALENDAR, **style, attempts=[],
                        retry={"eligible": False, "attempted": False,
                               "skipped": "calendar_deterministic",
                               "budget_s": retry_budget_s})
    if not prompt:
        return _silence(FAIL_INVALID, **style)
    t0 = now()
    kw = dict(clean_fn=clean_fn, guard_fn=guard_fn, spoken_fn=spoken_fn,
              warn_fn=warn_fn)
    first = _p2_attempt(packet, model_fn, prompt, context, **kw)
    attempts = [_attempt_record(1, first)]
    retry: dict[str, Any] = {"eligible": False, "attempted": False,
                             "skipped": None, "budget_s": retry_budget_s}
    final = first
    if first.get("action") == SAY_NOTHING:
        retry["eligible"] = first.get("reason") in P2_RETRY_REASONS
        if not retry["eligible"]:
            retry["skipped"] = "reason_not_eligible"
        elif retry_budget_s is None:
            retry["skipped"] = "no_deadline"
        else:
            elapsed = now() - t0
            first_s = ((first.get("generation") or {}).get("wall_ms") or 0) / 1000
            need = max(P2_RETRY_EST_MIN_S, P2_RETRY_EST_FACTOR * first_s)
            left = retry_budget_s - elapsed
            retry.update(elapsed_s=round(elapsed, 2), need_s=round(need, 2),
                         left_s=round(left, 2))
            if left < need:
                retry["skipped"] = "deadline"
            else:
                retry["attempted"] = True
                second = _p2_attempt(packet, model_fn, prompt, context,
                                     timeout_s=left, **kw)
                attempts.append(_attempt_record(2, second))
                final = second
    final = dict(final)
    final["attempts"] = attempts
    final["retry"] = retry
    if final.get("action") != SAY_NOTHING:
        final["accepted_attempt"] = len(attempts)
    return final
