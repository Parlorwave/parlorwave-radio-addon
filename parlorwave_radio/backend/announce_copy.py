"""Announcement copy: the directive parser, the deterministic renderer,
the Smart rewrite prompt and its grounding validator.

An announcement is operator programming, not DJ conversation. Everything
here is therefore held to a stricter contract than a music break:

- ANNOUNCE EXACT is the operator's approved wording. Nothing here may
  rewrite it; the only transformation it ever receives is the same
  speech.normalize_for_tts pass every finite audio render performs.
- Plain ANNOUNCE may be rewritten by the Smart writer, but only from the
  facts in the payload itself. validate_smart is the arbiter: a draft
  that changes a number, changes a time, introduces a number the payload
  never stated, drops a named subject, or balloons past the payload's
  own size is rejected outright. There is no repair loop -- a rejected
  draft falls to the deterministic renderer, which is grounded by
  construction.
- The deterministic renderer must work with NO model configured. It
  expands common operator shorthand (5p-10p, $11.99, 20%, &) into
  minimally natural broadcast copy and otherwise preserves the payload.

The parser is the single source of truth for what counts as a directive;
the integration mirrors only the cheap prefix test for routing (an
ANNOUNCE summary must never leak into ordinary DJ calendar context)."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal
from typing import Any

MODE_EXACT = "exact"
MODE_SMART = "smart"

#: Longest payload an announcement may carry, in characters. Long enough
#: for a paragraph of operator copy, short enough that one announcement
#: can never occupy the boundary for minutes.
MAX_PAYLOAD_CHARS = 500

INVALID_EMPTY = "empty_payload"
INVALID_TOO_LONG = "payload_too_long"

#: The routing test: does this summary CLAIM to be a directive at all?
#: Deliberately word-bounded -- "Announcements committee meeting" is an
#: ordinary calendar event, "announce Quiz night" is a directive. The
#: integration applies this same test (its own copy; documented mirror)
#: so directive summaries never reach the DJ's calendar context.
DIRECTIVE_RE = re.compile(r"^\s*announce\b", re.IGNORECASE)

_PARSE_RE = re.compile(
    r"^\s*announce(?:\s+(?P<exact>exact\b))?(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL)


def is_directive(summary: Any) -> bool:
    return bool(isinstance(summary, str) and DIRECTIVE_RE.match(summary))


def parse_summary(summary: str) -> dict[str, Any]:
    """{'mode': exact|smart, 'payload': str, 'invalid_reason': None|str}.

    Only call for summaries that pass is_directive. Whitespace inside the
    payload is collapsed; the prefix itself is never part of the payload.
    An empty payload is a directive with nothing to say -- it must never
    become a silent programming element, so it parses as invalid and the
    reason is surfaced to the operator instead."""
    m = _PARSE_RE.match(summary or "")
    if not m:                                   # not a directive at all
        return {"mode": MODE_SMART, "payload": "",
                "invalid_reason": INVALID_EMPTY}
    mode = MODE_EXACT if m.group("exact") else MODE_SMART
    payload = re.sub(r"\s+", " ", m.group("rest") or "").strip()
    invalid = None
    if not payload:
        invalid = INVALID_EMPTY
    elif len(payload) > MAX_PAYLOAD_CHARS:
        invalid = INVALID_TOO_LONG
    return {"mode": mode, "payload": payload, "invalid_reason": invalid}


# --- announcement voice request ----------------------------------------------
#
# A calendar event may ask for its announcement to be spoken by a named,
# explicitly authorized voice. The request lives in the event's DESCRIPTION,
# on its own line -- `voice: Evening Reader` -- never in the summary: the summary IS
# the spoken content (after the ANNOUNCE prefix), and no voice syntax may
# ever share a field with words that go to air. Structure, not parsing,
# keeps the directive out of the speech: a malformed voice line cannot
# corrupt a payload it never touches.
#
# The integration reads that one line (its own mirror of VOICE_LINE_RE,
# exactly as it mirrors DIRECTIVE_RE) and posts only the VALUE with the
# observation -- the description itself never leaves Home Assistant. The
# value is a REQUEST, not an identity: what it names, and whether that
# voice is authorized for announcements, is decided here at commit time
# (voice_library.resolve_announcement_voice), never by the calendar.

#: The first line of a description reading `voice: <name or id>`; the key
#: is case-insensitive, surrounding blanks are ignored.
VOICE_LINE_RE = re.compile(
    r"^[ \t]*voice[ \t]*:[ \t]*(?P<value>\S.*?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE)
#: Longer than any shelved voice id or display name (40); a request past
#: this can name nothing and is kept only as far as it can be shown.
MAX_VOICE_REQUEST_CHARS = 64


def voice_line(description: Any) -> str | None:
    """The requested voice named by a description's `voice:` line, or
    None. The canonical reader; the integration mirrors it."""
    if not isinstance(description, str):
        return None
    m = VOICE_LINE_RE.search(description)
    return normalize_voice_request(m.group("value")) if m else None


def normalize_voice_request(value: Any) -> str | None:
    """A request as the store may hold it: printable characters only,
    whitespace collapsed, bounded. Anything else is NO request -- never
    an error that could reach the payload, and never spoken."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join("".join(ch for ch in value if ch.isprintable()).split())
    if not cleaned:
        return None
    return cleaned[:MAX_VOICE_REQUEST_CHARS]


# --- explicit times -----------------------------------------------------------
#
# A number is a clock time ONLY when the operator wrote one. The forms:
#
#   meridiem    5pm  5 PM  5:30 pm  7 P.M.  9 a.m.   (hour 1-12 with AM/PM in
#               full -- always a time)
#   colon       6:30  18:05                          (a clock whose meridiem
#               is unknown: recognised for grounding, never re-spelled and
#               never given an AM/PM)
#   shorthand   5p  10a  5:30p  7P                   (hour 1-12, ONE letter
#               attached with no space) -- a time ONLY where the context
#               says so, because "Studio 4a", "Room 4a", "Gate 3A" and
#               "Stage 2a" are written exactly the same way:
#                 * a range of two shorthands across a tight dash, at
#                   least one of them p ("5p-10p", "11a-2p"), or across
#                   "to"/"until"/a spaced dash when the first is p
#                   ("9p to 1a", "5p - 10p");
#                 * right after a time word ("at 7p", "from 5p", "until
#                   10p", "by 6p", "starts 7p", "opens 5p", "doors 6p");
#                 * the next item after a time ("between 5p and 7p",
#                   "at 5p, 7p").
#               Anything else -- including an a-only range ("Gate 1a-3a",
#               which is as likely a run of gates as 1 to 3 AM) and a
#               shorthand next to a spelled-out time ("Room 4a-7pm") -- is
#               ambiguous, and ambiguous text is left exactly as written.
#               There is no noun list: the decision is the context's alone.

_MERIDIEM_RE = re.compile(
    r"(?<![\w$.,:])(?P<h>\d{1,2})(?::(?P<m>[0-5]\d))?\s?"
    r"(?P<mer>[AaPp])(?P<dots>\.[Mm]\.?|[Mm])(?![A-Za-z0-9])")
_SHORTHAND_RE = re.compile(
    r"(?<![\w$.,:])(?P<h>\d{1,2})(?::(?P<m>[0-5]\d))?"
    r"(?P<mer>[AaPp])(?![A-Za-z0-9])")
_COLON_RE = re.compile(r"(?<![\w$.,:])(?P<h>\d{1,2}):(?P<m>[0-5]\d)(?![\d:])")

#: Words after which a shorthand is a time.
_TIME_CUE_RE = re.compile(
    r"(?:\b(?:at|from|until|till|til|by|starts?|starting|opens?|opening|"
    r"doors|begins?|beginning|runs?|before|after|around|ends?|closes?|"
    r"through|thru)|@)\s*$", re.IGNORECASE)
#: A range word standing BETWEEN two shorthands is not a cue for the
#: second one: "Gates 1a through 3a" is a pair, judged by the pair rule.
_PAIR_WORD_AFTER_SHORT_RE = re.compile(
    r"(?<![\d:])\d{1,3}[A-Za-z]?\s*(?:until|till|til|through|thru|to)\s*$",
    re.IGNORECASE)
#: What may join a time to the next one ("5p and 7p", "5p, 7p", "5p to 7p").
_CHAIN_RE = re.compile(
    r"\s*(?:,|&|\band\b|\bor\b|\bto\b|\buntil\b|\btill\b|\btil\b|"
    r"\bthrough\b|\bthru\b|[-–—])\s*$", re.IGNORECASE)
#: The joint of a word-joined pair ("9p to 1a", "5p - 10p", "1a through 3a").
_PAIR_GAP_RE = re.compile(
    r"\s*(?:[-–—]|to|until|till|til|through|thru)\s*", re.IGNORECASE)
_TIGHT_DASH = "-–—"
#: A list neighbour that is plainly an identifier (a code whose letter is
#: not a or p): "3A/3B", "2a and 2b", "4a, 4b" -- then this one is too.
_LIST_JOIN = r"\s*(?:,|&|/|\band\b|\bor\b|[-–—])\s*"
_CODE_SIBLING_AFTER_RE = re.compile(
    _LIST_JOIN + r"\d{1,3}[b-oq-zB-OQ-Z](?![A-Za-z0-9])")
_CODE_SIBLING_BEFORE_RE = re.compile(
    r"(?<![\w$.,:])\d{1,3}[b-oq-zB-OQ-Z]" + _LIST_JOIN + r"$")


def _pair_is_time(text: str, first: re.Match, second: re.Match) -> bool:
    """Two shorthands side by side: are they a time range? Across a tight
    dash, when at least one end is p ("5p-10p", "11a-2p"). Across a word
    or a spaced dash, only when BOTH are p ("5p - 10p", "5p to 10p",
    "between 5p and 7p"): an a in that position is the shape of an
    identifier ("Bingo at 7p - 4a", "Classrooms 2B through 4A")."""
    gap = text[first.end():second.start()]
    a, b = first.group("mer").lower(), second.group("mer").lower()
    if gap in tuple(_TIGHT_DASH):
        return "p" in (a, b)
    head = text[max(0, first.start() - 16):first.start()]
    if _PAIR_GAP_RE.fullmatch(gap) or (
            re.fullmatch(r"\s*and\s*", gap, re.IGNORECASE)
            and re.search(r"\bbetween\s*$", head, re.IGNORECASE)):
        return a == b == "p"
    return False


def _shorthand_is_time(text: str, m: re.Match, shorts: dict[int, re.Match],
                       accepted: dict[int, str]) -> bool:
    """The context rules above, for one shorthand match. `accepted` maps
    the end offset of every time already accepted to its meridiem.

    Every path that can make an `a` shorthand a time is either a time word
    right before it or a tight pair with a p ("11a-2p"). Chains and
    word-joined pairs only ever reach p, because a letter-a code is the
    common identifier ("4a", "2a", "12A")."""
    start, end = m.start(), m.end()
    mine = m.group("mer").lower()
    ordered = sorted(shorts)
    idx = ordered.index(start)
    nxt = shorts[ordered[idx + 1]] if idx + 1 < len(ordered) else None
    prv = shorts[ordered[idx - 1]] if idx > 0 else None
    # a TIGHT pair decides on its own, either way
    for first, second in ((prv, m), (m, nxt)):
        if first is not None and second is not None and \
                text[first.end():second.start()] in tuple(_TIGHT_DASH):
            return _pair_is_time(text, first, second)
    # an all-p word-joined pair
    for first, second in ((prv, m), (m, nxt)):
        if first is not None and second is not None and \
                _pair_is_time(text, first, second):
            return True
    # an identifier list ("3A/3B", "4a and 4b") is never a time
    if _CODE_SIBLING_AFTER_RE.match(text, end) or \
            _CODE_SIBLING_BEFORE_RE.search(text[max(0, start - 24):start]):
        return False
    head = text[max(0, start - 40):start]
    if _TIME_CUE_RE.search(head) and not _PAIR_WORD_AFTER_SHORT_RE.search(head):
        return True
    # a p shorthand listed after an accepted time ("at 5p, 7p and 9p",
    # "Tours at 11a, 1p and 3p")
    if mine == "p":
        chain = _CHAIN_RE.search(head)
        if chain is not None and (start - len(head) + chain.start()) in accepted:
            return True
    return False


def _explicit_times(text: str) -> list[tuple[int, int, tuple[int, int, str | None]]]:
    """[(start, end, (hour % 12 or hour, minute, 'a'|'p'|None))] for every
    clock time the operator wrote explicitly, left to right, non-
    overlapping. Meridiem forms first; shorthand only in a time context;
    colon times last (meridiem None)."""
    found: list[tuple[int, int, tuple[int, int, str | None]]] = []
    taken: list[tuple[int, int]] = []

    def free(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in taken)

    for m in _MERIDIEM_RE.finditer(text):
        hour = int(m.group("h"))
        if 1 <= hour <= 12 and free(m.start(), m.end()):
            found.append((m.start(), m.end(), (hour % 12, int(m.group("m") or 0),
                                               m.group("mer").lower())))
            taken.append((m.start(), m.end()))
    shorts = {m.start(): m for m in _SHORTHAND_RE.finditer(text)
              if 1 <= int(m.group("h")) <= 12 and free(m.start(), m.end())}
    accepted = {e: v[2] for _s, e, v in found}
    for start in sorted(shorts):
        m = shorts[start]
        if _shorthand_is_time(text, m, shorts, accepted):
            hour = int(m.group("h"))
            found.append((m.start(), m.end(), (hour % 12, int(m.group("m") or 0),
                                               m.group("mer").lower())))
            taken.append((m.start(), m.end()))
            accepted[m.end()] = m.group("mer").lower()
    for m in _COLON_RE.finditer(text):
        hour = int(m.group("h"))
        if 0 <= hour <= 23 and free(m.start(), m.end()):
            found.append((m.start(), m.end(), (hour, int(m.group("m")), None)))
            taken.append((m.start(), m.end()))
    found.sort()
    return found


# --- deterministic renderer ------------------------------------------------

#: $2  $2.50  $11.99  $1,000  $1,000.50 -- comma groups are exact
#: thousands, and a comma that separates list items ("$2, $5") is not one.
#: A magnitude ("$1.5 million", "$5k") is not dollars-and-cents: anything
#: that does not parse cleanly is left as written.
_MONEY_RE = re.compile(
    r"(?<![\w$])\$(?P<d>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<c>\d{1,2}))?"
    r"(?!\d|,\d|\.\d|[A-Za-z])"
    r"(?!\s*(?:million|billion|trillion|thousand|hundred|grand|mil|bn)\b)",
    re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")


def _speak_time(hour: int, minute: int, meridiem: str) -> str:
    mer = "AM" if meridiem == "a" else "PM"
    hour = hour or 12
    if minute:
        return f"{hour}:{minute:02d} {mer}"
    return f"{hour} {mer}"


def _expand_times(text: str) -> str:
    """Re-spell explicit AM/PM times ("5p", "6:30pm", "7 P.M.") as
    "5 PM" / "6:30 PM". Colon times without a meridiem stay as written."""
    out, pos = [], 0
    for start, end, (hour, minute, mer) in _explicit_times(text):
        if mer is None:
            continue
        spoken = _speak_time(hour, minute, mer)
        # a dotted "p.m." swallowed the sentence's own full stop
        if text[start:end].endswith(".") and re.match(r"\s+[A-Z]", text[end:]):
            spoken += "."
        out.append(text[pos:start])
        out.append(spoken)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _expand_money(text: str) -> str:
    def sub(m: re.Match) -> str:
        written, cents = m.group("d"), m.group("c")
        dollars = int(written.replace(",", ""))
        cents_n = int(cents) if cents else 0
        if cents and len(cents) == 1:       # "$5.5" means 50 cents
            cents_n = int(cents) * 10
        parts = []
        if dollars:
            # the operator's own digits, commas included: "1,000 dollars"
            parts.append(f"{written} dollar" + ("s" if dollars != 1 else ""))
        if cents_n:
            parts.append(f"{cents_n} cent" + ("s" if cents_n != 1 else ""))
        return " and ".join(parts) if parts else "0 dollars"
    return _MONEY_RE.sub(sub, text)


#: A range whose right end is an explicit, already re-spelled time: the
#: dash is spoken as "to". The left end is kept exactly as re-spelled or as
#: written -- "7-9pm" becomes "7 to 9 PM", never "7 PM to 9 PM" (that
#: would be inventing a meridiem the operator did not write). A BARE left
#: number joins only across a tight dash ("7-9pm"): "Hall 2 - 7pm" is a
#: room and a time, not a range.
_RANGE_RE = re.compile(
    r"(?<![\w$.,:])(\d{1,2}(?::[0-5]\d)? [AP]M)\s*[-–—]\s*"
    r"(\d{1,2}(?::[0-5]\d)? [AP]M)\b"
    r"|(?<![\w$.,:])(\d{1,2}(?::[0-5]\d)?)[-–—]"
    r"(\d{1,2}(?::[0-5]\d)? [AP]M)\b")


def deterministic_copy(payload: str) -> str:
    """Minimally natural broadcast copy from operator shorthand. Grounded
    by construction: every fact in the output is a re-spelling of a fact
    in the payload, and nothing is re-spelled unless its meaning is
    explicit -- a bare number is never made a time, and a price is never
    split. Never returns empty for a non-empty payload."""
    text = _expand_times(payload)
    text = _RANGE_RE.sub(
        lambda m: (f"{m.group(1)} to {m.group(2)}" if m.group(1)
                   else f"{m.group(3)} to {m.group(4)}"), text)
    text = _expand_money(text)
    text = _PERCENT_RE.sub(lambda m: f"{m.group(1)} percent", text)
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


# --- Smart rewrite ----------------------------------------------------------

def smart_prompt(payload: str) -> str:
    """One strict instruction block. The payload is the ONLY source of
    fact; the instructions forbid everything the validator would reject,
    so an obedient model passes and a creative one falls back."""
    return (
        "You are a radio announcer reading a station announcement.\n"
        "Rewrite the announcement below as ONE short, natural spoken "
        "line.\n"
        "Rules:\n"
        "- Use ONLY facts stated in the announcement.\n"
        "- Do not add any place, price, time, date, offer, sponsor, "
        "urgency, or claim that is not in the announcement.\n"
        "- Keep every number, price and time exactly as written, in "
        "digits.\n"
        "- Do not write out numbers as words.\n"
        "- Keep every named subject.\n"
        "- No greetings, no station name, no hashtags, no emojis, no "
        "quotation marks.\n"
        "- Return only the spoken line, nothing else.\n\n"
        f"ANNOUNCEMENT: {payload}\n")


#: One number is one number: "1,500" is fifteen hundred, never "1" and
#: "500" (which would let a draft turn $1,500 into $500 and still pass).
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")

#: Spelled-out quantities. A Smart draft may use a number word only when
#: the payload itself did -- the prompt demands digits, so a spelled
#: number is a model paraphrasing values it was told to preserve.
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand "
    "noon midnight".split())

#: Connective / framing words a rewrite may freely add or drop. Anything
#: else in the payload is treated as a named subject that must survive.
_STOPWORDS = frozenset(
    "the and for with our your from until till all are you this that "
    "per off out not has have will its it's a an at in on to of is be "
    "we us they them there here come join don't dont forget every "
    "starting starts start runs run open opens".split())


def _times_of(text: str) -> set[tuple[int, int, str | None]]:
    return {value for _s, _e, value in _explicit_times(text)}


def _without_times(text: str) -> str:
    out, pos = [], 0
    for start, end, _value in _explicit_times(text):
        out.append(text[pos:start])
        out.append(" ")
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _canon_num(tok: str) -> str:
    """Exact value, never rounded: '11.90' == '11.9', '1,234,567' is one
    number, and 1,234,567 never collides with 1,234,570."""
    value = Decimal(tok.replace(",", ""))
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


#: A code a listener must hear exactly: digits and letters in one token
#: ("Gate 3A", "Room 101B", "B52", "5th"). Explicit times are not codes.
_CODE_RE = re.compile(r"(?<![\w$])(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]+(?!\w)")


#: A hyphen between a number and a lone letter ("3-A", "K-9") or the
#: apostrophe of a decade ("80's") is spelling, not a separator. A hyphen
#: between two codes ("4a-4d") is a range and keeps them apart.
_CODE_JOIN_RE = re.compile(
    r"(?<=\d)-(?=[A-Za-z](?![\w]))|(?<=\b[A-Za-z])-(?=\d)"
    r"|(?<=\d)['’](?=s\b)")


def _codes_of(text: str) -> set[str]:
    joined = _CODE_JOIN_RE.sub("", _without_times(text))
    return {c.lower() for c in _CODE_RE.findall(joined)}


def _numbers_of(text: str) -> set[str]:
    return {_canon_num(t) for t in _NUM_RE.findall(_without_times(text))}


def _subjects_of(payload: str) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(_without_times(payload))}
    return {w for w in words
            if len(w) >= 3 and w not in _STOPWORDS
            and w not in _NUMBER_WORDS}


def validate_smart(payload: str, copy: str) -> str | None:
    """None when the draft is grounded, else the rejection reason.

    The checks are set-containment, not prompt trust: every time and
    number the draft speaks must be one the payload stated; no spelled
    number may appear from nowhere; every named subject must survive;
    and the draft may not balloon far past the payload's own size."""
    if not copy or not copy.strip():
        return "empty_draft"
    if "\n" in copy.strip():
        return "not_a_single_line"
    if re.search(r"https?://|www\.", copy, re.IGNORECASE):
        return "link_added"
    if not _times_of(copy) <= _times_of(payload):
        return "unsupported_time"
    if not _numbers_of(copy) <= _numbers_of(payload):
        return "unsupported_number"
    if _codes_of(copy) != _codes_of(payload):
        return "code_changed"          # "Gate 3A" must not become "3B"
    payload_words = {w.lower() for w in _WORD_RE.findall(payload)}
    spelled = {w.lower() for w in _WORD_RE.findall(copy)} & _NUMBER_WORDS
    if spelled - payload_words:
        return "spelled_number_added"
    for subject in _subjects_of(payload):
        if not re.search(r"\b" + re.escape(subject) + r"s?\b", copy,
                         re.IGNORECASE):
            return "subject_dropped"
    payload_len = len(payload.split())
    if len(copy.split()) > max(26, 2 * payload_len + 10):
        return "draft_too_long"
    return None


# --- the raw-draft screen ------------------------------------------------------
#
# Runs on the model's RAW output, before anything is normalised away: a
# line break, a markdown bullet or a "Here you go:" is evidence that the
# model is talking to the operator rather than writing the line, and a
# cleaner that joined lines or stripped labels first would hide exactly
# that evidence. Anything it rejects falls to the deterministic renderer;
# nothing is repaired and the model is never asked again.

#: Typography folded for SCREENING only (the draft itself is never
#: rewritten): a writer model may type ’ as often as ', and a rule spelled
#: with one must not be defeated by the other.
_SCREEN_FOLD = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "′": "'",
                              "“": '"', "”": '"', "″": '"',
                              " ": " ", " ": " ", " ": " "})


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_SCREEN_FOLD)


#: Nouns a model uses to name its OWN output.
_OUTPUT_NOUN = (r"(?:version|rewrite|line|announcement|copy|script|draft|"
                r"text|read|spot|promo|blurb)s?")

#: Assistant/meta talk: the model addressing the person who asked, framing
#: its own output, offering more, or explaining itself. Word-bounded and
#: case-blind, matched against the folded text. A broadcast "Here's the
#: deal" or "Here's your chance" is NOT framing; "Here's your announcement"
#: is.
_CHATTER_RE = re.compile(
    r"\bhere(?:'s| is| are)? (?:you go|you are|it is|we go|goes)\b"
    r"|\bthere you go\b|\bhow about this\s*:|\btry this\s*:"
    r"|\bi(?:'d| would) (?:say|put it|go with)\s*:"
    r"|\bhere(?:'s| is| are) (?:(?:your|the|my|a|an)\s+)?"
    r"(?:(?:new|revised|rewritten|natural|spoken|radio|final|suggested|"
    r"updated|short|shorter|improved|broadcast|on-air|polished)\s+)*"
    + _OUTPUT_NOUN + r"(?![\w-])"
    # any wording, as long as it names the output and then presents it:
    # "Here's a short, natural spoken line:", "Here’s a more natural version -"
    r"|\bhere(?:'s| is| are)\s+(?:[\w'-]+,?\s+){0,6}?" + _OUTPUT_NOUN
    + r"(?![\w-])\s*(?::|—|–|-(?=\s))"
    r"|\bhope (?:this|that|it) (?:helps|works)\b"
    r"|\blet me know\b|\bfeel free\b"
    r"|\banything else (?:i can|you(?:'d| would)? (?:like|need))\b"
    r"|\bis there anything else\b"
    # offers made to the OPERATOR, not questions put to listeners
    r"|\b(?:would you like|do you want|want) (?:me\b|it\b|any\b|some\b|"
    r"other\b|more\b|a (?:few|more|shorter|longer|different|version)\b)"
    r"|\bwant me to\b|\banything else\s*\?\s*$"
    r"|\b(?:shall|should) i\b"
    r"|\bi can (?:make|adjust|shorten|change|rewrite)\b"
    r"|\bif you(?:'d| would) like,? i\b"
    r"|\b(?:i've|i have|i) (?:rewritten|rewrote|written|wrote|kept|made|"
    r"changed|removed|added|used|preserved|included|adjusted|revised)\b"
    r"|\brewritten (?:announcement|line|version)\b"
    r"|\b(?:this|the above|the following) (?:version|rewrite|line|"
    r"announcement)\b"
    r"|\bas an ai\b|\blanguage model\b"
    r"|\b(?:note|explanation)\s*:\s*(?:i|i've|this|the (?:above|rewrite|"
    r"line|announcement|version))\b"
    r"|\b(?:all )?(?:facts|prices|times) (?:preserved|kept|unchanged)\b"
    r"|\bword count\b",
    re.IGNORECASE)

#: An opening reply to a request. The unmistakable ones are rejected
#: before any punctuation ("Sure, ...", "Certainly! ..."); the ones a
#: radio host also says ("Okay, folks", "Great, ...") only when they stand
#: alone as a sentence ("Okay." / "Great!").
_PREAMBLE_RE = re.compile(
    r"^\s*\"?(?:(?:sure(?: thing)?|certainly|absolutely|you got it|"
    r"happy to help|gladly|understood|no problem|got it|note)\b"
    r"\s*(?:[!,.:;—–]|-(?=\s))"
    r"|(?:okay|ok|great|alright|all right|of course|done|here we go|"
    r"here goes)\b\s*(?:[!.:;—–]|-(?=\s)))", re.IGNORECASE)

#: A label naming the output rather than speaking it, near the start:
#: "Announcement:", "Station announcement:", "Revised version -",
#: "Output -", "VO:".
_LABEL_RE = re.compile(
    r"^\s*\"?(?:"
    # nouns that only ever name an output, after up to three words
    r"(?:[\w'-]+,?\s+){0,3}?(?:announcement|version|rewrite|rewritten|"
    r"draft|script|copy|output|result|answer|response|voice-?over|vo|"
    r"announcer|dj|text|host)"
    r"|option\s+\w{1,2}"                       # "Option A:", not "a sweet option:"
    # "line" only as the output's own label ("Spoken line:", not "The
    # bottom line:" or "Get in line:")
    r"|(?:(?:spoken|suggested|final|radio|new|revised|natural|broadcast|"
    r"on-air|rewritten|short|shorter|polished|updated)\s+)*line"
    r"|final|revised|suggested|spoken"
    # a delivery note: "Upbeat tone:", "In a warm, upbeat voice:" -- not
    # "Find your voice:"
    r"|in an? [\w ,'-]{0,30}\b(?:tone|voice)|[\w-]+ tone"
    r")(?![\w-])\s*(?::|—|–|-(?=\s))", re.IGNORECASE)

#: Markdown and code: emphasis (single or double, * or _), fences, inline
#: code, headings, bullets, numbered lists, block quotes, links, tables.
_MARKDOWN_RE = re.compile(
    r"`|\*|(?<!\w)_|_(?!\w)|^\s*#{1,6}\s|^\s*[-•+]\s|^\s*\d+[.)]\s|^\s*>"
    r"|\]\(|\|\s*\S+\s*\|", re.MULTILINE)

#: Tags (a thinking model's <think>, stray HTML/SSML), bracketed or
#: parenthesised stage directions at either end or before a colon, and a
#: quoted line followed by commentary.
_MARKUP_RE = re.compile(
    r"<\s*/?\s*[A-Za-z][^>]*>|\[[^\]]*\]"
    r"|\([^)]*\)"                            # any parenthetical ("(pause)")
    r"|^\s*\"[^\"]{4,}\"\s*\S"                # a quoted line, then more
    r"|:\s*\"[^\"]{4,}\"[.!?]?\s*$")            # "Try: \"...\""

_HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z]")
_EMOTICON_RE = re.compile(r"(?<!\w)[:;]-?[)(DPp](?!\w)")

# --- the fact-free backstop -----------------------------------------------------
#
# Deny lists alone cannot keep up with every way a model frames or follows
# up its answer. So, in addition: split the draft into sentences and
# label-like segments ("...:", " - ", " — "), and hold every segment that
# carries NONE of the payload's facts (a subject word, a number, a time, a
# code) to a short, plain hook: at most FACTLESS_MAX_WORDS words and no
# assistant vocabulary. "Great news!", "Heads up:", "Here's the deal:" and
# "See you there!" pass; "Here's a short, natural spoken line for the
# radio:", "Radio Announcer:", "Would you like any changes?" and
# "Anything else?" do not.
FACTLESS_MAX_WORDS = 8
_SEGMENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|:\s+|\s+[-–—]\s+|\s*—\s*")
_ASSISTANT_WORDS = frozenset(
    "i i'm i've i'd i'll me my mine version versions rewrite rewrites "
    "rewritten rewrote draft drafts script scripts text announcer dj spoken "
    "radio changes adjust adjustments adjusted tweak tweaks options "
    "alternatives alternative help anything else hope suggestion "
    "suggestions feedback formal shorter longer punchier output prompt kept "
    "preserved written original suggested revised sound sounds air".split())


def _in_payload(phrase: str, folded_payload: str) -> bool:
    return bool(phrase) and re.search(
        r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", folded_payload,
        re.IGNORECASE) is not None


def screen_raw_draft(payload: str, raw: Any) -> str | None:
    """None when the RAW model output looks like one broadcast line, else
    the rejection reason. Structure is judged before any normalisation;
    the patterns see a typography-folded COPY, never a rewritten draft.
    Something the operator's own payload says (the same symbol, or the
    same whole phrase) is never held against a draft."""
    if not isinstance(raw, str) or not raw.strip():
        return "empty_draft"
    if "\n" in raw.strip() or "\r" in raw.strip():
        return "not_a_single_line"
    text = _fold(raw.strip())
    folded_payload = _fold(payload or "")
    if re.fullmatch(r"([\"'])[^\"]*\1[.!?]?", text):
        return "quoted_draft"                # the prompt forbids quotes;
                                             # unwrapping would be a repair
    for m in _MARKDOWN_RE.finditer(text):
        mark = m.group(0)
        if mark in ("*", "_") and mark in folded_payload:
            continue                         # the operator typed it
        return "markdown_formatting"
    for m in _MARKUP_RE.finditer(text):
        span = m.group(0)
        if span.startswith("(") and _in_payload(span, folded_payload):
            continue                         # the operator's own aside
        if span.startswith(("[", "<")) and span in folded_payload:
            continue
        return "markup"
    if _HASHTAG_RE.search(text) and "#" not in folded_payload:
        return "markup"
    if any(unicodedata.category(ch) == "So" and ch not in folded_payload
           for ch in text):
        return "markup"                      # emoji / pictographs
    if _EMOTICON_RE.search(text) and not _EMOTICON_RE.search(folded_payload):
        return "markup"
    # An opening or a label is the operator's only if the payload OPENS
    # the same way ("Sure thing Saturday: ..."); "make sure" inside the
    # payload does not excuse a "Sure!" in front of the draft.
    for rx in (_PREAMBLE_RE, _LABEL_RE):
        m = rx.search(text)
        if m:
            mine = rx.search(folded_payload)
            if not mine or _norm_phrase(mine.group(0)) != _norm_phrase(
                    m.group(0)):
                return "assistant_chatter"
    for m in _CHATTER_RE.finditer(text):
        phrase = _norm_phrase(m.group(0))
        if phrase and not _in_payload(phrase, folded_payload):
            return "assistant_chatter"
    return _factless_segment(folded_payload, text)


def _factless_segment(payload: str, text: str) -> str | None:
    """The backstop above: a segment with none of the payload's facts must
    be a short, plain hook."""
    subjects = _subjects_of(payload)
    numbers, times, codes = (_numbers_of(payload), _times_of(payload),
                             _codes_of(payload))
    payload_words = {w.lower() for w in _WORD_RE.findall(payload)}
    for segment in _SEGMENT_SPLIT_RE.split(text):
        seg = segment.strip(" \"'“”:;,.!?-–—")
        if not seg:
            continue
        words = [w.lower() for w in _WORD_RE.findall(seg)]
        if (any(re.search(r"\b" + re.escape(sub) + r"s?\b", seg, re.I)
                for sub in subjects)
                or _numbers_of(seg) & numbers or _times_of(seg) & times
                or _codes_of(seg) & codes):
            continue                         # carries a payload fact
        if any(w in _ASSISTANT_WORDS and w not in payload_words
               for w in words):
            return "assistant_chatter"
        if len(words) > FACTLESS_MAX_WORDS:
            return "extraneous_sentence"
    return None


def _norm_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" \"'“:—–-!,.;")).lower()


def clean_smart_draft(raw: str) -> str:
    """Safe WHITESPACE normalisation of an already-screened draft: runs
    collapse, ends are trimmed. Nothing else -- quotes, labels and markdown
    are screen_raw_draft's to reject, never this function's to repair."""
    return re.sub(r"\s+", " ", raw or "").strip()


# --- calendar announcements: delivery inside a DJ break ----------------------
#
# When a plain ANNOUNCE rides an ordinary DJ break, the DJ writer is handed
# the message as REQUIRED, SUPPLIED content and asked to work it in in its
# own voice. Its words around the message are the DJ's (and are held to
# every DJ guard); the message itself must arrive intact. delivery_problem
# is the arbiter, by containment in the opposite direction to
# validate_smart: everything the MESSAGE states must be in the BREAK, which
# is free to say other things (a clock, a song, a weather reading) that the
# DJ's own guards already police. A break that fails it never airs as the
# announcement: the deterministic floor delivers the message verbatim.

#: A negation flips an announcement's meaning ("NOT open Monday"); the
#: subject check cannot see "not" (a stopword) or "no" (two letters), so a
#: negated message demands a negation in the break.
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|nobody|nothing|cannot|without)\b|n't\b",
    re.IGNORECASE)


def delivery_problem(message: str, text: str) -> str | None:
    """None when `text` carries every checkable fact of `message`, else the
    first reason: a time, a number or a code the message states and the
    break does not; a named subject the break dropped; a negation lost."""
    if not (text or "").strip():
        return "empty"
    if not _times_of(message) <= _times_of(text):
        return "time_missing"
    if not _numbers_of(message) <= _numbers_of(text):
        return "number_missing"
    if not _codes_of(message) <= _codes_of(text):
        return "code_missing"
    for subject in sorted(_subjects_of(message)):
        if not re.search(r"\b" + re.escape(subject) + r"s?\b", text,
                         re.IGNORECASE):
            return f"subject_missing:{subject}"
    if _NEGATION_RE.search(message) and not _NEGATION_RE.search(text):
        return "negation_missing"
    return None


def message_span(message: str, text: str) -> tuple[int, int] | None:
    """(first, last) character offsets in `text` of the message's subject
    words, or None when none is found: where the announcement sits."""
    hits = []
    for subject in _subjects_of(message):
        for m in re.finditer(r"\b" + re.escape(subject) + r"s?\b", text,
                             re.IGNORECASE):
            hits.append((m.start(), m.end()))
    if not hits:
        return None
    return min(h[0] for h in hits), max(h[1] for h in hits)
