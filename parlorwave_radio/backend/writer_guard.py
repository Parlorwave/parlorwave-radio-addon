"""Post-writer authority guard.

The prompt closes the OPPORTUNITY (a non-contrast transition offers no
transition fact and no transition instruction), and that gate works -- but
the writer model can still reach into its own pretrained knowledge of the
two NAMED songs and invent a musical relationship anyway ("going from that
dusty edge ... into something wide open").

This module applies the same principle to text: a successful upstream
component (the LLM returned words) does not automatically make its output safe
to air.  It runs after generation and before TTS, and answers exactly one
bounded question: did the writer make a KIND of claim it lacked authority to
make?  It never judges whether a claim is musically true -- "Ada Venn is
grittier than Wren Hollis" is invalid as an unauthorized COMPARISON, not
because the guard knows either band.

Two authority classes are checked:

  unauthorized_transition_relationship
      Comparative sonic/mood language or move-description between songs when
      no transition authority was supplied (transition_character is not a real
      contrast).  Naming both songs and handing off is always fine; grounded
      album/artist-run continuation phrasing carries no comparative or move
      language, so it passes untouched.

  unauthorized_station_pattern
      Generalized station/session habit claims ("we usually", "the pattern
      tonight", "you've been hearing a lot of").  These are NEVER authorized
      by a run fact: a supplied run authorizes a narrow current-run statement
      ("Ada Venn has owned this stretch"), which contains no habit marker and
      therefore never trips this class.

Supplied fact values (track, artist and album names) are masked out of the
text before matching, so a title like "Left Turn" can never trip the
move-noun check.  The check is a small, structured textual policy -- a
bounded lexicon of sonic/mood stems plus a handful of claim-shape patterns --
not a phrase blacklist of exact sentences.
"""

from __future__ import annotations

import re
from typing import Any

UNAUTHORIZED_TRANSITION = "unauthorized_transition_relationship"
UNAUTHORIZED_PATTERN = "unauthorized_station_pattern"
UNAUTHORIZED_ALBUM = "unauthorized_album_attribution"
UNAUTHORIZED_EARLIER_CHARACTER = "unauthorized_earlier_character"
UNAUTHORIZED_ARTIST = "unauthorized_artist_identity"
UNAUTHORIZED_GENRE = "unauthorized_genre_label"

# --- fact-gated invariants -------------------------------------------------
# Every one is a SEMANTIC GATE read from the supplied render context plus a
# surface DETECTOR. The gate is what keeps these invariants rather than
# blacklists: the identical sentence is legal when an authoritative fact
# licenses it and illegal when none does.
UNKNOWN_NEXT_NAMED = "unknown_next_named_act"        # BLOCKS
CALENDAR_OWNERSHIP = "calendar_ownership"            # BLOCKS
#: RETAINED AS A CATEGORY NAME, but nothing returns it from `check`
#: any more: biography/history is measured, not refused (see `warnings`).
#: Kept so a later re-promotion needs no new vocabulary, and so the
#: repair line below stays addressable.
UNSUPPORTED_BIOGRAPHY = "unsupported_biography"      # WARN ONLY

#: Biography / history lexicon, read together with the two structural
#: fixes below that keep it off false positives. Genuine catches are of the
#: shape "a quiet start to their career", "a few of their later
#: records", "how much these bands shaped each other".
_RE_BIO = re.compile(
    r"\b(career|debut|breakthrough|formed|founded|reunion|"
    r"(?:early|later|earlier|last|first|second|third|final)\s+"
    r"(?:work|works|record|records|album|albums|era|years|days|period)|"
    r"their\s+(?:first|second|third|early|later|last|best|biggest)|"
    r"back\s+(?:then|in\s+the)|years?\s+(?:later|earlier|ago)|decades?|"
    r"the\s+(?:sixties|seventies|eighties|nineties)|\b(?:19|20)\d\d\b|"
    r"influenc\w+|pioneer\w*)",
    re.IGNORECASE)
# DELIBERATELY NARROW. The lexicon carries no
# `legendary|seminal|iconic|landmark|classic`. Those are bare
# EVALUATIVE adjectives, and this station has always explicitly allowed
# absolute taste language -- "this one is moody and intense, classic
# late-night material" is a host's opinion, which the prompt permits and
# which the existing tests pin. A GENUINE biography catch is structural
# history and is still caught above:
# a career's start, an act's later records, how acts influenced one
# another. Taste is
# counted by the warn-only character telemetry, never refused here.

#: First-person OWNERSHIP of the listener's diary entry -- not merely
#: the word "I". "I hope your book club goes well" is correct
#: hosting and must not flag, so the marker has to be a construction that
#: takes possession, and the event noun has to follow it closely.
_RE_OWNS = re.compile(
    r"\b(?:i\s+(?:have|need|must|should|get|got)\b|i've\s+got\b"
    r"|i'?m\s+(?:going|heading|off|due|booked|expecting)\b|my\b)",
    re.IGNORECASE)

#: WARN ONLY. Forward phrasing with no authoritative next track.
_RE_FORWARD = re.compile(
    r"\b(up next|next up|coming up|coming right up|stay tuned|after this|"
    r"we'?ll (?:hear|play|have|be hearing)|here'?s|\bnext\b)", re.IGNORECASE)

#: Capitalized tokens that are ordinary words, never an act.
_COMMON_CAP = set((
    "that this it he she they we you i a an the and but so then now next up "
    "here there something anything nothing seems just still keep coming "
    "going well alright okay yes no oh quite certainly speaking hoping "
    "don't do not let's trash grocery book dentist").split())
_NUMLIKE = set((
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred degrees "
    "fahrenheit celsius monday tuesday wednesday thursday friday saturday "
    "sunday january february march april may june july august september "
    "october november december").split())
_RE_PRONOUN = re.compile(r"^(I|I'll|I'm|I've|I'd)$")
_RE_CAP_SPAN = re.compile(r"[A-Z][A-Za-z'&!.\-]*(?:\s+[A-Z][A-Za-z'&!.\-]*)*")
_RE_NAME_SPAN = re.compile(r"[A-Z][A-Za-z'&!.\-]*(?:\s+[A-Z][A-Za-z'&!.\-]*)+")


def _ordinary(tok: str) -> bool:
    """An everyday capitalized word rather than the name of an act. The
    contraction arm is a false-positive fix: "It's", "That's"
    and "Here's" open sentences constantly, and their bare stems are
    already known ordinary words."""
    base = tok.split("'")[0]
    return (tok in _COMMON_CAP or tok in _NUMLIKE
            or base in _COMMON_CAP or base in _NUMLIKE)


def _flatten(value: Any, out: list[str]) -> None:
    """Every string anywhere inside a supplied context value."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _flatten(v, out)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            _flatten(v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(str(value))


#: The render-context keys that carry AUTHORITATIVE material. A claim is
#: licensed when its words are already here; nothing else licenses it.
_FACT_KEYS = (
    "pre_name", "post_name", "ending_track_album", "next_track_album",
    "ending_track_character", "next_track_character", "transition_character",
    "current_weather", "weather_context", "daylight_line", "daylight_phase",
    "station_time", "station_time_forms", "calendar_event",
    "knowledge_facts", "album_sequence", "album_run", "semantic_themes",
    "semantic_tone", "daypart", "service_context",
    "station", "station_name",
)

#: The subset that is SERVICE context -- the room, the sky, the diary.
#: Used only to stop service wording reading as music history: a
#: supplied "The loft speaker was reconnected" is what makes "back
#: in the loft" a remark about the room and not about a career.
_SERVICE_KEYS = ("service_context", "weather_context", "current_weather",
                 "daylight_line", "calendar_event", "atmosphere_colour_text")


def _blob(ctx: dict[str, Any], keys) -> str:
    parts: list[str] = []
    for key in keys:
        if ctx.get(key):
            _flatten(ctx.get(key), parts)
    return " ".join(parts).lower()


def _proper_name_spans(text: str) -> list[tuple[int, int]]:
    """Capitalized MULTI-token spans -- i.e. names. A genre or biography
    word inside one is part of somebody's name, not a claim: a band with
    "Jazz" or "Soul" in its name, an album called Debut."""
    return [(m.start(), m.end()) for m in _RE_NAME_SPAN.finditer(text or "")]


def _inside_name(spans, pos: int) -> bool:
    return any(a <= pos < b for a, b in spans)


def _calendar_nouns(ctx: dict[str, Any]) -> tuple[str, ...]:
    """Substantive words from the listener's supplied diary entry."""
    event = ctx.get("calendar_event")
    if not isinstance(event, dict):
        return ()
    parts: list[str] = []
    for key in ("title", "summary"):
        if isinstance(event.get(key), str):
            parts.append(event[key])
    # NOT filtered against _COMMON_CAP: that list exists to stop the
    # unknown-next check reading "Dentist"/"Grocery" as an ACT, and those
    # are exactly the nouns an ownership claim takes possession of here.
    return tuple(sorted(set(re.findall(r"[a-z]{4,}",
                                       " ".join(parts).lower()))))

# Catalog artist names that are also ordinary words. These are matched only
# as a capitalized token that does not open a sentence ("Live is next" at a
# sentence start is ambiguous; "Bush" mid-sentence is the band), so a "live
# recording" or "tool" in prose never trips the guard.
_COMMON_WORD_ARTISTS = {
    "live", "bush", "tool", "yes", "cake", "train", "heart", "chicago",
    "boston", "genesis", "queen", "air", "blur", "muse", "garbage", "hole",
    "cream", "rush", "kiss", "wings", "america", "asia", "europe",
    "journey", "free", "television", "sparks", "war", "poison", "fun",
    "wham", "toto", "berlin", "kansas", "nazareth", "pulp", "suede",
}

# Genre labels the writer may use ONLY when a listed character line carries
# them (the existing prompt rule; this is its deterministic half). Kept to
# labels that are not ordinary English words -- 'alternative', 'rock',
# 'pop', 'country', 'folk', 'blues', 'soul', 'dance', 'house' are deliberately
# excluded because they read as plain words far more often than as genres.
_GENRE_LABELS = ("metal", "punk", "grunge", "hip hop", "hip-hop", "rap",
                 "jazz", "reggae", "ska", "techno", "emo", "indie",
                 "hardcore", "metalcore", "electronic")
_RE_GENRE = re.compile(r"\b(" + "|".join(re.escape(g) for g in _GENRE_LABELS)
                       + r")\b")
#: The same labels, case-insensitively, for the one scan that needs
#: ORIGINAL case alongside the match offsets (the proper-name exemption).
#: Lower-casing first would be wrong there, not merely redundant: a few
#: characters change LENGTH under `str.lower()` (Turkish dotted I among
#: them), which shifts every offset after them and would misalign the
#: name spans the exemption is read from.
_RE_GENRE_ANYCASE = re.compile(_RE_GENRE.pattern, re.IGNORECASE)

# Sonic / mood / character stems whose COMPARATIVE use asserts a relationship
# between songs.  Absolute feel language ("this one's moody", "high energy")
# is untouched -- the prompt explicitly allows it where character authority
# exists -- only "more/less X" and "X-er" claims are relationship claims.
_SONIC_STEMS = (
    "quiet", "loud", "heavy", "light", "soft", "hard", "gritty", "smooth",
    "mellow", "calm", "dark", "bright", "fast", "slow", "gentle", "raw",
    "polished", "aggressive", "energetic", "intense", "upbeat", "somber",
    "moody", "introspective", "expansive", "atmospheric", "abrasive",
    "delicate", "punchy", "subdued", "restrained", "muscular", "angry",
    "sparse", "dense", "warm", "cool", "big", "small", "grand",
)


def _comparative_forms(stem: str) -> list[str]:
    """The -er comparative(s) of a stem (quiet->quieter, gritty->grittier,
    gentle->gentler).  Multi-syllable stems that take no -er form in English
    simply never match; 'more <stem>' covers them."""
    if stem.endswith("y"):
        return [stem[:-1] + "ier"]
    if stem.endswith("e"):
        return [stem + "r"]
    return [stem + "er", stem + stem[-1] + "er"]   # big -> bigger


_ER_FORMS = sorted({f for s in _SONIC_STEMS for f in _comparative_forms(s)})

# --- claim-shape patterns (matched against masked, lowercased text) ---------

# a "more/less <sonic adjective>" comparison; \W{0,3} absorbs an ellipsis or
# dash the writer likes to put before the adjective ("a little more… expansive")
_RE_MORE_LESS = re.compile(
    r"\b(?:more|less)\W{0,3}(" + "|".join(_SONIC_STEMS) + r")\b")
# a bare -er comparative from the sonic lexicon ("a touch softer")
_RE_COMPARATIVE = re.compile(r"\b(?:" + "|".join(_ER_FORMS) + r")\b")
# describing the move itself as an object ("that's… a shift", "some segue",
# "a smooth transition", "a change of pace")
_RE_MOVE_NOUN = re.compile(
    r"\b(?:shift(?:s|ed|ing)?|transition(?:s|ed|ing)?|segue(?:s|d)?"
    r"|pivot(?:s|ed|ing)?|whiplash"
    r"|change\s+of\s+pace|gear\s+change|left\s+turn|about-face"
    r"|switch(?:es|ed|ing)?(?:-?up)?|an?\s+(?:\w+\s+)?change)\b")
# the "going from <X> to <Y>" construction
_RE_GOING_FROM = re.compile(
    r"\b(?:going|moving|switching|jumping|stepping|drifting|sliding)\s+from\b")

# --- session-comparison claim SHAPES ----------------------------------------
# The leak these close: characterize an earlier-session track from model
# memory, then relate it to the current/next one -- using vocabulary outside
# any bounded lexicon ("urgent", "restlessness", "intensity", "momentum").
# These match the FORM of that claim, not the words: whether the mood word is
# musically apt is never the question.
#
# similarity claims ("a similar kind of restlessness", "a similar intensity",
# "the same kind of energy") -- likeness between tracks is itself the claim
_RE_SIMILARITY = re.compile(
    r"\b(?:similar|(?:the\s+)?same)\s+(?:(?:kind|sort)s?\s+of\s+)?"
    r"(?!album\b|artist\b|band\b|record\b|year\b|name\b)\w+")
# a feel/seem/sound verb driving an open comparative ("felt a bit more
# urgent", "feels… further", "feels a bit… different") -- ANY adjective
# counts; \W{0,6} absorbs the writer's ellipses and dashes
_RE_FEEL_COMPARATIVE = re.compile(
    r"\b(?:feels?|felt|seems?|seemed|sounds?|sounded)\W{0,6}"
    r"(?:a\s+(?:bit|little)\W{0,6})?"
    r"(?:(?:more|less)\s+\w+|different|further|closer)\b")
# characterized-then-related: a characterization verb, then a contrast
# conjunction, then the current/next item (masked names appear as the mask
# glyph) -- "Slow Ferry had ..., but this one ...", "X felt ..., while
# the next ..."
_RE_CHARACTERIZE_RELATE = re.compile(
    r"\b(?:felt|had|seemed|sounded|carried|brought)\b[^.!?]{0,80}"
    r"\b(?:but|while|whereas|and\s+now)\b[^.!?]{0,80}"
    r"(?:§|¤|\bthis\b|\bthat\b|\bnext\b|\bnow\b|\bone\b)")

# --- earlier-session authority scope ----------------------------------------
# One flag could otherwise cover two different
# relationships: a real contrast authorizes commentary on the ADJACENT
# ending->next move, but the same flag also unlocked comparisons against
# arbitrary earlier_this_session tracks ("A few minutes ago, Cold Orchard
# brought another kind of edge").  Earlier-session comparisons have no
# authority source at all in the current system, so they are rejected under
# EVERY transition class.  A sentence is earlier-anchored conservatively:
# it names a known earlier-session identity (masked to ¤), or it carries a
# supplied-recency marker ("ago", "earlier", "<n> minutes/hours back") --
# deterministic context identity, not discourse analysis.
_RE_EARLIER_ANCHOR = re.compile(
    r"\b(?:ago|earlier|(?:minutes?|hours?|tracks?|songs?)\s+back)\b")
# difference claims join similarity as comparison shapes ("a different kind
# of intensity", "unlike that one"); kept to the kind-of / from forms so a
# harmless "something different now" never trips.
_RE_DIFFERENCE = re.compile(
    r"\b(?:different|unlike)\s+(?:kind|sort)s?\s+of\s+\w+"
    r"|\bdifferent\s+from\b|\bunlike\s+(?:§|¤|\w+)")


# --- earlier-session items carry OCCURRENCE authority only ------------------
# Character reaches the render context on exactly one path --
# derived_character(path) -> ending_track_character / next_track_character --
# and the played-session rows the earlier_this_session line is built from
# carry no path, so no per-item character fact for an earlier entry can
# exist.  Characterizing one is therefore unauthorized outright, with no
# comparison needed: "Marlow Bay, earlier today, brought a certain
# momentum" escapes the comparison shapes above because its relation to the
# next song is CONTINUITY ("and ... will keep that feeling going"), not contrast.
#
# The rule targets the claim TYPE, not an adjective vocabulary: a bounded
# list of character NOUNS (what a song is claimed to have) plus the copular
# feel verbs.  What a track sounded like is never in evidence for an
# occurrence-only entry, whatever adjective is chosen.
_CHARACTER_NOUNS = (
    "energy", "momentum", "intensity", "mood", "feel", "feeling", "vibe",
    "edge", "urgency", "weight", "warmth", "bite", "drive", "tone",
    "texture", "atmosphere", "character", "groove", "swagger", "calm",
    "restlessness", "melancholy", "punch", "sound", "spirit", "pace")
_NOUNS_RE = "|".join(_CHARACTER_NOUNS)

# "brought a certain drive", "had a nervous energy", "carried that warmth"
_RE_CHARACTER_ASCRIBED = re.compile(
    r"\b(?:had|brought|carried|gave|kept|packed|delivered|held)\b"
    r"(?:\W+\w+){0,3}?\W+(?:" + _NOUNS_RE + r")\b")
# "felt urgent", "seemed restless", "sounded enormous" -- an occurrence-only
# callback never needs these verbs ("were on", "came up", "we heard").
_RE_CHARACTER_FELT = re.compile(r"\b(?:felt|seemed|sounded)\b")
# the continuity tail: "keep THAT FEELING going", "staying in that same
# mood", "carries that energy forward" -- anaphora to a character the
# earlier entry was never given. Only checked when the break references an
# earlier item at all, so ordinary talk about the current song is untouched.
_RE_CHARACTER_ANAPHORA = re.compile(
    r"\b(?:that|this|the\s+same|its)\s+(?:" + _NOUNS_RE + r")\b")


def _comparison_shape(s: str):
    """Any shape that asserts a musical relationship between two tracks --
    the earlier-session check and the non-contrast adjacent check share it."""
    return (_RE_SIMILARITY.search(s) or _RE_DIFFERENCE.search(s)
            or _RE_FEEL_COMPARATIVE.search(s)
            or _RE_CHARACTERIZE_RELATE.search(s)
            or _RE_MORE_LESS.search(s) or _RE_COMPARATIVE.search(s))

# --- SERVICE SUBJECTS: what a "change" can be ABOUT besides the music ----
#
# The adjacent-transition rule exists to stop the writer inventing a
# relationship between the two SONGS. It was applied to the whole text with
# no idea what any sentence was about, so it refused WEATHER breaks for
# saying the weather had shifted (fictional examples):
#
#   "Now there's a shift -- fifty-one degrees and grey..."
#   "...quite a shift from this morning's sunshine..."
#   "The forecast has taken a real shift since lunchtime..."
#
# A shift in the temperature is not a claim about two records. The fix is
# not to drop "shift" from the lexicon -- that would blind the guard to the
# real thing it catches -- but to ask WHAT THE SENTENCE IS ABOUT before
# judging it.
#
# A sentence is MUSIC-REFERRING when it carries a masked adjacent identity
# (§), an earlier-session identity (¤), or an ordinary noun for a piece of
# music. Only those sentences can assert a song relationship, so only those
# are judged. A sentence that refers to no music at all is exempted ONLY
# when this break was actually supplied a service subject and the copy
# really is talking about it -- otherwise the old whole-text behaviour
# stands and the guard fails closed.
_RE_MUSIC_NOUN = re.compile(
    r"\b(?:track|song|tune|record|album|set|cut|number|medley|riff|jam|"
    r"demo|single|band|group|artist|performance|recording|side)s?\b")

#: Pointing at a piece of music without naming it. "A shift into something
#: quieter, THIS ONE" is a claim about the track, and the guard must catch
#: it even when a temperature in the sentence before makes the whole break
#: look like weather commentary.
#:
#: Noun-phrase anaphors ONLY. A bare "that" or "this" is exactly how
#: ordinary weather copy opens ("that's a shift, isn't it?"), so including
#: them would undo the fix this scoping exists for.
_RE_MUSIC_ANAPHORA = re.compile(
    r"\b(?:th(?:is|at)\s+one|the\s+one|th(?:ese|ose)\s+two|the\s+pair"
    r"|both\s+of\s+(?:them|these|those))\b")

#: Words that mean the sentence is talking about a supplied SERVICE
#: reading. Grouped by the concept that has to have been offered, so a
#: break that was never given the weather cannot claim weather scope.
#: Deliberately concrete nouns and readings -- no adjective that also
#: appears in the sonic lexicon (`warm`, `cool`, `bright`, `dark`), so
#: this can never launder a claim about how two songs sound.
_SERVICE_MARKERS: dict[str, re.Pattern] = {
    "weather": re.compile(
        r"\b(?:degrees?|temperature|forecast|cloudy|overcast|rain(?:y|ing)?"
        r"|sun(?:ny|shine)?|snow(?:y|ing)?|wind(?:y)?|fog(?:gy)?|humid(?:ity)?"
        r"|drizzle|storm(?:y)?|sleet|showers?|skies|sky|breeze|weather"
        r"|overnight\s+low|high\s+of)\b"),
    "daylight": re.compile(
        r"\b(?:sunrise|sunset|daylight|dusk|dawn|twilight|nightfall"
        r"|the\s+light|gets?\s+dark|getting\s+dark)\b"),
    "station_time": re.compile(
        r"\b\d{1,2}:\d{2}\b|\bo'?clock\b|\b[ap]\.?m\.?\b"
        r"|\b(?:this\s+)?(?:morning|afternoon|evening|tonight|midday|noon)\b"),
}


def _music_referring(sentence: str) -> bool:
    return ("§" in sentence or "¤" in sentence
            or bool(_RE_MUSIC_NOUN.search(sentence))
            or bool(_RE_MUSIC_ANAPHORA.search(sentence)))


def _service_scoped(sentence: str, text: str, subjects) -> bool:
    """Is this sentence commentary on a SUPPLIED service reading rather
    than on the music?

    The marker may sit in a neighbouring sentence: copy often opens with
    "Well, that's a shift, isn't it?" and only then gives the reading. So
    the sentence itself must name no music, and the BREAK must be visibly
    about that service. Both halves are required -- a break with no
    service subject supplied gets no exemption at all."""
    if not subjects:
        return False
    for concept in subjects:
        pat = _SERVICE_MARKERS.get(concept)
        if pat is not None and pat.search(text):
            return True
    return False


# station / session habit claims -- never authorized by a run fact
_RE_HABIT = [
    re.compile(r"\bwe\s+(?:usually|typically|often|always|tend|keep|rarely"
               r"|like\s+to\s+play)\b"),
    re.compile(r"\bthis\s+station\s+(?:usually|typically|often|always|tends"
               r"|keeps|plays|loves|comes\s+back)\b"),
    re.compile(r"\b(?:seems?|appears?)\s+to\s+be\s+the\s+pattern\b"),
    re.compile(r"\bthe\s+pattern\s+(?:tonight|today|lately|this\s+\w+"
               r"|around\s+here)\b"),
    re.compile(r"\b(?:we|you)(?:'ve|\s+have)\s+been\s+(?:leaning|hearing"
               r"|playing|favoring|favouring|heavy\s+on|deep\s+in)\b"),
    re.compile(r"\ba\s+lot\s+of\s+\w[\w\s]{0,30}\b(?:in\s+the\s+(?:mornings?"
               r"|afternoons?|evenings?)|tonight|lately)\b"),
    re.compile(r"\bas\s+(?:usual|always|ever)\b"),
]


def _album_scope(ctx: dict[str, Any],
                 known_albums) -> tuple[list, list]:
    """Album facts are TRACK-SCOPED. Returns

      owners  -- [(album_lower, own_identities, other_identities)] for the
                 albums supplied THIS break, so a supplied album can be
                 checked against the track it was actually supplied for;
      foreign -- catalog album titles NOT supplied this break, which the
                 writer therefore has no authority to name at all.

    An album title appearing in recently_aired prose, in history, or in the
    model's memory is not a supplied fact for THIS break -- that is how an
    album by one act ends up attached to another act's track.
    """
    pairs = [(ctx.get("ending_track_album"), ctx.get("pre_name")),
             (ctx.get("next_track_album"), ctx.get("post_name"))]
    artists = [(n.split(" - ")[0].strip().lower()
                if isinstance(n, str) and " - " in n else None)
               for _a, n in pairs]
    # A shared artist identifies neither track: on a same-artist pair the
    # TITLE is the only key that says which record an album belongs to.
    shared = artists[0] is not None and artists[0] == artists[1]
    ident: list[set[str]] = []
    for _album, name in pairs:
        keys = set()
        if isinstance(name, str):
            for part in [name] + name.split(" - "):
                part = part.strip().lower()
                if len(part) >= 3 and not (shared and part in artists):
                    keys.add(part)
        ident.append(keys)
    owners = []
    supplied = set()
    for i, (album, _name) in enumerate(pairs):
        if not isinstance(album, str) or not album.strip():
            continue
        a = album.strip().lower()
        supplied.add(a)
        owners.append((a, ident[i], ident[1 - i] - ident[i]))
    foreign = sorted(
        (a for a in (known_albums or ())
         if a not in supplied and not any(a in s or s in a for s in supplied)),
        key=len, reverse=True)
    return owners, foreign


def _explicit_album_claim(raw: str, album: str) -> bool:
    """Does the (lower-cased) text make an EXPLICIT album claim about
    `album`? Only album grammar counts -- "from the X album", "the X album",
    "X's/'s X album", "from the album X", "off (of) X", "from X" -- never a
    bare mention, which may be the title track. The bare "from X"/"off X"
    forms need the same 8-character floor the general rule uses, so "from
    nine years ago" is not a claim about the album 'Nine'."""
    a = re.escape(album)
    tail = r"(?![\w/&-])"
    if re.search(r"\b(?:from|off(?:\s+of)?)\s+the\s+" + a + r"\s+album\b", raw):
        return True
    if re.search(r"(?:\bthe|['\u2019]s)\s+" + a + r"\s+album\b", raw):
        return True
    if re.search(r"\b(?:from|off(?:\s+of)?)\s+the\s+album,?\s+" + a + tail, raw):
        return True
    if len(album) >= 8 and re.search(
            r"\b(?:from|off(?:\s+of)?)\s+(?:the\s+)?" + a + tail, raw):
        return True
    return False


def authority_flags(context: dict[str, Any] | None,
                    resolved: list | None = None,
                    known_albums=None, known_artists=None,
                    all_albums=None) -> dict[str, Any]:
    """What the render context actually authorized, derived from the SAME
    keys the prompt builder uses -- the guard never re-decides authority, it
    only reads what the deterministic prompt layer already established.

    `resolved` is the same played-session list the prompt's
    earlier_this_session line is built from; its identities let the guard
    tell an EARLIER-session reference apart from the two ADJACENT transition
    tracks (which contrast authority is local to)."""
    ctx = dict(context or {})
    if "station" not in ctx and "station_name" not in ctx:
        try:
            import settings
            ctx["station"] = {"name": settings.resolved_station_name()}
        except Exception:
            pass
    adjacent = _mask_values(ctx)
    album_owners, foreign_albums = _album_scope(ctx, known_albums)
    # Albums the writer may put in EXPLICIT album grammar: none but the
    # supplied ones. This set carries every catalog title, title-track
    # collisions included -- the prose saying "album" removes the ambiguity
    # the general foreign_albums filter exists for. Supplied albums (and
    # titles nested in / containing them) stay with the owners rule.
    supplied = {a for a, _own, _other in album_owners}
    explicit_albums = sorted(
        (a for a in (all_albums or ())
         if a not in supplied and not any(a in s or s in a for s in supplied)),
        key=len, reverse=True)
    authorized = _authorized_artists(ctx)
    return {
        "explicit_albums": explicit_albums,
        # Artist identities this break may NAME: the ending and next artists
        # (and, for a request liner, the requested artist and the current
        # song's artist). Never recently_aired, never prior prose, never a
        # prompt example -- those are not facts about this break.
        "authorized_artists": authorized,
        "known_artists": {a for a in (known_artists or set())
                          if a and a not in authorized},
        "supplied_genres": _supplied_genres(ctx),
        # Which SERVICE readings this break was actually handed. Only these
        # can scope a "change" away from the music, so a break that was
        # never given the weather can never claim weather scope. Read from
        # the same context keys dj_context writes and service_presence
        # prunes: a concept withheld as not-due is simply absent here.
        "service_subjects": _service_subjects(ctx),
        "album_owners": album_owners,
        "foreign_albums": foreign_albums,
        # album rules apply only to breaks that actually discuss songs (a
        # news intro or an ack names no tracks and claims no albums)
        "has_song_facts": bool(ctx.get("pre_name") or ctx.get("post_name")),
        # mirrors the prompt's opportunity gate: only a real contrast opens
        # commentary on the move between the ADJACENT songs.
        "transition_allowed": ctx.get("transition_character") in (
            "contrast", "major_contrast"),
        "masks": adjacent,
        "earlier_masks": _earlier_values(resolved, adjacent),
        # --- fact gates -------------------------------------------
        # Is there an AUTHORITATIVE next track? This is the gate for
        # UNKNOWN_NEXT_NAMED and the forward-phrasing warning: with no
        # next song supplied, the writer has nothing to name or point at.
        "next_supplied": bool(ctx.get("post_name")),
        # Everything this break was actually given, as one lowercase
        # blob and as a token set. A claim whose words are already in
        # here is licensed; that is the whole gate.
        "fact_text": _blob(ctx, _FACT_KEYS),
        "fact_tokens": set(re.findall(r"[a-z']+", _blob(ctx, _FACT_KEYS))),
        "service_text": _blob(ctx, _SERVICE_KEYS),
        "calendar_supplied": isinstance(ctx.get("calendar_event"), dict)
        and bool(ctx.get("calendar_event")),
        "calendar_nouns": _calendar_nouns(ctx),
        # Character telemetry gate only -- never a refusal. See `warnings`.
        "character_supplied": bool(ctx.get("ending_track_character")
                                   or ctx.get("next_track_character")
                                   or ctx.get("transition_character")),
    }


def _service_subjects(ctx: dict[str, Any]) -> tuple[str, ...]:
    """The service concepts this render actually supplied."""
    out = []
    if ctx.get("current_weather") or ctx.get("weather_context"):
        out.append("weather")
    if ctx.get("daylight_line"):
        out.append("daylight")
    if ctx.get("station_time"):
        out.append("station_time")
    # licensed atmospheric colour offered this break (dj.atmosphere): the
    # sky or the light it speaks of was supplied, even with no owed reading
    for subject in ctx.get("atmosphere_subjects") or ():
        if subject in ("weather", "daylight") and subject not in out:
            out.append(subject)
    return tuple(out)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _authorized_artists(ctx: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("pre_name", "post_name", "current_song"):
        val = ctx.get(key)
        if isinstance(val, str) and " - " in val:
            out.add(_norm(val.split(" - ", 1)[0]))
    for key in ("requested_artist", "pre_artist", "post_artist"):
        val = ctx.get(key)
        if isinstance(val, str) and val.strip():
            out.add(_norm(val))
    return {a for a in out if len(a) >= 3}


def _supplied_genres(ctx: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("ending_track_character", "next_track_character"):
        char = ctx.get(key)
        if isinstance(char, dict) and char.get("genre"):
            out.add(_norm(str(char["genre"]).replace("_", " ")))
    return out


def _masked_keep_case(text: str, masks: list[str]) -> str:
    for m in masks:
        text = re.sub(re.escape(m), "§", text, flags=re.IGNORECASE)
    return text


def _unauthorized_artist(text: str, flags: dict[str, Any]) -> str | None:
    """A known catalog artist named in the text that this break did not
    authorize, or None. Adjacent identities are already masked out; the
    deterministic callback span is already replaced; what remains that
    matches the catalog is the writer's own invention."""
    known = flags.get("known_artists") or set()
    if not known:
        return None
    kept = _masked_keep_case(text, flags.get("masks") or [])
    normed = " " + _norm(kept) + " "
    for name in sorted(known, key=len, reverse=True):
        if name in _COMMON_WORD_ARTISTS:
            # capitalized, and not the first word of a sentence
            pat = (r"(?<![.!?]\s)(?<![.!?])(?<!^)\b" + re.escape(name.title())
                   + r"\b")
            if re.search(pat, kept.strip()) and not re.match(
                    re.escape(name.title()) + r"\b", kept.strip()):
                return name
            continue
        if f" {name} " in normed:
            return name
    return None


def _earlier_values(resolved: list | None, adjacent: list[str]) -> list[str]:
    """Lower-cased earlier-session identities (artist and title of PLAYED
    entries), excluding anything that is also an adjacent-transition
    identity: the ending track appearing in history never re-classifies it
    as 'earlier session'."""
    adj = {a.lower() for a in adjacent}
    out: set[str] = set()
    for r in (resolved or [])[-12:]:
        if not isinstance(r, dict) or r.get("outcome") != "played":
            continue
        for key in ("artist", "title"):
            v = r.get(key)
            if (isinstance(v, str) and len(v.strip()) >= 3
                    and v.strip().lower() not in adj):
                out.add(v.strip().lower())
    return sorted(out, key=len, reverse=True)   # longest first


#: Release qualifiers Music Assistant carries inside a title and a writer
#: never speaks: "(live at the Cinder Rooms, ...)", "[Remastered 2011]".
_RE_TITLE_QUALIFIER = re.compile(r"\([^)]*\)|\[[^\]]*\]")


def _mask_forms(out: set[str], val: str) -> None:
    """Every form of ONE supplied fact string that could appear in copy.

    THE SPOKEN FORM MATTERS, NOT JUST THE STORED ONE. Take a next track
    stored as
    `Marlow Kites - Left Turn (live at the Cinder Rooms, Port Avel - spring
    2011)`: splitting that on " - " breaks INSIDE the parenthetical,
    yielding `Left Turn (live at the Cinder Rooms, Port Avel` -- never the
    bare words the writer actually says. Without the bare form "Left Turn"
    would reach the patterns unmasked, match the move-noun lexicon (where
    it sits as a metaphor for an abrupt move), and
    `unauthorized_transition_relationship` would fire on a SONG TITLE,
    rejecting the Smart draft and the deterministic floor alike so the
    hosting slot produces nothing.

    Masking a supplied value is not leniency: a title is the writer's
    INPUT, never its claim, which is this module's founding rule. The mask
    list must describe the spoken string the check reads, not only the
    stored one."""
    out.add(val)
    for part in val.split(" - "):
        if len(part.strip()) >= 3:
            out.add(part.strip())
    bare = _RE_TITLE_QUALIFIER.sub(" ", val)
    bare = re.sub(r"\s+", " ", bare).strip(" -,;:/")
    if bare and bare != val:
        out.add(bare)
        for part in bare.split(" - "):
            if len(part.strip()) >= 3:
                out.add(part.strip())


def _mask_values(ctx: dict[str, Any]) -> list[str]:
    """Supplied fact strings (song / artist / album names) that must never be
    able to trip a pattern: they are the writer's inputs, not its claims."""
    out: set[str] = set()
    for key in ("pre_name", "post_name", "ending_track_album",
                "next_track_album"):
        val = ctx.get(key)
        if isinstance(val, str) and val.strip():
            _mask_forms(out, val.strip())
    st = ctx.get("station")
    st_name = None
    if isinstance(st, dict):
        st_name = st.get("name")
    if not st_name and isinstance(ctx.get("station_name"), str):
        st_name = ctx["station_name"]
    if not st_name:
        try:
            import settings
            st_name = settings.resolved_station_name()
        except Exception:
            pass
    if isinstance(st_name, str) and st_name.strip():
        s = st_name.strip()
        out.add(s)
        for part in re.split(r"\s+-\s+|\s*[/–—]\s*", s):
            if len(part.strip()) >= 3:
                out.add(part.strip())
    return sorted(out, key=len, reverse=True)   # longest first


def _masked(text: str, masks: list[str]) -> str:
    for m in masks:
        text = re.sub(re.escape(m), "§", text, flags=re.IGNORECASE)
    return text.lower()


def check(text: str, flags: dict[str, Any]) -> str | None:
    """None when the text makes no unauthorized kind of claim; otherwise the
    violated category.  Purely textual and deterministic -- no model, no
    world knowledge, no opinion about whether the songs really differ."""
    raw = (text or "").lower()
    # Spans this engine composed itself from verified facts (the
    # deterministic earlier-session callback) are authorized as written: the
    # artist in them is an ARTIST identity, never a claim about an album of
    # the same name (a self-titled record must not read as misattribution).
    # Only the exact composed span is exempt, so cross-track album
    # protection everywhere else in the break is untouched.
    for span in flags.get("authorized_spans") or []:
        if isinstance(span, str) and span.strip():
            raw = raw.replace(span.strip().lower(), " \u00a4span\u00a4 ")
    if flags.get("has_song_facts"):
        # Album fidelity. Two track-scoped rules,
        # both needing only the SUPPLIED track->album associations plus the
        # grounded catalog; neither judges whether an album claim is true in
        # the world, only whether this break was given that association.
        for album in flags.get("foreign_albums") or []:
            # An album belonging to some other record entirely: not supplied
            # here, so there is no track it could correctly be attached to.
            # The boundary excludes / & - so an album name sitting INSIDE a
            # longer supplied title ("Lost Bells/Gullwing") is not a claim
            # about the album 'Gullwing'.
            if re.search(r"(?<![\w/&-])" + re.escape(album) + r"(?![\w/&-])",
                         raw):
                return UNAUTHORIZED_ALBUM
        # Explicit album grammar names an album that this break did not
        # supply: an unauthorized attribution even when the title is also a
        # track ("Cold Orchard, from the Gullwing album").
        for album in flags.get("explicit_albums") or []:
            if _explicit_album_claim(raw, album):
                return UNAUTHORIZED_ALBUM
        for sentence in re.split(r"(?<=[.!?])\s+", raw):
            for album, own, other in flags.get("album_owners") or []:
                if any(k in sentence for k in own):
                    continue          # its own track is named here: fine
                # Cross-track attribution is the BINDING order "<other
                # track> ... <album>" at close range -- "Marlow Bay ...
                # Brass Weather". The reverse order is ordinary radio prose
                # ("From <ending album>, next up is <next artist>";
                # "Staying with <album>, <next title> is coming up") and is
                # never treated as an attribution.
                for key in other:
                    if re.search(re.escape(key) + r"[^.!?]{0,30}?"
                                 + r"(?<![\w/&-])" + re.escape(album)
                                 + r"(?![\w/&-])", sentence):
                        return UNAUTHORIZED_ALBUM
    # Artist identity: the writer may name only
    # the artists this break supplied. An act named in a prompt example
    # can otherwise be copied into a break on a day that act never
    # played. Authorized spans and adjacent masks are already gone
    # from `raw`/the masked text, so any catalog name left is unauthorized.
    span_text = text or ""
    for span in flags.get("authorized_spans") or []:
        if isinstance(span, str) and span.strip():
            span_text = re.sub(re.escape(span.strip()), " ¤span¤ ", span_text,
                               flags=re.IGNORECASE)
    if _unauthorized_artist(span_text, flags):
        return UNAUTHORIZED_ARTIST
    t = _masked(text or "", flags.get("masks") or [])
    # THE DIARY IS THE LISTENER'S. A writer can adopt the listener's
    # appointment as its own: "I've got choir practice at eight." Prompting
    # makes that rarer but cannot eliminate it, and it is
    # a FACTUAL ownership error rather than a matter of style: the DJ has
    # no appointments, so precision here is ~100% by construction.
    # "Your book club is at seven" is the
    # correct form and carries no first-person marker at all.
    #
    # JUDGED ON THE MASKED TEXT. A supplied title is the writer's INPUT,
    # never its claim, and "Ada Venn - Under My Lantern" otherwise hands
    # this rule a "my" of its own: "Under My Lantern is next, and your book
    # club is at seven" would read as ownership because an event noun
    # happens to follow within the window.
    if flags.get("calendar_supplied"):
        nouns = flags.get("calendar_nouns") or ()
        if nouns:
            for m in _RE_OWNS.finditer(t):
                window = t[m.end():m.end() + 40]
                if any(n in window for n in nouns):
                    return CALENDAR_OWNERSHIP
    if flags.get("has_song_facts"):
        # Genre label (the deterministic half of the existing rule): only a
        # listed character line's genre may be named, and only that label.
        # EVERY recognized label is inspected -- the first match alone would
        # let "from ska to metal" pass when only ska was listed.
        # Two refinements to a rule that already existed:
        # (a) the LICENCE is widened from "a character line's genre
        #     field" to "any supplied fact", the same gate as the others:
        #     a listed feel of "Rock, high energy." licenses "rock", and
        #     a supplied fact that says post-punk licenses post-punk.
        # (b) a genre word inside a capitalized MULTI-token name is part
        #     of the name, not a label. That is the usual false-positive
        #     shape (a band with "Soul" or "Folk" in its name).
        #     Masking already removes the SUPPLIED names; this
        #     also covers a name the packet never listed, which is a
        #     different complaint (UNAUTHORIZED_ARTIST, above).
        supplied_genres = flags.get("supplied_genres") or set()
        fact_text = flags.get("fact_text") or ""
        kept = _masked_keep_case(span_text, flags.get("masks") or [])
        name_spans = _proper_name_spans(kept)
        for m in _RE_GENRE_ANYCASE.finditer(kept):
            label = _norm(m.group(1))
            if label in supplied_genres or label in fact_text:
                continue
            if _inside_name(name_spans, m.start()):
                continue
            return UNAUTHORIZED_GENRE


    for m in flags.get("earlier_masks") or []:
        t = re.sub(re.escape(m), "¤", t)        # t is already lower-cased
    # Earlier-session comparisons are rejected under EVERY
    # transition class -- contrast authority is local to the adjacent
    # ending->next pair and never extends to characterizing or comparing an
    # earlier_this_session track. Checked sentence by sentence so an
    # authorized adjacent remark elsewhere in the break cannot launder an
    # earlier-anchored comparison.
    anchored_anywhere = False
    for sentence in re.split(r"(?<=[.!?])\s+", t):
        anchored = bool("¤" in sentence
                        or _RE_EARLIER_ANCHOR.search(sentence))
        anchored_anywhere = anchored_anywhere or anchored
        if not anchored:
            continue
        if _comparison_shape(sentence):
            return UNAUTHORIZED_TRANSITION
        # Characterizing the earlier entry at all -- no comparison
        # required, and no contrast connective required either. A sentence
        # that names ONLY an adjacent track (and no earlier identity) is
        # left alone: the ending/next tracks can have their own supplied
        # character, and this rule is about an earlier item that has none.
        if "§" in sentence and "¤" not in sentence:
            continue
        if (_RE_CHARACTER_ASCRIBED.search(sentence)
                or _RE_CHARACTER_FELT.search(sentence)):
            return UNAUTHORIZED_EARLIER_CHARACTER
    if anchored_anywhere and _RE_CHARACTER_ANAPHORA.search(t):
        return UNAUTHORIZED_EARLIER_CHARACTER
    if not flags.get("transition_allowed"):
        # Judged SENTENCE BY SENTENCE and only where the sentence is
        # actually about the music. A weather break saying the afternoon
        # shifted is not claiming anything about two records, and refusing
        # it would silence every weather break that mentions a change. The
        # protection itself is unchanged: a sentence that names either song
        # (or an earlier one, or any piece of music) is judged exactly as
        # before, and a break supplied no service subject is judged exactly
        # as before.
        subjects = flags.get("service_subjects") or ()
        for sentence in re.split(r"(?<=[.!?])\s+", t):
            if not (_RE_GOING_FROM.search(sentence)
                    or _RE_MOVE_NOUN.search(sentence)
                    or _comparison_shape(sentence)):
                continue
            if (not _music_referring(sentence)
                    and _service_scoped(sentence, t, subjects)):
                continue
            return UNAUTHORIZED_TRANSITION
    # NO AUTHORITATIVE NEXT TRACK, AND A NAMED ACT ANYWAY. The
    # escape is "Something a little more driving now, from The Lantern
    # Quarter." with no next_track supplied -- which the artist rule
    # above cannot catch, because it only knows acts in the local catalog
    # and that band is not in it. This asks the opposite question: a
    # capitalized name whose words are not in the supplied facts AT ALL
    # is an invention, catalog or no.
    #
    # A HARD BACKSTOP ON PURPOSE. A well-prompted writer rarely invents an
    # act; that is a prompt behaviour, not a guarantee, and
    # this is the one failure in the class that actually misleads a
    # listener.
    #
    # LAST, so the older and more precise complaints win: a draft that
    # both invents a relationship and names an unlisted act is reported
    # as the relationship claim it has always been reported as.
    #
    # BOTH GATES MUST BE POSITIVELY PRESENT. `next_supplied` missing
    # means the caller did not establish it, and an empty `fact_tokens`
    # means there is no supplied vocabulary to judge against -- in
    # either case every capitalized word would look invented, so the
    # rule declines to run rather than guess.
    fact_tokens = flags.get("fact_tokens") or set()
    if (flags.get("has_song_facts") and fact_tokens
            and "next_supplied" in flags and not flags["next_supplied"]):
        kept_i2 = _masked_keep_case(span_text, flags.get("masks") or [])
        for sentence in re.split(r"(?<=[.!?])\s+", kept_i2):
            for m in _RE_CAP_SPAN.finditer(sentence):
                cand = m.group(0).strip(" .,;:!?\u2014\u2013")
                toks = [t.lower() for t in re.findall(r"[A-Za-z']+", cand)]
                if not toks or len(cand) < 4:
                    continue
                if _RE_PRONOUN.match(cand):
                    continue
                # a lone capitalized word opening a sentence is just a
                # sentence opening
                if m.start() == 0 and len(toks) == 1:
                    continue
                if all(t in fact_tokens for t in toks):
                    continue
                if all(_ordinary(t) for t in toks):
                    continue
                return UNKNOWN_NEXT_NAMED
    for pat in _RE_HABIT:
        if pat.search(t):
            return UNAUTHORIZED_PATTERN
    return None


#: TELEMETRY ONLY. The musical-character lexicon, deliberately not
#: enforced: it is the largest warning class, a few common words carry
#: much of it, and its false-positive rate survives every
#: structural fix -- because separating "that song feels like a quiet
#: moment" (a claim about the song) from "a fitting moment for a little
#: quiet reflection" (a host's observation) needs intent, and intent is
#: not in the facts. Blocking on this would make the guard a style cop.
#: Prompting keeps the class small; what is left
#: is counted here and never refused.
_RE_CHARACTER = re.compile(
    r"\b(energy|energetic|mood|moody|tempo|texture|groove|shimmer\w*|"
    r"driving|propulsi\w+|frantic|urgent|calm|quiet|gentle|soft|loud|heavy|"
    r"light|slow|fast|smoky|sparse|lush|dense|raw|abrasive|polished|smooth|"
    r"rough|warm|cool|bright|dark|wistful|melanchol\w+|upbeat|sombre|somber|"
    r"tender|jangly|fuzzy|hazy|shiny|punchy|swagger|grit|intensity|intense|"
    r"delicate|brooding|anthemic)\b", re.IGNORECASE)

#: The two structural fixes: the adjective is sitting on
#: the WEATHER, or on the hosting MOMENT rather than on a record.
#: Matched against the tail that FOLLOWS the adjective, which begins
#: mid-gap (" one out there..."), so it must not open with \b -- that can
#: only succeed on a word character and made this exemption dead code.
_RE_WEATHER_SCOPED = re.compile(
    r"[^.]{0,30}\b(out there|outside|degrees?|sunny|rain\w*|wind\w*"
    r"|cloud\w*|afternoon|evening|morning|tonight)\b",
    re.IGNORECASE)
_RE_MOMENT_NOUN = re.compile(
    r"\b(quiet|calm|gentle)\s+(moment|reflection|hour|interlude|spell|time)\b",
    re.IGNORECASE)

WARN_UNKNOWN_NEXT_FORWARD = "w1_unknown_next_forward"
WARN_UNSUPPORTED_CHARACTER = "x1_unsupported_character"
WARN_UNSUPPORTED_BIOGRAPHY = "i4_unsupported_biography"


# --- WHOSE DIARY IT IS, beyond the first person (opt-in P2 writer) ----------
#
# `check` refuses the host adopting the listener's appointment ("I've got
# choir practice"). The P2 writer can make the OTHER half of that mistake:
# it hands the listener's entry to the whole audience -- "A few of you have
# book club tonight", "a reminder for anyone who might need it",
# "if you're like some of our listeners". The calendar record carries no
# owner or audience (dj_context.calendar_packet: id, title, start_time,
# minutes_until, timing_band, all_day, lead) and the configured calendar is
# the listener's own, so the only correct framings are the listener's
# ("your", "you've got") or neutral ("choir practice is at 8 PM").
#
# Judged per SENTENCE that names the event (one of its nouns, or the event
# made plural), on text whose other supplied names are masked, so "Some of
# you know this one" elsewhere in the break is untouched. Called ONLY by
# the P2 writer's screen (smart_dj.p2_content_screen); `check` -- and so
# the production planner -- is unchanged.
CALENDAR_AUDIENCE = "calendar_audience"          # some of you / anyone / listeners
CALENDAR_OWNER = "calendar_owner"                # I / my / we / our
CALENDAR_PLURAL = "calendar_plural"              # the one event made many
CALENDAR_CONDITIONAL = "calendar_conditional"    # "if you have a ..."
CALENDAR_DETAIL = "calendar_detail"              # a person or place never supplied

_RE_CAL_AUDIENCE = re.compile(
    r"\b(?:(?:some|a few|few|many|most|all|any|those|several|plenty|lots|a lot"
    r"|a bunch|none|each|the rest) of (?:you|y'?all|our listeners|our audience"
    r"|the listeners)"
    r"|anyone|anybody|everyone|everybody|everyone's|everybody's|whoever"
    r"|someone|somebody|no one|nobody|listeners?|audience|folks|people"
    r"|you all|y'?all|you guys|all you|those (?:who|with|of|needing|heading"
    r"|going|planning|that|among|lucky|attending)|for those|calendars)\b",
    re.IGNORECASE)
_RE_CAL_OWNER = re.compile(
    r"\b(?:i\s+(?:have|need|must|should|get|got)|i've\s+got|i'?m\s+(?:going"
    r"|heading|off|due|booked|expecting)|my|we\s+(?:have|need|must|should"
    r"|get|got)|we've\s+got|we'?re\s+(?:going|heading|off|due|booked"
    r"|expecting)|our)\b", re.IGNORECASE)
_RE_CAL_CONDITIONAL = re.compile(
    r"\bif you(?:'ve| have)?\s+(?:have|got|happen to have|happen to be|are one"
    r"|'re one)\s+(?:a|an|any|one|of)\b", re.IGNORECASE)
#: An honorific, or "with/at <Name>", that the record never carried.
_RE_CAL_HONORIFIC = re.compile(
    r"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Prof|Professor|Coach|Nurse)\.?\s+[A-Z]")
_RE_CAL_WITH_NAME = re.compile(
    r"\b(?:with|at|over at|down at|to see)\s+(?:the\s+)?([A-Z][\w'’-]+"
    r"(?:\s+[A-Z][\w'’-]+)*)")
_CAL_NOT_DETAIL = {"pm", "am", "noon", "midnight", "monday", "tuesday",
                   "wednesday", "thursday", "friday", "saturday", "sunday",
                   "today", "tonight", "tomorrow"}
#: "a heads-up for some", "worth knowing for some of you": the event made the
#: audience's without a quantified "of you".
_RE_CAL_FOR_SOME = re.compile(
    r"\bfor some(?:\s+of\s+(?:you|us))?\b(?!\s+(?:time|reason|while|of the))",
    re.IGNORECASE)
#: A sentence NEXT TO the calendar sentence belongs to
#: its passage only when it is about the event too: time-keeping or plans
#: language, or (right after it) a back-reference. That is what separates
#: "I bet some of you are watching the clock. Choir practice
#: starts at eight." (the event handed to the audience across a full stop)
#: from a greeting beside
#: a correct reminder ("Hope everyone's evening is going well. Your book
#: club is at 7.").
_RE_CAL_PASSAGE_CUE = re.compile(
    r"\b(?:clock|on time|in time|running late|run late|be late|late for"
    r"|keep(?:ing)? an eye|watch(?:ing)? the|schedules?|scheduled|plans?"
    r"|planning|appointments?|reminders?|remind(?:ing|er)?|heading (?:out|off"
    r"|over)|head(?:ing)? out|get(?:ting)? going|on (?:the|their|your) way"
    r"|somewhere to be|places to be|be there|mak(?:e|es|ing) it|miss(?:es"
    r"|ing)? (?:it|that|this|out)|marker|calendars?|diar(?:y|ies)"
    r"|commitments?|errands?|out the door|busy"
    r"|(?:of you|everyone|anyone|anybody|those)\s+(?:going|attending"
    r"|heading|headed|driving|joining))\b", re.IGNORECASE)
_RE_CAL_BACKREF = re.compile(r"^\W*(?:that|it|this)(?:'s|’s|\s+is|\s+was)\b",
                             re.IGNORECASE)
#: How far a passage reaches from the calendar sentence, each way.
CAL_PASSAGE_HOPS = 2

#: Title words that name nothing ("Dinner WITH Ines"): never an event noun.
_CAL_STOP = {"with", "from", "that", "this", "have", "your", "about", "after",
             "before", "into", "over", "then", "than", "them", "they", "will",
             "just", "some", "more", "time", "next", "here", "there", "what",
             "when", "where", "while"}


def _cal_event_words(event: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(event nouns, their plurals) -- the event's substantive words."""
    nouns = set(_calendar_nouns({"calendar_event": event})) - _CAL_STOP
    plurals = set()
    for n in nouns:
        plurals.add(n + ("es" if n.endswith(("s", "x", "ch", "sh")) else "s"))
        if n.endswith("y") and len(n) > 4:
            plurals.add(n[:-1] + "ies")
    title_words = set(re.findall(r"[a-z]{4,}",
                                 str(event.get("title") or "").lower()))
    return nouns, plurals - title_words


def calendar_misattribution(text: str, event: dict[str, Any] | None,
                            masks: list[str] | None = None) -> str | None:
    """'<category>:<words>' when a sentence naming the listener's calendar
    event frames it as anyone's but the listener's, or adds a person or
    place the record never supplied; else None. Deterministic; no model."""
    if not isinstance(event, dict) or not event.get("title"):
        return None
    nouns, plurals = _cal_event_words(event)
    if not nouns:
        return None
    title = str(event.get("title") or "")
    keep = text or ""
    for mv in sorted(set(masks or []), key=len, reverse=True):
        if mv and mv.strip() and mv.strip().lower() != title.lower():
            # whole words only: a title "One" must not eat "everyONE's"
            keep = re.sub(r"(?<!\w)" + re.escape(mv.strip()) + r"(?!\w)",
                          " § ", keep, flags=re.IGNORECASE)
    sentences = [x for x in re.split(r"(?<=[.!?…])\s+", keep) if x.strip()]
    named = []
    for i, sentence in enumerate(sentences):
        words = set(re.findall(r"[a-z]+", sentence.lower()))
        if words & nouns or words & plurals:
            named.append(i)
    for i in named:
        sentence = sentences[i]
        words = set(re.findall(r"[a-z]+", sentence.lower()))
        plural = sorted(words & plurals)
        if plural:
            return f"{CALENDAR_PLURAL}:{plural[0]}"
        m = _RE_CAL_AUDIENCE.search(sentence) or _RE_CAL_FOR_SOME.search(sentence)
        if m:
            return f"{CALENDAR_AUDIENCE}:{m.group(0).lower()}"
        # Ownership, as `check` judges it: the possessive or "we've got" must
        # take the event (an event noun within 40 characters after it), so
        # "your book club is at 7, and our next song is..." stays legal.
        for m in _RE_CAL_OWNER.finditer(sentence):
            window = sentence[m.end():m.end() + 40].lower()
            if set(re.findall(r"[a-z]+", window)) & nouns:
                return f"{CALENDAR_OWNER}:{m.group(0).lower()}"
        m = _RE_CAL_CONDITIONAL.search(sentence)
        if m:
            return f"{CALENDAR_CONDITIONAL}:{m.group(0).lower()}"
        m = _RE_CAL_HONORIFIC.search(sentence)
        if m:
            return f"{CALENDAR_DETAIL}:{m.group(0)}"
        for m in _RE_CAL_WITH_NAME.finditer(sentence):
            name = m.group(1)
            if (name.lower() not in title.lower()
                    and name.lower().split()[0] not in _CAL_NOT_DETAIL):
                return f"{CALENDAR_DETAIL}:{m.group(0)}"
    # The passage around each calendar sentence -- neighbours that are
    # about the event (_RE_CAL_PASSAGE_CUE, or a back-reference right after
    # it), up to CAL_PASSAGE_HOPS each way -- may not hand it to the audience.
    for i in named:
        for step in (-1, 1):
            for hop in range(1, CAL_PASSAGE_HOPS + 1):
                j = i + step * hop
                if j < 0 or j >= len(sentences) or j in named:
                    break
                near = sentences[j]
                if not (_RE_CAL_PASSAGE_CUE.search(near)
                        or (step == 1 and hop == 1
                            and _RE_CAL_BACKREF.search(near))):
                    break
                m = _RE_CAL_AUDIENCE.search(near) or _RE_CAL_FOR_SOME.search(near)
                if m:
                    return f"{CALENDAR_AUDIENCE}:adjacent:{m.group(0).lower()}"
    return None


def unsupported_biography(text: str, flags: dict[str, Any]) -> list[str]:
    """Biography/history this break was given nothing to support.

    THE DETECTOR AND THE GATE ARE UNCHANGED from when this refused copy;
    only the consequence moved. As a refusal its marginal value is small:
    almost everything it would refuse is already caught
    upstream by `ungrounded_claim`, which owns invented facts and
    states them more precisely. Against that it carries a real
    false-positive rate, much of it copy whose
    only fault is the model doing arithmetic between two SUPPLIED years
    ("a full two decades before"). So it is counted,
    not enforced.

    Three exemptions: the phrase is already in the
    supplied facts; it sits inside a capitalized multi-token NAME; or its
    neighbourhood echoes supplied SERVICE context ("back in the loft"
    with a supplied speaker reconnect).
    """
    if not flags.get("has_song_facts"):
        return []
    fact_text = flags.get("fact_text") or ""
    kept = _masked_keep_case(text or "", flags.get("masks") or [])
    spans = _proper_name_spans(kept)
    service_words = set(re.findall(r"[a-z]{4,}",
                                   flags.get("service_text") or ""))
    hits: list[str] = []
    for m in _RE_BIO.finditer(kept):
        phrase = m.group(0).lower().strip()
        if phrase in fact_text:
            continue
        if _inside_name(spans, m.start()):
            continue
        window = kept[m.start():m.start() + 40].lower()
        if service_words and any(w in window for w in service_words):
            continue
        hits.append(phrase)
    return hits


def warnings(text: str, flags: dict[str, Any]) -> list[str]:
    """What this ACCEPTED break is worth counting, and nothing more.

    NOTHING HERE REFUSES COPY. `check` decides acceptance; this runs
    afterwards on a break that already passed, so the two classes
    deliberately left unenforced are still observable in forensics.

    WARN_UNKNOWN_NEXT_FORWARD -- forward phrasing with no authoritative
    next track ("Something a little different coming up"). The detector
    is lexicon-bound: it misses some forward phrasing outright, and
    widening it starts flagging "something different for a rainy
    evening", which is about the weather. UNKNOWN_NEXT_NAMED already
    BLOCKS the case that actually misleads anyone -- a named invented
    act -- so this arm only counts vagueness.

    WARN_UNSUPPORTED_CHARACTER -- unsupported musical character. See
    `_RE_CHARACTER`.

    WARN_UNSUPPORTED_BIOGRAPHY -- unsupported biography/history. See
    `unsupported_biography`.

    Never raises: telemetry must not be able to cost a break.
    """
    out: list[str] = []
    try:
        t = text or ""
        if not flags:
            return []
        if not flags.get("next_supplied"):
            m = _RE_FORWARD.search(t)
            if m:
                tail = t[m.end():m.end() + 40]
                # forward phrasing about the WEATHER or the diary is
                # not a promise about a song; inflected forms included
                # ("a rainy evening"), so the count stays about music
                # rather than about the sky
                if not re.match(r"[^.]{0,30}\b(degrees?|rain\w*|sun\w*|"
                                r"wind\w*|cloud\w*|weather|pickup|"
                                r"deliver\w*|appointment)\b",
                                tail, re.IGNORECASE):
                    out.append(f"{WARN_UNKNOWN_NEXT_FORWARD}:"
                               f"{m.group(0).strip().lower()}")
        bio = unsupported_biography(t, flags)
        if bio:
            out.append(f"{WARN_UNSUPPORTED_BIOGRAPHY}:"
                       + ",".join(sorted(set(bio))))
        if not flags.get("character_supplied"):
            fact_text = flags.get("fact_text") or ""
            kept = _masked_keep_case(t, flags.get("masks") or [])
            spans = _proper_name_spans(kept)
            hits: list[str] = []
            for m in _RE_CHARACTER.finditer(kept):
                word = m.group(0).lower()
                if word in fact_text or _inside_name(spans, m.start()):
                    continue
                after = kept[m.end():m.end() + 34]
                if _RE_WEATHER_SCOPED.match(after):
                    continue
                if word in ("quiet", "calm", "gentle") \
                        and _RE_MOMENT_NOUN.search(kept) \
                        and not re.search(
                            r"\b(song|track|piece|record|that one|it)\b"
                            r"[^.]{0,20}\b" + re.escape(word), kept,
                            re.IGNORECASE):
                    continue
                hits.append(word)
            if hits:
                out.append(f"{WARN_UNSUPPORTED_CHARACTER}:"
                           + ",".join(sorted(set(hits))))
    except Exception:  # noqa: BLE001 - telemetry never costs a break
        return out
    return out


def repair_instruction(category: str) -> str:
    """One compact repair line appended to the SAME prompt for the single
    rerender.  It names the violated authority class in writer terms; it adds
    no facts and no jargon a listener could hear."""
    if category == UNAUTHORIZED_EARLIER_CHARACTER:
        return ("\n\nYour previous draft said how an earlier song sounded or "
                "felt. The session list tells you only WHICH songs played "
                "and how long ago -- never what they were like. Rewrite it "
                "mentioning an earlier song only by name and when it "
                "played, and describe mood or energy only for the songs "
                "whose own character is listed.")
    if category == UNAUTHORIZED_ARTIST:
        return ("\n\nYour previous draft named an artist who is not among "
                "the listed facts. The only artists you may name are the "
                "ending track's and the next track's, exactly as listed. "
                "Rewrite it without any other artist name and without any "
                "claim about who has been playing.")
    if category == UNAUTHORIZED_GENRE:
        return ("\n\nYour previous draft used a genre label that no listed "
                "character line carries. Rewrite it naming no genre except "
                "one listed for these songs, exactly as listed.")
    if category == UNAUTHORIZED_ALBUM:
        return ("\n\nYour previous draft attached an album to the wrong "
                "song. Rewrite it naming, for each song, only the album "
                "listed for that song -- an album named in an earlier break "
                "or belonging to another record is not this song's album, "
                "and where no album is listed, name none.")
    if category == UNKNOWN_NEXT_NAMED:
        return ("\n\nYour previous draft named a song or artist as coming "
                "up, but no next track is listed for this break. You do "
                "not know what plays next. Rewrite it without naming or "
                "implying any upcoming act, hosting instead from what you "
                "were given -- the song that just ended, the weather, the "
                "listener's event, or a listed fact.")
    if category == UNSUPPORTED_BIOGRAPHY:
        # currently unreachable -- biography warns rather than refuses. Retained
        # against a re-promotion; see the constant above.
        return ("\n\nYour previous draft made a claim about an artist's "
                "career, era or history that no listed fact supplies. "
                "Rewrite it saying nothing about when anyone formed, what "
                "came earlier or later, or who influenced whom -- name the "
                "music and the moment instead.")
    if category == CALENDAR_OWNERSHIP:
        return ("\n\nYour previous draft took the listener's calendar "
                "entry as your own. The diary belongs to them and you have "
                "no appointments. Rewrite it in the second person -- "
                "'your grocery delivery is expected between two and four', "
                "never 'I have a dentist appointment'.")
    if category == UNAUTHORIZED_PATTERN:
        return ("\n\nYour previous draft claimed a station or session habit "
                "that is not among the listed facts. Rewrite it using only "
                "the listed facts, without saying what usually, often, or "
                "lately gets played.")
    return ("\n\nYour previous draft characterized the musical relationship "
            "or the move between the songs, which is not among the listed "
            "facts. Rewrite it using only the listed facts: name the songs "
            "and hand off without comparing them or describing the shift "
            "between them.")
