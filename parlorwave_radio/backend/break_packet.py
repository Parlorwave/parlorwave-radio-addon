"""The grounded break packet: every fact and every behaviour constraint a
DJ break may be written from, as one pure, typed structure.

Parlorwave Radio owns the facts; a writer -- deterministic, the current
Ollama writer, or a future Home Assistant / cloud writer -- only turns the
SAME packet into prose. Nothing here fetches anything: the packet is
composed from inputs the scheduler has already resolved (the render
context, the adjacent tracks, the service facts), so a writer that reads
only the packet can be proven never to invent a fact it was not given.

`render_prompt` is the existing DJ prompt, moved here verbatim and bound
to the packet instead of the loose context dict: the Ollama writer keeps
receiving byte-for-byte the prompt it received before this module
existed (tests/fixtures/prompt_goldens pins that). `build_break_prompt`
in dj_engine is now `render_prompt(build_packet(...))`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any

import dj_persona

#: same literals dj_engine uses; a test pins them equal
PURPOSE_SCHEDULED = "scheduled"
PURPOSE_STATION_REQUEST = "station_request"
MAX_WORDS = 60
ACK_MAX_WORDS = 30

#: the placeholder the legacy pre-name fallback prints for a missing field
_UNKNOWN = "Unknown"


# --- what a title SOUNDS like when a person says it ---------------------------
#
# A deterministic floor that reads the database field aloud says:
#
#     "Cold Orchard, Slow Ferry (live at the Cinder Rooms,
#      Port Avel - spring 2011)."
#
# That parenthesis is provenance -- where and when this particular
# recording was captured. A DJ says "Slow Ferry".
#
# WHAT THIS DELIBERATELY DOES NOT DO: strip parentheses. A great many
# song titles carry a meaningful one -- a leading phrase, a subtitle,
# "Lost Bells (Ask Teodor Vance)" -- and
# removing it would misname the song, which is a worse failure than
# reading an edition tag. So a bracketed segment is removed only when its
# whole content matches a KNOWN provenance shape below, and only from the
# END of the title where such tags live. Anything unrecognised stays: the
# song keeps its name.
_PROVENANCE = (
    # live captures, with or without a venue/date tail
    r"live",
    r"live\s+(?:at|in|from|on|version|recording|bootleg|session)\b.*",
    r".*\brecorded\s+live\b.*",
    r".*\blive\s+(?:at|in|from)\b.*",
    # remasters, editions and reissues
    r"(?:\d{4}\s+)?(?:digital\s+|hd\s+)?remaster(?:ed)?"
    r"(?:\s+version)?(?:\s+\d{4})?",
    r"remastered\s+\d{4}",
    r"(?:deluxe|expanded|anniversary|special|collector['’]?s|legacy)"
    r"(?:\s+edition|\s+version)?",
    r"\d{4}\s+(?:remaster|remix|version|edition|mix|re-?recording)",
    r"re-?issue(?:d)?(?:\s+\d{4})?",
    # edits, mixes and alternate cuts
    r"(?:radio|single|album|clean|explicit|edited|short|long|7in|12in)"
    r"\s*(?:edit|version|mix|cut)",
    r"(?:[\w'’\-]+\s+){0,3}(?:mix|remix)",
    r"(?:mono|stereo)(?:\s+version|\s+mix)?",
    r"(?:original|album)\s+version",
    r"acoustic(?:\s+version)?",
    r"unplugged.*",
    r"(?:[\w'’\-]+\s+){0,3}demo(?:\s+version)?",
    r"rough\s+mix|work\s+tape|outtake|rehearsal|alternate\s+take|take\s+\d+",
    r"bonus\s+track.*",
    r"re-?recorded(?:\s+version)?|taylor['’]?s\s+version",
)
_PROVENANCE_RE = re.compile(r"\s*(?:" + "|".join(_PROVENANCE) + r")\s*\Z",
                            re.IGNORECASE)
#: A trailing bracketed segment, and a trailing ' - ' tail. Those are the
#: two places a catalog puts an edition tag; nothing else is touched.
_TRAILING_BRACKET_RE = re.compile(r"\s*[\(\[]([^()\[\]]*)[\)\]]\s*\Z")
_TRAILING_DASH_RE = re.compile(r"\s+[-–—]\s+([^-–—]+)\Z")


def spoken_title(title: str | None, canonical: str | None = None) -> str:
    """The title as a HOST would say it.

    `canonical` -- a clean title the pack or the catalog already resolved
    -- wins outright when supplied; deriving what is already known is how
    two spellings of one song end up on air. Otherwise known
    edition/provenance tags are removed from the end of the title, one
    segment at a time, and everything else is left exactly as it is.
    Never returns empty: a title that is nothing but provenance keeps its
    original text, because misnaming a song beats having no name for it.
    """
    clean = str(canonical or "").strip()
    if clean:
        return clean
    original = str(title or "").strip()
    out = original
    for _ in range(3):                    # at most three stacked tags
        for rx in (_TRAILING_BRACKET_RE, _TRAILING_DASH_RE):
            m = rx.search(out)
            if m and _PROVENANCE_RE.fullmatch(" " + m.group(1).strip()):
                out = out[:m.start()].rstrip(" -–—")
                break
        else:
            break
    out = out.strip()
    return out or original


def split_name(name: str | None) -> tuple[str, str]:
    """'Artist - Title' -> (artist, SPOKEN title); a bare name is a title.
    The legacy 'Unknown' placeholder is NOT a fact and becomes ''.

    The title comes back in its spoken form (`spoken_title`), because this
    is the split every WRITER builds sentences from -- the raw MA name
    stays available as `ending_name`/`next_name` for identity work, where
    the catalog's exact string is the point."""
    if name and " - " in name:
        artist, title = name.split(" - ", 1)
    else:
        artist, title = "", (name or "")
    artist, title = artist.strip(), title.strip()
    if artist == _UNKNOWN:
        artist = ""
    if title == _UNKNOWN:
        title = ""
    return artist, spoken_title(title)


@dataclass(frozen=True)
class BreakPacket:
    """Facts and constraints only. Every field is either a resolved fact
    the scheduler supplied or a behaviour constraint already in force;
    '' / None / () mean 'not available' and a writer must abstain."""

    # what this copy is for
    purpose: str = PURPOSE_SCHEDULED
    max_words: int = MAX_WORDS
    target_words: tuple[int, int] | None = None
    # station / persona (resolved: never a missing profile or level)
    persona_profile: str = dj_persona.DEFAULT_PROFILE
    chattiness: str = dj_persona.DEFAULT_CHATTINESS
    # track facts -- ending_name/next_name are the MA names as listed to the
    # writer ("Artist - Title"); artist/title are the split, placeholder-free
    # forms a deterministic writer builds sentences from
    ending_name: str = ""
    ending_artist: str = ""
    ending_title: str = ""
    ending_album: str = ""
    ending_character: dict[str, Any] | None = None
    next_name: str = ""
    next_artist: str = ""
    next_title: str = ""
    next_album: str = ""
    next_character: dict[str, Any] | None = None
    transition_character: str | None = None
    album_run: bool = False
    album_sequence: str | None = None
    semantic_themes: tuple[str, ...] = ()
    semantic_tone: str = ""
    # clock facts
    station_time: str = ""
    station_time_forms: tuple[str, ...] = ()
    daypart: str = ""
    # environmental facts (already validated by dj_context / weather_context)
    current_weather: str = ""
    high_state: str = ""
    weather_changed: bool = False
    daylight_line: str = ""
    daylight_phase: str = ""
    daylight_salient: bool = False
    # optional editorial extras the deterministic topic editor selected
    weather_context: str = ""
    upcoming_event: str = ""
    news_headline: str = ""
    # continuity
    recent_break_texts: tuple[str, ...] = ()
    #: The `writer.template_id` of the last few AIRED breaks, newest
    #: first. Deterministic-writer rotation memory: without it the
    #: bank is memoryless and the same two track names always pick the
    #: same shape. A composite id ('both.x+clock.y') is stored whole and
    #: compared on its head.
    recent_template_ids: tuple[str, ...] = ()
    # constraints
    required_service: tuple[str, ...] = ()     # service_presence.due(...)
    #: The Smart DJ path's bounded calendar record (dj_context.calendar_packet):
    #: {id, title, start_time, minutes_until, timing_band, all_day, lead}.
    #: Present ONLY when an event is inside a band it has not already used,
    #: so presence IS due-ness -- there is no separate flag to disagree with.
    #: Carried here so a deterministic host can state a due event when the
    #: model's enrichment is rejected; every value is pre-computed in code.
    calendar_event: dict[str, Any] | None = None
    # request liner facts (purpose == station_request only)
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def has_track_facts(self) -> bool:
        return bool(self.ending_name or self.next_name)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _s(value: Any) -> str:
    return str(value) if value else ""


def build_packet(session: dict[str, Any] | None,
                 resolved: list[dict[str, Any]] | None,
                 current: dict[str, Any] | None,
                 context: dict[str, Any] | None = None) -> BreakPacket:
    """The packet for one render, from the SAME inputs build_break_prompt
    always took. Pure: no settings, no database, no network. `session` and
    `resolved` are accepted for signature parity; the prompt never read
    them (session history is deliberately not writer material)."""
    ctx = context or {}
    if ctx.get("purpose") == PURPOSE_STATION_REQUEST:
        profile = dj_persona.get_profile(ctx.get("persona_profile"))
        return BreakPacket(
            purpose=PURPOSE_STATION_REQUEST,
            max_words=int(ctx.get("max_words") or ACK_MAX_WORDS),
            target_words=ctx.get("target_words") or (6, 16),
            persona_profile=profile["persona_id"],
            # Personality no longer carries an airtime default; a liner
            # is always written at the production level.
            chattiness=dj_persona.PRODUCTION_CHATTINESS,
            request={k: ctx.get(k) for k in ("requested_artist",
                                             "current_song",
                                             "first_new_track")},
            recent_break_texts=tuple(ctx.get("recent_break_texts") or ()),
            recent_template_ids=tuple(ctx.get("recent_template_ids") or ()),
        )
    max_words = int(ctx.get("max_words") or MAX_WORDS)
    profile = dj_persona.get_profile(ctx.get("persona_profile"))
    # An explicit context level is the internal/research seam and still
    # wins; with none, the answer is the production level -- NEVER the
    # profile's own default, which is what used to make Late Night a
    # cadence setting.
    chattiness = ctx.get("chattiness")
    if chattiness not in dj_persona.CHATTINESS_LEVELS:
        chattiness = dj_persona.PRODUCTION_CHATTINESS
    target_words = ctx.get("target_words")
    if not target_words and max_words:
        wb = dj_persona.word_budget(max_words, chattiness)
        target_words = (wb["target_low"], wb["target_high"])
    pre = ctx.get("pre_name")
    if not pre and current and (current.get("title") or current.get("artist")):
        pre = (f"{current.get('artist') or _UNKNOWN} - "
               f"{current.get('title') or _UNKNOWN}")
    post = ctx.get("post_name")
    ending_artist, ending_title = split_name(pre)
    next_artist, next_title = split_name(post)
    import service_presence
    return BreakPacket(
        purpose=PURPOSE_SCHEDULED,
        max_words=max_words,
        target_words=target_words,
        persona_profile=profile["persona_id"],
        chattiness=chattiness,
        ending_name=_s(pre),
        ending_artist=ending_artist,
        ending_title=ending_title,
        ending_album=_s(ctx.get("ending_track_album")),
        ending_character=(dict(ctx["ending_track_character"])
                          if ctx.get("ending_track_character") else None),
        next_name=_s(post),
        next_artist=next_artist,
        next_title=next_title,
        next_album=_s(ctx.get("next_track_album")),
        next_character=(dict(ctx["next_track_character"])
                        if ctx.get("next_track_character") else None),
        transition_character=ctx.get("transition_character"),
        album_run=bool(ctx.get("album_run")),
        album_sequence=ctx.get("album_sequence"),
        semantic_themes=tuple(ctx.get("semantic_themes") or ()),
        semantic_tone=_s(ctx.get("semantic_tone")),
        station_time=_s(ctx.get("station_time")),
        station_time_forms=tuple(ctx.get("station_time_forms") or ()),
        daypart=_s(ctx.get("daypart")),
        current_weather=_s(ctx.get("current_weather")),
        high_state=_s(ctx.get("high_state")),
        weather_changed=bool(ctx.get("weather_changed")),
        daylight_line=_s(ctx.get("daylight_line")),
        daylight_phase=_s(ctx.get("daylight_phase")),
        daylight_salient=bool(ctx.get("daylight_salient")),
        weather_context=_s(ctx.get("weather_context")),
        upcoming_event=_s(ctx.get("upcoming_event")),
        news_headline=_s(ctx.get("news_headline")),
        recent_break_texts=tuple(ctx.get("recent_break_texts") or ()),
        recent_template_ids=tuple(ctx.get("recent_template_ids") or ()),
        required_service=tuple(ctx.get("required_service") or service_presence.due(ctx, max_words)),
        calendar_event=(dict(ctx["calendar_event"])
                        if isinstance(ctx.get("calendar_event"), dict)
                        else None),
    )


def character_line(char: dict[str, Any]) -> str:
    bits = []
    if char.get("genre"):
        bits.append(str(char["genre"]).replace("_", " "))
    if char.get("energy"):
        bits.append(f"{char['energy']} energy")
    for flag in ("live", "demo", "instrumental"):
        if char.get(flag):
            bits.append(f"a {flag} recording" if flag != "instrumental"
                        else "instrumental")
    return ", ".join(bits) or "unremarkable"




def _request_ack_prompt(p: BreakPacket) -> str:
    """The Start Radio liner: a quick spoken 'you got it', nothing else --
    the ack prompt moved from dj_engine unchanged (see its history)."""
    lines = ["requested_artist (what the listener just asked for more of): "
             f"{p.request.get('requested_artist')}"]
    if p.request.get('current_song'):
        lines.append("current_song (the song finishing as your words begin): "
                     f"{p.request['current_song']}")
    if p.request.get('first_new_track'):
        lines.append("first_new_track (starts right after you finish): "
                     f"{p.request['first_new_track']}")
    max_words = p.max_words
    profile = dj_persona.get_profile(p.persona_profile)
    return (
        "You are the DJ on the listener's personal radio station. "
        "The listener just asked you to keep the station going with more "
        "like what is playing. Write ONE short spoken acknowledgement, "
        f"maximum {max_words} words, plain spoken prose only.\n"
        "Acknowledge the request in your own voice and hand straight back "
        "to the music. You may name the artist, refer to current_song in "
        "the past tense (it ends as you speak), and point at "
        "first_new_track as what is next -- naming that title is optional.\n"
        + dj_persona.ack_line(profile) + "\n"
        "Use ONLY the facts below. Do NOT invent album, year, chart, "
        "biography, origin, recording or any other artist facts, and do "
        "NOT mention the time, weather, news, the listener's calendar, or "
        "other songs from this session. A joke or an opinion is fine; a "
        "fact is not. No lists, no markdown, no stage directions.\n"
        "Facts:\n" + "\n".join(lines)
        + "\nWrite only the spoken acknowledgement.")




def render_prompt(p: BreakPacket) -> str:
    """v2 music-only grounded prompt with EXPLICIT temporal roles -- the
    Ollama writer's exact input, moved here from dj_engine.build_break_prompt
    and bound to the packet. See that function's history for the reasoning
    behind every rule; none of it changed in the move."""
    if p.purpose == PURPOSE_STATION_REQUEST:
        return _request_ack_prompt(p)
    max_words = p.max_words
    profile = dj_persona.get_profile(p.persona_profile)
    pre = p.ending_name
    post = p.next_name
    lines = []
    if pre:
        lines.append("ending_track (the song finishing at the moment your "
                     f"words begin -- it has JUST ended): {pre}")
        if p.ending_album:
            lines.append("ending_track_album (grounded catalog fact): "
                         f"{p.ending_album}")
        if p.ending_character:
            lines.append("ending_track_character (derived descriptive "
                         "color, not literal fact): "
                         + character_line(p.ending_character))
    if post:
        lines.append(f"next_track (starts right after you finish): {post}")
        if p.next_album:
            lines.append("next_track_album (grounded catalog fact): "
                         f"{p.next_album}")
        if p.next_character:
            lines.append("next_track_character (derived descriptive "
                         "color, not literal fact): "
                         + character_line(p.next_character))
    # The transition-commentary OPPORTUNITY is closed unless a
    # real contrast exists. For continuous / gentle_change / unknown the
    # writer is handed NO basis to describe the move at all -- ending_track
    # and next_track already name both songs, and a clean handoff needs
    # nothing more. (A transition_context line for these classes invites
    # the model to paraphrase it into something like "a gentle step over"
    # -- an offered opportunity a negative rule cannot
    # undo.) An album/artist run is a separate, real continuation and is
    # still surfaced as such.
    _tc = p.transition_character
    if _tc in ("contrast", "major_contrast"):
        desc = ("a strong contrast in sound" if _tc == "major_contrast"
                else "a genuine contrast in sound")
        lines.append("transition_context (the ONLY basis for how you "
                     "describe the move between songs; an internal "
                     f"description, NOT words to repeat): {desc}")
    elif p.album_run:
        desc = "the same album continuing" + (
            " in normal order" if p.album_sequence == "ordered"
            else "")
        lines.append("album_run (a real run you MAY note as a continuation, "
                     f"never as a surprise): {desc}")
    if p.semantic_themes:
        bits = ", ".join(p.semantic_themes)
        tone = p.semantic_tone or ""
        lines.append("ending_track_feel (derived from the song itself; "
                     f"optional color, concepts only): themes {bits}"
                     + (f"; tone {tone}" if tone else ""))
    if p.station_time:
        # The clock and the part of the day are separate facts. Listing
        # them glued together ("8:40 PM (evening)") teaches the writer to
        # say them glued together -- "It's 8:40 PM, evening." -- a bare
        # tag that lands with continuation prosody (a clock sentence
        # ending on ", evening" / ", overnight").
        lines.append("station_time (the local time at the listener's home "
                     f"right now): {p.station_time}")
        if p.daypart:
            lines.append("daypart (which part of the day it is; colour for "
                         "phrasing elsewhere in the break, never a tag "
                         f"appended to the clock): {p.daypart}")
    if p.current_weather:
        lines.append("current_weather (fresh service weather at the "
                     f"listener's home): {p.current_weather}")
    if p.daylight_line:
        lines.append("daylight (resolved from sun position, authoritative; "
                     "context for your phrasing, never a line to recite): "
                     f"{p.daylight_line}")
    # Session history is NOT writer material: any earlier-session
    # callback is composed deterministically in dj_callback and joined to
    # the finished text outside the model, so no identity, sentence or
    # placeholder for it appears here at all.
    # Optional contextual roles, present ONLY when the deterministic topic
    # editor selected them (dj_context.select_topics). Each carries its own
    # semantics so the LLM never has to infer what a fact represents.
    if p.weather_context:
        lines.append("weather_context (current conditions at the listener's "
                     f"home right now): {p.weather_context}")
    if p.upcoming_event:
        lines.append("upcoming_event (on the listener's own calendar, "
                     f"starting soon): {p.upcoming_event}")
    if p.news_headline:
        lines.append("news_headline (one recent real headline from the "
                     f"listener's news source): {p.news_headline}")
    recent_texts = p.recent_break_texts or []
    if recent_texts:
        joined = " || ".join(recent_texts)
        lines.append("recently_aired (your last few breaks, about OTHER "
                     "songs -- prose to avoid repeating, NEVER a fact "
                     "source: any song, artist or album named in here "
                     "belongs to those breaks, not to the songs above, and "
                     "may not be reused as a fact; DO NOT REUSE their "
                     "observations, setups, or rhetorical constructions): "
                     f"{joined}")
    facts = "\n".join(lines) or "No track facts available."
    # Each service instruction exists ONLY when its fact does: an
    # instruction referencing an absent fact invites the model to invent it
    # (an instruction to "state the listed station_time" with no
    # station_time listed produces a fabricated clock).
    service_rule = ""
    if pre or post:
        service_rule += (
            "Music facts: the listed artist/title/album lines are the ONLY "
            "facts about these songs and artists you may state. Never "
            "invent or recall hometowns, band members, chart positions, "
            "producers, studios, release years, songwriting stories, or "
            "what a song is 'about' -- your own knowledge of these artists "
            "is NOT a source. Derived character lines may color your "
            "wording (a mood, an energy) within their own scope -- "
            "'keeping the energy up' from a high energy is fine -- but "
            "never assert concrete musical detail you were not given: no "
            "claims about chords, riffs, instrumentation, tempo, key, song "
            "structure, vocals, production, recording, lyrics, imagery, "
            "theme, or genre/subgenre unless a listed fact states it. A "
            "listed character line's genre is the ONLY genre label you may "
            "use, exactly as given -- never narrow, broaden, or relabel it: "
            "a listed 'rock' does not license 'hard rock', 'punk', 'grunge', "
            "'metal' or any other label, and with no listed genre you name "
            "no genre at all. When no song-feel line "
            "is listed, you have no lyric or theme information at all: do "
            "not describe imagery, mood-of-the-words, or what the song "
            "evokes from memory.\n")
    if pre and post:
        tc = p.transition_character
        if tc in ("contrast", "major_contrast"):
            service_rule += (
                "Transition: there is a REAL "
                + ("strong " if tc == "major_contrast" else "")
                + "contrast between these songs; you MAY describe it, "
                "grounded only in the listed character (mood/energy/"
                "genre), never quoting transition_context's wording.\n")
        elif p.album_run:
            service_rule += (
                "Transition: the SAME ALBUM is continuing -- fine to note "
                "we are staying with the artist or keeping the album "
                "rolling"
                + (" in normal album order"
                   if p.album_sequence == "ordered" else "")
                + "; never call same-album ordering unexpected or random.\n")
        # else (continuous / gentle_change / unknown, no run): NO transition
        # instruction and NO transition_context fact are emitted. The move is
        # not a topic -- the writer simply names ending_track and next_track
        # and hands off. We do not ask it to "avoid" the move; there is
        # nothing offered to describe.
    # The smarter-DJ priority + anti-filler stance, stated once whenever
    # there is music to talk about.
    if pre or post:
        service_rule += (
            "Priority for what to say: a real station/session pattern (an "
            "album or artist run) first; then a real "
            "musical contrast if the facts show one; then an eligible song "
            "feel; then genuinely useful, CHANGED service context; "
            "otherwise just a clean, short handoff. Silence about an "
            "uninteresting fact beats manufactured cleverness, and a "
            "music-only handoff is a perfectly good break. Do not reuse an "
            "observation, angle, or phrasing from recently_aired.\n")
        # Station-history / programming-tendency claims are
        # authority claims and need a supplied fact. The model's own sense
        # of what radio stations "usually" do is not a source.
        service_rule += (
            "Station-history claims need a listed fact: never say what this "
            "station 'usually', 'often', 'tends to', 'typically', 'always', "
            "or 'keeps' playing -- by genre, artist, era, mood, or time of "
            "day (e.g. 'we play a lot of rock in the mornings', 'this "
            "station always comes back to the same band', 'we've been at "
            "this all night') -- unless a listed run fact establishes it. A "
            "listed album_run fact CAN support a specific grounded remark "
            "('the last few have all been from the same record'); your "
            "general sense of programming patterns cannot. Name no artist "
            "other than the ending and next artists listed below.\n")
    if p.semantic_themes:
        service_rule += (
            "The ending_track_feel line is optional flavor for ONE brief, "
            "natural observation at most (e.g. 'a little darker and more "
            "reflective there'). Never quote or paraphrase lyrics, never "
            "explain what the song means or its story, never claim the "
            "song refers to the actual weather or time -- at most note "
            "that its mood fits the moment. Skipping it entirely is "
            "fine.\n")
    required = p.required_service
    if p.station_time and "station_time" in required:
        forms = p.station_time_forms or []
        service_rule += (
            "Time check (REQUIRED this break): state the local time once, "
            "briefly and naturally, as station_time is listed "
            f"('{p.station_time}')"
            + (" or in one of these equivalent forms: "
               + " / ".join(f"'{f}'" for f in forms) if forms else "")
            + ". Never any other time, never round it, never 'about', "
            "'almost' or 'nearly' -- only the listed clock or a listed "
            "form. Make the clock land cleanly: a complete sentence that "
            f"ends on the time ('It's {p.station_time}.') or a clause "
            "that genuinely goes on. Never tack the part of the day onto "
            "the clock as a bare tag -- not '"
            f"{p.station_time}, {p.daypart or 'evening'}', not "
            f"'{p.station_time} here, {p.daypart or 'evening'}' "
            "-- and do not end a sentence on 'in the evening' or 'in the "
            "morning'. If you name the part of the day, 'this evening' "
            "beside the clock or its own sentence is fine.\n")
    elif p.station_time:
        service_rule += (
            "station_time is OPTIONAL BACKGROUND, not a required talking "
            "point: do not announce the time merely because it is listed. "
            "A DJ can know the clock without calling it out every break -- "
            "mention it only when it genuinely adds something (a real "
            "daypart moment), and never state or guess a time that is not "
            "listed.\n")
    if p.current_weather:
        if "weather" in required:
            service_rule += (
                "Weather (REQUIRED this break): give current_weather once, "
                "in one short sentence"
                + (" -- it has CHANGED since it was last on air"
                   if p.weather_changed else "")
                + ". Use the listed numbers and keep their TENSE exactly as "
                "written -- the ")
        else:
            service_rule += (
                "current_weather is OPTIONAL BACKGROUND, not required "
                "material: do not mention weather merely because it is "
                "listed, and especially do not repeat weather a recent break "
                "already gave unless it materially changed. WHEN you do "
                "mention it, use the listed numbers and keep their TENSE "
                "exactly as written -- the ")
        service_rule += (
            "phrasing already says whether the high is still ahead "
            "('expected'), happening ('right around'), or over ('topped "
            "out'), and whether the low is tonight's. ")
        if p.high_state == "passed":
            service_rule += (
                "Today's high already happened: never present it as "
                "expected, upcoming, or still ahead. ")
        elif p.high_state == "upcoming":
            service_rule += (
                "The high is a forecast still ahead: do not say it already "
                "happened. ")
        else:
            # No high fact was supplied. The current temperature ALONE
            # must never let the writer reconstruct a high relationship
            # (say, 'close to today's high' with no high fact behind it).
            service_rule += (
                "No high or low relationship is listed: do NOT say or "
                "imply today's high or low, proximity to either, whether a "
                "peak has passed, forecast timing, or a precipitation "
                "trend. The current temperature is just the current "
                "temperature -- it is not 'near today's high'. ")
        service_rule += (
            "Never invent timing, history, or numbers beyond what is "
            "listed; a quip around them is welcome, but never replace, "
            "obscure or contradict them.\n")
    if p.daylight_line:
        if p.daylight_salient:
            service_rule += (
                "The light has CHANGED since it was last mentioned on air: "
                "one brief, natural mention of the listed daylight line is "
                "welcome here. ")
        service_rule += (
            "Daylight is OPTIONAL BACKGROUND: do not tack it on as a "
            "sign-off (e.g. ending with 'it's dark out') -- mention light "
            "only if it genuinely serves the moment. Daylight facts are "
            "context, not announcements: never recite the listed line as "
            "a sentence of its own. The listed daylight line is the ONLY "
            "light level you may imply. ")
        if p.daylight_phase == "dark":
            # Left alone, most 'dark' mentions are the bare fact string,
            # "It is dark out.", recited as its own sentence.
            service_rule += (
                "It is dark: do not merely announce that. If you use the "
                "fact at all, give it a natural radio payoff ('Dark "
                "outside; plenty of noise in here.'), connect it to a "
                "listed weather fact ('Night's settled in, and we're down "
                "to 68 degrees.'), or leave it out -- leaving it out beats "
                "anything vacuous. Generic advice is fine ('so lock up'); "
                "never assert anything about the listener's house, the "
                "roads, the sky (stars, the moon), or what anyone is doing "
                "that is not listed. ")
        if p.daylight_phase in ("late_day", "sunset_imminent",
                                          "twilight", "dark",
                                          "sunrise_imminent"):
            service_rule += (
                "Do NOT describe bright sun, sunshine, or plenty of "
                "daylight -- a clear sky at this hour is not brightness. ")
        service_rule += "\n"
    extras_rule = ""
    if any((p.weather_context, p.upcoming_event, p.news_headline)):
        extras_rule = (
            "If weather_context, upcoming_event, or news_headline appear "
            "below, they are optional editorial color: weave them in "
            "naturally and conversationally. Use only the listed values: no "
            "other temperatures, forecasts, events, times, or news, and no "
            "commentary presented as fact. Music remains the backbone of "
            "the break.\n")
    return (
        "You are the DJ on the listener's personal radio station, "
        "speaking briefly between songs mid-shift. Write ONE short spoken "
        f"break, maximum {max_words} words, plain spoken prose only, no "
        "lists, no markdown.\n"
        "Timing: your words play exactly when ending_track finishes. Refer "
        "to ending_track only in the past tense ('that was ...'); NEVER "
        "call it 'now playing' or currently on. Mention next_track only if "
        "it is listed, as what's coming up. The two track facts are "
        "context, not obligations: they need not both feature in every "
        "break.\n"
        "Use ONLY the facts below. Do NOT invent times, albums, years, chart "
        "positions, recording trivia, station history, or listener "
        "statistics. Do not promise any upcoming song other than a listed "
        "next_track. If a fact is not listed below, it does not exist.\n"
        "Session history: you do not know what this station played earlier "
        "-- not the songs, not the artists, not how long ago. Never claim a "
        "song or artist was on before, never say 'a couple of tracks ago', "
        "'earlier', or 'we heard', and never describe or compare anything "
        "from earlier in the shift. Write only about the songs listed "
        "below. Cultural or thematic references to music that is NOT "
        "listed are welcome, but never imply this station played it.\n"
        + service_rule + extras_rule +
        "Never announce counts or statistics (how many tracks played, how "
        "many skipped, session length).\n"
        + dj_persona.delivery_block(profile, p.chattiness) +
        "Jokes, metaphors, puns, and mood language may be freely inventive "
        "-- but commentary on the MOVE between the two songs is not a free "
        "angle: describe the transition only when a transition_context line "
        "is listed (a real contrast), and otherwise simply hand off. FACTUAL "
        "claims about artists "
        "or songs -- origin, hometown or geography, biography, release or "
        "album history, recording location or circumstances, dates, chart "
        "history, lineup, trivia -- are allowed ONLY when explicitly "
        "supplied in the facts below; never infer them from the artist's "
        "identity. A joke may be invented; a fact may not.\n"
        "Openings: vary how you enter every single time -- a direct "
        "transition, a dry aside, title or artist wordplay, a "
        "contextual observation, or no setup phrase at all. Do not fall "
        "into a habit of starting with 'Well, that was...', 'That was...', "
        "'Alright...', or 'Hello...'; greetings are rarely needed "
        "mid-shift.\n\n"
        f"FACTS:\n{facts}\n\nBreak:"
    )
